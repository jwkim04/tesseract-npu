// selftest.cpp — the `selftest` base kernel (docs/design/01-execution-model.md "EX" §4.6; docs/design/
// 06-software-verification.md "SW" §7.1, §7.6, SW-28): the card and simulator self-test programs
// that tsr-selftest launches (D15: the host never pushes tile ops itself). The arguments, every test's
// parameters and data layout, and the return codes are in sw/ctrl/ref/selftest_abi.hpp (shared with
// the host side); each test's sequence is described at its function below.
//
// Every test reads its parameter line, checks its arguments (RET < 0 on a misuse, before any other
// push), loads its inputs from the data area into SPM, runs its tile ops with hardware deps, and
// stores its outputs back. Ids 1, 7, 8 and 10 also check their results on the device (RET > 0 names
// the failed check); the other ids leave the numerics to the host, which compares the outputs with the
// tsr_ref references (sw/ctrl/tests/test_fw_selftest_sim.c) and, on the card, with the simulator
// running the same image (SW §7.6). Ids 4 and 5 build every VU function / reduction and every AT op
// through the generated builders (vu.hpp, at.hpp), so their encodings track the area documents.
//
// Determinism (SW §4.7): pushed words, deps and SPM stores depend only on the arguments and the data
// area; the device-side checks read results only after joining the ops that produced them (DONE_SEQ
// ends the waits; the error records of done tickets are exact, I2; err_units() reads ERR_CODE, not
// QSTATUS). Id 10 reads credits under a QHOLD after Q_DEPTH pushes to the unit, the one credit value
// SW §4.7 admits as exact: credit = min(Q_DEPTH - (ISSUED_SEQ - S_sync), FIFO free, 15) (CC §7.3),
// ISSUED_SEQ moves only with the RISC-V's own pushes, and S cannot move while the hold keeps every
// head from starting (the SYNC NOPs wait for a held LD, so none completes without running either),
// so the queue term is exactly 0 and the min is 0 whatever the lag of S_sync or of the FIFO pointers. Every test leaves no unit in error state (id 7 clears each provoked error),
// so a passing test answers DONE OK. Launch-stateless: everything is rebuilt per launch.
#include "tsr/kernel.hpp"
#include "tsr/log.hpp"
#include "tsr/rt.hpp"

#include "../ref/selftest_abi.hpp"

TSR_KERNEL(selftest, 5, 0, 0);

namespace {

using namespace tsr;

constexpr int32_t ret(uint32_t code) { return int32_t(code); }

struct st_ctx {
    dram area;            // a1: the data area (line 0 = the parameter line)
    uint32_t lines;       // a2
    uint32_t seed;        // a3
    uint32_t flags;       // a4
    uint32_t P[16];       // the parameter line
    uint32_t num_ch, t_max, caps, spm;
};

dram at(const st_ctx & c, uint32_t line) { return c.area + line; }

// Parameter line: LD into the line below the runtime argument range, join, read through the window.
bool load_params(st_ctx & c, uint32_t id) {
    const spm pl{argblk_line() - 1u};
    join(issue(LD(pl, c.area, 1)));
    for (uint32_t i = 0; i < 16u; ++i) c.P[i] = spm_ptr<uint32_t>(pl)[i];
    return c.P[0] == TSR_ST_MAGIC && c.P[1] == id;
}

uint32_t f32_lines_(uint32_t n) { return (n + 15u) / 16u; }
uint32_t qa_lines_(qa k, uint32_t K) { return tsr_st_qa_lines(k == qa::a256 ? 1u : 0u, K); }

// The 16 words of SPM line l, through the line-buffered window (one fill per line).
void read_line(spm l, uint32_t w[16]) {
    const volatile uint32_t * p = spm_lb_ptr<uint32_t>(l);
    for (uint32_t i = 0; i < 16u; ++i) w[i] = p[i];
}

// ---------------------------------------------------------------------------------------------
// 1 LDST: LD / ST round trips of 1, 63, 64, 65 and Lmax lines on every channel, the RISC-V comparing
// the loaded lines with the source; then cross-channel copies of the Lmax copy.
int32_t t_ldst(st_ctx & c) {
    const uint32_t lmax = c.P[2], S = c.P[3];
    if (lmax == 0u || lmax > TSR_ST1_LMAX_MAX) return ret(TSR_ST_RET_BAD_PARAMS);
    if (1u + lmax > c.lines) return ret(TSR_ST_RET_AREA);
    uint32_t lens[5];
    const uint32_t n = tsr_st1_lens(lmax, lens);
    uint32_t off[5], sum = 0;
    for (uint32_t k = 0; k < n; ++k) {
        off[k] = sum;
        sum += lens[k];
    }
    const spm A{0}, B{TSR_ST1_LMAX_MAX};
    for (uint32_t k = 0; k < n; ++k) {
        issue(LD(A, at(c, 1), lens[k]));
        for (uint32_t ch = 0; ch < c.num_ch; ++ch) {
            const dram d{uint8_t(ch), S + off[k]};
            issue(ST(A, d, lens[k]));
            const ticket t = issue(LD(B, d, lens[k]));
            join(t);
            if (error(t) != 0u) return ret(0x01000000u | (k << 8) | ch);
            for (uint32_t l = 0; l < lens[k]; ++l) {
                uint32_t a[16], b[16];
                read_line(A + l, a);
                read_line(B + l, b);
                for (uint32_t i = 0; i < 16u; ++i)
                    if (a[i] != b[i]) return ret(0x01000000u | (k << 8) | ch);
            }
        }
    }
    for (uint32_t ch = 0; ch < c.num_ch; ++ch) {
        issue(LD(B, dram{uint8_t(ch), S + off[n - 1u]}, lmax));
        issue(ST(B, dram{uint8_t((ch + 1u) % c.num_ch), S + sum}, lmax));
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 2 QACT: A32 and A256 of T host rows (one QACT each over all rows), stored for the host.
int32_t t_qact(st_ctx & c) {
    const uint32_t K = c.P[2], T = c.P[3];
    if (K < 256u || K > 8192u || K % 256u != 0u || T == 0u || T > c.t_max) return ret(TSR_ST_RET_BAD_PARAMS);
    const tsr_st2_layout L = tsr_st2(K, T);
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    const uint32_t xl = K / 16u, q32 = qa_lines_(qa::a32, K), q256 = qa_lines_(qa::a256, K);
    const spm X{0}, Q32 = X + T * xl, Q256 = Q32 + T * q32;
    const ticket ld = issue(LD(X, at(c, L.x), T * xl));
    const ticket a = issue(QACT(qa::a32, Q32, X, K, T, xl), deps{ld});
    const ticket b = issue(QACT(qa::a256, Q256, X, K, T, xl), deps{ld});
    issue(ST(Q32, at(c, L.qa32), T * q32), deps{a});
    issue(ST(Q256, at(c, L.qa256), T * q256), deps{b});
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 3 MVM: Q8_0 and Q4_K at T = 1 and T = T_MAX, then MVM ACC (Q4_K onto the Q8_0 T_MAX result).
int32_t t_mvm(st_ctx & c) {
    const uint32_t N = c.P[2], K = c.P[3], TM = c.t_max;
    const wref w8{c.P[4], uint8_t(FMT_Q8_0)}, w4{c.P[5], uint8_t(FMT_Q4_K)};
    if (N == 0u || N > 8192u || K < 256u || K > 8192u || K % 256u != 0u) return ret(TSR_ST_RET_BAD_PARAMS);
    const tsr_st3_layout L = tsr_st3(N, K, TM);
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    const uint32_t xl = K / 16u, yl = f32_lines_(N), q8l = qa_lines_(qa::a32, K), q4l = qa_lines_(qa::a256, K);
    const spm X{0}, Q8 = X + TM * xl, Q4 = Q8 + TM * q8l, Y0 = Q4 + TM * q4l, Y1 = Y0 + yl, Y2 = Y1 + TM * yl,
              Y3 = Y2 + yl;
    const ticket ld = issue(LD(X, at(c, L.x), TM * xl));
    const ticket q8 = issue(QACT(qa::a32, Q8, X, K, TM, xl), deps{ld});
    const ticket q4 = issue(QACT(qa::a256, Q4, X, K, TM, xl), deps{ld});
    const ticket m0 = issue(MVM(Y0, Q8, w8, N, K, 1, yl), deps{q8});
    const ticket m1 = issue(MVM(Y1, Q8, w8, N, K, TM, yl), deps{q8});
    const ticket m2 = issue(MVM(Y2, Q4, w4, N, K, 1, yl), deps{q4});
    const ticket m3 = issue(MVM(Y3, Q4, w4, N, K, TM, yl), deps{q4});
    issue(ST(Y0, at(c, L.y8_1), yl), deps{m0});
    const ticket s1 = issue(ST(Y1, at(c, L.y8_t), TM * yl), deps{m1});
    issue(ST(Y2, at(c, L.y4_1), yl), deps{m2});
    issue(ST(Y3, at(c, L.y4_t), TM * yl), deps{m3});
    const ticket acc = issue(MVM(Y1, Q4, w4, N, K, TM, yl, true), deps{s1});   // y8_T stored first (WAR)
    issue(ST(Y1, at(c, L.yacc), TM * yl), deps{acc});
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 4 VU: the data area is mirrored in SPM (area line l at SPM line l - 1) and loaded whole, so every
// output line is defined even where an op writes part of it; then every VOP function, the MULC SREG
// form, every RED kind and post-op, RED TO_S, every SOP function, ROPE in both modes (recurrence form,
// NORM plain and NEOX with FF and FS), TOPK and DQROW (when CAPS has them), all on the VU queue in
// this order; the outputs are stored after the last VU op.
int32_t t_vu(st_ctx & c) {
    if (!(c.caps & CAPS_VU)) return ret(TSR_ST_RET_CAPS);
    const uint32_t M = c.P[2], R = c.P[3];
    if (M == 0u || M > 64u || R == 0u || R > 16u) return ret(TSR_ST_RET_BAD_PARAMS);
    const tsr_st4_layout L = tsr_st4(M, R);
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    auto S = [](uint32_t area_line) { return area_line - 1u; };     // SPM line of an area line
    const ticket ld = issue(LD(spm{0}, at(c, 1), L.end - 1u));
    join(ld);                                                      // the RISC-V reads sv[12] (the SOP imm)
    const uint32_t imm = spm_ptr<uint32_t>(spm{S(L.sv)})[12];
    ticket last{};
    // s registers: LDS s0..s15 from sv, then MOVI the RMS and ROPE constants.
    for (uint32_t i = 0; i < 16u; ++i)
        last = issue(SOP({.sub = SOP_SUB_LDS, .sd = i, .addr = 16u * S(L.sv) + i}));
    const uint32_t movi[5][2] = { {1u, c.P[7]}, {2u, c.P[8]}, {4u, c.P[5]}, {5u, c.P[4]}, {6u, c.P[6]} };
    for (const auto & m : movi) last = issue(SOP({.sub = SOP_SUB_MOVI, .sd = m[0], .imm = m[1]}));
    // VOP: every function.
    for (uint32_t f = 0; f < VOP_FUNC_N; ++f) {
        const uint32_t rd = vop_func_reads(f);
        vop_f v{};
        v.func = f;
        v.cmode = (rd & VOP_READS_C) ? VOP_CMODE_VEC : VOP_CMODE_NONE;
        v.d = S(L.vop) + f * R * L.L;
        v.a = (rd & VOP_READS_A) ? S(L.a) : 0u;
        v.b = (rd & VOP_READS_B) ? S(L.b) : 0u;
        v.c = (rd & VOP_READS_C) ? S(L.c) : 0u;
        v.m = M;
        v.rows = R;
        v.d_stride = L.L;
        v.a_stride = (rd & VOP_READS_A) ? L.L : 0u;
        v.b_stride = (rd & VOP_READS_B) ? L.L : 0u;
        last = issue(VOP(v));
    }
    last = issue(VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_MULC, .si = 3u, .d = S(L.vop_sreg), .a = S(L.a), .m = M,
                      .rows = R, .d_stride = L.L, .a_stride = L.L}));
    // RED: kinds x post-ops (MAX: NONE only), per-row vectors; then SUM of row 0 to s7.
    const uint32_t reds[TSR_ST4_N_RED][2] = { {RED_KIND_SUM, RED_POST_NONE},   {RED_KIND_SUM, RED_POST_RCP},
                                              {RED_KIND_SUM, RED_POST_RMS},    {RED_KIND_SUMSQ, RED_POST_NONE},
                                              {RED_KIND_SUMSQ, RED_POST_RCP},  {RED_KIND_SUMSQ, RED_POST_RMS},
                                              {RED_KIND_MAX, RED_POST_NONE} };
    for (uint32_t j = 0; j < TSR_ST4_N_RED; ++j) {
        const bool rms = reds[j][1] == RED_POST_RMS;
        last = issue(RED({.kind = reds[j][0], .post = reds[j][1], .sk = rms ? 1u : 0u, .se = rms ? 2u : 0u,
                          .d = S(L.red) + j, .a = S(L.a), .m = M, .rows = R, .a_stride = L.L}));
    }
    last = issue(RED({.kind = RED_KIND_SUM, .to_s = 1u, .sd = 7u, .a = S(L.a), .m = M, .rows = 1u}));
    // SOP: every arithmetic function into s8, each stored by STS.
    const uint32_t subs[12] = { SOP_SUB_ADD, SOP_SUB_MUL, SOP_SUB_ADDI, SOP_SUB_MULI, SOP_SUB_SQRT, SOP_SUB_RECIP,
                                SOP_SUB_SUB, SOP_SUB_DIV, SOP_SUB_RSQRT, SOP_SUB_MAX, SOP_SUB_MIN, SOP_SUB_EXP };
    const uint32_t sop_word = 16u * S(L.sop);
    for (uint32_t j = 0; j < 12u; ++j) {
        const uint32_t u = sop_uses(subs[j]);
        sop_f s{};
        s.sub = subs[j];
        s.sd = 8u;
        s.sa = 10u;
        s.sb = (u & SOP_USES_SB) ? 11u : 0u;
        s.imm = (u & SOP_USES_IMM) ? imm : 0u;
        issue(SOP(s));
        last = issue(SOP({.sub = SOP_SUB_STS, .sa = 8u, .addr = sop_word + j}));
    }
    issue(SOP({.sub = SOP_SUB_STS, .sa = 7u, .addr = sop_word + 12u}));
    issue(SOP({.sub = SOP_SUB_STS, .sa = 0u, .addr = sop_word + 13u}));
    issue(SOP({.sub = SOP_SUB_MOVI, .sd = 9u, .imm = c.P[7]}));
    issue(SOP({.sub = SOP_SUB_STS, .sa = 9u, .addr = sop_word + 14u}));
    last = issue(SOP({.sub = SOP_SUB_STS, .sa = 15u, .addr = sop_word + 15u}));
    // ROPE: 2 tokens x 2 heads x 64.
    last = issue(ROPE({.mode = ROPE_MODE_NORM, .sm = 4u, .sts = 5u, .d = S(L.rope_norm), .src = S(L.rope_x),
                       .pos = S(L.rope_pos), .head_dim = 64u, .n_dims = 64u, .n_heads = 2u, .t = 2u}));
    last = issue(ROPE({.mode = ROPE_MODE_NEOX, .ff = 1u, .fs = 1u, .sm = 4u, .sts = 5u, .sfs = 6u, .d = S(L.rope_neox),
                       .src = S(L.rope_x), .pos = S(L.rope_pos), .table = S(L.rope_ff), .head_dim = 64u, .n_dims = 32u,
                       .n_heads = 2u, .t = 2u}));
    if (c.caps & CAPS_TOPK)
        last = issue(TOPK({.k = 4u, .d = S(L.topk), .a = S(L.a), .m = M, .rows = R, .a_stride = L.L}));
    if (c.caps & CAPS_DQROW) {
        issue(DQROW({.fmt = FMT_Q8_0, .d = S(L.dq8_out), .src = S(L.dq8), .k = 256u, .rows = 2u}));
        last = issue(DQROW({.fmt = FMT_Q4_K, .d = S(L.dq4_out), .src = S(L.dq4), .k = 256u, .rows = 2u}));
    }
    issue(ST(spm{S(L.vop)}, at(c, L.vop), L.end - L.vop), deps{last});
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 5 AT: one small attention (E = D = 64, one head, n_kv cells, T_q queries) through every AT op:
// QACT of K / V / Q rows, KVST K and V, ATTQK (plain and CAUSAL, MAXOUT), ASMAX, AMASK, ATTPV LSUM,
// AMERGE (INIT, and INIT followed by an online merge), ANORM. Area mirrored in SPM as in id 4.
int32_t t_at(st_ctx & c) {
    if (!(c.caps & CAPS_ATT)) return ret(TSR_ST_RET_CAPS);
    const uint32_t n_kv = c.P[2], Tq = c.P[3], scale = c.P[4], BK = c.P[5], BV = c.P[6], C = c.num_ch;
    if (n_kv == 0u || n_kv > 16u * C || Tq == 0u || Tq > c.t_max) return ret(TSR_ST_RET_BAD_PARAMS);
    const tsr_st5_layout L = tsr_st5(n_kv, Tq, C);
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    auto S = [](uint32_t area_line) { return area_line - 1u; };
    const uint32_t TM = c.t_max;
    const ticket ld = issue(LD(spm{0}, at(c, 1), L.end - 1u));
    // QACT A32 (K = 64): rows in groups of T_MAX tokens.
    auto qact_rows = [&](uint32_t src, uint32_t dst, uint32_t rows) {
        ticket t{};
        for (uint32_t r0 = 0; r0 < rows; r0 += TM) {
            const uint32_t tt = rows - r0 < TM ? rows - r0 : TM;
            t = issue(QACT(qa::a32, spm{S(dst) + r0 * L.qal}, spm{S(src) + r0 * 4u}, 64u, tt, 4u), deps{ld});
        }
        return t;
    };
    const ticket qk = qact_rows(L.k, L.kqa, n_kv), qv = qact_rows(L.v, L.vqa, n_kv), qq = qact_rows(L.q, L.qqa, Tq);
    const ticket kk = issue(KVST({.kv = KVST_KV_K, .n_rows_m1 = n_kv - 1u, .src = S(L.kqa), .base = BK, .e64 = 1u,
                                  .lg = 1u, .c0 = 0u}), deps{qk});
    const ticket kv = issue(KVST({.kv = KVST_KV_V, .n_rows_m1 = n_kv - 1u, .src = S(L.vqa), .base = BV, .e64 = 1u,
                                  .lg = 1u, .c0 = 0u}), deps{qv});
    const ticket s0 = issue(ATTQK({.maxout = 1u, .dst = S(L.scores), .x = S(L.qqa), .base = BK, .e64 = 1u,
                                   .tq_m1 = Tq - 1u, .u0 = 0u, .n_groups = L.n_groups, .c_lo = 0u, .c_hi = n_kv,
                                   .mdst = S(L.mvec)}), after(kk, qq));
    issue(ATTQK({.causal = 1u, .maxout = 1u, .dst = S(L.scores_c), .x = S(L.qqa), .base = BK, .e64 = 1u,
                 .tq_m1 = Tq - 1u, .u0 = 0u, .n_groups = L.n_groups, .c_lo = 0u, .c_hi = n_kv, .mdst = S(L.mvec_c)}));
    const ticket sm = issue(ASMAX({.dst = S(L.p), .src = S(L.scores), .mvec = S(L.mvec), .n_lines = L.n_groups,
                                   .r = L.R, .scale = scale}), deps{s0});
    issue(AMASK({.dst = S(L.sm), .src = S(L.scores), .mask = S(L.mask), .n_lines = L.n_groups, .r = L.R, .rpq = 1u,
                 .mstride = L.n_groups, .mdst = S(L.mvec_m)}));
    const ticket pv = issue(ATTPV({.lsum = 1u, .dst = S(L.parts), .p = S(L.p), .base = BV, .e64 = 1u, .tq_m1 = Tq - 1u,
                                   .u0 = 0u, .n_groups = L.n_groups, .lg = 1u}), after(sm, kv));
    issue(AMERGE({.init = 1u, .orun = S(L.orun1), .parts = S(L.parts), .mk = S(L.mvec), .st = S(L.st1), .r = L.R,
                  .scale = scale}), deps{pv});
    issue(AMERGE({.init = 1u, .orun = S(L.orun2), .parts = S(L.parts), .mk = S(L.mvec), .st = S(L.st2), .r = L.R,
                  .scale = scale}));
    issue(AMERGE({.init = 0u, .orun = S(L.orun2), .parts = S(L.parts), .mk = S(L.mvec_m), .st = S(L.st2), .r = L.R,
                  .scale = scale}));
    const ticket an = issue(ANORM({.out = S(L.out), .orun = S(L.orun2), .l_run = S(L.st2) + 1u, .r = L.R}));
    issue(ST(spm{S(L.kqa)}, at(c, L.kqa), L.end - L.kqa), after(an, qq));
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 6 CONC: docs/ctrl-riscv.md §8.7's matvec_batch on every unit: per job LD x, QACT, MVM, VOP SILU,
// RED SUM, ST, double-buffered with hardware deps only (the LD of job j+1 waits for the QACT of job
// j-1, the MVM for the VOP that read its Y, the VOP for the ST that read its Z); no join inside.
int32_t t_conc(st_ctx & c) {
    const uint32_t N = c.P[2], K = c.P[3], fmt = c.P[5], J = c.P[6];
    if (!(c.caps & CAPS_VU)) return ret(TSR_ST_RET_CAPS);
    if (N < 16u || N > 4096u || K < 256u || K > 4096u || K % 256u != 0u || fmt > FMT_Q4_K || J == 0u || J > 64u)
        return ret(TSR_ST_RET_BAD_PARAMS);
    const tsr_st6_layout L = tsr_st6(N, K, J);
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    const wref w{c.P[4], uint8_t(fmt)};
    const qa kind = qa_for(w);
    const uint32_t xl = K / 16u, yl = f32_lines_(N), ql = qa_lines_(kind, K);
    const spm X[2] = { spm{0}, spm{xl} }, Q[2] = { spm{2u * xl}, spm{2u * xl + ql} };
    const spm Y[2] = { spm{2u * (xl + ql)}, spm{2u * (xl + ql) + yl} };
    const spm Z[2] = { Y[1] + yl, Y[1] + 2u * yl };
    const spm Ssum = Z[1] + yl;
    ticket ld[2], q[2], mv[2], vo[2], st[2], red;
    ld[0] = issue(LD(X[0], at(c, L.x), xl));
    for (uint32_t j = 0; j < J; ++j) {
        const uint32_t b = j & 1u;
        q[b] = issue(QACT(kind, Q[b], X[b], K, 1), after(ld[b], mv[b]));
        mv[b] = issue(MVM(Y[b], Q[b], w, N, K, 1), after(q[b], vo[b]));
        if (j + 1u < J) ld[b ^ 1u] = issue(LD(X[b ^ 1u], at(c, L.x + (j + 1u) * xl), xl), deps{q[b ^ 1u]});
        vo[b] = issue(VOP({.func = VOP_FUNC_SILU, .d = Z[b].line, .a = Y[b].line, .m = N, .rows = 1u}), after(mv[b], st[b]));
        red = issue(RED({.kind = RED_KIND_SUM, .d = Ssum.line + j, .a = Z[b].line, .m = N, .rows = 1u}));
        st[b] = issue(ST(Z[b], at(c, L.y + j * yl), yl), deps{vo[b]});
    }
    issue(ST(Ssum, at(c, L.s), J), deps{red});
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 7 ERRORS (device self-checking): each provoked error is checked through error() / first_error()
// once its ticket is done, then cleared with QCLEAR (and the clear checked), so the launch ends with
// no unit in error state. RET = the number of the first failed check.
int32_t t_errors(st_ctx & c) {
    const tsr_st7_layout L = tsr_st7();
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    const uint32_t bad = c.P[2];
    const wref w{c.P[3], uint8_t(FMT_Q8_0)};
    const dram in = at(c, L.in), out = at(c, L.out);
    auto clear_ok = []() {
        join(qclear());
        return err_units() == 0u;
    };
    // DRAM_RANGE: the code, the record (ERR_INFO = the start beat, ERR_W0, the push index).
    uint32_t p = pidx_now();
    const op bad_ld = LD(spm{0}, dram{0, bad}, 1);
    ticket t = issue(bad_ld);
    join(t);
    if (error(t) != ERR_DRAM_RANGE) return 1;
    err_rec e{};
    if (!first_error(&e) || e.code != ERR_DRAM_RANGE || e.info != bad || e.w0 != bad_ld.w[0] || e.pidx != p || !(e.t == t))
        return 2;
    if (!clear_ok()) return 3;
    // ILLEGAL_OP: a reserved opcode (refused: SYNC), and a reserved bit of an LD.
    op end{};
    end.w[0] = OP_END;
    end = op_finish(end);
    t = issue(end);
    join(t);
    if (t.unit() != UNIT_SYNC || error(t) != ERR_ILLEGAL_OP) return 4;
    op rsv = LD(spm{0}, in, 1);
    rsv.w[0] |= 1u << 12;
    t = issue(rsv);
    join(t);
    if (error(t) != ERR_ILLEGAL_OP) return 5;
    if (!clear_ok()) return 6;
    // BAD_DEP: a dep on a DMA ticket not pushed yet.
    t = issue(LD(spm{0}, in, 2), deps{make_ticket(UNIT_DMA, issued_seq(UNIT_DMA) + 1000u)});
    join(t);
    if (error(t) != ERR_BAD_DEP) return 7;
    if (!clear_ok()) return 8;
    // DEP_FAILED chain: a failing LD, a QACT on it, an MVM on the QACT.
    p = pidx_now();
    const ticket a = issue(LD(spm{0}, dram{0, bad}, 2));
    const ticket b = issue(QACT(qa::a32, spm{2}, spm{0}, 32, 1), deps{a});
    const ticket m = issue(MVM(spm{4}, spm{2}, w, 16, 32, 1), deps{b});
    join(m);
    if (error(a) != ERR_DRAM_RANGE || error(b) != ERR_DEP_FAILED || error(m) != ERR_DEP_FAILED) return 9;
    if (!first_error(&e) || e.code != ERR_DRAM_RANGE || e.pidx != p) return 10;
    if (!clear_ok()) return 11;
    // QABORT with nothing pending, then a push: ABORTED; QCLEAR recovers; a round trip succeeds.
    join(issue(LD(spm{8}, in, 2)));
    qabort();
    t = issue(LD(spm{8}, in, 2));
    join(t);
    if (error(t) != ERR_ABORTED) return 12;
    if (!clear_ok()) return 13;
    const ticket r1 = issue(LD(spm{8}, in, 2));
    const ticket r2 = issue(ST(spm{8}, out, 2));
    join(r2);
    if (error(r1) != 0u || error(r2) != 0u || err_units() != 0u) return 14;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 8 SPMWIN (device self-checking): byte, half, word and misaligned stores and loads through the
// exact window, a store seen through the line-buffered alias after a fill, then an ST pushed right
// after the stores (CC §7.1: no SPM-window store is outstanding at a push) and an LD back.
volatile uint8_t * b8(spm l) { return spm_ptr<uint8_t>(l); }

void st32(volatile uint8_t * p, uint32_t v) {
#ifdef TSR_FW_HOST
    for (uint32_t i = 0; i < 4u; ++i) p[i] = uint8_t(v >> (8u * i));   // the host build has no split accesses
#else
    *reinterpret_cast<volatile uint32_t *>(p) = v;                       // misaligned: two accesses (CC, Ibex)
#endif
}
uint32_t ld32(const volatile uint8_t * p) {
#ifdef TSR_FW_HOST
    uint32_t v = 0;
    for (uint32_t i = 0; i < 4u; ++i) v |= uint32_t(p[i]) << (8u * i);
    return v;
#else
    return *reinterpret_cast<const volatile uint32_t *>(p);
#endif
}
void st16(volatile uint8_t * p, uint32_t v) {
#ifdef TSR_FW_HOST
    p[0] = uint8_t(v);
    p[1] = uint8_t(v >> 8);
#else
    *reinterpret_cast<volatile uint16_t *>(p) = uint16_t(v);
#endif
}

int32_t t_spmwin(st_ctx & c) {
    const tsr_st8_layout L = tsr_st8();
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    const uint32_t L0 = c.P[2], seed = c.seed;
    if (L0 + 4u > argblk_line() - 1u) return ret(TSR_ST_RET_BAD_PARAMS);
    volatile uint8_t * p = b8(spm{L0});
    // 1-2: bytes of line 0, read back as words (little-endian).
    for (uint32_t i = 0; i < 64u; ++i) p[i] = uint8_t(seed + 37u * i);
    for (uint32_t k = 0; k < 16u; ++k) {
        uint32_t want = 0;
        for (uint32_t i = 0; i < 4u; ++i) want |= uint32_t(uint8_t(seed + 37u * (4u * k + i))) << (8u * i);
        if (spm_ptr<uint32_t>(spm{L0})[k] != want) return 1;
    }
    // 3: halves of line 1, read back as bytes.
    for (uint32_t k = 0; k < 32u; ++k) spm_ptr<uint16_t>(spm{L0 + 1u})[k] = uint16_t((seed >> 8) ^ (k * 0x0101u));
    for (uint32_t k = 0; k < 32u; ++k) {
        const uint32_t h = ((seed >> 8) ^ (k * 0x0101u)) & 0xFFFFu;
        if (p[64u + 2u * k] != uint8_t(h) || p[65u + 2u * k] != uint8_t(h >> 8)) return 3;
    }
    // 4: an aligned word at byte 8: bytes and the upper half.
    const uint32_t w = seed * 2654435761u;
    spm_ptr<uint32_t>(spm{L0})[2] = w;
    for (uint32_t i = 0; i < 4u; ++i)
        if (p[8u + i] != uint8_t(w >> (8u * i))) return 4;
    if (spm_ptr<uint16_t>(spm{L0})[5] != uint16_t(w >> 16)) return 5;
    // 5: a misaligned word across the two lines (bytes 61..64) and a misaligned half (99, 100).
    st32(p + 61, ~seed);
    for (uint32_t i = 0; i < 4u; ++i)
        if (p[61u + i] != uint8_t(~seed >> (8u * i))) return 6;
    if (ld32(p + 61) != ~seed) return 7;
    const uint32_t mh = (seed ^ 0x5A5Au) & 0xFFFFu;
    st16(p + 99, mh);
    if (p[99] != uint8_t(mh) || p[100] != uint8_t(mh >> 8)) return 8;
    // 6: the line-buffered alias: a fill, a store through the exact window, the new value.
    volatile uint8_t * lb = spm_lb_ptr<uint8_t>(spm{L0});
    if (lb[32] != uint8_t(seed + 37u * 32u)) return 9;
    p[32] = uint8_t(seed ^ 0xC3u);
    if (lb[32] != uint8_t(seed ^ 0xC3u)) return 10;
    // 7: ST right after the stores, LD back, compare with the pattern.
    uint8_t want[128];
    tsr_st8_pattern(seed, want);
    issue(ST(spm{L0}, at(c, L.out), 2));
    const ticket t = issue(LD(spm{L0 + 2u}, at(c, L.out), 2));
    join(t);
    for (uint32_t i = 0; i < 128u; ++i) {
        if (p[i] != want[i]) return 11;
        if (p[128u + i] != want[i]) return 12;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 9 LOG: two FMT records and a kernel event (the host checks them; PING is the runtime's).
int32_t t_log(st_ctx & c) {
    TSR_LOG(TSR_LOG_INFO, "selftest 9: seed %x, id %u", c.seed, 9u);
    TSR_LOG(TSR_LOG_WARN, "selftest 9: %u %x", ~c.seed, c.seed + 1u);
    event(TSR_ST9_EVENT, c.seed & 0xFFFFu, c.seed, ~c.seed);
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 10 CREDITS (device self-checking): under QHOLD, Q_DEPTH ops into every built unit (SYNC: NOPs
// waiting for a held LD) must leave its credit at 0 (exact under the hold: file header, SW §4.7);
// then the hold is released and everything completes without an error. The held ops touch disjoint
// SPM per unit (no race once released).
int32_t t_credits(st_ctx & c) {
    const tsr_st10_layout L = tsr_st10();
    if (L.end > c.lines) return ret(TSR_ST_RET_AREA);
    const wref w{c.P[2], uint8_t(FMT_Q8_0)};
    const spm X{0}, Q{2}, Q2{4}, Y{6}, Z{7}, D{8};
    const ticket ld = issue(LD(X, at(c, L.x), 2));
    join(issue(QACT(qa::a32, Q, X, 32, 1, 2), deps{ld}));
    const uint32_t qd = core_csr(CSR_Q_DEPTH);
    auto depth = [qd](uint32_t u) { return (qd >> (Q_DEPTH_FIELD_W * u)) & 0x1Fu; };
    int32_t fail = 0;
    qhold(true);
    ticket held{};
    static const uint32_t units[4] = { UNIT_DMA, UNIT_QACT, UNIT_MVM, UNIT_VU };
    for (uint32_t u : units) {
        if (u == UNIT_VU && !(c.caps & CAPS_VU)) continue;
        const uint32_t d = depth(u);
        for (uint32_t i = 0; i < d; ++i) {
            ticket t{};
            if (u == UNIT_DMA) t = issue(LD(D, at(c, L.x), 1));
            else if (u == UNIT_QACT) t = issue(QACT(qa::a32, Q2, X, 32, 1, 2));
            else if (u == UNIT_MVM) t = issue(MVM(Y, Q, w, 16, 32, 1));
            else t = issue(VOP({.func = VOP_FUNC_COPY, .d = Z.line, .a = X.line, .m = 16u, .rows = 1u}));
            if (u == UNIT_DMA && i == 0u) held = t;
        }
        if (d != 0u && credit(u) != 0u && fail == 0) fail = int32_t(0x100u * (u + 1u) + 1u);
    }
    const uint32_t ds = depth(UNIT_SYNC);
    for (uint32_t i = 0; i < ds; ++i) issue(NOP(), deps{held});
    if (credit(UNIT_SYNC) != 0u && fail == 0) fail = int32_t(0x100u * (UNIT_SYNC + 1u) + 1u);
    qhold(false);
    join_all();
    if (err_units() != 0u && fail == 0) fail = 0x700;
    return fail;
}

}  // namespace

extern "C" int32_t selftest(const tsr_args * args) {
    st_ctx c{};
    const uint32_t id = args->a[0];
    c.area = from_handle(args->a[1]);
    c.lines = args->a[2];
    c.seed = args->a[3];
    c.flags = args->a[4];
    if (id == 0u || id > TSR_ST_N_IDS) return ret(TSR_ST_RET_BAD_ID);
    const uint32_t cfg0 = core_csr(CSR_CFG0);
    c.num_ch = field_get(cfg0, CFG0_NUM_CH, CFG0_NUM_CH_LSB);
    c.t_max = field_get(cfg0, CFG0_T_MAX, CFG0_T_MAX_LSB);
    c.caps = core_csr(CSR_CAPS);
    c.spm = spm_lines();
    if (c.lines == 0u || c.t_max == 0u || c.num_ch == 0u) return ret(TSR_ST_RET_BAD_PARAMS);
    if (!load_params(c, id)) return ret(TSR_ST_RET_BAD_PARAMS);
    set_info(id);
    switch (id) {
        case TSR_ST_ID_LDST:    return t_ldst(c);
        case TSR_ST_ID_QACT:    return t_qact(c);
        case TSR_ST_ID_MVM:     return t_mvm(c);
        case TSR_ST_ID_VU:      return t_vu(c);
        case TSR_ST_ID_AT:      return t_at(c);
        case TSR_ST_ID_CONC:    return t_conc(c);
        case TSR_ST_ID_ERRORS:  return t_errors(c);
        case TSR_ST_ID_SPMWIN:  return t_spmwin(c);
        case TSR_ST_ID_LOG:     return t_log(c);
        case TSR_ST_ID_CREDITS: return t_credits(c);
        default:                return ret(TSR_ST_RET_BAD_ID);
    }
}

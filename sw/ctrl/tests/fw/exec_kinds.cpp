// exec_kinds.cpp — test kinds of the graph executor tests (P2.4; docs/v2/06-software-verification.md
// "SW" §7.3, §10.2): handlers for kinds 0x30-0x3F (exec.hpp tsr_exec_test_kind), linked into the
// test image fw_exec (tsr_fw_image, sw/ctrl/tests/exec.cmake) and into the host driver
// test_fw_exec_host. They give the executor tests real data flow (copies through the ring, DRAM
// results to check), injected tile errors, handler failures, preload reads by the RISC-V and by the
// device, a cheap 5-op stub for the cycle measurement, and plan.hpp evaluated on the RISC-V.
//
// Unlike production handlers, PRE and PLAN join in software (test kinds only: they read or write
// SPM with the RISC-V and must order that against the ring's older users).
//
//   0x30 COPY   a0 src handle, a1 dst handle, a2 lines   LD ring block <- src (after d and the
//               ring's deps), ST dst <- block; ticket: the ST
//   0x31 STUB5  a0 src handle, a1 dst handle              LD (1 line), NOP, NOP, NOP, ST (1 line):
//               five pushes, each after the previous; ticket: the ST
//   0x32 FAIL   a0 lines                                 LD from beat 0x0FFF_FFC0 of ch0 (beyond
//               any test channel: DRAM_RANGE); ticket: the LD
//   0x33 PRE    a0 pre_byte, a1 n words (1..16), a2 dst   the RISC-V reads n words of the preload
//               area, stores word k ^ (0xA5A5_0000 + k) into a ring line (zeros after n), ST dst
//   0x34 PREST  a0 pre_byte, a1 lines, a2 dst             ST dst <- the preload area (device read)
//   0x35 RET    a0 code, a1 n NOPs                        pushes n NOPs (after d), then fails with
//               ctx.ret = code (the executor's handler-failure path)
//   0x36 PLAN   a0 table handle, a1 n (<= 64), a2 out     LDs n shapes {fmt, N, K, T} (16 B each),
//               runs plan::mm_plan / check_op for each with the executor's dev_cfg and writes
//               8 words per shape: status, kind, x_lines, qa_lines, y_lines, lpt, t_tile, n_tiles,
//               then 4 words: ring_budget, check_op(MUL_MAT record), dev t_max, dev k_max
//               (12 words, 48 B per shape, padded to 64 B: one line per shape)
//   0x37 RACE   a0 src handle, a1 with_dep                LD 2 lines into a ring block and a QACT
//               (A32, K = 32) reading them, with or without the dep: the race detector probe
//               (without the dep the QACT races the LD on the block, SW §4.6)
//   0x38 VU     a0 src handle, a1 dst handle, a2 m (f32      LD L = ceil(m/16) lines into a ring block,
//               elements, 1..8192), a3 n (VU ops, 1..3)   n VOPs on them (ADD x+x -> y, MIN y,x -> y,
//               MAX y,x -> y; full rate, short latency class), ST dst <- y: 2 + n pushes, each after
//               the previous; ticket: the ST. A stand-in for M7's short VU kinds (norm, RoPE, GLU,
//               residual: W2b's kinds do not exist yet) in fw_m6_perf's M7 projection (2026-10-30,
//               executor-optimisation integration); it goes through the generic handler call, as
//               W2b's kinds will
//   0x39 FASTVU a0 src, a1 dst, a2 m, a3 0               LD, VOP ADD, ST in the k_fast.hpp fast-path form
//               (W2b infra; tests/fw/exec_fast.cpp, -O2): fw_exec_perf's reference short VU record
//   0x3A SREG   a0 src, a1 dst, a2 m, a3 s, a4 v, a5 fl   y = x * s[i] with s[i] = v through the launch's
//               s-register shadow (k_fast.hpp sreg_movi; exec_fast.cpp)
#include "tsr/exec.hpp"
#include "tsr/kernel.hpp"
#include "tsr/plan.hpp"

namespace {

using namespace tsr;

constexpr uint32_t TK_COPY = 0x30u, TK_STUB5 = 0x31u, TK_FAIL = 0x32u, TK_PRE = 0x33u, TK_PREST = 0x34u,
                   TK_RET = 0x35u, TK_PLAN = 0x36u, TK_RACE = 0x37u, TK_VU = 0x38u, TK_FASTVU = 0x39u,
                   TK_SREG = 0x3Au;

// d and the ring's deps for a write to a fresh block, folded to <= 2 (group: join2 NOPs).
deps fold(deps a, deps b) {
    if (a.a.none() && a.b.none()) return b;
    if (b.a.none() && b.b.none()) return a;
    group g{};
    g.add(a);
    g.add(b);
    return g.as_deps();
}

void sw_join(deps w) {
    join(w.a);
    join(w.b);
}

ticket k_copy(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    deps w{};
    const spm blk = c.ring->alloc(a[2], 2u, &w);
    if (!valid(blk)) {
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const ticket ld = issue(LD(blk, from_handle(a[0]), a[2]), fold(d, w), c.push_flags);
    const ticket st = issue(ST(blk, from_handle(a[1]), a[2]), deps{ld}, c.push_flags);
    c.ring->release(blk, st);
    return st;
}

ticket k_stub5(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    deps w{};
    const spm blk = c.ring->alloc(1u, 2u, &w);
    if (!valid(blk)) {
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    ticket t = issue(LD(blk, from_handle(a[0]), 1u), fold(d, w), c.push_flags);
    for (int k = 0; k < 3; ++k) t = issue(NOP(), deps{t}, c.push_flags);
    t = issue(ST(blk, from_handle(a[1]), 1u), deps{t}, c.push_flags);
    c.ring->release(blk, t);
    return t;
}

ticket k_fail(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    deps w{};
    const spm blk = c.ring->alloc(a[0], 2u, &w);
    if (!valid(blk)) {
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const ticket t = issue(LD(blk, dram{0u, 0x0FFFFFC0u}, a[0]), fold(d, w), c.push_flags);
    c.ring->release(blk, t);
    return t;
}

ticket k_pre(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    const uint32_t n = a[1] < 16u ? a[1] : 16u;
    if (c.preload == nullptr || a[0] % 4u != 0u || a[0] + 4u * n > c.pre_lines * 64u) {
        c.ret = opl::RET_BAD_RECORD;
        return ticket{};
    }
    deps w{};
    const spm blk = c.ring->alloc(1u, 2u, &w);
    if (!valid(blk)) {
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    sw_join(w);                                          // the RISC-V writes the block itself
    const volatile uint32_t * src = reinterpret_cast<const volatile uint32_t *>(c.preload + a[0]);
    volatile uint32_t * dst = spm_ptr<uint32_t>(blk);
    for (uint32_t k = 0; k < 16u; ++k) dst[k] = k < n ? (src[k] ^ (0xA5A50000u + k)) : 0u;
    const ticket st = issue(ST(blk, from_handle(a[2]), 1u), d, c.push_flags);
    c.ring->release(blk, st);
    return st;
}

ticket k_prest(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    if (c.preload == nullptr || a[0] % 64u != 0u || a[1] == 0u || a[0] / 64u + a[1] > c.pre_lines) {
        c.ret = opl::RET_BAD_RECORD;
        return ticket{};
    }
    return issue(ST(spm{c.pre.line + a[0] / 64u}, from_handle(a[2]), a[1]), d, c.push_flags);
}

ticket k_ret(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    ticket t{};
    for (uint32_t k = 0; k < a[1]; ++k) t = issue(NOP(), k == 0u ? d : deps{t}, c.push_flags);
    c.ret = a[0];
    return t;
}

ticket k_plan(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    const uint32_t n = a[1];
    if (n == 0u || n > 64u) {
        c.ret = opl::RET_BAD_RECORD;
        return ticket{};
    }
    const uint32_t in_lines = (16u * n + 63u) / 64u;
    deps w{};
    const spm tab = c.ring->alloc(in_lines, 2u, &w);
    if (!valid(tab)) {
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const ticket ld = issue(LD(tab, from_handle(a[0]), in_lines), fold(d, w), c.push_flags);
    deps w2{};
    const spm out = c.ring->alloc(n, 2u, &w2);
    if (!valid(out)) {
        c.ring->release(tab, ld);
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    join(ld);                                            // the RISC-V reads the table
    sw_join(w2);                                         // and writes the out block
    const volatile uint32_t * t = spm_ptr<uint32_t>(tab);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t fmt = t[4 * i], N = t[4 * i + 1], K = t[4 * i + 2], T = t[4 * i + 3];
        plan::mm_tiling m{};
        const uint32_t st = plan::mm_plan(*c.cfg, fmt, N, K, T, &m);
        uint32_t rec[opl::mm::NARGS] = {0x00000000u, K / 16u, 0x10000000u, (N + 15u) / 16u, 0u, fmt, N, K, T,
                                        opl::HANDLE_NONE, 0u, 0u};
        const uint32_t chk = plan::check_op(*c.cfg, opl::K_MUL_MAT, rec, opl::mm::NARGS);
        const uint32_t v[16] = {st, m.kind, m.x_lines, m.qa_lines, m.y_lines, m.lpt, m.t_tile, m.n_tiles,
                                plan::ring_budget(*c.cfg), chk, c.cfg->t_max, c.cfg->k_max, 0u, 0u, 0u, 0u};
        volatile uint32_t * o = spm_ptr<uint32_t>(out + i);
        for (uint32_t k = 0; k < 16u; ++k) o[k] = v[k];
    }
    const ticket st = issue(ST(out, from_handle(a[2]), n), d, c.push_flags);
    c.ring->release(tab, ld);
    c.ring->release(out, st);
    return st;
}

ticket k_race(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    deps w{};
    const spm blk = c.ring->alloc(4u, 2u, &w);
    if (!valid(blk)) {
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const ticket ld = issue(LD(blk, from_handle(a[0]), 2u), fold(d, w), c.push_flags);
    const ticket q = issue(QACT(qa::a32, blk + 2u, blk, 32u, 1u), a[1] ? deps{ld} : fold(d, w), c.push_flags);
    const ticket t = join2(ld, q);
    c.ring->release(blk, t);
    return t;
}

ticket k_vu(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & c) {
    const uint32_t src = a[0], dst = a[1], m = a[2], n = a[3];   // read before the first push
    if (m == 0u || m > 8192u || n == 0u || n > 3u) {
        c.ret = opl::RET_BAD_RECORD;
        return ticket{};
    }
    const uint32_t L = (m + 15u) / 16u;
    deps w{};
    const spm x = c.ring->alloc(2u * L, 2u, &w);
    if (!valid(x)) {
        c.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const spm y = x + L;
    static constexpr uint32_t funcs[3] = {VOP_FUNC_ADD, VOP_FUNC_MIN, VOP_FUNC_MAX};
    ticket t = issue(LD(x, from_handle(src), L), fold(d, w), c.push_flags);
    for (uint32_t k = 0; k < n; ++k) {
        vop_f f{};
        f.func = funcs[k];
        f.d = y.line;
        f.a = k == 0u ? x.line : y.line;
        f.b = x.line;
        f.m = m;
        f.rows = 1u;
        f.d_stride = L;
        f.a_stride = L;
        f.b_stride = L;
        t = issue(VOP(f), deps{t}, c.push_flags);
    }
    t = issue(ST(y, from_handle(dst), L), deps{t}, c.push_flags);
    c.ring->release(x, t);
    return t;
}

}  // namespace

namespace tsr {
ticket exec_test_fastvu(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & ctx);   // exec_fast.cpp
ticket exec_test_sreg(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & ctx);     // exec_fast.cpp
}  // namespace tsr

extern "C" bool tsr_exec_test_kind(uint32_t kind, tsr::kind_entry * out) {
    switch (kind) {
        case TK_COPY:  *out = tsr::kind_entry{&k_copy, 3u};  return true;
        case TK_STUB5: *out = tsr::kind_entry{&k_stub5, 2u}; return true;
        case TK_FAIL:  *out = tsr::kind_entry{&k_fail, 1u};  return true;
        case TK_PRE:   *out = tsr::kind_entry{&k_pre, 3u};   return true;
        case TK_PREST: *out = tsr::kind_entry{&k_prest, 3u}; return true;
        case TK_RET:   *out = tsr::kind_entry{&k_ret, 2u};   return true;
        case TK_PLAN:  *out = tsr::kind_entry{&k_plan, 3u};  return true;
        case TK_RACE:  *out = tsr::kind_entry{&k_race, 2u};  return true;
        case TK_VU:    *out = tsr::kind_entry{&k_vu, 4u};    return true;
        case TK_FASTVU: *out = tsr::kind_entry{&tsr::exec_test_fastvu, 4u}; return true;
        case TK_SREG:  *out = tsr::kind_entry{&tsr::exec_test_sreg, 6u}; return true;
        default:       return false;
    }
}

// k_rope.cpp — the ROPE kind of the graph executor (op list kind 0x16; docs/design/06-software-
// verification.md "SW" §7.2, §7.5; handler interface tsr/exec.hpp, SW §7.3; planning
// tsr/plan_norm.hpp). ggml's ROPE (rope_ext) on n_tokens tokens of n_heads heads of head_dim f32:
// NORM (adjacent pairs) or NEOX (half-split pairs), the recurrence form of ggml-cpu's angles (TS =
// powf(freq_base, -2/n_dims), an optional freq-factor table, freq_scale) or the host-folded A table
// (YaRN, mode bit1 ATAB), attn_factor M (VU §5.4, §6.7).
//
// Tile-op sequence (the numerical contract, SW §7.5 last paragraph; sw/ctrl/ref/rope_ref.cpp runs the
// same tsr_ref_vu ROPE), per token tile of Tt tokens in one block [x | table] (the one-tile block for an
// op's only tile, SW-36; else the ring):
//   1. LD x: the tile's tokens (H = n_heads head_dim / 16 lines each; one LD when the DRAM tokens are
//      dense, src stride = H or Tt = 1, else one per token);
//   2. with a table: LD the freq-factor / A table (ceil(n_dims / 32) lines; per tile, so the tile
//      blocks stay independent in the ring: one tile for every decode op);
//   3. (first tile only) SOP.MOVI s5 = TS (not with ATAB), s4 = M, s6 = freq_scale (only when it is not
//      1.0f, and never with ATAB: FM(1.0f, phi) = phi for every VU angle, so FS off is exact), each
//      only when the launch's s-register shadow does not already hold the value (kind_ctx::
//      sreg_valid, k_fast.hpp sreg_movi: a list's ROPEs share TS / M);
//   4. ROPE in place on x (VU §4.5: same start and token stride, both 0 = H): mode NEOX = mode bit0,
//      FF = a table without ATAB, FS as above, ATAB = mode bit1; positions: the I32 words at the
//      record's preload byte offset (+ 4 t0 for the tile's first token t0, a multiple of 16: the
//      position vector starts on a preload line), read by the VU from the preload area, which the
//      executor loaded and joined before op 0 (SW §7.3 step 1);
//   5. ST x: the tokens to dst (one ST when dense, dst stride = H or Tt = 1, else one per token).
//
// Deps (exec.hpp contract), as RMS_NORM's (k_norm.cpp): d on the first LD (DMA tickets dropped) with
// the block's ring deps; the table LD follows on the DMA queue; the MOVIs carry none (s registers
// are VU state in VU queue order); ROPE depends on the tile's last LD (x, or the table after it:
// after() folds the two DMA tickets to the later); ST on ROPE. The returned ticket is the last ST.
// The positions need no dep: the executor's preload join and the RISC-V's program order put their
// LD before every push of the list.
//
// Push order of a multi-tile op (prefill): LD x_0 [LD t_0], MOVIs, then per tile: ROPE_t, the
// prefetch of tile t+1 (alloc, LD x_{t+1} [LD t_{t+1}]), ST_t (MUL_MAT's double buffering, SW §7.4).
//
// Decode fast path (SW §7.3 "Hot path", SW-35; the k_fast.hpp form): a one-token record (every
// decode ROPE) runs in kind_rope_fast (exec.hpp: the executor's first call) with no call: the argument
// words read in place (all before the first control-window access), plan_norm.hpp's checks for one
// token with the preload check (rope_t1_ok, equal to rope_check + kind_ctx::pre_ok by static_assert),
// kf::alloc_fast (the one-tile block, SW-36), the shadow's answer for the MOVIs, one ISSUE_CREDIT
// read covering the 3-7 pushes (wait_credits: it waits only while the RISC-V is ahead), the ROPE words
// without range checks (rope_fast: the plan bounds every field), no ISSUE_TICKET read for a push whose
// ticket nothing uses (push_nt), and exactly the general path's pushes (fw_kernels_normrope_host
// compares them push for push). Its hot entry (W2b fix) takes a record without a table and with
// freq_scale 1.0 while the s-register shadow holds TS and M, straight-line with the staged SPM line
// reused (kf::push_s); every other one-token record takes rope_fast_any (a tail call).
//
// Measured (Ibex maxperf, 2026-10-31, W2b fix; fw_exec_perf_m7 and fw_kernels_normrope_perf, spin-free
// work per record, executor included): 325-331 RV cycles per Qwen2.5 q / k decode record (0.5B and 32B
// shapes; the W2b target 350 is met); 439-475 through rope_fast_any (freq_scale != 1, Gemma 4's
// rope_freqs table: not met). Before the fix: 420 (the integration), 441 (the package).
#include "k_fast.hpp"
#include "tsr/plan.hpp"

// The op list kinds this source handles (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(ROPE);

namespace tsr {

// The kind handler (exec.hpp kind_fn; registered in graph.cpp's kind table).
ticket kind_rope(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);

#if defined(TSR_FW_HOST)
namespace nr {
uint32_t rope_fast_taken = 0;   // host builds only: records the fast path took (fw_kernels_normrope_host)
}  // namespace nr
#endif

namespace {

namespace r = opl::rope;
namespace pn = plan::norm;

constexpr uint32_t S_M = 4u, S_TS = 5u, S_FS = 6u;   // s registers: M, TS, freq_scale
constexpr uint32_t F32_ONE = 0x3F80'0000u;

using kf::drop_dma;

// Every push with issue()'s credit wait goes through push_g, out of line (the general path: compiled
// once); pushk / push_nt push inline only in the kf::nochk_t form (the fast path; as k_norm.cpp).
using kf::push_g;   // k_fast.hpp: one copy per image
// The general path's ring allocations, out of line (spm_ring::alloc is forced inline; the same calls).
using kf::alloc_g;
TSR_HOT ticket pushk(kf::nochk_t tag, const op & o, deps d, uint32_t pf) { return kf::push(tag, o, d, pf); }
TSR_HOT ticket pushk(kf::chk_t, const op & o, deps d, uint32_t pf) { return push_g(o, d, pf); }

// The fast path's credit rule (as k_norm.cpp's wait_credits): wait until one ISSUE_CREDIT read covers
// every push of the record (at most 3 DMA and 4 VU), then push them all without further reads.
TSR_HOT void wait_credits(uint32_t n_dma, uint32_t n_vu) {
    while (!kf::credits_for(credits(), n_dma, n_vu)) hw::pause();
}

// A push whose ticket nothing uses (as k_norm.cpp's push_nt: in the kf::nochk_t form on the RISC-V,
// hw::push without the ISSUE_TICKET read, CC §5.1; host builds kf::push; kf::chk_t push_g).
template <class C> TSR_HOT void push_nt(C tag, const op & o, deps d, uint32_t pf) {
    if constexpr (C::v) {
        (void)push_g(o, d, pf);
        return;
    }
#if defined(TSR_FW_HOST)
    (void)kf::push(tag, o, d, pf);
#else
    (void)tag;
    volatile uint32_t * w = hw::cw();
    const uint32_t used = o.used;
    w[CW_ISSUE_W(0) / 4u] = o.w[0];
    if (used & 0x02u) w[CW_ISSUE_W(1) / 4u] = o.w[1];
    if (used & 0x04u) w[CW_ISSUE_W(2) / 4u] = o.w[2];
    if (used & 0x08u) w[CW_ISSUE_W(3) / 4u] = o.w[3];
    if (used & 0x10u) w[CW_ISSUE_W(4) / 4u] = o.w[4];
    if (used & 0x20u) w[CW_ISSUE_W(5) / 4u] = o.w[5];
    if (used & 0x40u) w[CW_ISSUE_W(6) / 4u] = o.w[6];
    if (used & 0x80u) w[CW_ISSUE_W(7) / 4u] = o.w[7];
    if ((d.a.v >> 28) != UNIT_NONE) w[CW_ISSUE_DEP0 / 4u] = d.a.v;
    if ((d.b.v >> 28) != UNIT_NONE) w[CW_ISSUE_DEP1 / 4u] = d.b.v;
    w[CW_ISSUE_PUSH / 4u] = (used << ISSUE_PUSH_WMASK_LSB) | (pf & ~ISSUE_PUSH_WMASK);
#endif
}

// ROPE words without the generated builder's run-time range checks (as kf::vop_fast), for fields the
// plan has bounded (lines < 2^24, head_dim / n_dims / n_heads / T / strides < 2^16, mode / flags /
// s registers their enumerations): then the words are exactly vu.hpp's ROPE (static_assert below).
TSR_HOT constexpr op rope_fast(const rope_f & f) {
    op o{};
    o.w[0] = OP_ROPE | (f.mode << ROPE_MODE_LSB) | (f.ff << ROPE_FF_LSB) | (f.fs << ROPE_FS_LSB) |
             (f.atab << ROPE_ATAB_LSB) | (f.sm << ROPE_SM_LSB) | (f.sts << ROPE_STS_LSB) | (f.sfs << ROPE_SFS_LSB);
    o.w[1] = f.d;
    o.w[2] = f.src;
    o.w[3] = f.pos;
    o.w[4] = f.table;
    o.w[5] = f.head_dim | (f.n_dims << ROPE_N_DIMS_LSB);
    o.w[6] = f.n_heads | (f.t << ROPE_T_LSB);
    o.w[7] = f.src_stride | (f.d_stride << ROPE_D_STRIDE_LSB);
    o.used = uint8_t(ROPE_WORDS);
    return o;
}
constexpr bool same_op(const op & x, const op & y) {
    if (x.used != y.used) return false;
    for (uint32_t i = 0; i < 8u; ++i)
        if (x.w[i] != y.w[i]) return false;
    return true;
}
constexpr rope_f rope_sample(uint32_t mode, uint32_t ff, uint32_t fs, uint32_t atab) {
    rope_f f{};
    f.mode = mode;
    f.ff = ff;
    f.fs = fs;
    f.atab = atab;
    f.sm = 15u;
    f.sts = atab ? 0u : 14u;
    f.sfs = fs ? 13u : 0u;
    f.d = 0xFFFFFFu;
    f.src = 0x123456u;
    f.pos = 0xABCDEu;
    f.table = (ff || atab) ? 0xFFFFFEu : 0u;
    f.head_dim = 1024u;
    f.n_dims = 0xFFFFu;
    f.n_heads = 0xFFFFu;
    f.t = 0xFFFFu;
    f.src_stride = 0x8001u;
    f.d_stride = 0xFFFFu;
    return f;
}
static_assert(same_op(rope_fast(rope_sample(1, 0, 0, 0)), ROPE(rope_sample(1, 0, 0, 0))) &&
                  same_op(rope_fast(rope_sample(0, 1, 1, 0)), ROPE(rope_sample(0, 1, 1, 0))) &&
                  same_op(rope_fast(rope_sample(1, 0, 0, 1)), ROPE(rope_sample(1, 0, 0, 1))),
              "k_rope.cpp: rope_fast = vu.hpp ROPE for in-range fields");

// The record's fields the pushes need (read once).
struct rope_args {
    uint32_t dst, dst_stride, src, src_stride, pos, table, ts, m, fs, hd, nd, nh, T, mode;
};

// Pushed words (decision W2b normrope, as k_norm.cpp): op::used holds only the words that can be
// nonzero (ROPE: w7 token strides are 0, w4 table is 0 without one; SOP.MOVI: w3 is 0); the
// hardware sends the others as 0 (CC §5.1 ISSUE_PUSH WMASK): the same op words, one CW store less each.
constexpr uint32_t ROPE_USED = 0x6Fu, ROPE_USED_TAB = 0x10u, SOP_USED_MOVI = 0x07u;
static_assert((ROPE_USED | ROPE_USED_TAB | 0x80u) == ROPE_WORDS && (SOP_USED_MOVI | 0x08u) == SOP_WORDS,
              "k_rope.cpp: the used masks omit only w7 / w4 (ROPE) and w3 (MOVI)");
TSR_HOT op movi_op(uint32_t sd, uint32_t imm) {
    op o = kf::sop_movi(sd, imm);
    o.used = uint8_t(SOP_USED_MOVI);
    return o;
}

// Step 4: one ROPE over Tt tokens at x, positions at line pl, table at line tb. Token strides 0 (= H:
// the tile's tokens are dense in the block; VU §5.4).
TSR_HOT op rope_op(const rope_args & g, uint32_t x, uint32_t pl, uint32_t tb, uint32_t Tt, bool has_tab, bool fs) {
    const bool atab = (g.mode & r::MODE_ATAB) != 0u;
    rope_f f{};
    f.mode = (g.mode & r::MODE_NEOX) ? ROPE_MODE_NEOX : ROPE_MODE_NORM;
    f.ff = (has_tab && !atab) ? 1u : 0u;
    f.fs = fs ? 1u : 0u;
    f.atab = atab ? 1u : 0u;
    f.sm = S_M;
    f.sts = atab ? 0u : S_TS;
    f.sfs = fs ? S_FS : 0u;
    f.d = x;
    f.src = x;
    f.pos = pl;
    f.table = has_tab ? tb : 0u;
    f.head_dim = g.hd;
    f.n_dims = g.nd;
    f.n_heads = g.nh;
    f.t = Tt;
    op o = rope_fast(f);
    o.used = uint8_t(ROPE_USED | (has_tab ? ROPE_USED_TAB : 0u));
    return o;
}

// Step 3 (tag: kf::chk_t / nochk_t, k_fast.hpp): MOVIs through the launch's s-register shadow.
template <class C> TSR_HOT void sreg_set(C tag, kind_ctx & ctx, uint32_t i, uint32_t bits, uint32_t pf) {
    if (kf::sreg_holds(ctx, i, bits)) return;
    push_nt(tag, movi_op(i, bits), deps{}, pf);
    ctx.sreg_val[i] = bits;
    ctx.sreg_valid |= 1u << i;
}
template <class C> TSR_HOT void movis(C tag, kind_ctx & ctx, const rope_args & g, bool fs, uint32_t pf) {
    if (!(g.mode & r::MODE_ATAB)) sreg_set(tag, ctx, S_TS, g.ts, pf);
    sreg_set(tag, ctx, S_M, g.m, pf);
    if (fs) sreg_set(tag, ctx, S_FS, g.fs, pf);
}

// ---------------------------------------------------------------------------------------------
// General path (prefill tiles, strided tokens, refusals): compiled for size.
// ---------------------------------------------------------------------------------------------

// Tokens [t0, t0 + n) between DRAM (stride lines apart) and the block (H lines each): one DMA op
// when dense, else one per token; the first carries `first`; returns the last.
ticket tok_dma(bool store, spm b, uint32_t h, uint32_t stride, uint32_t H, uint32_t t0, uint32_t n, deps first,
               uint32_t pf) {
    const dram base = from_handle(h);
    if (n == 1u || stride == H)
        return push_g(store ? ST(b, base + t0 * stride, n * H) : LD(b, base + t0 * stride, n * H), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < n; ++i) {
        const spm s = b + i * H;
        const dram g = base + (t0 + i) * stride;
        last = push_g(store ? ST(s, g, H) : LD(s, g, H), i == 0u ? first : deps{}, pf);
    }
    return last;
}

struct ttile {
    uint32_t t0, n;
    spm blk;
    ticket ld;      // the tile's last LD
};

deps first_deps(deps dd, deps rd) {
    const deps rr = drop_dma(rd);
    if (rr.a.none()) return dd;
    if (dd.a.none()) return rr;
    return kf::fold_deps(dd, rr);
}

[[gnu::noinline, gnu::cold]] ticket rope_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                 kind_ctx & ctx) {
    // The planner itself (plan_norm.hpp plan_rope, the host's supports_op rule) on a copy of the
    // words, then the positions inside this list's preload area (known only here).
    uint32_t w[r::NARGS];
    for (uint32_t i = 0; i < r::NARGS; ++i) w[i] = a[i];
    kind_plan kp{};
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    if (!plan_rope(w, n_args, ctx.pl, &kp)) return ctx.fail(kp.status);
    rope_args g;
    g.dst = w[r::DST];
    g.dst_stride = w[r::DST_STRIDE];
    g.src = w[r::SRC];
    g.src_stride = w[r::SRC_STRIDE];
    g.pos = w[r::POS];
    g.table = w[r::TABLE];
    g.ts = w[r::TS];
    g.m = w[r::M];
    g.fs = w[r::FREQ_SCALE];
    g.hd = w[r::HEAD_DIM];
    g.nd = w[r::N_DIMS];
    g.nh = w[r::N_HEADS];
    g.T = w[r::N_TOKENS];
    g.mode = w[r::MODE];
    if (!ctx.pre_ok(g.pos, g.T, 2u, opl::PRE_POS_ALIGN - 1u)) return ctx.fail(opl::RET_BAD_RECORD);
    const uint32_t H = kp.row_lines, tab = kp.aux[0], Tt = kp.tile_rows, pf = ctx.push_flags;
    const bool fs = !(g.mode & r::MODE_ATAB) && g.fs != F32_ONE;
    spm_ring & ring = *ctx.ring;
    auto load = [&](ttile & t, deps first) {
        t.ld = tok_dma(false, t.blk, g.src, g.src_stride, H, t.t0, t.n, first, pf);
        if (tab != 0u) t.ld = push_g(LD(t.blk + t.n * H, from_handle(g.table), tab), deps{}, pf);
    };
    auto store = [&](const ttile & t, ticket rt) { return tok_dma(true, t.blk, g.dst, g.dst_stride, H, t.t0, t.n, deps{rt}, pf); };
    ttile tt2[2];
    ttile * cur = &tt2[0], * nxt = &tt2[1];
    cur->t0 = 0u;
    cur->n = Tt;
    // The op's only tile takes the one-tile block when it fits (kind_ctx::one: DMA-bracketed, no ring
    // record, no deps; W2b integration), as the fast path does; the tiles of a multi-tile op the ring.
    const bool one = Tt >= g.T && Tt * H + tab <= ctx.one_lines;
    deps rd{};
    cur->blk = one ? ctx.one : alloc_g(ring, Tt * H + tab, &rd);
    if (!valid(cur->blk)) {
        return ctx.fail(opl::RET_SCRATCH);
    }
    load(*cur, first_deps(drop_dma(d), rd));
    movis(kf::chk_t{}, ctx, g, fs, pf);
    const uint32_t pre_line = ctx.pre_line(g.pos).line;
    ticket last{};
    for (;;) {
        const spm x = cur->blk;
        const ticket rt = push_g(rope_op(g, x.line, pre_line + (cur->t0 >> 4), (x + cur->n * H).line, cur->n, tab != 0u, fs),
                                 deps{cur->ld}, pf);
        const uint32_t t1 = cur->t0 + cur->n;
        if (t1 >= g.T) {
            last = store(*cur, rt);
            if (!one) ring.release(cur->blk, last);
            break;
        }
        nxt->t0 = t1;
        nxt->n = g.T - t1 < Tt ? g.T - t1 : Tt;
        const uint32_t nl = nxt->n * H + tab;
        nxt->blk = alloc_g(ring, nl, &rd);
        bool stored = false;
        if (!valid(nxt->blk)) {                       // no room beside the live block: store it first
            last = store(*cur, rt);
            ring.release(cur->blk, last);
            stored = true;
            nxt->blk = alloc_g(ring, nl, &rd);
            if (!valid(nxt->blk)) {
                ctx.ret = opl::RET_SCRATCH;
                return last;
            }
        }
        load(*nxt, first_deps(deps{}, rd));
        if (!stored) {
            last = store(*cur, rt);
            ring.release(cur->blk, last);
        }
        ttile * t = cur;
        cur = nxt;
        nxt = t;
    }
    return last;
}

// ---------------------------------------------------------------------------------------------
// Decode fast path (one token)
// ---------------------------------------------------------------------------------------------
// The pushes of a one-token record whose block is blk (the one-tile block: no ring deps): w0 / w5 /
// w6 of the ROPE (rope_op's words for T = 1, built by the caller), the position line pl, the table
// handle (HANDLE_NONE: none) and its lines; need: bit0 MOVI TS, bit1 MOVI M, bit2 MOVI freq_scale
// (the s-register shadow's answer, computed once). Pushed in the kf::nochk_t form only, after
// wait_credits (as k_norm.cpp): one push form, inline, no call.
template <class C>
TSR_HOT ticket rope_pushes(C tag, kind_ctx & ctx, uint32_t blk, uint32_t H, uint32_t src, uint32_t dst, uint32_t table,
                           uint32_t tab, uint32_t w0, uint32_t pl, uint32_t w5, uint32_t w6, uint32_t ts, uint32_t m,
                           uint32_t fsv, uint32_t need, deps first) {
    const uint32_t pf = ctx.push_flags, tb = blk + H;
    ticket ld;
    if (tab != 0u) {
        push_nt(tag, LD(spm{blk}, from_handle(src), H), first, pf);
        ld = pushk(tag, LD(spm{tb}, from_handle(table), tab), deps{}, pf);
    } else {
        ld = pushk(tag, LD(spm{blk}, from_handle(src), H), first, pf);
    }
    // Step 3 as kf::sreg_movi with the shadow's answer known (TS, M, freq_scale in movis' order).
    uint32_t valid = ctx.sreg_valid;
    if (need & 1u) {
        push_nt(tag, movi_op(S_TS, ts), deps{}, pf);
        ctx.sreg_val[S_TS] = ts;
        valid |= 1u << S_TS;
    }
    if (need & 2u) {
        push_nt(tag, movi_op(S_M, m), deps{}, pf);
        ctx.sreg_val[S_M] = m;
        valid |= 1u << S_M;
    }
    if (need & 4u) {
        push_nt(tag, movi_op(S_FS, fsv), deps{}, pf);
        ctx.sreg_val[S_FS] = fsv;
        valid |= 1u << S_FS;
    }
    ctx.sreg_valid = valid;
    op o{};
    o.w[0] = w0;
    o.w[1] = blk;
    o.w[2] = blk;
    o.w[3] = pl;
    o.w[4] = tab != 0u ? tb : 0u;
    o.w[5] = w5;
    o.w[6] = w6;
    o.used = uint8_t(ROPE_USED | (tab != 0u ? ROPE_USED_TAB : 0u));
    const ticket rt = pushk(tag, o, deps{ld}, pf);
    return pushk(tag, ST(spm{blk}, from_handle(dst), H), deps{rt}, pf);
}


// plan_norm.hpp's rope_check for n_tokens = 1 (the strides do not matter) and the preload check
// (kind_ctx::pre_ok for one I32 at a line-aligned offset: pos < 64 pre_lines, as pos and the area are
// multiples of 64), in fewer instructions; equal for every argument (static_assert below).
TSR_HOT constexpr bool rope_t1_ok(uint32_t mode, bool has_tab, uint32_t hd, uint32_t nd, uint32_t nh, uint32_t pos,
                                  uint32_t pre_bytes) {
    return mode <= r::MODE_KNOWN && hd - 16u <= 1024u - 16u && (hd & 15u) == 0u && nd - 1u < hd &&
           (nd & ((mode & r::MODE_NEOX) ? 31u : 15u)) == 0u && (has_tab || !(mode & r::MODE_ATAB)) && nh - 1u < 0xFFFFu &&
           (pos & 63u) == 0u && pos < pre_bytes;
}
constexpr bool rope_t1_ok_matches() {
    const uint32_t hds[] = { 0u, 8u, 16u, 24u, 64u, 512u, 1008u, 1024u, 1040u, 0xFFFFFFF0u };
    const uint32_t nds[] = { 0u, 1u, 16u, 32u, 48u, 64u, 512u, 1024u, 1040u };
    const uint32_t nhs[] = { 0u, 1u, 14u, 0xFFFFu, 0x10000u };
    const uint32_t poss[] = { 0u, 4u, 64u, 128u, 16320u, 16384u, 0xFFFFFFC0u };
    const uint32_t pls[] = { 0u, 2u, 256u };
    for (uint32_t mode = 0; mode < 6u; ++mode)
        for (uint32_t tab = 0; tab < 2u; ++tab)
            for (uint32_t hd : hds)
                for (uint32_t nd : nds)
                    for (uint32_t nh : nhs)
                        for (uint32_t pos : poss)
                            for (uint32_t pl : pls) {
                                const bool ref = pn::rope_check(mode, tab != 0u, hd, nd, nh, 1u, 0u, pos) == opl::RET_OK &&
                                                 (pos & 63u) == 0u && pos <= (pl << 6) && 1u <= (((pl << 6) - pos) >> 2);
                                if (rope_t1_ok(mode, tab != 0u, hd, nd, nh, pos, pl << 6) != ref) return false;
                            }
    return true;
}
static_assert(rope_t1_ok_matches(), "k_rope.cpp: rope_t1_ok = rope_check (one token) + pre_ok");

// The decode fast path for any one-token record (a table, ATAB, freq_scale, MOVIs): the hot entry's
// tail call (kind_rope_fast below), else KIND_NOT_FAST with nothing changed. The argument words are
// read in two groups, both before the first control-window access (the ISSUE_CREDIT read; any
// control-window access invalidates the SPM line buffer): the fields the checks need, then, once the
// block is placed, the rest. The executor checked n_args.
[[gnu::noinline]] uint32_t rope_fast_any(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t T = a[r::N_TOKENS], mode = a[r::MODE], hd = a[r::HEAD_DIM], nd = a[r::N_DIMS], nh = a[r::N_HEADS];
    const uint32_t pos = a[r::POS], table = a[r::TABLE];
    const bool has_tab = table != opl::HANDLE_NONE;
    if (kflags == 0u && T == 1u && (ctx.pl.caps & plan::CAPS_VU) != 0u &&
        rope_t1_ok(mode, has_tab, hd, nd, nh, pos, ctx.pre_lines << 6)) [[likely]] {
        const uint32_t H = (nh * hd) >> 4, tab = has_tab ? plan::vu_lines(nd >> 1) : 0u;
        kf::block b;
        if (H + tab <= ctx.pl.budget && kf::alloc_fast(ctx, H + tab, &b)) [[likely]] {
            // Committed: blk = [x | table] in the one-tile block (no ring deps).
            const uint32_t dst = a[r::DST], src = a[r::SRC], ts = a[r::TS], m = a[r::M], fsv = a[r::FREQ_SCALE];
            const deps first = kf::first_deps(d);
            const bool atab = (mode & r::MODE_ATAB) != 0u, fs = !atab && fsv != F32_ONE;
            // The ROPE's words for T = 1 (rope_op's; rope_fast's layout), strides 0.
            const uint32_t w0 = OP_ROPE | ((mode & r::MODE_NEOX) << ROPE_MODE_LSB) | ((has_tab && !atab) ? ROPE_FF : 0u) |
                                (fs ? (ROPE_FS | (S_FS << ROPE_SFS_LSB)) : 0u) | (atab ? ROPE_ATAB : (S_TS << ROPE_STS_LSB)) |
                                (S_M << ROPE_SM_LSB);
            const uint32_t need = (atab ? 0u : kf::sreg_need(ctx, S_TS, ts)) | (kf::sreg_need(ctx, S_M, m) << 1) |
                                  (fs ? (kf::sreg_need(ctx, S_FS, fsv) << 2) : 0u);
            const uint32_t n_vu = 1u + (need & 1u) + ((need >> 1) & 1u) + (need >> 2);
            const uint32_t pl = ctx.pre_line(pos).line, w5 = hd | (nd << ROPE_N_DIMS_LSB), w6 = nh | (1u << ROPE_T_LSB);
            wait_credits(has_tab ? 3u : 2u, n_vu);
            const ticket st = rope_pushes(kf::nochk_t{}, ctx, b.at.line, H, src, dst, table, tab, w0, pl, w5, w6, ts, m, fsv,
                                          need, first);
            kf::release(b, st);
#if defined(TSR_FW_HOST)
            ++nr::rope_fast_taken;
#endif
            return st.v;
        }
    }
    return kf::NOT_FAST;
}

}  // namespace

// The decode fast path (header comment; exec.hpp kind_rope_fast: the executor's first call, a leaf in
// its common path). The hot entry (W2b fix): one token without a table (hence without ATAB) and with
// freq_scale 1.0 (no FS) while the launch's s-register shadow holds TS (s5) and M (s4) (every decode
// ROPE of a Qwen2.5 layer and of a Gemma 4 SWA layer after the list's first), straight-line: the
// checks of rope_fast_any for that case (rope_t1_ok), LD, ROPE, ST with the staged SPM line reused
// (the ROPE's d = the LD's line, w1; the ST's line = the ROPE's d, w1). Any other record:
// rope_fast_any, a tail call, which pushes the same ops for these records (fw_kernels_normrope_host
// compares both with the general path).
[[gnu::noinline]] uint32_t kind_rope_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t T = a[r::N_TOKENS], mode = a[r::MODE], hd = a[r::HEAD_DIM], nd = a[r::N_DIMS], nh = a[r::N_HEADS];
    const uint32_t pos = a[r::POS], table = a[r::TABLE], fsv = a[r::FREQ_SCALE], sv = ctx.sreg_valid;
    if (T == 1u && table == opl::HANDLE_NONE && fsv == F32_ONE && kflags == 0u && (ctx.pl.caps & plan::CAPS_VU) != 0u &&
        rope_t1_ok(mode, false, hd, nd, nh, pos, ctx.pre_lines << 6) && ((sv >> S_M) & 3u) == 3u) [[likely]] {
        const uint32_t ts = a[r::TS], m = a[r::M], H = (nh * hd) >> 4;
        if (ctx.sreg_val[S_TS] == ts && ctx.sreg_val[S_M] == m && H <= ctx.one_lines && H <= ctx.pl.budget) [[likely]] {
            // Committed: x in the one-tile block (no ring deps), no MOVI (rope_t1_ok without a table:
            // mode has no ATAB, so the ROPE reads TS from s5).
            const uint32_t dst = a[r::DST], src = a[r::SRC];
            const deps first = kf::first_deps(d);
            const uint32_t blk = ctx.one.line, pf = ctx.push_flags;
            op o{};
            o.w[0] = OP_ROPE | ((mode & r::MODE_NEOX) << ROPE_MODE_LSB) | (S_TS << ROPE_STS_LSB) | (S_M << ROPE_SM_LSB);
            o.w[1] = blk;
            o.w[2] = blk;
            o.w[3] = ctx.pre_line(pos).line;
            o.w[5] = hd | (nd << ROPE_N_DIMS_LSB);
            o.w[6] = nh | (1u << ROPE_T_LSB);
            o.used = uint8_t(ROPE_USED);
            wait_credits(2u, 1u);
            kf::stage_reset();
            const ticket ld = kf::push_s(LD(spm{blk}, from_handle(src), H), first, pf);
            const ticket rt = kf::push_s<kf::W1>(o, deps{ld}, pf);
            const ticket st = kf::push_s<kf::W1>(ST(spm{blk}, from_handle(dst), H), deps{rt}, pf);
#if defined(TSR_FW_HOST)
            ++nr::rope_fast_taken;
#endif
            return st.v;
        }
    }
    return rope_fast_any(a, kflags, d, ctx);
}

// The handler: the fast path, else the general path.
ticket kind_rope(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    if (n_args == r::NARGS) {
        const uint32_t t = kind_rope_fast(a, kflags, d, ctx);
        if (t != kf::NOT_FAST) return ticket{t};
    }
    return rope_general(a, n_args, kflags, d, ctx);
}

// The fast path's ROPE words are rope_op's for one token (the general path's; checked here).
static_assert(rope_fast(rope_f{ROPE_MODE_NEOX, 1u, 1u, 0u, S_M, S_TS, S_FS, 100u, 100u, 7u, 120u, 512u, 512u, 16u, 1u, 0u, 0u})
                      .w[0] == (OP_ROPE | (ROPE_MODE_NEOX << ROPE_MODE_LSB) | ROPE_FF | ROPE_FS | (S_M << ROPE_SM_LSB) |
                                (S_TS << ROPE_STS_LSB) | (S_FS << ROPE_SFS_LSB)) &&
                  rope_fast(rope_f{ROPE_MODE_NORM, 0u, 0u, 1u, S_M, 0u, 0u, 100u, 100u, 7u, 120u, 64u, 16u, 3u, 1u, 0u, 0u})
                          .w[0] == (OP_ROPE | ROPE_ATAB | (S_M << ROPE_SM_LSB)),
              "k_rope.cpp: the fast path's ROPE w0 = rope_op's");

#if defined(TSR_FW_HOST)
namespace nr {
// Host builds only (fw_kernels_normrope_host): the general path alone, which the test compares push
// for push with the fast path on the same record and state.
ticket rope_slow(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    return rope_general(a, n_args, kflags, d, ctx);
}
}  // namespace nr
#endif

}  // namespace tsr

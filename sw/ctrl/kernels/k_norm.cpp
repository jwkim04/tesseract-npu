// k_norm.cpp — the RMS_NORM kind of the graph executor (op list kind 0x14; docs/design/06-software-
// verification.md "SW" §7.2, §7.5; handler interface tsr/exec.hpp, SW §7.3; planning
// tsr/plan_norm.hpp). y_r = x_r * rms(x_r) [* w_(r mod p)] for `rows` rows of n f32: RMS_NORM and the
// fused RMS_NORM + MUL(weight) of ggml, including the per-head Q / K / V norms (rows = n_head T,
// n = head_dim, VU §8.1; Gemma 4, AT §7.5).
//
// Tile-op sequence (the numerical contract, SW §7.5 last paragraph; sw/ctrl/ref/rmsnorm_ref.cpp
// composes the same tsr_ref_vu calls), per row tile of R rows in one block [x | w | v] (the one-tile block
// for an op's only tile, SW-36; else the ring):
//   1. LD x: the tile's rows into x (L = ceil(n/16) lines each; one LD when the DRAM rows are dense,
//      src stride = L or R = 1, else one per row);
//   2. with a weight: LD w: one row (period 1) or the tile's R weight rows (period p > 1);
//   3. SOP.MOVI s2 = inv_n, SOP.MOVI s3 = eps (the host's f32 bits, SW-8), each only when the
//      launch's s-register shadow does not already hold the value (kind_ctx::sreg_valid,
//      k_fast.hpp sreg_movi: a list's norms share them);
//   4. RED.SUMSQ post RMS, sk = s2, se = s3, over the R rows of x: TO_S s1 when R = 1, else the
//      per-row vector v (VU §6.2: y_r = RC(SQ(FA(FM(sumsq_r, inv_n), eps))));
//   5. VOP in place on x (VU §4.5): MULCB (y = FM(FM(x, c_r), w), b = w with stride 0 for period 1,
//      L for p > 1) with a weight, MULC (y = FM(x, c_r)) without; c = s1 (cmode 1) when R = 1, else v
//      (cmode 2); the tail words of each row's last line become 0 (VU §4.2);
//   6. ST x: the rows to dst (one ST when dense, dst stride = L or R = 1, else one per row).
// The 2-op form (4-5) is bit-identical to the spec's 7-op lowering docs/spec.md:311-321 (VU §6.2,
// §8.1; fw_kernels_normrope_host checks it).
//
// Deps (exec.hpp contract). The op's predecessors d go on the first LD (DMA tickets dropped: queue
// order implies them), with the block's ring deps (releases of older blocks: STs, DMA); the weight
// LD follows on the DMA queue. The MOVIs carry none: they write s registers, which only VU ops read
// or write, and every earlier VU op is before them in the VU queue (VU §4.4, §4.5). RED depends on
// the tile's last LD (the weight, or the last x row: the DMA queue order puts every LD of the tile
// before it), so the VOP, after RED in the VU queue, needs no dep; the first ST depends on the VOP.
// The returned ticket is the last ST: it implies every op of the op (I1).
//
// Push order of a multi-tile op (prefill; plan_norm.hpp's tiles, as MUL_MAT's double buffering, SW
// §7.4): LD x_0 [LD w_0], MOVIs, then per tile t: RED_t, VOP_t, the prefetch of tile t+1 (alloc, LD
// x_{t+1} [LD w_{t+1}]), ST_t. On the DMA queue the LD of tile t+1 precedes ST_t and runs while the
// VU works on tile t; when the ring cannot place tile t+1 beside tile t, ST_t is pushed and its block
// released first. A tile never crosses a weight period block (its weight rows are contiguous).
//
// Decode fast path (SW §7.3 "Hot path", SW-35; the k_fast.hpp form): a record whose rows fit one
// tile with dense rows and weight period 1 (or none) — every decode RMS_NORM, per-head norms
// included — runs in kind_rms_norm_fast (exec.hpp: the executor's first call) with no call: the
// argument words read in place (all before the first control-window access), the field checks of
// plan_norm.hpp (no divide), kf::alloc_fast (the one-tile block, SW-36), the shadow's answer for
// the MOVIs, one ISSUE_CREDIT read covering the 4-7 pushes (wait_credits: it waits only while the
// RISC-V is ahead of the units), the VU words without range checks (kf::red_fast / vop_fast: the plan
// bounds every field), no ISSUE_TICKET read for a push whose ticket nothing uses, the staged words
// reused (kf::push_s, CC §5.1), and exactly the pushes, words and deps of the general path
// (fw_kernels_normrope_host compares them push for push). Its hot entry (W2b fix) takes one row with
// a weight while the s-register shadow holds 1/n and eps, straight-line; every other one-tile record
// takes rms_fast_any (a tail call). Anything else (a refused record included) goes to the general
// path, out of line and compiled for size (its pushes through push_g).
//
// Measured (Ibex maxperf, 2026-10-31, W2b fix; fw_exec_perf_m7 and fw_kernels_normrope_perf, spin-free
// work per record, executor included): 319-326 RV cycles per Qwen2.5 decode record (n 896, weight;
// the W2b target 350 is met), 347 at 32B's n 5,120 (unit-bound); the per-head norms of Gemma 4 (16 /
// 8 rows, rms_fast_any) 405-455 (not met). Before the fix: 395 (the integration), 445 (the package).
#include "k_fast.hpp"
#include "tsr/plan.hpp"

// The op list kinds this source handles (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(RMS_NORM);

namespace tsr {

#if defined(TSR_FW_HOST)
namespace nr {
uint32_t rms_fast_taken = 0;   // host builds only: records the fast path took (fw_kernels_normrope_host)
}  // namespace nr
#endif

namespace {

namespace r = opl::rms;
namespace pn = plan::norm;

constexpr uint32_t S_RES = 1u, S_INVN = 2u, S_EPS = 3u;   // s registers: RED TO_S result, sk, se

// Every push with issue()'s credit wait goes through push_g, out of line (the general path: compiled
// once); pushk / push_nt push inline only in the kf::nochk_t form (one credit read covered the
// record's pushes: the fast path).
using kf::push_g;   // k_fast.hpp: one copy per image
// The general path's ring allocations, out of line (spm_ring::alloc is forced inline; the same calls).
using kf::alloc_g;
TSR_HOT ticket pushk(kf::nochk_t tag, const op & o, deps d, uint32_t pf) { return kf::push(tag, o, d, pf); }
TSR_HOT ticket pushk(kf::chk_t, const op & o, deps d, uint32_t pf) { return push_g(o, d, pf); }

// The fast path's credit rule (decision W2b normrope): wait until one ISSUE_CREDIT read covers every
// push of the record (kf::credits_for: n_dma DMA and n_vu VU pushes and the issue FIFO), then push
// them all without further reads (kf::nochk_t). The counts are at most 3 DMA and 4 VU pushes, within
// every queue's depth and the FIFO's, and every op already queued can complete without a later
// push, so the wait ends (CC §7.3: no push waits in hardware, PERF PUSH_STALL 0). It only waits while
// the RISC-V is ahead of the units; the value ends a wait loop and decides nothing pushed (SW §4.7).
TSR_HOT void wait_credits(uint32_t n_dma, uint32_t n_vu) {
    while (!kf::credits_for(credits(), n_dma, n_vu)) hw::pause();
}

// A push whose ticket nothing uses: in the kf::nochk_t form on the RISC-V, hw::push without the
// ISSUE_TICKET read (the same staging, deps and ISSUE_PUSH value; CC §5.1: ISSUE_TICKET only holds the
// last push's ticket, reading it is optional);
// host builds push through kf::push (the simulator hook returns the ticket); kf::chk_t: push_g.
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

// Pushed words (decision W2b normrope): op::used holds only the words that can be nonzero in the
// form pushed (RED TO_S: w1 d and w7 stride are 0; VOP with one row: w4 c and w7 strides are 0, w3 b
// without a weight; SOP.MOVI: w3 addr is 0). The hardware sends the unstaged words as 0 (CC §5.1:
// WMASK), so the op words are exactly vu.hpp's builders'; only the ISSUE_PUSH WMASK bits differ from
// the <OP>_WORDS convention (both paths use these functions; one CW store less per omitted word).
constexpr uint32_t RED_USED_TO_S = 0x65u, RED_USED_VEC = 0xE7u;            // w0 w2 w5 w6 (+ w1 w7)
constexpr uint32_t VOP_USED_ONE = 0x67u, VOP_USED_VEC = 0xF7u, VOP_USED_B = 0x08u;   // + w3 with a weight
constexpr uint32_t SOP_USED_MOVI = 0x07u;                                  // w0 w1 w2

// SOP.MOVI s[sd] = imm (kf::sop_movi's words).
TSR_HOT op movi_op(uint32_t sd, uint32_t imm) {
    op o = kf::sop_movi(sd, imm);
    o.used = uint8_t(SOP_USED_MOVI);
    return o;
}
// kf::sreg_movi with movi_op: SOP.MOVI only when the launch's shadow does not hold the value.
template <class C> TSR_HOT void sreg_set(C tag, kind_ctx & ctx, uint32_t i, uint32_t bits, uint32_t pf);

// The RED's deps: the tile's last LD. The negative-test image fw_knorm_mut_nodep (TSR_KNORM_MUTANT 1,
// sw/ctrl/tests/normrope.cmake) drops it: the race detector must report the RED reading x unordered
// with the LD that writes it.
TSR_HOT deps red_deps(ticket ld) {
#if defined(TSR_KNORM_MUTANT) && TSR_KNORM_MUTANT == 1
    (void)ld;
    return deps{};
#else
    return deps{ld};
#endif
}

// Step 4: RED over R rows of x (L lines apart); TO_S s1 for one row, else the vector at line v.
TSR_HOT op red_op(uint32_t x, uint32_t v, uint32_t n, uint32_t R, uint32_t L) {
    red_f f{};
    f.kind = RED_KIND_SUMSQ;
    f.post = RED_POST_RMS;
    f.sk = S_INVN;
    f.se = S_EPS;
    f.a = x;
    f.m = n;
    f.rows = R;
    if (R == 1u) {
        f.to_s = 1u;
        f.sd = S_RES;
    } else {
        f.d = v;
        f.a_stride = L;
    }
    op o = kf::red_fast(f);
    o.used = uint8_t(R == 1u ? RED_USED_TO_S : RED_USED_VEC);
    return o;
}

// Step 5: the VOP in place on x. w = the weight block's line (has_w), w_stride its row stride (0
// for period 1).
TSR_HOT op vop_op(uint32_t x, bool has_w, uint32_t w, uint32_t w_stride, uint32_t v, uint32_t n, uint32_t R, uint32_t L) {
    vop_f f{};
    f.func = has_w ? VOP_FUNC_MULCB : VOP_FUNC_MULC;
    f.d = x;
    f.a = x;
    f.b = has_w ? w : 0u;
    f.m = n;
    f.rows = R;
    if (R == 1u) {
        f.cmode = VOP_CMODE_SREG;
        f.si = S_RES;
    } else {
        f.cmode = VOP_CMODE_VEC;
        f.c = v;
        f.d_stride = L;
        f.a_stride = L;
        f.b_stride = has_w ? w_stride : 0u;
    }
    op o = kf::vop_fast(f);
    o.used = uint8_t((R == 1u ? VOP_USED_ONE : VOP_USED_VEC) | (has_w ? VOP_USED_B : 0u));
    return o;
}

template <class C> TSR_HOT void sreg_set(C tag, kind_ctx & ctx, uint32_t i, uint32_t bits, uint32_t pf) {
    if (kf::sreg_holds(ctx, i, bits)) return;
    push_nt(tag, movi_op(i, bits), deps{}, pf);
    ctx.sreg_val[i] = bits;
    ctx.sreg_valid |= 1u << i;
}

// The omitted words are 0 in every form pushed (vu.hpp's builders on sample fields).
static_assert(RED({.kind = RED_KIND_SUMSQ, .to_s = 1u, .sd = S_RES, .post = RED_POST_RMS, .sk = S_INVN, .se = S_EPS,
                   .a = 77u, .m = 896u, .rows = 1u}).w[1] == 0u &&
                  RED({.kind = RED_KIND_SUMSQ, .to_s = 1u, .sd = S_RES, .post = RED_POST_RMS, .sk = S_INVN, .se = S_EPS,
                       .a = 77u, .m = 896u, .rows = 1u}).w[7] == 0u &&
                  (RED_WORDS & ~RED_USED_VEC) == 0u && (RED_USED_TO_S & ~RED_WORDS) == 0u &&
                  (VOP_USED_VEC | VOP_USED_B) == VOP_WORDS && (SOP_USED_MOVI | 0x08u) == SOP_WORDS,
              "k_norm.cpp: the used masks omit only words that are 0");

// Steps 3-5 of a tile (the RED's dep: the tile's last LD). Returns the VOP.
template <class C>
TSR_HOT ticket norm_ops(C tag, kind_ctx & ctx, uint32_t x, bool has_w, uint32_t w, uint32_t w_stride, uint32_t v,
                        uint32_t n, uint32_t R, uint32_t L, ticket ld, bool movis, uint32_t inv_n, uint32_t eps,
                        uint32_t pf) {
    if (movis) {
        sreg_set(tag, ctx, S_INVN, inv_n, pf);
        sreg_set(tag, ctx, S_EPS, eps, pf);
    }
    push_nt(tag, red_op(x, v, n, R, L), red_deps(ld), pf);
    if (R == 1u) kf::sreg_clobber(ctx, S_RES);          // RED TO_S writes s1 (exec.hpp's shadow rule)
    return pushk(tag, vop_op(x, has_w, w, w_stride, v, n, R, L), deps{}, pf);
}

// ---------------------------------------------------------------------------------------------
// General path (prefill tiles, strided rows, weight periods > 1, refusals): compiled for size.
// ---------------------------------------------------------------------------------------------

struct rms_op {
    uint32_t dst, dst_stride, src, src_stride, n, rows, eps, inv_n, w, period;
    bool has_w;
    uint32_t L;
    uint32_t pf;
};

struct rtile {
    uint32_t r0, nr, j0;       // first row, rows, first weight row (period > 1; else 0)
    spm blk;
    ticket ld;                 // the tile's last LD
};

// Rows [r0, r0 + nr) of a row tensor (stride lines apart) to / from the block's rows (L lines each):
// one DMA op when dense, else one per row; the first carries `first`; returns the last.
ticket rows_dma(bool store, spm b, uint32_t h, uint32_t stride, uint32_t L, uint32_t r0, uint32_t nr, deps first,
                uint32_t pf) {
    const dram base = from_handle(h);
    if (nr == 1u || stride == L)
        return push_g(store ? ST(b, base + r0 * stride, nr * L) : LD(b, base + r0 * stride, nr * L), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < nr; ++i) {
        const spm s = b + i * L;
        const dram g = base + (r0 + i) * stride;
        last = push_g(store ? ST(s, g, L) : LD(s, g, L), i == 0u ? first : deps{}, pf);
    }
    return last;
}

uint32_t w_lines_of(const rms_op & m, uint32_t nr) { return !m.has_w ? 0u : m.period == 1u ? m.L : nr * m.L; }
spm w_of(const rms_op & m, const rtile & t) { return t.blk + t.nr * m.L; }
spm v_of(const rms_op & m, const rtile & t) { return w_of(m, t) + w_lines_of(m, t.nr); }

// Steps 1-2 of a tile whose block is allocated.
void load_tile(const rms_op & m, rtile & t, deps first) {
    t.ld = rows_dma(false, t.blk, m.src, m.src_stride, m.L, t.r0, t.nr, first, m.pf);
    if (m.has_w) t.ld = push_g(LD(w_of(m, t), from_handle(m.w) + t.j0 * m.L, w_lines_of(m, t.nr)), deps{}, m.pf);
}

// Steps 3-5 (the MOVIs with the first tile); returns the VOP.
[[gnu::noinline]] ticket compute_tile(const rms_op & m, const rtile & t, kind_ctx & ctx, bool first) {
    return norm_ops(kf::chk_t{}, ctx, t.blk.line, m.has_w, w_of(m, t).line, m.period == 1u ? 0u : m.L, v_of(m, t).line,
                    m.n, t.nr, m.L, t.ld, first, m.inv_n, m.eps, m.pf);
}

// Both the op's deps and the ring deps without DMA tickets: folded to <= 2 (SYNC NOPs if needed).
deps first_deps(deps dd, deps rd) {
    const deps rr = kf::drop_dma(rd);
    if (rr.a.none()) return dd;
    if (dd.a.none()) return rr;
    return kf::fold_deps(dd, rr);
}

[[gnu::noinline, gnu::cold]] ticket rms_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                kind_ctx & ctx) {
    // The planner itself (plan_norm.hpp plan_rms_norm, the host's supports_op rule) on a copy of
    // the words: the general path refuses exactly what the host's planner refuses.
    uint32_t w[r::NARGS];
    for (uint32_t i = 0; i < r::NARGS; ++i) w[i] = a[i];
    kind_plan kp{};
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    if (!plan_rms_norm(w, n_args, ctx.pl, &kp)) return ctx.fail(kp.status);
    rms_op m;
    m.dst = w[r::DST];
    m.dst_stride = w[r::DST_STRIDE];
    m.src = w[r::SRC];
    m.src_stride = w[r::SRC_STRIDE];
    m.n = w[r::N];
    m.rows = w[r::ROWS];
    m.eps = w[r::EPS];
    m.inv_n = w[r::INV_N];
    m.w = w[r::W];
    m.has_w = m.w != opl::HANDLE_NONE;
    m.period = m.has_w ? w[r::W_PERIOD] : 1u;
    m.pf = ctx.push_flags;
    m.L = kp.row_lines;
    const uint32_t R = kp.tile_rows, p = m.period;
    spm_ring & ring = *ctx.ring;
    // The next tile after rows [r0, r0 + nr) (weight rows from j0): the rest of the period block,
    // at most R rows (period 1: no blocks).
    auto next = [&](uint32_t r0, uint32_t j0, rtile & t) {
        t.r0 = r0;
        t.j0 = (p > 1u && j0 == p) ? 0u : j0;
        uint32_t nr = m.rows - r0 < R ? m.rows - r0 : R;
        if (p > 1u && nr > p - t.j0) nr = p - t.j0;
        t.nr = nr;
    };
    rtile tl2[2];
    rtile * cur = &tl2[0], * nxt = &tl2[1];
    next(0u, 0u, *cur);
    // The op's only tile takes the one-tile block when it fits (kind_ctx::one: DMA-bracketed, no ring
    // record, no deps; W2b integration), as the fast path does; the tiles of a multi-tile op the ring.
    const uint32_t l0 = uint32_t(pn::rms_block_lines(m.L, cur->nr, m.has_w, p));
    const bool one = cur->nr == m.rows && l0 <= ctx.one_lines;
    deps rd{};
    cur->blk = one ? ctx.one : alloc_g(ring, l0, &rd);
    if (!valid(cur->blk)) return ctx.fail(opl::RET_SCRATCH);
    load_tile(m, *cur, first_deps(kf::drop_dma(d), rd));
    bool first = true;
    ticket last{};
    for (;;) {
        const ticket vt = compute_tile(m, *cur, ctx, first);
        first = false;
        const uint32_t r1 = cur->r0 + cur->nr;
        if (r1 >= m.rows) {
            last = rows_dma(true, cur->blk, m.dst, m.dst_stride, m.L, cur->r0, cur->nr, deps{vt}, m.pf);
            if (!one) ring.release(cur->blk, last);
            break;
        }
        next(r1, p > 1u ? cur->j0 + cur->nr : 0u, *nxt);
        const uint32_t nl = uint32_t(pn::rms_block_lines(m.L, nxt->nr, m.has_w, p));
        nxt->blk = alloc_g(ring, nl, &rd);
        bool stored = false;
        if (!valid(nxt->blk)) {                       // no room beside the live block: store it first
            last = rows_dma(true, cur->blk, m.dst, m.dst_stride, m.L, cur->r0, cur->nr, deps{vt}, m.pf);
            ring.release(cur->blk, last);
            stored = true;
            nxt->blk = alloc_g(ring, nl, &rd);
            if (!valid(nxt->blk)) {                   // cannot happen: one tile fits the budget
                ctx.ret = opl::RET_SCRATCH;
                return last;
            }
        }
        load_tile(m, *nxt, first_deps(deps{}, rd));
        if (!stored) {
            last = rows_dma(true, cur->blk, m.dst, m.dst_stride, m.L, cur->r0, cur->nr, deps{vt}, m.pf);
            ring.release(cur->blk, last);
        }
        rtile * t = cur;
        cur = nxt;
        nxt = t;
    }
    return last;
}

// ---------------------------------------------------------------------------------------------
// Decode fast path
// ---------------------------------------------------------------------------------------------
// The pushes of a one-tile record whose block is blk (dense rows; the one-tile block: no ring deps).
// need: bit0 MOVI inv_n, bit1 MOVI eps (the s-register shadow's answer, computed once). Pushed in the
// nochk_t form only (kf::push_s): the caller first waits until one ISSUE_CREDIT read covers all of
// them (wait_credits), so the fast path has one push form, inline, and makes no call. Staged words
// reused (kf::push_s, CC §5.1): the VOP follows the RED on the same rows, so its a line (w2 = x), m
// (w5 = n) and, for one row, w6 (rows 1, strides 0), for several rows w7 (a stride L, b stride 0 =
// the RED's a stride L) are the RED's; the ST follows the VOP, whose d line (w1 = x) is the ST's SPM
// line.
TSR_HOT ticket rms_pushes(kind_ctx & ctx, uint32_t blk, uint32_t src, uint32_t dst, uint32_t wh, uint32_t n,
                          uint32_t rows, uint32_t L, uint32_t inv_n, uint32_t eps, uint32_t need, deps first) {
    const uint32_t pf = ctx.push_flags, xl = rows * L;
    const bool has_w = wh != opl::HANDLE_NONE;
    const uint32_t wb = blk + xl, vb = wb + (has_w ? L : 0u);
    kf::stage_reset();
    ticket ld;
    if (has_w) [[likely]] {
        kf::push_s<0u, false>(LD(spm{blk}, from_handle(src), xl), first, pf);
        ld = kf::push_s(LD(spm{wb}, from_handle(wh), L), deps{}, pf);
    } else {
        ld = kf::push_s(LD(spm{blk}, from_handle(src), xl), first, pf);
    }
    // Step 3 as sreg_set with the shadow's answer known (the caller updates sreg_valid).
    if (need & 1u) [[unlikely]] {
        kf::push_s<0u, false>(movi_op(S_INVN, inv_n), deps{}, pf);
        ctx.sreg_val[S_INVN] = inv_n;
    }
    if (need & 2u) [[unlikely]] {
        kf::push_s<0u, false>(movi_op(S_EPS, eps), deps{}, pf);
        ctx.sreg_val[S_EPS] = eps;
    }
    kf::push_s<0u, false>(red_op(blk, vb, n, rows, L), red_deps(ld), pf);
    const op v = vop_op(blk, has_w, wb, 0u, vb, n, rows, L);
    const ticket vt = rows == 1u ? kf::push_s<kf::W2 | kf::W5 | kf::W6>(v, deps{}, pf)
                                 : kf::push_s<kf::W2 | kf::W5 | kf::W7>(v, deps{}, pf);
    return kf::push_s<kf::W1>(ST(spm{blk}, from_handle(dst), xl), deps{vt}, pf);
}

// The decode fast path for any one-tile record (several rows, no weight, MOVIs): the hot entry's tail
// call (kind_rms_norm_fast below), else KIND_NOT_FAST with nothing changed. The argument words are
// read in two groups, both before the first control-window access (the ISSUE_CREDIT read; any
// control-window access invalidates the SPM line buffer): the fields the checks need, then, once the
// block is placed, the rest (fewer values live at once).
[[gnu::noinline]] uint32_t rms_fast_any(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t n = a[r::N], rows = a[r::ROWS], wh = a[r::W];
    const bool has_w = wh != opl::HANDLE_NONE;
    const uint32_t L = plan::vu_lines(n);
    // One tile (plan_norm.hpp rms_check + rms_one_tile for weight period 1), dense rows. The sum
    // cannot overflow: rows, L <= 65,535 when rows > 1, and L < 2^20 for one row.
    if (kflags == 0u && (ctx.pl.caps & plan::CAPS_VU) != 0u && n - 1u < plan::VU_M_MAX &&
        rows - 1u < plan::VU_ROWS_MAX && (!has_w || a[r::W_PERIOD] == 1u) &&
        (rows == 1u || (a[r::SRC_STRIDE] == L && a[r::DST_STRIDE] == L && L <= plan::VU_STRIDE_MAX))) [[likely]] {
        const uint32_t lines = rows * L + (has_w ? L : 0u) + (rows > 1u ? plan::vu_lines(rows) : 0u);
        kf::block b;
        if (lines <= ctx.pl.budget && kf::alloc_fast(ctx, lines, &b)) [[likely]] {
            // Committed: blk = [x rows | w | v] in the one-tile block (no ring deps).
            const uint32_t dst = a[r::DST], src = a[r::SRC], eps = a[r::EPS], inv_n = a[r::INV_N];
            const deps first = kf::first_deps(d);
            const uint32_t need = kf::sreg_need(ctx, S_INVN, inv_n) | (kf::sreg_need(ctx, S_EPS, eps) << 1);
            ctx.sreg_valid = (ctx.sreg_valid | (3u << S_INVN)) & ~(rows == 1u ? 1u << S_RES : 0u);   // RED TO_S writes s1
            wait_credits(has_w ? 3u : 2u, 2u + (need & 1u) + (need >> 1));
            const ticket st = rms_pushes(ctx, b.at.line, src, dst, wh, n, rows, L, inv_n, eps, need, first);
            kf::release(b, st);
#if defined(TSR_FW_HOST)
            ++nr::rms_fast_taken;
#endif
            return st.v;
        }
    }
    return kf::NOT_FAST;
}

}  // namespace

// The decode fast path (header comment; exec.hpp kind_rms_norm_fast: the executor's first call, a leaf
// in its common path). The hot entry (W2b fix): one row with a weight of period 1 (every decode
// RMS_NORM of a Qwen2.5 / Gemma 4 layer but the per-head norms) while the launch's s-register shadow
// holds 1/n and eps (every norm of a list after its first), straight-line: the checks of rms_fast_any
// for that case, the five pushes with the staged words reused (rms_pushes', and the weight LD's lines
// word: w3 = L = the x LD's for one row). Any other record (several rows, no weight, a MOVI to push, a
// refusal): rms_fast_any, a tail call (no call in either path), which pushes the same ops for the
// records both take (fw_kernels_normrope_host compares both with the general path). A hot entry for
// any shape measured 279 RV cycles of work instead of 216 for this case (register pressure), so the
// per-head norms keep rms_fast_any (about 430 RV cycles per record at Gemma 4's shapes).
[[gnu::noinline]] uint32_t kind_rms_norm_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t n = a[r::N], rows = a[r::ROWS], wh = a[r::W], wp = a[r::W_PERIOD];
    const uint32_t L = plan::vu_lines(n), sv = ctx.sreg_valid;
    if (rows == 1u && wp == 1u && wh != opl::HANDLE_NONE && kflags == 0u && n - 1u < plan::VU_M_MAX &&
        (ctx.pl.caps & plan::CAPS_VU) != 0u && 2u * L <= ctx.one_lines && 2u * L <= ctx.pl.budget &&
        ((sv >> S_INVN) & 3u) == 3u) [[likely]] {
        const uint32_t eps = a[r::EPS], inv_n = a[r::INV_N];
        if (ctx.sreg_val[S_INVN] == inv_n && ctx.sreg_val[S_EPS] == eps) [[likely]] {
            // Committed: [x | w] in the one-tile block (no ring deps), no MOVI.
            const uint32_t dst = a[r::DST], src = a[r::SRC];
            const deps first = kf::first_deps(d);
            ctx.sreg_valid = sv & ~(1u << S_RES);                // RED TO_S writes s1
            wait_credits(3u, 2u);
            const uint32_t blk = ctx.one.line, wb = blk + L, pf = ctx.push_flags;
            kf::stage_reset();
            kf::push_s<0u, false>(LD(spm{blk}, from_handle(src), L), first, pf);
            const ticket ld = kf::push_s<kf::W3>(LD(spm{wb}, from_handle(wh), L), deps{}, pf);
            kf::push_s<0u, false>(red_op(blk, 0u, n, 1u, L), red_deps(ld), pf);
            const ticket vt = kf::push_s<kf::W2 | kf::W5 | kf::W6>(vop_op(blk, true, wb, 0u, 0u, n, 1u, L), deps{}, pf);
            const ticket st = kf::push_s<kf::W1>(ST(spm{blk}, from_handle(dst), L), deps{vt}, pf);
#if defined(TSR_FW_HOST)
            ++nr::rms_fast_taken;
#endif
            return st.v;
        }
    }
    return rms_fast_any(a, kflags, d, ctx);
}

// The handler: the fast path, else the general path.
ticket kind_rms_norm(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    if (n_args == r::NARGS) {
        const uint32_t t = kind_rms_norm_fast(a, kflags, d, ctx);
        if (t != kf::NOT_FAST) return ticket{t};
    }
    return rms_general(a, n_args, kflags, d, ctx);
}

#if defined(TSR_FW_HOST)
namespace nr {
// Host builds only (fw_kernels_normrope_host): the general path alone, which the test compares push
// for push with the fast path on the same record and state.
ticket rms_norm_slow(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    return rms_general(a, n_args, kflags, d, ctx);
}
}  // namespace nr
#endif

}  // namespace tsr

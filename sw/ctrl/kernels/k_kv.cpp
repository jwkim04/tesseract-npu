// k_kv.cpp — the SET_ROWS_KV kind of the graph executor (OPL v1 kind 0x18; docs/v2/06-software-
// verification.md "SW" §7.2, §7.5; docs/v2/03-attention-kv.md "AT" §4, §5.1, §7.2, §12; W2b P2.7).
//
// SET_ROWS into a KV buffer: n_rows f32 source rows of E values go to the cells idx[0 .. n_rows) of
// one K or V plane, quantized as llama.cpp's q8_0 (QACT A32 = quantize_row_q8_0_ref) and placed by
// KVST (AT §4.3 / §4.4; partial meta beats read-modify-written inside the DMA unit). Tile-op
// sequence, per run of consecutive cells (idx[i + 1] = idx[i] + 1, in index order, at most
// plan_attn.hpp's tile rows <= 64 per run: KVST's n_rows field):
//   1. LD the run's source rows (E/16 lines each) into the tile's block (one LD when the rows are
//      dense, src stride = E/16, else one per row);
//   2. QACT A32 (K = E) of the rows, at most T_MAX rows per QACT, into the block's QA images
//      (QA_LINES(A32, E) apart);
//   3. KVST (sub = which, n_rows = the run, src = the first QA image, stride 0 = QA_LINES, c0 = the
//      run's first cell, the plane base and GEOM's E/64 and LG).
// Numerics: per row, QACT A32 then the byte placement: tsr_ref_kvst of tsr_qact (AT §6.1; the KVST
// unit converts the QA scales to fp16 exactly). The grouping into runs and QACTs changes no byte
// (QACT is per row). A cell written twice gets the later row (the DMA unit runs KVSTs in queue
// order, as ggml's rows in order). sw/ctrl/ref/kv_ref.cpp composes the same calls.
//
// Block and deps (MUL_MAT's rules, SW §7.4): one block per run [rows | QA]: the one-tile block when
// it fits (a run is DMA-bracketed: LD ... KVST; V2-SW-36), else a ring block; the op's deps and a ring
// block's deps go on the run's first LD without DMA tickets (the DMA queue order implies them); QACT
// depends on the run's last LD, the later QACTs of the run follow it in the QACT queue; KVST depends
// on the run's last QACT; a ring block is released after its KVST. The returned ticket
// is the last KVST: it implies every LD and KVST (DMA queue order) and every QACT (each run's last
// QACT is a dep of its KVST; QACT queue order), I1.
//
// Decode fast path (SW §7.3 "Hot path", V2-SW-35; k_fast.hpp): n_rows = 1, the words read in place,
// the index read from the preload area (one I64), plan_attn.hpp's argument rules on the locals, the
// one-tile block from alloc_fast (V2-SW-36; the general path's runs take it too), one credit read for
// the three pushes (LD, QACT, KVST), no call; anything else (n_rows != 1, a refused record or index)
// goes to the general path, which pushes the same ops for the records both take.
//
// Data checks (the preload area is data: SW §7.2): the n_rows indices must lie inside the list's
// preload area (8-byte aligned, kind_ctx::pre_ok), and each must be < 2^32 (hi word 0; the host
// checks < 2^31) and < 16·C·LG (the plane's cells); else BAD_RECORD before any push.
#include "k_fast.hpp"
#include "k_mulmat.hpp"
#include "tsr/plan_attn.hpp"

namespace tsr {

namespace {

namespace sr = opl::setrows;
namespace at = plan::attn;

// The low word of index i, or ~0u when its hi word is not 0 (never a cell: refused).
TSR_HOT uint32_t idx_at(const volatile uint32_t * p, uint32_t i) {
    const uint32_t lo = p[2u * i], hi = p[2u * i + 1u];
    return hi == 0u ? lo : ~0u;
}

struct kv_op {
    uint32_t src, src_stride, which, plane, e64, lg, E;
    uint32_t row_lines, qa_lines, lpr;
};

// The general path: any n_rows, runs of consecutive cells, tiles of plan_attn.hpp's tile rows.
[[gnu::noinline, gnu::cold]] ticket set_rows_any(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                kind_ctx & ctx) {
    if (n_args != sr::NARGS || kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    uint32_t w[sr::NARGS];
    for (uint32_t i = 0; i < sr::NARGS; ++i) w[i] = a[i];
    const plan_limits l = plan::plan_limits_of(*ctx.cfg);
    kind_plan kp{};
    if (!plan_set_rows_kv(w, sr::NARGS, l, &kp)) return ctx.fail(kp.status);
    const uint32_t n = w[sr::N_ROWS];
    if (n == 0u) return ticket{};
    if (!ctx.pre_ok(w[sr::IDX], n, 3u, opl::PRE_KVIDX_ALIGN - 1u)) return ctx.fail(opl::RET_BAD_RECORD);
    kv_op k{};
    k.src = w[sr::SRC];
    k.src_stride = w[sr::SRC_STRIDE];
    k.which = w[sr::WHICH];
    k.plane = w[sr::PLANE];
    k.e64 = opl::geom_e64(w[sr::GEOM]);
    k.lg = opl::geom_lg(w[sr::GEOM]);
    k.E = k.e64 << 6;
    k.row_lines = at::kv_row_lines(k.e64);
    k.qa_lines = at::kv_qa_lines(k.e64);
    k.lpr = at::kv_lpr(k.e64);
    const uint32_t cells = at::kv_cells(l.num_ch, k.lg);
    // Every index first (nothing is pushed for a record with a bad one).
    const volatile uint32_t * ix = ctx.pre_lb(w[sr::IDX]);
    for (uint32_t i = 0; i < n; ++i) {
        if (idx_at(ix, i) >= cells) return ctx.fail(opl::RET_BAD_RECORD);
    }
    const uint32_t tr = kp.tile_rows, tq = l.t_max, pf = ctx.push_flags;
    ticket last{};
    for (uint32_t i = 0; i < n;) {
        // The run starting at row i (consecutive cells, at most tr rows).
        const uint32_t c0 = idx_at(ix, i);
        uint32_t run = 1;
        while (i + run < n && run < tr && idx_at(ix, i + run) == c0 + run) ++run;
        // A run is DMA-bracketed (LD ... QACT ... KVST): it takes the one-tile block whenever it fits
        // (kind_ctx::one; W2b integration): the DMA queue serialises the runs whatever their lines.
        const bool one = run * k.lpr <= ctx.one_lines;
        deps rd{};
        const spm blk = one ? ctx.one : ctx.ring->alloc(run * k.lpr, 2u, &rd);
        if (!valid(blk)) {                         // cannot happen: one tile fits the budget
            ctx.ret = opl::RET_SCRATCH;
            return last;
        }
        const deps first = i == 0u ? mmk::first_deps(d, rd) : kf::drop_dma(rd);
        const ticket ld = kf::load_rows(kf::chk_t{}, blk, from_handle(k.src), k.src_stride, k.row_lines, i, run, first, pf);
        const spm qa = blk + run * k.row_lines;
        ticket q{};
        for (uint32_t j = 0; j < run; j += tq) {
            const uint32_t t = run - j < tq ? run - j : tq;
            q = issue(QACT(qa::a32, qa + j * k.qa_lines, blk + j * k.row_lines, k.E, t, k.row_lines),
                      j == 0u ? deps{ld} : deps{}, pf);
        }
        last = issue(KVST({.kv = k.which, .n_rows_m1 = run - 1u, .src = qa.line, .base = k.plane, .e64 = k.e64,
                           .lg = k.lg, .c0 = c0, .src_stride = 0u}),
                     deps{q}, pf);
        if (!one) ctx.ring->release(blk, last);
        i += run;
    }
    return last;
}

}  // namespace

// The decode fast path (exec.hpp kind_set_rows_kv_fast: the executor's first call, a leaf in its
// common path): the words in place (one line fill), then the index (the preload area); else
// KIND_NOT_FAST with nothing changed. The executor checked n_args.
[[gnu::noinline]] uint32_t kind_set_rows_kv_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t src = a[sr::SRC], idx = a[sr::IDX], n = a[sr::N_ROWS], re = a[sr::ROW_ELEMS], which = a[sr::WHICH];
    const uint32_t plane = a[sr::PLANE], geom = a[sr::GEOM], pk = a[sr::PLANE_KIND];
    if (n == 1u && kflags == 0u) [[likely]] {
        const plan_limits l = plan::plan_limits_of(*ctx.cfg);
        const uint32_t e64 = opl::geom_e64(geom), lg = opl::geom_lg(geom);
        if (at::kv_args_ok(which, pk, geom, re, l) && ctx.pre_ok(idx, 1u, 3u, opl::PRE_KVIDX_ALIGN - 1u)) [[likely]] {
            const volatile uint32_t * ix = ctx.pre_lb(idx);
            const uint32_t cell = ix[0], hi = ix[1];
            kf::block b;
            if (hi == 0u && cell < at::kv_cells(l.num_ch, lg) && kf::alloc_fast(ctx, at::kv_lpr(e64), &b)) [[likely]] {
                const uint32_t rl = at::kv_row_lines(e64), pf = ctx.push_flags;
                const deps first = kf::first_deps(d);
                const ticket kv = kf::push_batch(2u, 0u, 0u, 1u, 0u, [&](auto tag) {
                    const ticket ld = kf::push(tag, LD(b.at, from_handle(src), rl), first, pf);
                    const ticket q = kf::push(tag, QACT(qa::a32, b.at + rl, b.at, e64 << 6, 1u, rl), deps{ld}, pf);
                    return kf::push(tag, KVST({.kv = which, .n_rows_m1 = 0u, .src = b.at.line + rl, .base = plane,
                                               .e64 = e64, .lg = lg, .c0 = cell, .src_stride = 0u}),
                                    deps{q}, pf);
                });
                kf::release(b, kv);
                return kv.v;
            }
        }
    }
    return kf::NOT_FAST;
}

// The handler: the fast path, else the general path.
ticket kind_set_rows_kv(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    if (n_args == sr::NARGS) {
        const uint32_t t = kind_set_rows_kv_fast(a, kflags, d, ctx);
        if (t != kf::NOT_FAST) return ticket{t};
    }
    return set_rows_any(a, n_args, kflags, d, ctx);
}

}  // namespace tsr

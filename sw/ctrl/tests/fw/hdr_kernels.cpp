// hdr_kernels.cpp — test kernels of the tile/VU header library (docs/design/06-software-
// verification.md "SW" §6.7, §6.8; CC §5, §7-§9, §12; EX §4.9). Built into the test image fw_hdr
// (tsr_fw_image, sw/ctrl/tests/hdr.cmake) for tsr_rv and, with TSR_FW_HOST, into the host tests
// (test_fw_hdr.cpp), so the same source pushes through both backends and the two push streams can
// be compared (SW §6.8).
//
// Every kernel obeys SW §4.7's timing rule (no DONE_SEQ / credit / QSTATUS / MTIME value reaches
// pushed words, deps, SPM stores or RET; join / join_irq / the ring's software joins only wait) and
// the launch-stateless rule (SW-4: every SPM line read was written by the launch; the ring is
// init()'d per launch). Exceptions, by design: hdr_stop (how long it spins depends on when the
// host writes LQ_STOP), excluded from the determinism comparisons.
//
//   hdr_stream  a0 x {ch, beat} (tiles x K f32), a1 W_OFF (beats, every channel), a2 y {ch, beat}
//               (tiles x ceil(N/16) lines), a3 N, a4 K, a5 fmt | tiles << 8, a6 ring lines.
//               Per tile, from an spm_ring (scratch at line 0): LD x, QACT, MVM, VOP SILU in place,
//               ST y; X / QA / Y blocks with parity 0 / 2 / 1, each write after the ring's
//               must_wait deps (folded with the producer through group: up to 3 tickets), each block
//               released after its last reader; per tile a group of the five tickets (4 units)
//               folded to one ticket (as_ticket: SYNC NOPs), join2 of consecutive tile tickets, and a
//               join_irq two tiles behind (bounded run-ahead), error() of each joined tile ticket.
//               RET = hash of every ticket relative to the launch's base sequences (>= 0), or -1 / -2
//               (ring failure / a tile in error).
//   hdr_credit  a0 src {ch, beat}, a1 lines of the long LDs, a2 count. Per unit (DMA, SYNC, QACT,
//               MVM, VU) a fresh long LD on DMA, then `count` ops of that unit behind it (DMA: queue
//               order; the others: a dep), so every queue fills while the long LD runs and issue()
//               waits for credits. RET = pushes (6 + 5 count).
//   hdr_err     a0 bad {ch, beat} (beyond the channel), a1 good {ch, beat}, a2 W_OFF. OK ops on DMA,
//               QACT, MVM; then LD from a0 (DRAM_RANGE), ST after it on DMA (ABORTED), QACT on its
//               result (DEP_FAILED), opcode 0x01 (refused: SYNC, ILLEGAL_OP), a NOP with a dep on
//               unit 7 (BAD_DEP); join_all; RET = bit mask of the checks that held (0x3FF = all):
//               error() of every ticket, first_error() and its record (ticket, ERR_INFO, ERR_W0,
//               ERR_PIDX), err_units().
//   hdr_qclear  a0 bad {ch, beat}, a1 good {ch, beat}: an error, join, qclear() and a join on it,
//               then an op that succeeds. RET = mask (0x1F = all).
//   hdr_irq     a0 src {ch, beat}, a1 lines: join_irq on a long LD (mie restored after it), on a
//               done ticket and on none; WAIT_TICKET disarmed afterwards. RET = mask (0x7 = all).
//   hdr_ring    a0 src {ch, beat}, a1 dst {ch, beat}, a2 ring first line, a3 ring lines, a4 blocks,
//               a5 seed, a6 max block lines. Random blocks (xorshift32 from the seed: lines,
//               parity, how long a block stays held, source offset): LD into the block after the
//               ring's must_wait deps, ST it to the next dst lines, release after the ST. A failed
//               alloc releases the oldest held block and retries. The race detector (fatal) checks
//               the deps (a missing one is a WAR race between a ST reading the old block and the LD
//               writing the new one). RET = dst lines written, or -(1 + block) on a failure /
//               parity or bound violation.
//   hdr_stop    a0 max spins: pushes a NOP and joins it until stop_requested(). RET = 1 if it saw
//               the stop, 0 if the spins ran out.
//   hdr_xring   a0 ring lines, a1 tiles, a2 W_OFF (Q8_0, N 32, K 64), a3 y {ch, beat} (tiles x 2
//               lines), a4 f32 bits c. Ring reuse across units with no queue-order chain between the
//               previous reader and the next writer (unlike LD ... ST pipelines, where the DMA queue
//               orders every reuse anyway): per tile, SOP MOVI s1 = c ^ (t << 4); A = VOP SETC (s1,
//               64 values, VU); B = QACT A (QACT); C = MVM B (MVM); ST C to y + 2t (DMA); A released
//               after the QACT, B after the MVM, C after the ST. Only must_wait orders a VOP writing
//               a reused A after the QACT / MVM that read the old block (the mutant races). RET = 0.
#include "tsr/kernel.hpp"
#include "tsr/rt.hpp"

using namespace tsr;

TSR_KERNEL(hdr_stream, 7, 0, 0);
TSR_KERNEL(hdr_credit, 3, 0, 0);
TSR_KERNEL(hdr_err, 3, 0, 0);
TSR_KERNEL(hdr_qclear, 2, 0, 0);
TSR_KERNEL(hdr_irq, 2, 0, 0);
TSR_KERNEL(hdr_ring, 7, 0, 0);
TSR_KERNEL(hdr_stop, 1, 0, 0);
TSR_KERNEL(hdr_xring, 5, 0, 0);

// HDR_MUTANT_NO_MUST_WAIT (the image fw_hdr_mut, test fw_hdr_rv_mutant): hdr_stream ignores
// spm_ring's must_wait deps, so a reused block can be written while its previous reader on another
// unit may still read it; the race detector must report it (it shows that must_wait is what orders
// the reuse in the real image).
#if defined(HDR_MUTANT_NO_MUST_WAIT)
#define HDR_MUST_WAIT(d) deps{}
#else
#define HDR_MUST_WAIT(d) (d)
#endif

namespace {

// A ticket relative to the launch's base sequences (the trace format's relative tickets, SW §5.2).
uint32_t rel(ticket t) {
    if (t.none() || t.unit() >= NUM_UNITS) return t.v;
    return make_ticket(t.unit(), t.seq() - launch().base[t.unit()]).v;
}
uint32_t mix(uint32_t h, uint32_t v) {
    for (uint32_t i = 0; i < 4u; ++i) h = (h ^ ((v >> (8u * i)) & 0xFFu)) * 16777619u;
    return h;
}

// xorshift32 (Marsaglia), deterministic from the seed; the host tests replay it.
struct xs32 {
    uint32_t s;
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    uint32_t below(uint32_t n) { return next() % n; }
};

uint32_t qa_lines_of(uint32_t kind, uint32_t K) {
    const uint32_t b = kind == QA_A32 ? 32u : 256u;
    const uint32_t qs = (K + 63u) / 64u, d = (4u * (K / b) + 63u) / 64u;
    const uint32_t bs = kind == QA_A256 ? (2u * (K / 16u) + 63u) / 64u : 0u;
    return qs + d + bs;
}

spm_ring g_ring;   // DTCM; init()'d by every launch that uses it (SW-4)

}  // namespace

// -----------------------------------------------------------------------------------------------
int32_t hdr_stream(const tsr_args * args) {
    const uint32_t * a = args->a;
    const dram x = from_handle(a[0]);
    const dram y = from_handle(a[2]);
    const uint32_t N = a[3], K = a[4], fmt = a[5] & 0xFFu, tiles = (a[5] >> 8) & 0xFFu;
    const wref w{a[1], uint8_t(fmt)};
    const qa kind = qa_for(w);
    const uint32_t xl = f32_lines(K), ql = qa_lines_of(uint32_t(kind), K), yl = f32_lines(N);
    g_ring.init(0, a[6]);
    uint32_t h = 2166136261u;
    ticket tile[256];
    for (uint32_t t = 0; t < tiles; ++t) {
        deps wx, wq, wy;
        const spm X = g_ring.alloc(xl, 0, &wx);
        if (!valid(X)) return -1;
        const ticket tl = issue(LD(X, x + t * xl, xl), HDR_MUST_WAIT(wx));
        const spm Q = g_ring.alloc(ql, 2, &wq);
        if (!valid(Q)) return -1;
        group gq;
        gq.add(tl);
        gq.add(HDR_MUST_WAIT(wq));
        const ticket tq = issue(QACT(kind, Q, X, K, 1), gq.as_deps());
        g_ring.release(X, tq);
        const spm Y = g_ring.alloc(yl, 1, &wy);
        if (!valid(Y)) return -1;
        group gy;
        gy.add(tq);
        gy.add(HDR_MUST_WAIT(wy));
        const ticket tm = issue(MVM(Y, Q, w, N, K, 1), gy.as_deps());
        g_ring.release(Q, tm);
        const ticket tv = issue(VOP({.func = VOP_FUNC_SILU, .d = Y.line, .a = Y.line, .m = N, .rows = 1}), after(tm));
        const ticket ts = issue(ST(Y, y + t * yl, yl), after(tv, tl));   // tl: a dep on the ST's own unit (legal)
        g_ring.release(Y, ts);
        group all;
        all.add(tl);
        all.add(tq);
        all.add(tm);
        all.add(tv);
        all.add(ts);
        tile[t] = all.as_ticket();                                     // 4 units: two SYNC NOPs
        if (t >= 1) tile[t] = join2(tile[t], tile[t - 1]);
        if (t >= 2) {
            join_irq(tile[t - 2]);
            if (error(tile[t - 2]) != 0u) return -2;
        }
        const ticket mine[7] = {tl, tq, tm, tv, ts, tile[t], wx.a};
        for (const ticket & k : mine) h = mix(h, rel(k));
        h = mix(h, X.line);
        h = mix(h, Q.line);
        h = mix(h, Y.line);
    }
    join_all();
    for (uint32_t t = 0; t < tiles; ++t)
        if (!done(tile[t]) || error(tile[t]) != 0u) return -2;
    if (first_error() != 0u) return -2;
    h = mix(h, pidx_now() - launch().pidx0);
    h = mix(h, g_ring.joins());
    return int32_t(h & 0x7FFF'FFFFu);
}

// -----------------------------------------------------------------------------------------------
int32_t hdr_credit(const tsr_args * args) {
    const dram src = from_handle(args->a[0]);
    const uint32_t long_lines = args->a[1], n = args->a[2];
    // SPM: [0, long_lines) the blockers' destination; then a 64-element f32 vector V, its QA image
    // (A32, K 64: 2 lines), the MVM / VOP outputs, the short LDs' lines.
    const spm L{0}, V{long_lines}, QA{long_lines + 4}, Y{long_lines + 8}, Z{long_lines + 12};
    uint32_t pushes = 0;
    auto blocker = [&]() {
        ++pushes;
        return issue(LD(L, src, long_lines));
    };
    const ticket tv = issue(LD(V, src, 4));
    ++pushes;
    // Each unit's ops wait for a fresh long LD, so its queue fills while that LD runs and issue()
    // has to wait for credits (DMA: queue order behind the blocker; the others: a dep on it).
    blocker();
    for (uint32_t i = 0; i < n; ++i, ++pushes) issue(LD(spm{long_lines + 16 + (i & 7u)}, src + i, 1));
    const ticket b2 = blocker();
    for (uint32_t i = 0; i < n; ++i, ++pushes) issue(NOP(), b2);
    const ticket b3 = blocker();
    ticket tq{};
    for (uint32_t i = 0; i < n; ++i, ++pushes) tq = issue(QACT(qa::a32, QA, V, 64, 1), deps{b3, tv});
    const ticket b4 = blocker();
    for (uint32_t i = 0; i < n; ++i, ++pushes) issue(MVM(Y, QA, wref{0, FMT_Q8_0}, 16, 64, 1), deps{b4, tq});
    const ticket b5 = blocker();
    for (uint32_t i = 0; i < n; ++i, ++pushes)
        issue(VOP({.func = VOP_FUNC_COPY, .d = Z.line, .a = V.line, .m = 16, .rows = 1}), deps{b5, tv});
    join_all();
    return first_error() == 0u ? int32_t(pushes) : -1;
}

// -----------------------------------------------------------------------------------------------
int32_t hdr_err(const tsr_args * args) {
    const dram bad = from_handle(args->a[0]), good = from_handle(args->a[1]);
    const wref w{args->a[2], uint8_t(FMT_Q8_0)};
    const spm X{0}, QA{8}, Y{16}, Z{24}, QB{32};
    const ticket ta = issue(LD(X, good, 4));                           // 64 f32
    const ticket tq = issue(QACT(qa::a32, QA, X, 64, 1), ta);
    const ticket tm = issue(MVM(Y, QA, w, 16, 64, 1), tq);
    const uint32_t p1 = pidx_now();
    const op ld_bad = LD(Z, bad, 4);
    const ticket t1 = issue(ld_bad);                                   // DRAM_RANGE (poisoned)
    const ticket t2 = issue(ST(Y, good + 100, 1), tm);                 // DMA in error state: ABORTED
    const ticket t3 = issue(QACT(qa::a32, QB, Z, 64, 1), t1);          // DEP_FAILED
    op ill{};
    ill.w[0] = OP_END;                                                 // refused: SYNC, ILLEGAL_OP
    ill.used = 1u;
    const ticket t5 = issue(ill);
    const ticket t6 = issue(NOP(), deps{make_ticket(7u, 0u)});         // BAD_DEP (unit 7)
    join_all();
    uint32_t m = 0;
    if (error(ta) == 0u && error(tq) == 0u && error(tm) == 0u) m |= 1u << 0;
    if (error(t1) == ERR_DRAM_RANGE) m |= 1u << 1;
    if (error(t2) == ERR_ABORTED) m |= 1u << 2;
    if (error(t3) == ERR_DEP_FAILED) m |= 1u << 3;
    if (t5.unit() == UNIT_SYNC && error(t5) == ERR_ILLEGAL_OP) m |= 1u << 4;
    if (t6.unit() == UNIT_SYNC && error(t6) == ERR_ABORTED) m |= 1u << 5;   // SYNC already in error (t5)
    err_rec r{};
    if (first_error() == ERR_DRAM_RANGE && first_error(&r)) m |= 1u << 6;
    if (r.t == t1 && r.pidx == p1) m |= 1u << 7;
    if (r.info == bad.beat && r.w0 == ld_bad.w[0]) m |= 1u << 8;
    const uint32_t eu = err_units();
    if (eu == ((1u << UNIT_DMA) | (1u << UNIT_QACT) | (1u << UNIT_SYNC))) m |= 1u << 9;
    return int32_t(m);
}

// -----------------------------------------------------------------------------------------------
int32_t hdr_qclear(const tsr_args * args) {
    const dram bad = from_handle(args->a[0]), good = from_handle(args->a[1]);
    uint32_t m = 0;
    const ticket t1 = issue(LD(spm{0}, bad, 2));
    join(t1);
    if (error(t1) == ERR_DRAM_RANGE && err_units() == (1u << UNIT_DMA)) m |= 1u << 0;
    const ticket tc = qclear();
    if (tc.unit() == UNIT_SYNC) m |= 1u << 1;
    join(tc);
    if (err_units() == 0u && first_error() == 0u) m |= 1u << 2;
    const ticket t2 = issue(LD(spm{0}, good, 2));
    const ticket t3 = issue(ST(spm{0}, good + 64, 2), t2);
    join(t3);
    if (error(t2) == 0u && error(t3) == 0u) m |= 1u << 3;
    if (err_units() == 0u) m |= 1u << 4;
    return int32_t(m);
}

// -----------------------------------------------------------------------------------------------
int32_t hdr_irq(const tsr_args * args) {
    const dram src = from_handle(args->a[0]);
    const uint32_t lines = args->a[1];
    uint32_t m = 0;
#if defined(TSR_FW_HOST)
    const uint32_t mie0 = 0;
    auto mie_now = []() { return 0u; };
#else
    auto mie_now = []() {
        uint32_t v;
        __asm__ volatile("csrr %0, mie" : "=r"(v));
        return v;
    };
    const uint32_t mie0 = mie_now();
#endif
    const ticket t = issue(LD(spm{0}, src, lines));
    join_irq(t);
    if (done(t) && error(t) == 0u && mie_now() == mie0) m |= 1u << 0;   // mie restored
    join_irq(t);                                                       // already done: returns at once
    join_irq(ticket{});                                                // none: always done
    if (hw::cw_read(CW_WAIT_TICKET) == TICKET_NONE) m |= 1u << 1;
    const ticket t2 = issue(ST(spm{0}, src + lines, lines), t);
    join_irq(t2);
    if (done(t2) && error(t2) == 0u) m |= 1u << 2;
    return int32_t(m);
}

// -----------------------------------------------------------------------------------------------
int32_t hdr_ring(const tsr_args * args) {
    const uint32_t * a = args->a;
    const dram src = from_handle(a[0]), dst = from_handle(a[1]);
    const uint32_t first = a[2], lines = a[3], nblk = a[4], maxb = a[6] ? a[6] : 1u;
    xs32 r{a[5] ? a[5] : 1u};
    g_ring.init(first, lines);
    struct held {
        spm b;
        ticket t;
    };
    held hq[4];
    uint32_t nh = 0, out = 0;
    for (uint32_t i = 0; i < nblk; ++i) {
        const uint32_t n = 1u + r.below(maxb), parity = r.below(3u), keep = r.below(4u), soff = r.below(256u);
        deps wd;
        spm b = g_ring.alloc(n, parity, &wd);
        while (!valid(b) && nh > 0u) {                                 // release the oldest held block, retry
            g_ring.release(hq[0].b, hq[0].t);
            for (uint32_t j = 1; j < nh; ++j) hq[j - 1] = hq[j];
            --nh;
            b = g_ring.alloc(n, parity, &wd);
        }
        if (!valid(b)) return -int32_t(1u + i);
        if ((parity < 2u && (b.line & 1u) != parity) || b.line < first || b.line + n > first + lines)
            return -int32_t(1u + i);
        const ticket tl = issue(LD(b, src + soff, n), wd);
        const ticket ts = issue(ST(b, dst + out, n), tl);
        out += n;
        if (keep == 0u || nh == 4u) {
            g_ring.release(b, ts);
            if (nh == 4u) {                                            // bounded holding
                g_ring.release(hq[0].b, hq[0].t);
                for (uint32_t j = 1; j < nh; ++j) hq[j - 1] = hq[j];
                --nh;
            }
        } else {
            hq[nh++] = held{b, ts};
        }
    }
    for (uint32_t j = 0; j < nh; ++j) g_ring.release(hq[j].b, hq[j].t);
    join_all();
    return first_error() == 0u ? int32_t(out) : -1000;
}

// -----------------------------------------------------------------------------------------------
int32_t hdr_xring(const tsr_args * args) {
    const uint32_t * a = args->a;
    const uint32_t tiles = a[1];
    const wref w{a[2], uint8_t(FMT_Q8_0)};
    const dram y = from_handle(a[3]);
    g_ring.init(0, a[0]);
    for (uint32_t t = 0; t < tiles; ++t) {
        deps wa, wb, wc;
        const ticket ts = issue(SOP({.sub = SOP_SUB_MOVI, .sd = 1, .imm = a[4] ^ (t << 4)}));
        const spm A = g_ring.alloc(4, 2, &wa);                         // 64 f32
        if (!valid(A)) return -1;
        group ga;
        ga.add(ts);                                                    // VU queue order anyway
        ga.add(HDR_MUST_WAIT(wa));
        const ticket tv = issue(VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_SETC, .si = 1, .d = A.line, .m = 64, .rows = 1}),
                                ga.as_deps());
        const spm B = g_ring.alloc(2, 2, &wb);                         // QA (A32, K 64)
        if (!valid(B)) return -1;
        group gb;
        gb.add(tv);
        gb.add(HDR_MUST_WAIT(wb));
        const ticket tq = issue(QACT(qa::a32, B, A, 64, 1), gb.as_deps());
        g_ring.release(A, tq);
        const spm C = g_ring.alloc(2, 2, &wc);                         // 32 f32
        if (!valid(C)) return -1;
        group gc;
        gc.add(tq);
        gc.add(HDR_MUST_WAIT(wc));
        const ticket tm = issue(MVM(C, B, w, 32, 64, 1), gc.as_deps());
        g_ring.release(B, tm);
        const ticket tst = issue(ST(C, y + 2u * t, 2), tm);
        g_ring.release(C, tst);
    }
    join_all();
    return first_error() == 0u ? 0 : -2;
}

// -----------------------------------------------------------------------------------------------
int32_t hdr_stop(const tsr_args * args) {
    for (uint32_t i = 0; i < args->a[0]; ++i) {
        if (stop_requested()) return 1;
        join(issue(NOP()));
    }
    return 0;
}

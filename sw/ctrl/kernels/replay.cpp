// replay.cpp — the `replay` base kernel (docs/design/01-execution-model.md "EX" §4.6; docs/design/
// 06-software-verification.md "SW" §7.7, §5.2, §5.4): replays a tile-only push stream (trace format
// ops/push.bin, 48-byte records) from device DRAM through the issue port. Used for the trace sets
// (traces/m2, gen, m3, vu, at, ctrl and, with SERIAL, model-v1conv), DB's SPM bulk dumps (a stream of
// ST ops, DB §5.7) and the divergence search; on the simulator through tsr_trace2_replay_sim's
// replay-kernel route (tsr_trace.c, "kernel_replay").
//
// Arguments (SW §7.1 a0-a2; a3 / a4 are this kernel's extension):
//   a0  push.bin handle {ch[31:28], beat[27:0]}       a1  record count
//   a2  flags: bit0 SERIAL (every op record depends on the previous op record instead of its
//       recorded deps; with I1 that is strictly sequential issue), bit1 BUF (a3 / a4 valid)
//   a3  BUF: first SPM line of the chunk buffers      a4  BUF: records per chunk (a multiple of 4,
//                                                         4 .. CHUNK_MAX; 0 = CHUNK_MAX)
// Without BUF the two buffers are the 2 x 1023 lines just below the runtime argument range. A
// replayer that knows the stream's SPM operands places the buffers outside them (the simulator route
// does), so the kernel's reads of the stream never conflict with the replayed ops.
//
// Algorithm (SW §7.7):
//   1. LD chunk 0 (and chunk 1) into the buffers, then read base_u = ISSUED_SEQ[u] for every unit
//      and pidx0 = ISSUE_PIDX: with at most two chunks no push of the kernel lies between the
//      replayed records, so the stream's relative tickets and push indices are the device's.
//   2. Stream push.bin in double-buffered chunks (64 KiB at most): before the records of chunk c
//      (c >= 1) the kernel pushes the LD of chunk c + 1 into the buffer chunk c - 1 used (the RISC-V
//      read all of it before: program order), then joins the LD of chunk c (a software wait on
//      DONE_SEQ: it only waits, SW §4.7) and checks it succeeded; a failed chunk LD (the stream put
//      the DMA unit into error state, or aborted the queues, before it) stops the replay
//      (RET STREAM).
//   3. Each record, read through the SPM line-buffered window:
//      kind 0  issue the words with the recorded push value (WMASK, REC) and the deps mapped back
//              as (u, base_u + rel_seq); on the DMA unit the kernel's own chunk LDs pushed after
//              base are added back (rel_seq counts only replayed ops); SERIAL replaces the deps by
//              the ticket of the previous kind-0 record. The push goes to the unit the issue path
//              chooses (issue(): refused opcodes go to SYNC and complete ILLEGAL_OP there, EX §4.6),
//              after a credit check on that unit;
//      kind 1  QCTRL = w0 (a QCLEAR waits for a SYNC credit, like qclear());
//      kind 2  TRACE_MARK = w0.
//      A soft stop (EX §4.9) is honoured at chunk boundaries.
//   4. join_all(); RET = 0 if no unit is in error state, else 1 + (ERR_PIDX of the first error -
//      pidx0 - the kernel's own pushes before it): the push index of the failing record relative to
//      the first replayed record, as the tile-only replay reports it (tsr_trace.c done_of_units).
//      When the first error is one of the kernel's own chunk LDs (a prefetch LD before pidx0, or a
//      later chunk LD: a0 / a1 put the chunk beyond the channel) no replayed record failed: RET
//      STREAM (SW §7.7). The runtime reports the error record itself in DONE r3-r6 (EX §4.6, SW §6.4).
//      Negative RET (KERNEL_ERR, or below TILE_ERR in EX §4.4's precedence): BAD_ARGS, TOO_LONG,
//      STREAM.
// Timing rule (SW §4.7): pushed words, deps and the RET come only from the stream, the arguments and
// exact counters (ISSUED_SEQ, ISSUE_PIDX); DONE_SEQ / credits only end waits. Launch-stateless: all
// state is rebuilt per launch.
#include "tsr/kernel.hpp"
#include "tsr/log.hpp"
#include "tsr/rt.hpp"

TSR_KERNEL(replay, 5, 0, 0);

namespace {

using namespace tsr;

// Same values as tsr_trace.h's TSR_REPLAY_* (the host build checks it, sw/ctrl/tests).
constexpr uint32_t F_SERIAL = 0x1u;
constexpr uint32_t F_BUF = 0x2u;
constexpr uint32_t CHUNK_MAX = 1364u;            // 64 KiB rounded down to 4 records: 1023 lines
constexpr uint32_t CHUNKS_MAX = 130u;
constexpr uint32_t RET_BAD_ARGS = 0x80000101u;
constexpr uint32_t RET_TOO_LONG = 0x80000102u;
constexpr uint32_t RET_STREAM = 0x80000103u;

constexpr uint32_t REC_BYTES = 48u;              // SW §5.2: w[8], dep[2], push, kind
constexpr uint32_t K_OP = 0u, K_QCTRL = 1u, K_MARK = 2u;

// DTCM bookkeeping of the chunk LDs pushed after base (step 2): how many replayed DMA ops preceded
// each, and its push index.
uint32_t g_own_dma[CHUNKS_MAX];
uint32_t g_own_pidx[CHUNKS_MAX];

struct stream {
    dram src;                // push.bin
    uint32_t count;          // records
    uint32_t chunk;          // records per chunk (a multiple of 4)
    uint32_t chunk_lines;    // chunk * 48 / 64
    spm buf[2];

    uint32_t chunks() const { return (count + chunk - 1u) / chunk; }
    // Lines of chunk c (the last one may be short; whole lines, the tail beyond the stream unused).
    uint32_t lines_of(uint32_t c) const {
        const uint32_t recs = count - c * chunk < chunk ? count - c * chunk : chunk;
        return (recs * REC_BYTES + LINE_BYTES - 1u) / LINE_BYTES;
    }
    ticket load(uint32_t c) const {
        return issue(LD(buf[c & 1u], src + c * chunk_lines, lines_of(c)));
    }
};

// Number of own chunk LDs (pushed after base) that precede replayed DMA op number r.
uint32_t own_before_dma(uint32_t n_own, uint32_t r) {
    uint32_t k = n_own;
    while (k > 0u && g_own_dma[k - 1u] > r) --k;
    return k;
}

// A recorded dep -> the device ticket (SW §7.7 step 3: (u, base_u + rel_seq)).
uint32_t map_dep(uint32_t rel, const uint32_t * base, uint32_t n_own) {
    const uint32_t u = rel >> 28;
    if (u >= NUM_UNITS) return rel;                       // none, marker, reserved: as recorded
    uint32_t s = rel & TICKET_SEQ;
    if (u == UNIT_DMA) s += own_before_dma(n_own, s);
    return (u << 28) | ((base[u] + s) & TICKET_SEQ);
}

}  // namespace

extern "C" int32_t replay(const tsr_args * args) {
    const uint32_t count = args->a[1], flags = args->a[2];
    if ((flags & ~(F_SERIAL | F_BUF)) != 0u) return int32_t(RET_BAD_ARGS);
    if (count == 0u) return 0;
    stream st{};
    st.src = from_handle(args->a[0]);
    st.count = count;
    st.chunk = CHUNK_MAX;
    if (flags & F_BUF) st.chunk = args->a[4] != 0u ? args->a[4] : CHUNK_MAX;
    if (st.chunk % 4u != 0u || st.chunk > CHUNK_MAX) return int32_t(RET_BAD_ARGS);
    st.chunk_lines = st.chunk * REC_BYTES / LINE_BYTES;
    const uint32_t nchunks = st.chunks();
    if (nchunks > CHUNKS_MAX) return int32_t(RET_TOO_LONG);
    const uint32_t nbuf = nchunks > 1u ? 2u : 1u;
    const uint32_t top = argblk_line();                   // the runtime argument range starts here
    const uint32_t first = (flags & F_BUF) ? args->a[3] : top - 2u * CHUNK_MAX * REC_BYTES / LINE_BYTES;
    if (first > top || nbuf * st.chunk_lines > top - first) return int32_t(RET_BAD_ARGS);
    st.buf[0] = spm{first};
    st.buf[1] = spm{first + st.chunk_lines};

    // 1. Prefetch, then the bases.
    ticket ld[2];
    ld[0] = st.load(0);
    if (nchunks > 1u) ld[1] = st.load(1);
    uint32_t base[NUM_UNITS];
    for (uint32_t u = 0; u < NUM_UNITS; ++u) base[u] = issued_seq(u);
    const uint32_t pidx0 = pidx_now();
    set_info(0);

    // 2. / 3. Stream and replay.
    const bool serial = (flags & F_SERIAL) != 0u;
    uint32_t n_own = 0, dma_ops = 0;
    ticket prev{};
    bool stream_failed = false;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t c = i / st.chunk, k = i % st.chunk;
        if (k == 0u) {
            if (c > 0u) {
                if (stop_requested()) break;                  // EX §4.9 safe point
                if (c + 1u < nchunks) {                       // the buffer of chunk c - 1 is free now
                    g_own_dma[n_own] = dma_ops;
                    g_own_pidx[n_own] = pidx_now();
                    ++n_own;
                    ld[(c + 1u) & 1u] = st.load(c + 1u);
                }
                set_info(i);
            }
            join(ld[c & 1u]);
            if (error(ld[c & 1u]) != 0u) {                    // the stream aborted its own chunk LD
                stream_failed = true;
                break;
            }
        }
        const volatile uint32_t * r = spm_lb_ptr<uint32_t>(st.buf[c & 1u], k * REC_BYTES);
        uint32_t w[12];
        for (uint32_t j = 0; j < 12u; ++j) w[j] = r[j];
        const uint32_t kind = w[11];
        if (kind == K_OP) {
            op o{};
            for (uint32_t j = 0; j < 8u; ++j) o.w[j] = w[j];
            const uint32_t push = w[10];
            o.used = uint8_t(push >> ISSUE_PUSH_WMASK_LSB);
            deps d;
            if (serial) d = deps{prev};
            else d = deps{ticket{map_dep(w[8], base, n_own)}, ticket{map_dep(w[9], base, n_own)}};
            const ticket t = issue(o, d, push & ~ISSUE_PUSH_WMASK);
            if (t.unit() == UNIT_DMA) ++dma_ops;
            prev = t;
        } else if (kind == K_QCTRL) {
            if (w[0] & QCTRL_QCLEAR) {
                while (credit(UNIT_SYNC) == 0u) hw::pause();
            }
            hw::qctrl(w[0]);
        } else if (kind == K_MARK) {
            trace_mark(w[0]);
        }
        // Other kinds: tsr_trace2_read refuses them; nothing to push.
    }

    // 4. Join, then the first error relative to the first replayed record.
    join_all();
    err_rec e{};
    if (first_error(&e)) {
        if (int32_t(e.pidx - pidx0) < 0) return int32_t(RET_STREAM);   // a prefetch LD failed (bad a0)
        uint32_t own = 0;
        for (uint32_t j = 0; j < n_own; ++j) {
            if (g_own_pidx[j] == e.pidx) return int32_t(RET_STREAM);    // a later chunk LD failed
            own += (g_own_pidx[j] - pidx0) < (e.pidx - pidx0) ? 1u : 0u;
        }
        return int32_t(1u + (e.pidx - pidx0) - own);
    }
    return stream_failed ? int32_t(RET_STREAM) : 0;
}

#ifdef TSR_FW_HOST
// The host build checks this kernel's ABI against the replayer's copy (tsr_trace.h).
#include "tsr_trace.h"
static_assert(F_SERIAL == TSR_REPLAY_F_SERIAL && F_BUF == TSR_REPLAY_F_BUF && CHUNK_MAX == TSR_REPLAY_CHUNK_MAX &&
                  CHUNKS_MAX == TSR_REPLAY_CHUNKS_MAX && RET_BAD_ARGS == TSR_REPLAY_RET_BAD_ARGS &&
                  RET_TOO_LONG == TSR_REPLAY_RET_TOO_LONG && RET_STREAM == TSR_REPLAY_RET_STREAM,
              "replay.cpp and tsr_trace.h disagree on the replay kernel's ABI");
#endif

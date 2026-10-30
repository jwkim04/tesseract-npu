// graph.cpp — the `graph` kernel: the OPL v1 op-list executor (docs/v2/06-software-verification.md
// "SW" §7.1-§7.3, §7.8; EX §4.6; V2-SW-12, V2-SW-15, V2-SW-29, V2-SW-30). Kernel arguments:
// a0 = DRAM handle of the op list, a1 = its size in bytes, a2 = flags (bit0 VERIFY_CRC).
//
// SPM layout (SW §7.3, tsr/plan.hpp exec_layout; L = SPM lines):
//   [L-16, L)                the runtime's argument range (untouched)
//   [L-16-4096, L-16)        OPL area (256 KiB). A list of more than 4096 lines streams through
//                            it in 128 KiB chunks: blob line b lives in SPM line opl + (b mod 4096),
//                            chunk c (lines [2048c, 2048c + 2048)) in half c mod 2. Chunks 0 and 1
//                            are loaded at the start; when the next record starts in chunk c, the
//                            half of chunk c - 1 is consumed and chunk c + 1 is loaded into it; a
//                            record is read only after its chunks' loads are joined (records are
//                            copied to DTCM first, so a chunk boundary inside a record is fine).
//   [opl - pre_lines, opl)   preload area (header pre_lines <= 256, V2-SW-15)
//   [0, opl - pre_lines)     scratch ring (spm_ring), handed to the kind handlers
// DTCM: ticket[4096] (the op's done ticket) and pidx[4096] (its first push index), 32 KiB, plus the
// record copy (1 KiB) and the per-launch state, all rebuilt at every launch (V2-SW-4).
//
// Algorithm (SW §7.3):
//   1. LD the list (or its first 256 KiB) into the OPL area and join; validate the header
//      (BAD_MAGIC, BAD_VERSION, TOO_MANY_OPS, layout -> BAD_RECORD) and, with VERIFY_CRC, the CRC
//      over [64, total_bytes) (a streamed list is read through once for it, then reloaded); LD every
//      preload entry and join once (the preload table is part of the list, so the list's own load
//      has to be joined before the preload loads can be pushed: two joins at the start when the
//      list has preload entries, one otherwise).
//   2. For op i = 0 ... n_ops - 1: copy the record to DTCM; check it (slot overflow, word0 [15:12],
//      n_args -> BAD_RECORD; no handler for the kind -> BAD_RECORD; deps -> BAD_DEP);
//      d = after(ticket[dep0], ticket[dep1]); MARK: TRACE_MARK NODE_BEGIN | i; pidx[i] =
//      ISSUE_PIDX; the handler pushes the op's tile ops and returns its ticket T_i (NOP: none;
//      JOIN: a SYNC NOP over d); ticket[i] = T_i. SERIAL joins T_i. Every 16 ops and before every
//      software join (SERIAL, a streamed chunk), with STOP_ON_ERROR, QSTATUS error bits stop the
//      pushing: timing-dependent, but it only decides which later ops are skipped (SW §7.3 2.6).
//      After SERIAL's join the error records (ERR_CODE) decide instead: exact once T_i is done (I2).
//   3. join_all().
//   4. The first error (smallest ERR_PIDX over the units in error state, relative to the kernel's
//      first push) is located by pidx: the op i with pidx[i] <= ERR_PIDX < pidx[i+1] -> RET i + 1.
//
// Return value (SW §6.4, §7.3; decision P2.4 for the combinations the table leaves open):
//   - a tile error: the failing op + 1 (1 ... 4096); 0 when the failing push was one of the
//     executor's own loads (list, chunk, preload), whose error record r3-r6 then names;
//   - otherwise a rejected list: 0x8000_0001 ... 0x8000_0007 (KERNEL_ERR). A list rejected after
//     some of its ops ran (records are checked as they are read) also had those ops joined first;
//     if one of them failed, the tile error wins (STATUS TILE_ERR > KERNEL_ERR, EX §4.4) and RET is
//     the failing op;
//   - else 0.
//
// Determinism (SW §4.7, V2-SW-30): pushed words, deps and records depend only on the list, the
// preload contents and ISSUE_PIDX; QSTATUS is read only for the STOP_ON_ERROR poll (allowed by
// SW §7.3). The first error after join_all and SERIAL's post-join stop come from the error records
// (ERR_CODE / ERR_PIDX, exact once the tickets concerned are done, I2; fix W2a: not QSTATUS, whose
// error bits I2 does not order with DONE_SEQ). No DONE_SEQ / credit / MTIME value is used.
//
// Profiling (EX §4.3 PROF): the executor behaves as if MARK were set (the runtime sets REC_CFG.ALL).
// REC (OPL flag) sets the ISSUE_PUSH REC bit on the executor's pushes (chunk and preload loads, JOIN
// NOPs; not the list's first load, pushed before the header is read) and is passed to the handlers
// in kind_ctx.push_flags for theirs. The header library's internal pushes (group / join2 NOPs, the
// join_all FENCE) carry no flags; a profiled launch's REC_CFG.ALL covers every entry.
#include "tsr/exec.hpp"
#include "tsr/kernel.hpp"
#include "tsr/log.hpp"
#include "tsr/rt.hpp"

TSR_KERNEL(graph, 3, 0, 0);

extern "C" {
uint32_t __tsr_exec_ticket[tsr::opl::MAX_OPS];
uint32_t __tsr_exec_pidx[tsr::opl::MAX_OPS];
tsr::exec_state __tsr_exec_state;
}

namespace tsr {

// The kind table (exec.hpp). TODO(W2b, P2.6 / P2.7): the M7 kinds' handlers.
kind_fn kind_handler(uint32_t kind) {
    switch (kind) {
        case opl::K_MUL_MAT: return &kind_mulmat;
        default:             return nullptr;
    }
}

}  // namespace tsr

namespace {

using namespace tsr;

// Register access beyond the SW §6.7 calls goes through kernel.hpp's tsr::hw (backend_dev.hpp on
// the RISC-V, backend_host.hpp on the host): QSTATUS, TRACE_MARK, CFG0 / CFG2 (core window). The
// list is read through the line-buffered SPM alias (CC §10.5: one bridge read per 64-byte line
// instead of 16); any control-window access invalidates the buffer, so each line is copied in one
// burst of loads.
inline const volatile uint32_t * spm_words(uint32_t line) { return spm_lb_ptr<uint32_t>(spm{line}); }

constexpr uint32_t MAX_CHUNKS = 36u;        // >= (1 + 4095 + 65536) lines / 2048: header, preload table
                                            // (inside the first 4096 lines), 4096 x 16 slots
constexpr uint32_t REC_WORDS = opl::MAX_REC_SLOTS * opl::SLOT_WORDS;

// Per-launch state (DTCM .bss; everything is set before use at every launch).
struct run_state {
    uint32_t opl_line;          // first SPM line of the OPL area
    dram     base;              // the list in DRAM
    uint32_t total_lines;       // blob lines (header total_bytes / 64)
    uint32_t n_chunks;          // 0: the list fits the OPL area; else its 128 KiB chunks
    uint32_t issued, ready;     // highest chunk whose LD was pushed / joined
    ticket   chunk_t[2];        // LD ticket of the chunk in each half
    uint32_t n_stream;          // chunk LDs pushed between ops (failing-op search)
    uint32_t late;              // of those, pushed by ensure() rather than prefetched
    uint32_t stream_pidx[MAX_CHUNKS];
    uint32_t pidx_base;         // ISSUE_PIDX at kernel entry
    uint32_t flags;             // header flags (| F_MARK for a PROF launch)
    uint32_t push_flags;        // PUSH_REC with F_REC
    bool     stop;              // stop pushing: STOP_ON_ERROR saw an error, or an executor load failed
};
constinit run_state g{};        // constant-initialised: no static constructor (crt0: the SDK has none)
uint32_t rec[REC_WORDS];        // the record being executed (DTCM copy)
plan::dev_cfg dev;
spm_ring ring;

inline ticket tk(uint32_t dep) { return dep == opl::DEP_NONE ? ticket{} : ticket{__tsr_exec_ticket[dep]}; }

// Copies n words of SPM line `line`, from word w, to dst.
inline void copy_line(uint32_t * dst, uint32_t line, uint32_t w, uint32_t n) {
    const volatile uint32_t * p = spm_words(line);
    for (uint32_t k = 0; k < n; ++k) dst[k] = p[w + k];
}
inline uint32_t blob_spm_line(uint32_t b) { return g.opl_line + (b & (plan::OPL_AREA_LINES - 1u)); }

// SW §7.3 2.6: with STOP_ON_ERROR, an error state on any unit stops the pushing. Returns true to stop.
// exact = false: the QSTATUS poll (one read, timing-dependent: it only decides which later ops are
// skipped); exact = true: the error records, for a decision right after a join (I2).
bool stop_on_error(bool exact = false) {
    if (!(g.flags & opl::F_STOP_ON_ERROR)) return false;
    if ((exact ? err_units() : err_units_poll()) != 0u) {
        g.stop = true;
        return true;
    }
    return false;
}

// Joins an executor load (list, chunk, preload). False if it failed: the executor stops pushing,
// and RET is 0 with the error record naming the load (SW §6.4 r3-r6).
bool sync_load(ticket t) {
    join(t);
    if (error(t) != 0u) {
        g.stop = true;
        return false;
    }
    return true;
}

// LD of chunk k of a streamed list into half k mod 2.
ticket load_chunk(uint32_t k) {
    const uint32_t first = k * plan::OPL_HALF_LINES;
    const uint32_t left = g.total_lines - first;
    const uint32_t n = left < plan::OPL_HALF_LINES ? left : plan::OPL_HALF_LINES;
    const ticket t = issue(LD(spm{g.opl_line + (k & 1u) * plan::OPL_HALF_LINES}, dram{g.base.ch, g.base.beat + first}, n),
                           deps{}, g.push_flags);
    g.chunk_t[k & 1u] = t;
    g.issued = k;
    return t;
}

// Blob line b is readable: the LD of its chunk was joined. False on a failed load or a stop.
bool ensure(uint32_t b) {
    if (g.n_chunks == 0u) return true;
    const uint32_t c = b / plan::OPL_HALF_LINES;
    while (g.ready < c) {
        const uint32_t k = g.ready + 1u;
        if (g.issued < k) {             // not reached with the prefetch rule; kept for safety
            g.stream_pidx[g.n_stream++] = pidx_now();
            g.late++;
            load_chunk(k);
        }
        if (stop_on_error()) return false;  // before every software join (SW §7.3 2.6)
        if (!sync_load(g.chunk_t[k & 1u])) return false;
        g.ready = k;
    }
    return true;
}

// The next record starts at blob line b: every chunk before b's chunk c is consumed, so chunk
// c + 1 can be loaded into the half of chunk c - 1.
inline void prefetch(uint32_t b) {
    if (g.n_chunks == 0u) return;
    const uint32_t c = b / plan::OPL_HALF_LINES;
    if (c + 1u < g.n_chunks && g.issued < c + 1u) {
        g.stream_pidx[g.n_stream++] = pidx_now();
        load_chunk(c + 1u);
    }
}

// CRC-32/ISO-HDLC of blob lines [first, last) as they are in the OPL area (line 0: words 0-15 are the
// header, skipped by the caller through first = 1).
uint32_t crc_lines(uint32_t c, uint32_t first, uint32_t last) {
    for (uint32_t b = first; b < last; ++b) {
        const volatile uint32_t * p = spm_words(blob_spm_line(b));
        for (uint32_t k = 0; k < opl::SLOT_WORDS; ++k) c = opl::crc_word(c, p[k]);
    }
    return c;
}

// First error of the launch (step 4): RET for a tile error, or ~0u if no unit is in error state.
uint32_t locate_error(uint32_t n_done, uint32_t pidx_end) {
    err_rec r{};
    if (!first_error(&r)) return ~0u;
    const uint32_t e = r.pidx - g.pidx_base;          // relative to the kernel's first push
    if (n_done == 0u) return 0u;
    if (e < __tsr_exec_pidx[0] - g.pidx_base || e >= pidx_end - g.pidx_base) return 0u;
    for (uint32_t k = 0; k < g.n_stream; ++k)
        if (e == g.stream_pidx[k] - g.pidx_base) return 0u;
    // The last op i < n_done with pidx[i] <= e (pidx is non-decreasing; empty ops share a value).
    uint32_t lo = 0, hi = n_done - 1u;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1u) / 2u;
        if (__tsr_exec_pidx[mid] - g.pidx_base <= e) lo = mid;
        else hi = mid - 1u;
    }
    return lo + 1u;
}

// Steps 1-2. Returns the KERNEL_ERR code that ended the run, or 0 (also after a stop). *n_done:
// ops whose pushes started.
uint32_t run(const tsr_args * a, uint32_t * n_done) {
    exec_state & st = __tsr_exec_state;
    const uint32_t L = spm_lines();
    g.opl_line = L - plan::ARG_LINES - plan::OPL_AREA_LINES;
    g.base = from_handle(a->a[0]);
    const uint32_t bytes = a->a[1];
    if (bytes < opl::HDR_BYTES) return opl::RET_BAD_MAGIC;   // not even a header
    g.flags = (a->flags & TSR_ARGS_FLAG_PROF) ? opl::F_MARK : 0u;   // until the header is read

    // --- 1. the list (its first 256 KiB), then the header -------------------------------------
    uint32_t first = bytes / opl::SLOT_BYTES + (bytes % opl::SLOT_BYTES != 0u ? 1u : 0u);
    if (first > plan::OPL_AREA_LINES) first = plan::OPL_AREA_LINES;
    {
        const uint32_t n0 = first < plan::OPL_HALF_LINES ? first : plan::OPL_HALF_LINES;
        ticket t = issue(LD(spm{g.opl_line}, g.base, n0));
        if (first > n0)
            t = issue(LD(spm{g.opl_line + plan::OPL_HALF_LINES}, dram{g.base.ch, g.base.beat + plan::OPL_HALF_LINES},
                         first - n0));
        if (!sync_load(t)) return 0u;
    }
    opl::header h{};
    {
        uint32_t w[opl::SLOT_WORDS];
        copy_line(w, g.opl_line, 0, opl::SLOT_WORDS);
        h.magic = w[opl::hw::MAGIC];
        h.version = uint16_t(w[opl::hw::VER] & 0xFFFFu);
        h.hdr_bytes = uint16_t(w[opl::hw::VER] >> 16);
        h.n_ops = w[opl::hw::N_OPS];
        h.n_slots = w[opl::hw::N_SLOTS];
        h.ops_off = w[opl::hw::OPS_OFF];
        h.flags = w[opl::hw::FLAGS];
        h.n_pre = w[opl::hw::N_PRE];
        h.pre_off = w[opl::hw::PRE_OFF];
        h.pre_lines = w[opl::hw::PRE_LINES];
        h.graph_id = w[opl::hw::GRAPH_ID];
        h.total_bytes = w[opl::hw::TOTAL];
        h.crc32 = w[opl::hw::CRC];
        for (uint32_t k = 0; k < 4u; ++k) h.rsvd[k] = w[12u + k];
    }
    if (const uint32_t e = opl::check_header(h, bytes)) return e;
    // The preload table must be in the first load (the executor reads it before streaming).
    if (h.n_pre != 0u && h.pre_off + opl::PRE_ENTRY_BYTES * h.n_pre > plan::OPL_AREA_LINES * opl::SLOT_BYTES)
        return opl::RET_BAD_RECORD;
    g.flags |= h.flags;
    g.push_flags = (g.flags & opl::F_REC) ? PUSH_REC : 0u;
    g.total_lines = h.total_bytes / opl::SLOT_BYTES;
    g.n_chunks = g.total_lines > plan::OPL_AREA_LINES
                     ? (g.total_lines + plan::OPL_HALF_LINES - 1u) / plan::OPL_HALF_LINES : 0u;
    g.issued = g.ready = 1u;
    st.n_ops = h.n_ops;
    st.graph_id = h.graph_id;
    st.flags = g.flags;
    st.chunks = g.n_chunks;
    TSR_LOG(TSR_LOG_INFO, "graph %08x: %u ops, %u slots, %u preload, flags %x", h.graph_id, h.n_ops, h.n_slots,
            h.n_pre, g.flags);

    // --- CRC (launch flag VERIFY_CRC) -----------------------------------------------------------
    if (a->a[2] & opl::LF_VERIFY_CRC) {
        uint32_t c = opl::CRC_INIT;
        if (g.n_chunks == 0u) {
            c = crc_lines(c, 1u, g.total_lines);
        } else {
            // One pass over the whole list, chunk by chunk, then chunks 0 and 1 again for step 2.
            for (uint32_t k = 0; k < g.n_chunks; ++k) {
                if (k >= 2u && !sync_load(load_chunk(k))) return 0u;
                const uint32_t lo = k * plan::OPL_HALF_LINES;
                const uint32_t hi = lo + plan::OPL_HALF_LINES < g.total_lines ? lo + plan::OPL_HALF_LINES : g.total_lines;
                c = crc_lines(c, lo == 0u ? 1u : lo, hi);
            }
            if (g.n_chunks > 2u) {
                load_chunk(0u);
                if (!sync_load(load_chunk(1u))) return 0u;
                if (error(g.chunk_t[0]) != 0u) { g.stop = true; return 0u; }
            }
            g.issued = g.ready = 1u;
        }
        if (opl::crc_final(c) != h.crc32) return opl::RET_CRC;
    }

    // --- preload (V2-SW-15): every entry, one join ----------------------------------------------
    plan::exec_layout lay{};
    if (!plan::layout(L, h.pre_lines, &lay)) return opl::RET_SCRATCH;
    if (h.n_pre != 0u) {
        ticket t{};
        for (uint32_t k = 0; k < h.n_pre; ++k) {
            const uint32_t off = h.pre_off + opl::PRE_ENTRY_BYTES * k;
            uint32_t w[4];
            copy_line(w, g.opl_line + off / opl::SLOT_BYTES, (off % opl::SLOT_BYTES) / 4u, 4u);
            const opl::pre_entry e{w[0], w[1], w[2], w[3]};
            if (opl::check_pre(e, h.pre_lines) != opl::RET_OK) {
                if (!t.none()) join(t);     // nothing may stay unjoinable; the error is BAD_RECORD
                return opl::RET_BAD_RECORD;
            }
            t = issue(LD(spm{lay.pre_line + e.pre_byte / opl::SLOT_BYTES}, from_handle(e.src),
                         (e.bytes + opl::SLOT_BYTES - 1u) / opl::SLOT_BYTES),
                      deps{}, g.push_flags);
        }
        if (!sync_load(t)) return 0u;
    }

    // --- 2. the ops -----------------------------------------------------------------------------
    dev.spm_lines = L;
    dev.t_max = field_get(core_csr(CSR_CFG0), CFG0_T_MAX, CFG0_T_MAX_LSB);
    dev.k_max = field_get(core_csr(CSR_CFG2), CFG2_K_MAX_256, CFG2_K_MAX_256_LSB) * 256u;
    ring.init(lay.ring_line, lay.ring_lines);
    kind_ctx ctx{};
    ctx.ring = &ring;
    ctx.preload = h.pre_lines ? spm_ptr<uint8_t>(spm{lay.pre_line}) : nullptr;
    ctx.pre_lines = h.pre_lines;
    ctx.pre = spm{lay.pre_line};
    ctx.cfg = &dev;
    ctx.push_flags = g.push_flags;

    const uint32_t ops_line = h.ops_off / opl::SLOT_BYTES;
    const uint32_t n_ops = h.n_ops, n_slots = h.n_slots, flags = g.flags, push_flags = g.push_flags;
    const bool streamed = g.n_chunks != 0u;
    uint32_t slot = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
        const uint32_t b = ops_line + slot;
        if (slot >= n_slots) return opl::RET_BAD_RECORD;            // the op area ended early
        if (streamed && !ensure(b)) return 0u;
        // Word 0 and 1 first, then only the argument words the record has (SW §7.3 2.1).
        const volatile uint32_t * p = spm_words(blob_spm_line(b));
        const uint32_t w0 = p[0], w1 = p[1];
        const uint32_t ns = opl::w0_slots(w0), kind = opl::w0_kind(w0), nargs = opl::w0_nargs(w0);

        // The kind's handler and argument count (test kinds first, exec.hpp).
        kind_fn fn = nullptr;
        uint32_t want = opl::NARGS_UNDEF;
        kind_entry ke{};
        if (tsr_exec_test_kind != nullptr && tsr_exec_test_kind(kind, &ke)) {
            fn = ke.fn;
            want = ke.nargs;
        } else if (kind == opl::K_NOP || kind == opl::K_JOIN) {
            want = 0u;
        } else if ((fn = kind_handler(kind)) != nullptr) {
            want = opl::kind_nargs(kind);
        }
        if (const uint32_t e = opl::check_record_n(w0, w1, i, slot, n_slots, want)) return e;
        if (nargs != 0u) {
            // n_args + 2 <= 16 ns (checked): the words spread over the record's lines.
            const uint32_t n0 = nargs + 2u < opl::SLOT_WORDS ? nargs + 2u : opl::SLOT_WORDS;
            for (uint32_t k = 2; k < n0; ++k) rec[k] = p[k];
            for (uint32_t s = 1, w = opl::SLOT_WORDS; w < nargs + 2u; ++s, w += opl::SLOT_WORDS) {
                if (streamed && !ensure(b + s)) return 0u;
                const uint32_t left = nargs + 2u - w;
                copy_line(rec + w, blob_spm_line(b + s), 0, left < opl::SLOT_WORDS ? left : opl::SLOT_WORDS);
            }
        }
        slot += ns;
        if (streamed) prefetch(ops_line + slot);

        const deps d = after(tk(opl::w1_dep0(w1)), tk(opl::w1_dep1(w1)));
        if (flags & opl::F_MARK) trace_mark((MARK_CLASS_NODE_BEGIN << MARK_CLASS_LSB) | i);
        __tsr_exec_pidx[i] = pidx_now();
        *n_done = i + 1u;
        set_info(i);                                                // RT_INFO: the op being pushed
        ticket t{};
        if (fn != nullptr) {
            ctx.op_index = i;
            ctx.ret = 0u;
            t = fn(rec + 2, nargs, opl::w0_kflags(w0), d, ctx);
            if (ctx.ret != 0u) return ctx.ret;
        } else if (kind == opl::K_JOIN) {
            t = issue(NOP(), d, push_flags);
        }
        __tsr_exec_ticket[i] = t.v;

        if (flags & opl::F_SERIAL) {
            // Before the join (SW §7.3 2.6) and after it: once T_i is done its error record is
            // visible (I2), so SERIAL + STOP_ON_ERROR stops exactly after the failing op, whatever
            // the schedule (the post-join check reads the records, not QSTATUS).
            if (stop_on_error()) return 0u;
            join(t);
            if (stop_on_error(true)) return 0u;
        } else if ((i & 15u) == 15u && stop_on_error()) {
            return 0u;
        }
    }
    return 0u;
}

}  // namespace

extern "C" int32_t graph(const tsr_args * a) {
    exec_state & st = __tsr_exec_state;
    st = exec_state{};
    g = run_state{};
    g.pidx_base = pidx_now();
    uint32_t n_done = 0;
    const uint32_t kret = run(a, &n_done);
    const uint32_t pidx_end = pidx_now();
    join_all();                                                     // 3.
    const uint32_t terr = locate_error(n_done, pidx_end);           // 4.
    const uint32_t ret = terr != ~0u ? terr : kret;
    st.n_done = n_done;
    st.stopped = g.stop ? 1u : 0u;
    st.pidx0 = n_done ? __tsr_exec_pidx[0] : pidx_end;
    st.pidx_end = pidx_end;
    st.late_chunks = g.late;
    st.ret = ret;
    if (ret != 0u)
        TSR_LOG(TSR_LOG_WARN, "graph %08x: ret %08x after %u of %u ops, stopped %u", st.graph_id, ret, n_done,
                st.n_ops, st.stopped);
    return int32_t(ret);
}

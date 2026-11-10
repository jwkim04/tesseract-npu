// graph.cpp — the `graph` kernel: the op list executor (docs/design/06-software-verification.md
// "SW" §7.1-§7.3, §7.8; EX §4.6; SW-12, SW-15, SW-29, SW-30). Kernel arguments:
// a0 = DRAM handle of the op list, a1 = its size in bytes, a2 = flags (bit0 VERIFY_CRC).
//
// SPM layout (SW §7.3, tsr/plan.hpp exec_layout; L = SPM lines):
//   [L-16, L)                the runtime's argument range (untouched)
//   [L-16-4096, L-16)        OPL area (256 KiB). A list of more than 4096 lines streams through
//                            it in 128 KiB chunks: blob line b lives in SPM line opl + (b mod 4096),
//                            chunk c (lines [2048c, 2048c + 2048)) in half c mod 2. Chunks 0 and 1
//                            are loaded at the start; once a record is checked and the next record
//                            starts in chunk c, the half of chunk c - 1 is consumed and chunk c + 1
//                            is loaded into it, before the record's own pushes;
//                            a record is read only after its chunks' loads are joined (a record
//                            whose lines wrap the 4096-line area, or that starts in chunk c - 1 when
//                            chunk c + 1 is loaded, is copied to DTCM; every other record is read
//                            in place).
//   [opl - pre_lines, opl)   preload area (header pre_lines <= 256, SW-15)
//   [0, opl - pre_lines)     scratch ring (spm_ring), handed to the kind handlers
// DTCM: ticket[4096] (the op's done ticket, with a sentinel word in front) and pidx[4096] (its first
// push index), 32 KiB, plus the record copy for wrapping records (1 KiB), the scratch ring's table
// (128 records, 2 KiB) and the per-launch state, all rebuilt at every launch (SW-4).
//
// Algorithm (SW §7.3):
//   1. LD the list (or its first 256 KiB) into the OPL area and join; validate the header
//      (BAD_MAGIC, BAD_VERSION, TOO_MANY_OPS, layout -> BAD_RECORD) and, with VERIFY_CRC, the CRC
//      over [64, total_bytes) (a streamed list is read through once for it, then reloaded); LD every
//      preload entry and join once (the preload table is part of the list, so the list's own load
//      has to be joined before the preload loads can be pushed: two joins at the start when the
//      list has preload entries, one otherwise).
//   2. For op i = 0 ... n_ops - 1 (op_loop): MARK: TRACE_MARK NODE_BEGIN | i; pidx[i] =
//      ISSUE_PIDX; RT_INFO = i (the op's control-window accesses come before the record is read:
//      they invalidate the SPM line buffer); read the record's words through the line-buffered
//      alias, in place; check it (slot overflow, word0 [15:12], n_args -> BAD_RECORD; no handler
//      for the kind -> BAD_RECORD; deps -> BAD_DEP); a streamed list's next chunk LD, when due
//      (then pidx[i] is read again: the op's first push follows it); d = after(ticket[dep0],
//      ticket[dep1]); the handler pushes the op's tile ops and returns its ticket T_i (NOP: none;
//      JOIN: a SYNC NOP over d; MUL_MAT: the decode fast path mmk::mulmat_t1_fast first,
//      k_mulmat.hpp, else kind_mulmat; the short M7 kinds: the table's fast entry
//      kind_<name>_fast first, else the handler out of line (call_m7_ool); FLASH_ATTN and test
//      kinds: the table's handler; kflags and d from the loop's registers); ticket[i] = T_i.
//      SERIAL joins T_i. Every 16 ops and
//      before every software join (SERIAL, a streamed chunk), with STOP_ON_ERROR, QSTATUS error
//      bits stop the pushing: timing-dependent, but it only decides which later ops are skipped
//      (SW §7.3 2.6). After SERIAL's join the error records (ERR_CODE) decide instead: exact once
//      T_i is done (I2).
//   3. join_all().
//   4. The first error (smallest ERR_PIDX over the units in error state, relative to the kernel's
//      first push) is located by pidx: the op i with pidx[i] <= ERR_PIDX < pidx[i+1] -> RET i + 1.
//
// Return value (SW §6.4, §7.3; the combinations the table leaves open are decided here):
//   - a tile error: the failing op + 1 (1 ... 4096); 0 when the failing push was one of the
//     executor's own loads (list, chunk, preload), whose error record r3-r6 then names;
//   - otherwise a rejected list: 0x8000_0001 ... 0x8000_0007 (KERNEL_ERR). A list rejected after
//     some of its ops ran (records are checked as they are read) also had those ops joined first;
//     if one of them failed, the tile error wins (STATUS TILE_ERR > KERNEL_ERR, EX §4.4) and RET is
//     the failing op;
//   - else 0.
//
// Determinism (SW §4.7, SW-30): pushed words, deps and records depend only on the list, the
// preload contents and ISSUE_PIDX; QSTATUS is read only for the STOP_ON_ERROR poll (allowed by
// SW §7.3). The first error after join_all and SERIAL's post-join stop come from the error records
// (ERR_CODE / ERR_PIDX, exact once the tickets concerned are done, I2; not QSTATUS, whose
// error bits I2 does not order with DONE_SEQ). No DONE_SEQ / credit / MTIME value is used.
//
// Profiling (EX §4.3 PROF): the executor behaves as if MARK were set (the runtime sets REC_CFG.ALL).
// REC (OPL flag) sets the ISSUE_PUSH REC bit on the executor's pushes (chunk and preload loads, JOIN
// NOPs; not the list's first load, pushed before the header is read) and is passed to the handlers
// in kind_ctx.push_flags for theirs. The header library's internal pushes (group / join2 NOPs, the
// join_all FENCE) carry no flags; a profiled launch's REC_CFG.ALL covers every entry.
#include "k_mulmat.hpp"
#include "tsr/exec.hpp"
#include "tsr/kernel.hpp"
#include "tsr/log.hpp"
#include "tsr/rt.hpp"

TSR_KERNEL(graph, 3, 0, 0);
// NOP and JOIN are the executor's own kinds (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(NOP);
TSR_KIND(JOIN);

extern "C" {
// The ticket table with a sentinel word in front: __tsr_exec_tick1[0] = TICKET_NONE ("no dep"),
// op i's ticket at i + 1, so the loop looks a dep up as tick1[(dep + 1) & 0xFFFF] with no test
// (DEP_NONE = 0xFFFF wraps to 0). __tsr_exec_ticket, the table the debugger and the tests read
// (exec.hpp), is the same storage from word 1 on (an assembler alias, no copy).
uint32_t __tsr_exec_tick1[1 + tsr::opl::MAX_OPS];
uint32_t __tsr_exec_pidx[tsr::opl::MAX_OPS];
tsr::exec_state __tsr_exec_state;
}
__asm__(".globl __tsr_exec_ticket\n\t.type __tsr_exec_ticket, @object\n\t"
        ".set __tsr_exec_ticket, __tsr_exec_tick1 + 4\n\t.size __tsr_exec_ticket, 16384\n");
static_assert(4u * tsr::opl::MAX_OPS == 16384u, "the alias's .size");

namespace tsr {

namespace {
// The kind table (exec.hpp): handler and exact argument count per kind 0x00-0x3F, in .rodata (one
// DTCM load per record instead of a switch and a call). NOP and JOIN have no handler (built into
// the loop) and 0 arguments; MUL_MAT and the M7 kinds 0x11-0x1B have theirs (an M7 kind
// whose source is not in the image has its weak stub, zz_kind_stubs.cpp: BAD_RECORD from the
// handler); other kinds have NARGS_UNDEF (BAD_RECORD from the record check).
// w0x / min_ns fold opl::check_record_n's record checks into two compares: (w0 & W0_CHECK_MASK)
// must equal w0x (n_args = the kind's, word0 [15:12] = 0; W0X_NONE, which has a bit outside the
// mask, never matches: no such kind here) and the record must have at least min_ns slots (its
// arguments fit).
constexpr uint32_t W0_CHECK_MASK = 0xFF00F000u, W0X_NONE = 1u;
// fast: the kind's decode fast path (exec.hpp kind_<name>_fast), called before fn;
// nullptr for kinds without one (MUL_MAT's is mulmat_t1_fast, called directly; FLASH_ATTN's records
// are long; test kinds).
struct kind_slot {
    kind_fn fn;
    kind_fast_fn fast;
    uint32_t w0x;
    uint32_t min_ns;
};
constexpr kind_slot slot_for(kind_fn fn, uint32_t nargs, kind_fast_fn fast = nullptr) {
    return kind_slot{fn, fast, nargs << 24, opl::slots_for_nargs(nargs)};
}
constexpr kind_fn m7_handler(uint32_t kind) {
    switch (kind) {
        case opl::K_ADD:         return &kind_add;
        case opl::K_MUL:         return &kind_mul;
        case opl::K_SCALE:       return &kind_scale;
        case opl::K_RMS_NORM:    return &kind_rms_norm;
        case opl::K_GLU:         return &kind_glu;
        case opl::K_ROPE:        return &kind_rope;
        case opl::K_GET_ROWS:    return &kind_get_rows;
        case opl::K_SET_ROWS_KV: return &kind_set_rows_kv;
        case opl::K_FLASH_ATTN:  return &kind_flash_attn;
        case opl::K_CPY:         return &kind_cpy;
        case opl::K_UNARY:       return &kind_unary;
        default:                 return nullptr;
    }
}
constexpr kind_fast_fn m7_fast(uint32_t kind) {
    switch (kind) {
        case opl::K_ADD:         return &kind_add_fast;
        case opl::K_MUL:         return &kind_mul_fast;
        case opl::K_SCALE:       return &kind_scale_fast;
        case opl::K_RMS_NORM:    return &kind_rms_norm_fast;
        case opl::K_GLU:         return &kind_glu_fast;
        case opl::K_ROPE:        return &kind_rope_fast;
        case opl::K_GET_ROWS:    return &kind_get_rows_fast;
        case opl::K_SET_ROWS_KV: return &kind_set_rows_kv_fast;
        case opl::K_CPY:         return &kind_cpy_fast;
        case opl::K_UNARY:       return &kind_unary_fast;
        default:                 return nullptr;   // FLASH_ATTN: no fast path
    }
}
constexpr kind_slot make_slot(uint32_t kind) {
    if (kind == opl::K_NOP || kind == opl::K_JOIN) return slot_for(nullptr, 0u);
    if (kind == opl::K_MUL_MAT) return slot_for(&kind_mulmat, opl::kind_nargs(kind));
#if !defined(TSR_TEST_NO_KIND_MUL_MAT_EPILOGUE) && !defined(TSR_TEST_NO_KIND_MUL_MAT)
    if (kind == opl::K_MUL_MAT_SCALE) return slot_for(&kind_mulmat_scale, opl::kind_nargs(kind));
    if (kind == opl::K_MUL_MAT_SOFTCAP) return slot_for(&kind_mulmat_softcap, opl::kind_nargs(kind));
#endif
    if (kind == opl::K_MOE_ROUTE) return slot_for(&kind_moe_route, opl::kind_nargs(kind));
    if (kind == opl::K_MUL_MAT_ID) return slot_for(&kind_mul_mat_id, opl::kind_nargs(kind));
    if (kind >= opl::K_ADD && kind <= opl::K_UNARY) return slot_for(m7_handler(kind), opl::kind_nargs(kind), m7_fast(kind));
    if (kind == opl::K_SOFTPLUS) return slot_for(&kind_softplus, opl::kind_nargs(kind));
    if (kind == opl::K_SSM_CONV) return slot_for(&kind_ssm_conv, opl::kind_nargs(kind));
    if (kind == opl::K_GATED_DELTA_NET) return slot_for(&kind_gated_delta_net, opl::kind_nargs(kind));
    return kind_slot{nullptr, nullptr, W0X_NONE, 1u};
}
struct kind_table {
    kind_slot s[opl::N_KINDS];
};
constexpr kind_table make_kind_table() {
    kind_table t{};
    for (uint32_t k = 0; k < opl::N_KINDS; ++k) t.s[k] = make_slot(k);
    return t;
}
constexpr kind_table k_kinds = make_kind_table();
}  // namespace

kind_fn kind_handler(uint32_t kind) { return kind < opl::N_KINDS ? k_kinds.s[kind].fn : nullptr; }

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
    bool     ring_used;         // the scratch ring was initialized for this launch (its counters are)
};
constinit run_state g{};        // constant-initialized: no static constructor (crt0: the SDK has none)
uint32_t rec[REC_WORDS];        // the record being executed (DTCM copy)
// constinit: dev_cfg's M7 fields have default member initializers, which made a
// default-initialized `dev` a dynamic initialization (a static constructor in .init_array, which the
// SDK's crt0 does not run and fw_elf_image rejects); value-initialized it is constant again.
constinit plan::dev_cfg dev{};
constinit spm_ring ring{};


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

// The first nw words of the record at blob line b to `rec` (a streamed record whose lines wrap the
// OPL area, or whose half the next chunk's LD is about to overwrite).
[[gnu::noinline]] void copy_record(uint32_t b, uint32_t nw) {
    for (uint32_t s = 0, w = 0; w < nw; ++s, w += opl::SLOT_WORDS) {
        const uint32_t left = nw - w;
        copy_line(rec + w, blob_spm_line(b + s), 0, left < opl::SLOT_WORDS ? left : opl::SLOT_WORDS);
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

// Step 2's loop (SW §7.3 2.1-2.6). Returns the KERNEL_ERR code that ended it, or 0 (also after a
// stop); *n_done: ops whose pushes started. STREAMED: the list streams through the OPL area
// (ensure per record, the next chunk's LD before the handler when due); else the records are read
// where the list was loaded, by a pointer walking the op area. The hot loop (SW §7.8): per record
// two control-window accesses (three with MARK) before the record is read (ISSUE_PIDX, RT_INFO,
// TRACE_MARK; they invalidate the SPM line buffer, CC §10.5), then the record's words through the
// line-buffered alias (one 64-byte line fill), its checks, the deps, the handler (which reads its
// argument words in place, exec.hpp; MUL_MAT records try the decode fast path first,
// k_mulmat.hpp), the ticket. Records are not copied (streamed lists: except the two cases above).
struct loop_in {
    uint32_t n_ops, n_slots, ops_line, flags, push_flags;
};

// A kind handler called with the record's fields re-read from its words (args[-2] = word0, args[-1]
// = word1, in place or the DTCM copy). The loop's MUL_MAT fallback (a record the decode fast path
// does not take) calls it out of line, so that nothing the fallback needs stays live across the
// fast path's call; other kinds call it inline.
TSR_HOT ticket call_handler(kind_fn fn, const volatile uint32_t * args, kind_ctx & ctx) {
    const uint32_t w0 = args[-2], w1 = args[-1];
    const uint32_t * const tick1 = __tsr_exec_tick1;
    const deps d = after(ticket{tick1[(opl::w1_dep0(w1) + 1u) & 0xFFFFu]}, ticket{tick1[(opl::w1_dep1(w1) + 1u) & 0xFFFFu]});
    return fn(args, opl::w0_nargs(w0), opl::w0_kflags(w0), d, ctx);
}
[[gnu::noinline]] ticket call_handler_ool(kind_fn fn, const volatile uint32_t * args, kind_ctx & ctx) {
    ++__tsr_exec_state.mm_fallback;
    return call_handler(fn, args, ctx);
}
// An M7 record its fast path did not take (nothing was pushed): the kind's handler (its general
// path), out of line, with the record's fields re-read (word0 / word1 in place or the DTCM copy; d
// recomputed from the ticket table), so that nothing but args and ctx stays live across the fast
// call in op_loop.
[[gnu::noinline, gnu::cold]] ticket call_m7_ool(const volatile uint32_t * args, kind_ctx & ctx) {
    return call_handler(k_kinds.s[opl::w0_kind(args[-2])].fn, args, ctx);
}

// PLAIN: the list has neither MARK nor SERIAL (every production list unless profiled): the
// two per-record flag tests fold away (about 6 RV cycles per record); instantiated for lists read in
// place only (a streamed list, > 4,096 lines, pays them).
template <bool STREAMED, bool PLAIN = false>
[[gnu::noinline]] uint32_t op_loop(const loop_in & in, kind_ctx & ctx, uint32_t * n_done) {
    const uint32_t n_ops = in.n_ops, n_slots = in.n_slots, ops_line = in.ops_line, flags = in.flags;
    const uint32_t push_flags = in.push_flags;
    uint32_t * const tick1 = __tsr_exec_tick1;
    uint32_t * const pidx = __tsr_exec_pidx;
    tick1[0] = TICKET_NONE;
    const volatile uint32_t * p = spm_words(blob_spm_line(ops_line));
    uint32_t slot = 0;
    uint32_t i = 0;
    uint32_t ret = 0;
    for (; i < n_ops; ++i) {
        if (slot >= n_slots) { ret = opl::RET_BAD_RECORD; break; }  // the op area ended early
        const uint32_t b = ops_line + slot;
        if constexpr (STREAMED) {
            if (!ensure(b)) break;
            p = spm_words(blob_spm_line(b));
        }
        // The op's control-window accesses come first: they invalidate the SPM line buffer, so
        // the record's words below are one line fill. MARK's NODE_BEGIN i is pushed here, before
        // the record is read (a record found malformed below has its marker but no pushes).
        if (!PLAIN && (flags & opl::F_MARK)) trace_mark((MARK_CLASS_NODE_BEGIN << MARK_CLASS_LSB) | i);
        pidx[i] = pidx_now();                                       // before the op's first push (exact)
        set_info(i);                                                // RT_INFO: the op being read / pushed
        const uint32_t w0 = p[0], w1 = p[1];
        const uint32_t kind = opl::w0_kind(w0), ns = opl::w0_slots(w0);

        // The kind's handler and record checks (exec.hpp: the table, then the test hook; MUL_MAT,
        // the decode list's kind, without the table load).
        kind_slot ks{nullptr, nullptr, W0X_NONE, 1u};
        if (kind == opl::K_MUL_MAT) ks = slot_for(&kind_mulmat, opl::mm::NARGS);
        else if (kind < opl::N_KINDS) ks = k_kinds.s[kind];
        if (ks.w0x == W0X_NONE && tsr_exec_test_kind != nullptr) {
            kind_entry ke{};
            if (tsr_exec_test_kind(kind, &ke)) ks = ke.nargs <= 0xFFu ? slot_for(ke.fn, ke.nargs) : kind_slot{nullptr, nullptr, W0X_NONE, 1u};
        }
        // opl::check_record_n's checks and codes: BAD_RECORD (word0 [15:12], n_args not the kind's,
        // slot overflow, arguments that do not fit the slots), then BAD_DEP (dep >= own index;
        // DEP_NONE = 0xFFFF wraps to 0).
        if ((w0 & W0_CHECK_MASK) != ks.w0x || ns > n_slots - slot || ns < ks.min_ns) {
            ret = opl::RET_BAD_RECORD;
            break;
        }
        const uint32_t d0 = (opl::w1_dep0(w1) + 1u) & 0xFFFFu, d1 = (opl::w1_dep1(w1) + 1u) & 0xFFFFu;
        if (d0 > i || d1 > i) {
            ret = opl::RET_BAD_DEP;
            break;
        }
        // The argument words in place (exec.hpp); a record whose lines wrap the streamed OPL area
        // is copied to DTCM instead.
        const volatile uint32_t * args = p + 2;
        if constexpr (STREAMED) {
            // The lines holding the record's words (not its
            // padding slots) are made readable, then, once the record is checked, the next chunk's
            // LD is pushed before the record's own pushes.
            const uint32_t nw = opl::w0_nargs(w0) + 2u;
            const uint32_t last = b + (nw - 1u) / opl::SLOT_WORDS;
            if (last != b) {
                if (!ensure(last)) break;
                if (blob_spm_line(b) + (last - b) != blob_spm_line(last)) {
                    copy_record(b, nw);
                    args = rec + 2;
                }
            }
            slot += ns;
            // prefetch(ops_line + slot): chunk c + 1 goes into the half of chunk c - 1. When this
            // record starts in chunk c - 1, its words are copied to DTCM first (at most once per
            // 2,048 lines). pidx[i] is re-read: the op's first push comes after the chunk LD.
            const uint32_t c = (ops_line + slot) / plan::OPL_HALF_LINES;
            if (c + 1u < g.n_chunks && g.issued < c + 1u) {
                if (args != rec + 2 && b / plan::OPL_HALF_LINES != c) {
                    copy_record(b, nw);
                    args = rec + 2;
                }
                g.stream_pidx[g.n_stream++] = pidx_now();
                load_chunk(c + 1u);
                pidx[i] = pidx_now();
            }
        } else {
            p += ns * opl::SLOT_WORDS;
            slot += ns;
        }

        ticket t{};
        if (ks.fn != nullptr) {
            ctx.op_index = i;
            if (kind == opl::K_MUL_MAT) {                           // deps after(t0, t1)
                t = ticket{mmk::mulmat_t1_fast(args, opl::w0_kflags(w0), ticket{tick1[d0]}, ticket{tick1[d1]}, ctx)};
                if (t.v == mmk::NOT_FAST) t = call_handler_ool(ks.fn, args, ctx);
            } else if (ks.fast != nullptr) {                         // the short M7 kinds: the decode fast
                // path first (a leaf, exec.hpp kind_<name>_fast), else the handler,
                // out of line; the record's fields from the loop's registers (w0; d0 / d1 already
                // offset for the sentinel table), not re-read through args.
                const uint32_t v = ks.fast(args, opl::w0_kflags(w0), after(ticket{tick1[d0]}, ticket{tick1[d1]}), ctx);
                t = v != KIND_NOT_FAST ? ticket{v} : call_m7_ool(args, ctx);
            } else {                                                // FLASH_ATTN, test kinds
                t = ks.fn(args, opl::w0_nargs(w0), opl::w0_kflags(w0), after(ticket{tick1[d0]}, ticket{tick1[d1]}), ctx);
            }
            if (ctx.ret != 0u) {
                ++i;                                                // its pushes started
                ret = ctx.ret;
                break;
            }
        } else if (kind == opl::K_JOIN) {
            t = issue(NOP(), after(ticket{tick1[d0]}, ticket{tick1[d1]}), push_flags);
        }
        tick1[i + 1u] = t.v;

        if (!PLAIN && (flags & opl::F_SERIAL)) {
            // Before the join (SW §7.3 2.6) and after it: once T_i is done its error record is
            // visible (I2), so SERIAL + STOP_ON_ERROR stops exactly after the failing op, whatever
            // the schedule (the post-join check reads the records, not QSTATUS).
            if (stop_on_error()) { ++i; break; }
            join(t);
            if (stop_on_error(true)) { ++i; break; }
        } else if ((i & 15u) == 15u && stop_on_error()) {
            ++i;
            break;
        }
    }
    *n_done = i;
    return ret;
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

    // --- preload (SW-15): every entry, one join ----------------------------------------------
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
    // Device parameters (plan.hpp dev_cfg), once per launch: CFG0 / CFG2 / CFG3 / CAPS through the
    // core window (hardware constants: not timing-dependent).
    {
        const uint32_t cfg0 = core_csr(CSR_CFG0), cfg2 = core_csr(CSR_CFG2);
        dev.spm_lines = L;
        dev.t_max = field_get(cfg0, CFG0_T_MAX, CFG0_T_MAX_LSB);
        dev.k_max = field_get(cfg2, CFG2_K_MAX_256, CFG2_K_MAX_256_LSB) * 256u;
        dev.num_ch = field_get(cfg0, CFG0_NUM_CH, CFG0_NUM_CH_LSB);
        dev.vl = field_get(cfg2, CFG2_VL, CFG2_VL_LSB);
        dev.pcap = field_get(core_csr(CSR_CFG3), CFG3_PCAP16, CFG3_PCAP16_LSB) * 16u;
        dev.caps = core_csr(CSR_CAPS);
    }
    ring.init(lay.ring_line, lay.ring_lines);
    g.ring_used = true;
    kind_ctx ctx{};
    ctx.ring = &ring;
    ctx.preload = h.pre_lines ? spm_ptr<uint8_t>(spm{lay.pre_line}) : nullptr;
    ctx.pre_lines = h.pre_lines;
    ctx.pre = spm{lay.pre_line};
    ctx.cfg = &dev;
    ctx.mml = plan::mm_limits_of(dev);
    ctx.pl = plan::plan_limits_of(dev);
    ctx.one = spm{lay.one_line};
    ctx.one_lines = lay.one_lines;
    ctx.push_flags = g.push_flags;

    loop_in in;
    in.n_ops = h.n_ops;
    in.n_slots = h.n_slots;
    in.ops_line = h.ops_off / opl::SLOT_BYTES;
    in.flags = g.flags;
    in.push_flags = g.push_flags;
    if (g.n_chunks != 0u) return op_loop<true>(in, ctx, n_done);
    return (g.flags & (opl::F_MARK | opl::F_SERIAL)) == 0u ? op_loop<false, true>(in, ctx, n_done)
                                                           : op_loop<false>(in, ctx, n_done);
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
    if (g.ring_used) {              // else they are the previous launch's (a list rejected before step 2)
        st.ring_joins = ring.joins();
        st.ring_slow = ring.slow_allocs();
    }
    if (ret != 0u)
        TSR_LOG(TSR_LOG_WARN, "graph %08x: ret %08x after %u of %u ops, stopped %u", st.graph_id, ret, n_done,
                st.n_ops, st.stopped);
    return int32_t(ret);
}

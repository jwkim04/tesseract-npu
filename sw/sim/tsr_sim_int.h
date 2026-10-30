// tsr_sim_int.h — simulator-internal interface shared by the sim's source files (not installed).
//
// Owner: the core (tsr_sim.c, tsr_q.c). Users: tsr_vu.c / tsr_at.c (unit semantics, P1.4 / P1.5),
// tsr_rv.c / tsr_bus.c (the RISC-V model and its bus, P1.7), and later the scheduler (P1.8), the
// race detector (P1.9), trace v2 (P1.10) and the debug models (P1.12 / P1.13). The public API is
// tsr_sim.h (host CSR port = EX §5 through tsr_sim_csr_read / tsr_sim_csr_write, time base, test
// hooks); nothing here is ABI. docs/v2/06-software-verification.md (SW) §4 is the model.
//
//   1. operand extents (CC §8.2): what a checker class reports for each op;
//   2. per-opcode handler table (tsr_op_handler, tsr_sim_register_op): checker class, execution
//      and cycle model of each opcode; unit registration;
//   3. state accessors: SPM, DRAM, the s bank, PERF, capture, configuration;
//   4. RISC-V side: control window (CC §5), core window (CC §5.3), run / reset state, TCMs,
//      interrupt levels, the RISC-V agent hook the time base steps;
//   5. time base (P1.6 deterministic model; P1.8 replaces its policy, not this interface) and
//      the hooks for trace / race detector / debug models;
//   6. (TSR_SIM_CORE_INTERNAL only) the core's own structures, shared by tsr_sim.c and tsr_q.c.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 1. Operand extents (CC §8.2, DB §5.3, SW §4.6)
//
// Up to four per op, each tagged with its memory kind and direction, in operand field order.
// SPM extents are in lines, DRAM extents in beats, SREG extents in s-register indices; the range
// is [lo, hi) with hi > lo (an empty operand, extent 0, is not listed). They are exactly the
// extents the checker's range checks use: an operand with start a and extent e is in range iff
// a + e <= limit (spec range-check rule), and its extent is [a, a + e). At most one DRAM extent;
// its channel is `ch`, or TSR_EXT_CH_ALL for an MVM-style weight stream (the same beat range on
// every channel, CC §8.2). The four-extent limit is the hardware's (SPM + DRAM); SREG extents are
// simulator bookkeeping for the race detector and the VU's s hazards (VU §7.8) and come on top,
// at most TSR_EXT_SREG_MAX, never range-checked. A read-modify-write operand (MVM ACC, VU
// accumulate forms) is listed once as WRITE and once as READ. An op lists at most
// TSR_EXT_MAX + TSR_EXT_SREG_MAX entries in all: ROPE has four SPM operands and reads three s
// registers (sm, sts, sfs); a merging AMERGE lists its four SPM operands in six entries (Orun
// and the running state are read-modify-write).
// ---------------------------------------------------------------------------
enum {
    TSR_EXT_SPM  = 1,
    TSR_EXT_DRAM = 2,
    TSR_EXT_SREG = 3,
};
#define TSR_EXT_READ     0u
#define TSR_EXT_WRITE    1u
#define TSR_EXT_CH_ALL   0xFFu
#define TSR_EXT_MAX      4u     // SPM + DRAM operands per op (CC §8.2)
#define TSR_EXT_SREG_MAX 3u     // s registers per op (ROPE: sm, sts, sfs; RED TO_S RMS: sd, sk, se)

typedef struct tsr_extent {
    uint8_t  kind;     // TSR_EXT_SPM / DRAM / SREG
    uint8_t  write;    // TSR_EXT_READ / TSR_EXT_WRITE
    uint8_t  ch;       // DRAM: channel or TSR_EXT_CH_ALL; else 0
    uint8_t  rsvd;
    uint64_t lo, hi;   // [lo, hi): lines, beats or s indices
} tsr_extent;

typedef struct tsr_extents {
    uint32_t   n;                                     // used entries
    uint32_t   overflow;                              // entries that did not fit (a checker bug)
    tsr_extent e[TSR_EXT_MAX + TSR_EXT_SREG_MAX];
} tsr_extents;

// Appends an extent [lo, hi) (nothing if hi <= lo or x is NULL). An entry beyond the array is
// counted in overflow, and the dispatcher aborts the simulation on it (with the per-kind limits
// above: at most one DRAM entry, TSR_EXT_SREG_MAX SREG entries, TSR_EXT_MAX SPM / DRAM operands
// with a read-modify-write pair counted once).
static inline void tsr_ext_add(tsr_extents * x, uint8_t kind, uint8_t write, uint8_t ch,
                               uint64_t lo, uint64_t hi) {
    if (x == NULL || hi <= lo) return;
    if (x->n >= TSR_EXT_MAX + TSR_EXT_SREG_MAX) {
        x->overflow++;
        return;
    }
    tsr_extent * e = &x->e[x->n++];
    e->kind = kind; e->write = write; e->ch = ch; e->rsvd = 0; e->lo = lo; e->hi = hi;
}

// ---------------------------------------------------------------------------
// 2. Per-opcode handler table (CC §8.2 checker class + unit semantics, SW §4.3-§4.5)
//
//   unit    CC §7.4 unit id (TSR_UNIT_*), as tsr_op_unit() of the tables. The issue path sends an
//           opcode to SYNC with ILLEGAL_OP (without calling check) when it is reserved / retired
//           (tsr_op_legal), when CAPS lacks its tsr_op_caps bits, when its unit is not built
//           (Q_DEPTH 0) or when no handler is registered (EX §7.2, CC §7.4).
//   check   the checker class. Returns 0 or an EX §8.1 code, the first in the order
//           ILLEGAL_OP -> BAD_FMT -> K_RANGE -> T_RANGE -> OPERAND -> SPM_RANGE -> DRAM_RANGE
//           (BAD_DEP is the queue model's, inserted after ILLEGAL_OP); range operands in field
//           order. *err_info (never NULL) = the start address for SPM_RANGE / DRAM_RANGE, the
//           offending value for OPERAND, else 0. On success *ext (n = 0 on entry; may be NULL)
//           receives the op's extents. Pure: it reads only w and the configuration (limits from
//           tsr_sim_spm_lines(), tsr_sim_dram_limit_beats(), tsr_sim_config()). Must accept x16
//           only where the op defines it (EX §7.1).
//   exec    applies an op that passed check: every architectural effect (SPM, DRAM, s bank),
//           the unit's PERF event counters through tsr_sim_perf_add (CH_RD_BEATS, VU_LINES,
//           KVST_RMW, ...) and capture through tsr_sim_cap_read / tsr_sim_cap_write. Never fails
//           (EX §7.2). It is called once, at the op's modelled completion time (SW §4.1: an op's
//           effects are applied atomically at completion), with check having passed at dispatch.
//           NULL for SYNC ops (NOP, FENCE).
//   cycles  the unit's busy time in core cycles for the op (>= 1; 0 is taken as 1),
//           deterministic, from w and the configuration only (the unit's cycle model, VU §7,
//           AT §9, the tile-unit model in tsr_sim.c; no issue cycle). The queue model uses it for
//           the completion time and the unit's busy PERF counter (MVM_CYCLES, VU_CYCLES,
//           DMA_CYCLES, ...), which exec must not touch. NULL for SYNC ops.
// ---------------------------------------------------------------------------
typedef struct tsr_op_handler {
    uint8_t  unit;
    uint32_t (*check)(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * err_info);
    void     (*exec)(tsr_sim * sim, const uint32_t w[8]);
    uint64_t (*cycles)(const tsr_sim * sim, const uint32_t w[8]);
} tsr_op_handler;

// Registers (or replaces) the handler of an opcode; *h is copied. h = NULL removes it. The table
// is per process: handlers are pure functions of (sim, w). The default registrations below run
// once (pthread_once) before the first explicit registration or simulator, so a test's own stubs,
// registered before it creates a simulator, replace them. Do not change a handler while a
// simulator holds entries of that opcode.
void tsr_sim_register_op(uint8_t opcode, const tsr_op_handler * h);
const tsr_op_handler * tsr_sim_op_handler(uint8_t opcode);   // NULL if none

// Unit registration functions, each registering its opcodes (idempotent). tsr_sim_create runs, once
// per process: tsr_core_register() (tsr_sim.c: NOP LD ST QACT MVM FENCE), then tsr_vu_register()
// if the library was built with TSR_SIM_HAVE_VU and tsr_at_register() if built with
// TSR_SIM_HAVE_AT. sw/sim/CMakeLists.txt defines those two macros on tsr_sim / tsr_sim_shared
// when the target's sources contain tsr_vu.c / tsr_at.c (explicit call list rather than weak
// symbols: a weak reference does not pull an object out of a static archive).
void tsr_core_register(void);
void tsr_vu_register(void);       // tsr_vu.c (P1.4)
void tsr_at_register(void);       // tsr_at.c (P1.5)

// CAPS bits the registered handlers implement: VU (VOP RED ROPE SOP all registered), TOPK, ATT
// (all seven AT ops), DQROW, VMASK. tsr_sim_create refuses a config whose caps ask for more.
// (Same as the public tsr_sim_caps_implemented().)

// ---------------------------------------------------------------------------
// 3. State accessors (tsr_sim.c)
// ---------------------------------------------------------------------------
// tsr_sim.h already has: tsr_sim_spm(sim) (SPM bytes, spm_bytes long; the RISC-V's SPM window
// 0x4000_0000 + i, CC §4.1, is byte i), tsr_sim_dram(sim, ch) (byte 0 of channel ch),
// tsr_sim_dram_limit_beats(sim), tsr_sim_config(sim) (NUM_CH, T_MAX, K_MAX, SPM size, VL, CAPS),
// tsr_sim_sreg(sim) (the s bank s0..s15 as 16 floats, VU §4.4 / V2-VU-11/12: written only by
// VU-queue ops' exec; read by the host at SREG0-15 = 0x9C0 and by the RISC-V through the core
// window; +0.0 after create, kept across SOFT_RESET, EX §7.4).
uint32_t tsr_sim_spm_lines(const tsr_sim * sim);       // SPM_BYTES / 64
uint32_t tsr_sim_num_ch(const tsr_sim * sim);

// PERF (EX §6). tsr_sim_perf_add: an event counter increment, gated by ACTIVE (the queue path) or
// always counted (the tsr_sim_op test hook). Units use it for their event counters; time
// counters are the time base's. idx < TSR_PERF_SLOTS, else ignored.
void tsr_sim_perf_add(tsr_sim * sim, uint32_t idx, uint64_t n);

// Access capture hooks for exec functions (SW §5.5): a DRAM read / write of beats [lo, lo + n) on
// channel ch. No-ops while capture is off. Call cap_read before reading (it copies the initial
// content) and cap_write after writing.
void tsr_sim_cap_read(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n);
void tsr_sim_cap_write(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n);

// Scratch for exec functions: >= n bytes owned by the simulator, 64-byte aligned, reused by the
// next call (not preserved between ops, not thread safe).
void * tsr_sim_scratch(tsr_sim * sim, size_t n);
// MVM-style row parallelism: runs fn(ctx, lo, hi) over [0, total) on the simulator's worker
// threads (tsr_sim_set_threads; results must not depend on the split), chunk rows at a time.
void tsr_sim_parallel_rows(tsr_sim * sim, void (*fn)(void * ctx, uint32_t lo, uint32_t hi),
                           void * ctx, uint32_t total, uint32_t chunk);

// ---------------------------------------------------------------------------
// 4. RISC-V side (CC §4-§5, §12; EX §4.5)
//
// Control window: offsets from 0x2000_0000 (TSR_CW_* in tsr_tables.h). Word accesses only: the
// bus model (tsr_bus.c) faults byte / half accesses before calling. tsr_sim_cw_access returns
//   TSR_CW_OK     done (*val holds the read value for a read);
//   TSR_CW_FAULT  access fault (LQ_POP with LQ_AVAIL = 0 or while ACTIVE; LCQ_POST with
//                 LCQ_FREE = 0 or ANSWER while not ACTIVE; EX §4.5) — nothing changed;
//   TSR_CW_WAIT   the store must wait and be retried later, nothing changed: ISSUE_PUSH or QCTRL
//                 without a credit / issue-FIFO entry (CC §7.3). The RISC-V model keeps the store
//                 pending, counts the clk_rv cycles it waits (PERF PUSH_STALL through
//                 tsr_sim_perf_add) and after TSR_ISSUE_WAIT_MAX cycles (not counted under DB's
//                 dbg_stop) calls tsr_sim_cw_push_timeout() and takes a store access fault.
// Reads have no side effects; reserved and write-only offsets read 0; writes to read-only or
// reserved offsets are ignored. 0x160-0x17C (except TRACE_MARK) and 0x400-0x7FC are DB's and go
// to the debug hooks (§5 below), reading 0 without them.
// tsr_sim_cw_read / tsr_sim_cw_write are the plain forms for tests and tools outside the time
// base: faults are dropped, and a store that would wait advances simulated time until it can
// complete (or times out). They must not be called from inside a time-base callback (agent step,
// listener): use tsr_sim_cw_access there (the simulator aborts on such a recursion).
// ---------------------------------------------------------------------------
#define TSR_CW_OK    0
#define TSR_CW_FAULT 1
#define TSR_CW_WAIT  2
int      tsr_sim_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val);
uint32_t tsr_sim_cw_read(tsr_sim * sim, uint32_t off);
void     tsr_sim_cw_write(tsr_sim * sim, uint32_t off, uint32_t val);
void     tsr_sim_cw_push_timeout(tsr_sim * sim);   // ISSUE_STATUS / STATUS PUSH_TIMEOUT, CC §7.3

// Core window (CC §5.3, 0x2000_1000 + CSR offset, word loads): the host CSR map (EX §5) read
// without side effects: no PERF snapshot (a PERF HI read returns the live high half), LOAD_DATA
// (0x0C4) and the DMI window (0xC00-0xDFC) read 0.
uint32_t tsr_sim_corewin_read(tsr_sim * sim, uint32_t csr_off);

// Run / reset state driven by the host CSR side (EX §3.4, §5.2, §8.5; CC §12.2):
//   tsr_sim_rv_run       CTRL.RV_RUN: 1 = the hart runs; 0 = held in reset (the RISC-V reset
//                        sequence completes at once in this model: RV_RESET_DONE = !RV_RUN)
//   tsr_sim_rv_boot      RV_BOOT (the hart fetches at RV_BOOT + TSR_BOOT_ENTRY_OFF)
//   tsr_sim_rv_boot_gen  +1 at every rise of RV_RUN: a hart that sees a new value restarts
//                        from its reset state (registers, CSRs, pc = RV_BOOT + 0x80)
//   tsr_sim_rv_set_state the RISC-V model reports its levels: bits TSR_STATUS_RV_HALTED and
//                        TSR_STATUS_RV_SLEEP (STATUS, DBG; QHALT gating of the PERF time counters)
//   tsr_sim_rv_ratio     F_RV / F_CORE (tsr_sim_opts.rv_ratio, default 166.7 / 276.3; the sim
//                        transport maps TSR_SIM_RV_RATIO onto it, SW §4.9)
uint32_t tsr_sim_rv_run(const tsr_sim * sim);
uint32_t tsr_sim_rv_boot(const tsr_sim * sim);
uint64_t tsr_sim_rv_boot_gen(const tsr_sim * sim);
void     tsr_sim_rv_set_state(tsr_sim * sim, uint32_t status_bits);
double   tsr_sim_rv_ratio(const tsr_sim * sim);
// The TCMs (CC §4.1-§4.2): byte arrays the loader writes (EX §3.3) and the hart fetches and
// accesses; *bytes (if not NULL) = their size (TSR_ITCM_BYTES / TSR_DTCM_BYTES by default).
uint8_t * tsr_sim_itcm(tsr_sim * sim, uint32_t * bytes);
uint8_t * tsr_sim_dtcm(tsr_sim * sim, uint32_t * bytes);
// Interrupt levels (CC §12.3), as an Ibex mip mask: bit 7 irq_timer (MTIME >= MTIMECMP), bit
// 16 + i irq_fast[i] (0 WAIT_TICKET done, 1 ISSUE_STATUS.ERR_IRQ, 2 LQ_AVAIL != 0 and not
// ACTIVE, 3 LCQ_FREE != 0, 4 DB's REC level (0 without the DB model), 5 WAIT_CREDIT).
// Recomputed from the current state at every call.
#define TSR_RV_MIP_MTIP     (1u << 7)
#define TSR_RV_MIP_FAST(i)  (1u << (16 + (i)))
uint32_t tsr_sim_rv_irq(tsr_sim * sim);

// The RISC-V agent (tsr_rv): the time base calls step(ctx, sim) whenever MTIME reaches the time
// the agent asked for, while RV_RUN = 1. step runs the hart for a while at the current time
// tsr_sim_now() (typically one instruction; its control-window and SPM accesses see the core
// state of that instant) and returns the core cycles until it wants to run again (>= 1), or
// TSR_SIM_AGENT_WAIT when it sleeps (WFI, debug halt, parked): it is then called again after the
// next core event (any completion, dispatch, timer or host action), and decides itself from
// tsr_sim_rv_irq() whether to wake. The RV-to-core clock conversion (tsr_sim_rv_ratio) is the
// agent's. NULL step removes the agent.
#define TSR_SIM_AGENT_WAIT UINT64_MAX
typedef uint64_t (*tsr_sim_agent_step)(void * ctx, tsr_sim * sim);
void tsr_sim_set_rv_agent(tsr_sim * sim, tsr_sim_agent_step step, void * ctx);
// The built-in agent: a library built with the RISC-V model (TSR_SIM_HAVE_RV: rv.cmake adds
// tsr_rv.c / tsr_bus.c) creates it in tsr_sim_create (tsr_rv_agent_create, tsr_bus.h) and destroys
// it in tsr_sim_destroy; NULL otherwise. A test that installs its own step with
// tsr_sim_set_rv_agent replaces it for the rest of the simulator's life (it is still destroyed by
// tsr_sim_destroy). tsr_rv_agent_rv(tsr_sim_rv_agent(sim)) is the hart (tsr_rv.h).
struct tsr_rv_agent;
struct tsr_rv_agent * tsr_sim_rv_agent(tsr_sim * sim);

// ---------------------------------------------------------------------------
// 5. Time base and hooks (SW §4.4)
//
// One clock: MTIME, core cycles since create (tsr_sim_now). The P1.6 policy ("fixed"):
//   - a push or QCTRL entry enters the issue FIFO at its time t and reaches the dispatcher at
//     max(t + 2 (crossing), previous dispatch + 1); the dispatcher checks it and enqueues it in
//     that cycle;
//   - each unit serves its queue in order, one entry at a time: the head starts at the first time
//     its deps (and a FENCE / QCLEAR snapshot) are done and no hold applies; a running op is done
//     at start + cycles(); a non-running completion (poisoned, ABORTED, DEP_FAILED, NOP, FENCE,
//     QCLEAR) takes 1 cycle; units run concurrently;
//   - at one time point: completions (exec, then DONE_SEQ) in unit order 0..5, then the
//     dispatch, then head starts in unit order, then the RISC-V agent;
//   - every host CSR access and control-window access acts at the current MTIME.
// RISC-V co-stepping (the built-in agent, tsr_bus.c; the placeholder for P1.8's scheduler): one
// clock, MTIME in core cycles; the agent converts with F_RV / F_CORE (tsr_sim_rv_ratio, default
// 166.7 / 276.3 MHz = 0.603, 1.0 in tests that want 1:1) from an anchor taken at each RISC-V reset.
// At its turn it runs the hart in a slice that ends before the next core event (completion,
// dispatch, MTIMECMP: tsr_sim_next_event, which leaves the running agent out) and right after the
// first access that leaves clk_rv (control window, core window, SPM bridge), so every such access
// sees the core state of its own instant and every core event is seen by the next instruction.
// Asleep (WFI), parked or with a waiting push it returns TSR_SIM_AGENT_WAIT and is stepped again
// after the next event. Deterministic for a given ratio; SW §4.7's observables do not depend on
// the ratio for a valid program (sim_rv_issue runs 0.603 and 1.0). Not modelled yet (P1.8): CDC
// lag of the RISC-V-side reads (2-8 cycle crossings), random CPI and durations, WFI / spin
// fast-forward beyond the event jump above.
// P1.8 replaces the policy (random durations and crossings, fast-forward) behind the same calls.
// Public (tsr_sim.h): tsr_sim_now, tsr_sim_advance (until a host-visible event or a budget),
// tsr_sim_run_until_idle, tsr_sim_push, tsr_sim_qctrl.
//   tsr_sim_next_event   time of the next pending event (completion, dispatch, agent, timer),
//                        UINT64_MAX if none; called from the agent's step it leaves the agent
//                        out and counts MTIMECMP (the end of the agent's slice)
//   tsr_sim_request_stop the running tsr_sim_advance returns after the current time point with
//                        TSR_SIM_EV_HALT (DB stop points, P1.12); callable from callbacks
// ---------------------------------------------------------------------------
uint64_t tsr_sim_next_event(const tsr_sim * sim);
void     tsr_sim_request_stop(tsr_sim * sim);

// Queue events (trace v2, race detector, DB history and records). ENQ: the dispatcher wrote the
// entry into its queue (code = the checker's code, 0 = runs; ext = the extents, valid only when
// code = 0); START: the unit accepted it; DONE: it completed (code 0 or the error code; ran = 1
// if the unit executed it; for a running op, exec has been applied before the call). MARK: a
// TRACE_MARK reached the dispatcher (w[1] = value, no ticket). Called synchronously from the time
// base; the callback may read state and call tsr_sim_request_stop, nothing else.
enum { TSR_QEV_ENQ = 1, TSR_QEV_START = 2, TSR_QEV_DONE = 3, TSR_QEV_MARK = 4 };
typedef struct tsr_q_event {
    uint32_t kind;              // TSR_QEV_*
    uint32_t unit;              // TSR_UNIT_*
    uint32_t ticket;            // {unit, seq}; TSR_TICKET_NONE for MARK
    uint32_t pidx;              // push index (MARK: the next push index)
    uint32_t code;
    uint32_t ran;
    uint32_t rec;               // ISSUE_PUSH.REC
    uint32_t dep[2];            // as dispatched (a BAD_DEP dep reads TSR_TICKET_NONE)
    uint64_t time;              // MTIME
    const uint32_t * w;         // the entry's words (valid during the call)
    const tsr_extents * ext;
} tsr_q_event;
typedef void (*tsr_q_listener)(void * ctx, tsr_sim * sim, const tsr_q_event * ev);
void tsr_sim_set_listener(tsr_sim * sim, tsr_q_listener fn, void * ctx);

// Debug-model hooks (DB, P1.12 / P1.13). Without them: the DBG block (0x800-0xAFC except SREG)
// and DMI (0xC00-0xDFC) read 0 and ignore writes (DBG_ID = 0: block absent, DB §10.1); DB's
// control-window registers read 0; no unit is held by DB.
typedef struct tsr_sim_dbg_hooks {
    void * ctx;
    // Host CSR 0x800-0xAFC (except SREG 0x9C0-0x9FC) and 0xC00-0xDFC. read returns 1 if it
    // handled the offset (*val set). `host` is 0 for a core-window read (no side effects).
    int      (*csr_read)(void * ctx, uint32_t off, uint32_t * val, int host);
    void     (*csr_write)(void * ctx, uint32_t off, uint32_t val);
    // Control window 0x160-0x174, 0x17C and 0x400-0x7FC (TRACE_MARK 0x178 is the queue model's).
    int      (*cw_read)(void * ctx, uint32_t off, uint32_t * val);
    void     (*cw_write)(void * ctx, uint32_t off, uint32_t val);
    // Bit u: DB's dbg_hold[u] (the hard hold of CC §9.2; ignored after a QABORT until the next
    // QCLEAR and during SR_BUSY).
    uint32_t (*unit_hold)(void * ctx);
    // SOFT_RESET step 1 (EX §8.5): clear DB's stop state.
    void     (*soft_reset)(void * ctx);
} tsr_sim_dbg_hooks;
void tsr_sim_set_dbg_hooks(tsr_sim * sim, const tsr_sim_dbg_hooks * hooks);   // NULL removes

#ifdef __cplusplus
}
#endif

// ===========================================================================
// 6. Core internals: tsr_sim.c <-> tsr_q.c only (define TSR_SIM_CORE_INTERNAL before including).
//    Other files use the accessors above; this section may change at any time.
// ===========================================================================
#ifdef TSR_SIM_CORE_INTERNAL
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TSR_SEQ_MASK   0x0FFFFFFFu   // 28-bit per-unit sequences (CC §7.5)
#define TSR_SEQ_HALF28 0x08000000u
#define TSR_Q_XING     2u            // fixed-schedule crossing, core cycles (SW §4.4)
#define TSR_QDEPTH_MAX 31u           // Q_DEPTH field width (5 bits)

// Issue-FIFO entry (CC §7.2).
typedef struct tsr_ifent {
    uint8_t  kind;          // TSR_IFIFO_KIND_OP / TSR_IFIFO_KIND_CTRL
    uint8_t  ctrl;          // CTRL: TSR_IFIFO_CTRL_CODE_*
    uint8_t  rec;           // OP: ISSUE_PUSH.REC
    uint8_t  refused;       // OP: opcode not accepted by the issue path -> SYNC, ILLEGAL_OP
    uint32_t w[8];          // OP: masked words; TRACE_MARK: w[1] = value
    uint32_t dep[2];
    uint32_t ticket;        // OP / QCLEAR
    uint32_t pidx;          // OP / QCLEAR: push index
    uint64_t t_disp;        // time the dispatcher takes it
} tsr_ifent;

// Queue entry kinds on the SYNC unit.
enum { TSR_QK_OP = 0, TSR_QK_NOP = 1, TSR_QK_FENCE = 2, TSR_QK_QCLEAR = 3 };
// Head states (Q_STATE / DBG_U_STATE).
enum { TSR_QH_EMPTY = 0, TSR_QH_DEP_WAIT = 1, TSR_QH_SNAP_WAIT = 2, TSR_QH_HELD = 3, TSR_QH_DBHOLD = 4 };

typedef struct tsr_qent {
    uint32_t    w[8];
    uint32_t    dep[2];     // TSR_TICKET_NONE for none (and for a BAD_DEP dep)
    uint32_t    seq, pidx;
    uint32_t    code, info; // dispatch result: code != 0 = poisoned
    uint8_t     kind;       // TSR_QK_*
    uint8_t     rec;
    uint32_t    snap[TSR_NUM_UNITS];   // FENCE / QCLEAR: per-unit pushes before it
    tsr_extents ext;
    uint64_t    t_enq, t_start;
} tsr_qent;

typedef struct tsr_qunit {
    uint32_t   depth;       // Q_DEPTH (0: unit absent)
    tsr_qent * ring;        // depth entries
    uint32_t   rd, n;       // front index, entries waiting
    tsr_qent   cur;         // the entry being completed (running op or 1-cycle completion)
    int        busy;        // cur valid
    int        ran;         // cur runs on the unit (else a non-running completion)
    uint32_t   cur_code;    // its completion code
    uint64_t   t_done;
    uint32_t   head;        // TSR_QH_* of the waiting head (after the last evaluation)
    uint32_t   issued;      // ISSUED_SEQ (RISC-V side, at push)
    uint32_t   cnt;         // dispatcher count (= ISSUED_SEQ at the dispatcher's position)
    uint32_t   S, D;        // accepts + non-running completions; completions (DONE_SEQ)
    uint32_t   clear_pt;
    uint32_t   err;         // error state
    uint32_t   err_seq, err_code, err_info, err_w0, err_pidx;
} tsr_qunit;

typedef void (*tsr_row_fn)(void * ctx, uint32_t lo, uint32_t hi);
struct tsr_pool;
struct tsr_cap_ch;

struct tsr_sim {
    // configuration and memories
    tsr_config cfg;
    uint32_t   num_ch;
    uint64_t   ch_bytes;
    uint64_t   limit_beats;     // min(ch_bytes/64, 2^addr_beat_w)
    uint32_t   spm_lines;
    uint8_t ** dram;
    int *      dram_fd;
    uint8_t *  spm;
    float      sreg[16];
    uint8_t *  itcm;
    uint8_t *  dtcm;
    uint32_t   itcm_bytes, dtcm_bytes;
    double     rv_ratio;

    // PERF (EX §6)
    uint64_t   perf[TSR_PERF_SLOTS];
    int        perf_force;      // tsr_sim_op: event counters not gated
    int        snap_valid;
    uint32_t   snap_idx;
    uint64_t   snap_val;

    // logging, capture, scratch, threads
    FILE *     log;
    uint64_t   n_ops;
    int        capturing;
    struct tsr_cap_ch * cap;
    void *     gaps;
    size_t     gaps_cap;
    float *    xbuf;
    float *    ybuf;
    size_t     ybuf_cap;
    void *     scratch;
    size_t     scratch_cap;
    unsigned   threads;
    struct tsr_pool * pool;

    // host CSR state (EX §5)
    uint32_t   ctrl;            // CTRL level bits
    uint32_t   sticky;          // STATUS sticky bits
    int        sr_busy;
    uint32_t   rv_boot;
    uint64_t   rv_boot_gen;
    uint32_t   rv_state;        // TSR_STATUS_RV_HALTED | TSR_STATUS_RV_SLEEP, from the RISC-V model
    uint32_t   wdog_cfg;
    uint32_t   rt_state, rt_info;
    uint32_t   load_addr, load_crc_state, load_count;

    // launch queue, completion ring, log ring (EX §4)
    uint32_t   lq_w[8];
    uint32_t   lq_ring[TSR_LQ_DEPTH][8];
    uint32_t   pushed, taken, answered;
    uint32_t   cancel_wm, stop_wm;
    int        cancel_valid, stop_valid;
    uint32_t   lcq_ring[TSR_LCQ_DEPTH][8];
    uint32_t   lcq_wr, lcq_rd;
    uint32_t   lcq_w[8];        // RISC-V staging
    uint32_t   log_ring[TSR_LOG_DEPTH];
    uint32_t   log_wr, log_rd, log_cfg, log_dropped;

    // RISC-V side issue port and queue control (CC §5.1, clk_rv registers)
    uint32_t   iss_w[8], iss_dep[2], iss_ticket, iss_status, iss_pidx;
    uint32_t   wait_ticket, wait_credit;
    uint64_t   mtimecmp;

    // queue model (tsr_q.c)
    tsr_qunit  u[TSR_NUM_UNITS];
    tsr_ifent  fifo[TSR_ISSUE_FIFO_DEPTH];
    uint32_t   f_rd, f_n;
    uint64_t   f_last;          // last dispatch time
    uint32_t   q_pidx;          // dispatcher's next push index (Q_PIDX)
    int        hold_rv;         // RISC-V hold source (QCTRL QHOLD, in push order)
    int        abort_active;    // a QABORT since the last QCLEAR (CC §9.2)
    int        err_any;         // some unit in error state (ERR_IRQ edge)

    // time base
    uint64_t   now;
    tsr_sim_agent_step agent;
    void *     agent_ctx;
    uint64_t   agent_next;
    int        agent_kick;      // something the agent may wait for changed outside an event
    tsr_q_listener lis;
    void *     lis_ctx;
    tsr_sim_dbg_hooks dbg;
    int        have_dbg;
    int        in_cb;           // inside an agent step or listener
    int        stop_req;
    struct tsr_rv_agent * rv_agent;   // built-in RISC-V agent (TSR_SIM_HAVE_RV), else NULL
};

// tsr_sim.c -> tsr_q.c
void     tsr_q_init(tsr_sim * sim, const uint8_t * depth);   // depth: 6 entries or NULL (U200)
void     tsr_q_free(tsr_sim * sim);
int      tsr_q_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val);   // CC §5.1, §5.2 timer
uint32_t tsr_q_csr_read(const tsr_sim * sim, uint32_t off);   // Q_* (0xE00-0xF1C)
uint32_t tsr_q_status(const tsr_sim * sim);                   // STATUS QIDLE, QQUIESCENT, QERR, HOLD_SRC
uint32_t tsr_q_depth_reg(const tsr_sim * sim);                // Q_DEPTH register
int      tsr_q_idle(const tsr_sim * sim);                     // QIDLE: nothing queued, running or in the FIFO
void     tsr_q_abort(tsr_sim * sim);                          // QABORT (any source), at once
int      tsr_q_host_clear(tsr_sim * sim);                     // host QCLEAR: 0 done, 1 refused
void     tsr_q_clear_all(tsr_sim * sim, const uint32_t * clear_pt);   // QCLEAR effects (NULL: current counts)
void     tsr_q_rv_reset(tsr_sim * sim);                       // clk_rv-side resets of the RISC-V reset sequence
// The opcode's handler if the issue path accepts it (legal, CAPS bits present, unit built, handler
// registered: EX §7.2-§7.3, CC §7.4), else NULL (the entry goes to SYNC with ILLEGAL_OP).
const tsr_op_handler * tsr_sim_accept_op(const tsr_sim * sim, uint32_t opcode);
// tsr_q.c -> tsr_sim.c
int      tsr_sim_lq_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val);   // 0x200-0x2FC
void     tsr_sim_sr_poll(tsr_sim * sim);                      // SOFT_RESET steps 2-4 once QIDLE
int      tsr_sim_active(const tsr_sim * sim);                 // ACTIVE = TAKEN != ANSWERED
void     tsr_sim_log_line(tsr_sim * sim, const char * fmt, ...) __attribute__((format(printf, 2, 3)));

#ifdef __cplusplus
}
#endif
#endif   // TSR_SIM_CORE_INTERNAL

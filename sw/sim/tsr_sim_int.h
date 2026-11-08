// tsr_sim_int.h — simulator-internal interface shared by the sim's source files (not installed).
//
// Owner: the core (tsr_sim.c, tsr_q.c). Users: tsr_vu.c / tsr_at.c (unit semantics, P1.4 / P1.5),
// tsr_rv.c / tsr_bus.c (the RISC-V model and its bus, P1.7), and later the scheduler (P1.8), the
// race detector (P1.9), the trace format (P1.10) and the debug models (P1.12 / P1.13). The public API is
// tsr_sim.h (host CSR port = EX §5 through tsr_sim_csr_read / tsr_sim_csr_write, time base, test
// hooks); nothing here is ABI. docs/design/06-software-verification.md (SW) §4 is the model.
//
//   1. operand extents (CC §8.2): what a checker class reports for each op;
//   2. per-opcode handler table (tsr_op_handler, tsr_sim_register_op): checker class, execution
//      and cycle model of each opcode; unit registration;
//   3. state accessors: SPM, DRAM, the s bank, PERF, capture, configuration;
//   4. RISC-V side: control window (CC §5), core window (CC §5.3), run / reset state, TCMs,
//      interrupt levels, the RISC-V agent hook the time base steps;
//   5. time base (P1.6 deterministic model) and the hooks for trace / race detector / debug models:
//      queue listeners (PUSH / QCTRL / MARK_PUSH / ENQ / START / DONE / MARK), access events, the
//      optional models attached at create (DM, DBG, RACE, TRACE) and their options;
//   5b. the schedule policy (P1.8: fixed / random, the CDC lag of the RISC-V-side reads, ties) and
//      fast-forward, TSR_SIM_LOG, ndmreset;
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
    TSR_EXT_ATT_LOCAL = 4,   // private FAST score/probability lines
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
// tsr_sim_sreg(sim) (the s bank s0..s15 as 16 floats, VU §4.4 / VU-11/12: written only by
// VU-queue ops' exec; read by the host at SREG0-15 = 0x9C0 and by the RISC-V through the core
// window; +0.0 after create, kept across SOFT_RESET, EX §7.4).
uint32_t tsr_sim_spm_lines(const tsr_sim * sim);       // SPM_BYTES / 64
uint32_t tsr_sim_num_ch(const tsr_sim * sim);
// CTL_ID (EX §5.1) of the simulator: version 2, SPM bridge, log ring, watchdog, SPM line-buffered
// alias, and bit 8 "debug module" only when the host can reach one: the DM model is attached and
// so is the DBG model, whose DMI window (0xC00-0xDFC, DB §3.2) is the host's only path to it
// (either one alone: DMI reads 0). TSR_SIM_CTL_ID_FULL is the value with every model, which the
// generated and converted trace sets record (tsr_trace2_ctl_id_full) whatever the build has.
#define TSR_SIM_CTL_ID_BASE (TSR_PUT(TSR_CTL_ID_VERSION, TSR_CTL_VERSION) | TSR_CTL_ID_BRIDGE | TSR_CTL_ID_LOG | \
                             TSR_CTL_ID_WDOG | TSR_CTL_ID_SPM_LB)
#define TSR_SIM_CTL_ID_FULL (TSR_SIM_CTL_ID_BASE | TSR_CTL_ID_DM)

// PERF (EX §6). tsr_sim_perf_add: an event counter increment, gated by ACTIVE (the queue path) or
// always counted (the tsr_sim_op test hook). Units use it for their event counters; time
// counters are the time base's. idx < TSR_PERF_SLOTS, else ignored.
void tsr_sim_perf_add(tsr_sim * sim, uint32_t idx, uint64_t n);

// Access capture hooks for exec functions (SW §5.5): a DRAM read / write of beats [lo, lo + n) on
// channel ch. Accesses reach the race detector even while capture is off. Call cap_read before
// reading (it copies the initial content when capturing) and cap_write after writing.
void tsr_sim_cap_read(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n);
void tsr_sim_cap_write(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n);
// Save preimages for masked-write replay checks without reporting a physical read to the race
// detector. Call before writing; a no-op when capture is off.
void tsr_sim_cap_preserve(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n);

// Scratch for exec functions: >= n bytes owned by the simulator, 64-byte aligned, reused by the
// next call (not preserved between ops, not thread safe).
void * tsr_sim_scratch(tsr_sim * sim, size_t n);
float * tsr_sim_att_local(tsr_sim * sim);
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
//   tsr_sim_rv_core      the Ibex configuration of the hart's cycle model (TSR_RV_CORE_* of
//                        tsr_rv.h; tsr_sim_opts.rv_core / TSR_SIM_RV_CORE, SW §4.2)
uint32_t tsr_sim_rv_run(const tsr_sim * sim);
uint32_t tsr_sim_rv_boot(const tsr_sim * sim);
uint64_t tsr_sim_rv_boot_gen(const tsr_sim * sim);
void     tsr_sim_rv_set_state(tsr_sim * sim, uint32_t status_bits);
double   tsr_sim_rv_ratio(const tsr_sim * sim);
// The VU's physical lanes (front, EXP, back) and register stages on each side of the lane array (VU §7.10): the cycle
// model's rates and latencies (tsr_vu.c, tsr_at.c). Default 16 / 16 / 8 and 0.
typedef struct tsr_vu_timing { uint32_t front, exp, back, pipe; } tsr_vu_timing;
tsr_vu_timing tsr_sim_vu_timing(const tsr_sim * sim);
uint32_t tsr_sim_rv_core(const tsr_sim * sim);
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
// One clock: MTIME, core cycles since create (tsr_sim_now). The P1.6 policy ("fixed"; P1.8's
// "random" varies durations, crossings and RV costs behind the same structure, §5b):
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
// the ratio for a valid program (sim_rv_issue runs 0.603 and 1.0; sim_sched_determinism adds 7
// random schedules at both ratios). The random policy, the CDC lag of the RISC-V-side reads and
// the spin fast-forward are §5b.
// Public (tsr_sim.h): tsr_sim_now, tsr_sim_advance (until a host-visible event or a budget),
// tsr_sim_run_until_idle, tsr_sim_push, tsr_sim_qctrl.
//   tsr_sim_next_event   time of the next pending event (completion, dispatch, agent, timer, a
//                        RISC-V-side visibility change under the random schedule), UINT64_MAX
//                        if none; called from the agent's step it leaves the agent out and
//                        counts MTIMECMP (the end of the agent's slice)
//   tsr_sim_request_stop the running tsr_sim_advance returns after the current time point with
//                        TSR_SIM_EV_HALT (DB stop points, P1.12); callable from callbacks
// ---------------------------------------------------------------------------
uint64_t tsr_sim_next_event(const tsr_sim * sim);
void     tsr_sim_request_stop(tsr_sim * sim);

// Queue events (trace format, race detector, DB history and records). Called synchronously from the
// time base (or from the push / QCTRL store that caused them); a listener may read state and call
// tsr_sim_request_stop, nothing else.
//   PUSH       an ISSUE_PUSH accepted into the issue FIFO (by the RISC-V or tsr_sim_push), at the
//              store: unit = the unit the issue path sends it to (SYNC for a refused opcode),
//              ticket, pidx = its push index, w = the words as pushed (masked words 0), dep[2] =
//              the deps as pushed (before the dispatcher's BAD_DEP rule), push = the ISSUE_PUSH
//              value, rec = its REC bit, pc = the push PC (CC §7.1; TSR_SIM_PC_HOOK for
//              tsr_sim_push and the other test hooks), code 0.
//              These and QCTRL / MARK_PUSH are the trace's push.bin records, in push order (SW §5.2)
//   QCTRL      a QCTRL write accepted into the issue FIFO (all of its entries, CC §5.1): w[0] = the
//              value written; unit SYNC with the QCLEAR's ticket and push index when bit 3 is set,
//              else unit TSR_UNIT_NONE, TSR_TICKET_NONE and pidx = the next push index; pc. A write
//              with no bit of [3:0] set enters nothing and has no event
//   MARK_PUSH  a TRACE_MARK accepted into the issue FIFO: w[0] = the value (push.bin kind 2),
//              unit TSR_UNIT_MARKER, no ticket, pidx = the next push index, pc
//   ENQ        the dispatcher wrote the entry into its queue (code = the checker's code, 0 = runs;
//              ext = the extents, valid only when code = 0; dep = as dispatched: a BAD_DEP dep
//              reads TSR_TICKET_NONE)
//   START      the unit accepted it (a running op; time = its start)
//   DONE       it completed (code 0 or the error code; ran = 1 if the unit executed it; for a
//              running op, exec has been applied before the call)
//   MARK       a TRACE_MARK reached the dispatcher (w[1] = value, no ticket)
// For ENQ / START / DONE, pc is the push PC of the entry and push its ISSUE_PUSH value.
enum {
    TSR_QEV_ENQ = 1, TSR_QEV_START = 2, TSR_QEV_DONE = 3, TSR_QEV_MARK = 4,
    TSR_QEV_PUSH = 5, TSR_QEV_QCTRL = 6, TSR_QEV_MARK_PUSH = 7,
};
typedef struct tsr_q_event {
    uint32_t kind;              // TSR_QEV_*
    uint32_t unit;              // TSR_UNIT_*
    uint32_t ticket;            // {unit, seq}; TSR_TICKET_NONE for MARK
    uint32_t pidx;              // push index (MARK: the next push index)
    uint32_t code;
    uint32_t ran;
    uint32_t rec;               // ISSUE_PUSH.REC
    uint32_t dep[2];            // ENQ / START / DONE: as dispatched (a BAD_DEP dep reads
                                // TSR_TICKET_NONE); PUSH: as pushed
    uint64_t time;              // MTIME
    const uint32_t * w;         // the entry's words (valid during the call)
    const tsr_extents * ext;
    uint32_t push;              // ISSUE_PUSH value (ops; 0 for control entries)
    uint32_t pc;                // push PC (CC §7.1, §7.2); TSR_SIM_PC_HOOK for the test hooks
} tsr_q_event;
typedef void (*tsr_q_listener)(void * ctx, tsr_sim * sim, const tsr_q_event * ev);
// Listener list: up to TSR_SIM_LISTENERS_MAX listeners, called in registration order for every
// event. add returns an id >= 0 (-1 if the list is full or fn is NULL); remove takes it (an
// unknown id is ignored; removing from inside a callback is allowed: a listener removed during an
// event is not called for the rest of it). tsr_sim_set_listener is the W1a single-listener form:
// it owns one entry of the list (NULL removes it; a new call replaces it, keeping its position
// only if fn is unchanged: it is re-added at the end otherwise).
#define TSR_SIM_LISTENERS_MAX 16
int  tsr_sim_add_listener(tsr_sim * sim, tsr_q_listener fn, void * ctx);
void tsr_sim_remove_listener(tsr_sim * sim, int id);
void tsr_sim_set_listener(tsr_sim * sim, tsr_q_listener fn, void * ctx);

// ---------------------------------------------------------------------------
// Access events (race detector SW §4.6, capture SW §5.5, DB watchpoints): every memory and
// register access by the RISC-V, the host and the debug module's system bus, synchronously at its
// MTIME (time). Tile ops are not here: their accesses are their extents (queue events).
//   RV_SPM_RD / RV_SPM_WR   the hart's SPM window loads / stores (0x4000_0000 and the 0x5000_0000
//                           alias, CC §4.1, §10.5): addr = SPM byte address, size = len = bytes
//                           (1, 2, 4; a line-buffer fill is one RV_SPM_RD of 64 at the line
//                           address, a hit is no access), val = the bytes (size <= 4), pc. A
//                           split misaligned access is two events
//   RV_CW_RD / RV_CW_WR     the hart's control-window accesses that completed (CC §5): off, val,
//                           pc (a waiting store is reported once, when it completes; a faulting
//                           access is not reported)
//   RV_CORE_RD              the hart's core-window loads (CC §5.3): off = the CSR offset, val, pc
//   HOST_CSR_RD / _WR       the host CSR port (tsr_sim_csr_read / _write, EX §5): off, val (this
//                           covers LQ_PUSH and the LCQ record reads)
//   HOST_DRAM_RD / _WR      host reads / writes of simulated DRAM through tsr_sim_dram_read /
//                           tsr_sim_dram_write / tsr_sim_dram_note_write (tsr_sim.h): ch, addr =
//                           byte offset, len. Accesses through the tsr_sim_dram() pointer are
//                           invisible unless the caller reports them with tsr_sim_dram_note_write
//   SBA_RD / SBA_WR         the debug module's system-bus accesses (DB §3.6), reported by the DM
//                           model with tsr_sim_acc_emit: addr, size, val
// Unused fields are 0. Listeners are called in registration order (at most
// TSR_SIM_LISTENERS_MAX); the same rules as the queue listeners apply.
// ---------------------------------------------------------------------------
enum {
    TSR_ACC_RV_SPM_RD = 1, TSR_ACC_RV_SPM_WR = 2,
    TSR_ACC_RV_CW_RD = 3, TSR_ACC_RV_CW_WR = 4,
    TSR_ACC_HOST_CSR_RD = 5, TSR_ACC_HOST_CSR_WR = 6,
    TSR_ACC_HOST_DRAM_RD = 7, TSR_ACC_HOST_DRAM_WR = 8,
    TSR_ACC_SBA_RD = 9, TSR_ACC_SBA_WR = 10,
    TSR_ACC_RV_CORE_RD = 11,
};
typedef struct tsr_sim_acc {
    uint32_t kind;              // TSR_ACC_*
    uint32_t size;              // bytes of an RV / SBA access
    uint32_t ch;                // DRAM channel
    uint32_t pc;                // the accessing instruction (RV kinds)
    uint64_t addr;              // SPM byte address, DRAM byte offset, SBA address
    uint64_t len;               // bytes (DRAM, SPM)
    uint32_t val;               // the value read or written (word-sized accesses)
    uint32_t off;               // control-window / CSR offset
    uint64_t time;              // MTIME
} tsr_sim_acc;
typedef void (*tsr_sim_acc_fn)(void * ctx, tsr_sim * sim, const tsr_sim_acc * a);
int  tsr_sim_add_acc_listener(tsr_sim * sim, tsr_sim_acc_fn fn, void * ctx);
void tsr_sim_remove_acc_listener(tsr_sim * sim, int id);
// Reports an access to the listeners (a copy is made; time is set to MTIME). Cheap when nothing
// listens (tsr_sim_acc_listened). Used by the bus model, the core and the DM model (SBA).
void tsr_sim_acc_emit(tsr_sim * sim, const tsr_sim_acc * a);
int  tsr_sim_acc_listened(const tsr_sim * sim);

// ---------------------------------------------------------------------------
// Optional models attached at create (P1.9, P1.10, P1.12, P1.13). Per model M a source file
// defines tsr_<m>_attach / tsr_<m>_detach; tsr_sim_create_opts calls the attach functions at its
// end in the order DM, DBG, RACE, TRACE, and tsr_sim_destroy calls the detach functions first, in
// the reverse order (before the RISC-V agent and the memories go). Each call is compiled in under
// TSR_SIM_HAVE_DM / _DBG / _RACE / _TRACE, which sw/sim/CMakeLists.txt defines when tsr_dm.c /
// tsr_dbg.c / tsr_race.c / tsr_trace.c are among the simulator's sources. A model keeps its state
// in its slot (NULL after create; the model frees it in detach) and reads its options through the
// accessors below. tsr_sim_model_slot returns NULL for an unknown id.
// ---------------------------------------------------------------------------
enum { TSR_MODEL_DM = 0, TSR_MODEL_DBG = 1, TSR_MODEL_RACE = 2, TSR_MODEL_TRACE = 3, TSR_MODEL_COUNT = 4 };
void ** tsr_sim_model_slot(tsr_sim * sim, int id);
void tsr_dm_attach(tsr_sim * sim);      void tsr_dm_detach(tsr_sim * sim);      // tsr_dm.c (P1.12)
void tsr_dbg_attach(tsr_sim * sim);     void tsr_dbg_detach(tsr_sim * sim);     // tsr_dbg.c (P1.13)
void tsr_race_attach(tsr_sim * sim);    void tsr_race_detach(tsr_sim * sim);    // tsr_race.c (P1.9)
// P1.9: tsr_sim_cap_read / _write pass every call (capture on or off) to the attached race
// detector: the exact DRAM beats a running op's exec touches (its DRAM access set, SW §4.6).
void tsr_race_cap_note(tsr_sim * sim, uint32_t ch, uint64_t lo, uint64_t n, int write);
void tsr_trace_attach(tsr_sim * sim);   void tsr_trace_detach(tsr_sim * sim);   // tsr_trace.c (P1.10)
// Options (SW §4.8 tsr_sim_opts, else the SW §4.9 environment read in tsr_sim_create_opts):
//   sched     the canonical schedule: "fixed" or "random:<seed>,<min>,<max>" (the seed in decimal,
//             min / max as %.17g of their Q16.16 values, e.g. "random:7,0.25,4"; parses back to
//             the same schedule)
//   race      0 off, 1 report, 2 fatal (TSR_SIM_RACE off / report / fatal)
//   trace_dir TSR_TRACE / tsr_sim_opts.trace_dir, or NULL
const char * tsr_sim_opt_sched(const tsr_sim * sim);
uint32_t     tsr_sim_opt_race(const tsr_sim * sim);
const char * tsr_sim_opt_trace_dir(const tsr_sim * sim);

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
    // P1.13: bit u = dbg_step[u] (CC §9.3, DB §5.2 QSTEP): the head of unit u may start or
    // complete this evaluation through DB's hard hold and the hold sources (DB arms it only while
    // the host and RISC-V holds are clear and clears it at the entry's START / DONE).
    uint32_t (*unit_step)(void * ctx);
    // P1.13: DB's STOP_ACTIVE: PERF DBG_STOP_CYCLES counts ACTIVE and (RV_HALTED or this) (EX §6).
    int      (*stop_active)(void * ctx);
    // P1.13: interrupt lines DB drives, as an Ibex mip mask (irq_fast[4], the REC level, CC §12.3),
    // ORed into tsr_sim_rv_irq.
    uint32_t (*rv_irq)(void * ctx);
    // P1.13: tsr_sim_rv_set_state changed the RISC-V's STATUS bits (called after the change, at
    // the current MTIME): DB's RV_HALTED edges (trace-writer RVHALT records).
    void     (*rv_state)(void * ctx);
} tsr_sim_dbg_hooks;
void tsr_sim_set_dbg_hooks(tsr_sim * sim, const tsr_sim_dbg_hooks * hooks);   // NULL removes

// ---------------------------------------------------------------------------
// 5b. Schedule policy (P1.8, tsr_sched.c; SW §4.4, §4.7) and fast-forward
//
// TSR_SIM_SCHED / tsr_sim_opts.sched:
//   fixed (default)            the P1.6 policy above, bit-identical to W1a: op durations from the
//                              cycle models, issue crossing 2, RISC-V-side reads see the core state
//                              of their own instant, RV CPI from the table (tsr_rv.h)
//   random:<seed>[,<min>,<max>] every running op's duration x f, f uniform over the Q16.16 values
//                              of [min, max], both included (default 0.25, 4; 0 < min <= max <=
//                              1024), rounded half up to >= 1 cycle; every
//                              crossing 2-8 core cycles (issue FIFO entry to dispatcher; the
//                              synchronised core -> RISC-V copies below; the LQ / LCQ / log, core-
//                              window and SPM-bridge handshakes as 0-6 extra core cycles of access
//                              latency); the RV cost of every instruction +-1 cycle (never below 1)
// Generator: xoshiro128** (Blackman / Vigna), one independent stream per purpose (TSR_RS_*): a
// stream's state comes from splitmix64(seed, stream), so a draw added in one place never shifts
// another's. Factors are Q16.16 integers (min / max rounded to 1/65536), durations
// round(base x f) computed in integers; no host time, no floating point in the draws.
//
// CDC lag (random only; CC §5.1 "synchronised copy; may lag, never leads", §11.2): the RISC-V side
// reads DONE_SEQ[u], ISSUE_CREDIT (S[u] and the FIFO read pointer), QSTATUS, ERR_* / ERR_PIDX and
// computes the interrupt levels irq_fast[0..3,5] (and ISSUE_STATUS.ERR_IRQ's edge) from a view of
// the core state that trails it: each change of that state at MTIME t becomes visible at
// max(t + crossing, the previous change's visibility), i.e. in order, never ahead of the core and
// with the core's own consistency (I2: an error record is visible no later than the DONE_SEQ that
// passes it). The push condition uses the same lagging credit (CC §7.1), so the queue never
// overflows. RISC-V-local registers (staging, deps, ISSUE_PIDX, ISSUED_SEQ, ISSUE_TICKET, WAIT_*,
// MTIMECMP), MTIME and the handshake registers (LQ / LCQ / log, RT_*, the core window, the SPM
// bridge) read the current core value. The view is reloaded from the core when the hart leaves
// reset (CC §12.2). Fixed keeps the W1a model (no lag) so that it stays bit-identical.
//
// Ties (SW §4.4) are broken by (time, event-kind priority, unit id, sequence): at one MTIME the
// time base runs (1) completions in unit order, (2) dispatches in issue-FIFO order, (3) head starts
// in unit order, (4) the SOFT_RESET poll, (5) RISC-V-side visibility changes in the order they were
// made, (6) the RISC-V agent; host CSR accesses act between tsr_sim_advance calls, at the current
// MTIME, after everything due at it.
//
// Fast-forward (SW §4.4): a hart asleep in WFI is stepped only at time points (it wakes at the
// first one that sets a pending enabled interrupt; the levels change only at time points, so this
// is exact). A spin is the same timing-dependent control-window register (DONE_SEQ[u],
// ISSUE_CREDIT, QSTATUS, LQ_AVAIL, LCQ_FREE, LOG_FREE) read 4 times in a row with an unchanged value
// and no store, push or other MMIO access in between (tsr_bus.c): the hart then idles to the next
// time point that may change one of them (any completion, dispatch, head start, visibility change
// or host action), re-reads, and jumps again while the value stays. Only poll counts, MTIME-derived
// values and PERF change (none is an SW §4.7 observable). tsr_sim_set_fast_forward(sim, 0)
// disables the spin jumps (the switch the determinism test compares against); default on.
// ---------------------------------------------------------------------------
typedef struct tsr_rng { uint32_t s[4]; } tsr_rng;
// xoshiro128** step (the reference algorithm); tsr_rng_seed: s from splitmix64 of (seed, stream),
// never all zero; tsr_rng_below: (next * n) >> 32 (n >= 1).
uint32_t tsr_rng_next(tsr_rng * r);
void     tsr_rng_seed(tsr_rng * r, uint64_t seed, uint32_t stream);
uint32_t tsr_rng_below(tsr_rng * r, uint32_t n);

enum {
    TSR_RS_DUR = 0,           // + unit: op durations of unit u (6 streams)
    TSR_RS_XING = 6,          // issue-FIFO crossings
    TSR_RS_VIEW = 7,          // core -> RISC-V visibility lags
    TSR_RS_HS = 8,            // handshake access latencies
    TSR_RS_CPI = 9,           // RV CPI jitter
    TSR_RS_COUNT = 10,
};
typedef struct tsr_sched_spec {
    int      random;          // 0 fixed, 1 random
    uint64_t seed;
    uint32_t fmin, fmax;      // duration factor bounds, Q16.16 (random)
} tsr_sched_spec;
// Parses a TSR_SIM_SCHED value (NULL or "" = fixed; seed decimal, or hex after 0x / 0X, 64 bits;
// a leading 0 is decimal, never octal). 0, or -1 with a message in err.
int  tsr_sched_parse(const char * text, tsr_sched_spec * out, char * err, size_t errlen);
void tsr_sched_format(const tsr_sched_spec * s, char * buf, size_t n);   // canonical form

int      tsr_sim_sched_random(const tsr_sim * sim);
// Duration of a running op of unit u whose cycle model gave base (>= 1).
uint64_t tsr_sched_op_cycles(tsr_sim * sim, uint32_t unit, uint64_t base);
// Extra RV cycles of one handshake access (LQ / LCQ / log registers, core window, SPM bridge):
// 0 fixed; random: round((x - 2) x F_RV / F_CORE) with x uniform in 2..8 core cycles.
uint32_t tsr_sched_rv_handshake(tsr_sim * sim);
// RV cost delta of one instruction: 0 fixed; random -1, 0 or +1.
int      tsr_sched_rv_cpi(tsr_sim * sim);

void     tsr_sim_set_fast_forward(tsr_sim * sim, int on);
int      tsr_sim_fast_forward(const tsr_sim * sim);

// Control-window access with the accessing instruction's pc (the push PC of CC §7.1 in PUSH /
// QCTRL / MARK_PUSH events, and the RV_CW_* access events). Two impossible pcs (bit 0 set: never
// a hart pc; 0 is one, the ITCM base and vector slot 0 with RV_BOOT = 0):
//   TSR_SIM_PC_HOOK  the test hooks and tools outside the hart: tsr_sim_cw_access, tsr_sim_cw_read
//                    / _write, tsr_sim_push, tsr_sim_qctrl (W1b fix; the race detector's harness
//                    rule keys on it)
//   TSR_SIM_PC_SBA   a debug-module system-bus access made through the bus: no RV_CW_* access
//                    event (the DM model reports SBA_* itself)
#define TSR_SIM_PC_HOOK 0xFFFFFFFDu
#define TSR_SIM_PC_SBA  0xFFFFFFFFu
int      tsr_sim_cw_access_pc(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val, uint32_t pc);
// Reports a core-window read (RV_CORE_RD access event); the agent calls it after
// tsr_sim_corewin_read.
void     tsr_sim_corewin_note(tsr_sim * sim, uint32_t csr_off, uint32_t val, uint32_t pc);

// The debug module's ndmreset (DB §3, CC §12.1-§12.2): level 1 applies the RISC-V reset sequence
// (QABORT, ANSWERED := TAKEN, the clk_rv-side resets) and holds the hart in reset (STATUS
// RV_RESET_DONE = 1, RV_RUNNING = 0, the agent is not stepped; CTRL.RV_RUN keeps its value);
// level 0 releases it: with RV_RUN = 1 the hart restarts at RV_BOOT + 0x80 (tsr_sim_rv_boot_gen
// steps). tsr_sim_advance reports RV_RESET_DONE rising as TSR_SIM_EV_RESET.
void     tsr_sim_rv_ndmreset(tsr_sim * sim, int level);
// P1.12: something a sleeping or parked RISC-V agent may wait for changed outside a time-base
// event (the DM model's resume, halt request or program buffer): the agent is stepped at the
// next time point (as after a host CSR write). Callable from callbacks.
void     tsr_sim_rv_kick(tsr_sim * sim);

// TSR_SIM_LOG (SW §4.9): the queue model writes push / qctrl / enq / start / done / mark lines and
// tsr_sim_op one line per op; with TSR_SIM_LOG_RV (tsr_sim.h) the agent also writes one line per
// retired RV instruction. Content is deterministic (MTIME, units, sequences, words, pcs).
// tsr_sim_log_line appends one line (no-op without a log).
int      tsr_sim_log_on(const tsr_sim * sim);
int      tsr_sim_log_rv(const tsr_sim * sim);
int      tsr_sim_log_words(const tsr_sim * sim);
void     tsr_sim_log_line(tsr_sim * sim, const char * fmt, ...) __attribute__((format(printf, 2, 3)));

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
    uint32_t push;          // OP: the ISSUE_PUSH value
    uint32_t pc;            // push PC (CC §7.2 [386:355])
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
    uint32_t    push, pc;   // ISSUE_PUSH value, push PC
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

// The RISC-V side's view of the core state under the random schedule (§5b): one entry per change,
// visible from `vis` on. view[v_rd] is the visible one, the v_n - 1 after it are pending.
typedef struct tsr_rvview {
    uint64_t made, vis;
    uint32_t D[TSR_NUM_UNITS], S[TSR_NUM_UNITS];
    uint32_t f_out;                    // issue-FIFO entries taken by the dispatcher (count)
    uint32_t qbits;                    // QSTATUS without QIDLE: holds, error states, non-empty queues
    uint32_t lines;                    // irq_fast[2] / [3] levels (TSR_RV_MIP_FAST bits)
    uint32_t err_any;
    uint32_t err[TSR_NUM_UNITS][6];    // error state, ERR_SEQ, ERR_CODE, ERR_INFO, ERR_W0, ERR_PIDX
} tsr_rvview;
#define TSR_VIEW_RING 16u

typedef struct tsr_lis_slot { int id; tsr_q_listener fn; void * ctx; } tsr_lis_slot;
typedef struct tsr_acc_slot { int id; tsr_sim_acc_fn fn; void * ctx; } tsr_acc_slot;

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
    float *    att_local;    // owner-banked FAST score/probability slots
    float      sreg[16];
    uint8_t *  itcm;
    uint8_t *  dtcm;
    uint32_t   itcm_bytes, dtcm_bytes;
    double     rv_ratio;
    uint32_t   rv_core;         // TSR_RV_CORE_* (tsr_rv.h)
    uint32_t   vu_front, vu_exp, vu_back, vu_pipe;   // the VU's lanes and lane-pipe stages (tsr_sim_opts.vu_timing)

    // PERF (EX §6)
    uint64_t   perf[TSR_PERF_SLOTS];
    int        perf_force;      // tsr_sim_op: event counters not gated
    int        snap_valid;
    uint32_t   snap_idx;
    uint64_t   snap_val;

    // logging, capture, scratch, threads
    FILE *     log;
    FILE *     log_owned;       // TSR_SIM_LOG=<file>: opened by create, closed by destroy
    uint32_t   log_flags;       // TSR_SIM_LOG_RV
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
    int        plat_err;        // P3.1 (transport, additive): plat_err_i level (tsr_sim_set_plat_err)
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
    uint32_t   f_in, f_out;     // entries written into / taken from the issue FIFO (counts)
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
    int        agent_wake_ev;   // P1.12: a store of the agent waits (TSR_CW_WAIT): due also at the
                                // next core event or host action, not only at its deadline
    tsr_lis_slot lis[TSR_SIM_LISTENERS_MAX];
    uint32_t   n_lis;
    int        lis_next_id, set_lis_id;   // set_lis_id: tsr_sim_set_listener's entry, -1 none
    tsr_acc_slot acc[TSR_SIM_LISTENERS_MAX];
    uint32_t   n_acc;
    int        acc_next_id;
    tsr_sim_dbg_hooks dbg;
    int        have_dbg;
    int        in_cb;           // inside an agent step or listener
    int        stop_req;
    int        ndmreset;        // tsr_sim_rv_ndmreset level
    int        ff;              // spin fast-forward enabled (tsr_sim_set_fast_forward)
    struct tsr_rv_agent * rv_agent;   // built-in RISC-V agent (TSR_SIM_HAVE_RV), else NULL

    // schedule policy (tsr_sched.c) and the RISC-V-side view (random)
    tsr_sched_spec sched;
    char       sched_text[80];
    tsr_rng    rs[TSR_RS_COUNT];
    int        view_on;
    tsr_rvview view[TSR_VIEW_RING];
    uint32_t   v_rd, v_n;

    // options read by the optional models, and their slots
    uint32_t   race;
    char *     trace_dir;
    void *     models[TSR_MODEL_COUNT];
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
int      tsr_q_host_clear(tsr_sim * sim, int rv_quiet);       // host QCLEAR: 0 done, 1 refused
void     tsr_q_clear_all(tsr_sim * sim, const uint32_t * clear_pt);   // QCLEAR effects (NULL: current counts)
void     tsr_q_rv_reset(tsr_sim * sim);                       // clk_rv-side resets of the RISC-V reset sequence
// The RISC-V-side view (random schedule, §5b): note = record the core state if it changed (after
// every change outside a time point: host CSR writes, control-window accesses, RV state reports);
// flush = the view equals the core state now (the hart leaves reset, CC §12.2). No-ops when fixed.
void     tsr_q_view_note(tsr_sim * sim);
void     tsr_q_view_flush(tsr_sim * sim);
// tsr_sched.c: sets sim->sched / sched_text / rs / view_on from a parsed spec; the issue-FIFO
// crossing of one entry (push store -> dispatcher) and the visibility lag of one view change,
// core cycles.
void     tsr_sched_init(tsr_sim * sim, const tsr_sched_spec * spec);
uint64_t tsr_sched_issue_crossing(tsr_sim * sim);
uint64_t tsr_sched_view_lag(tsr_sim * sim);
// The access listeners are called (copy of *a with time = MTIME).
void     tsr_sim_acc_call(tsr_sim * sim, const tsr_sim_acc * a);
// The opcode's handler if the issue path accepts it (legal, CAPS bits present, unit built, handler
// registered: EX §7.2-§7.3, CC §7.4), else NULL (the entry goes to SYNC with ILLEGAL_OP).
const tsr_op_handler * tsr_sim_accept_op(const tsr_sim * sim, uint32_t opcode);
// tsr_q.c -> tsr_sim.c
int      tsr_sim_lq_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val);   // 0x200-0x2FC
void     tsr_sim_sr_poll(tsr_sim * sim);                      // SOFT_RESET steps 2-4 once QIDLE
int      tsr_sim_active(const tsr_sim * sim);                 // ACTIVE = TAKEN != ANSWERED

#ifdef __cplusplus
}
#endif
#endif   // TSR_SIM_CORE_INTERNAL

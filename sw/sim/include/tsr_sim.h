// tsr_sim.h — Tesseract simulator (the oracle, spec 검증 전략; docs/v2/06-software-verification.md
// "SW" §4).
//
// The core model: SPM, per-channel DRAM, the s bank, the host CSR port (EX §5: identity, CTRL /
// STATUS, the firmware loader, the launch queue LQ, the completion ring LCQ, the log ring,
// watermarks, PERF v2, Q_*, SREG, SOFT_RESET), the RISC-V side control window (CC §5; internal
// interface in sw/sim/tsr_sim_int.h), the dispatcher with the checker, the six unit queues with
// tickets, dependencies, SYNC, errors and poisoning (tsr_q.c, EX §7-§8, CC §7-§9), the units with
// their checks and cycle models, per-channel beat counters and access capture. Every numeric
// result comes from tsr_ref.h. The RISC-V itself (tsr_rv, P1.7) plugs in as the time base's agent
// (tsr_sim_int.h); VU and AT ops plug in through the per-opcode handler table.
//
// Ops: NOP, LD, ST, QACT, MVM, FENCE (EX §7.2) here; VU / AT opcodes when their handlers are linked.
// Every other opcode is ILLEGAL_OP (END, LI and ADDI are reserved). Checks follow spec "구현
// 명확화" and EX §7.1 / §8.1: x16 (w0[31:16]) and every other field an op does not use must be zero
// (ILLEGAL_OP); check order ILLEGAL_OP -> BAD_DEP -> BAD_FMT -> K_RANGE -> T_RANGE -> OPERAND ->
// SPM_RANGE -> DRAM_RANGE; a failing op changes no state (it completes poisoned at its head).
//
// Time: one clock in core cycles (MTIME). Nothing moves unless the caller advances time
// (tsr_sim_advance, tsr_sim_run_until_idle, or a test hook that waits); CSR accesses act at the
// current time. Perf model (PERF_* CSRs; deterministic, not normative — hardware counters are
// compared against it, not bit-matched): see the "Cycle model" comment in tsr_sim.c.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_sim tsr_sim;

// CAPS bits this simulator implements: computed from the registered op handlers (TSR_CAPS_VU when
// VOP, RED, ROPE and SOP are present, TSR_CAPS_ATT when all seven AT ops are, TOPK, DQROW, VMASK).
// A config whose caps field asks for more is rejected by tsr_sim_create.
uint32_t tsr_sim_caps_implemented(void);
#define TSR_SIM_CAPS_IMPLEMENTED tsr_sim_caps_implemented()

// Options beyond the elaboration parameters (SW §4.8). Zero / NULL fields take the defaults.
typedef struct tsr_sim_opts {
    const char *    dram_dir;     // file-backed DRAM (see tsr_sim_create); NULL: anonymous memory
    uint32_t        threads;      // MVM worker threads (results independent of it); 0 = 1
    const char *    sched;        // NULL or "fixed" (P1.6); "random:<seed>[,<min>,<max>]" comes with P1.8
    double          rv_ratio;     // F_RV / F_CORE for the RISC-V agent; 0 = 166.7 / 276.3
    uint32_t        race;         // race detector (P1.9): 0 off; other values are refused until then
    void *          log;          // FILE*: per-op and queue event lines (TSR_SIM_LOG), or NULL
    const uint8_t * q_depth;      // NULL: U200 M6 depths (DMA 8, QACT 4, MVM 4, VU 16, unit 4
                                  // absent, SYNC 4); else 6 depths, 0 = unit absent, max 31, SYNC >= 1
    uint32_t        itcm_bytes;   // 0 = TSR_ITCM_BYTES; else a multiple of 1 KiB, <= 1 MiB
    uint32_t        dtcm_bytes;   // 0 = TSR_DTCM_BYTES; same rule
} tsr_sim_opts;

// Create a simulator. Channel DRAM is cfg->ch_bytes per channel, mapped lazily
// (MAP_NORESERVE) so a 4 x 16 GiB address space costs only what is touched; it starts zeroed.
// If dram_dir is non-NULL, each channel is backed by a sparse, already-unlinked temporary
// file created in dram_dir (unique name per process and channel, nothing is left behind),
// so models larger than RAM can live in the page cache.
// SPM starts filled with 0xA5 (spec: SPM contents are undefined at start; programs must not rely on them).
// Returns NULL (message on stderr) if the config is not supported: beat_bits != 512,
// num_ch not in [1,16] (TSR_NUM_CH_MAX: the LD/ST channel field is 4 bits), lanes not in
// [1,4095], t_max not in [1,8], k_max not a multiple of 256 in [256,65280], spm_bytes not a
// positive multiple of 64, vl not in [1,255], nsreg not in [1,16], ch_bytes not a positive
// multiple of 64, addr_beat_w not in [1,40], or caps beyond tsr_sim_caps_implemented().
// After create: MTIME 0, RV_RUN 0 (the RISC-V in reset), TCMs zero, queues empty, every sequence 0,
// PERF 0, s0..s15 = +0.0. tsr_sim_create(cfg, dir) = tsr_sim_create_opts(cfg, {.dram_dir = dir}).
tsr_sim * tsr_sim_create(const tsr_config * cfg, const char * dram_dir);
tsr_sim * tsr_sim_create_opts(const tsr_config * cfg, const tsr_sim_opts * opts);
void      tsr_sim_destroy(tsr_sim * sim);
const tsr_config * tsr_sim_config(const tsr_sim * sim);

// Direct DRAM access (host side). Returns a pointer to byte 0 of channel ch.
uint8_t * tsr_sim_dram(tsr_sim * sim, uint32_t ch);
uint8_t * tsr_sim_spm(tsr_sim * sim);            // debug view, spm_bytes long
// Addressable beats per channel: min(ch_bytes / 64, 2^ADDR_BEAT_W). A DRAM operand
// [a, a + n) is in range iff a + n <= this value (spec 구현 명확화, 범위 검사).
uint64_t  tsr_sim_dram_limit_beats(const tsr_sim * sim);

// Host CSR port (EX §5, the only host map): identity and configuration, CTRL / STATUS, RV_BOOT,
// RV_FREQ, RV_CFG, WDOG_CFG, RT_STATE / RT_INFO, the LQ staging and push, the sequence counters,
// the watermarks, the LCQ and log rings with their read pointers, LQ_CFG, PERF_INFO, Q_DEPTH,
// LOG_CFG / LOG_DROPPED, the loader (LOAD_*, receive CRC and VERIFY), PERF_0..63 (a LO read
// snapshots the counter for the next HI read), SREG0..15 (0x9C0), Q_* (0xE00-0xF1C). The DBG block
// and DMI read 0 until the debug models exist. Undefined offsets read 0 and ignore writes; reads
// have no side effects except the PERF snapshot. Accesses act at the current MTIME; actions that
// take time (SOFT_RESET's drain, ops started by a launch) need the caller to advance time.
uint32_t tsr_sim_csr_read(tsr_sim * sim, uint32_t off);
void     tsr_sim_csr_write(tsr_sim * sim, uint32_t off, uint32_t val);

// ---------------------------------------------------------------------------
// Time base (SW §4.4, §4.8)
// ---------------------------------------------------------------------------
#define TSR_SIM_EV_LCQ     1u   // LCQ_WR changed (a record was posted)
#define TSR_SIM_EV_LOG     2u   // log words available (LOG_WR changed)
#define TSR_SIM_EV_HALT    4u   // RV halted (debug) or parked after a TRAP record (RT_STATE TRAPPED), or a stop request
#define TSR_SIM_EV_RESET   8u   // RV_RESET_DONE rose or SR_BUSY fell
#define TSR_SIM_EV_QIDLE  16u   // STATUS.QIDLE changed
#define TSR_SIM_EV_BUDGET 32u   // max_cycles elapsed
// Runs the model from the current time until a host-visible event (returns its TSR_SIM_EV_* bits,
// possibly several) or until max_cycles core cycles have passed (TSR_SIM_EV_BUDGET). Events due at
// the current time are processed first; max_cycles = 0 processes only those.
uint32_t tsr_sim_advance(tsr_sim * sim, uint64_t max_cycles);
uint64_t tsr_sim_now(const tsr_sim * sim);                  // MTIME, core cycles
// Advances until QIDLE (nothing queued, running or in the issue FIFO) or until max_cycles passed.
// Returns 1 if idle.
int      tsr_sim_run_until_idle(tsr_sim * sim, uint64_t max_cycles);

// ---------------------------------------------------------------------------
// Test hooks: never reachable through tsr_transport (D15, V2-SW-19).
// ---------------------------------------------------------------------------
// As the RISC-V's ISSUE_W0..7 / ISSUE_DEP0 / ISSUE_DEP1 stores followed by an ISSUE_PUSH store of
// push_val ([15:8] WMASK: word i with bit i clear is sent as 0, w0 always; bit0 REC): pass 0xFF00
// to send all eight words. Without a credit it advances time like the hardware wait (CC §7.3)
// and, after ISSUE_WAIT_MAX clk_rv cycles, returns TSR_TICKET_NONE with PUSH_TIMEOUT set.
// Returns the ticket.
uint32_t tsr_sim_push(tsr_sim * sim, const uint32_t w[8], uint32_t dep0, uint32_t dep1,
                      uint32_t push_val);
// QCTRL store (CC §5.1): each set bit of [3:0] is one control entry, in bit order (QABORT, QHOLD
// set, QHOLD clear, QCLEAR). Returns the QCLEAR ticket, else TSR_TICKET_NONE.
uint32_t tsr_sim_qctrl(tsr_sim * sim, uint32_t bits);

// Checks and executes one op at once, outside the queues: w is its 8 instruction words. Returns 0
// or the op's error code; *err_info (if not NULL) receives ERR_INFO (the failing operand's start
// address for SPM_RANGE / DRAM_RANGE, the value for OPERAND, else 0). An op that fails changes no
// state except PERF. Accepts what the issue path accepts (an opcode refused there is ILLEGAL_OP).
// PERF, ungated by ACTIVE (legacy op-level view): RETIRED += 1 for every op; for a passing op
// CYCLES += 1 + cycles(op), MVM_CYCLES / VU_CYCLES += cycles(op) for MVM-unit / VU ops (and
// ATT_QK_CYCLES / ATT_PV_CYCLES for ATTQK / ATTPV), and the units' event counters (CH_RD_BEATS,
// ...). MTIME does not move.
uint32_t tsr_sim_op(tsr_sim * sim, const uint32_t w[8], uint32_t * err_info);

// Debug state.
float *  tsr_sim_sreg(tsr_sim * sim);                       // s0..s15, 16 floats (+0.0 after create)
uint64_t tsr_sim_perf(const tsr_sim * sim, uint32_t idx);   // PERF slot idx < 64 (EX §6), else 0
// RISC-V debug view (SW §4.8) of the built-in hart (a library built with the RISC-V model, tsr_rv
// + tsr_bus, creates it in tsr_sim_create; it runs on the time base while CTRL.RV_RUN = 1). 0 when
// the library has no RISC-V model. instret = instructions retired since create (model count, not
// minstret, which a RISC-V reset clears).
uint32_t tsr_sim_rv_reg(const tsr_sim * sim, unsigned i);   // x0..x31
uint32_t tsr_sim_rv_pc(const tsr_sim * sim);
uint64_t tsr_sim_rv_instret(const tsr_sim * sim);

// Optional log to a FILE* (NULL disables): one line per tsr_sim_op call, and one per queue event
// (push, enqueue, start, done) of the queue model.
void tsr_sim_set_log(tsr_sim * sim, void * file /* FILE* */);

// Worker threads used for large MVMs (rows are split across threads; results are identical
// for any thread count). Default 1. Values are clamped to [1, 64].
void tsr_sim_set_threads(tsr_sim * sim, unsigned n);

// ---------------------------------------------------------------------------
// Access capture for trace dumps (spec 트레이스 형식; trace v2 builds on it, SW §5.5)
// While capture is on, the simulator records, in beat granularity per channel:
//   - init: every DRAM range the ops READ (LD sources, MVM weight rows) whose bytes were not
//           written earlier during the capture; the bytes are copied at first read, i.e. the
//           content before execution.
//   - expect: every DRAM range the ops WRITE (ST); content taken at capture end.
// Adjacent/overlapping beats are merged into maximal ranges, sorted by (ch, offset).
// Host-side writes (tsr_sim_dram) are not tracked, so a capture should bracket exactly the ops
// of one run.
// ---------------------------------------------------------------------------
typedef struct tsr_region {
    uint32_t ch;
    uint64_t byte_off;     // multiple of 64
    uint64_t size;         // multiple of 64
    uint8_t * data;        // owned by the capture
} tsr_region;

typedef struct tsr_capture {
    tsr_region * init;   size_t n_init;
    tsr_region * expect; size_t n_expect;
} tsr_capture;

void tsr_sim_capture_begin(tsr_sim * sim);
// Ends capture and moves the recorded regions into *out (caller frees with tsr_capture_free).
// Returns an empty capture if capture was not on.
void tsr_sim_capture_end(tsr_sim * sim, tsr_capture * out);
void tsr_capture_free(tsr_capture * cap);

#ifdef __cplusplus
}
#endif

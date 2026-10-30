// tsr_rv.c — RV32IMC + Zicsr + Zifencei interpreter of the control core's Ibex (SW-1).
//
// Normative sources: docs/design/06-software-verification.md (SW) §4.2, docs/design/04-control-core.md
// (CC) §3 (Ibex configuration), §7.3 (waiting pushes), §12.3 (interrupts), docs/design/05-debug.md (DB)
// §3 (debug mode use), the RISC-V unprivileged / privileged specifications (20191213 / 1.12) and
// the Debug Specification 0.13.2. Where the specifications leave a choice to the implementation,
// this file follows Ibex; such facts are recalled from Ibex's documentation and RTL and marked
// *confirm*: they are checked against the pinned Ibex when it is vendored (CC §3.1) and by the
// RVFI lockstep bench (P5.2).
//
// Ibex behaviour modelled here (CC §3.2 configuration: WritebackStage 1, BranchTargetALU 1,
// RV32MSingleCycle (CC-34; only the cycle tables of tsr_rv.h depend on these three), RV32BNone,
// PMPEnable 0, MHPMCounterNum 0, DbgTriggerEn 1, DbgHwBreakNum 2, SecureIbex 0, ICache 0):
//   ISA        RV32I, M, C (no Zcb / Zba / Zbb), Zicsr, Zifencei. FENCE and FENCE.I are no-ops
//              (no cache; SW §4.2). Every other encoding is an illegal instruction (mcause 2,
//              mtval = the instruction bits, 16-bit ones zero-extended). Illegal compressed
//              forms: all-zero, C.ADDI4SPN with nzuimm 0, C.ADDI16SP / C.LUI with nzimm 0, shifts
//              with shamt[5] = 1, C.LWSP rd 0, C.JR rs1 0, the FP and RV64 forms and quadrant-0
//              funct3 100 (Zcb). SYSTEM funct3 0 needs rd = rs1 = 0 (*confirm*).
//   Privilege  M and U (Ibex always has U, misa.U = 1). mstatus: MIE, MPIE, MPP (WARL: values other
//              than M / U become U, *confirm*), MPRV (stored, no effect without PMP), TW. Reset
//              mstatus = MPIE with MPP = U (*confirm*): firmware must set MPP before an mret.
//              U-mode: every CSR access, mret and (with TW) WFI are illegal; ecall gives cause 8.
//   CSRs       mstatus misa(ro, 0x4010_1104: MXL 1, I M C U) mie mtvec mcounteren(0) menvcfg(h)(0)
//              mcountinhibit mhpmevent3-31(0) mscratch mepc mcause mtval mip(ro) tselect tdata1-3
//              mcontext / scontext(0) dcsr dpc dscratch0-1 cpuctrlsts secureseed(0) mcycle(h)
//              minstret(h) mhpmcounter3-31(h)(0) mvendorid(0) marchid(22) mimpid(0) mhartid(0)
//              mconfigptr(0). Not present (illegal): PMP CSRs and mseccfg (PMPEnable 0), the
//              unprivileged counters cycle / time / instret (*confirm*), every S-mode CSR.
//              Writes to a read-only CSR number ([11:10] = 3) are illegal; writes to misa and mip
//              are ignored. WARL: mtvec = {w[31:8], 0x01} (vectored only); mepc and dpc clear bit 0;
//              mcause keeps bit 31 and bits [4:0]; mie keeps MSIE, MTIE, MEIE and the 15 fast
//              bits; mcountinhibit keeps CY (0) and IR (2).
//   Counters   mcycle counts RV cycles (also asleep, stalled and in debug mode: Ibex has no
//              stopcount and its clock gate is a pass-through on the FPGA, CC §3.1); minstret
//              counts retired instructions (not trapped ones; program-buffer instructions count,
//              the debug ROM is not modelled). A CSR write to a counter suppresses that counter's
//              increment for the writing instruction.
//   cpuctrlsts sync_exc_seen (bit 6) set by each synchronous exception and cleared by mret;
//              double_fault_seen (bit 7) set by an exception while sync_exc_seen is set; both
//              writable; every other field reads 0 in this configuration (*confirm*).
//   Traps      exceptions to mtvec[31:8], interrupt c to mtvec[31:8] + 4c (vectored only;
//              reset entry is + 0x80). mtval: illegal = instruction bits; fetch fault = pc, or
//              pc + 2 if the second half of an uncompressed instruction faults; load / store
//              fault = the access address, or the word-aligned address of the second half of a
//              split access when only that half faults (*confirm*: Ibex's LSU addr_last; SW §4.2
//              says "the original address"); ecall, ebreak and interrupts write 0 (*confirm*).
//   Misaligned an access that crosses a word boundary is split into two word transactions in
//              address order; both are always issued, also when the first one faults (Ibex
//              issues the second while the first is outstanding), so a split store may write one
//              half and still fault. Accesses inside one word (e.g. a half at offset 1) are one
//              transaction with byte enables. Nothing ever raises a misaligned exception (with C,
//              jump targets only clear bit 0).
//   Interrupts levels sampled at instruction boundaries; taken when (mip & mie) != 0 and (priv U
//              or mstatus.MIE), not in debug mode, not while single-stepping, not while a store
//              waits (CC §7.3). Priority: irq_fast[0] .. irq_fast[14], MEI, MSI, MTI. mepc = the
//              next instruction. WFI: sleeps until (mip & mie) != 0 whatever mstatus.MIE is, or a
//              debug request; a no-op in debug mode and while single-stepping.
//   Debug      entry before an instruction, cause priority (Ibex DBG_TAKEN_IF, *confirm*; Debug
//              Spec 0.13.2 priorities trigger > ebreak > haltreq > step): trigger (an enabled
//              execute trigger matches the pc of the next instruction; it does not execute; also
//              right after a dret / resume to that pc), halt request (a level; also wakes WFI;
//              dpc = the next instruction), step (dcsr.step at the resume: the one instruction
//              that ran may have trapped, dpc = the next instruction, i.e. the handler then). A
//              debug entry has priority over interrupts. ebreak with dcsr.ebreakm (M) / ebreaku
//              (U) enters when it executes (dpc = the ebreak). Not while a store waits (CC §7.3;
//              DB §3.4: the halt follows the push). Entry saves dpc, dcsr.cause and dcsr.prv and
//              sets priv M; in debug mode interrupts are off, ebreak re-parks without touching
//              dpc / dcsr, an exception parks without touching any CSR (DmExceptionAddr), dret
//              leaves (illegal outside debug mode). While single-stepping interrupts are off and
//              WFI is a no-op (Ibex stepie = 0). dcsr: xdebugver 4, ebreakm, ebreaku, cause (read
//              only), step, prv (M / U; other values become M); stepie, stopcount, stoptime,
//              mprven, nmip, ebreaks read 0 and ignore writes (Ibex forces them, *confirm*), so
//              mcycle / minstret also count in debug mode (P1.12 decision: stopcount hardwired
//              0; DB §3.4's server reads it back). Triggers: tselect 0..1 (larger writes select
//              1), tdata1 reads 0x2800_1040 | execute << 2 (type 2, dmode 1, action 1 = enter
//              debug mode, m; *confirm* the u bit), tdata2 = address, tdata3 reads 0; tselect and
//              the trigger CSRs are writable in debug mode only (dmode = 1; M-mode writes are
//              ignored, *confirm*: Ibex gates tselect_we with debug_mode too).
//   Hooks      debug events (tsr_rv_set_debug_hook) ENTER / PARK / RESUME / RESET for the debug-
//              module model (tsr_dm.c, P1.12); data watch (tsr_rv_set_data_watch) on every data
//              transaction the hart issues, for DB's RV watchpoints (tsr_dbg.c, P1.13).
//
// Model-only conventions: the debug ROM's park loop is not interpreted (see tsr_rv.h);
// x1..x31 reset to 0. Timing: the cycle tables of tsr_rv.h (Ibex "small" and "maxperf", SW §4.2,
// selected by tsr_rv_set_core) with the two neighbour-dependent penalties computed here: a load
// whose rd the next instruction reads (load_use, WritebackStage) and a control transfer to a 32-bit
// instruction at pc[1] = 1 (target32); both peek at instruction memory without side effects.

#include "tsr_rv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONT 0xFFFFFFFFu   // rv_boundary: keep running

#define MISA_VALUE     0x40101104u     // MXL = 1 (RV32), U (20), M (12), I (8), C (2)
#define MARCHID_IBEX   22u             // lowRISC Ibex (RISC-V marchid registry)
#define MIE_MASK       0x7FFF0888u     // MSIE, MTIE, MEIE, fast 0..14
#define MSTATUS_MASK   (TSR_RV_MSTATUS_MIE | TSR_RV_MSTATUS_MPIE | TSR_RV_MSTATUS_MPP | \
                        TSR_RV_MSTATUS_MPRV | TSR_RV_MSTATUS_TW)
#define MPP_SHIFT      11u
#define DCSR_XDEBUGVER (4u << 28)
#define DCSR_EBREAKM   (1u << 15)
#define DCSR_EBREAKU   (1u << 12)
#define DCSR_CAUSE_SH  6u
#define DCSR_CAUSE     (7u << 6)
#define DCSR_STEP      (1u << 2)
#define DCSR_PRV       3u
#define TDATA1_BASE    0x28001040u     // type 2, dmode, action 1, m (execute at bit 2)
#define NUM_TRIGGERS   2u
#define CPUCTRL_SYNC_EXC     (1u << 6)
#define CPUCTRL_DOUBLE_FAULT (1u << 7)

struct tsr_rv {
    tsr_rv_bus bus;
    tsr_rv_cpi cpi;
    tsr_rv_stats st;

    uint32_t x[32];
    uint32_t pc;
    uint32_t priv;

    // CSRs (only the implemented bits are stored).
    uint32_t mstatus, mie, mtvec, mscratch, mepc, mcause, mtval, mcountinhibit, cpuctrlsts;
    uint64_t mcycle, minstret;
    uint32_t dcsr, dpc, dscratch[2];
    uint32_t tselect, tm_exec[NUM_TRIGGERS], tm_addr[NUM_TRIGGERS];

    // Inputs.
    uint32_t lines;          // irq levels sampled at the last boundary
    int      debug_req;

    // Execution state.
    int      dbg_mode;       // in debug mode
    int      parked;         // in debug mode, waiting for the debug module
    int      sleeping;       // WFI
    int      stepping;       // resumed with dcsr.step: the next instruction is a single step
    int      step_done;      // ... and it has executed: enter debug mode at the next boundary
    int      waiting;        // a store returned TSR_RV_BUS_WAIT; re-executed by the next step
    int      intr_next;      // the next record is the first instruction after an interrupt entry
    int      stop_req;
    int      wait_frozen;    // tsr_rv_freeze_wait
    uint64_t wait_cycles;
    uint64_t cycles, retired, order;

    tsr_rv_retire_fn retire_fn;  void * retire_ctx;
    tsr_rv_debug_fn  debug_fn;   void * debug_ctx;
    tsr_rv_data_watch_fn watch_fn; void * watch_ctx;
};

// One instruction in flight.
typedef struct ix {
    uint32_t insn, len, next_pc;
    uint32_t cost;
    int      status;          // 0, TSR_RV_BUS_WAIT or TSR_RV_BUS_YIELD
    int      exc;             // exception raised
    uint32_t cause, tval;
    uint32_t rd, rd_val;      // rd 0: no write
    uint32_t mem_addr, mem_val, mem_size, mem_write;
    uint32_t csr, csr_val, csr_written;
    int      wrote_mcycle, wrote_minstret;
    int      no_retire;       // completed but not retired: 1 ebreak in debug mode (parks), 2 ebreak into debug mode
    uint32_t cls;             // TSR_RV_CLS_* (statistics)
    int      io;              // a data transaction had a latency above 1 (left the single-cycle memories)
    int      pen_lu, pen_t32, pen_csr;   // penalties added to cost (statistics)
} ix;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static inline uint32_t sext(uint32_t v, unsigned bits) {
    uint32_t m = 1u << (bits - 1u);
    v &= (bits == 32u) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    return (v ^ m) - m;
}
static inline uint32_t sra32(uint32_t a, uint32_t sh) {
    sh &= 31u;
    if (sh == 0u) return a;
    uint32_t r = a >> sh;
    return (a & 0x80000000u) ? (r | ~(0xFFFFFFFFu >> sh)) : r;
}
static inline uint32_t bits(uint32_t v, unsigned hi, unsigned lo) {
    return (v >> lo) & ((hi - lo == 31u) ? 0xFFFFFFFFu : ((1u << (hi - lo + 1u)) - 1u));
}
static inline uint32_t ctz32(uint32_t v) {
    uint32_t n = 0;
    while (!(v & 1u)) { v >>= 1; ++n; }
    return n;
}

static void add_cycles(tsr_rv * rv, uint64_t n, int suppress_mcycle) {
    rv->cycles += n;
    if (!suppress_mcycle && !(rv->mcountinhibit & 1u)) rv->mcycle += n;
}

static void debug_event(tsr_rv * rv, uint32_t ev, uint32_t arg) {
    if (rv->debug_fn) rv->debug_fn(rv->debug_ctx, ev, arg);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
tsr_rv * tsr_rv_create(const tsr_rv_bus * bus) {
    tsr_rv * rv = (tsr_rv *)calloc(1, sizeof(*rv));
    if (rv == NULL) {
        fprintf(stderr, "tsr_rv: out of memory\n");
        abort();
    }
    if (bus) rv->bus = *bus;
    tsr_rv_cpi cpi = TSR_RV_CPI_DEFAULT;
    rv->cpi = cpi;
    tsr_rv_reset(rv, 0);
    return rv;
}

void tsr_rv_destroy(tsr_rv * rv) { free(rv); }

void tsr_rv_reset(tsr_rv * rv, uint32_t boot) {
    uint32_t base = boot & ~0xFFu;
    memset(rv->x, 0, sizeof(rv->x));
    rv->pc = base + 0x80u;
    rv->priv = TSR_RV_PRIV_M;
    rv->mstatus = TSR_RV_MSTATUS_MPIE;        // MPP = U (*confirm*)
    rv->mie = 0; rv->mtvec = base | 1u;
    rv->mscratch = rv->mepc = rv->mcause = rv->mtval = 0;
    rv->mcountinhibit = 0; rv->cpuctrlsts = 0;
    rv->mcycle = rv->minstret = 0;
    rv->dcsr = DCSR_XDEBUGVER | TSR_RV_PRIV_M;
    rv->dpc = 0; rv->dscratch[0] = rv->dscratch[1] = 0;
    rv->tselect = 0;
    for (unsigned i = 0; i < NUM_TRIGGERS; ++i) { rv->tm_exec[i] = 0; rv->tm_addr[i] = 0; }
    rv->dbg_mode = rv->parked = rv->sleeping = 0;
    rv->stepping = rv->step_done = rv->waiting = rv->intr_next = 0;
    rv->wait_cycles = 0;
    // The hart leaves reset here (the debug_req level is an input and survives): a halt request
    // raised by this event is taken before the first instruction (DB §3.8 halt-on-boot).
    debug_event(rv, TSR_RV_DBG_EV_RESET, boot);
}

void tsr_rv_set_retire_hook(tsr_rv * rv, tsr_rv_retire_fn fn, void * ctx) { rv->retire_fn = fn; rv->retire_ctx = ctx; }
void tsr_rv_set_debug_hook(tsr_rv * rv, tsr_rv_debug_fn fn, void * ctx)   { rv->debug_fn = fn; rv->debug_ctx = ctx; }
void tsr_rv_set_data_watch(tsr_rv * rv, tsr_rv_data_watch_fn fn, void * ctx) { rv->watch_fn = fn; rv->watch_ctx = ctx; }
void tsr_rv_set_cpi(tsr_rv * rv, const tsr_rv_cpi * cpi) {
    if (cpi) rv->cpi = *cpi;
    else { tsr_rv_cpi d = TSR_RV_CPI_DEFAULT; rv->cpi = d; }
}

int tsr_rv_core_parse(const char * name, uint32_t * core) {
    if (name == NULL) return -1;
    if (strcmp(name, "small") == 0)   { if (core) *core = TSR_RV_CORE_SMALL;   return 0; }
    if (strcmp(name, "maxperf") == 0) { if (core) *core = TSR_RV_CORE_MAXPERF; return 0; }
    return -1;
}

const char * tsr_rv_core_name(uint32_t core) {
    return core == TSR_RV_CORE_SMALL ? "small" : core == TSR_RV_CORE_MAXPERF ? "maxperf" : NULL;
}

void tsr_rv_core_cpi(uint32_t core, tsr_rv_cpi * out) {
    static const tsr_rv_cpi small = TSR_RV_CPI_SMALL, maxperf = TSR_RV_CPI_MAXPERF, dflt = TSR_RV_CPI_DEFAULT;
    *out = core == TSR_RV_CORE_SMALL ? small : core == TSR_RV_CORE_MAXPERF ? maxperf : dflt;
}

void tsr_rv_set_core(tsr_rv * rv, uint32_t core) { tsr_rv_core_cpi(core, &rv->cpi); }

void tsr_rv_get_stats(const tsr_rv * rv, tsr_rv_stats * out) { *out = rv->st; }

uint32_t tsr_rv_reg(const tsr_rv * rv, unsigned i) { return i < 32u ? rv->x[i] : 0u; }
void tsr_rv_set_reg(tsr_rv * rv, unsigned i, uint32_t v) { if (i > 0u && i < 32u) rv->x[i] = v; }
uint32_t tsr_rv_pc(const tsr_rv * rv) { return rv->parked ? TSR_RV_DM_HALT_ADDR : rv->pc; }
void tsr_rv_set_pc(tsr_rv * rv, uint32_t pc) { rv->pc = pc & ~1u; }
uint32_t tsr_rv_priv(const tsr_rv * rv) { return rv->priv; }
uint64_t tsr_rv_cycles(const tsr_rv * rv) { return rv->cycles; }
uint64_t tsr_rv_retired(const tsr_rv * rv) { return rv->retired; }
int tsr_rv_sleeping(const tsr_rv * rv) { return rv->sleeping; }
int tsr_rv_waiting(const tsr_rv * rv) { return rv->waiting; }
uint64_t tsr_rv_wait_cycles(const tsr_rv * rv) { return rv->wait_cycles; }
int tsr_rv_debug_mode(const tsr_rv * rv) { return rv->dbg_mode; }
int tsr_rv_halted(const tsr_rv * rv) { return rv->parked; }
void tsr_rv_set_debug_req(tsr_rv * rv, int level) { rv->debug_req = level != 0; }
void tsr_rv_request_stop(tsr_rv * rv) { rv->stop_req = 1; }

void tsr_rv_idle(tsr_rv * rv, uint64_t n) {
    add_cycles(rv, n, 0);
    if (rv->waiting && !rv->wait_frozen) rv->wait_cycles += n;
}
void tsr_rv_freeze_wait(tsr_rv * rv, int on) { rv->wait_frozen = on != 0; }

// ---------------------------------------------------------------------------
// CSRs
// ---------------------------------------------------------------------------
// Access check common to instructions and the debug API: 0 or -1 (illegal). `dbg` = the access
// executes in debug mode (M privileges, debug CSRs reachable).
static int csr_check(const tsr_rv * rv, uint32_t csr, int write, int dbg) {
    uint32_t priv = dbg ? TSR_RV_PRIV_M : rv->priv;
    if (((csr >> 8) & 3u) > priv) return -1;
    if (write && ((csr >> 10) & 3u) == 3u) return -1;
    if (csr >= 0x7B0u && csr <= 0x7B3u && !dbg) return -1;
    return 0;
}

static int csr_read_raw(const tsr_rv * rv, uint32_t csr, uint32_t * val) {
    uint32_t v = 0;
    switch (csr) {
    case 0x300: v = rv->mstatus; break;
    case 0x301: v = MISA_VALUE; break;
    case 0x304: v = rv->mie; break;
    case 0x305: v = rv->mtvec; break;
    case 0x306: case 0x30A: case 0x31A: v = 0; break;          // mcounteren, menvcfg(h)
    case 0x320: v = rv->mcountinhibit; break;
    case 0x340: v = rv->mscratch; break;
    case 0x341: v = rv->mepc; break;
    case 0x342: v = rv->mcause; break;
    case 0x343: v = rv->mtval; break;
    case 0x344: v = rv->lines & MIE_MASK; break;
    case 0x7A0: v = rv->tselect; break;
    case 0x7A1: v = TDATA1_BASE | (rv->tm_exec[rv->tselect] << 2); break;
    case 0x7A2: v = rv->tm_addr[rv->tselect]; break;
    case 0x7A3: case 0x7A8: case 0x7AA: v = 0; break;          // tdata3, mcontext, scontext
    case 0x7B0: v = rv->dcsr; break;
    case 0x7B1: v = rv->dpc; break;
    case 0x7B2: v = rv->dscratch[0]; break;
    case 0x7B3: v = rv->dscratch[1]; break;
    case 0x7C0: v = rv->cpuctrlsts; break;
    case 0x7C1: v = 0; break;                                   // secureseed
    case 0xB00: v = (uint32_t)rv->mcycle; break;
    case 0xB02: v = (uint32_t)rv->minstret; break;
    case 0xB80: v = (uint32_t)(rv->mcycle >> 32); break;
    case 0xB82: v = (uint32_t)(rv->minstret >> 32); break;
    case 0xF11: case 0xF13: case 0xF14: case 0xF15: v = 0; break;   // mvendorid mimpid mhartid mconfigptr
    case 0xF12: v = MARCHID_IBEX; break;
    default:
        if ((csr >= 0x323u && csr <= 0x33Fu) ||                 // mhpmevent3-31
            (csr >= 0xB03u && csr <= 0xB1Fu) ||                 // mhpmcounter3-31
            (csr >= 0xB83u && csr <= 0xB9Fu)) {                 // mhpmcounter3h-31h
            v = 0;
            break;
        }
        return -1;
    }
    *val = v;
    return 0;
}

// Writes an existing, writable CSR (after csr_check and csr_read_raw succeeded), applying its WARL
// rule. `dbg`: the write executes in debug mode (trigger CSRs are writable only then). cur (may be
// NULL): the instruction, told which counters it wrote.
static void csr_write_raw(tsr_rv * rv, uint32_t csr, uint32_t w, int dbg, ix * cur) {
    switch (csr) {
    case 0x300: {
        uint32_t m = w & MSTATUS_MASK;
        uint32_t mpp = (m >> MPP_SHIFT) & 3u;
        if (mpp != TSR_RV_PRIV_M && mpp != TSR_RV_PRIV_U) m &= ~TSR_RV_MSTATUS_MPP;   // -> U (*confirm*)
        rv->mstatus = m;
        break;
    }
    case 0x304: rv->mie = w & MIE_MASK; break;
    case 0x305: rv->mtvec = (w & ~0xFFu) | 1u; break;
    case 0x320: rv->mcountinhibit = w & 5u; break;
    case 0x340: rv->mscratch = w; break;
    case 0x341: rv->mepc = w & ~1u; break;
    case 0x342: rv->mcause = w & 0x8000001Fu; break;
    case 0x343: rv->mtval = w; break;
    case 0x7A0: if (dbg) rv->tselect = (w < NUM_TRIGGERS) ? w : NUM_TRIGGERS - 1u; break;
    case 0x7A1: if (dbg) rv->tm_exec[rv->tselect] = (w >> 2) & 1u; break;
    case 0x7A2: if (dbg) rv->tm_addr[rv->tselect] = w; break;
    case 0x7B0: {
        uint32_t prv = w & DCSR_PRV;
        if (prv != TSR_RV_PRIV_M && prv != TSR_RV_PRIV_U) prv = TSR_RV_PRIV_M;
        // stepie, stopcount, stoptime, mprven, nmip, ebreaks: forced 0 (Ibex, *confirm*).
        rv->dcsr = DCSR_XDEBUGVER | (w & (DCSR_EBREAKM | DCSR_EBREAKU | DCSR_STEP)) |
                   (rv->dcsr & DCSR_CAUSE) | prv;
        break;
    }
    case 0x7B1: rv->dpc = w & ~1u; break;
    case 0x7B2: rv->dscratch[0] = w; break;
    case 0x7B3: rv->dscratch[1] = w; break;
    case 0x7C0: rv->cpuctrlsts = w & (CPUCTRL_SYNC_EXC | CPUCTRL_DOUBLE_FAULT); break;
    case 0xB00: rv->mcycle = (rv->mcycle & 0xFFFFFFFF00000000ull) | w; if (cur) cur->wrote_mcycle = 1; break;
    case 0xB80: rv->mcycle = (rv->mcycle & 0xFFFFFFFFull) | ((uint64_t)w << 32); if (cur) cur->wrote_mcycle = 1; break;
    case 0xB02: rv->minstret = (rv->minstret & 0xFFFFFFFF00000000ull) | w; if (cur) cur->wrote_minstret = 1; break;
    case 0xB82: rv->minstret = (rv->minstret & 0xFFFFFFFFull) | ((uint64_t)w << 32); if (cur) cur->wrote_minstret = 1; break;
    default: break;   // misa, mip, mcounteren, menvcfg, hpm*, tdata3, contexts, secureseed: ignored
    }
}

int tsr_rv_csr_read(tsr_rv * rv, uint32_t csr, uint32_t * val) {
    uint32_t v;
    if (csr > 0xFFFu || csr_check(rv, csr, 0, 1) || csr_read_raw(rv, csr, &v)) return -1;
    if (val) *val = v;
    return 0;
}

int tsr_rv_csr_write(tsr_rv * rv, uint32_t csr, uint32_t val) {
    uint32_t v;
    if (csr > 0xFFFu || csr_check(rv, csr, 1, 1) || csr_read_raw(rv, csr, &v)) return -1;
    csr_write_raw(rv, csr, val, 1, NULL);
    return 0;
}

// ---------------------------------------------------------------------------
// Traps, interrupts, debug mode
// ---------------------------------------------------------------------------
static void trap_enter(tsr_rv * rv, uint32_t cause, uint32_t tval, uint32_t epc, int irq) {
    rv->mepc = epc & ~1u;
    rv->mcause = (irq ? TSR_RV_MCAUSE_IRQ : 0u) | (cause & 0x1Fu);
    rv->mtval = tval;
    uint32_t ms = rv->mstatus & ~(TSR_RV_MSTATUS_MPIE | TSR_RV_MSTATUS_MPP | TSR_RV_MSTATUS_MIE);
    if (rv->mstatus & TSR_RV_MSTATUS_MIE) ms |= TSR_RV_MSTATUS_MPIE;
    ms |= rv->priv << MPP_SHIFT;
    rv->mstatus = ms;
    rv->priv = TSR_RV_PRIV_M;
    if (!irq) {
        if (rv->cpuctrlsts & CPUCTRL_SYNC_EXC) rv->cpuctrlsts |= CPUCTRL_DOUBLE_FAULT;
        rv->cpuctrlsts |= CPUCTRL_SYNC_EXC;
    }
    rv->pc = (rv->mtvec & ~0xFFu) + (irq ? 4u * (cause & 0x1Fu) : 0u);
}

static void enter_debug(tsr_rv * rv, uint32_t cause, uint32_t dpc) {
    if (!rv->dbg_mode) {
        rv->dpc = dpc & ~1u;
        rv->dcsr = (rv->dcsr & ~(DCSR_CAUSE | DCSR_PRV)) | (cause << DCSR_CAUSE_SH) | rv->priv;
        rv->priv = TSR_RV_PRIV_M;
        rv->dbg_mode = 1;
    }
    rv->parked = 1;
    rv->sleeping = rv->stepping = rv->step_done = 0;
    rv->pc = TSR_RV_DM_HALT_ADDR;
    add_cycles(rv, rv->cpi.trap, 0);
    debug_event(rv, TSR_RV_DBG_EV_ENTER, cause);
}

static void leave_debug(tsr_rv * rv) {
    rv->pc = rv->dpc;
    rv->priv = (rv->dcsr & DCSR_PRV) == TSR_RV_PRIV_U ? TSR_RV_PRIV_U : TSR_RV_PRIV_M;
    if (rv->priv != TSR_RV_PRIV_M) rv->mstatus &= ~TSR_RV_MSTATUS_MPRV;
    rv->dbg_mode = rv->parked = 0;
    rv->stepping = (rv->dcsr & DCSR_STEP) != 0;
    rv->step_done = 0;
    debug_event(rv, TSR_RV_DBG_EV_RESUME, 0);
}

int tsr_rv_debug_resume(tsr_rv * rv) {
    if (!rv->parked) return -1;
    leave_debug(rv);
    return 0;
}

int tsr_rv_debug_exec(tsr_rv * rv, uint32_t pc) {
    if (!rv->parked) return -1;
    rv->parked = 0;
    rv->pc = pc & ~1u;
    return 0;
}

static int trigger_match(const tsr_rv * rv, uint32_t pc) {
    for (unsigned i = 0; i < NUM_TRIGGERS; ++i)
        if (rv->tm_exec[i] && rv->tm_addr[i] == pc) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------
static int fetch16(tsr_rv * rv, uint32_t addr, uint16_t * p) {
    uint32_t o = addr - rv->bus.fetch_base;
    if (rv->bus.fetch_mem && o < rv->bus.fetch_size && rv->bus.fetch_size - o >= 2u) {
        *p = (uint16_t)(rv->bus.fetch_mem[o] | (rv->bus.fetch_mem[o + 1u] << 8));
        return 0;
    }
    if (rv->bus.fetch == NULL) return TSR_RV_BUS_FAULT;
    return rv->bus.fetch(rv->bus.ctx, addr, (uint32_t)rv->dbg_mode, p) < 0 ? TSR_RV_BUS_FAULT : 0;
}

static int bus_xfer(tsr_rv * rv, int write, uint32_t addr, uint32_t n, uint32_t * v) {
    if (write) return rv->bus.store ? rv->bus.store(rv->bus.ctx, addr, n, (uint32_t)rv->dbg_mode, *v) : TSR_RV_BUS_FAULT;
    return rv->bus.load ? rv->bus.load(rv->bus.ctx, addr, n, (uint32_t)rv->dbg_mode, v) : TSR_RV_BUS_FAULT;
}

// Data watch (tsr_rv_set_data_watch, DB §3.7): one call per issued transaction, while the
// instruction is in flight (rv->pc = its pc).
static inline void data_watch(tsr_rv * rv, uint32_t addr, uint32_t n, int write) {
    if (rv->watch_fn) rv->watch_fn(rv->watch_ctx, addr, n, write, rv->pc);
}

// One load or store of `size` bytes at addr. Returns 0 on success; on WAIT / YIELD sets
// c->status; on a fault raises the exception in c. *v: the store value / the loaded bytes.
static int data_access(tsr_rv * rv, ix * c, uint32_t addr, uint32_t size, int write, uint32_t * v) {
    uint32_t off = addr & 3u;
    c->mem_addr = addr; c->mem_size = size; c->mem_write = (uint32_t)write;
    if (off + size <= 4u) {
        uint32_t val = write ? *v : 0u;
        int r = bus_xfer(rv, write, addr, size, &val);
        if (r == TSR_RV_BUS_WAIT || r == TSR_RV_BUS_YIELD) { c->status = r; return -1; }
        data_watch(rv, addr, size, write);
        if (r < 0) {
            c->exc = 1; c->cause = write ? TSR_RV_EXC_STORE_FAULT : TSR_RV_EXC_LOAD_FAULT; c->tval = addr;
            return -1;
        }
        c->cost += (uint32_t)r;
        if (r > 1) c->io = 1;
        if (!write) *v = val;
        c->mem_val = val;
        return 0;
    }
    // Split (Ibex LSU): the bytes in the first word, then the rest from the next word.
    uint32_t n1 = 4u - off, n2 = size - n1, a2 = (addr & ~3u) + 4u;
    uint32_t v1 = write ? (*v & ((1u << (8u * n1)) - 1u)) : 0u;
    uint32_t v2 = write ? (*v >> (8u * n1)) : 0u;
    int r1 = bus_xfer(rv, write, addr, n1, &v1);
    if (r1 == TSR_RV_BUS_WAIT || r1 == TSR_RV_BUS_YIELD) { c->status = r1; return -1; }
    int r2 = bus_xfer(rv, write, a2, n2, &v2);
    data_watch(rv, addr, n1, write);
    data_watch(rv, a2, n2, write);
    // WAIT / YIELD are only allowed on an instruction's first transaction (tsr_rv.h); a bus that
    // returns them here is treated as faulting.
    c->cost += 1u + (r1 > 0 ? (uint32_t)r1 : 0u) + (r2 > 0 ? (uint32_t)r2 : 0u);
    if (r1 > 1 || r2 > 1) c->io = 1;
    if (r1 < 0 || r2 < 0) {
        c->exc = 1; c->cause = write ? TSR_RV_EXC_STORE_FAULT : TSR_RV_EXC_LOAD_FAULT;
        c->tval = (r1 < 0) ? addr : a2;
        return -1;
    }
    uint32_t val = v1 | (v2 << (8u * n1));
    if (!write) *v = val;
    c->mem_val = write ? *v : val;
    return 0;
}

// ---------------------------------------------------------------------------
// Compressed expansion (RV32C quadrants 0-2 to the equivalent 32-bit instruction)
// ---------------------------------------------------------------------------
static inline uint32_t enc_i(uint32_t op, uint32_t rd, uint32_t f3, uint32_t rs1, uint32_t imm) {
    return op | (rd << 7) | (f3 << 12) | (rs1 << 15) | ((imm & 0xFFFu) << 20);
}
static inline uint32_t enc_r(uint32_t op, uint32_t rd, uint32_t f3, uint32_t rs1, uint32_t rs2, uint32_t f7) {
    return op | (rd << 7) | (f3 << 12) | (rs1 << 15) | (rs2 << 20) | (f7 << 25);
}
static inline uint32_t enc_s(uint32_t op, uint32_t f3, uint32_t rs1, uint32_t rs2, uint32_t imm) {
    return op | ((imm & 0x1Fu) << 7) | (f3 << 12) | (rs1 << 15) | (rs2 << 20) | (((imm >> 5) & 0x7Fu) << 25);
}
static inline uint32_t enc_b(uint32_t f3, uint32_t rs1, uint32_t rs2, uint32_t imm) {
    return 0x63u | (((imm >> 11) & 1u) << 7) | (((imm >> 1) & 0xFu) << 8) | (f3 << 12) | (rs1 << 15) |
           (rs2 << 20) | (((imm >> 5) & 0x3Fu) << 25) | (((imm >> 12) & 1u) << 31);
}
static inline uint32_t enc_j(uint32_t rd, uint32_t imm) {
    return 0x6Fu | (rd << 7) | (((imm >> 12) & 0xFFu) << 12) | (((imm >> 11) & 1u) << 20) |
           (((imm >> 1) & 0x3FFu) << 21) | (((imm >> 20) & 1u) << 31);
}

// Returns 0 and *out = the expanded instruction, or -1 for an illegal parcel.
static int expand_c(uint32_t c, uint32_t * out) {
    uint32_t q = c & 3u, f3 = bits(c, 15, 13);
    uint32_t rd = bits(c, 11, 7), rs2 = bits(c, 6, 2);
    uint32_t rdp = 8u + bits(c, 4, 2), rs1p = 8u + bits(c, 9, 7);
    if (q == 0u) {
        switch (f3) {
        case 0: {   // C.ADDI4SPN
            uint32_t imm = (bits(c, 12, 11) << 4) | (bits(c, 10, 7) << 6) | (bits(c, 6, 6) << 2) | (bits(c, 5, 5) << 3);
            if (imm == 0u) return -1;
            *out = enc_i(0x13u, rdp, 0, 2, imm);
            return 0;
        }
        case 2: case 6: {   // C.LW / C.SW
            uint32_t imm = (bits(c, 12, 10) << 3) | (bits(c, 6, 6) << 2) | (bits(c, 5, 5) << 6);
            *out = (f3 == 2u) ? enc_i(0x03u, rdp, 2, rs1p, imm) : enc_s(0x23u, 2, rs1p, rdp, imm);
            return 0;
        }
        default: return -1;   // C.FLD, C.FLW, reserved (Zcb), C.FSD, C.FSW
        }
    }
    if (q == 1u) {
        uint32_t imm6 = sext((bits(c, 12, 12) << 5) | bits(c, 6, 2), 6);
        switch (f3) {
        case 0: *out = enc_i(0x13u, rd, 0, rd, imm6); return 0;          // C.ADDI (rd 0: NOP / hint)
        case 1: case 5: {                                                  // C.JAL / C.J
            uint32_t imm = (bits(c, 12, 12) << 11) | (bits(c, 11, 11) << 4) | (bits(c, 10, 9) << 8) |
                           (bits(c, 8, 8) << 10) | (bits(c, 7, 7) << 6) | (bits(c, 6, 6) << 7) |
                           (bits(c, 5, 3) << 1) | (bits(c, 2, 2) << 5);
            *out = enc_j(f3 == 1u ? 1u : 0u, sext(imm, 12));
            return 0;
        }
        case 2: *out = enc_i(0x13u, rd, 0, 0, imm6); return 0;           // C.LI
        case 3:
            if (rd == 2u) {                                                // C.ADDI16SP
                uint32_t imm = (bits(c, 12, 12) << 9) | (bits(c, 6, 6) << 4) | (bits(c, 5, 5) << 6) |
                               (bits(c, 4, 3) << 7) | (bits(c, 2, 2) << 5);
                if (imm == 0u) return -1;
                *out = enc_i(0x13u, 2, 0, 2, sext(imm, 10));
            } else {                                                       // C.LUI
                if (bits(c, 12, 12) == 0u && bits(c, 6, 2) == 0u) return -1;
                *out = 0x37u | (rd << 7) | ((imm6 & 0xFFFFFu) << 12);
            }
            return 0;
        case 4: {
            uint32_t f2 = bits(c, 11, 10);
            if (f2 == 0u || f2 == 1u) {                                    // C.SRLI / C.SRAI
                if (bits(c, 12, 12)) return -1;
                *out = enc_i(0x13u, rs1p, 5, rs1p, bits(c, 6, 2) | (f2 == 1u ? 0x400u : 0u));
                return 0;
            }
            if (f2 == 2u) { *out = enc_i(0x13u, rs1p, 7, rs1p, imm6); return 0; }   // C.ANDI
            if (bits(c, 12, 12)) return -1;                                // C.SUBW / C.ADDW / reserved
            static const uint32_t f3s[4] = { 0, 4, 6, 7 };                 // SUB XOR OR AND
            uint32_t sel = bits(c, 6, 5);
            *out = enc_r(0x33u, rs1p, f3s[sel], rs1p, rdp, sel == 0u ? 0x20u : 0u);
            return 0;
        }
        default: {                                                         // C.BEQZ / C.BNEZ
            uint32_t imm = (bits(c, 12, 12) << 8) | (bits(c, 11, 10) << 3) | (bits(c, 6, 5) << 6) |
                           (bits(c, 4, 3) << 1) | (bits(c, 2, 2) << 5);
            *out = enc_b(f3 == 6u ? 0u : 1u, rs1p, 0, sext(imm, 9));
            return 0;
        }
        }
    }
    // q == 2
    switch (f3) {
    case 0:                                                                // C.SLLI
        if (bits(c, 12, 12)) return -1;
        *out = enc_i(0x13u, rd, 1, rd, rs2);
        return 0;
    case 2: {                                                              // C.LWSP
        if (rd == 0u) return -1;
        uint32_t imm = (bits(c, 12, 12) << 5) | (bits(c, 6, 4) << 2) | (bits(c, 3, 2) << 6);
        *out = enc_i(0x03u, rd, 2, 2, imm);
        return 0;
    }
    case 4:
        if (bits(c, 12, 12) == 0u) {
            if (rs2 == 0u) {                                               // C.JR
                if (rd == 0u) return -1;
                *out = enc_i(0x67u, 0, 0, rd, 0);
            } else {
                *out = enc_r(0x33u, rd, 0, 0, rs2, 0);                     // C.MV
            }
        } else {
            if (rs2 == 0u) {
                *out = (rd == 0u) ? 0x00100073u : enc_i(0x67u, 1, 0, rd, 0);   // C.EBREAK / C.JALR
            } else {
                *out = enc_r(0x33u, rd, 0, rd, rs2, 0);                    // C.ADD
            }
        }
        return 0;
    case 6: {                                                              // C.SWSP
        uint32_t imm = (bits(c, 12, 9) << 2) | (bits(c, 8, 7) << 6);
        *out = enc_s(0x23u, 2, 2, rs2, imm);
        return 0;
    }
    default: return -1;   // C.FLDSP, C.FLWSP, C.FSDSP, C.FSWSP
    }
}

// ---------------------------------------------------------------------------
// Timing penalties that depend on the neighbouring instruction (tsr_rv.h cycle model). They read
// the instruction memory without side effects (fetch16) and never change architectural state.
// ---------------------------------------------------------------------------
// The (expanded) instruction at pc: 0, or -1 if it cannot be fetched or is not a valid parcel.
static int peek_insn(tsr_rv * rv, uint32_t pc, uint32_t * e) {
    uint16_t p0, p1;
    if (fetch16(rv, pc, &p0)) return -1;
    if ((p0 & 3u) != 3u) return expand_c(p0, e);
    if (fetch16(rv, pc + 2u, &p1)) return -1;
    *e = (uint32_t)p0 | ((uint32_t)p1 << 16);
    return 0;
}

// 1 if the (expanded) instruction e reads register r (r != 0) through Ibex's register-file read
// ports (rf_ren_a / rf_ren_b of ibex_decoder: rs1 for JALR, loads, OP-IMM and register CSR ops;
// rs1 and rs2 for branches, stores and OP; none for LUI, AUIPC, JAL, fences, ecall / ebreak / mret
// / wfi and immediate CSR ops; *confirm*).
static int reads_reg(uint32_t e, uint32_t r) {
    const uint32_t opc = e & 0x7Fu, f3 = bits(e, 14, 12), rs1 = bits(e, 19, 15), rs2 = bits(e, 24, 20);
    int r1 = 0, r2 = 0;
    switch (opc) {
    case 0x67: case 0x03: case 0x13: r1 = 1; break;
    case 0x63: case 0x23: case 0x33: r1 = r2 = 1; break;
    case 0x73: r1 = f3 >= 1u && f3 <= 3u; break;
    default: break;
    }
    return (r1 && rs1 == r) || (r2 && rs2 == r);
}

// A control transfer to c->next_pc: + target32 if the target is a 32-bit instruction that starts
// in the upper half of a word (the IF stage needs a second word fetch before it is valid).
static void pen_target32(tsr_rv * rv, ix * c) {
    uint16_t p;
    if (rv->cpi.target32 == 0u || !(c->next_pc & 2u)) return;
    if (fetch16(rv, c->next_pc, &p) == 0 && (p & 3u) == 3u) {
        c->cost += rv->cpi.target32;
        c->pen_t32 = 1;
    }
}

// A completed load to rd: + load_use if the next instruction in memory reads rd (WritebackStage:
// Ibex does not forward load data from WB, stall_ld_hz).
static void pen_load_use(tsr_rv * rv, ix * c, uint32_t rd) {
    uint32_t e;
    if (rv->cpi.load_use == 0u || rd == 0u) return;
    if (peek_insn(rv, c->next_pc, &e) == 0 && reads_reg(e, rd)) {
        c->cost += rv->cpi.load_use;
        c->pen_lu = 1;
    }
}

// ---------------------------------------------------------------------------
// Execute one (expanded) instruction e. c->len and c->insn are set; c->next_pc defaults to
// pc + len. Raises exceptions in c; never changes architectural state except through c
// (rd is written by the caller) and through CSR / memory side effects.
// ---------------------------------------------------------------------------
static void illegal(ix * c) { c->exc = 1; c->cause = TSR_RV_EXC_ILLEGAL; c->tval = c->insn; }

static void exec_system(tsr_rv * rv, ix * c, uint32_t e) {
    uint32_t f3 = bits(e, 14, 12), rd = bits(e, 11, 7), rs1 = bits(e, 19, 15), csr = bits(e, 31, 20);
    if (f3 == 0u) {
        if (rd != 0u || rs1 != 0u) { illegal(c); return; }
        switch (csr) {
        case 0x000:                                    // ECALL
            c->exc = 1; c->cause = rv->priv == TSR_RV_PRIV_U ? TSR_RV_EXC_ECALL_U : TSR_RV_EXC_ECALL_M; c->tval = 0;
            return;
        case 0x001:                                    // EBREAK
            if (rv->dbg_mode) {
                // Ibex: re-enter the debug ROM's park loop, dpc / dcsr unchanged (parked by the caller).
                c->no_retire = 1;
                c->next_pc = TSR_RV_DM_HALT_ADDR;
                return;
            }
            if ((rv->priv == TSR_RV_PRIV_M && (rv->dcsr & DCSR_EBREAKM)) ||
                (rv->priv == TSR_RV_PRIV_U && (rv->dcsr & DCSR_EBREAKU))) {
                c->no_retire = 2;                      // debug entry, cause ebreak (done by the caller)
                return;
            }
            c->exc = 1; c->cause = TSR_RV_EXC_BREAKPOINT; c->tval = 0;
            return;
        case 0x302: {                                  // MRET
            if (rv->priv != TSR_RV_PRIV_M && !rv->dbg_mode) { illegal(c); return; }
            uint32_t mpp = (rv->mstatus >> MPP_SHIFT) & 3u;
            uint32_t ms = rv->mstatus & ~(TSR_RV_MSTATUS_MIE | TSR_RV_MSTATUS_MPP);
            if (rv->mstatus & TSR_RV_MSTATUS_MPIE) ms |= TSR_RV_MSTATUS_MIE;
            ms |= TSR_RV_MSTATUS_MPIE;                 // MPP := U (least privileged mode)
            if (mpp != TSR_RV_PRIV_M) ms &= ~TSR_RV_MSTATUS_MPRV;
            rv->mstatus = ms;
            rv->priv = mpp;
            rv->cpuctrlsts &= ~CPUCTRL_SYNC_EXC;
            c->next_pc = rv->mepc;
            c->cost = rv->cpi.trap;
            c->cls = TSR_RV_CLS_SYSTEM;
            return;
        }
        case 0x7B2:                                    // DRET
            if (!rv->dbg_mode) { illegal(c); return; }
            leave_debug(rv);
            c->next_pc = rv->pc;
            c->cost = rv->cpi.trap;
            c->cls = TSR_RV_CLS_SYSTEM;
            return;
        case 0x105:                                    // WFI
            if (rv->priv == TSR_RV_PRIV_U && (rv->mstatus & TSR_RV_MSTATUS_TW)) { illegal(c); return; }
            if (!rv->dbg_mode && !rv->stepping) rv->sleeping = 1;
            return;
        default:
            illegal(c);
            return;
        }
    }
    if (f3 == 4u) { illegal(c); return; }
    // Zicsr. CSRRW(I) always write; CSRRS(I) / CSRRC(I) write only with rs1 / uimm != 0.
    uint32_t op = f3 & 3u;
    uint32_t src = (f3 & 4u) ? rs1 : rv->x[rs1];
    int write = (op == 1u) || rs1 != 0u;
    uint32_t old;
    c->cls = TSR_RV_CLS_CSR;
    if (csr_check(rv, csr, write, rv->dbg_mode) || csr_read_raw(rv, csr, &old)) { illegal(c); return; }
    if (write) {
        // Ibex csr_pipe_flush: every CSR write / set / clear except mscratch and mepc (*confirm*).
        if (csr != 0x340u && csr != 0x341u) {
            c->cost += rv->cpi.csr_flush;
            c->pen_csr = 1;
        }
        uint32_t nv = (op == 1u) ? src : (op == 2u) ? (old | src) : (old & ~src);
        csr_write_raw(rv, csr, nv, rv->dbg_mode, c);
        uint32_t after = 0;
        (void)csr_read_raw(rv, csr, &after);
        c->csr = csr; c->csr_val = after; c->csr_written = 1;
    }
    c->rd = rd; c->rd_val = old;
}

static void execute(tsr_rv * rv, ix * c, uint32_t e) {
    uint32_t pc = rv->pc;
    uint32_t opc = e & 0x7Fu, rd = bits(e, 11, 7), f3 = bits(e, 14, 12);
    uint32_t a = rv->x[bits(e, 19, 15)], b = rv->x[bits(e, 24, 20)];
    uint32_t f7 = bits(e, 31, 25);
    uint32_t imm_i = sext(e >> 20, 12);
    c->cost = rv->cpi.alu;
    switch (opc) {
    case 0x37: c->rd = rd; c->rd_val = e & 0xFFFFF000u; return;                 // LUI
    case 0x17: c->rd = rd; c->rd_val = pc + (e & 0xFFFFF000u); return;          // AUIPC
    case 0x6F: {                                                                // JAL
        uint32_t imm = (bits(e, 31, 31) << 20) | (bits(e, 19, 12) << 12) | (bits(e, 20, 20) << 11) | (bits(e, 30, 21) << 1);
        c->rd = rd; c->rd_val = pc + c->len;
        c->next_pc = pc + sext(imm, 21);
        c->cost = rv->cpi.jump;
        c->cls = TSR_RV_CLS_JUMP;
        pen_target32(rv, c);
        return;
    }
    case 0x67:                                                                  // JALR
        if (f3 != 0u) { illegal(c); return; }
        c->rd = rd; c->rd_val = pc + c->len;
        c->next_pc = (a + imm_i) & ~1u;
        c->cost = rv->cpi.jump;
        c->cls = TSR_RV_CLS_JUMP;
        pen_target32(rv, c);
        return;
    case 0x63: {                                                                // BRANCH
        int t;
        switch (f3) {
        case 0: t = a == b; break;
        case 1: t = a != b; break;
        case 4: t = (int32_t)a < (int32_t)b; break;
        case 5: t = (int32_t)a >= (int32_t)b; break;
        case 6: t = a < b; break;
        case 7: t = a >= b; break;
        default: illegal(c); return;
        }
        c->cls = TSR_RV_CLS_BRANCH_NT;
        if (t) {
            uint32_t imm = (bits(e, 31, 31) << 12) | (bits(e, 7, 7) << 11) | (bits(e, 30, 25) << 5) | (bits(e, 11, 8) << 1);
            c->next_pc = pc + sext(imm, 13);
            c->cost = rv->cpi.taken;
            c->cls = TSR_RV_CLS_BRANCH_T;
            pen_target32(rv, c);
        }
        return;
    }
    case 0x03: {                                                                // LOAD
        uint32_t size;
        switch (f3) {
        case 0: case 4: size = 1; break;
        case 1: case 5: size = 2; break;
        case 2: size = 4; break;
        default: illegal(c); return;
        }
        uint32_t v = 0;
        c->cost = rv->cpi.lsu;
        c->cls = TSR_RV_CLS_LOAD;
        const int fail = data_access(rv, c, a + imm_i, size, 0, &v);
        if (c->io) c->cls = TSR_RV_CLS_LOAD_IO;
        if (fail) return;
        if (f3 == 0u) v = sext(v, 8);
        else if (f3 == 1u) v = sext(v, 16);
        c->rd = rd; c->rd_val = v;
        pen_load_use(rv, c, rd);
        return;
    }
    case 0x23: {                                                                // STORE
        if (f3 > 2u) { illegal(c); return; }
        uint32_t imm = sext((bits(e, 31, 25) << 5) | bits(e, 11, 7), 12);
        uint32_t size = 1u << f3;
        uint32_t v = (size == 4u) ? b : (b & ((1u << (8u * size)) - 1u));
        c->cost = rv->cpi.lsu;
        (void)data_access(rv, c, a + imm, size, 1, &v);
        c->cls = c->io ? TSR_RV_CLS_STORE_IO : TSR_RV_CLS_STORE;
        return;
    }
    case 0x13: {                                                                // OP-IMM
        uint32_t sh = bits(e, 24, 20), r;
        switch (f3) {
        case 0: r = a + imm_i; break;
        case 2: r = (int32_t)a < (int32_t)imm_i; break;
        case 3: r = a < imm_i; break;
        case 4: r = a ^ imm_i; break;
        case 6: r = a | imm_i; break;
        case 7: r = a & imm_i; break;
        case 1: if (f7 != 0u) { illegal(c); return; } r = a << sh; break;
        default:   // 5
            if (f7 == 0u) r = a >> sh;
            else if (f7 == 0x20u) r = sra32(a, sh);
            else { illegal(c); return; }
            break;
        }
        c->rd = rd; c->rd_val = r;
        return;
    }
    case 0x33: {                                                                // OP
        uint32_t r;
        if (f7 == 0x01u) {                                                      // M
            int32_t sa = (int32_t)a, sb = (int32_t)b;
            switch (f3) {
            case 0: r = a * b; break;
            case 1: r = (uint32_t)((uint64_t)((int64_t)sa * (int64_t)sb) >> 32); break;
            case 2: r = (uint32_t)((uint64_t)((int64_t)sa * (int64_t)(uint64_t)b) >> 32); break;
            case 3: r = (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32); break;
            case 4: r = (b == 0u) ? 0xFFFFFFFFu : (a == 0x80000000u && b == 0xFFFFFFFFu) ? a : (uint32_t)(sa / sb); break;
            case 5: r = (b == 0u) ? 0xFFFFFFFFu : a / b; break;
            case 6: r = (b == 0u) ? a : (a == 0x80000000u && b == 0xFFFFFFFFu) ? 0u : (uint32_t)(sa % sb); break;
            default: r = (b == 0u) ? a : a % b; break;
            }
            c->cost = f3 == 0u ? rv->cpi.mul : f3 < 4u ? rv->cpi.mulh : b == 0u ? rv->cpi.div0 : rv->cpi.div;
            c->cls = f3 < 4u ? TSR_RV_CLS_MUL : TSR_RV_CLS_DIV;
        } else if (f7 == 0u) {
            switch (f3) {
            case 0: r = a + b; break;
            case 1: r = a << (b & 31u); break;
            case 2: r = (int32_t)a < (int32_t)b; break;
            case 3: r = a < b; break;
            case 4: r = a ^ b; break;
            case 5: r = a >> (b & 31u); break;
            case 6: r = a | b; break;
            default: r = a & b; break;
            }
        } else if (f7 == 0x20u && f3 == 0u) {
            r = a - b;
        } else if (f7 == 0x20u && f3 == 5u) {
            r = sra32(a, b);
        } else {
            illegal(c);
            return;
        }
        c->rd = rd; c->rd_val = r;
        return;
    }
    case 0x0F:                                                                  // MISC-MEM
        if (f3 > 1u) { illegal(c); return; }                                    // FENCE / FENCE.I: no-ops
        if (f3 == 1u) {                                     // FENCE.I: Ibex jumps to the next instruction
            c->cost = rv->cpi.jump;
            c->cls = TSR_RV_CLS_JUMP;
            pen_target32(rv, c);
        }
        return;
    case 0x73:
        exec_system(rv, c, e);
        return;
    default:
        illegal(c);
        return;
    }
}

// ---------------------------------------------------------------------------
// Instruction boundary: debug entry, wake-up, interrupt, trigger, then one instruction.
// Returns CONT or a stop reason; *n counts executed instructions.
// ---------------------------------------------------------------------------
static void emit(tsr_rv * rv, const ix * c, uint32_t pc, uint32_t priv, uint32_t dbg) {
    tsr_rv_retire r;
    memset(&r, 0, sizeof(r));
    r.order = rv->order;
    r.pc = pc; r.next_pc = c->exc ? (dbg ? TSR_RV_DM_EXC_ADDR : rv->pc) : c->next_pc;
    r.insn = c->insn; r.len = (uint8_t)c->len;
    r.trap = (uint8_t)c->exc; r.intr = (uint8_t)rv->intr_next; r.dbg = (uint8_t)dbg; r.priv = (uint8_t)priv;
    if (!c->exc) {
        r.rd = (uint8_t)c->rd; r.rd_val = c->rd ? c->rd_val : 0u;
        r.mem_size = (uint8_t)c->mem_size; r.mem_write = (uint8_t)c->mem_write;
        r.mem_addr = c->mem_addr; r.mem_val = c->mem_val;
        r.csr = (uint16_t)c->csr; r.csr_written = (uint8_t)c->csr_written; r.csr_val = c->csr_val;
    } else {
        r.cause = c->cause;
    }
    if (rv->retire_fn(rv->retire_ctx, &r)) rv->stop_req = 1;
}

static uint32_t rv_boundary(tsr_rv * rv, uint64_t * n) {
    // The levels are sampled also while parked: mip read by the debugger is the live view.
    uint32_t lines = rv->bus.irq ? (rv->bus.irq(rv->bus.ctx) & TSR_RV_IRQ_LINES) : 0u;
    rv->lines = lines;
    if (rv->parked) return TSR_RV_STOP_HALTED;
    if (!rv->waiting) {
        // Debug entry before the next instruction, ahead of interrupts: trigger > haltreq > step
        // (header). A sleeping hart has no next instruction yet; a halt request wakes it.
        if (!rv->dbg_mode) {
            const int trig = !rv->sleeping && trigger_match(rv, rv->pc);
            if (trig || rv->debug_req || rv->step_done) {
                enter_debug(rv, trig ? TSR_RV_DBG_CAUSE_TRIGGER :
                                rv->debug_req ? TSR_RV_DBG_CAUSE_HALTREQ : TSR_RV_DBG_CAUSE_STEP, rv->pc);
                return TSR_RV_STOP_HALTED;
            }
        }
        uint32_t pend = lines & rv->mie;
        if (rv->sleeping) {
            if (pend == 0u) return TSR_RV_STOP_WFI;
            rv->sleeping = 0;
        }
        if (pend && !rv->dbg_mode && !rv->stepping &&
            (rv->priv == TSR_RV_PRIV_U || (rv->mstatus & TSR_RV_MSTATUS_MIE))) {
            uint32_t cause;
            if (pend & 0x7FFF0000u)          cause = 16u + ctz32(pend >> 16);
            else if (pend & TSR_RV_LINE_MEI) cause = TSR_RV_IRQ_MEI;
            else if (pend & TSR_RV_LINE_MSI) cause = TSR_RV_IRQ_MSI;
            else                             cause = TSR_RV_IRQ_MTI;
            trap_enter(rv, cause, 0, rv->pc, 1);
            rv->intr_next = 1;
            add_cycles(rv, rv->cpi.trap, 0);
        }
        // The instruction after a wake-up or an interrupt entry (the handler's first) may match.
        if (!rv->dbg_mode && trigger_match(rv, rv->pc)) {
            enter_debug(rv, TSR_RV_DBG_CAUSE_TRIGGER, rv->pc);
            return TSR_RV_STOP_HALTED;
        }
    }

    ix c;
    memset(&c, 0, sizeof(c));
    uint32_t pc = rv->pc, priv = rv->priv, dbg = (uint32_t)rv->dbg_mode;
    uint16_t p0 = 0, p1 = 0;
    if (fetch16(rv, pc, &p0)) {
        c.exc = 1; c.cause = TSR_RV_EXC_FETCH_FAULT; c.tval = pc; c.cost = rv->cpi.alu;
    } else if ((p0 & 3u) == 3u) {
        c.len = 4;
        if (fetch16(rv, pc + 2u, &p1)) {
            c.insn = p0; c.exc = 1; c.cause = TSR_RV_EXC_FETCH_FAULT; c.tval = pc + 2u; c.cost = rv->cpi.alu;
        } else {
            c.insn = (uint32_t)p0 | ((uint32_t)p1 << 16);
            c.next_pc = pc + 4u;
            if ((c.insn & 0x1Cu) == 0x1Cu) { c.cost = rv->cpi.alu; illegal(&c); }   // 48-bit and longer
            else execute(rv, &c, c.insn);
        }
    } else {
        uint32_t e;
        c.len = 2; c.insn = p0; c.next_pc = pc + 2u;
        if (expand_c(p0, &e)) { c.cost = rv->cpi.alu; illegal(&c); }
        else execute(rv, &c, e);
    }

    if (c.status == TSR_RV_BUS_WAIT) {
        if (!rv->waiting) { rv->waiting = 1; rv->wait_cycles = 0; }
        return TSR_RV_STOP_WAIT;
    }
    if (c.status == TSR_RV_BUS_YIELD) return TSR_RV_STOP_YIELD;
    rv->waiting = 0; rv->wait_cycles = 0;
    ++*n;
    rv->st.n[c.cls]++;
    rv->st.cyc[c.cls] += c.cost;
    rv->st.load_use += (uint64_t)c.pen_lu;
    rv->st.target32 += (uint64_t)c.pen_t32;
    rv->st.csr_flush += (uint64_t)c.pen_csr;

    if (c.no_retire == 2) {                                   // ebreak into debug mode
        enter_debug(rv, TSR_RV_DBG_CAUSE_EBREAK, pc);
        rv->intr_next = 0;
        return TSR_RV_STOP_HALTED;
    }
    if (c.exc) {
        if (dbg) {
            // Ibex: an exception in debug mode jumps to DmExceptionAddr without CSR updates; the
            // ROM reports it to the DM and parks again.
            add_cycles(rv, c.cost + rv->cpi.trap, 0);
            if (rv->retire_fn) emit(rv, &c, pc, priv, dbg);
            ++rv->order;
            rv->intr_next = 0;
            rv->parked = 1;
            rv->pc = TSR_RV_DM_HALT_ADDR;
            debug_event(rv, TSR_RV_DBG_EV_PARK, 1);
            return TSR_RV_STOP_HALTED;
        }
        trap_enter(rv, c.cause, c.tval, pc, 0);
        add_cycles(rv, c.cost + rv->cpi.trap, 0);
    } else {
        if (c.rd) rv->x[c.rd] = c.rd_val;
        rv->pc = c.next_pc;
        add_cycles(rv, c.cost, c.wrote_mcycle);
        if (!c.no_retire) {
            ++rv->retired;
            if (!c.wrote_minstret && !(rv->mcountinhibit & 4u)) ++rv->minstret;
        }
    }
    if (rv->retire_fn) emit(rv, &c, pc, priv, dbg);
    ++rv->order;
    rv->intr_next = 0;
    if (c.no_retire == 1) {                                   // ebreak in debug mode: park again
        rv->parked = 1;
        debug_event(rv, TSR_RV_DBG_EV_PARK, 0);
        return TSR_RV_STOP_HALTED;
    }
    // A single step is one instruction executed outside debug mode (not the dret that starts it).
    if (rv->stepping && !dbg && !rv->dbg_mode) rv->step_done = 1;
    return CONT;
}

uint32_t tsr_rv_run(tsr_rv * rv, uint64_t max_insns, uint64_t max_cycles) {
    uint64_t c0 = rv->cycles, n = 0;
    for (;;) {
        if (rv->stop_req) { rv->stop_req = 0; return TSR_RV_STOP_REQUEST; }
        if (n >= max_insns || rv->cycles - c0 >= max_cycles) return TSR_RV_STOP_BUDGET;
        uint32_t r = rv_boundary(rv, &n);
        if (r != CONT) {
            // A stop request raised by the last instruction is satisfied by any stop.
            rv->stop_req = 0;
            return r;
        }
    }
}

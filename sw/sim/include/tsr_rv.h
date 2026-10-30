// tsr_rv.h — RV32IMC + Zicsr + Zifencei interpreter modelling the control core's Ibex
// (docs/v2/06-software-verification.md "SW" §4.2, V2-SW-1; docs/v2/04-control-core.md "CC" §3).
//
// The interpreter is the architectural oracle of the RISC-V side: the firmware runs on it inside
// the simulator, and it is checked against Spike (riscv-tests, riscv-arch-test and random
// programs on the test bus profile, SW §10.1) and against Ibex RVFI (P5.2). Memory is reached only
// through a tsr_rv_bus (tsr_bus.h provides the firmware profile, CC §4.1, and the test profile,
// SW §4.2). Nothing here depends on host time: cycles are counted by a fixed table (SW §4.2,
// *estimate*, schedule only, never architectural).
//
// Ibex configuration modelled (CC §3.2): RV32M Fast, RV32B None, PMPEnable 0, MHPMCounterNum 0,
// DbgTriggerEn 1 with DbgHwBreakNum 2, DmHaltAddr 0x1000_0800, DmExceptionAddr 0x1000_0808,
// hart_id 0. Ibex always implements U-mode (misa.U = 1; mstatus.MPP is M or U); the firmware runs
// in M-mode only. Ibex facts recalled from its documentation and RTL are marked *confirm* in
// tsr_rv.c and are checked when Ibex is vendored (CC §3.1).
//
// Architectural summary (details and *confirm* items: the header comment of tsr_rv.c):
//   - reset (tsr_rv_reset): pc = boot[31:8] + 0x80, mtvec = boot[31:8] | 1 (vectored), mstatus =
//     MPIE (MPP = U), priv = M, x1..x31 = 0 (the FPGA register file is not reset; 0 here);
//   - traps: exceptions to mtvec base, interrupts to base + 4 x cause; mcause keeps bit 31 and
//     bits [4:0]; mtval = instruction bits (illegal), fetch address (pc, or pc + 2 when the second
//     half of an uncompressed instruction faults), the access address (load / store faults: the
//     original address, or the word-aligned address of the second half of a split access), 0
//     otherwise (ecall, ebreak, interrupts);
//   - misaligned loads / stores never trap: Ibex splits an access that crosses a word boundary
//     into two word accesses in address order, both always issued (not atomic);
//   - interrupts: level inputs sampled at instruction boundaries; priority irq_fast[0] (highest)
//     .. irq_fast[14], then MEI, MSI, MTI; enabled in M-mode by mstatus.MIE, always in U-mode;
//     WFI sleeps until (mip & mie) != 0 whatever mstatus.MIE is;
//   - debug mode (Debug Spec 0.13.2 as Ibex implements it): entered by a halt request
//     (tsr_rv_set_debug_req), an execute trigger (tselect / tdata1-2, two triggers), ebreak with
//     dcsr.ebreakm / ebreaku, or a single step (dcsr.step); before an instruction the cause
//     priority is trigger > haltreq > step, ahead of interrupts. dcsr.stepie / stopcount read 0
//     (Ibex): no interrupts during a step, counters count in debug mode. The debug ROM is not
//     interpreted: the hart "parks" (tsr_rv_halted) and the debug-module model (tsr_dm.c, P1.12,
//     DB §9.1) acts on it directly — abstract register access through tsr_rv_reg / tsr_rv_csr_*,
//     the program buffer through tsr_rv_debug_exec (runs from the DM region until ebreak, or an
//     exception, parks it again), resume through tsr_rv_debug_resume (as dret). DB's RISC-V
//     halted level RV_HALTED (last fetch in the DM range, DB §3.1) is tsr_rv_debug_mode here.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_rv tsr_rv;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// Exception causes (mcause, bit 31 = 0).
#define TSR_RV_EXC_FETCH_FAULT  1u
#define TSR_RV_EXC_ILLEGAL      2u
#define TSR_RV_EXC_BREAKPOINT   3u
#define TSR_RV_EXC_LOAD_FAULT   5u
#define TSR_RV_EXC_STORE_FAULT  7u
#define TSR_RV_EXC_ECALL_U      8u
#define TSR_RV_EXC_ECALL_M      11u
// Interrupt causes (mcause = bit 31 | cause) and mip / mie bits (bit = cause).
#define TSR_RV_IRQ_MSI          3u
#define TSR_RV_IRQ_MTI          7u
#define TSR_RV_IRQ_MEI          11u
#define TSR_RV_IRQ_FAST(i)      (16u + (i))     // irq_fast[i], i = 0..14
#define TSR_RV_MCAUSE_IRQ       0x80000000u

// Interrupt input lines as an Ibex mip mask (CC §12.3): bit 3 irq_software, 7 irq_timer,
// 11 irq_external, 16 + i irq_fast[i]. Bits outside TSR_RV_IRQ_LINES are ignored.
#define TSR_RV_LINE_MSI         (1u << 3)
#define TSR_RV_LINE_MTI         (1u << 7)
#define TSR_RV_LINE_MEI         (1u << 11)
#define TSR_RV_LINE_FAST(i)     (1u << (16u + (i)))
#define TSR_RV_IRQ_LINES        0x7FFF0888u

// Privilege levels.
#define TSR_RV_PRIV_U           0u
#define TSR_RV_PRIV_M           3u

// Debug mode (CC §3.2): riscv-dbg addresses at DmBaseAddress 0x1000_0000 (*confirm*).
#define TSR_RV_DM_HALT_ADDR     0x10000800u
#define TSR_RV_DM_EXC_ADDR      0x10000808u
// dcsr.cause values (Debug Spec 0.13.2).
#define TSR_RV_DBG_CAUSE_EBREAK  1u
#define TSR_RV_DBG_CAUSE_TRIGGER 2u
#define TSR_RV_DBG_CAUSE_HALTREQ 3u
#define TSR_RV_DBG_CAUSE_STEP    4u

// CSR numbers used by the API and the tests (the full list: tsr_rv.c).
#define TSR_RV_CSR_MSTATUS      0x300u
#define TSR_RV_CSR_MISA         0x301u
#define TSR_RV_CSR_MIE          0x304u
#define TSR_RV_CSR_MTVEC        0x305u
#define TSR_RV_CSR_MCOUNTINHIBIT 0x320u
#define TSR_RV_CSR_MSCRATCH     0x340u
#define TSR_RV_CSR_MEPC         0x341u
#define TSR_RV_CSR_MCAUSE       0x342u
#define TSR_RV_CSR_MTVAL        0x343u
#define TSR_RV_CSR_MIP          0x344u
#define TSR_RV_CSR_TSELECT      0x7A0u
#define TSR_RV_CSR_TDATA1       0x7A1u
#define TSR_RV_CSR_TDATA2       0x7A2u
#define TSR_RV_CSR_DCSR         0x7B0u
#define TSR_RV_CSR_DPC          0x7B1u
#define TSR_RV_CSR_DSCRATCH0    0x7B2u
#define TSR_RV_CSR_DSCRATCH1    0x7B3u
#define TSR_RV_CSR_MCYCLE       0xB00u
#define TSR_RV_CSR_MINSTRET     0xB02u
#define TSR_RV_CSR_MCYCLEH      0xB80u
#define TSR_RV_CSR_MINSTRETH    0xB82u
#define TSR_RV_CSR_MHARTID      0xF14u

#define TSR_RV_MSTATUS_MIE      (1u << 3)
#define TSR_RV_MSTATUS_MPIE     (1u << 7)
#define TSR_RV_MSTATUS_MPP      (3u << 11)
#define TSR_RV_MSTATUS_MPRV     (1u << 17)
#define TSR_RV_MSTATUS_TW       (1u << 21)

// ---------------------------------------------------------------------------
// Bus interface (implemented by tsr_bus.c, or by a test)
//
// Every data access the interpreter makes is one Ibex data-bus transaction: n = 1, 2 or 4 bytes
// that lie inside one aligned word (addr % 4 + n <= 4), little-endian, value in the low 8n bits
// (a split misaligned access is two such calls). Return values:
//   >= 0               done; the value is the access latency in RV cycles beyond the base cost
//   TSR_RV_BUS_FAULT   access fault (load / store / instruction access fault)
//   TSR_RV_BUS_WAIT    (store) the target cannot take the store now (an issue push without a
//                      credit, CC §7.3); nothing happened. The instruction does not complete: it
//                      is re-executed from scratch by the next step, and while it waits the hart
//                      takes neither interrupts nor a debug halt (CC §7.3)
//   TSR_RV_BUS_YIELD   the access must happen at a later time (the simulator agent's time
//                      slicing, tsr_bus.c); nothing happened; the instruction is re-executed by
//                      the next step, which is a normal instruction boundary
// The bus may only return WAIT or YIELD for the first transaction of an instruction (the
// interpreter guarantees nothing has changed before it).
// `dbg` = the hart is in debug mode.
// ---------------------------------------------------------------------------
#define TSR_RV_BUS_FAULT (-1)
#define TSR_RV_BUS_WAIT  (-2)
#define TSR_RV_BUS_YIELD (-3)

typedef struct tsr_rv_bus {
    void * ctx;
    // Instruction fetch of the 16-bit parcel at addr (2-byte aligned): 0 or TSR_RV_BUS_FAULT.
    int (*fetch)(void * ctx, uint32_t addr, uint32_t dbg, uint16_t * parcel);
    int (*load)(void * ctx, uint32_t addr, uint32_t n, uint32_t dbg, uint32_t * val);
    int (*store)(void * ctx, uint32_t addr, uint32_t n, uint32_t dbg, uint32_t val);
    // Interrupt input levels (TSR_RV_LINE_* mask), sampled at every instruction boundary.
    // NULL: all lines 0.
    uint32_t (*irq)(void * ctx);
    // Optional direct fetch window (speed only; same contents as fetch): parcels in
    // [fetch_base, fetch_base + fetch_size) are read from fetch_mem without calling fetch.
    const uint8_t * fetch_mem;
    uint32_t        fetch_base, fetch_size;
} tsr_rv_bus;

// ---------------------------------------------------------------------------
// Cycle model (SW §4.2 timing table, *estimate*; schedule only). Instruction cost in RV cycles:
// the class cost below plus the bus latency of each data transaction (one extra cycle per split
// half). P1.8 may replace the table (random schedules vary the CPI) through tsr_rv_set_cpi.
// ---------------------------------------------------------------------------
typedef struct tsr_rv_cpi {
    uint32_t alu;           // ALU, CSR, LUI / AUIPC, fence, not-taken branch, WFI, ecall/ebreak issue
    uint32_t taken;         // taken branch, JAL, JALR
    uint32_t lsu;           // load / store base (+ bus latency: TCM 1 -> 2 in total)
    uint32_t mul;           // MUL, MULH*
    uint32_t div;           // DIV*, REM*
    uint32_t trap;          // exception / interrupt / debug entry, mret, dret, fence.i
} tsr_rv_cpi;
#define TSR_RV_CPI_DEFAULT { /*alu*/ 1, /*taken*/ 3, /*lsu*/ 1, /*mul*/ 3, /*div*/ 37, /*trap*/ 3 }

// ---------------------------------------------------------------------------
// Retire records (RVFI-style; for lockstep against Spike's --log-commits and Ibex RVFI)
//
// One record per instruction that completes (retires) or raises an exception (trap = 1: rd and
// memory fields describe nothing; next_pc = the handler, or the debug exception address in debug
// mode). Interrupt and debug entries are not records: the first instruction of an interrupt
// handler has intr = 1. Not reported: instructions re-executed after WAIT / YIELD until they
// complete.
// ---------------------------------------------------------------------------
typedef struct tsr_rv_retire {
    uint64_t order;        // 0, 1, 2, ... over all records since create
    uint32_t pc;
    uint32_t next_pc;
    uint32_t insn;         // instruction bits (16-bit instructions: the parcel)
    uint8_t  len;          // 2 or 4 (0: fetch fault before the length was known)
    uint8_t  trap;         // exception raised (mcause in `cause`)
    uint8_t  intr;         // first instruction after an interrupt entry
    uint8_t  dbg;          // executed in debug mode
    uint8_t  priv;         // privilege it executed in (TSR_RV_PRIV_*)
    uint8_t  rd;           // destination register written (0: none)
    uint8_t  mem_size;     // 0 none, else 1 / 2 / 4 (the access as issued, possibly misaligned)
    uint8_t  mem_write;
    uint32_t rd_val;
    uint32_t mem_addr;
    uint32_t mem_val;      // load: the bytes read (before sign / zero extension); store: written
    uint32_t cause;        // trap: mcause
    uint16_t csr;          // CSR written by a Zicsr instruction (csr_written = 1)
    uint8_t  csr_written;
    uint8_t  rsvd;
    uint32_t csr_val;      // its value after the write (WARL applied)
} tsr_rv_retire;

// Return nonzero to stop the current tsr_rv_run / tsr_rv_step after this instruction
// (TSR_RV_STOP_REQUEST).
typedef int (*tsr_rv_retire_fn)(void * ctx, const tsr_rv_retire * r);

// Debug-module events (P1.12): ENTER (arg = dcsr.cause) when the hart enters debug mode and
// parks; PARK (arg = 0 ebreak, 1 exception) when program-buffer execution ends; RESUME when it
// leaves debug mode (tsr_rv_debug_resume or a dret); RESET (arg = the boot address) at the end
// of tsr_rv_reset, i.e. when the hart leaves reset: a debug request set from this event (or
// before) halts the hart before its first instruction (cause haltreq, dpc = boot + 0x80; DB §3.8
// halt-on-boot). The callback may call tsr_rv_set_debug_req and read state, nothing else.
#define TSR_RV_DBG_EV_ENTER  1u
#define TSR_RV_DBG_EV_PARK   2u
#define TSR_RV_DBG_EV_RESUME 3u
#define TSR_RV_DBG_EV_RESET  4u
typedef void (*tsr_rv_debug_fn)(void * ctx, uint32_t ev, uint32_t arg);

// Data watch (DB §3.7 RV watchpoints, P1.13): called once for every data transaction the hart
// issues on its data port (each half of a split misaligned access separately, with the bytes of
// that half: addr / size as the byte enables give them), after the bus took it and before the
// instruction completes, also when the transaction faults and in debug mode (program buffer);
// never for a WAIT / YIELD attempt (re-executed later), never for SBA (the bus's master, not the
// hart), never for fetches. pc = the accessing instruction. tsr_rv_set_debug_req from the
// callback halts the hart right after this instruction (cause haltreq, dpc = the next one).
typedef void (*tsr_rv_data_watch_fn)(void * ctx, uint32_t addr, uint32_t size, int is_store, uint32_t pc);

// ---------------------------------------------------------------------------
// Lifecycle and execution
// ---------------------------------------------------------------------------
// Creates a hart in reset state with boot address 0 (tsr_rv_reset(rv, 0)). *bus is copied.
tsr_rv * tsr_rv_create(const tsr_rv_bus * bus);
void     tsr_rv_destroy(tsr_rv * rv);

// Ibex reset with boot_addr_i = boot (CC §3.2): pc = (boot & ~0xFF) + 0x80, mtvec = (boot & ~0xFF)
// | 1, every CSR to its reset value, x1..x31 = 0, not in debug mode, not sleeping, no pending
// step / stall. Counters (cycles, order) of the model keep running; mcycle / minstret are reset.
void     tsr_rv_reset(tsr_rv * rv, uint32_t boot);

// Stop reasons of tsr_rv_run / tsr_rv_step.
#define TSR_RV_STOP_BUDGET   0u   // the instruction / cycle budget is used up
#define TSR_RV_STOP_WFI      1u   // asleep in WFI: no enabled interrupt pending, no debug request
#define TSR_RV_STOP_HALTED   2u   // in debug mode, parked (the debug module's turn)
#define TSR_RV_STOP_REQUEST  3u   // tsr_rv_request_stop or a retire callback asked for it
#define TSR_RV_STOP_WAIT     4u   // a store waits (TSR_RV_BUS_WAIT); retry later
#define TSR_RV_STOP_YIELD    5u   // the bus asked to retry the instruction later (TSR_RV_BUS_YIELD)

// Runs until max_insns instructions have been executed (retired or trapped; interrupt and debug
// entries do not count) or max_cycles RV cycles have passed since the call, or a stop reason
// occurs. Sleeping (WFI) and parked harts return at once without consuming cycles: the caller
// advances time with tsr_rv_idle. Returns the reason.
uint32_t tsr_rv_run(tsr_rv * rv, uint64_t max_insns, uint64_t max_cycles);
static inline uint32_t tsr_rv_step(tsr_rv * rv, uint64_t n) { return tsr_rv_run(rv, n, UINT64_MAX); }
// Lets n RV cycles pass with the hart asleep, parked or stalled (mcycle counts them unless
// inhibited: Ibex's clock gate is a pass-through on the FPGA, CC §3.1).
void     tsr_rv_idle(tsr_rv * rv, uint64_t n);
// Makes the running tsr_rv_run return TSR_RV_STOP_REQUEST after the current instruction
// (callable from bus callbacks and hooks; cleared when run returns it).
void     tsr_rv_request_stop(tsr_rv * rv);

// ---------------------------------------------------------------------------
// State access (debug view; the DM model's abstract commands, tests)
// ---------------------------------------------------------------------------
uint32_t tsr_rv_reg(const tsr_rv * rv, unsigned i);              // x0..x31
void     tsr_rv_set_reg(tsr_rv * rv, unsigned i, uint32_t v);    // x0 writes are ignored
// The next instruction (parked: TSR_RV_DM_HALT_ADDR). During a bus callback: the pc of the
// instruction making the access (e.g. the push PC of CC §7.1 for trace v2 and DB's history).
uint32_t tsr_rv_pc(const tsr_rv * rv);
void     tsr_rv_set_pc(tsr_rv * rv, uint32_t pc);                // bit 0 is cleared
uint32_t tsr_rv_priv(const tsr_rv * rv);
uint64_t tsr_rv_cycles(const tsr_rv * rv);    // RV cycles since create (model time, not mcycle)
uint64_t tsr_rv_retired(const tsr_rv * rv);   // instructions retired since create (not minstret)
int      tsr_rv_sleeping(const tsr_rv * rv);  // in WFI (STATUS.RV_SLEEP)
int      tsr_rv_waiting(const tsr_rv * rv);   // a store waits (TSR_RV_BUS_WAIT)
uint64_t tsr_rv_wait_cycles(const tsr_rv * rv);   // RV cycles the current waiting store waited
// While on, tsr_rv_idle does not add to the wait count (DB's dbg_stop level suspends the
// ISSUE_WAIT_MAX count, CC §7.3, DB §3.9; driven by the DB model, P1.13). Off after create.
void     tsr_rv_freeze_wait(tsr_rv * rv, int on);
// CSR access as an instruction executed in debug mode would do it (the debug module's abstract
// Access Register command, DB §3.5): M privilege, debug CSRs and trigger writes allowed, WARL
// rules applied, no side effect on counters. Return 0, or -1 if the CSR does not exist (or is
// read-only, for a write): the DM reports cmderr = exception.
int      tsr_rv_csr_read(tsr_rv * rv, uint32_t csr, uint32_t * val);
int      tsr_rv_csr_write(tsr_rv * rv, uint32_t csr, uint32_t val);

// Hooks (one slot each; fn = NULL removes). In the simulator the debug hook belongs to the DM
// model (tsr_dm.c forwards the events, tsr_dm.h) and the data watch to DB's model (tsr_dbg.c).
void     tsr_rv_set_retire_hook(tsr_rv * rv, tsr_rv_retire_fn fn, void * ctx);
void     tsr_rv_set_debug_hook(tsr_rv * rv, tsr_rv_debug_fn fn, void * ctx);
void     tsr_rv_set_data_watch(tsr_rv * rv, tsr_rv_data_watch_fn fn, void * ctx);
void     tsr_rv_set_cpi(tsr_rv * rv, const tsr_rv_cpi * cpi);

// ---------------------------------------------------------------------------
// Debug mode (for the DM model, P1.12; DB §3.3-§3.5, §9.1)
// ---------------------------------------------------------------------------
// Ibex debug_req_i (the DM's haltreq ORed with DB's dbg_req_rv, DB §3.3): a level. The hart
// enters debug mode (cause haltreq) at the next instruction boundary, also from WFI, but not
// while a store waits (CC §7.3).
void     tsr_rv_set_debug_req(tsr_rv * rv, int level);
int      tsr_rv_debug_mode(const tsr_rv * rv);   // in debug mode (parked or running the program buffer)
int      tsr_rv_halted(const tsr_rv * rv);       // parked in debug mode (STATUS.RV_HALTED)
// Leaves debug mode as dret does: pc = dpc, priv = dcsr.prv; with dcsr.step the hart executes one
// instruction (interrupts disabled) and re-enters debug mode with cause step. Returns -1 if the
// hart is not parked.
int      tsr_rv_debug_resume(tsr_rv * rv);
// Runs code in debug mode from pc (the program buffer in the DM region): un-parks the hart; it
// parks again on ebreak (debug event PARK, arg 0) or on an exception (PARK, arg 1; no CSR is
// changed, as Ibex jumps to DmExceptionAddr). Returns -1 if the hart is not parked.
int      tsr_rv_debug_exec(tsr_rv * rv, uint32_t pc);

#ifdef __cplusplus
}
#endif

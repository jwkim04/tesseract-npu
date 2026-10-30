// tsr_bus.h — the RISC-V bus of the simulator: the address maps tsr_rv runs on (SW §4.2).
//
// Two profiles (docs/design/06-software-verification.md "SW" §4.2):
//   firmware  the control core's map (docs/design/04-control-core.md "CC" §4.1, TSR_RV_* in
//             tsr_tables.h): ITCM (fetch, byte / half / word), DTCM (byte / half / word), debug
//             module memory (fetch, word), control window (word), core window (word loads),
//             SPM window and its line-buffered alias (byte / half / word). Everything else, and
//             every address of a region beyond its size, is an access fault. The control and
//             core windows and the SPM come from a tsr_bus_env (the simulator's core, or a test).
//   test      RAM at a base address (0x8000_0000 for riscv-tests -p, riscv-arch-test and random
//             lockstep programs, which Spike runs at the same place) with HTIF tohost / fromhost;
//             never used by firmware.
// Both hand the interpreter a tsr_rv_bus (tsr_bus_iface). The bus also provides the debug
// module's system-bus master (SBA) with CC §4.3's refusals, the ELF loader the test profile and
// the tools use, and (in the simulator library) the RISC-V agent that binds the firmware profile
// to a tsr_sim and runs the hart on the simulator's time base (tsr_rv_agent_*).
//
// Line buffer (CC §10.5): loads from the 0x5000_0000 alias hit a one-line buffer when its tag
// matches, else fill it from the SPM. Invalidated by any store to either SPM alias (hart or SBA),
// any control-window or core-window access, any rising edge of an interrupt line, and
// SPM_LB_INV (a control-window store, so covered by the previous rule).
//
// Latencies (RV cycles a data transaction adds to the instruction's base cost; SW §4.2
// *estimate*): TCM / DM / test RAM 1, control window 3, core window 9, SPM window 11, line-buffer
// hit 1 (miss: 11). The core window and the SPM bridge cross into the core clock domain: their
// latencies are TSR_BUS_XING_RV clk_rv cycles plus a core-domain part of TSR_BUS_XING_*_CC core
// cycles, i.e. TSR_BUS_XING_RV + round(cc x F_RV / F_CORE) RV cycles (tsr_bus_set_ratio; the
// agent sets the simulator's ratio). At the planning ratio 166.7 / 276.3 that is 9 and 11; at
// 250 / 276.3 it is 13 and 16, at 125 / 276.3 7 and 9. The control window is clk_rv registers
// (CC §5.1) and keeps 3 at every ratio. The instruction's own base cost (lsu) comes from the
// interpreter's table (tsr_rv.h: Ibex small 1, maxperf 0).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_rv.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_bus tsr_bus;

#define TSR_BUS_LAT_TCM   1
#define TSR_BUS_LAT_CW    3
#define TSR_BUS_LAT_CORE  9      // at the planning ratio (TSR_BUS_XING_RV + round(12 x 166.7 / 276.3))
#define TSR_BUS_LAT_SPM   11     // at the planning ratio (TSR_BUS_XING_RV + round(15 x 166.7 / 276.3))
#define TSR_BUS_LAT_LBHIT 1
#define TSR_BUS_XING_RV      2   // clk_rv cycles of a crossing access (request and response registers)
#define TSR_BUS_XING_CORE_CC 12  // core cycles of a core-window load (CC §5.3: 50-70 ns in total)
#define TSR_BUS_XING_SPM_CC  15  // core cycles of an SPM-bridge transaction (CC §10.5: 55-80 ns load)

// Return codes of the environment's control-window callback (the same values as tsr_sim_int.h's
// TSR_CW_OK / TSR_CW_FAULT / TSR_CW_WAIT).
#define TSR_BUS_ENV_OK    0
#define TSR_BUS_ENV_FAULT 1
#define TSR_BUS_ENV_WAIT  2

// Everything of the firmware profile outside the RISC-V clock domain.
typedef struct tsr_bus_env {
    void * ctx;
    // Control window (CC §5): one aligned word access at offset off (0 .. 0xFFC); a read fills
    // *val. Returns TSR_BUS_ENV_OK, TSR_BUS_ENV_FAULT (access fault, nothing changed) or, for a
    // store, TSR_BUS_ENV_WAIT (the store must be retried later: an issue push without a credit,
    // CC §7.3). NULL: the control window reads 0 and ignores writes.
    int      (*cw)(void * ctx, int is_write, uint32_t off, uint32_t * val);
    // Core window (CC §5.3): side-effect-free read of the host CSR at offset off. The bus answers
    // LOAD_DATA (0x0C4) and the DMI window (0xC00-0xDFC) with 0 itself, without calling it.
    // Returns TSR_BUS_ENV_OK or TSR_BUS_ENV_FAULT. NULL: reads 0.
    int      (*core_read)(void * ctx, uint32_t off, uint32_t * val);
    // SPM bytes for the SPM windows (CC §4.1: window offset i is SPM byte i). NULL: they fault.
    uint8_t * spm;
    uint32_t  spm_bytes;
    // Optional: every SPM transaction the bridge makes (an exact-window access of n bytes, or a
    // line-buffer fill of 64), after the bytes moved. For PERF, the race detector (SW §4.6) and
    // the agent's time slicing.
    void     (*spm_access)(void * ctx, uint32_t off, uint32_t n, int is_write);
    // Interrupt levels (TSR_RV_LINE_*). NULL: the levels set with tsr_bus_set_irq.
    uint32_t (*irq)(void * ctx);
    // Optional: called before every access that leaves the clk_rv domain (control window, core
    // window, SPM bridge); nonzero = the access must happen later (TSR_RV_BUS_YIELD).
    int      (*yield)(void * ctx);
    // Optional: RV_RUN for the SBA refusals (CC §4.3). NULL: the value of tsr_bus_set_rv_run.
    int      (*rv_run)(void * ctx);
    // Optional: extra RV cycles of latency for one hart access that completed (P1.8's random
    // handshake crossings): region TSR_BUS_REGION_CW (off = control-window offset), _CORE (off =
    // CSR offset), _SPM (an exact-window access or a line-buffer fill; off = SPM byte). Added to
    // the access latency; never called for SBA accesses or line-buffer hits. NULL: 0.
    uint32_t (*xlat)(void * ctx, uint32_t region, uint32_t off);
} tsr_bus_env;
#define TSR_BUS_REGION_CW   1u
#define TSR_BUS_REGION_CORE 2u
#define TSR_BUS_REGION_SPM  3u
// 1 while the bus serves a debug-module system-bus access (tsr_bus_sba_*): the env callbacks it
// calls then act for the SBA, not for the hart.
int       tsr_bus_in_sba(const struct tsr_bus * bus);

typedef struct tsr_bus_fw_cfg {
    uint8_t * itcm;          // NULL: allocated and owned by the bus (zeroed)
    uint32_t  itcm_bytes;    // 0: TSR_RV_ITCM_SIZE; at most the reserved range (CC §4.1)
    uint8_t * dtcm;
    uint32_t  dtcm_bytes;    // 0: TSR_RV_DTCM_SIZE
} tsr_bus_fw_cfg;

// Firmware profile (CC §4.1). cfg may be NULL (U200 sizes, owned TCMs); env may be NULL (no
// control window, core window or SPM). Returns NULL on bad sizes.
tsr_bus * tsr_bus_create_fw(const tsr_bus_fw_cfg * cfg, const tsr_bus_env * env);
// Test profile: RAM [ram_base, ram_base + ram_bytes), zeroed.
tsr_bus * tsr_bus_create_test(uint32_t ram_base, uint32_t ram_bytes);
void      tsr_bus_destroy(tsr_bus * bus);

// The interface the interpreter takes (valid while the bus lives): tsr_rv_create(tsr_bus_iface(b)).
const tsr_rv_bus * tsr_bus_iface(tsr_bus * bus);
// The hart HTIF exits stop (tsr_rv_request_stop); may be NULL.
void      tsr_bus_attach(tsr_bus * bus, tsr_rv * rv);

// Direct pointer to [addr, addr + n) if it lies inside one memory of the bus (ITCM, DTCM, the
// DM memory, test RAM), else NULL. For loaders and tests. The DM memory (0x1000_0000, 4 KiB) is a
// plain array the debug-module model (P1.12) lays out (program buffer, data registers): the hart
// fetches from it and accesses it with word loads / stores.
uint8_t * tsr_bus_mem(tsr_bus * bus, uint32_t addr, uint32_t n);

// Observer of the hart's data transactions (DB §3.7 RV watchpoints, P1.13): called once for every
// transaction the hart issues that completes or faults (each half of a split access separately;
// not for WAIT / YIELD attempts, not for SBA), with fault = 1 if it faulted. During the call
// tsr_rv_pc(rv) is the pc of the accessing instruction; tsr_rv_set_debug_req there halts the hart
// right after that instruction. NULL removes it.
typedef void (*tsr_bus_watch_fn)(void * ctx, uint32_t addr, uint32_t n, int is_write, uint32_t val, int fault);
void      tsr_bus_set_watch(tsr_bus * bus, tsr_bus_watch_fn fn, void * ctx);

void      tsr_bus_set_irq(tsr_bus * bus, uint32_t lines);   // levels when env->irq is NULL
// Crossing latencies for F_RV / F_CORE = ratio (see Latencies above); <= 0: the planning values
// TSR_BUS_LAT_CORE / TSR_BUS_LAT_SPM (also the state after create). Returns the SPM latency set.
uint32_t  tsr_bus_set_ratio(tsr_bus * bus, double ratio);
void      tsr_bus_lb_invalidate(tsr_bus * bus);
void      tsr_bus_set_rv_run(tsr_bus * bus, int run);

// The debug module's system-bus master (CC §4.3, DB §3.6): size 1, 2 or 4, naturally aligned.
// Returns 0, or -1 (sberror): unmapped or misaligned, the DM range, any write to the control or
// core window, a write to ITCM / DTCM while RV_RUN = 0. Reads of the windows have no side effects
// (control-window reads go through env->cw; reads have none, CC §5). SBA accesses are not watched
// and never yield.
int       tsr_bus_sba_read(tsr_bus * bus, uint32_t addr, uint32_t size, uint32_t * val);
int       tsr_bus_sba_write(tsr_bus * bus, uint32_t addr, uint32_t size, uint32_t val);

// HTIF (test profile): a store to the 64-bit tohost (low word, or high word for a device
// command) acts as Spike's front end does: odd value = exit with code value >> 1 (0 = pass);
// device 0 even value = syscall block address (SYS_write 64 to fd 1 / 2 and SYS_exit 93 are
// served, others answer -38); device 1 command 1 = putchar. A served command clears tohost and
// sets fromhost to 1. An exit stops the attached hart.
void      tsr_bus_set_htif(tsr_bus * bus, uint32_t tohost, uint32_t fromhost);
// 1 if the program exited through HTIF; *code (if not NULL) = its exit code.
int       tsr_bus_htif_exited(const tsr_bus * bus, uint32_t * code);

typedef struct tsr_bus_stats {
    uint64_t cw_reads, cw_writes, cw_waits, core_reads, spm_reads, spm_writes, lb_hits, lb_fills;
} tsr_bus_stats;
void      tsr_bus_get_stats(const tsr_bus * bus, tsr_bus_stats * out);

// ---------------------------------------------------------------------------
// ELF32 little-endian RISC-V executables (riscv-tests, riscv-arch-test, firmware images)
// ---------------------------------------------------------------------------
typedef struct tsr_rv_elf_seg {
    uint32_t paddr, filesz, memsz;
    const uint8_t * data;    // filesz bytes
} tsr_rv_elf_seg;

typedef struct tsr_rv_elf {
    uint32_t entry;
    uint32_t lo, hi;         // extent of the PT_LOAD segments (by physical address), [lo, hi)
    uint32_t nseg;
    tsr_rv_elf_seg * seg;
    // private
    uint8_t * buf; size_t size;
    uint32_t symoff, symnum, symentsz, stroff, strsize;
} tsr_rv_elf;

// Reads and checks an ELF file (class 32, little-endian, EM_RISCV). 0, or -1 with a message on
// stderr. Free with tsr_rv_elf_free.
int       tsr_rv_elf_read(const char * path, tsr_rv_elf * elf);
void      tsr_rv_elf_free(tsr_rv_elf * elf);
// Value of a symbol from .symtab: 0, or -1 if absent.
int       tsr_rv_elf_symbol(const tsr_rv_elf * elf, const char * name, uint32_t * value);
// Copies every PT_LOAD segment to the bus memories (memsz bytes, zero-filled past filesz).
// 0, or -1 if a segment does not fit one memory.
int       tsr_rv_elf_load(tsr_bus * bus, const tsr_rv_elf * elf);

// ---------------------------------------------------------------------------
// RISC-V agent of the simulator (built into libtsr_sim only; tsr_sim_int.h §4)
//
// Binds a firmware-profile bus to a tsr_sim (TCMs = tsr_sim_itcm / tsr_sim_dtcm, control window =
// tsr_sim_cw_access, core window = tsr_sim_corewin_read, SPM = tsr_sim_spm, interrupt levels =
// tsr_sim_rv_irq, RV_RUN = tsr_sim_rv_run) and registers the hart with tsr_sim_set_rv_agent. On
// the time base it:
//   - resets the hart (RV_BOOT) whenever tsr_sim_rv_boot_gen changes;
//   - maps core time T to RV cycles by F_RV / F_CORE (tsr_sim_rv_ratio) from an anchor taken at
//     each reset; time the hart spends asleep, parked or waiting is credited with tsr_rv_idle;
//   - runs instructions in a slice that ends before the next core event (tsr_sim_next_event) and
//     after the first instruction that left the clk_rv domain: such an access (control window,
//     core window, SPM bridge) is always the first transaction of a slice, so it sees the core
//     state of its own time (a later one yields and starts the next slice);
//   - keeps a push that gets TSR_CW_WAIT pending (no interrupts, no debug halt, CC §7.3), counts
//     its wait in PERF PUSH_STALL and after TSR_ISSUE_WAIT_MAX RV cycles calls
//     tsr_sim_cw_push_timeout and takes a store access fault; SPM bridge transactions count in
//     BRIDGE_ACC and BRIDGE_WAIT (the modelled latency);
//   - reports STATUS.RV_HALTED (debug mode) and RV_SLEEP (WFI) with tsr_sim_rv_set_state, and
//     sleeps (TSR_SIM_AGENT_WAIT) while asleep or parked;
//   - passes the accessing instruction's pc with every control-window access
//     (tsr_sim_cw_access_pc: the push PC of CC §7.1) and reports the hart's SPM-window and
//     core-window accesses as access events (tsr_sim_acc_emit / tsr_sim_corewin_note); accesses the
//     bus makes for the SBA are not the hart's and are not reported here;
//   - under the random schedule (tsr_sim_int.h §5b): adds tsr_sched_rv_cpi() to the cost of every
//     instruction (effective cycles = the hart's cycles + the accumulated deltas; each instruction
//     keeps >= 1 cycle; the slice budget is checked in effective cycles) and
//     tsr_sched_rv_handshake() cycles to every LQ / LCQ / log register, core-window and SPM-bridge
//     access;
//   - fast-forward (tsr_sim_int.h §5b): counts reads of the timing-dependent registers; at the 4th
//     identical read in a row with no store or other MMIO access in between it ends the slice
//     and sleeps until the next time point (TSR_SIM_AGENT_WAIT; never earlier than the hart's own
//     time), unless tsr_sim_fast_forward() is 0;
//   - TSR_SIM_LOG with TSR_SIM_LOG_RV: one line per retired instruction ("<MTIME> rv pc=.. insn=..
//     [x<rd>=..] [ld|st <addr>=<val>] [trap <mcause>]", MTIME = the core time the instruction ends).
// The agent owns the hart's retire hook (tsr_rv_set_retire_hook) when it needs it (random schedule
// or RV log lines); other models chain theirs with tsr_rv_agent_set_retire_hook instead.
// tsr_sim_create creates one (the built-in agent, tsr_sim_rv_agent in tsr_sim_int.h) and
// tsr_sim_destroy destroys it; the simulator's users never call these two. A test may still create
// another agent on the same simulator (it replaces the built-in one as the time base's agent, and
// the test destroys it before tsr_sim_destroy); test_rv_agent.c uses them on a stub time base.
// ---------------------------------------------------------------------------
typedef struct tsr_rv_agent tsr_rv_agent;
struct tsr_sim;
tsr_rv_agent * tsr_rv_agent_create(struct tsr_sim * sim);
void           tsr_rv_agent_destroy(tsr_rv_agent * a);
tsr_rv *       tsr_rv_agent_rv(tsr_rv_agent * a);
tsr_bus *      tsr_rv_agent_bus(tsr_rv_agent * a);
// Spin fast-forward jumps taken since create (tsr_sim_int.h §5b).
uint64_t       tsr_rv_agent_ff_jumps(const tsr_rv_agent * a);
// A retire hook called after the agent's own for every record (NULL removes it); its return value
// stops the slice as with tsr_rv_set_retire_hook.
void           tsr_rv_agent_set_retire_hook(tsr_rv_agent * a, tsr_rv_retire_fn fn, void * ctx);
// P1.12: reports STATUS.RV_HALTED / RV_SLEEP of the hart now (tsr_sim_rv_set_state), for the DM
// model, which changes the hart outside the agent's step (a resume); the agent's own reports
// continue from this state.
void           tsr_rv_agent_sync(tsr_rv_agent * a);

#ifdef __cplusplus
}
#endif

// tsr_bus.h — the RISC-V bus of the simulator: the address maps tsr_rv runs on (SW §4.2).
//
// Two profiles (docs/v2/06-software-verification.md "SW" §4.2):
//   firmware  the control core's map (docs/v2/04-control-core.md "CC" §4.1, TSR_RV_* in
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
// hit 1 (miss: 11).
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
#define TSR_BUS_LAT_CORE  9
#define TSR_BUS_LAT_SPM   11
#define TSR_BUS_LAT_LBHIT 1

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
} tsr_bus_env;

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
//     sleeps (TSR_SIM_AGENT_WAIT) while asleep or parked.
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

#ifdef __cplusplus
}
#endif

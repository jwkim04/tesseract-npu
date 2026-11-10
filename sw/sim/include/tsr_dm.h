// tsr_dm.h — debug-module model of the simulator (docs/design/05-debug.md "DB" §3, §9.1;
// docs/design/04-control-core.md "CC" §3.2, §4.3, §12.2; docs/design/06-software-verification.md "SW"
// §4.2). Implementation: sw/sim/tsr_dm.c.
//
// A behavioral riscv-dbg dm_top (Debug Spec 0.13.2, NrHarts 1, BusWidth 32, ProgBufSize 8,
// DataCount 2, DmBaseAddress 0x1000_0000, two-scratch ROM; DB §3.1, CC §3.2) on the simulator's
// built-in hart (tsr_rv) and bus (tsr_bus). It implements DB §9.1's DMI profile exactly; every
// other DM register reads 0 and ignores writes (abstractauto, haltsum*, authdata, nextdm, ...).
// Abstract commands act on the interpreter state directly and the program buffer runs in the
// interpreter in debug mode (DB-18); the debug ROM is not modeled (tsr_rv.h).
//
// DMI registers (7-bit addresses TSR_DMI_* in tsr_tables.h; host CSR 0xC00 + 4r through the DBG
// model's window, gated by DBG_CFG.DMI_EN, DB §3.2; in a library without the DBG model the core
// answers the whole window with 0 and drops writes, so the DM is reachable only through
// tsr_dm_dmi_read / _write):
//   data0-1 (0x04-0x05)    R/W. Memory-mapped for the hart at DATA_ADDR (0x1000_0380 / 0x384,
//                          riscv-dbg's DataAddr; hartinfo.dataaccess = 1)
//   dmcontrol (0x10)       haltreq [31] (per hart, a level into Ibex debug_req_i, ORed with DB's
//                          request, tsr_dm_set_ext_haltreq), resumereq [30] (W1: resumes a hart
//                          that is halted at the write; ignored with haltreq set in the same write;
//                          clears then sets resumeack), ackhavereset [28] (W1), hartsello [16]
//                          (1 bit, riscv-dbg's HartSelLen for one hart: hart 1 is nonexistent),
//                          ndmreset [1] (the RISC-V reset sequence, CC §12.2, held while 1),
//                          dmactive [0] (0: the DM's state takes its reset values; while 0 only
//                          dmactive can be written). hartreset, hasel, setresethaltreq /
//                          clrresethaltreq read 0 (resethaltreq is not relied on, DB §3.8).
//                          Reads return haltreq, hartsello, ndmreset, dmactive
//   dmstatus (0x11)        version 2, authenticated 1, impebreak 0, hasresethaltreq 0, and for the
//                          selected hart all* = any* (one hart): halted = RV_HALTED (the hart in
//                          debug mode, parked or running the program buffer: DB §3.1's "last fetch
//                          in the DM range", CC-29), running, unavail (the hart held in reset:
//                          CTRL.RV_RUN = 0, ndmreset, or a boot not yet taken by the time base),
//                          resumeack, havereset (1 after create; set again whenever the hart
//                          leaves a reset; ackhavereset clears it), nonexistent (hartsel 1)
//   hartinfo (0x12)        0x0021_2380: nscratch 2, dataaccess 1, datasize 2, dataaddr 0x380
//   abstractcs (0x16)      progbufsize 8, busy [12], cmderr [10:8] (W1C), datacount 2
//   command (0x17)         W (reads 0). cmdtype 0 Access Register only (others: cmderr 2):
//                          aarsize 2 (32-bit; others with transfer: cmderr 2), aarpostincrement
//                          (accepted; without abstractauto it has no visible effect), postexec,
//                          transfer, write, regno: 0x0000-0x0FFF CSR (tsr_rv_csr_read / _write in debug mode, WARL
//                          applied; an absent or read-only CSR: cmderr 3 exception), 0x1000-0x101F
//                          GPR x0-x31 (x0 writes ignored), 0x1020-0x103F FPR (no F: cmderr 3),
//                          others cmderr 2. pc is dpc (CSR 0x7B1). The hart must be halted and
//                          not running a command (cmderr 4 otherwise; with cmderr != 0 no command
//                          starts). postexec runs progbuf0-7 from PROGBUF_ADDR in debug mode until
//                          ebreak (done) or an exception (cmderr 3, no CSR changed: Ibex jumps to
//                          DmExceptionAddr). DB §3.5 uses only fence.i + ebreak; any program works
//   progbuf0-7 (0x20-0x27) R/W, at PROGBUF_ADDR (0x1000_0360, riscv-dbg's layout). impebreak 0:
//                          the debugger ends the program with ebreak (running past progbuf7 fetches
//                          data0 as an instruction)
//   sbcs (0x38)            sbversion 1, sbbusyerror [22] (W1C), sbbusy [21] (reads 0, below),
//                          sbreadonaddr [20], sbaccess [19:17] (reset 2), sbautoincrement [16],
//                          sbreadondata [15], sberror [14:12] (W1C), sbasize 32, sbaccess8/16/32
//   sbaddress0 (0x39)      R/W; a write starts a read with sbreadonaddr
//   sbdata0 (0x3C)         R/W; a write starts a write; a read returns the data and, with
//                          sbreadondata, starts the next read. With sberror or sbbusyerror != 0
//                          accesses do nothing. Sub-word reads are zero-extended (*confirm*
//                          riscv-dbg's lane handling)
// Accesses while busy (a running command): data / progbuf / command / abstractcs writes and data /
// progbuf reads set cmderr = 1 if it is 0 and do nothing else (Debug Spec 0.13.2 §3.12).
//
// SBA (DB §3.6, CC §4.3): through the RISC-V bus model's system-bus master (tsr_bus_sba_read /
// _write), so CC's refusals apply: the DM range, every control-window and core-window write, ITCM /
// DTCM writes while CTRL.RV_RUN = 0, unmapped addresses -> sberror 2 (bad address; the bus answers
// an error, *confirm* riscv-dbg's code for a bus error); an address not aligned to the access size
// -> sberror 3; sbaccess 3 or 4 (64 / 128-bit) -> sberror 4. sbautoincrement adds the size after a
// successful access only. Every SBA access is reported as an access event (tsr_sim_int.h,
// TSR_ACC_SBA_RD / TSR_ACC_SBA_WR: addr, size, val) and is never seen by the hart's data watch.
//
// Timing (model decisions): every DMI access acts at the current simulated time
// (tsr_sim_now), between time-base steps. SBA accesses complete within the DMI access that starts
// them, so sbbusy reads 0 and sbbusyerror is never set by the model (the bit and its W1C exist).
// An abstract command completes within the command write: busy reads 0 again at once, unless its
// program buffer has not parked after PB_SYNC_INSNS (1000) instructions or waits on a store
// (TSR_RV_BUS_WAIT): then busy stays 1 and the hart continues on the time base until it parks
// (cmderr 1 is reachable only then). A halt request takes effect at the hart's next instruction
// boundary on the time base (the caller advances time, e.g. tsr_sim_advance until
// TSR_SIM_EV_HALT), and not while a push waits for a credit (CC §7.3, DB §3.4: the hart halts
// after the push completes, dpc = the next instruction). The RISC-V agent runs the hart in slices
// that may end ahead of MTIME (up to the next core event, at most 4096 instructions; tsr_bus.c),
// so a request made between tsr_sim_advance calls is seen at the start of the next slice: the
// halt point is where the hart then is (the halt latency of DB §3.4, deterministic for a given
// schedule; never architectural). dmstatus and tsr_dm_rv_halted show the hart's own state at
// once; STATUS.RV_HALTED, the time base's copy, rises when MTIME reaches the core time at which
// the hart entered debug mode (the agent reports it then, tsr_bus.c). A resume acts at once (dpc,
// dcsr.prv, dcsr.step) and STATUS.RV_HALTED / allrunning follow at once. ndmreset 1 applies the RISC-V reset sequence
// (tsr_sim_rv_ndmreset) and holds the hart in reset; 0 releases it, and the hart restarts at
// RV_BOOT + 0x80 at the next time point (with haltreq still set it halts before its first
// instruction: "reset halt").
//
// State: the DM follows only the simulator's create / destroy (the hardware reset ap_rst_n): its
// registers, the haltreq level, data / progbuf / SBA state and dmactive survive RISC-V resets
// (RV_RUN 1 -> 0, ndmreset) and SOFT_RESET (DB §3.1, EX §8.5). An abstract command still running
// when the hart is reset ends with cmderr 4 (busy 0). A dret in the program buffer leaves debug
// mode and ends the command with cmderr 3 (unsupported, not used by DB).
//
// Halt-on-boot (DB §3.8) and DB's other halt sources (§3.3) live in tsr_dbg.c: it drives the
// external request level with tsr_dm_set_ext_haltreq and sees the hart's debug events through
// tsr_dm_set_event_hook. A request pending when the hart leaves reset (set before the release, or
// from the TSR_RV_DBG_EV_RESET event) halts it before its first instruction with cause haltreq and
// dpc = RV_BOOT + 0x80. The DM gates nothing: DB's one-shot rule (pend cleared by RV_HALTED) is
// the caller's.
#pragma once

#include <stdint.h>

#include "tsr_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_dm tsr_dm;

// DM memory layout (riscv-dbg dm_pkg / dm_mem at DmBaseAddress 0x1000_0000, *confirm*).
#define TSR_DM_PROGBUF_ADDR   0x10000360u   // progbuf0 (ProgBufBaseAddr = DataAddr - 4 x 8)
#define TSR_DM_DATA_ADDR      0x10000380u   // data0 (DataAddr)
#define TSR_DM_HARTINFO       0x00212380u
#define TSR_DM_PROGBUF_SIZE   8u
#define TSR_DM_DATA_COUNT     2u

// dmcontrol
#define TSR_DM_DMCONTROL_HALTREQ      (1u << 31)
#define TSR_DM_DMCONTROL_RESUMEREQ    (1u << 30)
#define TSR_DM_DMCONTROL_ACKHAVERESET (1u << 28)
#define TSR_DM_DMCONTROL_HARTSELLO_LSB 16u
#define TSR_DM_DMCONTROL_NDMRESET     (1u << 1)
#define TSR_DM_DMCONTROL_DMACTIVE     (1u << 0)
// dmstatus
#define TSR_DM_DMSTATUS_VERSION       0x0000000Fu   // 2 = Debug Spec 0.13
#define TSR_DM_DMSTATUS_AUTHENTICATED (1u << 7)
#define TSR_DM_DMSTATUS_ANYHALTED     (1u << 8)
#define TSR_DM_DMSTATUS_ALLHALTED     (1u << 9)
#define TSR_DM_DMSTATUS_ANYRUNNING    (1u << 10)
#define TSR_DM_DMSTATUS_ALLRUNNING    (1u << 11)
#define TSR_DM_DMSTATUS_ANYUNAVAIL    (1u << 12)
#define TSR_DM_DMSTATUS_ALLUNAVAIL    (1u << 13)
#define TSR_DM_DMSTATUS_ANYNONEXISTENT (1u << 14)
#define TSR_DM_DMSTATUS_ALLNONEXISTENT (1u << 15)
#define TSR_DM_DMSTATUS_ANYRESUMEACK  (1u << 16)
#define TSR_DM_DMSTATUS_ALLRESUMEACK  (1u << 17)
#define TSR_DM_DMSTATUS_ANYHAVERESET  (1u << 18)
#define TSR_DM_DMSTATUS_ALLHAVERESET  (1u << 19)
#define TSR_DM_DMSTATUS_IMPEBREAK     (1u << 22)
// abstractcs
#define TSR_DM_ABSTRACTCS_DATACOUNT   0x0000000Fu
#define TSR_DM_ABSTRACTCS_CMDERR_LSB  8u
#define TSR_DM_ABSTRACTCS_CMDERR      (7u << 8)
#define TSR_DM_ABSTRACTCS_BUSY        (1u << 12)
#define TSR_DM_ABSTRACTCS_PROGBUFSIZE_LSB 24u
// cmderr values
#define TSR_DM_CMDERR_NONE      0u
#define TSR_DM_CMDERR_BUSY      1u
#define TSR_DM_CMDERR_NOTSUP    2u
#define TSR_DM_CMDERR_EXCEPTION 3u
#define TSR_DM_CMDERR_HALTRESUME 4u
// command (Access Register)
#define TSR_DM_CMD_TYPE_LSB      24u
#define TSR_DM_CMD_AARSIZE_LSB   20u
#define TSR_DM_CMD_POSTINC       (1u << 19)
#define TSR_DM_CMD_POSTEXEC      (1u << 18)
#define TSR_DM_CMD_TRANSFER      (1u << 17)
#define TSR_DM_CMD_WRITE         (1u << 16)
#define TSR_DM_REGNO_GPR(i)      (0x1000u + (i))
// sbcs
#define TSR_DM_SBCS_SBVERSION_LSB    29u
#define TSR_DM_SBCS_SBBUSYERROR      (1u << 22)
#define TSR_DM_SBCS_SBBUSY           (1u << 21)
#define TSR_DM_SBCS_SBREADONADDR     (1u << 20)
#define TSR_DM_SBCS_SBACCESS_LSB     17u
#define TSR_DM_SBCS_SBACCESS         (7u << 17)
#define TSR_DM_SBCS_SBAUTOINCREMENT  (1u << 16)
#define TSR_DM_SBCS_SBREADONDATA     (1u << 15)
#define TSR_DM_SBCS_SBERROR_LSB      12u
#define TSR_DM_SBCS_SBERROR          (7u << 12)
#define TSR_DM_SBCS_SBASIZE_LSB      5u
#define TSR_DM_SBCS_ACCESS_8_16_32   0x7u
#define TSR_DM_SBERR_BADADDR    2u
#define TSR_DM_SBERR_ALIGN      3u
#define TSR_DM_SBERR_SIZE       4u

// Optional-model entry points (tsr_sim_int.h §5): attach creates the DM on the built-in hart
// (nothing without the RISC-V model); detach frees it. tsr_sim_dm returns the attached DM, NULL
// without it (a library built without tsr_dm.c has no tsr_sim_dm: test TSR_SIM_HAVE_DM).
void     tsr_dm_attach(tsr_sim * sim);
void     tsr_dm_detach(tsr_sim * sim);
tsr_dm * tsr_sim_dm(tsr_sim * sim);

// DMI access (DB §3.2; the DBG model forwards host CSR 0xC00 + 4r here while DMI_EN = 1, and the
// sim transport's DMI helpers call it directly, DB §9.2). addr7 = the DM register address; reads
// may have side effects (sbdata0 with sbreadondata, cmderr on a busy access). Must not be called
// from inside a time-base callback (agent step, listener).
uint32_t tsr_dm_dmi_read(tsr_dm * dm, uint32_t addr7);
void     tsr_dm_dmi_write(tsr_dm * dm, uint32_t addr7, uint32_t val);

// DB's dbg_req_rv (DB §3.3): a level ORed with dmcontrol.haltreq into the hart's debug request.
// Callable from the debug models' hooks (also from inside the time base: a queue listener, the
// hart's data watch, a debug event).
void     tsr_dm_set_ext_haltreq(tsr_dm * dm, int level);
// RV_HALTED (DB §3.1): the hart is out of reset and in debug mode (parked, or running the program
// buffer). Immediate (STATUS.RV_HALTED is the time base's copy, reported by the RISC-V agent).
int      tsr_dm_rv_halted(const tsr_dm * dm);

// The hart's debug events, forwarded (tsr_rv.h TSR_RV_DBG_EV_*: ENTER with the dcsr.cause, PARK,
// RESUME, RESET with the boot address) after the DM has updated its own state. The callback may
// read state and call tsr_dm_set_ext_haltreq; it runs inside the time base (agent step) or inside
// a DMI access. One slot; NULL removes it.
typedef void (*tsr_dm_event_fn)(void * ctx, uint32_t ev, uint32_t arg);
void     tsr_dm_set_event_hook(tsr_dm * dm, tsr_dm_event_fn fn, void * ctx);

#ifdef __cplusplus
}
#endif

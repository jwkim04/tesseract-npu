// tsr_dbg.h — DB simulator models (docs/design/05-debug.md "DB", docs/design/06-software-
// verification.md "SW" §4.3): the tile-level debug logic, the DBG register block, DB's RISC-V
// control-window registers, the profiler's record paths, and the decoders of their formats.
//
// The model (sw/sim/tsr_dbg.c) attaches to a simulator at create (tsr_sim_create_opts calls
// tsr_dbg_attach when the library was built with it, TSR_SIM_HAVE_DBG; after the DM model) and
// works only through the simulator's hooks (tsr_sim_int.h: the debug hooks, a queue listener,
// the hart's data watch, and the hart's debug events through the DM model's event hook, or the
// hart's own debug hook without the DM). It implements, per DB §9.1 "queue model, §5 semantics
// exactly":
//   - DB §10.1 DBG block, host CSR 0x800-0xAFC except SREG 0x9C0-0x9FC (the VU's), with every
//     register's access type; reads have no side effects and never wait (DB §7.1);
//   - DB §3.2 DMI window 0xC00-0xDFC forwarded to the DM model (tsr_dm.h) while DBG_CFG.DMI_EN = 1;
//     with 0 a read returns 0 and a write is dropped with DBG_STATUS.DMI_BLOCKED. Without the DM
//     model the window reads 0. (Without this model the core answers the whole window with 0,
//     tsr_sim_int.h §5: the DMI window needs the DBG model even when the DM model is present.)
//     DBG_STATUS.SBA_REFUSED: an SBA access through the window that CC's rules refuse;
//   - DB §5.2 stop logic (S, the match queue M, T, dbg_hold[u] as CC's hard hold, DRAINED,
//     STOP_NOW / STOP_AT / RELEASE / QSTEP with DBG_STEP_N and dbg_step[u], ERR_STOP, HUNG_STOP,
//     MATCHQ_CLEAR), DB §5.3 tile breakpoints on the dispatcher's enqueue tap, DB §5.4 serial
//     mode, DB §5.5 unit probes (model values), DB §5.6 dispatch history (512 entries, PRESENT,
//     HIST_FREEZE_ERR / HIST_FREEZE / HIST_CLEAR, push PC), DB §6.2 RV_SAMPLE (model values);
//   - DB §3.3 halt requests (pend, one-shot, RV_HALT_CANCEL), §3.7 RV watchpoints with the
//     RVWP_APPLY handshake, §3.8 HALT_ON_BOOT, §3.9 dbg_stop (DBG_FLAGS, the hart's
//     ISSUE_WAIT_MAX count frozen while it is 1);
//   - DB §8.2 REC FIFO (64 records, irq_fast[4]), §8.3 trace writer (32-record staging, DRAM ring
//     on DBG_TRW_BASE, flush rules, drops and DROP records), §8.4 markers, §8.1 PERF slots 61-63
//     (TRW_RECORDS, TRW_DROPS; DBG_STOP_CYCLES through the core's time integration);
//   - SOFT_RESET step 1 (EX §8.5): S, S_VALID, the match queue and a QSTEP in progress are
//     cleared; TBPs, SERIAL, the configuration and the evidence (history, TRW pointers, RV
//     sample) are kept, as they are across QABORT, QCLEAR and the RISC-V reset.
// DB §5.8 / §10.3: with no stop, serial mode, breakpoint or trace configured the model changes no
// SW §4.7 observable (it adds no cycle to dispatch or start); test_sim_dbg checks it.
//
// Model values (DB §9.1 "not compared"): unit probes DBG_Uu_*, DBG_QSTATE, DBG_RDOUT / WROUT (0:
// no memory-arbiter model), the RV sample (current_pc / next_pc = the hart's pc, last data
// address = the last watched data access, exception pc / address = mepc / mtval). Times come from
// the queue model's cycle model. Not modeled: DMI_RESP_ERR and RV_DFAULT (never set: the DM model
// answers 0, the hart reports no double fault), RVWP_BUSY (the apply completes at once), HUNG_STOP's
// trigger (the core has no progress watchdog yet: STATUS.UNIT_HUNG never rises).
// A tsr_sim_advance returns TSR_SIM_EV_HALT when DRAINED rises and when a QSTEP completes.
//
// The helpers below decode DB's record formats and read the DBG block through the host CSR port
// (tsr_sim_csr_read / _write: what the host sees), for tests, tools and hang dumps (DB §7.4).
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "tsr_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_dbg tsr_dbg;

// Optional-model entry points (tsr_sim_int.h): attach installs the debug hooks, a queue listener
// and the hart's data watch; detach removes them (a simulator without the model behaves as built
// without it). tsr_sim_dbg returns the attached model, NULL without it.
void      tsr_dbg_attach(tsr_sim * sim);
void      tsr_dbg_detach(tsr_sim * sim);
tsr_dbg * tsr_sim_dbg(tsr_sim * sim);

// ---------------------------------------------------------------------------
// Log records (DB §6.1; header fields TSR_LOG_HDR_* in tsr_tables.h)
// ---------------------------------------------------------------------------
typedef struct tsr_log_hdr {
    uint32_t n;         // payload words after the header
    uint32_t level;     // TSR_LOG_HDR_LEVEL_*
    uint32_t source;    // TSR_LOG_HDR_SOURCE_*
    uint32_t type;      // TSR_LOG_HDR_TYPE_*
    uint32_t ts;        // 1: payload word 0 is MTIME[31:0]
    uint32_t lseq;      // SEQ[7:0] of the running launch
    uint32_t sync;      // 0xA in a valid header
} tsr_log_hdr;

uint32_t tsr_log_hdr_make(uint32_t n, uint32_t level, uint32_t source, uint32_t type, int ts, uint32_t lseq);
void     tsr_log_hdr_decode(uint32_t hdr, tsr_log_hdr * out);

// DB §6.1 decoder check of the record that starts at w[0], with `avail` words readable: returns 0
// for a valid record (*words, if not NULL, = 1 + N), else one of the reasons below. fmt_lo /
// fmt_hi bound the .tsr_logfmt section for FMT records ([lo, hi); fmt_hi = 0 skips the check).
// A reader that gets a nonzero reason reports a firmware log bug and skips to LOG_WR (DB §6.1).
#define TSR_LOG_BAD_SYNC    1   // SYNC != 0xA
#define TSR_LOG_BAD_TYPE    2   // TYPE 5..7
#define TSR_LOG_BAD_LEVEL   3   // LEVEL > 5
#define TSR_LOG_BAD_LEN     4   // 1 + N words not available (torn record: never written so)
#define TSR_LOG_BAD_FMT     5   // FMT: the format address is outside .tsr_logfmt
#define TSR_LOG_BAD_TRAP    6   // TRAP: N != 44 (+1 with TS)
int tsr_log_rec_check(const uint32_t * w, size_t avail, uint32_t fmt_lo, uint32_t fmt_hi, uint32_t * words);

// ---------------------------------------------------------------------------
// Completion records to the RISC-V (DB §8.2): REC_DATA0..3 of the control window
// ---------------------------------------------------------------------------
typedef struct tsr_dbg_rec {
    uint32_t ticket;    // {unit, seq}; markers: unit 14 (TSR_DBG_REC_MARK_TICKET)
    uint32_t t_end;     // MTIME[31:0] at completion; markers: the dispatch time
    uint32_t run;       // start -> end core cycles (saturating), 0 if not run; markers: the value
    uint32_t code, ran, marker, opcode;
    uint32_t pidx16;    // push index [15:0] (markers: the next push index)
} tsr_dbg_rec;
#define TSR_DBG_REC_MARK_TICKET 0xE0000000u   // unit 14, seq 0
void tsr_dbg_rec_decode(const uint32_t d[4], tsr_dbg_rec * out);

// ---------------------------------------------------------------------------
// Trace-writer records (DB §8.3; word 0 fields TSR_TRW_W0_* in tsr_tables.h), 32 bytes
// little-endian, record k of the ring at byte (DBG_TRW_BASE.beat * 64 + 32 (k mod 2 LINES)) of
// channel DBG_TRW_BASE.ch.
// ---------------------------------------------------------------------------
typedef struct tsr_trw_rec {
    uint32_t w[8];      // raw words
    uint32_t kind;      // TSR_TRW_W0_KIND_*
    uint32_t unit, code, ran, rec, tbp, opcode, w0_hi;   // OP
    uint32_t a1;        // OP: push index; MARK: next push index; STOP: T; DROP: records dropped
    uint32_t a2;        // OP: ticket; MARK: value; STOP: cause bits (TSR_DBG_CAUSE_*); RVHALT: 1 / 0
    uint64_t time;      // OP: t_end; MARK: dispatch time; STOP / RVHALT / DROP: event time
    uint32_t run;       // OP: RUN = t_end - t_start (0 if not run); STOP: 1 activated / 0 released
    uint32_t qwait;     // OP: t_head - t_dispatch
    uint32_t hwait;     // OP: t_start - t_head (not run: t_end - t_head)
} tsr_trw_rec;
void tsr_trw_decode(const uint8_t rec[32], tsr_trw_rec * out);

// ---------------------------------------------------------------------------
// Host-port readers (DB §5.5, §5.6; each is a sequence of tsr_sim_csr_* accesses)
// ---------------------------------------------------------------------------
typedef struct tsr_dbg_hist {
    int      present;                    // INFO.PRESENT
    uint32_t info;                       // raw DBG_HIST_INFO
    uint32_t unit, code, rec, tbps, stop_point;
    uint32_t pidx, ticket, dep[2], time, pc;
    uint32_t w[8];
} tsr_dbg_hist;
// Writes DBG_HIST_IDX = pidx and reads the entry's fields (1 W + 17 R).
void tsr_dbg_hist_read(tsr_sim * sim, uint32_t pidx, tsr_dbg_hist * out);

typedef struct tsr_dbg_unit {
    uint32_t state;                      // raw DBG_Uu_STATE
    uint32_t accepted;                   // [1:0] ops accepted and not done
    uint32_t head_state;                 // [6:4] TSR_DBG_U_STATE_HEAD_* (CC §9.3 names)
    uint32_t hold_src;                   // [11:8] host, RISC-V, debug halt, DB (TSR_HOLD_SRC_*)
    uint32_t outstanding;                // [19:12]
    uint32_t oldest_ticket, oldest_pidx, age;
} tsr_dbg_unit;
void tsr_dbg_unit_read(tsr_sim * sim, uint32_t u, tsr_dbg_unit * out);

// Hang-dump helper (DB §7.4 "units.txt" / the DBG part of state.json, text form): every DBG
// register raw and decoded, the unit probes, and the last `hist_n` history entries.
void tsr_dbg_dump(tsr_sim * sim, FILE * f, uint32_t hist_n);

#ifdef __cplusplus
}
#endif

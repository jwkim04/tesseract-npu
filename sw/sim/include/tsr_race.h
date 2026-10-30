// tsr_race.h — the simulator's race detector and the launch-stateless check (P1.9;
// docs/design/06-software-verification.md "SW" §4.6, SW-3 / SW-4; memory model
// docs/ctrl-riscv.md §5.5 and EX §2.3). Implementation and the exact rules: tsr_race.c.
//
// The detector is an optional model of the simulator library (sw/sim/race.cmake adds tsr_race.c;
// tsr_sim_create_opts attaches it, tsr_sim_int.h §5). It is active when tsr_sim_opts.race != 0 or,
// with race left 0, when TSR_SIM_RACE is report / fatal (SW §4.9):
//   0 off      nothing is tracked (the default for model runs);
//   1 report   every report is collected, readable with tsr_sim_race_reports(), and printed to
//              stderr by tsr_sim_destroy;
//   2 fatal    the first report is printed to stderr and the process aborts (ctest's mode).
//
// What it checks, over the agents {units 0-5, the RISC-V, the host}:
//   - races (SW-3): two accesses to the same SPM byte, DRAM beat or issue-bank s register, at
//     least one a write, by two agents, not ordered by the happens-before edges of SW §4.6.
//     Reported once per (kind, pair of sites) with both sides; later hits of the same pair only
//     increase `count`;
//   - SPM_STALE (SW-4): while a launch runs (from its LQ_POP to its answer post), an op of the
//     launch reads an SPM line no byte of which was written since that LQ_POP, or the RISC-V
//     reads an SPM byte not written since it. Reported once per reading site.
// A site is an op's push (unit, opcode, push pc, ra at the push: the call site of a push helper)
// — or (unit, push index) for an op pushed by a test hook rather than by the hart — the
// RISC-V's pc, or the host's access direction.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

// Agents: units 0-5 are TSR_UNIT_* (CC §7.4).
#define TSR_RACE_AGENT_RV   6u
#define TSR_RACE_AGENT_HOST 7u
#define TSR_RACE_AGENTS     8u
#define TSR_RACE_AGENT_NONE 0xFFu

// Report kinds (first = the earlier access in simulated time, second = the later one).
#define TSR_RACE_RAW        1u   // write, then an unordered read
#define TSR_RACE_WAR        2u   // read, then an unordered write
#define TSR_RACE_WAW        3u   // write, then an unordered write
#define TSR_RACE_SPM_STALE  4u   // first = the read; second.agent = TSR_RACE_AGENT_NONE

// Location spaces of a report's [lo, hi).
#define TSR_RACE_SPACE_SPM  1u   // SPM byte addresses (an op's line l covers [64 l, 64 l + 64))
#define TSR_RACE_SPACE_DRAM 2u   // DRAM beats of channel ch
#define TSR_RACE_SPACE_SREG 3u   // issue-bank s-register indices 0..15

typedef struct tsr_race_side {
    uint32_t agent;       // 0-5 unit, TSR_RACE_AGENT_RV, TSR_RACE_AGENT_HOST (NONE: no side)
    uint32_t write;       // 1 write, 0 read
    uint32_t ticket;      // op: its ticket {unit, seq}; else TSR_TICKET_NONE
    uint32_t pidx;        // op: its push index; else 0
    uint32_t opcode;      // op: w0[7:0]; else 0
    uint32_t pc;          // op: the push pc (0: pushed by a test hook); RISC-V: the access's pc; host: 0
    uint32_t ra;          // op pushed by the hart: x1 (ra) at the push, i.e. the caller of a push helper
    uint32_t launch;      // the launch (descriptor sequence) it belongs to, 0xFFFFFFFF if none
    uint32_t detail;      // host: 1 DRAM, 2 CSR (SREG) access; op: 1 if it left the detector's
                          // history (only its ticket and time are known); else 0
    uint32_t rsvd;
    uint64_t time;        // MTIME of the access (an op's: its completion, when its effects apply)
} tsr_race_side;

typedef struct tsr_race_report {
    uint32_t      kind;     // TSR_RACE_*
    uint32_t      space;    // TSR_RACE_SPACE_*
    uint32_t      ch;       // DRAM channel (space DRAM), else 0
    uint32_t      count;    // occurrences folded into this report (>= 1)
    uint64_t      lo, hi;   // the first occurrence's locations, [lo, hi)
    uint32_t      launch;   // SPM_STALE: the running launch; else that of `second` (or 0xFFFFFFFF)
    uint32_t      rsvd;
    uint64_t      since;    // SPM_STALE: MTIME of that launch's LQ_POP; else 0
    tsr_race_side first;
    tsr_race_side second;
} tsr_race_report;

// The collected reports (report mode; fatal stops at the first), in the order they were found.
// Returns their number; *out (if not NULL) points to them until the next report or clear, or is
// NULL when there are none or the detector is off / not linked in.
size_t tsr_sim_race_reports(const tsr_sim * sim, const tsr_race_report ** out);
// Drops the collected reports and their dedup state (tests). The shadow state stays: a later
// access can report an earlier one again.
void   tsr_sim_race_clear(tsr_sim * sim);
// 1 if the detector is active on this simulator (opts.race / TSR_SIM_RACE), else 0.
int    tsr_sim_race_active(const tsr_sim * sim);
// One line describing r (no newline), snprintf-style: returns the length it needed.
int    tsr_race_format(const tsr_race_report * r, char * buf, size_t n);

// Counters (diagnostics, overhead measurement). Returns 0, or -1 when the detector is off.
typedef struct tsr_race_stats {
    uint64_t ops;            // completions processed (running and non-running)
    uint64_t op_lines;       // SPM lines, DRAM segments and s registers visited by op accesses
    uint64_t rv_accesses;    // RISC-V SPM / control-window / core-window accesses
    uint64_t host_accesses;  // host CSR and DRAM accesses
    uint64_t joins;          // vector-clock joins
    uint64_t dram_segments;  // DRAM shadow segments now (all channels)
    uint64_t dram_segments_max;
    uint64_t old_refs;       // joins on a ticket older than the per-unit history (conservative)
    uint64_t reports;        // reports found, folded duplicates included
} tsr_race_stats;
int    tsr_sim_race_stats(const tsr_sim * sim, tsr_race_stats * out);

#ifdef __cplusplus
}
#endif

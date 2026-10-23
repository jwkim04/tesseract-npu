// tsr_trace.h — trace directories (spec 트레이스 형식).
//
//   <dir>/meta.json                   config params, cmd_addr, cmd_count, args, region lists
//   <dir>/cmd.bin                     the instruction array (cmd_count * 32 bytes)
//   <dir>/init/ch<c>_<offset>.bin     DRAM content before execution (offset = hex byte offset)
//   <dir>/expect/ch<c>_<offset>.bin   DRAM content after execution of the ST-written ranges
//
// meta.json layout (all integers decimal, offsets in bytes):
// {
//   "format": "tesseract-trace", "version": 1,
//   "config": { "num_ch":4, "beat_bits":512, "lanes":128, "t_max":1, "k_max":32768,
//               "spm_bytes":4194304, "vl":16, "nireg":16, "nsreg":16, "loop_depth":4,
//               "rd_outstanding":16, "ch_bytes":17179869184, "addr_beat_w":28, "caps":0 },
//   "cmd_addr_beat": 0, "cmd_count": 5, "args": [0,0,0,0,0,0,0,0],
//   "status": 2, "err_code": 0, "err_pc": 0, "err_info": 0,
//   "init":   [ {"ch":0, "offset":0, "size":160, "file":"init/ch0_0000000000.bin"}, ... ],
//   "expect": [ ... same ... ],
//   "perf": { "cycles": N, "retired": N, "mvm_cycles": N, "ch_rd_beats": [..] }
// }
// File names use a 10-digit zero-padded lower-case hex byte offset.
//
// Semantics for a replayer (RTL testbench, hardware runner, tsr_trace_replay):
//   - DRAM outside the init regions is zero. Load cmd.bin at ch0 byte cmd_addr_beat*64, then
//     every init region (init wins where they overlap; tsr_trace_write keeps them consistent:
//     cmd.bin holds the init bytes where the command buffer was fetched and zero elsewhere).
//   - Run with CMD_ADDR = cmd_addr_beat, CMD_COUNT = cmd_count, ARG0..7 = args.
//   - Compare final STATUS, ERR_CODE, ERR_PC, ERR_INFO and every expect region bit-exactly.
//     The set of DRAM beats the run writes must equal the union of the expect regions.
//   - "err_info" and "perf" are optional on read. "perf" is the simulator's cycle model
//     (informational: tsr_trace_replay prints a warning, not a failure, if it differs).
#pragma once

#include <stdint.h>

#include "tsr_isa.h"
#include "tsr_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TSR_TRACE_MAX_CH 256u   // num_ch is an 8-bit CFG0 field

typedef struct tsr_trace_perf {
    int      valid;                              // 0: not recorded / not present in meta.json
    uint64_t cycles, retired, mvm_cycles;
    uint64_t ch_rd_beats[TSR_TRACE_MAX_CH];      // num_ch entries used
} tsr_trace_perf;

typedef struct tsr_trace {
    tsr_config  cfg;
    uint64_t    cmd_addr_beat;
    uint32_t    cmd_count;
    uint32_t    args[8];
    uint32_t    status, err_code, err_pc;   // expected final state
    tsr_capture cap;                        // init + expect regions
    // --- additions (v0.2.1) ---
    uint32_t    err_info;                   // expected ERR_INFO
    int         has_err_info;               // 0: not compared (meta.json without "err_info")
    tsr_trace_perf perf;                    // run's perf delta (informational)
    uint8_t *   cmd;                        // cmd_count*32 bytes (cmd.bin) or NULL. Owned by the
                                            // trace (freed by tsr_trace_free) when set by
                                            // tsr_trace_read. NULL on write: derived from init.
} tsr_trace;

// Write a trace directory (created if missing; an existing trace in it is replaced: its
// meta.json, cmd.bin and init/expect ch*_*.bin files are removed first). meta.json is written
// last, via rename, so a partially written directory has no meta.json. Returns 0 on success.
int  tsr_trace_write(const char * dir, const tsr_trace * tr);
// Read a trace directory. Returns 0 on success; free with tsr_trace_free.
// Checks: known format/version, every config field present, regions 64-byte aligned, inside
// the channel, file sizes equal to "size", cmd.bin size = cmd_count*32 and consistent with
// the init regions it overlaps.
int  tsr_trace_read(const char * dir, tsr_trace * tr);
void tsr_trace_free(tsr_trace * tr);

// Replay on a fresh simulator built from tr->cfg: load tr->cmd (if set) at the command
// address, then the init regions, run, compare final
// STATUS/ERR_CODE/ERR_PC (and ERR_INFO if present) and every expect region bit-exactly.
// Also fails if the run writes DRAM beats outside the expect regions or reads DRAM that is
// neither in an init region nor written earlier by the run. Returns 0 on match; on
// mismatch prints the first differing (ch, byte offset) to stderr and returns 1.
// Returns 2 if the simulator cannot be created for tr->cfg.
int  tsr_trace_replay(const tsr_trace * tr, int verbose);
// Same, with n simulator worker threads (tsr_sim_set_threads).
int  tsr_trace_replay_threads(const tsr_trace * tr, int verbose, unsigned threads);

#ifdef __cplusplus
}
#endif

// tsr_trace.h — trace format (docs/design/06-software-verification.md "SW" §5, SW-5): read,
// write and validate trace directories, capture one `launch` trace per launch on the simulator
// (SW §5.5, TSR_TRACE), and replay a trace on a fresh simulator with every comparison of SW §5.4.
// Implementation: sw/sim/tsr_trace.c (part of tsr_sim when sw/sim/trace.cmake is included).
//
// Directory (SW §5.1):
//   <trace>/meta.json                  version 2, every field below
//   <trace>/ops/push.bin               tsr_trace_push records (48 bytes, little-endian), push order
//   <trace>/init/ch<c>_<off>.bin       DRAM before the launch (<off> = 10 hex digits, bytes)
//   <trace>/expect/ch<c>_<off>.bin     DRAM written by the launch
//   <trace>/expect/tickets.bin         16-byte header {n_units, 0, 0, 0} + n_units tsr_trace_unit
//   <set>/images/<image id hex>.elf    firmware images (kind launch), shared by every trace of a set;
//                                      meta "image.file" is the path relative to <trace>
//
// meta.json (all integers decimal; "ret" may be negative; unknown keys are refused on read):
// { "format": "tesseract-trace", "version": 2, "kind": "launch" | "ops",
//   "config": { "num_ch", "beat_bits", "lanes", "t_max", "k_max", "spm_bytes", "vl", "nsreg",
//               "rd_outstanding", "ch_bytes", "addr_beat_w", "caps", "ctl_id", "cfg3",
//               "itcm_bytes", "dtcm_bytes", "q_depth": [6 depths] },
//   "image":  { "id": "<16 hex>", "file": "../images/<id>.elf" },             kind launch only
//   "launch": { "kernel": "<name>", "index", "prof", "args": [7] },           kind launch only
//   "ops":    { "file": "ops/push.bin", "count", "load": {"ch", "offset"} }, "load" kind ops only
//   "done":   { "status", "ret", "err_code", "err_ticket_rel", "err_pidx_rel", "err_info", "err_w0" },
//   "init":   [ {"ch", "offset", "size", "file"} ... ],  "expect": [ ... ],
//   "sched": "<schedule of the run>", "rv_ratio": <F_RV / F_CORE>,
//   "perf":   { "core_cycles", "rv_instret", "unit_busy": [6], "ch_rd_beats": [num_ch] } }   optional
//
// Relative tickets (SW §5.2): a ticket {u, seq} is stored as {u, (seq - base_u) mod 2^28}, with
// base_u = ISSUED_SEQ[u] at kernel entry (kind launch) or at the start of the replay (kind ops);
// unit fields >= 6 (reserved, marker, none) are stored as pushed. Push indices (ERR_PIDX, DONE
// r6) are stored relative to ISSUE_PIDX at the same instant, mod 2^32. "Kernel entry" in a
// capture is the first RISC-V read of ISSUE_PIDX after the launch's LQ_POP (SW §6.3 R5 reads
// ISSUE_PIDX, then ISSUED_SEQ[u], right before the kernel call); a firmware that never reads it
// gets the counters at its LQ_POP.
//
// What a replay compares (SW §5.4, §4.7): the DONE fields; every expect region bit-exactly; the set
// of written DRAM beats = the union of the expect regions; every DRAM read inside an init region or
// written earlier in the launch; the per-unit completion counts and error records (tickets.bin); the
// per-unit projections of push.bin (records grouped by the unit the issue path sends them to,
// tsr_trace2_push_unit; QCTRL records without QCLEAR and trace marks form one more stream). Never
// compared: MTIME, PERF ("perf" is informational), record word r7, completion records, log content.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"
#include "tsr_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TSR_TRACE2_FORMAT  "tesseract-trace"
#define TSR_TRACE2_VERSION 2u

// meta "kind"
#define TSR_TRACE2_KIND_LAUNCH 0u   // image + descriptor (capture, SW §5.5)
#define TSR_TRACE2_KIND_OPS    1u   // tile-only push stream (tsr-trace-gen, tsr-trace-conv)

// ---------------------------------------------------------------------------
// ops/push.bin (SW §5.2) and expect/tickets.bin (SW §5.3)
// ---------------------------------------------------------------------------
#define TSR_TRACE_PUSH_BYTES 48u
#define TSR_TRACE_PUSH_OP    0u     // w = instruction words as pushed (masked words 0)
#define TSR_TRACE_PUSH_QCTRL 1u     // w[0] = QCTRL value (CC §5.1), every other field 0
#define TSR_TRACE_PUSH_MARK  2u     // w[0] = TRACE_MARK value, every other field 0

typedef struct tsr_trace_push {     // 48 bytes, the file layout
    uint32_t w[8];
    uint32_t dep[2];                // relative tickets {unit[31:28], rel_seq[27:0]}; unit 15 = none
    uint32_t push;                  // the ISSUE_PUSH value (WMASK [15:8], REC bit0); 0 for kinds 1, 2
    uint32_t kind;                  // TSR_TRACE_PUSH_*
} tsr_trace_push;

#define TSR_TRACE_TICKETS_HDR_BYTES 16u
#define TSR_TRACE_UNIT_BYTES        32u
#define TSR_TRACE_UNITS_MAX         TSR_UNIT_SLOTS   // n_units on read: 1..8 (units >= 6 all zero)

typedef struct tsr_trace_unit {     // 32 bytes, the file layout
    uint32_t count;                 // entries of this unit completed during the launch / run
    uint32_t err_code;              // 0 = no error record (then every other field is 0)
    uint32_t err_rel_seq;           // ERR_SEQ - base_u (mod 2^28)
    uint32_t err_info, err_w0;
    uint32_t err_rel_pidx;          // ERR_PIDX - ISSUE_PIDX at kernel entry (mod 2^32)
    uint32_t reserved[2];           // 0
} tsr_trace_unit;

// DONE record r0-r6 (EX §4.4) with the ticket and push index relative (when r3 names no unit, as
// for OK / BAD_DESC answers, r3 and r6 are stored as posted). Kind ops: what the replay
// kernel would answer (SW §7.7): status OK / TILE_ERR / ABORTED by EX §4.4's precedence from the
// first error (smallest relative ERR_PIDX over the units in error), ret = 0 or 1 + its relative
// push index.
typedef struct tsr_trace_done {
    uint32_t status;                // r0[15:8]
    int32_t  ret;                   // r2
    uint32_t err_code;              // r0[23:16]
    uint32_t err_ticket_rel;        // r3, relative; TSR_TICKET_NONE = none
    uint32_t err_pidx_rel;          // r6 - pidx base
    uint32_t err_info;              // r4
    uint32_t err_w0;                // r5
} tsr_trace_done;

// meta "config": the elaboration parameters plus what tsr_sim_opts adds.
typedef struct tsr_trace2_config {
    tsr_config core;                // num_ch .. caps (tsr_isa.h)
    uint32_t   ctl_id, cfg3;        // CTL_ID / CFG3 of the capturing simulator (a replay warns on a difference)
    uint32_t   itcm_bytes, dtcm_bytes;
    uint8_t    q_depth[TSR_NUM_UNITS];
} tsr_trace2_config;

typedef struct tsr_trace2_perf {    // informational (SW §5.1)
    int      valid;
    uint64_t core_cycles;           // MTIME from LQ_POP to the answer (kind launch) / of the run (ops)
    uint64_t rv_instret;            // RISC-V instructions retired in that interval
    uint64_t unit_busy[TSR_NUM_UNITS];   // PERF DMA, QACT, MVM, VU, ATT busy cycles; SYNC 0
    uint64_t ch_rd_beats[TSR_NUM_CH_MAX];
} tsr_trace2_perf;

#define TSR_TRACE2_LOAD_AUTO 0xFFFFFFFFu   // load_ch: tsr_trace2_write picks "ops.load" (kind ops)

typedef struct tsr_trace2 {
    uint32_t          kind;         // TSR_TRACE2_KIND_*
    tsr_trace2_config cfg;
    // kind launch
    uint64_t          image_id;     // EX §3.2 (tsr_trace2_image_id of `image`)
    uint8_t *         image;        // the ELF bytes (owned)
    size_t            image_size;
    char              kernel[32];   // name from the image's .tsr_kernels, "" if it has none
    uint32_t          kernel_index; // descriptor KERNEL
    uint32_t          prof;         // descriptor PROF (0 / 1)
    uint32_t          args[7];      // ARG0..ARG6
    // the push stream (both kinds)
    tsr_trace_push *  push;
    size_t            n_push, cap_push;
    // kind ops: where a replay kernel would find push.bin (outside every region)
    uint32_t          load_ch;      // TSR_TRACE2_LOAD_AUTO: chosen by tsr_trace2_write
    uint64_t          load_offset;  // bytes, multiple of 64
    tsr_trace_done    done;
    tsr_capture       mem;          // init / expect regions (tsr_sim.h; owned, malloc'd data)
    uint32_t          n_units;      // tickets.bin units (TSR_NUM_UNITS on write)
    tsr_trace_unit    unit[TSR_TRACE_UNITS_MAX];
    char              sched[64];    // schedule of the run that produced it ("fixed" if empty)
    double            rv_ratio;     // F_RV / F_CORE of that run (0: the simulator default)
    tsr_trace2_perf   perf;
} tsr_trace2;

// ---------------------------------------------------------------------------
// Lifetime and building
// ---------------------------------------------------------------------------
tsr_trace2 * tsr_trace2_new(void);                 // empty kind-ops trace (tsr_trace2_init)
void         tsr_trace2_free(tsr_trace2 * t);
// For caller-owned structs: init sets an empty kind-ops trace (U200 depths, default TCMs, no regions,
// done = OK / none, n_units = 6, load_ch = AUTO, sched "fixed"); clear releases everything and
// re-inits; copy deep-copies (dst is cleared first). copy returns 0.
void         tsr_trace2_init(tsr_trace2 * t);
void         tsr_trace2_clear(tsr_trace2 * t);
int          tsr_trace2_copy(tsr_trace2 * dst, const tsr_trace2 * src);

// Sets cfg from a simulator's configuration (its tsr_config, TCM sizes, Q_DEPTH, CTL_ID, CFG3).
void tsr_trace2_config_from_sim(tsr_trace2_config * c, tsr_sim * sim);
// Sets cfg from a tsr_config with U200 depths and default TCMs; ctl_id / cfg3 0 (tsr_trace2_run_ops
// fills them from its simulator).
void tsr_trace2_config_from_core(tsr_trace2_config * c, const tsr_config * core);
// The CTL_ID a simulator with every model reports (CTL_ID.DM needs the DM and DBG models;
// tsr_sim_int.h §3). tsr-trace-gen and tsr-trace-conv store it in the sets they write instead of
// the CTL_ID of the library they run on, so that a set's bytes do not depend on which model
// snippets the build had (a replay only warns on a CTL_ID difference). Captures keep the
// capturing simulator's own value.
uint32_t tsr_trace2_ctl_id_full(void);

// Appends records (deps are relative tickets). Return 0.
int tsr_trace2_add_push(tsr_trace2 * t, const tsr_trace_push * p);
int tsr_trace2_add_op(tsr_trace2 * t, const uint32_t w[8], uint32_t dep0_rel, uint32_t dep1_rel, uint32_t push_val);
int tsr_trace2_add_qctrl(tsr_trace2 * t, uint32_t bits);
int tsr_trace2_add_mark(tsr_trace2 * t, uint32_t value);
// Appends an init / expect region (data copied; ch, offset and size checked against cfg when the
// trace is validated). Return 0.
int tsr_trace2_add_init(tsr_trace2 * t, uint32_t ch, uint64_t byte_off, const void * data, uint64_t size);
int tsr_trace2_add_expect(tsr_trace2 * t, uint32_t ch, uint64_t byte_off, const void * data, uint64_t size);

// The unit the issue path sends push record p to under configuration c (CC §7.1, §7.4): kind 0 ->
// tsr_op_unit(opcode) if the opcode is legal, its CAPS bits are in c, its unit is built and this
// simulator library has its handler, else SYNC; kind 1 with QCLEAR -> SYNC; otherwise
// TSR_UNIT_NONE (no ticket).
uint32_t tsr_trace2_push_unit(const tsr_trace2_config * c, const tsr_trace_push * p);
// The relative ticket of record i (what ISSUE_TICKET reads after it, relative to the trace's base),
// counting the tickets of records 0..i-1 per unit; TSR_TICKET_NONE for a record without a ticket.
// For building ops traces: a dependency on record i is tsr_trace2_ticket_of(t, i).
uint32_t tsr_trace2_ticket_of(const tsr_trace2 * t, size_t i);

// Relative <-> absolute tickets for base[TSR_NUM_UNITS] (unit fields >= 6 pass unchanged).
static inline uint32_t tsr_trace_rel_ticket(uint32_t abs, const uint32_t * base) {
    const uint32_t u = abs >> 28;
    return u < TSR_NUM_UNITS ? (u << 28) | ((abs - base[u]) & 0x0FFFFFFFu) : abs;
}
static inline uint32_t tsr_trace_abs_ticket(uint32_t rel, const uint32_t * base) {
    const uint32_t u = rel >> 28;
    return u < TSR_NUM_UNITS ? (u << 28) | ((rel + base[u]) & 0x0FFFFFFFu) : rel;
}

// EX §3.2 image id of an ELF32 RISC-V image: SHA-256 over its PT_LOAD segments in ascending
// physical address, each as (u32 paddr LE, u32 filesz padded to a word LE, padded bytes), with the
// 8 bytes of a ".tsr_id" section (if the ELF has section headers naming one) taken as zero; the
// first 8 digest bytes as a little-endian uint64. Returns 0 (*id set) or -1 (err filled: not a
// valid image).
int tsr_trace2_image_id(const uint8_t * elf, size_t n, uint64_t * id, char * err, size_t errlen);

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------
// Validates t (every rule tsr_trace2_read checks except the files). 0, or -1 with err filled.
int tsr_trace2_validate(const tsr_trace2 * t, char * err, size_t errlen);
// Writes t into dir (created; files of an earlier trace there are removed first; meta.json last,
// through a rename). Kind launch: the image goes to images_dir/<id>.elf (NULL: <dir>/../images),
// written once (an existing file with the same id must have the same bytes). Kind ops with
// load_ch = AUTO: "ops.load" = ch 0, the lowest 4 KiB-aligned offset at or above the OPL region
// where push.bin overlaps no region. Returns 0, or -1 with a message on stderr.
int tsr_trace2_write(const tsr_trace2 * t, const char * dir, const char * images_dir);
// Reads and strictly validates a trace directory: format / version / kind, every field present
// with its type and range, no unknown keys, config supported by the simulator's limits, regions
// 64-byte aligned, inside a channel, non-overlapping per list, files of exactly "size" bytes;
// push.bin = count x 48 bytes with valid kinds and zero padding fields; tickets.bin header and
// size; kind launch: the image file present, a valid image (EX §3.1: PT_LOADs inside the TCMs)
// whose id equals "image.id"; kind ops: "load" outside every region. On success *out is a new
// trace (tsr_trace2_free). Returns 0, or -1 with err filled (path and reason).
int tsr_trace2_read(const char * dir, tsr_trace2 ** out, char * err, size_t errlen);

// ---------------------------------------------------------------------------
// Replay on the simulator (SW §5.4)
// ---------------------------------------------------------------------------
// Comparison classes (tsr_trace2_result.fail_mask).
#define TSR_TRACE2_CMP_SETUP   0x001u   // simulator, image load, boot / HELLO, run did not finish
#define TSR_TRACE2_CMP_DONE    0x002u   // DONE fields
#define TSR_TRACE2_CMP_EXPECT  0x004u   // expect region content
#define TSR_TRACE2_CMP_WRITTEN 0x008u   // written beats != union of the expect regions
#define TSR_TRACE2_CMP_READS   0x010u   // a read outside init and not written earlier
#define TSR_TRACE2_CMP_COUNTS  0x020u   // per-unit completion counts
#define TSR_TRACE2_CMP_ERRREC  0x040u   // per-unit error records
#define TSR_TRACE2_CMP_PUSH    0x080u   // per-unit projections of push.bin

typedef struct tsr_trace2_result {
    int      ok;                    // 1: every comparison passed
    char     msg[512];              // the first failure, or a one-line summary when ok
    uint32_t fail_mask;             // TSR_TRACE2_CMP_* that failed
    uint32_t n_fail;                // failed comparisons (each class may count several)
    uint64_t core_cycles;           // simulated core cycles of the replayed launch / run
    uint64_t rv_instret;
    int      perf_differs;          // informational: "perf" present and different
    // the replay simulator's race detector at the end (TSR_SIM_RACE / opts.race report): reports,
    // and of those SPM_STALE (a read of SPM never written since the launch's LQ_POP, SW-4)
    uint32_t race_reports, race_stale;
} tsr_trace2_result;

typedef struct tsr_trace2_replay_opts {
    const tsr_config * cfg;         // NULL: the trace's config
    const char *       sched;       // NULL: TSR_SIM_SCHED or "fixed" (tsr_sim_create_opts' default)
    double             rv_ratio;    // 0: the trace's rv_ratio (0 there: the simulator default)
    uint32_t           threads;     // MVM worker threads (0: 1)
    const char *       dram_dir;    // file-backed DRAM (tsr_sim_opts), or NULL
    uint64_t           max_cycles;  // simulated-time budget of the run (0: 2^40 core cycles)
    int                verbose;     // 1: every failed comparison on stderr
    int                init_from_reads;  // tsr_trace2_run_ops: replace init by the regions the run read
    // kind ops through the `replay` kernel of a firmware image (SW §5.4, §7.7; EX §4.6) instead
    // of the tile-only route (the default, replay_image NULL). The image is booted like a kind-launch
    // trace's; push.bin is written at "ops.load" and launched as replay(handle, count, flags |
    // TSR_REPLAY_F_BUF, buffer line, chunk records), with the SPM buffer placed outside every SPM
    // extent of the trace's ops; the captured launch is translated back to the ops view (the kernel's
    // own chunk LDs and FENCEs removed, tickets and push indices renumbered, the push.bin reads dropped)
    // and compared like a tile-only replay. tsr_trace2_run_ops ignores these fields.
    const char *       replay_image;     // ELF path of an image with the kernel; NULL: tile-only route
    const char *       replay_kernel;    // its name in .tsr_kernels; NULL: "replay"
    uint32_t           replay_flags;     // TSR_REPLAY_F_SERIAL (traces/model-v1conv, SW §5.4)
    uint32_t           race;             // race detector of the replay simulator: 0 = TSR_SIM_RACE
                                         // (tsr_sim_opts default), 1 report, 2 fatal
} tsr_trace2_replay_opts;

// the `replay` kernel's arguments (SW §7.7, EX §4.6; sw/ctrl/kernels/replay.cpp holds the same
// constants): a0 push.bin handle {ch, beat}, a1 record count, a2 flags, a3 buffer line (F_BUF), a4
// records per chunk (F_BUF; a multiple of 4, at most TSR_REPLAY_CHUNK_MAX; buffer b is at line a3 +
// b x chunk x 48 / 64, b = 0, 1, and buffer 1 is used only when the stream has more than one chunk).
// Without F_BUF: chunks of TSR_REPLAY_CHUNK_MAX records in the two 1023-line buffers just below the
// runtime argument range. RET (SW §7.7 step 4): 0, or 1 + the push index of the first error relative to
// the first replayed record, the kernel's own chunk LDs not counted; negative: the codes below.
#define TSR_REPLAY_F_SERIAL      0x1u          // every op record depends on the previous op record only
#define TSR_REPLAY_F_BUF         0x2u          // a3 / a4 give the SPM buffer and the chunk size
#define TSR_REPLAY_CHUNK_MAX     1364u         // records per chunk: 64 KiB rounded down to 4 records (1023 lines)
#define TSR_REPLAY_CHUNKS_MAX    130u          // chunks per stream (DTCM bookkeeping of the own LDs)
#define TSR_REPLAY_RET_BAD_ARGS  0x80000101u   // flags, buffer or chunk out of range
#define TSR_REPLAY_RET_TOO_LONG  0x80000102u   // more than TSR_REPLAY_CHUNKS_MAX chunks
#define TSR_REPLAY_RET_STREAM    0x80000103u   // a chunk LD failed (DMA error state or QABORT from the trace)

// Replays t on a fresh simulator built from its config (or cfg_or_null) under sched_or_null and
// compares (SW §5.4). Kind launch: load the image through the host CSR loader (receive CRC and
// VERIFY), RV_BOOT, RV_RUN, wait for HELLO, push the descriptor, wait for its answer. Kind ops: a
// tile-only replay through the issue port (ISSUE_W / DEP / ISSUE_PUSH, QCTRL and TRACE_MARK
// stores, as tsr_sim_push / tsr_sim_qctrl do) with deps mapped back as (u, base_u + rel), each
// store waiting for its credit / FIFO space like the replay kernel (no ISSUE_WAIT_MAX timeout),
// then until QIDLE; with o->replay_image through the `replay` kernel (above). Returns 0 when every
// comparison passed, 1 on a mismatch, 2 on a setup failure (r filled in every case).
int tsr_trace2_replay_sim(const tsr_trace2 * t, const tsr_config * cfg_or_null, const char * sched_or_null,
                          tsr_trace2_result * r);
int tsr_trace2_replay_sim_ex(const tsr_trace2 * t, const tsr_trace2_replay_opts * o, tsr_trace2_result * r);

// For tsr-trace-gen (SW §5.5): runs a kind-ops trace like a replay and fills done, unit[] (counts and
// error records), expect (the beats the run wrote, content at the end), perf, cfg.ctl_id / cfg3 and
// sched / rv_ratio from the run; with o->init_from_reads also replaces init by the regions the run
// read before writing them (content as loaded, zero outside the old init). Without it a read
// outside init is reported (r->fail_mask READS) and nothing is filled. Returns 0 / 1 / 2 as replay.
int tsr_trace2_run_ops(tsr_trace2 * t, const tsr_trace2_replay_opts * o, tsr_trace2_result * r);

// ---------------------------------------------------------------------------
// Launch capture on the simulator (SW §4.8, §5.5; the TRACE model, attached at create)
// ---------------------------------------------------------------------------
// Capture covers one launch from its LQ_POP to its answer record (LCQ_POST with ANSWER): DRAM read
// before written (init, first-read copies), DRAM written (expect, content at the answer), the
// pushes (PUSH / QCTRL queue events, TRACE_MARK stores) with relative tickets (and the launch's
// own LAUNCH_BEGIN / LAUNCH_END markers with their SEQ field 0, SW §5.5), per-unit counts and
// error records at the answer, the DONE record, the image. Only LAUNCH descriptors without
// reserved bits answered by a DONE record become traces (PING, BAD_DESC types, launches answered
// by TRAP or killed by a reset are counted as skipped). The image is the ELF given to
// tsr_sim_capture_set_image, else one synthesized from what the host loader wrote (LOAD_ADDR /
// LOAD_DATA, contents at the last RV_RUN rise; entry RV_BOOT + 0x80). DRAM written by the host
// through the tsr_sim_dram pointer during a launch is invisible (H3: a race) unless announced
// with tsr_sim_dram_note_write; announced host writes during a capture are counted and warned.
// TSR_TRACE=<dir> (SW §4.9, read by tsr_sim_create_opts): every captured launch is written to
// <dir>/<seq, 6 digits>/ with its image in <dir>/images/. A second simulator of the same process
// that writes traces uses <dir>/sim<k>/<seq>/ (k = 1, 2, ... in the order of their first trace).
//   capture_launch  on = 1: queue each captured launch for capture_take (as well as TSR_TRACE)
//   capture_take    moves the oldest queued launch into *out (cleared first; from tsr_trace2_new or
//                   tsr_trace2_init); 0 = one launch taken, 1 = none queued, -1 = no TRACE model
//   set_image       the real ELF of the loaded image (copied; NULL clears). At every boot it is
//                   checked against the words the loader wrote and dropped (warning) if they differ
// The capture uses the simulator's DRAM access capture (tsr_sim_capture_begin / _end, tsr_sim.h)
// from LQ_POP to the answer: a caller must not run its own access capture across a captured
// launch. Captured launches stay queued until taken.
void tsr_sim_capture_launch(tsr_sim * sim, int on);
int  tsr_sim_capture_take(tsr_sim * sim, tsr_trace2 * out);
void tsr_sim_capture_set_image(tsr_sim * sim, const void * elf, size_t n);

typedef struct tsr_trace_capture_stats {
    uint32_t captured;              // launch traces completed
    uint32_t written;               // of which written to the TSR_TRACE directory
    uint32_t skipped;               // launches not captured (PING, bad descriptor, TRAP, reset)
    uint32_t failed;                // write failures
    uint32_t host_dram_writes;      // announced host DRAM writes during captured launches
    uint32_t queued;                // waiting for capture_take
} tsr_trace_capture_stats;
// Zeroes *s when the simulator has no TRACE model.
void tsr_sim_capture_stats(tsr_sim * sim, tsr_trace_capture_stats * s);

// The TRACE model's attach / detach (called by tsr_sim_create_opts / tsr_sim_destroy).
void tsr_trace_attach(tsr_sim * sim);
void tsr_trace_detach(tsr_sim * sim);

#ifdef __cplusplus
}
#endif

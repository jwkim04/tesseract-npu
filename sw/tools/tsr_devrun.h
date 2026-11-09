// tsr_devrun.h — launch targets and trace replays for the tools (docs/design/
// 06-software-verification.md "SW" §7.6, §9.5, SW-28; docs/design/01-execution-model.md "EX" §4).
//
// A target is either a device behind a transport (EX §10.1: any kind, the card included) or an
// in-process simulator booted the same way: the "shadow" of tsr-selftest, which runs every launch
// with the same image, inputs and arguments as the device (SW §7.6). Both answer the same calls,
// so a tool runs one sequence on both and compares what SW §4.7 makes comparable: the answer
// record r0-r6 (r0[31:24] and r1 are the target's own sequence, r3 / r6 compared relative to the
// target's queue counters before the launch), every output byte, the per-unit completion counts
// (Q_DONE[u] deltas, EX §5.4), the kernel events and the log words (LSEQ and the MTIME word
// masked). Never compared: DURATION (r7), MTIME, PERF.
//
//   tsr_tgt_transport  a booted or bootable device (borrows the transport)
//   tsr_tgt_shadow     a simulator with exactly the device's configuration (tsr_tgt_shadow_of)
//   tsr_tgt_boot       boot an image (no-op when the target runs that image already)
//   tsr_tgt_launch     one LAUNCH / PING: Q counters at quiescence before and after, the answer,
//                      the events and the log words of the launch
//   tsr_tgt_compare    the comparison above, one message per difference
//
// Trace replays (SW §5.4; tsr-trace-replay --device, tsr-selftest --traces): tsr_devrun_trace
// loads a trace's init regions (the expect regions are first filled with the complement of their
// expected bytes, so a missing write shows), runs it - kind ops through the image's `replay`
// kernel (SW §7.7, EX §4.6) with the arguments the simulator's kernel route uses (push.bin at
// "ops.load", the SPM buffer outside every SPM extent of the trace), kind launch as a full launch
// of the trace's own image and descriptor - and compares the answer with the trace's DONE and
// every expect region with the device's DRAM; with a shadow the same launch runs there too and is
// compared with tsr_tgt_compare (the per-unit counts included). A trace runs only where it means
// the same as where it was made (tsr_devrun_trace_fits: the same NUM_CH, LANES, K_MAX, SPM, VL,
// NSREG; every op routed to the same unit and given the same check result under both
// configurations; every region and push.bin inside the device's channels).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"
#include "tsr_sim.h"
#include "tsr_trace.h"
#include "tsr_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_tgt tsr_tgt;

// The device side of a configuration: what a shadow needs to equal the device (SW §7.6).
typedef struct tsr_tgt_cfg {
    tsr_config core;                // tsr_transport_config (for sim / mock: the inner simulator's)
    uint8_t    q_depth[TSR_NUM_UNITS];   // Q_DEPTH
    uint32_t   itcm_bytes, dtcm_bytes;   // RV_CFG
    double     rv_ratio;            // RV_FREQ / 2^16 (timing only), 0 = the simulator default
} tsr_tgt_cfg;

// Reads the configuration of a transport's device (CSRs; the inner simulator's tsr_config when the
// kind has one, tsr::transport_sim).
void tsr_tgt_cfg_of_transport(tsr_transport * t, tsr_tgt_cfg * c);

tsr_tgt * tsr_tgt_transport(tsr_transport * t);
// threads: MVM worker threads of the shadow (0: TSR_SIM_THREADS, else 1). NULL on failure.
tsr_tgt * tsr_tgt_shadow(const tsr_tgt_cfg * c, uint32_t threads);
void      tsr_tgt_free(tsr_tgt * g);

const char *       tsr_tgt_name(const tsr_tgt * g);             // "device (<kind>)" / "shadow"
const tsr_config * tsr_tgt_config(const tsr_tgt * g);
tsr_sim *          tsr_tgt_sim(tsr_tgt * g);                    // the shadow's simulator, else NULL
tsr_transport *    tsr_tgt_transport_of(tsr_tgt * g);           // the device's transport, else NULL

int      tsr_tgt_write(tsr_tgt * g, uint32_t ch, uint64_t off, const void * src, size_t n);
int      tsr_tgt_read(tsr_tgt * g, uint32_t ch, uint64_t off, void * dst, size_t n);
uint32_t tsr_tgt_csr(tsr_tgt * g, uint32_t off);
void     tsr_tgt_csr_write(tsr_tgt * g, uint32_t off, uint32_t val);

// Boots the ELF at path (a transport target: tsr_image_open + tsr_transport_boot; the shadow: the
// host loader with the receive CRC and VERIFY, RV_BOOT, RV_RUN, HELLO with the image id checked).
// Nothing happens when the target already runs an image with the same id. 0, or -1 (err filled).
int      tsr_tgt_boot(tsr_tgt * g, const char * elf_path, char * err, size_t errlen);
uint64_t tsr_tgt_image_id(const tsr_tgt * g);                   // 0 before a boot
// After a shadow launch that did not finish (a hang, a trap): a fresh simulator booted with the last
// image (its DRAM is lost: write the inputs after this). No-op for a device (the transport recovers
// by itself, DB §7.3) and for a healthy shadow. 0, or -1 (err filled).
int      tsr_tgt_ready(tsr_tgt * g, char * err, size_t errlen);
int      tsr_tgt_kernel(const tsr_tgt * g, const char * name);  // index in the booted image, -1
// Waits until the device is quiescent: RT_STATE IDLE, STATUS.QIDLE and RV_SLEEP, TAKEN =
// ANSWERED (the runtime's R9 / R10 after an answer included). 0, or -1 when it does not get there.
int      tsr_tgt_quiesce(tsr_tgt * g);

#define TSR_TGT_MAX_EVENTS 64u

typedef struct tsr_tgt_run {
    int      rc;                    // 0 answered; < 0 transport error (TSR_TE_*) or shadow failure
    uint32_t seq;                   // the launch's sequence number on its target
    uint32_t rec[8];                // the answer (DONE / PONG / TRAP)
    uint32_t n_ev;                  // kernel events posted during the launch (types 0xC0-0xFF)
    uint32_t ev[TSR_TGT_MAX_EVENTS][8];
    uint32_t * log;                 // log words drained after the answer (malloc'd; tsr_tgt_run_free)
    size_t   n_log;
    uint32_t q_done0[TSR_NUM_UNITS], q_done1[TSR_NUM_UNITS];   // Q_DONE at quiescence before / after
    uint32_t q_issued0[TSR_NUM_UNITS];                          // Q_ISSUED before (r3 relative)
    uint32_t q_pidx0;               // Q_PIDX before (r6 relative)
    uint64_t wall_ns;               // host time of the launch and wait
} tsr_tgt_run;
void tsr_tgt_run_free(tsr_tgt_run * r);

// One descriptor (type TSR_DESC_TYPE_LAUNCH / _PING; flags = w0[15:8]). The answer, events and log
// words of exactly this launch (the log is drained before and after). 0 or the error.
int tsr_tgt_launch(tsr_tgt * g, uint32_t type, uint32_t flags, uint32_t kernel, const uint32_t args[7], tsr_tgt_run * r);
// Drains and returns the log words (malloc'd *words, *n); 0 or -1.
int tsr_tgt_log_drain(tsr_tgt * g, uint32_t ** words, size_t * n);

// Compares two runs of the same launch (SW §7.6): answer r0[23:0], r2, r3 / r6 relative, r4, r5;
// the events (r0[23:0], r2-r7; r1 = the launch's own sequence on each side); the per-unit
// completion counts; the log words (header without LSEQ, payload without the MTIME word). Appends
// one line per difference to msg (NUL-terminated, truncated at msglen). Returns the number of
// differences.
uint32_t tsr_tgt_compare(const tsr_tgt_run * a, const tsr_tgt_run * b, char * msg, size_t msglen);

// ---------------------------------------------------------------------------
// Trace replays through targets
// ---------------------------------------------------------------------------
typedef struct tsr_devrun_opts {
    const char * image;             // ELF with the `replay` kernel (kind ops); the target's booted
                                    // image is used when NULL
    uint32_t     replay_flags;      // TSR_REPLAY_F_SERIAL (traces/model-v1conv, SW §5.4)
    int          verbose;           // every difference on stderr
} tsr_devrun_opts;

// Comparison classes (tsr_devrun_result.fail_mask); SETUP as tsr_trace.h.
#define TSR_DEVRUN_SETUP   0x001u   // writing, booting, launching failed
#define TSR_DEVRUN_DONE    0x002u   // the answer differs from the trace's DONE
#define TSR_DEVRUN_EXPECT  0x004u   // an expect region differs
#define TSR_DEVRUN_SHADOW  0x008u   // device and shadow differ (answer, counts, events, log)

typedef struct tsr_devrun_result {
    int      ok;                    // 1: every comparison passed
    int      skipped;               // 1: the trace does not fit the device (msg says why)
    char     msg[768];
    uint32_t fail_mask;
    uint32_t n_fail;
    uint64_t wall_ns;               // device launch and wait, host time
    uint32_t rec[8];                // the device's answer
} tsr_devrun_result;

// "" (fits) or why the trace cannot be replayed on a device of configuration dev (see the top).
// check_sim: any simulator of the device's configuration (the checker's limits), or NULL to skip
// the per-op check comparison (then a trace whose config differs in T_MAX, CAPS, Q_DEPTH or the DRAM
// limit does not fit).
const char * tsr_devrun_trace_fits(const tsr_trace2 * t, const tsr_config * dev, const uint8_t * dev_q_depth,
                                   tsr_sim * check_sim, char * why, size_t whylen);

// Replays t on dev (and shadow, if not NULL). Returns 0 passed, 1 mismatch, 2 setup failure, 3
// skipped (does not fit). r filled in every case.
int tsr_devrun_trace(tsr_tgt * dev, tsr_tgt * shadow, const tsr_trace2 * t, const tsr_devrun_opts * o,
                     tsr_devrun_result * r);

#ifdef __cplusplus
}
#endif

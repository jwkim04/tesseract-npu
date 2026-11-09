// tsr_transport.h — host <-> device transport (docs/design/01-execution-model.md "EX" §10.1, the
// API verbatim; docs/design/06-software-verification.md "SW" §9.1-§9.2, the conveniences and the
// environment).
//
// The only target-dependent part of the software stack. Every kind runs the same register
// protocol (SW-18, EX §10.2): the protocol layer in tsr_kdev.cpp over a device-access layer
// (kdev, tsr_kdev.h):
//   "sim"     the simulator in-process behind the core CSR port, no adapter (tsr_kdev_sim.cpp)
//   "mock"    the sim kdev plus the ap_ctrl_none adapter register model (tsr_kdev_mock.cpp)
//   "vkernel" the Verilated kernel in a server process (tsr_kdev_vkernel.cpp; the two-clock
//             server is P5.3)
//   "xrt"     the Alveo card through xrt::ip and one device-only buffer per channel
//             (tsr_transport_xrt.cpp, built with -DTSR_WITH_XRT=ON)
//   kinds registered at run time (tsr::register_kdev, tsr_kdev.h)
// Selected with TSR_DEVICE (default sim). The host never runs programs: it binds channel memory,
// loads a firmware image into the RISC-V's TCMs, boots it and launches kernels (EX §2.1). DRAM
// transfers are in EX §2.1.
//
// Environment (SW §9.2, read at open): TSR_DEVICE; TSR_IMAGE (tsr_image_open_default);
// TSR_LAUNCH_TIMEOUT_MS (60000; 0 = none), TSR_ABORT_MS (10000), TSR_BOOT_MS (2000),
// TSR_RESET_MS (15000); TSR_LOG (stderr | <file>: decode the device log there) and TSR_LOG_LEVEL
// (LOG_CFG.LEVEL, 0 FATAL .. 5 TRACE, default 3); TSR_ATTACH=1 (open without SOFT_RESET or bind,
// for diagnosis); TSR_STATS=1 (transport statistics and DB §8.6's PERF summary on stderr at
// close); DB §9.5: TSR_DUMP_DIR (hang dumps, recovery ladder level 0), TSR_GDB_WAIT (leave a
// trapped device for the debugger; with the embedded service also halt at boot or at the first
// launch), TSR_GDB / TSR_DEBUGGER (the embedded debug service, when one is registered:
// tsr_transport_set_debug_service), TSR_TILE_SERIAL / TSR_TILE_HIST (DBG_CFG at open). Kind-specific variables: tsr_kdev_*.cpp
// (TSR_SIM_*, TSR_MOCK_*, TSR_VKERNEL_*, TSR_XCLBIN / TSR_XRT_*). A malformed value makes the open
// fail with a message (never silently ignored).
//
// Time: on sim, mock and vkernel (whose server reports its time) every timeout and the poll policy
// count device time (core cycles at the nominal 276.3 MHz kernel clock), so results and recoveries
// do not depend on host load; on xrt they count host time (CLOCK_MONOTONIC).
//
// Threads: one mutex per transport serialises every call (EX §2.4). tsr_transport_wait releases it
// between polls, so another thread of the owning process (DB's in-process debug service) can use
// the device while a launch waits. The event callback runs on the waiting thread with the mutex
// held; it may call the transport (the mutex is recursive).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_transport tsr_transport;

// ---------------------------------------------------------------------------------------------
// EX §10.1 (verbatim). Return codes: 0 on success, < 0 one of TSR_TE_* below (message on stderr).
// ---------------------------------------------------------------------------------------------
// unchanged: open, close, name, config, write, read, csr_read
// Open by kind ("sim", "mock", "vkernel", "xrt", registered kinds); NULL or "" means
// getenv("TSR_DEVICE") or "sim". Checks the identity (ADAPTER_VERSION major 1 and NUM_CH on
// adapter kinds, ID, VERSION: Tesseract Core capability major 1 (EX §7.5), CTL_ID != 0 (a control
// core), CFG*, CAPS, CFG3, RV_CFG, Q_DEPTH, RV_FREQ != 0), clears ADAPTER_STATUS, then SOFT_RESET +
// bind (channel bases, MEM_EN), unless TSR_ATTACH=1 (SW §9.1). NULL on failure (message on stderr).
tsr_transport *    tsr_transport_open(const char * kind);   // SOFT_RESET + bind, unless TSR_ATTACH=1
void               tsr_transport_close(tsr_transport * t);
const char *       tsr_transport_name(const tsr_transport * t);
// Configuration as read from the CFG*/CAPS CSRs (+ ch_bytes, which the CSRs do not carry).
const tsr_config * tsr_transport_config(const tsr_transport * t);
// What the device layer says about itself, one item per line (the xrt kind: the device name and
// BDF, the xclbin, the kernel; the simulated kinds: their configuration). Returns the length, as
// snprintf; the text is truncated at n. No device access.
int                tsr_transport_describe(const tsr_transport * t, char * buf, size_t n);
// Host access to channel DRAM (EX §2.1 "Data"). Offsets / sizes in bytes, any
// alignment. 0 on success.
int      tsr_transport_write(tsr_transport * t, uint32_t ch, uint64_t off, const void * src, size_t size);
int      tsr_transport_read (tsr_transport * t, uint32_t ch, uint64_t off, void * dst, size_t size);
// Raw core CSR access (EX §5 offsets < 0x1000; PERF, debug, tools). A failed read returns
// 0xFFFFFFFF; csr_write returns 0 or a TSR_TE_* code.
uint32_t tsr_transport_csr_read(tsr_transport * t, uint32_t off);
int      tsr_transport_csr_write(tsr_transport * t, uint32_t off, uint32_t val);   // new (DB, tools)

// Firmware image (EX §3): a parsed static RV32 ELF. tsr_image_open checks the segments, the
// kernel table, the kind table and .note.tsr (EX §3.1, SW §6.2) and that the stored image id equals
// the computed one (EX §3.2); it refuses an image of another Basis ABI or without a Tesseract Basis
// version (SW §6.9). Images are reference counted: a transport keeps the image it booted alive
// after tsr_image_close.
typedef struct tsr_image tsr_image;                                   // parsed firmware ELF
int         tsr_image_open(const char * elf_path, tsr_image ** out);  // checks segments and .note.tsr
void        tsr_image_close(tsr_image * img);
int         tsr_image_kernel(const tsr_image * img, const char * name); // index, -1 if absent
uint64_t    tsr_image_id(const tsr_image * img);

// Boot (EX §3.3-§3.5, SW §9.1): refuses an image whose .note.tsr requirements the device does not
// meet (the Tesseract Core capability: same major, device minor >= the image's; CAPS; TCMs; the Basis
// ABI); VERIFY of every ITCM segment, a cold load (receive CRC, VERIFY CRC) of each one that does
// not match, every DTCM segment reloaded and verified (a warm restart when no ITCM segment had to
// be written); RV_BOOT = 0, RV_RUN = 1; HELLO within TSR_BOOT_MS with the image id, the Basis ABI,
// the kind table signature, the kernel count and the Tesseract Basis version checked, and no other
// record first. A booted device is soft-reset first.
int tsr_transport_boot(tsr_transport * t, const tsr_image * img);    // warm if possible, else cold; HELLO checked

// Launches (EX §4.3, §4.4, §4.8). tsr_launch: type TSR_DESC_TYPE_LAUNCH / _PING, flags = w0[15:8]
// (bit0 PROF; the rest must be 0), kernel = w0[31:16] (an index of tsr_image_kernel), arg = ARG0-6.
typedef struct tsr_launch { uint8_t type; uint8_t flags; uint16_t kernel; uint32_t arg[7]; } tsr_launch;
typedef struct tsr_record { uint32_t w[8]; } tsr_record;               // §4.4; accessors in tsr_isa.h
// Pushes a descriptor and returns its sequence number (EX §4.2). Blocks (consuming answers) only
// while LQ_DEPTH launches are outstanding (host-side credit, no device read).
int tsr_transport_launch(tsr_transport * t, const tsr_launch * l, uint32_t * seq);  // blocks only at LQ_DEPTH outstanding
// Waits for the answer of launch seq (once per launch). An OK DONE / PONG is read with EX §4.8's
// minimal reads: r0, and r2 unless the kernel's RET is 0 on success (graph, replay, SW §6.4); r1
// (the SEQ), r3 (TICKET_NONE) and r4-r6 (0) follow from EX §4.4; r7 (DURATION) reads as 0. Every
// other record is read whole. timeout_ns: TSR_TIMEOUT_DEFAULT (0) =
// TSR_LAUNCH_TIMEOUT_MS, TSR_TIMEOUT_NONE = none, else device-aware nanoseconds. On a timeout the
// transport runs DB's recovery ladder (§7.3: hang dump, LQ_CANCEL, SOFT_RESET + warm restart) and
// returns TSR_TE_TIMEOUT (rec = the ABORTED answer if the cancel produced one, else zeros). A TRAP
// answer returns 0 with rec = the TRAP record (status TRAP), after the log was drained, the trap
// reported and the device restarted (unless TSR_GDB_WAIT). Launches flushed by a SOFT_RESET return
// TSR_TE_LOST.
int tsr_transport_wait(tsr_transport * t, uint32_t seq, uint64_t timeout_ns, tsr_record * rec);
        // 0: answered (rec->w[0] has the status); <0: transport failure or timeout (device diagnosed, §8.6)
        // the timeout counts only while the RISC-V is not halted and no tile stop is active (§4.8)
int tsr_transport_call(tsr_transport * t, const tsr_launch * l, uint64_t timeout_ns, tsr_record * rec);  // launch + wait
// Watermarks (EX §4.9): every launch q with (seq_upto - q) mod 2^32 < 2^31 is covered. cancel: hard
// (QABORT of a covered running launch, covered queued launches answered ABORTED); stop: soft. A
// watermark stays valid until SOFT_RESET; 2^30 launches after it was written the transport moves it
// just below the oldest unanswered launch, so it never comes to cover new launches.
int tsr_transport_cancel(tsr_transport * t, uint32_t seq_upto);
int tsr_transport_stop(tsr_transport * t, uint32_t seq_upto);
// Kernel events (records 0xC0-0xFF, EX §4.4) and other unsolicited records are passed to cb as they
// are consumed (any wait or launch); NULL removes it (events are then dropped).
int tsr_transport_set_event_cb(tsr_transport * t, void (*cb)(void *, const tsr_record *), void * user);
// CTRL.SOFT_RESET, wait SR_BUSY = 0 (TSR_RESET_MS). The RISC-V is left in reset: boot again (warm).
// Outstanding launches are lost (their waits return TSR_TE_LOST).
int tsr_transport_soft_reset(tsr_transport * t);                        // SOFT_RESET, wait SR_BUSY = 0
// Raw log words (DB §6.1 records), oldest first: the words the transport drained earlier and kept
// (up to 64 Ki words), then the device ring. *n = words returned.
int tsr_transport_log_drain(tsr_transport * t, uint32_t * words, size_t max, size_t * n);

// ---------------------------------------------------------------------------------------------
// SW §9.1 conveniences (no second protocol)
// ---------------------------------------------------------------------------------------------
int               tsr_image_open_default(tsr_image ** out);   // $TSR_IMAGE or the build's tsr-fw.elf
const char *      tsr_image_path(const tsr_image * img);
// The booted image (ready or trapped) or NULL. The transport holds a reference: the pointer stays
// valid until a boot of another image or tsr_transport_close.
const tsr_image * tsr_transport_image(const tsr_transport * t);   // booted image or NULL

// Versions (SW §6.9): the device's Tesseract Core capability (VERSION, EX §7.5) and CAPS, an
// image's Tesseract Basis version, Basis ABI and kind table (.note.tsr, .tsr_kinds; on a booted
// device HELLO r4 / r6 proved them, EX §4.4).
typedef struct tsr_versions {
    uint32_t core;          // device Tesseract Core capability: VERSION[15:0] (major << 8 | minor); 0 = no device
    uint32_t caps;          // device CAPS
    uint32_t core_min;      // the image's .note.tsr cap_min: the capability it needs
    uint32_t basis;         // the image's Tesseract Basis version (major << 16 | minor << 8 | patch) = HELLO r6
    uint32_t abi;           // the image's Basis ABI = HELLO r4[15:0]
    uint32_t kinds_sig;     // = HELLO r4[31:16]
    uint32_t n_kinds;
    uint8_t  kind_rev[64];  // numerics revision of kind k, 0 = not in the image
    uint64_t image_id;
} tsr_versions;
uint32_t tsr_basis_version(void);                                       // this host library's Basis version
int      tsr_image_versions(const tsr_image * img, tsr_versions * v);   // the image's fields; core = caps = 0
int      tsr_transport_versions(const tsr_transport * t, tsr_versions * v);   // device fields + the booted
                                                                              // image's (zeros before a boot)
// "Tesseract Core 1.0 (VU ATT), Tesseract Basis 0.1.0 (ABI 1, 14 kinds)" (parts without a value are
// left out); verbose adds ", image <id>, kinds 0xfa6f: NOP r1 JOIN r1 ...". Returns the length (as
// snprintf; the text is truncated at n), or TSR_TE_INVAL.
int      tsr_versions_str(const tsr_versions * v, int verbose, char * buf, size_t n);

// Decoded view of a record (accessors over tsr_record.w[], EX §4.4)
typedef struct tsr_result {
    uint8_t  type, status, err_code;       // r0
    uint32_t seq;                          // r1
    int32_t  ret;                          // r2 (DONE)
    uint32_t err_ticket, err_info, err_w0, err_pidx;   // r3-r6 (DONE, first error)
    uint32_t duration;                     // r7, core cycles; 0 for an OK DONE / PONG (minimal reads)
} tsr_result;
void         tsr_record_decode(const tsr_record * rec, tsr_result * out);
const char * tsr_status_name(uint8_t status);   // "OK", "TILE_ERR", … (EX §4.4)

// Statistics since open. launch_wall_ns: host time from each answered launch's push to the
// consumption of its answer, summed; csr_*: core CSR and adapter register accesses; bytes_*:
// DRAM transfers; log_words: log words drained; log_records_dropped: records the firmware reported
// dropped ("dropped n" EVENT records, DB §6.1); staged_chunks: 0 here (the backend counts its
// staged chunks, DB §8.6). TSR_STATS=1 prints them at close.
typedef struct tsr_transport_stats {
    uint64_t launches, launch_wall_ns, csr_reads, csr_writes, bytes_written, bytes_read,
             log_words, log_records_dropped, staged_chunks;
} tsr_transport_stats;
void tsr_transport_get_stats(const tsr_transport * t, tsr_transport_stats * s);

// ---------------------------------------------------------------------------------------------
// DB §9.2 debug additions (P3.7 libtsr_dbg / tsr-dbg, P3.8 tsr-prof). Nothing above needs them.
// ---------------------------------------------------------------------------------------------
// DMI window (DB §3.2): debug-module register addr7 (0..127) at core CSR 0xC00 + 4 addr7, the same
// path on every kind (the sim kind's DBG model forwards it to tsr_dm). The window acts only while
// DBG_CFG.DMI_EN = 1 (tsr_transport_dmi_enable: read-modify-write of DBG_CFG); a DMI read may have
// side effects in the DM (sbdata0 with sbreadondata, cmderr on a busy access), unlike every other
// CSR read. No STATUS read is added per access: callers check STATUS.DMI_TIMEOUT (E01, DB §4.9)
// when a poll does not complete. 0, TSR_TE_INVAL or TSR_TE_IO.
int tsr_transport_dmi_enable(tsr_transport * t, int on);
int tsr_transport_dmi_read(tsr_transport * t, uint32_t addr7, uint32_t * val);
int tsr_transport_dmi_write(tsr_transport * t, uint32_t addr7, uint32_t val);

// The transport's time base (device time on sim, mock and vkernel; host time on xrt) and a pause
// of about ns in it (DB §9.1 "Stepping"): the simulated kinds advance the model by that much (it
// stops early at a host-visible event: a record, a log word, a halt), xrt sleeps. The mutex is
// released for the pause. For a debugger that polls while the hart runs and no launch waits.
uint64_t tsr_transport_now_ns(tsr_transport * t);
void     tsr_transport_idle(tsr_transport * t, uint64_t ns);

// Non-blocking form of tsr_transport_wait: consumes the records posted so far (events to the
// callback, the log drained when due, a TRAP handled as by a wait) and returns 1 with *rec when
// launch seq has been answered (it then counts as waited), 0 while it has not, TSR_TE_LOST when it
// was flushed, TSR_TE_INVAL for a seq that is not outstanding. No timeout, no recovery ladder: for
// a debugger that launches while it controls the hart (tsr-dbg's monitor launch).
int tsr_transport_poll(tsr_transport * t, uint32_t seq, tsr_record * rec);

// A debugger is attached (DB §4.1, §6.3): every wait reads DBG_STATUS / DBG_CFG at its backoff
// polls (no timeout while the hart is halted, a tile stop is active or SERIAL is on), and a TRAP
// record leaves the device TRAPPED for the debugger instead of the warm restart (as TSR_GDB_WAIT).
// 0 restores the environment's behaviour. Boot waits (HELLO, TSR_BOOT_MS) never count time while
// STATUS.RV_HALTED = 1 (halt-on-boot, DB §3.8), with or without this call.
int tsr_transport_set_debugger(tsr_transport * t, int attached);

// The embedded debug service (DB-17, §4.1, §9.5: TSR_GDB, TSR_GDB_WAIT, TSR_DEBUGGER). A library
// registers it once per process (libtsr_dbg: tsr_dbg_embed_install, or by linking tsr_dbg_embed);
// then every tsr_transport_open with TSR_GDB or TSR_DEBUGGER set calls start(t) after the open
// (NULL: not started), every boot calls boot(h, t, 0) before the image load and boot(h, t, 1)
// after HELLO (on the booting thread, with the transport usable), and tsr_transport_close calls
// stop(h) first. NULL unregisters. Without a registered service the variables are reported and
// ignored.
typedef struct tsr_debug_service {
    void * (*start)(tsr_transport * t);
    void   (*boot)(void * h, tsr_transport * t, int phase);
    void   (*stop)(void * h);
} tsr_debug_service;
void tsr_transport_set_debug_service(const tsr_debug_service * s);

// PERF over the session (DB §8.6, TSR_STATS; EX §6): the counters accumulated since open, folded
// in before every SOFT_RESET that zeroes them (EX §8.5), plus the host's launch wall time. PERF
// does not move between launches, so the values cover exactly the launches answered since open.
// core_hz: nominal (276.3 MHz, the U200 k4 clock) on the simulated kinds; on xrt fitted from
// DBG_MTIME against host time when at least 0.5 s passed (core_hz_fitted = 1).
typedef struct tsr_perf_summary {
    uint64_t perf[64];          // PERF_0..PERF_63 (EX §6 indices; TSR_PERF_*)
    uint32_t num_ch, perf_x;    // NUM_CH and X = 5 + 2 NUM_CH
    double   core_hz;
    int      core_hz_fitted;
    uint64_t launches, launch_wall_ns, session_wall_ns;
    uint64_t log_dropped_words; // LOG_DROPPED, accumulated the same way
} tsr_perf_summary;
int tsr_transport_perf_summary(tsr_transport * t, tsr_perf_summary * s);
// The DB §8.6 TSR_STATS text (lines without a prefix, '\n'-separated): device time, launch wall
// and host overhead, DBG_STOP_CYCLES (must be 0), MVM busy / wait / empty and RV_WFI as % of
// CYCLES, PUSH_STALL, LOG_DROPPED, TRW records / drops, per channel read GB/s and idle %. Returns
// the length (as snprintf).
int tsr_perf_summary_str(const tsr_perf_summary * s, char * buf, size_t n);

// Error codes of the int-returning calls, and wait timeouts.
#define TSR_TE_INVAL   (-1)   // bad argument or API misuse (unknown or already waited sequence, no image)
#define TSR_TE_IO      (-2)   // a device access failed (kdev error, connection lost)
#define TSR_TE_TIMEOUT (-3)   // wait timed out; the device was diagnosed and recovered (EX §8.6)
#define TSR_TE_LOST    (-4)   // the launch was flushed by a SOFT_RESET / restart without an answer
#define TSR_TE_DEVICE  (-5)   // device failure: unit hung ("card reset needed"), no clk_rv, reset stuck
#define TSR_TE_IMAGE   (-6)   // image refused: requirements, CRC / VERIFY mismatch, HELLO checks
#define TSR_TE_STATE   (-7)   // wrong device state for the call (not booted, trapped, failed)
const char * tsr_transport_strerror(int rc);
#define TSR_TIMEOUT_DEFAULT 0ull            // tsr_transport_wait / _call: TSR_LAUNCH_TIMEOUT_MS
#define TSR_TIMEOUT_NONE    UINT64_MAX      // no timeout

#ifdef __cplusplus
}
#endif

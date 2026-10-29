// tsr_sim.h — Tesseract ISA simulator (the oracle, spec 검증 전략).
//
// The op-level core of the simulator: SPM, per-channel DRAM, s0..s15, the identity /
// configuration / PERF CSRs, the tile units with their dispatch checks, per-channel beat
// counters and access capture. Every numeric result comes from tsr_ref.h. Ops execute one at a
// time, in call order, through tsr_sim_op(). v1's host command path (run registers, command
// fetch, R0..R15, LI / ADDI / END) is deleted (X0, docs/v2/01-execution-model.md §12); the v2
// model around this core (CSR map v2, launch queue and completion ring, issue queues, the RISC-V
// model) is docs/v2/06-software-verification.md §4 (W1).
//
// Ops: NOP, LD, ST, QACT, MVM, FENCE (EX §7.2). Every other opcode is ILLEGAL_OP (END, LI and ADDI
// are reserved, and this simulator implements no CAPS feature; see TSR_SIM_CAPS_IMPLEMENTED).
// Checks follow spec "구현 명확화" and EX §7.1 / §8.1: x16 (w0[31:16]) and every other field an op
// does not use must be zero (ILLEGAL_OP); check order ILLEGAL_OP -> BAD_FMT -> K_RANGE ->
// T_RANGE -> SPM_RANGE -> DRAM_RANGE; a failing op changes no state.
//
// Perf model (PERF_* CSRs; deterministic, not normative — hardware counters are compared
// against it, not bit-matched): see the "Cycle model" comment in tsr_sim.c.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_sim tsr_sim;

// CAPS bits this simulator implements (none of the optional M6+ features yet). A config
// whose caps field asks for more is rejected by tsr_sim_create.
#define TSR_SIM_CAPS_IMPLEMENTED 0u

// Create a simulator. Channel DRAM is cfg->ch_bytes per channel, mapped lazily
// (MAP_NORESERVE) so a 4 x 16 GiB address space costs only what is touched; it starts zeroed.
// If dram_dir is non-NULL, each channel is backed by a sparse, already-unlinked temporary
// file created in dram_dir (unique name per process and channel, nothing is left behind),
// so models larger than RAM can live in the page cache.
// SPM starts filled with 0xA5 (spec: SPM contents are undefined at start; programs must not rely on them).
// Returns NULL (message on stderr) if the config is not supported: beat_bits != 512,
// num_ch not in [1,16] (TSR_NUM_CH_MAX: the LD/ST channel field is 4 bits), lanes not in
// [1,4095], t_max not in [1,8], k_max not a multiple of 256 in [256,65280], spm_bytes not a
// positive multiple of 64, vl not in [1,255], nsreg not in [1,16], ch_bytes not a positive
// multiple of 64, addr_beat_w not in [1,40], or caps beyond TSR_SIM_CAPS_IMPLEMENTED.
tsr_sim * tsr_sim_create(const tsr_config * cfg, const char * dram_dir);
void      tsr_sim_destroy(tsr_sim * sim);
const tsr_config * tsr_sim_config(const tsr_sim * sim);

// Direct DRAM access (host side). Returns a pointer to byte 0 of channel ch.
uint8_t * tsr_sim_dram(tsr_sim * sim, uint32_t ch);
uint8_t * tsr_sim_spm(tsr_sim * sim);            // debug view, spm_bytes long
// Addressable beats per channel: min(ch_bytes / 64, 2^ADDR_BEAT_W). A DRAM operand
// [a, a + n) is in range iff a + n <= this value (spec 구현 명확화, 범위 검사).
uint64_t  tsr_sim_dram_limit_beats(const tsr_sim * sim);

// CSR port (host view). Reads: ID, VERSION (ISA 2.0), CFG0..2, CAPS and the PERF_* counters;
// every other offset reads 0 (EX §5.1: undefined offsets read 0). No register is writable yet
// (the v2 map, CTRL / STATUS and the rings, is P1.2): writes are ignored.
uint32_t tsr_sim_csr_read(tsr_sim * sim, uint32_t off);
void     tsr_sim_csr_write(tsr_sim * sim, uint32_t off, uint32_t val);

// Check and execute one op now: w is its 8 instruction words (w[0] = w0). Returns 0 or the op's
// error code (tsr_err); *err_info (if not NULL) receives ERR_INFO (the failing operand's start
// address for SPM_RANGE / DRAM_RANGE, else 0). An op that fails changes no state except PERF.
// There are no queues: NOP and FENCE complete at once, and ops never overlap.
uint32_t tsr_sim_op(tsr_sim * sim, const uint32_t w[8], uint32_t * err_info);

// Debug state.
float    tsr_sim_sreg(const tsr_sim * sim, uint32_t i);   // s0..s15 (+0.0 after create)
uint64_t tsr_sim_perf(const tsr_sim * sim, uint32_t idx);   // tsr_perf_idx (+ch)

// Optional per-op log to a FILE* (NULL disables): one line per tsr_sim_op call.
void tsr_sim_set_log(tsr_sim * sim, void * file /* FILE* */);

// Worker threads used for large MVMs (rows are split across threads; results are identical
// for any thread count). Default 1. Values are clamped to [1, 64].
void tsr_sim_set_threads(tsr_sim * sim, unsigned n);

// ---------------------------------------------------------------------------
// Access capture for trace dumps (spec 트레이스 형식; trace v2 builds on it, SW §5.5)
// While capture is on, the simulator records, in beat granularity per channel:
//   - init: every DRAM range the ops READ (LD sources, MVM weight rows) whose bytes were not
//           written earlier during the capture; the bytes are copied at first read, i.e. the
//           content before execution.
//   - expect: every DRAM range the ops WRITE (ST); content taken at capture end.
// Adjacent/overlapping beats are merged into maximal ranges, sorted by (ch, offset).
// Host-side writes (tsr_sim_dram) are not tracked, so a capture should bracket exactly the ops
// of one run.
// ---------------------------------------------------------------------------
typedef struct tsr_region {
    uint32_t ch;
    uint64_t byte_off;     // multiple of 64
    uint64_t size;         // multiple of 64
    uint8_t * data;        // owned by the capture
} tsr_region;

typedef struct tsr_capture {
    tsr_region * init;   size_t n_init;
    tsr_region * expect; size_t n_expect;
} tsr_capture;

void tsr_sim_capture_begin(tsr_sim * sim);
// Ends capture and moves the recorded regions into *out (caller frees with tsr_capture_free).
// Returns an empty capture if capture was not on.
void tsr_sim_capture_end(tsr_sim * sim, tsr_capture * out);
void tsr_capture_free(tsr_capture * cap);

#ifdef __cplusplus
}
#endif

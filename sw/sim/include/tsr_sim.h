// tsr_sim.h — Tesseract ISA simulator (the oracle, spec 검증 전략).
//
// Functional, instruction-at-a-time model of the core: decoder, R0..R15, s0..s15, SPM,
// per-channel DRAM, CSRs, error codes, per-channel beat counters. Every numeric result
// comes from tsr_ref.h. Execution is strictly sequential (spec 실행 모델).
//
// M1 instruction set: NOP, END, LI, ADDI, LD, ST, QACT, MVM, FENCE. Every other opcode is
// ILLEGAL_OP (this simulator implements no CAPS feature; see TSR_SIM_CAPS_IMPLEMENTED).
// Error semantics follow spec "구현 명확화": check order ILLEGAL_OP -> BAD_FMT -> K_RANGE ->
// T_RANGE -> SPM_RANGE -> DRAM_RANGE, a failing instruction changes no state, and the run
// stops with STATUS = DONE | ERR and ERR_PC / ERR_CODE / ERR_INFO set.
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
// positive multiple of 64, vl not in [1,255], nireg != 16, nsreg not in [1,16], ch_bytes not
// a positive multiple of 64, addr_beat_w not in [1,40], or caps beyond
// TSR_SIM_CAPS_IMPLEMENTED.
tsr_sim * tsr_sim_create(const tsr_config * cfg, const char * dram_dir);
void      tsr_sim_destroy(tsr_sim * sim);
const tsr_config * tsr_sim_config(const tsr_sim * sim);

// Direct DRAM access (host side). Returns a pointer to byte 0 of channel ch.
uint8_t * tsr_sim_dram(tsr_sim * sim, uint32_t ch);
uint8_t * tsr_sim_spm(tsr_sim * sim);            // debug view, spm_bytes long
// Addressable beats per channel: min(ch_bytes / 64, 2^ADDR_BEAT_W). A DRAM operand
// [a, a + n) is in range iff a + n <= this value (spec 구현 명확화, 범위 검사).
uint64_t  tsr_sim_dram_limit_beats(const tsr_sim * sim);

// CSR port. Writing CTRL.START latches CMD_ADDR/CMD_COUNT/ARGs, sets BUSY and clears
// DONE/ERR; execution happens in tsr_sim_run().
// CTRL.ABORT may be written from another thread while tsr_sim_run() is executing (it only
// sets an atomic flag, checked between instructions); every other CSR access must not race
// with tsr_sim_run().
uint32_t tsr_sim_csr_read(tsr_sim * sim, uint32_t off);
void     tsr_sim_csr_write(tsr_sim * sim, uint32_t off, uint32_t val);

// Run until END or an error. Returns the final STATUS value. No-op if not BUSY.
uint32_t tsr_sim_run(tsr_sim * sim);

// Convenience: write CMD_ADDR/COUNT/ARGs, START, run. args may be NULL (all zero).
uint32_t tsr_sim_exec(tsr_sim * sim, uint64_t cmd_addr_beat, uint32_t cmd_count,
                      const uint32_t args[8]);

// Debug state.
int32_t  tsr_sim_ireg(const tsr_sim * sim, uint32_t i);
float    tsr_sim_sreg(const tsr_sim * sim, uint32_t i);
uint64_t tsr_sim_perf(const tsr_sim * sim, uint32_t idx);   // tsr_perf_idx (+ch)

// Optional per-instruction log to a FILE* (NULL disables).
void tsr_sim_set_log(tsr_sim * sim, void * file /* FILE* */);

// Worker threads used for large MVMs (rows are split across threads; results are identical
// for any thread count). Default 1. Values are clamped to [1, 64].
void tsr_sim_set_threads(tsr_sim * sim, unsigned n);

// ---------------------------------------------------------------------------
// Access capture for trace dumps (spec 트레이스 형식)
// While capture is on, the simulator records, in beat granularity per channel:
//   - init: every DRAM range the program READS (command fetch, LD sources, MVM weight rows,
//           ...) whose bytes were not written earlier in the same run; the bytes are
//           copied at first read, i.e. the content before execution.
//   - expect: every DRAM range the program WRITES (ST); content taken at the end of run.
// Adjacent/overlapping beats are merged into maximal ranges, sorted by (ch, offset).
// Command fetch reads one beat (two instructions) at a time, only for instructions that are
// actually fetched. Host-side writes (tsr_sim_dram) are not tracked, so a capture should
// bracket exactly one run.
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

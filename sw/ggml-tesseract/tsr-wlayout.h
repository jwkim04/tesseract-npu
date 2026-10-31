// tsr-wlayout.h — device weight layout helpers without ggml (spec "Device weight layout",
// "Symmetric allocation"): bytes of a row of device-format blocks, and the row scatter / gather between
// ggml-layout blocks and the repacked rows of a symmetrically allocated tensor on the channels.
//
// Shared by the ggml backend (tsr_store_weight_rows / tsr_load_weight_rows in ggml-tesseract.cpp)
// and the ggml-free runtime (sw/rt), so both put every weight byte at the same place.
// Part of the tsr_lower library (with tsr-lower and tsr-alloc). No ggml dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

// Bytes of one row of K weights in ggml block layout of the device format
// (Q8_0: K/32 * 34, Q4_K: K/256 * 144).
size_t tsr_dev_blocks_row_bytes(uint32_t fmt, uint32_t K);

// Row r of the tensor in ggml blocks of the device format (tsr_dev_blocks_row_bytes bytes). The
// callee may convert into `scratch` (tsr_dev_blocks_row_bytes bytes) and return it, or return a
// pointer to blocks it already has.
using tsr_row_blocks_fn = std::function<const void * (uint32_t r, void * scratch)>;
// Host -> device write of `size` bytes at byte offset `off` of channel `ch`.
using tsr_dram_write_fn = std::function<void (uint32_t ch, uint64_t off, const void * src, size_t size)>;
// Device -> host read.
using tsr_dram_read_fn  = std::function<void (uint32_t ch, uint64_t off, void * dst, size_t size)>;
// Receives the ggml blocks of row r (valid only during the call).
using tsr_row_sink_fn   = std::function<void (uint32_t r, const void * blocks)>;

// Store rows [r0, r1) of an N x K weight in device format `fmt` at W_OFF = w_off (bytes, the same
// on every channel; symmetric allocation over num_ch channels: R = ceil(N / C), row r on channel
// r / R at w_off + (r mod R) * ROW_BEATS * 64). Each row is repacked with tsr_repack_row; the rows
// of one channel are written in batches of max(1, max_batch_bytes / row bytes) rows.
// threads > 1: that many workers convert and repack batches while the calling thread writes the
// finished ones, in completion order (the batches are disjoint). row_blocks is then called
// concurrently, each worker with its own scratch, and must be thread-safe; write is only ever
// called from the calling thread. Host memory: (workers + 1) batches.
void tsr_weight_store_rows(uint32_t num_ch, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off,
                           uint32_t r0, uint32_t r1, size_t max_batch_bytes,
                           const tsr_row_blocks_fn & row_blocks, const tsr_dram_write_fn & write,
                           unsigned threads = 1);

// Inverse: read rows [r0, r1) in batches as above, un-repack each (tsr_unrepack_row) and pass its
// ggml blocks to sink.
void tsr_weight_load_rows(uint32_t num_ch, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off,
                          uint32_t r0, uint32_t r1, size_t max_batch_bytes,
                          const tsr_dram_read_fn & read, const tsr_row_sink_fn & sink);

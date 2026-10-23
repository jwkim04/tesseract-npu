// tsr-weights.h — ggml type <-> device weight format (spec "디바이스 가중치 레이아웃",
// "로드 시 변환").
//
// The hardware stores Q8_0 and Q4_K only. Q8_0 and Q4_K tensors keep their ggml blocks
// (bit-identical, only repacked by tsr_repack_row). Every other type that ggml can convert
// to f32 is converted at load: ggml type_traits.to_float, then quantize_row_q8_0_ref.
#pragma once

#include <cstddef>
#include <cstdint>

#include "ggml.h"

// Device format for a ggml weight type: Q4_K -> TSR_FMT_Q4_K, everything else -> TSR_FMT_Q8_0.
uint32_t tsr_dev_fmt_for_type(enum ggml_type type);

// True if rows of `type` can be turned into device-format blocks (native Q8_0/Q4_K, F32, or
// any type with a ggml to_float).
bool tsr_type_loadable(enum ggml_type type);

// True if the device format is the tensor's own type (no conversion, exact round trip).
bool tsr_type_is_native(enum ggml_type type);

// Bytes of one row of K weights in ggml block layout of the device format
// (Q8_0: K/32 * 34, Q4_K: K/256 * 144).
size_t tsr_dev_blocks_row_bytes(uint32_t fmt, uint32_t K);

// One row (K elements of `type`, ggml layout) -> ggml blocks of the device format.
// Native types return src_row itself; otherwise the row is converted into dst_blocks
// (tsr_dev_blocks_row_bytes bytes) using f32_tmp (K floats) and dst_blocks is returned.
const void * tsr_row_to_dev_blocks(enum ggml_type type, uint32_t fmt, const void * src_row, uint32_t K,
                                   void * dst_blocks, float * f32_tmp);

// Inverse, for the debug get_tensor of weights: device-format blocks -> one row of `type`
// (ggml_row_size(type, K) bytes). Exact for native types. For converted types the row is
// dequantized and re-encoded with the type's from_float_ref, i.e. lossy. Returns false if the
// type has no from_float_ref (e.g. I-quants needing an imatrix).
bool tsr_dev_blocks_to_row(enum ggml_type type, uint32_t fmt, const void * blocks, uint32_t K,
                           void * dst_row, float * f32_tmp);

// tsr-weights.cpp — see tsr-weights.h
#include "tsr-weights.h"

#include <cstring>

#include "ggml-quants.h"   // quantize_row_q8_0_ref, block_q8_0, block_q4_K (ggml/src, internal)
#include "tsr_isa.h"
#include "tsr_ref.h"

static_assert(sizeof(block_q8_0) == sizeof(tsr_block_q8_0), "block_q8_0 layout mismatch");
static_assert(sizeof(block_q4_K) == sizeof(tsr_block_q4_K), "block_q4_K layout mismatch");
static_assert(QK8_0 == TSR_QK8_0 && QK_K == TSR_QK_K, "block size mismatch");

uint32_t tsr_dev_fmt_for_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_K ? TSR_FMT_Q4_K : TSR_FMT_Q8_0;
}

bool tsr_type_is_native(enum ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K;
}

bool tsr_type_loadable(enum ggml_type type) {
    if ((int) type < 0 || (int) type >= (int) GGML_TYPE_COUNT) {
        return false;
    }
    if (tsr_type_is_native(type) || type == GGML_TYPE_F32) {
        return true;
    }
    const struct ggml_type_traits * tr = ggml_get_type_traits(type);
    return tr != nullptr && tr->to_float != nullptr && tr->type_size > 0 && tr->blck_size > 0;
}

const void * tsr_row_to_dev_blocks(enum ggml_type type, uint32_t fmt, const void * src_row, uint32_t K,
                                   void * dst_blocks, float * f32_tmp) {
    if ((type == GGML_TYPE_Q8_0 && fmt == TSR_FMT_Q8_0) || (type == GGML_TYPE_Q4_K && fmt == TSR_FMT_Q4_K)) {
        return src_row;
    }
    GGML_ASSERT(fmt == TSR_FMT_Q8_0 && "only Q8_0 is a conversion target");
    GGML_ASSERT(K % QK8_0 == 0);
    if (type == GGML_TYPE_F32) {
        quantize_row_q8_0_ref((const float *) src_row, (block_q8_0 *) dst_blocks, K);
    } else {
        const struct ggml_type_traits * tr = ggml_get_type_traits(type);
        GGML_ASSERT(tr->to_float != nullptr);
        tr->to_float(src_row, f32_tmp, K);
        quantize_row_q8_0_ref(f32_tmp, (block_q8_0 *) dst_blocks, K);
    }
    return dst_blocks;
}

bool tsr_dev_blocks_to_row(enum ggml_type type, uint32_t fmt, const void * blocks, uint32_t K,
                           void * dst_row, float * f32_tmp) {
    if ((type == GGML_TYPE_Q8_0 && fmt == TSR_FMT_Q8_0) || (type == GGML_TYPE_Q4_K && fmt == TSR_FMT_Q4_K)) {
        memcpy(dst_row, blocks, tsr_dev_blocks_row_bytes(fmt, K));
        return true;
    }
    GGML_ASSERT(fmt == TSR_FMT_Q8_0);
    dequantize_row_q8_0((const block_q8_0 *) blocks, f32_tmp, K);
    if (type == GGML_TYPE_F32) {
        memcpy(dst_row, f32_tmp, (size_t) K * sizeof(float));
        return true;
    }
    const struct ggml_type_traits * tr = ggml_get_type_traits(type);
    if (tr->from_float_ref == nullptr) {
        return false;
    }
    tr->from_float_ref(f32_tmp, dst_row, K);
    return true;
}

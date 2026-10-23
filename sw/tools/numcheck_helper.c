// numcheck_helper.c — small C entry points for sw/tools/numcheck.py (docs/numcheck.md), loaded
// with ctypes. Two jobs:
//   1. ggml's own conversion functions, so the comparer reproduces the backend's device-format
//      weights bit for bit: ggml to_float (type traits) and quantize_row_q8_0_ref, combined
//      exactly like sw/ggml-tesseract/tsr-weights.cpp tsr_row_to_dev_blocks (native Q8_0/Q4_K
//      bytes as they are; F32 -> quantize_row_q8_0_ref; every other type -> to_float of the row,
//      then quantize_row_q8_0_ref).
//   2. the bit-exact reference model of a weight MUL_MAT through libtsr_ref (B1): per token
//      tsr_qact, per row tsr_repack_row + tsr_dot_row, looped in C over rows and tokens
//      (threads over rows) so Python does not make one call per element.
// Nothing here does arithmetic of its own: every number comes from ggml or tsr_ref.

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "tsr_isa.h"
#include "tsr_ref.h"

// ggml/src/ggml-quants.h (internal header); exported by libggml-base. The block pointer is
// written as void * here; the ABI is the same.
extern void quantize_row_q8_0_ref(const float * x, void * y, int64_t k);

#define NC_API __attribute__((visibility("default")))

NC_API int nc_abi_version(void);
NC_API int nc_type_count(void);
NC_API const char * nc_type_name(int type);
NC_API int64_t nc_blck_size(int type);
NC_API int64_t nc_type_size(int type);
NC_API int64_t nc_row_size(int type, int64_t ne0);
NC_API int nc_has_to_float(int type);
NC_API int nc_to_float(int type, const void * src, float * dst, int64_t n);
NC_API void nc_quantize_q8_0_ref(const float * x, void * dst, int64_t n);
NC_API int nc_dev_fmt(int type);
NC_API int nc_to_dev_blocks(int type, const void * src, int64_t nrows, int64_t K, void * dst);
NC_API uint32_t nc_qa_lines(uint32_t kind, uint32_t K);
NC_API int nc_qact(uint32_t kind, const float * x, uint32_t K, uint32_t T, uint8_t * qa);
NC_API int nc_model_mm(uint32_t fmt, const void * blocks, uint32_t N, uint32_t K, const float * x, uint32_t T,
                       float * y, int nthreads);
NC_API int nc_model_mm_qa(uint32_t fmt, const void * blocks, uint32_t N, uint32_t K, const uint8_t * qa,
                          uint32_t T, float * y, int nthreads);

int nc_abi_version(void) { return 1; }
int nc_type_count(void) { return (int) GGML_TYPE_COUNT; }

static int type_ok(int type) { return type >= 0 && type < (int) GGML_TYPE_COUNT && ggml_blck_size((enum ggml_type) type) > 0; }

const char * nc_type_name(int type) { return type_ok(type) ? ggml_type_name((enum ggml_type) type) : "?"; }
int64_t nc_blck_size(int type) { return type_ok(type) ? ggml_blck_size((enum ggml_type) type) : 0; }
int64_t nc_type_size(int type) { return type_ok(type) ? (int64_t) ggml_type_size((enum ggml_type) type) : 0; }
int64_t nc_row_size(int type, int64_t ne0) {
    return type_ok(type) ? (int64_t) ggml_row_size((enum ggml_type) type, ne0) : 0;
}

int nc_has_to_float(int type) {
    if (!type_ok(type)) return 0;
    if (type == GGML_TYPE_F32) return 1;
    const struct ggml_type_traits * tr = ggml_get_type_traits((enum ggml_type) type);
    return tr != NULL && tr->to_float != NULL;
}

// dst[n] = ggml to_float(src) (F32: copy). Returns 0, or -1 (no to_float), -2 (n not a block multiple).
int nc_to_float(int type, const void * src, float * dst, int64_t n) {
    if (!nc_has_to_float(type)) return -1;
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t) n * sizeof(float));
        return 0;
    }
    if (n % ggml_blck_size((enum ggml_type) type) != 0) return -2;
    ggml_get_type_traits((enum ggml_type) type)->to_float(src, dst, n);
    return 0;
}

void nc_quantize_q8_0_ref(const float * x, void * dst, int64_t n) { quantize_row_q8_0_ref(x, dst, n); }

// Device format of a GGUF type (tsr-weights.cpp tsr_dev_fmt_for_type): Q4_K -> Q4_K, else Q8_0.
int nc_dev_fmt(int type) { return type == GGML_TYPE_Q4_K ? (int) TSR_FMT_Q4_K : (int) TSR_FMT_Q8_0; }

// nrows rows of K elements of `type` (GGUF bytes, rows contiguous) -> the ggml-layout blocks of
// the device format (Q8_0 34 B / 32, Q4_K 144 B / 256), row by row as tsr_row_to_dev_blocks.
// Returns the device fmt, or -1 (type without to_float), -2 (K not a multiple of 32 / the block).
int nc_to_dev_blocks(int type, const void * src, int64_t nrows, int64_t K, void * dst) {
    const int fmt = nc_dev_fmt(type);
    if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K) {
        if (K % ggml_blck_size((enum ggml_type) type) != 0) return -2;
        memcpy(dst, src, (size_t) nrows * ggml_row_size((enum ggml_type) type, K));
        return fmt;
    }
    if (!nc_has_to_float(type)) return -1;
    if (K % 32 != 0 || K % ggml_blck_size((enum ggml_type) type) != 0) return -2;
    const size_t src_row = ggml_row_size((enum ggml_type) type, K);
    const size_t dst_row = ggml_row_size(GGML_TYPE_Q8_0, K);
    float * tmp = (float *) malloc((size_t) K * sizeof(float));
    if (!tmp) return -3;
    for (int64_t r = 0; r < nrows; ++r) {
        const uint8_t * s = (const uint8_t *) src + (size_t) r * src_row;
        uint8_t * d = (uint8_t *) dst + (size_t) r * dst_row;
        if (type == GGML_TYPE_F32) {
            quantize_row_q8_0_ref((const float *) (const void *) s, d, K);
        } else {
            ggml_get_type_traits((enum ggml_type) type)->to_float(s, tmp, K);
            quantize_row_q8_0_ref(tmp, d, K);
        }
    }
    free(tmp);
    return fmt;
}

uint32_t nc_qa_lines(uint32_t kind, uint32_t K) { return tsr_qa_lines(kind, K); }

// T QA images of tsr_qa_lines(kind, K) * 64 bytes each, consecutive (QACT semantics, stride QA_LINES).
int nc_qact(uint32_t kind, const float * x, uint32_t K, uint32_t T, uint8_t * qa) {
    if (kind > TSR_QA_A256 || K == 0 || K % tsr_qa_block_size(kind) != 0) return -1;
    const size_t img = (size_t) tsr_qa_lines(kind, K) * TSR_LINE_BYTES;
    for (uint32_t t = 0; t < T; ++t) {
        tsr_qact(kind, x + (size_t) t * K, K, qa + (size_t) t * img);
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// B1 model: y[t * N + n] = tsr_dot_row(fmt, repack(row n), K, QA_t)

typedef struct {
    uint32_t fmt, N, K, T;
    const uint8_t * blocks;     // N rows of ggml-layout blocks of fmt
    const uint8_t * qa;         // T QA images (aligned)
    size_t img;                 // bytes per QA image
    float * y;
    uint32_t r0, r1;
    int err;
} nc_job;

static void * nc_worker(void * arg) {
    nc_job * j = (nc_job *) arg;
    const size_t row_in  = tsr_fmt_ggml_block_bytes(j->fmt) * (size_t) (j->K / tsr_fmt_block_size(j->fmt));
    const size_t row_dev = (size_t) tsr_row_beats(j->fmt, j->K) * TSR_BEAT_BYTES;
    uint8_t * buf = (uint8_t *) aligned_alloc(64, (row_dev + 63) / 64 * 64);
    if (!buf) {
        j->err = -3;
        return NULL;
    }
    for (uint32_t n = j->r0; n < j->r1; ++n) {
        tsr_repack_row(j->fmt, j->blocks + (size_t) n * row_in, j->K, buf);
        for (uint32_t t = 0; t < j->T; ++t) {
            j->y[(size_t) t * j->N + n] = tsr_dot_row(j->fmt, buf, j->K, j->qa + (size_t) t * j->img);
        }
    }
    free(buf);
    return NULL;
}

static int run_mm(uint32_t fmt, const void * blocks, uint32_t N, uint32_t K, const uint8_t * qa, size_t img,
                  uint32_t T, float * y, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    if ((uint32_t) nthreads > N) nthreads = N > 0 ? (int) N : 1;
    nc_job jobs[64];
    pthread_t th[64];
    const uint32_t per = (N + (uint32_t) nthreads - 1) / (uint32_t) nthreads;
    int joined[64];
    for (int i = 0; i < nthreads; ++i) {
        const uint32_t r0 = (uint32_t) i * per < N ? (uint32_t) i * per : N;
        const uint32_t r1 = r0 + per < N ? r0 + per : N;
        jobs[i] = (nc_job){ fmt, N, K, T, (const uint8_t *) blocks, qa, img, y, r0, r1, 0 };
        joined[i] = 0;
        if (i == nthreads - 1 || pthread_create(&th[i], NULL, nc_worker, &jobs[i]) != 0) {
            nc_worker(&jobs[i]);       // last share (or a failed create) on this thread
        } else {
            joined[i] = 1;
        }
    }
    int err = 0;
    for (int i = 0; i < nthreads; ++i) {
        if (joined[i]) pthread_join(th[i], NULL);
        if (jobs[i].err) err = jobs[i].err;
    }
    return err;
}

static int mm_args_ok(uint32_t fmt, uint32_t K) { return tsr_fmt_valid(fmt) && K > 0 && K % tsr_fmt_block_size(fmt) == 0; }

// x: T tokens of K f32 (row-major [T][K]); y: [T][N]. Returns 0 or < 0.
int nc_model_mm(uint32_t fmt, const void * blocks, uint32_t N, uint32_t K, const float * x, uint32_t T,
                float * y, int nthreads) {
    if (!mm_args_ok(fmt, K)) return -1;
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    const size_t img = (size_t) tsr_qa_lines(kind, K) * TSR_LINE_BYTES;
    uint8_t * qa = (uint8_t *) aligned_alloc(64, img * (T ? T : 1));
    if (!qa) return -3;
    nc_qact(kind, x, K, T, qa);
    const int r = run_mm(fmt, blocks, N, K, qa, img, T, y, nthreads);
    free(qa);
    return r;
}

// Same with caller-made QA images (T images of tsr_qa_lines(kind, K) * 64 bytes, consecutive).
int nc_model_mm_qa(uint32_t fmt, const void * blocks, uint32_t N, uint32_t K, const uint8_t * qa,
                   uint32_t T, float * y, int nthreads) {
    if (!mm_args_ok(fmt, K)) return -1;
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    const size_t img = (size_t) tsr_qa_lines(kind, K) * TSR_LINE_BYTES;
    uint8_t * a = (uint8_t *) aligned_alloc(64, img * (T ? T : 1));   // 4-byte alignment for the D/BS views
    if (!a) return -3;
    memcpy(a, qa, img * T);
    const int r = run_mm(fmt, blocks, N, K, a, img, T, y, nthreads);
    free(a);
    return r;
}

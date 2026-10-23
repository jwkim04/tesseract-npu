// test-tsr-scalar-nmse.cpp — the single-output MUL_MAT case of test-backend-ops,
//   MUL_MAT(type_a=f32, type_b=f32, m=1, n=1, k=2048)
// checked with a stricter oracle than test-backend-ops can apply to it.
//
// Why this test exists: test-backend-ops judges MUL_MAT by NMSE <= 5e-4 against the CPU
// (sum (a-b)^2 / sum a^2). With one output element that is the squared relative error of a
// single dot product of U(-1,1) data. The spec's device path quantizes the F32 weight to Q8_0
// at load (spec 로드 시 변환) and the activation to A32 int8 (spec QA 레이아웃), which leaves an
// absolute error of about 0.08 on a dot product whose value is ~N(0, 15^2); whenever the true
// value happens to be small (|y| < ~3.6) the relative error exceeds the tolerance. Because
// test-backend-ops draws fresh random data every run (std::random_device), that one case fails
// in about 15% of runs on any device that implements the spec. The ctest
// ggml_tesseract_backend_ops therefore excludes exactly this shape, and this test covers it:
//
//   (1) BIT-EXACT: on every trial the device output equals the spec model built from tsr_ref
//       (quantize_row_q8_0_ref -> tsr_repack_row, tsr_qact A32, tsr_dot_row).
//   (2) The NMSE failures are caused by the 8-bit operands alone: the exact (double) dot
//       product of the dequantized Q8_0 weight and dequantized A32 activation fails the 5e-4
//       limit on the same trials (up to borderline flips by f32 accumulation rounding).
//   (3) The NMSE noise floor (median) is far below the tolerance.
// Deterministic seed; prints the measured failure rates.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"      // quantize_row_q8_0_ref, dequantize_row_q8_0 (ggml/src)
#include "ggml-tesseract.h"

#include "tsr_isa.h"
#include "tsr_ref.h"

static double nmse(const float * a, const float * b, size_t n) {   // test-backend-ops definition
    double ab = 0.0, a0 = 0.0;
    for (size_t i = 0; i < n; i++) {
        ab += (double) (a[i] - b[i]) * (a[i] - b[i]);
        a0 += (double) a[i] * a[i];
    }
    return ab / a0;
}

static float mul_mat_1x1(ggml_backend_t be, const std::vector<float> & a, const std::vector<float> & b, int K) {
    ggml_init_params ip = { 3 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * to = ggml_mul_mat(ctx, ta, tb);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, to);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) {
        fprintf(stderr, "buffer allocation failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(ta, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(tb, b.data(), 0, b.size() * sizeof(float));
    if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed\n");
        exit(1);
    }
    float y = 0.0f;
    ggml_backend_tensor_get(to, &y, 0, sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return y;
}

int main(int argc, char ** argv) {
    const int    K      = 2048;
    const int    TRIALS = argc > 1 ? atoi(argv[1]) : 2000;
    const double TOL    = 5e-4;   // test_mul_mat::max_nmse_err() at the pin

    ggml_backend_t tsr = ggml_backend_tesseract_init();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (tsr == nullptr || cpu == nullptr) {
        fprintf(stderr, "backend init failed (Tesseract %p, CPU %p)\n", (void *) tsr, (void *) cpu);
        return 1;
    }

    std::mt19937 rng(20261022);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);   // test-backend-ops init_tensor_uniform

    int mismatch = 0, fail_dev = 0, fail_q = 0, fail_dev_only = 0;
    std::vector<double> nm;
    std::vector<float> a(K), b(K), wdq(K);
    std::vector<block_q8_0> wq(K / QK8_0);
    std::vector<uint8_t> row((size_t) tsr_row_beats(TSR_FMT_Q8_0, K) * TSR_BEAT_BYTES);
    std::vector<uint8_t> qa((size_t) tsr_qa_lines(TSR_QA_A32, K) * TSR_LINE_BYTES);

    for (int tr = 0; tr < TRIALS; ++tr) {
        for (auto & v : a) v = u(rng);
        for (auto & v : b) v = u(rng);
        const float y_dev = mul_mat_1x1(tsr, a, b, K);
        const float y_cpu = mul_mat_1x1(cpu, a, b, K);

        // spec model: F32 weight -> Q8_0 (load conversion), repack, QACT A32, DOT
        quantize_row_q8_0_ref(a.data(), wq.data(), K);
        tsr_repack_row(TSR_FMT_Q8_0, wq.data(), K, row.data());
        tsr_qact(TSR_QA_A32, b.data(), K, qa.data());
        const float y_ref = tsr_dot_row(TSR_FMT_Q8_0, row.data(), K, qa.data());
        if (tsr_f32_bits(y_dev) != tsr_f32_bits(y_ref)) {
            if (mismatch < 5) {
                fprintf(stderr, "  FAIL trial %d: device 0x%08x (%.9g) != tsr_ref model 0x%08x (%.9g)\n", tr,
                        tsr_f32_bits(y_dev), (double) y_dev, tsr_f32_bits(y_ref), (double) y_ref);
            }
            mismatch++;
        }

        // the 8-bit operands, multiplied and summed exactly (double; every product is exact)
        dequantize_row_q8_0(wq.data(), wdq.data(), K);
        const int8_t * q = tsr_qa_qs(TSR_QA_A32, K, qa.data());
        const float *  D = tsr_qa_d(TSR_QA_A32, K, qa.data());
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            acc += (double) wdq[k] * ((double) D[k / QK8_0] * (double) q[k]);
        }
        const float y_q = (float) acc;

        const double nd = nmse(&y_cpu, &y_dev, 1);
        const double nq = nmse(&y_cpu, &y_q, 1);
        nm.push_back(nd);
        fail_dev += nd > TOL;
        fail_q   += nq > TOL;
        // a device-only failure must be a borderline one (f32 accumulation rounding)
        fail_dev_only += nd > TOL && nq <= TOL && nd > 1.1 * TOL;
    }
    std::sort(nm.begin(), nm.end());
    const double median = nm[nm.size() / 2];

    printf("MUL_MAT f32 x f32 m=1 n=1 k=%d, %d trials of U(-1,1) data:\n", K, TRIALS);
    printf("  device vs tsr_ref spec model : %d bit mismatches\n", mismatch);
    printf("  NMSE(cpu, device) > %.0e    : %d (%.1f%%)\n", TOL, fail_dev, 100.0 * fail_dev / TRIALS);
    printf("  NMSE(cpu, exact dot of the 8-bit operands) > %.0e : %d (%.1f%%)\n", TOL, fail_q, 100.0 * fail_q / TRIALS);
    printf("  NMSE(cpu, device) median %.2e, p90 %.2e, max %.2e\n", median, nm[nm.size() * 9 / 10], nm.back());

    int bad = 0;
    if (mismatch != 0) {
        fprintf(stderr, "FAIL: device output differs from the spec model\n");
        bad = 1;
    }
    if (fail_dev_only != 0) {
        fprintf(stderr, "FAIL: %d trials fail the tolerance clearly beyond what the 8-bit operands explain\n", fail_dev_only);
        bad = 1;
    }
    if (std::abs(fail_dev - fail_q) > std::max(2, TRIALS / 200)) {
        fprintf(stderr, "FAIL: device failure count %d differs from the 8-bit-operand failure count %d\n", fail_dev, fail_q);
        bad = 1;
    }
    if (!(median < 1e-4)) {
        fprintf(stderr, "FAIL: median NMSE %.3e is not well below the tolerance\n", median);
        bad = 1;
    }
    ggml_backend_free(tsr);
    ggml_backend_free(cpu);
    printf("test-tsr-scalar-nmse: %s\n", bad ? "FAILED" : "OK");
    return bad;
}

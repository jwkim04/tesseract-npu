// test-tsr-moe.cpp — the MoE graph of llama.cpp's build_moe_ffn (softmax gating, normalised top-k) through the
// ggml-tesseract backend on whatever transport TSR_DEVICE selects (docs/design/06-software-verification.md "SW" §9.3,
// SW-60 to SW-65, SW-77). Two layers of two models:
//
// Gemma 4 26B-A4B (a per-expert output scale): the node sequence the pinned llama.cpp emits (build_moe_ffn in
// src/llama-graph.cpp and the router of src/models/gemma4.cpp): rms_norm, scale, mul, mul_mat (the router),
// soft_max, argsort_top_k, get_rows, sum_rows, clamp, div, mul_mat_id (gate-up), geglu_split of two views,
// mul_mat_id (down), repeat + get_rows + mul (the scale), mul (the weights), eight views and seven adds.
//
// Qwen3-30B-A3B (src/models/qwen3moe.cpp): the router reads the normed input as the experts do, the gate and up
// experts are two tensors (two mul_mat_id with the same ids and input), swiglu_split of their two results, no scale
// chain (the down result goes straight to the mul by the weights). The routing group is the same.
//
// The expert tensors are in a WEIGHTS buffer (placed with the per-channel expert stride, SW-60), the activations in
// compute buffers of ggml_gallocr (the allocator reuses the dead tensors, as llama.cpp's does).
//
// For T = 1, 2, 5, 17 and 70 tokens the device output is compared with the same graph on the ggml CPU backend:
// at least nine tenths of the tokens must agree (the others selected other experts: a router near a tie can differ
// by the Q8_0 router and the A32 activations; the inputs are drawn so that it is rare), and the layer output of the
// agreeing tokens be within NMSE 5e-3 (the
// CPU multiplies Q4_K by Q8_K and the device accumulates in another order; the router and the exponentials
// differ at rounding level). Which ops the device takes follows TSR_KINDS: the default takes the routing group
// (one MOE_ROUTE record) and the scale group, `m6,+MUL_MAT_ID` only the matrix products (the CPU selects),
// `all,-MOE_ROUTE` takes everything but the routing; TSR_TEST_ROUTE / TSR_TEST_MMID state what supports_op must answer
// for SOFT_MAX / ARGSORT and for MUL_MAT_ID. Exit code 0 on success.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-tesseract.h"

static int g_fail = 0;
#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);     \
            fprintf(stderr, __VA_ARGS__);                                         \
            fprintf(stderr, "\n");                                                \
            g_fail++;                                                             \
        }                                                                         \
    } while (0)

static ggml_backend_t g_tsr = nullptr, g_cpu = nullptr;
static std::mt19937 g_rng(77);

static constexpr int64_t N_EMBD = 256, N_EXPERT = 64, N_USED = 8, N_FF = 128;

static std::vector<float> randn(size_t n, float sd) {
    std::normal_distribution<float> u(0.0f, sd);
    std::vector<float> v(n);
    for (auto & x : v) x = u(g_rng);
    return v;
}

static std::vector<uint8_t> encode(ggml_type type, const std::vector<float> & f, int64_t rows, int64_t K) {
    std::vector<uint8_t> out(ggml_row_size(type, K) * rows);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), f.data(), out.size());
    } else {
        ggml_quantize_chunk(type, f.data(), out.data(), 0, rows, K, nullptr);
    }
    return out;
}

struct weights {
    ggml_tensor * gate_inp = nullptr, * gate_inp_s = nullptr, * gate_up = nullptr, * down = nullptr, * down_s = nullptr;
    ggml_tensor * q3_inp = nullptr, * q3_gate = nullptr, * q3_up = nullptr;      // Qwen3 layer: router, separate gate and up experts
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx2 = nullptr;          // the scale table in host memory (Gemma 4's: its loader probe has no kind)
    ggml_backend_buffer_t buf2 = nullptr;
};

// The layer's weights in backend `be`: data from the fixed-seed host copies below.
struct host_weights {
    std::vector<uint8_t> gate_inp, gate_up, down, q3_inp, q3_gate, q3_up;
    std::vector<float> gate_inp_s, down_s;
};

static host_weights make_host_weights() {
    host_weights h;
    // router weights ~ N(0, 4): logits of standard deviation about 4 (the rms-normalised input has elements of 1 / 16), so the
    // gap between the 8th and 9th of 64 is far above the device router's error
    h.gate_inp = encode(GGML_TYPE_F32, randn(N_EMBD * N_EXPERT, 4.0f), N_EXPERT, N_EMBD);
    h.gate_up = encode(GGML_TYPE_Q4_K, randn((size_t) N_EMBD * 2 * N_FF * N_EXPERT, 0.1f), 2 * N_FF * N_EXPERT, N_EMBD);
    h.down = encode(GGML_TYPE_Q8_0, randn((size_t) N_FF * N_EMBD * N_EXPERT, 0.1f), N_EMBD * N_EXPERT, N_FF);
    // Qwen3: the router reads the rms-normed input (elements of 1, not 1 / 16): the same logit spread with weights of 1 / 16 of the size
    h.q3_inp = encode(GGML_TYPE_F32, randn(N_EMBD * N_EXPERT, 0.25f), N_EXPERT, N_EMBD);
    h.q3_gate = encode(GGML_TYPE_Q4_K, randn((size_t) N_EMBD * N_FF * N_EXPERT, 0.1f), N_FF * N_EXPERT, N_EMBD);
    h.q3_up = encode(GGML_TYPE_Q4_K, randn((size_t) N_EMBD * N_FF * N_EXPERT, 0.1f), N_FF * N_EXPERT, N_EMBD);
    h.gate_inp_s.assign(N_EMBD, 1.0f);
    for (auto & x : h.gate_inp_s) x = 0.5f + (float) (g_rng() % 1000) / 1000.0f;
    h.down_s.assign(N_EXPERT, 1.0f);
    for (auto & x : h.down_s) x = 0.5f + (float) (g_rng() % 1000) / 500.0f;
    return h;
}

static weights make_weights(ggml_backend_t be, const host_weights & h, bool weights_usage, bool scale_on_cpu = false) {
    weights w;
    ggml_init_params ip = { ggml_tensor_overhead() * 12, nullptr, true };
    w.ctx = ggml_init(ip);
    w.gate_inp = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_EXPERT);
    w.gate_inp_s = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EMBD);
    w.gate_up = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q4_K, N_EMBD, 2 * N_FF, N_EXPERT);
    w.down = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q8_0, N_FF, N_EMBD, N_EXPERT);
    w.q3_inp = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_EXPERT);
    w.q3_gate = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERT);
    w.q3_up = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERT);
    if (scale_on_cpu) {
        w.ctx2 = ggml_init(ip);
        w.down_s = ggml_new_tensor_1d(w.ctx2, GGML_TYPE_F32, N_EXPERT);
    } else {
        w.down_s = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EXPERT);
    }
    ggml_set_name(w.gate_inp, "blk.0.ffn_gate_inp.weight");
    ggml_set_name(w.gate_inp_s, "blk.0.ffn_gate_inp.scale");
    ggml_set_name(w.gate_up, "blk.0.ffn_gate_up_exps.weight");
    ggml_set_name(w.down, "blk.0.ffn_down_exps.weight");
    ggml_set_name(w.down_s, "blk.0.ffn_down_exps.scale");
    ggml_set_name(w.q3_inp, "blk.1.ffn_gate_inp.weight");
    ggml_set_name(w.q3_gate, "blk.1.ffn_gate_exps.weight");
    ggml_set_name(w.q3_up, "blk.1.ffn_up_exps.weight");
    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, be);
    if (weights_usage) {
        ggml_backend_buffer_set_usage(w.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (scale_on_cpu) {
        w.buf2 = ggml_backend_alloc_ctx_tensors(w.ctx2, g_cpu);
    }
    ggml_backend_tensor_set(w.gate_inp, h.gate_inp.data(), 0, h.gate_inp.size());
    ggml_backend_tensor_set(w.gate_inp_s, h.gate_inp_s.data(), 0, h.gate_inp_s.size() * 4);
    ggml_backend_tensor_set(w.gate_up, h.gate_up.data(), 0, h.gate_up.size());
    ggml_backend_tensor_set(w.down, h.down.data(), 0, h.down.size());
    ggml_backend_tensor_set(w.down_s, h.down_s.data(), 0, h.down_s.size() * 4);
    ggml_backend_tensor_set(w.q3_inp, h.q3_inp.data(), 0, h.q3_inp.size());
    ggml_backend_tensor_set(w.q3_gate, h.q3_gate.data(), 0, h.q3_gate.size());
    ggml_backend_tensor_set(w.q3_up, h.q3_up.data(), 0, h.q3_up.size());
    return w;
}

struct layer {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * x = nullptr, * out = nullptr, * ids = nullptr, * probs = nullptr, * logits = nullptr, * wnorm = nullptr;
    std::vector<ggml_tensor *> chain;       // the routing and scale chain ops
};

// llama.cpp's build_moe_ffn (norm_w, softmax gating) in Gemma 4's form (GELU, one gate-up tensor, down_exps_s, the router
// behind rms_norm / scale / mul) or, with `qwen3`, in Qwen3's (SILU, gate and up tensors, no scale, the router on the normed input).
static layer build_layer(const weights & w, int64_t T, bool qwen3 = false) {
    layer l;
    ggml_init_params ip = { ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(256, false), nullptr, true };
    l.ctx = ggml_init(ip);
    ggml_context * ctx = l.ctx;
    l.gf = ggml_new_graph_custom(ctx, 256, false);
    l.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, T);
    ggml_set_name(l.x, "x");
    ggml_set_input(l.x);
    ggml_tensor * tmp = ggml_rms_norm(ctx, l.x, 1e-6f);
    if (qwen3) {
        tmp = ggml_mul(ctx, tmp, w.gate_inp_s);             // the ffn_norm weight
    } else {
        tmp = ggml_scale(ctx, tmp, 1.0f / std::sqrt((float) N_EMBD));
        tmp = ggml_mul(ctx, tmp, w.gate_inp_s);
    }
    ggml_tensor * logits = ggml_mul_mat(ctx, qwen3 ? w.q3_inp : w.gate_inp, tmp);
    ggml_set_name(logits, "ffn_moe_logits");
    l.logits = logits;
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    ggml_set_name(probs, "ffn_moe_probs");
    l.chain.push_back(probs);
    ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, N_USED);
    ggml_set_name(selected, "ffn_moe_topk");
    l.ids = selected;
    l.chain.push_back(selected->src[0]);
    probs = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, T);
    ggml_tensor * weights_ = ggml_get_rows(ctx, probs, selected);
    l.chain.push_back(weights_);
    weights_ = ggml_reshape_2d(ctx, weights_, N_USED, T);
    ggml_tensor * sum = ggml_sum_rows(ctx, weights_);
    l.chain.push_back(sum);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    l.chain.push_back(sum);
    weights_ = ggml_div(ctx, weights_, sum);
    l.chain.push_back(weights_);
    weights_ = ggml_reshape_3d(ctx, weights_, 1, N_USED, T);
    l.wnorm = weights_;
    ggml_build_forward_expand(l.gf, weights_);
    ggml_tensor * cur = qwen3 ? tmp : ggml_rms_norm(ctx, l.x, 1e-6f);
    cur = ggml_reshape_3d(ctx, cur, N_EMBD, 1, T);
    if (qwen3) {
        ggml_tensor * up = ggml_mul_mat_id(ctx, w.q3_up, cur, selected);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, w.q3_gate, cur, selected);
        cur = ggml_swiglu_split(ctx, gate, up);
    } else {
        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, w.gate_up, cur, selected);
        ggml_tensor * gate = ggml_view_3d(ctx, gate_up, N_FF, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor * up = ggml_view_3d(ctx, gate_up, N_FF, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2],
                                        N_FF * gate_up->nb[0]);
        cur = ggml_geglu_split(ctx, gate, up);
    }
    ggml_tensor * experts = ggml_mul_mat_id(ctx, w.down, cur, selected);
    if (!qwen3) {
        ggml_tensor * s = ggml_reshape_3d(ctx, w.down_s, 1, N_EXPERT, 1);
        s = ggml_repeat_4d(ctx, s, 1, N_EXPERT, T, 1);
        l.chain.push_back(s);
        s = ggml_get_rows(ctx, s, selected);
        l.chain.push_back(s);
        experts = ggml_mul(ctx, experts, s);
    }
    experts = ggml_mul(ctx, experts, weights_);
    ggml_build_forward_expand(l.gf, experts);
    ggml_tensor * moe = ggml_view_2d(ctx, experts, N_EMBD, T, experts->nb[2], 0);
    ggml_build_forward_expand(l.gf, moe);
    for (int64_t i = 1; i < N_USED; ++i) {
        ggml_tensor * v = ggml_view_2d(ctx, experts, N_EMBD, T, experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(l.gf, v);
        moe = ggml_add(ctx, moe, v);
        ggml_build_forward_expand(l.gf, moe);
    }
    l.out = ggml_rms_norm(ctx, moe, 1e-6f);
    ggml_set_name(l.out, "out");
    ggml_set_output(l.out);
    ggml_build_forward_expand(l.gf, l.out);
    return l;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den == 0 ? num : num / den;
}

struct result {
    std::vector<float> out;
};

// The graph on the scheduler: `dev` set: the device first and the CPU for what it does not take (as llama.cpp's loader
// does); else the CPU alone.
static result run_on(bool dev, const weights & w, int64_t T, const std::vector<float> & x, bool qwen3) {
    layer l = build_layer(w, T, qwen3);
    ggml_backend_t bes[2] = { g_tsr, g_cpu };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(g_tsr), ggml_backend_get_default_buffer_type(g_cpu) };
    ggml_backend_sched_t sched = dev ? ggml_backend_sched_new(bes, bufts, 2, 256, false, false)
                                     : ggml_backend_sched_new(bes + 1, bufts + 1, 1, 256, false, false);
    CHECK(ggml_backend_sched_alloc_graph(sched, l.gf), "alloc_graph");
    ggml_backend_tensor_set(l.x, x.data(), 0, x.size() * 4);
    CHECK(ggml_backend_sched_graph_compute(sched, l.gf) == GGML_STATUS_SUCCESS, "compute");
    result r;
    r.out.resize(N_EMBD * T);
    ggml_backend_tensor_get(l.out, r.out.data(), 0, r.out.size() * 4);
    ggml_backend_sched_free(sched);
    ggml_free(l.ctx);
    return r;
}

// Tokens whose output differs from the CPU's by more than this (relative squared error) selected other experts: the
// device's router (Q8_0 weights, A32 activations) and exponentials differ at rounding level, so a token whose
// 8th and 9th probabilities are closer than that may pick the other one. The inputs are drawn so that it is rare.
static constexpr double OTHER_EXPERTS = 0.02;

static void run_case(const weights & wd, const weights & wc, int64_t T, bool qwen3 = false) {
    const std::vector<float> x = randn(N_EMBD * T, 1.0f);
    const result d = run_on(true, wd, T, x, qwen3);
    const result c = run_on(false, wc, T, x, qwen3);
    int same = 0;
    double worst = 0;
    std::vector<float> od, oc;
    for (int64_t t = 0; t < T; ++t) {
        const std::vector<float> a(d.out.begin() + t * N_EMBD, d.out.begin() + (t + 1) * N_EMBD);
        const std::vector<float> b(c.out.begin() + t * N_EMBD, c.out.begin() + (t + 1) * N_EMBD);
        const double e = nmse(a, b);
        if (e > OTHER_EXPERTS) continue;
        ++same;
        worst = std::max(worst, e);
        od.insert(od.end(), a.begin(), a.end());
        oc.insert(oc.end(), b.begin(), b.end());
    }
    const double e = od.empty() ? 1.0 : nmse(od, oc);
    printf("  T=%-3lld: %d of %lld tokens agree with the CPU (other experts otherwise), NMSE %.3g, worst token %.3g\n", (long long) T,
           same, (long long) T, e, worst);
    CHECK(same * 10 >= T * 9, "T=%lld: %lld of %lld tokens differ from the CPU by more than %g", (long long) T, (long long) (T - same),
          (long long) T, OTHER_EXPERTS);
    CHECK(e <= 5e-3, "T=%lld: NMSE %.3g vs CPU", (long long) T, e);
}

int main() {
    g_tsr = ggml_backend_tesseract_init();
    if (g_tsr == nullptr) {
        fprintf(stderr, "test-tsr-moe: no Tesseract backend (TSR_DEVICE=%s)\n", getenv("TSR_DEVICE") ? getenv("TSR_DEVICE") : "");
        return 1;
    }
    g_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(g_cpu != nullptr, "no CPU backend");
    printf("device: %s; TSR_KINDS=%s\n", ggml_backend_dev_description(ggml_backend_get_device(g_tsr)),
           getenv("TSR_KINDS") ? getenv("TSR_KINDS") : "(unset)");
    const host_weights h = make_host_weights();
    const weights wd = make_weights(g_tsr, h, true);
    const weights wc = make_weights(g_cpu, h, false);

    // supports_op follows TSR_KINDS: TSR_TEST_ROUTE / TSR_TEST_MMID (1 or 0) say what the run expects (CMake sets them)
    {
        layer l = build_layer(wd, 3);
        const char * er = getenv("TSR_TEST_ROUTE"), * em = getenv("TSR_TEST_MMID");
        if (er != nullptr) {
            for (ggml_tensor * t : { l.chain[0], l.chain[1] }) {                  // SOFT_MAX, ARGSORT
                CHECK(ggml_backend_supports_op(g_tsr, t) == (atoi(er) != 0), "supports_op(%s), expected %s", ggml_op_name(t->op), er);
            }
        }
        if (em != nullptr) {
            for (int i = 0; i < ggml_graph_n_nodes(l.gf); ++i) {
                if (ggml_graph_node(l.gf, i)->op == GGML_OP_MUL_MAT_ID) {
                    CHECK(ggml_backend_supports_op(g_tsr, ggml_graph_node(l.gf, i)) == (atoi(em) != 0), "supports_op(MUL_MAT_ID), expected %s", em);
                }
            }
        }
        ggml_free(l.ctx);
    }
    for (int64_t T : { 1, 2, 5, 17, 70 }) {
        run_case(wd, wc, T);
    }
    // Qwen3-30B-A3B's layer: separate gate and up experts, SWIGLU, no scale chain (SW-77)
    {
        layer l = build_layer(wd, 3, true);
        const char * er = getenv("TSR_TEST_ROUTE"), * em = getenv("TSR_TEST_MMID");
        if (er != nullptr) {
            for (ggml_tensor * t : { l.chain[0], l.chain[1] }) {
                CHECK(ggml_backend_supports_op(g_tsr, t) == (atoi(er) != 0), "qwen3: supports_op(%s), expected %s", ggml_op_name(t->op), er);
            }
        }
        if (em != nullptr) {
            int n_mmid = 0;
            for (int i = 0; i < ggml_graph_n_nodes(l.gf); ++i) {
                if (ggml_graph_node(l.gf, i)->op == GGML_OP_MUL_MAT_ID) {
                    n_mmid++;
                    CHECK(ggml_backend_supports_op(g_tsr, ggml_graph_node(l.gf, i)) == (atoi(em) != 0), "qwen3: supports_op(MUL_MAT_ID), expected %s", em);
                }
            }
            CHECK(n_mmid == 3, "qwen3 layer: %d MUL_MAT_ID nodes", n_mmid);
        }
        ggml_free(l.ctx);
        printf("qwen3 layer (separate gate / up experts, SWIGLU, no scale):\n");
        // up to one tile (64 tokens): with several tiles the allocator's placement of this graph makes the route's outputs
        // overlap a later tile's logits, which the backend refuses (SW-64; the several-tile case is not part of this test)
        for (int64_t T : { 1, 2, 5, 17, 64 }) {
            run_case(wd, wc, T, true);
        }
    }
    // the scale table in host memory, as llama.cpp's loader places it for Gemma 4: the scheduler copies the reshaped view
    // to the device (a leaf of its own), and the scale group must still fuse (without it graph_compute aborts)
    {
        const weights wh = make_weights(g_tsr, h, true, true);
        printf("scale table in host memory:\n");
        for (int64_t T : { 1, 17 }) {
            run_case(wh, wc, T);
        }
        ggml_backend_buffer_free(wh.buf);
        ggml_backend_buffer_free(wh.buf2);
        ggml_free(wh.ctx);
        ggml_free(wh.ctx2);
    }
    ggml_backend_buffer_free(wd.buf);
    ggml_backend_buffer_free(wc.buf);
    ggml_free(wd.ctx);
    ggml_free(wc.ctx);
    ggml_backend_free(g_cpu);
    ggml_backend_free(g_tsr);
    if (g_fail) {
        printf("test-tsr-moe: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-moe: OK\n");
    return 0;
}

// test_tsr_rt_graph.cpp — the graph API of tsr::rt::device (graph_create / graph_run / graph_run_ref) on the transport TSR_DEVICE
// selects (sim, mock, ...). Registered as rt_graph_sim, rt_graph_sim_tmax4, rt_graph_sim_tmax8 and rt_graph_mock.
//
// Three oracles, independent of one another:
//   - graph_run_ref: the op lists of the plan executed on the host with tsr::ref::opl_exec (the executor's dispatch);
//   - the per-kind references called directly here with argument words written by hand on an image of this file's own
//     (rmsnorm_ref, rope_ref, set_rows_kv_ref + flash_attn_ref) and device::linear_ref (libtsr_ref's QACT / dot rows) for
//     MUL_MAT; ADD, MUL, SCALE, CPY against host f32 math, GLU and UNARY against host math with a tolerance (the vector
//     unit's functions are a few ulp);
//   - the device (graph_run).
// graph_run must equal graph_run_ref bit for bit in every case; the direct references must equal both.
//
// Cases: each op alone at T in {1, 2, 5, 8, 17} (and more); N parts, K parts, an addend with K parts (falls to ADD), Q8_0 and
// Q4_K, strided arguments; the mask forms of attn; a Qwen2.5-0.5B decoder layer and three layers as one program, the OPL cache
// and the plan cache, a second T reusing the planes; the limits and their error texts; programs that are malformed or that the
// device cannot run; const_free / graph_free returning memory; two threads; negative controls (the comparison must fail for a
// wrong inv_n, the other RoPE pairing, a missing FA_NEW where the device is sensitive to it).
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "ggml-quants.h"   // quantize_row_q8_0_ref (ggml/src)
#include "mulmat_ref.hpp"
#include "rmsnorm_ref.hpp"
#include "rope_ref.hpp"
#include "kvfa_ref.hpp"
#include "tsr-kinds.h"
#include "tsr-wlayout.h"
#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_rt.h"
#include "tsr/opl.hpp"

using namespace tsr::rt;
namespace opl = tsr::opl;

static int g_fail = 0;
static int g_cases = 0;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);     \
            fprintf(stderr, __VA_ARGS__);                                         \
            fprintf(stderr, "\n");                                                \
            g_fail++;                                                             \
        }                                                                         \
    } while (0)

static std::mt19937 g_rng(20241113);
static float rnd(float scale = 1.0f) { return std::uniform_real_distribution<float>(-scale, scale)(g_rng); }
static float rnd_n(float sigma) { return std::normal_distribution<float>(0.0f, sigma)(g_rng); }
static uint32_t rnd_u(uint32_t n) { return (uint32_t) (g_rng() % n); }
static uint32_t f2b(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

static std::vector<float> rnd_vec(size_t n, float scale) {
    std::vector<float> v(n);
    for (float & x : v) x = rnd(scale);
    return v;
}

static const uint32_t TS[] = { 1, 2, 5, 8, 17 };

static device * D = nullptr;
static tsr_kenv g_kenv;

// ---------------------------------------------------------------------------------------------
// Program builder
// ---------------------------------------------------------------------------------------------
struct PB {
    graph_desc g;
    int val(gv k, uint32_t n, bool out = false, const_handle h = 0, uint32_t rows = 1) {
        graph_value v;
        v.kind = k;
        v.n = n;
        v.out = out;
        v.handle = h;
        v.rows = rows;
        g.values.push_back(v);
        return (int) g.values.size() - 1;
    }
    int in(uint32_t n) { return val(gv::in, n); }
    int tmp(uint32_t n, bool out = false) { return val(gv::tmp, n, out); }
    int cst(const_handle h, uint32_t n, uint32_t rows = 1) { return val(gv::konst, n, false, h, rows); }
    int pos() { return val(gv::pos, 0); }
    int mask(uint32_t cols = 0) {
        const int m = val(gv::mask, 0);
        g.values[(size_t) m].cols = cols;
        return m;
    }
    graph_op & op(gop k, int dst) {
        graph_op o;
        o.op = k;
        o.dst = dst;
        o.tag = 1000 + (int) g.ops.size();
        g.ops.push_back(o);
        return g.ops.back();
    }
    void mul_mat(weight_handle w, int x, int dst, int addend = -1) {
        graph_op & o = op(gop::mul_mat, dst);
        o.w = w; o.x = x; o.addend = addend;
    }
    void ew(gop k, int a, int b, int dst) {
        graph_op & o = op(k, dst);
        o.a = a; o.b = b;
    }
    void rms(int x, int dst, int w, float eps) {
        graph_op & o = op(gop::rms_norm, dst);
        o.x = x; o.weight = w; o.eps = eps;
    }
    void rope(int x, int dst, uint32_t nh, uint32_t hd, uint32_t nd, int table, int pos, float factor = 1.0f) {
        graph_op & o = op(gop::rope, dst);
        o.x = x; o.n_heads = nh; o.head_dim = hd; o.n_dims = nd; o.table = table; o.pos = pos; o.factor = factor;
    }
    void attn(int q, int k, int v, int dst, uint32_t nh, uint32_t nkv, uint32_t d, float scale, int mask, int kv_slot = -1) {
        graph_op & o = op(gop::attn, dst);
        o.q = q; o.k = k; o.v = v; o.n_head = nh; o.n_head_kv = nkv; o.d = d; o.scale = scale; o.mask = mask; o.kv = kv_slot;
    }
};

// An argument of a run, with the data that backs it.
struct arg_t {
    gv kind;
    std::vector<float> f;        // in, mask
    std::vector<int32_t> p;      // pos
};

struct result_t {
    std::vector<std::vector<float>> dev, ref;
};

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b, size_t * first = nullptr) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], 4) != 0) {
            if (first) *first = i;
            return false;
        }
    }
    return true;
}

// graph_run and graph_run_ref on the same arguments; checks they agree bit for bit; the device's outputs in *res. pad: extra floats
// between rows of the inputs and outputs (strided arguments).
static bool run_cmp(graph_handle h, uint32_t T, const std::vector<arg_t> & args, const char * what, result_t * res = nullptr, uint32_t pad = 0,
                    const graph_state & st = graph_state()) {
    g_cases++;
    const graph_shape sh = D->graph_shape_of(h);
    graph_inputs gi;
    std::vector<std::vector<float>> padded(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        graph_arg a;
        if (sh.arg_kind[i] == gv::in) {
            const uint32_t n = sh.arg_n[i];
            padded[i].assign((size_t) T * (n + pad), 12345.0f);
            for (uint32_t t = 0; t < T; ++t) memcpy(&padded[i][(size_t) t * (n + pad)], &args[i].f[(size_t) t * n], (size_t) n * 4);
            a.data = padded[i].data();
            a.stride = pad != 0 ? n + pad : 0;
        } else if (sh.arg_kind[i] == gv::pos) {
            a.data = args[i].p.data();
        } else {
            a.data = args[i].f.data();
        }
        gi.args.push_back(a);
    }
    result_t r;
    r.dev.resize(sh.out_n.size());
    r.ref.resize(sh.out_n.size());
    std::vector<std::vector<float>> dbuf(sh.out_n.size()), rbuf(sh.out_n.size());
    graph_outputs od, orf;
    for (size_t i = 0; i < sh.out_n.size(); ++i) {
        const uint32_t n = sh.out_n[i];
        dbuf[i].assign((size_t) T * (n + pad), -7.0f);
        rbuf[i].assign((size_t) T * (n + pad), -7.0f);
        od.outs.push_back({ dbuf[i].data(), pad != 0 ? (size_t) n + pad : 0 });
        orf.outs.push_back({ rbuf[i].data(), pad != 0 ? (size_t) n + pad : 0 });
    }
    bool ok = true;
    try {
        D->graph_run(h, T, gi, od, st);
        D->graph_run_ref(h, T, gi, orf, st);
    } catch (const std::exception & e) {
        CHECK(false, "%s: %s", what, e.what());
        return false;
    }
    for (size_t i = 0; i < sh.out_n.size(); ++i) {
        const uint32_t n = sh.out_n[i];
        r.dev[i].resize((size_t) T * n);
        r.ref[i].resize((size_t) T * n);
        for (uint32_t t = 0; t < T; ++t) {
            memcpy(&r.dev[i][(size_t) t * n], &dbuf[i][(size_t) t * (n + pad)], (size_t) n * 4);
            memcpy(&r.ref[i][(size_t) t * n], &rbuf[i][(size_t) t * (n + pad)], (size_t) n * 4);
            // the padding between rows must be untouched
            for (uint32_t j = 0; j < pad; ++j) {
                CHECK(dbuf[i][(size_t) t * (n + pad) + n + j] == -7.0f, "%s: the output padding was written", what);
            }
        }
        size_t first = 0;
        const bool same = same_bits(r.dev[i], r.ref[i], &first);
        CHECK(same, "%s: output %zu: graph_run differs from graph_run_ref (first at %zu: %g vs %g)", what, i, first,
              (double) r.dev[i][first], (double) r.ref[i][first]);
        ok = ok && same;
    }
    if (res) *res = std::move(r);
    return ok;
}

static graph_handle create(const graph_desc & g, const char * what) {
    try {
        const char * why = D->graph_unsupported(g);
        if (why != nullptr) {
            CHECK(false, "%s: graph_unsupported: %s", what, why);
            return 0;
        }
        return D->graph_create(g);
    } catch (const std::exception & e) {
        CHECK(false, "%s: graph_create: %s", what, e.what());
        return 0;
    }
}

static const_handle up_const(const std::vector<float> & v, uint32_t rows, uint32_t n) { return D->const_upload(v.data(), rows, n); }

static arg_t in_arg(const std::vector<float> & v) { arg_t a; a.kind = gv::in; a.f = v; return a; }
static arg_t pos_arg(const std::vector<int32_t> & p) { arg_t a; a.kind = gv::pos; a.p = p; return a; }
static arg_t mask_arg(const std::vector<float> & m) { arg_t a; a.kind = gv::mask; a.f = m; return a; }

// ---------------------------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------------------------
struct W {
    weight_handle h = 0;
    uint32_t N = 0, K = 0;
    dtype t = dtype::f32;
};

static W up_weight(uint32_t N, uint32_t K, float sigma, const weight_options & o = weight_options()) {
    std::vector<float> w((size_t) N * K);
    for (float & x : w) x = rnd_n(sigma);
    W r;
    r.N = N; r.K = K;
    r.h = D->upload(dtype::f32, w.data(), N, K, 0, o);
    return r;
}

static W up_q4k(uint32_t N, uint32_t K) {
    const size_t rb = dtype_row_bytes(dtype::q4_k, K);
    std::vector<uint8_t> blocks((size_t) N * rb);
    for (size_t o = 0; o + sizeof(tsr_block_q4_K) <= blocks.size(); o += sizeof(tsr_block_q4_K)) {
        tsr_block_q4_K q;
        q.d = (uint16_t) (0x2400u + rnd_u(0x200u));
        q.dmin = (uint16_t) (0x2000u + rnd_u(0x200u));
        for (auto & s : q.scales) s = (uint8_t) rnd_u(256u);
        for (auto & s : q.qs) s = (uint8_t) rnd_u(256u);
        memcpy(blocks.data() + o, &q, sizeof(q));
    }
    W r;
    r.N = N; r.K = K; r.t = dtype::q4_k;
    r.h = D->upload(dtype::q4_k, blocks.data(), N, K);
    return r;
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT
// ---------------------------------------------------------------------------------------------
enum { MM_PLAIN = 0, MM_BIAS = 1, MM_RESID = 2 };

static void test_mul_mat(const W & w, uint32_t T, int mode, const char * label, uint32_t pad = 0) {
    char what[160];
    snprintf(what, sizeof(what), "mul_mat %s [%u x %u] T %u mode %d pad %u", label, w.N, w.K, T, mode, pad);
    PB pb;
    const int x = pb.in(w.K);
    int add = -1;
    std::vector<float> bias, resid;
    if (mode == MM_BIAS) {
        bias = rnd_vec(w.N, 1.0f);
        add = pb.cst(up_const(bias, 1, w.N), w.N);
    } else if (mode == MM_RESID) {
        add = pb.in(w.N);
        resid = rnd_vec((size_t) T * w.N, 1.0f);
    }
    const int y = pb.tmp(w.N, true);
    pb.mul_mat(w.h, x, y, add);
    pb.g.kv_size = 0;
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    std::vector<arg_t> args;
    const std::vector<float> xv = rnd_vec((size_t) T * w.K, 2.0f);
    args.push_back(in_arg(xv));
    if (mode == MM_RESID) args.push_back(in_arg(resid));
    result_t r;
    if (run_cmp(h, T, args, what, &r, pad)) {
        // the independent model: linear_ref (libtsr_ref on the weight rows read back), then the host f32 add
        std::vector<float> want((size_t) T * w.N);
        D->linear_ref(w.h, xv.data(), T, want.data());
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t n = 0; n < w.N; ++n) {
                float & v = want[(size_t) t * w.N + n];
                if (mode == MM_BIAS) v = v + bias[n];
                if (mode == MM_RESID) v = v + resid[(size_t) t * w.N + n];
            }
        }
        size_t first = 0;
        CHECK(same_bits(r.dev[0], want, &first), "%s: differs from linear_ref (+ addend) at %zu: %g vs %g", what, first,
              (double) r.dev[0][first], (double) want[first]);
    }
    D->graph_free(h);
}

static void test_mul_mats() {
    printf("mul_mat\n");
    const W a = up_weight(128, 64, 0.1f), b = up_weight(300, 96, 0.1f), c = up_weight(96, 256, 0.1f);
    const W q4 = up_q4k(64, 512);
    weight_options ok;
    ok.max_part_k = 64;                       // K parts: 256 = 4 x 64
    const W kp = up_weight(96, 256, 0.1f, ok);
    CHECK(D->info(kp.h).k_parts == 4, "K parts");
    weight_options on;
    on.max_part_rows = 48;                    // N parts: 96 = 2 x 48 (multiples of 16)
    const W np = up_weight(96, 128, 0.1f, on);
    CHECK(D->info(np.h).n_parts == 2, "N parts");
    weight_options onk;
    onk.max_part_rows = 32;
    onk.max_part_k = 64;
    const W npkp = up_weight(96, 192, 0.1f, onk);
    for (uint32_t T : TS) {
        test_mul_mat(a, T, MM_PLAIN, "q8_0");
        test_mul_mat(b, T, MM_PLAIN, "q8_0");
        test_mul_mat(c, T, MM_BIAS, "q8_0 + bias");
        test_mul_mat(a, T, MM_RESID, "q8_0 + residual");
        test_mul_mat(q4, T, MM_PLAIN, "q4_k");
        test_mul_mat(q4, T, MM_BIAS, "q4_k + bias");
        test_mul_mat(kp, T, MM_PLAIN, "4 K parts");
        test_mul_mat(kp, T, MM_BIAS, "4 K parts + bias (ADD)");
        test_mul_mat(kp, T, MM_RESID, "4 K parts + residual (ADD)");
        test_mul_mat(np, T, MM_PLAIN, "2 N parts");
        test_mul_mat(np, T, MM_BIAS, "2 N parts + bias");
        test_mul_mat(np, T, MM_RESID, "2 N parts + residual");
        test_mul_mat(npkp, T, MM_RESID, "2 N x 3 K parts + residual");
    }
    test_mul_mat(a, 5, MM_PLAIN, "q8_0", 3);          // strided inputs and outputs
    test_mul_mat(c, 5, MM_RESID, "q8_0 + residual", 7);
}

// ---------------------------------------------------------------------------------------------
// Element-wise ops
// ---------------------------------------------------------------------------------------------
static double h_silu(double x) { return x / (1.0 + std::exp(-x)); }
static double h_gelu(double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }

static bool close_to(const std::vector<float> & got, const std::vector<double> & want, double rtol, double atol, size_t * first) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (!(std::fabs((double) got[i] - want[i]) <= rtol * std::fabs(want[i]) + atol)) {
            *first = i;
            return false;
        }
    }
    return true;
}

static void test_ew(gop k, uint32_t n, uint32_t T, bool b_const) {
    char what[96];
    snprintf(what, sizeof(what), "%s n %u T %u b %s", gop_name(k), n, T, b_const ? "const row" : "value");
    PB pb;
    const int a = pb.in(n);
    std::vector<float> bv;
    int b;
    std::vector<arg_t> args;
    const std::vector<float> av = rnd_vec((size_t) T * n, 3.0f);
    args.push_back(in_arg(av));
    if (b_const) {
        const std::vector<float> row = rnd_vec(n, 3.0f);
        b = pb.cst(up_const(row, 1, n), n);
        for (uint32_t t = 0; t < T; ++t) bv.insert(bv.end(), row.begin(), row.end());
    } else {
        b = pb.in(n);
        bv = rnd_vec((size_t) T * n, 3.0f);
        args.push_back(in_arg(bv));
    }
    const int y = pb.tmp(n, true);
    pb.ew(k, a, b, y);
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    result_t r;
    if (run_cmp(h, T, args, what, &r)) {
        std::vector<float> want(av.size());
        for (size_t i = 0; i < want.size(); ++i) want[i] = k == gop::add ? av[i] + bv[i] : av[i] * bv[i];
        size_t first = 0;
        CHECK(same_bits(r.dev[0], want, &first), "%s: differs from host f32 math at %zu", what, first);
    }
    D->graph_free(h);
}

static void test_scale(uint32_t n, uint32_t T, float s, float bias) {
    char what[96];
    snprintf(what, sizeof(what), "scale n %u T %u s %g b %g", n, T, (double) s, (double) bias);
    PB pb;
    const int x = pb.in(n), y = pb.tmp(n, true);
    graph_op & o = pb.op(gop::scale, y);
    o.x = x; o.s = s; o.sb = bias;
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    const std::vector<float> xv = rnd_vec((size_t) T * n, 4.0f);
    result_t r;
    if (run_cmp(h, T, { in_arg(xv) }, what, &r)) {
        std::vector<float> want(xv.size());
        for (size_t i = 0; i < want.size(); ++i) {
            volatile float m = xv[i] * s;
            want[i] = bias != 0.0f ? (float) m + bias : (float) m;
        }
        size_t first = 0;
        CHECK(same_bits(r.dev[0], want, &first), "%s: differs from host math at %zu", what, first);
    }
    D->graph_free(h);
}

static void test_glu(uint32_t n, uint32_t T, uint32_t variant) {
    char what[96];
    snprintf(what, sizeof(what), "glu variant %u n %u T %u", variant, n, T);
    PB pb;
    const int g = pb.in(n), u = pb.in(n), y = pb.tmp(n, true);
    graph_op & o = pb.op(gop::glu, y);
    o.gate = g; o.up = u; o.variant = variant;
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    const std::vector<float> gv_ = rnd_vec((size_t) T * n, 4.0f), uv = rnd_vec((size_t) T * n, 2.0f);
    result_t r;
    if (run_cmp(h, T, { in_arg(gv_), in_arg(uv) }, what, &r)) {
        std::vector<double> want(gv_.size());
        for (size_t i = 0; i < want.size(); ++i) want[i] = (variant == 0 ? h_silu(gv_[i]) : h_gelu(gv_[i])) * (double) uv[i];
        size_t first = 0;
        CHECK(close_to(r.dev[0], want, 2e-3, 1e-4, &first), "%s: differs from host math at %zu (%g vs %g)", what, first, (double) r.dev[0][first], want[first]);
    }
    D->graph_free(h);
}

static void test_unary(uint32_t n, uint32_t T, uint32_t func) {
    char what[96];
    snprintf(what, sizeof(what), "unary func %u n %u T %u", func, n, T);
    PB pb;
    const int x = pb.in(n), y = pb.tmp(n, true);
    graph_op & o = pb.op(gop::unary, y);
    o.x = x; o.func = func;
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    const std::vector<float> xv = rnd_vec((size_t) T * n, func == 3 ? 6.0f : 8.0f);
    result_t r;
    if (run_cmp(h, T, { in_arg(xv) }, what, &r)) {
        std::vector<double> want(xv.size());
        for (size_t i = 0; i < want.size(); ++i) {
            const double v = xv[i];
            switch (func) {
                case 0: want[i] = std::tanh(v); break;
                case 2: want[i] = h_silu(v); break;
                case 3: want[i] = std::exp(v); break;
                case 4: want[i] = h_gelu(v); break;
                default: want[i] = 1.0 / (1.0 + std::exp(-v)); break;
            }
        }
        size_t first = 0;
        CHECK(close_to(r.dev[0], want, 2e-3, 1e-4, &first), "%s: differs from host math at %zu", what, first);
    }
    D->graph_free(h);
}

static void test_cpy(uint32_t n, uint32_t T) {
    char what[96];
    snprintf(what, sizeof(what), "cpy n %u T %u", n, T);
    PB pb;
    const int x = pb.in(n), y = pb.tmp(n, true);
    graph_op & o = pb.op(gop::cpy, y);
    o.x = x;
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    const std::vector<float> xv = rnd_vec((size_t) T * n, 5.0f);
    result_t r;
    if (run_cmp(h, T, { in_arg(xv) }, what, &r)) {
        size_t first = 0;
        CHECK(same_bits(r.dev[0], xv, &first), "%s: differs from the source at %zu", what, first);
    }
    D->graph_free(h);
}

static void test_ew_ops() {
    printf("add mul scale glu unary cpy\n");
    for (uint32_t T : TS) {
        for (gop k : { gop::add, gop::mul }) {
            for (bool c : { false, true }) {
                test_ew(k, 64, T, c);
                test_ew(k, 200, T, c);
            }
        }
        test_scale(96, T, 0.37f, 0.0f);
        test_scale(130, T, -1.5f, 0.25f);
        test_glu(64, T, 0);
        test_glu(256, T, 0);
        test_glu(96, T, 2);
        for (uint32_t f : { 0u, 2u, 3u, 4u, 5u }) {
            test_unary(64, T, f);
            test_unary(100, T, f);
        }
        test_cpy(64, T);
        test_cpy(150, T);
    }
}

// ---------------------------------------------------------------------------------------------
// The direct references' image (independent of the runtime's placement)
// ---------------------------------------------------------------------------------------------
static constexpr uint64_t IMG_BYTES = TSR_REGION_WEIGHT_OFF + (160ull << 20);

struct image_t {
    uint8_t * ch[TSR_NUM_CH_MAX] = {};
    uint32_t C = 1;
    uint64_t io = TSR_REGION_IO_OFF, aux = TSR_REGION_AUX_OFF, heap = TSR_REGION_WEIGHT_OFF;
    image_t(uint32_t num_ch) : C(num_ch) {
        for (uint32_t c = 0; c < C; ++c) {
            void * p = mmap(nullptr, IMG_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) { fprintf(stderr, "cannot map the host image\n"); exit(1); }
            ch[c] = (uint8_t *) p;
        }
    }
    ~image_t() { for (uint32_t c = 0; c < C; ++c) munmap(ch[c], IMG_BYTES); }
    tsr::ref::dram_view view() { return tsr::ref::dram_view{ ch, C, IMG_BYTES }; }
    uint64_t alloc_io(uint64_t n) { const uint64_t r = io; io += (n + 63u) / 64u * 64u + 64u; return r; }
    uint64_t alloc_aux(uint64_t n) { const uint64_t r = aux; aux += (n + 63u) / 64u * 64u + 64u; return r; }
    uint64_t alloc_heap(uint64_t n) { const uint64_t r = heap; heap += (n + 4095u) / 4096u * 4096u; return r; }
    // f32 rows [rows][n] at off, rows `stride` lines apart (padding zero); returns the handle
    uint32_t put_rows(uint64_t off, const std::vector<float> & v, uint32_t rows, uint32_t n, uint32_t stride) {
        for (uint32_t r = 0; r < rows; ++r) {
            memset(ch[0] + off + (uint64_t) r * stride * 64, 0, (size_t) stride * 64);
            memcpy(ch[0] + off + (uint64_t) r * stride * 64, &v[(size_t) r * n], (size_t) n * 4);
        }
        return tsr_dram_handle(0, off);
    }
    std::vector<float> rows_out(uint32_t h, uint32_t rows, uint32_t n, uint32_t stride) {
        std::vector<float> v((size_t) rows * n);
        const uint64_t off = (uint64_t) (h & 0x0FFFFFFFu) * 64;
        for (uint32_t r = 0; r < rows; ++r) memcpy(&v[(size_t) r * n], ch[0] + off + (uint64_t) r * stride * 64, (size_t) n * 4);
        return v;
    }
};

static uint32_t Lc(uint32_t n) { return (n + 15u) / 16u; }

// ---------------------------------------------------------------------------------------------
// RMS_NORM
// ---------------------------------------------------------------------------------------------
// The direct reference: words written here (inv_n = 1.0f / n as a host division).
static std::vector<float> rms_direct(const std::vector<float> & x, const std::vector<float> & wv, uint32_t n, uint32_t T, float eps, uint32_t inv_bits) {
    image_t img(D->config().num_ch);
    const uint32_t L = Lc(n);
    const uint32_t xh = img.put_rows(img.alloc_io((uint64_t) T * L * 64), x, T, n, L);
    const uint32_t yh = tsr_dram_handle(0, img.alloc_io((uint64_t) T * L * 64));
    uint32_t wh = opl::HANDLE_NONE;
    if (!wv.empty()) wh = img.put_rows(img.alloc_aux((uint64_t) L * 64), wv, 1, n, L);
    const uint32_t a[10] = { yh, L, xh, L, n, T, f2b(eps), inv_bits, wh, 1u };
    CHECK(tsr::ref::rmsnorm_ref(img.view(), a) == 0, "direct rmsnorm_ref");
    return img.rows_out(yh, T, n, L);
}

static void test_rms(uint32_t n, uint32_t T, bool weight) {
    char what[96];
    snprintf(what, sizeof(what), "rms_norm n %u T %u weight %d", n, T, (int) weight);
    PB pb;
    const int x = pb.in(n);
    std::vector<float> wv;
    int w = -1;
    if (weight) {
        wv = rnd_vec(n, 1.5f);
        w = pb.cst(up_const(wv, 1, n), n);
    }
    const int y = pb.tmp(n, true);
    pb.rms(x, y, w, 1e-6f);
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    const std::vector<float> xv = rnd_vec((size_t) T * n, 2.0f);
    result_t r;
    if (run_cmp(h, T, { in_arg(xv) }, what, &r)) {
        const std::vector<float> want = rms_direct(xv, wv, n, T, 1e-6f, f2b(1.0f / (float) n));
        size_t first = 0;
        CHECK(same_bits(r.dev[0], want, &first), "%s: differs from the direct reference at %zu", what, first);
        if (n == 896 && T == 5) {
            // negative control: a wrong inv_n (one ulp off 1 / n) must give another result than the device's
            const std::vector<float> bad = rms_direct(xv, wv, n, T, 1e-6f, f2b(1.0f / (float) n) + 1u);
            CHECK(!same_bits(r.dev[0], bad), "negative control: a wrong inv_n word was not caught by the comparison");
            printf("  negative control: wrong inv_n changes the result: ok\n");
        }
    }
    D->graph_free(h);
}

static void test_rms_ops() {
    printf("rms_norm\n");
    for (uint32_t T : TS) {
        test_rms(64, T, true);
        test_rms(128, T, false);
        test_rms(896, T, true);
        test_rms(1000, T, true);
    }
}

// ---------------------------------------------------------------------------------------------
// ROPE
// ---------------------------------------------------------------------------------------------
static std::vector<float> rope_direct(const std::vector<float> & x, const std::vector<float> & tab, const std::vector<int32_t> & pos, uint32_t nh,
                                      uint32_t hd, uint32_t nd, uint32_t T, float factor, bool neox) {
    image_t img(D->config().num_ch);
    const uint32_t H = nh * hd / 16u;
    const uint32_t xh = img.put_rows(img.alloc_io((uint64_t) T * H * 64), x, T, nh * hd, H);
    const uint32_t yh = tsr_dram_handle(0, img.alloc_io((uint64_t) T * H * 64));
    const uint32_t th = img.put_rows(img.alloc_aux((uint64_t) Lc(nd / 2) * 64), tab, 1, nd / 2, Lc(nd / 2));
    const uint32_t a[14] = { yh, H, xh, H, 0u, th, 0u, f2b(factor), f2b(1.0f), hd, nd, nh, T,
                             (neox ? opl::rope::MODE_NEOX : 0u) | opl::rope::MODE_ATAB };
    std::vector<uint8_t> pre(64u * ((T * 4u + 63u) / 64u), 0);
    memcpy(pre.data(), pos.data(), (size_t) T * 4);
    CHECK(tsr::ref::rope_ref(img.view(), a, pre.data(), (uint32_t) pre.size()) == 0, "direct rope_ref");
    return img.rows_out(yh, T, nh * hd, H);
}

static void test_rope(uint32_t nh, uint32_t hd, uint32_t nd, uint32_t T, float factor) {
    char what[96];
    snprintf(what, sizeof(what), "rope %u x %u (%u rotated) T %u factor %g", nh, hd, nd, T, (double) factor);
    PB pb;
    std::vector<float> tab(nd / 2);
    for (uint32_t i = 0; i < nd / 2; ++i) tab[i] = (float) std::pow(1000000.0, -2.0 * (double) i / (double) nd);
    const int x = pb.in(nh * hd), t = pb.cst(up_const(tab, 1, nd / 2), nd / 2), p = pb.pos(), y = pb.tmp(nh * hd, true);
    pb.rope(x, y, nh, hd, nd, t, p, factor);
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    const std::vector<float> xv = rnd_vec((size_t) T * nh * hd, 2.0f);
    std::vector<int32_t> pos(T);
    for (uint32_t i = 0; i < T; ++i) pos[i] = (int32_t) (rnd_u(1500u) + i);
    result_t r;
    if (run_cmp(h, T, { in_arg(xv), pos_arg(pos) }, what, &r)) {
        const std::vector<float> want = rope_direct(xv, tab, pos, nh, hd, nd, T, factor, true);
        size_t first = 0;
        CHECK(same_bits(r.dev[0], want, &first), "%s: differs from the direct reference at %zu", what, first);
        if (nh == 14 && T == 5) {
            // negative control: the other pairing (2 i, 2 i + 1) must give another result
            const std::vector<float> bad = rope_direct(xv, tab, pos, nh, hd, nd, T, factor, false);
            CHECK(!same_bits(r.dev[0], bad), "negative control: the non-NEOX pairing was not caught by the comparison");
            printf("  negative control: the other RoPE pairing changes the result: ok\n");
        }
    }
    D->graph_free(h);
}

static void test_rope_ops() {
    printf("rope\n");
    for (uint32_t T : TS) {
        test_rope(2, 64, 64, T, 1.0f);
        test_rope(14, 64, 64, T, 1.0f);
        test_rope(4, 128, 128, T, 1.0f);
        test_rope(2, 64, 32, T, 1.0f);       // the tail of every head is copied
        test_rope(2, 64, 64, T, 1.25f);      // attn_factor
    }
}

// ---------------------------------------------------------------------------------------------
// attn
// ---------------------------------------------------------------------------------------------
enum { MK_NONE_CAUSAL, MK_USER_CAUSAL, MK_SLIDING, MK_RANDOM, MK_ALL };

static std::vector<float> make_mask(int kind, uint32_t T) {
    std::vector<float> m((size_t) T * T, -INFINITY);
    for (uint32_t q = 0; q < T; ++q) {
        for (uint32_t j = 0; j < T; ++j) {
            bool vis = j <= q;
            if (kind == MK_SLIDING) vis = vis && j + 3u > q;
            if (kind == MK_RANDOM) vis = j == q || (vis && rnd_u(10u) >= 3u);
            if (kind == MK_ALL) vis = true;
            if (vis) m[(size_t) q * T + j] = 0.0f;
        }
    }
    return m;
}

// SET_ROWS_KV x 2 and FLASH_ATTN by hand on the image: the preload area by hand, indices at 0, the summary behind them. no_fa_new:
// the FLASH_ATTN record without FA_NEW. m: the mask as the device sees it (T x T, 0 / -inf), empty: causal.
static std::vector<float> attn_direct(const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v, uint32_t nh, uint32_t nkv,
                                      uint32_t D_, uint32_t T, float scale, const std::vector<float> & m, bool no_fa_new) {
    const uint32_t C = D->config().num_ch, E = nkv * D_, G = nh / nkv;
    image_t img(C);
    tsr_k_kv kv;
    std::string why;
    CHECK(tsr_k_kv_geom(E, T, C, &kv, &why), "kv geometry: %s", why.c_str());
    const uint32_t kp = (uint32_t) (img.alloc_heap(kv.bytes) / 64), vp = (uint32_t) (img.alloc_heap(kv.bytes) / 64);
    const uint32_t ql = nh * D_ / 16u, el = E / 16u;
    const uint32_t qh = img.put_rows(img.alloc_io((uint64_t) T * ql * 64), q, T, nh * D_, ql);
    const uint32_t kh = img.put_rows(img.alloc_io((uint64_t) T * el * 64), k, T, E, el);
    const uint32_t vh = img.put_rows(img.alloc_io((uint64_t) T * el * 64), v, T, E, el);
    const uint32_t ah = tsr_dram_handle(0, img.alloc_io((uint64_t) T * ql * 64));
    std::vector<uint16_t> m16;
    if (!m.empty()) {
        m16.resize(m.size());
        for (size_t i = 0; i < m.size(); ++i) m16[i] = m[i] == 0.0f ? 0x0000u : 0xFC00u;
    }
    const uint32_t f32_h = tsr_dram_handle(0, img.alloc_io((uint64_t) T * Lc(T) * 64));
    tsr_k_fa_mask mk;
    CHECK(tsr_k_fa_mask_build(g_kenv, E, D_, G, scale, T, T, f32_h, m.empty() ? nullptr : m16.data(), T, 0, &mk, &why), "mask: %s", why.c_str());
    if (mk.need_f32) memcpy(img.ch[0] + (uint64_t) (f32_h & 0x0FFFFFFFu) * 64, mk.f32.data(), mk.f32.size() * 4);
    std::vector<uint64_t> idx64(((T + 7u) / 8u) * 8u, 0);
    for (uint32_t t = 0; t < T; ++t) idx64[t] = t;
    const uint32_t msum_pre = (uint32_t) ((idx64.size() * 8 + 63u) / 64u * 64u);
    std::vector<uint8_t> pre(msum_pre + (mk.msum.size() * 4 + 63u) / 64u * 64u, 0);
    memcpy(pre.data(), idx64.data(), (size_t) T * 8);
    memcpy(pre.data() + msum_pre, mk.msum.data(), mk.msum.size() * 4);
    const uint32_t geom = opl::geom_word(E / 64u, kv.LG);
    const uint32_t ska[12] = { tsr_dram_handle(0, (uint64_t) kp * 64), 0u, kh, el, 0u, T, E, opl::setrows::WHICH_K, kp, geom, 0u, 0u };
    const uint32_t sva[12] = { tsr_dram_handle(0, (uint64_t) vp * 64), 0u, vh, el, 0u, T, E, opl::setrows::WHICH_V, vp, geom, 0u, 0u };
    const tsr::ref::dram_view vw = img.view();
    std::vector<uint8_t *> kpB(C), vpB(C);
    std::vector<const uint8_t *> kpC(C), vpC(C);
    for (uint32_t c = 0; c < C; ++c) {
        kpB[c] = img.ch[c] + (uint64_t) kp * 64;
        vpB[c] = img.ch[c] + (uint64_t) vp * 64;
        kpC[c] = kpB[c];
        vpC[c] = vpB[c];
    }
    CHECK(tsr::ref::set_rows_kv_ref(vw, ska, pre.data(), pre.size(), kpB.data(), C) == 0, "direct SET_ROWS_KV K");
    CHECK(tsr::ref::set_rows_kv_ref(vw, sva, pre.data(), pre.size(), vpB.data(), C) == 0, "direct SET_ROWS_KV V");
    const uint32_t faa[22] = { ah, ql, qh, D_ / 16u, ql, tsr_dram_handle(0, (uint64_t) kp * 64), tsr_dram_handle(0, (uint64_t) vp * 64), T, D_, nh, nkv, T,
                               msum_pre, mk.need_f32 ? f32_h : opl::HANDLE_NONE, f2b(scale), 0u, kp, vp, geom, D_, G,
                               no_fa_new ? 0u : tsr::plan::attn::fa_flags(0u, T) };
    tsr_att_cfg att = g_kenv.att;
    CHECK(tsr::ref::flash_attn_ref(vw, faa, pre.data(), pre.size(), kpC.data(), vpC.data(), att) == 0, "direct FLASH_ATTN");
    return img.rows_out(ah, T, nh * D_, ql);
}

static void test_attn(uint32_t nh, uint32_t nkv, uint32_t d, uint32_t T, int mk, uint32_t kv_size) {
    char what[128];
    snprintf(what, sizeof(what), "attn %u / %u heads of %u T %u mask %d kv_size %u", nh, nkv, d, T, mk, kv_size);
    PB pb;
    const uint32_t E = nkv * d;
    const int q = pb.in(nh * d), k = pb.in(E), v = pb.in(E);
    int m = -1;
    if (mk != MK_NONE_CAUSAL) m = pb.mask();
    const int y = pb.tmp(nh * d, true);
    const float scale = 1.0f / std::sqrt((float) d);
    pb.attn(q, k, v, y, nh, nkv, d, scale, m);
    pb.g.kv_size = kv_size;
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 2.0f), kv = rnd_vec((size_t) T * E, 2.0f), vv = rnd_vec((size_t) T * E, 2.0f);
    std::vector<arg_t> args = { in_arg(qv), in_arg(kv), in_arg(vv) };
    std::vector<float> mask;
    if (mk != MK_NONE_CAUSAL) {
        mask = make_mask(mk, T);
        args.push_back(mask_arg(mask));
    }
    result_t r;
    if (run_cmp(h, T, args, what, &r)) {
        const std::vector<float> want = attn_direct(qv, kv, vv, nh, nkv, d, T, scale, mk == MK_NONE_CAUSAL ? std::vector<float>() : mask, false);
        size_t first = 0;
        CHECK(same_bits(r.dev[0], want, &first), "%s: differs from the direct references at %zu", what, first);
    }
    D->graph_free(h);
}

static void test_attn_ops() {
    printf("attn\n");
    for (uint32_t T : TS) {
        test_attn(2, 1, 64, T, MK_NONE_CAUSAL, 64);
        test_attn(14, 2, 64, T, MK_NONE_CAUSAL, T);
        test_attn(4, 2, 128, T, MK_NONE_CAUSAL, 64);
        test_attn(2, 1, 64, T, MK_USER_CAUSAL, 64);
        test_attn(4, 2, 64, T, MK_SLIDING, 64);
        test_attn(4, 2, 64, T, MK_RANDOM, 64);      // holes: the GENERAL path with the f32 copy
        test_attn(2, 1, 64, T, MK_ALL, 64);
    }
    test_attn(2, 1, 64, 300, MK_NONE_CAUSAL, 512);   // several chunks
    test_attn(4, 2, 128, 200, MK_RANDOM, 200);
    test_attn(2, 1, 64, graph_max_attn_tokens, MK_NONE_CAUSAL, graph_max_attn_tokens);
}


// ---------------------------------------------------------------------------------------------
// Liveness: random programs of ADD, MUL and SCALE over a few inputs, evaluated on the host in f32 (the same bits: one IEEE
// operation per op). Values die and their IO ranges are reused; some middle values are outs. graph_run_ref cannot see a wrong
// placement (it runs the same lists), the host model can.
// ---------------------------------------------------------------------------------------------
static void test_random_dag(uint32_t n, uint32_t T, uint32_t n_ops, uint32_t seed) {
    char what[96];
    snprintf(what, sizeof(what), "random DAG n %u T %u ops %u seed %u", n, T, n_ops, seed);
    std::mt19937 rng(seed);
    PB pb;
    std::vector<int> ids;
    std::vector<std::vector<float>> host;
    std::vector<arg_t> args;
    for (int i = 0; i < 3; ++i) {
        ids.push_back(pb.in(n));
        std::vector<float> v((size_t) T * n);
        for (float & x : v) x = 0.5f + (float) (rng() % 1000) / 1000.0f;
        host.push_back(v);
        args.push_back(in_arg(v));
    }
    std::vector<int> tmp_pos;                     // indices (in ids) of the tmp values
    std::vector<int> outs;
    for (uint32_t k = 0; k < n_ops; ++k) {
        const int a = (int) (rng() % ids.size()), b = (int) (rng() % ids.size());
        const int kind = (int) (rng() % 3);
        const bool is_out = rng() % 7 == 0 || k + 1 == n_ops;
        const int dst = pb.tmp(n, is_out);
        std::vector<float> r(host[0].size());
        if (kind == 2) {
            const float sc = 0.6f + (float) (rng() % 800) / 1000.0f;
            graph_op & o = pb.op(gop::scale, dst);
            o.x = ids[a]; o.s = sc; o.sb = 0.0f;
            for (size_t i = 0; i < r.size(); ++i) { volatile float m = host[a][i] * sc; r[i] = m; }
        } else {
            const gop g = kind == 0 ? gop::add : gop::mul;
            pb.ew(g, ids[a], ids[b], dst);
            for (size_t i = 0; i < r.size(); ++i) { volatile float m = kind == 0 ? host[a][i] + host[b][i] : host[a][i] * host[b][i]; r[i] = m; }
        }
        ids.push_back(dst);
        host.push_back(r);
        if (is_out) outs.push_back((int) ids.size() - 1);
    }
    const graph_handle h = create(pb.g, what);
    if (!h) return;
    result_t r;
    if (run_cmp(h, T, args, what, &r)) {
        for (size_t i = 0; i < outs.size(); ++i) {
            size_t first = 0;
            CHECK(same_bits(r.dev[i], host[(size_t) outs[i]], &first), "%s: out %zu differs from the host model at %zu (%g vs %g)", what, i, first,
                  (double) r.dev[i][first], (double) host[(size_t) outs[i]][first]);
        }
        const graph_plan_info info = D->graph_info(h, T);
        // with liveness the IO use is far below one range per value
        const uint64_t all = (uint64_t) (n_ops + 3) * T * ((n + 15) / 16) * 64;
        CHECK(n_ops < 20 || info.io_bytes < all, "%s: IO bytes %" PRIu64 " (no reuse: %" PRIu64 ")", what, info.io_bytes, all);
    }
    D->graph_free(h);
}

static void test_dags() {
    printf("random programs (liveness)\n");
    for (uint32_t seed = 1; seed <= 6; ++seed) {
        test_random_dag(64, 1 + seed, 40, seed);
        test_random_dag(150, 5, 120, 100 + seed);
    }
}

// ---------------------------------------------------------------------------------------------
// A list that is full: more than 4096 records are cut into several lists; the attn op after the chain gets its FA_NEW from the
// map kept across the cut, whichever of its three records land in the first list.
// ---------------------------------------------------------------------------------------------
static void test_cut() {
    printf("lists of more than 4096 records\n");
    const uint32_t T = 5, nh = 2, nkv = 1, d = 64, n = 64;
    for (uint32_t chain : { 4000u, 4093u, 4094u, 4095u, 4096u, 4097u, 4100u }) {
        char what[96];
        snprintf(what, sizeof(what), "chain of %u ops, then attn", chain);
        PB pb;
        const int x = pb.in(n), q = pb.in(nh * d), k = pb.in(nkv * d), v = pb.in(nkv * d);
        int cur = x;
        for (uint32_t i = 0; i < chain; ++i) {
            const bool last = i + 1 == chain;
            const int nx = pb.tmp(n, last);
            graph_op & o = pb.op(gop::scale, nx);
            o.x = cur; o.s = 1.0f; o.sb = 0.0f;
            cur = nx;
        }
        const int y = pb.tmp(nh * d, true);
        pb.attn(q, k, v, y, nh, nkv, d, 0.125f, -1);
        pb.g.kv_size = 16;
        const graph_handle h = create(pb.g, what);
        if (!h) continue;
        const std::vector<float> xv = rnd_vec((size_t) T * n, 3.0f), qv = rnd_vec((size_t) T * nh * d, 2.0f), kv = rnd_vec((size_t) T * d, 2.0f),
                                 vv = rnd_vec((size_t) T * d, 2.0f);
        result_t r;
        if (run_cmp(h, T, { in_arg(xv), in_arg(qv), in_arg(kv), in_arg(vv) }, what, &r)) {
            CHECK(same_bits(r.dev[0], xv), "%s: the chain of scale by 1 returns its input", what);
            const std::vector<float> want = attn_direct(qv, kv, vv, nh, nkv, d, T, 0.125f, {}, false);
            CHECK(same_bits(r.dev[1], want), "%s: attn differs from the direct references", what);
            const graph_plan_info info = D->graph_info(h, T);
            CHECK(info.fa_new == 1, "%s: FA_NEW", what);
            printf("  %s: %u lists, %u records\n", what, info.lists, info.records);
            if (chain >= 4100) CHECK(info.lists >= 2, "%s: the list was not cut (%u lists)", what, info.lists);
        }
        D->graph_free(h);
    }
}

// ---------------------------------------------------------------------------------------------
// A decoder layer of Qwen2.5-0.5B's shape (the layer of fw_m7_layer_qwen05b) as graph ops
// ---------------------------------------------------------------------------------------------
struct layer_w {
    W q, k, v, o, g, u, d;
    const_handle an, fn, bq, bk, bv, tab;
};

static constexpr uint32_t Q_EMBD = 896, Q_HEAD = 14, Q_KV = 2, Q_D = 64, Q_FF = 4864;

static const_handle row_const(uint32_t n, float sc, float mean) {
    std::vector<float> v(n);
    for (float & x : v) x = mean + rnd(sc);
    return up_const(v, 1, n);
}

static layer_w make_layer() {
    layer_w w;
    w.q = up_weight(Q_HEAD * Q_D, Q_EMBD, 0.04f);
    w.k = up_weight(Q_KV * Q_D, Q_EMBD, 0.04f);
    w.v = up_weight(Q_KV * Q_D, Q_EMBD, 0.04f);
    w.o = up_weight(Q_EMBD, Q_HEAD * Q_D, 0.04f);
    w.g = up_weight(Q_FF, Q_EMBD, 0.04f);
    w.u = up_weight(Q_FF, Q_EMBD, 0.04f);
    w.d = up_weight(Q_EMBD, Q_FF, 0.04f);
    w.an = row_const(Q_EMBD, 0.3f, 1.0f);
    w.fn = row_const(Q_EMBD, 0.3f, 1.0f);
    w.bq = row_const(Q_HEAD * Q_D, 0.5f, 0.0f);
    w.bk = row_const(Q_KV * Q_D, 0.5f, 0.0f);
    w.bv = row_const(Q_KV * Q_D, 0.5f, 0.0f);
    std::vector<float> tab(Q_D / 2);
    for (uint32_t i = 0; i < Q_D / 2; ++i) tab[i] = (float) std::pow(1000000.0, -2.0 * (double) i / (double) Q_D);
    w.tab = up_const(tab, 1, Q_D / 2);
    return w;
}

static void free_layer(layer_w & w) {
    for (const W * x : { &w.q, &w.k, &w.v, &w.o, &w.g, &w.u, &w.d }) D->free_weight(x->h);
    for (const_handle c : { w.an, w.fn, w.bq, w.bk, w.bv, w.tab }) D->const_free(c);
}

// Appends one layer to pb: x (an activation value of Q_EMBD) -> the layer's output value.
static int add_layer(PB & pb, const layer_w & w, int x, int pos, bool out, int kv_slot = -1) {
    const int an = pb.cst(w.an, Q_EMBD), fn = pb.cst(w.fn, Q_EMBD), bq = pb.cst(w.bq, Q_HEAD * Q_D), bk = pb.cst(w.bk, Q_KV * Q_D), bv = pb.cst(w.bv, Q_KV * Q_D);
    const int tab = pb.cst(w.tab, Q_D / 2);
    const int xn = pb.tmp(Q_EMBD), q = pb.tmp(Q_HEAD * Q_D), k = pb.tmp(Q_KV * Q_D), v = pb.tmp(Q_KV * Q_D);
    const int qr = pb.tmp(Q_HEAD * Q_D), kr = pb.tmp(Q_KV * Q_D), att = pb.tmp(Q_HEAD * Q_D), hh = pb.tmp(Q_EMBD), hn = pb.tmp(Q_EMBD);
    const int g = pb.tmp(Q_FF), u = pb.tmp(Q_FF), gl = pb.tmp(Q_FF), y = pb.tmp(Q_EMBD, out);
    pb.rms(x, xn, an, 1e-6f);
    pb.mul_mat(w.q.h, xn, q, bq);
    pb.mul_mat(w.k.h, xn, k, bk);
    pb.mul_mat(w.v.h, xn, v, bv);
    pb.rope(q, qr, Q_HEAD, Q_D, Q_D, tab, pos);
    pb.rope(k, kr, Q_KV, Q_D, Q_D, tab, pos);
    pb.attn(qr, kr, v, att, Q_HEAD, Q_KV, Q_D, 0.125f, -1, kv_slot);
    pb.mul_mat(w.o.h, att, hh, x);
    pb.rms(hh, hn, fn, 1e-6f);
    pb.mul_mat(w.g.h, hn, g);
    pb.mul_mat(w.u.h, hn, u);
    graph_op & o = pb.op(gop::glu, gl);
    o.gate = g; o.up = u; o.variant = 0;
    pb.mul_mat(w.d.h, gl, y, hh);
    return y;
}

static std::vector<int32_t> iota(uint32_t T) {
    std::vector<int32_t> p(T);
    for (uint32_t t = 0; t < T; ++t) p[t] = (int32_t) t;
    return p;
}

static bool sane(const std::vector<float> & v) {
    double s = 0;
    for (float x : v) {
        if (!std::isfinite(x)) return false;
        s += std::fabs((double) x);
    }
    return s > 0.0;
}

static void test_layers() {
    printf("decoder layers\n");
    layer_w l1 = make_layer();
    {
        PB pb;
        const int x = pb.in(Q_EMBD), p = pb.pos();
        add_layer(pb, l1, x, p, true);
        pb.g.kv_size = 64;
        const graph_handle h = create(pb.g, "Qwen2.5-0.5B layer");
        if (h) {
            for (uint32_t T : { 1u, 5u, 17u }) {
                char what[64];
                snprintf(what, sizeof(what), "Qwen2.5-0.5B layer T %u", T);
                result_t r;
                const memory_info before = D->memory();
                const bool ok = run_cmp(h, T, { in_arg(rnd_vec((size_t) T * Q_EMBD, 1.0f)), pos_arg(iota(T)) }, what, &r);
                CHECK(ok && sane(r.dev[0]), "%s: the output is not sane", what);
                const graph_plan_info info = D->graph_info(h, T);
                printf("  %s: %u lists, %u records (%u JOIN, %u MUL_MAT_BEGIN), %.2f MiB of IO\n", what, info.lists, info.records, info.joins,
                       info.mul_mat_begin, (double) info.io_bytes / (1024.0 * 1024.0));
                CHECK(info.lists >= 1 && info.by_kind[opl::K_FLASH_ATTN] == 1 && info.fa_new == 1 && info.by_kind[opl::K_SET_ROWS_KV] == 2 &&
                      info.by_kind[opl::K_ROPE] == 2 && info.by_kind[opl::K_RMS_NORM] == 2 && info.by_kind[opl::K_GLU] == 1,
                      "%s: the records of the layer", what);
                CHECK(D->memory().weight_free == before.weight_free, "%s: a run does not allocate weight memory", what);
            }
            // the OPL cache hits on a repeat, the plan cache too; a second T reuses the planes
            const counters c0 = D->get_counters();
            const std::vector<float> x5 = rnd_vec((size_t) 5 * Q_EMBD, 1.0f);
            result_t r1, r2;
            run_cmp(h, 5, { in_arg(x5), pos_arg(iota(5)) }, "layer repeat 1", &r1);
            const counters c1 = D->get_counters();
            run_cmp(h, 5, { in_arg(x5), pos_arg(iota(5)) }, "layer repeat 2", &r2);
            const counters c2 = D->get_counters();
            CHECK(same_bits(r1.dev[0], r2.dev[0]), "a repeat gives the same bits");
            CHECK(c2.prog_hits > c1.prog_hits && c2.prog_misses == c1.prog_misses, "the OPL cache hits on a repeat (hits %" PRIu64 " -> %" PRIu64 ")",
                  c1.prog_hits, c2.prog_hits);
            CHECK(c2.graph_plan_hits > c1.graph_plan_hits && c2.graph_plan_misses == c1.graph_plan_misses, "the plan of T = 5 is cached");
            CHECK(c1.graph_launches > c0.graph_launches && c2.graph_records > c1.graph_records, "launch and record counters");
            const memory_info m0 = D->memory();
            run_cmp(h, 11, { in_arg(rnd_vec((size_t) 11 * Q_EMBD, 1.0f)), pos_arg(iota(11)) }, "layer T 11 (a second T)");
            CHECK(D->memory().weight_free == m0.weight_free, "a second T reuses the KV planes");
            D->graph_free(h);
        }
    }
    {
        // three layers in one program (the second and third with the weights of the first: distinct KV planes per attn op)
        layer_w l2 = make_layer(), l3 = make_layer();
        PB pb;
        const int x = pb.in(Q_EMBD), p = pb.pos();
        const int y1 = add_layer(pb, l1, x, p, false), y2 = add_layer(pb, l2, y1, p, false);
        add_layer(pb, l3, y2, p, true);
        pb.g.kv_size = 32;
        const memory_info m0 = D->memory();
        const graph_handle h = create(pb.g, "three layers");
        if (h) {
            const graph_shape sh = D->graph_shape_of(h);
            CHECK(sh.kv_bytes > 0 && m0.weight_free - D->memory().weight_free >= sh.kv_bytes, "the KV planes of three attn ops are allocated");
            for (uint32_t T : { 1u, 8u, 17u }) {
                char what[64];
                snprintf(what, sizeof(what), "three layers T %u", T);
                result_t r;
                const bool ok = run_cmp(h, T, { in_arg(rnd_vec((size_t) T * Q_EMBD, 1.0f)), pos_arg(iota(T)) }, what, &r);
                CHECK(ok && sane(r.dev[0]), "%s: the output is not sane", what);
                const graph_plan_info info = D->graph_info(h, T);
                printf("  %s: %u lists, %u records\n", what, info.lists, info.records);
            }
            D->graph_free(h);
            CHECK(D->memory().weight_free == m0.weight_free, "graph_free returns the KV planes (%" PRIu64 " vs %" PRIu64 ")", D->memory().weight_free, m0.weight_free);
        }
        free_layer(l2);
        free_layer(l3);
    }
    free_layer(l1);
}

static bool throws_with(const std::function<void()> & f, const char * needle, bool * invalid = nullptr) {
    try {
        f();
    } catch (const std::invalid_argument & e) {
        if (invalid) *invalid = true;
        return strstr(e.what(), needle) != nullptr;
    } catch (const std::exception & e) {
        if (invalid) *invalid = false;
        return strstr(e.what(), needle) != nullptr;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Cached attention: KV objects, attn ops with a KV slot
// ---------------------------------------------------------------------------------------------
static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size() && i < ref.size(); ++i) {
        const double e = (double) a[i] - (double) ref[i];
        num += e * e;
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0 ? num / den : num;
}

// The stored form of one KV cell row: the E / 32 ggml q8_0 blocks (fp16 d, 32 int8) that QACT A32 gives, packed (34 B each).
static std::vector<uint8_t> q8_row(const float * x, uint32_t E) {
    std::vector<uint8_t> qa((size_t) tsr_qa_lines(TSR_QA_A32, E) * 64);
    tsr_qact(TSR_QA_A32, x, E, qa.data());
    const int8_t * qs = tsr_qa_qs(TSR_QA_A32, E, qa.data());
    const float * dd = tsr_qa_d(TSR_QA_A32, E, qa.data());
    std::vector<uint8_t> out((size_t) E / 32 * 34);
    for (uint32_t b = 0; b < E / 32; ++b) {
        const uint16_t h = tsr_f32_to_fp16(dd[b]);
        memcpy(&out[(size_t) b * 34], &h, 2);
        memcpy(&out[(size_t) b * 34 + 2], qs + (size_t) b * 32, 32);
    }
    return out;
}

// The rows [c0, c0 + n) of both planes of a KV object against tsr_qact's blocks of the f32 rows (k, v: n rows of E, at row0 of them).
static bool kv_cells_equal(kv_handle h, uint32_t c0, uint32_t n, const std::vector<float> & k, const std::vector<float> & v, uint32_t E,
                           size_t row0, const char * what) {
    const size_t rb = (size_t) E / 32 * 34;
    std::vector<uint8_t> kb((size_t) n * rb), vb((size_t) n * rb);
    D->kv_read(h, c0, n, kb.data(), vb.data());
    bool ok = true;
    for (uint32_t t = 0; t < n && ok; ++t) {
        const std::vector<uint8_t> ek = q8_row(&k[(row0 + t) * E], E), ev = q8_row(&v[(row0 + t) * E], E);
        const bool a = memcmp(ek.data(), &kb[(size_t) t * rb], rb) == 0, b = memcmp(ev.data(), &vb[(size_t) t * rb], rb) == 0;
        CHECK(a && b, "%s: the cell %u: the K plane %s tsr_qact, the V plane %s", what, c0 + t, a ? "equals" : "DIFFERS from", b ? "equals" : "DIFFERS from");
        ok = a && b;
    }
    return ok;
}

// SET_ROWS_KV x 2 and FLASH_ATTN by hand on an image whose planes hold the history rows [0, pos0) (tsr_qact + tsr_ref_kvst: the
// same device image the kind tests build): the cells of the new rows are pos0 + t + idx_shift, the summary is built for pos0 +
// mask_shift. m: [T, cells] 0 / -inf (a given mask), empty: causal from pos0. kall, vall: rows 0 .. pos0 + T - 1 of E values.
static std::vector<float> attn_direct_cached(const std::vector<float> & q, const std::vector<float> & kall, const std::vector<float> & vall,
                                             uint32_t nh, uint32_t nkv, uint32_t D_, uint32_t pos0, uint32_t T, uint32_t cells, float scale,
                                             const std::vector<float> & m, int idx_shift = 0, int mask_shift = 0) {
    const uint32_t C = D->config().num_ch, E = nkv * D_, G = nh / nkv;
    image_t img(C);
    tsr_k_kv kv;
    std::string why;
    CHECK(tsr_k_kv_geom(E, cells, C, &kv, &why), "kv geometry: %s", why.c_str());
    const uint32_t kp = (uint32_t) (img.alloc_heap(kv.bytes) / 64), vp = (uint32_t) (img.alloc_heap(kv.bytes) / 64);
    const tsr_kv_geom geo = { E, C, kv.LG };
    std::vector<uint8_t *> kpP(C), vpP(C);
    for (uint32_t c = 0; c < C; ++c) {
        kpP[c] = img.ch[c] + (uint64_t) kp * 64;
        vpP[c] = img.ch[c] + (uint64_t) vp * 64;
    }
    std::vector<uint8_t> qa((size_t) tsr_qa_lines(TSR_QA_A32, E) * 64);
    for (uint32_t j = 0; j < pos0; ++j) {
        tsr_qact(TSR_QA_A32, &kall[(size_t) j * E], E, qa.data());
        tsr_ref_kvst(TSR_KV_K, &geo, kpP.data(), j, 1u, qa.data(), 0u, nullptr);
        tsr_qact(TSR_QA_A32, &vall[(size_t) j * E], E, qa.data());
        tsr_ref_kvst(TSR_KV_V, &geo, vpP.data(), j, 1u, qa.data(), 0u, nullptr);
    }
    const std::vector<float> knew(kall.begin() + (size_t) pos0 * E, kall.end()), vnew(vall.begin() + (size_t) pos0 * E, vall.end());
    const uint32_t ql = nh * D_ / 16u, el = E / 16u;
    const uint32_t qh = img.put_rows(img.alloc_io((uint64_t) T * ql * 64), q, T, nh * D_, ql);
    const uint32_t kh = img.put_rows(img.alloc_io((uint64_t) T * el * 64), knew, T, E, el);
    const uint32_t vh = img.put_rows(img.alloc_io((uint64_t) T * el * 64), vnew, T, E, el);
    const uint32_t ah = tsr_dram_handle(0, img.alloc_io((uint64_t) T * ql * 64));
    std::vector<uint16_t> m16;
    if (!m.empty()) {
        m16.resize(m.size());
        for (size_t i = 0; i < m.size(); ++i) m16[i] = m[i] == 0.0f ? 0x0000u : 0xFC00u;
    }
    const uint32_t f32_h = tsr_dram_handle(0, img.alloc_io((uint64_t) T * Lc(cells) * 64));
    tsr_k_fa_mask mk;
    CHECK(tsr_k_fa_mask_build(g_kenv, E, D_, G, scale, T, cells, f32_h, m.empty() ? nullptr : m16.data(), cells, (uint32_t) ((int) pos0 + mask_shift), &mk, &why),
          "mask: %s", why.c_str());
    if (mk.need_f32) memcpy(img.ch[0] + (uint64_t) (f32_h & 0x0FFFFFFFu) * 64, mk.f32.data(), mk.f32.size() * 4);
    std::vector<uint64_t> idx64(((T + 7u) / 8u) * 8u, 0);
    for (uint32_t t = 0; t < T; ++t) idx64[t] = (uint64_t) ((int) (pos0 + t) + idx_shift);
    const uint32_t msum_pre = (uint32_t) ((idx64.size() * 8 + 63u) / 64u * 64u);
    std::vector<uint8_t> pre(msum_pre + (mk.msum.size() * 4 + 63u) / 64u * 64u, 0);
    memcpy(pre.data(), idx64.data(), (size_t) T * 8);
    memcpy(pre.data() + msum_pre, mk.msum.data(), mk.msum.size() * 4);
    const uint32_t geom = opl::geom_word(E / 64u, kv.LG);
    const uint32_t ska[12] = { tsr_dram_handle(0, (uint64_t) kp * 64), 0u, kh, el, 0u, T, E, opl::setrows::WHICH_K, kp, geom, 0u, 0u };
    const uint32_t sva[12] = { tsr_dram_handle(0, (uint64_t) vp * 64), 0u, vh, el, 0u, T, E, opl::setrows::WHICH_V, vp, geom, 0u, 0u };
    const tsr::ref::dram_view vw = img.view();
    std::vector<const uint8_t *> kpC(kpP.begin(), kpP.end()), vpC(vpP.begin(), vpP.end());
    CHECK(tsr::ref::set_rows_kv_ref(vw, ska, pre.data(), pre.size(), kpP.data(), C) == 0, "direct SET_ROWS_KV K");
    CHECK(tsr::ref::set_rows_kv_ref(vw, sva, pre.data(), pre.size(), vpP.data(), C) == 0, "direct SET_ROWS_KV V");
    const uint32_t faa[22] = { ah, ql, qh, D_ / 16u, ql, tsr_dram_handle(0, (uint64_t) kp * 64), tsr_dram_handle(0, (uint64_t) vp * 64), cells, D_, nh, nkv, T,
                               msum_pre, mk.need_f32 ? f32_h : opl::HANDLE_NONE, f2b(scale), 0u, kp, vp, geom, D_, G, tsr::plan::attn::fa_flags(0u, T) };
    CHECK(tsr::ref::flash_attn_ref(vw, faa, pre.data(), pre.size(), kpC.data(), vpC.data(), g_kenv.att) == 0, "direct FLASH_ATTN");
    return img.rows_out(ah, T, nh * D_, ql);
}

// A cached attn program alone: q, k, v (in), a [T, cells] mask when `given`, the output. One KV slot of `cells` cells.
struct att_prog {
    graph_handle h = 0;
    uint32_t nh = 0, nkv = 0, d = 0, cells = 0;
    bool given = false;
    float scale = 0.125f;
};

static att_prog make_att(uint32_t nh, uint32_t nkv, uint32_t d, uint32_t cells, bool given, const char * what) {
    PB pb;
    const int q = pb.in(nh * d), k = pb.in(nkv * d), v = pb.in(nkv * d);
    const int m = given ? pb.mask(cells) : -1;
    const int y = pb.tmp(nh * d, true);
    att_prog a;
    a.nh = nh; a.nkv = nkv; a.d = d; a.cells = cells; a.given = given;
    a.scale = 1.0f / std::sqrt((float) d);
    pb.attn(q, k, v, y, nh, nkv, d, a.scale, m, 0);
    pb.g.kv_slots = { cells };
    a.h = create(pb.g, what);
    return a;
}

// A [T, cols] mask of the cached form for the rows pos0 ..: row t sees cells 0 .. pos0 + t, with `holes` (fraction of the visible
// cells below pos0 + t that are hidden); a cell of the same call or before may be hidden but the row's own cell stays visible.
static std::vector<float> cached_mask(uint32_t T, uint32_t pos0, uint32_t cols, uint32_t holes_pct) {
    std::vector<float> m((size_t) T * cols, -INFINITY);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j <= pos0 + t; ++j) {
            if (j != pos0 + t && holes_pct != 0 && rnd_u(100u) < holes_pct) continue;
            m[(size_t) t * cols + j] = 0.0f;
        }
    }
    return m;
}

// The global [R, R] mask of the stateless run from the per-step [T, cols] masks (rows concatenated; the columns below R).
struct seq_t {
    std::vector<float> q, k, v, y;       // all rows so far
    std::vector<float> mask;             // [R, cols] given masks, rows concatenated
};

// One sequence on one KV object through the cached attn program in steps of the given row counts: every step device == reference
// (run_cmp), == the direct reference of the kind tests (history by tsr_qact + tsr_ref_kvst) bit for bit, the cells just written ==
// tsr_qact's blocks; at the end the outputs == the stateless run of all rows (the NMSE is returned in *nm). Returns the cached outputs.
static bool test_cached_seq(const att_prog & a, kv_handle kvh, const std::vector<uint32_t> & steps, uint32_t holes_pct, const char * label,
                            double * nm, std::vector<float> * out_all = nullptr, bool check_direct = true) {
    const uint32_t E = a.nkv * a.d;
    seq_t sq;
    uint32_t pos0 = 0;
    bool ok = true;
    for (size_t si = 0; si < steps.size(); ++si) {
        const uint32_t T = steps[si];
        char what[160];
        snprintf(what, sizeof(what), "cached attn %u / %u heads of %u, %u cells, %s: step %zu (pos0 %u, T %u)", a.nh, a.nkv, a.d, a.cells, label, si, pos0, T);
        const std::vector<float> qv = rnd_vec((size_t) T * a.nh * a.d, 2.0f), kv = rnd_vec((size_t) T * E, 2.0f), vv = rnd_vec((size_t) T * E, 2.0f);
        std::vector<arg_t> args = { in_arg(qv), in_arg(kv), in_arg(vv) };
        std::vector<float> mk;
        if (a.given) {
            mk = cached_mask(T, pos0, a.cells, holes_pct);
            args.push_back(mask_arg(mk));
        }
        sq.q.insert(sq.q.end(), qv.begin(), qv.end());
        sq.k.insert(sq.k.end(), kv.begin(), kv.end());
        sq.v.insert(sq.v.end(), vv.begin(), vv.end());
        sq.mask.insert(sq.mask.end(), mk.begin(), mk.end());
        graph_state st;
        st.kv = { kvh };
        st.pos0 = pos0;
        result_t r;
        if (!run_cmp(a.h, T, args, what, &r, 0, st)) { ok = false; break; }
        sq.y.insert(sq.y.end(), r.dev[0].begin(), r.dev[0].end());
        if (check_direct) {
            const std::vector<float> want = attn_direct_cached(qv, sq.k, sq.v, a.nh, a.nkv, a.d, pos0, T, a.cells, a.scale, mk);
            size_t first = 0;
            const bool same = same_bits(r.dev[0], want, &first);
            CHECK(same, "%s: differs from the direct references at %zu (%g vs %g)", what, first, (double) r.dev[0][first], (double) want[first]);
            ok = ok && same;
        }
        ok = kv_cells_equal(kvh, pos0, T, sq.k, sq.v, E, pos0, what) && ok;
        pos0 += T;
    }
    if (!ok || nm == nullptr) { if (out_all) *out_all = sq.y; return ok; }
    // the stateless run of all the rows: the existing path (own planes, [R, R] mask)
    const uint32_t R = pos0;
    PB pb;
    const int q = pb.in(a.nh * a.d), k = pb.in(E), v = pb.in(E);
    int m = a.given ? pb.mask() : -1;
    const int y = pb.tmp(a.nh * a.d, true);
    pb.attn(q, k, v, y, a.nh, a.nkv, a.d, a.scale, m);
    pb.g.kv_size = R;
    const graph_handle h2 = create(pb.g, "stateless");
    if (!h2) return false;
    std::vector<arg_t> args = { in_arg(sq.q), in_arg(sq.k), in_arg(sq.v) };
    std::vector<float> mm;
    if (a.given) {
        mm.assign((size_t) R * R, -INFINITY);
        for (uint32_t r = 0; r < R; ++r) memcpy(&mm[(size_t) r * R], &sq.mask[(size_t) r * a.cells], (size_t) R * 4);
        args.push_back(mask_arg(mm));
    }
    result_t r2;
    run_cmp(h2, R, args, "stateless run of the same rows", &r2);
    D->graph_free(h2);
    *nm = r2.dev.empty() ? 1.0 : nmse(sq.y, r2.dev[0]);
    if (out_all) *out_all = sq.y;
    return ok;
}

// The cycles of one decode step (T = 1 at pos0) of an attn op over a KV slot of `cells` cells.
static uint64_t decode_cycles(uint32_t nh, uint32_t nkv, uint32_t d, uint32_t cells, uint32_t pos0) {
    const att_prog a = make_att(nh, nkv, d, cells, false, "decode cycles");
    if (!a.h) return 0;
    const uint32_t E = nkv * d;
    const kv_handle kvh = D->kv_create(E, cells);
    if (pos0 != 0) {
        const std::vector<float> kh = rnd_vec((size_t) pos0 * E, 2.0f), vh = rnd_vec((size_t) pos0 * E, 2.0f);
        D->kv_store(kvh, 0, pos0, kh.data(), vh.data());
    }
    const std::vector<float> qv = rnd_vec((size_t) nh * d, 2.0f), kv = rnd_vec(E, 2.0f), vv = rnd_vec(E, 2.0f);
    graph_inputs gi;
    gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { vv.data(), 0 } };
    std::vector<float> out((size_t) nh * d);
    graph_outputs go;
    go.outs = { { out.data(), 0 } };
    graph_state st;
    st.kv = { kvh };
    st.pos0 = pos0;
    D->graph_run(a.h, 1, gi, go, st);          // plan, op list
    const counters c0 = D->get_counters();
    D->graph_run(a.h, 1, gi, go, st);
    const counters c1 = D->get_counters();
    D->graph_free(a.h);
    D->kv_free(kvh);
    return c1.perf_cycles - c0.perf_cycles;
}

static void measure_nkv() {
    printf("cached attention: device cycles of a decode step against the planes' cells\n");
    for (const uint32_t pos0 : { 15u, 127u, 1023u }) {
        const uint64_t c256 = pos0 + 1 <= 256 ? decode_cycles(14, 2, 64, 256, pos0) : 0;
        const uint64_t c4096 = decode_cycles(14, 2, 64, 4096, pos0);
        const uint64_t c1024 = pos0 + 1 <= 1024 ? decode_cycles(14, 2, 64, 1024, pos0) : 0;
        printf("  %u visible cells: planes of 256 cells %" PRIu64 " cycles, 1024 cells %" PRIu64 ", 4096 cells %" PRIu64 " (4096 / 256: %.4f)\n", pos0 + 1,
               c256, c1024, c4096, c256 ? (double) c4096 / (double) c256 : 0.0);
    }
}

// Measured (simulator, the shapes and seeds of the tests below): the NMSE of the cached steps against the stateless run of the same rows is 0
// for the causal masks and at most 3.2e-15 for given masks (a different row order of the same sums), for a decoder layer 0. The gate has a
// margin of more than 1e4 over that; a wrong pos0, cell index or mask offset gives an NMSE of order 1e-2 and more (negative controls).
static constexpr double NMSE_BOUND = 1e-10;
static double g_nm_attn = 0, g_nm_layer = 0;   // largest NMSE seen (measurement)

static void test_cached_attn_cases() {
    printf("cached attention (attn alone)\n");
    struct shape { uint32_t nh, nkv, d; };
    const shape shapes[] = { { 2, 1, 64 }, { 14, 2, 64 }, { 4, 2, 128 } };
    for (const shape & sh : shapes) {
        for (uint32_t seed = 0; seed < 3; ++seed) {
            for (const bool given : { false, true }) {
                for (const uint32_t cells : { 160u, 256u }) {
                    const att_prog a = make_att(sh.nh, sh.nkv, sh.d, cells, given, "cached attn");
                    if (!a.h) continue;
                    const kv_handle kvh = D->kv_create(sh.nkv * sh.d, cells);
                    // a prefill of 1 + seed * 7 rows (T = 1 .. 15), then decode steps and one multi-row step
                    std::vector<uint32_t> steps = { 1 + seed * 7 + 2, 1, 1, 3, 1, 1 + seed };
                    double nm = 0;
                    test_cached_seq(a, kvh, steps, given ? 20u : 0u, given ? "given mask with holes" : "causal", &nm);
                    g_nm_attn = std::max(g_nm_attn, nm);
                    CHECK(nm < NMSE_BOUND, "cached attn %u / %u heads of %u, %u cells: the steps differ from the stateless run (NMSE %.3g)", sh.nh, sh.nkv, sh.d, cells, nm);
                    printf("  %u / %u heads of %u, %u cells, %s, seed %u: NMSE of the cached steps against the stateless run %.3g\n", sh.nh, sh.nkv, sh.d, cells,
                           given ? "given mask" : "causal", seed, nm);
                    D->kv_free(kvh);
                    D->graph_free(a.h);
                }
            }
        }
    }
    printf("  largest NMSE of an attention output against the stateless run: %.3g\n", g_nm_attn);
}

// ---- KV objects
static void test_kv_objects() {
    printf("KV objects\n");
    const memory_info m0 = D->memory();
    const counters c0 = D->get_counters();
    uint64_t creates = 0, stores = 0, store_rows = 0;
    size_t q8ref_bad = 0, q8ref_blocks = 0;
    for (const uint32_t E : { 64u, 128u, 256u, 1024u }) {
        for (const uint32_t cells : { 1u, 16u, 100u, 700u }) {
            char what[96];
            snprintf(what, sizeof(what), "KV object E %u cells %u", E, cells);
            const kv_handle h = D->kv_create(E, cells);
            creates++;
            const kv_info info = D->kv_info_of(h);
            tsr_k_kv geom;
            std::string why;
            CHECK(tsr_k_kv_geom(E, cells, D->config().num_ch, &geom, &why), "%s: %s", what, why.c_str());
            CHECK(info.E == E && info.cells == cells && info.bytes == 2 * geom.bytes, "%s: kv_info_of: E %u cells %u bytes %" PRIu64 " (%" PRIu64 " expected)", what,
                  info.E, info.cells, info.bytes, 2 * geom.bytes);
            CHECK(info.k_off % 4096 == 0 && info.v_off % 4096 == 0 && info.k_off != info.v_off && info.k_off >= TSR_REGION_WEIGHT_OFF, "%s: plane offsets 0x%" PRIx64 " 0x%" PRIx64, what, info.k_off, info.v_off);
            const size_t rb = (size_t) E / 32 * 34;
            std::vector<uint8_t> kb((size_t) cells * rb, 0xAA), vb((size_t) cells * rb, 0xAA);
            D->kv_read(h, 0, cells, kb.data(), vb.data());
            bool zero = true;
            for (size_t i = 0; i < kb.size(); ++i) zero = zero && kb[i] == 0 && vb[i] == 0;
            CHECK(zero, "%s: a new object is zero in every cell", what);
            // store the middle of the cells (a row stride above E: strided input), then the cells around it are still zero
            const uint32_t c1 = cells / 3, n = std::max(1u, cells / 2);
            const uint32_t nn = std::min(n, cells - c1);
            const size_t stride = E + 5;
            std::vector<float> ks((size_t) nn * stride, 777.0f), vs((size_t) nn * stride, 777.0f), kk((size_t) nn * E), vv((size_t) nn * E);
            for (uint32_t t = 0; t < nn; ++t) {
                for (uint32_t e = 0; e < E; ++e) {
                    kk[(size_t) t * E + e] = ks[(size_t) t * stride + e] = rnd_n(1.5f);
                    vv[(size_t) t * E + e] = vs[(size_t) t * stride + e] = rnd_n(1.5f);
                }
            }
            D->kv_store(h, c1, nn, ks.data(), vs.data(), stride, stride);
            stores++;
            store_rows += nn;
            kv_cells_equal(h, c1, nn, kk, vv, E, 0, what);
            // against ggml's quantizer too (ordinary random rows: equal)
            std::vector<uint8_t> kr((size_t) nn * rb), vr((size_t) nn * rb);
            D->kv_read(h, c1, nn, kr.data(), vr.data());
            for (uint32_t t = 0; t < nn; ++t) {
                std::vector<block_q8_0> ref(E / 32);
                quantize_row_q8_0_ref(&kk[(size_t) t * E], ref.data(), E);
                for (uint32_t b = 0; b < E / 32; ++b) {
                    q8ref_blocks++;
                    q8ref_bad += memcmp(&ref[b], &kr[(size_t) t * rb + (size_t) b * 34], 34) != 0;
                }
            }
            D->kv_read(h, 0, cells, kb.data(), vb.data());
            bool around = true;
            for (uint32_t c = 0; c < cells; ++c) {
                if (c >= c1 && c < c1 + nn) continue;
                for (size_t i = 0; i < rb; ++i) around = around && kb[(size_t) c * rb + i] == 0 && vb[(size_t) c * rb + i] == 0;
            }
            CHECK(around, "%s: the cells that were not stored are still zero", what);
            // overwrite the same cells: the new rows are read, the neighbours are unchanged
            const std::vector<float> k2 = rnd_vec((size_t) nn * E, 3.0f), v2 = rnd_vec((size_t) nn * E, 3.0f);
            D->kv_store(h, c1, nn, k2.data(), v2.data());
            stores++;
            store_rows += nn;
            kv_cells_equal(h, c1, nn, k2, v2, E, 0, what);
            D->kv_free(h);
        }
    }
    CHECK(q8ref_bad == 0, "the stored K blocks differ from quantize_row_q8_0_ref in %zu of %zu blocks", q8ref_bad, q8ref_blocks);
    CHECK(D->memory().weight_free == m0.weight_free && D->memory().weight_largest_free == m0.weight_largest_free, "kv_free returns the planes");
    // more rows than one list takes (the chunks of graph_max_attn_tokens), a cell0 that is not a multiple of a group
    {
        const uint32_t E = 128, cells = 1700, n = 1500, c1 = 37;
        const kv_handle h = D->kv_create(E, cells);
        const std::vector<float> kk = rnd_vec((size_t) n * E, 2.0f), vv = rnd_vec((size_t) n * E, 2.0f);
        const counters a = D->get_counters();
        D->kv_store(h, c1, n, kk.data(), vv.data());
        stores++;
        store_rows += n;
        const counters b = D->get_counters();
        CHECK(b.execs - a.execs == 3, "kv_store of %u rows: %" PRIu64 " launches (3 expected: lists of at most %u rows)", n, b.execs - a.execs, graph_max_attn_tokens);
        kv_cells_equal(h, c1, n, kk, vv, E, 0, "kv_store of 1500 rows");
        D->kv_free(h);
    }
    const counters c1 = D->get_counters();
    CHECK(c1.kv_creates - c0.kv_creates == creates + 1, "kv_creates %" PRIu64, c1.kv_creates - c0.kv_creates);
    CHECK(c1.kv_stores - c0.kv_stores == stores, "kv_stores %" PRIu64 " (%" PRIu64 " expected)", c1.kv_stores - c0.kv_stores, stores);
    CHECK(c1.kv_store_rows - c0.kv_store_rows == store_rows, "kv_store_rows %" PRIu64 " (%" PRIu64 " expected)", c1.kv_store_rows - c0.kv_store_rows, store_rows);
    // refusals
    {
        const kv_handle h = D->kv_create(64, 32);
        const std::vector<float> rows = rnd_vec(64 * 8, 1.0f);
        bool inv = false;
        CHECK(throws_with([&] { D->kv_store(h, 30, 4, rows.data(), rows.data()); }, "beyond", &inv) && inv, "kv_store beyond the cells: invalid_argument");
        CHECK(throws_with([&] { D->kv_read(h, 31, 2, nullptr, nullptr); }, "null", &inv) && inv, "kv_read to null");
        std::vector<uint8_t> blocks(4 * 68);
        CHECK(throws_with([&] { D->kv_read(h, 31, 2, blocks.data(), blocks.data()); }, "beyond", &inv) && inv, "kv_read beyond the cells: invalid_argument");
        CHECK(throws_with([&] { D->kv_store(h + 1000, 0, 1, rows.data(), rows.data()); }, "unknown KV handle", &inv) && inv, "unknown handle: kv_store");
        CHECK(throws_with([&] { D->kv_info_of(h + 1000); }, "unknown KV handle", &inv) && inv, "unknown handle: kv_info_of");
        CHECK(throws_with([&] { D->kv_create(100, 16); }, "multiple of 64", &inv) && inv, "kv_create E = 100: invalid_argument");
        CHECK(throws_with([&] { D->kv_create(64, 0); }, "cells", &inv) && inv, "kv_create 0 cells: invalid_argument");
        // a plane that no channel can hold: the geometry (LG above 16 bits) or the heap refuses it, "does not fit", not invalid_argument
        const memory_info m = D->memory();
        const uint32_t huge = 16u * D->config().num_ch * 65535u;
        CHECK(throws_with([&] { D->kv_create(511 * 64, huge); }, "does not fit", &inv) && !inv, "a plane of 511 * 64 x %u cells does not fit", huge);
        CHECK(D->memory().weight_free == m.weight_free, "a refused kv_create allocates nothing");
        D->kv_free(h);
        CHECK(throws_with([&] { D->kv_free(h); }, "unknown KV handle", &inv) && inv, "kv_free twice");
        const kv_handle h1 = D->kv_create(64, 1);
        D->kv_store(h1, 0, 0, nullptr, nullptr);   // n = 0 is a no-op
        D->kv_free(h1);
    }
}

// ---- the decoder layer with KV slots
static void test_cached_layers() {
    printf("cached attention (decoder layers)\n");
    struct cfg { uint32_t layers, cells; std::vector<uint32_t> steps; };
    const cfg cfgs[] = { { 1, 64, { 5, 1, 1, 1, 1 } }, { 1, 160, { 1, 7, 1, 1, 3, 1 } }, { 2, 96, { 9, 1, 1, 1 } } };
    for (const cfg & c : cfgs) {
        std::vector<layer_w> ls;
        for (uint32_t i = 0; i < c.layers; ++i) ls.push_back(make_layer());
        PB pb;
        const int x = pb.in(Q_EMBD), p = pb.pos();
        int cur = x;
        for (uint32_t i = 0; i < c.layers; ++i) {
            cur = add_layer(pb, ls[i], cur, p, i + 1 == c.layers, (int) i);
            pb.g.kv_slots.push_back(c.cells + 16 * i);       // other cells for the second layer: its own geometry
        }
        char what[96];
        snprintf(what, sizeof(what), "cached %u layer(s) of Qwen2.5-0.5B, %u cells", c.layers, c.cells);
        const graph_handle h = create(pb.g, what);
        if (h) {
            std::vector<kv_handle> kvs;
            for (uint32_t i = 0; i < c.layers; ++i) kvs.push_back(D->kv_create(Q_KV * Q_D, c.cells + 16 * i));
            std::vector<float> xs, ys;
            uint32_t pos0 = 0;
            std::set<uint32_t> seen;
            for (size_t si = 0; si < c.steps.size(); ++si) {
                const uint32_t T = c.steps[si];
                char w2[160];
                snprintf(w2, sizeof(w2), "%s: step %zu (pos0 %u, T %u)", what, si, pos0, T);
                const std::vector<float> xv = rnd_vec((size_t) T * Q_EMBD, 1.0f);
                std::vector<int32_t> pv(T);
                for (uint32_t t = 0; t < T; ++t) pv[t] = (int32_t) (pos0 + t);
                graph_state st;
                st.kv = kvs;
                st.pos0 = pos0;
                const counters before = D->get_counters();
                result_t r;
                const bool ok = run_cmp(h, T, { in_arg(xv), pos_arg(pv) }, w2, &r, 0, st);
                CHECK(ok && sane(r.dev[0]), "%s: the output is not sane", w2);
                const counters after = D->get_counters();
                if (seen.count(T) != 0) {
                    // a T seen before at another pos0: the plan and the op lists are the same (only the data differs): no plan, no op list built
                    CHECK(after.graph_plan_misses == before.graph_plan_misses, "%s: a plan was built for a T seen before", w2);
                    CHECK(after.prog_misses == before.prog_misses && after.prog_hits > before.prog_hits, "%s: the op list was written again (OPL cache misses %" PRIu64 " -> %" PRIu64 ")", w2,
                          before.prog_misses, after.prog_misses);
                }
                seen.insert(T);
                xs.insert(xs.end(), xv.begin(), xv.end());
                if (ok) ys.insert(ys.end(), r.dev[0].begin(), r.dev[0].end());
                pos0 += T;
            }
            // the stateless run of all the rows
            PB p2;
            const int x2 = p2.in(Q_EMBD), pp = p2.pos();
            int cur2 = x2;
            for (uint32_t i = 0; i < c.layers; ++i) cur2 = add_layer(p2, ls[i], cur2, pp, i + 1 == c.layers);
            p2.g.kv_size = pos0;
            const graph_handle h2 = create(p2.g, "stateless layers");
            if (h2) {
                result_t r2;
                run_cmp(h2, pos0, { in_arg(xs), pos_arg(iota(pos0)) }, "stateless layers", &r2);
                const double nm = nmse(ys, r2.dev[0]);
                g_nm_layer = std::max(g_nm_layer, nm);
                printf("  %s: NMSE of the cached steps against the stateless run of %u rows: %.3g\n", what, pos0, nm);
                CHECK(nm < NMSE_BOUND, "%s: the cached steps differ from the stateless run (NMSE %.3g)", what, nm);
                D->graph_free(h2);
            }
            for (kv_handle k : kvs) D->kv_free(k);
            D->graph_free(h);
        }
        for (layer_w & l : ls) free_layer(l);
    }
}

// ---- sequences on several KV objects
struct step_data {
    std::vector<float> q, k, v;
};

static std::vector<step_data> make_steps(const att_prog & a, const std::vector<uint32_t> & steps) {
    std::vector<step_data> d;
    for (uint32_t T : steps) d.push_back({ rnd_vec((size_t) T * a.nh * a.d, 2.0f), rnd_vec((size_t) T * a.nkv * a.d, 2.0f), rnd_vec((size_t) T * a.nkv * a.d, 2.0f) });
    return d;
}

// One causal step of an attn program (run_cmp: device == reference); the outputs are appended to *out.
static bool run_step(const att_prog & a, kv_handle kvh, const step_data & sd, uint32_t pos0, std::vector<float> * out, const char * what) {
    const uint32_t T = (uint32_t) (sd.q.size() / (a.nh * a.d));
    graph_state st;
    st.kv = { kvh };
    st.pos0 = pos0;
    result_t r;
    if (!run_cmp(a.h, T, { in_arg(sd.q), in_arg(sd.k), in_arg(sd.v) }, what, &r, 0, st)) return false;
    out->insert(out->end(), r.dev[0].begin(), r.dev[0].end());
    return true;
}

static void test_cached_objects() {
    printf("cached attention: several KV objects on one graph, reused planes\n");
    const att_prog a = make_att(14, 2, 64, 128, false, "two objects");
    if (!a.h) return;
    const uint32_t E = a.nkv * a.d;
    const std::vector<uint32_t> sa = { 6, 1, 1, 1, 1 }, sb = { 10, 1, 1, 1 };
    const std::vector<step_data> da = make_steps(a, sa), db = make_steps(a, sb);
    // each sequence alone on its own object
    std::vector<float> alone_a, alone_b;
    {
        const kv_handle ka = D->kv_create(E, 128), kb = D->kv_create(E, 128);
        uint32_t p = 0;
        for (size_t i = 0; i < sa.size(); ++i) { run_step(a, ka, da[i], p, &alone_a, "alone A"); p += sa[i]; }
        p = 0;
        for (size_t i = 0; i < sb.size(); ++i) { run_step(a, kb, db[i], p, &alone_b, "alone B"); p += sb[i]; }
        D->kv_free(ka);
        D->kv_free(kb);
    }
    // alternating: A0, B0, A1, B1, ...; each step binds its own object to the one graph
    {
        const kv_handle ka = D->kv_create(E, 128), kb = D->kv_create(E, 128);
        std::vector<float> oa, ob;
        uint32_t pa = 0, pb = 0;
        for (size_t i = 0; i < std::max(sa.size(), sb.size()); ++i) {
            if (i < sa.size()) { run_step(a, ka, da[i], pa, &oa, "interleaved A"); pa += sa[i]; }
            if (i < sb.size()) { run_step(a, kb, db[i], pb, &ob, "interleaved B"); pb += sb[i]; }
        }
        CHECK(same_bits(oa, alone_a), "two KV objects alternating: sequence A differs from its run alone");
        CHECK(same_bits(ob, alone_b), "two KV objects alternating: sequence B differs from its run alone");
        // the objects do not touch each other: A's cells hold A's rows
        std::vector<float> ka_rows, va_rows;
        for (const step_data & s : da) { ka_rows.insert(ka_rows.end(), s.k.begin(), s.k.end()); va_rows.insert(va_rows.end(), s.v.begin(), s.v.end()); }
        kv_cells_equal(ka, 0, pa, ka_rows, va_rows, E, 0, "object A after the alternation");
        D->kv_free(ka);
        D->kv_free(kb);
    }
    // planes reused after another sequence (no zero fill: the cells above the new sequence's length hold the old one's rows)
    {
        const kv_handle kx = D->kv_create(E, 128), ky = D->kv_create(E, 128);
        const std::vector<step_data> d1 = make_steps(a, { 40, 1, 1, 1 });
        std::vector<float> junk;
        uint32_t p = 0;
        for (size_t i = 0; i < d1.size(); ++i) { run_step(a, kx, d1[i], p, &junk, "old sequence"); p += (uint32_t) (d1[i].q.size() / (a.nh * a.d)); }
        std::vector<float> reused, fresh;
        p = 0;
        for (size_t i = 0; i < sa.size(); ++i) { run_step(a, kx, da[i], p, &reused, "reused planes"); p += sa[i]; }
        p = 0;
        for (size_t i = 0; i < sa.size(); ++i) { run_step(a, ky, da[i], p, &fresh, "fresh planes"); p += sa[i]; }
        CHECK(same_bits(reused, fresh), "planes reused after another sequence give other bits than fresh planes");
        CHECK(same_bits(reused, alone_a), "reused planes differ from the first run of the sequence");
        D->kv_free(kx);
        D->kv_free(ky);
    }
    D->graph_free(a.h);
}

// ---- refusals and masks
static void test_cached_refusals() {
    printf("cached attention: refusals and masks\n");
    const uint32_t nh = 2, nkv = 1, d = 64, E = nkv * d, cells = 64;
    const att_prog a = make_att(nh, nkv, d, cells, true, "refusals");
    if (!a.h) return;
    const kv_handle kvh = D->kv_create(E, cells);
    const std::vector<float> hk = rnd_vec((size_t) 10 * E, 2.0f), hv = rnd_vec((size_t) 10 * E, 2.0f);
    D->kv_store(kvh, 0, 10, hk.data(), hv.data());
    auto try_run = [&](uint32_t T, uint32_t pos0, const std::vector<float> & mask, const std::vector<kv_handle> & handles) {
        const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 1.0f), kv = rnd_vec((size_t) T * E, 1.0f);
        graph_inputs gi;
        gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { kv.data(), 0 }, { mask.data(), 0 } };
        std::vector<float> out((size_t) T * nh * d);
        graph_outputs go;
        go.outs = { { out.data(), 0 } };
        graph_state st;
        st.kv = handles;
        st.pos0 = pos0;
        D->graph_run(a.h, T, gi, go, st);
    };
    const std::vector<float> ok2 = cached_mask(2, 10, cells, 0);
    // a mask that shows a cell at or above pos0 + T
    {
        std::vector<float> m = ok2;
        m[(size_t) 1 * cells + 20] = 0.0f;
        bool inv = false;
        std::string text;
        try { try_run(2, 10, m, { kvh }); } catch (const std::exception & e) { text = e.what(); inv = dynamic_cast<const std::invalid_argument *>(&e) != nullptr; }
        CHECK(text.find("does not fit") != std::string::npos && text.find("mask row 1 shows cell 20") != std::string::npos && !inv,
              "a mask showing a cell at or above pos0 + T: '%s'", text.c_str());
        printf("  %s\n", text.c_str());
        // nothing was written
        std::vector<uint8_t> kb(2 * (E / 32 * 34)), vb(kb.size());
        D->kv_read(kvh, 10, 2, kb.data(), vb.data());
        bool zero = true;
        for (size_t i = 0; i < kb.size(); ++i) zero = zero && kb[i] == 0 && vb[i] == 0;
        CHECK(zero, "a refused run did not write the planes");
    }
    {
        std::vector<float> m = ok2;
        m[3] = 0.5f;
        CHECK(throws_with([&] { try_run(2, 10, m, { kvh }); }, "does not fit"), "a mask value 0.5: the error says does not fit");
    }
    CHECK(throws_with([&] { try_run(2, 63, ok2, { kvh }); }, "does not fit"), "pos0 + T above the cells: the error says does not fit");
    CHECK(throws_with([&] { try_run(65, 0, cached_mask(65, 0, cells, 0), { kvh }); }, "does not fit"), "T above the cells: the error says does not fit");
    // the bound objects
    bool inv = false;
    const kv_handle other_e = D->kv_create(2 * E, cells), other_c = D->kv_create(E, cells + 16);
    CHECK(throws_with([&] { try_run(2, 10, ok2, { other_e }); }, "KV slot 0 needs E", &inv) && inv, "a KV object of another E: invalid_argument");
    CHECK(throws_with([&] { try_run(2, 10, ok2, { other_c }); }, "KV slot 0 needs E", &inv) && inv, "a KV object of other cells: invalid_argument");
    CHECK(throws_with([&] { try_run(2, 10, ok2, {}); }, "0 KV handles given", &inv) && inv, "no handle: invalid_argument");
    CHECK(throws_with([&] { try_run(2, 10, ok2, { kvh, kvh }); }, "2 KV handles given", &inv) && inv, "two handles for one slot: invalid_argument");
    CHECK(throws_with([&] { try_run(2, 10, ok2, { kvh + 1000 }); }, "unknown KV handle", &inv) && inv, "an unknown handle: invalid_argument");
    D->kv_free(other_e);
    D->kv_free(other_c);
    // a freed object is refused
    {
        const kv_handle gone = D->kv_create(E, cells);
        D->kv_free(gone);
        CHECK(throws_with([&] { try_run(2, 10, ok2, { gone }); }, "unknown KV handle", &inv) && inv, "a freed KV object: invalid_argument");
    }
    // the graph still works, and a mask that shows a later cell of the same call (below pos0 + T) is taken
    {
        const uint32_t T = 3, pos0 = 10;
        std::vector<float> m = cached_mask(T, pos0, cells, 20);
        m[(size_t) 0 * cells + pos0 + 2] = 0.0f;            // row 0 sees the cell of row 2 (a non-causal mask)
        const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 2.0f);
        std::vector<float> ka = hk, va = hv;
        const std::vector<float> kn = rnd_vec((size_t) T * E, 2.0f), vn = rnd_vec((size_t) T * E, 2.0f);
        ka.insert(ka.end(), kn.begin(), kn.end());
        va.insert(va.end(), vn.begin(), vn.end());
        graph_state st;
        st.kv = { kvh };
        st.pos0 = pos0;
        result_t r;
        if (run_cmp(a.h, T, { in_arg(qv), in_arg(kn), in_arg(vn), mask_arg(m) }, "a mask that shows a later cell of the call", &r, 0, st)) {
            const std::vector<float> want = attn_direct_cached(qv, ka, va, nh, nkv, d, pos0, T, cells, a.scale, m);
            CHECK(same_bits(r.dev[0], want), "a mask that shows a later cell of the call: differs from the direct references");
        }
    }
    D->kv_free(kvh);
    D->graph_free(a.h);
    // a program that is malformed for slots: invalid_argument naming the op
    {
        auto bad = [&](const char * label, const std::function<void(PB &)> & build, const char * needle) {
            PB pb;
            build(pb);
            bool invalid = false;
            const bool ok = throws_with([&] { D->graph_unsupported(pb.g); }, needle, &invalid);
            CHECK(ok && invalid, "malformed (%s): invalid_argument with '%s'", label, needle);
        };
        bad("slot out of range", [](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), y = pb.tmp(128, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1, 1);
            pb.g.kv_slots = { 32 };
        }, "out of range");
        bad("slot used by two ops", [](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), y = pb.tmp(128, true), y2 = pb.tmp(128, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1, 0);
            pb.attn(q, k, v, y2, 2, 1, 64, 0.125f, -1, 0);
            pb.g.kv_slots = { 32 };
        }, "one op per slot");
        bad("slot used by no op", [](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), y = pb.tmp(128, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1, 0);
            pb.g.kv_slots = { 32, 32 };
        }, "used by no attn op");
        bad("mask cols != cells", [](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), m = pb.mask(33), y = pb.tmp(128, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, m, 0);
            pb.g.kv_slots = { 32 };
        }, "cols = 33");
        bad("cached op with a [T, T] mask", [](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), m = pb.mask(), y = pb.tmp(128, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, m, 0);
            pb.g.kv_slots = { 32 };
        }, "cols = 0");
        bad("own planes with a cols mask", [](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), m = pb.mask(32), y = pb.tmp(128, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, m, -1);
            pb.g.kv_size = 32;
        }, "belongs to cached attention");
        bad("own planes without kv_size", [](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), y = pb.tmp(128, true), q2 = pb.in(128), y2 = pb.tmp(128, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1, 0);
            pb.attn(q2, k, v, y2, 2, 1, 64, 0.125f, -1, -1);
            pb.g.kv_slots = { 32 };
        }, "needs kv_size >= 1");
        bad("cols on a non-mask value", [](PB & pb) {
            const int x = pb.in(64), y = pb.tmp(64, true);
            pb.g.values[(size_t) x].cols = 4;
            pb.op(gop::cpy, y).x = x;
        }, "only a mask value has cols");
    }
    // a graph without slots takes no handles
    {
        PB pb;
        const int x = pb.in(64), y = pb.tmp(64, true);
        pb.op(gop::cpy, y).x = x;
        const graph_handle h = create(pb.g, "cpy");
        const kv_handle k1 = D->kv_create(64, 16);
        if (h) {
            const std::vector<float> xv(64, 1.0f);
            std::vector<float> yv(64);
            graph_inputs gi;
            gi.args = { { xv.data(), 0 } };
            graph_outputs go;
            go.outs = { { yv.data(), 0 } };
            graph_state st;
            st.kv = { k1 };
            bool inv = false;
            CHECK(throws_with([&] { D->graph_run(h, 1, gi, go, st); }, "no KV slots", &inv) && inv, "handles for a graph without slots: invalid_argument");
            st.kv.clear();
            st.pos0 = 99;                                // ignored
            D->graph_run(h, 1, gi, go, st);
            CHECK(same_bits(yv, xv), "pos0 of a graph without slots is ignored");
            D->graph_free(h);
        }
        D->kv_free(k1);
    }
}

// ---- equal bytes from kv_store and from an attn run; negative controls
static void test_cached_store_and_controls() {
    printf("cached attention: kv_store against an attn run, negative controls\n");
    const uint32_t nh = 14, nkv = 2, d = 64, E = nkv * d, cells = 96;
    const att_prog a = make_att(nh, nkv, d, cells, false, "store and controls");
    if (!a.h) return;
    // (1) the same rows stored by kv_store into one object and by a cached attn run into another: the planes are byte-equal
    {
        const uint32_t n = 37, c0 = 11;
        const kv_handle ks = D->kv_create(E, cells), kr = D->kv_create(E, cells);
        const std::vector<float> hk = rnd_vec((size_t) c0 * E, 2.0f), hv = rnd_vec((size_t) c0 * E, 2.0f);
        D->kv_store(ks, 0, c0, hk.data(), hv.data());
        D->kv_store(kr, 0, c0, hk.data(), hv.data());
        const std::vector<float> qv = rnd_vec((size_t) n * nh * d, 2.0f), kn = rnd_vec((size_t) n * E, 2.0f), vn = rnd_vec((size_t) n * E, 2.0f);
        D->kv_store(ks, c0, n, kn.data(), vn.data());
        graph_state st;
        st.kv = { kr };
        st.pos0 = c0;
        result_t r;
        run_cmp(a.h, n, { in_arg(qv), in_arg(kn), in_arg(vn) }, "rows stored by an attn run", &r, 0, st);
        const size_t rb = (size_t) E / 32 * 34;
        std::vector<uint8_t> k1(cells * rb), v1(cells * rb), k2(cells * rb), v2(cells * rb);
        D->kv_read(ks, 0, cells, k1.data(), v1.data());
        D->kv_read(kr, 0, cells, k2.data(), v2.data());
        size_t bad = SIZE_MAX;
        for (size_t i = 0; i < k1.size() && bad == SIZE_MAX; ++i) if (k1[i] != k2[i] || v1[i] != v2[i]) bad = i;
        CHECK(bad == SIZE_MAX, "kv_store and a cached attn run store other bytes (first difference at byte %zu of the %zu cell rows)", bad, (size_t) cells);
        // and the cells above the rows are untouched (zero) in both
        bool zero = true;
        for (size_t i = (size_t) (c0 + n) * rb; i < k2.size(); ++i) zero = zero && k2[i] == 0 && v2[i] == 0;
        CHECK(zero, "an attn run wrote cells above pos0 + T");
        D->kv_free(ks);
        D->kv_free(kr);
    }
    // (2) negative controls: the direct reference (history by tsr_qact + tsr_ref_kvst, then the kinds' encoders) agrees with the run, and
    //     disagrees when the run's pos0 is wrong or when the reference is given a wrong cell index or a wrong mask offset
    {
        const uint32_t pos0 = 20, T = 3;
        const kv_handle kvh = D->kv_create(E, cells);
        const std::vector<float> kall = rnd_vec((size_t) (pos0 + T) * E, 2.0f), vall = rnd_vec((size_t) (pos0 + T) * E, 2.0f);
        D->kv_store(kvh, 0, pos0, kall.data(), vall.data());
        const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 2.0f);
        const std::vector<float> kn(kall.begin() + (size_t) pos0 * E, kall.end()), vn(vall.begin() + (size_t) pos0 * E, vall.end());
        graph_state st;
        st.kv = { kvh };
        st.pos0 = pos0;
        result_t r;
        run_cmp(a.h, T, { in_arg(qv), in_arg(kn), in_arg(vn) }, "controls: the run", &r, 0, st);
        const std::vector<float> none;
        CHECK(same_bits(r.dev[0], attn_direct_cached(qv, kall, vall, nh, nkv, d, pos0, T, cells, a.scale, none)), "controls: the right pos0 equals the direct references");
        CHECK(!same_bits(r.dev[0], attn_direct_cached(qv, kall, vall, nh, nkv, d, pos0, T, cells, a.scale, none, 1, 0)),
              "controls: a reference with a cell index one too high must differ");
        CHECK(!same_bits(r.dev[0], attn_direct_cached(qv, kall, vall, nh, nkv, d, pos0, T, cells, a.scale, none, 0, -1)),
              "controls: a reference with a causal mask one cell too short must differ");
        CHECK(!same_bits(r.dev[0], attn_direct_cached(qv, kall, vall, nh, nkv, d, pos0, T, cells, a.scale, none, 0, 1)),
              "controls: a reference with a causal mask one cell too long must differ");
        // a run at another pos0 (the planes hold the history at cells 0 .. 19 only; cell 20 .. are written by the run itself)
        graph_state wrong = st;
        wrong.pos0 = pos0 + 1;
        result_t rw;
        run_cmp(a.h, T, { in_arg(qv), in_arg(kn), in_arg(vn) }, "controls: wrong pos0", &rw, 0, wrong);
        CHECK(!same_bits(r.dev[0], rw.dev[0]), "controls: a run at pos0 + 1 must give other bits");
        printf("  control: NMSE of a run at pos0 + 1 against the run at pos0: %.3g\n", nmse(rw.dev[0], r.dev[0]));
        CHECK(nmse(rw.dev[0], r.dev[0]) > 1e3 * NMSE_BOUND, "controls: the wrong pos0 is far above the bound");
        D->kv_free(kvh);
    }
    D->graph_free(a.h);
}

// ---- chunks, plan counters, graph_info with slots
static void test_cached_chunks() {
    printf("cached attention: chunks, counters, graph_info\n");
    const uint32_t nh = 2, nkv = 1, d = 64, E = nkv * d, cells = 1536, R = 1500;
    const att_prog a = make_att(nh, nkv, d, cells, true, "chunks");
    if (!a.h) return;
    const std::vector<float> qv = rnd_vec((size_t) R * nh * d, 2.0f), kv = rnd_vec((size_t) R * E, 2.0f), vv = rnd_vec((size_t) R * E, 2.0f);
    const std::vector<float> M = cached_mask(R, 0, cells, 10);
    auto rows = [&](const std::vector<float> & x, uint32_t w, uint32_t r0, uint32_t n) { return std::vector<float>(x.begin() + (size_t) r0 * w, x.begin() + (size_t) (r0 + n) * w); };
    // the shape reports the slot, graph_info before any run uses fake offsets
    CHECK(D->graph_shape_of(a.h).kv_slots == std::vector<uint32_t>({ cells }), "graph_shape: kv_slots");
    CHECK(D->graph_shape_of(a.h).kv_bytes == 0, "graph_shape: a graph with only slots has no planes of its own");
    CHECK(D->graph_info(a.h, 8).lists >= 1, "graph_info of a graph with slots before any run");
    CHECK(throws_with([&] { D->graph_info(a.h, cells + 1); }, "does not fit"), "graph_info above the cells of the slot: does not fit");
    CHECK(D->graph_info(a.h, graph_max_attn_tokens + 1).T <= graph_max_attn_tokens, "graph_info above graph_max_attn_tokens: the plan of a chunk");
    // one call of 1500 rows: chunks 680, 680, 140 (counters on a cold graph, plain graph_run)
    const kv_handle k1 = D->kv_create(E, cells);
    std::vector<float> one((size_t) R * nh * d);
    {
        graph_inputs gi;
        gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { vv.data(), 0 }, { M.data(), 0 } };
        graph_outputs go;
        go.outs = { { one.data(), 0 } };
        graph_state st;
        st.kv = { k1 };
        const counters c0 = D->get_counters();
        D->graph_run(a.h, R, gi, go, st);
        const counters c1 = D->get_counters();
        const graph_plan_info i680 = D->graph_info(a.h, 680), i140 = D->graph_info(a.h, 140);
        // graph_info of the run's T: the plan of its first chunk, which the run already made
        {
            const counters b0 = D->get_counters();
            const graph_plan_info ir = D->graph_info(a.h, R);
            const counters b1 = D->get_counters();
            CHECK(ir.T == 680 && ir.T < R, "graph_info(h, %u) after a chunked run: T %u (the first chunk, 680 rows)", R, ir.T);
            CHECK(ir.lists == i680.lists && ir.records == i680.records && ir.joins == i680.joins && ir.io_bytes == i680.io_bytes &&
                  memcmp(ir.by_kind, i680.by_kind, sizeof(ir.by_kind)) == 0, "graph_info of a chunked run's T equals graph_info of the chunk's T");
            CHECK(b1.graph_plan_misses == b0.graph_plan_misses, "graph_info of a chunked run's T makes no new plan (%" PRIu64 " misses)",
                  b1.graph_plan_misses - b0.graph_plan_misses);
        }
        CHECK(c1.graph_runs - c0.graph_runs == 1, "a chunked run counts once in graph_runs (%" PRIu64 ")", c1.graph_runs - c0.graph_runs);
        CHECK(c1.graph_launches - c0.graph_launches == 2ull * i680.lists + i140.lists, "graph_launches %" PRIu64 " (lists of the chunks 680, 680, 140: %u)",
              c1.graph_launches - c0.graph_launches, 2 * i680.lists + i140.lists);
        CHECK(c1.graph_plan_misses - c0.graph_plan_misses == 2 && c1.graph_plan_hits - c0.graph_plan_hits == 1, "chunk plans: %" PRIu64 " misses, %" PRIu64 " hits (2 and 1 expected)",
              c1.graph_plan_misses - c0.graph_plan_misses, c1.graph_plan_hits - c0.graph_plan_hits);
        CHECK(c1.execs - c0.execs == c1.graph_launches - c0.graph_launches, "execs equal the graph launches");
    }
    // the same rows in three calls of 500 rows on another object: bit-equal outputs, byte-equal planes
    {
        const kv_handle k2 = D->kv_create(E, cells);
        std::vector<float> three;
        for (uint32_t r0 = 0; r0 < R; r0 += 500) {
            graph_state st;
            st.kv = { k2 };
            st.pos0 = r0;
            result_t r;
            const std::vector<float> mm = rows(M, cells, r0, 500);
            run_cmp(a.h, 500, { in_arg(rows(qv, nh * d, r0, 500)), in_arg(rows(kv, E, r0, 500)), in_arg(rows(vv, E, r0, 500)), mask_arg(mm) }, "500 rows", &r, 0, st);
            three.insert(three.end(), r.dev[0].begin(), r.dev[0].end());
        }
        size_t first = 0;
        CHECK(same_bits(one, three, &first), "1500 rows in one call (chunks of 680) differ from three calls of 500 rows at %zu", first);
        kv_cells_equal(k1, 0, R, kv, vv, E, 0, "the cells of the chunked run");
        D->kv_free(k2);
    }
    // the same call under run_cmp (device == reference over the chunks, one image)
    {
        const kv_handle k3 = D->kv_create(E, cells);
        graph_state st;
        st.kv = { k3 };
        result_t r;
        run_cmp(a.h, R, { in_arg(qv), in_arg(kv), in_arg(vv), mask_arg(M) }, "1500 rows, device and reference", &r, 0, st);
        CHECK(same_bits(r.dev[0], one), "a second chunked run on a new object gives other bits");
        D->kv_free(k3);
    }
    // a given mask that shows a cell above the end of its row's chunk is refused (a cell the chunk does not write)
    {
        std::vector<float> m = M;
        m[(size_t) 3 * cells + 700] = 0.0f;           // row 3 is in chunk [0, 680): cell 700 is above its end
        graph_inputs gi;
        gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { vv.data(), 0 }, { m.data(), 0 } };
        graph_outputs go;
        go.outs = { { one.data(), 0 } };
        graph_state st;
        st.kv = { k1 };
        std::string text;
        try { D->graph_run(a.h, R, gi, go, st); } catch (const std::exception & e) { text = e.what(); }
        CHECK(text.find("does not fit") != std::string::npos && text.find("mask row 3 shows cell 700") != std::string::npos && text.find("chunk") != std::string::npos,
              "a mask showing a cell above its chunk: '%s'", text.c_str());
    }
    // pos0 + T above the cells
    {
        graph_inputs gi;
        gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { vv.data(), 0 }, { M.data(), 0 } };
        graph_outputs go;
        go.outs = { { one.data(), 0 } };
        graph_state st;
        st.kv = { k1 };
        st.pos0 = 100;
        CHECK(throws_with([&] { D->graph_run(a.h, R, gi, go, st); }, "does not fit"), "a chunked run with pos0 + T above the cells");
    }
    // a small IO region: the chunk is halved until its plan fits; the bits equal those of the big IO region
    {
        open_options o;
        o.io_bytes = 1 << 18;
        device small(o);
        device * saved = D;
        D = &small;
        const att_prog s = make_att(nh, nkv, d, cells, true, "chunks, small IO");
        if (s.h) {
            const kv_handle ks = small.kv_create(E, cells);
            graph_state st;
            st.kv = { ks };
            result_t r;
            const counters c0 = small.get_counters();
            run_cmp(s.h, R, { in_arg(qv), in_arg(kv), in_arg(vv), mask_arg(M) }, "1500 rows in a small IO region", &r, 0, st);
            const counters c1 = small.get_counters();
            CHECK(same_bits(r.dev[0], one), "a small IO region (smaller chunks) gives other bits than chunks of 680");
            CHECK(c1.graph_launches - c0.graph_launches >= 6, "a small IO region runs more chunks (%" PRIu64 " launches)", c1.graph_launches - c0.graph_launches);
            // a mask that shows a cell of the next chunk is refused with the chunk that the small region sets, not with 680
            std::vector<float> m = M;
            m[(size_t) 0 * cells + 300] = 0.0f;
            graph_inputs gi;
            gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { vv.data(), 0 }, { m.data(), 0 } };
            graph_outputs go;
            std::vector<float> out((size_t) R * nh * d);
            go.outs = { { out.data(), 0 } };
            CHECK(throws_with([&] { small.graph_run(s.h, R, gi, go, st); }, "does not fit"), "small IO: a mask showing a cell of a later chunk");
            small.graph_free(s.h);
        }
        D = saved;
    }
    {
        // a region too small for one row
        open_options o;
        o.io_bytes = 1 << 10;
        device tiny(o);
        device * saved = D;
        D = &tiny;
        PB pb;
        const int q = pb.in(nh * d), k = pb.in(E), v = pb.in(E), y = pb.tmp(nh * d, true);
        pb.attn(q, k, v, y, nh, nkv, d, 0.125f, -1, 0);
        pb.g.kv_slots = { 64 };
        bool refused = false;
        try { (void) tiny.graph_create(pb.g); } catch (const std::exception & e) { refused = strstr(e.what(), "does not fit") != nullptr || strstr(e.what(), "IO") != nullptr; }
        CHECK(refused, "an IO region too small for one row is refused");
        D = saved;
    }
    D->kv_free(k1);
    D->graph_free(a.h);
    // a decoder layer in chunks: the rows of the inputs, the positions and the outputs follow the chunk
    {
        const uint32_t T = 700, lcells = 768;
        layer_w lw = make_layer();
        PB pb;
        const int x = pb.in(Q_EMBD), p = pb.pos();
        add_layer(pb, lw, x, p, true, 0);
        pb.g.kv_slots = { lcells };
        const graph_handle h = create(pb.g, "layer in chunks");
        if (h) {
            const std::vector<float> xv = rnd_vec((size_t) T * Q_EMBD, 1.0f);
            const kv_handle ka = D->kv_create(Q_KV * Q_D, lcells), kb = D->kv_create(Q_KV * Q_D, lcells);
            graph_state sa;
            sa.kv = { ka };
            result_t r;
            const counters c0 = D->get_counters();
            run_cmp(h, T, { in_arg(xv), pos_arg(iota(T)) }, "a layer of 700 rows (chunks 680, 20)", &r, 0, sa);
            CHECK(D->get_counters().graph_launches - c0.graph_launches == D->graph_info(h, 680).lists + D->graph_info(h, 20).lists,
                  "a layer of 700 rows runs the lists of the chunks 680 and 20 (%" PRIu64 " launches)", D->get_counters().graph_launches - c0.graph_launches);
            std::vector<float> halves;
            for (uint32_t r0 = 0; r0 < T; r0 += 350) {
                graph_state sb;
                sb.kv = { kb };
                sb.pos0 = r0;
                std::vector<int32_t> pv(350);
                for (uint32_t t = 0; t < 350; ++t) pv[t] = (int32_t) (r0 + t);
                result_t r2;
                run_cmp(h, 350, { in_arg(rows(xv, Q_EMBD, r0, 350)), pos_arg(pv) }, "a layer of 350 rows", &r2, 0, sb);
                halves.insert(halves.end(), r2.dev[0].begin(), r2.dev[0].end());
            }
            size_t first = 0;
            CHECK(sane(r.dev[0]) && same_bits(r.dev[0], halves, &first), "a layer of 700 rows in chunks differs from two calls of 350 rows at %zu", first);
            D->kv_free(ka);
            D->kv_free(kb);
            D->graph_free(h);
        }
        free_layer(lw);
    }
}

// ---- graph_info of a graph with slots, mixed graphs, chunks at pos0 > 0, one object in two slots
static std::vector<float> rows_of(const std::vector<float> & x, uint32_t w, uint32_t r0, uint32_t n) {
    return std::vector<float>(x.begin() + (size_t) r0 * w, x.begin() + (size_t) (r0 + n) * w);
}

static void kv_planes_equal(kv_handle a, kv_handle b, uint32_t n, uint32_t E, const char * what) {
    const size_t rb = (size_t) E / 32 * 34;
    std::vector<uint8_t> ka(n * rb), va(n * rb), kb(n * rb), vb(n * rb);
    D->kv_read(a, 0, n, ka.data(), va.data());
    D->kv_read(b, 0, n, kb.data(), vb.data());
    CHECK(ka == kb && va == vb, "%s: the planes differ", what);
}

static void test_graph_info_slots() {
    printf("cached attention: graph_info does not depend on pos0, is refused above the cells\n");
    const uint32_t nh = 2, nkv = 1, d = 64, E = 64, cells = 1024;
    const att_prog a = make_att(nh, nkv, d, cells, false, "info");
    if (!a.h) return;
    const kv_handle k = D->kv_create(E, cells);
    const uint32_t T = 700;
    const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 1.0f), kv = rnd_vec((size_t) T * E, 1.0f);
    graph_inputs gi;
    gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { kv.data(), 0 } };
    std::vector<float> out((size_t) T * nh * d);
    graph_outputs go;
    go.outs = { { out.data(), 0 } };
    graph_state st;
    st.kv = { k };
    D->graph_run(a.h, T, gi, go, st);
    const graph_plan_info i700 = D->graph_info(a.h, T), i680 = D->graph_info(a.h, 680);
    CHECK(i700.T == 680 && i700.records == i680.records && i700.lists == i680.lists, "graph_info(h, 700) after a run of 700 rows: the plan of the chunk of 680 rows (T %u)", i700.T);
    // a run at the last cell, then graph_info of two rows
    st.pos0 = cells - 1;
    D->graph_run(a.h, 1, gi, go, st);
    graph_plan_info i2;
    try { i2 = D->graph_info(a.h, 2); } catch (const std::exception & e) { CHECK(false, "graph_info(h, 2) after a run at pos0 %u: %s", cells - 1, e.what()); }
    CHECK(i2.T == 2 && i2.lists >= 1, "graph_info(h, 2) after a run at pos0 %u: T %u", cells - 1, i2.T);
    CHECK(throws_with([&] { D->graph_info(a.h, cells + 1); }, "does not fit"), "graph_info above the cells of the slot is refused");
    CHECK(throws_with([&] { D->graph_info(a.h, 0); }, "T must be >= 1"), "graph_info of T = 0 is refused");
    D->kv_free(k);
    D->graph_free(a.h);
    // a graph with an attn op that has no slot: T above graph_max_attn_tokens is refused as before, with or without a slot beside it
    {
        PB pb;
        const int q = pb.in(nh * d), k2 = pb.in(E), v2 = pb.in(E), y1 = pb.tmp(nh * d, true), y2 = pb.tmp(nh * d, true);
        pb.attn(q, k2, v2, y1, nh, nkv, d, 0.125f, -1, 0);
        pb.attn(q, k2, v2, y2, nh, nkv, d, 0.125f, -1);
        pb.g.kv_slots = { cells };
        pb.g.kv_size = 1024;
        const graph_handle hm = create(pb.g, "mixed info");
        if (hm) {
            CHECK(throws_with([&] { D->graph_info(hm, 700); }, "does not fit an attn op"), "graph_info of a mixed graph above graph_max_attn_tokens: does not fit");
            CHECK(D->graph_info(hm, 680).T == 680, "graph_info of a mixed graph at graph_max_attn_tokens");
            D->graph_free(hm);
        }
    }
}

// A chunked run that starts at pos0 > 0, causal, with a given mask and with a strided mask, against unchunked calls of 500 rows.
static void test_chunk_at_pos0(bool given, uint32_t mstride_pad) {
    printf("cached attention: a chunked run at pos0 > 0, %s mask, mask stride pad %u\n", given ? "given" : "causal", mstride_pad);
    const uint32_t nh = 2, nkv = 1, d = 64, E = nkv * d, cells = 1536, P = 100, R = 1000, N = P + R;
    const att_prog a = make_att(nh, nkv, d, cells, given, "chunk at pos0");
    if (!a.h) return;
    const std::vector<float> qv = rnd_vec((size_t) N * nh * d, 2.0f), kv = rnd_vec((size_t) N * E, 2.0f), vv = rnd_vec((size_t) N * E, 2.0f);
    const std::vector<float> M = cached_mask(N, 0, cells, given ? 10u : 0u);
    auto args_of = [&](uint32_t r0, uint32_t n) {
        std::vector<arg_t> args = { in_arg(rows_of(qv, nh * d, r0, n)), in_arg(rows_of(kv, E, r0, n)), in_arg(rows_of(vv, E, r0, n)) };
        if (given) args.push_back(mask_arg(rows_of(M, cells, r0, n)));
        return args;
    };
    const kv_handle k1 = D->kv_create(E, cells), k2 = D->kv_create(E, cells);
    graph_state s1, s2;
    s1.kv = { k1 };
    s2.kv = { k2 };
    result_t r;
    run_cmp(a.h, P, args_of(0, P), "prefix (object 1)", &r, 0, s1);
    run_cmp(a.h, P, args_of(0, P), "prefix (object 2)", &r, 0, s2);
    // object 1: one call of R rows at pos0 = P (chunks of 680 and 320 rows); the mask may be strided
    std::vector<float> one((size_t) R * nh * d);
    {
        s1.pos0 = P;
        const std::vector<float> q1 = rows_of(qv, nh * d, P, R), k1v = rows_of(kv, E, P, R), v1v = rows_of(vv, E, P, R);
        std::vector<float> ms;
        graph_inputs gi;
        gi.args = { { q1.data(), 0 }, { k1v.data(), 0 }, { v1v.data(), 0 } };
        if (given) {
            ms.assign((size_t) R * (cells + mstride_pad), 12345.0f);
            for (uint32_t t = 0; t < R; ++t) memcpy(&ms[(size_t) t * (cells + mstride_pad)], &M[(size_t) (P + t) * cells], (size_t) cells * 4);
            gi.args.push_back({ ms.data(), mstride_pad ? (size_t) cells + mstride_pad : 0 });
        }
        graph_outputs go;
        go.outs = { { one.data(), 0 } };
        const counters c0 = D->get_counters();
        try { D->graph_run(a.h, R, gi, go, s1); } catch (const std::exception & e) { CHECK(false, "chunked run at pos0 %u: %s", P, e.what()); }
        CHECK(D->get_counters().graph_launches - c0.graph_launches >= 2, "a run of %u rows is cut into chunks", R);
    }
    // device == reference over the chunks (a third object with the same prefix)
    {
        const kv_handle k3 = D->kv_create(E, cells);
        graph_state s3;
        s3.kv = { k3 };
        run_cmp(a.h, P, args_of(0, P), "prefix (object 3)", &r, 0, s3);
        s3.pos0 = P;
        result_t rr;
        run_cmp(a.h, R, args_of(P, R), "chunked at pos0 > 0 (device and reference)", &rr, 0, s3);
        size_t first = 0;
        CHECK(same_bits(rr.dev[0], one, &first), "a chunked run at pos0 > 0 gives other bits with a strided or contiguous mask at %zu", first);
        D->kv_free(k3);
    }
    // object 2: calls of 500 rows (no chunks)
    std::vector<float> two;
    for (uint32_t r0 = P; r0 < N; r0 += 500) {
        s2.pos0 = r0;
        result_t rr;
        run_cmp(a.h, 500, args_of(r0, 500), "500 rows", &rr, 0, s2);
        two.insert(two.end(), rr.dev[0].begin(), rr.dev[0].end());
    }
    size_t first = 0;
    CHECK(same_bits(one, two, &first), "a chunked run at pos0 %u differs from calls of 500 rows at %zu", P, first);
    kv_planes_equal(k1, k2, N, E, "a chunked run at pos0 > 0 against calls of 500 rows");
    D->kv_free(k1);
    D->kv_free(k2);
    D->graph_free(a.h);
}

// A graph with one attn op that has a KV slot and one that has its own planes, same q, k, v: each output against the graph of that
// op alone, over a prefill and decode steps at pos0 > 0.
static void test_mixed_graph(bool given) {
    printf("cached attention: a graph with a cached and an own-plane attn op, %s masks\n", given ? "given" : "causal");
    const uint32_t nh = 2, nkv = 1, d = 64, E = nkv * d, cells = 64;
    const float sc = 0.125f;
    PB pb;
    const int q = pb.in(nh * d), k = pb.in(E), v = pb.in(E);
    const int mc = given ? pb.mask(cells) : -1, mo = given ? pb.mask(0) : -1;
    const int y1 = pb.tmp(nh * d, true), y2 = pb.tmp(nh * d, true);
    pb.attn(q, k, v, y1, nh, nkv, d, sc, mc, 0);
    pb.attn(q, k, v, y2, nh, nkv, d, sc, mo);
    pb.g.kv_slots = { cells };
    pb.g.kv_size = 16;
    const graph_handle hm = create(pb.g, "mixed");
    if (!hm) return;
    const att_prog ac = make_att(nh, nkv, d, cells, given, "cached alone");
    PB po;
    {
        const int q2 = po.in(nh * d), k2 = po.in(E), v2 = po.in(E), m2 = given ? po.mask(0) : -1, y = po.tmp(nh * d, true);
        po.attn(q2, k2, v2, y, nh, nkv, d, sc, m2);
        po.g.kv_size = 16;
    }
    const graph_handle ho = create(po.g, "own alone");
    const kv_handle ka = D->kv_create(E, cells), kb = D->kv_create(E, cells);
    uint32_t pos0 = 0;
    for (uint32_t T : { 5u, 1u, 3u, 1u }) {
        const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 2.0f), kv = rnd_vec((size_t) T * E, 2.0f), vv = rnd_vec((size_t) T * E, 2.0f);
        const std::vector<float> mcv = given ? cached_mask(T, pos0, cells, 20) : std::vector<float>();
        std::vector<float> mov;
        if (given) {
            mov.assign((size_t) T * T, -INFINITY);
            for (uint32_t r = 0; r < T; ++r) for (uint32_t j = 0; j < T; ++j) if (j == r || rnd_u(2)) mov[(size_t) r * T + j] = 0.0f;
        }
        std::vector<arg_t> am = { in_arg(qv), in_arg(kv), in_arg(vv) };
        if (given) { am.push_back(mask_arg(mcv)); am.push_back(mask_arg(mov)); }
        graph_state sm;
        sm.kv = { ka };
        sm.pos0 = pos0;
        result_t rm, rc, ro;
        char w[96];
        snprintf(w, sizeof(w), "mixed pos0 %u T %u", pos0, T);
        run_cmp(hm, T, am, w, &rm, 0, sm);
        std::vector<arg_t> ac_args = { in_arg(qv), in_arg(kv), in_arg(vv) };
        if (given) ac_args.push_back(mask_arg(mcv));
        graph_state sa;
        sa.kv = { kb };
        sa.pos0 = pos0;
        run_cmp(ac.h, T, ac_args, "cached alone", &rc, 0, sa);
        std::vector<arg_t> ao_args = { in_arg(qv), in_arg(kv), in_arg(vv) };
        if (given) ao_args.push_back(mask_arg(mov));
        run_cmp(ho, T, ao_args, "own alone", &ro);
        if (!rm.dev.empty() && !rc.dev.empty() && !ro.dev.empty()) {
            CHECK(same_bits(rm.dev[0], rc.dev[0]), "%s: the cached op's output differs from the cached op alone", w);
            CHECK(same_bits(rm.dev[1], ro.dev[0]), "%s: the own-plane op's output differs from that op alone", w);
        }
        pos0 += T;
    }
    D->kv_free(ka);
    D->kv_free(kb);
    D->graph_free(hm);
    D->graph_free(ac.h);
    D->graph_free(ho);
}

static void test_kv_in_two_slots() {
    printf("cached attention: one KV object in two slots\n");
    const uint32_t cells = 64;
    PB pb;
    const int q1 = pb.in(128), k1 = pb.in(64), v1 = pb.in(64), q2 = pb.in(128), k2 = pb.in(64), v2 = pb.in(64);
    const int y1 = pb.tmp(128, true), y2 = pb.tmp(128, true);
    pb.attn(q1, k1, v1, y1, 2, 1, 64, 0.125f, -1, 0);
    pb.attn(q2, k2, v2, y2, 2, 1, 64, 0.125f, -1, 1);
    pb.g.kv_slots = { cells, cells };
    const graph_handle h = create(pb.g, "two slots");
    if (!h) return;
    const kv_handle ka = D->kv_create(64, cells), kb = D->kv_create(64, cells);
    const std::vector<float> a = rnd_vec(128, 1.0f), b = rnd_vec(64, 1.0f);
    graph_inputs gi;
    gi.args = { { a.data(), 0 }, { b.data(), 0 }, { b.data(), 0 }, { a.data(), 0 }, { b.data(), 0 }, { b.data(), 0 } };
    std::vector<float> o1(128), o2(128);
    graph_outputs go;
    go.outs = { { o1.data(), 0 }, { o2.data(), 0 } };
    graph_state st;
    st.kv = { ka, ka };
    bool invalid = false;
    CHECK(throws_with([&] { D->graph_run(h, 1, gi, go, st); }, "is bound to the slots 0 and 1", &invalid) && invalid, "one KV object in two slots: invalid_argument naming the slots");
    st.kv = { ka, kb };
    D->graph_run(h, 1, gi, go, st);
    D->kv_free(ka);
    D->kv_free(kb);
    D->graph_free(h);
}

// ---------------------------------------------------------------------------------------------
// Limits, refusals, memory, threads
// ---------------------------------------------------------------------------------------------
static void test_limits() {
    printf("limits and refusals\n");
    // T above kv_size, T above the attn limit, a mask value that is neither 0 nor -inf
    {
        PB pb;
        const uint32_t nh = 2, nkv = 1, d = 64;
        const int q = pb.in(nh * d), k = pb.in(nkv * d), v = pb.in(nkv * d), m = pb.mask(), y = pb.tmp(nh * d, true);
        pb.attn(q, k, v, y, nh, nkv, d, 0.125f, m);
        pb.g.kv_size = 700;
        const graph_handle h = create(pb.g, "limits graph");
        if (h) {
            auto run = [&](uint32_t T, float mval) {
                const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 1.0f), kv = rnd_vec((size_t) T * d, 1.0f);
                std::vector<float> mask = make_mask(MK_USER_CAUSAL, T);
                if (mval != 0.0f) mask[1] = mval;
                graph_inputs gi;
                gi.args = { { qv.data(), 0 }, { kv.data(), 0 }, { kv.data(), 0 }, { mask.data(), 0 } };
                std::vector<float> out((size_t) T * nh * d);
                graph_outputs go;
                go.outs = { { out.data(), 0 } };
                D->graph_run(h, T, gi, go);
            };
            CHECK(throws_with([&] { run(graph_max_attn_tokens + 1, 0.0f); }, "does not fit"), "T above the attn limit: the error says does not fit");
            CHECK(throws_with([&] { run(701, 0.0f); }, "does not fit"), "T above kv_size: the error says does not fit");
            CHECK(throws_with([&] { run(5, 0.5f); }, "does not fit"), "a mask value 0.5: the error says does not fit");
            CHECK(throws_with([&] { run(0, 0.0f); }, "T must be"), "T = 0");
            // and the graph still works after the refusals
            std::vector<arg_t> args = { in_arg(rnd_vec(5 * nh * d, 1.0f)), in_arg(rnd_vec(5 * d, 1.0f)), in_arg(rnd_vec(5 * d, 1.0f)),
                                        mask_arg(make_mask(MK_USER_CAUSAL, 5)) };
            run_cmp(h, 5, args, "after the refusals");
            D->graph_free(h);
        }
    }
    // the IO region: a device opened with a small IO area
    {
        open_options o;
        o.io_bytes = 1 << 16;
        device small(o);
        std::vector<float> w((size_t) 256 * 256, 0.01f);
        const weight_handle wh = small.upload(dtype::f32, w.data(), 256, 256);
        PB pb;
        const int x = pb.in(256), y = pb.tmp(256, true);
        pb.mul_mat(wh, x, y);
        const graph_handle h = small.graph_create(pb.g);
        const uint32_t T = 100;
        std::vector<float> xv((size_t) T * 256, 1.0f), yv((size_t) T * 256);
        graph_inputs gi;
        gi.args = { { xv.data(), 0 } };
        graph_outputs go;
        go.outs = { { yv.data(), 0 } };
        CHECK(throws_with([&] { small.graph_run(h, T, gi, go); }, "does not fit"), "the IO region is too small: the error says does not fit");
        small.graph_run(h, 10, gi, go);   // 10 tokens fit
        small.graph_free(h);
    }
    // a program the device cannot run: NEOX RoPE of n_dims 48 (the encoder refuses it); the reason names the op
    {
        PB pb;
        std::vector<float> tab(24, 1.0f);
        const int x = pb.in(64), t = pb.cst(up_const(tab, 1, 24), 24), p = pb.pos(), y = pb.tmp(64, true);
        pb.rope(x, y, 1, 64, 48, t, p);
        const char * why = D->graph_unsupported(pb.g);
        CHECK(why != nullptr && strstr(why, "op #0") && strstr(why, "tag 1000") && strstr(why, "rope"), "unsupported rope: %s", why ? why : "(supported)");
        printf("  unsupported: %s\n", why ? why : "-");
        CHECK(throws_with([&] { D->graph_create(pb.g); }, "rope"), "graph_create of it throws");
    }
    // an N part that is not a multiple of 16
    {
        weight_options o;
        o.max_part_rows = 40;
        const W w = up_weight(100, 64, 0.1f, o);
        PB pb;
        const int x = pb.in(64), y = pb.tmp(100, true);
        pb.mul_mat(w.h, x, y);
        const char * why = D->graph_unsupported(pb.g);
        CHECK(why != nullptr && strstr(why, "multiple of 16") && strstr(why, "mul_mat"), "N parts of 40 rows: %s", why ? why : "(supported)");
    }
    // a weight by shape only (handle 0): a Q8_0 weight [N, K] checked as such
    {
        PB pb;
        const int x = pb.in(128), y = pb.tmp(64, true);
        pb.mul_mat(0, x, y);
        CHECK(D->graph_unsupported(pb.g) == nullptr, "a mul_mat with w = 0 is checked by shape");
        PB pb2;
        const int x2 = pb2.in(100), y2 = pb2.tmp(64, true);   // K = 100 is not a multiple of 32
        pb2.mul_mat(0, x2, y2);
        const char * why = D->graph_unsupported(pb2.g);
        CHECK(why != nullptr && strstr(why, "mul_mat"), "K = 100: %s", why ? why : "(supported)");
        CHECK(throws_with([&] { D->graph_create(pb.g); }, "weight handle 0"), "graph_create needs the weight handle");
    }
    // malformed programs: invalid_argument, naming the op
    {
        auto bad = [&](const char * label, const std::function<void(PB &)> & build, const char * needle) {
            PB pb;
            build(pb);
            bool invalid = false;
            const bool ok = throws_with([&] { D->graph_unsupported(pb.g); }, needle, &invalid);
            CHECK(ok && invalid, "malformed (%s): invalid_argument with '%s'", label, needle);
        };
        bad("read before written", [](PB & pb) { const int t = pb.tmp(64), y = pb.tmp(64, true); pb.op(gop::cpy, y).x = t; }, "read before it is written");
        bad("dst is an operand", [](PB & pb) { const int x = pb.in(64), y = pb.tmp(64, true); pb.ew(gop::add, x, y, y); }, "read before");
        bad("two writers", [](PB & pb) {
            const int x = pb.in(64), y = pb.tmp(64, true);
            pb.op(gop::cpy, y).x = x;
            pb.op(gop::cpy, y).x = x;
        }, "written by an earlier op");
        bad("shape", [](PB & pb) { const int x = pb.in(64), y = pb.tmp(32, true); pb.op(gop::cpy, y).x = x; }, "values per row");
        bad("dst not tmp", [](PB & pb) { const int x = pb.in(64), y = pb.in(64); pb.tmp(64, true); pb.op(gop::cpy, y).x = x; }, "dst must be a tmp");
        bad("no out", [](PB & pb) { const int x = pb.in(64), y = pb.tmp(64); pb.op(gop::cpy, y).x = x; }, "no out value");
        bad("bad id", [](PB & pb) { const int y = pb.tmp(64, true); pb.op(gop::cpy, y).x = 17; }, "out of range");
        bad("glu variant", [](PB & pb) {
            const int a = pb.in(64), b = pb.in(64), y = pb.tmp(64, true);
            graph_op & o = pb.op(gop::glu, y);
            o.gate = a; o.up = b; o.variant = 1;
        }, "variant");
        bad("attn heads", [](PB & pb) {
            const int q = pb.in(192), k = pb.in(64), v = pb.in(64), y = pb.tmp(192, true);
            pb.attn(q, k, v, y, 3, 2, 64, 0.125f, -1);
            pb.g.kv_size = 8;
        }, "multiple of n_head_kv");
    }
    // run-time argument errors
    {
        PB pb;
        const int x = pb.in(64), y = pb.tmp(64, true);
        pb.op(gop::cpy, y).x = x;
        const graph_handle h = create(pb.g, "cpy");
        if (h) {
            graph_inputs gi;
            graph_outputs go;
            CHECK(throws_with([&] { D->graph_run(h, 1, gi, go); }, "arguments given"), "wrong argument count");
            CHECK(throws_with([&] { D->graph_run(h + 1000, 1, gi, go); }, "unknown graph handle"), "unknown graph");
            D->graph_free(h);
            CHECK(throws_with([&] { D->graph_free(h); }, "unknown graph handle"), "graph_free twice");
        }
    }
}

static void test_memory() {
    printf("const_free and graph_free\n");
    // the AUX region (192 MiB): 100 consts of 4 MiB would not fit without the release
    for (int i = 0; i < 100; ++i) {
        std::vector<float> v((size_t) 4 * 1024 * 1024 / 4, 1.0f);
        const const_handle c = D->const_upload(v.data(), 1, (uint32_t) v.size());
        D->const_free(c);
    }
    CHECK(throws_with([] { D->const_free(999999); }, "unknown const handle"), "unknown const handle");
    const memory_info m0 = D->memory();
    {
        PB pb;
        const int q = pb.in(128), k = pb.in(64), v = pb.in(64), y = pb.tmp(128, true);
        pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1);
        pb.g.kv_size = 4096;
        const graph_handle h = create(pb.g, "kv planes");
        if (h) {
            const memory_info m1 = D->memory();
            CHECK(m1.weight_free < m0.weight_free, "the KV planes take heap memory");
            const graph_shape sh = D->graph_shape_of(h);
            CHECK(m0.weight_free - m1.weight_free >= sh.kv_bytes, "the planes: %" PRIu64 " B per channel taken, %" PRIu64 " expected",
                  m0.weight_free - m1.weight_free, sh.kv_bytes);
            D->graph_free(h);
            CHECK(D->memory().weight_free == m0.weight_free && D->memory().weight_largest_free == m0.weight_largest_free,
                  "graph_free returns the planes");
        }
    }
    // a freed weight or const: the run says so
    {
        const W w = up_weight(64, 64, 0.1f);
        const_handle bc = up_const(rnd_vec(64, 1.0f), 1, 64);
        PB pb;
        const int x = pb.in(64), b = pb.cst(bc, 64), y = pb.tmp(64, true);
        pb.mul_mat(w.h, x, y, b);
        const graph_handle h = create(pb.g, "freed");
        if (h) {
            std::vector<arg_t> args = { in_arg(rnd_vec(64, 1.0f)) };
            run_cmp(h, 1, args, "before the free");
            D->const_free(bc);
            const std::vector<float> xv = rnd_vec(64, 1.0f);
            std::vector<float> yv(64);
            graph_inputs gi;
            gi.args = { { xv.data(), 0 } };
            graph_outputs go;
            go.outs = { { yv.data(), 0 } };
            CHECK(throws_with([&] { D->graph_run(h, 1, gi, go); }, "was freed"), "a freed const is reported");
            D->graph_free(h);
        }
        D->free_weight(w.h);
    }
}

static void test_threads() {
    printf("two threads\n");
    const W w1 = up_weight(128, 64, 0.1f), w2 = up_weight(96, 128, 0.1f);
    auto mk = [&](const W & w) {
        PB pb;
        const int x = pb.in(w.K), y = pb.tmp(w.N, true);
        pb.mul_mat(w.h, x, y);
        return create(pb.g, "thread graph");
    };
    const graph_handle g1 = mk(w1), g2 = mk(w2);
    if (!g1 || !g2) return;
    std::atomic<int> bad{ 0 };
    auto work = [&](graph_handle h, const W & w, uint32_t seed) {
        std::mt19937 rng(seed);
        for (int it = 0; it < 12; ++it) {
            const uint32_t T = 1 + (uint32_t) (rng() % 9);
            std::vector<float> x((size_t) T * w.K), y((size_t) T * w.N), want((size_t) T * w.N);
            for (float & v : x) v = std::uniform_real_distribution<float>(-1, 1)(rng);
            graph_inputs gi;
            gi.args = { { x.data(), 0 } };
            graph_outputs go;
            go.outs = { { y.data(), 0 } };
            D->graph_run(h, T, gi, go);
            D->linear_ref(w.h, x.data(), T, want.data());
            if (memcmp(y.data(), want.data(), y.size() * 4) != 0) bad++;
            // linear() between the graph runs, on the same IO region
            std::vector<float> y2((size_t) T * w.N);
            D->linear(w.h, x.data(), T, y2.data());
            if (memcmp(y2.data(), want.data(), y2.size() * 4) != 0) bad++;
        }
    };
    std::thread a(work, g1, std::cref(w1), 1u), b(work, g2, std::cref(w2), 2u);
    a.join();
    b.join();
    CHECK(bad == 0, "%d results differ from linear_ref in the threads", bad.load());
    D->graph_free(g1);
    D->graph_free(g2);
}

// ---------------------------------------------------------------------------------------------
// Negative control: FA_NEW. The comparison with the direct reference without FA_NEW must differ where the device is sensitive to it
// (T_MAX = 1, chunks of a long attention); elsewhere the FLASH_ATTN result does not depend on it (see test-tsr-kinds.cpp).
// ---------------------------------------------------------------------------------------------
static void neg_fa_new() {
    printf("negative control: FA_NEW\n");
    size_t max_bad = 0;
    for (uint32_t T : { 300u, 600u, 680u }) {
        const uint32_t nh = 2, nkv = 1, d = 64;
        PB pb;
        const int q = pb.in(nh * d), k = pb.in(d), v = pb.in(d), y = pb.tmp(nh * d, true);
        const float scale = 0.125f;
        pb.attn(q, k, v, y, nh, nkv, d, scale, -1);
        pb.g.kv_size = T;
        const graph_handle h = create(pb.g, "FA_NEW graph");
        if (!h) return;
        const std::vector<float> qv = rnd_vec((size_t) T * nh * d, 2.0f), kv = rnd_vec((size_t) T * d, 2.0f), vv = rnd_vec((size_t) T * d, 2.0f);
        result_t r;
        run_cmp(h, T, { in_arg(qv), in_arg(kv), in_arg(vv) }, "FA_NEW control", &r);
        const std::vector<float> with = attn_direct(qv, kv, vv, nh, nkv, d, T, scale, {}, false);
        const std::vector<float> without = attn_direct(qv, kv, vv, nh, nkv, d, T, scale, {}, true);
        CHECK(same_bits(r.dev[0], with), "T %u: the device equals the direct reference with FA_NEW", T);
        // the plan's FLASH_ATTN record carries FA_NEW for the T cells its SET_ROWS_KV records wrote
        const graph_plan_info info = D->graph_info(h, T);
        CHECK(info.by_kind[opl::K_FLASH_ATTN] == 1 && info.fa_new == 1, "T %u: FA_NEW in the FLASH_ATTN record (%u of %u)", T, info.fa_new, info.by_kind[opl::K_FLASH_ATTN]);
        size_t bad = 0;
        for (size_t i = 0; i < with.size(); ++i) bad += memcmp(&with[i], &without[i], 4) != 0;
        printf("  T %u: %zu of %zu outputs differ between the references with and without FA_NEW\n", T, bad, with.size());
        max_bad = std::max(max_bad, bad);
        D->graph_free(h);
    }
    if (D->config().t_max == 1) {
        printf("  (T_MAX 1: reported only; the graph form has every cell new, so the order of the chunks may not show)\n");
    } else {
        printf("  (T_MAX %u: the FLASH_ATTN result does not depend on FA_NEW; the flag word is checked in the encoders' test)\n", D->config().t_max);
    }
}

// ---------------------------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------------------------
static void test_counters() {
    printf("counters\n");
    D->reset_counters();
    const W w = up_weight(64, 64, 0.1f);
    const const_handle c = up_const(rnd_vec(64, 1.0f), 1, 64);
    PB pb;
    const int x = pb.in(64), b = pb.cst(c, 64), y = pb.tmp(64, true);
    pb.mul_mat(w.h, x, y, b);
    const graph_handle h = create(pb.g, "counters");
    if (!h) return;
    for (uint32_t T : { 3u, 3u, 4u }) run_cmp(h, T, { in_arg(rnd_vec((size_t) T * 64, 1.0f)) }, "counters");
    const counters k = D->get_counters();
    // run_cmp runs graph_run and graph_run_ref: three plan lookups of each kind (T = 3 miss, T = 3 hit, T = 4 miss) for each call
    CHECK(k.graphs == 1 && k.consts == 1, "graphs %" PRIu64 " consts %" PRIu64, k.graphs, k.consts);
    CHECK(k.graph_runs == 3, "graph_runs %" PRIu64, k.graph_runs);
    CHECK(k.graph_launches >= 3 && k.graph_records >= 3, "launches %" PRIu64 " records %" PRIu64, k.graph_launches, k.graph_records);
    CHECK(k.graph_plan_misses == 2 && k.graph_plan_hits == 4, "plans: %" PRIu64 " misses, %" PRIu64 " hits (2 and 4 expected)", k.graph_plan_misses, k.graph_plan_hits);
    D->graph_free(h);
}

// ---------------------------------------------------------------------------------------------
// Transfers: the out values that touch are read in one transfer, the inputs that touch are written in one
// ---------------------------------------------------------------------------------------------
// graph_run with the counters taken around it alone, then graph_run_ref; the bits must be equal. `delta` gets the counters of the run.
static bool run_counted(graph_handle h, uint32_t T, const std::vector<arg_t> & args, const graph_state & st, counters * delta, const char * what) {
    g_cases++;
    const graph_shape sh = D->graph_shape_of(h);
    graph_inputs gi;
    for (size_t i = 0; i < args.size(); ++i) {
        graph_arg a;
        a.data = sh.arg_kind[i] == gv::pos ? (const void *) args[i].p.data() : (const void *) args[i].f.data();
        gi.args.push_back(a);
    }
    std::vector<std::vector<float>> dv(sh.out_n.size()), rv(sh.out_n.size());
    graph_outputs od, orf;
    for (size_t i = 0; i < sh.out_n.size(); ++i) {
        dv[i].assign((size_t) T * sh.out_n[i], -7.0f);
        rv[i].assign((size_t) T * sh.out_n[i], -7.0f);
        od.outs.push_back({ dv[i].data(), 0 });
        orf.outs.push_back({ rv[i].data(), 0 });
    }
    try {
        const counters c0 = D->get_counters();
        D->graph_run(h, T, gi, od, st);
        const counters c1 = D->get_counters();
        if (delta) {
            *delta = c1;
            delta->h2d_bytes -= c0.h2d_bytes;
            delta->d2h_bytes -= c0.d2h_bytes;
            delta->h2d_writes -= c0.h2d_writes;
            delta->d2h_reads -= c0.d2h_reads;
        }
        D->graph_run_ref(h, T, gi, orf, st);
    } catch (const std::exception & e) {
        CHECK(false, "%s: %s", what, e.what());
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < dv.size(); ++i) {
        size_t first = 0;
        const bool same = same_bits(dv[i], rv[i], &first);
        CHECK(same, "%s: output %zu: graph_run differs from graph_run_ref at %zu", what, i, first);
        ok = ok && same;
    }
    return ok;
}

static uint64_t row_bytes(uint32_t n) { return (uint64_t) Lc(n) * 64; }

static void unary_op(PB & pb, int x, int y) {
    graph_op & o = pb.op(gop::unary, y);
    o.x = x; o.func = 2;
}

static void test_transfers() {
    printf("transfers: out values in one read, inputs in few writes\n");
    // two cached attn ops with their own slots; four out values side by side: y1, y2, z1 = silu(q1), z2 = silu(y1)
    {
        const uint32_t cells = 64;
        PB pb;
        const int q1 = pb.in(128), k1 = pb.in(64), v1 = pb.in(64), q2 = pb.in(128), k2 = pb.in(64), v2 = pb.in(64);
        const int y1 = pb.tmp(128, true), y2 = pb.tmp(128, true), z1 = pb.tmp(128, true), z2 = pb.tmp(128, true);
        pb.attn(q1, k1, v1, y1, 2, 1, 64, 0.125f, -1, 0);
        pb.attn(q2, k2, v2, y2, 2, 1, 64, 0.125f, -1, 1);
        unary_op(pb, q1, z1);
        unary_op(pb, y1, z2);
        pb.g.kv_slots = { cells, cells };
        const graph_handle h = create(pb.g, "transfers: two slots");
        if (h) {
            const kv_handle ka = D->kv_create(64, cells), kb = D->kv_create(64, cells);
            graph_state st;
            st.kv = std::vector<kv_handle>{ ka, kb };
            uint32_t pos0 = 0;
            for (uint32_t T : { 1u, 5u }) {
                st.pos0 = pos0;
                const std::vector<arg_t> args = { in_arg(rnd_vec((size_t) T * 128, 1.0f)), in_arg(rnd_vec((size_t) T * 64, 1.0f)), in_arg(rnd_vec((size_t) T * 64, 1.0f)),
                                                  in_arg(rnd_vec((size_t) T * 128, 1.0f)), in_arg(rnd_vec((size_t) T * 64, 1.0f)), in_arg(rnd_vec((size_t) T * 64, 1.0f)) };
                counters dc;
                char what[96];
                snprintf(what, sizeof(what), "transfers: two slots, T %u at pos0 %u", T, pos0);
                run_counted(h, T, args, st, &dc, what);
                CHECK(dc.d2h_reads == 1, "%s: %" PRIu64 " reads of the out values (1 expected)", what, dc.d2h_reads);
                CHECK(dc.d2h_bytes == (uint64_t) T * 4 * row_bytes(128), "%s: d2h_bytes %" PRIu64, what, dc.d2h_bytes);
                pos0 += T;
            }
            D->kv_free(ka);
            D->kv_free(kb);
            D->graph_free(h);
        }
    }
    // a chunked run: one read per chunk (1500 rows: chunks of 680, 680, 140), two out values that touch
    {
        const uint32_t cells = 1536, R = 1500;
        PB pb;
        const int q = pb.in(128), k = pb.in(64), v = pb.in(64);
        const int y = pb.tmp(128, true), z = pb.tmp(128, true);
        pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1, 0);
        unary_op(pb, q, z);
        pb.g.kv_slots = { cells };
        const graph_handle h = create(pb.g, "transfers: chunks");
        if (h) {
            const kv_handle kh = D->kv_create(64, cells);
            graph_state st;
            st.kv = std::vector<kv_handle>{ kh };
            counters dc;
            run_counted(h, R, { in_arg(rnd_vec((size_t) R * 128, 1.0f)), in_arg(rnd_vec((size_t) R * 64, 1.0f)), in_arg(rnd_vec((size_t) R * 64, 1.0f)) }, st, &dc,
                        "transfers: 1500 rows");
            CHECK(dc.d2h_reads == 3, "transfers: 1500 rows: %" PRIu64 " reads (one per chunk, 3 expected)", dc.d2h_reads);
            CHECK(dc.d2h_bytes == (uint64_t) R * 2 * row_bytes(128), "transfers: 1500 rows: d2h_bytes %" PRIu64, dc.d2h_bytes);
            D->kv_free(kh);
            D->graph_free(h);
        }
    }
    // the writes of a run with a plan hit: T = 1, a cached graph whose mask needs no f32 copy. The static block (the cell indices, then the
    // mask summary: they touch) is one write. The f32 mask area of the slot is reserved but not written, so the in rows, which start
    // behind it, are the second write.
    {
        const uint32_t cells = 64;
        PB pb;
        const int q = pb.in(128), k = pb.in(64), v = pb.in(64), y = pb.tmp(128, true);
        pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1, 0);
        pb.g.kv_slots = { cells };
        const graph_handle h = create(pb.g, "transfers: writes");
        if (h) {
            const kv_handle kh = D->kv_create(64, cells);
            graph_state st;
            st.kv = std::vector<kv_handle>{ kh };
            const uint32_t T = 1;
            for (int round = 0; round < 3; ++round) {
                const counters p0 = D->get_counters();
                counters dc;
                run_counted(h, T, { in_arg(rnd_vec(128, 1.0f)), in_arg(rnd_vec(64, 1.0f)), in_arg(rnd_vec(64, 1.0f)) }, st, &dc, "transfers: writes");
                const counters p1 = D->get_counters();
                // the first round builds the plan and writes its op lists; the later rounds hit the plan and the op list cache
                if (round == 0) continue;
                CHECK(p1.graph_plan_hits > p0.graph_plan_hits && p1.prog_misses == p0.prog_misses, "transfers: writes: a plan hit in round %d", round);
                const uint64_t stat = ((uint64_t) T * opl::PRE_KVIDX_BYTES + 63) / 64 * 64 + (opl::msum_bytes(T) + 63) / 64 * 64;
                const uint64_t rows = row_bytes(128) + 2 * row_bytes(64);
                CHECK(dc.h2d_writes == 2, "transfers: writes: %" PRIu64 " writes (2 expected: the static block and the in rows)", dc.h2d_writes);
                CHECK(dc.h2d_bytes == stat + rows, "transfers: writes: h2d_bytes %" PRIu64 " = %" PRIu64 " (static) + %" PRIu64 " (in rows)", dc.h2d_bytes, stat, rows);
            }
            D->kv_free(kh);
            D->graph_free(h);
        }
    }
    // the fallback: placing the out values first costs IO space (in writer order y2 takes the hole that t1 leaves; with the out values
    // first the side chain lies one wide value higher)
    {
        // T is the largest row count that graph_unsupported probes, so the small region is sized for every plan it builds
        const uint32_t cells = 64, W = 2048, T = D->config().t_max + 1;
        auto build = [&](PB & pb) {
            const int q = pb.in(128), k = pb.in(64), v = pb.in(64), xw = pb.in(W);
            const int y = pb.tmp(128, true), t1 = pb.tmp(W), t2 = pb.tmp(W), y2 = pb.tmp(W, true);
            pb.attn(q, k, v, y, 2, 1, 64, 0.125f, -1, 0);
            unary_op(pb, xw, t1);
            unary_op(pb, t1, t2);
            unary_op(pb, t2, y2);
            pb.g.kv_slots = { cells };
        };
        auto args = [&] {
            return std::vector<arg_t>{ in_arg(rnd_vec((size_t) T * 128, 1.0f)), in_arg(rnd_vec((size_t) T * 64, 1.0f)), in_arg(rnd_vec((size_t) T * 64, 1.0f)),
                                       in_arg(rnd_vec((size_t) T * W, 1.0f)) };
        };
        PB pb;
        build(pb);
        const graph_handle h = create(pb.g, "fallback: big device");
        uint64_t P1 = 0;
        if (h) {
            const kv_handle kh = D->kv_create(64, cells);
            graph_state st;
            st.kv = std::vector<kv_handle>{ kh };
            run_counted(h, T, args(), st, nullptr, "fallback: big device");
            P1 = D->graph_info(h, T).io_bytes;
            D->kv_free(kh);
            D->graph_free(h);
        }
        if (P1 > 64) {
            open_options o;
            o.io_bytes = P1 - 64;
            device small(o);
            device * saved = D;
            D = &small;
            PB pb2;
            build(pb2);
            const graph_handle hs = create(pb2.g, "fallback: small device");
            if (hs) {
                const kv_handle ks = small.kv_create(64, cells);
                graph_state st;
                st.kv = std::vector<kv_handle>{ ks };
                run_counted(hs, T, args(), st, nullptr, "fallback: small device");
                const graph_plan_info info = small.graph_info(hs, T);
                CHECK(info.T == T, "fallback: the whole run of %u rows is one chunk (T %u)", T, info.T);
                CHECK(info.io_bytes < P1, "fallback: io_bytes %" PRIu64 " below the plan with the out values first (%" PRIu64 ")", info.io_bytes, P1);
                CHECK(P1 - info.io_bytes >= (uint64_t) T * row_bytes(W) - 64, "fallback: the out values first cost a wide value (%" PRIu64 " against %" PRIu64 ")", P1, info.io_bytes);
                // the plan holds nothing of the attempt that did not fit: one attn op with new cells, one list
                CHECK(info.fa_new == 1 && info.lists == 1, "fallback: the plan has %u attn ops with new cells and %u lists (1 and 1 expected)", info.fa_new, info.lists);
                small.kv_free(ks);
                small.graph_free(hs);
            }
            D = saved;
        }
    }
    // a graph with a pos argument and no attn op: the position array ends where the in rows begin, so a run that hits the plan and
    // the op list cache writes its inputs in one transfer (the positions first, then the in rows: address order)
    {
        const uint32_t nh = 2, hd = 64, T = 3;
        PB pb;
        std::vector<float> tab(hd / 2);
        for (uint32_t i = 0; i < hd / 2; ++i) tab[i] = (float) std::pow(10000.0, -2.0 * (double) i / (double) hd);
        const int x = pb.in(nh * hd), t = pb.cst(up_const(tab, 1, hd / 2), hd / 2), p = pb.pos(), y = pb.tmp(nh * hd, true);
        pb.rope(x, y, nh, hd, hd, t, p);
        const graph_handle h = create(pb.g, "transfers: pos");
        if (h) {
            counters dc;
            for (int round = 0; round < 2; ++round) {
                run_counted(h, T, { in_arg(rnd_vec((size_t) T * nh * hd, 1.0f)), pos_arg({ 3, 4, 5 }) }, graph_state(), &dc, "transfers: pos");
            }
            CHECK(dc.h2d_writes == 1, "transfers: pos: %" PRIu64 " writes (1 expected: the positions and the in rows touch)", dc.h2d_writes);
            CHECK(dc.h2d_bytes == 64 + (uint64_t) T * row_bytes(nh * hd), "transfers: pos: h2d_bytes %" PRIu64, dc.h2d_bytes);
            CHECK(dc.d2h_reads == 1, "transfers: pos: %" PRIu64 " reads (1 expected)", dc.d2h_reads);
            D->graph_free(h);
        }
    }
    // a graph without slots is planned in writer order. a (out), d, b live together, c (out) is placed above d: the two out values do
    // not touch, so there are two reads and the bytes of d are not read.
    {
        const uint32_t n = 256, T = 3;
        PB pb;
        const int x = pb.in(n), a = pb.tmp(n, true), d = pb.tmp(n), b = pb.tmp(n), c = pb.tmp(n, true);
        unary_op(pb, x, a);
        unary_op(pb, x, d);
        unary_op(pb, d, b);
        pb.ew(gop::add, b, d, c);
        const graph_handle h = create(pb.g, "transfers: no slots");
        if (h) {
            counters dc;
            run_counted(h, T, { in_arg(rnd_vec((size_t) T * n, 1.0f)) }, graph_state(), &dc, "transfers: no slots");
            run_counted(h, T, { in_arg(rnd_vec((size_t) T * n, 1.0f)) }, graph_state(), &dc, "transfers: no slots");   // the op lists are cached now
            CHECK(dc.d2h_reads == 2, "transfers: no slots: %" PRIu64 " reads (2 expected, the out values do not touch)", dc.d2h_reads);
            CHECK(dc.d2h_bytes == (uint64_t) T * 2 * row_bytes(n), "transfers: no slots: d2h_bytes %" PRIu64 " (no gap is read)", dc.d2h_bytes);
            CHECK(D->graph_info(h, T).io_bytes == (uint64_t) T * 4 * row_bytes(n), "transfers: no slots: io_bytes %" PRIu64, D->graph_info(h, T).io_bytes);
            CHECK(dc.h2d_writes == 1, "transfers: no slots: %" PRIu64 " writes (1 expected)", dc.h2d_writes);
            D->graph_free(h);
        }
    }
}

int main() {
    try {
        D = new device();
    } catch (const std::exception & e) {
        fprintf(stderr, "test_tsr_rt_graph: cannot open the device: %s\n", e.what());
        return 1;
    }
    g_kenv = tsr_kenv_make(D->config(), D->pcap());
    printf("%s\n", D->description());
    test_mul_mats();
    test_ew_ops();
    test_rms_ops();
    test_rope_ops();
    test_attn_ops();
    test_dags();
    test_cut();
    test_layers();
    measure_nkv();
    test_kv_objects();
    test_cached_attn_cases();
    test_cached_layers();
    test_cached_objects();
    test_cached_refusals();
    test_cached_store_and_controls();
    test_cached_chunks();
    test_graph_info_slots();
    test_chunk_at_pos0(false, 0);
    test_chunk_at_pos0(true, 0);
    test_chunk_at_pos0(true, 16);
    test_mixed_graph(false);
    test_mixed_graph(true);
    test_kv_in_two_slots();
    test_limits();
    test_memory();
    test_threads();
    test_counters();
    test_transfers();
    neg_fa_new();
    delete D;
    printf("%s (%d cases, %d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_cases, g_fail);
    return g_fail == 0 ? 0 : 1;
}

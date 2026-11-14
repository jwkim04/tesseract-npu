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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mulmat_ref.hpp"
#include "rmsnorm_ref.hpp"
#include "rope_ref.hpp"
#include "kvfa_ref.hpp"
#include "tsr-kinds.h"
#include "tsr-wlayout.h"
#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_rt.h"

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
    int mask() { return val(gv::mask, 0); }
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
    void attn(int q, int k, int v, int dst, uint32_t nh, uint32_t nkv, uint32_t d, float scale, int mask) {
        graph_op & o = op(gop::attn, dst);
        o.q = q; o.k = k; o.v = v; o.n_head = nh; o.n_head_kv = nkv; o.d = d; o.scale = scale; o.mask = mask;
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
static bool run_cmp(graph_handle h, uint32_t T, const std::vector<arg_t> & args, const char * what, result_t * res = nullptr, uint32_t pad = 0) {
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
        D->graph_run(h, T, gi, od);
        D->graph_run_ref(h, T, gi, orf);
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
static int add_layer(PB & pb, const layer_w & w, int x, int pos, bool out) {
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
    pb.attn(qr, kr, v, att, Q_HEAD, Q_KV, Q_D, 0.125f, -1);
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

// ---------------------------------------------------------------------------------------------
// Limits, refusals, memory, threads
// ---------------------------------------------------------------------------------------------
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
    test_limits();
    test_memory();
    test_threads();
    test_counters();
    neg_fa_new();
    delete D;
    printf("%s (%d cases, %d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_cases, g_fail);
    return g_fail == 0 ? 0 : 1;
}

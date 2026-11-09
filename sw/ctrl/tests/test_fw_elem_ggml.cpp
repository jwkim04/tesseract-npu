// test_fw_elem_ggml.cpp — fw_kernels_elem_ggml: the element-wise and row kinds' numerics (sw/ctrl/ref/
// elem_ref.cpp, bit-exact with the firmware by fw_kernels_elem_rv / _host) against the pinned
// ggml-cpu's ops on the same inputs (docs/design/06-software-verification.md "SW" §7.5; VU §6.9 "Accuracy
// and differences from the ggml CPU backend"). Built with TSR_WITH_LLAMA only (links ggml-cpu).
//
// Per op: inputs N(0, sigma) f32 (no subnormals, no specials: those are VU §4.1's DAZ / FTZ / canonical
// NaN rules, which differ from the CPU by design), 16 rows of 4,096 elements, sigma 1 and 4; the
// figures printed: the fraction of bit-identical elements, the largest error in ulps of the CPU
// value (ulp at max(floor(log2|y|), -126)), the largest absolute error, and the KL divergence of the
// rows' softmax (the model-level criterion, klcheck, applied per row). Asserted bounds (VU §6.9):
//   ADD, MUL, ADD / MUL by a scalar or a per-row scalar, SCALE without bias, CPY, GET_ROWS: bit-exact;
//   SCALE with bias: |error| <= ulp(x s) + ulp(y) per element (the CPU's FMA x s + b against the VU's
//     FA(FM(x, s), b): the extra rounding of x s, which cancellation can make many ulps of y);
//   SILU <= 4 ulp, SWIGLU <= 6 ulp (the CPU's own SIMD expf vs EXPK; VU §6.9 SILU 2.87 ulp vs exact;
//     SWIGLU's product rounds once more);
//   EXP: <= 2 ulp (EXPK 0.53 ulp vs the CPU's expf);
//   TANH: <= 40 ulp and 4e-7 absolute (TANH 16.8 ulp / 1.2e-7 vs exact, plus libm tanhf);
//   GELU16: fp16 values within 2 fp16 ulps of the CPU table's (VU §6.5: 272 of 65,536 table entries
//     differ, by at most 2 fp16 ulps; they cluster in the negative tail, so the share of differing
//     elements depends on the inputs: ~0.05 % at sigma 1, ~5 % at sigma 4); GEGLU16 (= GELU16 x up,
//     one more f32 rounding): relative error <= 2^-9 where |y| > 1e-3;
//   GELU, GEGLU (the exact tanh form, PyTorch) against the CPU's fp16 GELU: the up to 3.9e-3
//     relative difference of VU §6.9 (reported; bound 1e-2 relative where |y| > 1e-3).
// Exit 0 / 1.
#undef _GNU_SOURCE
#include "test_util.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

#include "../ref/elem_ref.hpp"
#include "tsr_ref_math.h"
#include "tsr/opl.hpp"
#include "tsr_isa.h"

namespace {

namespace opl = tsr::opl;
constexpr int N = 4096, R = 16;

// ulps of e = |a - b| relative to b (ulp of b's binade, at least the smallest normal's).
double ulp_err(float a, float b) {
    if (a == b) return 0.0;
    int e = 0;
    std::frexp(b, &e);
    e = std::max(e - 1, -126);
    return std::fabs(double(a) - double(b)) / std::ldexp(1.0, e - 23);
}

struct stats {
    double exact = 0, n = 0, max_ulp = 0, max_abs = 0, max_rel = 0, kl = 0;
    void add(float ours, float cpu, double rel_floor = 1e-3) {
        n += 1;
        if (f2u(ours) == f2u(cpu)) {
            exact += 1;
            return;
        }
        max_ulp = std::max(max_ulp, ulp_err(ours, cpu));
        max_abs = std::max(max_abs, std::fabs(double(ours) - double(cpu)));
        if (std::fabs(cpu) > rel_floor) max_rel = std::max(max_rel, std::fabs(double(ours) - double(cpu)) / std::fabs(double(cpu)));
    }
};

// KL(softmax(cpu row) || softmax(our row)) in double.
double kl_row(const float * ours, const float * cpu, int n) {
    double mo = -1e300, mc = -1e300;
    for (int i = 0; i < n; ++i) { mo = std::max(mo, double(ours[i])); mc = std::max(mc, double(cpu[i])); }
    double so = 0, sc = 0;
    for (int i = 0; i < n; ++i) { so += std::exp(double(ours[i]) - mo); sc += std::exp(double(cpu[i]) - mc); }
    double kl = 0;
    for (int i = 0; i < n; ++i) {
        const double p = std::exp(double(cpu[i]) - mc) / sc, q = std::exp(double(ours[i]) - mo) / so;
        if (p > 0) kl += p * std::log(p / std::max(q, 1e-300));
    }
    return kl;
}

struct gg {
    ggml_context * ctx = nullptr;
    gg() {
        ggml_init_params p{size_t(512) << 20, nullptr, false};
        ctx = ggml_init(p);
    }
    ~gg() { ggml_free(ctx); }
    ggml_tensor * t2(int ne0, int ne1, const std::vector<float> & v) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        memcpy(t->data, v.data(), sizeof(float) * size_t(ne0) * size_t(ne1));
        return t;
    }
    void run(ggml_tensor * out) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
    }
};

std::vector<float> gen(rng_t * r, size_t n, float sigma) {
    std::vector<float> v(n);
    for (auto & x : v) x = float(rng_normal(r)) * sigma;
    return v;
}

// Our rows: elem_ref_row per row (the tsr_ref_vop calls of the kernels).
std::vector<float> ours_rows(uint32_t kind, uint32_t func, const std::vector<float> & a, const std::vector<float> * b, int bmode,
                             const std::vector<float> * c, uint32_t c2 = 0u) {
    // bmode: 0 none, 1 rows, 2 broadcast row, 3 one scalar, 4 per-row scalar
    std::vector<float> y(size_t(R) * N);
    std::vector<uint32_t> row(size_t(N) + 16u);
    for (int r = 0; r < R; ++r) {
        const uint32_t * ap = reinterpret_cast<const uint32_t *>(a.data() + size_t(r) * N);
        const uint32_t * bp = nullptr;
        uint32_t cv = 0;
        bool scalar = false;
        if (bmode == 1) bp = reinterpret_cast<const uint32_t *>(b->data() + size_t(r) * N);
        if (bmode == 2) bp = reinterpret_cast<const uint32_t *>(b->data());
        if (bmode == 3) { cv = f2u((*c)[0]); scalar = true; }
        if (bmode == 4) { cv = f2u((*c)[size_t(r)]); scalar = true; }
        if (kind == opl::K_SCALE) cv = f2u((*c)[0]);
        tsr::ref::elem_ref_row(kind, func, N, row.data(), ap, bp, cv, scalar, c2);
        memcpy(y.data() + size_t(r) * N, row.data(), sizeof(float) * N);
    }
    return y;
}

void report(const char * name, const std::vector<float> & ours, const ggml_tensor * cpu, stats & s, double rel_floor = 1e-3) {
    const float * c = static_cast<const float *>(cpu->data);
    for (size_t i = 0; i < ours.size(); ++i) s.add(ours[i], c[i], rel_floor);
    double kl = 0;
    for (int r = 0; r < R; ++r) kl = std::max(kl, kl_row(ours.data() + size_t(r) * N, c + size_t(r) * N, N));
    s.kl = std::max(s.kl, kl);
    printf("  %-34s exact %7.3f %%  max %8.3f ulp  max abs %.3g  max rel %.3g  KL(row softmax) %.3g\n", name, 100.0 * s.exact / s.n,
           s.max_ulp, s.max_abs, s.max_rel, kl);
}

}  // namespace

int main() {
    printf("fw_kernels_elem_ggml: elem_ref (the firmware's numerics) against ggml-cpu, %d rows x %d f32\n", R, N);
    for (float sigma : {1.0f, 4.0f}) {
        printf("sigma %.1f\n", sigma);
        rng_t rg = {0xC0FFEEu + uint64_t(sigma)};
        const std::vector<float> a = gen(&rg, size_t(R) * N, sigma), b = gen(&rg, size_t(R) * N, sigma);
        const std::vector<float> brow = gen(&rg, N, sigma), sc1 = {0.37f}, scr = gen(&rg, R, 2.0f);
        gg g;
        {   // ADD rows, broadcast, scalar, per-row scalar
            stats s1, s2, s3, s4, s5, s6;
            ggml_tensor * A = g.t2(N, R, a);
            ggml_tensor * t;
            g.run(t = ggml_add(g.ctx, A, g.t2(N, R, b)));
            report("ADD rows", ours_rows(opl::K_ADD, TSR_VOP_FUNC_ADD, a, &b, 1, nullptr), t, s1);
            g.run(t = ggml_add(g.ctx, A, g.t2(N, 1, brow)));
            report("ADD broadcast row (bias)", ours_rows(opl::K_ADD, TSR_VOP_FUNC_ADD, a, &brow, 2, nullptr), t, s2);
            g.run(t = ggml_add(g.ctx, A, g.t2(1, 1, sc1)));
            report("ADD one scalar (ADDC)", ours_rows(opl::K_ADD, TSR_VOP_FUNC_ADDC, a, nullptr, 3, &sc1), t, s3);
            g.run(t = ggml_mul(g.ctx, A, g.t2(N, R, b)));
            report("MUL rows", ours_rows(opl::K_MUL, TSR_VOP_FUNC_MUL, a, &b, 1, nullptr), t, s4);
            g.run(t = ggml_mul(g.ctx, A, g.t2(1, R, scr)));
            report("MUL per-row scalar (inp_scale_rows)", ours_rows(opl::K_MUL, TSR_VOP_FUNC_MULC, a, nullptr, 4, &scr), t, s5);
            g.run(t = ggml_mul(g.ctx, A, g.t2(N, 1, brow)));
            report("MUL broadcast row (norm weight)", ours_rows(opl::K_MUL, TSR_VOP_FUNC_MUL, a, &brow, 2, nullptr), t, s6);
            for (stats * s : {&s1, &s2, &s3, &s4, &s5, &s6}) CHECK(s->exact == s->n, "ADD / MUL: not bit-exact with ggml-cpu");
        }
        {   // SCALE
            stats s1, s2;
            ggml_tensor * t;
            const std::vector<float> sv = {1.0f / 30.0f};
            g.run(t = ggml_scale(g.ctx, g.t2(N, R, a), sv[0]));
            report("SCALE", ours_rows(opl::K_SCALE, TSR_VOP_FUNC_MULC, a, nullptr, 0, &sv), t, s1);
            const float bias = -0.625f;
            g.run(t = ggml_scale_bias(g.ctx, g.t2(N, R, a), sv[0], bias));
            report("SCALE with bias", ours_rows(opl::K_SCALE, TSR_VOP_FUNC_MULC, a, nullptr, 0, &sv, f2u(bias)), t, s2);
            CHECK(s1.exact == s1.n, "SCALE: not bit-exact with ggml-cpu");
            // Per element: |ours - cpu| <= ulp(x s) + ulp(y).
            const std::vector<float> ob = ours_rows(opl::K_SCALE, TSR_VOP_FUNC_MULC, a, nullptr, 0, &sv, f2u(bias));
            const float * cb = static_cast<const float *>(t->data);
            int bad = 0;
            for (size_t i = 0; i < ob.size(); ++i) {
                const double xs = double(a[i]) * double(sv[0]);
                auto ulp = [](double v) { int e = 0; std::frexp(v, &e); return std::ldexp(1.0, std::max(e - 1, -126) - 23); };
                if (std::fabs(double(ob[i]) - double(cb[i])) > ulp(xs) + ulp(double(cb[i]))) ++bad;
            }
            CHECK(bad == 0, "SCALE with bias: %d elements beyond ulp(x s) + ulp(y)", bad);
        }
        {   // GLU
            stats s1, s2, s3;
            ggml_tensor * t;
            g.run(t = ggml_swiglu_split(g.ctx, g.t2(N, R, a), g.t2(N, R, b)));
            report("GLU SWIGLU", ours_rows(opl::K_GLU, TSR_VOP_FUNC_SWIGLU, a, &b, 1, nullptr), t, s1);
            g.run(t = ggml_geglu_split(g.ctx, g.t2(N, R, a), g.t2(N, R, b)));
            report("GLU GEGLU16", ours_rows(opl::K_GLU, TSR_VOP_FUNC_GEGLU16, a, &b, 1, nullptr), t, s2);
            report("GLU GEGLU (exact) vs CPU GEGLU", ours_rows(opl::K_GLU, TSR_VOP_FUNC_GEGLU, a, &b, 1, nullptr), t, s3);
            CHECK(s1.max_ulp <= 6.0, "SWIGLU: %.3f ulp (bound 6)", s1.max_ulp);
            CHECK(s2.max_rel <= 1.0 / 512.0, "GEGLU16: rel %.3g (bound 2^-9)", s2.max_rel);
            CHECK(s3.max_rel <= 1e-2, "GEGLU exact form: rel %.3g (bound 1e-2)", s3.max_rel);
        }
        {   // UNARY
            stats st, sg, ss, se, sx;
            ggml_tensor * A = g.t2(N, R, a);
            ggml_tensor * t;
            g.run(t = ggml_tanh(g.ctx, A));
            report("UNARY TANH", ours_rows(opl::K_UNARY, TSR_VOP_FUNC_TANH, a, nullptr, 0, nullptr), t, st);
            g.run(t = ggml_gelu(g.ctx, A));
            report("UNARY GELU16", ours_rows(opl::K_UNARY, TSR_VOP_FUNC_GELU16, a, nullptr, 0, nullptr), t, sg);
            report("UNARY GELU (exact) vs CPU GELU", ours_rows(opl::K_UNARY, TSR_VOP_FUNC_GELU, a, nullptr, 0, nullptr), t, sx);
            g.run(t = ggml_silu(g.ctx, A));
            report("UNARY SILU", ours_rows(opl::K_UNARY, TSR_VOP_FUNC_SILU, a, nullptr, 0, nullptr), t, ss);
            // EXP on values that do not overflow
            std::vector<float> ae(a);
            for (auto & x : ae) x = std::fmax(std::fmin(x, 80.0f), -80.0f);
            g.run(t = ggml_exp(g.ctx, g.t2(N, R, ae)));
            report("UNARY EXP", ours_rows(opl::K_UNARY, TSR_VOP_FUNC_EXP, ae, nullptr, 0, nullptr), t, se);
            CHECK(st.max_ulp <= 40.0 && st.max_abs <= 4e-7, "TANH: %.3f ulp, abs %.3g", st.max_ulp, st.max_abs);
            {   // GELU16: both sides are fp16 values; their distance in fp16 ulps.
                const std::vector<float> og = ours_rows(opl::K_UNARY, TSR_VOP_FUNC_GELU16, a, nullptr, 0, nullptr);
                ggml_tensor * tg = ggml_gelu(g.ctx, A);
                g.run(tg);
                const float * cg = static_cast<const float *>(tg->data);
                auto key = [](uint16_t h) { return (h & 0x8000u) ? -int32_t(h & 0x7FFFu) : int32_t(h); };
                int32_t worst = 0;
                for (size_t i = 0; i < og.size(); ++i) {
                    const int32_t dk = key(tsr_f32_to_f16_rne(og[i])) - key(tsr_f32_to_f16_rne(cg[i]));
                    worst = std::max(worst, dk < 0 ? -dk : dk);
                }
                printf("  %-34s largest difference %d fp16 ulps\n", "UNARY GELU16 (fp16 values)", worst);
                CHECK(worst <= 2, "GELU16: %d fp16 ulps from the CPU table (bound 2, VU §6.5)", worst);
            }
            CHECK(ss.max_ulp <= 4.0, "SILU: %.3f ulp (bound 4)", ss.max_ulp);
            CHECK(sg.max_rel <= 1.0 / 512.0, "GELU16: rel %.3g (bound 2^-9)", sg.max_rel);
            CHECK(se.max_ulp <= 2.0, "EXP: %.3f ulp (bound 2)", se.max_ulp);
            CHECK(sx.max_rel <= 1e-2, "GELU exact form: rel %.3g (bound 1e-2)", sx.max_rel);
        }
        {   // CPY, GET_ROWS (bit copies)
            ggml_tensor * A = g.t2(N, R, a);
            ggml_tensor * D = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N, R);
            ggml_tensor * t;
            g.run(t = ggml_cpy(g.ctx, A, D));
            CHECK(memcmp(t->data, a.data(), sizeof(float) * a.size()) == 0, "CPY: not bit-exact");
            ggml_tensor * ids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 5);
            const int32_t iv[5] = {3, 0, 15, 7, 7};
            memcpy(ids->data, iv, sizeof(iv));
            g.run(t = ggml_get_rows(g.ctx, A, ids));
            bool same = true;
            for (int i = 0; i < 5; ++i)
                same = same && memcmp(static_cast<const float *>(t->data) + size_t(i) * N, a.data() + size_t(iv[i]) * N, sizeof(float) * N) == 0;
            CHECK(same, "GET_ROWS: not bit-exact");
            printf("  %-34s bit-exact (copies)\n", "CPY, GET_ROWS");
        }
    }
    return test_finish("fw_kernels_elem_ggml");
}

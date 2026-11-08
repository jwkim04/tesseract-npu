// test_fw_gdn_ggml.cpp — fw_gdn_ggml: the Gated DeltaNet kinds' numerics (sw/ctrl/ref/gdn_ref.cpp,
// bit-exact with the firmware by fw_gdn_rv) against the pinned ggml-cpu's ops on the same inputs
// (docs/design/06-software-verification.md "SW" §7.9, decisions GDN-1 to GDN-4). Built with
// TSR_WITH_LLAMA only (links ggml-cpu). Also against float64 math, so a defect of the reference cannot
// hide behind the CPU kernel's own rounding.
//
//   softplus  x ~ N(0, sigma), sigma 1, 4, 12, and a linear sweep of [-40, 40]: against float64
//             log1p(exp(x)): relative error <= 4.2e-7 (all results above 1e-30), <= 5 ulp above 1e-3;
//             against ggml's ggml_softplus (logf(1 + expf(x)), x > 20 passes through): absolute
//             error <= 1.2e-6 (the CPU form is the less accurate one for results below 1e-3: reported).
//   ssm_conv  ggml_concat + ggml_ssm_conv + ggml_silu of the pinned graph against sconv_ref: relative
//             error <= 2e-6 of the largest |y| of the channel (the CPU adds with FMAs, the VU rounds
//             each product), exact state.
//   gdn       ggml_gated_delta_net (K = 1) against gdn_ref over 1, 16 and 512 tokens, H_k < H_v,
//             S_v 128: the largest relative error of the outputs and states against the CPU result
//             and against a float64 recurrence; asserted: both below 1e-3 at 512 tokens, and ours
//             no worse than 4x the CPU's own error against float64.
// Exit 0 / 1.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#undef _GNU_SOURCE
#include "test_util.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include "../ref/gdn_ref.hpp"
#include "tsr/opl.hpp"

namespace {

namespace opl = tsr::opl;
constexpr uint64_t LINE = 64u;

double ulp_of(float b) {
    int e = 0;
    std::frexp(b, &e);
    e = std::max(e - 1, -126);
    return std::ldexp(1.0, e - 23);
}

// ---------------------------------------------------------------------------------------------
// A tiny CPU runner
// ---------------------------------------------------------------------------------------------
struct cpu_ctx {
    ggml_context * ctx = nullptr;
    std::vector<uint8_t> buf;
    explicit cpu_ctx(size_t bytes = 256u << 20) {
        buf.resize(bytes);
        ggml_init_params p = {bytes, buf.data(), false};
        ctx = ggml_init(p);
    }
    ~cpu_ctx() { ggml_free(ctx); }
    void run(ggml_tensor * out) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
    }
};

// ---------------------------------------------------------------------------------------------
// DRAM view helper for the references
// ---------------------------------------------------------------------------------------------
struct dview {
    std::vector<std::vector<uint8_t>> ch;
    std::vector<uint8_t *> chp;
    uint64_t next = 0;
    explicit dview(size_t bytes) : ch(4, std::vector<uint8_t>(bytes)), chp(4) {
        for (int c = 0; c < 4; ++c) chp[size_t(c)] = ch[size_t(c)].data();
    }
    tsr::ref::dram_view view() { return tsr::ref::dram_view{chp.data(), 4u, ch[0].size()}; }
    // lines of storage on channel 0; returns the handle
    uint32_t take(uint64_t lines) {
        const uint32_t h = uint32_t(next);
        next += (lines + 7u) & ~7ull;
        return h;
    }
    uint8_t * at(uint32_t h) { return ch[0].data() + uint64_t(h) * LINE; }
};
void put_rows(dview & d, uint32_t h, uint32_t row, uint32_t stride, const float * src, uint32_t n) {
    float * p = reinterpret_cast<float *>(d.at(h) + uint64_t(row) * stride * LINE);
    memcpy(p, src, size_t(n) * 4u);
}
void get_rows(dview & d, uint32_t h, uint32_t row, uint32_t stride, float * dst, uint32_t n) {
    memcpy(dst, d.at(h) + uint64_t(row) * stride * LINE, size_t(n) * 4u);
}

// ---------------------------------------------------------------------------------------------
// softplus
// ---------------------------------------------------------------------------------------------
void test_softplus() {
    std::mt19937_64 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int n = 4096;
    std::vector<float> x;
    for (float sigma : {1.0f, 4.0f, 12.0f})
        for (int i = 0; i < n; ++i) x.push_back(nd(rng) * sigma);
    for (int i = 0; i <= 8000; ++i) x.push_back(-40.0f + 80.0f * float(i) / 8000.0f);
    for (int i = 0; i < 2000; ++i) x.push_back(std::ldexp(nd(rng), -(i % 40)));   // tiny |x|
    const int N = int(x.size());
    cpu_ctx c;
    ggml_tensor * t = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, N);
    memcpy(t->data, x.data(), size_t(N) * 4u);
    ggml_tensor * y = ggml_softplus(c.ctx, t);
    c.run(y);
    const float * cpu = static_cast<const float *>(y->data);
    std::vector<uint32_t> ours(size_t(N) + 16);
    std::vector<uint32_t> xin(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) memcpy(&xin[size_t(i)], &x[size_t(i)], 4);
    tsr::ref::softplus_ref_row(uint32_t(N), ours.data(), xin.data());
    double max_rel = 0, max_ulp = 0, max_abs_cpu = 0, cpu_max_rel = 0;
    for (int i = 0; i < N; ++i) {
        float o;
        memcpy(&o, &ours[size_t(i)], 4);
        const double xd = x[size_t(i)];
        const double truth = std::max(xd, 0.0) + std::log1p(std::exp(-std::fabs(xd)));
        if (truth > 1e-30) {
            max_rel = std::max(max_rel, std::fabs(double(o) - truth) / truth);
            cpu_max_rel = std::max(cpu_max_rel, std::fabs(double(cpu[i]) - truth) / truth);
        }
        if (truth > 1e-3) max_ulp = std::max(max_ulp, std::fabs(double(o) - truth) / ulp_of(float(truth)));
        max_abs_cpu = std::max(max_abs_cpu, std::fabs(double(o) - double(cpu[i])));
    }
    printf("softplus: %d values; ours vs float64: max rel %.3e, max %.2f ulp (results > 1e-3); ggml-cpu vs float64: max rel %.3e; "
           "ours vs ggml-cpu: max abs %.3e\n", N, max_rel, max_ulp, cpu_max_rel, max_abs_cpu);
    CHECK(max_rel <= 4.2e-7, "softplus: relative error %.3e against float64", max_rel);
    CHECK(max_ulp <= 5.0, "softplus: %.2f ulp against float64", max_ulp);
    CHECK(max_abs_cpu <= 1.2e-6, "softplus: absolute difference to ggml-cpu %.3e", max_abs_cpu);
}

// ---------------------------------------------------------------------------------------------
// ssm_conv (+ SILU) with the state, as qwen35's graph builds it
// ---------------------------------------------------------------------------------------------
void test_ssm_conv() {
    std::mt19937_64 rng(2);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int T : {1, 4, 33}) {
        const int C = 2048, dc = 4, Lc = C / 16;
        std::vector<float> st(size_t(dc - 1) * C), x(size_t(T) * C), w(size_t(dc) * C);
        for (auto & v : st) v = nd(rng);
        for (auto & v : x) v = nd(rng);
        for (auto & v : w) v = nd(rng) * 0.5f;
        cpu_ctx c;
        // ggml layouts: state [dc-1, C], x [C, T] (transposed to [T, C]), kernel [dc, C] time-fastest
        ggml_tensor * ts = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, dc - 1, C);
        ggml_tensor * tx = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, C, T);
        ggml_tensor * tw = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, dc, C);
        for (int ch = 0; ch < C; ++ch) {
            for (int j = 0; j < dc - 1; ++j) static_cast<float *>(ts->data)[ch * (dc - 1) + j] = st[size_t(j) * C + ch];
            for (int j = 0; j < dc; ++j) static_cast<float *>(tw->data)[ch * dc + j] = w[size_t(j) * C + ch];
        }
        memcpy(tx->data, x.data(), x.size() * 4u);
        ggml_tensor * cin = ggml_concat(c.ctx, ts, ggml_transpose(c.ctx, tx), 0);
        ggml_tensor * out = ggml_silu(c.ctx, ggml_ssm_conv(c.ctx, cin, tw));
        c.run(out);
        const float * cpu = static_cast<const float *>(out->data);   // [C, T]
        // ours
        dview d(8u << 20);
        const uint32_t hw = d.take(uint64_t(dc) * Lc), hsi = d.take(uint64_t(dc - 1) * Lc), hso = d.take(uint64_t(dc - 1) * Lc),
                       hx = d.take(uint64_t(T) * Lc), hd = d.take(uint64_t(T) * Lc);
        for (int j = 0; j < dc; ++j) put_rows(d, hw, uint32_t(j), uint32_t(Lc), &w[size_t(j) * C], uint32_t(C));
        for (int j = 0; j < dc - 1; ++j) put_rows(d, hsi, uint32_t(j), uint32_t(Lc), &st[size_t(j) * C], uint32_t(C));
        for (int t = 0; t < T; ++t) put_rows(d, hx, uint32_t(t), uint32_t(Lc), &x[size_t(t) * C], uint32_t(C));
        const uint32_t a[12] = {hd, uint32_t(Lc), hx, uint32_t(Lc), hsi, hso, hw, uint32_t(C), uint32_t(dc), uint32_t(T), 1u,
                                opl::sconv::FLAG_SILU};
        CHECK(tsr::ref::sconv_ref(d.view(), a) == tsr::ref::GD_OK, "ssm_conv: reference refused");
        double max_rel = 0;
        std::vector<float> row(static_cast<size_t>(C));
        for (int t = 0; t < T; ++t) {
            get_rows(d, hd, uint32_t(t), uint32_t(Lc), row.data(), uint32_t(C));
            for (int ch = 0; ch < C; ++ch) {
                const double ref = cpu[size_t(t) * C + ch];
                double scale = 0;                      // the largest |product| of the channel's window
                for (int j = 0; j < dc; ++j) {
                    const int tt = t + j;
                    const double xv = tt < dc - 1 ? st[size_t(tt) * C + ch] : x[size_t(tt - (dc - 1)) * C + ch];
                    scale = std::max(scale, std::fabs(xv * w[size_t(j) * C + ch]));
                }
                max_rel = std::max(max_rel, std::fabs(double(row[size_t(ch)]) - ref) / std::max(scale, 1e-3));
            }
        }
        // the new state: the last dc-1 window rows
        bool state_ok = true;
        for (int j = 0; j < dc - 1; ++j) {
            get_rows(d, hso, uint32_t(j), uint32_t(Lc), row.data(), uint32_t(C));
            for (int ch = 0; ch < C; ++ch) {
                const int tt = T + j;
                const float want = tt < dc - 1 ? st[size_t(tt) * C + ch] : x[size_t(tt - (dc - 1)) * C + ch];
                state_ok = state_ok && std::memcmp(&row[size_t(ch)], &want, 4) == 0;
            }
        }
        printf("ssm_conv T=%d: max error %.3e of the largest product; state %s\n", T, max_rel, state_ok ? "exact" : "WRONG");
        CHECK(max_rel <= 2e-6, "ssm_conv T=%d: error %.3e", T, max_rel);
        CHECK(state_ok, "ssm_conv T=%d: the new state differs", T);
    }
}

// ---------------------------------------------------------------------------------------------
// gated delta net
// ---------------------------------------------------------------------------------------------
// float64 recurrence (token order, rows independent)
void gdn_f64(int S, int Hk, int Hv, int T, const std::vector<float> & q, const std::vector<float> & k,
             const std::vector<float> & v, const std::vector<float> & g, const std::vector<float> & b,
             std::vector<double> & state, std::vector<double> & o) {
    o.assign(size_t(S) * Hv * T, 0.0);
    const double scale = 1.0 / std::sqrt(double(S));
    std::vector<double> delta(static_cast<size_t>(S));
    for (int h = 0; h < Hv; ++h) {
        double * Sm = &state[size_t(h) * S * S];   // row j: S[.][j]
        for (int t = 0; t < T; ++t) {
            const float * qq = &q[(size_t(t) * Hk + h % Hk) * S];
            const float * kk = &k[(size_t(t) * Hk + h % Hk) * S];
            const float * vv = &v[(size_t(t) * Hv + h) * S];
            const double al = std::exp(double(g[size_t(t) * Hv + h])), be = b[size_t(t) * Hv + h];
            for (int j = 0; j < S; ++j) {
                double sum = 0;
                for (int i = 0; i < S; ++i) {
                    Sm[size_t(j) * S + i] *= al;
                    sum += Sm[size_t(j) * S + i] * kk[i];
                }
                delta[size_t(j)] = (vv[j] - sum) * be;
            }
            for (int j = 0; j < S; ++j) {
                double w = 0;
                for (int i = 0; i < S; ++i) {
                    Sm[size_t(j) * S + i] += kk[i] * delta[size_t(j)];
                    w += Sm[size_t(j) * S + i] * qq[i];
                }
                o[(size_t(t) * Hv + h) * S + j] = w * scale;
            }
        }
    }
}

void test_gdn(int T) {
    std::mt19937_64 rng(3 + uint64_t(T));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ug(-1.5f, -0.002f), ub(0.0f, 1.0f);
    const int S = 128, Hk = 4, Hv = 16, L = S / 16;   // Hv a multiple of 16: g / beta rows are one line
    std::vector<float> q(size_t(T) * Hk * S), k(q.size()), v(size_t(T) * Hv * S), g(size_t(T) * Hv), b(g.size()), st0(size_t(Hv) * S * S);
    auto l2 = [&](std::vector<float> & a) {
        for (size_t r = 0; r < a.size() / S; ++r) {
            double n2 = 0;
            for (int i = 0; i < S; ++i) n2 += double(a[r * S + size_t(i)]) * a[r * S + size_t(i)];
            const float inv = float(1.0 / std::sqrt(n2));
            for (int i = 0; i < S; ++i) a[r * S + size_t(i)] *= inv;
        }
    };
    for (auto & e : q) e = nd(rng);
    for (auto & e : k) e = nd(rng);
    l2(q);
    l2(k);
    for (auto & e : v) e = nd(rng);
    for (auto & e : g) e = ug(rng);
    for (auto & e : b) e = ub(rng);
    for (auto & e : st0) e = nd(rng) * 0.2f;

    // ggml-cpu
    cpu_ctx c;
    ggml_tensor * tq = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, S, Hk, T, 1), * tk = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, S, Hk, T, 1),
                * tv = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, S, Hv, T, 1), * tg = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 1, Hv, T, 1),
                * tb = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 1, Hv, T, 1), * ts = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, S, S, Hv, 1);
    memcpy(tq->data, q.data(), q.size() * 4u);
    memcpy(tk->data, k.data(), k.size() * 4u);
    memcpy(tv->data, v.data(), v.size() * 4u);
    memcpy(tg->data, g.data(), g.size() * 4u);
    memcpy(tb->data, b.data(), b.size() * 4u);
    memcpy(ts->data, st0.data(), st0.size() * 4u);
    ggml_tensor * out = ggml_gated_delta_net(c.ctx, tq, tk, tv, tg, tb, ts, 1);
    c.run(out);
    const float * cpu = static_cast<const float *>(out->data);
    const float * cpu_o = cpu;
    const float * cpu_st = cpu + size_t(S) * Hv * T;

    // float64
    std::vector<double> st64(st0.begin(), st0.end()), o64;
    gdn_f64(S, Hk, Hv, T, q, k, v, g, b, st64, o64);

    // ours
    const uint32_t Lg = 1;
    dview d(24u << 20);
    const uint32_t hq = d.take(uint64_t(T) * Hk * L), hk = d.take(uint64_t(T) * Hk * L), hv = d.take(uint64_t(T) * Hv * L),
                   hg = d.take(uint64_t(T) * Lg), hb = d.take(uint64_t(T) * Lg), hsi = d.take(uint64_t(Hv) * S * L),
                   hso = d.take(uint64_t(Hv) * S * L), hd = d.take(uint64_t(T) * Hv * L);
    for (int t = 0; t < T; ++t) {
        put_rows(d, hq, uint32_t(t), uint32_t(Hk * L), &q[size_t(t) * Hk * S], uint32_t(Hk * S));
        put_rows(d, hk, uint32_t(t), uint32_t(Hk * L), &k[size_t(t) * Hk * S], uint32_t(Hk * S));
        put_rows(d, hv, uint32_t(t), uint32_t(Hv * L), &v[size_t(t) * Hv * S], uint32_t(Hv * S));
        put_rows(d, hg, uint32_t(t), Lg, &g[size_t(t) * Hv], uint32_t(Hv));
        put_rows(d, hb, uint32_t(t), Lg, &b[size_t(t) * Hv], uint32_t(Hv));
    }
    put_rows(d, hsi, 0, 1, st0.data(), uint32_t(st0.size()));
    namespace gg = opl::gdn;
    uint32_t a[gg::NARGS] = {};
    const float sc = 1.0f / std::sqrt(float(S));
    uint32_t scb;
    memcpy(&scb, &sc, 4);
    a[gg::DST] = hd; a[gg::STATE_OUT] = hso; a[gg::STATE_IN] = hsi;
    a[gg::Q] = hq; a[gg::Q_HEAD_STRIDE] = L; a[gg::Q_TOKEN_STRIDE] = Hk * L;
    a[gg::K] = hk; a[gg::K_HEAD_STRIDE] = L; a[gg::K_TOKEN_STRIDE] = Hk * L;
    a[gg::V] = hv; a[gg::V_HEAD_STRIDE] = L; a[gg::V_TOKEN_STRIDE] = Hv * L;
    a[gg::G] = hg; a[gg::G_TOKEN_STRIDE] = Lg; a[gg::BETA] = hb; a[gg::BETA_TOKEN_STRIDE] = Lg;
    a[gg::S_V] = S; a[gg::H_K] = Hk; a[gg::H_V] = Hv; a[gg::N_TOKENS] = uint32_t(T); a[gg::N_SEQS] = 1; a[gg::SCALE] = scb;
    CHECK(tsr::ref::gdn_ref(d.view(), a) == tsr::ref::GD_OK, "gdn: reference refused");
    std::vector<float> our_o(size_t(S) * Hv * T), our_st(st0.size());
    for (int n = 0; n < T * Hv; ++n) get_rows(d, hd, uint32_t(n), L, &our_o[size_t(n) * S], uint32_t(S));
    get_rows(d, hso, 0, 1, our_st.data(), uint32_t(our_st.size()));

    auto rel = [](const float * a_, const double * ref, size_t n) {   // error / rms of the reference
        double e2 = 0, r2 = 0;
        for (size_t i = 0; i < n; ++i) {
            e2 += (double(a_[i]) - ref[i]) * (double(a_[i]) - ref[i]);
            r2 += ref[i] * ref[i];
        }
        return std::sqrt(e2 / std::max(r2, 1e-300));
    };
    std::vector<double> cpu_o64(cpu_o, cpu_o + our_o.size()), cpu_st64(cpu_st, cpu_st + our_st.size());
    const double e_ours_o = rel(our_o.data(), o64.data(), our_o.size()), e_cpu_o = rel(cpu_o, o64.data(), our_o.size());
    const double e_ours_s = rel(our_st.data(), st64.data(), our_st.size()), e_cpu_s = rel(cpu_st, st64.data(), our_st.size());
    const double e_vs_cpu_o = rel(our_o.data(), cpu_o64.data(), our_o.size()), e_vs_cpu_s = rel(our_st.data(), cpu_st64.data(), our_st.size());
    printf("gdn T=%d: rms error vs float64  out ours %.3e cpu %.3e | state ours %.3e cpu %.3e | ours vs cpu  out %.3e state %.3e\n", T,
           e_ours_o, e_cpu_o, e_ours_s, e_cpu_s, e_vs_cpu_o, e_vs_cpu_s);
    CHECK(e_ours_o <= 4.0 * e_cpu_o + 2e-6 && e_ours_s <= 4.0 * e_cpu_s + 2e-6,
          "gdn T=%d: ours (%.3e / %.3e) is worse than 4x the CPU's (%.3e / %.3e) against float64", T, e_ours_o, e_ours_s, e_cpu_o, e_cpu_s);
    CHECK(e_vs_cpu_o <= 1e-3 && e_vs_cpu_s <= 1e-3, "gdn T=%d: ours vs cpu %.3e / %.3e", T, e_vs_cpu_o, e_vs_cpu_s);
}

}  // namespace

int main() {
    test_softplus();
    test_ssm_conv();
    for (int T : {1, 16, 512}) test_gdn(T);
    return test_finish("fw_gdn_ggml");
}

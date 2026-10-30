// test_fw_normrope_ggml.cpp — fw_kernels_normrope_ggml: the RMS_NORM and ROPE kinds' outputs (the
// RISC-V image on tsr_rv through `graph`, sw/ctrl/kernels/k_norm.cpp / k_rope.cpp) against ggml-cpu's
// rms_norm (+ mul by the weight) and rope_ext on the same inputs (docs/design/06-software-verification.md
// "SW" §7.5; VU §6.9: no VU op is bit-exact with the CPU in general, the differences are measured
// here at kernel level). TSR_WITH_LLAMA builds only (links the pinned ggml-cpu).
//
//   RMS_NORM: n = 896, 5120, 3840 (one row per token, weight) and the per-head 256 / 512 rows, 64
//   rows each with per-row scales 1e-3 .. 1e3 and rare outliers, eps 1e-6: the ulp distance of every
//   output to ggml_mul(ggml_rms_norm(x, eps), w) (the CPU sums in double and divides by n; the VU sums
//   in f32 with 128 partials and multiplies by f32(1/n), VU §6.9). Gate: <= 8 ulp (W1a measured the
//   scale within 2 ulp; the weight multiply adds at most one rounding).
//   ROPE: Qwen2.5-0.5B / 32B (NEOX, base 1e6, head_dim 64 / 128), Gemma 4 SWA (NEOX 256, base 1e4) and
//   global (NEOX 512, base 1e6, rope_freqs: 1 for the first quarter, 1e30 after), 2 heads, 256 tokens
//   at positions spread over 0 .. 32767 (0, 1 and 32767 included): unit pairs (x0 = 1, x1 = 0: the
//   outputs are cos / sin of the angles; the VU's COS / SIN against libm, VU §6.6) and random inputs
//   (the full op). Max |difference| reported; gate 8e-6 (W1a: 4.77e-7 on the full op, ref level).
// Exit 0 / 1; 77 (skipped) without the image.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <sys/stat.h>

#include <cmath>
#include <vector>

#include "../ref/fw_sim_host.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tsr/opl.hpp"
#include "tsr_ref.h"

namespace {

namespace opl = tsr::opl;
namespace rr = tsr::opl::rms;
namespace ro = tsr::opl::rope;
constexpr uint64_t LINE = 64u;
const char * g_image = nullptr;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t L16(uint32_t n) { return (n + 15u) / 16u; }
uint32_t fb(float f) { return f2u(f); }

tsr_config cfg_u200() {
    tsr_config c = TSR_CONFIG_U200_V1;
    c.t_max = 8;
    c.ch_bytes = 1ull << 30;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}

// One launch of `blob` (the list at ch0 beat 0x40000) on a fresh simulator whose DRAM `fill` set up;
// `read` collects the outputs. Returns false on a failed launch.
template <class Fill, class Read>
bool run_list(const std::vector<uint8_t> & blob, Fill fill, Read read) {
    const tsr_config cfg = cfg_u200();
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = 2;
    so.threads = 4;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &so);
    fill(sim);
    tsr_sim_dram_write(sim, 0, 0x40000ull * LINE, blob.data(), blob.size());
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    if (ok) {
        const uint32_t a[7] = { H(0, 0x40000u), uint32_t(blob.size()), 0, 0, 0, 0, 0 };
        ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0 && fwh_status(&h) == 0 && fwh_ret(&h) == 0;
    }
    CHECK(ok, "launch: DONE 0x%08x RET 0x%08x", h.done[0], h.done[2]);
    if (ok) read(sim);
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

uint32_t ulp_dist(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return (std::isnan(a) && std::isnan(b)) ? 0u : 0xFFFFFFFFu;
    const int32_t ia = int32_t(fb(a)), ib = int32_t(fb(b));
    const int64_t ka = ia < 0 ? int64_t(INT32_MIN) - ia : ia, kb = ib < 0 ? int64_t(INT32_MIN) - ib : ib;   // monotone keys
    const int64_t d = ka > kb ? ka - kb : kb - ka;
    return d > 0xFFFFFFFFll ? 0xFFFFFFFFu : uint32_t(d);
}

// ---------------------------------------------------------------------------
// RMS_NORM
// ---------------------------------------------------------------------------
void test_rms() {
    struct shp { const char * name; uint32_t n, rows; bool w; };
    const shp shapes[] = {
        { "Qwen2.5-0.5B 896 (weight)", 896, 64, true },     { "Qwen2.5-32B 5120 (weight)", 5120, 64, true },
        { "Gemma 4 12B 3840 (weight)", 3840, 64, true },    { "Gemma 4 per-head 256 (weight)", 256, 64, true },
        { "Gemma 4 per-head 512 (weight)", 512, 64, true }, { "Gemma 4 v_norm 256 (no weight)", 256, 64, false },
        { "Qwen2.5-0.5B 896 (no weight)", 896, 64, false },
    };
    const float eps = 1e-6f;
    rng_t r = { 0x99E7ull };
    for (const auto & sh : shapes) {
        const uint32_t n = sh.n, R = sh.rows, L = L16(n);
        std::vector<float> x(size_t(n) * R), w(n);
        for (uint32_t i = 0; i < R; ++i) {
            const double sc = std::pow(10.0, double(rng_below(&r, 7)) - 3.0);
            for (uint32_t j = 0; j < n; ++j)
                x[size_t(i) * n + j] = float(rng_normal(&r) * sc * (rng_below(&r, 100) == 0 ? 30.0 : 1.0));
        }
        // Without a weight, word 0 of every row is 1.0: the kernel's output there is its row scale
        // FM(1, s) = s, compared with the CPU's 1.0f / sqrtf(f32(double sum / n) + eps) (ops.cpp).
        if (!sh.w)
            for (uint32_t i = 0; i < R; ++i) x[size_t(i) * n] = 1.0f;
        for (uint32_t j = 0; j < n; ++j) w[j] = float(1.0 + 0.2 * rng_normal(&r));
        // The device: one record (rows dense, L lines each).
        const uint32_t src = H(1, 0x10000u), dst = H(2, 0x10000u), wh = H(3, 0x10000u);
        opl::writer wr(opl::F_STOP_ON_ERROR, 0x6E);
        const opl::rms_norm_args ra{dst, L, src, L, n, R, fb(eps), fb(tsr_fdiv(1.0f, float(n))), sh.w ? wh : opl::HANDLE_NONE,
                                    1u};
        wr.add_rms_norm(ra);
        const std::vector<uint8_t> blob = wr.finish();
        std::vector<float> dev(size_t(n) * R);
        run_list(blob, [&](tsr_sim * sim) {
            std::vector<uint8_t> buf(size_t(L) * LINE);
            for (uint32_t i = 0; i < R; ++i) {
                memset(buf.data(), 0, buf.size());
                put_f32_le(buf.data(), &x[size_t(i) * n], n);
                tsr_sim_dram_write(sim, 1, (0x10000ull + uint64_t(i) * L) * LINE, buf.data(), buf.size());
            }
            memset(buf.data(), 0, buf.size());
            put_f32_le(buf.data(), w.data(), n);
            tsr_sim_dram_write(sim, 3, 0x10000ull * LINE, buf.data(), buf.size());
        }, [&](tsr_sim * sim) {
            for (uint32_t i = 0; i < R; ++i) {
                const uint8_t * p = tsr_sim_dram(sim, 2) + (0x10000ull + uint64_t(i) * L) * LINE;
                for (uint32_t j = 0; j < n; ++j) dev[size_t(i) * n + j] = u2f(get_u32_le(p + 4u * j));
            }
        });
        // ggml-cpu.
        struct ggml_init_params ip = { size_t(4) * n * R * sizeof(float) + (16u << 20), nullptr, false };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, R);
        memcpy(a->data, x.data(), x.size() * sizeof(float));
        struct ggml_tensor * y = ggml_rms_norm(ctx, a, eps);
        if (sh.w) {
            struct ggml_tensor * wt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            memcpy(wt->data, w.data(), n * sizeof(float));
            y = ggml_mul(ctx, y, wt);
        }
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        const float * yc = static_cast<const float *>(y->data);
        uint32_t max_ulp = 0;
        long same = 0;
        double max_rel = 0;
        for (size_t k = 0; k < dev.size(); ++k) {
            const uint32_t u = ulp_dist(dev[k], yc[k]);
            max_ulp = std::max(max_ulp, u);
            same += u == 0u;
            if (yc[k] != 0.0f) max_rel = std::max(max_rel, std::fabs(double(dev[k]) - double(yc[k])) / std::fabs(double(yc[k])));
        }
        uint32_t scale_ulp = 0;
        long scale_same = 0;
        if (!sh.w) {
            for (uint32_t i = 0; i < R; ++i) {
                double sum = 0;
                for (uint32_t j = 0; j < n; ++j) sum += double(x[size_t(i) * n + j] * x[size_t(i) * n + j]);
                const float mean = float(sum / n);
                const float scale = 1.0f / sqrtf(mean + eps);
                const uint32_t u = ulp_dist(dev[size_t(i) * n], scale);
                scale_ulp = std::max(scale_ulp, u);
                scale_same += u == 0u;
            }
        }
        printf("  rms_norm %-32s %3u rows x %4u: outputs max %u ulp (max rel %.3g), %.1f%% bit-identical to ggml-cpu",
               sh.name, R, n, max_ulp, max_rel, 100.0 * double(same) / double(dev.size()));
        if (!sh.w) printf("; row scale max %u ulp, %ld of %u rows identical", scale_ulp, scale_same, R);
        printf("\n");
        if (!sh.w) CHECK(scale_ulp <= 4u, "rms_norm %s: row scale %u ulp from the CPU's (gate 4)", sh.name, scale_ulp);
        CHECK(max_ulp <= 8u, "rms_norm %s: %u ulp from ggml-cpu (gate 8)", sh.name, max_ulp);
        ggml_free(ctx);
    }
}

// ---------------------------------------------------------------------------
// ROPE
// ---------------------------------------------------------------------------
struct rmodel {
    const char * name;
    float base;
    uint32_t hd;
    bool gemma_ff;
};

void test_rope() {
    const rmodel models[] = {
        { "Qwen2.5-0.5B (NEOX 64, base 1e6)", 1e6f, 64, false },
        { "Qwen2.5-32B (NEOX 128, base 1e6)", 1e6f, 128, false },
        { "Gemma 4 SWA (NEOX 256, base 1e4)", 1e4f, 256, false },
        { "Gemma 4 global (NEOX 512, base 1e6, rope_freqs)", 1e6f, 512, true },
    };
    const uint32_t T = 256, NH = 2;
    rng_t r = { 0x40E1ull };
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = t == 0 ? 0 : t == 1 ? 1 : t == T - 1 ? 32767 : int32_t(rng_below(&r, 32768u));
    for (const auto & m : models) {
        const uint32_t hd = m.hd, nd = hd, half = nd / 2, Hl = NH * hd / 16;
        std::vector<float> ff(half, 1.0f);
        if (m.gemma_ff)
            for (uint32_t i = half / 4; i < half; ++i) ff[i] = 1e30f;
        for (int unit = 1; unit >= 0; --unit) {
            std::vector<float> x(size_t(T) * NH * hd, 0.0f);
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t h = 0; h < NH; ++h)
                    for (uint32_t j = 0; j < hd; ++j)
                        x[(size_t(t) * NH + h) * hd + j] = unit ? (j < half ? 1.0f : 0.0f) : float(rng_normal(&r));
            // The device: one ROPE record over the T tokens (tiles of the planning budget).
            const uint32_t src = H(1, 0x10000u), dst = H(2, 0x10000u), tab = H(3, 0x10000u), posh = H(0, 0x80000u);
            opl::writer wr(opl::F_STOP_ON_ERROR, 0x6F);
            const uint32_t pb = wr.add_preload(posh, 4u * T);
            const opl::rope_args ra{dst, Hl, src, Hl, pb, m.gemma_ff ? tab : opl::HANDLE_NONE,
                                    fb(powf(m.base, -2.0f / float(nd))), fb(1.0f), fb(1.0f), hd, nd, NH, T, ro::MODE_NEOX};
            wr.add_rope(ra);
            const std::vector<uint8_t> blob = wr.finish();
            std::vector<float> dev(x.size());
            run_list(blob, [&](tsr_sim * sim) {
                std::vector<uint8_t> b(size_t(T) * Hl * LINE);
                put_f32_le(b.data(), x.data(), x.size());
                tsr_sim_dram_write(sim, 1, 0x10000ull * LINE, b.data(), b.size());
                std::vector<uint8_t> pbuf(size_t(L16(T)) * LINE, 0);
                for (uint32_t t = 0; t < T; ++t) memcpy(pbuf.data() + 4u * t, &pos[t], 4u);
                tsr_sim_dram_write(sim, 0, 0x80000ull * LINE, pbuf.data(), pbuf.size());
                std::vector<uint8_t> tb(size_t(L16(half)) * LINE, 0);
                put_f32_le(tb.data(), ff.data(), half);
                tsr_sim_dram_write(sim, 3, 0x10000ull * LINE, tb.data(), tb.size());
            }, [&](tsr_sim * sim) {
                const uint8_t * p = tsr_sim_dram(sim, 2) + 0x10000ull * LINE;
                for (size_t k = 0; k < dev.size(); ++k) dev[k] = u2f(get_u32_le(p + 4u * k));
            });
            // ggml-cpu rope_ext: [hd, NH, T], positions as given.
            struct ggml_init_params ip = { 4 * x.size() * sizeof(float) + (64u << 20), nullptr, false };
            struct ggml_context * ctx = ggml_init(ip);
            struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, NH, T);
            memcpy(a->data, x.data(), x.size() * sizeof(float));
            struct ggml_tensor * pt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
            memcpy(pt->data, pos.data(), T * sizeof(int32_t));
            struct ggml_tensor * fft = nullptr;
            if (m.gemma_ff) {
                fft = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half);
                memcpy(fft->data, ff.data(), half * sizeof(float));
            }
            struct ggml_tensor * y = ggml_rope_ext(ctx, a, pt, fft, int(nd), GGML_ROPE_TYPE_NEOX, 32768, m.base, 1.0f, 0.0f,
                                                   1.0f, 32.0f, 1.0f);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_graph_compute_with_ctx(ctx, gf, 4);
            const float * yc = static_cast<const float *>(y->data);
            double maxd = 0;
            long same = 0;
            for (size_t k = 0; k < dev.size(); ++k) {
                maxd = std::max(maxd, std::fabs(double(dev[k]) - double(yc[k])));
                same += fb(dev[k]) == fb(yc[k]);
            }
            printf("  rope %-48s %s, %u tokens over 0..32767: max |kernel - ggml-cpu| %.3g, %.1f%% bit-identical\n", m.name,
                   unit ? "unit pairs (cos / sin)" : "random x (full op)  ", T, maxd, 100.0 * double(same) / double(dev.size()));
            CHECK(maxd < 8e-6, "rope %s: max |diff| %g (gate 8e-6)", m.name, maxd);
            ggml_free(ctx);
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s IMAGE\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    struct stat st;
    if (stat(g_image, &st) != 0) {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    printf("fw_kernels_normrope_ggml: kernel outputs (tsr_rv, `graph`) against ggml-cpu\n");
    test_rms();
    test_rope();
    return test_finish("fw_kernels_normrope_ggml");
}

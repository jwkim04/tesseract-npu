// Host references of the Ising machine kernel (ising_ref.hpp).
#include "ising_ref.hpp"

#include <math.h>
#include <string.h>

#include "tsr_ref.h"
#include "tsr_ref_vu.h"

namespace ism {

namespace {

// The constants of the step as the host sees them. They are written here again, in float notation, and
// not shared with the kernel: a wrong constant in the kernel must show as a difference.
const float BIG = ldexpf(1.0f, 100);
const float ONE = 1.0f;
const float MINUS_ONE = -1.0f;
const float P24 = 16777216.0f;
const float M24 = -16777216.0f;
const uint32_t SCALE_BITS = 0x3C010204u;   // the f32 nearest to 1/127 (checked in the test)

inline float mint(float a, float b) { return tsr_bits_f32(tsr_ref_vu_mint(tsr_f32_bits(a), tsr_f32_bits(b))); }
inline float maxt(float a, float b) { return tsr_bits_f32(tsr_ref_vu_maxt(tsr_f32_bits(a), tsr_f32_bits(b))); }
inline float clip(float v) { return maxt(mint(v, ONE), MINUS_ONE); }
inline float f32_of_bits(uint32_t b) { return tsr_bits_f32(b); }

}  // namespace

schedule make_schedule(double a0, double c0, double dt, uint32_t steps) {
    auto bits = [](double v) { return tsr_f32_bits(float(v)); };
    return {bits(-a0 * dt), bits(steps ? a0 * dt / steps : 0.0), bits(c0 * dt), bits(a0 * dt)};
}

uint32_t advance_ca(uint32_t ca, uint32_t da, uint32_t steps) {
    float c = f32_of_bits(ca);
    const float d = f32_of_bits(da);
    for (uint32_t i = 0; i < steps; ++i) c = tsr_fadd(c, d);
    return tsr_f32_bits(c);
}

dense_j::dense_j(const problem & p) : n(p.n), a_(size_t(p.n) * p.n) {
    for (uint32_t i = 0; i < p.n; ++i) p.row(i, a_.data() + size_t(i) * p.n);
}

void bitexact_launch(const dense_j & J, uint32_t t, uint32_t kblk, uint32_t steps, uint32_t snap_every, bool energy,
                     const schedule & sch, float * x, float * y, bitexact_out * out) {
    const uint32_t n = J.n;
    const uint32_t nblk = (n + kblk - 1) / kblk;
    float ca = f32_of_bits(sch.ca0);
    const float da = f32_of_bits(sch.da), cf = f32_of_bits(sch.cf), cd = f32_of_bits(sch.cd);
    std::vector<float> s(size_t(t) * n), f(size_t(t) * n), tmp(n);
    std::vector<uint8_t> qa;
    // The weights as the upload packs them: for every column tensor the n rows in the device row layout, with
    // the block scale 1.0.
    std::vector<std::vector<uint8_t>> packed(nblk);
    for (uint32_t b = 0; b < nblk; ++b) {
        const uint32_t col0 = b * kblk, k = (b + 1 < nblk) ? kblk : n - col0;
        const size_t rb = size_t(tsr_row_beats(TSR_FMT_Q8_0, k)) * 64;
        std::vector<tsr_block_q8_0> blocks(k / 32);
        packed[b].resize(size_t(n) * rb);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < k / 32; ++j) {
                blocks[j].d = tsr_f32_to_fp16(1.0f);
                memcpy(blocks[j].qs, J.row(i) + col0 + j * 32, 32);
            }
            tsr_repack_row(TSR_FMT_Q8_0, blocks.data(), k, packed[b].data() + i * rb);
        }
    }
    for (uint32_t step = 0; step < steps; ++step) {
        // s = clip(x * 2^100)
        for (size_t i = 0; i < s.size(); ++i) s[i] = clip(tsr_fmul(x[i], BIG));
        const float * act = s.data();
        // f = J s: per column tensor and replica a QA image with the replaced scale, the row sums of 16
        // partials and the tree, the tensors added in order.
        for (uint32_t r = 0; r < t; ++r) {
            const float * sr = act + size_t(r) * n;
            float * fr = f.data() + size_t(r) * n;
            for (uint32_t b = 0; b < nblk; ++b) {
                const uint32_t col0 = b * kblk, k = (b + 1 < nblk) ? kblk : n - col0;
                qa.assign(size_t(tsr_qa_lines(TSR_QA_A32, k)) * 64, 0);
                tsr_qact(TSR_QA_A32, sr + col0, k, qa.data());
                float * dx = reinterpret_cast<float *>(qa.data() + 64 * size_t(tsr_qa_qs_lines(TSR_QA_A32, k)));   // the scale words
                for (uint32_t j = 0; j < k / 32; ++j) dx[j] = f32_of_bits(SCALE_BITS);
                const size_t rb = packed[b].size() / n;
                for (uint32_t i = 0; i < n; ++i) {
                    const float y_row = tsr_dot_row(TSR_FMT_Q8_0, packed[b].data() + i * rb, k, qa.data());
                    fr[i] = b == 0 ? y_row : tsr_fadd(fr[i], y_row);
                }
            }
        }
        // the energy: e = sum_i s_i f_i (RED SUM over the products)
        if (energy) {
            for (uint32_t r = 0; r < t; ++r) {
                for (uint32_t i = 0; i < n; ++i) tmp[i] = tsr_fmul(act[size_t(r) * n + i], f[size_t(r) * n + i]);
                uint32_t words[1];
                words[0] = tsr_ref_red_row(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, reinterpret_cast<const uint32_t *>(tmp.data()), n, 0, 0);
                out->energy.push_back(tsr_bits_f32(words[0]));
            }
        }
        // y1 = x cA + y, y2 = f cF + y1, x1 = y2 cD + x, x2 = clip(x1)
        for (size_t i = 0; i < s.size(); ++i) {
            const float y1 = tsr_fadd(tsr_fmul(x[i], ca), y[i]);
            const float y2 = tsr_fadd(tsr_fmul(f[i], cf), y1);
            const float x1 = tsr_fadd(tsr_fmul(y2, cd), x[i]);
            const float x2 = clip(x1);
            // m = min(2^24 - 2^24 x2 x2, 1): 1 inside the walls, 0 at |x2| = 1
            const float m = mint(tsr_fadd(tsr_fmul(tsr_fmul(x2, M24), x2), P24), ONE);
            x[i] = x2;
            y[i] = tsr_fmul(y2, m);
        }
        if (snap_every != 0 && (step + 1) % snap_every == 0) out->snapshots.insert(out->snapshots.end(), x, x + size_t(t) * n);
        ca = tsr_fadd(ca, da);
    }
}

}  // namespace ism

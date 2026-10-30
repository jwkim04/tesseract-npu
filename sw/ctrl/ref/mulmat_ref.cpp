// mulmat_ref.cpp — see mulmat_ref.hpp (SW §7.4: the MUL_MAT kind's tile-op sequence as tsr_ref calls).
#include "mulmat_ref.hpp"

#include <cstring>
#include <thread>
#include <vector>

#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_ref.h"

namespace tsr {
namespace ref {

namespace {

constexpr uint64_t LINE = 64u;

// Byte pointer of handle h {ch[31:28], beat[27:0]} plus `lines` lines, `bytes` long; nullptr if
// the range is outside the view.
uint8_t * at(const dram_view & d, uint32_t h, uint64_t lines, uint64_t bytes) {
    const uint32_t ch = h >> 28;
    const uint64_t off = (uint64_t(h & 0x0FFFFFFFu) + lines) * LINE;
    if (ch >= d.num_ch || off + bytes > d.ch_bytes) return nullptr;
    return d.ch[ch] + off;
}

float rdf(const uint8_t * p) {
    const uint32_t u = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void wrf(uint8_t * p, float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    p[0] = uint8_t(u); p[1] = uint8_t(u >> 8); p[2] = uint8_t(u >> 16); p[3] = uint8_t(u >> 24);
}

}  // namespace

void mulmat_ref_token(const dram_view & d, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off, const float * x,
                      const float * add, float * y, uint32_t threads) {
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    std::vector<uint8_t> qa(size_t(tsr_qa_lines(kind, K)) * LINE + 64u);
    // QA views need 4-byte alignment (tsr_qa_d): the vector's storage is.
    tsr_qact(kind, x, K, qa.data());
    auto rows = [&](uint32_t lo, uint32_t hi) {
        for (uint32_t n = lo; n < hi; ++n) {
            uint32_t ch;
            uint64_t boff;
            tsr_row_location(fmt, N, K, d.num_ch, n, &ch, &boff);
            const uint8_t * row = d.ch[ch] + (w_off + boff) * LINE;
            const float v = tsr_dot_row(fmt, row, K, qa.data());
            y[n] = add ? tsr_fadd(add[n], v) : v;      // MVM ACC: tsr_fadd(old, W.x), as the simulator
        }
    };
    if (threads <= 1 || N < 256u) {
        rows(0, N);
        return;
    }
    std::vector<std::thread> pool;
    const uint32_t chunk = (N + threads - 1u) / threads;
    for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t lo = t * chunk, hi = lo + chunk < N ? lo + chunk : N;
        if (lo >= hi) break;
        pool.emplace_back(rows, lo, hi);
    }
    for (auto & th : pool) th.join();
}

int mulmat_ref(const dram_view & d, const uint32_t a[12], uint32_t threads) {
    namespace mm = opl::mm;
    const uint32_t fmt = a[mm::FMT], N = a[mm::N], K = a[mm::K], T = a[mm::T];
    const bool has_add = a[mm::ADD] != opl::HANDLE_NONE;
    const bool acc_dst = (a[mm::FLAGS] & mm::FLAG_ACC_DST) != 0u;
    if ((a[mm::FLAGS] & ~mm::FLAG_ACC_DST) != 0u || (has_add && acc_dst)) return MM_BAD_RECORD;
    // Shape rules: the kernel's (plan.hpp, any device large enough for one token).
    const plan::dev_cfg cfg = {65536u, 8u, 32768u};
    if (plan::mm_plan(cfg, fmt, N, K, T, nullptr) == opl::RET_BAD_RECORD) return MM_BAD_RECORD;
    const uint32_t x_lines = K / 16u, y_lines = plan::ceil_div(N, 16u);
    const uint64_t row_beats = tsr_row_beats(fmt, K);
    const uint64_t w_need = uint64_t(tsr_rows_per_ch(N, d.num_ch)) * row_beats;
    if ((uint64_t(a[mm::W_OFF]) + w_need) * LINE > d.ch_bytes) return MM_RANGE;
    std::vector<float> x(K), add(N), y(N);
    for (uint32_t t = 0; t < T; ++t) {
        const uint8_t * xp = at(d, a[mm::X], uint64_t(t) * a[mm::X_STRIDE], uint64_t(x_lines) * LINE);
        uint8_t * yp = at(d, a[mm::Y], uint64_t(t) * a[mm::Y_STRIDE], uint64_t(y_lines) * LINE);
        if (xp == nullptr || yp == nullptr) return MM_RANGE;
        for (uint32_t k = 0; k < K; ++k) x[k] = rdf(xp + 4u * k);
        const float * addp = nullptr;
        if (has_add || acc_dst) {
            const uint8_t * ap = has_add ? at(d, a[mm::ADD], uint64_t(t) * a[mm::ADD_STRIDE], uint64_t(y_lines) * LINE)
                                         : yp;
            if (ap == nullptr) return MM_RANGE;
            for (uint32_t n = 0; n < N; ++n) add[n] = rdf(ap + 4u * n);
            addp = add.data();
        }
        mulmat_ref_token(d, fmt, N, K, a[mm::W_OFF], x.data(), addp, y.data(), threads);
        for (uint32_t n = 0; n < N; ++n) wrf(yp + 4u * n, y[n]);
        std::memset(yp + 4u * uint64_t(N), 0, size_t(uint64_t(y_lines) * LINE - 4u * uint64_t(N)));
    }
    return MM_OK;
}

}  // namespace ref
}  // namespace tsr

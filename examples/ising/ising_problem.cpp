// The problem of the Ising machine example (ising_problem.hpp).
#include "ising_problem.hpp"

#include <math.h>
#include <string.h>

namespace ism {

uint64_t splitmix64(uint64_t & state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void problem::row(uint32_t i, int8_t * dst) const {
    memset(dst, 0, n);
    if (i >= spins) return;
    for (uint32_t j = 0; j < spins; ++j)
        if (j != i) dst[j] = int8_t(pair_sign(seed, i, j));
}

void problem::initial_state(uint32_t t, uint64_t sd, std::vector<float> * x, std::vector<float> * y) const {
    x->assign(size_t(t) * n, 0.f);
    y->assign(size_t(t) * n, 0.f);
    uint64_t st = sd;
    auto uni = [](uint64_t & state) {   // uniform in (-0.1, 0.1): 24 random bits, never exactly 0
        const uint32_t u = uint32_t(splitmix64(state) >> 40);
        return (float(u) + 0.5f) * (0.2f / 16777216.0f) - 0.1f;
    };
    for (uint32_t r = 0; r < t; ++r)
        for (uint32_t i = 0; i < spins; ++i) {
            (*x)[size_t(r) * n + i] = uni(st);
            (*y)[size_t(r) * n + i] = uni(st);
        }
}

double problem::default_c0() const { return 0.5 * sqrt((double(spins) - 1.0) / (double(spins) * (double(spins) - 1.0))); }

void spins_from_x(const problem & p, const float * x, uint32_t t, std::vector<int8_t> * spins) {
    spins->resize(size_t(t) * p.spins);
    for (uint32_t r = 0; r < t; ++r)
        for (uint32_t i = 0; i < p.spins; ++i) (*spins)[size_t(r) * p.spins + i] = x[size_t(r) * p.n + i] < 0.f ? -1 : 1;
}

void field_energy_sums(const problem & p, const int8_t * s, uint32_t t, std::vector<int64_t> * out) {
    const uint32_t n = p.n;
    out->assign(t, 0);
    std::vector<int8_t> row(n);
    for (uint32_t i = 0; i < n; ++i) {
        p.row(i, row.data());
        for (uint32_t r = 0; r < t; ++r) {
            const int8_t * sr = s + size_t(r) * n;
            if (sr[i] == 0) continue;
            int32_t f = 0;
            for (uint32_t j = 0; j < n; ++j) f += int32_t(row[j]) * int32_t(sr[j]);
            (*out)[r] += int64_t(sr[i]) * f;
        }
    }
}

void evaluate(const problem & p, const int8_t * s, uint32_t t, std::vector<evaluation> * out) {
    const uint32_t n = p.spins;
    std::vector<int64_t> sum(t, 0);
    std::vector<int8_t> row(p.n);
    for (uint32_t i = 0; i < n; ++i) {
        p.row(i, row.data());
        for (uint32_t r = 0; r < t; ++r) {
            const int8_t * sr = s + size_t(r) * n;
            int32_t f = 0;
            for (uint32_t j = 0; j < n; ++j) f += int32_t(row[j]) * int32_t(sr[j]);
            sum[r] += int64_t(sr[i]) * f;
        }
    }
    out->resize(t);
    for (uint32_t r = 0; r < t; ++r) {
        const double e = -0.5 * double(sum[r]);
        (*out)[r] = {e, e / pow(double(n), 1.5)};
    }
}

}  // namespace ism

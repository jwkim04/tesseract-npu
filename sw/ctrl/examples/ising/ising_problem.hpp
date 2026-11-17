// The problem of the Ising machine example: the Sherrington-Kirkpatrick spin glass. The couplings are J_ij = +1 or -1 at
// random for every pair of spins i != j, J_ii = 0, and the energy of the signs s is E = -1/2 sum_ij J_ij s_i s_j. The
// result is quoted as the energy per spin in units of N^1.5, e = E / N^1.5: the ground state tends to -0.7632 for large N
// (Parisi), with a finite-size correction that goes about as N^(-2/3).
//
// The sign of the pair {i, j} comes from splitmix64 of (seed, min, max). It does not depend on the C++ library, and the
// video script sk_video.py has a copy of it.
//
// The problem has `spins` spins. The machine has n = spins padded to a multiple of 32 (the padding spins have no
// couplings). The matrix is dense and its rows are generated on the fly; the block scales of its Q8_0 upload are all 1.0.
#pragma once

#include <stdint.h>

#include <vector>

namespace ism {

uint64_t splitmix64(uint64_t & state);

// The +-1 of the unordered pair {i, j}.
inline int pair_sign(uint64_t seed, uint32_t i, uint32_t j) {
    const uint32_t a = i < j ? i : j, b = i < j ? j : i;
    uint64_t st = seed ^ ((uint64_t(a) << 32) | b);
    return (splitmix64(st) >> 32) & 1u ? 1 : -1;
}

struct problem {
    uint32_t spins = 0;   // at least 2
    uint32_t n = 0;       // the machine's spins: `spins` padded to a multiple of 32
    uint64_t seed = 1;

    problem(uint32_t spins_, uint64_t seed_) : spins(spins_), n((spins_ + 31u) / 32u * 32u), seed(seed_) {}

    // Row i of the machine's matrix: n int8 values (zero for the diagonal and the padding).
    void row(uint32_t i, int8_t * dst) const;
    // The start values of t replicas: t rows of n floats each, x and y uniform in (-0.1, 0.1) from the seed, zero for the padding.
    void initial_state(uint32_t t, uint64_t seed, std::vector<float> * x, std::vector<float> * y) const;
    // The default scale of the coupling term: 0.5 sqrt((N - 1) / sum_ij J_ij^2) with N the spins.
    double default_c0() const;
};

// The signs of t machine states (t rows of n floats): +1 where x >= 0. Out: t rows of `spins` values.
void spins_from_x(const problem & p, const float * x, uint32_t t, std::vector<int8_t> * spins);

// The exact energy E of t rows of `spins` signs, and e = E / N^1.5.
struct evaluation {
    double energy;
    double e;
};
void evaluate(const problem & p, const int8_t * s, uint32_t t, std::vector<evaluation> * out);

// sum_i s_i (J s)_i of the machine's matrix for t rows of n values in {-1, 0, +1}: the exact value of the kernel's energy
// word (which is -2 E for signs without padding).
void field_energy_sums(const problem & p, const int8_t * s, uint32_t t, std::vector<int64_t> * out);

}  // namespace ism

// The bit-exact model of the Ising machine kernel (ising.cpp), host only: the step of the kernel with the operations
// of the simulator's reference library (tsr_ref.h, tsr_ref_vu.h): the same QACT with the replaced scale, and the row
// dot product tsr_dot_row on the weight rows packed as the upload packs them (so the 16 partial sums and the tree are
// the library's). Launch by launch with the same f32 constants. It is the oracle of the tests and of `tsr-ising --check`:
// the device must agree bit for bit (state, snapshots, energies).
#pragma once

#include <stdint.h>

#include <vector>

#include "ising_problem.hpp"

namespace ism {

// The constants of a run as f32 bits. cA starts at -a0 dt and rises by dA = a0 dt / steps per step.
struct schedule {
    uint32_t ca0;   // cA at the start of the launch (a run that is cut into launches carries it over)
    uint32_t da;
    uint32_t cf;    // c0 dt
    uint32_t cd;    // a0 dt
};
schedule make_schedule(double a0, double c0, double dt, uint32_t steps);
// cA after `steps` steps (the kernel adds dA with one f32 addition per step).
uint32_t advance_ca(uint32_t ca, uint32_t da, uint32_t steps);
// J as a dense matrix of int8 (n x n, n the machine's spins), for the model.
class dense_j {
public:
    explicit dense_j(const problem & p);
    uint32_t n;
    const int8_t * row(uint32_t i) const { return a_.data() + size_t(i) * n; }
private:
    std::vector<int8_t> a_;
};

struct bitexact_out {
    std::vector<float> energy;      // steps x t words (the e of every step and replica)
    std::vector<float> snapshots;   // (steps / snap_every) x t x n floats
};
// One launch of `steps` steps of the kernel's step on t replicas: x and y (t rows of n floats) are
// updated, the energies (when `energy`) and the snapshots (snap_every != 0) are appended to *out.
// sch.ca0 is the cA at the start; the caller advances it with advance_ca for the next launch.
// kblk is the columns per weight tensor of the device layout (rows are accumulated per tensor).
void bitexact_launch(const dense_j & J, uint32_t t, uint32_t kblk, uint32_t steps, uint32_t snap_every, bool energy,
                     const schedule & sch, float * x, float * y, bitexact_out * out);

}  // namespace ism

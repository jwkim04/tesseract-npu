// The dump of tsr-ising: one binary file with the trajectory of a run, for sk_video.py. The layout is documented in
// ising_dump.cpp.
#pragma once

#include <stdint.h>

#include <string>
#include <vector>

#include "ising_problem.hpp"
#include "ising_run.hpp"

namespace ism {

struct dump_input {
    const problem * prob = nullptr;
    uint32_t t = 0;
    uint32_t steps_requested = 0;
    uint32_t snap_every = 0;
    bool energy_words = false;     // the energy monitor was on
    double a0 = 1, c0 = 0, dt = 1;
    const run_result * res = nullptr;
    const std::vector<evaluation> * final_eval = nullptr;   // the energy of every replica at the end
};

// The energy E for every replica after the steps of a snapshot: from the snapshots (nsnap x t rows of n floats), rows
// nsnap x t of doubles.
void energy_of_snapshots(const problem & p, const std::vector<float> & snaps, uint32_t t, std::vector<double> * out);

// Writes the dump; false when the file cannot be written.
bool write_dump(const char * path, const dump_input & in);

}  // namespace ism

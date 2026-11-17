// The host side of the Ising machine example over tsr_transport (any device kind: the simulator, the card):
// places the problem and the buffers in device memory by the DRAM map of EX §11, uploads J as Q8_0 rows
// (all block scales 1.0) with the symmetric allocation, fills the parameter block, runs the launches and reads the results back.
// The test and the tool tsr-ising both use it.
#pragma once

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>

#include "ising_abi.hpp"
#include "ising_problem.hpp"
#include "ising_ref.hpp"
#include "tsr_transport.h"

namespace ism {

struct options {
    uint32_t t = 1;            // replicas
    uint32_t kblk = 0;         // columns per weight tensor, a multiple of 32; 0: the whole row, as far as K_MAX allows
    uint32_t snap_every = 0;   // 0: no snapshots
    bool energy = true;        // the energy monitor
    uint32_t chunk = 0;        // steps per launch; 0: as many as one launch can do (ism::MAX_STEPS)
};

// Where everything is in device memory (beats; handles are {channel, beat}).
struct placement {
    uint32_t kblk = 0, nblk = 0;
    uint32_t w_off = 0, tensor_stride = 0;       // weights: the same beat offsets on every channel
    uint32_t param = 0;                          // the parameter block, channel 0
    uint32_t x = 0, y = 0, energy = 0;           // channel 0
    uint32_t snap_ch = 0, snap = 0;              // the snapshot area of one launch
    uint32_t chunk = 0;                          // steps per launch (a multiple of snap_every when there are snapshots)
};

struct run_result {
    std::vector<float> energy;      // steps x t words: e = sum_i s_i f_i of every step and replica (the energy of the signs is -e / 2)
    std::vector<float> snapshots;   // one snapshot every snap_every steps: t rows of n floats each
    std::vector<float> x, y;        // the state at the end (t rows of n floats)
    uint32_t steps_done = 0;        // below the requested steps when a soft stop ended the run
    bool stopped = false;           // a soft stop ended the run early (the state is the state after steps_done steps)
};

class machine {
public:
    // Places the problem and the buffers for the device behind `t` (the image with the kernel `ising`
    // must be booted; `kernel` is its index) and uploads J. false with a message in *err when the
    // problem does not fit (weights, snapshots, SPM) or the device cannot run it.
    bool open(tsr_transport * t, int kernel, const problem & prob, const options & opt, std::string * err);

    const placement & place() const { return pl_; }
    const options & opts() const { return opt_; }
    uint32_t n() const { return n_; }

    // The state x, y in device memory (t rows of n floats each).
    bool write_state(const float * x, const float * y);
    bool read_state(std::vector<float> * x, std::vector<float> * y);

    // The parameter block of a launch of `steps` steps with the constants of sch.
    params make_params(uint32_t steps, const schedule & sch) const;
    // Writes the block, launches the kernel and waits for the answer: false when the transport fails. *res is the
    // launch record (status), *ret the kernel's return value (0, or an ism::status).
    bool launch(const params & p, tsr_result * res, int32_t * ret, std::string * err);
    // Runs `steps` steps in launches of the chunk size from the state in device memory. sch.ca0 is the cA at the
    // start (the run carries it over from launch to launch). Collects the energies, the snapshots and the final state.
    // A soft stop (EX §4.9, see set_stop_poll) ends the run early and is not an error: the kernel has stored the state
    // of the steps it did, out->stopped is set and out->steps_done says how many steps the energies and snapshots
    // (those of whole snapshot periods of the last launch) and the state belong to. To go on, run again with
    // sch.ca0 = advance_ca(sch.ca0, sch.da, steps_done).
    bool run(const schedule & sch, uint32_t steps, run_result * out, std::string * err);

    // While a launch runs the machine calls f about every 20 us of the transport's time; when it returns true the machine
    // asks the device to stop the launch between two steps, and starts no further launch. nullptr: no polling (the
    // launch is a plain transport call with its timeout).
    void set_stop_poll(std::function<bool()> f) { stop_poll_ = std::move(f); }

private:
    bool upload_weights(const problem & prob, std::string * err);
    tsr_transport * t_ = nullptr;
    int kernel_ = -1;
    uint32_t n_ = 0;
    uint32_t nch_ = 0;
    uint64_t timeout_per_step_ns_ = 0;
    options opt_;
    placement pl_;
    std::function<bool()> stop_poll_;
};

}  // namespace ism

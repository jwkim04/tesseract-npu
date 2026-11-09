// tsr_kdev_sim.h — the "sim" kdev: the simulator in-process behind the core CSR port, with no
// adapter (docs/design/01-execution-model.md "EX" §10.2; docs/design/06-software-verification.md "SW"
// §9.1, §4.9, SW-18). The
// protocol layer drives it through the same registers as the card.
//
// Time (EX §10.2): nothing moves unless the host lets it. A host read of LCQ_WR, STATUS or LOG_WR
// first runs the model until a record or a log word appears, the RISC-V halts or parks, SOFT_RESET
// ends, or nothing can happen without host input (no pending event: the RISC-V asleep with no op
// running, held in reset, halted), at most run_ahead core cycles per read; idle() advances by the
// poll interval. Results are therefore deterministic and independent of host timing. now_ns() is
// device time: MTIME at CORE_HZ_NOMINAL.
//
// TSR_DEVICE=sim reads (a malformed value makes the open fail): TSR_SIM_TMAX (1..8, default 1),
// TSR_SIM_CH_GIB (1..16, default 16; lazily mapped), TSR_SIM_THREADS (MVM worker threads, default
// min(online CPUs, 8)), TSR_SIM_DRAM_DIR (file-backed DRAM), TSR_SIM_VU (0: the device of the reduced
// card image tsr_u200_k4, built without the vector unit and attention: CAPS 0 and no VU queue,
// as tsr_core reads with HAS_VU = 0 and HAS_ATT = 0; default 1); the simulator itself reads
// TSR_SIM_SCHED, TSR_SIM_RV_RATIO, TSR_SIM_RV_CORE, TSR_SIM_RACE, TSR_SIM_LOG and TSR_TRACE (SW §4.9;
// tsr_sim.h). Configuration: 4 channels, the U200 tile core, CAPS = what this simulator implements
// (EX §7.5), the U200 M6 queue depths and TCM sizes.
#pragma once

#include "tsr_kdev.h"
#include "tsr_sim.h"

#include <functional>

namespace tsr {

struct sim_options {
    tsr_config   cfg;                       // as tsr_sim_create takes it
    const char * dram_dir = nullptr;
    unsigned     threads = 1;
    bool         no_vu = false;             // a core built without the VU and AT (HAS_VU = 0): the VU queue depth is 0
    uint64_t     run_ahead = 1ull << 20;    // core cycles one run-ahead read may simulate
    double       rv_ratio = 0.0;            // F_RV / F_CORE (tsr_sim_opts.rv_ratio; 0: TSR_SIM_RV_RATIO, else the default)
    const char * vu_timing = nullptr;       // tsr_sim_opts.vu_timing (nullptr: TSR_SIM_VU_TIMING, else "u200")
};

// Options from the TSR_SIM_* environment (who names the kind in messages). false if malformed.
bool sim_options_from_env(const char * who, sim_options * o);

class sim_kdev : public kdev {
public:
    explicit sim_kdev(const sim_options & o);
    ~sim_kdev() override;
    bool ok() const { return sim_ != nullptr; }

    const char * name() const override { return "sim"; }
    uint32_t num_ch() const override { return cfg_.num_ch; }
    uint64_t ch_bytes(uint32_t) const override { return cfg_.ch_bytes; }
    uint64_t ch_addr(uint32_t) const override { return 0; }
    std::string describe() const override;
    bool has_adapter() const override { return false; }
    int reg_read(uint32_t off, uint32_t * val) override;
    int reg_write(uint32_t off, uint32_t val) override;
    int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    uint64_t now_ns() override;
    void idle(uint32_t poll_iter, uint64_t interval_ns) override;
    tsr_sim * sim() override { return sim_; }
    bool simulated() const override { return true; }

    // Core CSR port (offset < 0x1000), with the run-ahead rule above for LCQ_WR, STATUS, LOG_WR.
    uint32_t csr_read(uint32_t off);
    void     csr_write(uint32_t off, uint32_t val);
    // For the mock's adapter model: while may_advance() returns false the model does not move
    // (held memory requests, a hung unit) and idle() only adds to the device clock.
    std::function<bool()> may_advance;
    uint64_t cycles_of(uint64_t ns) const;

    // Observations (tests): run-ahead reads, the cycles they simulated, idle() calls.
    uint64_t run_ahead_reads = 0, run_ahead_cycles = 0, idles = 0;

private:
    void run_to_event();
    bool can_move() const { return !may_advance || may_advance(); }

    tsr_config cfg_;
    uint64_t   run_ahead_;
    tsr_sim *  sim_ = nullptr;
    uint64_t   frozen_cycles_ = 0;   // device time that passed while the model could not move
};

} // namespace tsr

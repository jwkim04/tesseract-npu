// tsr_kdev_sim.cpp — the "sim" kdev (tsr_kdev_sim.h): the simulator behind the core CSR port,
// advanced by the host's reads and idle() (EX §10.2, SW §9.1).

#include "tsr_kdev_sim.h"

#include "tsr_sim_int.h"   // tsr_sim_next_event: "nothing can happen without host input"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

namespace tsr {

bool sim_options_from_env(const char * who, sim_options * o) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.caps = tsr_sim_caps_implemented();   // EX §7.5: the simulator reports what it implements
    uint64_t v = 0;
    int r;
    if ((r = env_u64(who, "TSR_SIM_TMAX", 1, TSR_T_MAX_ARCH, &v)) < 0) return false;
    if (r) cfg.t_max = (uint32_t) v;
    if ((r = env_u64(who, "TSR_SIM_CH_GIB", 1, 16, &v)) < 0) return false;
    if (r) cfg.ch_bytes = v << 30;
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned threads = ncpu < 1 ? 1u : (ncpu > 8 ? 8u : (unsigned) ncpu);
    if ((r = env_u64(who, "TSR_SIM_THREADS", 1, 64, &v)) < 0) return false;
    if (r) threads = (unsigned) v;
    const char * dd = getenv("TSR_SIM_DRAM_DIR");
    o->cfg = cfg;
    o->dram_dir = (dd != nullptr && dd[0] != '\0') ? dd : nullptr;
    o->threads = threads;
    return true;
}

sim_kdev::sim_kdev(const sim_options & o) : cfg_(o.cfg), run_ahead_(o.run_ahead ? o.run_ahead : 1) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.dram_dir = o.dram_dir;
    so.threads = o.threads;
    sim_ = tsr_sim_create_opts(&cfg_, &so);
    if (sim_ == nullptr) return;
    tsr_sim_set_threads(sim_, o.threads);
    cfg_ = *tsr_sim_config(sim_);
}

sim_kdev::~sim_kdev() { tsr_sim_destroy(sim_); }

std::string sim_kdev::describe() const {
    char b[256];
    snprintf(b, sizeof(b), "simulator in-process: %u channels of %.2f GiB, T_MAX %u, CAPS 0x%08x, F_RV / F_CORE %.4f, schedule %s\n",
             cfg_.num_ch, (double) cfg_.ch_bytes / (1u << 30), cfg_.t_max, cfg_.caps, tsr_sim_rv_ratio(sim_),
             tsr_sim_opt_sched(sim_));
    return b;
}

uint64_t sim_kdev::cycles_of(uint64_t ns) const { return (uint64_t) ((double) ns * CORE_HZ_NOMINAL / 1e9 + 0.5); }

uint64_t sim_kdev::now_ns() {
    return (uint64_t) ((double) (tsr_sim_now(sim_) + frozen_cycles_) * 1e9 / CORE_HZ_NOMINAL);
}

// EX §10.2: run until a record or log word appears, the RISC-V halts or parks, SOFT_RESET ends,
// nothing can happen without the host, or the run-ahead budget is spent.
void sim_kdev::run_to_event() {
    run_ahead_reads++;
    uint64_t spent = 0;
    while (spent < run_ahead_ && can_move()) {
        if (tsr_sim_next_event(sim_) == UINT64_MAX) return;   // waits for host input
        const uint64_t t0 = tsr_sim_now(sim_);
        const uint32_t ev = tsr_sim_advance(sim_, run_ahead_ - spent);
        const uint64_t dt = tsr_sim_now(sim_) - t0;
        spent += dt;
        run_ahead_cycles += dt;
        if (ev & (TSR_SIM_EV_LCQ | TSR_SIM_EV_LOG | TSR_SIM_EV_HALT | TSR_SIM_EV_RESET)) return;
        // else QIDLE edges or the budget: go on
    }
}

uint32_t sim_kdev::csr_read(uint32_t off) {
    if (off == TSR_CSR_LCQ_WR || off == TSR_CSR_STATUS || off == TSR_CSR_LOG_WR) run_to_event();
    return tsr_sim_csr_read(sim_, off);
}

void sim_kdev::csr_write(uint32_t off, uint32_t val) { tsr_sim_csr_write(sim_, off, val); }

int sim_kdev::reg_read(uint32_t off, uint32_t * val) {
    if (off < KREG_CSR_WINDOW || off >= KREG_SPACE || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(sim): register read at 0x%x: the sim kind has the core CSR port only (no adapter)\n", off);
        return -1;
    }
    *val = csr_read(off - KREG_CSR_WINDOW);
    return 0;
}

int sim_kdev::reg_write(uint32_t off, uint32_t val) {
    if (off < KREG_CSR_WINDOW || off >= KREG_SPACE || (off & 3u) != 0) {
        fprintf(stderr, "tsr_transport(sim): register write at 0x%x: the sim kind has the core CSR port only (no adapter)\n", off);
        return -1;
    }
    csr_write(off - KREG_CSR_WINDOW, val);
    return 0;
}

// The visible forms (tsr_sim.h): the race detector and the launch capture see host DRAM accesses.
int sim_kdev::mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) {
    return tsr_sim_dram_write(sim_, ch, off, src, n) == 0 ? 0 : -1;
}

int sim_kdev::mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) {
    return tsr_sim_dram_read(sim_, ch, off, dst, n) == 0 ? 0 : -1;
}

void sim_kdev::idle(uint32_t, uint64_t interval_ns) {
    idles++;
    const uint64_t c = cycles_of(interval_ns) ? cycles_of(interval_ns) : 1;
    if (!can_move()) {
        frozen_cycles_ += c;
        return;
    }
    tsr_sim_advance(sim_, c);   // returns early at a host-visible event; time passes when idle
}

std::unique_ptr<kdev> open_sim_kdev() {
    sim_options o;
    if (!sim_options_from_env("sim", &o)) return nullptr;
    std::unique_ptr<sim_kdev> k(new sim_kdev(o));
    if (!k->ok()) {
        fprintf(stderr, "tsr_transport(sim): simulator creation failed%s%s%s\n", o.dram_dir ? " (TSR_SIM_DRAM_DIR=" : "",
                o.dram_dir ? o.dram_dir : "", o.dram_dir ? ")" : "");
        return nullptr;
    }
    return k;
}

} // namespace tsr

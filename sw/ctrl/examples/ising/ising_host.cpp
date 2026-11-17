// tsr-ising: the host tool of the Ising machine example (kernel ising.cpp). It makes a Sherrington-Kirkpatrick spin
// glass, uploads it, runs discrete simulated bifurcation on the device through tsr_transport (the simulator or the card:
// --device or $TSR_DEVICE), and prints the energy E and e = E / N^1.5 of every replica and of the best.
//
//   tsr-ising [--device KIND] [--image ELF] [--n N] [--seed S] [--t REPLICAS] [--steps N] [--dt X] [--a0 X] [--c0 X]
//             [--chunk N] [--kblk N] [--snap-every N] [--no-energy] [--dump FILE] [--check]
//
//   --n           spins of the problem (default 1024, at least 2); the machine pads them to a multiple of 32. A size above
//                 the device's K_MAX columns is stored in several column tensors.
//   --seed        the seed of the couplings and of the start values
//   --t           replicas that run side by side with different start values (1 to the device's T_MAX, at most 8)
//   --steps, --dt, --a0, --c0   the schedule: a(t) rises from 0 to a0 in `steps` steps of time step dt; c0 is the
//                 scale of the coupling term (default 0.5 sqrt((N - 1) / sum J^2))
//   --chunk       steps per launch (default: as many as one launch can do)
//   --kblk        columns per weight tensor (default: the whole row, as far as the device's K_MAX allows)
//   --snap-every  store x of every replica after every N-th step (the video needs them); with --dump the
//                 default is about 200 snapshots
//   --no-energy   leave out the energy words of the kernel
//   --check       run the bit-exact model of the kernel next to the device and compare the energies, the
//                 snapshots and the final state bit for bit (the model needs the dense matrix and is slow for large n)
//   --dump FILE   write the trajectory to FILE (the layout is in ising_dump.cpp; sk_video.py reads it)
//
// The device kind is taken from --device, else $TSR_DEVICE, else "sim". The simulator runs one replica
// unless TSR_SIM_TMAX says more: this tool sets TSR_SIM_TMAX=8, the T_MAX of the card, when it is not set.
// The image is --image, else $TSR_IMAGE, else the example image of this build.
// Ctrl-C asks the device for a soft stop: the run ends between two steps, the results and the dump are those of the
// steps done, and the exit status is 0.
//
// Exit status: 0 ok, 1 a run or a device error, 2 a bad command line, 3 --check found a difference.
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "ising_abi.hpp"
#include "ising_dump.hpp"
#include "ising_problem.hpp"
#include "ising_ref.hpp"
#include "ising_run.hpp"
#include "tsr_fw_paths.h"
#include "tsr_ref.h"
#include "tsr_transport.h"

using namespace ism;

namespace {

constexpr double DEFAULT_DT = 1.0;
constexpr uint32_t DEFAULT_STEPS = 1000u;

volatile sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }

struct cli {
    const char * device = nullptr;
    const char * image = nullptr;
    const char * dump = nullptr;
    uint32_t n = 1024, t = 4, steps = DEFAULT_STEPS, chunk = 0, kblk = 0, snap_every = 0;
    uint64_t seed = 1;
    double dt = DEFAULT_DT, a0 = 1.0, c0 = -1.0;   // c0 < 0: the default
    bool energy = true, check = false, snap_given = false;
};

[[noreturn]] void usage_exit(const char * msg) {
    if (msg) fprintf(stderr, "tsr-ising: %s\n", msg);
    fprintf(stderr,
            "usage: tsr-ising [--device KIND] [--image ELF] [--n N] [--seed S] [--t REPLICAS] [--steps N] [--dt X] [--a0 X]\n"
            "                 [--c0 X] [--chunk N] [--kblk N] [--snap-every N] [--no-energy] [--dump FILE] [--check]\n"
            "see the comment at the top of ising_host.cpp\n");
    exit(2);
}

uint64_t parse_uint(const char * opt, const char * s, uint64_t max) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || s[0] == '-' || *end != '\0' || errno != 0 || v > max)
        usage_exit((std::string(opt) + " needs a whole number up to " + std::to_string(max) + ", not \"" + s + "\"").c_str());
    return v;
}

double parse_real(const char * opt, const char * s) {
    char * end = nullptr;
    const double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !(v == v) || v > 1e300 || v < -1e300)
        usage_exit((std::string(opt) + " needs a number, not \"" + s + "\"").c_str());
    return v;
}

cli parse(int argc, char ** argv) {
    cli c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * {
            if (i + 1 >= argc) usage_exit((a + " needs a value").c_str());
            return argv[++i];
        };
        if (a == "--device") c.device = val();
        else if (a == "--image") c.image = val();
        else if (a == "--dump") c.dump = val();
        else if (a == "--n") c.n = uint32_t(parse_uint("--n", val(), MAX_N));
        else if (a == "--t") c.t = uint32_t(parse_uint("--t", val(), TSR_T_MAX_ARCH));
        else if (a == "--steps") c.steps = uint32_t(parse_uint("--steps", val(), 1u << 24));
        else if (a == "--chunk") c.chunk = uint32_t(parse_uint("--chunk", val(), MAX_STEPS));
        else if (a == "--kblk") c.kblk = uint32_t(parse_uint("--kblk", val(), 1u << 20));
        else if (a == "--snap-every") { c.snap_every = uint32_t(parse_uint("--snap-every", val(), MAX_STEPS)); c.snap_given = true; }
        else if (a == "--seed") c.seed = parse_uint("--seed", val(), UINT64_MAX);
        else if (a == "--dt") c.dt = parse_real("--dt", val());
        else if (a == "--a0") c.a0 = parse_real("--a0", val());
        else if (a == "--c0") c.c0 = parse_real("--c0", val());
        else if (a == "--no-energy") c.energy = false;
        else if (a == "--check") c.check = true;
        else if (a == "--help" || a == "-h") usage_exit(nullptr);
        else usage_exit(("unknown option " + a).c_str());
    }
    if (c.t == 0) usage_exit("--t must be at least 1");
    if (c.steps == 0) usage_exit("--steps must be at least 1");
    if (!(c.dt > 0.0) || !(c.a0 > 0.0)) usage_exit("--dt and --a0 must be positive");
    if (c.n < 2) usage_exit("--n must be at least 2");
    if (c.dump != nullptr && !c.snap_given) c.snap_every = std::max(1u, c.steps / 200u);
    return c;
}

std::string number(double v) {
    char buf[40];
    if (v == double(int64_t(v)) && v > -1e15 && v < 1e15) snprintf(buf, sizeof(buf), "%" PRId64, int64_t(v));
    else snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

// The per-replica lines and the best one.
std::vector<evaluation> report(const problem & p, const std::vector<float> & x, uint32_t t) {
    std::vector<int8_t> spins;
    spins_from_x(p, x.data(), t, &spins);
    std::vector<evaluation> ev;
    evaluate(p, spins.data(), t, &ev);
    uint32_t best = 0;
    for (uint32_t r = 0; r < t; ++r) {
        if (ev[r].e < ev[best].e) best = r;
        printf("replica %u  energy %s  e %.4f\n", r, number(ev[r].energy).c_str(), ev[r].e);
    }
    printf("best: replica %u  energy %s  e %.4f\n", best, number(ev[best].energy).c_str(), ev[best].e);
    return ev;
}

// The first word at which two float arrays differ (bitwise); -1 when equal.
long first_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (memcmp(&a[i], &b[i], 4) != 0) return long(i);
    return -1;
}

bool same(const char * what, const std::vector<float> & dev, const std::vector<float> & ref) {
    if (dev.size() != ref.size()) {
        printf("check: %s: %zu words on the device, %zu in the model\n", what, dev.size(), ref.size());
        return false;
    }
    const long d = first_diff(dev, ref);
    if (d >= 0) {
        printf("check: %s differs at word %ld: device %.9g (0x%08x), model %.9g (0x%08x)\n", what, d, dev[d],
               tsr_f32_bits(dev[d]), ref[d], tsr_f32_bits(ref[d]));
        return false;
    }
    printf("check: %s equal (%zu words)\n", what, dev.size());
    return true;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int fail(const std::string & msg) {
    fprintf(stderr, "tsr-ising: %s\n", msg.c_str());
    return 1;
}

}  // namespace

int main(int argc, char ** argv) {
    signal(SIGINT, on_sigint);   // from here on Ctrl-C asks for a soft stop (nothing polls it before the run starts)
    cli c = parse(argc, argv);

    // The problem.
    std::string err;
    const auto t_start = std::chrono::steady_clock::now();
    const problem prob(c.n, c.seed);
    const double c0 = c.c0 >= 0.0 ? c.c0 : prob.default_c0();
    printf("problem: Sherrington-Kirkpatrick, n %u (padded %u), seed %" PRIu64 "\n", prob.spins, prob.n, prob.seed);
    printf("schedule: a0 %g, c0 %g, dt %g, %u steps, %u replicas\n", c.a0, c0, c.dt, c.steps, c.t);

    std::vector<float> x0, y0;
    prob.initial_state(c.t, c.seed ^ 0x5151u, &x0, &y0);

    // The device. The simulator runs one replica unless TSR_SIM_TMAX says more.
    const char * kind = c.device != nullptr ? c.device : getenv("TSR_DEVICE");
    if (kind == nullptr || kind[0] == '\0') kind = "sim";
    if (strcmp(kind, "sim") == 0) setenv("TSR_SIM_TMAX", "8", 0);
    tsr_image * img = nullptr;
    const char * image_path = c.image != nullptr ? c.image : getenv("TSR_IMAGE");
    if (image_path == nullptr || image_path[0] == '\0') image_path = TSR_FW_TEST_IMAGE_fw_example_ising;
    if (tsr_image_open(image_path, &img) != 0 || img == nullptr) return fail(std::string("cannot open the firmware image ") + image_path);
    tsr_transport * tr = tsr_transport_open(kind);
    if (!tr) { tsr_image_close(img); return fail(std::string("cannot open the \"") + kind + "\" transport"); }
    int rc = 1;
    do {
        if (tsr_transport_boot(tr, img) != 0) { fail("cannot boot the image " + std::string(image_path)); break; }
        const int kernel = tsr_image_kernel(img, "ising");
        if (kernel < 0) { fail(std::string("the image ") + image_path + " has no kernel \"ising\""); break; }
        const tsr_config * cfg = tsr_transport_config(tr);
        printf("device %s: %u channels, T_MAX %u\n", tsr_transport_name(tr), cfg->num_ch, cfg->t_max);

        options opt;
        opt.t = c.t; opt.kblk = c.kblk; opt.snap_every = c.snap_every; opt.energy = c.energy; opt.chunk = c.chunk;
        machine m;
        if (!m.open(tr, kernel, prob, opt, &err)) { fail(err); break; }
        const placement & pl = m.place();
        printf("weights: %u column tensor(s) of %u columns; %u steps per launch\n", pl.nblk, pl.kblk, pl.chunk);
        if (!m.write_state(x0.data(), y0.data())) { fail("writing the start values failed"); break; }
        const double upload_s = seconds_since(t_start);

        m.set_stop_poll([]() { return g_stop != 0; });
        const schedule sch = make_schedule(c.a0, c0, c.dt, c.steps);
        run_result res;
        tsr_perf_summary p0{}, p1{};
        const bool have_perf = tsr_transport_perf_summary(tr, &p0) == 0;
        const auto t_run = std::chrono::steady_clock::now();
        if (!m.run(sch, c.steps, &res, &err)) { fail(err); break; }
        const double run_s = seconds_since(t_run);
        if (res.stopped) printf("stopped: %u of %u steps done\n", res.steps_done, c.steps);
        if (have_perf && res.steps_done != 0 && tsr_transport_perf_summary(tr, &p1) == 0) {
            const uint64_t cyc = p1.perf[TSR_PERF_CYCLES] - p0.perf[TSR_PERF_CYCLES];
            printf("device: %" PRIu64 " core cycles, %.0f per step", cyc, double(cyc) / res.steps_done);
            if (p1.core_hz > 0) printf(", %.1f us per step at %.0f MHz (%s)", double(cyc) / res.steps_done / p1.core_hz * 1e6, p1.core_hz / 1e6,
                                       tsr_core_hz_source_str(p1.core_hz_fitted));
            printf("\n");
        }
        printf("time: upload %.1f s, run %.1f s\n", upload_s, run_s);

        const std::vector<evaluation> ev = report(prob, res.x, c.t);

        bool ok = true;
        if (c.check) {
            const auto t_chk = std::chrono::steady_clock::now();
            const dense_j J(prob);
            std::vector<float> x = x0, y = y0;
            bitexact_out ref;
            schedule s = sch;
            for (uint32_t done = 0; done < res.steps_done;) {   // launch by launch like the device
                const uint32_t cnt = std::min(pl.chunk, res.steps_done - done);
                bitexact_launch(J, c.t, pl.kblk, cnt, c.snap_every, c.energy, s, x.data(), y.data(), &ref);
                s.ca0 = advance_ca(s.ca0, s.da, cnt);
                done += cnt;
            }
            ok &= same("energy words", res.energy, ref.energy);
            ok &= same("snapshots", res.snapshots, ref.snapshots);
            ok &= same("final x", res.x, x);
            ok &= same("final y", res.y, y);
            printf("check: %s (%.1f s)\n", ok ? "the device agrees with the bit-exact model" : "DIFFERENT", seconds_since(t_chk));
        }
        if (c.dump != nullptr) {
            dump_input in;
            in.prob = &prob; in.t = c.t; in.steps_requested = c.steps; in.snap_every = c.snap_every; in.energy_words = c.energy;
            in.a0 = c.a0; in.c0 = c0; in.dt = c.dt; in.res = &res; in.final_eval = &ev;
            const auto t_dump = std::chrono::steady_clock::now();
            if (!write_dump(c.dump, in)) { fail(std::string("cannot write ") + c.dump); break; }
            printf("dump: %s (%.1f s)\n", c.dump, seconds_since(t_dump));
        }
        rc = ok ? 0 : 3;
    } while (false);
    tsr_transport_close(tr);
    tsr_image_close(img);
    return rc;
}

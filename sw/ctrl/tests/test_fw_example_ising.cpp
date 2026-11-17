// fw_example_ising: the Ising machine example (sw/ctrl/examples/ising/) on the simulator, through the transport.
//
//   1. the numeric constants of the kernel: the scale 0x3C010204 makes the block term exact, the wall factor m is 0 or 1
//   2. bit-exact agreement with the host model (ising_ref.cpp): every snapshot, every energy word, the final state,
//      under the fixed schedule and under random schedules with the race detector fatal; several column tensors, t = 8,
//      a run cut into launches; every energy word is the integer sum_i s_i (J s)_i of the signs before it
//   3. the status codes of bad parameter blocks, and a good launch after them; a soft stop between steps with the
//      state stored, and a run that goes on from it
//   4. what the device and the model share, checked against the design on its own: the schedule constants, the
//      couplings, the energy of a sign vector against a direct sum, and a directed start state that covers the sign step
//   5. quality: the best energy per spin of 8 replicas at n = 512 is below -0.73
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "ising_abi.hpp"
#include "ising_problem.hpp"
#include "ising_ref.hpp"
#include "ising_run.hpp"
#include "tsr_fw_paths.h"
#include "tsr_ref.h"
#include "tsr_ref_vu.h"
#include "tsr_transport.h"

using namespace ism;

static int g_fail = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
            ++g_fail;                                     \
        }                                                 \
    } while (0)
// A booted simulator with the image that holds the kernel.
struct device {
    tsr_image * img = nullptr;
    tsr_transport * t = nullptr;
    int kernel = -1;
    bool open(const char * sched, const char * tmax) {
        setenv("TSR_SIM_SCHED", sched, 1);
        setenv("TSR_SIM_RACE", "fatal", 1);
        setenv("TSR_SIM_TMAX", tmax, 1);
        if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_example_ising, &img) != 0) return false;
        t = tsr_transport_open("sim");
        if (!t || tsr_transport_boot(t, img) != 0) return false;
        kernel = tsr_image_kernel(img, "ising");
        return kernel >= 0;
    }
    ~device() {
        if (t) tsr_transport_close(t);
        if (img) tsr_image_close(img);
    }
};

static size_t first_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (memcmp(&a[i], &b[i], 4) != 0) return i;
    return size_t(-1);
}

static bool same(const char * what, const std::vector<float> & dev, const std::vector<float> & ref) {
    if (dev.size() != ref.size()) {
        printf("  %s: %zu words on the device, %zu in the model\n", what, dev.size(), ref.size());
        return false;
    }
    const size_t d = first_diff(dev, ref);
    if (d != size_t(-1)) {
        printf("  %s differs from the model at word %zu: device %.9g (0x%08x), model %.9g (0x%08x)\n", what, d, dev[d],
               tsr_f32_bits(dev[d]), ref[d], tsr_f32_bits(ref[d]));
        return false;
    }
    return true;
}
struct case_cfg {
    const char * name;
    uint32_t n, t, kblk, steps, snap_every, chunk;
    uint64_t seed;
};

// The constants of the runs: a0 = 1, dt = 0.5 and the default c0.
static schedule schedule_for(const problem & p, uint32_t steps) { return make_schedule(1.0, p.default_c0(), 0.5, steps); }

// Runs the case on the device and in the model and compares them. With cut launches (chunk < steps) the model is
// run in the same launches and also in one piece. x0: written start values of x (t rows of n floats; y keeps the seeded
// values); the energy words are then not the integer sums of the signs (|x| < 2^-100 gives a fractional s).
static bool run_case(const case_cfg & c, const char * sched, const std::vector<float> * x0_given = nullptr) {
    std::string err;
    const problem prob(c.n, c.seed);
    const dense_j J(prob);
    const schedule sch = schedule_for(prob, c.steps);
    std::vector<float> x0, y0;
    prob.initial_state(c.t, c.seed ^ 0x5151, &x0, &y0);
    if (x0_given) x0 = *x0_given;

    device dev;
    if (!dev.open(sched, "8")) { printf("%s: cannot boot the image\n", c.name); return false; }
    options opt;
    opt.t = c.t; opt.kblk = c.kblk; opt.snap_every = c.snap_every; opt.energy = true; opt.chunk = c.chunk;
    machine m;
    if (!m.open(dev.t, dev.kernel, prob, opt, &err)) { printf("%s: %s\n", c.name, err.c_str()); return false; }
    if (!m.write_state(x0.data(), y0.data())) { printf("%s: writing the state failed\n", c.name); return false; }
    run_result res;
    if (!m.run(sch, c.steps, &res, &err)) { printf("%s: %s\n", c.name, err.c_str()); return false; }
    const uint32_t kblk = m.place().kblk, chunk = m.place().chunk;

    // The model, launch by launch like the device.
    std::vector<float> x = x0, y = y0;
    bitexact_out ref;
    schedule s = sch;
    for (uint32_t done = 0; done < c.steps;) {
        const uint32_t cnt = std::min(chunk, c.steps - done);
        bitexact_launch(J, c.t, kblk, cnt, c.snap_every, true, s, x.data(), y.data(), &ref);
        s.ca0 = advance_ca(s.ca0, s.da, cnt);
        done += cnt;
    }
    bool ok = same("energy", res.energy, ref.energy);
    ok &= same("snapshots", res.snapshots, ref.snapshots);
    ok &= same("x", res.x, x);
    ok &= same("y", res.y, y);
    if (chunk < c.steps) {   // a cut run is the run in one piece
        std::vector<float> xw = x0, yw = y0;
        bitexact_out whole;
        bitexact_launch(J, c.t, kblk, c.steps, c.snap_every, true, sch, xw.data(), yw.data(), &whole);
        ok &= same("energy (cut run against one piece)", res.energy, whole.energy);
        ok &= same("x (cut run against one piece)", res.x, xw);
    }

    // The energy words are exact: e of step k is sum_i s_i (J s)_i of the signs that enter it, which are the
    // signs of the snapshot of step k - 1 (and of the start values for k = 0).
    if (c.snap_every == 1 && !x0_given) {
        for (uint32_t k = 0; k < c.steps && ok; ++k) {
            std::vector<int8_t> s8(size_t(c.t) * prob.n);
            const float * src = k == 0 ? x0.data() : res.snapshots.data() + size_t(k - 1) * c.t * prob.n;
            for (size_t i = 0; i < s8.size(); ++i) s8[i] = src[i] > 0.f ? 1 : (src[i] < 0.f ? -1 : 0);
            std::vector<int64_t> sums;
            field_energy_sums(prob, s8.data(), c.t, &sums);
            for (uint32_t r = 0; r < c.t; ++r) {
                const float want = float(sums[r]);
                if (memcmp(&want, &res.energy[size_t(k) * c.t + r], 4) != 0) {
                    printf("  energy word of step %u replica %u: %.9g, the integer sum is %lld\n", k, r,
                           res.energy[size_t(k) * c.t + r], (long long) sums[r]);
                    ok = false;
                    break;
                }
            }
        }
    }
    printf("%-38s [%s]: %s (n %u t %u kblk %u steps %u, %zu snapshot words, launches of %u)\n", c.name, sched,
           ok ? "bit-exact" : "DIFFERS", prob.n, c.t, kblk, c.steps, res.snapshots.size(), chunk);
    return ok;
}

// 1. The constants of the kernel.
static void test_constants() {
    const float D = tsr_bits_f32(0x3C010204u);
    CHECK(tsr_f32_bits(tsr_fdiv(1.0f, 127.0f)) == 0x3C010204u, "0x3C010204 is the f32 nearest to 1/127");
    // The term of a Q8_0 block of J (scale 1.0) against s = +-1 (q = +-127, scale D) is (1.0 * D) * f32(127 m) = m.
    uint32_t bad = 0;
    for (int32_t m = -32768; m < 32768; ++m)
        if (tsr_fmul(tsr_fmul(1.0f, D), tsr_i32_to_f32(127 * m)) != float(m)) ++bad;
    CHECK(bad == 0, "%u block terms are not exact", bad);
    // The wall factor m = min(2^24 - 2^24 x x, 1) is 1 for |x| < 1 and 0 for |x| = 1: all f32 x just below 1 and a sweep
    // of the whole range (the rest follows from monotonicity: every rounding is monotone in |x|).
    auto wall = [](float x) {
        const float m = tsr_bits_f32(tsr_ref_vu_mint(tsr_f32_bits(tsr_fadd(tsr_fmul(tsr_fmul(x, -16777216.0f), x), 16777216.0f)), tsr_f32_bits(1.0f)));
        return m;
    };
    uint32_t wbad = 0;
    for (uint32_t b = 0x3F800000u - (1u << 22); b < 0x3F800000u; ++b)
        if (wall(tsr_bits_f32(b)) != 1.0f || wall(tsr_bits_f32(b | 0x80000000u)) != 1.0f) ++wbad;
    for (uint32_t b = 0; b < 0x3F800000u; b += 997u)
        if (wall(tsr_bits_f32(b)) != 1.0f) ++wbad;
    CHECK(wall(1.0f) == 0.0f && wall(-1.0f) == 0.0f, "the wall factor at |x| = 1 is not 0");
    CHECK(wbad == 0, "%u wall factors are not 1 inside the walls", wbad);
    printf("constants: scale bits, block terms and wall factor ok\n");
}
// 8a. The schedule constants against the design (the device and the model take them from make_schedule and advance_ca, so
// a mistake there would agree on both sides): cA starts at -a0 dt, rises by dA = a0 dt / steps per step to about 0.
static void test_schedule() {
    // Powers of two: every constant and the sum are exact.
    schedule s = make_schedule(2.0, 0.5, 0.25, 64);
    CHECK(s.ca0 == 0xBF000000u, "cA0 = -a0 dt: 0x%08x, wanted -0.5", s.ca0);
    CHECK(s.cd == 0x3F000000u, "cD = a0 dt: 0x%08x, wanted 0.5", s.cd);
    CHECK(s.cf == 0x3E000000u, "cF = c0 dt: 0x%08x, wanted 0.125", s.cf);
    CHECK(s.da == 0x3C000000u, "dA = a0 dt / steps: 0x%08x, wanted 2^-7", s.da);
    CHECK(tsr_f32_bits(tsr_bits_f32(advance_ca(s.ca0, s.da, 64))) == 0u, "cA after all the steps is not 0");
    CHECK(tsr_bits_f32(advance_ca(s.ca0, s.da, 32)) == -0.25f, "cA at half time is not -a0 dt / 2");
    // The tool's default (a0 = 1, dt = 1, 1000 steps): cA rises in 1000 f32 additions and ends within rounding of 0.
    s = make_schedule(1.0, 0.25, 1.0, 1000);
    CHECK(tsr_bits_f32(s.ca0) == -1.0f && tsr_bits_f32(s.cd) == 1.0f && tsr_bits_f32(s.cf) == 0.25f, "constants of dt 1");
    CHECK(tsr_bits_f32(s.da) == 0.001f, "dA of 1000 steps");
    CHECK(fabsf(tsr_bits_f32(advance_ca(s.ca0, s.da, 1000))) < 1e-4f, "cA after 1000 steps is %g, not about 0", tsr_bits_f32(advance_ca(s.ca0, s.da, 1000)));
    CHECK(fabsf(tsr_bits_f32(advance_ca(s.ca0, s.da, 500)) + 0.5f) < 1e-4f, "cA at half time is not about -0.5");
    printf("schedule: constants and cA as designed\n");
}
// 4b. The couplings against the design: J symmetric with a zero diagonal, +-1 off the diagonal, the padding spins
// uncoupled, the rows equal to the pair signs; and the energy of a sign vector against a direct sum over the pairs.
static void test_couplings() {
    for (uint32_t spins : {70u, 64u}) {
        const problem p(spins, 6);
        const uint32_t n = p.n;
        std::vector<int8_t> J(size_t(n) * n);
        for (uint32_t i = 0; i < n; ++i) p.row(i, J.data() + size_t(i) * n);
        uint32_t asym = 0, diag = 0, pad = 0, off = 0, wrong = 0;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j) {
                const int8_t a = J[size_t(i) * n + j];
                asym += a != J[size_t(j) * n + i];
                diag += i == j && a != 0;
                pad += (i >= spins || j >= spins) && a != 0;
                off += i != j && i < spins && j < spins && a != 1 && a != -1;
                wrong += i != j && i < spins && j < spins && a != pair_sign(6, i, j);
            }
        CHECK(asym == 0 && diag == 0 && pad == 0 && off == 0 && wrong == 0, "n %u: %u asymmetric, %u diagonal, %u padding, %u not +-1, %u differ from pair_sign",
              spins, asym, diag, pad, off, wrong);
    }
    const problem p(70, 33);
    std::vector<int8_t> s(70);
    uint64_t st = 5;
    for (auto & v : s) v = (splitmix64(st) >> 40) & 1 ? 1 : -1;
    int64_t direct = 0;
    for (uint32_t i = 0; i < 70; ++i)
        for (uint32_t j = i + 1; j < 70; ++j) direct -= int64_t(pair_sign(33, i, j)) * s[i] * s[j];
    std::vector<evaluation> ev;
    evaluate(p, s.data(), 1, &ev);
    CHECK(ev[0].energy == double(direct), "energy %g, the direct sum %lld", ev[0].energy, (long long) direct);
    CHECK(fabs(ev[0].e - double(direct) / pow(70.0, 1.5)) < 1e-12, "e %g", ev[0].e);
    CHECK(fabs(p.default_c0() - 0.5 / sqrt(70.0)) < 1e-12, "c0 %g", p.default_c0());
    printf("couplings: symmetric, zero diagonal, padding uncoupled, energy against the direct sum\n");
}

// 8c. A directed start state for the sign step s = clip(x 2^100) and the wall: values around the edges of the rule
// (|x| = 2^-90 and above is a full sign, 2^-100 is the first, 2^-101 and below a fraction that quantizes to a half or to 0,
// ties to the subnormal range, which the vector unit flushes to 0), +-0 (s = 0), +-1 (on the wall) and ordinary values. The
// model keeps its own constant, so a wrong constant of the kernel shows as a difference.
static void test_sign_step() {
    std::vector<float> special;
    for (int e : {-60, -90, -99, -100, -101, -102, -105, -110, -126}) {
        special.push_back(ldexpf(1.0f, e));
        special.push_back(-ldexpf(1.0f, e));
    }
    for (uint32_t b : {0x00000001u, 0x007FFFFFu, 0x00400000u}) {   // subnormals
        special.push_back(tsr_bits_f32(b));
        special.push_back(tsr_bits_f32(b | 0x80000000u));
    }
    for (float v : {0.0f, -0.0f, 1.0f, -1.0f, 0.999999940f, -0.999999940f, 0.05f, -0.07f}) special.push_back(v);
    const uint32_t n = 64, t = 2;
    const problem prob(n, 31);
    std::vector<float> x0, y0;
    prob.initial_state(t, 31 ^ 0x5151, &x0, &y0);   // the ordinary values of the other spins
    for (uint32_t r = 0; r < t; ++r)
        for (uint32_t i = 0; i < special.size() && i < n; ++i) x0[size_t(r) * n + (i + 7 * r) % n] = special[i];
    const case_cfg c = {"directed start values (sign step, wall)", n, t, 64, 6, 1, 0, 31};
    CHECK(run_case(c, "fixed", &x0), "the directed start values differ from the model");
}
// 6. Bad parameter blocks.
static void test_status() {
    std::string err;
    const problem prob(64, 7);
    device dev;
    if (!dev.open("fixed", "8")) { CHECK(false, "cannot boot the image"); return; }
    options opt;
    opt.t = 2; opt.snap_every = 0; opt.energy = true;
    machine m;
    if (!m.open(dev.t, dev.kernel, prob, opt, &err)) { CHECK(false, "%s", err.c_str()); return; }
    std::vector<float> x, y;
    prob.initial_state(2, 3, &x, &y);
    m.write_state(x.data(), y.data());
    const schedule sch = schedule_for(prob, 4);
    const params good = m.make_params(4, sch);
    struct bad_case { const char * what; params p; int32_t want; };
    std::vector<bad_case> cases;
    auto add = [&](const char * what, int32_t want, auto edit) { params p = good; edit(p); cases.push_back({what, p, want}); };
    add("bad magic", ST_BAD_MAGIC, [](params & p) { p.magic ^= 1; });
    add("bad version", ST_BAD_MAGIC, [](params & p) { p.version = 2; });
    add("n = 0", ST_BAD_N, [](params & p) { p.n = 0; });
    add("n = 48", ST_BAD_N, [](params & p) { p.n = 48; });
    add("n above the limit", ST_BAD_N, [](params & p) { p.n = MAX_N + 32; });
    add("t = 0", ST_BAD_T, [](params & p) { p.t = 0; });
    add("t = 9", ST_BAD_T, [](params & p) { p.t = 9; });
    add("steps above the limit", ST_BAD_STEPS, [](params & p) { p.steps = MAX_STEPS + 1; });
    add("kblk = 0", ST_BAD_KBLK, [](params & p) { p.kblk = 0; });
    add("kblk = 48", ST_BAD_KBLK, [](params & p) { p.kblk = 48; });
    add("kblk above K_MAX", ST_BAD_KBLK, [](params & p) { p.n = 65536; p.kblk = 32768 + 32; });
    add("more than 32 column tensors", ST_BAD_KBLK, [](params & p) { p.n = 1056; p.kblk = 32; });
    add("unknown flag", ST_BAD_FLAGS, [](params & p) { p.flags = 2; });
    add("unknown flag with the known one", ST_BAD_FLAGS, [](params & p) { p.flags = FLAG_ENERGY | 4; });
    add("buffers larger than the SPM", ST_BAD_SPM, [](params & p) { p.n = 1u << 18; p.t = 8; p.kblk = 32768; });
    for (const auto & c : cases) {
        tsr_result res;
        int32_t ret = -1;
        const bool ok = m.launch(c.p, &res, &ret, &err);
        CHECK(ok && res.status == TSR_LCQ_R0_STATUS_OK && ret == c.want, "%s: launch %s, status %u, return value %d, wanted %d", c.what,
              ok ? "answered" : err.c_str(), res.status, ret, c.want);
    }
    // t above this device's T_MAX: a device with T_MAX 2 answers BAD_T to t = 3 (the block is the host's, not m's checks).
    {
        device small;
        if (!small.open("fixed", "2")) { CHECK(false, "cannot boot the image"); return; }
        options o2;
        o2.t = 2;
        machine m2;
        CHECK(m2.open(small.t, small.kernel, prob, o2, &err), "%s", err.c_str());
        params p = m2.make_params(1, sch);
        p.t = 3;
        tsr_result res;
        int32_t ret = -1;
        CHECK(m2.launch(p, &res, &ret, &err) && res.status == TSR_LCQ_R0_STATUS_OK && ret == ST_BAD_T, "t above T_MAX: return value %d", ret);
        options o3;
        o3.t = 3;
        machine m3;
        CHECK(!m3.open(small.t, small.kernel, prob, o3, &err), "the host accepted t = 3 on a device with T_MAX 2");
    }
    // After the refusals the device runs a good launch, and the state was not touched by them.
    run_result res;
    CHECK(m.run(sch, 4, &res, &err), "%s", err.c_str());
    const dense_j J(prob);
    std::vector<float> xr = x, yr = y;
    bitexact_out ref;
    bitexact_launch(J, 2, m.place().kblk, 4, 0, true, sch, xr.data(), yr.data(), &ref);
    CHECK(same("x after the refusals", res.x, xr) && same("energy after the refusals", res.energy, ref.energy), "a good launch after bad ones differs from the model");
    printf("status codes: %zu bad blocks answered as expected\n", cases.size());
}

// 6b. A soft stop ends the launch between steps, with the state of the steps done stored: machine::run() returns what
// was done (not an error), and the run can go on from it. The stop request comes from the poll function of the machine
// once the device clock has passed a time; a stop requested before the first launch starts none.
static void test_stop() {
    const uint64_t STOP_AFTER_NS = 1000000;   // device time (about 20 us per step here)
    const uint32_t n = 128, t = 2, steps = 3000;
    std::string err;
    const problem prob(n, 21);
    const dense_j J(prob);
    device dev;
    if (!dev.open("fixed", "8")) { CHECK(false, "cannot boot the image"); return; }
    options opt;
    opt.t = t; opt.energy = true;
    machine m;
    if (!m.open(dev.t, dev.kernel, prob, opt, &err)) { CHECK(false, "%s", err.c_str()); return; }
    std::vector<float> x0, y0;
    prob.initial_state(t, 5, &x0, &y0);
    m.write_state(x0.data(), y0.data());
    const schedule sch = schedule_for(prob, steps);
    // Let the kernel run a while, then ask it to stop.
    const uint64_t t_start = tsr_transport_now_ns(dev.t);
    uint32_t polls = 0;
    m.set_stop_poll([&]() { ++polls; return tsr_transport_now_ns(dev.t) >= t_start + STOP_AFTER_NS; });
    run_result res;
    CHECK(m.run(sch, steps, &res, &err), "%s", err.c_str());
    const uint32_t done = res.steps_done;
    CHECK(res.stopped && done > 0 && done < steps, "stopped %d after %u of %u steps (%u polls)", int(res.stopped), done, steps, polls);
    CHECK(res.energy.size() == size_t(done) * t, "%zu energy words for %u steps", res.energy.size(), done);
    // The state after `done` steps is the model's, and so are the energy words of those steps.
    std::vector<float> xr = x0, yr = y0;
    bitexact_out ref;
    bitexact_launch(J, t, m.place().kblk, done, 0, true, sch, xr.data(), yr.data(), &ref);
    CHECK(same("x after the stop", res.x, xr) && same("y after the stop", res.y, yr) && same("energy after the stop", res.energy, ref.energy),
          "the stopped run differs from the model");
    // The run goes on from the stored state: 60 more steps with the cA of the steps done.
    m.set_stop_poll(nullptr);
    schedule rest = sch;
    rest.ca0 = advance_ca(sch.ca0, sch.da, done);
    run_result more;
    CHECK(m.run(rest, 60, &more, &err) && !more.stopped && more.steps_done == 60, "the run after the stop: %s", err.c_str());
    bitexact_launch(J, t, m.place().kblk, 60, 0, true, rest, xr.data(), yr.data(), &ref);
    CHECK(same("x after the stop and 60 more steps", more.x, xr) && same("y after the stop and 60 more steps", more.y, yr), "the run after a stop differs from the model");
    // A stop that is already requested starts no launch and leaves the state as it was.
    m.set_stop_poll([]() { return true; });
    run_result none;
    CHECK(m.run(rest, 10, &none, &err) && none.stopped && none.steps_done == 0 && none.energy.empty(), "a stop before the first launch: %s", err.c_str());
    CHECK(same("x after a stop before the first launch", none.x, xr), "the state changed");
    printf("soft stop: stopped after %u of %u steps, state and energies as the model; 60 more steps as the model\n", done, steps);
}

// 5. Quality: the best e of 8 replicas on the device (n 512, 1000 steps, dt 1.0, the tool's defaults). A ground state at
// this size has e about -0.75 (the Parisi value -0.7632 plus a correction that goes as N^(-2/3)); random spins have e about 0.
static void test_quality() {
    const uint32_t n = 512, t = 8, steps = 1000;
    const uint64_t seed = 1;
    std::string err;
    const problem prob(n, seed);
    std::vector<float> x0, y0;
    prob.initial_state(t, seed ^ 0x5151, &x0, &y0);
    device dev;
    if (!dev.open("fixed", "8")) { CHECK(false, "cannot boot the image"); return; }
    options opt;
    opt.t = t; opt.energy = false;
    machine m;
    if (!m.open(dev.t, dev.kernel, prob, opt, &err)) { CHECK(false, "%s", err.c_str()); return; }
    m.write_state(x0.data(), y0.data());
    run_result res;
    CHECK(m.run(make_schedule(1.0, prob.default_c0(), 1.0, steps), steps, &res, &err), "%s", err.c_str());
    std::vector<int8_t> sd;
    spins_from_x(prob, res.x.data(), t, &sd);
    std::vector<evaluation> ed;
    evaluate(prob, sd.data(), t, &ed);
    double best = 1e9;
    for (uint32_t r = 0; r < t; ++r) best = std::min(best, ed[r].e);
    printf("quality: best e %.4f (n %u, %u steps, %u replicas)\n", best, n, steps, t);
    CHECK(best < -0.73, "the best e %.4f is not below -0.73", best);
}

int main() {
    test_constants();
    test_schedule();
    test_couplings();
    int bad = 0;
    // Bit-exact, exact energies; the fixed schedule and random ones, the race detector fatal.
    const case_cfg small = {"n 64, t 3, snapshot every step", 64, 3, 64, 40, 1, 0, 11};
    for (const char * s : { "fixed", "random:1", "random:2", "random:3" }) bad += !run_case(small, s);
    // Several column tensors (96 = 3 x 32, and 64 + 32; 200 spins pad to 224: seven tensors of 32, or 96 + 96 + 32), t = 8,
    // padded sizes.
    bad += !run_case({"n 96, kblk 32 (3 tensors)", 96, 2, 32, 24, 4, 0, 12}, "fixed");
    bad += !run_case({"n 96, kblk 64 (64 + 32)", 96, 3, 64, 24, 3, 0, 13}, "random:4");
    bad += !run_case({"n 64, t 8", 64, 8, 64, 20, 5, 0, 14}, "random:5");
    bad += !run_case({"n 100 (padded to 128), t 4", 100, 4, 0, 30, 6, 0, 15}, "fixed");
    bad += !run_case({"n 200 (224), kblk 32 (7 tensors)", 200, 3, 32, 30, 5, 0, 18}, "random:7");
    bad += !run_case({"n 200 (224), kblk 96, launches of 10", 200, 2, 96, 30, 5, 10, 19}, "random:8");
    // A run cut into launches (a snapshot period of 4 divides the launch of 12 steps).
    bad += !run_case({"cut: 40 steps in launches of 12", 64, 3, 64, 40, 4, 12, 16}, "fixed");
    bad += !run_case({"cut: 3 tensors, launches of 8", 96, 2, 32, 24, 1, 8, 17}, "random:6");
    CHECK(bad == 0, "%d of the bit-exact cases differ", bad);
    test_sign_step();
    test_status();
    test_stop();
    test_quality();
    printf("fw_example_ising: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}

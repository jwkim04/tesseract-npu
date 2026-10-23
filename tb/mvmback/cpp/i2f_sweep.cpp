// i2f_sweep.cpp — Verilator C++ harness: tsr_mvm_i2f (rtl/core/tsr_mvm_back.sv, the 3-stage
// int -> f32 converter of the MVM back end) against libtsr_ref tsr_i32_to_f32 (sw/sim/tsr_ref.c)
// for every IW-bit two's complement input in [lo, hi) (default: all 2^IW).
//
// Toplevel: tb/mvmback/hdl/tsr_tb_mvm_i2f.sv (LANES converters, latency 3). Built and run by
// tb/mvmback/test_mvm_back.py test_mvm_back_i2f_sweep. Usage: i2f_sweep [lo hi] (signed decimal).
// Exit status 0 iff no mismatch; prints the count of checked inputs and the first mismatches.

#include <verilated.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "Vtsr_tb_mvm_i2f.h"

extern "C" {
#include "tsr_ref.h"
}

#ifndef LANES
#define LANES 16
#endif
#ifndef IW
#define IW 26
#endif
static const int LAT = 3;

template <typename W>
static inline void setb(W& w, unsigned lsb, unsigned width, uint64_t v) {
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        const uint32_t m = 1u << (b % 32);
        if ((v >> i) & 1u) w[b / 32] |= m; else w[b / 32] &= ~m;
    }
}
template <typename W>
static inline uint32_t get32(const W& w, unsigned lane) { return w[lane]; }

static uint32_t ref_bits(int64_t v) {
    const float f = tsr_i32_to_f32((int32_t)v);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

int main(int argc, char** argv) {
    const int64_t full_lo = -(1ll << (IW - 1)), full_hi = 1ll << (IW - 1);
    int64_t lo = full_lo, hi = full_hi;
    if (argc >= 3) {
        lo = std::strtoll(argv[1], nullptr, 0);
        hi = std::strtoll(argv[2], nullptr, 0);
    }
    if (lo < full_lo) lo = full_lo;
    if (hi > full_hi) hi = full_hi;
    auto ctx = std::make_unique<VerilatedContext>();
    auto top = std::make_unique<Vtsr_tb_mvm_i2f>(ctx.get());
    // pipeline of the inputs presented in the last LAT cycles (per lane; -1 = none)
    int64_t hist[LAT + 1][LANES];
    for (auto& h : hist) for (auto& x : h) x = INT64_MIN;
    uint64_t checked = 0, bad = 0;
    int64_t next = lo;
    top->clk = 0;
    top->eval();
    for (uint64_t cyc = 0;; ++cyc) {
        // present LANES inputs (or repeat lo when done; marked as not to check)
        int64_t cur[LANES];
        for (int l = 0; l < LANES; ++l) {
            if (next < hi) { cur[l] = next++; } else { cur[l] = INT64_MIN; }
            const uint64_t raw = (cur[l] == INT64_MIN ? 0 : (uint64_t)cur[l]) & ((1ull << IW) - 1);
            setb(top->v, (unsigned)(l * IW), IW, raw);
        }
        top->eval();
        top->clk = 1;
        top->eval();
        top->clk = 0;
        top->eval();
        // shift history: hist[0] = presented this cycle
        for (int s = LAT; s > 0; --s) std::memcpy(hist[s], hist[s - 1], sizeof(hist[0]));
        std::memcpy(hist[0], cur, sizeof(cur));
        // after LAT edges the result of hist[LAT - 1] is on y
        bool any = false;
        for (int l = 0; l < LANES; ++l) {
            const int64_t vin = hist[LAT - 1][l];
            if (vin == INT64_MIN) continue;
            any = true;
            const uint32_t got = get32(top->y, (unsigned)l);
            const uint32_t exp = ref_bits(vin);
            ++checked;
            if (got != exp) {
                if (bad < 10) std::printf("MISMATCH v=%" PRId64 " got 0x%08x exp 0x%08x\n", vin, got, exp);
                ++bad;
            }
        }
        if (!any && next >= hi && cyc > (uint64_t)LAT) break;
    }
    std::printf("i2f_sweep IW=%d [%" PRId64 ", %" PRId64 "): %" PRIu64 " checked, %" PRIu64 " mismatches\n",
                IW, lo, hi, checked, bad);
    return (bad == 0 && checked == (uint64_t)(hi - lo)) ? 0 : 1;
}

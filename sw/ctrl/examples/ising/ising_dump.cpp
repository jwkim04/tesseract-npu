// The dump of tsr-ising (ising_dump.hpp).
//
// The dump (little-endian; words are 32 bits unless said):
//   header, 32 words:
//      0 magic 0x50445349 ("ISDP")    1 version (2)                  2 problem kind id (2: Sherrington-Kirkpatrick)
//      3 spins                        4 n, the machine's spins       5 t replicas
//      6 steps done                   7 snap_every (0: none)         8 flags: bit 0 is 1 (a lower score is better),
//                                        bit 2 the curve comes from the device's energy words (else from the snapshots),
//                                        bit 3 a soft stop ended the run early
//      9 number of snapshots         10 reserved (0)
//     11 n, the spins padded to 32   12, 13 seed (uint64)            14, 15, 16 f32 bits of a0, c0, dt
//     17 rows of the curve           18 step of the first curve row  19 steps between curve rows
//     20 reserved (0)                21 steps requested               22..31 reserved (0)
//   results, t records of 16 bytes: float64 energy E and float64 e = E / N^1.5 for the final signs of every replica
//     (the signs are +1 where x >= 0)
//   curve: `rows` rows of t float64: the energy E of every replica at the steps first, first + every, ... A run with
//     energy words gets it from the device's words (E = -e / 2 of the signs that enter a step: row k is the state after
//     k steps; first = 0, every = 1); a run without them from the snapshots (first = every = snap_every). No energy
//     words and no snapshots: no rows.
//   final x: t rows of n float32
//   snapshots: `number of snapshots` blocks of t rows of n float32: x of every replica after step snap_every,
//     2 snap_every, ... of the launch (the snapshots of a run in several launches continue at the same period)
#include "ising_dump.hpp"

#include <stdio.h>

#include "tsr_ref.h"

namespace ism {

namespace {

constexpr uint32_t DUMP_MAGIC = 0x50445349u;   // "ISDP"
constexpr uint32_t DUMP_VERSION = 2u;
constexpr uint32_t DUMP_HEADER_WORDS = 32u;
constexpr uint32_t KIND_SK = 2u;

void put32(std::vector<uint8_t> * b, uint32_t v) { for (int i = 0; i < 4; ++i) b->push_back(uint8_t(v >> (8 * i))); }
void put64(std::vector<uint8_t> * b, uint64_t v) { put32(b, uint32_t(v)); put32(b, uint32_t(v >> 32)); }
uint64_t bits64(double d) { uint64_t u; static_assert(sizeof(u) == sizeof(d), ""); __builtin_memcpy(&u, &d, 8); return u; }

}  // namespace

void energy_of_snapshots(const problem & p, const std::vector<float> & snaps, uint32_t t, std::vector<double> * out) {
    const size_t rows = t ? snaps.size() / (size_t(t) * p.n) * t : 0;   // snapshots x replicas
    std::vector<int8_t> spins;
    spins_from_x(p, snaps.data(), uint32_t(rows), &spins);   // one row per (snapshot, replica) pair
    std::vector<evaluation> ev;
    evaluate(p, spins.data(), uint32_t(rows), &ev);
    out->resize(rows);
    for (size_t i = 0; i < rows; ++i) (*out)[i] = ev[i].energy;
}

bool write_dump(const char * path, const dump_input & in) {
    const problem & p = *in.prob;
    const run_result & res = *in.res;
    const uint32_t t = in.t, n = p.n;
    const uint64_t nsnap = res.snapshots.size() / (size_t(t) * n);

    // The curve.
    const bool from_words = in.energy_words;
    std::vector<double> curve;
    uint32_t first = 0, every = 0;
    if (from_words) {
        for (float e : res.energy) curve.push_back(-0.5 * double(e));
        first = 0;
        every = 1;
    } else if (nsnap != 0) {
        energy_of_snapshots(p, res.snapshots, t, &curve);
        first = in.snap_every;
        every = in.snap_every;
    }

    std::vector<uint8_t> h;
    const uint32_t flags = 1u | (from_words ? 4u : 0u) | (res.stopped ? 8u : 0u);
    const uint32_t head[DUMP_HEADER_WORDS] = {
        DUMP_MAGIC, DUMP_VERSION, KIND_SK, p.spins, n, t, res.steps_done, in.snap_every, flags, uint32_t(nsnap),
        0u, n, uint32_t(p.seed), uint32_t(p.seed >> 32),
        tsr_f32_bits(float(in.a0)), tsr_f32_bits(float(in.c0)), tsr_f32_bits(float(in.dt)),
        uint32_t(t ? curve.size() / t : 0), first, every, 0u, in.steps_requested,
    };
    for (uint32_t w : head) put32(&h, w);
    for (uint32_t r = 0; r < t; ++r) {
        put64(&h, bits64((*in.final_eval)[r].energy));
        put64(&h, bits64((*in.final_eval)[r].e));
    }
    for (double c : curve) put64(&h, bits64(c));
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(h.data(), 1, h.size(), f) == h.size();
    ok &= fwrite(res.x.data(), 4, res.x.size(), f) == res.x.size();
    ok &= fwrite(res.snapshots.data(), 4, res.snapshots.size(), f) == res.snapshots.size();
    ok &= fclose(f) == 0;
    return ok;
}

}  // namespace ism

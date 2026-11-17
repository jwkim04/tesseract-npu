// The ABI of the Ising machine example: what the kernel `ising` (ising.cpp) and its host side
// (ising_run.cpp, the reference ising_ref.cpp, the tests) agree on. Plain C++17 with no library
// header, so the firmware (freestanding RV32) and the host both include it.
//
// A launch has one argument, ARG0 = the DRAM handle {channel[31:28], beat[27:0]} of the parameter
// block (`params`, two lines of 64 bytes; the runtime loads it into the SPM before the kernel starts,
// TSR_KFLAG_ARGBLK). The kernel returns 0, or one of `status` below (positive) for a bad block, or
// ST_STOPPED_FLAG | steps done when a soft stop ended the launch early (EX §4.9; the runtime then
// answers STOPPED). The status is the launch record's RET word (r2).
//
// State and results, all f32, t replicas side by side:
//   x, y      t rows of n floats each (n a multiple of 32), row r at beat r * n / 16 of the area. The
//             kernel reads both at the start of a launch and writes both back at the end, so a run
//             can be cut into launches.
//   snapshots every snap_every steps the t rows of x after the step, one after the other (t * n / 16
//             beats apart), counted from the start of the launch.
//   energy    one 64-byte line per step (beat k = step k of the launch): word r is
//             e_r = sum_i s_i f_i of replica r (the signs that enter the step; the Ising energy of
//             them is -e_r / 2); words t..15 are 0.
//   weights   J as Q8_0 tensors: tensor b holds the columns [b * kblk, min(n, (b + 1) * kblk)) of all
//             n rows (n rows, K = its number of columns), block scale 1.0, symmetric allocation
//             (spec §5): the W_OFF of tensor b is w_off + b * tensor_stride beats on every channel.
#pragma once

#include <stdint.h>

// The namespace is `ism` (Ising machine): the kernel itself is the function `ising`.
namespace ism {

constexpr uint32_t MAGIC = 0x474E5349u;    // "ISNG" read as little-endian bytes
constexpr uint32_t VERSION = 1u;
constexpr uint32_t PARAM_LINES = 2u;       // the block is two SPM lines (32 words)
constexpr uint32_t MAX_N = 1u << 20;       // spins (a multiple of 32)
constexpr uint32_t MAX_STEPS = 4096u;      // steps in one launch (one SPM line of energy each)
constexpr uint32_t MAX_BLK = 32u;          // column tensors
constexpr uint32_t FLAG_ENERGY = 1u;       // flags bit 0: store the energy line of every step

// The parameter block (little-endian words, in this order).
struct params {
    uint32_t magic;          // MAGIC
    uint32_t version;        // VERSION
    uint32_t n;              // spins, a multiple of 32, at most MAX_N
    uint32_t t;              // replicas, 1 .. the device's T_MAX (at most 8)
    uint32_t steps;          // steps of this launch, 1 .. MAX_STEPS (0 is legal: a launch that only moves state)
    uint32_t snap_every;     // 0: no snapshots
    uint32_t flags;          // FLAG_ENERGY
    uint32_t w_off;          // weight tensor 0: beat offset, the same on every channel
    uint32_t kblk;           // columns per weight tensor: a multiple of 32, at most the device's K_MAX
    uint32_t tensor_stride;  // beats between consecutive column tensors
    uint32_t x;              // DRAM handles: state x, state y, snapshot area, energy area
    uint32_t y;
    uint32_t snap;
    uint32_t energy;
    uint32_t ca0;            // f32 bits: cA at the start of the launch, dA (cA += dA per step), cF, cD
    uint32_t da;
    uint32_t cf;
    uint32_t cd;
    uint32_t reserved[14];   // 0
};
static_assert(sizeof(params) == 32 * 4, "params is two SPM lines");

enum status : int32_t {
    ST_OK = 0,
    ST_BAD_MAGIC = 1,        // magic or version
    ST_BAD_N = 2,            // n not a multiple of 32, 0 or above MAX_N
    ST_BAD_T = 3,            // t outside 1 .. the device's T_MAX
    ST_BAD_STEPS = 4,        // steps above MAX_STEPS
    ST_BAD_KBLK = 5,         // kblk not a multiple of 32, 0, above the device's K_MAX, or more than MAX_BLK tensors
    ST_BAD_FLAGS = 6,        // an unknown flag bit
    ST_BAD_SPM = 7,          // the buffers do not fit in the SPM
};
constexpr int32_t ST_STOPPED_FLAG = 1 << 16;   // returned with the number of steps done in the low bits

// The SPM layout, in lines, from line 0 up. L = n / 16 is the line distance of two rows.
//   x, y, s (signs), f (fields), tmp: t rows each; one, neg: constant rows of +1 and -1 (one row);
//   qa[b]: the QA image of column tensor b for every replica (t images, QA_LINES apart);
//   e: the energy lines (steps of this launch, 0 with the monitor off).
struct layout {
    uint32_t nblk;           // column tensors
    uint32_t L;
    uint32_t x, y, s, f, tmp, one, neg;
    uint32_t qa[MAX_BLK];
    uint32_t kb[MAX_BLK];    // K of column tensor b
    uint32_t e;
    uint32_t total;          // lines used
};

// QA_LINES of an A32 image of k values (spec §6): ceil(k / 64) lines of int8, ceil(k / 32 * 4 / 64) of scales.
constexpr uint32_t qa_lines(uint32_t k) { return (k + 63u) / 64u + (k + 511u) / 512u; }
// First line of the scale words of an A32 image (the int8 region comes first).
constexpr uint32_t qa_scale_line(uint32_t k) { return (k + 63u) / 64u; }

// Checks the block against the device (spm_lines = the lines the kernel may use, t_max, k_max) and fills
// the layout. Returns ST_OK or the first status that applies.
inline int32_t make_layout(const params & p, uint32_t spm_lines, uint32_t t_max, uint32_t k_max, layout * lay) {
    if (p.magic != MAGIC || p.version != VERSION) return ST_BAD_MAGIC;
    if (p.n == 0u || p.n > MAX_N || p.n % 32u != 0u) return ST_BAD_N;
    if (p.t == 0u || p.t > t_max || p.t > 8u) return ST_BAD_T;
    if (p.steps > MAX_STEPS) return ST_BAD_STEPS;
    if (p.kblk == 0u || p.kblk % 32u != 0u || p.kblk > k_max) return ST_BAD_KBLK;
    const uint32_t nblk = (p.n + p.kblk - 1u) / p.kblk;
    if (nblk > MAX_BLK) return ST_BAD_KBLK;
    if ((p.flags & ~FLAG_ENERGY) != 0u) return ST_BAD_FLAGS;
    layout l{};
    l.nblk = nblk;
    l.L = p.n / 16u;
    const uint32_t tl = p.t * l.L;
    l.x = 0u;
    l.y = tl;
    l.s = 2u * tl;
    l.f = 3u * tl;
    l.tmp = 4u * tl;
    l.one = 5u * tl;
    l.neg = 5u * tl + l.L;
    uint32_t at = 5u * tl + 2u * l.L;
    for (uint32_t b = 0; b < nblk; ++b) {
        const uint32_t k = (b + 1u < nblk) ? p.kblk : p.n - b * p.kblk;
        l.kb[b] = k;
        l.qa[b] = at;
        at += p.t * qa_lines(k);
    }
    l.e = at;
    l.total = at + ((p.flags & FLAG_ENERGY) != 0u ? p.steps : 0u);
    if (l.total > spm_lines) return ST_BAD_SPM;
    *lay = l;
    return ST_OK;
}

}  // namespace ism

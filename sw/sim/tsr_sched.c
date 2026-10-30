// tsr_sched.c — Tesseract simulator, schedule policy (P1.8; docs/design/06-software-verification.md
// "SW" §4.4, §4.7, §4.9): TSR_SIM_SCHED parsing, the xoshiro128** streams and the draws the time
// base (tsr_q.c) and the RISC-V agent (tsr_bus.c) make under the random schedule. The policy
// itself (what is drawn where, the CDC lag of the RISC-V-side reads, ties, fast-forward) is
// described in tsr_sim_int.h §5b; this file only owns the numbers.
//
// Determinism: every draw comes from a stream of its own purpose (TSR_RS_*), seeded from
// splitmix64 of (seed, stream), so adding a draw to one purpose never shifts another. Durations are
// computed in integers from a Q16.16 factor; the only floating-point step is the handshake latency
// conversion by F_RV / F_CORE (one IEEE multiply and add, -ffp-contract=off), the same conversion
// the agent already makes for its clock.

#define _GNU_SOURCE
#define TSR_SIM_CORE_INTERNAL

#include "tsr_sim.h"
#include "tsr_sim_int.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FACTOR_ONE   65536u          // Q16.16 1.0
#define FACTOR_MIN_D 0.25            // SW §4.4 defaults
#define FACTOR_MAX_D 4.0
#define FACTOR_LIMIT 1024.0
#define XING_MIN     2u              // crossings, core cycles (SW §4.4)
#define XING_MAX     8u

// ---------------------------------------------------------------------------
// xoshiro128** 1.1 (D. Blackman, S. Vigna, public domain reference algorithm) and splitmix64
// ---------------------------------------------------------------------------
static inline uint32_t rotl32(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

uint32_t tsr_rng_next(tsr_rng * r) {
    uint32_t * s = r->s;
    const uint32_t result = rotl32(s[1] * 5u, 7) * 9u;
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11);
    return result;
}

static uint64_t splitmix64(uint64_t * x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void tsr_rng_seed(tsr_rng * r, uint64_t seed, uint32_t stream) {
    // Each stream starts splitmix64 at its own point (an odd multiplier, not the splitmix
    // increment, so two streams never share outputs within one seed).
    uint64_t x = seed + (uint64_t)(stream + 1u) * 0x632BE59BD9B4E019ull;
    const uint64_t a = splitmix64(&x), b = splitmix64(&x);
    r->s[0] = (uint32_t)a;
    r->s[1] = (uint32_t)(a >> 32);
    r->s[2] = (uint32_t)b;
    r->s[3] = (uint32_t)(b >> 32);
    if ((r->s[0] | r->s[1] | r->s[2] | r->s[3]) == 0u) r->s[0] = 1u;   // the one forbidden state
}

uint32_t tsr_rng_below(tsr_rng * r, uint32_t n) {
    return (uint32_t)(((uint64_t)tsr_rng_next(r) * (uint64_t)(n ? n : 1u)) >> 32);
}

// ---------------------------------------------------------------------------
// TSR_SIM_SCHED
// ---------------------------------------------------------------------------
static int factor_q16(double v, uint32_t * out) {
    if (!(v > 0.0) || !(v <= FACTOR_LIMIT)) return -1;            // also rejects NaN
    const double q = floor(v * (double)FACTOR_ONE + 0.5);
    if (q < 1.0) return -1;
    *out = (uint32_t)q;
    return 0;
}

int tsr_sched_parse(const char * text, tsr_sched_spec * out, char * err, size_t errlen) {
    tsr_sched_spec s;
    memset(&s, 0, sizeof(s));
    if (err && errlen) err[0] = '\0';
    if (text == NULL || text[0] == '\0' || strcmp(text, "fixed") == 0) {
        *out = s;
        return 0;
    }
    if (strncmp(text, "random:", 7) != 0) {
        if (err) snprintf(err, errlen, "\"%s\": expected \"fixed\" or \"random:<seed>[,<min>,<max>]\"", text);
        return -1;
    }
    const char * p = text + 7;
    char * end = NULL;
    if (*p == '\0' || *p == '-' || *p == '+' || *p == ' ') {
        if (err) snprintf(err, errlen, "\"%s\": missing or signed seed", text);
        return -1;
    }
    // Decimal, or hex after an explicit 0x / 0X; a leading 0 is decimal ("random:010" is seed 10,
    // never octal). Digits only (no sign, space or second prefix), 64 bits.
    const int hex = p[0] == '0' && (p[1] == 'x' || p[1] == 'X');
    const char * q = hex ? p + 2 : p;
    uint64_t seed = 0;
    int nd = 0, ovf = 0;
    for (;; ++q, ++nd) {
        uint32_t dv;
        if (*q >= '0' && *q <= '9') dv = (uint32_t)(*q - '0');
        else if (hex && *q >= 'a' && *q <= 'f') dv = (uint32_t)(*q - 'a') + 10u;
        else if (hex && *q >= 'A' && *q <= 'F') dv = (uint32_t)(*q - 'A') + 10u;
        else break;
        const uint64_t base = hex ? 16u : 10u;
        if (seed > (UINT64_MAX - dv) / base) ovf = 1;
        seed = seed * base + dv;
    }
    if (nd == 0 || ovf || (*q != '\0' && *q != ',')) {
        if (err) snprintf(err, errlen, "\"%s\": the seed is not a 64-bit decimal or 0x number", text);
        return -1;
    }
    s.random = 1;
    s.seed = seed;
    double fmin = FACTOR_MIN_D, fmax = FACTOR_MAX_D;
    if (*q == ',') {
        p = q + 1;
        fmin = strtod(p, &end);
        if (end == p || *end != ',') {
            if (err) snprintf(err, errlen, "\"%s\": expected <seed>,<min>,<max>", text);
            return -1;
        }
        p = end + 1;
        fmax = strtod(p, &end);
        if (end == p || *end != '\0') {
            if (err) snprintf(err, errlen, "\"%s\": expected <seed>,<min>,<max>", text);
            return -1;
        }
    }
    if (factor_q16(fmin, &s.fmin) != 0 || factor_q16(fmax, &s.fmax) != 0 || s.fmin > s.fmax) {
        if (err) snprintf(err, errlen, "\"%s\": need 0 < min <= max <= %g (duration factors)", text, FACTOR_LIMIT);
        return -1;
    }
    *out = s;
    return 0;
}

void tsr_sched_format(const tsr_sched_spec * s, char * buf, size_t n) {
    if (!s->random) {
        snprintf(buf, n, "fixed");
        return;
    }
    // %.17g: the Q16.16 bounds are exact doubles, so the text parses back to the same spec.
    snprintf(buf, n, "random:%llu,%.17g,%.17g", (unsigned long long)s->seed, (double)s->fmin / FACTOR_ONE,
             (double)s->fmax / FACTOR_ONE);
}

void tsr_sched_init(tsr_sim * sim, const tsr_sched_spec * spec) {
    sim->sched = *spec;
    tsr_sched_format(spec, sim->sched_text, sizeof(sim->sched_text));
    for (uint32_t k = 0; k < TSR_RS_COUNT; ++k) tsr_rng_seed(&sim->rs[k], spec->seed, k);
    sim->view_on = spec->random;
}

int tsr_sim_sched_random(const tsr_sim * sim) { return sim->sched.random; }
const char * tsr_sim_opt_sched(const tsr_sim * sim) { return sim->sched_text; }

// ---------------------------------------------------------------------------
// Draws (tsr_sim_int.h §5b)
// ---------------------------------------------------------------------------
uint64_t tsr_sched_op_cycles(tsr_sim * sim, uint32_t unit, uint64_t base) {
    if (base == 0) base = 1;
    if (!sim->sched.random) return base;
    const tsr_sched_spec * s = &sim->sched;
    tsr_rng * r = &sim->rs[TSR_RS_DUR + (unit < TSR_NUM_UNITS ? unit : 0u)];
    // f uniform over the Q16.16 values of [fmin, fmax], both bounds included (SW §4.4 [min, max])
    const uint64_t f = s->fmin + (((uint64_t)(s->fmax - s->fmin) + 1u) * tsr_rng_next(r) >> 32);
    // round(base x f / 2^16) in integers: f <= 2^26 (max 1024), so both terms fit 64 bits for
    // base < 2^54; only the low part is rounded (the high part is an exact integer)
    const uint64_t d = (base >> 16) * f + (((base & 0xFFFFu) * f + 0x8000u) >> 16);
    return d ? d : 1;
}

// Issue-FIFO crossing of one entry (push store -> dispatcher), core cycles.
uint64_t tsr_sched_issue_crossing(tsr_sim * sim) {
    if (!sim->sched.random) return TSR_Q_XING;
    return XING_MIN + tsr_rng_below(&sim->rs[TSR_RS_XING], XING_MAX - XING_MIN + 1u);
}

// Visibility lag of one change of the RISC-V-side view, core cycles.
uint64_t tsr_sched_view_lag(tsr_sim * sim) {
    return XING_MIN + tsr_rng_below(&sim->rs[TSR_RS_VIEW], XING_MAX - XING_MIN + 1u);
}

uint32_t tsr_sched_rv_handshake(tsr_sim * sim) {
    if (!sim->sched.random) return 0;
    const uint32_t x = XING_MIN + tsr_rng_below(&sim->rs[TSR_RS_HS], XING_MAX - XING_MIN + 1u);
    return (uint32_t)floor((double)(x - XING_MIN) * sim->rv_ratio + 0.5);
}

int tsr_sched_rv_cpi(tsr_sim * sim) {
    if (!sim->sched.random) return 0;
    return (int)tsr_rng_below(&sim->rs[TSR_RS_CPI], 3u) - 1;
}

void tsr_sim_set_fast_forward(tsr_sim * sim, int on) { sim->ff = on != 0; }
int  tsr_sim_fast_forward(const tsr_sim * sim) { return sim->ff; }

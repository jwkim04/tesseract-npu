// test_util.h — tiny helpers shared by the tsr_ref unit tests (no framework).
#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static long g_checks = 0;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(cond)) {                                                            \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);       \
            fprintf(stderr, __VA_ARGS__);                                         \
            fputc('\n', stderr);                                                  \
            if (++g_fail >= 40) {                                                 \
                fprintf(stderr, "too many failures, stopping\n");                 \
                exit(1);                                                          \
            }                                                                     \
        }                                                                         \
    } while (0)

static inline int test_finish(const char * name) {
    if (g_fail) {
        fprintf(stderr, "%s: %d FAILED of %ld checks\n", name, g_fail, g_checks);
        return 1;
    }
    printf("%s: all %ld checks passed\n", name, g_checks);
    return 0;
}

// splitmix64 — deterministic, seedable.
typedef struct { uint64_t s; } rng_t;
static inline uint64_t rng_u64(rng_t * r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline uint32_t rng_u32(rng_t * r) { return (uint32_t)(rng_u64(r) >> 32); }
static inline uint32_t rng_below(rng_t * r, uint32_t n) { return (uint32_t)(((rng_u64(r) >> 32) * (uint64_t)n) >> 32); }
static inline double rng_unit(rng_t * r) { return (double)(rng_u64(r) >> 11) * 0x1.0p-53; }
static inline double rng_normal(rng_t * r) {
    double u1 = rng_unit(r), u2 = rng_unit(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static inline void rng_bytes(rng_t * r, void * dst, size_t n) {
    uint8_t * p = (uint8_t *)dst;
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rng_u32(r);
}

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

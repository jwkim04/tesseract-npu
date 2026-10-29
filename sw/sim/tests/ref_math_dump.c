// ref_math_dump.c — evaluate one tsr_ref_math function on a strided sweep of f32 bit patterns
// and write the result bit patterns to stdout (uint32, little-endian, raw). The independent
// numpy model (sw/sim/tests/ref_math_np.py, VU §10 item 1, V2-SW-23) has the same command
// line, so the two byte streams compare with cmp(1); --check compares here instead.
//
//   ref_math_dump <func> <start_hex> <count> <stride> [<b_hex>]
//   ref_math_dump --check <func> <start_hex> <count> <stride> [<b_hex>]   (expected on stdin)
//
// Input i (0 <= i < count) is the f32 with bits start + i·stride (uint32 wraparound). count
// is 0 .. 2^32 and stride 0 .. 2^32-1, decimal or 0x-prefixed hex; start and b are hex with
// or without 0x. Output: count words (sincos: 2·count, sin then cos per input).
//
// Functions (names are the VU §6.1 reference names without 'tsr_m_'; the 'tsr_m_' prefix and
// upper case are accepted too):
//   unary   exp, expk_p1, expk_n1, expk_n2, expk_ng, sigmoid, silu, gelu_tanh, gelu16, tanh,
//           sin, cos, sincos
//           expk with a κ argument instead of b: 'expk <start> <count> <stride> <kappa>',
//           kappa = p1 | n1 | n2 | ng or 0 | 1 | 2 | 3 (TSR_M_P1 .. TSR_M_NG)
//   binary  the swept input is the FIRST operand, b (hex bits) is fixed for every element:
//           geglu16    tsr_m_geglu16(a, b)           (VOP GEGLU16)
//           expsub     tsr_m_expsub(a, b)            (VOP EXPSUB / EXPSUBC: EXPSUB(a, x = b))
//           swiglu     FM(tsr_m_silu(a), b)          (VOP SWIGLU)
//           geglu_tanh FM(tsr_m_gelu_tanh(a), b)     (VOP GEGLU; alias geglu)
//   binary16 helpers (VU §6.1), one word per input:
//           f32_to_f16_rne  tsr_f32_to_f16_rne(x), the 16-bit result zero-extended
//           f16_to_f32      tsr_f16_to_f32(low 16 bits of the input word)
//   f32 primitives of tsr_ref.c (VU §6.1 FA FS FM FD SQ RC), so the numpy model's primitive
//   emulation is checked against the C ones: fsqrt, frecip (unary); fadd, fsub, fmul, fdiv (binary)
//   aliases sig = sigmoid, gelu = gelu_tanh

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsr_ref.h"
#include "tsr_ref_math.h"

typedef enum { K_UNARY, K_EXPK, K_BINARY, K_PAIR, K_RAW } fkind;

typedef struct {
    const char * name;
    fkind kind;
    float (*f1)(float);
    float (*f2)(float, float);
    uint32_t (*fr)(uint32_t);   // K_RAW: input bits -> output word
} fdesc;

static float f_expk_p1(float x) { return tsr_m_expk(x, TSR_M_P1); }
static float f_expk_n1(float x) { return tsr_m_expk(x, TSR_M_N1); }
static float f_expk_n2(float x) { return tsr_m_expk(x, TSR_M_N2); }
static float f_expk_ng(float x) { return tsr_m_expk(x, TSR_M_NG); }
static float f_swiglu(float a, float b) { return tsr_fmul(tsr_m_silu(a), b); }
static float f_geglu(float a, float b)  { return tsr_fmul(tsr_m_gelu_tanh(a), b); }
static uint32_t f_f32_to_f16(uint32_t x) { return tsr_f32_to_f16_rne(tsr_bits_f32(x)); }
static uint32_t f_f16_to_f32(uint32_t x) { return tsr_f32_bits(tsr_f16_to_f32((uint16_t)(x & 0xFFFFu))); }

static const fdesc k_funcs[] = {
    { "exp",        K_UNARY,  tsr_m_exp,       NULL, NULL },
    { "expk_p1",    K_UNARY,  f_expk_p1,       NULL, NULL },
    { "expk_n1",    K_UNARY,  f_expk_n1,       NULL, NULL },
    { "expk_n2",    K_UNARY,  f_expk_n2,       NULL, NULL },
    { "expk_ng",    K_UNARY,  f_expk_ng,       NULL, NULL },
    { "expk",       K_EXPK,   NULL,            NULL, NULL },
    { "sigmoid",    K_UNARY,  tsr_m_sigmoid,   NULL, NULL },
    { "sig",        K_UNARY,  tsr_m_sigmoid,   NULL, NULL },
    { "silu",       K_UNARY,  tsr_m_silu,      NULL, NULL },
    { "gelu_tanh",  K_UNARY,  tsr_m_gelu_tanh, NULL, NULL },
    { "gelu",       K_UNARY,  tsr_m_gelu_tanh, NULL, NULL },
    { "gelu16",     K_UNARY,  tsr_m_gelu16,    NULL, NULL },
    { "tanh",       K_UNARY,  tsr_m_tanh,      NULL, NULL },
    { "sin",        K_UNARY,  tsr_m_sin,       NULL, NULL },
    { "cos",        K_UNARY,  tsr_m_cos,       NULL, NULL },
    { "sincos",     K_PAIR,   NULL,            NULL, NULL },
    { "geglu16",    K_BINARY, NULL,            tsr_m_geglu16, NULL },
    { "expsub",     K_BINARY, NULL,            tsr_m_expsub,  NULL },
    { "swiglu",     K_BINARY, NULL,            f_swiglu,      NULL },
    { "geglu_tanh", K_BINARY, NULL,            f_geglu,       NULL },
    { "geglu",      K_BINARY, NULL,            f_geglu,       NULL },
    { "f32_to_f16_rne", K_RAW, NULL,           NULL,          f_f32_to_f16 },
    { "f16_to_f32", K_RAW,    NULL,            NULL,          f_f16_to_f32 },
    { "fsqrt",      K_UNARY,  tsr_fsqrt,       NULL, NULL },
    { "frecip",     K_UNARY,  tsr_frecip,      NULL, NULL },
    { "fadd",       K_BINARY, NULL,            tsr_fadd,      NULL },
    { "fsub",       K_BINARY, NULL,            tsr_fsub,      NULL },
    { "fmul",       K_BINARY, NULL,            tsr_fmul,      NULL },
    { "fdiv",       K_BINARY, NULL,            tsr_fdiv,      NULL },
};

static void usage(void) {
    fprintf(stderr,
            "usage: ref_math_dump [--check] <func> <start_hex> <count> <stride> [<b_hex> | <kappa>]\n"
            "  writes count uint32 result words (little-endian) for the inputs start + i*stride\n"
            "  (uint32 wraparound); --check reads the expected words from stdin and compares.\n"
            "  unary : exp expk_p1 expk_n1 expk_n2 expk_ng sigmoid silu gelu_tanh gelu16 tanh sin cos\n"
            "          sincos (2 words per input: sin, cos); expk <start> <count> <stride> <p1|n1|n2|ng|0..3>\n"
            "  binary: geglu16 expsub swiglu geglu_tanh: the swept input is the first operand a,\n"
            "          <b_hex> the fixed second operand (expsub: EXPSUB(a, x = b)); default b = 0x3F800000\n"
            "  helpers: f32_to_f16_rne (result zero-extended), f16_to_f32 (low 16 input bits)\n"
            "  primitives: fsqrt frecip; fadd fsub fmul fdiv (binary, as above)\n"
            "  names may carry the prefix tsr_m_ and any case; aliases: sig, gelu, geglu\n");
    exit(2);
}

static int parse_u64(const char * s, int hex_default, uint64_t * out) {
    int base = hex_default ? 16 : 10;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
        base = 16;
    }
    if (*s == '\0') {
        return 0;
    }
    char * end = NULL;
    const unsigned long long v = strtoull(s, &end, base);
    if (*end != '\0' || s[0] == '-' || s[0] == '+') {
        return 0;
    }
    *out = (uint64_t)v;
    return 1;
}

static const fdesc * find_func(const char * arg) {
    char name[64];
    size_t n = strlen(arg);
    if (n >= sizeof name) {
        return NULL;
    }
    for (size_t i = 0; i <= n; ++i) {
        name[i] = (char)tolower((unsigned char)arg[i]);
    }
    const char * p = name;
    if (strncmp(p, "tsr_m_", 6) == 0) {
        p += 6;
    }
    for (size_t i = 0; i < sizeof k_funcs / sizeof k_funcs[0]; ++i) {
        if (strcmp(p, k_funcs[i].name) == 0) {
            return &k_funcs[i];
        }
    }
    return NULL;
}

static int parse_kappa(const char * arg, unsigned * kappa) {
    static const char * names[4] = { "p1", "n1", "n2", "ng" };
    for (unsigned k = 0; k < 4; ++k) {
        if ((tolower((unsigned char)arg[0]) == names[k][0]) &&
            (tolower((unsigned char)arg[1]) == names[k][1]) && arg[2] == '\0') {
            *kappa = k;
            return 1;
        }
    }
    uint64_t v;
    if (parse_u64(arg, 0, &v) && v <= TSR_M_NG) {
        *kappa = (unsigned)v;
        return 1;
    }
    return 0;
}

static void put_le(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint32_t get_le(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(int argc, char ** argv) {
    int check = 0;
    int ai = 1;
    if (ai < argc && strcmp(argv[ai], "--check") == 0) {
        check = 1;
        ++ai;
    }
    if (argc - ai != 4 && argc - ai != 5) {
        usage();
    }
    const fdesc * fd = find_func(argv[ai]);
    if (!fd) {
        fprintf(stderr, "ref_math_dump: unknown function '%s'\n", argv[ai]);
        usage();
    }
    uint64_t start, count, stride;
    if (!parse_u64(argv[ai + 1], 1, &start) || start > 0xFFFFFFFFull ||
        !parse_u64(argv[ai + 2], 0, &count) || count > (1ull << 32) ||
        !parse_u64(argv[ai + 3], 0, &stride) || stride > 0xFFFFFFFFull) {
        fprintf(stderr, "ref_math_dump: bad start / count / stride\n");
        usage();
    }
    uint32_t bbits = 0x3F800000u;
    unsigned kappa = 0;
    if (fd->kind == K_EXPK) {
        if (argc - ai != 5 || !parse_kappa(argv[ai + 4], &kappa)) {
            fprintf(stderr, "ref_math_dump: expk needs a kappa argument (p1, n1, n2, ng or 0..3)\n");
            usage();
        }
    } else if (argc - ai == 5) {
        uint64_t b;
        if (fd->kind != K_BINARY || !parse_u64(argv[ai + 4], 1, &b) || b > 0xFFFFFFFFull) {
            fprintf(stderr, "ref_math_dump: a fifth argument is only taken by binary functions and expk\n");
            usage();
        }
        bbits = (uint32_t)b;
    }
    const float bval = tsr_bits_f32(bbits);
    const unsigned words_per = (fd->kind == K_PAIR) ? 2u : 1u;

    enum { CHUNK = 1 << 16 };
    static uint8_t buf[CHUNK * 2 * 4];
    static uint8_t exp_buf[CHUNK * 2 * 4];
    uint32_t x = (uint32_t)start;
    uint64_t done = 0, mismatches = 0;
    while (done < count) {
        const uint64_t left = count - done;
        const uint32_t n = (left < CHUNK) ? (uint32_t)left : (uint32_t)CHUNK;
        for (uint32_t i = 0; i < n; ++i, x += (uint32_t)stride) {
            const float xf = tsr_bits_f32(x);
            uint8_t * p = buf + (size_t)i * words_per * 4;
            switch (fd->kind) {
            case K_UNARY:  put_le(p, tsr_f32_bits(fd->f1(xf))); break;
            case K_EXPK:   put_le(p, tsr_f32_bits(tsr_m_expk(xf, kappa))); break;
            case K_BINARY: put_le(p, tsr_f32_bits(fd->f2(xf, bval))); break;
            case K_RAW:    put_le(p, fd->fr(x)); break;
            case K_PAIR: {
                float s, c;
                tsr_m_sincos(xf, &s, &c);
                put_le(p, tsr_f32_bits(s));
                put_le(p + 4, tsr_f32_bits(c));
                break;
            }
            }
        }
        const size_t bytes = (size_t)n * words_per * 4;
        if (check) {
            if (fread(exp_buf, 1, bytes, stdin) != bytes) {
                fprintf(stderr, "ref_math_dump --check: expected stream ended after %llu inputs\n",
                        (unsigned long long)done);
                return 1;
            }
            for (uint32_t i = 0; i < n * words_per; ++i) {
                const uint32_t got = get_le(buf + 4 * (size_t)i), want = get_le(exp_buf + 4 * (size_t)i);
                if (got != want) {
                    const uint64_t idx = done + i / words_per;
                    if (mismatches < 20) {
                        fprintf(stderr, "mismatch %s(0x%08x)%s: C 0x%08x, expected 0x%08x\n", fd->name,
                                (uint32_t)(start + idx * stride), (words_per == 2) ? ((i & 1) ? " cos" : " sin") : "",
                                got, want);
                    }
                    ++mismatches;
                }
            }
        } else if (fwrite(buf, 1, bytes, stdout) != bytes) {
            fprintf(stderr, "ref_math_dump: write error\n");
            return 1;
        }
        done += n;
    }
    if (check) {
        if (fgetc(stdin) != EOF) {
            fprintf(stderr, "ref_math_dump --check: expected stream is longer than %llu inputs\n",
                    (unsigned long long)count);
            return 1;
        }
        fprintf(stderr, "ref_math_dump --check %s: %llu inputs, %llu mismatching words\n", fd->name,
                (unsigned long long)count, (unsigned long long)mismatches);
        return mismatches ? 1 : 0;
    }
    return fflush(stdout) == 0 ? 0 : 1;
}

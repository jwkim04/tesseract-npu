// libc.cpp — the four C library functions a freestanding GCC may call on its own (struct and array
// copies and initialisations; docs/design/06-software-verification.md "SW" §6.1: no libc, libgcc only).
// Plain loops, word-wise (4 words per step) when both pointers and the size are word aligned.
// Compiled without loop-pattern recognition, so GCC does not turn the loops back into calls to
// themselves.
#include <stddef.h>
#include <stdint.h>

#define TSR_NO_LDP __attribute__((optimize("no-tree-loop-distribute-patterns")))

extern "C" {

TSR_NO_LDP void * memset(void * d, int c, size_t n) {
    uint8_t * p = static_cast<uint8_t *>(d);
    const uint8_t b = uint8_t(c);
    if (((uintptr_t(p) | n) & 3u) == 0u) {
        const uint32_t w = 0x01010101u * b;
        uint32_t * q = reinterpret_cast<uint32_t *>(p);
        for (; n >= 16u; n -= 16u, q += 4) q[0] = q[1] = q[2] = q[3] = w;   // op{} and records: 32 bytes
        for (; n; n -= 4u) *q++ = w;
        return d;
    }
    while (n--) *p++ = b;
    return d;
}

TSR_NO_LDP void * memcpy(void * __restrict d, const void * __restrict s, size_t n) {
    uint8_t * p = static_cast<uint8_t *>(d);
    const uint8_t * q = static_cast<const uint8_t *>(s);
    if (((uintptr_t(p) | uintptr_t(q) | n) & 3u) == 0u) {
        uint32_t * pw = reinterpret_cast<uint32_t *>(p);
        const uint32_t * qw = reinterpret_cast<const uint32_t *>(q);
        for (; n >= 16u; n -= 16u, pw += 4, qw += 4) {
            const uint32_t a = qw[0], b = qw[1], c = qw[2], e = qw[3];
            pw[0] = a; pw[1] = b; pw[2] = c; pw[3] = e;
        }
        for (; n; n -= 4u) *pw++ = *qw++;
        return d;
    }
    while (n--) *p++ = *q++;
    return d;
}

TSR_NO_LDP void * memmove(void * d, const void * s, size_t n) {
    uint8_t * p = static_cast<uint8_t *>(d);
    const uint8_t * q = static_cast<const uint8_t *>(s);
    if (p == q || n == 0u) return d;
    if (p < q || p >= q + n) {
        while (n--) *p++ = *q++;
    } else {
        p += n;
        q += n;
        while (n--) *--p = *--q;
    }
    return d;
}

TSR_NO_LDP int memcmp(const void * a, const void * b, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(a);
    const uint8_t * q = static_cast<const uint8_t *>(b);
    for (; n; --n, ++p, ++q)
        if (*p != *q) return *p < *q ? -1 : 1;
    return 0;
}

}  // extern "C"

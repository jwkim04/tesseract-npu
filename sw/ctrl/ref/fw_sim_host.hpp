// fw_sim_host.hpp — the host side of a firmware image on the simulator, for the kernel tests of
// P2.5 (sw/ctrl/tests/test_fw_kernels*.cpp, test_fw_selftest*.c, test_fw_replay*.c). Plain C (also
// valid C++): the C tests include it like a .h file.
//
// What the transport will do with a kdev (EX §3.3-§3.4, §4.8; docs/design/06-software-verification.md
// "SW" §9.1), reduced to what the tests need:
//   fwh_open        a simulator with an image: cold load through LOAD_ADDR / LOAD_DATA with the
//                   receive CRC and VERIFY (EX §3.3), RV_BOOT, RV_RUN, LOG_CFG, wait for HELLO
//                   (checked: first record, image id = the one tsr-elf stamped, kernel count)
//   fwh_kernel      a kernel's index by name from the image's .tsr_kernels (SW §6.2; entries of 32
//                   or 36 bytes, the name at byte 12, whichever stride gives a valid table)
//   fwh_launch      push one LAUNCH descriptor and run until its answer: every record and log word
//                   on the way is consumed (EX §4.8 consume(); kernel events and log words kept)
//   fwh_ping        PING -> PONG (EX §4.4)
// Every function reports failures on stderr and returns nonzero; nothing aborts.
#ifndef TSR_FW_SIM_HOST_HPP
#define TSR_FW_SIM_HOST_HPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsr_bus.h"
#include "tsr_isa.h"
#include "tsr_sim.h"
#include "tsr_trace.h"

#define FWH_MAX_EVENTS 256u
#define FWH_MAX_LOG    4096u

typedef struct fwh {
    tsr_sim * sim;
    uint8_t * image;          // the ELF bytes
    size_t    image_n;
    uint64_t  image_id;
    uint32_t  lcq_rd, log_rd;
    uint32_t  hello[8];
    // the last launch
    uint32_t  done[8];                       // its answer record (DONE / PONG / TRAP)
    uint32_t  n_ev, ev[FWH_MAX_EVENTS][8];   // events posted while it ran (types 0xC0-0xFF, HELLO)
    uint32_t  n_log, log[FWH_MAX_LOG];       // log words drained while it ran
    uint64_t  cycles;                        // core cycles from the push to the answer
} fwh;

static uint32_t fwh_crc32(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

static uint32_t fwh_rd32(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Section [*off, *off + *size) of the ELF named `name`; 0, or -1.
static int fwh_elf_section(const uint8_t * b, size_t n, const char * name, uint32_t * off, uint32_t * size) {
    if (n < 52 || memcmp(b, "\177ELF", 4) != 0) return -1;
    const uint32_t shoff = fwh_rd32(b + 32);
    const uint32_t shentsize = (uint32_t)b[46] | ((uint32_t)b[47] << 8);
    const uint32_t shnum = (uint32_t)b[48] | ((uint32_t)b[49] << 8);
    const uint32_t shstrndx = (uint32_t)b[50] | ((uint32_t)b[51] << 8);
    if (shoff == 0 || shentsize != 40 || (uint64_t)shoff + 40ull * shnum > n || shstrndx >= shnum) return -1;
    const uint8_t * ss = b + shoff + 40u * shstrndx;
    const uint32_t str_off = fwh_rd32(ss + 16), str_size = fwh_rd32(ss + 20);
    if ((uint64_t)str_off + str_size > n) return -1;
    for (uint32_t i = 0; i < shnum; ++i) {
        const uint8_t * sh = b + shoff + 40u * i;
        const uint32_t nm = fwh_rd32(sh);
        if (nm >= str_size) continue;
        const char * s = (const char *)b + str_off + nm;
        if (strncmp(s, name, str_size - nm) != 0) continue;
        *off = fwh_rd32(sh + 16);
        *size = fwh_rd32(sh + 20);
        return (uint64_t)*off + *size <= n ? 0 : -1;
    }
    return -1;
}

// Index of kernel `name` in .tsr_kernels, or -1.
static int fwh_kernel_in(const uint8_t * b, size_t n, const char * name) {
    uint32_t off = 0, size = 0;
    if (fwh_elf_section(b, n, ".tsr_kernels", &off, &size)) return -1;
    static const uint32_t strides[2] = { 32, 36 };
    for (int s = 0; s < 2; ++s) {
        const uint32_t st = strides[s];
        if (size == 0 || size % st != 0) continue;
        int valid = 1, found = -1;
        for (uint32_t i = 0; i < size / st && valid; ++i) {
            const uint8_t * nm = b + off + st * i + 12;
            const uint32_t cap = st - 12;
            uint32_t L = 0;
            while (L < cap && nm[L] >= 0x20 && nm[L] < 0x7F) ++L;
            if (L == 0 || L == cap) { valid = 0; break; }
            for (uint32_t k = L; k < cap; ++k)
                if (nm[k] != 0) valid = 0;
            if (valid && strlen(name) == L && memcmp(nm, name, L) == 0 && found < 0) found = (int)i;
        }
        if (valid) return found;
    }
    return -1;
}

static int fwh_kernel(const fwh * h, const char * name) { return fwh_kernel_in(h->image, h->image_n, name); }

static uint8_t * fwh_read_file(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t * b = sz > 0 ? (uint8_t *)malloc((size_t)sz) : NULL;
    if (b == NULL || fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

// Drains the LCQ and the log ring (EX §4.8 consume()). The answer of `seq` (if want_seq) goes to
// done; returns 1 when it arrived.
static int fwh_consume(fwh * h, int want_seq, uint32_t seq) {
    int answered = 0;
    const uint32_t wr = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_WR);
    for (uint32_t k = h->lcq_rd; k != wr; ++k) {
        uint32_t r[8];
        for (uint32_t j = 0; j < 8u; ++j) r[j] = tsr_sim_csr_read(h->sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, j));
        const uint32_t type = TSR_GET(r[0], TSR_LCQ_R0_TYPE);
        const int is_answer = type == TSR_LCQ_R0_TYPE_DONE || type == TSR_LCQ_R0_TYPE_PONG ||
                              (type == TSR_LCQ_R0_TYPE_TRAP && want_seq);
        if (want_seq && is_answer && r[1] == seq) {
            memcpy(h->done, r, sizeof(r));
            answered = 1;
        } else if (h->n_ev < FWH_MAX_EVENTS) {
            memcpy(h->ev[h->n_ev++], r, sizeof(r));
        }
    }
    h->lcq_rd = wr;
    tsr_sim_csr_write(h->sim, TSR_CSR_LCQ_RD, wr);
    const uint32_t lwr = tsr_sim_csr_read(h->sim, TSR_CSR_LOG_WR);
    for (uint32_t k = h->log_rd; k != lwr; ++k)
        if (h->n_log < FWH_MAX_LOG) h->log[h->n_log++] = tsr_sim_csr_read(h->sim, TSR_CSR_LOG_RING(k % TSR_LOG_DEPTH));
    h->log_rd = lwr;
    tsr_sim_csr_write(h->sim, TSR_CSR_LOG_RD, lwr);
    return answered;
}

// Advances until the answer of seq (want_seq) or HELLO (!want_seq) or max_cycles. 0 / -1.
static int fwh_run(fwh * h, int want_seq, uint32_t seq, uint64_t max_cycles, const char * what) {
    const uint64_t t0 = tsr_sim_now(h->sim);
    for (;;) {
        const uint32_t ev = tsr_sim_advance(h->sim, 1ull << 22);
        if (want_seq) {
            if (fwh_consume(h, 1, seq)) break;
        } else {
            const uint32_t before = h->n_ev;
            fwh_consume(h, 0, 0);
            if (h->n_ev > before) break;
        }
        const uint32_t rt = tsr_sim_csr_read(h->sim, TSR_CSR_RT_STATE);
        if (TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            fwh_consume(h, want_seq, seq);
            fprintf(stderr, "%s: the firmware trapped (RT_INFO 0x%08x, pc 0x%08x)\n", what,
                    tsr_sim_csr_read(h->sim, TSR_CSR_RT_INFO), tsr_sim_rv_pc(h->sim));
            return -1;
        }
        if (tsr_sim_now(h->sim) - t0 > max_cycles) {
            fprintf(stderr, "%s: no answer within %llu core cycles\n", what, (unsigned long long)max_cycles);
            return -1;
        }
        (void)ev;
    }
    h->cycles = tsr_sim_now(h->sim) - t0;
    return 0;
}

// Loads and boots the image on sim (fresh: RV_RUN 0). 0, or -1.
static int fwh_open(fwh * h, tsr_sim * sim, const char * elf_path) {
    memset(h, 0, sizeof(*h));
    h->sim = sim;
    h->image = fwh_read_file(elf_path, &h->image_n);
    if (h->image == NULL) {
        fprintf(stderr, "fwh_open: cannot read %s\n", elf_path);
        return -1;
    }
    char err[256];
    if (tsr_trace2_image_id(h->image, h->image_n, &h->image_id, err, sizeof(err))) {
        fprintf(stderr, "fwh_open: %s: %s\n", elf_path, err);
        return -1;
    }
    tsr_rv_elf elf;
    if (tsr_rv_elf_read(elf_path, &elf) != 0) return -1;
    int rc = 0;
    for (uint32_t s = 0; s < elf.nseg && rc == 0; ++s) {
        const tsr_rv_elf_seg * g = &elf.seg[s];
        if (g->filesz == 0) continue;
        const uint32_t bytes = (g->filesz + 3u) & ~3u;
        uint8_t * buf = (uint8_t *)calloc(bytes, 1);
        memcpy(buf, g->data, g->filesz);
        const uint32_t crc = fwh_crc32(buf, bytes);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        for (uint32_t i = 0; i < bytes; i += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, fwh_rd32(buf + i));
        const uint32_t rx = tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_COUNT, bytes / 4u);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_CTRL, TSR_LOAD_CTRL_VERIFY);
        for (int i = 0; i < 1000 && (tsr_sim_csr_read(sim, TSR_CSR_LOAD_STATUS) & TSR_LOAD_STATUS_VERIFY_BUSY); ++i)
            tsr_sim_advance(sim, 1000);
        const uint32_t vf = tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC);
        if (rx != crc || vf != crc) {
            fprintf(stderr, "fwh_open: segment 0x%08x: CRC 0x%08x / VERIFY 0x%08x, expected 0x%08x\n", g->paddr, rx, vf, crc);
            rc = -1;
        }
        free(buf);
    }
    const uint32_t entry = elf.entry;
    tsr_rv_elf_free(&elf);
    if (rc) return rc;
    tsr_sim_capture_set_image(sim, h->image, h->image_n);
    tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, 5u));
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, entry - TSR_BOOT_ENTRY_OFF);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, tsr_sim_csr_read(sim, TSR_CSR_CTRL) | TSR_CTRL_RV_RUN);
    if (fwh_run(h, 0, 0, 1ull << 30, "boot")) return -1;
    memcpy(h->hello, h->ev[0], sizeof(h->hello));
    const uint32_t type = TSR_GET(h->hello[0], TSR_LCQ_R0_TYPE);
    const uint64_t id = ((uint64_t)h->hello[3] << 32) | h->hello[2];
    if (type != TSR_LCQ_R0_TYPE_HELLO || id != h->image_id) {
        fprintf(stderr, "fwh_open: first record type 0x%02x id %016llx, expected HELLO with %016llx\n", type,
                (unsigned long long)id, (unsigned long long)h->image_id);
        return -1;
    }
    h->n_ev = 0;
    h->n_log = 0;
    return 0;
}

static void fwh_close(fwh * h) {
    free(h->image);
    h->image = NULL;
}

// One LAUNCH of kernel `index` with args a[7]; its answer in h->done. 0, or -1 (trap, timeout).
static int fwh_launch(fwh * h, uint32_t index, const uint32_t a[7], uint64_t max_cycles) {
    h->n_ev = 0;
    h->n_log = 0;
    memset(h->done, 0, sizeof(h->done));
    const uint32_t seq = tsr_sim_csr_read(h->sim, TSR_CSR_LQ_SEQ);
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(0), TSR_PUT(TSR_DESC_TYPE, TSR_DESC_TYPE_LAUNCH) | TSR_PUT(TSR_DESC_KERNEL, index));
    for (uint32_t i = 0; i < 7u; ++i) tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(1 + i), a[i]);
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_PUSH, 1u);
    return fwh_run(h, 1, seq, max_cycles ? max_cycles : (1ull << 40), "launch");
}

static int fwh_ping(fwh * h, uint32_t arg0) {
    const uint32_t seq = tsr_sim_csr_read(h->sim, TSR_CSR_LQ_SEQ);
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(0), TSR_PUT(TSR_DESC_TYPE, TSR_DESC_TYPE_PING));
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(1), arg0);
    for (uint32_t i = 2; i < 8u; ++i) tsr_sim_csr_write(h->sim, TSR_CSR_LQ_W(i), 0u);
    tsr_sim_csr_write(h->sim, TSR_CSR_LQ_PUSH, 1u);
    if (fwh_run(h, 1, seq, 1ull << 30, "ping")) return -1;
    return TSR_GET(h->done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_PONG && h->done[2] == arg0 ? 0 : -1;
}

static uint32_t fwh_status(const fwh * h) { return TSR_GET(h->done[0], TSR_LCQ_R0_STATUS); }
static int32_t fwh_ret(const fwh * h) { return (int32_t)h->done[2]; }

#endif  // TSR_FW_SIM_HOST_HPP

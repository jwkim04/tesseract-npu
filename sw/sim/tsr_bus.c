// tsr_bus.c — RISC-V bus of the simulator: firmware and test profiles, SBA, HTIF, ELF loading,
// and the RISC-V agent that runs tsr_rv on the simulator's time base (SW §4.2, §4.4; CC §4-§5,
// §7.3, §10.5, §12.3). Interface and rules: tsr_bus.h.
//
// The agent part (tsr_rv_agent_*) uses the simulator-internal interface (tsr_sim_int.h §4-§5) and
// is compiled only into the simulator library (TSR_RV_WITH_SIM); the standalone tsr_rv library
// (unit tests, the riscv-tests runner) has the two profiles without it.

#include "tsr_bus.h"
#include "tsr_isa.h"     // tsr_tables.h: TSR_RV_* map, TSR_CW_*, TSR_CSR_*, PERF slots

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PROF_FW = 0, PROF_TEST = 1 };
enum { R_NONE = 0, R_ITCM, R_DTCM, R_DM, R_CW, R_CORE, R_SPM, R_SPMLB, R_RAM };

struct tsr_bus {
    int        profile;
    tsr_rv_bus iface;
    tsr_rv *   rv;

    // Firmware profile.
    uint8_t *  itcm;  uint32_t itcm_bytes;  int own_itcm;
    uint8_t *  dtcm;  uint32_t dtcm_bytes;  int own_dtcm;
    uint8_t    dm[TSR_RV_DM_SIZE];
    tsr_bus_env env;
    uint32_t   spm_bytes;           // window size actually mapped (<= TSR_RV_SPM_SIZE)
    int        rv_run;
    uint32_t   irq_lines, irq_prev;
    uint8_t    lb[TSR_LINE_BYTES];
    uint32_t   lb_line;
    int        lb_valid;

    // Test profile.
    uint8_t *  ram;   uint32_t ram_base, ram_bytes;
    int        htif;  uint32_t tohost, fromhost;
    int        htif_exited;  uint32_t htif_code;

    tsr_bus_watch_fn watch;
    void *     watch_ctx;
    tsr_bus_stats stats;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void * xcalloc(size_t n, size_t sz) {
    void * p = calloc(n ? n : 1, sz ? sz : 1);
    if (p == NULL) {
        fprintf(stderr, "tsr_bus: out of memory (%zu x %zu bytes)\n", n, sz);
        abort();
    }
    return p;
}
static inline uint32_t rd_le(const uint8_t * p, uint32_t n) {
    uint32_t v = 0;
    for (uint32_t i = 0; i < n; ++i) v |= (uint32_t)p[i] << (8u * i);
    return v;
}
static inline void wr_le(uint8_t * p, uint32_t n, uint32_t v) {
    for (uint32_t i = 0; i < n; ++i) p[i] = (uint8_t)(v >> (8u * i));
}
static inline uint64_t rd64_le(const uint8_t * p) { return (uint64_t)rd_le(p, 4) | ((uint64_t)rd_le(p + 4, 4) << 32); }
static inline void wr64_le(uint8_t * p, uint64_t v) { wr_le(p, 4, (uint32_t)v); wr_le(p + 4, 4, (uint32_t)(v >> 32)); }

static int decode(const tsr_bus * b, uint32_t addr, uint32_t * off) {
    uint32_t o;
    if (b->profile == PROF_TEST) {
        o = addr - b->ram_base;
        if (o < b->ram_bytes) { *off = o; return R_RAM; }
        return R_NONE;
    }
    if ((o = addr - TSR_RV_ITCM_BASE) < b->itcm_bytes)  { *off = o; return R_ITCM; }
    if ((o = addr - TSR_RV_DTCM_BASE) < b->dtcm_bytes)  { *off = o; return R_DTCM; }
    if ((o = addr - TSR_RV_DM_BASE) < TSR_RV_DM_SIZE)   { *off = o; return R_DM; }
    if ((o = addr - TSR_RV_CTL_BASE) < TSR_RV_CTL_SIZE) { *off = o; return R_CW; }
    if ((o = addr - TSR_RV_CORE_BASE) < TSR_RV_CORE_SIZE) { *off = o; return R_CORE; }
    if ((o = addr - TSR_RV_SPM_BASE) < b->spm_bytes)    { *off = o; return R_SPM; }
    if ((o = addr - TSR_RV_SPM_LB_BASE) < b->spm_bytes) { *off = o; return R_SPMLB; }
    return R_NONE;
}

// Memory of a plain region (TCM, DM memory, RAM), or NULL.
static uint8_t * region_mem(tsr_bus * b, int r) {
    switch (r) {
    case R_ITCM: return b->itcm;
    case R_DTCM: return b->dtcm;
    case R_DM:   return b->dm;
    case R_RAM:  return b->ram;
    default:     return NULL;
    }
}

static inline int word_only(uint32_t addr, uint32_t n) { return n == 4u && (addr & 3u) == 0u; }

// Core-window offsets the bus answers with 0 itself (CC §5.3): LOAD_DATA and the DMI window.
static inline int core_masked(uint32_t off) {
    return off == TSR_CSR_LOAD_DATA || (off >= 0xC00u && off < 0xE00u);
}

static int bus_yield(tsr_bus * b) { return b->env.yield && b->env.yield(b->env.ctx); }

static uint32_t bus_rv_run(tsr_bus * b) {
    return b->env.rv_run ? (uint32_t)(b->env.rv_run(b->env.ctx) != 0) : (uint32_t)(b->rv_run != 0);
}

// ---------------------------------------------------------------------------
// HTIF (test profile)
// ---------------------------------------------------------------------------
static void htif_command(tsr_bus * b, uint32_t lo, uint32_t hi) {
    uint32_t dev = hi >> 24, cmd = (hi >> 16) & 0xFFu;
    uint64_t payload = ((uint64_t)(hi & 0xFFFFu) << 32) | lo;
    uint8_t * th = b->ram + (b->tohost - b->ram_base);
    if (dev == 0u && cmd == 0u && (payload & 1u)) {
        b->htif_exited = 1;
        b->htif_code = (uint32_t)(payload >> 1);
        if (b->rv) tsr_rv_request_stop(b->rv);
        return;                                   // tohost keeps the exit value (as Spike leaves it)
    }
    if (dev == 0u && cmd == 0u) {
        // Syscall block: 8 x u64 {number, a0, a1, a2, ...} at `payload`; result into word 0.
        uint32_t a = (uint32_t)payload;
        uint8_t * m = tsr_bus_mem(b, a, 64);
        if (m != NULL) {
            uint64_t num = rd64_le(m), a0 = rd64_le(m + 8), a1 = rd64_le(m + 16), a2 = rd64_le(m + 24);
            int64_t ret = -38;                    // ENOSYS
            if (num == 93u) {                     // SYS_exit
                b->htif_exited = 1;
                b->htif_code = (uint32_t)a0;
                if (b->rv) tsr_rv_request_stop(b->rv);
                ret = 0;
            } else if (num == 64u && (a0 == 1u || a0 == 2u)) {   // SYS_write
                uint8_t * p = (a1 <= 0xFFFFFFFFu && a2 <= 0xFFFFFFFFu) ? tsr_bus_mem(b, (uint32_t)a1, (uint32_t)a2) : NULL;
                if (p != NULL) {
                    fwrite(p, 1, (size_t)a2, a0 == 1u ? stdout : stderr);
                    fflush(a0 == 1u ? stdout : stderr);
                    ret = (int64_t)a2;
                } else {
                    ret = -14;                    // EFAULT
                }
            }
            wr64_le(m, (uint64_t)ret);
        }
    } else if (dev == 1u && cmd == 1u) {          // console putchar
        fputc((int)(payload & 0xFFu), stdout);
        fflush(stdout);
    }
    wr64_le(th, 0);
    if (b->fromhost >= b->ram_base && b->fromhost - b->ram_base <= b->ram_bytes - 8u)
        wr64_le(b->ram + (b->fromhost - b->ram_base), dev == 1u ? (((uint64_t)dev << 56) | ((uint64_t)cmd << 48)) : 1u);
}

static void htif_store(tsr_bus * b, uint32_t addr, uint32_t n) {
    uint32_t t = b->tohost;
    if ((uint64_t)addr + n <= t || (uint64_t)addr >= (uint64_t)t + 8u) return;
    uint8_t * th = b->ram + (t - b->ram_base);
    uint32_t lo = rd_le(th, 4), hi = rd_le(th + 4, 4);
    int low_touched = addr < t + 4u;
    if (low_touched && hi == 0u && lo != 0u) htif_command(b, lo, 0);
    else if (!low_touched && hi != 0u) htif_command(b, lo, hi);
}

void tsr_bus_set_htif(tsr_bus * b, uint32_t tohost, uint32_t fromhost) {
    if (b->profile != PROF_TEST || tohost - b->ram_base > b->ram_bytes - 8u || (tohost & 7u)) {
        b->htif = 0;
        return;
    }
    b->htif = 1; b->tohost = tohost; b->fromhost = fromhost;
}

int tsr_bus_htif_exited(const tsr_bus * b, uint32_t * code) {
    if (code) *code = b->htif_code;
    return b->htif_exited;
}

// ---------------------------------------------------------------------------
// Interpreter interface
// ---------------------------------------------------------------------------
static int if_fetch(void * ctx, uint32_t addr, uint32_t dbg, uint16_t * parcel) {
    tsr_bus * b = (tsr_bus *)ctx;
    uint32_t off;
    int r = decode(b, addr, &off);
    (void)dbg;
    if (r != R_ITCM && r != R_DM && r != R_RAM) return TSR_RV_BUS_FAULT;
    *parcel = (uint16_t)rd_le(region_mem(b, r) + off, 2);
    return 0;
}

static void lb_fill(tsr_bus * b, uint32_t line) {
    memcpy(b->lb, b->env.spm + (size_t)line * TSR_LINE_BYTES, TSR_LINE_BYTES);
    b->lb_line = line;
    b->lb_valid = 1;
    ++b->stats.lb_fills;
    if (b->env.spm_access) b->env.spm_access(b->env.ctx, line * TSR_LINE_BYTES, TSR_LINE_BYTES, 0);
}

static int bus_load(tsr_bus * b, uint32_t addr, uint32_t n, uint32_t * val) {
    uint32_t off, v = 0;
    int r = decode(b, addr, &off);
    switch (r) {
    case R_ITCM: case R_DTCM: case R_RAM:
        *val = rd_le(region_mem(b, r) + off, n);
        return TSR_BUS_LAT_TCM;
    case R_DM:
        if (!word_only(addr, n)) return TSR_RV_BUS_FAULT;
        *val = rd_le(b->dm + off, 4);
        return TSR_BUS_LAT_TCM;
    case R_CW:
        if (!word_only(addr, n)) return TSR_RV_BUS_FAULT;
        if (bus_yield(b)) return TSR_RV_BUS_YIELD;
        b->lb_valid = 0;
        ++b->stats.cw_reads;
        if (b->env.cw && b->env.cw(b->env.ctx, 0, off, &v) != TSR_BUS_ENV_OK) return TSR_RV_BUS_FAULT;
        *val = v;
        return TSR_BUS_LAT_CW;
    case R_CORE:
        if (!word_only(addr, n)) return TSR_RV_BUS_FAULT;
        if (bus_yield(b)) return TSR_RV_BUS_YIELD;
        b->lb_valid = 0;
        ++b->stats.core_reads;
        if (!core_masked(off) && b->env.core_read &&
            b->env.core_read(b->env.ctx, off, &v) != TSR_BUS_ENV_OK) return TSR_RV_BUS_FAULT;
        *val = core_masked(off) ? 0u : v;
        return TSR_BUS_LAT_CORE;
    case R_SPM:
        if (bus_yield(b)) return TSR_RV_BUS_YIELD;
        *val = rd_le(b->env.spm + off, n);
        ++b->stats.spm_reads;
        if (b->env.spm_access) b->env.spm_access(b->env.ctx, off, n, 0);
        return TSR_BUS_LAT_SPM;
    case R_SPMLB: {
        uint32_t line = off / TSR_LINE_BYTES;
        if (b->lb_valid && b->lb_line == line) {
            ++b->stats.lb_hits;
            *val = rd_le(b->lb + off % TSR_LINE_BYTES, n);
            return TSR_BUS_LAT_LBHIT;
        }
        if (bus_yield(b)) return TSR_RV_BUS_YIELD;
        lb_fill(b, line);
        *val = rd_le(b->lb + off % TSR_LINE_BYTES, n);
        return TSR_BUS_LAT_SPM;
    }
    default:
        return TSR_RV_BUS_FAULT;
    }
}

static int bus_store(tsr_bus * b, uint32_t addr, uint32_t n, uint32_t val) {
    uint32_t off;
    int r = decode(b, addr, &off);
    switch (r) {
    case R_ITCM: case R_DTCM:
        wr_le(region_mem(b, r) + off, n, val);
        return TSR_BUS_LAT_TCM;
    case R_RAM:
        wr_le(b->ram + off, n, val);
        if (b->htif) htif_store(b, addr, n);
        return TSR_BUS_LAT_TCM;
    case R_DM:
        if (!word_only(addr, n)) return TSR_RV_BUS_FAULT;
        wr_le(b->dm + off, 4, val);
        return TSR_BUS_LAT_TCM;
    case R_CW: {
        if (!word_only(addr, n)) return TSR_RV_BUS_FAULT;
        if (bus_yield(b)) return TSR_RV_BUS_YIELD;
        b->lb_valid = 0;
        if (b->env.cw == NULL) { ++b->stats.cw_writes; return TSR_BUS_LAT_CW; }
        uint32_t v = val;
        int e = b->env.cw(b->env.ctx, 1, off, &v);
        if (e == TSR_BUS_ENV_WAIT) { ++b->stats.cw_waits; return TSR_RV_BUS_WAIT; }
        if (e != TSR_BUS_ENV_OK) return TSR_RV_BUS_FAULT;
        ++b->stats.cw_writes;
        return TSR_BUS_LAT_CW;
    }
    case R_SPM: case R_SPMLB:
        if (bus_yield(b)) return TSR_RV_BUS_YIELD;
        wr_le(b->env.spm + off, n, val);
        b->lb_valid = 0;
        ++b->stats.spm_writes;
        if (b->env.spm_access) b->env.spm_access(b->env.ctx, off, n, 1);
        return TSR_BUS_LAT_SPM;
    default:   // R_CORE (read only), unmapped
        return TSR_RV_BUS_FAULT;
    }
}

static int if_load(void * ctx, uint32_t addr, uint32_t n, uint32_t dbg, uint32_t * val) {
    tsr_bus * b = (tsr_bus *)ctx;
    (void)dbg;
    int r = bus_load(b, addr, n, val);
    if (b->watch && r != TSR_RV_BUS_WAIT && r != TSR_RV_BUS_YIELD)
        b->watch(b->watch_ctx, addr, n, 0, r < 0 ? 0u : *val, r < 0);
    return r;
}

static int if_store(void * ctx, uint32_t addr, uint32_t n, uint32_t dbg, uint32_t val) {
    tsr_bus * b = (tsr_bus *)ctx;
    (void)dbg;
    int r = bus_store(b, addr, n, val);
    if (b->watch && r != TSR_RV_BUS_WAIT && r != TSR_RV_BUS_YIELD)
        b->watch(b->watch_ctx, addr, n, 1, val, r < 0);
    return r;
}

static uint32_t if_irq(void * ctx) {
    tsr_bus * b = (tsr_bus *)ctx;
    uint32_t lines = (b->env.irq ? b->env.irq(b->env.ctx) : b->irq_lines) & TSR_RV_IRQ_LINES;
    if (lines & ~b->irq_prev) b->lb_valid = 0;      // CC §10.5: rising edge of an interrupt line
    b->irq_prev = lines;
    return lines;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
static void set_iface(tsr_bus * b) {
    b->iface.ctx = b;
    b->iface.fetch = if_fetch;
    b->iface.load = if_load;
    b->iface.store = if_store;
    b->iface.irq = if_irq;
    if (b->profile == PROF_TEST) {
        b->iface.fetch_mem = b->ram; b->iface.fetch_base = b->ram_base; b->iface.fetch_size = b->ram_bytes;
    } else {
        b->iface.fetch_mem = b->itcm; b->iface.fetch_base = TSR_RV_ITCM_BASE; b->iface.fetch_size = b->itcm_bytes;
    }
}

tsr_bus * tsr_bus_create_fw(const tsr_bus_fw_cfg * cfg, const tsr_bus_env * env) {
    tsr_bus_fw_cfg c = { NULL, 0, NULL, 0 };
    if (cfg) c = *cfg;
    if (c.itcm_bytes == 0u) c.itcm_bytes = TSR_RV_ITCM_SIZE;
    if (c.dtcm_bytes == 0u) c.dtcm_bytes = TSR_RV_DTCM_SIZE;
    if ((c.itcm_bytes & 3u) || c.itcm_bytes > TSR_RV_ITCM_LIMIT - TSR_RV_ITCM_BASE ||
        (c.dtcm_bytes & 3u) || c.dtcm_bytes > TSR_RV_DTCM_LIMIT - TSR_RV_DTCM_BASE) {
        fprintf(stderr, "tsr_bus: bad TCM sizes (ITCM %u, DTCM %u bytes)\n", c.itcm_bytes, c.dtcm_bytes);
        return NULL;
    }
    tsr_bus * b = (tsr_bus *)xcalloc(1, sizeof(*b));
    b->profile = PROF_FW;
    b->itcm_bytes = c.itcm_bytes;
    b->dtcm_bytes = c.dtcm_bytes;
    b->itcm = c.itcm ? c.itcm : (uint8_t *)xcalloc(c.itcm_bytes, 1);
    b->dtcm = c.dtcm ? c.dtcm : (uint8_t *)xcalloc(c.dtcm_bytes, 1);
    b->own_itcm = c.itcm == NULL;
    b->own_dtcm = c.dtcm == NULL;
    if (env) b->env = *env;
    b->spm_bytes = 0;
    if (b->env.spm) {
        b->spm_bytes = b->env.spm_bytes & ~(TSR_LINE_BYTES - 1u);
        if (b->spm_bytes > TSR_RV_SPM_SIZE) b->spm_bytes = TSR_RV_SPM_SIZE;
    }
    set_iface(b);
    return b;
}

tsr_bus * tsr_bus_create_test(uint32_t ram_base, uint32_t ram_bytes) {
    if (ram_bytes == 0u || (ram_bytes & 7u) || (uint64_t)ram_base + ram_bytes > 0x100000000ull) {
        fprintf(stderr, "tsr_bus: bad test RAM [0x%08x, +0x%x)\n", ram_base, ram_bytes);
        return NULL;
    }
    tsr_bus * b = (tsr_bus *)xcalloc(1, sizeof(*b));
    b->profile = PROF_TEST;
    b->ram = (uint8_t *)xcalloc(ram_bytes, 1);
    b->ram_base = ram_base;
    b->ram_bytes = ram_bytes;
    set_iface(b);
    return b;
}

void tsr_bus_destroy(tsr_bus * b) {
    if (b == NULL) return;
    if (b->own_itcm) free(b->itcm);
    if (b->own_dtcm) free(b->dtcm);
    free(b->ram);
    free(b);
}

const tsr_rv_bus * tsr_bus_iface(tsr_bus * b) { return &b->iface; }
void tsr_bus_attach(tsr_bus * b, tsr_rv * rv) { b->rv = rv; }
void tsr_bus_set_irq(tsr_bus * b, uint32_t lines) { b->irq_lines = lines; }
void tsr_bus_set_watch(tsr_bus * b, tsr_bus_watch_fn fn, void * ctx) { b->watch = fn; b->watch_ctx = ctx; }
void tsr_bus_lb_invalidate(tsr_bus * b) { b->lb_valid = 0; }
void tsr_bus_set_rv_run(tsr_bus * b, int run) { b->rv_run = run != 0; }
void tsr_bus_get_stats(const tsr_bus * b, tsr_bus_stats * out) { *out = b->stats; }

uint8_t * tsr_bus_mem(tsr_bus * b, uint32_t addr, uint32_t n) {
    uint32_t off;
    int r = decode(b, addr, &off);
    uint8_t * m = region_mem(b, r);
    if (m == NULL) return NULL;
    uint32_t size = r == R_ITCM ? b->itcm_bytes : r == R_DTCM ? b->dtcm_bytes : r == R_DM ? TSR_RV_DM_SIZE : b->ram_bytes;
    if (n > size - off) return NULL;
    return m + off;
}

// ---------------------------------------------------------------------------
// SBA (CC §4.3, DB §3.6)
// ---------------------------------------------------------------------------
int tsr_bus_sba_read(tsr_bus * b, uint32_t addr, uint32_t size, uint32_t * val) {
    uint32_t off, v = 0;
    if ((size != 1u && size != 2u && size != 4u) || (addr & (size - 1u))) return -1;
    int r = decode(b, addr, &off);
    switch (r) {
    case R_ITCM: case R_DTCM: case R_RAM:
        *val = rd_le(region_mem(b, r) + off, size);
        return 0;
    case R_CW:
        if (size != 4u) return -1;
        if (b->env.cw && b->env.cw(b->env.ctx, 0, off, &v) != TSR_BUS_ENV_OK) return -1;
        *val = v;
        return 0;
    case R_CORE:
        if (size != 4u) return -1;
        if (!core_masked(off) && b->env.core_read && b->env.core_read(b->env.ctx, off, &v) != TSR_BUS_ENV_OK) return -1;
        *val = core_masked(off) ? 0u : v;
        return 0;
    case R_SPM: case R_SPMLB:     // SPM bytes directly; the hart's line buffer is not touched
        *val = rd_le(b->env.spm + off, size);
        if (b->env.spm_access) b->env.spm_access(b->env.ctx, off, size, 0);
        return 0;
    default:                      // DM range, unmapped
        return -1;
    }
}

int tsr_bus_sba_write(tsr_bus * b, uint32_t addr, uint32_t size, uint32_t val) {
    uint32_t off;
    if ((size != 1u && size != 2u && size != 4u) || (addr & (size - 1u))) return -1;
    int r = decode(b, addr, &off);
    switch (r) {
    case R_ITCM: case R_DTCM:
        if (!bus_rv_run(b)) return -1;            // the host load port owns the TCM write port
        wr_le(region_mem(b, r) + off, size, val);
        return 0;
    case R_RAM:
        wr_le(b->ram + off, size, val);
        return 0;
    case R_SPM: case R_SPMLB:
        wr_le(b->env.spm + off, size, val);
        b->lb_valid = 0;
        if (b->env.spm_access) b->env.spm_access(b->env.ctx, off, size, 1);
        return 0;
    default:                      // control and core windows (D15, V2-CC-33), DM range, unmapped
        return -1;
    }
}

// ---------------------------------------------------------------------------
// ELF32
// ---------------------------------------------------------------------------
static uint32_t e16(const uint8_t * p) { return rd_le(p, 2); }
static uint32_t e32(const uint8_t * p) { return rd_le(p, 4); }

int tsr_rv_elf_read(const char * path, tsr_rv_elf * elf) {
    memset(elf, 0, sizeof(*elf));
    FILE * f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "tsr_rv_elf: cannot open %s\n", path); return -1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < 52 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); fprintf(stderr, "tsr_rv_elf: %s: too small\n", path); return -1; }
    elf->buf = (uint8_t *)xcalloc((size_t)sz, 1);
    elf->size = (size_t)sz;
    size_t got = fread(elf->buf, 1, (size_t)sz, f);
    fclose(f);
    const uint8_t * h = elf->buf;
    if (got != (size_t)sz || memcmp(h, "\177ELF", 4) != 0 || h[4] != 1u || h[5] != 1u || e16(h + 18) != 243u) {
        fprintf(stderr, "tsr_rv_elf: %s: not an ELF32 little-endian RISC-V file\n", path);
        tsr_rv_elf_free(elf);
        return -1;
    }
    elf->entry = e32(h + 24);
    uint32_t phoff = e32(h + 28), shoff = e32(h + 32);
    uint32_t phentsize = e16(h + 42), phnum = e16(h + 44), shentsize = e16(h + 46), shnum = e16(h + 48);
    if ((uint64_t)phoff + (uint64_t)phnum * phentsize > elf->size || (phnum && phentsize < 32u)) {
        fprintf(stderr, "tsr_rv_elf: %s: bad program headers\n", path);
        tsr_rv_elf_free(elf);
        return -1;
    }
    elf->seg = (tsr_rv_elf_seg *)xcalloc(phnum ? phnum : 1, sizeof(tsr_rv_elf_seg));
    elf->lo = 0xFFFFFFFFu; elf->hi = 0;
    for (uint32_t i = 0; i < phnum; ++i) {
        const uint8_t * ph = h + phoff + (size_t)i * phentsize;
        if (e32(ph) != 1u) continue;                          // PT_LOAD
        uint32_t offset = e32(ph + 4), paddr = e32(ph + 12), filesz = e32(ph + 16), memsz = e32(ph + 20);
        if (memsz == 0u) continue;
        if ((uint64_t)offset + filesz > elf->size || filesz > memsz || (uint64_t)paddr + memsz > 0x100000000ull) {
            fprintf(stderr, "tsr_rv_elf: %s: bad PT_LOAD segment %u\n", path, i);
            tsr_rv_elf_free(elf);
            return -1;
        }
        tsr_rv_elf_seg * s = &elf->seg[elf->nseg++];
        s->paddr = paddr; s->filesz = filesz; s->memsz = memsz; s->data = h + offset;
        if (paddr < elf->lo) elf->lo = paddr;
        if (paddr + memsz > elf->hi) elf->hi = paddr + memsz;
    }
    if (elf->nseg == 0u) elf->lo = elf->hi = 0;
    // .symtab and its string table (optional).
    if (shoff && shnum && shentsize >= 40u && (uint64_t)shoff + (uint64_t)shnum * shentsize <= elf->size) {
        for (uint32_t i = 0; i < shnum; ++i) {
            const uint8_t * sh = h + shoff + (size_t)i * shentsize;
            if (e32(sh + 4) != 2u) continue;                  // SHT_SYMTAB
            uint32_t off = e32(sh + 16), size = e32(sh + 20), link = e32(sh + 24), entsz = e32(sh + 36);
            if (link >= shnum || entsz < 16u || (uint64_t)off + size > elf->size) break;
            const uint8_t * st = h + shoff + (size_t)link * shentsize;
            uint32_t soff = e32(st + 16), ssize = e32(st + 20);
            if ((uint64_t)soff + ssize > elf->size) break;
            elf->symoff = off; elf->symnum = size / entsz; elf->symentsz = entsz;
            elf->stroff = soff; elf->strsize = ssize;
            break;
        }
    }
    return 0;
}

void tsr_rv_elf_free(tsr_rv_elf * elf) {
    free(elf->seg);
    free(elf->buf);
    memset(elf, 0, sizeof(*elf));
}

int tsr_rv_elf_symbol(const tsr_rv_elf * elf, const char * name, uint32_t * value) {
    size_t len = strlen(name);
    for (uint32_t i = 0; i < elf->symnum; ++i) {
        const uint8_t * s = elf->buf + elf->symoff + (size_t)i * elf->symentsz;
        uint32_t nm = e32(s);
        if (nm >= elf->strsize || elf->strsize - nm <= len) continue;
        const char * str = (const char *)elf->buf + elf->stroff + nm;
        if (memcmp(str, name, len) == 0 && str[len] == '\0' && e16(s + 14) != 0u) {   // defined
            if (value) *value = e32(s + 4);
            return 0;
        }
    }
    return -1;
}

int tsr_rv_elf_load(tsr_bus * b, const tsr_rv_elf * elf) {
    for (uint32_t i = 0; i < elf->nseg; ++i) {
        const tsr_rv_elf_seg * s = &elf->seg[i];
        uint8_t * m = tsr_bus_mem(b, s->paddr, s->memsz);
        if (m == NULL) {
            fprintf(stderr, "tsr_rv_elf: segment [0x%08x, +0x%x) is outside the bus memories\n", s->paddr, s->memsz);
            return -1;
        }
        memcpy(m, s->data, s->filesz);
        memset(m + s->filesz, 0, s->memsz - s->filesz);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// RISC-V agent on the simulator's time base (tsr_sim_int.h §4-§5)
// ---------------------------------------------------------------------------
#ifdef TSR_RV_WITH_SIM
#include "tsr_sim_int.h"

// Slice bounds: a slice never runs past the next core event; without one it stops after 1024 RV
// cycles, so the hart is never far ahead of MTIME when the host acts between tsr_sim_advance
// calls (host actions are seen at the next slice, as if made up to 1024 RV cycles later).
#define SLICE_MAX_INSNS  4096u
#define SLICE_MAX_CYCLES 1024u

struct tsr_rv_agent {
    tsr_sim *  sim;
    tsr_bus *  bus;
    tsr_rv *   rv;
    double     ratio;               // F_RV / F_CORE
    uint64_t   gen;                 // tsr_sim_rv_boot_gen the hart was reset for
    int        started;
    uint64_t   t0, c0;              // anchor: core time t0 <-> RV cycle count c0
    uint64_t   slice_c0;            // RV cycles at the start of the running slice
    int        in_step;
    uint32_t   state;               // STATUS bits last reported
    uint32_t   perf_x;              // PERF extension base X = 5 + 2C
};

// RV cycles elapsed by core time t (t >= t0).
static uint64_t ag_rv_at(const tsr_rv_agent * a, uint64_t t) {
    return a->c0 + (uint64_t)((double)(t - a->t0) * a->ratio);
}
// First core time at which the RV cycle count c has been reached.
static uint64_t ag_time_of(const tsr_rv_agent * a, uint64_t c) {
    if (c <= a->c0) return a->t0;
    uint64_t t = a->t0 + (uint64_t)((double)(c - a->c0) / a->ratio);
    while (ag_rv_at(a, t) < c) ++t;
    while (t > a->t0 && ag_rv_at(a, t - 1u) >= c) --t;
    return t;
}

// sleep_ok = 0: the hart sleeps, but the core time it reached the WFI lies ahead (reported later).
static void ag_report_state(tsr_rv_agent * a, int sleep_ok) {
    uint32_t s = (tsr_rv_debug_mode(a->rv) ? TSR_STATUS_RV_HALTED : 0u) |
                 (sleep_ok && tsr_rv_sleeping(a->rv) ? TSR_STATUS_RV_SLEEP : 0u);
    if (s != a->state) {
        a->state = s;
        tsr_sim_rv_set_state(a->sim, s);
    }
}

// An access left the clk_rv domain: end the slice after this instruction, so that the next
// instruction sees core events the access may have caused (a new MTIMECMP, a push, ...).
static void ag_external(tsr_rv_agent * a) {
    if (a->in_step) tsr_rv_request_stop(a->rv);
}

static int ag_cw(void * ctx, int is_write, uint32_t off, uint32_t * val) {
    tsr_rv_agent * a = (tsr_rv_agent *)ctx;
    int r = tsr_sim_cw_access(a->sim, is_write, off, val);
    ag_external(a);
    if (r == TSR_CW_OK) return TSR_BUS_ENV_OK;
    if (r == TSR_CW_WAIT && is_write) {
        if (tsr_rv_waiting(a->rv) && tsr_rv_wait_cycles(a->rv) >= TSR_ISSUE_WAIT_MAX) {
            tsr_sim_cw_push_timeout(a->sim);       // CC §7.3: store access fault + PUSH_TIMEOUT
            return TSR_BUS_ENV_FAULT;
        }
        return TSR_BUS_ENV_WAIT;
    }
    return TSR_BUS_ENV_FAULT;
}

static int ag_core_read(void * ctx, uint32_t off, uint32_t * val) {
    tsr_rv_agent * a = (tsr_rv_agent *)ctx;
    *val = tsr_sim_corewin_read(a->sim, off);
    ag_external(a);
    return TSR_BUS_ENV_OK;
}

static void ag_spm_access(void * ctx, uint32_t off, uint32_t n, int is_write) {
    tsr_rv_agent * a = (tsr_rv_agent *)ctx;
    (void)off; (void)n; (void)is_write;
    tsr_sim_perf_add(a->sim, a->perf_x + TSR_PERF_XOFF_BRIDGE_ACC, 1);
    tsr_sim_perf_add(a->sim, a->perf_x + TSR_PERF_XOFF_BRIDGE_WAIT, TSR_BUS_LAT_SPM - TSR_BUS_LAT_TCM);
    ag_external(a);
}

static uint32_t ag_irq(void * ctx) { return tsr_sim_rv_irq(((tsr_rv_agent *)ctx)->sim); }
static int ag_rv_run(void * ctx) { return tsr_sim_rv_run(((tsr_rv_agent *)ctx)->sim) != 0u; }

// Accesses leaving clk_rv happen only as the first transaction of a slice (tsr_bus.h).
static int ag_yield(void * ctx) {
    tsr_rv_agent * a = (tsr_rv_agent *)ctx;
    return a->in_step && tsr_rv_cycles(a->rv) != a->slice_c0;
}

static uint64_t ag_step(void * ctx, tsr_sim * sim) {
    tsr_rv_agent * a = (tsr_rv_agent *)ctx;
    uint64_t now = tsr_sim_now(sim);
    uint64_t gen = tsr_sim_rv_boot_gen(sim);
    if (!a->started || gen != a->gen) {
        tsr_rv_reset(a->rv, tsr_sim_rv_boot(sim));
        tsr_bus_lb_invalidate(a->bus);
        a->gen = gen;
        a->started = 1;
        a->ratio = tsr_sim_rv_ratio(sim);
        if (!(a->ratio > 0.0)) a->ratio = 166.7 / 276.3;
        a->t0 = now;
        a->c0 = tsr_rv_cycles(a->rv);
    }
    // Credit the time that passed since the hart last ran (asleep, parked, waiting, rounding).
    uint64_t c = tsr_rv_cycles(a->rv), target = ag_rv_at(a, now);
    if (target > c) {
        if (tsr_rv_waiting(a->rv)) tsr_sim_perf_add(sim, a->perf_x + TSR_PERF_XOFF_PUSH_STALL, target - c);
        tsr_rv_idle(a->rv, target - c);
        c = target;
    }
    // Slice: instructions that start before the next core event. While this step runs, the core's
    // tsr_sim_next_event() leaves the agent itself out (tsr_q.c) and counts MTIMECMP, so the
    // slice ends before the next completion, dispatch or timer edge.
    uint64_t next = tsr_sim_next_event(sim), max_cycles = 1;
    if (next == UINT64_MAX) {
        max_cycles = SLICE_MAX_CYCLES;
    } else if (next > now) {
        uint64_t last = ag_rv_at(a, next - 1u);
        max_cycles = last >= c ? last - c + 1u : 1u;
    }
    a->slice_c0 = c;
    a->in_step = 1;
    uint32_t r = tsr_rv_run(a->rv, SLICE_MAX_INSNS, max_cycles);
    a->in_step = 0;
    if (r == TSR_RV_STOP_WFI) {
        // The slice ran ahead of MTIME: the hart reached the WFI at its own cycle count. STATUS
        // RV_SLEEP and PERF RV_WFI start at that core time, so the agent wakes there to report it
        // (a sleeping hart with no pending interrupt stops again at once, without using cycles).
        const uint64_t t = ag_time_of(a, tsr_rv_cycles(a->rv));
        if (t > now) {
            ag_report_state(a, 0);
            return t - now;
        }
    }
    ag_report_state(a, 1);
    switch (r) {
    case TSR_RV_STOP_WFI:
    case TSR_RV_STOP_HALTED:
        return TSR_SIM_AGENT_WAIT;
    case TSR_RV_STOP_WAIT: {
        // A push waits for a credit: credits change only at core events, so retry at the next
        // one, or at the CC §7.3 deadline.
        uint64_t w = tsr_rv_wait_cycles(a->rv);
        uint64_t left = w < TSR_ISSUE_WAIT_MAX ? TSR_ISSUE_WAIT_MAX - w : 0u;
        uint64_t t = ag_time_of(a, tsr_rv_cycles(a->rv) + left);
        uint64_t nx = tsr_sim_next_event(sim);
        if (nx < t) t = nx;
        return t > now ? t - now : 1u;
    }
    default: {
        uint64_t t = ag_time_of(a, tsr_rv_cycles(a->rv));
        return t > now ? t - now : 1u;
    }
    }
}

tsr_rv_agent * tsr_rv_agent_create(tsr_sim * sim) {
    tsr_rv_agent * a = (tsr_rv_agent *)xcalloc(1, sizeof(*a));
    a->sim = sim;
    a->perf_x = TSR_PERF_X(tsr_sim_num_ch(sim));
    tsr_bus_fw_cfg cfg;
    cfg.itcm = tsr_sim_itcm(sim, &cfg.itcm_bytes);
    cfg.dtcm = tsr_sim_dtcm(sim, &cfg.dtcm_bytes);
    tsr_bus_env env;
    memset(&env, 0, sizeof(env));
    env.ctx = a;
    env.cw = ag_cw;
    env.core_read = ag_core_read;
    env.spm = tsr_sim_spm(sim);
    env.spm_bytes = tsr_sim_config(sim)->spm_bytes;
    env.spm_access = ag_spm_access;
    env.irq = ag_irq;
    env.yield = ag_yield;
    env.rv_run = ag_rv_run;
    a->bus = tsr_bus_create_fw(&cfg, &env);
    if (a->bus == NULL) { free(a); return NULL; }
    a->rv = tsr_rv_create(tsr_bus_iface(a->bus));
    tsr_bus_attach(a->bus, a->rv);
    tsr_sim_set_rv_agent(sim, ag_step, a);
    return a;
}

void tsr_rv_agent_destroy(tsr_rv_agent * a) {
    if (a == NULL) return;
    tsr_sim_set_rv_agent(a->sim, NULL, NULL);
    tsr_rv_destroy(a->rv);
    tsr_bus_destroy(a->bus);
    free(a);
}

tsr_rv * tsr_rv_agent_rv(tsr_rv_agent * a) { return a->rv; }
tsr_bus * tsr_rv_agent_bus(tsr_rv_agent * a) { return a->bus; }
#endif   // TSR_RV_WITH_SIM

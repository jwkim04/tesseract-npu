// test_rv_run.c — tsr_rv_run: runs an RV32 ELF on tsr_rv's test bus profile (SW §4.2, §10.1).
//
// The runner of the RV32 conformance harness (sw/sim/rvtests.cmake: rv_isa, rv_random), with the
// contract of sw/sim/tests/rvtests/rvcommon.py:
//
//   tsr_rv_run [--log-commits FILE] [--signature FILE] [--max-insns N] [--ram BYTES] [-v] ELF
//
//   * RAM at 0x8000_0000 covering the ELF's PT_LOAD segments (rounded up to 1 MiB, plus 1 MiB:
//     Spike's RAM is larger, so a program that strays a little past its image still matches),
//     reset PC = e_entry, x1..x31 = 0, mtvec = 0x0000_0001 (Ibex reset with boot address 0),
//     HTIF tohost / fromhost by symbol (tsr_bus.h);
//   * exit status: the HTIF exit code (0 = pass: the program stored 1 to tohost; (n << 1) | 1
//     gives n, truncated to 8 bits by the OS, and 1 if that would read 0); 2 if the instruction
//     budget ran out (default 50,000,000), 3 for usage, ELF or load errors;
//   * --log-commits FILE: one line per retired instruction in Spike's --log-commits format
//     ("core   0: 3 0x80000000 (0x00000297) x5  0x80000000", loads " mem 0x<addr>", stores
//     " mem 0x<addr> 0x<data>", explicit Zicsr writes " c<num>_<name> 0x<value>"); trapping
//     instructions print no line; the log ends with the instruction that ended the run;
//   * --signature FILE: the words from begin_signature to end_signature, "%08x" per line.

#include "tsr_bus.h"
#include "tsr_rv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RAM_BASE 0x80000000u
#define MiB      (1024u * 1024u)

static const char * csr_name(uint32_t c) {
    switch (c) {
    case 0x300: return "mstatus";   case 0x301: return "misa";      case 0x304: return "mie";
    case 0x305: return "mtvec";     case 0x306: return "mcounteren"; case 0x30A: return "menvcfg";
    case 0x31A: return "menvcfgh";  case 0x320: return "mcountinhibit";
    case 0x340: return "mscratch";  case 0x341: return "mepc";      case 0x342: return "mcause";
    case 0x343: return "mtval";     case 0x344: return "mip";       case 0x7A0: return "tselect";
    case 0x7A1: return "tdata1";    case 0x7A2: return "tdata2";    case 0x7A3: return "tdata3";
    case 0x7A8: return "mcontext";  case 0x7AA: return "scontext";  case 0x7B0: return "dcsr";
    case 0x7B1: return "dpc";       case 0x7B2: return "dscratch0"; case 0x7B3: return "dscratch1";
    case 0x7C0: return "cpuctrlsts"; case 0x7C1: return "secureseed";
    case 0xB00: return "mcycle";    case 0xB02: return "minstret";  case 0xB80: return "mcycleh";
    case 0xB82: return "minstreth";
    default:
        if (c >= 0x323u && c <= 0x33Fu) return "mhpmevent";
        if ((c >= 0xB03u && c <= 0xB1Fu) || (c >= 0xB83u && c <= 0xB9Fu)) return "mhpmcounter";
        return "csr";
    }
}

typedef struct run_ctx {
    FILE *   log;
    uint64_t retired;
} run_ctx;

static int on_retire(void * p, const tsr_rv_retire * r) {
    run_ctx * c = (run_ctx *)p;
    if (r->trap) return 0;
    ++c->retired;
    if (c->log == NULL) return 0;
    FILE * f = c->log;
    fprintf(f, "core%4d: %u 0x%08x ", 0, (unsigned)r->priv, (unsigned)r->pc);
    if (r->len == 2) fprintf(f, "(0x%04x)", (unsigned)(r->insn & 0xFFFFu));
    else fprintf(f, "(0x%08x)", (unsigned)r->insn);
    if (r->rd) fprintf(f, " x%-2u 0x%08x", (unsigned)r->rd, (unsigned)r->rd_val);
    if (r->csr_written) fprintf(f, " c%u_%s 0x%08x", (unsigned)r->csr, csr_name(r->csr), (unsigned)r->csr_val);
    if (r->mem_size && !r->mem_write) fprintf(f, " mem 0x%08x", (unsigned)r->mem_addr);
    if (r->mem_size && r->mem_write)
        fprintf(f, " mem 0x%08x 0x%0*x", (unsigned)r->mem_addr, 2 * (int)r->mem_size,
                (unsigned)(r->mem_size == 4u ? r->mem_val : r->mem_val & ((1u << (8u * r->mem_size)) - 1u)));
    fputc('\n', f);
    return 0;
}

static int usage(void) {
    fprintf(stderr, "usage: tsr_rv_run [--log-commits FILE] [--signature FILE] [--max-insns N] "
                    "[--ram BYTES] [-v] ELF\n");
    return 3;
}

int main(int argc, char ** argv) {
    const char * log_path = NULL, * sig_path = NULL, * elf_path = NULL;
    uint64_t max_insns = 50000000ull;
    uint64_t ram_bytes = 0;
    int verbose = 0;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (strcmp(a, "--log-commits") == 0 && i + 1 < argc) log_path = argv[++i];
        else if (strcmp(a, "--signature") == 0 && i + 1 < argc) sig_path = argv[++i];
        else if (strcmp(a, "--max-insns") == 0 && i + 1 < argc) max_insns = strtoull(argv[++i], NULL, 0);
        else if (strcmp(a, "--ram") == 0 && i + 1 < argc) ram_bytes = strtoull(argv[++i], NULL, 0);
        else if (strcmp(a, "-v") == 0) verbose = 1;
        else if (a[0] == '-' || elf_path) return usage();
        else elf_path = a;
    }
    if (elf_path == NULL) return usage();

    tsr_rv_elf elf;
    if (tsr_rv_elf_read(elf_path, &elf)) return 3;
    if (elf.nseg == 0u || elf.lo < RAM_BASE) {
        fprintf(stderr, "tsr_rv_run: %s: PT_LOAD segments must lie at or above 0x%08x\n", elf_path, RAM_BASE);
        tsr_rv_elf_free(&elf);
        return 3;
    }
    if (ram_bytes == 0u) {
        uint64_t need = (uint64_t)elf.hi - RAM_BASE;
        ram_bytes = ((need + MiB - 1u) / MiB) * MiB + MiB;
    }
    if (ram_bytes > 0x40000000ull) ram_bytes = 0x40000000ull;
    tsr_bus * bus = tsr_bus_create_test(RAM_BASE, (uint32_t)ram_bytes);
    if (bus == NULL || tsr_rv_elf_load(bus, &elf)) {
        tsr_rv_elf_free(&elf);
        tsr_bus_destroy(bus);
        return 3;
    }
    uint32_t tohost = 0, fromhost = 0;
    if (tsr_rv_elf_symbol(&elf, "tohost", &tohost)) {
        fprintf(stderr, "tsr_rv_run: %s: no 'tohost' symbol\n", elf_path);
        tsr_rv_elf_free(&elf);
        tsr_bus_destroy(bus);
        return 3;
    }
    if (tsr_rv_elf_symbol(&elf, "fromhost", &fromhost)) fromhost = tohost + 64u;
    tsr_bus_set_htif(bus, tohost, fromhost);

    tsr_rv * rv = tsr_rv_create(tsr_bus_iface(bus));
    tsr_bus_attach(bus, rv);
    tsr_rv_reset(rv, 0);
    tsr_rv_set_pc(rv, elf.entry);

    run_ctx ctx = { NULL, 0 };
    if (log_path) {
        ctx.log = fopen(log_path, "w");
        if (ctx.log == NULL) {
            fprintf(stderr, "tsr_rv_run: cannot write %s\n", log_path);
            return 3;
        }
    }
    tsr_rv_set_retire_hook(rv, on_retire, &ctx);

    // Run in chunks; the budget counts retired instructions (traps do not count, as for Spike's
    // instruction count; a trap loop still ends because each chunk also bounds the steps).
    uint32_t code = 0;
    int exited = 0, stuck = 0;
    uint64_t steps = 0;
    while (!exited) {
        if (ctx.retired >= max_insns || steps >= 4u * max_insns + 1000u) break;
        uint32_t why = tsr_rv_run(rv, 4096, UINT64_MAX);
        steps += 4096;
        exited = tsr_bus_htif_exited(bus, &code);
        if (!exited && (why == TSR_RV_STOP_WFI || why == TSR_RV_STOP_HALTED)) { stuck = 1; break; }
    }
    if (ctx.log) fclose(ctx.log);

    int status;
    if (exited) {
        status = (int)(code & 0xFFu);
        if (status == 0 && code != 0u) status = 1;
        if (verbose || code != 0u)
            fprintf(stderr, "tsr_rv_run: %s: exit code %u after %llu instructions\n", elf_path, code,
                    (unsigned long long)ctx.retired);
    } else {
        uint32_t mcause = 0, mepc = 0;
        (void)tsr_rv_csr_read(rv, 0x342, &mcause);
        (void)tsr_rv_csr_read(rv, 0x341, &mepc);
        status = 2;
        fprintf(stderr, "tsr_rv_run: %s: %s after %llu instructions (pc 0x%08x, mcause 0x%x, mepc 0x%08x)\n",
                elf_path, stuck ? "hart asleep with no wake-up source" : "instruction budget exhausted",
                (unsigned long long)ctx.retired, tsr_rv_pc(rv), mcause, mepc);
    }

    if (sig_path) {
        uint32_t b = 0, e = 0;
        if (tsr_rv_elf_symbol(&elf, "begin_signature", &b) || tsr_rv_elf_symbol(&elf, "end_signature", &e) || e < b) {
            fprintf(stderr, "tsr_rv_run: %s: no begin_signature / end_signature\n", elf_path);
            if (status == 0) status = 3;
        } else {
            FILE * f = fopen(sig_path, "w");
            uint8_t * m = tsr_bus_mem(bus, b, e - b);
            if (f == NULL || (m == NULL && e > b)) {
                fprintf(stderr, "tsr_rv_run: cannot write the signature %s\n", sig_path);
                if (status == 0) status = 3;
            } else {
                for (uint32_t a = 0; a + 4u <= e - b; a += 4u)
                    fprintf(f, "%08x\n", (unsigned)(m[a] | (m[a + 1] << 8) | (m[a + 2] << 16) | ((uint32_t)m[a + 3] << 24)));
            }
            if (f) fclose(f);
        }
    }
    tsr_rv_destroy(rv);
    tsr_bus_destroy(bus);
    tsr_rv_elf_free(&elf);
    return status;
}

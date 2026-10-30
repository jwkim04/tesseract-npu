// test_sim_dm.c — the debug-module model (P1.12; ctest sim_dm, sim_dm_fw, sim_dm_ubsan).
//
// The DM (sw/sim/tsr_dm.c, include/tsr_dm.h) is driven through its DMI API (tsr_dm_dmi_read /
// _write: what the DBG model's window at host CSR 0xC00 + 4r forwards, and what the sim
// transport's DMI helpers call, DB §9.2), with the DMI sequences of docs/design/05-debug.md (DB)
// §3.5 and §3.6 verbatim. The hart runs hand-encoded RV32 programs written into ITCM while
// RV_RUN = 0 (no toolchain needed), booted through the host CSR port. Checks:
//   - the DMI profile (DB §9.1): reset values, dmactive gating, hartinfo, abstractcs, sbcs, the
//     registers outside the profile read 0, hartsel discovery and the nonexistent hart;
//   - halt / resume / ndmreset sequences, dmstatus all* bits, havereset / ackhavereset, "reset
//     halt", STATUS.RV_HALTED / RV_RUNNING / RV_RESET_DONE;
//   - Access Register for every GPR, CSRs, dpc (resume goes there), postexec, the cmderr paths
//     (not supported, exception, halt / resume, busy) and the W 0x16 = 0x700 clear;
//   - the program buffer: fence.i (DB §3.5), computation, data0 as memory, an exception, a
//     program longer than the synchronous budget (busy on the time base), one that never parks
//     (ended by ndmreset), dret;
//   - a halt request waking a hart asleep in WFI;
//   - SBA: n-word reads with autoincrement and the last-word rule, n-word writes, 16-bit and
//     8-bit accesses, every CC §4.3 refusal with sberror (DM range, control- and core-window
//     writes, TCM writes while RV_RUN = 0, unmapped), alignment / size errors, sberror blocking
//     and W1C, the SBA access events;
//   - single step over 20 instructions (a branch, loads, stores, compressed instructions)
//     against a free-running reference hart, at F_RV / F_CORE 1.0 and 0.603, and under a random
//     schedule (P1.8);
//   - both hardware triggers, ebreak with dcsr.ebreakm (and the trap without it), software
//     breakpoints written through SBA (c.ebreak with the 16-bit sequence, ebreak) + fence.i;
//   - a halt requested while a push waits for a credit: the hart halts after the push;
//   - halt-on-boot through the external request (tsr_dm_set_ext_haltreq, also from the RESET
//     event): halted before the first instruction, dpc = RV_BOOT + 0x80;
//   - a timer interrupt that becomes pending during a halt is taken after the resume;
//   - the DM's state across SOFT_RESET;
//   - with the DBG model built in: the DMI window through host CSR 0xC00 + 4r, gated by
//     DBG_CFG.DMI_EN (reads 0, writes dropped while 0); without it the window reads 0.
// With "--fw <fw_dm.elf>" (ctest sim_dm_fw) it runs instead a GDB-style session on a C program
// (tests/dm/fw_dm.c); exit 77 when the ELF does not exist (no RISC-V compiler at configure time).
#include "test_sim_util.h"

#include "tsr_bus.h"
#include "tsr_dm.h"
#include "tsr_rv.h"
#include "tsr_sim_int.h"

// ---------------------------------------------------------------------------
// Encoders (the subset of test_rv_isa.c this test uses; rv_unit_enc checks those against the
// GNU assembler)
// ---------------------------------------------------------------------------
static uint32_t eR(uint32_t f7, uint32_t rs2, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t op) {
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t eI(int32_t imm, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t op) {
    return (((uint32_t)imm & 0xFFFu) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t eS(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 5) & 0x7Fu) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((u & 0x1Fu) << 7) | 0x23u;
}
static uint32_t eB(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 12) & 1u) << 31) | (((u >> 5) & 0x3Fu) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) |
           (((u >> 1) & 0xFu) << 8) | (((u >> 11) & 1u) << 7) | 0x63u;
}
static uint32_t eJ(int32_t imm, uint32_t rd) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 20) & 1u) << 31) | (((u >> 1) & 0x3FFu) << 21) | (((u >> 11) & 1u) << 20) |
           (((u >> 12) & 0xFFu) << 12) | (rd << 7) | 0x6Fu;
}
#define LUI(rd, imm)        (((uint32_t)(imm) & 0xFFFFF000u) | ((rd) << 7) | 0x37u)
#define JAL(rd, off)        eJ(off, rd)
#define BNE(a, b, off)      eB(off, b, a, 1)
#define LW(rd, rs1, imm)    eI(imm, rs1, 2, rd, 0x03u)
#define SW(rs2, rs1, imm)   eS(imm, rs2, rs1, 2)
#define ADDI(rd, rs1, imm)  eI(imm, rs1, 0, rd, 0x13u)
#define XOR(rd, a, b)       eR(0x00u, b, a, 4, rd, 0x33u)
#define ADD(rd, a, b)       eR(0x00u, b, a, 0, rd, 0x33u)
#define FENCE_I             0x0000100Fu
#define DRET                0x7B200073u
#define WFI                 0x10500073u
#define EBREAK              0x00100073u
#define MRET                0x30200073u
#define NOP                 0x00000013u
#define CSRRW(rd, c, rs1)   eI((int32_t)(c), rs1, 1, rd, 0x73u)
#define CSRRSI(rd, c, u)    eI((int32_t)(c), u, 6, rd, 0x73u)
static uint32_t c_ci(uint32_t f3, uint32_t rd, int32_t imm, uint32_t op) {
    uint32_t u = (uint32_t)imm;
    return (f3 << 13) | (((u >> 5) & 1u) << 12) | (rd << 7) | ((u & 0x1Fu) << 2) | op;
}
#define C_ADDI(rd, imm)     c_ci(0, rd, imm, 1)
#define C_NOP               0x0001u
#define C_EBREAK            0x9002u

// DMI addresses and the DB §3.5 / §3.6 constants.
#define D_DATA0  TSR_DMI_DATA0
#define D_DATA1  TSR_DMI_DATA1
#define D_CTRL   TSR_DMI_DMCONTROL
#define D_STAT   TSR_DMI_DMSTATUS
#define D_HINFO  TSR_DMI_HARTINFO
#define D_ACS    TSR_DMI_ABSTRACTCS
#define D_CMD    TSR_DMI_COMMAND
#define D_PB(i)  (TSR_DMI_PROGBUF0 + (i))
#define D_SBCS   TSR_DMI_SBCS
#define D_SBADDR TSR_DMI_SBADDRESS0
#define D_SBDATA TSR_DMI_SBDATA0
#define ALLHALTED    TSR_DM_DMSTATUS_ALLHALTED
#define ALLRUNNING   TSR_DM_DMSTATUS_ALLRUNNING
#define ALLUNAVAIL   TSR_DM_DMSTATUS_ALLUNAVAIL
#define ALLRESUMEACK TSR_DM_DMSTATUS_ALLRESUMEACK
#define ALLHAVERESET TSR_DM_DMSTATUS_ALLHAVERESET
#define CSR_MSTATUS 0x300u
#define CSR_MIE     0x304u
#define CSR_MTVEC   0x305u
#define CSR_MSCRATCH 0x340u
#define CSR_MEPC    0x341u
#define CSR_MCAUSE  0x342u
#define CSR_TSELECT 0x7A0u
#define CSR_TDATA1  0x7A1u
#define CSR_TDATA2  0x7A2u
#define CSR_DCSR    0x7B0u
#define CSR_DPC     0x7B1u
#define CSR_MINSTRET 0xB02u

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
static tsr_sim * SIM;
static tsr_dm *  DM;
static uint8_t * ITCM;
static uint8_t * DTCM;
static uint32_t  PCW;

static void put16(uint32_t a, uint32_t h) { ITCM[a] = (uint8_t)h; ITCM[a + 1] = (uint8_t)(h >> 8); }
static void put32(uint32_t a, uint32_t w) { put16(a, w & 0xFFFFu); put16(a + 2, w >> 16); }
static uint32_t get32(const uint8_t * p) { return get_u32_le(p); }
static void org(uint32_t a) { PCW = a; }
static void i32(uint32_t w) { put32(PCW, w); PCW += 4; }
static void i16(uint32_t h) { put16(PCW, h); PCW += 2; }
static void li(uint32_t rd, uint32_t v) {                    // lui + addi (the low part sign-extends)
    const uint32_t lo = v & 0xFFFu, hi = (v + 0x800u) & 0xFFFFF000u;
    if (hi) i32(LUI(rd, hi));
    i32(ADDI(rd, hi ? rd : 0u, (int32_t)(lo ^ 0x800u) - 0x800));
}

static void sim_new_sched(double ratio, const char * sched) {
    tsr_config cfg = TSR_CONFIG_U200;
    cfg.num_ch = 1;
    cfg.ch_bytes = 1ull << 28;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.rv_ratio = ratio;
    o.sched = sched;
    SIM = tsr_sim_create_opts(&cfg, &o);
    if (SIM == NULL) { fprintf(stderr, "tsr_sim_create_opts failed\n"); exit(1); }
    DM = tsr_sim_dm(SIM);
    if (DM == NULL) { fprintf(stderr, "no DM model in the simulator (TSR_SIM_HAVE_DM)\n"); exit(1); }
    ITCM = tsr_sim_itcm(SIM, NULL);
    DTCM = tsr_sim_dtcm(SIM, NULL);
}
static void sim_new(double ratio) { sim_new_sched(ratio, "fixed"); }
static void sim_free(void) {
    tsr_sim_destroy(SIM);
    SIM = NULL;
    DM = NULL;
}
static tsr_rv * hart(void) { return tsr_rv_agent_rv(tsr_sim_rv_agent(SIM)); }

static uint32_t R(uint32_t a) { return tsr_dm_dmi_read(DM, a); }
static void W(uint32_t a, uint32_t v) { tsr_dm_dmi_write(DM, a, v); }
static uint32_t csr_r(uint32_t off) { return tsr_sim_csr_read(SIM, off); }
static void csr_w(uint32_t off, uint32_t v) { tsr_sim_csr_write(SIM, off, v); }

static void boot(uint32_t rv_boot) {
    csr_w(TSR_CSR_RV_BOOT, rv_boot);
    csr_w(TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
}
// Advances simulated time by n core cycles (tsr_sim_advance returns early at host events).
static void run_for(uint64_t n) {
    const uint64_t t = tsr_sim_now(SIM) + n;
    while (tsr_sim_now(SIM) < t) tsr_sim_advance(SIM, t - tsr_sim_now(SIM));
}
// Polls dmstatus for `bit`, advancing time between reads (a debugger's poll loop on sim).
static int poll(uint32_t bit, uint64_t max_cycles) {
    for (uint64_t c = 0; c <= max_cycles; c += 50) {
        if (R(D_STAT) & bit) return 1;
        tsr_sim_advance(SIM, 50);
    }
    return (R(D_STAT) & bit) != 0;
}

// STATUS.RV_HALTED is the time base's copy: it rises when MTIME reaches the core time the hart
// entered debug mode, which may lie ahead after a slice (tsr_bus.c); dmstatus is immediate.
static int status_halted(void) {
    for (int k = 0; k < 400; ++k) {
        if (csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) return 1;
        tsr_sim_advance(SIM, 50);
    }
    return 0;
}

// DB §3.5 sequences.
static int dm_halt(void) {                     // W 0x10 = 0x8000_0001, poll allhalted, W 0x10 = 1
    W(D_CTRL, 0x80000001u);
    const int ok = poll(ALLHALTED, 200000);
    W(D_CTRL, 0x00000001u);
    return ok;
}
static int dm_resume(void) {                   // W 0x10 = 0x4000_0001, poll allresumeack, W 0x10 = 1
    W(D_CTRL, 0x40000001u);
    const int ok = poll(ALLRESUMEACK, 1000);
    W(D_CTRL, 0x00000001u);
    return ok;
}
static uint32_t acs_wait(void) {               // R 0x16 until busy = 0; returns cmderr
    for (int k = 0; k < 10000; ++k) {
        const uint32_t v = R(D_ACS);
        if (!(v & TSR_DM_ABSTRACTCS_BUSY)) return (v & TSR_DM_ABSTRACTCS_CMDERR) >> TSR_DM_ABSTRACTCS_CMDERR_LSB;
        tsr_sim_advance(SIM, 100);
    }
    return 0xFFu;
}
static void cmderr_clear(void) { W(D_ACS, 0x00000700u); }
static uint32_t reg_get(uint32_t regno, uint32_t * err) {   // W 0x17 = 0x0022_0000 + regno; R 0x16; R 0x04
    W(D_CMD, 0x00220000u + regno);
    const uint32_t e = acs_wait();
    if (err) *err = e;
    return R(D_DATA0);
}
static uint32_t reg_set(uint32_t regno, uint32_t v) {       // W 0x04; W 0x17 = 0x0023_0000 + regno; R 0x16
    W(D_DATA0, v);
    W(D_CMD, 0x00230000u + regno);
    return acs_wait();
}
static uint32_t xr(unsigned i) {
    uint32_t e = 0;
    const uint32_t v = reg_get(TSR_DM_REGNO_GPR(i), &e);
    CHECK(e == 0u, "read x%u: cmderr %u", i, e);
    return v;
}
static void xw(unsigned i, uint32_t v) { CHECK(reg_set(TSR_DM_REGNO_GPR(i), v) == 0u, "write x%u", i); }
static uint32_t cr(uint32_t c) {
    uint32_t e = 0;
    const uint32_t v = reg_get(c, &e);
    CHECK(e == 0u, "read CSR 0x%03x: cmderr %u", c, e);
    return v;
}
static void cw(uint32_t c, uint32_t v) { CHECK(reg_set(c, v) == 0u, "write CSR 0x%03x", c); }
static uint32_t dcause(void) { return (cr(CSR_DCSR) >> 6) & 7u; }
static uint32_t progbuf_run(const uint32_t * p, unsigned n) {   // W 0x20.. ; W 0x17 = 0x0004_0000; R 0x16
    for (unsigned i = 0; i < n; ++i) W(D_PB(i), p[i]);
    W(D_CMD, 0x00040000u);
    return acs_wait();
}
static uint32_t fence_i(void) {                // progbuf0 = 0x0000_100F, progbuf1 = 0x0010_0073
    const uint32_t p[2] = { 0x0000100Fu, 0x00100073u };
    return progbuf_run(p, 2);
}

// DB §3.6 sequences.
static void sba_read_n(uint32_t a, uint32_t n, uint32_t * out) {
    W(D_SBCS, 0x0055F000u);
    W(D_SBADDR, a);
    for (uint32_t i = 0; i + 1u < n; ++i) out[i] = R(D_SBDATA);
    W(D_SBCS, 0x00457000u);
    out[n - 1u] = R(D_SBDATA);
}
static void sba_write_n(uint32_t a, uint32_t n, const uint32_t * in) {
    W(D_SBCS, 0x00457000u);
    W(D_SBADDR, a);
    for (uint32_t i = 0; i < n; ++i) W(D_SBDATA, in[i]);
}
static void sba_write16(uint32_t a, uint32_t v) {
    W(D_SBCS, 0x00427000u);
    W(D_SBADDR, a);
    W(D_SBDATA, v);
}
static uint32_t sba_err(void) { return R(D_SBCS) & (TSR_DM_SBCS_SBBUSYERROR | TSR_DM_SBCS_SBERROR); }
static uint32_t sba_rd32(uint32_t a) { uint32_t v = 0; sba_read_n(a, 1, &v); return v; }
static void sba_clear(void) { W(D_SBCS, 0x00407000u); }     // W1C sbbusyerror, sberror; sbaccess 0 (8-bit)

// SBA access events (tsr_sim_int.h TSR_ACC_SBA_*).
static tsr_sim_acc ACC[32];
static int NACC;
static void acc_cb(void * ctx, tsr_sim * sim, const tsr_sim_acc * a) {
    (void)ctx; (void)sim;
    if ((a->kind == TSR_ACC_SBA_RD || a->kind == TSR_ACC_SBA_WR) && NACC < 32) ACC[NACC++] = *a;
}

// ---------------------------------------------------------------------------
// Program A: a loop of mixed instructions (stores and loads to DTCM, compressed instructions, a
// branch taken in two of three iterations from the second on, a jump). x5 counts iterations, x8
// every third one (the second iteration first).
// ---------------------------------------------------------------------------
#define A_LOOP 0x8Cu
#define A_LW   0x94u
#define A_CADD 0x98u
#define A_ADD8 0xA4u
#define A_XOR  0xA8u
static void prog_a(void) {
    org(0x80);
    i32(LUI(10, TSR_RV_DTCM_BASE));                       // 0x80 x10 = DTCM
    i32(ADDI(5, 0, 0));                                   // 0x84
    i32(ADDI(6, 0, 2));                                   // 0x88 (2: the second iteration falls through)
    i32(ADDI(5, 5, 1));                                   // 0x8C loop
    i32(SW(5, 10, 0));                                    // 0x90
    i32(LW(7, 10, 0));                                    // 0x94
    i16(C_ADDI(6, -1));                                   // 0x98
    i16(C_NOP);                                           // 0x9A
    i32(BNE(6, 0, 12));                                   // 0x9C -> 0xA8
    i32(ADDI(6, 0, 3));                                   // 0xA0
    i32(ADDI(8, 8, 1));                                   // 0xA4
    i32(XOR(9, 9, 5));                                    // 0xA8
    i32(JAL(0, (int32_t)A_LOOP - 0xAC));                  // 0xAC
    // Vector 0 (exceptions): park in a loop (x31 = 0xEE marks it).
    org(0x00); i32(ADDI(31, 0, 0xEE)); i32(JAL(0, 0));
}

// ---------------------------------------------------------------------------
// The profile's registers and reset values
// ---------------------------------------------------------------------------
static void test_profile(void) {
    sim_new(1.0);
    // Before dmactive: only dmactive is writable.
    CHECK(R(D_CTRL) == 0u, "dmcontrol after reset 0x%08x", R(D_CTRL));
    W(D_CTRL, 0x80000002u);                               // haltreq + ndmreset without dmactive
    CHECK(R(D_CTRL) == 0u, "writes ignored while dmactive = 0");
    W(D_DATA0, 0x1234u); W(D_SBADDR, 0x40u);
    CHECK(R(D_DATA0) == 0u && R(D_SBADDR) == 0u, "data / sbaddress ignored while inactive");
    W(D_CTRL, 0x80000003u);                               // activates only
    CHECK(R(D_CTRL) == 0x00000001u, "dmactive only (0x%08x)", R(D_CTRL));
    // dmstatus: version 2, authenticated; the hart is in reset (RV_RUN = 0): unavailable; havereset.
    uint32_t st = R(D_STAT);
    CHECK((st & 0xFu) == 2u && (st & TSR_DM_DMSTATUS_AUTHENTICATED) && !(st & TSR_DM_DMSTATUS_IMPEBREAK), "dmstatus 0x%08x", st);
    CHECK((st & 0x000FFF00u) == (TSR_DM_DMSTATUS_ANYUNAVAIL | ALLUNAVAIL | TSR_DM_DMSTATUS_ANYHAVERESET | ALLHAVERESET),
          "in reset: unavailable, havereset (0x%08x)", st);
    CHECK(R(D_HINFO) == 0x00212380u, "hartinfo 0x%08x", R(D_HINFO));
    CHECK(R(D_ACS) == 0x08000002u, "abstractcs 0x%08x", R(D_ACS));
    CHECK(R(D_SBCS) == 0x20040407u, "sbcs reset 0x%08x", R(D_SBCS));
    CHECK(R(D_CMD) == 0u && R(D_SBDATA) == 0u && R(TSR_DMI_DATA1) == 0u, "command reads 0");
    static const uint32_t absent[] = { 0x00, 0x01, 0x03, 0x06, 0x0F, 0x13, 0x14, 0x15, 0x18, 0x19, 0x1A, 0x1D,
                                       0x28, 0x30, 0x32, 0x34, 0x3A, 0x3B, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x7F };
    for (unsigned i = 0; i < sizeof(absent) / sizeof(absent[0]); ++i) {
        W(absent[i], 0xFFFFFFFFu);
        CHECK(R(absent[i]) == 0u, "DM register 0x%02x outside the profile reads 0", absent[i]);
    }
    CHECK(R(D_HINFO) == 0x00212380u && R(D_ACS) == 0x08000002u, "read-only registers unchanged");
    // Data and program buffer: R/W, mapped at riscv-dbg's addresses in the DM memory.
    for (unsigned i = 0; i < 8; ++i) W(D_PB(i), 0x11110000u + i);
    W(D_DATA0, 0xD0D0D0D0u); W(D_DATA1, 0xD1D1D1D1u);
    uint8_t * dmm = tsr_bus_mem(tsr_rv_agent_bus(tsr_sim_rv_agent(SIM)), TSR_RV_DM_BASE, TSR_RV_DM_SIZE);
    CHECK(dmm != NULL && get32(dmm + 0x360) == 0x11110000u && get32(dmm + 0x37C) == 0x11110007u &&
          get32(dmm + 0x380) == 0xD0D0D0D0u && get32(dmm + 0x384) == 0xD1D1D1D1u, "progbuf / data in the DM memory");
    CHECK(R(D_PB(7)) == 0x11110007u && R(D_DATA1) == 0xD1D1D1D1u, "progbuf / data read back");
    // hartsel: one bit (riscv-dbg HartSelLen 1); hart 1 does not exist.
    W(D_CTRL, 0x03FFFFC1u);                               // hartsello / hartselhi all ones
    CHECK(R(D_CTRL) == 0x00010001u, "hartsel discovery: hartsello = 1, hartselhi = 0 (0x%08x)", R(D_CTRL));
    st = R(D_STAT);
    CHECK((st & 0x000FFF00u) == (TSR_DM_DMSTATUS_ANYNONEXISTENT | TSR_DM_DMSTATUS_ALLNONEXISTENT), "hart 1 nonexistent (0x%08x)", st);
    W(D_CMD, 0x00221005u);
    CHECK(acs_wait() == TSR_DM_CMDERR_HALTRESUME, "command on a nonexistent hart: cmderr 4");
    cmderr_clear();
    W(D_CTRL, 0x80010001u);                               // haltreq for hart 1 does not reach hart 0
    W(D_CTRL, 0x00000001u);
    CHECK(R(D_CTRL) == 0x00000001u, "back to hart 0, no halt request");
    // dmactive 1 -> 0 resets the DM's registers.
    W(D_SBCS, 0x00150000u); W(D_SBADDR, 0x40u);
    W(D_CTRL, 0x00000000u);
    W(D_CTRL, 0x00000001u);
    CHECK(R(D_DATA0) == 0u && R(D_PB(7)) == 0u && R(D_SBCS) == 0x20040407u && R(D_SBADDR) == 0u,
          "dmactive 0 resets data, progbuf and SBA state");
    CHECK(tsr_sim_dm(SIM) == DM && !tsr_dm_rv_halted(DM), "tsr_sim_dm, not halted in reset");
    sim_free();
}

// ---------------------------------------------------------------------------
// Halt / resume (DB §3.5) and register access
// ---------------------------------------------------------------------------
static void test_halt_resume_regs(void) {
    sim_new(1.0);
    prog_a();
    W(D_CTRL, 0x00000001u);
    boot(0);
    run_for(2000);
    uint32_t st = R(D_STAT);
    CHECK((st & (ALLRUNNING | TSR_DM_DMSTATUS_ANYRUNNING)) == (ALLRUNNING | TSR_DM_DMSTATUS_ANYRUNNING) &&
          !(st & (ALLHALTED | ALLUNAVAIL)) && (st & ALLHAVERESET), "running (0x%08x)", st);
    CHECK(!(csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED), "STATUS.RV_HALTED = 0");
    W(D_CMD, 0x00221005u);
    CHECK(acs_wait() == TSR_DM_CMDERR_HALTRESUME, "abstract command while running: cmderr 4");
    cmderr_clear();
    CHECK(R(D_ACS) == 0x08000002u, "W 0x16 = 0x700 clears cmderr");
    // Halt.
    CHECK(dm_halt(), "halt: allhalted");
    st = R(D_STAT);
    CHECK((st & 0x000FFF00u & ~(TSR_DM_DMSTATUS_ANYHAVERESET | ALLHAVERESET)) == (TSR_DM_DMSTATUS_ANYHALTED | ALLHALTED),
          "halted only (0x%08x)", st);
    CHECK(tsr_dm_rv_halted(DM) && status_halted() && (csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_RUNNING),
          "tsr_dm_rv_halted, STATUS.RV_HALTED");
    CHECK(dcause() == TSR_RV_DBG_CAUSE_HALTREQ && (cr(CSR_DCSR) >> 28) == 4u && (cr(CSR_DCSR) & 3u) == 3u, "dcsr cause haltreq, xdebugver 4, prv M");
    const uint32_t dpc = cr(CSR_DPC);
    CHECK(dpc >= A_LOOP && dpc <= 0xACu, "dpc 0x%x inside the loop", dpc);
    const uint64_t ret0 = tsr_rv_retired(hart());
    run_for(5000);
    CHECK(tsr_rv_retired(hart()) == ret0 && (R(D_STAT) & ALLHALTED), "stays halted, nothing retires");
    // Every GPR: write distinct values, read them back; x0 stays 0.
    const uint32_t x5 = xr(5);
    CHECK(x5 >= 1u && (get32(DTCM) == x5 || get32(DTCM) + 1u == x5), "x5 %u / DTCM counter %u consistent", x5, get32(DTCM));
    uint32_t keep[32];
    for (unsigned i = 0; i < 32; ++i) keep[i] = xr(i);
    for (unsigned i = 0; i < 32; ++i) xw(i, 0xA5000000u + i * 0x01010101u);
    for (unsigned i = 0; i < 32; ++i)
        CHECK(xr(i) == (i ? 0xA5000000u + i * 0x01010101u : 0u), "x%u read back 0x%08x", i, xr(i));
    for (unsigned i = 1; i < 32; ++i) xw(i, keep[i]);
    CHECK(tsr_rv_reg(hart(), 5) == x5 && tsr_rv_reg(hart(), 10) == TSR_RV_DTCM_BASE, "abstract writes act on the hart");
    // CSRs: read / write with WARL; the debug CSRs; read-only and absent CSRs give cmderr 3.
    cw(CSR_MSCRATCH, 0xCAFEF00Du);
    CHECK(cr(CSR_MSCRATCH) == 0xCAFEF00Du, "mscratch");
    cw(CSR_MTVEC, 0x12345677u);
    CHECK(cr(CSR_MTVEC) == 0x12345601u, "mtvec WARL 0x%08x", cr(CSR_MTVEC));
    cw(CSR_MTVEC, 0x00000001u);
    CHECK(cr(0x301u) == 0x40101104u && cr(0xF12u) == 22u, "misa, marchid");
    cw(0x7B2u, 0x11u); cw(0x7B3u, 0x22u);
    CHECK(cr(0x7B2u) == 0x11u && cr(0x7B3u) == 0x22u, "dscratch0-1");
    cw(CSR_DCSR, 0xFFFFFFFFu);
    CHECK(cr(CSR_DCSR) == (0x40009007u | (TSR_RV_DBG_CAUSE_HALTREQ << 6)), "dcsr: ebreakm, ebreaku, step, prv; stepie / stopcount 0 (0x%08x)",
          cr(CSR_DCSR));
    cw(CSR_DCSR, 0x40000003u);
    uint32_t e = 0;
    (void)reg_get(0x3A0u, &e);
    CHECK(e == TSR_DM_CMDERR_EXCEPTION, "absent CSR (pmpcfg0): cmderr 3 (%u)", e);
    W(D_DATA0, 0x77u);
    W(D_CMD, 0x00221005u);
    CHECK(acs_wait() == TSR_DM_CMDERR_EXCEPTION && R(D_DATA0) == 0x77u, "no command starts while cmderr != 0");
    cmderr_clear();
    CHECK(reg_set(0xF14u, 1u) == TSR_DM_CMDERR_EXCEPTION, "write to read-only mhartid: cmderr 3");
    cmderr_clear();
    W(D_CMD, 0x00321005u);                                // aarsize 3 (64-bit)
    CHECK(acs_wait() == TSR_DM_CMDERR_NOTSUP, "aarsize 3: cmderr 2");
    cmderr_clear();
    W(D_CMD, 0x02000000u);                                // cmdtype 2 (Access Memory)
    CHECK(acs_wait() == TSR_DM_CMDERR_NOTSUP, "Access Memory: cmderr 2");
    cmderr_clear();
    W(D_CMD, 0x01000000u);                                // cmdtype 1 (Quick Access)
    CHECK(acs_wait() == TSR_DM_CMDERR_NOTSUP, "Quick Access: cmderr 2");
    cmderr_clear();
    (void)reg_get(0x1020u, &e);
    CHECK(e == TSR_DM_CMDERR_EXCEPTION, "FPR f0 (no F): cmderr 3");
    cmderr_clear();
    (void)reg_get(0xC000u, &e);
    CHECK(e == TSR_DM_CMDERR_NOTSUP, "custom regno: cmderr 2");
    cmderr_clear();
    W(D_CMD, 0x00300000u);                                // aarsize 3 with transfer 0: nothing, no error
    CHECK(acs_wait() == 0u, "transfer 0 ignores aarsize");
    W(D_CMD, 0x002A1005u);                                // aarpostincrement: accepted
    CHECK(acs_wait() == 0u && R(D_DATA0) == x5, "aarpostincrement accepted");
    // dpc: the pc the hart resumes at. Point it at the loop head with x6 = 1 and x5 = 1000.
    cw(CSR_DPC, A_LOOP);
    xw(5, 1000u); xw(6, 1u);
    CHECK(cr(CSR_DPC) == A_LOOP, "dpc write");
    // resumereq with haltreq in the same write is ignored; resume.
    W(D_CTRL, 0xC0000001u);
    CHECK((R(D_STAT) & ALLHALTED) && !(R(D_STAT) & ALLRESUMEACK), "resumereq ignored with haltreq set");
    W(D_CTRL, 0x00000001u);
    CHECK(dm_resume(), "resume: allresumeack");
    st = R(D_STAT);
    CHECK((st & ALLRUNNING) && !(st & ALLHALTED) && !(csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED), "running at once, STATUS.RV_HALTED 0");
    run_for(300);
    // resumereq while running: nothing (resumeack stays 1).
    W(D_CTRL, 0x40000001u); W(D_CTRL, 0x00000001u);
    CHECK((R(D_STAT) & ALLRESUMEACK) && (R(D_STAT) & ALLRUNNING), "resumereq while running: no effect");
    CHECK(dm_halt(), "halt again");
    CHECK(xr(5) > 1000u && xr(5) < 1300u && get32(DTCM) >= 1001u, "resumed at dpc with the written registers (x5 %u)", xr(5));
    CHECK(xr(8) >= 1u, "x6 = 1 took the not-taken branch path (x8 %u)", xr(8));
    // ackhavereset clears havereset.
    W(D_CTRL, 0x10000001u);
    CHECK(!(R(D_STAT) & (ALLHAVERESET | TSR_DM_DMSTATUS_ANYHAVERESET)), "ackhavereset");
    W(D_CTRL, 0x00000001u);
    sim_free();
}

// ---------------------------------------------------------------------------
// Program buffer (DB §3.5) and the abstract-command cmderr paths
// ---------------------------------------------------------------------------
static void test_progbuf(void) {
    sim_new(1.0);
    prog_a();
    W(D_CTRL, 0x00000001u);
    boot(0);
    run_for(500);
    CHECK(dm_halt(), "halt");
    const uint32_t dpc = cr(CSR_DPC), dcsr = cr(CSR_DCSR), mepc = cr(CSR_MEPC), mcause = cr(CSR_MCAUSE);
    // fence.i (DB §3.5 verbatim).
    CHECK(fence_i() == 0u, "fence.i through the program buffer");
    CHECK((R(D_STAT) & ALLHALTED) && tsr_rv_halted(hart()), "parked again");
    // Computation, data0 as memory (sw to DataAddr), postexec after a transfer.
    const uint32_t p1[6] = {
        ADDI(11, 0, 42),
        LUI(12, TSR_DM_DATA_ADDR & 0xFFFFF000u),
        SW(11, 12, (int32_t)(TSR_DM_DATA_ADDR & 0xFFFu)),  // data0 = 42
        LW(13, 12, (int32_t)((TSR_DM_DATA_ADDR & 0xFFFu) + 4u)),   // x13 = data1
        EBREAK, 0,
    };
    W(D_DATA1, 0x5151u);
    CHECK(progbuf_run(p1, 6) == 0u, "program buffer ran");
    CHECK(R(D_DATA0) == 42u && xr(11) == 42u && xr(13) == 0x5151u, "data0 / data1 as memory");
    const uint32_t p2[2] = { ADDI(14, 6, 1), EBREAK };
    W(D_PB(0), p2[0]); W(D_PB(1), p2[1]);
    W(D_DATA0, 0x600u);
    W(D_CMD, 0x00271006u);                                // write x6 from data0, then postexec
    CHECK(acs_wait() == 0u && xr(6) == 0x600u && xr(14) == 0x601u, "transfer then postexec");
    CHECK(cr(CSR_DPC) == dpc && cr(CSR_DCSR) == dcsr && cr(CSR_MEPC) == mepc && cr(CSR_MCAUSE) == mcause,
          "program buffer leaves dpc, dcsr, mepc, mcause");
    // An exception in the program buffer: cmderr 3, no CSR changed, parked again.
    const uint32_t p3[2] = { LW(15, 0, -4), EBREAK };    // load from 0xFFFF_FFFC: access fault
    CHECK(progbuf_run(p3, 2) == TSR_DM_CMDERR_EXCEPTION, "exception: cmderr 3");
    cmderr_clear();
    CHECK(cr(CSR_MEPC) == mepc && cr(CSR_MCAUSE) == mcause && cr(CSR_DPC) == dpc && (R(D_STAT) & ALLHALTED),
          "no CSR update, still halted");
    const uint32_t p4[1] = { 0 };                         // an illegal instruction
    CHECK(progbuf_run(p4, 1) == TSR_DM_CMDERR_EXCEPTION, "illegal instruction: cmderr 3");
    cmderr_clear();
    // Longer than the synchronous budget: busy stays 1, the hart continues on the time base.
    const uint32_t p5[4] = { ADDI(16, 0, 1500), ADDI(16, 16, -1), BNE(16, 0, -4), EBREAK };
    for (unsigned i = 0; i < 4; ++i) W(D_PB(i), p5[i]);
    W(D_CMD, 0x00040000u);
    CHECK(R(D_ACS) & TSR_DM_ABSTRACTCS_BUSY, "busy after the command write");
    CHECK((R(D_STAT) & ALLHALTED) && tsr_dm_rv_halted(DM) && status_halted() && (R(D_ACS) & TSR_DM_ABSTRACTCS_BUSY),
          "RV_HALTED while the program buffer runs (DB §3.1)");
    // While busy: command, data and progbuf accesses set cmderr = busy and do nothing else.
    W(D_DATA0, 0xBADu);
    CHECK(((R(D_ACS) >> 8) & 7u) == TSR_DM_CMDERR_BUSY, "data0 write while busy: cmderr 1");
    W(D_ACS, 0x700u);
    CHECK(((R(D_ACS) >> 8) & 7u) == TSR_DM_CMDERR_BUSY, "abstractcs write while busy ignored");
    W(D_PB(0), 0);
    W(D_CMD, 0x00221005u);
    W(D_CTRL, 0x40000001u); W(D_CTRL, 0x00000001u);       // resume while busy: ignored
    CHECK(acs_wait() == TSR_DM_CMDERR_BUSY, "completes with cmderr 1 kept");
    cmderr_clear();
    CHECK(R(D_DATA0) != 0xBADu && R(D_PB(0)) == p5[0] && xr(16) == 0u && (R(D_STAT) & ALLHALTED),
          "busy accesses did nothing; the loop ran to its end");
    // A data0 read while busy also sets cmderr 1; a program that never parks keeps busy until
    // the hart is reset (ndmreset ends the command with cmderr 4 unless one is set).
    const uint32_t p6[1] = { JAL(0, 0) };
    W(D_PB(0), p6[0]);
    W(D_CMD, 0x00040000u);
    run_for(5000);
    CHECK((R(D_ACS) & TSR_DM_ABSTRACTCS_BUSY) && ((R(D_ACS) >> 8) & 7u) == 0u, "j . in the program buffer: busy");
    (void)R(D_DATA0);
    CHECK(((R(D_ACS) >> 8) & 7u) == TSR_DM_CMDERR_BUSY, "data0 read while busy: cmderr 1");
    cmderr_clear();
    CHECK(((R(D_ACS) >> 8) & 7u) == TSR_DM_CMDERR_BUSY, "cannot clear while busy");
    W(D_CTRL, 0x80000003u);                               // ndmreset with haltreq: reset halt
    CHECK(!(R(D_ACS) & TSR_DM_ABSTRACTCS_BUSY), "the reset ends the command");
    W(D_CTRL, 0x80000001u);
    CHECK(poll(ALLHALTED, 2000), "halted after the reset");
    W(D_CTRL, 0x00000001u);
    cmderr_clear();
    CHECK(R(D_ACS) == 0x08000002u && cr(CSR_DPC) == 0x80u, "cmderr cleared, at the reset entry");
    // dret in the program buffer leaves debug mode: the command ends with cmderr 3.
    const uint32_t p7[1] = { DRET };
    CHECK(progbuf_run(p7, 1) == TSR_DM_CMDERR_EXCEPTION, "dret in the program buffer: cmderr 3");
    CHECK((R(D_STAT) & ALLRUNNING) && (R(D_STAT) & ALLRESUMEACK), "the hart runs");
    cmderr_clear();
    run_for(200);
    CHECK(dm_halt() && xr(5) >= 1u, "runs again");
    sim_free();
}

// ---------------------------------------------------------------------------
// A halt request wakes a hart asleep in WFI (dpc = the instruction after it)
// ---------------------------------------------------------------------------
static void test_halt_wfi(void) {
    sim_new(1.0);
    org(0x80);
    i32(ADDI(5, 0, 1));
    i32(WFI);                                             // mie = 0: sleeps for good
    i32(ADDI(5, 0, 2));
    i32(JAL(0, 0));
    W(D_CTRL, 0x00000001u);
    boot(0);
    run_for(1000);
    CHECK((csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_SLEEP) && tsr_rv_sleeping(hart()), "asleep (STATUS.RV_SLEEP)");
    CHECK(dm_halt(), "halt from WFI");
    CHECK(cr(CSR_DPC) == 0x88u && xr(5) == 1u && dcause() == TSR_RV_DBG_CAUSE_HALTREQ, "dpc after the WFI");
    CHECK(status_halted(), "STATUS.RV_HALTED");
    const uint32_t st = csr_r(TSR_CSR_STATUS);
    CHECK((st & TSR_STATUS_RV_HALTED) && !(st & TSR_STATUS_RV_SLEEP), "RV_HALTED, not RV_SLEEP");
    CHECK(dm_resume(), "resume");
    run_for(200);
    CHECK(tsr_rv_reg(hart(), 5) == 2u, "continues after the WFI");
    sim_free();
}

// ---------------------------------------------------------------------------
// SBA (DB §3.6) with CC §4.3's refusals
// ---------------------------------------------------------------------------
static void test_sba(void) {
    sim_new(1.0);
    prog_a();
    W(D_CTRL, 0x00000001u);
    const int lid = tsr_sim_add_acc_listener(SIM, acc_cb, NULL);
    CHECK(lid >= 0, "access listener");
    // RV_RUN = 0: TCM reads work, TCM writes are refused (the load port owns them), SPM writes work.
    uint32_t w[8];
    sba_read_n(0x80, 4, w);
    CHECK(w[0] == LUI(10, TSR_RV_DTCM_BASE) && w[1] == ADDI(5, 0, 0) && w[3] == ADDI(5, 5, 1) && sba_err() == 0u,
          "read 4 ITCM words in reset");
    CHECK(R(D_SBADDR) == 0x90u, "autoincrement: sbaddress = a + 4n");
    const uint32_t two[2] = { 1, 2 };
    sba_write_n(TSR_RV_DTCM_BASE, 2, two);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR && get32(DTCM) == 0u, "DTCM write refused while RV_RUN = 0");
    sba_clear();
    sba_write_n(0x80, 1, two);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR && get32(ITCM + 0x80) == LUI(10, TSR_RV_DTCM_BASE),
          "ITCM write refused while RV_RUN = 0");
    sba_clear();
    W(D_DATA0, 0xDA7A0000u);
    sba_write_n(TSR_DM_DATA_ADDR, 1, two);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR && R(D_DATA0) == 0xDA7A0000u, "DM range write refused");
    sba_clear();
    uint8_t * spm = tsr_sim_spm(SIM);
    const uint32_t pat[8] = { 0x03020100u, 0x07060504u, 0x0B0A0908u, 0x0F0E0D0Cu, 0x13121110u, 0x17161514u, 0x1B1A1918u, 0x1F1E1D1Cu };
    NACC = 0;
    sba_write_n(TSR_RV_SPM_BASE + 0x100u, 8, pat);
    CHECK(sba_err() == 0u && memcmp(spm + 0x100, pat, 32) == 0 && R(D_SBADDR) == TSR_RV_SPM_BASE + 0x120u,
          "write 8 SPM words (in reset)");
    CHECK(NACC == 8 && ACC[0].kind == TSR_ACC_SBA_WR && ACC[0].addr == TSR_RV_SPM_BASE + 0x100u && ACC[0].size == 4u &&
          ACC[0].val == pat[0] && ACC[7].addr == TSR_RV_SPM_BASE + 0x11Cu, "SBA_WR access events (%d)", NACC);
    // Read n words: n - 1 reads with sbreadondata, then the last word without starting another.
    uint32_t got[8];
    NACC = 0;
    sba_read_n(TSR_RV_SPM_LB_BASE + 0x100u, 8, got);
    CHECK(memcmp(got, pat, 32) == 0 && sba_err() == 0u, "read 8 words through the SPM alias");
    CHECK(NACC == 8 && ACC[7].kind == TSR_ACC_SBA_RD && ACC[7].addr == TSR_RV_SPM_LB_BASE + 0x11Cu && ACC[7].val == pat[7],
          "exactly n bus reads (%d)", NACC);
    CHECK(R(D_SBADDR) == TSR_RV_SPM_LB_BASE + 0x120u && R(D_SBDATA) == pat[7], "last-word rule: no read past the block");
    boot(0);
    run_for(1000);
    // DTCM while running: the program's counter, then an n-word write.
    sba_read_n(TSR_RV_DTCM_BASE, 1, got);
    CHECK(sba_err() == 0u && got[0] >= 1u, "DTCM read while running (%u)", got[0]);
    const uint32_t six[6] = { 0, 0x11u, 0x22u, 0x33u, 0x44u, 0x55u };
    sba_write_n(TSR_RV_DTCM_BASE + 0x100u, 6, six);
    CHECK(sba_err() == 0u && get32(DTCM + 0x104) == 0x11u && get32(DTCM + 0x114) == 0x55u, "DTCM write while RV_RUN = 1");
    // 16-bit write (DB §3.6 verbatim) touches two bytes only; 8-bit access.
    sba_write16(TSR_RV_DTCM_BASE + 0x106u, 0xBEEFu);
    CHECK(sba_err() == 0u && get32(DTCM + 0x104) == 0xBEEF0011u, "16-bit write 0x%08x", get32(DTCM + 0x104));
    W(D_SBCS, 0x00107000u);                               // 8-bit, read on address
    W(D_SBADDR, TSR_RV_DTCM_BASE + 0x107u);
    CHECK(R(D_SBDATA) == 0xBEu && sba_err() == 0u, "8-bit read, zero-extended");
    W(D_SBCS, 0x00007000u);
    W(D_SBADDR, TSR_RV_DTCM_BASE + 0x105u);
    W(D_SBDATA, 0x1234567Au);
    CHECK(get32(DTCM + 0x104) == 0xBEEF7A11u, "8-bit write 0x%08x", get32(DTCM + 0x104));
    W(D_SBCS, 0x00127000u);                               // 16-bit read on address
    W(D_SBADDR, TSR_RV_DTCM_BASE + 0x106u);
    CHECK(R(D_SBDATA) == 0xBEEFu, "16-bit read");
    // Refusals (CC §4.3, CC-33) and errors: each sets sberror, changes nothing; W1C clears it.
    W(D_SBCS, 0x00140000u | 0x7000u);                     // 32-bit, read on address
    W(D_SBADDR, TSR_RV_DM_BASE);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR, "DM range read refused");
    sba_clear();
    // Autoincrement only after a successful access: a refused read or write leaves sbaddress0.
    W(D_SBCS, 0x00157000u);                               // 32-bit, autoincrement, read on address
    W(D_SBADDR, TSR_RV_DM_BASE + 0x100u);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR && R(D_SBADDR) == TSR_RV_DM_BASE + 0x100u,
          "refused autoincrement read: sbaddress0 unchanged (0x%08x)", R(D_SBADDR));
    sba_clear();
    const uint32_t staged = 0x5A5A0001u;
    tsr_sim_cw_write(SIM, TSR_CW_ISSUE_W(0), staged);
    const uint32_t pidx0 = tsr_sim_cw_read(SIM, TSR_CW_ISSUE_PIDX);
    static const uint32_t cwoff[] = { TSR_CW_ISSUE_W(0), TSR_CW_ISSUE_PUSH, TSR_CW_QCTRL, TSR_CW_TRACE_MARK, TSR_CW_LQ_POP,
                                      TSR_CW_LCQ_POST, TSR_CW_LOG_PUSH, TSR_CW_MTIMECMP_LO, TSR_CW_RT_STATE };
    NACC = 0;
    for (unsigned i = 0; i < sizeof(cwoff) / sizeof(cwoff[0]); ++i) {
        sba_write_n(TSR_RV_CTL_BASE + cwoff[i], 1, &six[1]);
        CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR && R(D_SBADDR) == TSR_RV_CTL_BASE + cwoff[i],
              "control-window write 0x%03x refused, sbaddress0 not incremented", cwoff[i]);
        sba_clear();
    }
    CHECK(tsr_sim_cw_read(SIM, TSR_CW_ISSUE_W(0)) == staged && tsr_sim_cw_read(SIM, TSR_CW_ISSUE_PIDX) == pidx0 &&
          NACC == 0 && csr_r(TSR_CSR_RT_STATE) == 0u, "refused writes did nothing (D15), no access events");
    CHECK(sba_rd32(TSR_RV_CTL_BASE + TSR_CW_ISSUE_W(0)) == staged && sba_err() == 0u, "control-window read");
    CHECK(sba_rd32(TSR_RV_CTL_BASE + TSR_CW_ISSUE_PIDX) == pidx0, "ISSUE_PIDX read");
    sba_write_n(TSR_RV_CORE_BASE + TSR_CSR_RV_BOOT, 1, &six[1]);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR && csr_r(TSR_CSR_RV_BOOT) == 0u, "core-window write refused");
    sba_clear();
    CHECK(sba_rd32(TSR_RV_CORE_BASE + TSR_CSR_ID) == TSR_ID_MAGIC, "core-window read (ID)");
    CHECK(sba_rd32(TSR_RV_CORE_BASE + TSR_CSR_LOAD_DATA) == 0u && sba_rd32(TSR_RV_CORE_BASE + TSR_CSR_DMI(TSR_DMI_DMSTATUS)) == 0u &&
          sba_err() == 0u, "core window: LOAD_DATA and the DMI window read 0");
    W(D_SBCS, 0x00140000u);
    W(D_SBADDR, 0x30000000u);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR, "unmapped address: sberror 2");
    // While sberror != 0 accesses do nothing.
    W(D_SBADDR, TSR_RV_DTCM_BASE + 0x104u);
    W(D_SBDATA, 0xFFFFFFFFu);
    CHECK(get32(DTCM + 0x104) == 0xBEEF7A11u && R(D_SBADDR) == TSR_RV_DTCM_BASE + 0x104u, "blocked by sberror; sbaddress still written");
    W(D_SBCS, 0x00140000u | 0x1000u);                     // W1C of one sberror bit (2 = 0b010: bit 13 stays)
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_BADADDR, "W1C of another bit keeps it");
    W(D_SBCS, 0x00142000u);
    CHECK(((R(D_SBCS) >> 12) & 7u) == 0u, "W1C clears it");
    W(D_SBCS, 0x00140000u);
    W(D_SBADDR, TSR_RV_DTCM_BASE + 0x102u);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_ALIGN, "misaligned 32-bit: sberror 3");
    sba_clear();
    W(D_SBCS, 0x00160000u);                               // sbaccess 3 (64-bit)
    W(D_SBADDR, TSR_RV_DTCM_BASE);
    CHECK(((R(D_SBCS) >> 12) & 7u) == TSR_DM_SBERR_SIZE && (R(D_SBCS) & TSR_DM_SBCS_SBACCESS) == (3u << 17), "64-bit: sberror 4");
    sba_clear();
    CHECK(!(R(D_SBCS) & (TSR_DM_SBCS_SBBUSY | TSR_DM_SBCS_SBBUSYERROR)), "sbbusy 0, sbbusyerror 0 (accesses complete at once)");
    // SBA leaves the hart alone: still running, its program intact.
    CHECK((R(D_STAT) & ALLRUNNING) && get32(ITCM + 0x8C) == ADDI(5, 5, 1), "hart running");
    tsr_sim_remove_acc_listener(SIM, lid);
    sim_free();
}

// ---------------------------------------------------------------------------
// Single step over 20 instructions against a free-running reference hart
// ---------------------------------------------------------------------------
#define NSTEP 20
static void test_step(double ratio, const char * sched) {
    // Reference: the same program on a bare hart (firmware profile, no core).
    sim_new_sched(ratio, sched);
    prog_a();
    tsr_bus * rb = tsr_bus_create_fw(NULL, NULL);
    memcpy(tsr_bus_mem(rb, 0, 0x100), ITCM, 0x100);
    tsr_rv * ref = tsr_rv_create(tsr_bus_iface(rb));
    tsr_rv_reset(ref, 0);
    uint32_t rpc[NSTEP + 1], rx[NSTEP + 1][5];
    int kinds_br = 0, kinds_ld = 0, kinds_c = 0;
    for (int k = 0; k <= NSTEP; ++k) {
        rpc[k] = tsr_rv_pc(ref);
        for (unsigned i = 0; i < 5; ++i) rx[k][i] = tsr_rv_reg(ref, 5 + i);
        if (k < NSTEP) {
            if (rpc[k] == 0x9Cu) ++kinds_br;
            if (rpc[k] == A_LW) ++kinds_ld;
            if (rpc[k] == A_CADD || rpc[k] == 0x9Au) ++kinds_c;
            tsr_rv_step(ref, 1);
        }
    }
    int not_taken = 0;
    for (int k = 0; k < NSTEP; ++k) not_taken |= rpc[k] == 0x9Cu && rpc[k + 1] == 0xA0u;
    CHECK(kinds_br >= 2 && not_taken && kinds_ld >= 2 && kinds_c >= 2,
          "the 20 steps cover taken and not-taken branches, loads, compressed instructions");
    // The DM: halt before the first instruction ("reset halt": haltreq held over the boot).
    W(D_CTRL, 0x00000001u);
    W(D_CTRL, 0x80000001u);
    boot(0);
    CHECK(poll(ALLHALTED, 1000), "halted at boot");
    W(D_CTRL, 0x00000001u);
    CHECK(cr(CSR_DPC) == 0x80u && dcause() == TSR_RV_DBG_CAUSE_HALTREQ && cr(CSR_MINSTRET) == 0u, "before the first instruction");
    cw(CSR_DCSR, cr(CSR_DCSR) | 4u);                      // dcsr.step
    for (int k = 1; k <= NSTEP; ++k) {
        CHECK(dm_resume(), "step %d: resume", k);
        CHECK(poll(ALLHALTED, 1000), "step %d: halted again", k);
        const uint32_t dpc = cr(CSR_DPC);
        CHECK(dpc == rpc[k], "step %d: dpc 0x%x, reference 0x%x", k, dpc, rpc[k]);
        CHECK(dcause() == TSR_RV_DBG_CAUSE_STEP, "step %d: cause step", k);
        for (unsigned i = 0; i < 5; ++i) CHECK(xr(5 + i) == rx[k][i], "step %d: x%u 0x%x, reference 0x%x", k, 5 + i, xr(5 + i), rx[k][i]);
        CHECK(cr(CSR_MINSTRET) == (uint32_t)k, "step %d: minstret", k);
    }
    CHECK(get32(DTCM) == rx[NSTEP][0] || rpc[NSTEP] == 0x90u, "the stores reached DTCM");
    cw(CSR_DCSR, cr(CSR_DCSR) & ~4u);
    CHECK(dm_resume(), "resume without step");
    run_for(500);
    CHECK(R(D_STAT) & ALLRUNNING, "free running");
    tsr_rv_destroy(ref);
    tsr_bus_destroy(rb);
    sim_free();
}

// ---------------------------------------------------------------------------
// Hardware breakpoints (Ibex triggers 0-1, DB §3.7), ebreak with dcsr.ebreakm, software
// breakpoints written through SBA
// ---------------------------------------------------------------------------
static void test_breakpoints(void) {
    sim_new(1.0);
    prog_a();
    W(D_CTRL, 0x00000001u);
    boot(0);
    run_for(300);
    CHECK(dm_halt(), "halt");
    // Trigger 0 on the every-third-iteration instruction.
    cw(CSR_TSELECT, 0);
    cw(CSR_TDATA1, 0x28001044u);
    cw(CSR_TDATA2, A_ADD8);
    CHECK(cr(CSR_TDATA1) == 0x28001044u && cr(CSR_TDATA2) == A_ADD8, "tdata1 / tdata2 read back");
    const uint32_t x8 = xr(8);
    CHECK(dm_resume(), "resume");
    CHECK(poll(ALLHALTED, 5000), "trigger 0 hit");
    CHECK(cr(CSR_DPC) == A_ADD8 && dcause() == TSR_RV_DBG_CAUSE_TRIGGER && xr(8) == x8, "dpc = the address, cause 2, not executed");
    // Trigger 1 on the load; trigger 0 off (resuming onto it would re-enter at once).
    cw(CSR_TDATA1, 0);
    cw(CSR_TSELECT, 1);
    CHECK(cr(CSR_TSELECT) == 1u, "tselect 1");
    cw(CSR_TDATA1, 0x28001044u);
    cw(CSR_TDATA2, A_LW);
    CHECK(dm_resume() && poll(ALLHALTED, 5000), "trigger 1 hit");
    CHECK(cr(CSR_DPC) == A_LW && dcause() == TSR_RV_DBG_CAUSE_TRIGGER && xr(8) == x8 + 1u, "at the load, after the add");
    cw(CSR_TDATA1, 0);
    // ebreak with ebreakm: a software breakpoint at the 32-bit xor (ebreak) and at the compressed
    // c.addi (c.ebreak through the DB §3.6 16-bit write), then fence.i.
    const uint32_t mcause = cr(CSR_MCAUSE), mepc = cr(CSR_MEPC);
    const uint32_t orig_c = get32(ITCM + A_CADD) & 0xFFFFu, orig_x = get32(ITCM + A_XOR);
    sba_write16(A_CADD, C_EBREAK);
    CHECK(sba_err() == 0u && (get32(ITCM + A_CADD) & 0xFFFFu) == C_EBREAK && (get32(ITCM + A_CADD) >> 16) == C_NOP, "c.ebreak written");
    sba_write_n(A_XOR, 1, &(const uint32_t){ EBREAK });
    CHECK(sba_err() == 0u && get32(ITCM + A_XOR) == EBREAK, "ebreak written");
    CHECK(fence_i() == 0u, "fence.i");
    cw(CSR_DCSR, cr(CSR_DCSR) | (1u << 15));              // ebreakm
    CHECK(dm_resume() && poll(ALLHALTED, 5000), "c.ebreak hit");
    CHECK(cr(CSR_DPC) == A_CADD && dcause() == TSR_RV_DBG_CAUSE_EBREAK && cr(CSR_MCAUSE) == mcause && cr(CSR_MEPC) == mepc,
          "dpc = the c.ebreak, cause 1, no M-mode trap");
    sba_write16(A_CADD, orig_c);
    CHECK(fence_i() == 0u && sba_err() == 0u, "original restored");
    CHECK(dm_resume() && poll(ALLHALTED, 5000), "ebreak hit");
    CHECK(cr(CSR_DPC) == A_XOR && dcause() == TSR_RV_DBG_CAUSE_EBREAK, "dpc = the ebreak, cause 1");
    // Without ebreakm the ebreak is an M-mode breakpoint exception (vector 0 parks with x31).
    cw(CSR_DCSR, cr(CSR_DCSR) & ~(1u << 15));
    CHECK(dm_resume(), "resume onto the ebreak");
    run_for(300);
    CHECK(dm_halt() && xr(31) == 0xEEu && cr(CSR_MCAUSE) == TSR_RV_EXC_BREAKPOINT && cr(CSR_MEPC) == A_XOR,
          "breakpoint exception without ebreakm (mcause %u)", cr(CSR_MCAUSE));
    sba_write_n(A_XOR, 1, &orig_x);
    CHECK(fence_i() == 0u && get32(ITCM + A_XOR) == orig_x, "restored");
    sim_free();
}

// ---------------------------------------------------------------------------
// A halt request while a push waits for a credit (DB §3.4, CC §7.3): the hart halts after the push
// ---------------------------------------------------------------------------
static void test_halt_push_wait(void) {
    sim_new(1.0);
    const tsr_insn ld = tsr_enc_ld(0, 0, 0, 1);
    org(0x80);
    i32(LUI(10, TSR_RV_CTL_BASE));
    for (unsigned i = 0; i < 8; ++i) {
        if (ld.w[i] == 0u) continue;
        li(11, ld.w[i]);
        i32(SW(11, 10, (int32_t)TSR_CW_ISSUE_W(i)));
    }
    li(12, 0xFF00u);                                      // WMASK all words
    i32(ADDI(13, 0, 0));
    const uint32_t loop = PCW;
    i32(SW(12, 10, (int32_t)TSR_CW_ISSUE_PUSH));
    const uint32_t after = PCW;
    i32(ADDI(13, 13, 1));
    i32(JAL(0, (int32_t)loop - (int32_t)PCW));
    W(D_CTRL, 0x00000001u);
    csr_w(TSR_CSR_RV_BOOT, 0);
    csr_w(TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_HOST_HOLD);   // the DMA head is held: credits run out
    run_for(3000);
    CHECK(tsr_rv_waiting(hart()) && tsr_rv_reg(hart(), 13) == 8u, "the 9th push waits (x13 %u)", tsr_rv_reg(hart(), 13));
    W(D_CTRL, 0x80000001u);
    run_for(20000);
    CHECK(!(R(D_STAT) & ALLHALTED) && (R(D_STAT) & ALLRUNNING) && tsr_rv_waiting(hart()), "no halt while the push waits");
    csr_w(TSR_CSR_CTRL, TSR_CTRL_RV_RUN);                 // release the hold: the LDs run, credits return
    CHECK(poll(ALLHALTED, 20000), "halted after the push");
    W(D_CTRL, 0x00000001u);
    CHECK(cr(CSR_DPC) == after && dcause() == TSR_RV_DBG_CAUSE_HALTREQ && xr(13) == 8u, "dpc = the instruction after the push (0x%x)", cr(CSR_DPC));
    CHECK(sba_rd32(TSR_RV_CTL_BASE + TSR_CW_ISSUE_PIDX) == 9u && !(csr_r(TSR_CSR_STATUS) & TSR_STATUS_PUSH_TIMEOUT),
          "the push completed (ISSUE_PIDX 9), no timeout");
    sim_free();
}

// ---------------------------------------------------------------------------
// Halt-on-boot through DB's request (DB §3.8), "reset halt", ndmreset (DB §3.5)
// ---------------------------------------------------------------------------
static int RESET_HALT;
static uint32_t EVS[16], NEV;
static void ev_cb(void * ctx, uint32_t ev, uint32_t arg) {
    (void)ctx; (void)arg;
    if (NEV < 16) EVS[NEV++] = ev;
    if (ev == TSR_RV_DBG_EV_RESET && RESET_HALT) tsr_dm_set_ext_haltreq(DM, 1);
    if (ev == TSR_RV_DBG_EV_ENTER) tsr_dm_set_ext_haltreq(DM, 0);   // DB §3.3: one request, one halt
}

static void test_boot_reset(void) {
    sim_new(1.0);
    org(0x280);                                           // RV_BOOT = 0x200
    i32(LUI(10, TSR_RV_CTL_BASE));
    li(11, 0x1234u);
    i32(SW(11, 10, (int32_t)TSR_CW_ISSUE_W(0)));
    i32(ADDI(5, 5, 1));
    i32(JAL(0, -4));
    W(D_CTRL, 0x00000001u);
    tsr_dm_set_event_hook(DM, ev_cb, NULL);
    // External request pending at the reset release: halted before the first instruction.
    tsr_dm_set_ext_haltreq(DM, 1);
    boot(0x200);
    const uint32_t ev = tsr_sim_advance(SIM, 10000);
    CHECK(ev & TSR_SIM_EV_HALT, "tsr_sim_advance reports the halt (0x%x)", ev);
    CHECK((R(D_STAT) & ALLHALTED) && tsr_dm_rv_halted(DM) && (csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED), "halted on boot");
    CHECK(cr(CSR_DPC) == 0x280u && dcause() == TSR_RV_DBG_CAUSE_HALTREQ && cr(CSR_MINSTRET) == 0u && tsr_rv_retired(hart()) == 0u,
          "dpc = RV_BOOT + 0x80, nothing executed");
    CHECK(NEV >= 2 && EVS[0] == TSR_RV_DBG_EV_RESET && EVS[1] == TSR_RV_DBG_EV_ENTER, "events RESET, ENTER forwarded");
    CHECK(dm_resume(), "resume");
    run_for(500);
    CHECK((R(D_STAT) & ALLRUNNING) && tsr_rv_reg(hart(), 5) > 10u, "one request, one halt");
    // The same through the RESET event, after a previous halt (STATUS.RV_HALTED must rise again).
    CHECK(dm_halt(), "halt");
    csr_w(TSR_CSR_CTRL, 0);
    CHECK(R(D_STAT) & ALLUNAVAIL, "RV_RUN 0: unavailable");
    W(D_CMD, 0x00221005u);
    CHECK(acs_wait() == TSR_DM_CMDERR_HALTRESUME, "command while in reset: cmderr 4");
    cmderr_clear();
    RESET_HALT = 1; NEV = 0;
    boot(0x200);
    CHECK(tsr_sim_advance(SIM, 10000) & TSR_SIM_EV_HALT, "halt event");
    RESET_HALT = 0;
    CHECK((csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) && cr(CSR_DPC) == 0x280u && cr(CSR_MINSTRET) == 0u,
          "halted before the first instruction from the RESET event");
    CHECK(R(D_STAT) & ALLHAVERESET, "havereset after the reboot");
    W(D_CTRL, 0x10000001u);
    CHECK(!(R(D_STAT) & ALLHAVERESET), "ackhavereset");
    W(D_CTRL, 0x00000001u);
    CHECK(dm_resume(), "resume");
    run_for(500);
    CHECK(sba_rd32(TSR_RV_CTL_BASE + TSR_CW_ISSUE_W(0)) == 0x1234u, "the program staged ISSUE_W0");
    // ndmreset (DB §3.5: W 0x10 = 0x0000_0003, then 0x0000_0001).
    W(D_CTRL, 0x00000003u);
    uint32_t st = csr_r(TSR_CSR_STATUS);
    CHECK((st & TSR_STATUS_RV_RESET_DONE) && !(st & TSR_STATUS_RV_RUNNING) && (csr_r(TSR_CSR_CTRL) & TSR_CTRL_RV_RUN),
          "ndmreset: the reset sequence, CTRL.RV_RUN kept (STATUS 0x%08x)", st);
    CHECK((R(D_STAT) & ALLUNAVAIL) && R(D_CTRL) == 0x00000003u, "unavailable while held");
    CHECK(sba_rd32(TSR_RV_CTL_BASE + TSR_CW_ISSUE_W(0)) == 0u, "staging reset by the RISC-V reset sequence");
    run_for(2000);
    CHECK(R(D_STAT) & ALLUNAVAIL, "held in reset");
    W(D_CTRL, 0x00000001u);
    run_for(500);
    st = R(D_STAT);
    CHECK((st & ALLRUNNING) && (st & ALLHAVERESET) && (csr_r(TSR_CSR_STATUS) & TSR_STATUS_RV_RUNNING), "released: running, havereset");
    CHECK(sba_rd32(TSR_RV_CTL_BASE + TSR_CW_ISSUE_W(0)) == 0x1234u && tsr_rv_reg(hart(), 5) < 600u, "restarted from RV_BOOT + 0x80");
    // "reset halt": ndmreset with haltreq, then release with haltreq still set.
    W(D_CTRL, 0x80000003u);
    W(D_CTRL, 0x80000001u);
    CHECK(poll(ALLHALTED, 2000), "reset halt");
    W(D_CTRL, 0x00000001u);
    CHECK(cr(CSR_DPC) == 0x280u && xr(5) == 0u && xr(11) == 0u, "halted at the first instruction, registers reset");
    tsr_dm_set_event_hook(DM, NULL, NULL);
    sim_free();
}

// ---------------------------------------------------------------------------
// A timer interrupt pending during a halt is taken after the resume
// ---------------------------------------------------------------------------
static void test_timer(void) {
    sim_new(1.0);
    org(0x1C);                                            // MTI vector (mtvec base 0, vectored)
    i32(CSRRW(0, CSR_MIE, 0)); i32(ADDI(20, 0, 0xAA)); i32(MRET);
    org(0x80);
    i32(LUI(10, TSR_RV_CTL_BASE));
    i32(LW(11, 10, (int32_t)TSR_CW_MTIME_LO));
    li(13, 30000u);                                       // beyond one agent slice (tsr_bus.c)
    i32(ADD(11, 11, 13));
    i32(SW(11, 10, (int32_t)TSR_CW_MTIMECMP_LO));
    i32(SW(0, 10, (int32_t)TSR_CW_MTIMECMP_HI));
    i32(ADDI(12, 0, 0x80));
    i32(CSRRW(0, CSR_MIE, 12));                           // MTIE
    i32(CSRRSI(0, CSR_MSTATUS, 8));                       // MIE
    i32(ADDI(5, 5, 1));
    i32(JAL(0, -4));
    W(D_CTRL, 0x00000001u);
    boot(0);
    run_for(400);
    CHECK(dm_halt(), "halted before the timer");
    const uint32_t cmp = sba_rd32(TSR_RV_CTL_BASE + TSR_CW_MTIMECMP_LO);
    CHECK(tsr_sim_now(SIM) < cmp, "MTIME %llu < MTIMECMP %u", (unsigned long long)tsr_sim_now(SIM), cmp);
    const uint32_t dpc = cr(CSR_DPC);
    run_for(cmp + 2000u - tsr_sim_now(SIM));
    CHECK(tsr_sim_rv_irq(SIM) & TSR_RV_MIP_MTIP, "the timer is pending");
    CHECK((R(D_STAT) & ALLHALTED) && cr(CSR_MCAUSE) == 0u && xr(20) == 0u && cr(0x344u) & 0x80u, "not taken in debug mode (mip.MTIP)");
    CHECK(dm_resume(), "resume");
    run_for(200);
    CHECK(dm_halt(), "halt");
    CHECK(xr(20) == 0xAAu && cr(CSR_MCAUSE) == (TSR_RV_MCAUSE_IRQ | TSR_RV_IRQ_MTI) && cr(CSR_MEPC) == dpc,
          "taken after the resume (mcause 0x%x, mepc 0x%x, dpc 0x%x)", cr(CSR_MCAUSE), cr(CSR_MEPC), dpc);
    sim_free();
}

// ---------------------------------------------------------------------------
// The DM's state survives SOFT_RESET (DB §3.1, EX §8.5)
// ---------------------------------------------------------------------------
static void test_soft_reset(void) {
    sim_new(1.0);
    prog_a();
    W(D_CTRL, 0x00000001u);
    boot(0);
    run_for(500);
    CHECK(dm_halt(), "halt");
    W(D_DATA0, 0x0D0D0D0Du); W(D_DATA1, 0x1D1D1D1Du);
    W(D_PB(3), 0x33333333u);
    W(D_SBCS, 0x00150000u);                               // 32-bit, autoincrement, read on address
    W(D_SBADDR, TSR_RV_DTCM_BASE + 0x40u);
    const uint32_t sbcs = R(D_SBCS);
    W(D_CTRL, 0x80000001u);                               // keep a halt request across the reset
    csr_w(TSR_CSR_CTRL, TSR_CTRL_SOFT_RESET);
    for (int i = 0; i < 100 && (csr_r(TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY); ++i) tsr_sim_advance(SIM, 1000);
    CHECK(!(csr_r(TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY) && !(csr_r(TSR_CSR_CTRL) & TSR_CTRL_RV_RUN), "SOFT_RESET done, RV_RUN 0");
    CHECK(R(D_CTRL) == 0x80000001u && R(D_DATA0) == 0x0D0D0D0Du && R(D_DATA1) == 0x1D1D1D1Du && R(D_PB(3)) == 0x33333333u &&
          R(D_SBCS) == sbcs && R(D_SBADDR) == TSR_RV_DTCM_BASE + 0x44u, "registers kept");
    CHECK(R(D_STAT) & ALLUNAVAIL, "the hart is in reset");
    boot(0);
    CHECK(poll(ALLHALTED, 2000), "the kept halt request halts the reboot");
    W(D_CTRL, 0x00000001u);
    CHECK(cr(CSR_DPC) == 0x80u && (R(D_STAT) & ALLHAVERESET), "at RV_BOOT + 0x80, havereset");
    sim_free();
}

// ---------------------------------------------------------------------------
// The DMI window at host CSR 0xC00 + 4r (DB §3.2): through the DBG model, gated by DMI_EN
// ---------------------------------------------------------------------------
static void test_csr_window(void) {
    sim_new(1.0);
    prog_a();
    boot(0);
    run_for(200);
    const uint32_t stat_off = TSR_CSR_DMI(TSR_DMI_DMSTATUS), ctrl_off = TSR_CSR_DMI(TSR_DMI_DMCONTROL);
    if (csr_r(TSR_CSR_DBG_ID) == 0u) {
        // No DBG model: the core answers the window with 0 and drops writes.
        csr_w(ctrl_off, 0x00000001u);
        CHECK(csr_r(stat_off) == 0u && R(D_CTRL) == 0u, "without the DBG model the window reads 0, writes do not reach the DM");
        printf("sim_dm: no DBG model in this build: CSR window checked as absent\n");
        sim_free();
        return;
    }
    CHECK(csr_r(TSR_CSR_CTL_ID) & TSR_CTL_ID_DM, "CTL_ID.DM");
    // DMI_EN = 0 (reset): reads 0, writes dropped.
    csr_w(ctrl_off, 0x00000001u);
    CHECK(csr_r(stat_off) == 0u && R(D_CTRL) == 0u, "DMI_EN = 0: reads 0, the write did not reach the DM");
    csr_w(TSR_CSR_DBG_CFG, csr_r(TSR_CSR_DBG_CFG) | TSR_DBG_CFG_DMI_EN);
    CHECK(csr_r(stat_off) == R(D_STAT) && csr_r(TSR_CSR_DMI(TSR_DMI_HARTINFO)) == 0x00212380u, "DMI_EN = 1: dmstatus through the window");
    // The halt sequence through the CSR port.
    csr_w(ctrl_off, 0x00000001u);
    csr_w(ctrl_off, 0x80000001u);
    int ok = 0;
    for (int i = 0; i < 1000 && !ok; ++i) {
        ok = (csr_r(stat_off) & ALLHALTED) != 0u;
        if (!ok) tsr_sim_advance(SIM, 50);
    }
    csr_w(ctrl_off, 0x00000001u);
    CHECK(ok && tsr_dm_rv_halted(DM), "halted through host CSR 0xC40");
    csr_w(TSR_CSR_DMI(TSR_DMI_COMMAND), 0x00221005u);
    CHECK(csr_r(TSR_CSR_DMI(TSR_DMI_DATA0)) == tsr_rv_reg(hart(), 5), "x5 through the window");
    CHECK(tsr_sim_corewin_read(SIM, stat_off) == 0u, "the core window reads the DMI window as 0 (CC §5.3)");
    sim_free();
}

// ---------------------------------------------------------------------------
// sim_dm_fw: a GDB-style session on a C program (tests/dm/fw_dm.c)
// ---------------------------------------------------------------------------
static uint32_t sym(const tsr_rv_elf * e, const char * name) {
    uint32_t v = 0;
    CHECK(tsr_rv_elf_symbol(e, name, &v) == 0, "symbol %s", name);
    return v;
}

static int test_fw(const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        printf("sim_dm_fw: %s not found (no RISC-V compiler at configure time): skipped\n", path);
        return 77;
    }
    fclose(f);
    tsr_rv_elf elf;
    if (tsr_rv_elf_read(path, &elf) != 0) { CHECK(0, "read %s", path); return test_finish("sim_dm_fw"); }
    sim_new(166.7 / 276.3);
    CHECK(tsr_rv_elf_load(tsr_rv_agent_bus(tsr_sim_rv_agent(SIM)), &elf) == 0, "image in the TCMs");
    const uint32_t f_square = sym(&elf, "dm_square"), f_report = sym(&elf, "dm_report");
    const uint32_t g_input = sym(&elf, "g_input"), g_result = sym(&elf, "g_result"), g_count = sym(&elf, "g_count");
    const uint32_t g_trace = sym(&elf, "g_trace");
    // Attach and halt at boot (the server's --halt-on-boot, DB §9.3).
    W(D_CTRL, 0x00000001u);
    W(D_CTRL, 0x80000001u);
    boot(0);
    CHECK(poll(ALLHALTED, 2000), "halted at boot");
    W(D_CTRL, 0x00000001u);
    CHECK(cr(CSR_DPC) == 0x80u, "at _start");
    // Hardware breakpoint at dm_square: the argument is g_input = 5; change it to 7.
    cw(CSR_TSELECT, 0); cw(CSR_TDATA1, 0x28001044u); cw(CSR_TDATA2, f_square);
    CHECK(dm_resume() && poll(ALLHALTED, 20000), "breakpoint at dm_square");
    CHECK(cr(CSR_DPC) == f_square && dcause() == TSR_RV_DBG_CAUSE_TRIGGER && xr(10) == 5u, "dpc dm_square, a0 = 5 (%u)", xr(10));
    xw(10, 7u);
    cw(CSR_TDATA1, 0);
    // Software breakpoint at dm_report (c.ebreak or ebreak by the instruction's size, DB §3.7).
    const uint32_t word = sba_rd32(f_report & ~3u);
    const uint32_t parcel = (f_report & 2u) ? word >> 16 : word & 0xFFFFu;
    const int is32 = (parcel & 3u) == 3u;
    uint32_t saved[2] = { parcel, 0 };
    if (is32) {
        saved[1] = (sba_rd32((f_report + 2u) & ~3u) >> (((f_report + 2u) & 2u) ? 16 : 0)) & 0xFFFFu;
        sba_write16(f_report, EBREAK & 0xFFFFu);
        sba_write16(f_report + 2u, EBREAK >> 16);
    } else {
        sba_write16(f_report, C_EBREAK);
    }
    CHECK(sba_err() == 0u && fence_i() == 0u, "breakpoint inserted (%d-bit), fence.i", is32 ? 32 : 16);
    cw(CSR_DCSR, cr(CSR_DCSR) | (1u << 15));
    CHECK(dm_resume() && poll(ALLHALTED, 20000), "software breakpoint hit");
    CHECK(cr(CSR_DPC) == f_report && dcause() == TSR_RV_DBG_CAUSE_EBREAK && xr(10) == 49u, "dm_report(49) (a0 %u)", xr(10));
    CHECK(sba_rd32(g_result) == 49u, "g_result = 49 through SBA");
    // Remove it, step once, continue.
    sba_write16(f_report, saved[0]);
    if (is32) sba_write16(f_report + 2u, saved[1]);
    CHECK(sba_err() == 0u && fence_i() == 0u, "breakpoint removed");
    cw(CSR_DCSR, (cr(CSR_DCSR) | 4u) & ~(1u << 15));
    CHECK(dm_resume() && poll(ALLHALTED, 2000), "step");
    CHECK(cr(CSR_DPC) != f_report && dcause() == TSR_RV_DBG_CAUSE_STEP, "stepped off dm_report (dpc 0x%x)", cr(CSR_DPC));
    cw(CSR_DCSR, cr(CSR_DCSR) & ~4u);
    CHECK(dm_resume(), "continue");
    run_for(20000);
    // Live variables through SBA while the hart runs (DB-3).
    const uint32_t c0 = sba_rd32(g_count);
    run_for(20000);
    const uint32_t c1 = sba_rd32(g_count);
    CHECK(c1 > c0 && c0 >= 1u, "g_count advances while running (%u -> %u)", c0, c1);
    sba_write_n(g_input, 1, &(const uint32_t){ 9u });
    run_for(20000);
    CHECK(sba_rd32(g_result) == 81u, "the program sees the SBA write (g_result %u)", sba_rd32(g_result));
    uint32_t tr[16];
    sba_read_n(g_trace, 16, tr);
    int seen49 = 0, seen81 = 0;
    for (int i = 0; i < 16; ++i) { seen49 |= tr[i] == 49u; seen81 |= tr[i] == 81u; }
    CHECK(seen81 && (seen49 || c1 > 16u), "g_trace through an n-word SBA read");
    CHECK(dm_halt(), "final halt");
    tsr_rv_elf_free(&elf);
    sim_free();
    return test_finish("sim_dm_fw");
}

int main(int argc, char ** argv) {
    if (argc > 2 && strcmp(argv[1], "--fw") == 0) return test_fw(argv[2]);
    test_profile();
    test_halt_resume_regs();
    test_progbuf();
    test_halt_wfi();
    test_sba();
    test_step(1.0, "fixed");
    test_step(166.7 / 276.3, "fixed");
    test_step(166.7 / 276.3, "random:5,0.25,4");      // P1.8: the steps are architectural
    test_breakpoints();
    test_halt_push_wait();
    test_boot_reset();
    test_timer();
    test_soft_reset();
    test_csr_window();
    return test_finish("sim_dm");
}

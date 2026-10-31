// fw_ctrl.c — the tb/ctrl test program (W4 item 3): exercises the TCMs, the control window, the
// issue port, the core window and the SPM windows of the control subsystem (CC §4-§5, §7, §10.5).
// It runs on the RTL (tb/ctrl, Ibex RVFI) and on the simulator's interpreter (tsr_rv with the
// simulator's control window, tsr_q.c), and the two are compared instruction by instruction and
// by their issue-FIFO entries, LCQ records, log words, RT_* mirrors, SPM writes and the final DTCM.
//
// Rules for a lockstep-comparable program (CC §15.4): the values of timing-dependent registers
// (ISSUE_CREDIT, DONE_SEQ, QSTATUS, ISSUE_STATUS.ERR_IRQ, error records, MTIME, LQ / LCQ / log
// state, the core window) may differ between the two environments; the lockstep gives the
// interpreter the RTL's value for those loads, so the program may branch on them. Interrupts stay
// disabled (mstatus.MIE = 0); WFI wakes on mie.
#include <stdint.h>

#include "tsr_isa.h"

#define CW(o)   (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CTL_BASE + (o)))
#define CWB(o)  (*(volatile uint8_t *)(uintptr_t)(TSR_RV_CTL_BASE + (o)))
#define CWH(o)  (*(volatile uint16_t *)(uintptr_t)(TSR_RV_CTL_BASE + (o)))
#define CORE(o) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CORE_BASE + (o)))
#define SPMW(a) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_SPM_BASE + (a)))
#define SPMH(a) (*(volatile uint16_t *)(uintptr_t)(TSR_RV_SPM_BASE + (a)))
#define SPMB(a) (*(volatile uint8_t *)(uintptr_t)(TSR_RV_SPM_BASE + (a)))
#define LBW(a)  (*(volatile uint32_t *)(uintptr_t)(TSR_RV_SPM_LB_BASE + (a)))
#define LBB(a)  (*(volatile uint8_t *)(uintptr_t)(TSR_RV_SPM_LB_BASE + (a)))

#define NONE TSR_TICKET_NONE

// Trap log (crt0.S): mcause, mepc, mtval per exception.
volatile uint32_t trap_n;
volatile uint32_t trap_log[16][4];

// Results (compared through the final DTCM image).
volatile uint32_t res[256];
static unsigned rn;
static void put(uint32_t v) { if (rn < 256) res[rn] = v; rn++; }

// Code copied into ITCM at run time: li a0, 0x55; ret.
__attribute__((section(".itcm_buf"))) volatile uint32_t itcm_buf[4];

// Unit of an opcode for the bench's CAPS (VU | ATT) and queue configuration (CC §7.4).
static unsigned unit_of(uint32_t op) {
    switch (op) {
    case TSR_OP_LD: case TSR_OP_ST: case TSR_OP_KVST: return TSR_UNIT_DMA;
    case TSR_OP_QACT: return TSR_UNIT_QACT;
    case TSR_OP_MVM: case TSR_OP_ATTQK: case TSR_OP_ATTPV: return TSR_UNIT_MVM;
    case TSR_OP_VOP: case TSR_OP_RED: case TSR_OP_ROPE: case TSR_OP_SOP:
    case TSR_OP_ASMAX: case TSR_OP_AMASK: case TSR_OP_AMERGE: case TSR_OP_ANORM: return TSR_UNIT_VU;
    default: return TSR_UNIT_SYNC;    // NOP, FENCE, refused (TOPK, DQROW, VMASK without CAPS, reserved)
    }
}

static uint32_t push(const uint32_t * w, uint32_t dep0, uint32_t dep1, uint32_t pv) {
    const unsigned u = unit_of(w[0] & 0xFFu);
    while (((CW(TSR_CW_ISSUE_CREDIT) >> (4u * u)) & 0xFu) == 0) {
    }
    for (unsigned i = 0; i < 8; ++i) CW(TSR_CW_ISSUE_W(i)) = w[i];
    CW(TSR_CW_ISSUE_DEP0) = dep0;
    CW(TSR_CW_ISSUE_DEP1) = dep1;
    CW(TSR_CW_ISSUE_PUSH) = pv;
    const uint32_t t = CW(TSR_CW_ISSUE_TICKET);
    put(t);
    put(CW(TSR_CW_ISSUE_PIDX));
    put(CW(TSR_CW_ISSUE_DEP0));
    put(CW(TSR_CW_ISSUE_DEP1));
    return t;
}

static void tcm_tests(void) {
    static volatile uint32_t buf[64];
    volatile uint8_t * b8 = (volatile uint8_t *)buf;
    volatile uint16_t * b16 = (volatile uint16_t *)buf;
    uint32_t h = 0x811C9DC5u;
    for (unsigned i = 0; i < 64; ++i) buf[i] = (i * 0x9E3779B9u) ^ (i << 7);
    for (unsigned i = 0; i < 32; ++i) b8[3 * i + 1] = (uint8_t)(i * 37u);
    for (unsigned i = 0; i < 16; ++i) b16[5 * i + 3] = (uint16_t)(i * 4099u);
    for (unsigned i = 0; i < 61; ++i) {
        // misaligned word loads and stores (Ibex splits them, CC §4.1 / tsr_rv.h)
        volatile uint32_t * p = (volatile uint32_t *)(void *)(b8 + 4 * i + (i % 4));
        uint32_t v = *p;
        h = (h ^ v) * 0x01000193u;
        if (i % 7 == 3) *p = v + i;
    }
    for (unsigned i = 0; i < 64; ++i) h = (h ^ buf[i]) * 0x01000193u;
    put(h);
    {   // a load of the word stored by the instruction before (back to back on the bus, where the
        // store's SRAM write and the load's read share a cycle: the bus forwards the store)
        volatile uint32_t * q = &buf[5];
        uint32_t r0, r1, r2;
        __asm__ volatile("sw %3, 0(%4)\n\tlw %0, 0(%4)\n\t"
                         "sb %3, 1(%4)\n\tlw %1, 0(%4)\n\t"
                         "sh %5, 2(%4)\n\tlhu %2, 2(%4)"
                         : "=&r"(r0), "=&r"(r1), "=&r"(r2) : "r"(0x11223344u), "r"(q), "r"(0xBEEFu) : "memory");
        put(r0); put(r1); put(r2);
    }
    for (unsigned i = 0; i < 24; ++i) {
        int32_t a = (int32_t)(buf[i] ^ 0x80000000u), b = (int32_t)buf[63 - i] >> (i % 31);
        if (i == 5) b = 0;
        if (i == 6) { a = (int32_t)0x80000000u; b = -1; }
        uint32_t m;
        __asm__ volatile("mulh %0, %1, %2" : "=r"(m) : "r"(a), "r"(b));
        h ^= m;
        __asm__ volatile("mulhsu %0, %1, %2" : "=r"(m) : "r"(a), "r"(b));
        h += m;
        __asm__ volatile("mulhu %0, %1, %2" : "=r"(m) : "r"(a), "r"(b));
        h ^= m * 3u;
        __asm__ volatile("div %0, %1, %2" : "=r"(m) : "r"(a), "r"(b));
        h += m;
        __asm__ volatile("divu %0, %1, %2" : "=r"(m) : "r"(a), "r"(b));
        h ^= m;
        __asm__ volatile("rem %0, %1, %2" : "=r"(m) : "r"(a), "r"(b));
        h += m;
        __asm__ volatile("remu %0, %1, %2" : "=r"(m) : "r"(a), "r"(b));
        h ^= m;
        h *= (uint32_t)a | 1u;
    }
    put(h);
    // ITCM data reads (CC §4.1: they steal the fetch port), then code written to ITCM and run.
    uintptr_t ca = 0x80u;                       // the reset entry (crt0.S)
    __asm__ volatile("" : "+r"(ca));
    const volatile uint32_t * code = (const volatile uint32_t *)ca;
    for (unsigned i = 0; i < 8; ++i) put(code[i]);
    itcm_buf[0] = 0x05500513u;   // li a0, 0x55
    itcm_buf[1] = 0x00008067u;   // ret
    __asm__ volatile("fence.i" ::: "memory");
    uint32_t (*fn)(void) = (uint32_t (*)(void))(uintptr_t)&itcm_buf[0];
    put(fn());
    uint32_t v;
    __asm__ volatile("csrr %0, misa" : "=r"(v));
    put(v);
    __asm__ volatile("csrr %0, mhartid" : "=r"(v));
    put(v);
    __asm__ volatile("csrr %0, mstatus" : "=r"(v));
    put(v);
}

static void fault_tests(void) {
    uint32_t v;
    CWB(TSR_CW_ISSUE_W(0)) = 0x5A;                                  // byte store: fault
    v = CWH(TSR_CW_ISSUE_W(1));                                     // half load: fault
    put(v);
    CORE(TSR_CSR_ID) = 1;                                           // core-window store: fault
    v = *(volatile uint32_t *)(uintptr_t)0x30000000u;               // unmapped: fault
    put(v);
    uintptr_t a = TSR_RV_CTL_BASE + 2u;                             // misaligned window word: Ibex
    __asm__ volatile("lw %0, 0(%1)" : "=r"(v) : "r"(a));            // splits it, the first half faults
    put(v);
    v = *(volatile uint32_t *)(uintptr_t)(TSR_RV_SPM_BASE + TSR_RV_SPM_SIZE);   // past the SPM: fault
    put(v);
    __asm__ volatile("ebreak");
    __asm__ volatile("ecall");
    __asm__ volatile(".word 0x00000000");                           // illegal instruction
    // fetch from DTCM (not executable): instruction access fault, the handler returns to ra
    __asm__ volatile("la t0, 1f\n\tmv t1, ra\n\tmv ra, t0\n\tli t0, 0x00100000\n\tjr t0\n1:\n\tmv ra, t1"
                     ::: "t0", "t1", "memory");
    put(trap_n);
}

static void window_tests(void) {
    // reset values and constants
    put(CW(TSR_CW_ISSUE_UNITS));
    put(CW(TSR_CW_ISSUE_PIDX));
    put(CW(TSR_CW_ISSUE_TICKET));
    put(CW(TSR_CW_ISSUE_DEP0));
    put(CW(TSR_CW_WAIT_TICKET));
    put(CW(TSR_CW_WAIT_CREDIT));
    put(CW(TSR_CW_MTIMECMP_LO));
    put(CW(TSR_CW_MTIMECMP_HI));
    for (unsigned u = 0; u < 8; ++u) put(CW(TSR_CW_ISSUED_SEQ(u)));
    // reserved offsets read 0
    put(CW(0x150) | CW(0x1C0) | CW(0x310) | CW(0x800) | CW(0xFFC) | CW(0x2FC) | CW(0x060));
    // staging words read back
    for (unsigned i = 0; i < 8; ++i) CW(TSR_CW_ISSUE_W(i)) = 0xA0000000u + i * 0x01010101u;
    for (unsigned i = 0; i < 8; ++i) put(CW(TSR_CW_ISSUE_W(i)));
    // timing-dependent registers (the lockstep takes the RTL's values)
    uint32_t t = CW(TSR_CW_ISSUE_CREDIT) ^ CW(TSR_CW_QSTATUS) ^ CW(TSR_CW_MTIME_LO) ^ CW(TSR_CW_DONE_SEQ(0));
    t ^= CW(TSR_CW_ERR_SEQ(0)) ^ CW(TSR_CW_ERR_CODE(2)) ^ CW(TSR_CW_ERR_PIDX(3)) ^ CW(TSR_CW_ISSUE_STATUS);
    put(t);

    // pushes (CC §7.1): every unit, refused opcodes, word masks, REC, dependencies
    uint32_t w[8];
    uint32_t prev = NONE, first;
    for (unsigned i = 0; i < 8; ++i) w[i] = 0;
    first = prev = push(w, NONE, NONE, 0xFF00u);                   // NOP
    static const uint32_t ops[] = { TSR_OP_LD, TSR_OP_ST, TSR_OP_QACT, TSR_OP_MVM, TSR_OP_VOP, TSR_OP_RED,
                                    TSR_OP_SOP, TSR_OP_FENCE, 0x05u, TSR_OP_DQROW, TSR_OP_TOPK, TSR_OP_KVST,
                                    TSR_OP_ATTQK, TSR_OP_ASMAX, 0x7Eu, TSR_OP_NOP };
    for (unsigned k = 0; k < 3; ++k) {
        for (unsigned j = 0; j < sizeof(ops) / sizeof(ops[0]); ++j) {
            const unsigned n = k * 16u + j;
            w[0] = ops[j] | ((n & 0xFu) << 8) | (k == 2 ? 0x00010000u : 0u);   // x16 set in round 2
            for (unsigned i = 1; i < 8; ++i) w[i] = (n << 24) | (i << 16) | (n * 7u + i);
            const uint32_t wmask = (k == 0) ? 0xFFu : (k == 1) ? (1u << (j % 8)) : (uint32_t)(n * 37u) & 0xFFu;
            const uint32_t dep1 = (j % 3 == 0) ? NONE : (j % 3 == 1) ? first : 0x06000001u;  // reserved unit
            prev = push(w, prev, dep1, (wmask << 8) | (n & 1u) | 0xFFFF0000u);
        }
    }
    // a push without a credit check: the safety net waits (CC §7.3)
    w[0] = TSR_OP_NOP;
    for (unsigned i = 1; i < 8; ++i) w[i] = 0;
    for (unsigned i = 0; i < 8; ++i) CW(TSR_CW_ISSUE_W(i)) = w[i];
    for (unsigned r = 0; r < 6; ++r) CW(TSR_CW_ISSUE_PUSH) = 0x100u;
    put(CW(TSR_CW_ISSUE_TICKET));
    put(CW(TSR_CW_ISSUE_PIDX));

    // QCTRL (one entry per set bit, in bit order; QCLEAR ticketed) and TRACE_MARK
    CW(TSR_CW_QCTRL) = 0x2u;
    put(CW(TSR_CW_ISSUE_PIDX));
    CW(TSR_CW_QCTRL) = 0x4u;
    CW(TSR_CW_QCTRL) = 0x6u;
    CW(TSR_CW_QCTRL) = 0x0u;
    CW(TSR_CW_QCTRL) = 0x8u;
    put(CW(TSR_CW_ISSUE_TICKET));
    put(CW(TSR_CW_ISSUE_PIDX));
    CW(TSR_CW_TRACE_MARK) = 0xC0FFEE01u;
    CW(TSR_CW_QCTRL) = 0xFu;
    put(CW(TSR_CW_ISSUE_TICKET));
    put(CW(TSR_CW_ISSUE_PIDX));
    CW(TSR_CW_TRACE_MARK) = 0xC0FFEE02u;
    CW(TSR_CW_QCTRL) = 0x8u;
    for (unsigned u = 0; u < 8; ++u) put(CW(TSR_CW_ISSUED_SEQ(u)));
    put(CW(TSR_CW_ISSUE_PIDX));

    // WFI on WAIT_TICKET (irq_fast[0], CC §12.3) for the last ticket
    w[0] = TSR_OP_NOP;
    for (unsigned i = 1; i < 8; ++i) w[i] = 0;
    const uint32_t last = push(w, NONE, NONE, 0xFF00u);
    CW(TSR_CW_WAIT_TICKET) = last;
    put(CW(TSR_CW_WAIT_TICKET));
    __asm__ volatile("csrw mie, %0" :: "r"(1u << 16));
    __asm__ volatile("wfi");
    __asm__ volatile("csrw mie, zero");
    CW(TSR_CW_WAIT_TICKET) = NONE;
    put(CW(TSR_CW_DONE_SEQ(TSR_UNIT_SYNC)) - (last & 0x0FFFFFFFu));   // RTL value; > 0 after the wake-up

    // WFI on the timer (irq_timer while MTIME >= MTIMECMP, CC §5.2) and on WAIT_CREDIT (irq_fast[5])
    {
        const uint32_t lo = CW(TSR_CW_MTIME_LO), hi = CW(TSR_CW_MTIME_HI);   // the RTL's values
        const uint64_t when = (((uint64_t)hi << 32) | lo) + 3000u;
        CW(TSR_CW_MTIMECMP_HI) = 0xFFFFFFFFu;
        CW(TSR_CW_MTIMECMP_LO) = (uint32_t)when;
        CW(TSR_CW_MTIMECMP_HI) = (uint32_t)(when >> 32);
        __asm__ volatile("csrw mie, %0" :: "r"(1u << 7));
        __asm__ volatile("wfi");
        __asm__ volatile("csrw mie, zero");
        CW(TSR_CW_WAIT_CREDIT) = 0x80000000u | TSR_UNIT_DMA;
        __asm__ volatile("csrw mie, %0" :: "r"(1u << (16 + 5)));
        __asm__ volatile("wfi");
        __asm__ volatile("csrw mie, zero");
        CW(TSR_CW_WAIT_CREDIT) = 0;
        CW(TSR_CW_MTIMECMP_LO) = 0xFFFFFFFFu;       // (mip is timing-dependent: not read here)
        CW(TSR_CW_MTIMECMP_HI) = 0xFFFFFFFFu;
    }

    // WAIT_CREDIT, MTIMECMP, ISSUE_STATUS, SPM_LB_INV
    CW(TSR_CW_WAIT_CREDIT) = 0xFFFFFFF3u;
    put(CW(TSR_CW_WAIT_CREDIT));
    CW(TSR_CW_WAIT_CREDIT) = 0;
    CW(TSR_CW_MTIMECMP_LO) = 0x12345678u;
    CW(TSR_CW_MTIMECMP_HI) = 0x9ABCDEF0u;
    put(CW(TSR_CW_MTIMECMP_LO));
    put(CW(TSR_CW_MTIMECMP_HI));
    CW(TSR_CW_MTIMECMP_LO) = 0xFFFFFFFFu;
    CW(TSR_CW_MTIMECMP_HI) = 0xFFFFFFFFu;
    CW(TSR_CW_ISSUE_STATUS) = 0x3u;
    CW(TSR_CW_SPM_LB_INV) = 1u;
    CW(TSR_CW_ISSUE_UNITS) = 0xFFu;                // read-only: ignored
    put(CW(TSR_CW_ISSUE_UNITS));

    // LQ / LCQ / log / RT (EX §4.5)
    for (unsigned i = 0; i < 8; ++i) CW(TSR_CW_LCQ_W(i)) = 0xEC000000u + i;
    for (unsigned i = 0; i < 8; ++i) put(CW(TSR_CW_LCQ_W(i)));
    CW(TSR_CW_LQ_POP) = 0;                          // LQ empty: fault
    CW(TSR_CW_LCQ_POST) = 0;                        // event record
    CW(TSR_CW_LCQ_W(0)) = 0xEC0000FFu;
    CW(TSR_CW_LCQ_POST) = 0;                        // a second one
    CW(TSR_CW_LCQ_POST) = 1;                        // ANSWER while not ACTIVE: fault
    CW(TSR_CW_LOG_PUSH) = 0x10C00001u;
    CW(TSR_CW_LOG_PUSH) = 0x10C00002u;
    CW(TSR_CW_LOG_PUSH) = 0x10C00003u;
    t = CW(TSR_CW_LQ_AVAIL) ^ CW(TSR_CW_LQ_HEAD_SEQ) ^ CW(TSR_CW_LQ_FLAGS) ^ CW(TSR_CW_LCQ_FREE);
    t ^= CW(TSR_CW_LOG_FREE) ^ CW(TSR_CW_LOG_CFG) ^ CW(TSR_CW_LQ_HEAD_W(0)) ^ CW(TSR_CW_LQ_CANCEL);
    t ^= CW(TSR_CW_LQ_STOP);
    put(t);
    put(CW(TSR_CW_LQ_POP) | CW(TSR_CW_LCQ_POST) | CW(TSR_CW_LOG_PUSH));   // write-only: read 0
    CW(TSR_CW_RT_STATE) = 0x00010003u;
    CW(TSR_CW_RT_INFO) = 0x00001234u;
    put(CW(TSR_CW_RT_STATE));
    put(CW(TSR_CW_RT_INFO));

    // core window (CC §5.3)
    put(CORE(TSR_CSR_LOAD_DATA));                   // 0 without an access
    put(CORE(TSR_CSR_DMI_BASE) | CORE(TSR_CSR_DMI_BASE + 0x1FCu));
    t = CORE(TSR_CSR_ID) ^ CORE(TSR_CSR_STATUS) ^ CORE(0x9C0u);
    put(t);
}

static void spm_tests(void) {
    // exact window: word / half / byte stores and loads (CC §10.5)
    for (unsigned i = 0; i < 16; ++i) SPMW(64u * 3u + 4u * i) = 0x53000000u | (i * 0x111u);
    SPMH(64u * 4u + 2u) = 0xBEEFu;
    SPMB(64u * 4u + 9u) = 0x7Cu;
    SPMB(64u * 4u + 63u) = 0x81u;
    uint32_t h = 0;
    for (unsigned i = 0; i < 16; ++i) h = h * 31u + SPMW(64u * 4u + 4u * i);
    put(h);
    put(SPMH(64u * 4u + 2u));
    put(SPMB(64u * 4u + 9u));
    uint32_t v;
    uintptr_t a = TSR_RV_SPM_BASE + 64u * 3u + 6u;                  // misaligned: split
    __asm__ volatile("lw %0, 0(%1)" : "=r"(v) : "r"(a));
    put(v);
    a = TSR_RV_SPM_BASE + 64u * 5u + 61u;                           // crosses two lines
    __asm__ volatile("sw %0, 0(%1)" :: "r"(0x01020304u), "r"(a) : "memory");
    put(SPMW(64u * 5u + 60u));
    put(SPMW(64u * 6u));
    // line-buffered alias: fill, hits, invalidation by an SPM store, by a control-window access
    h = 0;
    for (unsigned i = 0; i < 16; ++i) h = h * 33u + LBW(64u * 3u + 4u * i);
    put(h);
    put(LBB(64u * 3u + 5u));
    SPMW(64u * 3u + 8u) = 0xA11A5ED1u;              // exact-window store: invalidates
    put(LBW(64u * 3u + 8u));
    put(LBW(64u * 7u));                             // other line: fill
    SPMW(64u * 7u) = 0x77777777u;
    (void)CW(TSR_CW_ISSUE_UNITS);                   // control-window access: invalidates
    put(LBW(64u * 7u));
    *(volatile uint32_t *)(uintptr_t)(TSR_RV_SPM_LB_BASE + 64u * 7u + 4u) = 0x0BADF00Du;  // store via the alias
    put(LBW(64u * 7u + 4u));
    put(SPMW(64u * 7u + 4u));
}

// Mode word (in .data: the bench may overwrite it through the load port before the boot):
//   0 the lockstep program (everything above), 1 DMI target (a loop with x5 = 0x12345678, x6 a
//   counter), 2 push safety net (pushes without credit checks; the bench stops the dispatcher),
//   3 two SPM-window loads (the bench silences the bridge for the first), 4 a push that waits
//   until the RISC-V reset drops it, 5 line-buffer invalidation events, 6 a full issue FIFO.
__attribute__((section(".data"))) volatile uint32_t fw_mode = 0;
volatile uint32_t flag5;                            // mode 5: stepped by the bench (SBA)   // not .bss: crt0 zeroes that

static void mode_pushes(unsigned n) {
    for (unsigned i = 0; i < 8; ++i) CW(TSR_CW_ISSUE_W(i)) = 0;
    for (unsigned i = 0; i < n; ++i) {
        CW(TSR_CW_ISSUE_PUSH) = 0xFF00u;
        put(CW(TSR_CW_ISSUE_PIDX));
        put(trap_n);
    }
    put(CW(TSR_CW_ISSUE_STATUS));
    CW(TSR_CW_ISSUE_STATUS) = 1u;
    put(CW(TSR_CW_ISSUE_STATUS));
}

int main(void) {
    CW(TSR_CW_RT_STATE) = 1;
    switch (fw_mode) {
    case 1:
        __asm__ volatile("li x5, 0x12345678\n\tli x6, 0\n1:\n\taddi x6, x6, 1\n\tj 1b" ::: "x5", "x6");
        break;
    case 2:
        mode_pushes(6);
        CW(TSR_CW_RT_STATE) = 0x00000D0Eu;
        return 0;
    case 3:
        put(SPMW(64u * 9u));
        put(trap_n);
        put(SPMW(64u * 10u + 4u));                  // another line: a late answer must not be taken
        put(trap_n);
        CW(TSR_CW_RT_STATE) = 0x00000D0Eu;
        return 0;
    case 4:
        CW(TSR_CW_RT_STATE) = 0x00000004u;
        mode_pushes(8);
        CW(TSR_CW_RT_STATE) = 0x00000D0Eu;
        return 0;
    case 5: {
        // line-buffer invalidation events (CC §10.5): the bench changes SPM lines behind the
        // hart's back and steps flag5 through SBA (DTCM writes: no invalidation event)
        uint32_t v1 = LBW(64u * 8u);                // fill
        while (flag5 == 0) {
        }
        uint32_t v2 = LBW(64u * 8u);                // a hit: the old value
        (void)CW(TSR_CW_ISSUE_UNITS);               // control-window access: invalidates
        uint32_t v3 = LBW(64u * 8u);                // refilled: the new value
        uint32_t w1 = LBW(64u * 9u);                // fill
        while (flag5 == 1) {
        }
        uint32_t w2 = LBW(64u * 9u);                // after a rising interrupt line: refilled
        uint32_t x1 = LBW(64u * 11u);               // fill
        while (flag5 == 2) {
        }
        uint32_t x2 = LBW(64u * 11u);               // no event (an SBA DTCM write only): a hit
        put(v1); put(v2); put(v3); put(w1); put(w2); put(x1); put(x2);
        CW(TSR_CW_RT_STATE) = 0x00000D0Eu;
        return 0;
    }
    case 6: {
        // issue-FIFO credit limit (CC §7.3): with the dispatcher stopped, 32 pushes fill the FIFO
        // while the queues still have credits (VU 16 + DMA 8 + QACT 4 + MVM 4); the 33rd waits
        static const uint32_t op6[4] = { TSR_OP_VOP, TSR_OP_LD, TSR_OP_QACT, TSR_OP_MVM };
        static const unsigned n6[4] = { 16, 8, 4, 4 };
        for (unsigned i = 1; i < 8; ++i) CW(TSR_CW_ISSUE_W(i)) = 0;
        for (unsigned k = 0; k < 4; ++k) {
            CW(TSR_CW_ISSUE_W(0)) = op6[k];
            for (unsigned i = 0; i < n6[k]; ++i) CW(TSR_CW_ISSUE_PUSH) = 0x0100u;
        }
        put(CW(TSR_CW_ISSUE_PIDX));
        CW(TSR_CW_ISSUE_W(0)) = TSR_OP_NOP;
        CW(TSR_CW_ISSUE_PUSH) = 0x0100u;            // waits for FIFO space (SYNC has 4 credits)
        put(CW(TSR_CW_ISSUE_PIDX));
        put(CW(TSR_CW_ISSUE_TICKET));
        CW(TSR_CW_RT_STATE) = 0x00000D0Eu;
        return 0;
    }
    default:
        break;
    }
    tcm_tests();
    fault_tests();
    window_tests();
    spm_tests();
    for (unsigned i = 0; i < 16; ++i) put(trap_log[i][0] ^ (trap_log[i][1] << 1) ^ (trap_log[i][2] << 2));
    put(trap_n);
    uint32_t v;
    __asm__ volatile("csrr %0, minstret" : "=r"(v));
    put(v);
    put(rn);
    CW(TSR_CW_RT_STATE) = 0x00000D0Eu;              // done
    return 0;
}

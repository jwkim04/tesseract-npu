// fw_race.c — the RISC-V program of sim_race_rv (sw/sim/tests/test_sim_race.c): a minimal EX §4.7
// runtime (as tests/rv_issue/fw_issue.c, same crt0.S and link.ld) whose kernels are the race
// detector's RISC-V-side scenarios (SW §4.6, P1.9). Built by the pinned xPack compiler
// (sw/sim/race.cmake). The host checks what the detector reports for each launch.
//
// Runtime: R1 boot (QCLEAR + join, HELLO), R2 pop, R5 run kernel KERNEL (w0[31:16]) with ARG0..6,
// R6 join_all (FENCE, then DONE_SEQ polling: an observation of every op of the launch; skipped for
// kernel 12), R8 answer DONE with r2 = RET, R9 QCLEAR if a unit is in error. PING answers PONG.
//
// Kernels (SPM lines and DRAM handles {ch[31:28], beat[27:0]} from the descriptor):
//   0 store_push      RISC-V stores line A (16 words), then pushes ST A -> y: ordered (E4), fresh
//   1 read_nowait     push LD x -> B, read B at once: unordered with the LD (race) and stale
//   2 read_doneseq    push LD x -> B, poll DONE_SEQ until its ticket is done, read B: ordered
//   3 read_waitirq    push LD x -> B, WAIT_TICKET + one WFI on irq_fast[0], read B: ordered (E5)
//   4 op_stale        push ST C -> y without writing C in the launch: SPM_STALE (op)
//   5 op_fresh        push LD x -> C, ST C -> y after it: neither stale nor racy
//   6 ldst            push LD x -> D (n lines), ST D -> y after it (ARG3 = n): the host's H1 / H2 cases
//   7 rv_stale        read line E word 0 (not written in the launch: SPM_STALE, RISC-V), then write
//                     word 1 and read it back (fresh)
//   8 write_after_st  RISC-V writes line F, pushes ST F -> y, then rewrites F at once: the ST's read
//                     and the second store are unordered (race)
//   9 lb_reads        write word 0 of line G, read it through the line-buffered alias (a 64-byte
//                     fill: fresh at line level), then read line H through the alias (stale)
//  10 sreg_nowait     write line G, push RED SUM G -> s0, read SREG0 through the core window at once:
//                     the RED's s0 write and the read are unordered (race on s0)
//  11 qidle_wait      push LD x -> B, poll QSTATUS until QIDLE, read B: ordered (QIDLE observes all)
//  12 ldst_nojoin     as 6, but the runtime answers without its R6 join (EX §4.7; a deliberate
//                     runtime bug): the RISC-V clock at the post does not cover the ST, so only
//                     H2's join of the launch's completed ops orders it before the host's y read
#include <stdint.h>

#include "tsr_isa.h"
#include "../../../ctrl/include/tsr/version.h"   // TSR_SDK_ABI, TSR_BASIS_VERSION (HELLO r4 / r6, SW §6.9)

#define CW(o)   (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CTL_BASE + (o)))
#define SPMW(l, i) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_SPM_BASE + 64u * (l) + 4u * (i)))
#define SPMLB(l, i) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_SPM_LB_BASE + 64u * (l) + 4u * (i)))
#define CORE(o) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CORE_BASE + (o)))
#define MIE_FAST(i) (1u << (16u + (i)))

void * memset(void * d, int c, __SIZE_TYPE__ n);
void * memcpy(void * d, const void * s, __SIZE_TYPE__ n);
void * memset(void * d, int c, __SIZE_TYPE__ n) {
    volatile uint8_t * p = (volatile uint8_t *)d;
    while (n--) *p++ = (uint8_t)c;
    return d;
}
void * memcpy(void * d, const void * s, __SIZE_TYPE__ n) {
    volatile uint8_t * p = (volatile uint8_t *)d;
    const uint8_t * q = (const uint8_t *)s;
    while (n--) *p++ = *q++;
    return d;
}

static inline void set_mie(uint32_t v) { __asm__ volatile("csrw mie, %0" : : "r"(v) : "memory"); }
static inline void wfi(void) { __asm__ volatile("wfi" : : : "memory"); }

static uint32_t credit(uint32_t u) { return (CW(TSR_CW_ISSUE_CREDIT) >> (TSR_CREDIT_FIELD_W * u)) & 0xFu; }

static uint32_t push(const tsr_insn * i, uint32_t dep0, uint32_t dep1) {
    const uint32_t u = tsr_op_unit(i->w[0]);
    while (credit(u) == 0u) {}
    for (uint32_t k = 0; k < TSR_INSN_WORDS; ++k) CW(TSR_CW_ISSUE_W(k)) = i->w[k];
    CW(TSR_CW_ISSUE_DEP0) = dep0;
    CW(TSR_CW_ISSUE_DEP1) = dep1;
    CW(TSR_CW_ISSUE_PUSH) = TSR_ISSUE_PUSH_WMASK;
    return CW(TSR_CW_ISSUE_TICKET);
}

static int ticket_done(uint32_t t) {
    const uint32_t u = TSR_GET(t, TSR_TICKET_UNIT), s = t & TSR_TICKET_SEQ;
    const uint32_t diff = (CW(TSR_CW_DONE_SEQ(u)) - s) & TSR_TICKET_SEQ;
    return diff >= 1u && diff < 0x08000000u;
}

static void join_poll(uint32_t t) { while (!ticket_done(t)) {} }

static void join_all(void) {
    const tsr_insn f = tsr_enc_fence();
    join_poll(push(&f, TSR_TICKET_NONE, TSR_TICKET_NONE));
}

static void qclear(void) {
    CW(TSR_CW_QCTRL) = TSR_QCTRL_QCLEAR;
    join_poll(CW(TSR_CW_ISSUE_TICKET));
}

static void post(const uint32_t r[8], uint32_t answer) {
    while (CW(TSR_CW_LCQ_FREE) == 0u) {}
    for (uint32_t k = 0; k < 8u; ++k) CW(TSR_CW_LCQ_W(k)) = r[k];
    CW(TSR_CW_LCQ_POST) = answer ? TSR_LCQ_POST_ANSWER : 0u;
}

static uint32_t hch(uint32_t h) { return TSR_GET(h, TSR_TICKET_UNIT); }

static tsr_insn enc_red_sum_s(uint32_t sd, uint32_t a, uint32_t m) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_RED;
    TSR_INSN_SET(i.w, TSR_RED_KIND, TSR_RED_KIND_SUM);
    TSR_INSN_SET(i.w, TSR_RED_TO_S, 1u);
    TSR_INSN_SET(i.w, TSR_RED_SD, sd);
    TSR_INSN_SET(i.w, TSR_RED_POST, TSR_RED_POST_NONE);
    TSR_INSN_SET(i.w, TSR_RED_A, a);
    TSR_INSN_SET(i.w, TSR_RED_M, m);
    TSR_INSN_SET(i.w, TSR_RED_ROWS, 1u);
    return i;
}
static uint32_t hbeat(uint32_t h) { return h & 0x0FFFFFFFu; }

static int32_t kernel(uint32_t k, const uint32_t * a) {
    const uint32_t none = TSR_TICKET_NONE;
    tsr_insn i;
    uint32_t t, v = 0;
    switch (k) {
    case 0:                                                // store_push
        for (uint32_t w = 0; w < 16u; ++w) SPMW(a[0], w) = 0x1000u + w;
        i = tsr_enc_st(hch(a[1]), a[0], hbeat(a[1]), 1);
        (void)push(&i, none, none);
        return 0;
    case 1:                                                // read_nowait
        i = tsr_enc_ld(hch(a[1]), a[0], hbeat(a[1]), 1);
        (void)push(&i, none, none);
        return (int32_t)(SPMW(a[0], 0) & 0xFFFFu);
    case 2:                                                // read_doneseq
        i = tsr_enc_ld(hch(a[1]), a[0], hbeat(a[1]), 1);
        join_poll(push(&i, none, none));
        return (int32_t)(SPMW(a[0], 0) & 0xFFFFu);
    case 3:                                                // read_waitirq
        i = tsr_enc_ld(hch(a[1]), a[0], hbeat(a[1]), 1);
        t = push(&i, none, none);
        CW(TSR_CW_WAIT_TICKET) = t;                        // irq_fast[0] while t is done
        set_mie(MIE_FAST(0));
        wfi();                                             // mstatus.MIE = 0: wakes without a trap
        set_mie(0);
        v = SPMW(a[0], 0);
        CW(TSR_CW_WAIT_TICKET) = none;
        return (int32_t)(v & 0xFFFFu);
    case 4:                                                // op_stale
        i = tsr_enc_st(hch(a[1]), a[0], hbeat(a[1]), 1);
        (void)push(&i, none, none);
        return 0;
    case 5:                                                // op_fresh
        i = tsr_enc_ld(hch(a[1]), a[0], hbeat(a[1]), 1);
        t = push(&i, none, none);
        i = tsr_enc_st(hch(a[2]), a[0], hbeat(a[2]), 1);
        (void)push(&i, t, none);
        return 0;
    case 6:                                                // ldst
        i = tsr_enc_ld(hch(a[0]), a[2], hbeat(a[0]), a[3]);
        t = push(&i, none, none);
        i = tsr_enc_st(hch(a[1]), a[2], hbeat(a[1]), a[3]);
        (void)push(&i, t, none);
        return 0;
    case 7:                                                // rv_stale
        v = SPMW(a[0], 0);
        SPMW(a[0], 1) = 0x77u;
        return (int32_t)((v + SPMW(a[0], 1)) & 0xFFFFu);
    case 8:                                                // write_after_st
        for (uint32_t w = 0; w < 16u; ++w) SPMW(a[0], w) = 0x2000u + w;
        i = tsr_enc_st(hch(a[1]), a[0], hbeat(a[1]), 1);
        (void)push(&i, none, none);
        SPMW(a[0], 0) = 0xDEADu;
        return 0;
    case 9:                                                // lb_reads
        SPMW(a[0], 0) = 0x99u;
        v = SPMLB(a[0], 0);
        v += SPMLB(a[1], 0);
        return (int32_t)(v & 0xFFFFu);
    case 10:                                               // sreg_nowait
        for (uint32_t w = 0; w < 16u; ++w) SPMW(a[0], w) = 0x3F800000u;   // 1.0f
        i = enc_red_sum_s(0, a[0], 16);
        (void)push(&i, none, none);
        return (int32_t)(CORE(TSR_CSR_SREG(0)) & 0xFFFFu);
    case 11:                                               // qidle_wait
        i = tsr_enc_ld(hch(a[1]), a[0], hbeat(a[1]), 1);
        (void)push(&i, none, none);
        while (!(CW(TSR_CW_QSTATUS) & TSR_QSTATUS_QIDLE)) {}
        return (int32_t)(SPMW(a[0], 0) & 0xFFFFu);
    case 12:                                               // ldst_nojoin (no R6 join: main)
        i = tsr_enc_ld(hch(a[0]), a[2], hbeat(a[0]), a[3]);
        t = push(&i, none, none);
        i = tsr_enc_st(hch(a[1]), a[2], hbeat(a[1]), a[3]);
        (void)push(&i, t, none);
        return 0;
    default:
        return -1;
    }
}

int main(void);
int main(void) {
    CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_BOOT;
    qclear();
    CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_IDLE;
    // r4 Basis ABI | the signature of an empty kind table (tsr/opl.hpp kinds_sig), r6 the Basis version
    const uint32_t hello[8] = { TSR_LCQ_R0_TYPE_HELLO, CW(TSR_CW_LQ_HEAD_SEQ), 0x5EED0002u, 0u, (0xB51Bu << 16) | TSR_SDK_ABI,
                                9u, TSR_BASIS_VERSION, 0u };
    post(hello, 0);
    for (;;) {
        set_mie(MIE_FAST(2));
        while (CW(TSR_CW_LQ_AVAIL) == 0u) wfi();
        set_mie(0);
        uint32_t d[8];
        for (uint32_t k = 0; k < 8u; ++k) d[k] = CW(TSR_CW_LQ_HEAD_W(k));
        const uint32_t seq = CW(TSR_CW_LQ_HEAD_SEQ);
        CW(TSR_CW_LQ_POP) = 1u;
        uint32_t r[8] = { 0u, seq, 0u, TSR_TICKET_NONE, 0u, 0u, 0u, 0u };
        const uint32_t type = TSR_GET(d[0], TSR_DESC_TYPE);
        if (type == TSR_DESC_TYPE_PING) {
            r[0] = TSR_LCQ_R0_TYPE_PONG;
            r[2] = d[1];
        } else {
            CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_KERNEL;
            const uint32_t kn = TSR_GET(d[0], TSR_DESC_KERNEL);
            const int32_t ret = kernel(kn, d + 1);
            CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_JOIN;
            if (kn != 12u) join_all();                     // kernel 12: R6 broken on purpose
            r[0] = TSR_LCQ_R0_TYPE_DONE | TSR_PUT(TSR_LCQ_R0_STATUS, ret < 0 ? TSR_LCQ_R0_STATUS_KERNEL_ERR : 0u);
            r[2] = (uint32_t)ret;
        }
        r[0] |= TSR_PUT(TSR_LCQ_R0_SEQ, seq & 0xFFu);
        CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_ANSWER;
        post(r, 1);
        if (TSR_GET(CW(TSR_CW_QSTATUS), TSR_QSTATUS_ERR) != 0u) qclear();
        CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_IDLE;
    }
}

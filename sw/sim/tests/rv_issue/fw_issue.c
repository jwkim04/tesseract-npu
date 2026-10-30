// fw_issue.c — the RISC-V program of sim_rv_issue (sw/sim/tests/test_sim_rv_issue.c): a minimal
// runtime after EX §4.7 with one kernel, compiled by the pinned xPack riscv-none-elf-gcc for the
// control core (SW §6.1 flags, C11 instead of C++20) and run on tsr_rv inside the simulator.
// Everything it knows about the device comes from the generated tables (tsr_tables.h through
// tsr_isa.h, which is freestanding: <stdint.h> only) and the tile-op encoders of tsr_isa.h.
//
// Runtime (EX §4.7):
//   R1  RT_STATE BOOT, QCTRL.QCLEAR and a join on its ticket, RT_STATE IDLE, HELLO event
//   R2  wait for LQ_AVAIL (WFI on irq_fast[2]), copy LQ_HEAD_W*, LQ_HEAD_SEQ, LQ_POP
//   R4  PING -> PONG (r2 = ARG0); unknown TYPE, reserved bits or KERNEL beyond the table -> BAD_DESC
//   R5  RT_STATE KERNEL | index, ISSUE_PIDX as the launch's first push index, call the kernel
//   R6  join: FENCE, then WAIT_TICKET + WFI on irq_fast[0]
//   R7  first error = the smallest ERR_PIDX (relative to the first push index) over the units in
//       error (QSTATUS); STATUS by EX §4.4's precedence (TILE_ERR / ABORTED > PLAT_ERR >
//       KERNEL_ERR > OK); r7 = MTIME from the pop to the post
//   R8  wait for LCQ_FREE (WFI on irq_fast[3]), post with ANSWER
//   R9  QCLEAR and a join if a unit is in error; R10 RT_STATE IDLE
// Not here (the SDK's job, W2): cancel / stop (R3), PROF, argument blocks, image id, TRAP records
// (crt0 parks with RT_STATE TRAPPED), FMT log records.
//
// Kernel 0 "mvm_act", one token: y = act(W x), s0 = sum(y), y stored to DRAM.
//   ARG0 x handle {ch[31:28], beat[27:0]}: K f32 values      ARG3 N
//   ARG1 W_OFF beat of the repacked N x K weights            ARG4 K
//   ARG2 y handle {ch, beat}: ceil(N/16) lines out           ARG5 fmt[7:0], QA_LINES(kind, K)[31:16]
//   ARG6 VOP function [7:0] (a one-input function: cmode NONE, reads a)
//   SPM: X = line 0 (K/16 lines), QA, Y (MVM out), Z (VOP out), packed in that order.
//   Pushes (each depends on its producer; the RED and the ST both on the VOP):
//     LD X <- x;  QACT QA <- X;  MVM Y <- W . QA;  VOP Z <- act(Y);  RED s0 <- sum Z (TO_S);  ST y <- Z
//   then waits for the RED, reads s0 through the core window (CC §5.3), posts a kernel event
//   (type 0xC0: r1 SEQ, r2 s0 bits, r3 first push index, r4 ops pushed, r5 the ST's ticket) and an
//   INFO / EVENT log record (payload N, K, s0 bits), and returns RET = 6 (ops pushed). The host
//   checks every value against direct references.
#include <stdint.h>

#include "tsr_isa.h"
#include "../../../ctrl/include/tsr/version.h"   // TSR_SDK_ABI, TSR_BASIS_VERSION (HELLO r4 / r6, SW §6.9)

#define CW(o)   (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CTL_BASE + (o)))    // control window, CC §5
#define CORE(o) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CORE_BASE + (o)))   // core window, CC §5.3
#define MIE_FAST(i) (1u << (16u + (i)))                                        // irq_fast[i] (CC §12.3)

#define N_KERNELS   1u
#define KINDS_SIG   0xB51Bu          // HELLO r4[31:16] of an image without op list kinds (tsr/opl.hpp kinds_sig)
#define IMAGE_ID_LO 0x5EED0001u      // a constant instead of EX §3.2's SHA-256 id (no tsr-elf here)
#define IMAGE_ID_HI 0x0000F00Du

// Freestanding GCC may call these for aggregate copies and initialisations.
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
// mstatus.MIE stays 0: WFI wakes on (mip & mie) != 0 without taking the interrupt (SW §4.2).
static inline void wfi(void) { __asm__ volatile("wfi" : : : "memory"); }

// ---------------------------------------------------------------------------
// Issue path (CC §5.1, §7)
// ---------------------------------------------------------------------------
static uint32_t credit(uint32_t u) { return (CW(TSR_CW_ISSUE_CREDIT) >> (TSR_CREDIT_FIELD_W * u)) & 0xFu; }

// Pushes one op after checking its unit's credit (CC §7.3: software checks first; WAIT_CREDIT
// wakes it); returns the ticket.
static uint32_t push(const tsr_insn * i, uint32_t dep0, uint32_t dep1) {
    const uint32_t u = tsr_op_unit(i->w[0]);
    if (credit(u) == 0u) {
        CW(TSR_CW_WAIT_CREDIT) = TSR_WAIT_CREDIT_EN | u;
        set_mie(MIE_FAST(5));
        while (credit(u) == 0u) wfi();
        set_mie(0);
        CW(TSR_CW_WAIT_CREDIT) = 0;
    }
    for (uint32_t k = 0; k < TSR_INSN_WORDS; ++k) CW(TSR_CW_ISSUE_W(k)) = i->w[k];
    CW(TSR_CW_ISSUE_DEP0) = dep0;
    CW(TSR_CW_ISSUE_DEP1) = dep1;
    CW(TSR_CW_ISSUE_PUSH) = TSR_ISSUE_PUSH_WMASK;          // all eight words, no completion record
    return CW(TSR_CW_ISSUE_TICKET);
}

// Ticket (u, s) is done iff (DONE_SEQ[u] - s) mod 2^28 is in [1, 2^27) (CC §7.5).
static int ticket_done(uint32_t t) {
    const uint32_t u = TSR_GET(t, TSR_TICKET_UNIT), s = t & TSR_TICKET_SEQ;
    const uint32_t diff = (CW(TSR_CW_DONE_SEQ(u)) - s) & TSR_TICKET_SEQ;
    return diff >= 1u && diff < 0x08000000u;
}

static void wait_ticket(uint32_t t) {
    CW(TSR_CW_WAIT_TICKET) = t;                            // irq_fast[0] while t is done
    set_mie(MIE_FAST(0));
    while (!ticket_done(t)) wfi();
    set_mie(0);
    CW(TSR_CW_WAIT_TICKET) = TSR_TICKET_NONE;
}

static void join_all(void) {
    const tsr_insn f = tsr_enc_fence();
    wait_ticket(push(&f, TSR_TICKET_NONE, TSR_TICKET_NONE));
}

static void qclear(void) {
    CW(TSR_CW_QCTRL) = TSR_QCTRL_QCLEAR;                  // a SYNC ticket in ISSUE_TICKET
    wait_ticket(CW(TSR_CW_ISSUE_TICKET));
}

// ---------------------------------------------------------------------------
// LCQ and log (EX §4.4, §4.5, §4.10)
// ---------------------------------------------------------------------------
static void post(const uint32_t r[8], uint32_t answer) {
    set_mie(MIE_FAST(3));                                  // irq_fast[3] while LCQ_FREE != 0
    while (CW(TSR_CW_LCQ_FREE) == 0u) wfi();
    set_mie(0);
    for (uint32_t k = 0; k < 8u; ++k) CW(TSR_CW_LCQ_W(k)) = r[k];
    CW(TSR_CW_LCQ_POST) = answer ? TSR_LCQ_POST_ANSWER : 0u;
}

static uint32_t g_log_dropped;   // records dropped by this firmware (LOG_FREE too small, no BLOCK)

static void log_rec(uint32_t level, uint32_t type, uint32_t lseq, const uint32_t * p, uint32_t n) {
    const uint32_t cfg = CW(TSR_CW_LOG_CFG);
    if (!(cfg & TSR_LOG_CFG_ENABLE) || level > TSR_GET(cfg, TSR_LOG_CFG_LEVEL)) return;
    while (CW(TSR_CW_LOG_FREE) < 1u + n) {                 // never tear a record (EX §4.10)
        if (!(cfg & TSR_LOG_CFG_BLOCK)) {
            ++g_log_dropped;
            return;
        }
    }
    CW(TSR_CW_LOG_PUSH) = TSR_PUT(TSR_LOG_HDR_N, n) | TSR_PUT(TSR_LOG_HDR_LEVEL, level) |
                          TSR_PUT(TSR_LOG_HDR_SOURCE, TSR_LOG_HDR_SOURCE_KERNEL) |
                          TSR_PUT(TSR_LOG_HDR_TYPE, type) | TSR_PUT(TSR_LOG_HDR_LSEQ, lseq & 0xFFu) |
                          TSR_PUT(TSR_LOG_HDR_SYNC, TSR_LOG_HDR_SYNC_MAGIC);
    for (uint32_t k = 0; k < n; ++k) CW(TSR_CW_LOG_PUSH) = p[k];
}

// ---------------------------------------------------------------------------
// Kernel 0: mvm_act
// ---------------------------------------------------------------------------
static tsr_insn enc_vop1(uint32_t func, uint32_t d, uint32_t a, uint32_t m) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_VOP;
    TSR_INSN_SET(i.w, TSR_VOP_CMODE, TSR_VOP_CMODE_NONE);
    TSR_INSN_SET(i.w, TSR_VOP_FUNC, func);
    TSR_INSN_SET(i.w, TSR_VOP_D, d);
    TSR_INSN_SET(i.w, TSR_VOP_A, a);
    TSR_INSN_SET(i.w, TSR_VOP_M, m);
    TSR_INSN_SET(i.w, TSR_VOP_ROWS, 1u);
    return i;
}

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

static int32_t k_mvm_act(const uint32_t * a, uint32_t seq) {
    const uint32_t xh = a[0], w_off = a[1], yh = a[2], N = a[3], K = a[4];
    const uint32_t fmt = a[5] & 0xFFu, qa_lines = a[5] >> 16, func = a[6] & 0xFFu;
    const uint32_t kind = fmt == TSR_FMT_Q4_K ? TSR_QA_A256 : TSR_QA_A32;
    const uint32_t xl = K / 16u, yl = (N + 15u) / 16u;
    const uint32_t X = 0, QA = X + xl, Y = QA + qa_lines, Z = Y + yl;
    const uint32_t pidx0 = CW(TSR_CW_ISSUE_PIDX);
    const uint32_t none = TSR_TICKET_NONE;
    tsr_insn i;

    CW(TSR_CW_RT_INFO) = 0;                                // progress word: ops pushed so far
    i = tsr_enc_ld(TSR_GET(xh, TSR_TICKET_UNIT), X, xh & 0x0FFFFFFFu, xl);
    const uint32_t t_ld = push(&i, none, none);
    i = tsr_enc_qact(kind, QA, X, K, 1, 0);
    const uint32_t t_qa = push(&i, t_ld, none);
    i = tsr_enc_mvm(fmt, 0, Y, QA, w_off, N, K, 1, 0);
    const uint32_t t_mvm = push(&i, t_qa, none);
    CW(TSR_CW_RT_INFO) = 3;
    i = enc_vop1(func, Z, Y, N);
    const uint32_t t_vop = push(&i, t_mvm, none);
    i = enc_red_sum_s(0, Z, N);
    const uint32_t t_red = push(&i, t_vop, none);
    i = tsr_enc_st(TSR_GET(yh, TSR_TICKET_UNIT), Z, yh & 0x0FFFFFFFu, yl);
#ifdef FW_ISSUE_MUTANT_ST_NODEP
    // P1.9 (sw/sim/race.cmake): the mutant without the ST's dependency on the VOP; only the race
    // detector catches it (sim_race_rv). Off by default.
    const uint32_t t_st = push(&i, none, none);
#else
    const uint32_t t_st = push(&i, t_vop, none);
#endif
    CW(TSR_CW_RT_INFO) = 6;

    wait_ticket(t_red);                                    // done also when it failed or was aborted
    const uint32_t s0 = CORE(TSR_CSR_SREG(0));
    const uint32_t ev[8] = { TSR_LCQ_R0_TYPE_EVENT_FIRST | (seq & 0xFFu) << TSR_LCQ_R0_SEQ_LSB, seq, s0,
                             pidx0, 6u, t_st, 0u, 0u };
    post(ev, 0);
    const uint32_t lg[3] = { N, K, s0 };
    log_rec(TSR_LOG_HDR_LEVEL_INFO, TSR_LOG_HDR_TYPE_EVENT, seq, lg, 3);
    return 6;
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------
typedef struct { uint32_t code, ticket, info, w0, pidx; } first_err;

// R7: the error of the smallest push index since pidx0 over the units in error state.
static int first_error(uint32_t pidx0, first_err * e) {
    const uint32_t q = CW(TSR_CW_QSTATUS);
    uint32_t best = 0;
    int found = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        if (!(TSR_GET(q, TSR_QSTATUS_ERR) & (1u << u))) continue;
        const uint32_t pidx = CW(TSR_CW_ERR_PIDX(u));
        if (found && pidx - pidx0 >= best) continue;
        found = 1;
        best = pidx - pidx0;
        e->code = CW(TSR_CW_ERR_CODE(u)) & 0xFFu;
        e->ticket = TSR_PUT(TSR_TICKET_UNIT, u) | (CW(TSR_CW_ERR_SEQ(u)) & TSR_TICKET_SEQ);
        e->info = CW(TSR_CW_ERR_INFO(u));
        e->w0 = CW(TSR_CW_ERR_W0(u));
        e->pidx = pidx;
    }
    return found;
}

int main(void);
int main(void) {
    // R1
    CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_BOOT;
    qclear();
    CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_IDLE;
    const uint32_t hello[8] = { TSR_LCQ_R0_TYPE_HELLO, CW(TSR_CW_LQ_HEAD_SEQ), IMAGE_ID_LO, IMAGE_ID_HI,
                                (KINDS_SIG << 16) | TSR_SDK_ABI, N_KERNELS, TSR_BASIS_VERSION, CW(TSR_CW_MTIME_LO) };
    post(hello, 0);

    for (;;) {
        // R2
        set_mie(MIE_FAST(2));                              // LQ_AVAIL != 0 and not ACTIVE
        while (CW(TSR_CW_LQ_AVAIL) == 0u) wfi();
        set_mie(0);
        uint32_t d[8];
        for (uint32_t k = 0; k < 8u; ++k) d[k] = CW(TSR_CW_LQ_HEAD_W(k));
        const uint32_t seq = CW(TSR_CW_LQ_HEAD_SEQ);
        CW(TSR_CW_LQ_POP) = 1u;
        const uint32_t t0 = CW(TSR_CW_MTIME_LO);
        CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_DISPATCH;

        uint32_t r[8] = { 0u, seq, 0u, TSR_TICKET_NONE, 0u, 0u, 0u, 0u };
        const uint32_t type = TSR_GET(d[0], TSR_DESC_TYPE), kernel = TSR_GET(d[0], TSR_DESC_KERNEL);
        if (type == TSR_DESC_TYPE_PING && (d[0] & ~TSR_DESC_TYPE) == 0u) {          // R4
            r[0] = TSR_LCQ_R0_TYPE_PONG;
            r[2] = d[1];
        } else if (type != TSR_DESC_TYPE_LAUNCH || (d[0] & 0x0000FE00u) != 0u || kernel >= N_KERNELS) {
            r[0] = TSR_LCQ_R0_TYPE_DONE | TSR_PUT(TSR_LCQ_R0_STATUS, TSR_LCQ_R0_STATUS_BAD_DESC);
        } else {
            // R5, R6
            CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_KERNEL | TSR_PUT(TSR_RT_STATE_KERNEL, kernel);
            const uint32_t pidx0 = CW(TSR_CW_ISSUE_PIDX);
            const int32_t ret = k_mvm_act(d + 1, seq);
            CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_JOIN;
            join_all();
            // R7
            first_err e = { 0u, TSR_TICKET_NONE, 0u, 0u, 0u };
            uint32_t status = TSR_LCQ_R0_STATUS_OK;
            if (first_error(pidx0, &e)) {
                status = e.code == TSR_ERR_ABORTED ? TSR_LCQ_R0_STATUS_ABORTED : TSR_LCQ_R0_STATUS_TILE_ERR;
                r[3] = e.ticket;
                r[4] = e.info;
                r[5] = e.w0;
                r[6] = e.pidx;
            } else if (CW(TSR_CW_LQ_FLAGS) & TSR_LQ_FLAGS_PLAT_ERR) {
                status = TSR_LCQ_R0_STATUS_PLAT_ERR;
            } else if (ret < 0) {
                status = TSR_LCQ_R0_STATUS_KERNEL_ERR;
            }
            r[0] = TSR_LCQ_R0_TYPE_DONE | TSR_PUT(TSR_LCQ_R0_STATUS, status) | TSR_PUT(TSR_LCQ_R0_ERR_CODE, e.code);
            r[2] = (uint32_t)ret;
        }
        r[0] |= TSR_PUT(TSR_LCQ_R0_SEQ, seq & 0xFFu);
        // R8
        CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_ANSWER;
        r[7] = CW(TSR_CW_MTIME_LO) - t0;
        post(r, 1);
        // R9, R10
        if (TSR_GET(CW(TSR_CW_QSTATUS), TSR_QSTATUS_ERR) != 0u) qclear();
        CW(TSR_CW_RT_STATE) = TSR_RT_STATE_PHASE_IDLE;
    }
}

// fw_core.c — the tb/core smoke program: a MUL_MAT-shaped op stream pushed by the RISC-V
// through the issue path (CC §5.1, §7-§9) into the units of tsr_core, run on the RTL (Ibex) and on
// the simulator (tsr_rv inside tsr_sim) from the same image and the same arguments; the host side
// (tb/core/test_core.py) compares DRAM, SPM (dumped to DRAM by the last op), the log words, the LCQ
// record and the CSR state.
//
// Arguments: args[] (in .data, so crt0 does not clear it) is overwritten by the host through the
// loader after the ELF segments (EX §3.3); its layout is ARG_* below. Op stream (each op depends on
// its producer; deps cross units, SYNC included):
//   LD    SPM lines [0, DUMP_LINES) <- DRAM ch1 (background bytes: every dumped byte is defined)
//   LD    X   <- DRAM ch0 x (T x K f32)
//   QACT  QA1 <- X (A32, for Q8_0)            QACT QA2 <- X (A256, for Q4_K)        (both on LD)
//   MVM   Y1  <- W1 (Q8_0, N1 x K) . QA1      MVM  Y2  <- W2 (Q4_K, N2 x K) . QA2
//   ST    y1  <- Y1                           ST   y2  <- Y2
//   FENCE, then ST dump <- SPM lines [0, DUMP_LINES) (on the FENCE)
// with a join on each MVM right after its push and SPM bridge reads of the first and last line of its
// output (exact window); then a join on the last ticket (DONE_SEQ polling), the error state
// (QSTATUS), a bridge store + load back, core-window reads (CC §5.3); then an op that
// fails (LD past the end of the SPM: SPM_RANGE) and a FENCE, whose completion implies the error
// record crossed to clk_rv (CC §9.4 I2): the ERR_* registers of the DMA unit, then QCTRL.QCLEAR and a
// join on its ticket (CC §9.5); then log words, an LCQ event record (EX §4.5), RT_INFO = checksum,
// RT_STATE = DONE. Interrupts stay disabled.
#include <stdint.h>

#include "tsr_isa.h"

#define CW(o)   (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CTL_BASE + (o)))
#define CORE(o) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_CORE_BASE + (o)))
#define SPMW(a) (*(volatile uint32_t *)(uintptr_t)(TSR_RV_SPM_BASE + (a)))

#define NONE    TSR_TICKET_NONE
#define RT_DONE 0xD0Eu

enum {
    ARG_K, ARG_T, ARG_N1, ARG_N2, ARG_X_BEAT, ARG_W1_OFF, ARG_W2_OFF, ARG_X_LINE, ARG_QA1_LINE,
    ARG_QA2_LINE, ARG_Y1_LINE, ARG_Y2_LINE, ARG_X_LINES, ARG_Y1_LINES, ARG_Y2_LINES, ARG_Y1_DST,
    ARG_Y2_DST, ARG_DUMP_DST, ARG_DUMP_LINES, ARG_SCRATCH_LINE, ARG_BG_SRC, ARG_SPM_LINES,
    // GDN (CAPS bit 8; 0 = off): two heads of rows x m, inputs k, q, v, S per head from DRAM ch0 beat GDN_IN, outputs S'
    // and o per head to GDN_OUT; SPM from line GDN_SPM; alpha and beta of each head
    ARG_GDN_ON, ARG_GDN_IN, ARG_GDN_OUT, ARG_GDN_SPM, ARG_GDN_M, ARG_GDN_ROWS, ARG_GDN_AL0, ARG_GDN_BE0, ARG_GDN_AL1,
    ARG_GDN_BE1, ARG_N
};
// {ch[31:28], beat[27:0]} for the *_DST arguments.
#define DST_CH(v)   ((v) >> 28)
#define DST_BEAT(v) ((v) & 0x0FFFFFFFu)

// Freestanding GCC calls these for aggregate copies and initialisations (as fw_issue.c).
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

volatile uint32_t args[32] = { 0xA5A5A5A5u };

// Trap log of crt0.S (tb/ctrl/fw/crt0.S): mcause, mepc, mtval per exception (none expected).
volatile uint32_t trap_n;
volatile uint32_t trap_log[16][4];

// Results (also visible to the host through the RT mirrors, the log and the LCQ).
volatile uint32_t res[16];

static uint32_t credit(uint32_t u) { return (CW(TSR_CW_ISSUE_CREDIT) >> (TSR_CREDIT_FIELD_W * u)) & 0xFu; }

static uint32_t push(const tsr_insn * i, uint32_t dep0, uint32_t dep1) {
    const uint32_t u = tsr_op_unit(i->w[0]);
    while (credit(u) == 0u) {
    }
    for (uint32_t k = 0; k < TSR_INSN_WORDS; ++k) CW(TSR_CW_ISSUE_W(k)) = i->w[k];
    CW(TSR_CW_ISSUE_DEP0) = dep0;
    CW(TSR_CW_ISSUE_DEP1) = dep1;
    CW(TSR_CW_ISSUE_PUSH) = TSR_ISSUE_PUSH_WMASK;
    return CW(TSR_CW_ISSUE_TICKET);
}

// Ticket (u, s) is done iff (DONE_SEQ[u] - s) mod 2^28 is in [1, 2^27) (CC §7.5).
static int ticket_done(uint32_t t) {
    const uint32_t u = TSR_GET(t, TSR_TICKET_UNIT), s = t & TSR_TICKET_SEQ;
    const uint32_t diff = (CW(TSR_CW_DONE_SEQ(u)) - s) & TSR_TICKET_SEQ;
    return diff >= 1u && diff < 0x08000000u;
}

static uint32_t fnv(uint32_t h, uint32_t v) { return (h ^ v) * 0x01000193u; }

int main(void);
int main(void) {
    const uint32_t K = args[ARG_K], T = args[ARG_T];
    tsr_insn i;
    uint32_t tk[12];

    CW(TSR_CW_RT_STATE) = 1u;                                          // running
    const uint32_t pidx0 = CW(TSR_CW_ISSUE_PIDX);
    i = tsr_enc_ld(DST_CH(args[ARG_BG_SRC]), 0, DST_BEAT(args[ARG_BG_SRC]), args[ARG_DUMP_LINES]);
    tk[9] = push(&i, NONE, NONE);
    i = tsr_enc_ld(0, args[ARG_X_LINE], args[ARG_X_BEAT], args[ARG_X_LINES]);
    tk[0] = push(&i, tk[9], NONE);
    i = tsr_enc_qact(TSR_QA_A32, args[ARG_QA1_LINE], args[ARG_X_LINE], K, T, 0);
    tk[1] = push(&i, tk[0], NONE);
    i = tsr_enc_qact(TSR_QA_A256, args[ARG_QA2_LINE], args[ARG_X_LINE], K, T, 0);
    tk[2] = push(&i, tk[0], NONE);
    i = tsr_enc_mvm(TSR_FMT_Q8_0, 0, args[ARG_Y1_LINE], args[ARG_QA1_LINE], args[ARG_W1_OFF], args[ARG_N1], K, T, 0);
    tk[3] = push(&i, tk[1], NONE);
    i = tsr_enc_mvm(TSR_FMT_Q4_K, 0, args[ARG_Y2_LINE], args[ARG_QA2_LINE], args[ARG_W2_OFF], args[ARG_N2], K, T, 0);
    tk[4] = push(&i, tk[2], tk[1]);

    // SPM bridge (CC §10.5, §7.1): an op's SPM effects are visible to the RISC-V once its ticket is
    // done. Join on each MVM right after pushing it, then read the first and the last line of its
    // output through the exact window (the MVM is still running when DONE_SEQ shows its accept).
    uint32_t h = 0x811C9DC5u;
    const uint32_t y1_last = args[ARG_Y1_LINE] + args[ARG_Y1_LINES] - 1u;
    const uint32_t y2_last = args[ARG_Y2_LINE] + args[ARG_Y2_LINES] - 1u;
    while (!ticket_done(tk[3])) {
    }
    for (uint32_t k = 0; k < 16u; ++k) h = fnv(h, SPMW(64u * args[ARG_Y1_LINE] + 4u * k));
    for (uint32_t k = 0; k < 16u; ++k) h = fnv(h, SPMW(64u * y1_last + 4u * k));
    while (!ticket_done(tk[4])) {
    }
    for (uint32_t k = 0; k < 16u; ++k) h = fnv(h, SPMW(64u * args[ARG_Y2_LINE] + 4u * k));
    for (uint32_t k = 0; k < 16u; ++k) h = fnv(h, SPMW(64u * y2_last + 4u * k));
    i = tsr_enc_st(DST_CH(args[ARG_Y1_DST]), args[ARG_Y1_LINE], DST_BEAT(args[ARG_Y1_DST]), args[ARG_Y1_LINES]);
    tk[5] = push(&i, tk[3], NONE);
    i = tsr_enc_st(DST_CH(args[ARG_Y2_DST]), args[ARG_Y2_LINE], DST_BEAT(args[ARG_Y2_DST]), args[ARG_Y2_LINES]);
    tk[6] = push(&i, tk[4], NONE);
    CW(TSR_CW_RT_INFO) = 7u;                                           // progress: ops pushed
    i = tsr_enc_fence();
    tk[7] = push(&i, NONE, NONE);
    i = tsr_enc_st(DST_CH(args[ARG_DUMP_DST]), 0, DST_BEAT(args[ARG_DUMP_DST]), args[ARG_DUMP_LINES]);
    tk[8] = push(&i, tk[7], NONE);
    CW(TSR_CW_RT_INFO) = 9u;
    while (!ticket_done(tk[8])) {
    }
    // GDN: head 0 to its own tile, head 1 in place (the fused op on the VU, VU §7.12); joined on its stores
    if (args[ARG_GDN_ON] != 0u) {
        const uint32_t m = args[ARG_GDN_M], rows = args[ARG_GDN_ROWS], nl = m / 16u, lv = (rows + 15u) / 16u, tile = rows * nl;
        const uint32_t inh = 2u * nl + lv + tile, hs = inh + tile + lv, sp = args[ARG_GDN_SPM];
        const uint32_t al[2] = { args[ARG_GDN_AL0], args[ARG_GDN_AL1] }, be[2] = { args[ARG_GDN_BE0], args[ARG_GDN_BE1] };
        uint32_t t_last = NONE;
        for (uint32_t hd = 0; hd < 2u; ++hd) {
            const uint32_t b = sp + hd * hs, kl = b, ql = b + nl, vl = b + 2u * nl, sl = vl + lv, tl = b + inh, ol = tl + tile;
            i = tsr_enc_ld(0, b, args[ARG_GDN_IN] + hd * inh, inh);
            const uint32_t t_ld = push(&i, NONE, NONE);
            for (uint32_t k = 0; k < 2u; ++k) {                         // MOVI s(2 hd + k) = alpha / beta
                memset(&i, 0, sizeof(i));
                i.w[0] = TSR_OP_SOP;
                TSR_INSN_SET(i.w, TSR_SOP_SUB, TSR_SOP_SUB_MOVI);
                TSR_INSN_SET(i.w, TSR_SOP_SD, 2u * hd + k);
                TSR_INSN_SET(i.w, TSR_SOP_IMM, k ? be[hd] : al[hd]);
                (void)push(&i, NONE, NONE);
            }
            memset(&i, 0, sizeof(i));
            i.w[0] = TSR_OP_GDN;
            TSR_INSN_SET(i.w, TSR_GDN_SA, 2u * hd);
            TSR_INSN_SET(i.w, TSR_GDN_SB, 2u * hd + 1u);
            TSR_INSN_SET(i.w, TSR_GDN_D, hd ? sl : tl);
            TSR_INSN_SET(i.w, TSR_GDN_A, sl);
            TSR_INSN_SET(i.w, TSR_GDN_K, kl);
            TSR_INSN_SET(i.w, TSR_GDN_Q, ql);
            TSR_INSN_SET(i.w, TSR_GDN_V, vl);
            TSR_INSN_SET(i.w, TSR_GDN_O, ol);
            TSR_INSN_SET(i.w, TSR_GDN_M, m);
            TSR_INSN_SET(i.w, TSR_GDN_ROWS, rows);
            const uint32_t t_g = push(&i, t_ld, NONE);
            i = tsr_enc_st(0, hd ? sl : tl, args[ARG_GDN_OUT] + hd * (tile + lv), tile);
            (void)push(&i, t_g, NONE);
            i = tsr_enc_st(0, ol, args[ARG_GDN_OUT] + hd * (tile + lv) + tile, lv);
            t_last = push(&i, t_g, NONE);
        }
        while (!ticket_done(t_last)) {
        }
    }
    // QSTATUS's error field only: QIDLE and NONEMPTY are synchronised levels (may lag, CC §5.1)
    const uint32_t qstatus = CW(TSR_CW_QSTATUS) & TSR_QSTATUS_ERR;

    // A bridge store into the scratch line and its load back.
    const uint32_t sa = 64u * args[ARG_SCRATCH_LINE];
    SPMW(sa + 4u) = 0x5AC3F00Du ^ args[ARG_K];                       // (independent of h: the readback
    h = fnv(h, SPMW(sa + 4u));                                         //  must not cancel the hash)
    h = fnv(h, SPMW(sa));                                              // a background word of that line
    // Core window (CC §5.3): identification and the DMA unit's completion count.
    const uint32_t id = CORE(TSR_CSR_ID);
    const uint32_t qdone_dma = CORE(TSR_CSR_Q_DONE_BASE);

    // The error path (CC §9.4 I2, §9.5).
    i = tsr_enc_ld(0, args[ARG_SPM_LINES], args[ARG_X_BEAT], 1);
    tk[10] = push(&i, NONE, NONE);
    i = tsr_enc_fence();
    const uint32_t t_f2 = push(&i, NONE, NONE);
    while (!ticket_done(t_f2)) {
    }
    const uint32_t q_err = TSR_GET(CW(TSR_CW_QSTATUS), TSR_QSTATUS_ERR);
    const uint32_t e[5] = { CW(TSR_CW_ERR_SEQ(0)), CW(TSR_CW_ERR_CODE(0)), CW(TSR_CW_ERR_INFO(0)),
                            CW(TSR_CW_ERR_W0(0)), CW(TSR_CW_ERR_PIDX(0)) - pidx0 };
    for (uint32_t k = 0; k < 5u; ++k) h = fnv(h, e[k]);
    CW(TSR_CW_QCTRL) = TSR_QCTRL_QCLEAR;
    const uint32_t t_clr = CW(TSR_CW_ISSUE_TICKET);
    while (!ticket_done(t_clr)) {
    }
    const uint32_t q_err2 = TSR_GET(CW(TSR_CW_QSTATUS), TSR_QSTATUS_ERR);
    h = fnv(fnv(h, q_err), q_err2);
    h = fnv(h, CW(TSR_CW_ERR_CODE(0)));

    res[0] = qstatus;
    res[1] = h;
    res[2] = id;
    res[3] = qdone_dma;
    res[4] = CW(TSR_CW_ISSUE_PIDX) - pidx0;
    for (uint32_t k = 0; k < 10u; ++k) res[5u + k] = tk[k];

    CW(TSR_CW_LOG_PUSH) = 0x10C0u;                                     // a raw word stream (no record framing)
    CW(TSR_CW_LOG_PUSH) = h;
    CW(TSR_CW_LOG_PUSH) = tk[8];
    while (CW(TSR_CW_LCQ_FREE) == 0u) {
    }
    const uint32_t r[8] = { TSR_LCQ_R0_TYPE_EVENT_FIRST, h, qstatus, id, qdone_dma, e[1] | q_err << 8 | q_err2 << 16,
                            e[2], res[4] };
    for (uint32_t k = 0; k < 8u; ++k) CW(TSR_CW_LCQ_W(k)) = r[k];
    CW(TSR_CW_LCQ_POST) = 0u;
    CW(TSR_CW_RT_INFO) = h;
    CW(TSR_CW_RT_STATE) = RT_DONE;
    return 0;
}

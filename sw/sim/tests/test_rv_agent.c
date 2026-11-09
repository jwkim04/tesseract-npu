// test_rv_agent.c — the RISC-V agent of tsr_bus.c (TSR_RV_WITH_SIM) against a stub time base
// (ctest rv_agent).
//
// The test defines the tsr_sim_int.h / tsr_sim.h functions the agent uses, with a minimal event
// loop that follows the core's ordering rule (tsr_sim_int.h §5: at one time point the core
// events first, then the RISC-V agent; an agent that returned TSR_SIM_AGENT_WAIT is called after
// the next core event). It checks, with F_RV / F_CORE = 0.5 (one RV cycle = 2 core cycles):
//   - accesses that leave clk_rv happen at their own time (yield + slice), not at the slice start;
//   - slices end before the next core event, so an interrupt raised at core time t is taken at the
//     first instruction boundary at or after t;
//   - a push that gets TSR_CW_WAIT completes at the core event that gives the credit, counts its
//     wait in PUSH_STALL, and times out after TSR_ISSUE_WAIT_MAX RV cycles with a store access
//     fault and tsr_sim_cw_push_timeout (CC §7.3);
//   - WFI sleeps (TSR_SIM_AGENT_WAIT, STATUS.RV_SLEEP) and wakes after the core event; RV_SLEEP
//     starts at the core time the hart reaches the WFI, not at the start of its slice;
//   - a new tsr_sim_rv_boot_gen resets the hart to RV_BOOT + 0x80;
//   - BRIDGE_ACC / BRIDGE_WAIT for SPM bridge transactions, core-window reads at their time.

#include "tsr_bus.h"
#include "tsr_isa.h"
#include "tsr_rv.h"
#include "tsr_sim_int.h"

#include "test_util.h"

// ---------------------------------------------------------------------------
// Stub simulator
// ---------------------------------------------------------------------------
struct tsr_sim { int unused; };
static tsr_sim SIM;

#define NCH 4u
static uint8_t ITCM[TSR_ITCM_BYTES], DTCMEM[TSR_DTCM_BYTES], SPMMEM[64 * 1024];
static tsr_config CFG;
static uint64_t NOW;
static uint64_t BOOT_GEN = 1;
static uint32_t BOOT;
static uint32_t IRQ;
static uint32_t CREDIT;
static uint32_t STATE, STATE_SETS;
static uint64_t PERF[256];
static uint64_t T_PUSH = UINT64_MAX, T_TIMEOUT = UINT64_MAX, T_CORE = UINT64_MAX, T_BRIDGE = UINT64_MAX, T_RT = UINT64_MAX;
static uint64_t T_SLEEP = UINT64_MAX;
static int TIMEOUTS, PUSHES, WAITS;
static tsr_sim_agent_step STEP;
static void * STEP_CTX;

typedef struct ev { uint64_t t; int kind; uint32_t arg; } ev;
enum { EV_IRQ = 1, EV_CREDIT = 2 };
static ev EVS[8];
static int NEV;

uint64_t tsr_sim_now(const tsr_sim * sim) { (void)sim; return NOW; }
uint32_t tsr_sim_rv_run(const tsr_sim * sim) { (void)sim; return 1; }
uint32_t tsr_sim_rv_boot(const tsr_sim * sim) { (void)sim; return BOOT; }
uint64_t tsr_sim_rv_boot_gen(const tsr_sim * sim) { (void)sim; return BOOT_GEN; }
double   tsr_sim_rv_ratio(const tsr_sim * sim) { (void)sim; return 0.5; }
// The expected timings below are the Ibex "small" table's (tsr_rv.h), whatever the default.
uint32_t tsr_sim_rv_core(const tsr_sim * sim) { (void)sim; return TSR_RV_CORE_SMALL; }
void     tsr_sim_rv_set_state(tsr_sim * sim, uint32_t s) {
    (void)sim; STATE = s; ++STATE_SETS;
    if ((s & TSR_STATUS_RV_SLEEP) && T_SLEEP == UINT64_MAX) T_SLEEP = NOW;
}
uint32_t tsr_sim_rv_irq(tsr_sim * sim) { (void)sim; return IRQ; }
uint32_t tsr_sim_num_ch(const tsr_sim * sim) { (void)sim; return NCH; }
uint8_t * tsr_sim_itcm(tsr_sim * sim, uint32_t * bytes) { (void)sim; if (bytes) *bytes = sizeof(ITCM); return ITCM; }
uint8_t * tsr_sim_dtcm(tsr_sim * sim, uint32_t * bytes) { (void)sim; if (bytes) *bytes = sizeof(DTCMEM); return DTCMEM; }
uint8_t * tsr_sim_spm(tsr_sim * sim) { (void)sim; return SPMMEM; }
const tsr_config * tsr_sim_config(const tsr_sim * sim) { (void)sim; return &CFG; }
void tsr_sim_perf_add(tsr_sim * sim, uint32_t idx, uint64_t n) {
    (void)sim;
    if (idx < 256u) PERF[idx] += n;
    if (idx == TSR_PERF_X(NCH) + TSR_PERF_XOFF_BRIDGE_ACC && T_BRIDGE == UINT64_MAX) T_BRIDGE = NOW;
}
void tsr_sim_set_rv_agent(tsr_sim * sim, tsr_sim_agent_step step, void * ctx) { (void)sim; STEP = step; STEP_CTX = ctx; }
void tsr_sim_cw_push_timeout(tsr_sim * sim) { (void)sim; ++TIMEOUTS; T_TIMEOUT = NOW; }
uint32_t tsr_sim_corewin_read(tsr_sim * sim, uint32_t off) { (void)sim; if (T_CORE == UINT64_MAX) T_CORE = NOW; return 0xC0000000u | off; }

int tsr_sim_cw_access(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val) {
    (void)sim;
    if (is_write && off == TSR_CW_ISSUE_PUSH) {
        if (CREDIT == 0u) { ++WAITS; return TSR_CW_WAIT; }
        --CREDIT; ++PUSHES; T_PUSH = NOW;
        return TSR_CW_OK;
    }
    if (is_write && off == TSR_CW_RT_STATE && T_RT == UINT64_MAX) T_RT = NOW;
    if (!is_write) *val = 0;
    return TSR_CW_OK;
}

// the agent's scheduler / access-event / fast-forward interface (tsr_sim_int.h §5, §5b), as
// the fixed schedule without listeners or a log: the agent behaves exactly as before.
int tsr_sim_cw_access_pc(tsr_sim * sim, int is_write, uint32_t off, uint32_t * val, uint32_t pc) {
    (void)pc;
    return tsr_sim_cw_access(sim, is_write, off, val);
}
void tsr_sim_corewin_note(tsr_sim * sim, uint32_t off, uint32_t val, uint32_t pc) { (void)sim; (void)off; (void)val; (void)pc; }
void tsr_sim_acc_emit(tsr_sim * sim, const tsr_sim_acc * a) { (void)sim; (void)a; }
int  tsr_sim_acc_listened(const tsr_sim * sim) { (void)sim; return 0; }
int  tsr_sim_sched_random(const tsr_sim * sim) { (void)sim; return 0; }
uint32_t tsr_sched_rv_handshake(tsr_sim * sim) { (void)sim; return 0; }
int  tsr_sched_rv_cpi(tsr_sim * sim) { (void)sim; return 0; }
int  tsr_sim_fast_forward(const tsr_sim * sim) { (void)sim; return 1; }
int  tsr_sim_log_rv(const tsr_sim * sim) { (void)sim; return 0; }
void tsr_sim_log_line(tsr_sim * sim, const char * fmt, ...) { (void)sim; (void)fmt; }

uint64_t tsr_sim_next_event(const tsr_sim * sim) {
    (void)sim;
    uint64_t t = UINT64_MAX;
    for (int i = 0; i < NEV; ++i) if (EVS[i].t < t) t = EVS[i].t;
    return t;
}

static void add_event(uint64_t t, int kind, uint32_t arg) { ev e = { t, kind, arg }; EVS[NEV++] = e; }

// The time base: core events, then the agent, until `limit`.
static uint64_t AGENT_AT;
static int AGENT_WAITING;
static void run_until(uint64_t limit) {
    for (;;) {
        int k = -1;
        for (int i = 0; i < NEV; ++i) if (k < 0 || EVS[i].t < EVS[k].t) k = i;
        uint64_t t_ev = k >= 0 ? EVS[k].t : UINT64_MAX;
        uint64_t t_ag = AGENT_WAITING ? UINT64_MAX : AGENT_AT;
        uint64_t t = t_ev < t_ag ? t_ev : t_ag;
        if (t > limit || t == UINT64_MAX) { if (limit > NOW) NOW = limit; return; }
        NOW = t;
        if (t_ev <= t_ag) {
            ev e = EVS[k];
            EVS[k] = EVS[--NEV];
            if (e.kind == EV_IRQ) IRQ = e.arg;
            else if (e.kind == EV_CREDIT) CREDIT += e.arg;
            if (AGENT_WAITING) { AGENT_WAITING = 0; AGENT_AT = NOW; }
            continue;
        }
        uint64_t d = STEP(STEP_CTX, &SIM);
        CHECK(d >= 1u, "agent step returned 0");
        if (d == TSR_SIM_AGENT_WAIT) AGENT_WAITING = 1;
        else AGENT_AT = NOW + d;
    }
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------
#define LUI(rd, imm)        (((uint32_t)(imm) & 0xFFFFF000u) | ((rd) << 7) | 0x37u)
#define ADDI(rd, rs1, imm)  ((((uint32_t)(imm) & 0xFFFu) << 20) | ((rs1) << 15) | ((rd) << 7) | 0x13u)
#define SW(rs2, rs1, imm)   ((((uint32_t)(imm) >> 5 & 0x7Fu) << 25) | ((rs2) << 20) | ((rs1) << 15) | (2u << 12) | (((uint32_t)(imm) & 0x1Fu) << 7) | 0x23u)
#define LW(rd, rs1, imm)    ((((uint32_t)(imm) & 0xFFFu) << 20) | ((rs1) << 15) | (2u << 12) | ((rd) << 7) | 0x03u)
#define CSRRW(rd, c, rs1)   (((uint32_t)(c) << 20) | ((rs1) << 15) | (1u << 12) | ((rd) << 7) | 0x73u)
#define CSRRSI(rd, c, u)    (((uint32_t)(c) << 20) | ((u) << 15) | (6u << 12) | ((rd) << 7) | 0x73u)
#define NOP                 0x00000013u
#define WFI                 0x10500073u
#define JSELF               0x0000006Fu      // jal x0, 0

static uint32_t PCW;
static void org(uint32_t a) { PCW = a; }
static void put(uint32_t w) {
    ITCM[PCW] = (uint8_t)w; ITCM[PCW + 1] = (uint8_t)(w >> 8); ITCM[PCW + 2] = (uint8_t)(w >> 16); ITCM[PCW + 3] = (uint8_t)(w >> 24);
    PCW += 4;
}

static tsr_rv_agent * A;
static void fresh(void) {
    if (A) tsr_rv_agent_destroy(A);
    CHECK(STEP == NULL, "destroy unregisters the agent");
    memset(ITCM, 0, sizeof(ITCM));
    memset(PERF, 0, sizeof(PERF));
    NOW = 0; BOOT = 0; ++BOOT_GEN; IRQ = 0; CREDIT = 0; STATE = 0; STATE_SETS = 0;
    T_PUSH = T_TIMEOUT = T_CORE = T_BRIDGE = T_RT = T_SLEEP = UINT64_MAX;
    TIMEOUTS = PUSHES = WAITS = 0; NEV = 0;
    AGENT_AT = 0; AGENT_WAITING = 0;
    CFG.spm_bytes = sizeof(SPMMEM);
    A = tsr_rv_agent_create(&SIM);
    CHECK(A != NULL && STEP != NULL && STEP_CTX == A, "agent registered");
    org(0x00); put(JSELF);                                              // exception handler (mtvec base 0)
    org(0x40); for (int i = 0; i < 8; ++i) put(NOP); put(JSELF);       // irq_fast[0] entry (vectored)
    org(0x80);
}
static uint32_t csr(uint32_t c) { uint32_t v = 0; tsr_rv_csr_read(tsr_rv_agent_rv(A), c, &v); return v; }

int main(void) {
    const uint32_t X = TSR_PERF_X(NCH);

    // 1. An access that leaves clk_rv happens at its own time: LUI + 99 NOPs = 100 RV cycles, so
    //    the store starts at core time 200; the slice that ran them started at 0.
    fresh();
    put(LUI(10, TSR_RV_CTL_BASE));
    for (int i = 0; i < 99; ++i) put(NOP);
    put(SW(0, 10, TSR_CW_RT_STATE));
    put(LUI(11, TSR_RV_CORE_BASE)); put(LW(12, 11, 0x24)); put(LUI(13, TSR_RV_SPM_BASE)); put(LW(14, 13, 0x40)); put(JSELF);
    run_until(1000);
    CHECK(T_RT == 200u, "control-window store at core time 200, got %llu", (unsigned long long)T_RT);
    // After the store (4 RV cycles: lsu 1 + window 3): LUI at 104, the core-window load at 105 ->
    // core time 210; then lsu 1 + the core-window latency at ratio 0.5 (tsr_bus.h: 2 + round(12 x
    // 0.5) = 8) -> LUI at 114, the SPM load at 115 -> 230. The SPM latency is 2 + round(15 x 0.5) = 10.
    const uint32_t lat_spm = TSR_BUS_XING_RV + (uint32_t)(TSR_BUS_XING_SPM_CC * 0.5 + 0.5);
    CHECK(T_CORE == 210u, "core-window read at core time 210, got %llu", (unsigned long long)T_CORE);
    CHECK(T_BRIDGE == 230u, "SPM bridge access at core time 230, got %llu", (unsigned long long)T_BRIDGE);
    CHECK(tsr_rv_reg(tsr_rv_agent_rv(A), 12) == (0xC0000000u | 0x24u), "core-window value");
    CHECK(PERF[X + TSR_PERF_XOFF_BRIDGE_ACC] == 1u && PERF[X + TSR_PERF_XOFF_BRIDGE_WAIT] == (uint64_t)(lat_spm - TSR_BUS_LAT_TCM),
          "BRIDGE_ACC / BRIDGE_WAIT %llu", (unsigned long long)PERF[X + TSR_PERF_XOFF_BRIDGE_WAIT]);
    CHECK(tsr_rv_pc(tsr_rv_agent_rv(A)) == 0x80u + 4u * 105u, "spinning at the end");
    // The hart runs again exactly when core time reaches its RV time (ratio 0.5: 2 core cycles per
    // RV cycle), at most one slice (1024 RV cycles) ahead of MTIME.
    uint64_t c = tsr_rv_cycles(tsr_rv_agent_rv(A));
    CHECK(AGENT_AT == 2u * c && c <= 500u + 1024u + 3u, "RV time %llu vs next run at %llu", (unsigned long long)c, (unsigned long long)AGENT_AT);

    // 2. A core event at time 50 raises irq_fast[0]: the interrupt is taken at the first boundary
    //    at or after core time 50, i.e. RV cycle 25: LUI 1, the mie and mstatus writes 2 each (Ibex
    //    pipeline flush, tsr_rv.h), NOPs 1 each -> the 24th instruction, at 0x80 + 4 x 23.
    fresh();
    put(LUI(5, 0x10000)); put(CSRRW(0, 0x304, 5)); put(CSRRSI(0, 0x300, 8));
    for (int i = 0; i < 64; ++i) put(NOP);
    put(JSELF);
    add_event(50, EV_IRQ, TSR_RV_LINE_FAST(0));
    run_until(200);
    CHECK(csr(0x342) == 0x80000010u && csr(0x341) == 0x80u + 4u * 23u, "irq at RV cycle 25: mcause 0x%x mepc 0x%x", csr(0x342), csr(0x341));

    // 3. A push without a credit waits (no interrupt meanwhile) and completes at the core event
    //    that gives the credit; the wait counts in PUSH_STALL (RV cycles).
    fresh();
    put(LUI(5, 0x10000)); put(CSRRW(0, 0x304, 5)); put(CSRRSI(0, 0x300, 8));
    put(LUI(10, TSR_RV_CTL_BASE)); put(SW(0, 10, TSR_CW_ISSUE_PUSH)); put(ADDI(1, 0, 1)); put(JSELF);
    add_event(300, EV_IRQ, TSR_RV_LINE_FAST(0));
    add_event(1000, EV_CREDIT, 1);
    run_until(1100);
    CHECK(PUSHES == 1 && T_PUSH == 1000u, "push at the credit event: %llu", (unsigned long long)T_PUSH);
    CHECK(WAITS >= 1 && WAITS <= 3, "retried only at events (%d attempts)", WAITS);
    // The push starts at RV cycle 6 (LUI 1, two flushing CSR writes 2 each, LUI 1) and waits to 500.
    CHECK(PERF[X + TSR_PERF_XOFF_PUSH_STALL] == 500u - 6u, "PUSH_STALL %llu", (unsigned long long)PERF[X + TSR_PERF_XOFF_PUSH_STALL]);
    CHECK(csr(0x342) == 0x80000010u && csr(0x341) == 0x80u + 4u * 5u, "interrupt only after the push: mepc 0x%x", csr(0x341));

    // 4. No credit ever: after TSR_ISSUE_WAIT_MAX RV cycles the push faults (CC §7.3).
    fresh();
    put(LUI(10, TSR_RV_CTL_BASE)); put(SW(0, 10, TSR_CW_ISSUE_PUSH)); put(JSELF);
    run_until(200000);
    CHECK(TIMEOUTS == 1 && PUSHES == 0, "one push timeout");
    CHECK(T_TIMEOUT == 2u * (1u + TSR_ISSUE_WAIT_MAX), "timeout at core time %llu", (unsigned long long)T_TIMEOUT);
    CHECK(csr(0x342) == TSR_RV_EXC_STORE_FAULT && csr(0x341) == 0x84u && csr(0x343) == TSR_RV_CTL_BASE + TSR_CW_ISSUE_PUSH, "store access fault");
    CHECK(PERF[X + TSR_PERF_XOFF_PUSH_STALL] == TSR_ISSUE_WAIT_MAX, "PUSH_STALL = the bound");

    // 5. WFI: the agent sleeps (STATUS.RV_SLEEP) and wakes after the core event; MIE = 0, so the
    //    hart continues after the WFI.
    fresh();
    put(LUI(5, 0x10000)); put(CSRRW(0, 0x304, 5)); put(WFI); put(ADDI(1, 0, 7)); put(JSELF);
    run_until(400);
    CHECK(AGENT_WAITING && STATE == TSR_STATUS_RV_SLEEP && tsr_rv_reg(tsr_rv_agent_rv(A), 1) == 0u, "asleep");
    add_event(500, EV_IRQ, TSR_RV_LINE_FAST(0));
    run_until(600);
    CHECK(STATE == 0u && tsr_rv_reg(tsr_rv_agent_rv(A), 1) == 7u, "woke at the event");
    CHECK(csr(0xB00) >= 250u, "mcycle counted the sleep: %u", csr(0xB00));

    // 6. A new boot generation resets the hart to RV_BOOT + 0x80.
    org(0x280); put(ADDI(2, 0, 9)); put(JSELF);
    BOOT = 0x200; ++BOOT_GEN;
    AGENT_AT = NOW; AGENT_WAITING = 0;
    run_until(700);
    CHECK(tsr_rv_reg(tsr_rv_agent_rv(A), 2) == 9u && tsr_rv_reg(tsr_rv_agent_rv(A), 1) == 0u && csr(0x305) == 0x201u, "reset to the new boot address");

    // 7. Debug halt: STATUS.RV_HALTED and the agent sleeps.
    tsr_rv_set_debug_req(tsr_rv_agent_rv(A), 1);
    run_until(NOW + 5000u);       // the next slice starts within 1024 RV cycles
    CHECK(STATE == TSR_STATUS_RV_HALTED && AGENT_WAITING, "halted");
    tsr_rv_set_debug_req(tsr_rv_agent_rv(A), 0);

    // 8. With no core event pending one slice runs the whole program ahead of MTIME: STATUS.RV_SLEEP
    //    (and so PERF RV_WFI) starts at the core time the hart reaches the WFI, not at core time 0.
    fresh();
    put(LUI(5, 0x10000)); put(CSRRW(0, 0x304, 5));
    for (int i = 0; i < 200; ++i) put(NOP);
    put(WFI); put(ADDI(1, 0, 7)); put(JSELF);
    run_until(2000);
    const uint64_t c_wfi = tsr_rv_cycles(tsr_rv_agent_rv(A));
    CHECK(AGENT_WAITING && STATE == TSR_STATUS_RV_SLEEP && c_wfi > 200u && T_SLEEP == 2u * c_wfi,
          "RV_SLEEP from core time %llu, WFI reached at RV cycle %llu", (unsigned long long)T_SLEEP, (unsigned long long)c_wfi);
    add_event(2100, EV_IRQ, TSR_RV_LINE_FAST(0));
    run_until(2200);
    CHECK(STATE == 0u && tsr_rv_reg(tsr_rv_agent_rv(A), 1) == 7u, "woke at the event");

    tsr_rv_agent_destroy(A);
    A = NULL;
    CHECK(STEP == NULL, "unregistered");
    return test_finish("rv_agent");
}

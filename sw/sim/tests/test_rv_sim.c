// test_rv_sim.c — the RISC-V agent on the real simulator (ctest rv_sim): firmware in ITCM, booted
// through the host CSR port (EX §3.4), runs on tsr_rv against the core model (tsr_sim.c, tsr_q.c).
//
// The firmware pushes a NOP through the issue port (CC §5.1) and reads its ticket, polls
// DONE_SEQ[SYNC] until it completes, stores to the SPM window, writes RT_STATE / RT_INFO (mirrored
// to the host, EX §4.5), reads ID through the core window (CC §5.3), arms MTIMECMP and sleeps in
// WFI with only MTIE enabled (CC §12.3), then posts a kernel event record to the LCQ (EX §4.4) and
// parks in WFI. The host checks each effect through its CSR port and the SPM.

#include "tsr_bus.h"
#include "tsr_isa.h"
#include "tsr_rv.h"
#include "tsr_sim.h"
#include "tsr_sim_int.h"

#include "test_util.h"

#define LUI(rd, imm)        (((uint32_t)(imm) & 0xFFFFF000u) | ((rd) << 7) | 0x37u)
#define ADDI(rd, rs1, imm)  ((((uint32_t)(imm) & 0xFFFu) << 20) | ((rs1) << 15) | ((rd) << 7) | 0x13u)
#define SW(rs2, rs1, imm)   ((((uint32_t)(imm) >> 5 & 0x7Fu) << 25) | ((rs2) << 20) | ((rs1) << 15) | (2u << 12) | (((uint32_t)(imm) & 0x1Fu) << 7) | 0x23u)
#define LW(rd, rs1, imm)    ((((uint32_t)(imm) & 0xFFFu) << 20) | ((rs1) << 15) | (2u << 12) | ((rd) << 7) | 0x03u)
#define CSRRW(rd, c, rs1)   (((uint32_t)(c) << 20) | ((rs1) << 15) | (1u << 12) | ((rd) << 7) | 0x73u)
#define WFI                 0x10500073u
static uint32_t eB(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 12) & 1u) << 31) | (((u >> 5) & 0x3Fu) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) |
           (((u >> 1) & 0xFu) << 8) | (((u >> 11) & 1u) << 7) | 0x63u;
}
static uint32_t eJ(int32_t imm, uint32_t rd) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 20) & 1u) << 31) | (((u >> 1) & 0x3FFu) << 21) | (((u >> 11) & 1u) << 20) | (((u >> 12) & 0xFFu) << 12) | (rd << 7) | 0x6Fu;
}

static uint8_t * ITCM;
static uint32_t PCW;
static void put(uint32_t w) {
    ITCM[PCW] = (uint8_t)w; ITCM[PCW + 1] = (uint8_t)(w >> 8); ITCM[PCW + 2] = (uint8_t)(w >> 16); ITCM[PCW + 3] = (uint8_t)(w >> 24);
    PCW += 4;
}

int main(void) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = 1;
    cfg.ch_bytes = 1ull << 28;
    tsr_sim * sim = tsr_sim_create(&cfg, NULL);
    CHECK(sim != NULL, "tsr_sim_create");
    if (sim == NULL) return test_finish("rv_sim");
    tsr_rv_agent * a = tsr_sim_rv_agent(sim);            // built in by tsr_sim_create
    CHECK(a != NULL, "built-in agent");
    uint32_t itcm_bytes = 0;
    ITCM = tsr_sim_itcm(sim, &itcm_bytes);
    CHECK(itcm_bytes == TSR_ITCM_BYTES, "ITCM size");

    // Firmware at RV_BOOT + 0x80 (RV_BOOT = 0).
    PCW = 0x80;
    put(LUI(10, TSR_RV_CTL_BASE));
    put(ADDI(6, 0, (int32_t)TSR_OP_NOP));
    put(SW(6, 10, TSR_CW_ISSUE_W(0)));
    put(SW(0, 10, TSR_CW_ISSUE_PUSH));                    // WMASK 0, REC 0
    put(LW(7, 10, TSR_CW_ISSUE_TICKET));
    uint32_t poll = PCW;
    put(LW(8, 10, TSR_CW_DONE_SEQ(TSR_UNIT_SYNC)));
    put(eB((int32_t)(poll - PCW), 0, 8, 0));             // beq x8, x0, poll
    put(LUI(11, TSR_RV_SPM_BASE));
    put(LUI(12, 0x12345000)); put(ADDI(12, 12, 0x678));
    put(SW(12, 11, 0x100));
    put(ADDI(13, 0, 2));
    put(SW(13, 10, TSR_CW_RT_STATE));
    put(SW(7, 10, TSR_CW_RT_INFO));
    put(LUI(14, TSR_RV_CORE_BASE));
    put(LW(15, 14, TSR_CSR_ID));
    put(SW(15, 10, TSR_CW_LCQ_W(1)));
    put(LW(16, 10, TSR_CW_MTIME_LO));
    put(ADDI(16, 16, 500));
    put(SW(16, 10, TSR_CW_MTIMECMP_LO));
    put(SW(0, 10, TSR_CW_MTIMECMP_HI));
    put(SW(16, 10, TSR_CW_LCQ_W(3)));
    put(ADDI(17, 0, 0x80));
    put(CSRRW(0, 0x304, 17));                             // mie = MTIE; mstatus.MIE stays 0
    put(WFI);
    put(LW(18, 10, TSR_CW_MTIME_LO));
    put(SW(18, 10, TSR_CW_LCQ_W(2)));
    put(ADDI(19, 0, 0xC1));
    put(SW(19, 10, TSR_CW_LCQ_W(0)));
    put(SW(0, 10, TSR_CW_LCQ_POST));                      // event (ANSWER = 0)
    uint32_t park = PCW;
    put(CSRRW(0, 0x304, 0));
    put(WFI);
    put(eJ((int32_t)(park - PCW), 0));

    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    uint32_t ev = 0;
    for (int i = 0; i < 200 && !(ev & TSR_SIM_EV_LCQ); ++i) ev = tsr_sim_advance(sim, 100000);
    CHECK(ev & TSR_SIM_EV_LCQ, "an LCQ record was posted (events 0x%x)", ev);
    CHECK(tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR) == 1u, "LCQ_WR = 1");
    uint32_t r0 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(0, 0)), r1 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(0, 1));
    uint32_t r2 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(0, 2)), r3 = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(0, 3));
    CHECK(r0 == 0xC1u, "record type 0x%x", r0);
    CHECK(r1 == TSR_ID_MAGIC, "core-window ID 0x%08x", r1);
    CHECK(r2 >= r3 && r2 - r3 < 64u, "woke from WFI by the timer at MTIME %u (MTIMECMP %u)", r2, r3);
    CHECK(tsr_sim_csr_read(sim, TSR_CSR_RT_STATE) == 2u, "RT_STATE mirror");
    uint32_t ticket = tsr_sim_csr_read(sim, TSR_CSR_RT_INFO);
    CHECK(ticket >> 28 == TSR_UNIT_SYNC && (ticket & 0x0FFFFFFFu) == 0u, "NOP ticket 0x%08x", ticket);
    const uint8_t * spm = tsr_sim_spm(sim);
    CHECK(spm[0x100] == 0x78 && spm[0x101] == 0x56 && spm[0x102] == 0x34 && spm[0x103] == 0x12, "SPM window store");
    tsr_rv * rv = tsr_rv_agent_rv(a);
    CHECK(tsr_rv_reg(rv, 7) == ticket, "ISSUE_TICKET read");
    // The hart parks in WFI: STATUS.RV_SLEEP.
    tsr_sim_advance(sim, 10000);
    CHECK(tsr_rv_sleeping(rv) && (tsr_sim_csr_read(sim, TSR_CSR_STATUS) & TSR_STATUS_RV_SLEEP), "parked, STATUS.RV_SLEEP");
    uint32_t mcause = 0;
    tsr_rv_csr_read(rv, 0x342, &mcause);
    CHECK(mcause == 0u, "no trap (mcause %u)", mcause);
    CHECK(tsr_sim_rv_instret(sim) == tsr_rv_retired(rv) && tsr_sim_rv_reg(sim, 7) == ticket &&
          tsr_sim_rv_pc(sim) == tsr_rv_pc(rv), "SW §4.8 debug view");
    printf("rv_sim: %llu instructions retired, MTIME %llu\n", (unsigned long long)tsr_rv_retired(rv),
           (unsigned long long)tsr_sim_now(sim));
    tsr_sim_destroy(sim);                                 // destroys the built-in agent
    return test_finish("rv_sim");
}

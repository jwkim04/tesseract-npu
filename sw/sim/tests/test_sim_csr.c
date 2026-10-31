// test_sim_csr.c — host CSR port (EX §5) and the RISC-V side of the rings (EX §4.5): identity and
// configuration (VERSION 0x0100: Tesseract Core capability 1.0), the firmware loader with receive
// CRC and VERIFY (EX §3.3, incl. a corrupted TCM and an address-decode fault), the launch queue
// and completion ring (ordering, sequence numbers, ACTIVE, flow control, faults, read-pointer rule),
// watermarks and the hard cancel, the log ring (DB §6.1 records, overflow drop with LOG_DROPPED),
// SOFT_RESET (EX §8.5: what it clears and keeps, SR_BUSY over a running op), PERF gating by ACTIVE
// and the LO/HI snapshot, STATUS levels / sticky bits, MTIME and the timer, and one launch end to
// end through a scripted RISC-V agent on the time base, PERF RV_WFI gated by ACTIVE.
#include "test_sim_util.h"
#include "tsr_sim_int.h"

static tsr_sim * mk_sim(void) {
    tsr_config c = cfg_u200(1);
    tsr_sim * s = tsr_sim_create(&c, NULL);
    CHECK(s != NULL, "create");
    if (s == NULL) exit(1);
    return s;
}
static uint32_t rd(tsr_sim * s, uint32_t off) { return tsr_sim_csr_read(s, off); }
static void wr(tsr_sim * s, uint32_t off, uint32_t v) { tsr_sim_csr_write(s, off, v); }
static uint32_t cwr(tsr_sim * s, uint32_t off) { return tsr_sim_cw_read(s, off); }
static int cww(tsr_sim * s, uint32_t off, uint32_t v) { return tsr_sim_cw_access(s, 1, off, &v); }

// Independent zlib crc32 (table driven; the simulator's is bitwise).
static uint32_t g_crc_tab[256];
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
        g_crc_tab[i] = c;
    }
}
static uint32_t crc32_bytes(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = g_crc_tab[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static uint32_t crc32_words(const uint32_t * w, size_t n) {
    uint8_t * b = (uint8_t *)malloc(4 * n + 1);
    for (size_t i = 0; i < n; ++i) { b[4 * i] = (uint8_t)w[i]; b[4 * i + 1] = (uint8_t)(w[i] >> 8);
                                     b[4 * i + 2] = (uint8_t)(w[i] >> 16); b[4 * i + 3] = (uint8_t)(w[i] >> 24); }
    const uint32_t c = crc32_bytes(b, 4 * n);
    free(b);
    return c;
}

// ---------------------------------------------------------------------------
static void test_identity(void) {
    tsr_sim * s = mk_sim();
    CHECK(rd(s, TSR_CSR_ID) == 0x54535231u, "ID");
    CHECK(rd(s, TSR_CSR_VERSION) == 0x00000100u, "VERSION 0x%x (Tesseract Core capability 1.0)", rd(s, TSR_CSR_VERSION));
    CHECK(rd(s, TSR_CSR_CFG0) == (4u | (128u << 8) | (1u << 20)), "CFG0");
    CHECK(rd(s, TSR_CSR_CFG1) == 65536u, "CFG1");
    CHECK(rd(s, TSR_CSR_CFG2) == (16u | (16u << 8) | (128u << 16)), "CFG2");
    CHECK(rd(s, TSR_CSR_CAPS) == 0 && rd(s, TSR_CSR_CFG3) == 0, "CAPS / CFG3 without VU and AT");
    CHECK(TSR_GET(rd(s, TSR_CSR_CTL_ID), TSR_CTL_ID_VERSION) == 1 && (rd(s, TSR_CSR_CTL_ID) & TSR_CTL_ID_LOG), "CTL_ID 0x%x",
          rd(s, TSR_CSR_CTL_ID));
    CHECK(rd(s, TSR_CSR_RV_CFG) == (128u | (64u << 16)), "RV_CFG 0x%x", rd(s, TSR_CSR_RV_CFG));
    const uint32_t f = rd(s, TSR_CSR_RV_FREQ);
    CHECK(f > 39000 && f < 40000, "RV_FREQ %u (166.7 / 276.3 x 2^16)", f);
    CHECK(rd(s, TSR_CSR_WDOG_CFG) == 24 && rd(s, TSR_CSR_LOG_CFG) == 0xF1, "reset values WDOG_CFG / LOG_CFG");
    wr(s, TSR_CSR_WDOG_CFG, 0xFFFFFFFFu);
    wr(s, TSR_CSR_LOG_CFG, 0xFFFFFFFFu);
    CHECK(rd(s, TSR_CSR_WDOG_CFG) == 0x1F && rd(s, TSR_CSR_LOG_CFG) == 0xF3, "RW masks");
    CHECK(rd(s, TSR_CSR_LQ_CFG) == (8u | (8u << 8) | (256u << 16)), "LQ_CFG 0x%x", rd(s, TSR_CSR_LQ_CFG));
    CHECK(rd(s, TSR_CSR_PERF_INFO) == (64u | (13u << 8)), "PERF_INFO 0x%x", rd(s, TSR_CSR_PERF_INFO));
    CHECK(rd(s, TSR_CSR_Q_DEPTH) == TSR_QDEPTH_U200_REG, "Q_DEPTH 0x%x", rd(s, TSR_CSR_Q_DEPTH));
    const uint32_t st = rd(s, TSR_CSR_STATUS);
    CHECK(st == (TSR_STATUS_RV_CLK_OK | TSR_STATUS_RV_RESET_DONE | TSR_STATUS_QIDLE | TSR_STATUS_QQUIESCENT),
          "STATUS after create 0x%x", st);
    CHECK(rd(s, TSR_CSR_CTRL) == 0, "CTRL");
    // RV_BOOT: 256-byte aligned
    wr(s, TSR_CSR_RV_BOOT, 0x12345u);
    CHECK(rd(s, TSR_CSR_RV_BOOT) == 0x12300u && tsr_sim_rv_boot(s) == 0x12300u, "RV_BOOT [7:0] read 0");
    // Unmapped / reserved / absent blocks read 0
    const uint32_t zero_offs[] = { 0x08C, 0x0A0, 0x0D8, 0x0FC, 0x800, 0x804, 0x9BC, 0xA00, 0xB00, 0xC00, 0xDFC, 0xF20, 0xFFC,
                                   0x1000, 0x2001 };
    for (size_t i = 0; i < sizeof(zero_offs) / 4; ++i) {
        // P1.13: with the DBG model attached (tsr_dbg.c) DBG_ID reports the block (sim_dbg checks it)
        void ** dbg_slot = tsr_sim_model_slot(s, TSR_MODEL_DBG);
        if (zero_offs[i] == TSR_CSR_DBG_ID && dbg_slot != NULL && *dbg_slot != NULL) continue;
        CHECK(rd(s, zero_offs[i]) == 0, "offset 0x%x reads 0", zero_offs[i]);
    }
    // SREG: the s bank, also through the core window
    tsr_sim_sreg(s)[3] = 1.5f;
    tsr_sim_sreg(s)[15] = -0.0f;
    CHECK(rd(s, TSR_CSR_SREG(3)) == f2u(1.5f) && rd(s, TSR_CSR_SREG(15)) == 0x80000000u && rd(s, TSR_CSR_SREG(0)) == 0, "SREG");
    CHECK(tsr_sim_corewin_read(s, TSR_CSR_SREG(3)) == f2u(1.5f), "SREG core window");
    wr(s, TSR_CSR_SREG(3), 0);
    CHECK(rd(s, TSR_CSR_SREG(3)) == f2u(1.5f), "SREG read-only");
    // The RISC-V view of the configuration
    CHECK(cwr(s, TSR_CW_ISSUE_UNITS) == 0x2Fu, "ISSUE_UNITS 0x%x (unit 4 absent)", cwr(s, TSR_CW_ISSUE_UNITS));
    CHECK(tsr_sim_corewin_read(s, TSR_CSR_VERSION) == 0x100u && tsr_sim_corewin_read(s, TSR_CSR_Q_DEPTH) == TSR_QDEPTH_U200_REG,
          "core window identity");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
static uint32_t load_segment(tsr_sim * s, uint32_t addr, const uint32_t * w, uint32_t n) {
    wr(s, TSR_CSR_LOAD_ADDR, addr);
    CHECK(rd(s, TSR_CSR_LOAD_CRC) == 0, "LOAD_ADDR restarts the CRC (CRC of zero bytes)");
    for (uint32_t i = 0; i < n; ++i) wr(s, TSR_CSR_LOAD_DATA, w[i]);
    return rd(s, TSR_CSR_LOAD_CRC);
}
static uint32_t verify_segment(tsr_sim * s, uint32_t addr, uint32_t n) {
    wr(s, TSR_CSR_LOAD_ADDR, addr);
    wr(s, TSR_CSR_LOAD_COUNT, n);
    wr(s, TSR_CSR_LOAD_CTRL, TSR_LOAD_CTRL_VERIFY);
    CHECK((rd(s, TSR_CSR_LOAD_STATUS) & TSR_LOAD_STATUS_VERIFY_BUSY) == 0, "VERIFY_BUSY clears");
    return rd(s, TSR_CSR_LOAD_CRC);
}

static void test_loader(void) {
    crc_init();
    CHECK(crc32_bytes((const uint8_t *)"123456789", 9) == 0xCBF43926u, "reference crc32 check value");
    tsr_sim * s = mk_sim();
    rng_t r = { 77 };
    enum { NI = 4096, ND = 300 };                      // 16 KiB ITCM segment, 1200 B DTCM segment
    uint32_t * itw = (uint32_t *)malloc(NI * 4), * dtw = (uint32_t *)malloc(ND * 4);
    for (int i = 0; i < NI; ++i) itw[i] = rng_u32(&r);
    for (int i = 0; i < ND; ++i) dtw[i] = rng_u32(&r);
    const uint32_t it_crc = crc32_words(itw, NI), dt_crc = crc32_words(dtw, ND);

    CHECK(load_segment(s, 0x0, itw, NI) == it_crc, "receive CRC ITCM");
    CHECK(rd(s, TSR_CSR_LOAD_ADDR) == 4u * NI, "LOAD_ADDR advanced");
    CHECK(load_segment(s, TSR_RV_DTCM_BASE + 0x40, dtw, ND) == dt_crc, "receive CRC DTCM");
    uint32_t ib = 0, db = 0;
    const uint8_t * itcm = tsr_sim_itcm(s, &ib), * dtcm = tsr_sim_dtcm(s, &db);
    CHECK(ib == TSR_ITCM_BYTES && db == TSR_DTCM_BYTES, "TCM sizes");
    CHECK(get_u32_le(itcm + 4 * 123) == itw[123] && get_u32_le(dtcm + 0x40 + 4 * 299) == dtw[299], "TCM content");
    CHECK(verify_segment(s, 0x0, NI) == it_crc && verify_segment(s, TSR_RV_DTCM_BASE + 0x40, ND) == dt_crc, "VERIFY CRC");
    // LOAD_DATA read: the word at LOAD_ADDR, no increment
    wr(s, TSR_CSR_LOAD_ADDR, 4 * 10);
    CHECK(rd(s, TSR_CSR_LOAD_DATA) == itw[10] && rd(s, TSR_CSR_LOAD_DATA) == itw[10] && rd(s, TSR_CSR_LOAD_ADDR) == 40,
          "LOAD_DATA read");
    CHECK(tsr_sim_corewin_read(s, TSR_CSR_LOAD_DATA) == 0, "core window LOAD_DATA reads 0");
    wr(s, TSR_CSR_LOAD_ADDR, 0x103);
    CHECK(rd(s, TSR_CSR_LOAD_ADDR) == 0x100, "LOAD_ADDR word aligned");

    // A corrupted TCM word (a broken write the receive CRC cannot see): VERIFY mismatches.
    tsr_sim_itcm(s, NULL)[4 * 2000 + 1] ^= 0x10;
    CHECK(verify_segment(s, 0x0, NI) != it_crc, "VERIFY detects the corrupted word");
    tsr_sim_itcm(s, NULL)[4 * 2000 + 1] ^= 0x10;
    CHECK(verify_segment(s, 0x0, NI) == it_crc, "VERIFY clean again");
    // Address decode: a segment beyond ITCM is received (CRC matches) but VERIFY reads zeros.
    CHECK(load_segment(s, TSR_ITCM_BYTES, dtw, 16) == crc32_words(dtw, 16), "receive CRC of an unmapped segment");
    CHECK(verify_segment(s, TSR_ITCM_BYTES, 16) != crc32_words(dtw, 16), "VERIFY catches the address-decode fault");
    // Refused while RV_RUN = 1: writes dropped with LOAD_REFUSED, reads 0, VERIFY refused.
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    wr(s, TSR_CSR_LOAD_ADDR, 0);
    wr(s, TSR_CSR_LOAD_DATA, 0xDEADBEEFu);
    CHECK(get_u32_le(tsr_sim_itcm(s, NULL)) == itw[0] && rd(s, TSR_CSR_LOAD_ADDR) == 0, "write refused");
    CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOAD_REFUSED, "LOAD_REFUSED");
    CHECK(rd(s, TSR_CSR_LOAD_DATA) == 0, "read refused returns 0");
    wr(s, TSR_CSR_STATUS, TSR_STATUS_LOAD_REFUSED);
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOAD_REFUSED), "W1C");
    (void)rd(s, TSR_CSR_LOAD_DATA);
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOAD_REFUSED), "reads never set LOAD_REFUSED");
    verify_segment(s, 0x0, NI);
    CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOAD_REFUSED, "VERIFY refused");
    wr(s, TSR_CSR_CTRL, 0);
    CHECK(verify_segment(s, 0x0, NI) == it_crc, "loader back after RV_RUN = 0");
    free(itw);
    free(dtw);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
static void lq_push(tsr_sim * s, const uint32_t d[8]) {
    for (int i = 0; i < 8; ++i) wr(s, TSR_CSR_LQ_W(i), d[i]);
    wr(s, TSR_CSR_LQ_PUSH, 0);
}
static void lcq_post(tsr_sim * s, const uint32_t r[8], uint32_t answer, int expect_rc) {
    for (int i = 0; i < 8; ++i) CHECK(cww(s, TSR_CW_LCQ_W(i), r[i]) == TSR_CW_OK, "LCQ_W");
    CHECK(cww(s, TSR_CW_LCQ_POST, answer) == expect_rc, "LCQ_POST answer=%u rc expected %d", answer, expect_rc);
}

static void test_lq_lcq(void) {
    tsr_sim * s = mk_sim();
    uint32_t d[8];
    // 3 descriptors, popped in order by the RISC-V side
    for (uint32_t k = 0; k < 3; ++k) {
        for (int i = 0; i < 8; ++i) d[i] = 0x1000u * (k + 1) + (uint32_t)i;
        lq_push(s, d);
    }
    CHECK(rd(s, TSR_CSR_LQ_SEQ) == 3 && rd(s, TSR_CSR_LQ_TAKEN) == 0, "PUSHED 3");
    CHECK(rd(s, TSR_CSR_LQ_W(5)) == 0x3005u, "staging kept after a push");
    CHECK(cwr(s, TSR_CW_LQ_AVAIL) == 3 && cwr(s, TSR_CW_LQ_HEAD_SEQ) == 0, "LQ_AVAIL");
    CHECK(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(2), "irq_fast[2]: LQ_AVAIL != 0 and not ACTIVE");
    for (uint32_t k = 0; k < 3; ++k) {
        CHECK(cwr(s, TSR_CW_LQ_HEAD_SEQ) == k, "HEAD_SEQ %u", k);
        for (int i = 0; i < 8; ++i)
            CHECK(cwr(s, TSR_CW_LQ_HEAD_W(i)) == 0x1000u * (k + 1) + (uint32_t)i, "head word %d of %u", i, k);
        CHECK(cww(s, TSR_CW_LQ_POP, 0) == TSR_CW_OK, "pop %u", k);
        CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_ACTIVE, "ACTIVE");
        CHECK(cwr(s, TSR_CW_LQ_FLAGS) & TSR_LQ_FLAGS_ACTIVE, "LQ_FLAGS.ACTIVE");
        CHECK(!(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(2)), "irq_fast[2] off while ACTIVE");
        CHECK(cww(s, TSR_CW_LQ_POP, 0) == TSR_CW_FAULT, "pop while ACTIVE faults");
        CHECK(rd(s, TSR_CSR_LQ_TAKEN) == k + 1, "TAKEN");
        uint32_t r[8] = { 0x01u | (k << 24), k, 0, TSR_TICKET_NONE, 0, 0, 0, 100 + k };
        lcq_post(s, r, 1, TSR_CW_OK);
        CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_ACTIVE) && rd(s, TSR_CSR_LCQ_ANSWERED) == k + 1, "answered %u", k);
        lcq_post(s, r, 1, TSR_CW_FAULT);              // ANSWER while not ACTIVE
    }
    CHECK(cww(s, TSR_CW_LQ_POP, 0) == TSR_CW_FAULT && cwr(s, TSR_CW_LQ_AVAIL) == 0, "pop of an empty LQ faults");
    CHECK(cwr(s, TSR_CW_LQ_HEAD_W(0)) == 0, "head words read 0 while empty");
    // Host side of the ring: records 0..2 at 0x300 + 32 k
    CHECK(rd(s, TSR_CSR_LCQ_WR) == 3 && rd(s, TSR_CSR_LCQ_RD) == 0, "LCQ_WR");
    CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LCQ_PENDING, "LCQ_PENDING");
    for (uint32_t k = 0; k < 3; ++k)
        CHECK(rd(s, TSR_CSR_LCQ_REC(k, 0)) == (0x01u | (k << 24)) && rd(s, TSR_CSR_LCQ_REC(k, 7)) == 100 + k, "record %u", k);
    wr(s, TSR_CSR_LCQ_RD, 3);
    CHECK(rd(s, TSR_CSR_LCQ_RD) == 3 && !(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LCQ_PENDING), "LCQ_RD consumed");
    wr(s, TSR_CSR_LCQ_RD, 4);                          // beyond LCQ_WR: refused
    CHECK(rd(s, TSR_CSR_LCQ_RD) == 3 && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_RING_RD_BAD), "RING_RD_BAD");
    wr(s, TSR_CSR_STATUS, 0xFFFFFFFFu);
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_RING_RD_BAD), "W1C RING_RD_BAD");

    // LCQ flow control: events (ANSWER = 0) need no ACTIVE; 8 fill the ring, the 9th faults.
    CHECK(cwr(s, TSR_CW_LCQ_FREE) == 8 && (tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(3)), "LCQ_FREE 8, irq_fast[3]");
    uint32_t ev[8] = { 0xC0u, 0, 0, 0, 0, 0, 0, 0 };
    for (uint32_t k = 0; k < 8; ++k) { ev[2] = k; lcq_post(s, ev, 0, TSR_CW_OK); }
    CHECK(cwr(s, TSR_CW_LCQ_FREE) == 0 && !(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(3)), "LCQ full");
    lcq_post(s, ev, 0, TSR_CW_FAULT);
    CHECK(rd(s, TSR_CSR_LCQ_WR) == 11, "the faulting post changed nothing");
    // record k lives at slot k mod 8 until consumed
    for (uint32_t k = 3; k < 11; ++k) CHECK(rd(s, TSR_CSR_LCQ_REC(k % 8, 2)) == k - 3, "event record %u", k);
    wr(s, TSR_CSR_LCQ_RD, 7);
    CHECK(cwr(s, TSR_CW_LCQ_FREE) == 4, "partial consumption frees 4");
    wr(s, TSR_CSR_LCQ_RD, 11);

    // LQ flow control: PUSHED - TAKEN <= LQ_DEPTH; the 9th push is dropped with LQ_OVERFLOW.
    for (uint32_t k = 0; k < 8; ++k) { d[0] = k; lq_push(s, d); }
    CHECK(rd(s, TSR_CSR_LQ_SEQ) == 11 && !(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LQ_OVERFLOW), "8 queued");
    d[0] = 99;
    lq_push(s, d);
    CHECK(rd(s, TSR_CSR_LQ_SEQ) == 11 && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_LQ_OVERFLOW), "overflow dropped");
    CHECK(cwr(s, TSR_CW_LQ_HEAD_W(0)) == 0 && cwr(s, TSR_CW_LQ_HEAD_SEQ) == 3, "head unchanged");
    CHECK(cww(s, TSR_CW_LQ_POP, 0) == TSR_CW_OK && cwr(s, TSR_CW_LQ_HEAD_W(0)) == 1, "next head after a pop");
    // RT_STATE / RT_INFO mirror
    CHECK(cww(s, TSR_CW_RT_STATE, TSR_PUT(TSR_RT_STATE_PHASE, 4) | TSR_PUT(TSR_RT_STATE_KERNEL, 7)) == TSR_CW_OK, "RT_STATE");
    cww(s, TSR_CW_RT_INFO, 1234);
    CHECK(rd(s, TSR_CSR_RT_STATE) == (4u | (7u << 16)) && rd(s, TSR_CSR_RT_INFO) == 1234, "RT mirror");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
static void test_watermarks(void) {
    tsr_sim * s = mk_sim();
    uint32_t d[8] = { 1, 0, 0, 0, 0, 0, 0, 0 };
    for (int k = 0; k < 5; ++k) lq_push(s, d);
    // Launch 0 runs; a cancel watermark that does not cover it (s = 0 - 1) changes nothing.
    CHECK(cww(s, TSR_CW_LQ_POP, 0) == TSR_CW_OK, "pop 0");
    wr(s, TSR_CSR_LQ_STOP, 3);
    CHECK((rd(s, TSR_CSR_LQ_CFG) & TSR_LQ_CFG_STOP_VALID) && rd(s, TSR_CSR_LQ_STOP) == 3, "STOP_VALID");
    CHECK(cwr(s, TSR_CW_LQ_STOP) == 3 && (cwr(s, TSR_CW_LQ_FLAGS) & TSR_LQ_FLAGS_STOP_VALID), "RISC-V copy of the stop watermark");
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR), "a soft stop has no hardware action");
    wr(s, TSR_CSR_LQ_CANCEL, 0xFFFFFFFFu);
    CHECK((rd(s, TSR_CSR_LQ_CFG) & TSR_LQ_CFG_CANCEL_VALID) && (cwr(s, TSR_CW_LQ_FLAGS) & TSR_LQ_FLAGS_CANCEL_VALID) &&
          cwr(s, TSR_CW_LQ_CANCEL) == 0xFFFFFFFFu, "CANCEL_VALID");
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR), "an uncovering cancel aborts nothing ((s - q) mod 2^32 >= 2^31)");
    // Covering cancel while launch 0 runs: host QABORT at once, every built unit in error (ABORTED).
    wr(s, TSR_CSR_LQ_CANCEL, 2);
    CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR, "covered: QABORT");
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const uint32_t want = (u == TSR_UNIT_ATT) ? 0u : TSR_ERR_ABORTED;
        CHECK(rd(s, TSR_CSR_Q_ERR_CODE(u)) == want && cwr(s, TSR_CW_ERR_CODE(u)) == want, "unit %u error code", u);
        CHECK(rd(s, TSR_CSR_Q_ERR_SEQ(u)) == 0, "ERR_SEQ = S = 0");
    }
    // The cover boundary (EX §4.9): s - q = 2^31 does not cover, 2^31 - 1 does (q = ANSWERED = 0).
    tsr_sim * s3 = mk_sim();
    lq_push(s3, d);
    CHECK(cww(s3, TSR_CW_LQ_POP, 0) == TSR_CW_OK, "pop 0");
    wr(s3, TSR_CSR_LQ_CANCEL, 0x80000000u);
    CHECK(!(rd(s3, TSR_CSR_STATUS) & TSR_STATUS_QERR), "s = ANSWERED + 2^31: not covered");
    wr(s3, TSR_CSR_LQ_CANCEL, 0x7FFFFFFFu);
    CHECK(rd(s3, TSR_CSR_STATUS) & TSR_STATUS_QERR, "s = ANSWERED + 2^31 - 1: covered, QABORT");
    tsr_sim_destroy(s3);
    // Not ACTIVE: a cancel has no hardware action (the runtime answers covered launches ABORTED).
    tsr_sim * s2 = mk_sim();
    lq_push(s2, d);
    wr(s2, TSR_CSR_LQ_CANCEL, 5);
    CHECK(!(rd(s2, TSR_CSR_STATUS) & TSR_STATUS_QERR), "cancel while not ACTIVE: no abort");
    tsr_sim_destroy(s2);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
static uint32_t log_hdr(uint32_t n, uint32_t level, uint32_t src, uint32_t type, uint32_t lseq) {
    return TSR_PUT(TSR_LOG_HDR_N, n) | TSR_PUT(TSR_LOG_HDR_LEVEL, level) | TSR_PUT(TSR_LOG_HDR_SOURCE, src) |
           TSR_PUT(TSR_LOG_HDR_TYPE, type) | TSR_PUT(TSR_LOG_HDR_LSEQ, lseq) | TSR_PUT(TSR_LOG_HDR_SYNC, TSR_LOG_HDR_SYNC_MAGIC);
}

static void test_log(void) {
    tsr_sim * s = mk_sim();
    CHECK(cwr(s, TSR_CW_LOG_FREE) == 256 && cwr(s, TSR_CW_LOG_CFG) == 0xF1, "LOG_FREE, LOG_CFG copy");
    // 40 records of 1 + (k % 5) words, the way the SDK writes them (only with LOG_FREE >= 1 + N).
    uint32_t words = 0;
    for (uint32_t k = 0; k < 40; ++k) {
        const uint32_t n = k % 5;
        CHECK(cwr(s, TSR_CW_LOG_FREE) >= 1 + n, "space");
        cww(s, TSR_CW_LOG_PUSH, log_hdr(n, TSR_LOG_HDR_LEVEL_INFO, TSR_LOG_HDR_SOURCE_KERNEL, TSR_LOG_HDR_TYPE_EVENT, k));
        for (uint32_t i = 0; i < n; ++i) cww(s, TSR_CW_LOG_PUSH, 1000 * k + i);
        words += 1 + n;
    }
    CHECK(rd(s, TSR_CSR_LOG_WR) == words && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOG_PENDING), "LOG_WR %u", words);
    // Host parse: header SYNC / N / LSEQ, payload
    uint32_t k = rd(s, TSR_CSR_LOG_RD), rec = 0;
    while (k != rd(s, TSR_CSR_LOG_WR)) {
        const uint32_t h = rd(s, TSR_CSR_LOG_RING(k % 256));
        CHECK(TSR_GET(h, TSR_LOG_HDR_SYNC) == 0xA && TSR_GET(h, TSR_LOG_HDR_LSEQ) == rec && TSR_GET(h, TSR_LOG_HDR_N) == rec % 5,
              "record %u header 0x%08x", rec, h);
        for (uint32_t i = 0; i < TSR_GET(h, TSR_LOG_HDR_N); ++i)
            CHECK(rd(s, TSR_CSR_LOG_RING((k + 1 + i) % 256)) == 1000 * rec + i, "record %u payload %u", rec, i);
        k += 1 + TSR_GET(h, TSR_LOG_HDR_N);
        rec++;
    }
    CHECK(rec == 40, "40 records parsed");
    wr(s, TSR_CSR_LOG_RD, k);
    CHECK(cwr(s, TSR_CW_LOG_FREE) == 256 && !(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOG_PENDING), "consumed");
    // Overflow: fill 256 words, then 5 more are dropped and counted; nothing faults.
    for (uint32_t i = 0; i < 256; ++i) CHECK(cww(s, TSR_CW_LOG_PUSH, i) == TSR_CW_OK, "push");
    CHECK(cwr(s, TSR_CW_LOG_FREE) == 0 && rd(s, TSR_CSR_LOG_DROPPED) == 0, "full");
    for (uint32_t i = 0; i < 5; ++i) CHECK(cww(s, TSR_CW_LOG_PUSH, 0xBAD0u + i) == TSR_CW_OK, "dropping push never faults");
    CHECK(rd(s, TSR_CSR_LOG_DROPPED) == 5 && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOG_DROP), "LOG_DROPPED 5, LOG_DROP");
    CHECK(rd(s, TSR_CSR_LOG_WR) == k + 256, "dropped words not accepted");
    for (uint32_t i = 0; i < 256; ++i) CHECK(rd(s, TSR_CSR_LOG_RING((k + i) % 256)) == i, "ring word %u kept (drop-new)", i);
    wr(s, TSR_CSR_LOG_RD, k + 300);                    // beyond LOG_WR
    CHECK(rd(s, TSR_CSR_LOG_RD) == k && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_RING_RD_BAD), "LOG_RD rule");
    wr(s, TSR_CSR_LOG_RD, k + 256);
    CHECK(cwr(s, TSR_CW_LOG_FREE) == 256, "free again");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// PERF: gated by ACTIVE (EX §6); the LO read snapshot.
static const uint32_t PUSH_ALL = 0xFF00u;

static void test_perf(void) {
    tsr_sim * s = mk_sim();
    const uint32_t C = 4;
    // Not ACTIVE: ops run but no counter moves.
    CHECK(tsr_sim_push(s, tsr_enc_ld(1, 0, 0, 4).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL) == 0u, "ticket {0, 0}");
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) CHECK(tsr_sim_perf(s, i) == 0, "PERF %u = 0 outside ACTIVE", i);
    // ACTIVE: one launch popped by the RISC-V side.
    uint32_t d[8] = { 1, 0, 0, 0, 0, 0, 0, 0 };
    lq_push(s, d);
    CHECK(cww(s, TSR_CW_LQ_POP, 0) == TSR_CW_OK, "pop");
    const uint64_t t0 = tsr_sim_now(s);
    tsr_sim_push(s, tsr_enc_ld(1, 0, 0, 7).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_push(s, tsr_enc_ld(2, 100, 0, 3).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_push(s, tsr_enc_nop().w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_push(s, insn_raw(0x01u, 0, 0, 0, 0, 0, 0, 0).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);   // END: ILLEGAL_OP
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    const uint64_t dt = tsr_sim_now(s) - t0;
    CHECK(tsr_sim_perf(s, TSR_PERF_CYCLES) == dt, "CYCLES %llu = elapsed %llu", (unsigned long long)tsr_sim_perf(s, 0),
          (unsigned long long)dt);
    CHECK(tsr_sim_perf(s, TSR_PERF_RETIRED) == 4, "RETIRED 4 (any outcome)");
    CHECK(tsr_sim_perf(s, TSR_PERF_CH_RD_BEATS(1)) == 7 && tsr_sim_perf(s, TSR_PERF_CH_RD_BEATS(2)) == 3, "CH_RD_BEATS");
    CHECK(tsr_sim_perf(s, TSR_PERF_DMA_CYCLES(C)) == (64 + 7) + (64 + 3), "DMA_CYCLES %llu",
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_DMA_CYCLES(C)));
    CHECK(tsr_sim_perf(s, TSR_PERF_PUSHES(C)) == 4, "PUSHES");
    CHECK(tsr_sim_perf(s, TSR_PERF_MVM_EMPTY(C)) == dt && tsr_sim_perf(s, TSR_PERF_MVM_CYCLES) == 0, "MVM idle and empty");
    // An MVM: MVM_CYCLES = its cycle model; CYCLES - MVM_CYCLES - MVM_WAIT - MVM_EMPTY small.
    const uint32_t K = 256, N = 64;
    const tsr_insn mvm = tsr_enc_mvm(TSR_FMT_Q8_0, 0, 1000, 2000, 0, N, K, 1, 0);
    const uint32_t lt = tsr_sim_push(s, tsr_enc_ld(0, 3000, 0, 50).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_push(s, mvm.w, lt, TSR_TICKET_NONE, PUSH_ALL);   // waits for the LD: MVM_WAIT
    CHECK(tsr_sim_run_until_idle(s, 100000), "idle");
    const uint64_t mvm_cyc = 64 + tsr_qa_lines(TSR_QA_A32, K) + (uint64_t)(N / 4) * tsr_row_beats(TSR_FMT_Q8_0, K);
    CHECK(tsr_sim_perf(s, TSR_PERF_MVM_CYCLES) == mvm_cyc, "MVM_CYCLES %llu, expected %llu",
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_MVM_CYCLES), (unsigned long long)mvm_cyc);
    CHECK(tsr_sim_perf(s, TSR_PERF_MVM_WAIT(C)) >= 100, "MVM_WAIT %llu", (unsigned long long)tsr_sim_perf(s, TSR_PERF_MVM_WAIT(C)));
    const uint64_t rem = tsr_sim_perf(s, TSR_PERF_CYCLES) - tsr_sim_perf(s, TSR_PERF_MVM_CYCLES) -
                         tsr_sim_perf(s, TSR_PERF_MVM_WAIT(C)) - tsr_sim_perf(s, TSR_PERF_MVM_EMPTY(C));
    CHECK(rem <= 3, "CYCLES - MVM_CYCLES - MVM_WAIT - MVM_EMPTY = %llu <= 3 per MVM entry", (unsigned long long)rem);
    // Answer: LAUNCHES counted; then nothing changes between launches.
    uint32_t r[8] = { 1, 0, 0, TSR_TICKET_NONE, 0, 0, 0, 0 };
    lcq_post(s, r, 1, TSR_CW_OK);
    CHECK(tsr_sim_perf(s, TSR_PERF_LAUNCHES(C)) == 1, "LAUNCHES");
    uint64_t before[TSR_PERF_SLOTS];
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) before[i] = tsr_sim_perf(s, i);
    tsr_sim_push(s, tsr_enc_ld(1, 0, 0, 4).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_run_until_idle(s, 100000);
    CHECK(tsr_sim_advance(s, 5000) == TSR_SIM_EV_BUDGET, "budget");
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) CHECK(tsr_sim_perf(s, i) == before[i], "PERF %u unchanged between launches", i);
    // PERF CSR view and the LO snapshot: CYCLES past 2^32 within one launch.
    lq_push(s, d);
    cww(s, TSR_CW_LQ_POP, 0);
    tsr_sim_advance(s, (1ull << 32) + 5);
    const uint64_t c1 = tsr_sim_perf(s, TSR_PERF_CYCLES);
    CHECK(c1 > (1ull << 32), "CYCLES above 2^32");
    const uint32_t lo = rd(s, TSR_CSR_PERF_LO(TSR_PERF_CYCLES));
    tsr_sim_advance(s, 1ull << 32);                    // the live counter moves on
    CHECK(rd(s, TSR_CSR_PERF_HI(TSR_PERF_CYCLES)) == (uint32_t)(c1 >> 32) && lo == (uint32_t)c1, "HI from the LO snapshot");
    CHECK(tsr_sim_corewin_read(s, TSR_CSR_PERF_HI(TSR_PERF_CYCLES)) == (uint32_t)(tsr_sim_perf(s, 0) >> 32),
          "core window HI is live (no snapshot)");
    (void)rd(s, TSR_CSR_PERF_LO(TSR_PERF_RETIRED));    // a LO read of another counter replaces the snapshot
    CHECK(rd(s, TSR_CSR_PERF_HI(TSR_PERF_CYCLES)) == (uint32_t)(tsr_sim_perf(s, 0) >> 32), "snapshot is per counter");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// SOFT_RESET (EX §8.5): clears and keeps; SR_BUSY over a running op.
static void test_soft_reset(void) {
    tsr_sim * s = mk_sim();
    uint8_t * dram = tsr_sim_dram(s, 0);
    for (int i = 0; i < 4096; ++i) dram[i] = (uint8_t)(i * 7);
    // Configuration and staging that must survive
    wr(s, TSR_CSR_RV_BOOT, 0x400);
    wr(s, TSR_CSR_LOG_CFG, 0x43);
    wr(s, TSR_CSR_WDOG_CFG, 9);
    uint32_t d[8] = { 1, 11, 22, 33, 44, 55, 66, 77 };
    for (int k = 0; k < 3; ++k) lq_push(s, d);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
    cww(s, TSR_CW_LQ_POP, 0);                          // launch 0 runs
    cww(s, TSR_CW_RT_STATE, 4);
    wr(s, TSR_CSR_LQ_CANCEL, 0xFFFFFFF0u);
    wr(s, TSR_CSR_LQ_STOP, 0xFFFFFFF0u);
    uint32_t ev[8] = { 0xC1u, 0, 0, 0, 0, 0, 0, 0 };
    lcq_post(s, ev, 0, TSR_CW_OK);
    cww(s, TSR_CW_LOG_PUSH, 0xA0000000u);
    for (uint32_t i = 0; i < 258; ++i) cww(s, TSR_CW_LOG_PUSH, i);   // 255 fill the ring, 3 dropped
    CHECK(rd(s, TSR_CSR_LOG_DROPPED) == 3 && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_LOG_DROP), "LOG_DROPPED 3 before SOFT_RESET");
    tsr_sim_sreg(s)[2] = 3.25f;
    tsr_sim_itcm(s, NULL)[100] = 0x5A;
    for (int k = 0; k < 7; ++k) wr(s, TSR_CSR_LQ_PUSH, 0);   // 2 + 6 fill the LQ, the 7th: LQ_OVERFLOW
    CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_LQ_OVERFLOW, "sticky set");
    // Work: a long MVM running, an error poisoning the DMA unit, PERF counting (ACTIVE).
    const uint32_t K = 4096, N = 2048;
    tsr_sim_push(s, tsr_enc_mvm(TSR_FMT_Q8_0, 0, 2000, 1000, 0, N, K, 1, 0).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_push(s, tsr_enc_ld(0, 0xFFFFFFF0u, 0, 1).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);   // SPM_RANGE
    tsr_sim_push(s, tsr_enc_ld(0, 10, 0, 1).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_advance(s, 200);
    CHECK(rd(s, TSR_CSR_Q_STATE(TSR_UNIT_MVM)) & TSR_Q_STATE_RUNNING, "MVM running");
    CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR, "DMA in error");
    CHECK(tsr_sim_perf(s, TSR_PERF_CYCLES) > 0, "PERF counting");
    const uint32_t pidx = cwr(s, TSR_CW_ISSUE_PIDX), iss_mvm = cwr(s, TSR_CW_ISSUED_SEQ(TSR_UNIT_MVM));
    const uint32_t lq_seq = rd(s, TSR_CSR_LQ_SEQ), lcq_wr = rd(s, TSR_CSR_LCQ_WR), log_wr = rd(s, TSR_CSR_LOG_WR);
    const uint8_t spm10 = tsr_sim_spm(s)[10 * 64];

    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN | TSR_CTRL_SOFT_RESET);
    uint32_t st = rd(s, TSR_CSR_STATUS);
    CHECK((st & TSR_STATUS_SR_BUSY) && !(st & TSR_STATUS_RV_RUNNING) && (st & TSR_STATUS_RV_RESET_DONE),
          "SR_BUSY while the MVM runs; RISC-V reset at once (0x%x)", st);
    CHECK(rd(s, TSR_CSR_RT_STATE) == 0 && !(st & TSR_STATUS_ACTIVE), "RISC-V reset: RT_STATE 0, ANSWERED := TAKEN");
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
    CHECK(!(rd(s, TSR_CSR_CTRL) & TSR_CTRL_RV_RUN) && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_RV_RUN_REFUSED), "RV_RUN refused");
    uint32_t evs = 0;
    for (int i = 0; i < 100 && !(evs & TSR_SIM_EV_RESET); ++i) evs = tsr_sim_advance(s, 1000000);
    CHECK(evs & TSR_SIM_EV_RESET, "SR_BUSY fell (advance event)");
    st = rd(s, TSR_CSR_STATUS);
    CHECK(!(st & TSR_STATUS_SR_BUSY) && (st & TSR_STATUS_QIDLE) && !(st & TSR_STATUS_QERR), "drained, errors cleared 0x%x", st);
    CHECK((st & 0x0FFF0000u) == 0 && rd(s, TSR_CSR_LOG_DROPPED) == 0, "sticky bits cleared");
    CHECK(rd(s, TSR_CSR_LQ_TAKEN) == lq_seq && rd(s, TSR_CSR_LCQ_ANSWERED) == lq_seq, "TAKEN := PUSHED, ANSWERED := TAKEN");
    CHECK(rd(s, TSR_CSR_LCQ_RD) == lcq_wr && rd(s, TSR_CSR_LOG_RD) == log_wr, "LCQ_RD := LCQ_WR, LOG_RD := LOG_WR");
    CHECK(!(rd(s, TSR_CSR_LQ_CFG) & (TSR_LQ_CFG_STOP_VALID | TSR_LQ_CFG_CANCEL_VALID)), "watermarks invalid");
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) CHECK(tsr_sim_perf(s, i) == 0, "PERF %u zeroed", i);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) CHECK(rd(s, TSR_CSR_Q_ERR_CODE(u)) == 0, "record %u cleared", u);
    // Kept
    CHECK(rd(s, TSR_CSR_LQ_SEQ) == lq_seq && rd(s, TSR_CSR_LCQ_WR) == lcq_wr && rd(s, TSR_CSR_LOG_WR) == log_wr, "counters kept");
    CHECK(cwr(s, TSR_CW_ISSUE_PIDX) == pidx && cwr(s, TSR_CW_ISSUED_SEQ(TSR_UNIT_MVM)) == iss_mvm &&
          cwr(s, TSR_CW_DONE_SEQ(TSR_UNIT_MVM)) == iss_mvm, "sequences kept");
    CHECK(rd(s, TSR_CSR_CTRL) == (TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN), "level bits kept, RV_RUN 0");
    CHECK(rd(s, TSR_CSR_RV_BOOT) == 0x400 && rd(s, TSR_CSR_LOG_CFG) == 0x43 && rd(s, TSR_CSR_WDOG_CFG) == 9, "config kept");
    CHECK(rd(s, TSR_CSR_LQ_W(7)) == 77, "LQ staging kept");
    CHECK(f2u(tsr_sim_sreg(s)[2]) == f2u(3.25f) && tsr_sim_itcm(s, NULL)[100] == 0x5A && dram[4095] == (uint8_t)(4095 * 7),
          "s bank, TCM, DRAM kept");
    CHECK(tsr_sim_spm(s)[10 * 64] == spm10, "SPM line 10 kept (the LD after the poisoned one was aborted)");
    // The machine works again; tickets keep counting.
    const uint32_t t = tsr_sim_push(s, tsr_enc_ld(0, 10, 0, 1).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    CHECK(t == 2u, "ticket continues {0, 2}: 0x%08x", t);
    tsr_sim_run_until_idle(s, 10000);
    CHECK(cwr(s, TSR_CW_ERR_CODE(0)) == 0 && cwr(s, TSR_CW_DONE_SEQ(0)) == 3, "runs normally after SOFT_RESET");
    // A SOFT_RESET with nothing in flight completes at once.
    wr(s, TSR_CSR_CTRL, TSR_CTRL_SOFT_RESET);
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_SR_BUSY), "immediate SOFT_RESET");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
static void test_status_ctrl(void) {
    tsr_sim * s = mk_sim();
    wr(s, TSR_CSR_CTRL, TSR_CTRL_HOST_HOLD);
    CHECK(TSR_GET(rd(s, TSR_CSR_STATUS), TSR_STATUS_HOLD_SRC) == TSR_HOLD_SRC_HOST, "HOLD_SRC host");
    CHECK(cwr(s, TSR_CW_QSTATUS) & TSR_QSTATUS_HOLD_HOST, "QSTATUS host hold");
    // Pulse bits read 0; the host QABORT and QCLEAR
    wr(s, TSR_CSR_CTRL, TSR_CTRL_HOST_HOLD | TSR_CTRL_QABORT);
    CHECK(rd(s, TSR_CSR_CTRL) == TSR_CTRL_HOST_HOLD && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR), "QABORT");
    CHECK(cwr(s, TSR_CW_ISSUE_STATUS) & TSR_ISSUE_STATUS_ERR_IRQ, "ERR_IRQ on the rising edge");
    CHECK(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(1), "irq_fast[1]");
    cww(s, TSR_CW_ISSUE_STATUS, TSR_ISSUE_STATUS_ERR_IRQ);
    CHECK(!(cwr(s, TSR_CW_ISSUE_STATUS) & TSR_ISSUE_STATUS_ERR_IRQ), "W1C ERR_IRQ");
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);              // RISC-V running: host QCLEAR refused
    CHECK(rd(s, TSR_CSR_STATUS) & TSR_STATUS_RV_RUNNING, "RV_RUNNING");
    CHECK(tsr_sim_rv_run(s) == 1 && tsr_sim_rv_boot_gen(s) == 1, "boot generation");
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_QCLEAR);
    CHECK((rd(s, TSR_CSR_STATUS) & (TSR_STATUS_QCLEAR_REFUSED | TSR_STATUS_QERR)) == (TSR_STATUS_QCLEAR_REFUSED | TSR_STATUS_QERR),
          "QCLEAR refused while the RISC-V runs");
    wr(s, TSR_CSR_CTRL, 0);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_QCLEAR);
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR), "QCLEAR accepted: idle, RISC-V in reset");
    // A QCLEAR in a write that changes RV_RUN is judged on the state before the change (EX §5.2 bit 9):
    // RV_RUN 1 -> 0 starts a reset that has not completed, so the QCLEAR is refused (the reset's own
    // abort keeps the errors); RV_RUN 0 -> 1 with the hart in reset is accepted.
    wr(s, TSR_CSR_STATUS, 0xFFFFFFFFu);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_QABORT);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_QCLEAR);               // RV_RUN 1 -> 0 and QCLEAR
    CHECK((rd(s, TSR_CSR_STATUS) & (TSR_STATUS_QCLEAR_REFUSED | TSR_STATUS_QERR)) == (TSR_STATUS_QCLEAR_REFUSED | TSR_STATUS_QERR),
          "QCLEAR refused in the write that drops RV_RUN");
    wr(s, TSR_CSR_STATUS, 0xFFFFFFFFu);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_QCLEAR);   // RV_RUN 0 -> 1 and QCLEAR
    CHECK((rd(s, TSR_CSR_STATUS) & (TSR_STATUS_QCLEAR_REFUSED | TSR_STATUS_QERR)) == 0 && tsr_sim_rv_run(s) == 1,
          "QCLEAR accepted in the write that releases the RISC-V from reset");
    wr(s, TSR_CSR_CTRL, 0);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_QCLEAR);
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR), "QCLEAR accepted again with the RISC-V in reset");
    wr(s, TSR_CSR_STATUS, 0xFFFFFFFFu);
    // Not idle: refused
    wr(s, TSR_CSR_STATUS, 0xFFFFFFFFu);
    tsr_sim_push(s, tsr_enc_ld(0, 0, 0, 1000).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_QABORT | TSR_CTRL_QCLEAR);
    CHECK((rd(s, TSR_CSR_STATUS) & TSR_STATUS_QCLEAR_REFUSED) && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR), "refused: not QIDLE");
    tsr_sim_run_until_idle(s, 100000);
    CHECK(cwr(s, TSR_CW_ERR_CODE(0)) == TSR_ERR_ABORTED && cwr(s, TSR_CW_ERR_SEQ(0)) == 0, "the LD was aborted before it started");
    // MTIME and the timer
    CHECK(cwr(s, TSR_CW_MTIME_LO) == (uint32_t)tsr_sim_now(s) && cwr(s, TSR_CW_MTIMECMP_HI) == 0xFFFFFFFFu, "MTIME, MTIMECMP reset");
    cww(s, TSR_CW_MTIMECMP_HI, 0);
    cww(s, TSR_CW_MTIMECMP_LO, (uint32_t)tsr_sim_now(s) + 500);
    CHECK(!(tsr_sim_rv_irq(s) & TSR_RV_MIP_MTIP), "timer not yet");
    tsr_sim_advance(s, 499);
    CHECK(!(tsr_sim_rv_irq(s) & TSR_RV_MIP_MTIP), "timer not yet (499)");
    tsr_sim_advance(s, 1);
    CHECK(tsr_sim_rv_irq(s) & TSR_RV_MIP_MTIP, "irq_timer at MTIMECMP");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// RISC-V reset sequence (RV_RUN 1 -> 0; EX §8.5, CC §12.2): QABORT at once, ANSWERED := TAKEN, the
// clk_rv-side registers reset; rings, sequences and queued descriptors kept.
static void test_rv_reset(void) {
    tsr_sim * s = mk_sim();
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    uint32_t d[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    lq_push(s, d);
    lq_push(s, d);
    cww(s, TSR_CW_LQ_POP, 0);
    cww(s, TSR_CW_ISSUE_W(3), 7);
    cww(s, TSR_CW_ISSUE_DEP0, 5);
    cww(s, TSR_CW_WAIT_TICKET, 5);
    cww(s, TSR_CW_WAIT_CREDIT, TSR_WAIT_CREDIT_EN | 1u);
    cww(s, TSR_CW_LCQ_W(2), 9);
    cww(s, TSR_CW_RT_STATE, 4);
    cww(s, TSR_CW_RT_INFO, 5);
    cww(s, TSR_CW_MTIMECMP_HI, 0);
    cww(s, TSR_CW_MTIMECMP_LO, 100);
    tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_SET);
    const uint32_t t = tsr_sim_push(s, tsr_enc_ld(0, 0, 0, 1).w, TSR_TICKET_NONE, TSR_TICKET_NONE, PUSH_ALL);
    tsr_sim_advance(s, 50);                            // dispatched and held
    CHECK(rd(s, TSR_CSR_Q_STATE(0)) & TSR_Q_STATE_HELD, "held by the RISC-V hold source");
    const uint32_t pidx = cwr(s, TSR_CW_ISSUE_PIDX);
    wr(s, TSR_CSR_CTRL, 0);
    CHECK(cwr(s, TSR_CW_ISSUE_W(3)) == 0 && cwr(s, TSR_CW_ISSUE_DEP0) == TSR_TICKET_NONE, "staging, deps reset");
    CHECK(cwr(s, TSR_CW_WAIT_TICKET) == TSR_TICKET_NONE && cwr(s, TSR_CW_WAIT_CREDIT) == 0, "WAIT_* reset");
    CHECK(cwr(s, TSR_CW_LCQ_W(2)) == 0 && cwr(s, TSR_CW_RT_STATE) == 0 && rd(s, TSR_CSR_RT_INFO) == 0, "LCQ_W, RT_* reset");
    CHECK(cwr(s, TSR_CW_MTIMECMP_LO) == 0xFFFFFFFFu && cwr(s, TSR_CW_ISSUE_STATUS) == 0, "MTIMECMP, ISSUE_STATUS reset");
    CHECK(!(cwr(s, TSR_CW_QSTATUS) & TSR_QSTATUS_HOLD_RV), "RISC-V hold source reset");
    CHECK(!(rd(s, TSR_CSR_STATUS) & TSR_STATUS_ACTIVE) && rd(s, TSR_CSR_LCQ_ANSWERED) == 1 && rd(s, TSR_CSR_LCQ_WR) == 0,
          "ANSWERED := TAKEN without a record");
    CHECK(cwr(s, TSR_CW_LQ_AVAIL) == 1 && cwr(s, TSR_CW_ISSUE_PIDX) == pidx && cwr(s, TSR_CW_ISSUED_SEQ(0)) == 1,
          "queued descriptor and sequences kept");
    CHECK(tsr_sim_run_until_idle(s, 1000), "idle");
    CHECK(cwr(s, TSR_CW_ERR_CODE(0)) == TSR_ERR_ABORTED && cwr(s, TSR_CW_ERR_SEQ(0)) == (t & TSR_TICKET_SEQ) &&
          (rd(s, TSR_CSR_STATUS) & TSR_STATUS_QERR), "the held op was aborted; error state kept for the next boot's R1");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// One launch end to end with a scripted RISC-V agent on the time base (EX §4.7 R2-R8 in miniature):
// pop, push an LD described by the descriptor, join by polling DONE_SEQ, answer.
typedef struct { int state; uint32_t seq, ticket, a0, a1, steps; uint64_t t_pop; } fake_rt;

static uint64_t fake_step(void * ctx, tsr_sim * s) {
    fake_rt * f = (fake_rt *)ctx;
    uint32_t v;
    f->steps++;
    switch (f->state) {
    case 0:                                            // R2: take
        tsr_sim_cw_access(s, 0, TSR_CW_LQ_AVAIL, &v);
        if (v == 0) return TSR_SIM_AGENT_WAIT;        // WFI on irq_fast[2]
        tsr_sim_cw_access(s, 0, TSR_CW_LQ_HEAD_SEQ, &f->seq);
        tsr_sim_cw_access(s, 0, TSR_CW_LQ_HEAD_W(1), &f->a0);
        tsr_sim_cw_access(s, 0, TSR_CW_LQ_HEAD_W(2), &f->a1);
        CHECK(tsr_sim_cw_access(s, 1, TSR_CW_LQ_POP, &v) == TSR_CW_OK, "agent pop");
        f->t_pop = tsr_sim_now(s);
        f->state = 1;
        return 12;
    case 1: {                                          // R5: the kernel pushes one LD
        const tsr_insn ld = tsr_enc_ld(0, f->a1, f->a0, 4);
        for (int i = 0; i < 8; ++i) { v = ld.w[i]; tsr_sim_cw_access(s, 1, TSR_CW_ISSUE_W(i), &v); }
        v = PUSH_ALL;
        if (tsr_sim_cw_access(s, 1, TSR_CW_ISSUE_PUSH, &v) == TSR_CW_WAIT) return 3;
        tsr_sim_cw_access(s, 0, TSR_CW_ISSUE_TICKET, &f->ticket);
        f->state = 2;
        return 20;
    }
    case 2:                                            // R6: join by polling
        tsr_sim_cw_access(s, 0, TSR_CW_DONE_SEQ(0), &v);
        if (((v - (f->ticket & TSR_TICKET_SEQ)) & 0x0FFFFFFFu) == 0 || ((v - (f->ticket & TSR_TICKET_SEQ)) & 0x0FFFFFFFu) >= 0x08000000u)
            return 7;
        f->state = 3;
        return 5;
    case 3: {                                          // R7 / R8: answer DONE / OK
        uint32_t r[8] = { TSR_LCQ_R0_TYPE_DONE | ((f->seq & 0xFF) << 24), f->seq, 0, TSR_TICKET_NONE, 0, 0, 0,
                          (uint32_t)(tsr_sim_now(s) - f->t_pop) };
        for (int i = 0; i < 8; ++i) tsr_sim_cw_access(s, 1, TSR_CW_LCQ_W(i), &r[i]);
        v = TSR_LCQ_POST_ANSWER;
        if (tsr_sim_cw_access(s, 1, TSR_CW_LCQ_POST, &v) != TSR_CW_OK) return 10;
        f->state = 0;
        return 4;
    }
    default: return TSR_SIM_AGENT_WAIT;
    }
}

static void test_launch_agent(void) {
    tsr_sim * s = mk_sim();
    fake_rt f;
    memset(&f, 0, sizeof(f));
    tsr_sim_set_rv_agent(s, fake_step, &f);
    uint8_t * dram = tsr_sim_dram(s, 0);
    for (int i = 0; i < 4 * 64; ++i) dram[5 * 64 + i] = (uint8_t)(i ^ 0x3C);
    CHECK(tsr_sim_advance(s, 100) == TSR_SIM_EV_BUDGET && f.steps == 0, "the agent does not run in reset");
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    CHECK(tsr_sim_advance(s, 1000) == TSR_SIM_EV_BUDGET && f.steps == 1, "the agent sleeps with an empty LQ (steps %u)", f.steps);
    for (uint32_t k = 0; k < 3; ++k) {
        uint32_t d[8] = { TSR_DESC_TYPE_LAUNCH, 5, 200 + 10 * k, 0, 0, 0, 0, 0 };
        lq_push(s, d);
        const uint64_t t0 = tsr_sim_now(s);
        uint32_t ev = 0;
        while (!(ev & TSR_SIM_EV_LCQ)) ev = tsr_sim_advance(s, 100000);
        const uint32_t r0 = rd(s, TSR_CSR_LCQ_REC(k, 0));
        CHECK(TSR_GET(r0, TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && rd(s, TSR_CSR_LCQ_REC(k, 1)) == k, "DONE record of launch %u", k);
        CHECK(rd(s, TSR_CSR_LCQ_REC(k, 7)) > 64 && tsr_sim_now(s) - t0 > 64, "duration covers the LD");
        CHECK(memcmp(tsr_sim_spm(s) + (200 + 10 * k) * 64, dram + 5 * 64, 4 * 64) == 0, "the LD of launch %u ran", k);
        wr(s, TSR_CSR_LCQ_RD, rd(s, TSR_CSR_LCQ_WR));
    }
    CHECK(tsr_sim_perf(s, TSR_PERF_LAUNCHES(4)) == 3 && tsr_sim_perf(s, TSR_PERF_RETIRED) == 3 &&
          tsr_sim_perf(s, TSR_PERF_CH_RD_BEATS(0)) == 12, "PERF of three launches");
    // RV_RUN 1 -> 0 stops the agent; 0 -> 1 restarts it (new boot generation).
    wr(s, TSR_CSR_CTRL, 0);
    const uint32_t steps = f.steps;
    tsr_sim_advance(s, 1000);
    CHECK(f.steps == steps && tsr_sim_rv_boot_gen(s) == 1, "no steps in reset");
    tsr_sim_destroy(s);
}

// An agent that arms the timer and sleeps (WFI): the time base wakes it at MTIMECMP.
typedef struct { int armed; uint64_t woke, calls; } timer_rt;
static uint64_t timer_step(void * ctx, tsr_sim * s) {
    timer_rt * t = (timer_rt *)ctx;
    t->calls++;
    if (!t->armed) {
        uint32_t v = 0;
        tsr_sim_cw_access(s, 1, TSR_CW_MTIMECMP_HI, &v);
        v = (uint32_t)tsr_sim_now(s) + 5000;
        tsr_sim_cw_access(s, 1, TSR_CW_MTIMECMP_LO, &v);
        t->armed = 1;
        tsr_sim_rv_set_state(s, TSR_STATUS_RV_SLEEP);
        return TSR_SIM_AGENT_WAIT;
    }
    if (!(tsr_sim_rv_irq(s) & TSR_RV_MIP_MTIP)) return TSR_SIM_AGENT_WAIT;   // spurious wake: sleep on
    if (t->woke == 0) t->woke = tsr_sim_now(s);
    tsr_sim_rv_set_state(s, 0);
    uint32_t v = 0xFFFFFFFFu;
    tsr_sim_cw_access(s, 1, TSR_CW_MTIMECMP_HI, &v);     // disarm
    return TSR_SIM_AGENT_WAIT;
}

static void test_agent_timer(void) {
    tsr_sim * s = mk_sim();
    timer_rt t;
    memset(&t, 0, sizeof(t));
    tsr_sim_set_rv_agent(s, timer_step, &t);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    const uint64_t t0 = tsr_sim_now(s);
    tsr_sim_advance(s, 2000);
    CHECK(t.armed && t.woke == 0 && (rd(s, TSR_CSR_STATUS) & TSR_STATUS_RV_SLEEP), "asleep");
    tsr_sim_advance(s, 100000);
    CHECK(t.woke == t0 + 5000 && !(rd(s, TSR_CSR_STATUS) & TSR_STATUS_RV_SLEEP), "woken at MTIMECMP (%llu)",
          (unsigned long long)(t.woke - t0));
    CHECK(t.calls <= 4, "no busy stepping while asleep (%llu calls)", (unsigned long long)t.calls);
    tsr_sim_destroy(s);
}

// PERF RV_WFI (EX §6 X + 12): core cycles with STATUS.RV_SLEEP, a time counter (ACTIVE only).
static void test_wfi_perf(void) {
    const uint32_t C = 4;
    tsr_sim * s = mk_sim();                            // between launches: nothing counts
    timer_rt t;
    memset(&t, 0, sizeof(t));
    tsr_sim_set_rv_agent(s, timer_step, &t);
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    tsr_sim_advance(s, 100000);
    CHECK(t.woke != 0 && tsr_sim_perf(s, TSR_PERF_RV_WFI(C)) == 0, "RV_WFI 0 outside ACTIVE");
    tsr_sim_destroy(s);
    s = mk_sim();                                      // during a launch: exactly the sleep
    memset(&t, 0, sizeof(t));
    tsr_sim_set_rv_agent(s, timer_step, &t);
    uint32_t d[8] = { 1, 0, 0, 0, 0, 0, 0, 0 };
    wr(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    lq_push(s, d);
    CHECK(cww(s, TSR_CW_LQ_POP, 0) == TSR_CW_OK, "pop");
    const uint64_t t0 = tsr_sim_now(s);
    tsr_sim_advance(s, 100000);
    CHECK(t.woke == t0 + 5000, "woken at MTIMECMP");
    CHECK(tsr_sim_perf(s, TSR_PERF_RV_WFI(C)) == 5000 && tsr_sim_perf(s, TSR_PERF_CYCLES) == tsr_sim_now(s) - t0,
          "RV_WFI %llu = the sleep, CYCLES %llu", (unsigned long long)tsr_sim_perf(s, TSR_PERF_RV_WFI(C)),
          (unsigned long long)tsr_sim_perf(s, TSR_PERF_CYCLES));
    tsr_sim_destroy(s);
}

int main(void) {
    test_identity();
    test_loader();
    test_lq_lcq();
    test_watermarks();
    test_log();
    test_perf();
    test_soft_reset();
    test_status_ctrl();
    test_rv_reset();
    test_launch_agent();
    test_agent_timer();
    test_wfi_perf();
    return test_finish("sim_csr");
}

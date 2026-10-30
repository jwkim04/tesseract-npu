// test_sim_queue.c — the queue model (tsr_q.c; EX §7-§8, CC §5.1, §7-§9) through the RISC-V side
// control window and the push / QCTRL test hooks: checker codes in EX §8.1 order for every tile
// op (with ERR_INFO / ERR_W0 / ERR_SEQ / ERR_PIDX records and "a poisoned op changes no state"),
// refused opcodes on SYNC, BAD_DEP rules and its place in the order, OPERAND through a stub VU
// handler, the four tagged extents and their limits, tickets and ISSUE_PIDX (+1 per push and per
// QCLEAR, not per QABORT / QHOLD / TRACE_MARK), staging / WMASK / REC, deps and DEP_FAILED
// propagation (and CC §9.3's order: unit in error before a failed dep), I1 for
// poisoned entries, credits and back-pressure (queue full, issue FIFO full, the push safety net),
// SYNC (NOP join, FENCE, QCLEAR), QABORT, holds, effects at completion with a serial-chain
// bit-exact comparison against tsr_sim_op, advance events, and a random stress run for the
// internal invariants (I1-I3) and determinism.
#include "test_sim_util.h"
#include "tsr_sim_int.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>

static const uint32_t ALL = 0xFF00u;   // ISSUE_PUSH: send all eight words

static uint32_t TK(uint32_t u, uint32_t s) { return (u << 28) | (s & 0x0FFFFFFFu); }
static uint32_t tku(uint32_t t) { return t >> 28; }
static uint32_t tks(uint32_t t) { return t & 0x0FFFFFFFu; }

static tsr_sim * mk_opts(uint32_t t_max, const uint8_t * depth, uint32_t caps) {
    tsr_config c = cfg_u200(t_max);
    c.caps = caps;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.q_depth = depth;
    tsr_sim * s = tsr_sim_create_opts(&c, &o);
    CHECK(s != NULL, "create");
    if (s == NULL) exit(1);
    return s;
}
static tsr_sim * mk_sim(void) { return mk_opts(1, NULL, 0); }
static uint32_t cwr(tsr_sim * s, uint32_t off) { return tsr_sim_cw_read(s, off); }
static int cww(tsr_sim * s, uint32_t off, uint32_t v) { return tsr_sim_cw_access(s, 1, off, &v); }
static uint32_t push(tsr_sim * s, tsr_insn i, uint32_t d0, uint32_t d1) { return tsr_sim_push(s, i.w, d0, d1, ALL); }
static void idle(tsr_sim * s) { CHECK(tsr_sim_run_until_idle(s, 100000000ull), "run until idle"); }

// ---------------------------------------------------------------------------
// Event recorder (listener)
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t kind, unit, ticket, pidx, code, ran, rec, dep[2];
    uint64_t time;
    uint32_t w[8];
    tsr_extents ext;
} ev_t;
typedef struct { ev_t * v; size_t n, cap; uint64_t hash; } evlog;

static void ev_cb(void * ctx, tsr_sim * s, const tsr_q_event * e) {
    (void)s;
    evlog * L = (evlog *)ctx;
    if (L->n == L->cap) {
        L->cap = L->cap ? 2 * L->cap : 256;
        L->v = (ev_t *)realloc(L->v, L->cap * sizeof(ev_t));
    }
    ev_t * r = &L->v[L->n++];
    memset(r, 0, sizeof(*r));
    r->kind = e->kind; r->unit = e->unit; r->ticket = e->ticket; r->pidx = e->pidx; r->code = e->code;
    r->ran = e->ran; r->rec = e->rec; r->dep[0] = e->dep[0]; r->dep[1] = e->dep[1]; r->time = e->time;
    if (e->w) memcpy(r->w, e->w, 32);
    if (e->ext) r->ext = *e->ext;
    const uint64_t f[6] = { e->kind, e->unit, e->ticket, e->code, e->ran, e->time };
    for (int i = 0; i < 6; ++i) { L->hash ^= f[i]; L->hash *= 1099511628211ull; }
}
static const ev_t * ev_find(const evlog * L, uint32_t kind, uint32_t ticket) {
    for (size_t i = L->n; i-- > 0;)
        if (L->v[i].kind == kind && L->v[i].ticket == ticket) return &L->v[i];
    return NULL;
}
static uint64_t t_of(const evlog * L, uint32_t kind, uint32_t ticket) {
    const ev_t * e = ev_find(L, kind, ticket);
    CHECK(e != NULL, "event %u of ticket 0x%08x", kind, ticket);
    return e ? e->time : 0;
}

// QCLEAR from the RISC-V side, joined.
static void qclear_join(tsr_sim * s) {
    const uint32_t t = tsr_sim_qctrl(s, TSR_QCTRL_QCLEAR);
    CHECK(tku(t) == TSR_UNIT_SYNC, "QCLEAR ticket on SYNC");
    idle(s);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) CHECK(cwr(s, TSR_CW_ERR_CODE(u)) == 0, "unit %u cleared", u);
}

// ---------------------------------------------------------------------------
// 1. Checker codes in order, through the queues
// ---------------------------------------------------------------------------
static uint64_t mem_hash(tsr_sim * s) {
    return fnv64(tsr_sim_spm(s), tsr_sim_config(s)->spm_bytes) ^ (fnv64(tsr_sim_dram(s, 1), 1u << 16) * 3u);
}

static void expect_q(tsr_sim * s, tsr_insn in, uint32_t d0, uint32_t d1, uint32_t unit, uint32_t code, uint32_t info,
                     const char * what) {
    CHECK(cwr(s, TSR_CW_ERR_CODE(unit)) == 0, "%s: unit %u clean before", what, unit);
    const uint32_t pidx = cwr(s, TSR_CW_ISSUE_PIDX);
    const uint64_t h0 = mem_hash(s);
    const uint32_t t = push(s, in, d0, d1);
    CHECK(tku(t) == unit, "%s: ticket unit %u, expected %u", what, tku(t), unit);
    idle(s);
    CHECK(cwr(s, TSR_CW_DONE_SEQ(unit)) == ((tks(t) + 1) & 0x0FFFFFFFu), "%s: completed", what);
    CHECK(cwr(s, TSR_CW_ERR_CODE(unit)) == code, "%s: ERR_CODE %u, expected %u", what, cwr(s, TSR_CW_ERR_CODE(unit)), code);
    if (code == TSR_ERR_NONE) return;
    CHECK(cwr(s, TSR_CW_ERR_INFO(unit)) == info, "%s: ERR_INFO 0x%x, expected 0x%x", what, cwr(s, TSR_CW_ERR_INFO(unit)), info);
    CHECK(cwr(s, TSR_CW_ERR_W0(unit)) == in.w[0] && cwr(s, TSR_CW_ERR_SEQ(unit)) == tks(t) && cwr(s, TSR_CW_ERR_PIDX(unit)) == pidx,
          "%s: ERR_W0 / ERR_SEQ / ERR_PIDX", what);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_ERR_CODE(unit)) == code && tsr_sim_csr_read(s, TSR_CSR_Q_ERR_PIDX(unit)) == pidx,
          "%s: host Q_ERR_*", what);
    CHECK(mem_hash(s) == h0, "%s: a poisoned op changes no state", what);
    qclear_join(s);
}

static void test_checker_order(void) {
    tsr_sim * s = mk_sim();
    const uint32_t SL = 65536;                           // SPM lines
    const uint32_t DL = (uint32_t)tsr_sim_dram_limit_beats(s);
    const uint32_t BIG = 0x7FFFFFF0u;
    tsr_insn i;
    // LD / ST
    i = tsr_enc_ld(1, BIG, BIG, 4); i.w[0] |= 1u << 16;  expect_q(s, i, 0xF0000000u, 0xF0000000u, 0, TSR_ERR_ILLEGAL_OP, 0, "LD x16 + ranges");
    i = tsr_enc_ld(1, 0, 0, 4); i.w[0] |= 1u << 12;      expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_ILLEGAL_OP, 0, "LD flags");
    i = tsr_enc_ld(4, 0, 0, 4);                          expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_ILLEGAL_OP, 0, "LD ch >= NUM_CH");
    i = tsr_enc_ld(1, BIG, 0, 4); i.w[4] = 1;            expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_ILLEGAL_OP, 0, "LD w4 + range");
    i = tsr_enc_ld(1, SL - 3, DL - 3, 4);                expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_SPM_RANGE, SL - 3, "LD SPM before DRAM");
    i = tsr_enc_ld(1, 7, DL - 3, 4);                     expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_DRAM_RANGE, DL - 3, "LD DRAM");
    i = tsr_enc_ld(1, SL - 4, DL - 4, 4);                expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_NONE, 0, "LD at the limits");
    i = tsr_enc_st(1, SL - 1, 0, 2);                     expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_SPM_RANGE, SL - 1, "ST SPM");
    i = tsr_enc_st(1, 0, DL, 1);                         expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_DRAM_RANGE, DL, "ST DRAM");
    i = tsr_enc_st(1, 0, 0, 1); i.w[7] = 9;              expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 0, TSR_ERR_ILLEGAL_OP, 0, "ST w7");
    // QACT
    i = tsr_enc_qact(2, BIG, 0, 33, 0, 0);               expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 1, TSR_ERR_ILLEGAL_OP, 0, "QACT kind 2");
    i = tsr_enc_qact(0, 0, 0, 64, 1, 0); i.w[6] = 1;     expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 1, TSR_ERR_ILLEGAL_OP, 0, "QACT w6");
    i = tsr_enc_qact(0, BIG, 0, 33, 0, 0);               expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 1, TSR_ERR_K_RANGE, 0, "QACT K before T");
    i = tsr_enc_qact(1, 0, 0, 32768 + 256, 1, 0);        expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 1, TSR_ERR_K_RANGE, 0, "QACT K > K_MAX");
    i = tsr_enc_qact(0, BIG, BIG, 64, 2, 0);             expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 1, TSR_ERR_T_RANGE, 0, "QACT T before SPM");
    i = tsr_enc_qact(0, BIG, BIG, 64, 1, 0);             expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 1, TSR_ERR_SPM_RANGE, BIG, "QACT dst before src");
    i = tsr_enc_qact(0, 0, SL - 3, 64, 1, 0);            expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 1, TSR_ERR_SPM_RANGE, SL - 3, "QACT src");
    // MVM
    i = tsr_enc_mvm(3, 2, 0, 0, 0, 1, 256, 1, 0);         expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_ILLEGAL_OP, 0, "MVM flags bit1 + fmt");
    i = tsr_enc_mvm(0, 0, 0, 0, 0, 1, 256, 1, 0); i.w[5] |= 1u << 20; expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_ILLEGAL_OP, 0, "MVM w5[20]");
    i = tsr_enc_mvm(0, 0, 0, 0, 0, 1, 256, 1, 0); i.w[7] = 1; expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_ILLEGAL_OP, 0, "MVM w7");
    i = tsr_enc_mvm(2, 0, 0, 0, 0, 1, 33, 9, 0);          expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_BAD_FMT, 0, "MVM fmt before K");
    i = tsr_enc_mvm(0, 0, 0, 0, 0, 1, 33, 9, 0);          expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_K_RANGE, 0, "MVM K before T");
    i = tsr_enc_mvm(1, 0, 0, 0, 0, 1, 128, 1, 0);         expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_K_RANGE, 0, "MVM Q4_K K 128");
    i = tsr_enc_mvm(0, 0, BIG, BIG, BIG, 1, 256, 0, 0);   expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_T_RANGE, 0, "MVM T 0 before ranges");
    i = tsr_enc_mvm(0, 0, BIG, BIG, BIG, 1, 256, 1, 0);   expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_SPM_RANGE, BIG, "MVM dst first");
    i = tsr_enc_mvm(0, 0, 0, BIG, BIG, 1, 256, 1, 0);     expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_SPM_RANGE, BIG, "MVM x second");
    i = tsr_enc_mvm(0, 0, 0, 100, DL - 1, 64, 256, 1, 0); expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 2, TSR_ERR_DRAM_RANGE, DL - 1, "MVM W");
    // NOP / FENCE use no words
    i = tsr_enc_nop(); i.w[1] = 1;                       expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 5, TSR_ERR_ILLEGAL_OP, 0, "NOP w1");
    i = tsr_enc_nop(); i.w[0] |= 1u << 8;                expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 5, TSR_ERR_ILLEGAL_OP, 0, "NOP sub");
    i = tsr_enc_fence(); i.w[0] |= 1u << 31;             expect_q(s, i, TSR_TICKET_NONE, TSR_TICKET_NONE, 5, TSR_ERR_ILLEGAL_OP, 0, "FENCE x16");
    expect_q(s, tsr_enc_fence(), TSR_TICKET_NONE, TSR_TICKET_NONE, 5, TSR_ERR_NONE, 0, "FENCE");
    // Reserved, retired and absent opcodes: accepted, SYNC ticket, ILLEGAL_OP (EX §7.2)
    const uint32_t refused[] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0F, 0x12, 0x13, 0x14, 0x15, 0x19,
                                 0x21, 0x23, 0x30, 0x31, 0x32, 0x33, 0x34, 0x38, 0x40, 0x48, 0x70, 0x7E, 0x80, 0xFF };
    for (size_t k = 0; k < sizeof(refused) / 4; ++k) {
        char what[32];
        snprintf(what, sizeof(what), "opcode 0x%02x refused", refused[k]);
        expect_q(s, insn_raw(refused[k], 0, 0, 0, 0, 0, 0, 0), TSR_TICKET_NONE, TSR_TICKET_NONE, 5, TSR_ERR_ILLEGAL_OP, 0, what);
    }
    // BAD_DEP after ILLEGAL_OP, before everything else
    i = tsr_enc_ld(1, 0, 0, 1); i.w[0] |= 1u << 20;      expect_q(s, i, TK(7, 0), TSR_TICKET_NONE, 0, TSR_ERR_ILLEGAL_OP, 0, "ILLEGAL_OP before BAD_DEP");
    i = tsr_enc_mvm(2, 0, 0, 0, 0, 1, 256, 1, 0);         expect_q(s, i, TSR_TICKET_NONE, TK(9, 0), 2, TSR_ERR_BAD_DEP, 0, "BAD_DEP before BAD_FMT");
    i = tsr_enc_qact(0, 0, 0, 33, 1, 0);                 expect_q(s, i, TK(2, 999), TSR_TICKET_NONE, 1, TSR_ERR_BAD_DEP, 0, "BAD_DEP before K_RANGE");
    i = tsr_enc_ld(1, BIG, 0, 1);                        expect_q(s, i, TK(4, 0), TSR_TICKET_NONE, 0, TSR_ERR_BAD_DEP, 0, "BAD_DEP before SPM_RANGE");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 1b. OPERAND and a VU checker class through stub handlers (the real VU is P1.4's)
// ---------------------------------------------------------------------------
static uint32_t stub_check(tsr_sim * s, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (w[0] >> 16) return TSR_ERR_ILLEGAL_OP;
    if (w[1] == 0) { *info = w[2]; return TSR_ERR_OPERAND; }
    if (w[3] >= tsr_sim_spm_lines(s)) { *info = w[3]; return TSR_ERR_SPM_RANGE; }
    tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_WRITE, 0, w[3], (uint64_t)w[3] + 1);
    tsr_ext_add(ext, TSR_EXT_SREG, TSR_EXT_WRITE, 0, 1, 2);
    return TSR_ERR_NONE;
}
static void stub_exec(tsr_sim * s, const uint32_t w[8]) {
    tsr_sim_spm(s)[(size_t)w[3] * 64] = (uint8_t)w[1];
    tsr_sim_sreg(s)[1] = 2.0f;
}
static uint64_t stub_cycles(const tsr_sim * s, const uint32_t w[8]) { (void)s; return 10u + w[1]; }

static void test_operand_stub(void) {
    const tsr_op_handler h = { TSR_UNIT_VU, stub_check, stub_exec, stub_cycles };
    const uint8_t ops[4] = { TSR_OP_VOP, TSR_OP_RED, TSR_OP_ROPE, TSR_OP_SOP };
    for (int k = 0; k < 4; ++k) tsr_sim_register_op(ops[k], &h);
    CHECK(tsr_sim_caps_implemented() & TSR_CAPS_VU, "CAPS_VU implemented with the four VU handlers");
    tsr_sim * s = mk_opts(1, NULL, TSR_CAPS_VU);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_CAPS) == TSR_CAPS_VU && TSR_GET(tsr_sim_csr_read(s, TSR_CSR_CFG3), TSR_CFG3_VU_FRONT) == 16,
          "CAPS / CFG3 with the VU");
    evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_set_listener(s, ev_cb, &L);
    expect_q(s, insn_raw(TSR_OP_VOP, 0, 0x1234, 5, 0, 0, 0, 0), TSR_TICKET_NONE, TSR_TICKET_NONE, 3, TSR_ERR_OPERAND, 0x1234, "OPERAND");
    expect_q(s, insn_raw(TSR_OP_RED, 0, 77, 5, 0, 0, 0, 0), TK(3, 500), TSR_TICKET_NONE, 3, TSR_ERR_BAD_DEP, 0, "BAD_DEP before OPERAND");
    expect_q(s, insn_raw(TSR_OP_SOP | (1u << 16), 0, 77, 5, 0, 0, 0, 0), TK(3, 500), TSR_TICKET_NONE, 3, TSR_ERR_ILLEGAL_OP, 0,
             "ILLEGAL_OP before BAD_DEP");
    expect_q(s, insn_raw(TSR_OP_ROPE, 1, 0, 70000, 0, 0, 0, 0), TSR_TICKET_NONE, TSR_TICKET_NONE, 3, TSR_ERR_SPM_RANGE, 70000, "stub range");
    // A passing stub op: effects at completion, SREG visible to the host, cycles from the handler.
    const uint32_t t = push(s, insn_raw(TSR_OP_VOP, 0x5A, 0, 40, 0, 0, 0, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    tsr_sim_advance(s, 2);                           // dispatched at +2 and started
    CHECK(tsr_sim_spm(s)[40 * 64] != 0x5A, "no effect before completion");
    idle(s);
    CHECK(tsr_sim_spm(s)[40 * 64] == 0x5A && tsr_sim_csr_read(s, TSR_CSR_SREG(1)) == f2u(2.0f), "effects at completion");
    CHECK(t_of(&L, TSR_QEV_DONE, t) - t_of(&L, TSR_QEV_START, t) == 10 + 0x5A, "duration = cycles()");
    const ev_t * e = ev_find(&L, TSR_QEV_ENQ, t);
    CHECK(e && e->ext.n == 2 && e->ext.e[1].kind == TSR_EXT_SREG && e->ext.e[1].lo == 1 && e->ext.e[1].hi == 2, "SREG extent");
    // VMASK (no handler, no CAPS) stays refused even with CAPS_VU
    expect_q(s, insn_raw(TSR_OP_VMASK, 0, 0, 0, 0, 0, 0, 0), TSR_TICKET_NONE, TSR_TICKET_NONE, 5, TSR_ERR_ILLEGAL_OP, 0, "VMASK");
    tsr_sim_destroy(s);
    free(L.v);
    for (int k = 0; k < 4; ++k) tsr_sim_register_op(ops[k], NULL);
    CHECK(!(tsr_sim_caps_implemented() & TSR_CAPS_VU), "handlers removed");
}

// ---------------------------------------------------------------------------
// 2. Extents
// ---------------------------------------------------------------------------
static void ext_is(const ev_t * e, uint32_t k, uint8_t kind, uint8_t wr, uint8_t ch, uint64_t lo, uint64_t hi, const char * what) {
    CHECK(e && k < e->ext.n && e->ext.e[k].kind == kind && e->ext.e[k].write == wr && e->ext.e[k].ch == ch &&
              e->ext.e[k].lo == lo && e->ext.e[k].hi == hi,
          "%s: extent %u {%u %u %u [%llu, %llu)}", what, k, e ? e->ext.e[k].kind : 0, e ? e->ext.e[k].write : 0,
          e ? e->ext.e[k].ch : 0, e ? (unsigned long long)e->ext.e[k].lo : 0, e ? (unsigned long long)e->ext.e[k].hi : 0);
}

static void test_extents(void) {
    tsr_sim * s = mk_opts(4, NULL, 0);
    evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_set_listener(s, ev_cb, &L);
    uint32_t t;
    const ev_t * e;
    t = push(s, tsr_enc_ld(2, 10, 100, 5), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s); e = ev_find(&L, TSR_QEV_ENQ, t);
    CHECK(e && e->ext.n == 2, "LD 2 extents");
    ext_is(e, 0, TSR_EXT_SPM, TSR_EXT_WRITE, 0, 10, 15, "LD");
    ext_is(e, 1, TSR_EXT_DRAM, TSR_EXT_READ, 2, 100, 105, "LD");
    t = push(s, tsr_enc_st(1, 20, 7, 3), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s); e = ev_find(&L, TSR_QEV_ENQ, t);
    ext_is(e, 0, TSR_EXT_SPM, TSR_EXT_READ, 0, 20, 23, "ST");
    ext_is(e, 1, TSR_EXT_DRAM, TSR_EXT_WRITE, 1, 7, 10, "ST");
    t = push(s, tsr_enc_ld(0, 10, 100, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s); e = ev_find(&L, TSR_QEV_ENQ, t);
    CHECK(e && e->ext.n == 0, "LD of 0 lines: no extents");
    const uint32_t qa64 = tsr_qa_lines(TSR_QA_A32, 64);
    t = push(s, tsr_enc_qact(TSR_QA_A32, 100, 200, 64, 3, 10), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s); e = ev_find(&L, TSR_QEV_ENQ, t);
    CHECK(e && e->ext.n == 2, "QACT 2 extents");
    ext_is(e, 0, TSR_EXT_SPM, TSR_EXT_WRITE, 0, 100, 100 + 3 * qa64, "QACT dst");
    ext_is(e, 1, TSR_EXT_SPM, TSR_EXT_READ, 0, 200, 200 + 2 * 10 + 4, "QACT strided src");
    const uint32_t qa256 = tsr_qa_lines(TSR_QA_A32, 256), rb = tsr_row_beats(TSR_FMT_Q8_0, 256);
    t = push(s, tsr_enc_mvm(TSR_FMT_Q8_0, TSR_MVM_FLAG_ACC, 300, 400, 50, 100, 256, 2, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s); e = ev_find(&L, TSR_QEV_ENQ, t);
    CHECK(e && e->ext.n == 4, "MVM ACC: 4 extents");
    ext_is(e, 0, TSR_EXT_SPM, TSR_EXT_WRITE, 0, 300, 300 + 7 + 7, "MVM dst (T = 2, dense stride 7)");
    ext_is(e, 1, TSR_EXT_SPM, TSR_EXT_READ, 0, 400, 400 + 2 * qa256, "MVM x");
    ext_is(e, 2, TSR_EXT_DRAM, TSR_EXT_READ, TSR_EXT_CH_ALL, 50, 50 + 25 * rb, "MVM weights on every channel");
    ext_is(e, 3, TSR_EXT_SPM, TSR_EXT_READ, 0, 300, 314, "MVM ACC reads dst");
    t = push(s, tsr_enc_nop(), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s); e = ev_find(&L, TSR_QEV_ENQ, t);
    CHECK(e && e->ext.n == 0 && e->code == 0, "NOP: no extents");
    t = push(s, tsr_enc_ld(1, 0x7FFFFFFF, 0, 2), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s); e = ev_find(&L, TSR_QEV_ENQ, t);
    CHECK(e && e->ext.n == 0 && e->code == TSR_ERR_SPM_RANGE, "poisoned: code in ENQ, no extents");
    tsr_sim_destroy(s);
    free(L.v);
}

// Extent-list limits (tsr_sim_int.h §1): tsr_ext_add records an overflow instead of dropping it
// silently, and the dispatcher aborts on a list that breaks a limit (checked in a child process
// with a stub LD checker; the child's stderr is discarded).
static int g_bad_ext;
static uint32_t bad_ext_check(tsr_sim * s, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    (void)s; (void)w; (void)info;
    switch (g_bad_ext) {
    case 0:                                            // 4 operands, 2 of them read-modify-write: fine
        for (uint32_t k = 0; k < 4; ++k) tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_WRITE, 0, 10u * k, 10u * k + 2u);
        tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_READ, 0, 0, 2);
        tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_READ, 0, 10, 12);
        tsr_ext_add(ext, TSR_EXT_SREG, TSR_EXT_READ, 0, 1, 2);
        break;
    case 1:                                            // 8 entries: one does not fit
        for (uint32_t k = 0; k < 8; ++k) tsr_ext_add(ext, TSR_EXT_SREG, TSR_EXT_READ, 0, k, k + 1u);
        break;
    case 2:                                            // two DRAM extents
        tsr_ext_add(ext, TSR_EXT_DRAM, TSR_EXT_READ, 0, 0, 1);
        tsr_ext_add(ext, TSR_EXT_DRAM, TSR_EXT_WRITE, 1, 0, 1);
        break;
    case 3:                                            // five SPM operands
        for (uint32_t k = 0; k < 5; ++k) tsr_ext_add(ext, TSR_EXT_SPM, TSR_EXT_READ, 0, 10u * k, 10u * k + 1u);
        break;
    default:                                           // four SREG entries
        for (uint32_t k = 0; k < 4; ++k) tsr_ext_add(ext, TSR_EXT_SREG, TSR_EXT_WRITE, 0, k, k + 1u);
        break;
    }
    return TSR_ERR_NONE;
}
static void noop_exec(tsr_sim * s, const uint32_t w[8]) { (void)s; (void)w; }
static uint64_t one_cycle(const tsr_sim * s, const uint32_t w[8]) { (void)s; (void)w; return 1; }

// Pushes one LD with the stub checker variant v in a child; returns its wait status.
static int run_bad_ext(int v) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        const int fd = open("/dev/null", O_WRONLY);
        if (fd >= 0) dup2(fd, 2);
        g_bad_ext = v;
        const tsr_op_handler h = { TSR_UNIT_DMA, bad_ext_check, noop_exec, one_cycle };
        tsr_sim_register_op(TSR_OP_LD, &h);
        tsr_sim * s = mk_sim();
        push(s, tsr_enc_ld(0, 0, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE);
        const int ok = tsr_sim_run_until_idle(s, 1000000);
        tsr_sim_destroy(s);
        _exit(ok ? 0 : 3);
    }
    int st = -1;
    if (pid < 0 || waitpid(pid, &st, 0) != pid) return -1;
    return st;
}

static void test_extent_limits(void) {
    tsr_extents x;
    memset(&x, 0, sizeof(x));
    for (uint32_t k = 0; k < 9; ++k) tsr_ext_add(&x, TSR_EXT_SREG, TSR_EXT_READ, 0, k, k + 1u);
    CHECK(x.n == TSR_EXT_MAX + TSR_EXT_SREG_MAX && x.overflow == 2, "tsr_ext_add: n %u, overflow %u", x.n, x.overflow);
    tsr_ext_add(&x, TSR_EXT_SPM, TSR_EXT_READ, 0, 5, 5);
    CHECK(x.overflow == 2, "an empty extent is not an entry");
    int st = run_bad_ext(0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "4 operands with 2 read-modify-write pairs: accepted (status 0x%x)", st);
    for (int v = 1; v <= 4; ++v) {
        st = run_bad_ext(v);
        CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT, "bad extent list %d: the dispatcher aborts (status 0x%x)", v, st);
    }
}

// ---------------------------------------------------------------------------
// 3. Tickets, ISSUE_PIDX, ISSUED_SEQ, staging, WMASK, REC, TRACE_MARK
// ---------------------------------------------------------------------------
static void test_tickets_pidx(void) {
    tsr_sim * s = mk_sim();
    evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_set_listener(s, ev_cb, &L);
    CHECK(cwr(s, TSR_CW_ISSUE_PIDX) == 0 && cwr(s, TSR_CW_ISSUE_TICKET) == TSR_TICKET_NONE, "reset");
    CHECK(cwr(s, TSR_CW_ISSUE_DEP0) == TSR_TICKET_NONE && cwr(s, TSR_CW_WAIT_TICKET) == TSR_TICKET_NONE, "dep / wait reset values");
    CHECK(push(s, tsr_enc_ld(0, 0, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE) == TK(0, 0), "{0, 0}");
    CHECK(push(s, tsr_enc_qact(0, 10, 20, 32, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE) == TK(1, 0), "{1, 0}");
    CHECK(push(s, tsr_enc_ld(0, 0, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE) == TK(0, 1), "{0, 1}");
    CHECK(push(s, insn_raw(0x01, 0, 0, 0, 0, 0, 0, 0), TSR_TICKET_NONE, TSR_TICKET_NONE) == TK(5, 0), "END on SYNC {5, 0}");
    CHECK(cwr(s, TSR_CW_ISSUE_PIDX) == 4 && cwr(s, TSR_CW_ISSUE_TICKET) == TK(5, 0), "PIDX 4");
    CHECK(cwr(s, TSR_CW_ISSUED_SEQ(0)) == 2 && cwr(s, TSR_CW_ISSUED_SEQ(1)) == 1 && cwr(s, TSR_CW_ISSUED_SEQ(5)) == 1, "ISSUED_SEQ");
    tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_SET);
    tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_CLR);
    tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, 0x30000005u);
    CHECK(tsr_sim_qctrl(s, TSR_QCTRL_QABORT) == TSR_TICKET_NONE, "QABORT has no ticket");
    tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_SET | TSR_QCTRL_QHOLD_CLR);
    CHECK(cwr(s, TSR_CW_ISSUE_PIDX) == 4 && cwr(s, TSR_CW_ISSUE_TICKET) == TK(5, 0), "QHOLD, TRACE_MARK, QABORT: no PIDX step");
    const uint32_t qc = tsr_sim_qctrl(s, TSR_QCTRL_QCLEAR);
    CHECK(qc == TK(5, 1) && cwr(s, TSR_CW_ISSUE_TICKET) == qc && cwr(s, TSR_CW_ISSUE_PIDX) == 5, "QCLEAR: SYNC ticket, PIDX + 1");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_FIFO) == 11 && tsr_sim_csr_read(s, TSR_CSR_Q_PIDX) == 0, "all 11 entries in the issue FIFO");
    // An op that depends on the QCLEAR runs normally.
    const uint32_t t = push(s, tsr_enc_ld(0, 5, 0, 1), qc, TSR_TICKET_NONE);
    CHECK(t == TK(0, 2) && cwr(s, TSR_CW_ISSUE_PIDX) == 6, "PIDX 6");
    idle(s);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_PIDX) == 6 && tsr_sim_csr_read(s, TSR_CSR_Q_FIFO) == 0, "Q_PIDX = ISSUE_PIDX when drained");
    CHECK(cwr(s, TSR_CW_ERR_CODE(0)) == 0 && cwr(s, TSR_CW_DONE_SEQ(0)) == 3, "post-QCLEAR op ran clean");
    const ev_t * d = ev_find(&L, TSR_QEV_DONE, t);
    CHECK(d && d->ran && d->code == 0, "ran");
    // The marker reached the dispatcher in push order, between pidx 4 and 5.
    const ev_t * m = NULL;
    for (size_t k = 0; k < L.n; ++k) if (L.v[k].kind == TSR_QEV_MARK) m = &L.v[k];
    CHECK(m && m->w[1] == 0x30000005u && m->pidx == 4 && m->ticket == TSR_TICKET_NONE, "TRACE_MARK event");
    // The aborted entries: END (SYNC, ILLEGAL_OP) completed before the abort or ABORTED; never ran.
    CHECK(ev_find(&L, TSR_QEV_DONE, TK(5, 0)) && !ev_find(&L, TSR_QEV_DONE, TK(5, 0))->ran, "END did not run");
    // Staging is kept, deps reset to none by every push, WMASK and REC.
    cww(s, TSR_CW_ISSUE_DEP0, t);                      // tickets from after the QCLEAR (earlier ones are BAD_DEP now)
    cww(s, TSR_CW_ISSUE_DEP1, qc);
    const tsr_insn ld = tsr_enc_ld(1, 30, 40, 2);
    for (int k = 0; k < 8; ++k) cww(s, TSR_CW_ISSUE_W(k), ld.w[k]);
    CHECK(cww(s, TSR_CW_ISSUE_PUSH, ALL | TSR_ISSUE_PUSH_REC) == TSR_CW_OK, "push");
    const uint32_t t1 = cwr(s, TSR_CW_ISSUE_TICKET);
    CHECK(cwr(s, TSR_CW_ISSUE_DEP0) == TSR_TICKET_NONE && cwr(s, TSR_CW_ISSUE_DEP1) == TSR_TICKET_NONE, "deps reset by the push");
    CHECK(cwr(s, TSR_CW_ISSUE_W(1)) == 30 && cwr(s, TSR_CW_ISSUE_W(3)) == 2, "staging kept");
    CHECK(cww(s, TSR_CW_ISSUE_PUSH, 0x0100u) == TSR_CW_OK, "push with WMASK = w0 only");
    const uint32_t t2 = cwr(s, TSR_CW_ISSUE_TICKET);
    CHECK(cww(s, TSR_CW_ISSUE_PUSH, 0x0B00u) == TSR_CW_OK, "push with WMASK = w0, w1, w3");
    const uint32_t t3 = cwr(s, TSR_CW_ISSUE_TICKET);
    idle(s);
    const ev_t * e1 = ev_find(&L, TSR_QEV_ENQ, t1), * e2 = ev_find(&L, TSR_QEV_ENQ, t2), * e3 = ev_find(&L, TSR_QEV_ENQ, t3);
    CHECK(e1 && e1->rec == 1 && e1->dep[0] == t && e1->dep[1] == qc && e1->w[2] == 40 && e1->code == 0, "REC, deps, words");
    CHECK(e2 && e2->rec == 0 && e2->w[0] == ld.w[0] && e2->w[1] == 0 && e2->w[3] == 0 && e2->code == 0, "WMASK: w1-w7 sent as 0");
    CHECK(e3 && e3->w[1] == 30 && e3->w[2] == 0 && e3->w[3] == 2, "WMASK 0x0B");
    tsr_sim_destroy(s);
    free(L.v);
}

// ---------------------------------------------------------------------------
// 4. Dependencies, DEP_FAILED, BAD_DEP rules, I1 for poisoned entries
// ---------------------------------------------------------------------------
static void test_deps(void) {
    tsr_sim * s = mk_sim();
    evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_set_listener(s, ev_cb, &L);
    const uint32_t A = push(s, tsr_enc_ld(0, 0, 0, 100), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t B = push(s, tsr_enc_qact(0, 200, 0, 64, 1, 0), A, TSR_TICKET_NONE);
    const uint32_t D = push(s, tsr_enc_mvm(0, 0, 300, 400, 0, 16, 256, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t J = push(s, tsr_enc_nop(), A, D);
    const uint32_t P = push(s, tsr_enc_qact(0, 0, 0, 33, 1, 0), J, TSR_TICKET_NONE);   // poisoned (K_RANGE), waits for J (I1)
    idle(s);
    CHECK(t_of(&L, TSR_QEV_START, B) >= t_of(&L, TSR_QEV_DONE, A), "a dependent starts after its producer");
    CHECK(t_of(&L, TSR_QEV_START, D) < t_of(&L, TSR_QEV_DONE, A), "independent units run concurrently");
    const uint64_t jd = t_of(&L, TSR_QEV_DONE, J);
    CHECK(jd > t_of(&L, TSR_QEV_DONE, A) && jd > t_of(&L, TSR_QEV_DONE, D), "NOP join after both deps");
    CHECK(t_of(&L, TSR_QEV_DONE, P) > jd && ev_find(&L, TSR_QEV_DONE, P)->code == TSR_ERR_K_RANGE, "I1: a poisoned entry completes after its deps");
    qclear_join(s);

    // BAD_DEP rules (CC §8.3)
    const uint32_t LD = 0;
    const tsr_insn ok = tsr_enc_ld(1, 0, 0, 1);
    const uint32_t bad[] = { TK(6, 0), TK(13, 0), TK(14, 0), TK(4, 0),                 // reserved, marker, absent unit
                             TK(1, cwr(s, TSR_CW_ISSUED_SEQ(1)) + 5),                  // QACT ticket not pushed yet
                             TK(0, cwr(s, TSR_CW_ISSUED_SEQ(0))) };                    // the op's own ticket
    for (size_t k = 0; k < sizeof(bad) / 4; ++k) {
        char what[48];
        snprintf(what, sizeof(what), "BAD_DEP 0x%08x in DEP%zu", bad[k], k & 1);
        if (k == 5) {
            expect_q(s, ok, TSR_TICKET_NONE, TK(0, cwr(s, TSR_CW_ISSUED_SEQ(0))), LD, TSR_ERR_BAD_DEP, 0, what);
        } else {
            expect_q(s, ok, (k & 1) ? TSR_TICKET_NONE : bad[k], (k & 1) ? bad[k] : TSR_TICKET_NONE, LD, TSR_ERR_BAD_DEP, 0, what);
        }
    }
    // (each expect_q above ended with a QCLEAR: every earlier ticket is below the clear point now)
    const uint32_t prev = push(s, ok, TSR_TICKET_NONE, TSR_TICKET_NONE);
    expect_q(s, ok, prev, 0xF0000123u, LD, TSR_ERR_NONE, 0, "dep on an earlier op of the own unit; unit 15 = none");
    // Below the clear point: a ticket from before the last QCLEAR
    const uint32_t old_q = push(s, tsr_enc_qact(0, 500, 600, 32, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s);
    tsr_sim_qctrl(s, TSR_QCTRL_QCLEAR);
    idle(s);
    expect_q(s, ok, old_q, TSR_TICKET_NONE, LD, TSR_ERR_BAD_DEP, 0, "dep below the clear point");
    const uint32_t qc = tsr_sim_qctrl(s, TSR_QCTRL_QCLEAR);
    idle(s);
    expect_q(s, ok, qc, TSR_TICKET_NONE, LD, TSR_ERR_NONE, 0, "dep on the last QCLEAR itself");

    // DEP_FAILED and propagation; first error = smallest ERR_PIDX.
    const uint32_t L0 = push(s, tsr_enc_ld(0, 100, 0, 2), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t pP = cwr(s, TSR_CW_ISSUE_PIDX);
    const uint32_t Pz = push(s, tsr_enc_ld(0, 0x7FFFFFFF, 0, 2), TSR_TICKET_NONE, TSR_TICKET_NONE);   // SPM_RANGE
    const uint32_t Q = push(s, tsr_enc_qact(0, 500, 600, 32, 1, 0), Pz, TSR_TICKET_NONE);
    const uint32_t Q2 = push(s, tsr_enc_qact(0, 700, 600, 32, 1, 0), L0, TSR_TICKET_NONE);   // unit already in error
    const uint32_t M = push(s, tsr_enc_mvm(0, 0, 800, 900, 0, 4, 256, 1, 0), Q, TSR_TICKET_NONE);
    const uint32_t L2 = push(s, tsr_enc_ld(0, 100, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t S0 = push(s, tsr_enc_st(1, 100, 0, 1), L0, TSR_TICKET_NONE);   // DMA: after Pz in queue order -> ABORTED
    // Unit in error and a failed dep at once: CC §9.3 decides "unit in error" first -> ABORTED.
    const uint32_t S1 = push(s, tsr_enc_st(1, 100, 0, 1), Pz, TSR_TICKET_NONE);
    const uint32_t Q4 = push(s, tsr_enc_qact(0, 900, 600, 32, 1, 0), Pz, TSR_TICKET_NONE);
    idle(s);
    CHECK(ev_find(&L, TSR_QEV_DONE, L0)->ran && ev_find(&L, TSR_QEV_DONE, L0)->code == 0, "L0 ran");
    CHECK(ev_find(&L, TSR_QEV_DONE, Pz)->code == TSR_ERR_SPM_RANGE, "Pz poisoned");
    CHECK(ev_find(&L, TSR_QEV_DONE, Q)->code == TSR_ERR_DEP_FAILED && cwr(s, TSR_CW_ERR_CODE(1)) == TSR_ERR_DEP_FAILED, "DEP_FAILED");
    CHECK(ev_find(&L, TSR_QEV_DONE, Q2)->code == TSR_ERR_ABORTED, "later QACT entry ABORTED");
    CHECK(ev_find(&L, TSR_QEV_DONE, M)->code == TSR_ERR_DEP_FAILED && cwr(s, TSR_CW_ERR_CODE(2)) == TSR_ERR_DEP_FAILED, "propagates");
    CHECK(ev_find(&L, TSR_QEV_DONE, L2)->code == TSR_ERR_ABORTED && ev_find(&L, TSR_QEV_DONE, S0)->code == TSR_ERR_ABORTED,
          "the errored DMA aborts later entries");
    CHECK(ev_find(&L, TSR_QEV_DONE, S1)->code == TSR_ERR_ABORTED && ev_find(&L, TSR_QEV_DONE, Q4)->code == TSR_ERR_ABORTED,
          "unit in error before a failed dep (CC §9.3): ABORTED, not DEP_FAILED");
    CHECK(cwr(s, TSR_CW_ERR_PIDX(0)) == pP && cwr(s, TSR_CW_ERR_PIDX(1)) > pP && cwr(s, TSR_CW_ERR_PIDX(2)) > cwr(s, TSR_CW_ERR_PIDX(1)),
          "first error by smallest ERR_PIDX is the cause");
    CHECK(cwr(s, TSR_CW_ERR_SEQ(0)) == tks(Pz) && cwr(s, TSR_CW_ERR_INFO(0)) == 0x7FFFFFFFu, "cause record");
    CHECK((cwr(s, TSR_CW_QSTATUS) & TSR_QSTATUS_ERR) == (0x7u << 8), "QSTATUS error bits");
    qclear_join(s);
    // A dep on an op of an errored unit from before its error is not failed.
    const uint32_t L3 = push(s, tsr_enc_ld(0, 100, 0, 2), TSR_TICKET_NONE, TSR_TICKET_NONE);
    push(s, tsr_enc_ld(0, 0x7FFFFFFF, 0, 2), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t Q3 = push(s, tsr_enc_qact(0, 500, 600, 32, 1, 0), L3, TSR_TICKET_NONE);
    idle(s);
    CHECK(ev_find(&L, TSR_QEV_DONE, Q3)->ran && cwr(s, TSR_CW_ERR_CODE(1)) == 0, "dep before ERR_SEQ: runs");
    tsr_sim_destroy(s);
    free(L.v);
}

// ---------------------------------------------------------------------------
// 5. Credits and back-pressure
// ---------------------------------------------------------------------------
static void test_credits(void) {
    tsr_sim * s = mk_sim();
    CHECK(cwr(s, TSR_CW_ISSUE_CREDIT) == (8u | (4u << 4) | (4u << 8) | (15u << 12) | (0u << 16) | (4u << 20)),
          "credits 0x%08x (VU saturates at 15, unit 4 absent)", cwr(s, TSR_CW_ISSUE_CREDIT));
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_HOST_HOLD);
    const tsr_insn ld = tsr_enc_ld(0, 0, 0, 1);
    for (int k = 0; k < 8; ++k) cww(s, TSR_CW_ISSUE_W(k), ld.w[k]);
    for (uint32_t k = 0; k < 8; ++k) {
        CHECK(TSR_GET(cwr(s, TSR_CW_ISSUE_CREDIT), TSR_ISSUE_CREDIT_DMA) == 8 - k, "DMA credit %u", 8 - k);
        CHECK(cww(s, TSR_CW_ISSUE_PUSH, ALL) == TSR_CW_OK, "push %u", k);
    }
    cww(s, TSR_CW_WAIT_CREDIT, TSR_WAIT_CREDIT_EN | TSR_UNIT_DMA);
    CHECK(!(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(5)), "WAIT_CREDIT: no credit");
    const uint32_t pidx = cwr(s, TSR_CW_ISSUE_PIDX);
    CHECK(cww(s, TSR_CW_ISSUE_PUSH, ALL) == TSR_CW_WAIT, "no credit: the store waits");
    CHECK(cwr(s, TSR_CW_ISSUE_PIDX) == pidx && cwr(s, TSR_CW_ISSUED_SEQ(0)) == 8, "a waiting push changes nothing");
    tsr_sim_advance(s, 100);
    CHECK(TSR_GET(tsr_sim_csr_read(s, TSR_CSR_Q_STATE(0)), TSR_Q_STATE_ENTRIES) == 8 &&
          (tsr_sim_csr_read(s, TSR_CSR_Q_STATE(0)) & TSR_Q_STATE_HELD), "8 entries held");
    CHECK((cwr(s, TSR_CW_QSTATUS) & (TSR_QSTATUS_HOLD | TSR_QSTATUS_HOLD_HOST | (1u << 16))) == (TSR_QSTATUS_HOLD | TSR_QSTATUS_HOLD_HOST | (1u << 16)),
          "QSTATUS hold, queue 0 non-empty");
    // The hook's push waits ISSUE_WAIT_MAX clk_rv cycles, then times out.
    const uint64_t t0 = tsr_sim_now(s);
    CHECK(tsr_sim_push(s, ld.w, TSR_TICKET_NONE, TSR_TICKET_NONE, ALL) == TSR_TICKET_NONE, "push safety net");
    CHECK(tsr_sim_now(s) - t0 >= 100000, "waited %llu core cycles", (unsigned long long)(tsr_sim_now(s) - t0));
    CHECK((cwr(s, TSR_CW_ISSUE_STATUS) & TSR_ISSUE_STATUS_PUSH_TIMEOUT) && (tsr_sim_csr_read(s, TSR_CSR_STATUS) & TSR_STATUS_PUSH_TIMEOUT),
          "PUSH_TIMEOUT");
    cww(s, TSR_CW_ISSUE_STATUS, TSR_ISSUE_STATUS_PUSH_TIMEOUT);
    tsr_sim_csr_write(s, TSR_CSR_CTRL, 0);
    idle(s);
    CHECK(tsr_sim_rv_irq(s) & TSR_RV_MIP_FAST(5), "WAIT_CREDIT pending again");
    CHECK(TSR_GET(cwr(s, TSR_CW_ISSUE_CREDIT), TSR_ISSUE_CREDIT_DMA) == 8 && cwr(s, TSR_CW_DONE_SEQ(0)) == 8, "drained");
    // A waiting push that completes: one long LD running, eight queued, the tenth waits for a start.
    const tsr_insn big = tsr_enc_ld(0, 0, 0, 1000);
    for (int k = 0; k < 9; ++k) push(s, big, TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint64_t t1 = tsr_sim_now(s);
    CHECK(tsr_sim_push(s, big.w, TSR_TICKET_NONE, TSR_TICKET_NONE, ALL) == TK(0, 17), "tenth push gets its ticket");
    CHECK(tsr_sim_now(s) - t1 >= 1064, "after the second LD started (%llu cycles)", (unsigned long long)(tsr_sim_now(s) - t1));
    idle(s);
    // SYNC full: QCLEAR waits for a SYNC credit, QHOLD only for FIFO space.
    const uint32_t A = push(s, tsr_enc_ld(0, 0, 0, 2000), TSR_TICKET_NONE, TSR_TICKET_NONE);
    for (int k = 0; k < 4; ++k) push(s, tsr_enc_nop(), A, TSR_TICKET_NONE);
    CHECK(TSR_GET(cwr(s, TSR_CW_ISSUE_CREDIT), TSR_ISSUE_CREDIT_SYNC) == 0, "SYNC full");
    CHECK(cww(s, TSR_CW_QCTRL, TSR_QCTRL_QCLEAR) == TSR_CW_WAIT && cww(s, TSR_CW_QCTRL, TSR_QCTRL_QHOLD_CLR) == TSR_CW_OK,
          "QCLEAR waits, QHOLD does not");
    idle(s);
    tsr_sim_destroy(s);

    // Issue-FIFO full: credits include free FIFO entries (32).
    const uint8_t dep[6] = { 31, 31, 31, 0, 0, 31 };
    s = mk_opts(1, dep, 0);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_DEPTH) == (31u | (31u << 5) | (31u << 10) | (31u << 25)) && cwr(s, TSR_CW_ISSUE_UNITS) == 0x27,
          "custom depths, VU absent");
    for (int k = 0; k < 31; ++k) push(s, tsr_enc_ld(0, 0, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(TSR_GET(cwr(s, TSR_CW_ISSUE_CREDIT), TSR_ISSUE_CREDIT_QACT) == 1, "one FIFO entry left");
    // The QACTs use line 100, away from the LDs' line 0: without a dependency they would race with
    // them (W1b integration: race-free, so the test also runs with TSR_SIM_RACE=fatal, SW §4.6).
    push(s, tsr_enc_qact(0, 100, 100, 32, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(tsr_sim_csr_read(s, TSR_CSR_Q_FIFO) == 32 && cwr(s, TSR_CW_ISSUE_CREDIT) == 0, "FIFO full: every credit 0");
    const tsr_insn q = tsr_enc_qact(0, 100, 100, 32, 1, 0);
    for (int k = 0; k < 8; ++k) cww(s, TSR_CW_ISSUE_W(k), q.w[k]);
    CHECK(cww(s, TSR_CW_ISSUE_PUSH, ALL) == TSR_CW_WAIT, "push waits for FIFO space");
    CHECK(cww(s, TSR_CW_TRACE_MARK, 1) == TSR_CW_WAIT, "TRACE_MARK waits for FIFO space");
    tsr_sim_advance(s, 3);
    CHECK(TSR_GET(cwr(s, TSR_CW_ISSUE_CREDIT), TSR_ISSUE_CREDIT_QACT) >= 1 && cww(s, TSR_CW_ISSUE_PUSH, ALL) == TSR_CW_OK,
          "dispatch frees FIFO entries");
    idle(s);
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 6. SYNC, QABORT, holds
// ---------------------------------------------------------------------------
static void test_sync_abort_hold(void) {
    tsr_sim * s = mk_sim();
    evlog L;
    memset(&L, 0, sizeof(L));
    tsr_sim_set_listener(s, ev_cb, &L);
    // FENCE: completes after everything pushed before it; not a barrier for later ops without a dep.
    const uint32_t A = push(s, tsr_enc_ld(0, 0, 0, 500), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t M = push(s, tsr_enc_mvm(0, 0, 1000, 2000, 0, 64, 256, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t Q1 = push(s, tsr_enc_qact(0, 3000, 3100, 64, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t F = push(s, tsr_enc_fence(), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t Q2 = push(s, tsr_enc_qact(0, 3200, 3100, 64, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t Q3 = push(s, tsr_enc_qact(0, 3300, 3100, 64, 1, 0), F, TSR_TICKET_NONE);
    idle(s);
    const uint64_t fd = t_of(&L, TSR_QEV_DONE, F);
    CHECK(fd > t_of(&L, TSR_QEV_DONE, A) && fd > t_of(&L, TSR_QEV_DONE, M) && fd > t_of(&L, TSR_QEV_DONE, Q1), "FENCE after all earlier");
    CHECK(t_of(&L, TSR_QEV_START, Q2) < fd, "a later op without a dep is not held by the FENCE");
    CHECK(t_of(&L, TSR_QEV_START, Q3) >= fd, "after(fence)");

    // RISC-V QABORT: the running op finishes, every later entry completes ABORTED, all units in error.
    const uint32_t B0 = push(s, tsr_enc_ld(0, 100, 0, 300), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t B1 = push(s, tsr_enc_ld(0, 400, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t C1 = push(s, tsr_enc_qact(0, 500, 600, 32, 1, 0), B0, TSR_TICKET_NONE);
    tsr_sim_qctrl(s, TSR_QCTRL_QABORT);
    const uint32_t D1 = push(s, tsr_enc_mvm(0, 0, 1000, 2000, 0, 4, 256, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    idle(s);
    CHECK(ev_find(&L, TSR_QEV_DONE, B0)->ran && ev_find(&L, TSR_QEV_DONE, B0)->code == 0, "the running op completes");
    CHECK(ev_find(&L, TSR_QEV_DONE, B1)->code == TSR_ERR_ABORTED && ev_find(&L, TSR_QEV_DONE, C1)->code == TSR_ERR_ABORTED &&
          ev_find(&L, TSR_QEV_DONE, D1)->code == TSR_ERR_ABORTED, "queued and later entries ABORTED");
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        CHECK(cwr(s, TSR_CW_ERR_CODE(u)) == (u == TSR_UNIT_ATT ? 0u : TSR_ERR_ABORTED), "unit %u ABORTED", u);
    CHECK(cwr(s, TSR_CW_ERR_SEQ(0)) == tks(B1) && cwr(s, TSR_CW_ERR_W0(0)) == tsr_enc_ld(0, 400, 0, 1).w[0], "DMA: ERR_SEQ = S (first not started)");
    CHECK(cwr(s, TSR_CW_ERR_SEQ(2)) == tks(D1) && cwr(s, TSR_CW_ERR_PIDX(2)) == ev_find(&L, TSR_QEV_ENQ, D1)->pidx,
          "MVM: the first entry not started was still to be pushed");
    qclear_join(s);                                   // QCLEAR clears the SYNC unit's own abort state too

    // RISC-V hold: heads stop; non-running completions drain; QHOLD clear releases.
    tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_SET);
    const uint32_t H = push(s, tsr_enc_ld(0, 100, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t Pz = push(s, tsr_enc_qact(0, 0, 0, 33, 1, 0), TSR_TICKET_NONE, TSR_TICKET_NONE);
    const uint32_t N0 = push(s, tsr_enc_nop(), TSR_TICKET_NONE, TSR_TICKET_NONE);
    tsr_sim_advance(s, 1000);
    CHECK(!ev_find(&L, TSR_QEV_START, H) && (tsr_sim_csr_read(s, TSR_CSR_Q_STATE(0)) & TSR_Q_STATE_HELD), "held");
    CHECK((cwr(s, TSR_CW_QSTATUS) & TSR_QSTATUS_HOLD_RV) && TSR_GET(tsr_sim_csr_read(s, TSR_CSR_STATUS), TSR_STATUS_HOLD_SRC) == TSR_HOLD_SRC_RV,
          "RISC-V hold source");
    CHECK(ev_find(&L, TSR_QEV_DONE, Pz) && ev_find(&L, TSR_QEV_DONE, N0), "poisoned entries and NOPs drain under a hold");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) & TSR_STATUS_QQUIESCENT, "QQUIESCENT while only held entries remain");
    tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_CLR);
    idle(s);
    CHECK(ev_find(&L, TSR_QEV_DONE, H) && ev_find(&L, TSR_QEV_DONE, H)->ran, "released");
    qclear_join(s);
    // Debug-halt hold: HOLD_ON_HALT with the RISC-V halted.
    tsr_sim_csr_write(s, TSR_CSR_CTRL, TSR_CTRL_RV_RUN | TSR_CTRL_HOLD_ON_HALT);
    tsr_sim_rv_set_state(s, TSR_STATUS_RV_HALTED);
    const uint32_t H2 = push(s, tsr_enc_ld(0, 100, 0, 1), TSR_TICKET_NONE, TSR_TICKET_NONE);
    tsr_sim_advance(s, 100);
    CHECK(!ev_find(&L, TSR_QEV_START, H2) && (cwr(s, TSR_CW_QSTATUS) & TSR_QSTATUS_HOLD_HALT), "debug-halt hold");
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED, "RV_HALTED");
    tsr_sim_rv_set_state(s, 0);
    idle(s);
    CHECK(ev_find(&L, TSR_QEV_DONE, H2) != NULL, "resumed");
    tsr_sim_destroy(s);
    free(L.v);
}

// ---------------------------------------------------------------------------
// 7. Effects at completion; a dependent chain equals the op-by-op run bit for bit
// ---------------------------------------------------------------------------
typedef struct { tsr_insn i; } prog_op;

static void gen_program(rng_t * r, tsr_sim * s, prog_op * p, uint32_t * n_out) {
    // Inputs: f32 activations at DRAM ch0 beat 1000+, weights repacked at W_OFF 20000 (both fmts).
    uint32_t n = 0;
    const uint32_t C = tsr_sim_config(s)->num_ch;
    for (uint32_t step = 0; step < 6; ++step) {
        const uint32_t fmt = rng_below(r, 2);
        const uint32_t K = fmt == TSR_FMT_Q4_K ? 256u * (1 + rng_below(r, 4)) : 32u * (1 + rng_below(r, 24));
        const uint32_t N = 1 + rng_below(r, 200);
        const uint32_t woff = 20000 + step * 4000;
        const size_t row_bytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
        uint8_t * blocks = (uint8_t *)malloc(row_bytes * N);
        for (uint32_t row = 0; row < N; ++row) gen_row_blocks(r, fmt, K, blocks + row * row_bytes);
        put_weights(s, fmt, N, K, woff, blocks);
        free(blocks);
        float * x = (float *)malloc(K * sizeof(float));
        gen_f32(r, x, K);
        const uint32_t xb = 1000 + step * 200;
        put_f32_le(tsr_sim_dram(s, 0) + (uint64_t)xb * 64, x, K);
        free(x);
        const uint32_t xl = K * 4 / 64, kind = tsr_qa_kind_for_fmt(fmt);
        const uint32_t f32_line = 100 + step * 1000, qa_line = f32_line + 400, y_line = qa_line + 300;
        p[n++].i = tsr_enc_ld(0, f32_line, xb, xl);
        p[n++].i = tsr_enc_qact(kind, qa_line, f32_line, K, 1, 0);
        p[n++].i = tsr_enc_mvm(fmt, rng_below(r, 2) ? TSR_MVM_FLAG_ACC : 0, y_line, qa_line, woff, N, K, 1, 0);
        p[n++].i = tsr_enc_st(1 + rng_below(r, C - 1), y_line, 500 + step * 64, (uint32_t)tsr_ceil_div_u64(4ull * N, 64));
    }
    *n_out = n;
}

static void test_serial_chain(void) {
    rng_t r1 = { 4242 }, r2 = { 4242 };
    tsr_sim * a = mk_sim(), * b = mk_sim();
    prog_op pa[64], pb[64];
    uint32_t na = 0, nb = 0;
    gen_program(&r1, a, pa, &na);
    gen_program(&r2, b, pb, &nb);
    CHECK(na == nb && memcmp(pa, pb, na * sizeof(prog_op)) == 0, "same program");
    // a: queues, each op depending on the previous one (one pipeline per step runs concurrently
    // with the others through the units; the chain is per step).
    uint32_t prev = TSR_TICKET_NONE;
    for (uint32_t k = 0; k < na; ++k) {
        if (k % 4 == 0) prev = TSR_TICKET_NONE;          // independent steps: disjoint SPM / DRAM
        prev = push(a, pa[k].i, prev, TSR_TICKET_NONE);
    }
    idle(a);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) CHECK(cwr(a, TSR_CW_ERR_CODE(u)) == 0, "no errors (unit %u)", u);
    // b: op by op
    for (uint32_t k = 0; k < nb; ++k) CHECK(tsr_sim_op(b, pb[k].i.w, NULL) == 0, "op %u", k);
    CHECK(memcmp(tsr_sim_spm(a), tsr_sim_spm(b), tsr_sim_config(a)->spm_bytes) == 0, "SPM bit-exact vs tsr_sim_op");
    for (uint32_t c = 0; c < 4; ++c)
        CHECK(memcmp(tsr_sim_dram(a, c), tsr_sim_dram(b, c), 2u << 20) == 0, "DRAM ch%u bit-exact", c);
    tsr_sim_destroy(a);
    tsr_sim_destroy(b);
}

// ---------------------------------------------------------------------------
// 8. Advance events
// ---------------------------------------------------------------------------
static void test_advance_events(void) {
    tsr_sim * s = mk_sim();
    CHECK(tsr_sim_advance(s, 0) == TSR_SIM_EV_BUDGET && tsr_sim_now(s) == 0, "advance 0");
    CHECK(tsr_sim_advance(s, 1000) == TSR_SIM_EV_BUDGET && tsr_sim_now(s) == 1000, "idle budget");
    push(s, tsr_enc_ld(0, 0, 0, 10), TSR_TICKET_NONE, TSR_TICKET_NONE);
    CHECK(tsr_sim_advance(s, 1000000) == TSR_SIM_EV_QIDLE && tsr_sim_now(s) == 1000 + 2 + 74, "QIDLE event at the completion (now %llu)",
          (unsigned long long)tsr_sim_now(s));
    CHECK(tsr_sim_csr_read(s, TSR_CSR_STATUS) & TSR_STATUS_QIDLE, "QIDLE");
    CHECK(tsr_sim_next_event(s) == UINT64_MAX, "nothing pending");
    tsr_sim_destroy(s);
}

// ---------------------------------------------------------------------------
// 9. Random stress: internal invariants (tsr_q.c aborts on a violation) and determinism
// ---------------------------------------------------------------------------
static uint64_t stress_run(uint64_t seed, uint64_t * final_hash) {
    tsr_sim * s = mk_sim();
    evlog L;
    memset(&L, 0, sizeof(L));
    L.hash = 1469598103934665603ull;
    tsr_sim_set_listener(s, ev_cb, &L);
    rng_t r = { seed };
    uint32_t tickets[256];
    uint32_t nt = 0;
    for (int it = 0; it < 3000; ++it) {
        const uint32_t k = rng_below(&r, 100);
        uint32_t d0 = TSR_TICKET_NONE, d1 = TSR_TICKET_NONE;
        if (nt && rng_below(&r, 2)) d0 = tickets[rng_below(&r, nt)];
        if (nt && rng_below(&r, 4) == 0) d1 = tickets[rng_below(&r, nt)];
        if (rng_below(&r, 50) == 0) d1 = TK(rng_below(&r, 16), rng_u32(&r));   // mostly BAD_DEP
        tsr_insn in;
        if (k < 30) in = tsr_enc_ld(rng_below(&r, 5), rng_below(&r, 70000), rng_below(&r, 1000), rng_below(&r, 64));
        else if (k < 45) in = tsr_enc_st(rng_below(&r, 4), rng_below(&r, 66000), 50000 + rng_below(&r, 1000), rng_below(&r, 16));
        else if (k < 60) in = tsr_enc_qact(rng_below(&r, 3), rng_below(&r, 66000), rng_below(&r, 66000), 32 * (1 + rng_below(&r, 20)), 1, 0);
        else if (k < 72) in = tsr_enc_mvm(rng_below(&r, 3), 0, rng_below(&r, 66000), rng_below(&r, 66000), rng_below(&r, 1000),
                                           1 + rng_below(&r, 64), 256, rng_below(&r, 2), 0);
        else if (k < 80) in = tsr_enc_nop();
        else if (k < 84) in = tsr_enc_fence();
        else if (k < 88) in = insn_raw(rng_below(&r, 256), 0, 0, 0, 0, 0, 0, 0);
        else {
            const uint32_t c = rng_below(&r, 20);
            if (c < 6) {
                const uint32_t qc = tsr_sim_qctrl(s, TSR_QCTRL_QCLEAR);
                if (nt < 256) tickets[nt++] = qc;
            } else if (c < 8) tsr_sim_qctrl(s, TSR_QCTRL_QABORT);
            else if (c < 11) tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_SET);
            else if (c < 16) tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_CLR);
            else tsr_sim_cw_write(s, TSR_CW_TRACE_MARK, rng_u32(&r));
            continue;
        }
        const uint32_t t = push(s, in, d0, d1);
        CHECK(t != TSR_TICKET_NONE || (cwr(s, TSR_CW_QSTATUS) & TSR_QSTATUS_HOLD_RV), "push");
        if (t == TSR_TICKET_NONE) {                    // timed out behind a hold: release it
            tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_CLR);
            cww(s, TSR_CW_ISSUE_STATUS, TSR_ISSUE_STATUS_PUSH_TIMEOUT);
            continue;
        }
        if (nt < 256) tickets[nt++] = t; else tickets[rng_below(&r, 256)] = t;
        if (rng_below(&r, 8) == 0) tsr_sim_advance(s, rng_below(&r, 300));
    }
    tsr_sim_qctrl(s, TSR_QCTRL_QHOLD_CLR);
    qclear_join(s);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        CHECK(cwr(s, TSR_CW_DONE_SEQ(u)) == cwr(s, TSR_CW_ISSUED_SEQ(u)), "unit %u: every ticket completed", u);
    size_t done = 0, ran = 0;
    for (size_t k = 0; k < L.n; ++k) {
        done += L.v[k].kind == TSR_QEV_DONE;
        ran += L.v[k].kind == TSR_QEV_DONE && L.v[k].ran;
    }
    CHECK(done == (size_t)cwr(s, TSR_CW_ISSUE_PIDX), "one completion per ticketed entry (%zu of %u)", done, cwr(s, TSR_CW_ISSUE_PIDX));
    CHECK(ran > 20 && ran < done, "a mix of runs (%zu) and non-running completions", ran);
    *final_hash = fnv64(tsr_sim_spm(s), 4u << 20) ^ fnv64(tsr_sim_dram(s, 1), 1u << 20);
    const uint64_t h = L.hash;
    free(L.v);
    tsr_sim_destroy(s);
    return h;
}

static void test_stress(void) {
    uint64_t f1 = 0, f2 = 0, f3 = 0;
    const uint64_t h1 = stress_run(1, &f1), h2 = stress_run(1, &f2), h3 = stress_run(2, &f3);
    CHECK(h1 == h2 && f1 == f2, "deterministic: same seed, same events and state");
    CHECK(h1 != h3, "different seeds differ");
}

int main(void) {
    test_checker_order();
    test_operand_stub();
    test_extents();
    test_extent_limits();
    test_tickets_pidx();
    test_deps();
    test_credits();
    test_sync_abort_hold();
    test_serial_chain();
    test_advance_events();
    test_stress();
    return test_finish("sim_queue");
}

// test_sim_sched.c — ctest sim_sched_determinism (docs/design/06-software-verification.md "SW"
// §4.4, §4.7, §10.2): the determinism check over the fixed schedule and 7 random seeds.
//
// SW §4.7's observables, collected through the listeners of tsr_sim_int.h §5 on every run:
//   - the per-unit op streams: every PUSH (words as pushed, ISSUE_PUSH value, deps as relative
//     tickets {unit, seq - base_u}) and QCTRL entry (on SYNC), projected per unit (SW §5.2);
//   - the LCQ records r0-r6;
//   - the RISC-V's SPM store stream (RV_SPM_WR access events: address, size, value, in order);
//   - the final written DRAM and written SPM lines: the union of the write extents of every op that
//     ran (DONE events) and of the RISC-V's SPM stores, with their content at the end.
// Timing (MTIME, poll counts, PERF, r7) is not compared; that it differs shows the schedules differ.
//
// Parts:
//   1. the sim_rv_issue firmware (test_sim_rvhost.h: every check of sim_rv_issue holds on every run)
//      under fixed + random:1..7 at F_RV / F_CORE 0.603 and 1.0: 16 runs, identical observables;
//   2. random tile programs pushed through tsr_sim_push with correct dependencies (RAW, WAR, WAW
//      tracked per SPM slot and DRAM region, joined through NOPs when more than two) under the 8
//      schedules: identical final state;
//   3. a racy program (LD and VOP writing the same slots, no dependency between them): some random
//      seed ends in another final state than the fixed schedule, so the schedules do perturb;
//   4. fast-forward: a hand-encoded RV32 loop spinning on DONE_SEQ across a ~140 K-cycle MVM, then
//      an SPM read, two SPM stores, an ST and a spin on the DMA's DONE_SEQ, then an LCQ record: with
//      fast-forward the hart retires a few dozen instructions (the jump count is printed), without
//      it thousands; the observables are equal (fixed, without fast-forward, and 7 random seeds).
// Usage: sim_sched_determinism <fw_issue.elf>. Part 1 reports "Skipped" if the ELF was not built
// (no RISC-V compiler at configure time); parts 2-4 always run.
#include "test_sim_rvhost.h"
#include "test_sim_obs.h"         // SW §4.7 observables (obs_*)

#define N_SCHED 8
static const char * const k_sched[N_SCHED] = {
    "fixed", "random:1", "random:2", "random:3", "random:4", "random:5", "random:6", "random:7",
};

// ---------------------------------------------------------------------------
// Part 1: the sim_rv_issue firmware under 8 schedules x 2 ratios
// ---------------------------------------------------------------------------
static void p1_create(tsr_sim * sim, void * ctx) { obs_attach((obs *)ctx, sim); }
static void p1_end(tsr_sim * sim, void * ctx) { obs_finish((obs *)ctx, sim); }

static int part1(const char * elf) {
    FILE * f = fopen(elf, "rb");
    if (f == NULL) {
        printf("sim_sched_determinism part 1: %s not built (no RISC-V compiler at configure time); skipped\n", elf);
        return 0;
    }
    fclose(f);
    static observed rec[2 * N_SCHED];
    static obs ob[2 * N_SCHED];
    const double ratios[2] = { 0.0, 1.0 };
    for (int r = 0; r < 2; ++r) {
        for (int s = 0; s < N_SCHED; ++s) {
            const int k = r * N_SCHED + s;
            memset(&rec[k], 0, sizeof(rec[k]));
            rvh_run run = { ratios[r], k_sched[s], p1_create, p1_end, &ob[k], NULL };
            scenario(elf, &run, &rec[k]);
        }
    }
    int distinct_mtime = 0;
    for (int k = 0; k < 2 * N_SCHED; ++k) {
        char what[96];
        snprintf(what, sizeof(what), "part 1, %s, ratio %s", k_sched[k % N_SCHED], k < N_SCHED ? "0.603" : "1.0");
        CHECK(rec[k].n_rec == 13u && rec[k].n_rec == rec[0].n_rec && memcmp(rec[k].rec, rec[0].rec, sizeof(rec[0].rec)) == 0,
              "%s: LCQ records r0-r6 equal", what);
        CHECK(rec[k].out_hash == rec[0].out_hash && rec[k].s0 == rec[0].s0, "%s: outputs and s0 equal", what);
        CHECK(ob[k].stream_n[TSR_UNIT_DMA] == 8u && ob[k].stream_n[TSR_UNIT_SYNC] == 6u && ob[k].n_written_beats > 0,
              "%s: streams seen (DMA %u, SYNC %u entries)", what, ob[k].stream_n[TSR_UNIT_DMA], ob[k].stream_n[TSR_UNIT_SYNC]);
        obs_equal(&ob[0], &ob[k], what);
        for (int j = 0; j < k; ++j) if (rec[j].mtime == rec[k].mtime) goto same;
        ++distinct_mtime;
    same:;
    }
    CHECK(distinct_mtime >= 12, "the schedules change the timing: %d distinct final MTIMEs of 16", distinct_mtime);
    printf("sim_sched_determinism part 1: 16 runs, observables equal; %d distinct MTIMEs (fixed %llu / %llu, random:1 %llu / %llu)\n",
           distinct_mtime, (unsigned long long)rec[0].mtime, (unsigned long long)rec[N_SCHED].mtime,
           (unsigned long long)rec[1].mtime, (unsigned long long)rec[N_SCHED + 1].mtime);
    for (int k = 0; k < 2 * N_SCHED; ++k) obs_free(&ob[k]);
    return 1;
}

// ---------------------------------------------------------------------------
// Part 2 / 3: tile programs through tsr_sim_push
// ---------------------------------------------------------------------------
#define SLOTS       8u
#define SLOT_LINES  64u
#define K_ACT       256u
#define N_OUT       64u
#define IN_BEATS    1024u
#define W_OFF       4096u
#define OUT_BEAT    8192u
#define N_OUTREG    4u

typedef struct tk { uint32_t t; uint32_t order; } tk;   // a ticket and its push order

typedef struct prog {
    tsr_sim * sim;
    rng_t     r;
    uint32_t  order;
    tk        writer[SLOTS];
    tk        readers[SLOTS][TSR_NUM_UNITS];
    tk        out_writer[N_OUTREG];
    uint32_t  n_ops, n_nops;
    int       vu;                               // VOP available (CAPS VU)
} prog;

static const tk TK_NONE = { TSR_TICKET_NONE, 0 };

static tsr_sim * mk_tile_sim(const char * sched) {
    tsr_config cfg = TSR_CONFIG_U200;
    cfg.num_ch = 2;
    cfg.ch_bytes = 64ull << 20;
    cfg.caps = tsr_sim_caps_implemented() & TSR_CAPS_VU;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = sched;
    tsr_sim * s = tsr_sim_create_opts(&cfg, &o);
    if (s == NULL) {
        fprintf(stderr, "create (%s) failed\n", sched);
        exit(1);
    }
    return s;
}

// The same DRAM inputs and weights for every schedule (host writes through the API).
static void tile_data(tsr_sim * sim, uint64_t seed) {
    rng_t r = { seed };
    float x[IN_BEATS * 16u];
    for (uint32_t c = 0; c < 2; ++c) {
        gen_f32(&r, x, IN_BEATS * 16u);
        uint8_t * b = (uint8_t *)malloc(IN_BEATS * 64u);
        put_f32_le(b, x, IN_BEATS * 16u);
        tsr_sim_dram_write(sim, c, 0, b, IN_BEATS * 64u);
        free(b);
    }
    const size_t row_bytes = (size_t)(K_ACT / 32u) * 34u;
    uint8_t * blocks = (uint8_t *)malloc(row_bytes * N_OUT);
    for (uint32_t n = 0; n < N_OUT; ++n) gen_row_blocks(&r, TSR_FMT_Q8_0, K_ACT, blocks + row_bytes * n);
    put_weights(sim, TSR_FMT_Q8_0, N_OUT, K_ACT, W_OFF, blocks);
    free(blocks);
}

static tsr_insn enc_vop1(uint32_t func, uint32_t d, uint32_t a, uint32_t m, uint32_t rows) {
    tsr_insn i = {{0}};
    i.w[0] = TSR_OP_VOP;
    TSR_INSN_SET(i.w, TSR_VOP_CMODE, TSR_VOP_CMODE_NONE);
    TSR_INSN_SET(i.w, TSR_VOP_FUNC, func);
    TSR_INSN_SET(i.w, TSR_VOP_D, d);
    TSR_INSN_SET(i.w, TSR_VOP_A, a);
    TSR_INSN_SET(i.w, TSR_VOP_M, m);
    TSR_INSN_SET(i.w, TSR_VOP_ROWS, rows);
    return i;
}

static void dep_add(tk * best, tk t, uint32_t self_unit) {
    if (t.t == TSR_TICKET_NONE) return;
    const uint32_t u = t.t >> 28;
    if (u == self_unit) return;                      // implied by the unit's queue order
    if (best[u].t == TSR_TICKET_NONE || t.order > best[u].order) best[u] = t;
}

static tk pushed(prog * p, uint32_t t) { tk k = { t, ++p->order }; return k; }

// Pushes op i with dependencies on everything in best[] (per unit the latest); more than two are
// joined through NOPs on SYNC first.
static tk push_with(prog * p, const tsr_insn * i, tk * best) {
    tk list[TSR_NUM_UNITS];
    uint32_t n = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) if (best[u].t != TSR_TICKET_NONE) list[n++] = best[u];
    while (n > 2) {
        const tsr_insn nop = tsr_enc_nop();
        const uint32_t t = tsr_sim_push(p->sim, nop.w, list[n - 1].t, list[n - 2].t, TSR_ISSUE_PUSH_WMASK);
        p->n_nops++;
        n -= 2;
        list[n++] = pushed(p, t);
    }
    const uint32_t t = tsr_sim_push(p->sim, i->w, n > 0 ? list[0].t : TSR_TICKET_NONE, n > 1 ? list[1].t : TSR_TICKET_NONE,
                                    TSR_ISSUE_PUSH_WMASK);
    p->n_ops++;
    CHECK(t != TSR_TICKET_NONE, "push accepted");
    return pushed(p, t);
}

// One op reading slots rd[0..nr) and writing slot wr (or an output region), with every hazard
// (RAW, WAR, WAW) as a dependency unless `racy` drops the WAW / WAR ones.
static tk tile_op(prog * p, const tsr_insn * i, const uint32_t * rd, uint32_t nr, int wr, int out_reg, int racy) {
    const uint32_t self = tsr_op_unit(i->w[0] & 0xFFu);
    tk best[TSR_NUM_UNITS];
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) best[u] = TK_NONE;
    for (uint32_t k = 0; k < nr; ++k) dep_add(best, p->writer[rd[k]], self);            // RAW
    if (wr >= 0 && !racy) {
        dep_add(best, p->writer[wr], self);                                              // WAW
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) dep_add(best, p->readers[wr][u], self);   // WAR
    }
    if (out_reg >= 0) dep_add(best, p->out_writer[out_reg], self);                      // DRAM WAW
    const tk t = push_with(p, i, best);
    for (uint32_t k = 0; k < nr; ++k) {
        tk * rr = &p->readers[rd[k]][self];
        if (rr->t == TSR_TICKET_NONE || t.order > rr->order) *rr = t;
    }
    if (wr >= 0) {
        p->writer[wr] = t;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) p->readers[wr][u] = TK_NONE;
    }
    if (out_reg >= 0) p->out_writer[out_reg] = t;
    return t;
}

static void prog_init(prog * p, tsr_sim * sim, uint64_t seed) {
    memset(p, 0, sizeof(*p));
    p->sim = sim;
    p->r.s = seed;
    for (uint32_t s = 0; s < SLOTS; ++s) {
        p->writer[s] = TK_NONE;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) p->readers[s][u] = TK_NONE;
    }
    for (uint32_t k = 0; k < N_OUTREG; ++k) p->out_writer[k] = TK_NONE;
    p->vu = (tsr_sim_config(sim)->caps & TSR_CAPS_VU) != 0;
}

// A random valid program: LD (DMA), QACT, MVM, VOP (VU), ST over 8 SPM slots of 64 lines.
static void random_program(prog * p, uint32_t n_ops) {
    const uint32_t qa = tsr_qa_lines(TSR_QA_A32, K_ACT);
    static const uint32_t funcs[4] = { TSR_VOP_FUNC_SILU, TSR_VOP_FUNC_TANH, TSR_VOP_FUNC_GELU, TSR_VOP_FUNC_SILU };
    for (uint32_t k = 0; k < n_ops; ++k) {
        const uint32_t kind = rng_below(&p->r, p->vu ? 5u : 4u);
        const uint32_t d = rng_below(&p->r, SLOTS), s = rng_below(&p->r, SLOTS);
        tsr_insn i;
        uint32_t rd[1] = { s };
        switch (kind) {
        case 0: {                                                 // LD slot d <- input
            const uint32_t ch = rng_below(&p->r, 2u), b = rng_below(&p->r, IN_BEATS - 16u);
            i = tsr_enc_ld(ch, d * SLOT_LINES, b, 16u);
            tile_op(p, &i, rd, 0, (int)d, -1, 0);
            break;
        }
        case 1:                                                   // QACT slot d <- slot s (K f32)
            if (d == s) { --k; continue; }
            i = tsr_enc_qact(TSR_QA_A32, d * SLOT_LINES, s * SLOT_LINES, K_ACT, 1, 0);
            tile_op(p, &i, rd, 1, (int)d, -1, 0);
            break;
        case 2:                                                   // MVM slot d <- W . slot s (QA)
            if (d == s) { --k; continue; }
            i = tsr_enc_mvm(TSR_FMT_Q8_0, 0, d * SLOT_LINES, s * SLOT_LINES, W_OFF, N_OUT, K_ACT, 1, 0);
            (void)qa;
            tile_op(p, &i, rd, 1, (int)d, -1, 0);
            break;
        case 3: {                                                 // ST output region <- slot s
            const uint32_t reg = rng_below(&p->r, N_OUTREG);
            i = tsr_enc_st(reg % 2u, s * SLOT_LINES, OUT_BEAT + 64u * reg, 4u);
            tile_op(p, &i, rd, 1, -1, (int)reg, 0);
            break;
        }
        default:                                                  // VOP slot d <- f(slot s)
            if (d == s) { --k; continue; }
            i = enc_vop1(funcs[rng_below(&p->r, 4u)], d * SLOT_LINES, s * SLOT_LINES, N_OUT, 1);
            tile_op(p, &i, rd, 1, (int)d, -1, 0);
            break;
        }
    }
}

// Final state of a tile program: every slot and every output region (all of SPM's slots: a valid
// program's unwritten lines are the same 0xA5 fill everywhere).
static uint64_t tile_state(tsr_sim * sim) {
    uint64_t h = 1469598103934665603ull;
    h = h64(h, fnv64(tsr_sim_spm(sim), (size_t)SLOTS * SLOT_LINES * 64u));
    for (uint32_t c = 0; c < 2; ++c) h = h64(h, fnv64(tsr_sim_dram(sim, c) + (uint64_t)OUT_BEAT * 64u, N_OUTREG * 64u * 64u));
    h = h64(h, fnv64(tsr_sim_sreg(sim), 64));
    return h;
}

static void part2(void) {
    int distinct_total = 0;
    for (uint64_t pseed = 1; pseed <= 4; ++pseed) {
        uint64_t st[N_SCHED], mt[N_SCHED];
        uint32_t nops = 0, nnops = 0;
        for (int s = 0; s < N_SCHED; ++s) {
            tsr_sim * sim = mk_tile_sim(k_sched[s]);
            tile_data(sim, 1000u + pseed);
            prog p;
            prog_init(&p, sim, 0xC0DE0000u + pseed);
            random_program(&p, 80);
            CHECK(tsr_sim_run_until_idle(sim, 100000000ull), "program %llu, %s: idle", (unsigned long long)pseed, k_sched[s]);
            const uint32_t st_reg = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
            CHECK(!(st_reg & TSR_STATUS_QERR), "program %llu, %s: no errors (STATUS 0x%08x)", (unsigned long long)pseed,
                  k_sched[s], st_reg);
            st[s] = tile_state(sim);
            mt[s] = tsr_sim_now(sim);
            nops = p.n_ops;
            nnops = p.n_nops;
            tsr_sim_destroy(sim);
        }
        int distinct = 0;
        for (int s = 0; s < N_SCHED; ++s) {
            CHECK(st[s] == st[0], "program %llu: %s final state equals fixed", (unsigned long long)pseed, k_sched[s]);
            int seen = 0;
            for (int j = 0; j < s; ++j) seen |= mt[j] == mt[s];
            distinct += !seen;
        }
        CHECK(distinct >= 6, "program %llu: the schedules change the timing (%d distinct MTIMEs)", (unsigned long long)pseed, distinct);
        distinct_total += distinct;
        printf("sim_sched_determinism part 2: program %llu (%u ops + %u NOP joins): 8 schedules, one final state; MTIME fixed %llu, "
               "random:1 %llu\n", (unsigned long long)pseed, nops - nnops, nnops, (unsigned long long)mt[0],
               (unsigned long long)mt[1]);
    }
    (void)distinct_total;
}

// Part 3: LD (DMA) and VOP (VU) write the same slots 1..6 with no dependency between them; each
// VOP waits for LD 0 (its input), each LD k for nothing. Which writer finishes last decides the
// slot, so the final state depends on the schedule.
static uint64_t racy_run(const char * sched, int * have_vu) {
    tsr_sim * sim = mk_tile_sim(sched);
    tile_data(sim, 77);
    prog p;
    prog_init(&p, sim, 5);
    *have_vu = p.vu;
    uint64_t h = 0;
    if (p.vu) {
        tsr_insn i = tsr_enc_ld(0, 0, 0, 16u);
        tile_op(&p, &i, NULL, 0, 0, -1, 0);
        const uint32_t rd0[1] = { 0 };
        for (uint32_t k = 1; k <= 6; ++k) {
            i = enc_vop1(TSR_VOP_FUNC_SILU, k * SLOT_LINES, 0, 256u, 1);   // 16 lines
            tile_op(&p, &i, rd0, 1, (int)k, -1, 1);
            i = tsr_enc_ld(1, k * SLOT_LINES, 16u * k, 16u);
            tile_op(&p, &i, NULL, 0, (int)k, -1, 1);
        }
        tsr_sim_run_until_idle(sim, 100000000ull);
        h = tile_state(sim);
    }
    tsr_sim_destroy(sim);
    return h;
}

static void part3(void) {
    // The program is racy on purpose: a race detector in fatal mode (TSR_SIM_RACE, SW §4.6) would
    // stop it, so this part runs with the detector off (tsr_sim_opts.race 0 takes the environment).
    char race_env[32] = "";
    const char * re = getenv("TSR_SIM_RACE");
    if (re != NULL) snprintf(race_env, sizeof(race_env), "%s", re);
    unsetenv("TSR_SIM_RACE");
    int vu = 0;
    uint64_t st[N_SCHED];
    int differ = 0;
    for (int s = 0; s < N_SCHED; ++s) {
        st[s] = racy_run(k_sched[s], &vu);
        if (!vu) break;
        if (s > 0 && st[s] != st[0]) ++differ;
    }
    if (vu) {
        CHECK(differ >= 1, "the racy program ends in another state under some random seed (%d of 7 differ)", differ);
        // Deterministic per seed: the same schedule twice gives the same (racy) result.
        CHECK(racy_run(k_sched[3], &vu) == st[3], "a random schedule is reproducible");
        printf("sim_sched_determinism part 3: racy program: %d of 7 random seeds end in another state than fixed\n", differ);
    } else {
        printf("sim_sched_determinism part 3: no VU in this build; skipped\n");
    }
    if (race_env[0] != '\0') setenv("TSR_SIM_RACE", race_env, 1);
}

// ---------------------------------------------------------------------------
// Part 4: spin fast-forward
// ---------------------------------------------------------------------------
#define LUI(rd, imm)        (((uint32_t)(imm) & 0xFFFFF000u) | ((uint32_t)(rd) << 7) | 0x37u)
#define ADDI(rd, rs1, imm)  ((((uint32_t)(imm) & 0xFFFu) << 20) | ((uint32_t)(rs1) << 15) | ((uint32_t)(rd) << 7) | 0x13u)
#define SW(rs2, rs1, imm)   ((((uint32_t)(imm) >> 5 & 0x7Fu) << 25) | ((uint32_t)(rs2) << 20) | ((uint32_t)(rs1) << 15) | \
                             (2u << 12) | (((uint32_t)(imm) & 0x1Fu) << 7) | 0x23u)
#define LW(rd, rs1, imm)    ((((uint32_t)(imm) & 0xFFFu) << 20) | ((uint32_t)(rs1) << 15) | (2u << 12) | ((uint32_t)(rd) << 7) | 0x03u)
#define CSRRW(rd, c, rs1)   (((uint32_t)(c) << 20) | ((uint32_t)(rs1) << 15) | (1u << 12) | ((uint32_t)(rd) << 7) | 0x73u)
#define WFI                 0x10500073u

static uint32_t enc_b(uint32_t f3, uint32_t rs1, uint32_t rs2, int32_t off) {
    const uint32_t o = (uint32_t)off;
    return (((o >> 12) & 1u) << 31) | (((o >> 5) & 0x3Fu) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) |
           (((o >> 1) & 0xFu) << 8) | (((o >> 11) & 1u) << 7) | 0x63u;
}
static uint32_t enc_jal(uint32_t rd, int32_t off) {
    const uint32_t o = (uint32_t)off;
    return (((o >> 20) & 1u) << 31) | (((o >> 1) & 0x3FFu) << 21) | (((o >> 11) & 1u) << 20) | (((o >> 12) & 0xFFu) << 12) |
           (rd << 7) | 0x6Fu;
}

#define SP_Y      0u          // MVM output, 256 lines
#define SP_X      1024u       // QA input, 68 lines
#define SP_OUT    2048u       // the RISC-V's stores
#define SP_N      4096u
#define SP_K      4096u
#define SP_STBEAT 500000u

typedef struct pw { uint8_t * itcm; uint32_t pc; } pw;
static void put(pw * p, uint32_t w) {
    for (int b = 0; b < 4; ++b) p->itcm[p->pc + (uint32_t)b] = (uint8_t)(w >> (8 * b));
    p->pc += 4;
}

static uint32_t g_push_pc[2];                    // where spin_program put its two ISSUE_PUSH stores

static void spin_program(tsr_sim * sim) {
    pw p = { tsr_sim_itcm(sim, NULL), 0x80 };
    uint8_t * dtcm = tsr_sim_dtcm(sim, NULL);
    const tsr_insn mvm = tsr_enc_mvm(TSR_FMT_Q8_0, 0, SP_Y, SP_X, 0, SP_N, SP_K, 1, 0);
    const tsr_insn st = tsr_enc_st(0, SP_OUT, SP_STBEAT, 1);
    for (int k = 0; k < 8; ++k) {
        for (int b = 0; b < 4; ++b) {
            dtcm[4 * k + b] = (uint8_t)(mvm.w[k] >> (8 * b));
            dtcm[32 + 4 * k + b] = (uint8_t)(st.w[k] >> (8 * b));
        }
    }
    put(&p, LUI(10, TSR_RV_CTL_BASE));
    put(&p, LUI(11, TSR_RV_SPM_BASE));
    put(&p, LUI(12, TSR_RV_DTCM_BASE));
    put(&p, LUI(16, TSR_RV_SPM_BASE + SP_OUT * 64u));
    put(&p, LW(7, 10, TSR_CW_DONE_SEQ(TSR_UNIT_MVM)));          // DONE_SEQ before the push
    for (int k = 0; k < 8; ++k) { put(&p, LW(5, 12, 4 * k)); put(&p, SW(5, 10, TSR_CW_ISSUE_W(k))); }
    put(&p, LUI(5, 0x10000)); put(&p, ADDI(5, 5, -256));         // 0xFF00: all eight words
    g_push_pc[0] = p.pc;
    put(&p, SW(5, 10, TSR_CW_ISSUE_PUSH));
    const uint32_t spin1 = p.pc;
    put(&p, LW(8, 10, TSR_CW_DONE_SEQ(TSR_UNIT_MVM)));            // the spin (SW §4.4)
    put(&p, enc_b(0, 8, 7, (int32_t)spin1 - (int32_t)p.pc));     // beq x8, x7, spin1
    put(&p, LW(13, 11, SP_Y * 64u));                            // y[0] through the SPM window
    put(&p, SW(13, 16, 0));                                      // SPM stores
    put(&p, SW(8, 16, 4));
    put(&p, LW(9, 10, TSR_CW_DONE_SEQ(TSR_UNIT_DMA)));
    for (int k = 0; k < 8; ++k) { put(&p, LW(5, 12, 32 + 4 * k)); put(&p, SW(5, 10, TSR_CW_ISSUE_W(k))); }
    put(&p, LUI(5, 0x10000)); put(&p, ADDI(5, 5, -256));
    g_push_pc[1] = p.pc;
    put(&p, SW(5, 10, TSR_CW_ISSUE_PUSH));                       // ST of the stored line
    const uint32_t spin2 = p.pc;
    put(&p, LW(14, 10, TSR_CW_DONE_SEQ(TSR_UNIT_DMA)));
    put(&p, enc_b(0, 14, 9, (int32_t)spin2 - (int32_t)p.pc));
    put(&p, ADDI(15, 0, 0xC1)); put(&p, SW(15, 10, TSR_CW_LCQ_W(0)));
    put(&p, SW(13, 10, TSR_CW_LCQ_W(1)));
    put(&p, SW(8, 10, TSR_CW_LCQ_W(2)));
    put(&p, SW(14, 10, TSR_CW_LCQ_W(3)));
    put(&p, SW(0, 10, TSR_CW_LCQ_POST));                          // an event record (no ANSWER)
    const uint32_t park = p.pc;
    put(&p, CSRRW(0, 0x304, 0));
    put(&p, WFI);
    put(&p, enc_jal(0, (int32_t)park - (int32_t)p.pc));
}

typedef struct spin_res { obs o; uint32_t r[7]; int posted; uint32_t push_pc[2], n_push; } spin_res;

// The push PC of every PUSH event (CC §7.1: the ISSUE_PUSH store's pc).
static void pc_cb(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    (void)sim;
    spin_res * r = (spin_res *)ctx;
    if (ev->kind == TSR_QEV_PUSH && r->n_push < 2u) r->push_pc[r->n_push++] = ev->pc;
}

static void spin_run(const char * sched, int ff, spin_res * res) {
    tsr_config cfg = TSR_CONFIG_U200;
    cfg.num_ch = 2;
    cfg.ch_bytes = 64ull << 20;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = sched;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    CHECK(sim != NULL, "create (%s)", sched);
    if (sim == NULL) return;
    tsr_sim_set_fast_forward(sim, ff);
    // Weights (N x K Q8_0 rows on both channels) and the QA input: arbitrary bytes, the same for
    // every run (any bit pattern is a valid Q8_0 row and QA image).
    uint64_t x = 0x9E3779B97F4A7C15ull;
    const uint64_t wbeats = (uint64_t)(SP_N / 2u) * tsr_row_beats(TSR_FMT_Q8_0, SP_K);
    for (uint32_t c = 0; c < 2; ++c) {
        uint64_t * d = (uint64_t *)tsr_sim_dram(sim, c);
        for (uint64_t k = 0; k < wbeats * 8u; ++k) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; d[k] = x; }
        tsr_sim_dram_note_write(sim, c, 0, wbeats * 64u);
    }
    uint8_t * spm = tsr_sim_spm(sim);
    for (uint32_t k = 0; k < tsr_qa_lines(TSR_QA_A32, SP_K) * 64u; ++k) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; spm[SP_X * 64u + k] = (uint8_t)x; }
    spin_program(sim);
    obs_attach(&res->o, sim);
    res->n_push = 0;
    tsr_sim_add_listener(sim, pc_cb, res);
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    res->posted = 0;
    for (int it = 0; it < 1000 && !res->posted; ++it) {
        const uint32_t ev = tsr_sim_advance(sim, 1000000);
        if (ev & TSR_SIM_EV_LCQ) res->posted = 1;
    }
    tsr_sim_advance(sim, 1000);
    for (int j = 0; j < 7; ++j) res->r[j] = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(0, j));
    CHECK(res->posted && res->r[0] == 0xC1u && res->r[2] == 1u && res->r[3] == 1u, "%s ff %d: record posted (r0 0x%x)", sched,
          ff, res->r[0]);
    const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
    CHECK((st & TSR_STATUS_QIDLE) && !(st & TSR_STATUS_QERR) && (st & TSR_STATUS_RV_SLEEP), "%s ff %d: idle, parked (0x%08x)",
          sched, ff, st);
    obs_finish(&res->o, sim);
    tsr_sim_destroy(sim);
}

static void part4(void) {
    static spin_res base, noff, rnd[N_SCHED], rnd_noff;
    spin_run("fixed", 1, &base);
    spin_run("fixed", 0, &noff);
    printf("sim_sched_determinism part 4: MVM %u x %u spin: fast-forward %llu RV instructions, %llu jumps, MTIME %llu; "
           "without %llu RV instructions, MTIME %llu\n", SP_N, SP_K, (unsigned long long)base.o.instret,
           (unsigned long long)base.o.ff_jumps, (unsigned long long)base.o.mtime, (unsigned long long)noff.o.instret,
           (unsigned long long)noff.o.mtime);
    CHECK(base.o.ff_jumps >= 1u && base.o.instret < 300u, "fast-forward: %llu instructions, %llu jumps",
          (unsigned long long)base.o.instret, (unsigned long long)base.o.ff_jumps);
    CHECK(noff.o.ff_jumps == 0u && noff.o.instret > 5000u, "without fast-forward the loop spins (%llu instructions)",
          (unsigned long long)noff.o.instret);
    CHECK(memcmp(base.r, noff.r, sizeof(base.r)) == 0, "records equal with and without fast-forward");
    // A jump lands on the event that ends the spin (SW §4.4), so the fast-forward run is never later
    // and at most one poll iteration earlier per spin loop (two loops; one iteration = lw from the
    // control window + taken beq, (1 + 3 + 3) RV cycles = 12 core cycles at 166.7 / 276.3).
    CHECK(base.o.mtime <= noff.o.mtime && noff.o.mtime - base.o.mtime <= 2u * 12u,
          "fast-forward MTIME %llu within two poll iterations before %llu", (unsigned long long)base.o.mtime,
          (unsigned long long)noff.o.mtime);
    CHECK(base.o.spm_st_n == 2u, "two RISC-V SPM stores (%u)", base.o.spm_st_n);
    CHECK(base.n_push == 2u && base.push_pc[0] == g_push_pc[0] && base.push_pc[1] == g_push_pc[1],
          "PUSH events carry the ISSUE_PUSH store's pc (0x%x, 0x%x; expected 0x%x, 0x%x)", base.push_pc[0], base.push_pc[1],
          g_push_pc[0], g_push_pc[1]);
    obs_equal(&base.o, &noff.o, "part 4, fixed without fast-forward");
    for (int s = 1; s < N_SCHED; ++s) {
        char what[64];
        snprintf(what, sizeof(what), "part 4, %s", k_sched[s]);
        spin_run(k_sched[s], 1, &rnd[s]);
        CHECK(memcmp(base.r, rnd[s].r, sizeof(base.r)) == 0, "%s: records equal", what);
        CHECK(rnd[s].o.instret < 300u, "%s: fast-forward bounds the spin (%llu instructions, %llu jumps)", what,
              (unsigned long long)rnd[s].o.instret, (unsigned long long)rnd[s].o.ff_jumps);
        obs_equal(&base.o, &rnd[s].o, what);
        obs_free(&rnd[s].o);
    }
    spin_run("random:1", 0, &rnd_noff);
    CHECK(memcmp(base.r, rnd_noff.r, sizeof(base.r)) == 0, "random:1 without fast-forward: records equal");
    obs_equal(&base.o, &rnd_noff.o, "part 4, random:1 without fast-forward");
    obs_free(&base.o);
    obs_free(&noff.o);
    obs_free(&rnd_noff.o);
}

int main(int argc, char ** argv) {
    unsetenv("TSR_SIM_SCHED");                   // every run names its schedule
    const char * elf = argc > 1 ? argv[1] : "";
    const int p1 = part1(elf);
    part2();
    part3();
    part4();
    const int rc = test_finish("sim_sched_determinism");
    if (rc == 0 && !p1) {
        printf("sim_sched_determinism: part 1 (firmware) skipped\n");
        return 77;
    }
    return rc;
}

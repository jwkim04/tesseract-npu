// tsr_at.c — the simulator's AT ops (P1.5): decode, dispatch checks, operand extents, execution and
// cycle model of KVST 0x14 (unit 0 DMA), ATTQK 0x21 and ATTPV 0x22 (unit 2 MVM), and of the vector
// ops ASMAX 0x34, AMASK 0x35, AMERGE 0x36 and ANORM 0x37 (unit 3, executed by the VU).
//
// Normative sources: docs/v2/03-attention-kv.md (AT) §4 (KV planes), §5 (encodings and semantics),
// §5.8 (checks), §5.10 (CAPS, CFG3, units), §8–§10 and §13.4 (structure, rates, PERF, the cycle
// model); docs/v2/02-vector-unit.md (VU) §5.11, §7.4, §7.9 (the vector ops on the VU and their
// rates); docs/v2/01-execution-model.md (EX) §6, §7.1, §8.1. Every numeric result and every byte
// placement comes from tsr_ref_at.c; this file decodes with the generated field constants
// (tsr_tables.h), checks, and points the references at the simulator's SPM and channel DRAM.
// CAPS bit5 (ATT) is implemented when all seven ops are registered (tsr_at_register).
//
// Checks (AT §5.8; EX §8.1 order ILLEGAL_OP -> BAD_FMT -> K_RANGE -> T_RANGE -> OPERAND ->
// SPM_RANGE -> DRAM_RANGE; BAD_DEP is the queue model's):
//   ILLEGAL_OP  the reserved bits of the encoding tables (TSR_*_RSV_W*), KVST sub >= 2, ATTQK w7 != 0
//               without MAXOUT.
//   BAD_FMT     ATTQK / ATTPV sub (KV format) != 0.
//   K_RANGE     E/64 = 0 (KVST, ATTQK, ATTPV); G·E > K_MAX (ATTQK, ATTPV). KVST has no K_MAX bound
//               (AT §5.8 lists only E/64 = 0; the DMA moves beats).
//   T_RANGE     ATTQK / ATTPV: T_q·GS > T_MAX.
//   OPERAND     AT §5.8's conditions, ERR_INFO = the value in parentheses there, in this order:
//               KVST     LG = 0 (0); c0 + n_rows > 16·C·LG (c0)
//               ATTQK    E % D != 0 (E); hg0 + GS > G (hg0 + GS); n_groups not in [1, 4095]
//                        (n_groups); c_lo > c_hi or c_hi > 16·n_groups (c_hi); pass rows
//                        (E/D)·GS·T_q > 128 (rows); 0 < rstride < n_groups (rstride)
//               ATTPV    as ATTQK without c_lo / c_hi and rstride, then ℓ_max >= LG
//                        (u0 + n_groups - 1); (E/D)·16·ceil(n_groups/C) > PCAP (n_groups);
//                        0 < pstride < n_groups (pstride)
//               ASMAX    n_lines not in [1, 4095] (n_lines); R not in [1, 1024] (R); scale not
//                        finite, not normal or <= 0 (the scale bits)
//               AMASK    n_lines; R; rpq = 0 (0)
//               AMERGE   R; scale as ASMAX        ANORM  R
//   SPM_RANGE / DRAM_RANGE  operands in field order with AT §5.8's extents, ERR_INFO = the start.
//               PCAP = QA_LINES(A256, K_MAX)·16 (CFG3[31:16]·16, AT §5.8). The DRAM extent is
//               checked on every channel (CC §8.2) and listed with TSR_EXT_CH_ALL.
//
// Extents (tsr_sim_int.h §1): the range-check extents in field order. KVST lists its plane as one
// DRAM write (its read-modify-write of partial meta beats only re-writes bytes it read, so the
// write extent already covers every conflict); a merging AMERGE lists Orun and the running state
// as WRITE and READ (INIT only writes them). Rows of other passes inside an ATTQK / ATTPV extent
// are left unchanged by the op (tsr_ref_at.c) but lie inside the listed extent.
//
// Execution: at completion (SW §4.1), on the simulator's SPM (f32 words / QA bytes) and the KV
// planes in channel DRAM (plane[c] = channel c byte 64·B). Capture (SW §5.5): ATTQK / ATTPV report
// the beats each channel streams (whole local groups); KVST reports the partial meta beats it
// reads (read-modify-write) and every beat it writes.
//
// Cycle model (unit busy time, deterministic; AT §13.4 and VU §7.9 *estimates*, not part of the
// bit-exact contract). beats_c = the groups of channel c times GB = 17·E/64 (one beat per cycle
// per channel), lines = pass rows · n_groups (the gather writes / the p broadcast reads one SPM
// line per cycle, AT §8.1–§8.2):
//   KVST    40 + the beats written (meta + quant, all channels) + the meta beats read (RMW)
//   ATTQK   350 + max(max_c beats_c, lines)               (fill ≈ 300–400, AT §9.1–§9.2)
//   ATTPV   350 + 68 + max(max_c beats_c, lines)          (+ the 64-beat segment delay, AT §13.4)
//   ASMAX   R·n_lines + 64                                (VU §7.9)
//   AMASK   R·n_lines + R if n_lines is odd (pair reads of two sources) + 45
//   AMERGE  6·ceil(R/16) + 64 + R·D/16·(C + 1) (INIT: max(1, C - 1)): C partial line reads + the
//           Orun line per output line (VU §7.4: ≈ 5 at C = 4, INIT ≈ 3; AT §9.2)
//   ANORM   4·ceil(R/16) + 64 + R·D/16
// PERF (EX §6, AT §10): MVM_CYCLES, ATT_QK_CYCLES, ATT_PV_CYCLES, DMA_CYCLES and VU_CYCLES are the
// queue model's busy time. Counted here at completion (events, gated by ACTIVE): CH_RD_BEATS[c]
// += beats_c (ATTQK, ATTPV) or the RMW reads of channel c (KVST); KVST_RMW += the meta beats
// read-modify-written; VU_LINES += the lines the vector ops write (ASMAX R·n_lines, AMASK
// R·n_lines + ceil(R/16), AMERGE R·D/16 + 2·ceil(R/16), ANORM R·D/16); ATT_PATH_STALL (a time
// counter) += max(0, lines - max_c beats_c), the gather / broadcast-bound share of the op, added at
// completion.

#include "tsr_sim.h"
#include "tsr_sim_int.h"
#include "tsr_ref.h"
#include "tsr_ref_at.h"

#include <string.h>

#define AT_KVST_LAT   40u    // AT §13.4
#define AT_FILL       350u   // AT §9.1-§9.2: dispatch, broadcast, pipeline, gather drain (*estimate*)
#define AT_PV_SEG_LAT 68u    // AT §13.4: ATTPV's segment delay (64 beats) and drain
#define VU_ASMAX_LAT  64u    // VU §7.9
#define VU_AMASK_LAT  45u
#define VU_PREPASS    64u    // VU §7.9: AMERGE / ANORM row-vector prepass latency

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline uint32_t ceil16(uint32_t n) { return (n + 15u) >> 4; }

static int rsv_bad(const uint32_t w[8], const uint32_t rsv[8]) {
    for (unsigned i = 0; i < TSR_INSN_WORDS; ++i) {
        if (w[i] & rsv[i]) return 1;
    }
    return 0;
}

// Spec range-check rule (docs/spec.md:1053): [a, a + e) must end at or below the limit.
static int spm_bad(const tsr_sim * sim, uint32_t a, uint64_t e, uint32_t * info) {
    if ((uint64_t)a + e <= tsr_sim_spm_lines(sim)) return 0;
    *info = a;
    return 1;
}
static int dram_bad(const tsr_sim * sim, uint32_t a, uint64_t e, uint32_t * info) {
    if ((uint64_t)a + e <= tsr_sim_dram_limit_beats(sim)) return 0;
    *info = a;
    return 1;
}
static inline void ext_spm(tsr_extents * x, uint8_t write, uint32_t a, uint64_t e) {
    tsr_ext_add(x, TSR_EXT_SPM, write, 0, a, (uint64_t)a + e);
}
static inline void ext_dram(tsr_extents * x, uint8_t write, uint32_t a, uint64_t e) {
    tsr_ext_add(x, TSR_EXT_DRAM, write, TSR_EXT_CH_ALL, a, (uint64_t)a + e);
}

static inline float * spm_f32(tsr_sim * sim, uint32_t line) {
    return (float *)(void *)(tsr_sim_spm(sim) + (size_t)line * TSR_LINE_BYTES);
}
static inline uint8_t * spm_bytes(tsr_sim * sim, uint32_t line) {
    return tsr_sim_spm(sim) + (size_t)line * TSR_LINE_BYTES;
}

// plane[c] = channel c byte 64·B (AT §4.2: the same beat range in every channel).
static void plane_ptrs(tsr_sim * sim, uint32_t B, uint8_t * plane[TSR_NUM_CH_MAX]) {
    const uint32_t C = tsr_sim_num_ch(sim);
    for (uint32_t c = 0; c < C; ++c) plane[c] = tsr_sim_dram(sim, c) + (uint64_t)B * TSR_BEAT_BYTES;
}

static inline uint32_t pcap_of(const tsr_sim * sim) {
    return tsr_qa_lines(TSR_QA_A256, tsr_sim_config(sim)->k_max) * 16u;   // AT §5.8, CFG3[31:16]·16
}

// ASMAX / AMERGE scale: finite, normal and > 0 (AT §5.4).
static inline int scale_bad(uint32_t s) {
    const uint32_t e = (s >> 23) & 0xFFu;
    return (s >> 31) != 0u || e == 0u || e == 0xFFu;
}

static void vu_lines(tsr_sim * sim, uint64_t n) {
    if (n != 0u) tsr_sim_perf_add(sim, TSR_PERF_VU_LINES(tsr_sim_num_ch(sim)), n);
}

// ---------------------------------------------------------------------------
// KVST 0x14 (AT §5.1)
// ---------------------------------------------------------------------------
static const uint32_t k_kvst_rsv[8] = { TSR_KVST_RSV_W0, TSR_KVST_RSV_W1, TSR_KVST_RSV_W2, TSR_KVST_RSV_W3,
                                        TSR_KVST_RSV_W4, TSR_KVST_RSV_W5, TSR_KVST_RSV_W6, TSR_KVST_RSV_W7 };

typedef struct kvst_dec {
    uint32_t kv, n_rows, src, B, E64, E, LG, c0, ss;
    uint32_t qa_lines;    // QA_LINES(A32, E)
} kvst_dec;

static void kvst_decode(const uint32_t w[8], kvst_dec * k) {
    k->kv       = TSR_INSN_GET(w, TSR_KVST_KV);
    k->n_rows   = TSR_INSN_GET(w, TSR_KVST_N_ROWS_M1) + 1u;
    k->src      = TSR_INSN_GET(w, TSR_KVST_SRC);
    k->B        = TSR_INSN_GET(w, TSR_KVST_BASE);
    k->E64      = TSR_INSN_GET(w, TSR_KVST_E64);
    k->E        = 64u * k->E64;
    k->LG       = TSR_INSN_GET(w, TSR_KVST_LG);
    k->c0       = TSR_INSN_GET(w, TSR_KVST_C0);
    k->ss       = TSR_INSN_GET(w, TSR_KVST_SRC_STRIDE);
    k->qa_lines = k->E ? tsr_qa_lines(TSR_QA_A32, k->E) : 0u;
}

static uint32_t chk_kvst(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_kvst_rsv)) return TSR_ERR_ILLEGAL_OP;
    kvst_dec k;
    kvst_decode(w, &k);
    if (k.kv > TSR_KVST_KV_V) return TSR_ERR_ILLEGAL_OP;
    if (k.E64 == 0u) return TSR_ERR_K_RANGE;
    const uint32_t C = tsr_sim_num_ch(sim);
    if (k.LG == 0u) { *info = 0u; return TSR_ERR_OPERAND; }
    if ((uint64_t)k.c0 + k.n_rows > 16ull * C * k.LG) { *info = k.c0; return TSR_ERR_OPERAND; }
    const uint64_t ssp = k.ss ? k.ss : k.qa_lines;
    const uint64_t es = (uint64_t)(k.n_rows - 1u) * ssp + k.qa_lines;
    const uint64_t ep = (uint64_t)k.LG * tsr_kv_group_beats(k.E);
    if (spm_bad(sim, k.src, es, info)) return TSR_ERR_SPM_RANGE;
    if (dram_bad(sim, k.B, ep, info)) return TSR_ERR_DRAM_RANGE;
    ext_spm(ext, TSR_EXT_READ, k.src, es);
    ext_dram(ext, TSR_EXT_WRITE, k.B, ep);
    return TSR_ERR_NONE;
}

// Capture of a KVST's beats (AT §4.3 / §4.4, the beat walk of tsr_kvst_count): phase 0 reports the
// partial meta beats it reads before writing (RMW), phase 1 every beat it wrote.
static void kvst_capture(tsr_sim * sim, const kvst_dec * k, int phase) {
    const uint32_t C = tsr_sim_num_ch(sim), nb = k->E / 32u, slices = k->E64, GB = tsr_kv_group_beats(k->E);
    const uint32_t end = k->c0 + k->n_rows;
    for (uint32_t cell = k->c0; cell < end;) {
        const uint32_t u = cell >> 4, ch = u % C, l = u / C;
        const uint32_t r_lo = cell & 15u, r_hi = (end - 16u * u) < 16u ? end - 16u * u : 16u;
        if (k->kv == TSR_KVST_KV_V) {
            for (uint32_t s = 0; s < slices; ++s) {
                const uint64_t chunk = (uint64_t)k->B + ((uint64_t)s * k->LG + l) * 17u;
                if (phase == 0) {
                    if (r_hi - r_lo < 16u) tsr_sim_cap_read(sim, ch, chunk, 1u);
                } else {
                    tsr_sim_cap_write(sim, ch, chunk, 1u);
                    tsr_sim_cap_write(sim, ch, chunk + 1u + r_lo, r_hi - r_lo);
                }
            }
        } else {
            const uint64_t b_lo = (uint64_t)r_lo * nb, b_hi = (uint64_t)r_hi * nb;    // beta range
            for (uint64_t kap = b_lo >> 5; kap <= (b_hi - 1u) >> 5; ++kap) {
                const uint64_t base = (uint64_t)k->B + (uint64_t)l * GB + 17u * kap;
                if (phase == 0) {
                    if (32u * kap < b_lo || 32u * kap + 32u > b_hi) tsr_sim_cap_read(sim, ch, base, 1u);
                } else {
                    const uint64_t o_lo = (b_lo > 32u * kap ? b_lo : 32u * kap) - 32u * kap;
                    const uint64_t o_hi = (b_hi < 32u * kap + 32u ? b_hi : 32u * kap + 32u) - 32u * kap;
                    tsr_sim_cap_write(sim, ch, base, 1u);
                    tsr_sim_cap_write(sim, ch, base + 1u + (o_lo >> 1), ((o_hi - 1u) >> 1) - (o_lo >> 1) + 1u);
                }
            }
        }
        cell = 16u * u + r_hi;
    }
}

static void ex_kvst(tsr_sim * sim, const uint32_t w[8]) {
    kvst_dec k;
    kvst_decode(w, &k);
    const uint32_t C = tsr_sim_num_ch(sim);
    const tsr_kv_geom g = { k.E, C, k.LG };
    uint8_t * plane[TSR_NUM_CH_MAX];
    plane_ptrs(sim, k.B, plane);
    kvst_capture(sim, &k, 0);
    tsr_kvst_stats st;
    tsr_ref_kvst(k.kv, &g, plane, k.c0, k.n_rows, spm_bytes(sim, k.src), k.ss, &st);
    kvst_capture(sim, &k, 1);
    uint64_t rmw = 0;
    for (uint32_t c = 0; c < C; ++c) {
        rmw += st.meta_rmw[c];
        if (st.meta_rmw[c]) tsr_sim_perf_add(sim, TSR_PERF_CH_RD_BEATS(c), st.meta_rmw[c]);
    }
    if (rmw) tsr_sim_perf_add(sim, TSR_PERF_KVST_RMW(C), rmw);
}

static uint64_t cyc_kvst(const tsr_sim * sim, const uint32_t w[8]) {
    kvst_dec k;
    kvst_decode(w, &k);
    const uint32_t C = tsr_sim_num_ch(sim);
    const tsr_kv_geom g = { k.E, C, k.LG };
    tsr_kvst_stats st;
    tsr_kvst_count(k.kv, &g, k.c0, k.n_rows, &st);
    uint64_t n = AT_KVST_LAT;
    for (uint32_t c = 0; c < C; ++c) n += (uint64_t)st.meta_beats[c] + st.quant_beats[c] + st.meta_rmw[c];
    return n;
}

// ---------------------------------------------------------------------------
// ATTQK 0x21 / ATTPV 0x22 (AT §5.2, §5.3): shared shape decode
// ---------------------------------------------------------------------------
static const uint32_t k_attqk_rsv[8] = { TSR_ATTQK_RSV_W0, TSR_ATTQK_RSV_W1, TSR_ATTQK_RSV_W2, TSR_ATTQK_RSV_W3,
                                         TSR_ATTQK_RSV_W4, TSR_ATTQK_RSV_W5, TSR_ATTQK_RSV_W6, TSR_ATTQK_RSV_W7 };
static const uint32_t k_attpv_rsv[8] = { TSR_ATTPV_RSV_W0, TSR_ATTPV_RSV_W1, TSR_ATTPV_RSV_W2, TSR_ATTPV_RSV_W3,
                                         TSR_ATTPV_RSV_W4, TSR_ATTPV_RSV_W5, TSR_ATTPV_RSV_W6, TSR_ATTPV_RSV_W7 };

typedef struct att_dec {
    uint32_t fmt, flags;              // flags: w0[15:12] >> 12 (TSR_ATT_*)
    tsr_att_shape sh;                 // E, D, G, hg0, GS, T_q
    uint32_t E64, rs;                 // rs: ATTQK rstride / ATTPV pstride field
    uint32_t dst, x, B, u0, n, w6, w7;
    uint32_t n_head, R, pass_rows, max_row;
} att_dec;

// The two ops share w0's x16, w4's shape, w5's group range; field constants are ATTQK's and ATTPV's
// own (identical positions, AT §5.2-§5.3).
static void att_decode(const uint32_t w[8], att_dec * a) {
    a->fmt     = TSR_INSN_GET(w, TSR_ATTQK_FMT);
    a->flags   = (w[0] >> 12) & 0xFu;
    a->E64     = TSR_INSN_GET(w, TSR_ATTQK_E64);
    a->sh.E    = 64u * a->E64;
    a->sh.D    = 64u << TSR_INSN_GET(w, TSR_ATTQK_DCODE);
    a->sh.G    = TSR_INSN_GET(w, TSR_ATTQK_G_M1) + 1u;
    a->sh.hg0  = TSR_INSN_GET(w, TSR_ATTQK_HG0);
    a->sh.GS   = TSR_INSN_GET(w, TSR_ATTQK_GS_M1) + 1u;
    a->sh.T_q  = TSR_INSN_GET(w, TSR_ATTQK_TQ_M1) + 1u;
    a->rs      = TSR_INSN_GET(w, TSR_ATTQK_RSTRIDE);
    a->dst     = w[1];
    a->x       = w[2];
    a->B       = w[3];
    a->u0      = TSR_INSN_GET(w, TSR_ATTQK_U0);
    a->n       = TSR_INSN_GET(w, TSR_ATTQK_N_GROUPS);
    a->w6      = w[6];
    a->w7      = w[7];
    const uint32_t nseg = a->sh.E / a->sh.D;                 // 0 when E < D (OPERAND before use)
    a->n_head    = a->sh.G * nseg;
    a->R         = a->sh.T_q * a->n_head;
    a->pass_rows = nseg * a->sh.GS * a->sh.T_q;
    // r(T_q - 1, E/D - 1, GS - 1): the last pass row (only meaningful when nseg >= 1).
    a->max_row = (a->sh.T_q - 1u) * a->n_head + (nseg ? nseg - 1u : 0u) * a->sh.G + a->sh.hg0 + a->sh.GS - 1u;
}

// Checks shared up to the first op-specific OPERAND condition (AT §5.8: K_RANGE, T_RANGE, then E,
// hg0 + GS, n_groups). Returns 0 or the code.
static uint32_t att_chk_common(const tsr_sim * sim, const att_dec * a, uint32_t * info) {
    const tsr_config * cfg = tsr_sim_config(sim);
    if (a->fmt != TSR_KVFMT_Q8_0) return TSR_ERR_BAD_FMT;
    if (a->E64 == 0u || (uint64_t)a->sh.G * a->sh.E > cfg->k_max) return TSR_ERR_K_RANGE;
    if (a->sh.T_q * a->sh.GS > cfg->t_max) return TSR_ERR_T_RANGE;
    if (a->sh.E % a->sh.D != 0u) { *info = a->sh.E; return TSR_ERR_OPERAND; }
    if (a->sh.hg0 + a->sh.GS > a->sh.G) { *info = a->sh.hg0 + a->sh.GS; return TSR_ERR_OPERAND; }
    if (a->n == 0u || a->n > TSR_ATT_MAX_GROUPS) { *info = a->n; return TSR_ERR_OPERAND; }
    return TSR_ERR_NONE;
}

// Groups of [u0, u0 + n) on channel c and the local index of the first one.
static uint32_t att_ch_first_l(uint32_t C, uint32_t u0, uint32_t c) { return (u0 + (c + C - u0 % C) % C) / C; }

static uint64_t att_max_beats(uint32_t C, const att_dec * a) {
    uint64_t mx = 0;
    for (uint32_t c = 0; c < C; ++c) {
        const uint64_t b = (uint64_t)tsr_att_ch_groups(C, a->u0, a->n, c) * tsr_kv_group_beats(a->sh.E);
        if (b > mx) mx = b;
    }
    return mx;
}

// Stream accounting of an ATTQK / ATTPV: per channel the beats streamed (CH_RD_BEATS, capture of
// the K groups or of the V slices) and the gather / broadcast-bound share (ATT_PATH_STALL).
static void att_stream(tsr_sim * sim, const att_dec * a, int is_v, uint32_t LG) {
    const uint32_t C = tsr_sim_num_ch(sim), GB = tsr_kv_group_beats(a->sh.E);
    for (uint32_t c = 0; c < C; ++c) {
        const uint32_t ng = tsr_att_ch_groups(C, a->u0, a->n, c);
        if (ng == 0u) continue;
        const uint64_t l0 = att_ch_first_l(C, a->u0, c);
        if (!is_v) {
            tsr_sim_cap_read(sim, c, (uint64_t)a->B + l0 * GB, (uint64_t)ng * GB);
        } else {
            for (uint32_t s = 0; s < a->E64; ++s)                // slice s: [B + (s·LG + l)·17, ...)
                tsr_sim_cap_read(sim, c, (uint64_t)a->B + ((uint64_t)s * LG + l0) * 17u, (uint64_t)ng * 17u);
        }
        tsr_sim_perf_add(sim, TSR_PERF_CH_RD_BEATS(c), (uint64_t)ng * GB);
    }
    const uint64_t lines = (uint64_t)a->pass_rows * a->n, beats = att_max_beats(C, a);
    if (lines > beats) tsr_sim_perf_add(sim, TSR_PERF_ATT_PATH_STALL(C), lines - beats);
}

static uint64_t att_cycles(const tsr_sim * sim, const att_dec * a) {
    const uint64_t lines = (uint64_t)a->pass_rows * a->n, beats = att_max_beats(tsr_sim_num_ch(sim), a);
    return AT_FILL + (lines > beats ? lines : beats);
}

// ---------------------------------------------------------------------------
// ATTQK 0x21 (AT §5.2, §6.2, §6.3)
// ---------------------------------------------------------------------------
static uint32_t chk_attqk(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_attqk_rsv)) return TSR_ERR_ILLEGAL_OP;
    att_dec a;
    att_decode(w, &a);
    const int maxout = (a.flags & TSR_ATT_MAXOUT) != 0;
    if (!maxout && a.w7 != 0u) return TSR_ERR_ILLEGAL_OP;
    uint32_t rc = att_chk_common(sim, &a, info);
    if (rc) return rc;
    const uint32_t c_lo = TSR_INSN_GET(w, TSR_ATTQK_C_LO), c_hi = TSR_INSN_GET(w, TSR_ATTQK_C_HI);
    if (c_lo > c_hi || c_hi > 16u * a.n) { *info = c_hi; return TSR_ERR_OPERAND; }
    if (a.pass_rows > TSR_ATT_MAX_PASS_ROWS) { *info = a.pass_rows; return TSR_ERR_OPERAND; }
    if (a.rs != 0u && a.rs < a.n) { *info = a.rs; return TSR_ERR_OPERAND; }
    const uint64_t rsp = a.rs ? a.rs : a.n;
    const uint64_t ed = (uint64_t)a.max_row * rsp + a.n;
    const uint64_t ex = (uint64_t)a.sh.T_q * tsr_qa_lines(TSR_QA_A32, a.sh.G * a.sh.E);
    const uint64_t em = maxout ? ceil16(a.max_row + 1u) : 0u;
    const uint32_t C = tsr_sim_num_ch(sim);
    const uint64_t ep = (uint64_t)((a.u0 + a.n - 1u) / C + 1u) * tsr_kv_group_beats(a.sh.E);
    if (spm_bad(sim, a.dst, ed, info)) return TSR_ERR_SPM_RANGE;
    if (spm_bad(sim, a.x, ex, info)) return TSR_ERR_SPM_RANGE;
    if (maxout && spm_bad(sim, a.w7, em, info)) return TSR_ERR_SPM_RANGE;
    if (dram_bad(sim, a.B, ep, info)) return TSR_ERR_DRAM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, a.dst, ed);
    ext_spm(ext, TSR_EXT_READ, a.x, ex);
    ext_dram(ext, TSR_EXT_READ, a.B, ep);
    if (maxout) ext_spm(ext, TSR_EXT_WRITE, a.w7, em);
    return TSR_ERR_NONE;
}

static void ex_attqk(tsr_sim * sim, const uint32_t w[8]) {
    att_dec a;
    att_decode(w, &a);
    uint8_t * plane[TSR_NUM_CH_MAX];
    plane_ptrs(sim, a.B, plane);
    att_stream(sim, &a, 0, 0);
    const int maxout = (a.flags & TSR_ATT_MAXOUT) != 0;
    tsr_ref_attqk(&a.sh, tsr_sim_num_ch(sim), (const uint8_t * const *)plane, a.u0, a.n,
                  TSR_INSN_GET(w, TSR_ATTQK_C_LO), TSR_INSN_GET(w, TSR_ATTQK_C_HI), a.flags,
                  spm_bytes(sim, a.x), spm_f32(sim, a.dst), a.rs, maxout ? spm_f32(sim, a.w7) : NULL);
}

static uint64_t cyc_attqk(const tsr_sim * sim, const uint32_t w[8]) {
    att_dec a;
    att_decode(w, &a);
    return att_cycles(sim, &a);
}

// ---------------------------------------------------------------------------
// ATTPV 0x22 (AT §5.3, §6.4, §6.5)
// ---------------------------------------------------------------------------
static uint32_t chk_attpv(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_attpv_rsv)) return TSR_ERR_ILLEGAL_OP;
    att_dec a;
    att_decode(w, &a);
    uint32_t rc = att_chk_common(sim, &a, info);
    if (rc) return rc;
    const uint32_t C = tsr_sim_num_ch(sim), LG = TSR_INSN_GET(w, TSR_ATTPV_LG);
    if (a.pass_rows > TSR_ATT_MAX_PASS_ROWS) { *info = a.pass_rows; return TSR_ERR_OPERAND; }
    const uint32_t l_max = (a.u0 + a.n - 1u) / C;
    if (l_max >= LG) { *info = a.u0 + a.n - 1u; return TSR_ERR_OPERAND; }
    if ((uint64_t)(a.sh.E / a.sh.D) * 16u * ((a.n + C - 1u) / C) > pcap_of(sim)) { *info = a.n; return TSR_ERR_OPERAND; }
    if (a.rs != 0u && a.rs < a.n) { *info = a.rs; return TSR_ERR_OPERAND; }
    const int lsum = (a.flags & TSR_ATT_LSUM) != 0;
    const uint64_t rl = (uint64_t)a.R * a.sh.D / 16u, cs = a.w7 ? a.w7 : rl;
    uint64_t ed = (uint64_t)(C - 1u) * cs + rl;
    if (lsum && (uint64_t)C * cs + (uint64_t)C * ceil16(a.R) > ed) ed = (uint64_t)C * cs + (uint64_t)C * ceil16(a.R);
    const uint64_t ep = (uint64_t)a.max_row * (a.rs ? a.rs : a.n) + a.n;
    const uint64_t ev = ((uint64_t)(a.E64 - 1u) * LG + l_max + 1u) * 17u;
    if (spm_bad(sim, a.dst, ed, info)) return TSR_ERR_SPM_RANGE;
    if (spm_bad(sim, a.x, ep, info)) return TSR_ERR_SPM_RANGE;
    if (dram_bad(sim, a.B, ev, info)) return TSR_ERR_DRAM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, a.dst, ed);
    ext_spm(ext, TSR_EXT_READ, a.x, ep);
    ext_dram(ext, TSR_EXT_READ, a.B, ev);
    return TSR_ERR_NONE;
}

static void ex_attpv(tsr_sim * sim, const uint32_t w[8]) {
    att_dec a;
    att_decode(w, &a);
    const uint32_t LG = TSR_INSN_GET(w, TSR_ATTPV_LG);
    uint8_t * plane[TSR_NUM_CH_MAX];
    plane_ptrs(sim, a.B, plane);
    att_stream(sim, &a, 1, LG);
    tsr_ref_attpv(&a.sh, tsr_sim_num_ch(sim), LG, (const uint8_t * const *)plane, a.u0, a.n, a.flags,
                  spm_f32(sim, a.x), a.rs, spm_f32(sim, a.dst), a.w7);
}

static uint64_t cyc_attpv(const tsr_sim * sim, const uint32_t w[8]) {
    att_dec a;
    att_decode(w, &a);
    return att_cycles(sim, &a) + AT_PV_SEG_LAT;
}

// ---------------------------------------------------------------------------
// ASMAX 0x34 (AT §5.4)
// ---------------------------------------------------------------------------
static const uint32_t k_asmax_rsv[8] = { TSR_ASMAX_RSV_W0, TSR_ASMAX_RSV_W1, TSR_ASMAX_RSV_W2, TSR_ASMAX_RSV_W3,
                                         TSR_ASMAX_RSV_W4, TSR_ASMAX_RSV_W5, TSR_ASMAX_RSV_W6, TSR_ASMAX_RSV_W7 };

static uint32_t chk_asmax(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_asmax_rsv)) return TSR_ERR_ILLEGAL_OP;
    const uint32_t nl = TSR_INSN_GET(w, TSR_ASMAX_N_LINES), st = TSR_INSN_GET(w, TSR_ASMAX_STRIDE);
    const uint32_t R = TSR_INSN_GET(w, TSR_ASMAX_R), sc = TSR_INSN_GET(w, TSR_ASMAX_SCALE);
    if (nl == 0u || nl > TSR_ATT_MAX_LINES) { *info = nl; return TSR_ERR_OPERAND; }
    if (R == 0u || R > TSR_ATT_MAX_R) { *info = R; return TSR_ERR_OPERAND; }
    if (scale_bad(sc)) { *info = sc; return TSR_ERR_OPERAND; }
    const uint64_t e = (uint64_t)(R - 1u) * (st ? st : nl) + nl, em = ceil16(R);
    if (spm_bad(sim, w[1], e, info) || spm_bad(sim, w[2], e, info) || spm_bad(sim, w[3], em, info))
        return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, w[1], e);
    ext_spm(ext, TSR_EXT_READ, w[2], e);
    ext_spm(ext, TSR_EXT_READ, w[3], em);
    return TSR_ERR_NONE;
}

static void ex_asmax(tsr_sim * sim, const uint32_t w[8]) {
    const uint32_t nl = TSR_INSN_GET(w, TSR_ASMAX_N_LINES), R = TSR_INSN_GET(w, TSR_ASMAX_R);
    tsr_ref_asmax(spm_f32(sim, w[1]), spm_f32(sim, w[2]), spm_f32(sim, w[3]), nl,
                  TSR_INSN_GET(w, TSR_ASMAX_STRIDE), R, tsr_bits_f32(TSR_INSN_GET(w, TSR_ASMAX_SCALE)));
    vu_lines(sim, (uint64_t)R * nl);
}

static uint64_t cyc_asmax(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    return (uint64_t)TSR_INSN_GET(w, TSR_ASMAX_R) * TSR_INSN_GET(w, TSR_ASMAX_N_LINES) + VU_ASMAX_LAT;
}

// ---------------------------------------------------------------------------
// AMASK 0x35 (AT §5.5)
// ---------------------------------------------------------------------------
static const uint32_t k_amask_rsv[8] = { TSR_AMASK_RSV_W0, TSR_AMASK_RSV_W1, TSR_AMASK_RSV_W2, TSR_AMASK_RSV_W3,
                                         TSR_AMASK_RSV_W4, TSR_AMASK_RSV_W5, TSR_AMASK_RSV_W6, TSR_AMASK_RSV_W7 };

static uint32_t chk_amask(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_amask_rsv)) return TSR_ERR_ILLEGAL_OP;
    const uint32_t nl = TSR_INSN_GET(w, TSR_AMASK_N_LINES), st = TSR_INSN_GET(w, TSR_AMASK_STRIDE);
    const uint32_t R = TSR_INSN_GET(w, TSR_AMASK_R), rpq = TSR_INSN_GET(w, TSR_AMASK_RPQ);
    const uint32_t ms = TSR_INSN_GET(w, TSR_AMASK_MSTRIDE);
    if (nl == 0u || nl > TSR_ATT_MAX_LINES) { *info = nl; return TSR_ERR_OPERAND; }
    if (R == 0u || R > TSR_ATT_MAX_R) { *info = R; return TSR_ERR_OPERAND; }
    if (rpq == 0u) { *info = 0u; return TSR_ERR_OPERAND; }
    const uint64_t e = (uint64_t)(R - 1u) * (st ? st : nl) + nl;
    const uint64_t emask = (uint64_t)((R + rpq - 1u) / rpq - 1u) * ms + nl, em = ceil16(R);
    if (spm_bad(sim, w[1], e, info) || spm_bad(sim, w[2], e, info) || spm_bad(sim, w[3], emask, info) ||
        spm_bad(sim, w[7], em, info))
        return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, w[1], e);
    ext_spm(ext, TSR_EXT_READ, w[2], e);
    ext_spm(ext, TSR_EXT_READ, w[3], emask);
    ext_spm(ext, TSR_EXT_WRITE, w[7], em);
    return TSR_ERR_NONE;
}

static void ex_amask(tsr_sim * sim, const uint32_t w[8]) {
    const uint32_t nl = TSR_INSN_GET(w, TSR_AMASK_N_LINES), R = TSR_INSN_GET(w, TSR_AMASK_R);
    tsr_ref_amask(spm_f32(sim, w[1]), spm_f32(sim, w[2]), spm_f32(sim, w[3]), nl, TSR_INSN_GET(w, TSR_AMASK_STRIDE),
                  R, TSR_INSN_GET(w, TSR_AMASK_RPQ), TSR_INSN_GET(w, TSR_AMASK_MSTRIDE), spm_f32(sim, w[7]));
    vu_lines(sim, (uint64_t)R * nl + ceil16(R));
}

static uint64_t cyc_amask(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    const uint64_t nl = TSR_INSN_GET(w, TSR_AMASK_N_LINES), R = TSR_INSN_GET(w, TSR_AMASK_R);
    return R * nl + ((nl & 1u) ? R : 0u) + VU_AMASK_LAT;
}

// ---------------------------------------------------------------------------
// AMERGE 0x36 (AT §5.6)
// ---------------------------------------------------------------------------
static const uint32_t k_amerge_rsv[8] = { TSR_AMERGE_RSV_W0, TSR_AMERGE_RSV_W1, TSR_AMERGE_RSV_W2, TSR_AMERGE_RSV_W3,
                                          TSR_AMERGE_RSV_W4, TSR_AMERGE_RSV_W5, TSR_AMERGE_RSV_W6, TSR_AMERGE_RSV_W7 };

typedef struct amerge_dec {
    uint32_t init, orun, parts, mk, st, R, D, scale, cs;
} amerge_dec;

static void amerge_decode(const uint32_t w[8], amerge_dec * m) {
    m->init  = TSR_INSN_GET(w, TSR_AMERGE_INIT);
    m->orun  = TSR_INSN_GET(w, TSR_AMERGE_ORUN);
    m->parts = TSR_INSN_GET(w, TSR_AMERGE_PARTS);
    m->mk    = TSR_INSN_GET(w, TSR_AMERGE_MK);
    m->st    = TSR_INSN_GET(w, TSR_AMERGE_ST);
    m->R     = TSR_INSN_GET(w, TSR_AMERGE_R);
    m->D     = 64u << TSR_INSN_GET(w, TSR_AMERGE_DCODE);
    m->scale = TSR_INSN_GET(w, TSR_AMERGE_SCALE);
    m->cs    = TSR_INSN_GET(w, TSR_AMERGE_CSTRIDE);
}

static uint32_t chk_amerge(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_amerge_rsv)) return TSR_ERR_ILLEGAL_OP;
    amerge_dec m;
    amerge_decode(w, &m);
    if (m.R == 0u || m.R > TSR_ATT_MAX_R) { *info = m.R; return TSR_ERR_OPERAND; }
    if (scale_bad(m.scale)) { *info = m.scale; return TSR_ERR_OPERAND; }
    const uint32_t C = tsr_sim_num_ch(sim);
    const uint64_t rl = (uint64_t)m.R * m.D / 16u, cs = m.cs ? m.cs : rl;
    const uint64_t eparts = (uint64_t)C * cs + (uint64_t)C * ceil16(m.R), emk = ceil16(m.R), est = 2u * emk;
    if (spm_bad(sim, m.orun, rl, info) || spm_bad(sim, m.parts, eparts, info) || spm_bad(sim, m.mk, emk, info) ||
        spm_bad(sim, m.st, est, info))
        return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, m.orun, rl);
    if (!m.init) ext_spm(ext, TSR_EXT_READ, m.orun, rl);
    ext_spm(ext, TSR_EXT_READ, m.parts, eparts);
    ext_spm(ext, TSR_EXT_READ, m.mk, emk);
    ext_spm(ext, TSR_EXT_WRITE, m.st, est);
    if (!m.init) ext_spm(ext, TSR_EXT_READ, m.st, est);
    return TSR_ERR_NONE;
}

static void ex_amerge(tsr_sim * sim, const uint32_t w[8]) {
    amerge_dec m;
    amerge_decode(w, &m);
    tsr_ref_amerge(spm_f32(sim, m.orun), spm_f32(sim, m.parts), spm_f32(sim, m.mk), spm_f32(sim, m.st), m.R, m.D,
                   tsr_bits_f32(m.scale), tsr_sim_num_ch(sim), m.cs, m.init ? TSR_ATT_INIT : 0u);
    vu_lines(sim, (uint64_t)m.R * m.D / 16u + 2u * ceil16(m.R));
}

static uint64_t cyc_amerge(const tsr_sim * sim, const uint32_t w[8]) {
    amerge_dec m;
    amerge_decode(w, &m);
    const uint64_t C = tsr_sim_num_ch(sim);
    const uint64_t per_line = m.init ? (C > 1u ? C - 1u : 1u) : C + 1u;
    return 6u * (uint64_t)ceil16(m.R) + VU_PREPASS + (uint64_t)m.R * m.D / 16u * per_line;
}

// ---------------------------------------------------------------------------
// ANORM 0x37 (AT §5.7)
// ---------------------------------------------------------------------------
static const uint32_t k_anorm_rsv[8] = { TSR_ANORM_RSV_W0, TSR_ANORM_RSV_W1, TSR_ANORM_RSV_W2, TSR_ANORM_RSV_W3,
                                         TSR_ANORM_RSV_W4, TSR_ANORM_RSV_W5, TSR_ANORM_RSV_W6, TSR_ANORM_RSV_W7 };

static uint32_t chk_anorm(tsr_sim * sim, const uint32_t w[8], tsr_extents * ext, uint32_t * info) {
    if (rsv_bad(w, k_anorm_rsv)) return TSR_ERR_ILLEGAL_OP;
    const uint32_t R = TSR_INSN_GET(w, TSR_ANORM_R), D = 64u << TSR_INSN_GET(w, TSR_ANORM_DCODE);
    if (R == 0u || R > TSR_ATT_MAX_R) { *info = R; return TSR_ERR_OPERAND; }
    const uint64_t rl = (uint64_t)R * D / 16u, el = ceil16(R);
    if (spm_bad(sim, w[1], rl, info) || spm_bad(sim, w[2], rl, info) || spm_bad(sim, w[3], el, info))
        return TSR_ERR_SPM_RANGE;
    ext_spm(ext, TSR_EXT_WRITE, w[1], rl);
    ext_spm(ext, TSR_EXT_READ, w[2], rl);
    ext_spm(ext, TSR_EXT_READ, w[3], el);
    return TSR_ERR_NONE;
}

static void ex_anorm(tsr_sim * sim, const uint32_t w[8]) {
    const uint32_t R = TSR_INSN_GET(w, TSR_ANORM_R), D = 64u << TSR_INSN_GET(w, TSR_ANORM_DCODE);
    tsr_ref_anorm(spm_f32(sim, w[1]), spm_f32(sim, w[2]), spm_f32(sim, w[3]), R, D);
    vu_lines(sim, (uint64_t)R * D / 16u);
}

static uint64_t cyc_anorm(const tsr_sim * sim, const uint32_t w[8]) {
    (void)sim;
    const uint64_t R = TSR_INSN_GET(w, TSR_ANORM_R), D = 64u << TSR_INSN_GET(w, TSR_ANORM_DCODE);
    return 4u * ((R + 15u) / 16u) + VU_PREPASS + R * D / 16u;
}

// ---------------------------------------------------------------------------
// Registration (tsr_sim_int.h §2)
// ---------------------------------------------------------------------------
void tsr_at_register(void) {
    const tsr_op_handler kvst   = { TSR_UNIT_DMA, chk_kvst, ex_kvst, cyc_kvst };
    const tsr_op_handler attqk  = { TSR_UNIT_MVM, chk_attqk, ex_attqk, cyc_attqk };
    const tsr_op_handler attpv  = { TSR_UNIT_MVM, chk_attpv, ex_attpv, cyc_attpv };
    const tsr_op_handler asmax  = { TSR_UNIT_VU, chk_asmax, ex_asmax, cyc_asmax };
    const tsr_op_handler amask  = { TSR_UNIT_VU, chk_amask, ex_amask, cyc_amask };
    const tsr_op_handler amerge = { TSR_UNIT_VU, chk_amerge, ex_amerge, cyc_amerge };
    const tsr_op_handler anorm  = { TSR_UNIT_VU, chk_anorm, ex_anorm, cyc_anorm };
    tsr_sim_register_op(TSR_OP_KVST, &kvst);
    tsr_sim_register_op(TSR_OP_ATTQK, &attqk);
    tsr_sim_register_op(TSR_OP_ATTPV, &attpv);
    tsr_sim_register_op(TSR_OP_ASMAX, &asmax);
    tsr_sim_register_op(TSR_OP_AMASK, &amask);
    tsr_sim_register_op(TSR_OP_AMERGE, &amerge);
    tsr_sim_register_op(TSR_OP_ANORM, &anorm);
}

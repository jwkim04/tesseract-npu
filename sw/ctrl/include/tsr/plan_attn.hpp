// tsr/plan_attn.hpp — planners of SET_ROWS_KV (0x18) and FLASH_ATTN (0x19), and the FA chunk planner
// shared with tsr_ref_fa (docs/design/06-software-verification.md "SW" §7.2, §7.5; docs/design/03-attention-
// kv.md "AT" §4, §5, §6.6, §7, §12; W2b P2.7). Freestanding, C++17-compatible, constexpr-friendly:
// the firmware's kind handlers (sw/ctrl/kernels/k_kv.cpp, k_fattn.cpp) and the host's supports_op
// (W3b) call the same functions (SW-20).
//
// The planner contract and the declarations are tsr/plan.hpp's ("M7 planning"), which includes this
// file after them; including this file first works too (it includes plan.hpp). Owner: W2b kvfa.
//
// FA chunk planner. The per-tile chunk list (AT §6.6 steps 2-3, §7.3; W1a's rules 1-8) is
// sw/sim/include/tsr_fa_plan.h, ONE freestanding C / C++ source that tsr_ref_at.c (tsr_fa_plan, hence
// tsr_ref_fa) also compiles: the firmware's tile-op sequence and the reference's are the same by
// construction (sw/ctrl/tests/test_fw_kvfa_plan.cpp compares it with W1a's code). It is included
// here by a path relative to this file (the firmware's include path is sw/ctrl/include only); its
// pass rule and NCG are constexpr in C++.
//
// SET_ROWS_KV (SW §7.2 0x18, §7.5; AT §5.1, §7.2). Per run of consecutive cells (indices from the
// preload area, I64, the host checks idx < 2^31): LD the source rows (f32, E values = E/16 lines
// each), QACT A32 (K = E, at most T_MAX rows per QACT), KVST of the run (at most 64 rows per KVST)
// into the K or V plane. A tile of R rows takes one ring block [rows R·E/16 | QA R·QA_LINES(A32, E)].
// Argument rules (BAD_RECORD): n_args 12; which 0 (K) / 1 (V); plane kind 0 (the KV format: q8_0,
// AT §5.2 sub; the only one); GEOM E/64 in [1, 511] (the KVST field, 9 bits) and LG >= 1; row
// elements = E; E <= K_MAX (QACT); a device with the AT ops (caps ATT, num_ch 1..16, T_MAX >= 1).
// Words 0 / 1 (the ggml cache tensor's handle and row stride) and 11 (stream) identify the cache for
// the host and the debugger; the plane base (word 8) is where the rows go, so the kernel does not
// use them. Tile rows: min(n_rows, 64), fewer only when two tiles do not fit the budget (as
// plan_rows); RET_SCRATCH when one row does not fit. The cells themselves are data (the preload
// area): the kernel checks each index against the plane (hi word 0, cell < 16·C·LG; else
// BAD_RECORD) and splits runs at the tile size.
// kind_plan: row_lines = E/16, tile_rows, n_tiles = ceil(n_rows / tile_rows) (one run; more runs give
// more tiles), scratch_lines = tile_rows · lpr, aux[0] = QA_LINES(A32, E), aux[1] = E, aux[2] = LG,
// aux[3] = rows per QACT (min(T_MAX, 8)).
//
// FLASH_ATTN (SW §7.2 0x19, §7.5; AT §6.6, §7). The sequence of tsr_ref_fa (k_fattn.cpp's header).
// Argument rules (BAD_RECORD): n_args 22; softcap +0.0; D (word 19) in {64, 128, 256, 512} and
// head_dim (word 8) = D; E = 64·GEOM E/64 with E/64 in [1, 511], E % D = 0 and n_head_kv = E / D;
// G in [1, 64] and n_head = G·n_head_kv; LG >= 1; n_kv <= 16·C·LG and <= 2^20 (the ATTQK / ATTPV
// group fields); n_tokens >= 1; scale finite, normal, > 0 (ASMAX / AMERGE); G·E <= K_MAX (QACT of the
// Q rows, ATTQK); q head stride = D/16 (each token's heads contiguous: one QA vector per token, AT
// §2.6); dst token stride >= n_head·D/16 when n_tokens > 1; the flags word below; the pass shape of
// AT §6.6 step 2 (GS = min(G, T_MAX), T_q = floor(T_MAX / GS)): pass rows (E/D)·GS·T_q <= 128 and
// R = T_q·n_head <= 1024; NCG > 0 (PCAP >= 16·E/D); a device with the AT ops. Words 5 / 6 (the ggml
// cache tensors' handles) identify the caches for the host; the kernel uses the plane bases (16, 17).
// The MSUM (word 12) and the new-cell indices are data: the kernel checks their preload ranges, the
// MSUM header (magic, n_q = n_tokens, n_kv = word 7) and, per tile, the planner's verdict; a tile
// that needs the GENERAL path needs the f32 mask copy (word 13 != 0xFFFF_FFFF), else BAD_RECORD.
// Scratch: one ring block per op, all tiles (tiles run one after another in it), laid out by
// fa_layout below for T_q tokens, R = T_q·n_head rows and NCG_MAX = min(128, C·floor(PCAP/(16·E/D)))
// groups per chunk (the largest chunk any tile can get): Q rows / output (T_q·K/16, K = n_head·D: the
// LD target of Q, read by QACT, then ANORM's output, read by the ST), QA (T_q·QA_LINES(A32, K)), Orun
// (R·D/16), running state (2·ceil(R/16)), MK[4] (chunk row maxima, ceil(R/16) each), S[3] (scores /
// p, R·NCG_MAX each), P[2] (ATTPV partials with LSUM, C·R·D/16 + C·ceil(R/16) each), M[2] (the GENERAL
// mask rows of a chunk, T_q·NCG_MAX each). RET_SCRATCH when the block exceeds the budget.
// kind_plan: row_lines = K/16, tile_rows = T_q, n_tiles = ceil(n_tokens / T_q), scratch_lines = the
// block, aux[0] = GS, aux[1] = NCG_MAX, aux[2] = R, aux[3] = E.
//
// FLASH_ATTN flags (word 21; AT §12 names the word without defining it: W2b decision). Rule 7 of
// the chunk planner ("newest last", AT §6.6 step 3) needs the cells written in this launch; they are
// the indices of the layer's SET_ROWS_KV records (the same preload entry), so the word refers to them:
//   [0] FA_NEW      the cells written in this launch are the I64 indices below (else: none)
//   [7:1]           0
//   [15:8]          the preload line of the indices (byte offset / 64; the writer 64-aligns entries)
//   [31:16]         their count
// Without FA_NEW the whole word is 0. A chunk then holds a new cell iff one of the indices lies in
// its groups (tsr_ref_fa's new_runs = the indices' runs of consecutive cells: the same predicate).
#pragma once

#include <stdint.h>

#include "tsr/plan.hpp"
#include "../../../sim/include/tsr_fa_plan.h"   // the FA chunk planner (shared with tsr_ref_at.c)

namespace tsr {
namespace plan {
namespace attn {

namespace sr = opl::setrows;
namespace fa = opl::fa;

// Encoding limits of the AT ops (AT §5; tsr_isa.h TSR_ATT_*, TSR_KVST_MAX_ROWS).
constexpr uint32_t E64_MAX = 511u;             // KVST / ATTQK / ATTPV E/64: 9 bits
constexpr uint32_t KVST_ROWS_MAX = 64u;        // KVST n_rows
constexpr uint32_t ATT_G_MAX = 64u;            // x16[5:0] G - 1
constexpr uint32_t ATT_PASS_ROWS_MAX = 128u;   // ATTQK / ATTPV pass rows
constexpr uint32_t ATT_R_MAX = 1024u;          // ASMAX / AMASK / AMERGE / ANORM R
constexpr uint32_t ATT_NCG_MAX = 128u;         // AT §6.6: NCG <= 128 groups
constexpr uint32_t ATT_GROUPS_MAX = 65536u;    // u0 is 16 bits: n_kv <= 16·65,536 cells
constexpr uint32_t KV_FMT_Q8_0 = 0u;           // ATTQK / ATTPV sub, SET_ROWS_KV plane kind
constexpr uint32_t NUM_CH_MAX = 16u;

// FLASH_ATTN flags (header comment).
constexpr uint32_t FA_NEW = 1u << 0;
constexpr uint32_t FA_RSVD = 0xFEu;
constexpr uint32_t FA_NEW_LINE_LSB = 8u, FA_N_NEW_LSB = 16u;
constexpr uint32_t fa_flags(uint32_t idx_pre_byte, uint32_t n_idx) {
    return FA_NEW | ((idx_pre_byte >> 6) & 0xFFu) << FA_NEW_LINE_LSB | (n_idx & 0xFFFFu) << FA_N_NEW_LSB;
}
constexpr uint32_t fa_new_pre_byte(uint32_t flags) { return ((flags >> FA_NEW_LINE_LSB) & 0xFFu) << 6; }
constexpr uint32_t fa_n_new(uint32_t flags) { return flags >> FA_N_NEW_LSB; }

// f32 scale bits that ASMAX / AMERGE accept: finite, normal, > 0.
TSR_PLAN_HOT constexpr bool scale_ok(uint32_t s) {
    const uint32_t e = (s >> 23) & 0xFFu;
    return (s >> 31) == 0u && e != 0u && e != 0xFFu;
}
// D = 64·2^dcode (AT §5): the dcode of a valid D, else 4.
TSR_PLAN_HOT constexpr uint32_t dcode_of(uint32_t D) {
    return D == 64u ? 0u : D == 128u ? 1u : D == 256u ? 2u : D == 512u ? 3u : 4u;
}
// A device with the AT ops (caps ATT; C, PCAP and T_MAX known).
TSR_PLAN_HOT constexpr bool att_device(const plan_limits & l) {
    return (l.caps & CAPS_ATT) != 0u && l.num_ch - 1u < NUM_CH_MAX && l.pcap != 0u && l.t_max - 1u < T_MAX_ARCH;
}

// ---------------------------------------------------------------------------------------------
// SET_ROWS_KV
// ---------------------------------------------------------------------------------------------
// Lines of one row's block share: the f32 row (E/16 = 4·E/64) and its QA(A32, E) image (QS E/64 +
// D ceil(E/512) = E/64 + ceil(E/64 / 8)).
TSR_PLAN_HOT constexpr uint32_t kv_qa_lines(uint32_t e64) { return e64 + ((e64 + 7u) >> 3); }
TSR_PLAN_HOT constexpr uint32_t kv_row_lines(uint32_t e64) { return e64 << 2; }
TSR_PLAN_HOT constexpr uint32_t kv_lpr(uint32_t e64) { return kv_row_lines(e64) + kv_qa_lines(e64); }
static_assert(kv_qa_lines(2u) == qa_lines(QA_A32, 128u) && kv_qa_lines(16u) == qa_lines(QA_A32, 1024u) &&
                  kv_qa_lines(9u) == qa_lines(QA_A32, 576u) && kv_qa_lines(511u) == qa_lines(QA_A32, 32704u),
              "plan_attn.hpp: KV QA lines = QA_LINES(A32, E)");

// The record-level rules (header comment), on the words a handler read: which, plane kind, GEOM, row
// elements. The fast path evaluates the same function.
TSR_PLAN_HOT constexpr bool kv_args_ok(uint32_t which, uint32_t plane_kind, uint32_t geom, uint32_t row_elems,
                                       const plan_limits & l) {
    const uint32_t e64 = opl::geom_e64(geom), lg = opl::geom_lg(geom);
    return which <= sr::WHICH_V && plane_kind == KV_FMT_Q8_0 && e64 - 1u < E64_MAX && lg != 0u && row_elems == (e64 << 6) &&
           (e64 << 6) <= l.k_max && att_device(l);
}
// Cells a plane holds: 16·C·LG (< 2^25 for C <= 16, LG < 2^16).
TSR_PLAN_HOT constexpr uint32_t kv_cells(uint32_t num_ch, uint32_t lg) { return (num_ch * lg) << 4; }

}  // namespace attn
}  // namespace plan

TSR_PLAN_HOT constexpr bool plan_set_rows_kv(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    namespace at = plan::attn;
    namespace sr = opl::setrows;
    if (n_args != sr::NARGS || !at::kv_args_ok(args[sr::WHICH], args[sr::PLANE_KIND], args[sr::GEOM], args[sr::ROW_ELEMS], l))
        return plan::plan_fail(out, opl::RET_BAD_RECORD);
    const uint32_t e64 = opl::geom_e64(args[sr::GEOM]);
    const uint32_t lpr = at::kv_lpr(e64), n = args[sr::N_ROWS];
    if (lpr > l.budget) return plan::plan_fail(out, opl::RET_SCRATCH);
    uint32_t tr = n < at::KVST_ROWS_MAX ? n : at::KVST_ROWS_MAX;
    if (tr > 1u && tr * lpr > (l.budget >> 1)) {     // two tiles live (as plan_rows), at least one row
        tr = (l.budget >> 1) / lpr;
        if (tr == 0u) tr = 1u;
    }
    out->status = opl::RET_OK;
    out->row_lines = at::kv_row_lines(e64);
    out->tile_rows = tr;
    out->n_tiles = n == 0u ? 0u : n <= tr ? 1u : plan::ceil_div(n, tr);
    out->scratch_lines = (tr != 0u ? tr : 1u) * lpr;
    out->flags = out->n_tiles <= 1u ? plan::KP_ONE_TILE : 0u;
    out->aux[0] = at::kv_qa_lines(e64);
    out->aux[1] = e64 << 6;
    out->aux[2] = opl::geom_lg(args[sr::GEOM]);
    out->aux[3] = l.t_max;
    return true;
}

namespace plan {
namespace attn {

// ---------------------------------------------------------------------------------------------
// FLASH_ATTN
// ---------------------------------------------------------------------------------------------
// The op's shape, pass rule and scratch layout (lines from the block's first line).
struct fa_layout {
    uint32_t D, dcode, E, LG, G, n_head, n_head_kv;
    uint32_t K;          // n_head·D: values of one token's Q row (QACT K)
    uint32_t GS, Tq;     // pass rule: heads per head subset, query tokens per tile
    uint32_t R;          // Tq·n_head: rows of a full tile
    uint32_t ncg;        // NCG_MAX = min(128, C·floor(PCAP/(16·E/D)))
    uint32_t xl;         // QA_LINES(A32, K)
    uint32_t rl;         // ceil(R/16)
    uint32_t qs, x, orun, st, mk, s, p, m, total;      // buffer offsets and the block size
    uint32_t s_lines, p_lines, m_lines;                 // one S / P / M buffer
    uint32_t n_tiles;
};

// RET_OK (and *f), RET_BAD_RECORD or RET_SCRATCH (header comment).
TSR_PLAN_HOT constexpr uint32_t fa_plan(const uint32_t * a, const plan_limits & l, fa_layout * f) {
    if (!att_device(l) || a[fa::SOFTCAP] != 0u) return opl::RET_BAD_RECORD;
    const uint32_t D = a[fa::D], dc = dcode_of(D);
    const uint32_t e64 = opl::geom_e64(a[fa::GEOM]), LG = opl::geom_lg(a[fa::GEOM]);
    const uint32_t G = a[fa::G], T = a[fa::N_TOKENS], C = l.num_ch;
    if (dc > 3u || a[fa::HEAD_DIM] != D || e64 - 1u >= E64_MAX || LG == 0u || G - 1u >= ATT_G_MAX || T == 0u)
        return opl::RET_BAD_RECORD;
    const uint32_t E = e64 << 6;
    if ((E & (D - 1u)) != 0u) return opl::RET_BAD_RECORD;
    const uint32_t nkvh = E >> (6u + dc);
    if (a[fa::N_HEAD_KV] != nkvh || a[fa::N_HEAD] != G * nkvh) return opl::RET_BAD_RECORD;
    const uint64_t cells = uint64_t(kv_cells(C, LG));
    if (uint64_t(a[fa::N_KV]) > cells || a[fa::N_KV] > 16u * ATT_GROUPS_MAX) return opl::RET_BAD_RECORD;
    if (!scale_ok(a[fa::SCALE]) || uint64_t(G) * E > l.k_max) return opl::RET_BAD_RECORD;
    const uint32_t n_head = G * nkvh, K = n_head * D;
    if (a[fa::Q_HEAD_STRIDE] != (D >> 4) || (T > 1u && a[fa::DST_STRIDE] < (K >> 4))) return opl::RET_BAD_RECORD;
    const uint32_t fl = a[fa::FLAGS];
    if ((fl & FA_RSVD) != 0u || ((fl & FA_NEW) == 0u && fl != 0u)) return opl::RET_BAD_RECORD;
    uint32_t GS = 0, Tq = 0;
    tsr_fap_pass_rule(l.t_max, G, &GS, &Tq);
    const uint32_t R = Tq * n_head;
    if (nkvh * GS * Tq > ATT_PASS_ROWS_MAX || R > ATT_R_MAX) return opl::RET_BAD_RECORD;
    // NCG_MAX: ncg() for the largest n_total (its third term only grows with n_total).
    const uint64_t pc = uint64_t(C) * (l.pcap / (16u * (E >> (6u + dc))));
    const uint32_t ncg = pc < ATT_NCG_MAX ? uint32_t(pc) : ATT_NCG_MAX;
    if (ncg == 0u) return opl::RET_BAD_RECORD;
    fa_layout r{};
    r.D = D;
    r.dcode = dc;
    r.E = E;
    r.LG = LG;
    r.G = G;
    r.n_head = n_head;
    r.n_head_kv = nkvh;
    r.K = K;
    r.GS = GS;
    r.Tq = Tq;
    r.R = R;
    r.ncg = ncg;
    r.xl = qa_lines(QA_A32, K);
    r.rl = (R + 15u) >> 4;
    const uint32_t rd = R * (D >> 4);                 // R rows of D values
    r.s_lines = R * ncg;
    r.p_lines = C * rd + C * r.rl;
    r.m_lines = Tq * ncg;
    r.qs = 0u;
    r.x = r.qs + Tq * (K >> 4);
    r.orun = r.x + Tq * r.xl;
    r.st = r.orun + rd;
    r.mk = r.st + 2u * r.rl;
    r.s = r.mk + 4u * r.rl;
    r.p = r.s + 3u * r.s_lines;
    r.m = r.p + 2u * r.p_lines;
    r.total = r.m + 2u * r.m_lines;
    r.n_tiles = T <= Tq ? 1u : ceil_div(T, Tq);
    *f = r;
    return r.total <= l.budget ? opl::RET_OK : opl::RET_SCRATCH;
}

}  // namespace attn
}  // namespace plan

TSR_PLAN_HOT constexpr bool plan_flash_attn(const uint32_t * args, uint32_t n_args, const plan_limits & l, kind_plan * out) {
    if (n_args != opl::fa::NARGS) return plan::plan_fail(out, opl::RET_BAD_RECORD);
    plan::attn::fa_layout f{};
    const uint32_t st = plan::attn::fa_plan(args, l, &f);
    if (st != opl::RET_OK) return plan::plan_fail(out, st);
    out->status = opl::RET_OK;
    out->scratch_lines = f.total;
    out->row_lines = f.K >> 4;
    out->tile_rows = f.Tq;
    out->n_tiles = f.n_tiles;
    out->flags = f.n_tiles == 1u ? plan::KP_ONE_TILE : 0u;
    out->aux[0] = f.GS;
    out->aux[1] = f.ncg;
    out->aux[2] = f.R;
    out->aux[3] = f.E;
    return true;
}

namespace plan {
namespace attn {
// The model shapes plan on the U200 (AT §2.6, §7.4: the 32B decode block is ~18.9 K lines).
constexpr bool fa_u200_ok(uint32_t D, uint32_t nkvh, uint32_t G, uint32_t T, uint32_t n_kv, uint32_t LG, uint32_t * total) {
    uint32_t a[fa::NARGS] = {};
    a[fa::DST_STRIDE] = G * nkvh * D / 16u;
    a[fa::Q_HEAD_STRIDE] = D / 16u;
    a[fa::Q_TOKEN_STRIDE] = G * nkvh * D / 16u;
    a[fa::N_KV] = n_kv;
    a[fa::HEAD_DIM] = D;
    a[fa::N_HEAD] = G * nkvh;
    a[fa::N_HEAD_KV] = nkvh;
    a[fa::N_TOKENS] = T;
    a[fa::MASK] = opl::HANDLE_NONE;
    a[fa::SCALE] = 0x3F800000u;
    a[fa::GEOM] = opl::geom_word(D * nkvh / 64u, LG);
    a[fa::D] = D;
    a[fa::G] = G;
    kind_plan kp{};
    const bool ok = plan_flash_attn(a, fa::NARGS, LIMITS_U200, &kp);
    *total = kp.scratch_lines;
    return ok;
}
constexpr uint32_t fa_u200_total(uint32_t D, uint32_t nkvh, uint32_t G, uint32_t T, uint32_t n_kv, uint32_t LG) {
    uint32_t t = 0;
    return fa_u200_ok(D, nkvh, G, T, n_kv, LG, &t) ? t : 0u;
}
static_assert(fa_u200_total(64, 2, 7, 1, 4096, 64) != 0u && fa_u200_total(128, 8, 5, 1, 4096, 64) == 18948u &&
                  fa_u200_total(256, 8, 2, 8, 1536, 24) != 0u && fa_u200_total(512, 1, 16, 1, 4096, 64) != 0u &&
                  fa_u200_total(128, 8, 5, 512, 32768, 512) != 0u,
              "plan_attn.hpp: the Qwen2.5 / Gemma 4 FA shapes plan on the U200 (32B decode: 18,948 lines)");
}  // namespace attn
}  // namespace plan

}  // namespace tsr

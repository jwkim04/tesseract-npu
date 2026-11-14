// tsr-kinds.h — one encoder per op-list kind over plain descriptors, without ggml
// (docs/design/06-software-verification.md "SW" §7.2 OPL kinds and their argument words, §9.3 builder passes 2 and 6).
//
// The ggml lowering (tsr-build.cpp, tsr_lower_m7) turns a ggml node into a tsr_rec. A framework without ggml (the PyTorch
// runtime) has no node: it knows device handles, row counts, row lengths and strides. These functions take exactly that
// and write the same argument words as the ggml lowerers (tsr_lower_ew / _scale / _rms / _glu / _rope / _set_rows / _fa /
// _cpy / _unary) and as sw/ctrl/tests/test_fw_m7_layer.cpp build() for a Qwen2.5-0.5B layer.
//
// Conventions
//   - A handle is a DRAM handle {ch[31:28], beat[27:0]} of a 64-byte-aligned byte (tsr_dram_handle).
//   - A row set (tsr_k_rows) is the handle of row 0 and the stride in lines (64 bytes) from one row to the next. A dense f32
//     row of n values takes L(n) = ceil(n / 16) lines; the kinds load and store whole lines, so a stored row has its tail
//     line padded (the device writes zeros there).
//   - f32 parameters are given as float; the record holds their bit patterns. The caller rounds to float32 (Python rounds).
//   - Every encoder validates its record with the firmware's own checks (tsr::opl::check_args, tsr::plan::plan_kind or
//     mm_plan with the device's limits) and refuses a kind that the booted image does not have. On refusal it returns false
//     and *why says what is wrong; *out is left empty.
//   - An encoder only builds the record. What the record reads must be in DRAM when the list starts or be written by an
//     earlier record of the list: the preload sources (positions, KV cell indices, the mask summary) must be written before
//     the launch and not be overwritten by a record of the list (SW-15). Each doc comment says what the caller places.
//   - tsr_k_emit adds a record to a tsr_opl_builder (preload entries, FA_NEW, hazards, JOINs, MUL_MAT deferral).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr-opl.h"
#include "tsr-rec.h"
#include "tsr_isa.h"
#include "tsr_ref_at.h"
#include "tsr_transport.h"

// ---------------------------------------------------------------------------------------------
// Device view
// ---------------------------------------------------------------------------------------------

// What the encoders check against: the planner's device view, the channel count, the attention parameters and the kinds the
// booted image provides.
struct tsr_kenv {
    tsr::plan::dev_cfg     cfg = {};     // spm_lines, t_max, k_max, num_ch, pcap, vl, caps
    tsr::plan::plan_limits pl = {};      // plan_limits_of(cfg)
    uint32_t               num_ch = 1;
    tsr_att_cfg            att = {};     // the chunk planner's parameters (C, T_MAX, PCAP, K_MAX), for the mask summaries
    uint64_t               kinds = ~0ull;   // bit k: the image has kind k (tsr_kinds_mask); every kind by default
    std::string            image;        // named in the text of a refusal
    // The booted image provides `kind` and JOIN (every list may need one).
    bool has(uint32_t kind) const {
        return kind < tsr::opl::N_KINDS && ((kinds >> kind) & 1u) != 0 && ((kinds >> tsr::opl::K_JOIN) & 1u) != 0;
    }
};

// Bit k set iff the image's .tsr_kinds has kind k (tsr_versions::kind_rev[k] != 0).
uint64_t tsr_kinds_mask(const tsr_versions & v);

// The device view of a transport's configuration. pcap: AT PCAP = CFG3[31:16] x 16 (read from TSR_CSR_CFG3 with
// TSR_CFG3_PCAP16 / TSR_CFG3_PCAP16_LSB). kinds: tsr_kinds_mask of the booted image (~0 for every kind).
tsr_kenv tsr_kenv_make(const tsr_config & cfg, uint32_t pcap, uint64_t kinds = ~0ull, const char * image = "");

// ---------------------------------------------------------------------------------------------
// Descriptors (tsr_kd_*: what the caller knows; the encoders tsr_k_* make the record)
// ---------------------------------------------------------------------------------------------

// Rows of f32 in DRAM.
struct tsr_k_rows {
    uint32_t h = tsr::opl::HANDLE_NONE;   // handle of row 0
    uint32_t stride = 0;                  // lines between the starts of two rows
};

// L(n): lines of n f32 values.
inline uint32_t tsr_k_lines(uint64_t n) { return (uint32_t) ((n + 15u) / 16u); }

// MUL_MAT y[t, 0 .. N) = W x[t, 0 .. K) (+ addend row t) for t < T, one N part x one K part of a weight.
//   x       T rows of K f32 (the activation columns of this K part: x.h already advanced by k0 / 16 beats for a K part
//           that starts at column k0; the stride is the row stride of the whole activation)
//   y       T rows of L(N) lines
//   add     none (HANDLE_NONE), or T rows of N f32 (stride 0: one row for every token, a bias); not with acc_dst
//   acc_dst y += W x (y is read and written: the K parts after the first of an N part)
//   w_off   W_OFF: the beat offset of the part's weight in every channel (symmetric allocation)
//   fmt     TSR_FMT_Q8_0 or TSR_FMT_Q4_K; N rows, K columns (K a multiple of the block size)
struct tsr_kd_mm {
    tsr_k_rows x, y, add;
    uint32_t w_off = 0;
    uint32_t fmt = TSR_FMT_Q8_0;
    uint32_t N = 0, K = 0, T = 0;
    bool acc_dst = false;
};

// ADD / MUL: dst = a (+|x) b, rows of n values, `rows` rows.
//   b is row (r mod b_period) of a row set (b_period 1: one row for every row, any stride is taken as 0), or, with b_scalar,
//   one f32 word at b.h (b_period words, word r mod b_period; a per-row-period scalar).
struct tsr_kd_ew {
    uint32_t kind = tsr::opl::K_ADD;      // K_ADD or K_MUL
    tsr_k_rows dst, a, b;
    uint32_t n = 0, rows = 0;
    uint32_t b_period = 1;
    bool b_scalar = false;
};

// SCALE: dst = src * s, then + b when b is not +0.0 (the ggml CPU form).
struct tsr_kd_scale {
    tsr_k_rows dst, src;
    uint32_t n = 0, rows = 0;
    float s = 1.0f, b = 0.0f;
};

// RMS_NORM, optionally with the multiplication by a weight: y = ((x * s) * w), s = RC(SQ(FA(FM(sum, inv_n), eps))).
//   weight  none (HANDLE_NONE), or w_period dense rows of n f32 (L(n) lines apart), row r mod w_period for row r
//   the record's inv_n word is 1 / n (tsr_fdiv), computed here (SW §7.2).
struct tsr_kd_rms {
    tsr_k_rows dst, src;
    uint32_t n = 0, rows = 0;
    float eps = 1e-6f;
    uint32_t weight = tsr::opl::HANDLE_NONE;
    uint32_t w_period = 1;
};

// GLU: dst = act(gate) * up. variant: tsr::opl::glu::SWIGLU, GEGLU16 (the CPU's fp16-table form) or GEGLU (exact tanh form).
struct tsr_kd_glu {
    tsr_k_rows dst, gate, up;
    uint32_t n = 0, rows = 0;
    uint32_t variant = tsr::opl::glu::SWIGLU;
};

// UNARY: func tsr::opl::unary::TANH, SILU, EXP, GELU (exact tanh form) or SIGMOID.
struct tsr_kd_unary {
    tsr_k_rows dst, src;
    uint32_t n = 0, rows = 0;
    uint32_t func = tsr::opl::unary::SILU;
};

// ROPE of T tokens, each row of n_heads heads of head_dim values (heads contiguous): the first n_dims values of every head are
// rotated, the rest is copied.
//   pos_src  handle of T I32 positions, line-aligned. A preload source: written before the launch, read at the start of it.
//   table    handle of n_dims / 2 f32 angles A_i (the model's inv_freq), padded to lines, in AUX (not written by the list).
//            The angle of position p and pair i is FM(f32(p), A_i): mode ATAB. HANDLE_NONE: no table, the angle recurrence of
//            the CPU (TS = powf(freq_base, -2 / n_dims)) with freq_base given below.
//   neox     pairing (i, i + n_dims / 2) when true (HF / NEOX), else (2 i, 2 i + 1)
//   attn_factor  multiplies cos and sin (the word M)
struct tsr_kd_rope {
    tsr_k_rows dst, src;                  // T token rows of n_heads * head_dim values
    uint32_t n_heads = 0, head_dim = 0, n_dims = 0, T = 0;
    uint32_t pos_src = tsr::opl::HANDLE_NONE;
    uint32_t table = tsr::opl::HANDLE_NONE;
    bool neox = true;
    float attn_factor = 1.0f;
    float freq_base = 10000.0f;           // only without a table
};

// CPY: `rows` rows of row_lines lines (f32 -> f32), whole lines.
struct tsr_kd_cpy {
    tsr_k_rows dst, src;
    uint32_t row_lines = 0, rows = 0;
};

// The geometry of a KV plane (AT §4): E = n_head_kv * D values per cell (a multiple of 64), C channels, LG local groups per
// channel (16 C LG cells). One plane takes `bytes` bytes in every channel at the same offset; the K and the V plane of a layer
// are two planes of one geometry.
struct tsr_k_kv {
    uint32_t E = 0, C = 0, LG = 0;
    uint64_t bytes = 0;                   // per channel: tsr_kv_plane_beats * 64
    uint64_t cells() const { return 16ull * C * LG; }
    uint32_t geom_word() const { return tsr::opl::geom_word(E / 64u, LG); }
};
// The smallest geometry that holds kv_size cells; false with *why when E is not a multiple of 64 (or above 511 * 64), or
// the plane would not fit a channel.
bool tsr_k_kv_geom(uint32_t E, uint32_t kv_size, uint32_t num_ch, tsr_k_kv * out, std::string * why = nullptr);

// SET_ROWS_KV: quantizes (QACT A32) n_rows rows of E f32 and stores them into cells idx[r] of a K or V plane.
//   src        n_rows rows of E f32
//   idx_src    handle of n_rows I64 cell indices (the low word is the cell, the high word 0), 8-byte aligned. A preload source.
//   plane      base beat of the plane (the same in every channel): 4 KiB aligned
//   which      tsr::opl::setrows::WHICH_K or WHICH_V
// tsr_k_emit notes the indices of a K plane so that the FLASH_ATTN that reads the plane gets FA_NEW.
struct tsr_kd_set_rows {
    tsr_k_rows src;
    uint32_t n_rows = 0;
    tsr_k_kv kv;
    uint32_t which = tsr::opl::setrows::WHICH_K;
    uint32_t plane = 0;
    uint32_t idx_src = tsr::opl::HANDLE_NONE;
};

// The summary of a FLASH_ATTN's mask (AT §7.3) and, when some tile needs the GENERAL path, its f32 copy.
//   msum   tsr_msum_words(n_q) words: the image to store at the preload source handle (msum_h of tsr_k_flash_attn)
//   f32    when need_f32: n_q rows of L(n_kv) lines of floats (the mask divided by the scale, -inf in the padding): the image to
//          store at mask_h; the record reads it
//   binary every mask value is 0 or -inf (the form of a causal or padding mask); other finite values enter the scores as
//          scale * s + m, up to f32 rounding
struct tsr_k_fa_mask {
    uint32_t n_q = 0, n_kv = 0;
    std::vector<uint32_t> msum;
    bool need_f32 = false;
    std::vector<float> f32;
    bool binary = true;
};
// Builds the summary for n_q query rows over cells 0 .. n_kv - 1 for a FLASH_ATTN of shape (E = n_head_kv * D, D, G = n_head /
// n_head_kv) and scale. mask: n_q rows of n_kv f16 values (row q at mask + q * mask_stride), 0x0000 = visible, 0xFC00 = -inf;
// nullptr: causal, row q sees cells 0 .. pos0 + q (pos0 + n_q <= n_kv). f32_h: the handle the caller reserved for the f32 copy
// (n_q * L(n_kv) lines); the summary names it when it needs the copy, so give it always (HANDLE_NONE only when the caller can
// make no room: a mask that needs the copy is then refused). Fills *out; false with *why when the planner refuses the shape.
bool tsr_k_fa_mask_build(const tsr_kenv & env, uint32_t E, uint32_t D, uint32_t G, float scale, uint32_t n_q, uint32_t n_kv,
                         uint32_t f32_h, const uint16_t * mask, size_t mask_stride, uint32_t pos0, tsr_k_fa_mask * out,
                         std::string * why = nullptr);

// FLASH_ATTN over n_kv cells of a K and a V plane (q8_0 K / V, D in {64, 128, 256, 512}), T query tokens.
//   q        T token rows of n_head * D f32, heads contiguous (head h at + h * D / 16 lines)
//   dst      T token rows of n_head * D f32, same layout
//   msum_h   handle of the mask summary (tsr_k_fa_mask::msum), a preload source
//   mask_h   handle of the f32 mask copy (tsr_k_fa_mask::f32), HANDLE_NONE when the summary needs none
//   k_plane / v_plane  base beats of the two planes, of geometry kv
// FA_NEW (the cells that SET_ROWS_KV wrote into the K plane earlier in the same graph) is added by tsr_encode / tsr_k_emit
// from the tsr_fanew_map; give the record the same map that holds the SET_ROWS_KV records' indices.
// T <= 680: the summary and the K indices must fit the 256-line preload area.
struct tsr_kd_flash_attn {
    tsr_k_rows dst, q;
    uint32_t n_kv = 0, D = 0, n_head = 0, n_head_kv = 0, T = 0;
    float scale = 1.0f;
    tsr_k_kv kv;
    uint32_t k_plane = 0, v_plane = 0;
    uint32_t msum_h = tsr::opl::HANDLE_NONE, mask_h = tsr::opl::HANDLE_NONE;
};

// ---------------------------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------------------------
bool tsr_k_mul_mat(const tsr_kenv & env, const tsr_kd_mm & d, tsr_rec * out, std::string * why);
bool tsr_k_add(const tsr_kenv & env, const tsr_kd_ew & d, tsr_rec * out, std::string * why);      // d.kind = K_ADD or K_MUL
bool tsr_k_scale(const tsr_kenv & env, const tsr_kd_scale & d, tsr_rec * out, std::string * why);
bool tsr_k_rms_norm(const tsr_kenv & env, const tsr_kd_rms & d, tsr_rec * out, std::string * why);
bool tsr_k_glu(const tsr_kenv & env, const tsr_kd_glu & d, tsr_rec * out, std::string * why);
bool tsr_k_unary(const tsr_kenv & env, const tsr_kd_unary & d, tsr_rec * out, std::string * why);
bool tsr_k_rope(const tsr_kenv & env, const tsr_kd_rope & d, tsr_rec * out, std::string * why);
bool tsr_k_cpy(const tsr_kenv & env, const tsr_kd_cpy & d, tsr_rec * out, std::string * why);
bool tsr_k_set_rows_kv(const tsr_kenv & env, const tsr_kd_set_rows & d, tsr_rec * out, std::string * why);
bool tsr_k_flash_attn(const tsr_kenv & env, const tsr_kd_flash_attn & d, tsr_rec * out, std::string * why);

// Is a MUL_MAT record deferrable (SW §9.3): the image has MUL_MAT_BEGIN and MUL_MAT_END, the record is one T tile and has no
// ACC_DST / addend problem for the planner. The builder then lowers it as BEGIN / END when emitted with defer = true.
bool tsr_k_mm_deferrable(const tsr_kenv & env, const tsr_rec & mm);

// The checks of every encoder on a finished record: the kind is in the image, tsr::opl::check_args, the firmware's planner
// (plan_kind; mm_plan for MUL_MAT). False with *why.
bool tsr_k_check(const tsr_kenv & env, const tsr_rec & r, std::string * why);

// Adds a record to the builder. MUL_MAT goes through tsr_opl_builder::mul_mat (defer: pending, MUL_MAT_BEGIN / MUL_MAT_END;
// answer tsr_opl_builder::PENDING), every other kind through tsr_encode (preload entries, FA_NEW from *fanew, ranges,
// dependencies). A SET_ROWS_KV records its indices in *fanew first (a null map: not recorded). Returns the op index, NO_ROOM
// (run the list, settle() first, and add the record to a new one) or FAILED, as tsr_encode.
uint32_t tsr_k_emit(tsr_opl_builder & b, tsr_rec & r, int32_t tag, tsr_fanew_map * fanew, bool defer = false);

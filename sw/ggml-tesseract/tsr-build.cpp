// tsr-build.cpp — see tsr-build.h: the M7 records of ggml nodes (probe and build mode), supports_op for
// the M7 kinds, the fusion pass, the encoding of one record and the graph key.
#include "tsr-build.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

// ggml_node_has_n_uses (use counts of the whole graph), struct ggml_cgraph; its static helpers are
// not all used here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "ggml-impl.h"
#pragma GCC diagnostic pop

#include "tsr/plan.hpp"

namespace opl = tsr::opl;

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

static uint32_t tsr_l16(uint64_t n) {            // L(n): lines of n f32 values (VU §4.2)
    return (uint32_t) ((n + 15u) / 16u);
}

static uint32_t tsr_bits(float f) {
    return tsr::opl::writer::f32_bits(f);
}

static float tsr_param_f32(const ggml_tensor * t, int i) {
    float f;
    memcpy(&f, (const char *) t->op_params + 4 * i, 4);
    return f;
}

static int32_t tsr_param_i32(const ggml_tensor * t, int i) {
    int32_t v;
    memcpy(&v, (const char *) t->op_params + 4 * i, 4);
    return v;
}

static bool tsr_fail(std::string * why, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
static bool tsr_fail(std::string * why, const char * fmt, ...) {
    if (why != nullptr) {
        char buf[384];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        *why = buf;
    }
    return false;
}

bool tsr_node_is_noop(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

// The tensor a view reads from (itself if not a view).
static const ggml_tensor * tsr_base(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src : t;
}

// A leaf: a graph input or data, written before the launch (never by a node of the graph). The preload
// table may only carry such tensors: the executor loads it before op 0 (SW §7.3, V2-SW-15).
static bool tsr_is_leaf(const ggml_tensor * t) {
    return tsr_base(t)->op == GGML_OP_NONE;
}

// The rows of t (SW §7.2: shapes are flattened to rows, rows = ne1·ne2·ne3): its elements contiguous
// (nb0 = the element size), and each dimension above 0 continuing the row sequence (a dimension of 1
// is skipped). *stride: bytes between consecutive rows (0 for one row).
static bool tsr_rows_shape(const ggml_tensor * t, uint64_t * rows, uint64_t * stride) {
    if (t->nb[0] != ggml_type_size(t->type) || ggml_blck_size(t->type) != 1) {
        return false;
    }
    uint64_t n = 1, st = 0, next = 0;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (t->ne[d] == 1) {
            continue;
        }
        if (st == 0) {
            st = t->nb[d];
        } else if ((uint64_t) t->nb[d] != next) {
            return false;
        }
        n *= (uint64_t) t->ne[d];
        next = (uint64_t) t->nb[d] * (uint64_t) t->ne[d];
    }
    *rows = n;
    *stride = n > 1 ? st : 0;
    return true;
}

// ggml's broadcast of b over a's rows (ggml_can_repeat) as one row period (SW §7.2: b row r mod p):
// b's row dimensions equal a's up to some dimension and are 1 above it. *p = b's row count.
static bool tsr_row_period(const ggml_tensor * a, const ggml_tensor * b, uint32_t * p) {
    bool ones = false;
    uint64_t n = 1;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (!ones && b->ne[d] == a->ne[d]) {
            n *= (uint64_t) b->ne[d];
            continue;
        }
        if (b->ne[d] != 1) {
            return false;
        }
        ones = true;
    }
    if (n == 0 || n > UINT32_MAX) {
        return false;
    }
    *p = (uint32_t) n;
    return true;
}

// One row operand of a record: the handle of row 0, the rows and their stride in lines.
struct tsr_rows {
    uint32_t h = 0;
    uint64_t rows = 0;
    uint32_t stride = 0;      // lines between rows; one row: L(ne0)
    uint64_t elems = 0;       // ne0
};

// The device address of t's first byte: build mode env.loc (64-aligned, a handle's beat field);
// probe mode only the alignment / raw-ness that env.raw_ok reports. *h = 0 in probe mode.
static bool tsr_addr(const tsr_build_env & env, bool probe, const ggml_tensor * t, const char * what, uint32_t * h,
                     std::string * why) {
    if (probe) {
        if (env.raw_ok && !env.raw_ok(t)) {
            return tsr_fail(why, "%s '%s' is not addressable in place as raw device rows (64-byte base, not a device-format "
                            "weight, not in host memory)", what, t->name);
        }
        *h = 0;
        return true;
    }
    tsr_tloc l;
    if (!env.loc || !env.loc(t, &l)) {
        return tsr_fail(why, "%s '%s' is not in device memory the kinds can address in place", what, t->name);
    }
    if (l.off % TSR_BEAT_BYTES != 0 || l.off / TSR_BEAT_BYTES > 0x0FFFFFFFull) {
        return tsr_fail(why, "%s '%s' at ch%u+0x%llx is not 64-byte aligned or beyond a handle", what, t->name, l.ch,
                        (unsigned long long) l.off);
    }
    *h = tsr_dram_handle(l.ch, l.off);
    return true;
}

// An F32 row operand (the 64-byte rule: base and row stride multiples of 64 bytes, SW §9.3).
static bool tsr_f32_rows(const tsr_build_env & env, bool probe, const ggml_tensor * t, const char * what, tsr_rows * o,
                         std::string * why) {
    if (t->type != GGML_TYPE_F32) {
        return tsr_fail(why, "%s '%s' is %s, not F32", what, t->name, ggml_type_name(t->type));
    }
    uint64_t rows = 0, stride = 0;
    if (!tsr_rows_shape(t, &rows, &stride)) {
        return tsr_fail(why, "%s '%s' is not a sequence of equally spaced contiguous rows", what, t->name);
    }
    if (rows > 1 && (stride % TSR_LINE_BYTES != 0 || stride / TSR_LINE_BYTES > UINT32_MAX)) {
        return tsr_fail(why, "%s '%s': row stride %llu bytes is not a multiple of 64", what, t->name, (unsigned long long) stride);
    }
    if (rows == 0 || rows > UINT32_MAX || t->ne[0] <= 0) {
        return tsr_fail(why, "%s '%s' is empty or has too many rows", what, t->name);
    }
    o->rows = rows;
    o->elems = (uint64_t) t->ne[0];
    o->stride = rows > 1 ? (uint32_t) (stride / TSR_LINE_BYTES) : tsr_l16(o->elems);
    return tsr_addr(env, probe, t, what, &o->h, why);
}

// A record's result rows: as tsr_f32_rows, and the kinds store whole lines (L(n) per row), so a row
// whose n·4 bytes end inside a line writes up to the line's end. Allowed for a tensor of its own (its
// rows are then one row, or rows of whole lines by the stride rule; ggml-alloc pads every block to the
// buffer's 64-byte alignment), not for a view: the bytes after each of its rows belong to the tensor
// it views and may be live (W3b fix: also for views of several rows, whose stride is the viewed
// tensor's).
static bool tsr_out_rows(const tsr_build_env & env, bool probe, const ggml_tensor * t, const char * what, tsr_rows * o,
                         std::string * why) {
    if (!tsr_f32_rows(env, probe, t, what, o, why)) {
        return false;
    }
    if ((o->elems * 4) % TSR_LINE_BYTES != 0 && t->view_src != nullptr) {
        return tsr_fail(why, "%s '%s' is a view whose rows end inside a line: the device's whole-line stores would "
                        "overwrite the viewed tensor's bytes after each row", what, t->name);
    }
    return true;
}

// An index input of the preload table (ROPE positions, GET_ROWS ids: I32; SET_ROWS_KV cells: I64):
// a contiguous leaf of n elements.
static bool tsr_index_input(const tsr_build_env & env, bool probe, const ggml_tensor * t, enum ggml_type type, uint64_t n,
                            const char * what, uint32_t * h, std::string * why) {
    if (t->type != type || !ggml_is_contiguous(t) || (uint64_t) ggml_nelements(t) != n) {
        return tsr_fail(why, "%s '%s' must be a contiguous %s vector of %llu elements", what, t->name, ggml_type_name(type),
                        (unsigned long long) n);
    }
    if (!tsr_is_leaf(t)) {
        return tsr_fail(why, "%s '%s' is computed in the graph: the preload table carries graph inputs only (V2-SW-15)",
                        what, t->name);
    }
    return tsr_addr(env, probe, t, what, h, why);
}

// The planner of the kind on the record's words (V2-SW-20), and the host reader's structural rules.
static bool tsr_plan_ok(const tsr_build_env & env, const tsr_rec & r, std::string * why) {
    const uint32_t e = opl::check_args(r.kind, r.args, r.n_args, tsr::plan::PRE_MAX_LINES);
    if (e != opl::RET_OK) {
        return tsr_fail(why, "%s record: argument rules (opl::check_args) refuse it", tsr_kind_name(r.kind));
    }
    tsr::plan::kind_plan kp{};
    if (!tsr::plan::plan_kind(env.pl, r.kind, r.args, r.n_args, &kp)) {
        return tsr_fail(why, "%s record: the firmware's planner refuses it (%s)", tsr_kind_name(r.kind),
                        kp.status == opl::RET_SCRATCH ? "SCRATCH: one tile does not fit the SPM ring" : "BAD_RECORD");
    }
    return true;
}

const char * tsr_kind_name(uint32_t kind) {
    switch (kind) {
        case opl::K_NOP:         return "NOP";
        case opl::K_JOIN:        return "JOIN";
        case opl::K_MUL_MAT:     return "MUL_MAT";
        case opl::K_ADD:         return "ADD";
        case opl::K_MUL:         return "MUL";
        case opl::K_SCALE:       return "SCALE";
        case opl::K_RMS_NORM:    return "RMS_NORM";
        case opl::K_GLU:         return "GLU";
        case opl::K_ROPE:        return "ROPE";
        case opl::K_GET_ROWS:    return "GET_ROWS";
        case opl::K_SET_ROWS_KV: return "SET_ROWS_KV";
        case opl::K_FLASH_ATTN:  return "FLASH_ATTN";
        case opl::K_CPY:         return "CPY";
        case opl::K_UNARY:       return "UNARY";
        default:                 return "?";
    }
}

// ---------------------------------------------------------------------------------------------
// Lowering, one function per kind (SW §7.2 argument words; the ggml semantics they implement)
// ---------------------------------------------------------------------------------------------

// ADD / MUL: dst = a (+|x) b, b broadcast by whole rows (a row period) or per-row scalars
// (FLAG_B_SCALAR: b.ne0 = 1, a vector of p f32 words; Gemma's inp_scale_rows [1, T] and its
// layer_output_scale [1], SW §8 table, W2b-elem-2).
static bool tsr_lower_ew(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (a == nullptr || b == nullptr || !ggml_are_same_shape(a, node)) {
        return tsr_fail(why, "%s: a and the result differ in shape", ggml_op_name(node->op));
    }
    tsr_rows D, A;
    if (!tsr_out_rows(env, probe, node, "result", &D, why) || !tsr_f32_rows(env, probe, a, "a", &A, why)) {
        return false;
    }
    if (b->type != GGML_TYPE_F32) {
        return tsr_fail(why, "b '%s' is %s, not F32", b->name, ggml_type_name(b->type));
    }
    uint32_t p = 0;
    if (!tsr_row_period(node, b, &p)) {
        return tsr_fail(why, "b '%s' is not broadcast by whole rows or as one scalar per row period", b->name);
    }
    namespace e = opl::ew;
    uint32_t bh = 0, bstride = 0, flags = 0;
    if (b->ne[0] == 1 && node->ne[0] > 1) {
        // per-row scalars: p dense f32 words from a 64-byte-aligned base
        if (!ggml_is_contiguous(b)) {
            return tsr_fail(why, "b '%s' (per-row scalars) is not contiguous", b->name);
        }
        if (!tsr_addr(env, probe, b, "b", &bh, why)) {
            return false;
        }
        flags = e::FLAG_B_SCALAR;
    } else if (b->ne[0] == node->ne[0]) {
        tsr_rows B;
        if (!tsr_f32_rows(env, probe, b, "b", &B, why)) {
            return false;
        }
        bh = B.h;
        bstride = p == 1 ? 0u : B.stride;
    } else {
        return tsr_fail(why, "b '%s' has %lld columns, neither %lld nor 1", b->name, (long long) b->ne[0], (long long) node->ne[0]);
    }
    r->kind = node->op == GGML_OP_ADD ? opl::K_ADD : opl::K_MUL;
    r->n_args = e::NARGS;
    const opl::ew_args w{ D.h, D.stride, A.h, A.stride, bh, bstride, (uint32_t) D.elems, (uint32_t) D.rows, p, flags };
    opl::put_args(w, r->args);
    return true;
}

static bool tsr_lower_scale(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * s = node->src[0];
    if (s == nullptr || !ggml_are_same_shape(s, node)) {
        return tsr_fail(why, "SCALE: source and result differ in shape");
    }
    tsr_rows D, S;
    if (!tsr_out_rows(env, probe, node, "result", &D, why) || !tsr_f32_rows(env, probe, s, "source", &S, why)) {
        return false;
    }
    r->kind = opl::K_SCALE;
    r->n_args = opl::scl::NARGS;
    // y = x·s, then + b when b's bits are not 0 (SW §7.2; ggml-cpu: x·s, or x·s + b when b != 0)
    const opl::scale_args w{ D.h, D.stride, S.h, S.stride, (uint32_t) D.elems, (uint32_t) D.rows,
                             tsr_bits(tsr_param_f32(node, 0)), tsr_bits(tsr_param_f32(node, 1)) };
    opl::put_args(w, r->args);
    return true;
}

// RMS_NORM, optionally with the MUL by a weight folded in (fused: the MUL; its b is the weight, its
// result the record's dst). The weight rows are dense, L(n) lines apart (SW §7.2), row r mod p.
static bool tsr_lower_rms(const tsr_build_env & env, bool probe, const ggml_tensor * node, const ggml_tensor * fused,
                          tsr_rec * r, std::string * why) {
    const ggml_tensor * x = node->src[0];
    const ggml_tensor * out = fused != nullptr ? fused : node;
    if (x == nullptr || !ggml_are_same_shape(x, node) || !ggml_are_same_shape(out, node)) {
        return tsr_fail(why, "RMS_NORM: source and result differ in shape");
    }
    tsr_rows D, X;
    if (!tsr_out_rows(env, probe, out, "result", &D, why) || !tsr_f32_rows(env, probe, x, "source", &X, why)) {
        return false;
    }
    uint32_t wh = opl::HANDLE_NONE, wp = 1;
    if (fused != nullptr) {
        const ggml_tensor * w = fused->src[1];
        if (fused->src[0] != node || w == nullptr || w->type != GGML_TYPE_F32 || w->ne[0] != node->ne[0] ||
            !tsr_row_period(node, w, &wp)) {
            return tsr_fail(why, "RMS_NORM + MUL: the weight is not an F32 row (period) of the norm's width");
        }
        if (wp > 1 && (!ggml_is_contiguous(w) || (node->ne[0] * 4) % TSR_LINE_BYTES != 0)) {
            return tsr_fail(why, "RMS_NORM + MUL: per-row weights must be dense rows of whole lines");
        }
        if (!tsr_addr(env, probe, w, "weight", &wh, why)) {
            return false;
        }
    }
    r->kind = opl::K_RMS_NORM;
    r->n_args = opl::rms::NARGS;
    const uint32_t n = (uint32_t) D.elems;
    // eps as ggml stores it; inv_n = 1/n (correctly rounded, = tsr_fdiv(1, n): SW §7.2)
    const opl::rms_norm_args w{ D.h, D.stride, X.h, X.stride, n, (uint32_t) D.rows, tsr_bits(tsr_param_f32(node, 0)),
                                tsr_bits(1.0f / (float) n), wh, wp };
    opl::put_args(w, r->args);
    return true;
}

// GLU: SWIGLU, or GEGLU as GEGLU16 (the CPU's fp16-table form, V2-SW-31); split form (gate = src0,
// up = src1) or the non-swapped single form (gate = the first half of src0's row, up = the second).
static bool tsr_lower_glu(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const enum ggml_glu_op op = ggml_get_glu_op(node);
    uint32_t variant = 0;
    if (op == GGML_GLU_OP_SWIGLU) {
        variant = opl::glu::SWIGLU;
    } else if (op == GGML_GLU_OP_GEGLU) {
        variant = opl::glu::GEGLU16;
    } else {
        return tsr_fail(why, "GLU %s is not on the device (SWIGLU, GEGLU)", ggml_glu_op_name(op));
    }
    const ggml_tensor * g = node->src[0];
    const ggml_tensor * u = node->src[1];
    tsr_rows D, G, U;
    if (!tsr_out_rows(env, probe, node, "result", &D, why)) {
        return false;
    }
    if (u != nullptr) {
        if (!ggml_are_same_shape(g, node) || !ggml_are_same_shape(u, node)) {
            return tsr_fail(why, "GLU: gate, up and the result differ in shape");
        }
        if (!tsr_f32_rows(env, probe, g, "gate", &G, why) || !tsr_f32_rows(env, probe, u, "up", &U, why)) {
            return false;
        }
    } else {
        if (tsr_param_i32(node, 1) != 0) {
            return tsr_fail(why, "GLU: the swapped single form is not supported");
        }
        if (g->ne[0] != 2 * node->ne[0] || ggml_nrows(g) != ggml_nrows(node)) {
            return tsr_fail(why, "GLU: the single form's source is not [2n, rows]");
        }
        if ((node->ne[0] * 4) % TSR_LINE_BYTES != 0) {
            return tsr_fail(why, "GLU single form: the up half does not start on a line (n %% 16 != 0)");
        }
        if (!tsr_f32_rows(env, probe, g, "gate|up", &G, why)) {
            return false;
        }
        U = G;
        if (!probe) {
            U.h = G.h + (uint32_t) ((uint64_t) node->ne[0] * 4 / TSR_BEAT_BYTES);   // same channel, n·4 bytes on
        }
    }
    r->kind = opl::K_GLU;
    r->n_args = opl::glu::NARGS;
    const opl::glu_args w{ D.h, D.stride, G.h, G.stride, U.h, U.stride, (uint32_t) D.elems, (uint32_t) D.rows, variant };
    opl::put_args(w, r->args);
    return true;
}

// The ROPE view rule (SW §7.2 W2b obligations): [head_dim, n_heads, n_tokens] with each token's heads
// contiguous (nb1 = 4·head_dim) and tokens a multiple of 64 bytes apart (a fused-qkv view's token
// stride is the qkv row). *h, *stride (lines between tokens). out: the result, stored in whole lines
// per token, so a view must have tokens of whole lines (tsr_out_rows; holds by itself while the
// planner requires head_dim % 16 = 0).
static bool tsr_rope_view(const tsr_build_env & env, bool probe, const ggml_tensor * t, const char * what, bool out, uint32_t * h,
                          uint32_t * stride, std::string * why) {
    if (t->type != GGML_TYPE_F32 || t->nb[0] != 4 || (uint64_t) t->nb[1] != 4 * (uint64_t) t->ne[0] || t->ne[3] != 1) {
        return tsr_fail(why, "ROPE %s '%s' is not F32 [head_dim, n_heads, n_tokens] with contiguous heads", what, t->name);
    }
    const uint64_t H = (uint64_t) t->ne[0] * (uint64_t) t->ne[1] * 4;     // bytes of one token's heads
    if (t->ne[2] > 1 && (t->nb[2] % TSR_LINE_BYTES != 0 || (uint64_t) t->nb[2] < H)) {
        return tsr_fail(why, "ROPE %s '%s': token stride %zu bytes is not a multiple of 64 (or overlaps)", what, t->name, t->nb[2]);
    }
    if (out && t->view_src != nullptr && H % TSR_LINE_BYTES != 0) {
        return tsr_fail(why, "ROPE %s '%s' is a view whose tokens end inside a line (whole-line stores)", what, t->name);
    }
    *stride = t->ne[2] > 1 ? (uint32_t) (t->nb[2] / TSR_LINE_BYTES) : (uint32_t) ((H + 63) / 64);
    return tsr_addr(env, probe, t, what, h, why);
}

static bool tsr_lower_rope(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * x = node->src[0];
    const ggml_tensor * pos = node->src[1];
    const ggml_tensor * ff = node->src[2];
    int32_t n_dims = tsr_param_i32(node, 1);
    const int32_t mode = tsr_param_i32(node, 2);
    const float freq_base = tsr_param_f32(node, 5), freq_scale = tsr_param_f32(node, 6), ext_factor = tsr_param_f32(node, 7);
    if (probe && n_dims == 0 && freq_base == 0.0f && ff != nullptr) {
        // llama.cpp's weight-placement probe (src/llama-model-loader.cpp:998-1008: ggml_rope_ext with
        // every parameter 0) asks whether the device reads a freq_factors weight (Gemma 4's
        // rope_freqs); planned as a full rotation, so the weight is placed on the device
        n_dims = (int32_t) node->ne[0];
    }
    const float attn_factor = tsr_param_f32(node, 8);
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return tsr_fail(why, "ROPE mode %d (multi-section / vision) is not on the device", mode);
    }
    if (ext_factor != 0.0f) {
        return tsr_fail(why, "ROPE with YaRN ext_factor %g (the device form needs the host-folded A table, not built)", ext_factor);
    }
    if (tsr_param_i32(node, 15) != 0) {
        return tsr_fail(why, "ROPE with a position offset (ggml_rope_set_offset)");
    }
    if (x == nullptr || !ggml_are_same_shape(x, node) || pos == nullptr) {
        return tsr_fail(why, "ROPE: source and result differ in shape");
    }
    const uint32_t hd = (uint32_t) node->ne[0], nh = (uint32_t) node->ne[1], T = (uint32_t) node->ne[2];
    uint32_t dh = 0, ds = 0, sh = 0, ss = 0, ph = 0, th = opl::HANDLE_NONE;
    if (!tsr_rope_view(env, probe, node, "result", true, &dh, &ds, why) ||
        !tsr_rope_view(env, probe, x, "source", false, &sh, &ss, why)) {
        return false;
    }
    if (!tsr_index_input(env, probe, pos, GGML_TYPE_I32, T, "positions", &ph, why)) {
        return false;
    }
    if (ff != nullptr) {
        if (ff->type != GGML_TYPE_F32 || !ggml_is_contiguous(ff) || (ff->ne[0] < n_dims / 2 && !probe)) {
            return tsr_fail(why, "ROPE freq_factors '%s' is not an F32 vector of n_dims/2", ff->name);
        }
        if (!tsr_addr(env, probe, ff, "freq_factors", &th, why)) {
            return false;
        }
    }
    r->kind = opl::K_ROPE;
    r->n_args = opl::rope::NARGS;
    // TS = powf(freq_base, -2/n_dims) with the host's libm, as ggml-cpu's theta_scale (SW §7.2)
    const opl::rope_args w{ dh, ds, sh, ss, 0u, th, tsr_bits(powf(freq_base, -2.0f / (float) n_dims)), tsr_bits(attn_factor),
                            tsr_bits(freq_scale), hd, (uint32_t) n_dims, nh, T,
                            mode == GGML_ROPE_TYPE_NEOX ? opl::rope::MODE_NEOX : 0u };
    opl::put_args(w, r->args);
    r->pre.push_back({ opl::rope::POS, ph, opl::PRE_POS_BYTES * T });
    return true;
}

// GET_ROWS on f32 rows (llama.cpp's inp_out_ids): ids I32, a graph input.
static bool tsr_lower_get_rows(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * s = node->src[0];
    const ggml_tensor * ids = node->src[1];
    if (s == nullptr || ids == nullptr || s->ne[2] != 1 || s->ne[3] != 1 || ids->ne[1] != 1 || ids->ne[2] != 1 ||
        s->ne[0] != node->ne[0]) {
        return tsr_fail(why, "GET_ROWS: only 2D sources with one id vector");
    }
    tsr_rows D, S;
    if (!tsr_out_rows(env, probe, node, "result", &D, why) || !tsr_f32_rows(env, probe, s, "source", &S, why)) {
        return false;
    }
    uint32_t ih = 0;
    const uint64_t n = (uint64_t) ids->ne[0];
    if (!tsr_index_input(env, probe, ids, GGML_TYPE_I32, n, "ids", &ih, why)) {
        return false;
    }
    r->kind = opl::K_GET_ROWS;
    r->n_args = opl::getrows::NARGS;
    const opl::get_rows_args w{ D.h, D.stride, S.h, S.stride, 0u, (uint32_t) n, tsr_l16(D.elems) };
    opl::put_args(w, r->args);
    r->pre.push_back({ opl::getrows::IDS, ih, (uint32_t) (opl::PRE_ID_BYTES * n) });
    r->aux = S.rows;
    return true;
}

// The plane words of a KV view (build: the placed plane; probe: the shape, base 0).
struct tsr_plane_words {
    uint32_t which = 0, stream = 0, base = 0, geom = 0, handle = 0, E = 0, LG = 0;
};
static bool tsr_kv_words(const tsr_build_env & env, bool probe, const ggml_tensor * t, const char * what, tsr_plane_words * o,
                         std::string * why) {
    if (probe) {
        tsr_kv_shape s;
        if (!env.kv_probe || !env.kv_probe(t, &s)) {
            return tsr_fail(why, "%s '%s' is not a view of a KV-cache tensor in a Tesseract KV buffer", what, t->name);
        }
        if (s.n_stream != 1) {
            return tsr_fail(why, "%s '%s': %u KV streams (M7: one, SW §8 R10)", what, t->name, s.n_stream);
        }
        // as the build branch below: the view starts at the plane's cell 0 (llama.cpp's get_k / get_v /
        // cpy_k / cpy_v with one stream), so supports_op refuses what the builder cannot encode
        if (t->view_src != nullptr && t->view_offs != 0) {
            return tsr_fail(why, "%s '%s' starts at byte %zu of the cache, not at the plane's cell 0", what, t->name, t->view_offs);
        }
        o->which = s.which;
        o->E = s.E;
        o->LG = s.LG;
        o->geom = opl::geom_word(s.E / 64u, s.LG);
        return true;
    }
    tsr_kv_view v;
    if (!env.kv_view || !env.kv_view(t, &v) || v.plane == nullptr) {
        return tsr_fail(why, "%s '%s' is not a view of a placed KV-cache tensor", what, t->name);
    }
    if (v.cell0 != 0 || v.byte_in_row != 0) {
        return tsr_fail(why, "%s '%s' starts at cell %u byte %u, not at the plane's cell 0", what, t->name, v.cell0, v.byte_in_row);
    }
    if (v.t != nullptr && v.t->shape.n_stream != 1) {
        return tsr_fail(why, "%s '%s': %u KV streams (M7: one, SW §8 R10)", what, t->name, v.t->shape.n_stream);
    }
    o->which = v.plane->which;
    o->stream = v.plane->stream;
    o->base = v.plane->base_beat();
    o->geom = v.plane->geom_word();
    o->handle = v.plane->handle();
    o->E = v.plane->geom.E;
    o->LG = v.plane->geom.LG;
    return true;
}

// SET_ROWS into a KV cache (llama.cpp's cpy_k / cpy_v): QACT A32 + KVST per run of cells (SW §7.5).
static bool tsr_lower_set_rows(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    // ggml_set_rows(a, b, c): src[0] = b (the rows), src[1] = c (the indices), src[2] = a (the
    // destination; the node is a view of it), ggml.c's legacy order
    const ggml_tensor * s = node->src[0];
    const ggml_tensor * idx = node->src[1];
    const ggml_tensor * cache = node->src[2] != nullptr ? node->src[2] : node->view_src;
    if (cache == nullptr || s == nullptr || idx == nullptr) {
        return tsr_fail(why, "SET_ROWS without operands");
    }
    tsr_plane_words pw;
    if (!tsr_kv_words(env, probe, cache, "destination", &pw, why)) {
        return false;     // R1: a SET_ROWS that does not write a KV cache stays on the CPU
    }
    if (cache->type != GGML_TYPE_Q8_0 || (uint32_t) s->ne[0] != pw.E) {
        return tsr_fail(why, "SET_ROWS: rows of %lld values into a q8_0 cache of E = %u", (long long) s->ne[0], pw.E);
    }
    // the destination's rows are the plane's cells (E values each), as cpy_k / cpy_v view the cache
    if (cache->ne[0] != (int64_t) pw.E || cache->nb[1] != ggml_row_size(GGML_TYPE_Q8_0, pw.E)) {
        return tsr_fail(why, "SET_ROWS: the destination's rows are not the cache's cells of E = %u values", pw.E);
    }
    tsr_rows S;
    if (!tsr_f32_rows(env, probe, s, "source", &S, why)) {
        return false;
    }
    uint32_t ih = 0;
    if (!tsr_index_input(env, probe, idx, GGML_TYPE_I64, S.rows, "cell indices", &ih, why)) {
        return false;
    }
    r->kind = opl::K_SET_ROWS_KV;
    r->n_args = opl::setrows::NARGS;
    // words 0 / 1 / 11 informational (the cache's handle, row stride 0, stream), 8-10 the plane (SW §7.2)
    const opl::set_rows_kv_args w{ pw.handle, 0u, S.h, S.stride, 0u, (uint32_t) S.rows, pw.E, pw.which, pw.base, pw.geom,
                                   0u, pw.stream };
    opl::put_args(w, r->args);
    r->pre.push_back({ opl::setrows::IDX, ih, (uint32_t) (opl::PRE_KVIDX_BYTES * S.rows) });
    r->kv_plane = pw.base;
    r->kv_idx_src = ih;
    r->kv_idx_n = (uint32_t) S.rows;
    return true;
}

// FLASH_ATTN_EXT with K / V in KV planes (AT §2.6; SW §7.2 0x19).
static bool tsr_lower_fa(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    if (q == nullptr || k == nullptr || v == nullptr) {
        return tsr_fail(why, "FLASH_ATTN_EXT without q / k / v");
    }
    if (node->src[4] != nullptr) {
        return tsr_fail(why, "FLASH_ATTN_EXT with attention sinks");
    }
    const float scale = tsr_param_f32(node, 0), max_bias = tsr_param_f32(node, 1), softcap = tsr_param_f32(node, 2);
    const int32_t prec = tsr_param_i32(node, 3), n_kv_max = tsr_param_i32(node, 4);
    if (max_bias != 0.0f || softcap != 0.0f || n_kv_max != 0 || (prec != GGML_PREC_DEFAULT && prec != GGML_PREC_F32)) {
        return tsr_fail(why, "FLASH_ATTN_EXT with ALiBi, a softcap, n_kv_max or a precision other than F32 (AT §2.6)");
    }
    const uint32_t D = (uint32_t) q->ne[0], T = (uint32_t) q->ne[1], n_head = (uint32_t) q->ne[2];
    const uint32_t n_kv = (uint32_t) k->ne[1], n_head_kv = (uint32_t) k->ne[2];
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q8_0 || v->type != GGML_TYPE_Q8_0 || node->type != GGML_TYPE_F32) {
        return tsr_fail(why, "FLASH_ATTN_EXT: q F32, K / V q8_0 only");
    }
    if ((D != 64 && D != 128 && D != 256 && D != 512) || k->ne[0] != (int64_t) D || v->ne[0] != (int64_t) D ||
        v->ne[1] != (int64_t) n_kv || v->ne[2] != (int64_t) n_head_kv || n_head_kv == 0 || n_head % n_head_kv != 0 ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) {
        return tsr_fail(why, "FLASH_ATTN_EXT shape: D = DK = DV in {64, 128, 256, 512}, n_head %% n_head_kv = 0, one stream");
    }
    // q: [D, T, n_head] (ggml's permute of [D, n_head, T]): heads D values apart (contiguous per token)
    if (q->nb[0] != 4 || (uint64_t) q->nb[2] != 4ull * D || (T > 1 && q->nb[1] % TSR_LINE_BYTES != 0)) {
        return tsr_fail(why, "FLASH_ATTN_EXT: q's heads are not contiguous per token or its token stride is not 64-byte");
    }
    // the result: [D, n_head, T], heads contiguous, tokens a multiple of 64 bytes apart
    if (node->nb[0] != 4 || (uint64_t) node->nb[1] != 4ull * D || (T > 1 && node->nb[2] % TSR_LINE_BYTES != 0) ||
        node->ne[0] != (int64_t) D || node->ne[1] != (int64_t) n_head || node->ne[2] != (int64_t) T) {
        return tsr_fail(why, "FLASH_ATTN_EXT: the result's layout");
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[0] != (int64_t) n_kv || mask->ne[1] != (int64_t) T ||
                            mask->ne[2] != 1 || mask->ne[3] != 1)) {
        return tsr_fail(why, "FLASH_ATTN_EXT: the mask is not F16 [n_kv, n_tokens]");
    }
    tsr_plane_words kp, vp;
    if (!tsr_kv_words(env, probe, k, "K", &kp, why) || !tsr_kv_words(env, probe, v, "V", &vp, why)) {
        return false;
    }
    if (kp.which != TSR_KV_K || vp.which != TSR_KV_V || kp.E != n_head_kv * D || vp.E != kp.E || vp.LG != kp.LG) {
        return tsr_fail(why, "FLASH_ATTN_EXT: K / V are not the K and V planes of one layer with E = n_head_kv·D");
    }
    // K / V as get_k / get_v view the cache (then permuted to [D, n_kv, n_head_kv]): blocks contiguous,
    // cells E values apart, heads D values apart (the record has no stride words for them)
    const size_t cell_b = ggml_row_size(GGML_TYPE_Q8_0, kp.E), head_b = ggml_row_size(GGML_TYPE_Q8_0, D);
    for (const ggml_tensor * t : { k, v }) {
        if (t->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || (n_kv > 1 && t->nb[1] != cell_b) || (n_head_kv > 1 && t->nb[2] != head_b)) {
            return tsr_fail(why, "FLASH_ATTN_EXT: '%s' is not the cache's cells (%zu bytes apart) with heads %zu bytes apart",
                            t->name, cell_b, head_b);
        }
    }
    // The preload entries of this record alone, as in a list cut just before it: the MSUM of T rows and
    // FA_NEW's K indices (the SET_ROWS_KV of the same ubatch: T cells) must fit the 256-line area
    // together, each entry from a 64-byte boundary: T <= 680 (the MSUM alone would allow 1022). A
    // longer ubatch's FLASH_ATTN stays on the CPU (W3b fix; V2-SW-43).
    const uint64_t pre_lines = (4ull * (opl::MSUM_HDR_WORDS + (uint64_t) opl::MSUM_ROW_WORDS * T) + opl::PRE_ALIGN - 1) / opl::PRE_ALIGN +
                               ((uint64_t) opl::PRE_KVIDX_BYTES * T + opl::PRE_ALIGN - 1) / opl::PRE_ALIGN;
    if (pre_lines > opl::PRE_MAX_LINES) {
        return tsr_fail(why, "FLASH_ATTN_EXT of %u tokens: its MSUM and new-cell indices need %llu preload lines (at most %u: "
                        "T <= 680)", T, (unsigned long long) pre_lines, opl::PRE_MAX_LINES);
    }
    uint32_t qh = 0, dh = 0;
    if (!tsr_addr(env, probe, q, "q", &qh, why) || !tsr_addr(env, probe, node, "result", &dh, why)) {
        return false;
    }
    tsr_msum_ref m;
    if (probe) {
        if (mask != nullptr && env.fa_mask_probe && !env.fa_mask_probe(node)) {
            return tsr_fail(why, "FLASH_ATTN_EXT: the mask '%s' cannot be summarized (AT §7.3)", mask->name);
        }
        m.valid = true;
        m.n_q = T;
        m.n_kv = n_kv;
        m.msum_handle = 0;
        m.mask_handle = opl::HANDLE_NONE;
    } else {
        if (env.fa_mask) {
            m = env.fa_mask(node);
        }
        if (!m.valid || m.n_q != T || m.n_kv != n_kv) {
            return tsr_fail(why, "FLASH_ATTN_EXT: no mask summary for '%s' (n_q %u, n_kv %u)", mask ? mask->name : "(none)", m.n_q,
                            m.n_kv);
        }
    }
    const uint32_t token_lines = n_head * D / 16;
    r->kind = opl::K_FLASH_ATTN;
    r->n_args = opl::fa::NARGS;
    const opl::flash_attn_args w{ dh, T > 1 ? (uint32_t) (node->nb[2] / TSR_LINE_BYTES) : token_lines, qh, D / 16,
                                  T > 1 ? (uint32_t) (q->nb[1] / TSR_LINE_BYTES) : token_lines, kp.handle, vp.handle, n_kv, D,
                                  n_head, n_head_kv, T, 0u, m.mask_handle, tsr_bits(scale), 0u, kp.base, vp.base, kp.geom, D,
                                  n_head / n_head_kv, 0u };
    opl::put_args(w, r->args);
    r->pre.push_back({ opl::fa::MSUM, m.msum_handle, opl::msum_bytes(T) });
    r->kv_plane = kp.base;
    r->mask = m;
    return true;
}

// CPY (f32 -> f32, into src1) and CONT (f32, a new contiguous tensor): whole rows.
static bool tsr_lower_cpy(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * s = node->src[0];
    if (s == nullptr || s->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || s->ne[0] != node->ne[0] ||
        ggml_nrows(s) != ggml_nrows(node)) {
        return tsr_fail(why, "%s: F32 -> F32 with the same rows only", ggml_op_name(node->op));
    }
    tsr_rows D, S;
    if (!tsr_out_rows(env, probe, node, "destination", &D, why) || !tsr_f32_rows(env, probe, s, "source", &S, why)) {
        return false;
    }
    r->kind = opl::K_CPY;
    r->n_args = opl::cpy::NARGS;
    const opl::cpy_args w{ D.h, D.stride, S.h, S.stride, tsr_l16(D.elems), (uint32_t) D.rows };
    opl::put_args(w, r->args);
    return true;
}

static bool tsr_lower_unary(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const enum ggml_unary_op op = ggml_get_unary_op(node);
    uint32_t func = 0;
    if (op == GGML_UNARY_OP_TANH) {
        func = opl::unary::TANH;
    } else if (op == GGML_UNARY_OP_GELU) {
        func = opl::unary::GELU16;    // llama.cpp's GELU: ggml-cpu's fp16 table form (V2-SW-31)
    } else {
        return tsr_fail(why, "UNARY %s is not on the device (TANH, GELU)", ggml_unary_op_name(op));
    }
    const ggml_tensor * s = node->src[0];
    if (s == nullptr || !ggml_are_same_shape(s, node)) {
        return tsr_fail(why, "UNARY: source and result differ in shape");
    }
    tsr_rows D, S;
    if (!tsr_out_rows(env, probe, node, "result", &D, why) || !tsr_f32_rows(env, probe, s, "source", &S, why)) {
        return false;
    }
    r->kind = opl::K_UNARY;
    r->n_args = opl::unary::NARGS;
    const opl::unary_args w{ D.h, D.stride, S.h, S.stride, (uint32_t) D.elems, (uint32_t) D.rows, func };
    opl::put_args(w, r->args);
    return true;
}

bool tsr_lower_m7(const tsr_build_env & env, bool probe, const ggml_tensor * node, const ggml_tensor * fused, tsr_rec * out,
                  std::string * why) {
    *out = tsr_rec();
    bool ok = false;
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:            ok = tsr_lower_ew(env, probe, node, out, why); break;
        case GGML_OP_SCALE:          ok = tsr_lower_scale(env, probe, node, out, why); break;
        case GGML_OP_RMS_NORM:       ok = tsr_lower_rms(env, probe, node, fused, out, why); break;
        case GGML_OP_GLU:            ok = tsr_lower_glu(env, probe, node, out, why); break;
        case GGML_OP_ROPE:           ok = tsr_lower_rope(env, probe, node, out, why); break;
        case GGML_OP_GET_ROWS:       ok = tsr_lower_get_rows(env, probe, node, out, why); break;
        case GGML_OP_SET_ROWS:       ok = tsr_lower_set_rows(env, probe, node, out, why); break;
        case GGML_OP_FLASH_ATTN_EXT: ok = tsr_lower_fa(env, probe, node, out, why); break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:           ok = tsr_lower_cpy(env, probe, node, out, why); break;
        case GGML_OP_UNARY:          ok = tsr_lower_unary(env, probe, node, out, why); break;
        default:
            return tsr_fail(why, "op %s has no device kind", ggml_op_desc(node));
    }
    return ok && tsr_plan_ok(env, *out, why);
}

bool tsr_build_supports(const tsr_build_env & env, const ggml_tensor * op, std::string * why) {
    if (tsr_node_is_noop(op)) {
        return true;
    }
    if (op->op == GGML_OP_DUP) {
        return tsr_fail(why, "DUP stays on the CPU (SW §8 R1)");
    }
    if (env.pl.caps == 0 || env.pl.num_ch == 0) {
        return tsr_fail(why, "the device configuration has no VU / AT parameters");
    }
    tsr_rec r;
    return tsr_lower_m7(env, true, op, nullptr, &r, why);
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT + ADD (pass 3)
// ---------------------------------------------------------------------------------------------
// The ADD's operand that is not the MUL_MAT's result.
static const ggml_tensor * tsr_add_other(const ggml_tensor * mm, const ggml_tensor * add) {
    return add->src[0] == mm ? add->src[1] : add->src[0];
}

bool tsr_fuse_mm_add(const tsr_build_env & env, const ggml_tensor * mm, const ggml_tensor * add, uint32_t * a,
                     std::string * why) {
    namespace m = opl::mm;
    const ggml_tensor * b = tsr_add_other(mm, add);
    if (b == nullptr || b == mm || b->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || !ggml_are_same_shape(add, mm) ||
        b->ne[0] != mm->ne[0]) {
        return tsr_fail(why, "MUL_MAT + ADD: not a bias or residual of the MUL_MAT's shape");
    }
    tsr_rows Y, B;
    if (!tsr_out_rows(env, false, add, "ADD result", &Y, why) || !tsr_f32_rows(env, false, b, "addend", &B, why)) {
        return false;
    }
    const uint64_t T = a[m::T];
    if (Y.rows != T || (B.rows != 1 && B.rows != T)) {
        return tsr_fail(why, "MUL_MAT + ADD: the addend has %llu rows, the MUL_MAT %llu tokens", (unsigned long long) B.rows,
                        (unsigned long long) T);
    }
    const uint32_t y_lines = tsr_l16(mm->ne[0]);
    if (T > 1 && Y.stride < y_lines) {
        return tsr_fail(why, "MUL_MAT + ADD: the result's rows overlap");
    }
    a[m::Y] = Y.h;
    a[m::Y_STRIDE] = T > 1 ? Y.stride : y_lines;
    a[m::ADD] = B.h;
    a[m::ADD_STRIDE] = B.rows == 1 ? 0u : B.stride;     // a bias row is repeated for every token
    return true;
}

// ---------------------------------------------------------------------------------------------
// Fusion pairs (pass 3)
// ---------------------------------------------------------------------------------------------
struct tsr_vrange {
    uint64_t lo, hi;
};
static tsr_vrange tsr_bytes_of(const ggml_tensor * t) {
    const uint64_t lo = (uint64_t) (uintptr_t) t->data;
    return { lo, lo == 0 ? 0 : lo + ggml_nbytes(t) };
}
static bool tsr_vr_overlap(const tsr_vrange & a, const tsr_vrange & b) {
    return a.lo < a.hi && b.lo < b.hi && a.lo < b.hi && b.lo < a.hi;
}

// Can node k's work move to node i's position? (other: k's operand that is not i's result; out: k's
// result.) No node in between writes `other`, or reads or writes `out`.
static bool tsr_can_hoist(const ggml_cgraph * cg, int i, int k, const ggml_tensor * other, const ggml_tensor * out) {
    const tsr_vrange ob = tsr_bytes_of(other), rb = tsr_bytes_of(out);
    if (ob.lo == 0 || rb.lo == 0) {
        return false;
    }
    for (int j = i + 1; j < k; ++j) {
        const ggml_tensor * n = cg->nodes[j];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || tsr_node_is_noop(n)) {
            continue;
        }
        const tsr_vrange w = tsr_bytes_of(n);
        if (tsr_vr_overlap(w, ob) || tsr_vr_overlap(w, rb)) {
            return false;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] != nullptr && tsr_vr_overlap(tsr_bytes_of(n->src[s]), rb)) {
                return false;
            }
        }
    }
    return true;
}

tsr_fusion tsr_build_fusion(const tsr_build_env & env, const ggml_cgraph * cg) {
    tsr_fusion f;
    const int n = cg->n_nodes;
    f.with.assign(n, -1);
    f.into.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * node = cg->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || f.into[i] >= 0 ||
            (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_RMS_NORM) || !ggml_node_has_n_uses(cg, i, 1)) {
            continue;
        }
        int k = -1;
        for (int j = i + 1; j < n && k < 0; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (cg->nodes[j]->src[s] == node) {
                    k = j;
                }
            }
        }
        if (k < 0) {
            continue;    // its user is in another graph (another split, an observed node's next call)
        }
        const ggml_tensor * u = cg->nodes[k];
        if ((u->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || f.into[k] >= 0 || f.with[k] >= 0) {
            continue;
        }
        tsr_tloc l;
        if (node->op == GGML_OP_MUL_MAT) {
            if (u->op != GGML_OP_ADD || (u->src[0] != node && u->src[1] != node) || u->src[0] == u->src[1]) {
                continue;
            }
            const ggml_tensor * b = tsr_add_other(node, u);
            const ggml_tensor * x = node->src[1];
            // the MUL_MAT must run in place (not staged): x, the ADD result and the addend on the device,
            // and its rows on lines when T > 1 (SW §9.3 step 2)
            if (b->type != GGML_TYPE_F32 || b->ne[0] != node->ne[0] || !ggml_are_same_shape(u, node) ||
                (ggml_nrows(b) != 1 && !ggml_are_same_shape(b, node)) || !env.loc || !env.loc(x, &l) || !env.loc(u, &l) ||
                !env.loc(b, &l) || (ggml_nrows(x) > 1 && (node->ne[0] * 4) % TSR_LINE_BYTES != 0)) {
                continue;
            }
            if (!tsr_can_hoist(cg, i, k, b, u)) {
                continue;
            }
        } else {
            if (u->op != GGML_OP_MUL || u->src[0] != node || u->src[1] == node || !ggml_are_same_shape(u, node)) {
                continue;
            }
            const ggml_tensor * w = u->src[1];
            uint32_t p = 0;
            if (w->type != GGML_TYPE_F32 || w->ne[0] != node->ne[0] || !tsr_row_period(node, w, &p) ||
                (p > 1 && (!ggml_is_contiguous(w) || (node->ne[0] * 4) % TSR_LINE_BYTES != 0)) || !env.loc || !env.loc(w, &l) ||
                !env.loc(u, &l)) {
                continue;
            }
            if (!tsr_can_hoist(cg, i, k, w, u)) {
                continue;
            }
        }
        f.with[i] = k;
        f.into[k] = i;
        f.n_pairs++;
    }
    return f;
}

// ---------------------------------------------------------------------------------------------
// Encoding (passes 4-6)
// ---------------------------------------------------------------------------------------------
void tsr_note_kv_write(const tsr_rec & r, tsr_fanew_map * m) {
    if (r.kind == opl::K_SET_ROWS_KV && r.kv_idx_n != 0) {
        (*m)[r.kv_plane] = { r.kv_idx_src, r.kv_idx_n };
    }
}

uint32_t tsr_encode(tsr_opl_builder & b, tsr_rec & r, int32_t tag, const tsr_fanew_map & fanew) {
    std::vector<std::pair<uint32_t, uint32_t>> ents;
    for (const tsr_pre_ref & p : r.pre) {
        ents.push_back({ p.src, p.bytes });
    }
    const auto nw = r.kind == opl::K_FLASH_ATTN ? fanew.find(r.kv_plane) : fanew.end();
    if (nw != fanew.end()) {
        ents.push_back({ nw->second.first, opl::PRE_KVIDX_BYTES * nw->second.second });
    }
    if (b.preload_lines_with(ents) > opl::PRE_MAX_LINES) {
        return tsr_opl_builder::NO_ROOM;
    }
    for (const tsr_pre_ref & p : r.pre) {
        const uint32_t off = b.preload(p.src, p.bytes);
        if (off == tsr_opl_builder::NO_ROOM || off == tsr_opl_builder::FAILED) {
            return off;
        }
        r.args[p.word] = off;
    }
    if (r.kind == opl::K_FLASH_ATTN) {
        r.args[opl::fa::FLAGS] = 0;
        if (nw != fanew.end()) {
            // FA_NEW: the cells SET_ROWS_KV wrote into this K plane (AT §6.6 rule 7, §12)
            const uint32_t off = b.preload(nw->second.first, opl::PRE_KVIDX_BYTES * nw->second.second);
            if (off == tsr_opl_builder::NO_ROOM || off == tsr_opl_builder::FAILED) {
                return off;
            }
            r.args[opl::fa::FLAGS] = tsr::plan::attn::fa_flags(off, nw->second.second);
        }
    }
    std::vector<tsr_dram_range> rd, wr;
    b.kind_ranges(r.kind, r.args, r.aux, rd, wr);
    return b.add(r.kind, r.args, r.n_args, rd, wr, tag);
}

// ---------------------------------------------------------------------------------------------
// Graph identity (R7)
// ---------------------------------------------------------------------------------------------
static inline void tsr_mix(uint64_t * h, uint64_t v) {
    *h ^= v + 0x9e3779b97f4a7c15ull + (*h << 6) + (*h >> 2);
    *h *= 0x100000001b3ull;
}

static void tsr_mix_tensor(uint64_t * h, const ggml_tensor * t) {
    tsr_mix(h, (uint64_t) (uintptr_t) t);
    tsr_mix(h, (uint64_t) (uintptr_t) t->data);
    tsr_mix(h, (uint64_t) t->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        tsr_mix(h, (uint64_t) t->ne[d]);
        tsr_mix(h, (uint64_t) t->nb[d]);
    }
}

uint64_t tsr_graph_key(const ggml_cgraph * cg, uint64_t salt) {
    uint64_t h = 0xcbf29ce484222325ull;
    tsr_mix(&h, salt);
    tsr_mix(&h, (uint64_t) cg->n_nodes);
    for (int i = 0; i < cg->n_nodes; ++i) {
        const ggml_tensor * n = cg->nodes[i];
        tsr_mix_tensor(&h, n);
        tsr_mix(&h, (uint64_t) n->op);
        tsr_mix(&h, (uint64_t) n->flags);
        for (size_t w = 0; w < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++w) {
            tsr_mix(&h, (uint64_t) (uint32_t) n->op_params[w]);
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] != nullptr) {
                tsr_mix(&h, (uint64_t) s);
                tsr_mix_tensor(&h, n->src[s]);
            }
        }
    }
    return h;
}

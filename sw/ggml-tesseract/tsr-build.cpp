// tsr-build.cpp — see tsr-build.h: the M7 records of ggml nodes (probe and build mode), supports_op for
// the M7 kinds, the fusion pass, the encoding of one record and the graph key.
#include "tsr-build.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <strings.h>

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

// An allocated tensor that is not in a Tesseract buffer (the op would read or write another backend's memory
// in place). Unallocated tensors (graph planning) and tensors with no on_device hook pass.
static bool tsr_foreign_buffer(const tsr_build_env & env, const ggml_tensor * t) {
    const ggml_tensor * b = tsr_base(t);
    return env.on_device && b->buffer != nullptr && !env.on_device(b);
}

// A leaf: a graph input or data, written before the launch (never by a node of the graph). The preload
// table may only carry such tensors: the executor loads it before op 0 (SW §7.3, SW-15).
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
        return tsr_fail(why, "%s '%s' is computed in the graph: the preload table carries graph inputs only (SW-15)",
                        what, t->name);
    }
    return tsr_addr(env, probe, t, what, h, why);
}

// The planner of the kind on the record's words (SW-20), and the host reader's structural rules.
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

const char * tsr_kind_name(uint32_t kind) { return opl::kind_name(kind); }

bool tsr_kinds_parse(const char * spec, uint64_t * mask, std::string * why) {
    const uint64_t keep = 1ull << opl::K_NOP | 1ull << opl::K_JOIN;   // every list may need them
    if (spec == nullptr || *spec == '\0') {
        *mask = ~0ull;
        return true;
    }
    uint64_t m = spec[0] == '-' ? ~0ull : keep;
    const std::string s(spec);
    for (size_t i = 0; i <= s.size();) {
        size_t j = s.find(',', i);
        j = j == std::string::npos ? s.size() : j;
        const std::string w = s.substr(i, j - i);
        i = j + 1;
        if (w == "all") {
            m = ~0ull;
            continue;
        }
        if (w == "m6") {
            m = keep | 1ull << opl::K_MUL_MAT;
            continue;
        }
        const bool del = !w.empty() && w[0] == '-';
        const std::string name = !w.empty() && (w[0] == '-' || w[0] == '+') ? w.substr(1) : w;
        uint32_t k = 0;
        while (k < opl::N_KINDS && (opl::kind_nargs(k) == opl::NARGS_UNDEF || strcasecmp(name.c_str(), opl::kind_name(k)) != 0)) {
            ++k;
        }
        if (k == opl::N_KINDS) {
            return tsr_fail(why, "TSR_KINDS='%s': '%s' is not all, m6 or [+|-] a kind name (MUL_MAT, ADD, RMS_NORM, "
                            "FLASH_ATTN, ...)", spec, w.c_str());
        }
        if (del && ((keep >> k) & 1u) != 0) {
            return tsr_fail(why, "TSR_KINDS='%s': %s cannot be removed", spec, opl::kind_name(k));
        }
        m = del ? m & ~(1ull << k) : m | 1ull << k;
    }
    *mask = m;
    return true;
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
    if (node->view_src != nullptr && tsr_foreign_buffer(env, node)) {
        return tsr_fail(why, "SCALE writes '%s', which is in another backend's memory", tsr_base(node)->name);
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
    if (!tsr_out_rows(env, probe, out, "result", &D, why)) {
        return false;
    }
    // A source whose tokens are not spaced like its heads (a view into a wider row: Gated DeltaNet's q and k in the
    // convolution output [128, H, T] with the token stride of the whole channel row) is normalised one token at a
    // time: one record per token, without a weight (SW §7.9).
    bool per_token = false;
    std::string why_x;
    if (!tsr_f32_rows(env, probe, x, "source", &X, &why_x)) {
        per_token = fused == nullptr && x->type == GGML_TYPE_F32 && x->nb[0] == 4 && x->ne[3] == 1 && x->ne[2] > 1 &&
                    (x->ne[1] == 1 || x->nb[1] % TSR_LINE_BYTES == 0) && x->nb[2] % TSR_LINE_BYTES == 0 &&
                    (uint64_t) x->nb[1] * x->ne[1] <= (uint64_t) x->nb[2] && (x->ne[0] * 4) % TSR_LINE_BYTES == 0;
        if (!per_token) {
            return tsr_fail(why, "%s", why_x.c_str());
        }
        X.rows = (uint64_t) x->ne[1];
        X.elems = (uint64_t) x->ne[0];
        X.stride = x->ne[1] > 1 ? (uint32_t) (x->nb[1] / TSR_LINE_BYTES) : tsr_l16(X.elems);
        if (!tsr_addr(env, probe, x, "source", &X.h, why)) {
            return false;
        }
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
    const uint32_t rows = per_token ? (uint32_t) X.rows : (uint32_t) D.rows;
    // eps as ggml stores it; inv_n = 1/n (correctly rounded, = tsr_fdiv(1, n): SW §7.2)
    const opl::rms_norm_args w{ D.h, D.stride, X.h, X.stride, n, rows, tsr_bits(tsr_param_f32(node, 0)),
                                tsr_bits(1.0f / (float) n), wh, wp };
    opl::put_args(w, r->args);
    if (per_token && !probe) {
        for (int64_t t = 1; t < x->ne[2]; ++t) {
            tsr_rec q = *r;
            q.args[opl::rms::SRC] = X.h + (uint32_t) (t * (int64_t) (x->nb[2] / TSR_LINE_BYTES));
            q.args[opl::rms::DST] = D.h + (uint32_t) (t * (int64_t) rows * D.stride);
            r->more.push_back(q);
        }
    }
    return true;
}

// GLU: SWIGLU, or GEGLU as GEGLU16 (the CPU's fp16-table form, SW-31); split form (gate = src0,
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
    // IMROPE (Qwen3.5 / 3.6 / 3.8 interleaved M-RoPE): for text tokens the position streams are [p, p, p, 0]
    // (llama-batch.cpp), the sections then pick the same position for every pair and the rotation is NEOX's;
    // the device reads the first n_tokens positions (the temporal stream). A batch with image or video
    // positions would differ: the backend checks the inp_pos input when it is written (tsr_buffer_set_tensor).
    const bool imrope = mode == GGML_ROPE_TYPE_IMROPE;
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && !imrope) {
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
    if (!tsr_index_input(env, probe, pos, GGML_TYPE_I32, imrope ? 4ull * T : T, "positions", &ph, why)) {
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
                            (mode == GGML_ROPE_TYPE_NEOX || imrope) ? opl::rope::MODE_NEOX : 0u };
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
    if (tsr_is_leaf(s) && tsr_foreign_buffer(env, s)) {
        return tsr_fail(why, "GET_ROWS reads '%s', which is in another backend's memory", tsr_base(s)->name);
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
    // longer ubatch's FLASH_ATTN stays on the CPU (W3b fix; SW-43).
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

// CPY (f32 -> f32, into src1) and CONT (f32, a new contiguous tensor): whole rows. A copy whose two sides
// have different row lengths but the same number of values is regrouped into the source's rows when the
// destination is one dense run (a CONT of a head-strided view, the Gated DeltaNet state into its cache
// cell): the rows are then the source's ne0 values, which are whole lines.
static bool tsr_lower_cpy(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * s = node->src[0];
    if (s == nullptr || s->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || ggml_nelements(s) != ggml_nelements(node)) {
        return tsr_fail(why, "%s: F32 -> F32 with the same number of values only", ggml_op_name(node->op));
    }
    if (node->view_src != nullptr && tsr_foreign_buffer(env, node)) {
        return tsr_fail(why, "%s writes '%s', which is in another backend's memory", ggml_op_name(node->op), tsr_base(node)->name);
    }
    tsr_rows D, S;
    if (s->ne[0] == node->ne[0] && ggml_nrows(s) == ggml_nrows(node)) {
        if (!tsr_out_rows(env, probe, node, "destination", &D, why) || !tsr_f32_rows(env, probe, s, "source", &S, why)) {
            return false;
        }
        r->kind = opl::K_CPY;
        r->n_args = opl::cpy::NARGS;
        const opl::cpy_args w{ D.h, D.stride, S.h, S.stride, tsr_l16(D.elems), (uint32_t) D.rows };
        opl::put_args(w, r->args);
        return true;
    }
    if (!ggml_is_contiguous(node) || (s->ne[0] * 4) % TSR_LINE_BYTES != 0) {
        return tsr_fail(why, "%s: different row lengths need a dense destination and a source whose rows are whole lines",
                        ggml_op_name(node->op));
    }
    if (!tsr_f32_rows(env, probe, s, "source", &S, why)) {
        return false;
    }
    const uint32_t rl = (uint32_t) (s->ne[0] * 4 / TSR_LINE_BYTES);
    if (node->view_src != nullptr && (ggml_nelements(node) * 4) % TSR_LINE_BYTES != 0) {
        return tsr_fail(why, "%s: the destination view ends inside a line", ggml_op_name(node->op));
    }
    uint32_t dh = 0;
    if (!tsr_addr(env, probe, node, "destination", &dh, why)) {
        return false;
    }
    r->kind = opl::K_CPY;
    r->n_args = opl::cpy::NARGS;
    const opl::cpy_args w{ dh, rl, S.h, S.stride, rl, (uint32_t) S.rows };
    opl::put_args(w, r->args);
    return true;
}

// A contiguous F32 tensor as one row of all its elements: the element-wise kinds (UNARY, SOFTPLUS) are
// indifferent to the shape, so a [1, H, T] gate tensor is one row of H·T values (its rows of one
// element would break the 64-byte row rule). out: the result (a view must end on a line, as tsr_out_rows).
static bool tsr_flat_rows(const tsr_build_env & env, bool probe, const ggml_tensor * t, const char * what, bool out,
                          tsr_rows * o, std::string * why) {
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) <= 0) {
        return tsr_fail(why, "%s '%s' is not a non-empty contiguous F32 tensor", what, t->name);
    }
    o->rows = 1;
    o->elems = (uint64_t) ggml_nelements(t);
    o->stride = tsr_l16(o->elems);
    if (out && t->view_src != nullptr && (o->elems * 4) % TSR_LINE_BYTES != 0) {
        return tsr_fail(why, "%s '%s' is a view whose row ends inside a line", what, t->name);
    }
    return tsr_addr(env, probe, t, what, &o->h, why);
}

// The source and result rows of an element-wise node: rows of ne0 when they satisfy the 64-byte row
// rule, else one flat row (contiguous tensors).
static bool tsr_ew_operands(const tsr_build_env & env, bool probe, const ggml_tensor * node, const ggml_tensor * s, tsr_rows * D,
                            tsr_rows * S, std::string * why) {
    std::string w1;
    if (tsr_out_rows(env, probe, node, "result", D, &w1) && tsr_f32_rows(env, probe, s, "source", S, &w1)) {
        return true;
    }
    if (tsr_flat_rows(env, probe, node, "result", true, D, nullptr) && tsr_flat_rows(env, probe, s, "source", false, S, nullptr)) {
        return true;
    }
    return tsr_fail(why, "%s", w1.c_str());
}

static bool tsr_lower_unary(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const enum ggml_unary_op op = ggml_get_unary_op(node);
    uint32_t func = 0;
    bool softplus = false;
    if (op == GGML_UNARY_OP_TANH) {
        func = opl::unary::TANH;
    } else if (op == GGML_UNARY_OP_GELU) {
        func = opl::unary::GELU16;    // llama.cpp's GELU: ggml-cpu's fp16 table form (SW-31)
    } else if (op == GGML_UNARY_OP_SILU) {
        func = opl::unary::SILU;
    } else if (op == GGML_UNARY_OP_SIGMOID) {
        func = opl::unary::SIGMOID;
    } else if (op == GGML_UNARY_OP_SOFTPLUS) {
        softplus = true;              // composed from VU functions (SW §7.9, GDN-3)
    } else {
        return tsr_fail(why, "UNARY %s is not on the device (TANH, GELU, SILU, SIGMOID, SOFTPLUS)", ggml_unary_op_name(op));
    }
    const ggml_tensor * s = node->src[0];
    if (s == nullptr || !ggml_are_same_shape(s, node)) {
        return tsr_fail(why, "UNARY: source and result differ in shape");
    }
    tsr_rows D, S;
    if (!tsr_ew_operands(env, probe, node, s, &D, &S, why)) {
        return false;
    }
    if (softplus) {
        r->kind = opl::K_SOFTPLUS;
        r->n_args = opl::softplus::NARGS;
        const uint32_t w[opl::softplus::NARGS] = { D.h, D.stride, S.h, S.stride, (uint32_t) D.elems, (uint32_t) D.rows };
        memcpy(r->args, w, sizeof(w));
        return true;
    }
    r->kind = opl::K_UNARY;
    r->n_args = opl::unary::NARGS;
    const opl::unary_args w{ D.h, D.stride, S.h, S.stride, (uint32_t) D.elems, (uint32_t) D.rows, func };
    opl::put_args(w, r->args);
    return true;
}

// GATED_DELTA_NET (ggml_gated_delta_net, scalar gate, K = 1): q, k [S, H_k, T, n_seqs], v [S, H_v, T, n_seqs],
// g, beta [1, H_v, T, n_seqs], the state [S, S, H_v, n_seqs]; the result [S H_v, T n_seqs + S n_seqs] holds the
// attention rows (S H_v per token) and then the new state (SW §7.9, opl::gdn). Token rows of q, k and v keep
// the heads S values apart or more (the record has a head stride), the tokens (and sequences) a constant
// stride; the gate rows are H_v values, so H_v is a multiple of 16 for more than one token.
static bool tsr_gdn_token_rows(const tsr_build_env & env, bool probe, const ggml_tensor * t, const char * what, uint32_t S,
                               uint32_t H, uint32_t * h, uint32_t * hs, uint32_t * ts, std::string * why) {
    if (t->type != GGML_TYPE_F32 || t->nb[0] != 4 || t->ne[0] != (int64_t) S || t->ne[1] != (int64_t) H) {
        return tsr_fail(why, "%s '%s' is not F32 [%u, %u, T, n_seqs] with contiguous rows", what, t->name, S, H);
    }
    if ((H > 1 && (t->nb[1] % TSR_LINE_BYTES != 0 || t->nb[1] < 4ull * S)) ||
        (t->ne[2] * t->ne[3] > 1 && t->nb[2] % TSR_LINE_BYTES != 0) ||
        (t->ne[2] > 1 && t->ne[3] > 1 && (uint64_t) t->nb[3] != (uint64_t) t->nb[2] * (uint64_t) t->ne[2])) {
        return tsr_fail(why, "%s '%s': heads / tokens are not 64-byte spaced rows (nb1 %zu, nb2 %zu, nb3 %zu)", what, t->name,
                        t->nb[1], t->nb[2], t->nb[3]);
    }
    *hs = H > 1 ? (uint32_t) (t->nb[1] / TSR_LINE_BYTES) : tsr_l16(S);
    *ts = t->ne[2] * t->ne[3] > 1 ? (uint32_t) (t->nb[2] / TSR_LINE_BYTES) : (uint32_t) ((H * 4ull * S + 63) / 64);
    return tsr_addr(env, probe, t, what, h, why);
}

static bool tsr_lower_gdn(const tsr_build_env & env, bool probe, const ggml_tensor * node, tsr_rec * r, std::string * why) {
    const ggml_tensor * q = node->src[0], * k = node->src[1], * v = node->src[2], * g = node->src[3], * b = node->src[4],
                      * s = node->src[5];
    if (!q || !k || !v || !g || !b || !s) {
        return tsr_fail(why, "GATED_DELTA_NET without all six operands");
    }
    if (node->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32) {
        return tsr_fail(why, "GATED_DELTA_NET: operands are not F32");
    }
    if (tsr_param_i32(node, 0) != 1) {
        return tsr_fail(why, "GATED_DELTA_NET with K = %d state snapshots (rollback): only the final state", tsr_param_i32(node, 0));
    }
    if (g->ne[0] != 1) {
        return tsr_fail(why, "GATED_DELTA_NET with a per-row gate (KDA) is not on the device");
    }
    const int64_t Sv = v->ne[0], Hv = v->ne[1], T = v->ne[2], ns = v->ne[3], Hk = q->ne[1];
    if (Sv <= 0 || (Sv & 15) != 0 || Sv > opl::gdn::S_V_MAX || Hk <= 0 || Hv % Hk != 0 || T <= 0 || ns <= 0 ||
        k->ne[0] != Sv || k->ne[1] != Hk || q->ne[2] != T || k->ne[2] != T || q->ne[3] != ns || k->ne[3] != ns ||
        q->ne[0] != Sv) {
        return tsr_fail(why, "GATED_DELTA_NET shape: S_v = %lld (a multiple of 16 up to %u), H_k = %lld, H_v = %lld, T = %lld",
                        (long long) Sv, opl::gdn::S_V_MAX, (long long) Hk, (long long) Hv, (long long) T);
    }
    if (!ggml_is_contiguous(g) || !ggml_is_contiguous(b) || g->ne[1] != Hv || b->ne[1] != Hv || g->ne[2] != T || b->ne[2] != T ||
        g->ne[3] != ns || b->ne[3] != ns || b->ne[0] != 1 || (T * ns > 1 && (Hv & 15) != 0)) {
        return tsr_fail(why, "GATED_DELTA_NET: g / beta are not contiguous [1, H_v, T, n_seqs] with H_v a multiple of 16");
    }
    if (!ggml_is_contiguous(s) || s->ne[0] != Sv || s->ne[1] != Sv || s->ne[2] != Hv || s->ne[3] != ns) {
        return tsr_fail(why, "GATED_DELTA_NET: the state is not contiguous [S_v, S_v, H_v, n_seqs]");
    }
    if (!ggml_is_contiguous(node) || node->ne[0] != Sv * Hv || node->ne[1] != T * ns + Sv * ns || node->view_src != nullptr ||
        node->ne[2] * node->ne[3] != 1) {
        return tsr_fail(why, "GATED_DELTA_NET: the result is not one contiguous [S_v H_v, rows] tensor");
    }
    uint32_t qh = 0, qhs = 0, qts = 0, kh = 0, khs = 0, kts = 0, vh = 0, vhs = 0, vts = 0, gh = 0, bh = 0, sh = 0, dh = 0;
    if (!tsr_gdn_token_rows(env, probe, q, "q", (uint32_t) Sv, (uint32_t) Hk, &qh, &qhs, &qts, why) ||
        !tsr_gdn_token_rows(env, probe, k, "k", (uint32_t) Sv, (uint32_t) Hk, &kh, &khs, &kts, why) ||
        !tsr_gdn_token_rows(env, probe, v, "v", (uint32_t) Sv, (uint32_t) Hv, &vh, &vhs, &vts, why) ||
        !tsr_addr(env, probe, g, "g", &gh, why) || !tsr_addr(env, probe, b, "beta", &bh, why) ||
        !tsr_addr(env, probe, s, "state", &sh, why) || !tsr_addr(env, probe, node, "result", &dh, why)) {
        return false;
    }
    const uint32_t L = (uint32_t) Sv / 16u, nt = (uint32_t) (T * ns), Lg = tsr_l16((uint64_t) Hv);
    namespace gd = opl::gdn;
    r->kind = opl::K_GATED_DELTA_NET;
    r->n_args = gd::NARGS;
    uint32_t * a = r->args;
    a[gd::DST] = dh;
    a[gd::STATE_OUT] = probe ? 0u : dh + nt * (uint32_t) Hv * L;   // the state follows the attention rows, same channel
    a[gd::STATE_IN] = sh;
    a[gd::Q] = qh; a[gd::Q_HEAD_STRIDE] = qhs; a[gd::Q_TOKEN_STRIDE] = qts;
    a[gd::K] = kh; a[gd::K_HEAD_STRIDE] = khs; a[gd::K_TOKEN_STRIDE] = kts;
    a[gd::V] = vh; a[gd::V_HEAD_STRIDE] = vhs; a[gd::V_TOKEN_STRIDE] = vts;
    a[gd::G] = gh; a[gd::G_TOKEN_STRIDE] = Lg;
    a[gd::BETA] = bh; a[gd::BETA_TOKEN_STRIDE] = Lg;
    a[gd::S_V] = (uint32_t) Sv; a[gd::H_K] = (uint32_t) Hk; a[gd::H_V] = (uint32_t) Hv;
    a[gd::N_TOKENS] = (uint32_t) T; a[gd::N_SEQS] = (uint32_t) ns;
    a[gd::SCALE] = tsr_bits(1.0f / sqrtf((float) Sv));          // ggml-cpu: the output is scaled by 1 / sqrtf(S_v)
    return true;
}

// ---------------------------------------------------------------------------------------------
// SSM_CONV with its state (SW §7.9): the pinned graph builds  CONCAT(conv state, qkv^T) -> SSM_CONV -> SILU  and
// copies the last d_conv - 1 columns of the CONCAT back into the state cache. The device keeps the state as
// d_conv - 1 rows of C channels (the channel-vector layout of the kind; the cache tensors of a Tesseract buffer
// belong to the device). The group is three steps at three places of the graph, because ggml-alloc frees the
// CONCAT's inputs after the CONCAT and may give their memory to the nodes between it and the SSM_CONV:
//   - at the CONCAT: two CPY records fill the CONCAT's result buffer (kept alive until the SSM_CONV and the state
//     CPY use it) with the window of rows: the n_seqs·(d_conv - 1) state rows, then the n_seqs·T token rows;
//   - at the SSM_CONV: one SSM_CONV record reads the window, writes the convolution (with a following SILU
//     folded in) and the new state rows into the state cache; the state CPY is absorbed;
//   - nothing reads the CONCAT's result but the SSM_CONV and the state CPY.
// ---------------------------------------------------------------------------------------------
struct tsr_conv_parts {
    const ggml_tensor * concat = nullptr;   // CONCAT(state, x^T)
    const ggml_tensor * state = nullptr;    // [d_conv - 1, C, n_seqs]: the gathered state rows
    const ggml_tensor * x = nullptr;        // [C, T, n_seqs]: the tokens' channel rows
    const ggml_tensor * w = nullptr;        // [d_conv, C] taps
    uint32_t dc = 0, C = 0, T = 0, ns = 0;
};

// Where the CONCAT's conv state comes from: the GET_ROWS of a state cache (through the reshape and view of
// build_rs). The cache must be in a Tesseract buffer; a CPU cache (-nkvo) keeps the whole group on the CPU.
static bool tsr_conv_state_in_device(const tsr_build_env & env, const ggml_tensor * concat) {
    const ggml_tensor * t = concat->src[0];
    while (t != nullptr && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->src[0] != nullptr) {
        t = t->src[0];
    }
    if (t == nullptr || t->op != GGML_OP_GET_ROWS || t->src[0] == nullptr) {
        return true;       // not the build_rs form: the later checks decide
    }
    return !tsr_foreign_buffer(env, t->src[0]);
}

static bool tsr_conv_concat_ok(const ggml_tensor * concat, tsr_conv_parts * p, std::string * why) {
    if (concat == nullptr || concat->op != GGML_OP_CONCAT || concat->type != GGML_TYPE_F32 || tsr_param_i32(concat, 0) != 0) {
        return tsr_fail(why, "SSM_CONV: its input is not a CONCAT along the time axis");
    }
    const ggml_tensor * st = concat->src[0];
    const ggml_tensor * tr = concat->src[1];
    if (st == nullptr || tr == nullptr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32 || !ggml_is_contiguous(st) ||
        tr->op != GGML_OP_TRANSPOSE || tr->src[0] == nullptr || concat->ne[3] != 1 || st->ne[1] != concat->ne[1] ||
        st->ne[2] != concat->ne[2] || st->ne[3] != 1 || tr->ne[0] + st->ne[0] != concat->ne[0] || tr->ne[1] != concat->ne[1] ||
        tr->ne[2] != concat->ne[2]) {
        return tsr_fail(why, "SSM_CONV: the CONCAT is not (a contiguous conv state, a transposed token matrix)");
    }
    const ggml_tensor * x = tr->src[0];
    if (x->type != GGML_TYPE_F32 || x->ne[0] != concat->ne[1] || x->ne[1] != tr->ne[0] || x->ne[2] != concat->ne[2] ||
        x->ne[3] != 1) {
        return tsr_fail(why, "SSM_CONV: the transposed operand is not [C, T, n_seqs]");
    }
    if (p != nullptr) {
        p->concat = concat;
        p->state = st;
        p->x = x;
        p->dc = (uint32_t) st->ne[0] + 1u;
        p->C = (uint32_t) concat->ne[1];
        p->T = (uint32_t) tr->ne[0];
        p->ns = (uint32_t) concat->ne[2];
    }
    return true;
}

static bool tsr_conv_parts_of(const ggml_tensor * conv, tsr_conv_parts * p, std::string * why) {
    if (conv->type != GGML_TYPE_F32 || conv->src[1] == nullptr || conv->src[1]->type != GGML_TYPE_F32) {
        return tsr_fail(why, "SSM_CONV: F32 operands only");
    }
    if (!tsr_conv_concat_ok(conv->src[0], p, why)) {
        return false;
    }
    const ggml_tensor * w = conv->src[1];
    if (w->ne[0] != (int64_t) p->dc || w->ne[1] != (int64_t) p->C || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) ||
        conv->ne[0] != (int64_t) p->C || conv->ne[1] != (int64_t) p->T || conv->ne[2] != (int64_t) p->ns || conv->ne[3] != 1 ||
        !ggml_is_contiguous(conv) || p->dc < 2 || p->dc > opl::sconv::D_CONV_MAX || (p->C & 15u) != 0) {
        return tsr_fail(why, "SSM_CONV shape: d_conv %u (2 .. 8), C %u (a multiple of 16), T %u, n_seqs %u", p->dc, p->C, p->T, p->ns);
    }
    p->w = w;
    return true;
}

// The state CPY of the group: copies the last d_conv - 1 columns of the CONCAT's result (a view of it) into
// d_conv - 1 rows of C values per sequence of a state cache.
static bool tsr_conv_state_cpy_ok(const ggml_tensor * cpy, const ggml_tensor ** concat) {
    const ggml_tensor * v = cpy->src[0];
    if (cpy->op != GGML_OP_CPY || v == nullptr || v->op != GGML_OP_VIEW || v->view_src == nullptr ||
        v->view_src->op != GGML_OP_CONCAT || v->type != GGML_TYPE_F32 || cpy->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * c = v->view_src;
    if (!tsr_conv_concat_ok(c, nullptr, nullptr)) {
        return false;
    }
    // the last d_conv - 1 columns of every channel row of the CONCAT
    const int64_t keep = v->ne[0];
    if (v->ne[1] != c->ne[1] || v->ne[2] != c->ne[2] || v->ne[3] != 1 || keep != c->src[0]->ne[0] ||
        v->view_offs != (size_t) ((c->ne[0] - keep) * 4) || (size_t) v->nb[1] != (size_t) c->nb[1] || v->nb[2] != c->nb[2] ||
        ggml_nelements(cpy) != ggml_nelements(v) || !ggml_is_contiguous(cpy)) {
        return false;
    }
    if (concat != nullptr) {
        *concat = c;
    }
    return true;
}

// The window of a conv group, at the CONCAT: recs[0] copies the n_seqs·(d_conv - 1) state rows into the CONCAT's
// result buffer, recs[1] the n_seqs·T token rows after them. *n = 2.
bool tsr_lower_conv_window(const tsr_build_env & env, bool probe, const ggml_tensor * concat, tsr_rec * recs, uint32_t * n,
                           std::string * why) {
    tsr_conv_parts p;
    if (!tsr_conv_concat_ok(concat, &p, why)) {
        return false;
    }
    if (!tsr_conv_state_in_device(env, concat) || !tsr_kind_in_image(env.kinds, opl::K_SSM_CONV) ||
        !tsr_kind_in_image(env.kinds, opl::K_CPY)) {
        return tsr_fail(why, "CONCAT of a conv state: SSM_CONV or CPY is not in the booted image (%s) or the state cache is "
                        "not in Tesseract memory", env.image.c_str());
    }
    if ((p.C & 15u) != 0 || p.dc < 2 || p.dc > opl::sconv::D_CONV_MAX) {
        return tsr_fail(why, "CONCAT of a conv state: d_conv %u (2 .. 8), C %u (a multiple of 16)", p.dc, p.C);
    }
    const uint32_t Lc = p.C / 16u;
    tsr_rows X;
    uint32_t wh = 0, sh = 0;
    if (!tsr_addr(env, probe, concat, "conv window", &wh, why) || !tsr_addr(env, probe, p.state, "state", &sh, why) ||
        !tsr_f32_rows(env, probe, p.x, "tokens", &X, why)) {
        return false;
    }
    const uint64_t nt = uint64_t(p.T) * p.ns;
    if (nt > 1 && (X.rows != nt || X.stride < Lc)) {
        return tsr_fail(why, "CONCAT of a conv state: the token rows are not a single run of T·n_seqs rows");
    }
    const uint32_t srows = (p.dc - 1u) * p.ns;
    recs[0] = tsr_rec();
    recs[0].kind = opl::K_CPY;
    recs[0].n_args = opl::cpy::NARGS;
    const opl::cpy_args w0{ wh, Lc, sh, Lc, Lc, srows };
    opl::put_args(w0, recs[0].args);
    recs[1] = tsr_rec();
    recs[1].kind = opl::K_CPY;
    recs[1].n_args = opl::cpy::NARGS;
    const opl::cpy_args w1{ wh + srows * Lc, Lc, X.h, nt > 1 ? X.stride : Lc, Lc, (uint32_t) nt };
    opl::put_args(w1, recs[1].args);
    *n = 2;
    for (uint32_t i = 0; i < 2; ++i) {
        if (!tsr_plan_ok(env, recs[i], why)) {
            return false;
        }
    }
    return true;
}

// The fused record. conv: the SSM_CONV node; silu: the SILU node that follows it (or nullptr); cpy: the state
// CPY node (or nullptr in probe mode, supports_op asks the SSM_CONV alone).
bool tsr_lower_conv_group(const tsr_build_env & env, bool probe, const ggml_tensor * conv, const ggml_tensor * silu,
                          const ggml_tensor * cpy, tsr_rec * r, std::string * why) {
    tsr_conv_parts p;
    if (!tsr_conv_parts_of(conv, &p, why)) {
        return false;
    }
    if (!probe && cpy == nullptr) {
        return tsr_fail(why, "SSM_CONV: no state CPY in the group (the record would not write the state)");
    }
    if (!tsr_conv_state_in_device(env, p.concat) || tsr_foreign_buffer(env, p.w)) {
        return tsr_fail(why, "SSM_CONV: its state cache or taps are in another backend's memory");
    }
    const ggml_tensor * out = silu != nullptr ? silu : conv;
    if (silu != nullptr && (silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != conv ||
                            !ggml_are_same_shape(silu, conv))) {
        return tsr_fail(why, "SSM_CONV + SILU: not a SILU of the convolution");
    }
    tsr_rows D;
    uint32_t win = 0, sout = 0, wh = 0;
    if (!tsr_out_rows(env, probe, out, "result", &D, why) || !tsr_addr(env, probe, p.concat, "conv window", &win, why) ||
        !tsr_addr(env, probe, p.w, "taps", &wh, why)) {
        return false;
    }
    if (p.T * p.ns > 1 && D.rows != uint64_t(p.T) * p.ns) {
        return tsr_fail(why, "SSM_CONV: the token rows are not a single run of T·n_seqs rows");
    }
    const uint32_t Lc = p.C / 16u;
    if (p.T * p.ns > 1 && D.stride < Lc) {
        return tsr_fail(why, "SSM_CONV: overlapping result rows");
    }
    if (cpy != nullptr) {
        if (!tsr_conv_state_cpy_ok(cpy, nullptr) || cpy->src[0]->view_src != p.concat ||
            !tsr_addr(env, probe, cpy, "state cache", &sout, why)) {
            return tsr_fail(why, "SSM_CONV: the state CPY is not the CONCAT's last columns into a contiguous run of cache rows");
        }
        if (cpy->view_src != nullptr && (ggml_nelements(cpy) * 4) % TSR_LINE_BYTES != 0) {
            return tsr_fail(why, "SSM_CONV: the state cache run ends inside a line");
        }
    }
    namespace c = opl::sconv;
    r->kind = opl::K_SSM_CONV;
    r->n_args = c::NARGS;
    uint32_t * a = r->args;
    a[c::DST] = D.h; a[c::DST_TOKEN_STRIDE] = p.T * p.ns > 1 ? D.stride : Lc;
    a[c::X] = win + (p.dc - 1u) * p.ns * Lc; a[c::X_TOKEN_STRIDE] = Lc;      // the window: state rows, then token rows
    a[c::STATE_IN] = win; a[c::STATE_OUT] = sout; a[c::W] = wh;
    a[c::N_CH] = p.C; a[c::D_CONV] = p.dc; a[c::N_TOKENS] = p.T; a[c::N_SEQS] = p.ns;
    a[c::FLAGS] = silu != nullptr ? c::FLAG_SILU : 0u;
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
        case GGML_OP_GATED_DELTA_NET: ok = tsr_lower_gdn(env, probe, node, out, why); break;
        case GGML_OP_SSM_CONV:       ok = tsr_lower_conv_group(env, probe, node, nullptr, nullptr, out, why); break;
        default:
            return tsr_fail(why, "op %s has no device kind", ggml_op_desc(node));
    }
    if (ok && !tsr_kind_in_image(env.kinds, out->kind)) {   // SW §6.9: a kind the booted image lacks
        return tsr_fail(why, "kind %s is not in the booted image (%s)", tsr_kind_name(out->kind), env.image.c_str());
    }
    return ok && tsr_plan_ok(env, *out, why);
}

bool tsr_build_supports(const tsr_build_env & env, const ggml_tensor * op, std::string * why) {
    if (tsr_node_is_noop(op)) {
        return true;
    }
    if (ggml_is_empty(op)) {
        return true;     // an empty node (llama.cpp's recurrent-state copies of zero rows) does nothing; graph_compute skips it
    }
    if (op->op == GGML_OP_DUP) {
        return tsr_fail(why, "DUP stays on the CPU (SW §8 R1)");
    }
    if (env.pl.caps == 0 || env.pl.num_ch == 0) {
        return tsr_fail(why, "the device configuration has no VU / AT parameters");
    }
    // The two nodes of an SSM_CONV group that have no record of their own (the group is the SSM_CONV's):
    // the CONCAT of the conv state with the tokens and the CPY of its last columns into the state cache.
    if (op->op == GGML_OP_CONCAT && tsr_conv_concat_ok(op, nullptr, nullptr)) {
        tsr_rec recs[2];
        uint32_t nr = 0;
        return tsr_lower_conv_window(env, true, op, recs, &nr, why);
    }
    if (op->op == GGML_OP_CPY && tsr_conv_state_cpy_ok(op, nullptr)) {
        return (tsr_kind_in_image(env.kinds, opl::K_SSM_CONV) && !tsr_foreign_buffer(env, op)) ||
               tsr_fail(why, "conv state CPY: SSM_CONV is not in the booted image (%s) or the cache is not in Tesseract memory",
                        env.image.c_str());
    }
    // llama.cpp asks, before it places an SSM_CONV weight, whether the device runs ggml_ssm_conv(conv_x, w) on a
    // plain F32 [d_conv - 1 + 512, C, 3] input (llama-model-loader.cpp weight_buft_supported). A graph's SSM_CONV
    // has the CONCAT and the state copy in front of it (tsr_lower_conv_group); the weight is stored as tap rows.
    if (op->op == GGML_OP_SSM_CONV && op->src[0] != nullptr && op->src[0]->op == GGML_OP_NONE && op->src[0]->data == nullptr &&
        op->src[1] != nullptr && op->src[1]->type == GGML_TYPE_F32 && op->src[1]->ne[0] >= 2 &&
        op->src[1]->ne[0] <= (int64_t) opl::sconv::D_CONV_MAX && op->src[1]->ne[1] % 16 == 0 &&
        op->src[0]->ne[0] == op->src[1]->ne[0] - 1 + 512) {
        return tsr_kind_in_image(env.kinds, opl::K_SSM_CONV) ||
               tsr_fail(why, "kind SSM_CONV is not in the booted image (%s)", env.image.c_str());
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

static bool tsr_mm_epilogue_supported(const tsr_build_env & env, uint32_t kind) {
    const uint32_t cap = kind == opl::K_MUL_MAT_SCALE ? tsr::plan::CAPS_MVM_LOCAL_SCALE :
                                                      tsr::plan::CAPS_MVM_LOCAL_SOFTCAP;
    return env.cfg.num_ch == 4u && (env.cfg.caps & cap) && tsr_kind_in_image(env.kinds, kind);
}

static bool tsr_mm_scale_rec(const tsr_build_env & env, const ggml_tensor * scale, const uint32_t * a, tsr_rec * r) {
    namespace m = opl::mm;
    namespace sc = opl::scl;
    return scale->op == GGML_OP_SCALE && !scale->view_src && tsr_lower_scale(env, false, scale, r, nullptr) &&
           r->args[sc::B] == 0u && r->args[sc::ROWS] == a[m::T] && r->args[sc::N] == a[m::N];
}

static bool tsr_mm_epilogue_dst(const tsr_rec & r, uint32_t * a) {
    namespace m = opl::mm;
    namespace sc = opl::scl;
    const uint32_t y = r.args[sc::DST];
    const auto x_range = tsr_rows_range(tsr_handle_ch(a[m::X]), tsr_handle_byte(a[m::X]), a[m::T],
                                       uint64_t(a[m::K]) * 4u, uint64_t(a[m::X_STRIDE]) * TSR_LINE_BYTES);
    const auto y_range = tsr_rows_range(tsr_handle_ch(y), tsr_handle_byte(y), a[m::T],
                                       uint64_t(tsr_l16(a[m::N])) * TSR_LINE_BYTES,
                                       uint64_t(r.args[sc::DST_STRIDE]) * TSR_LINE_BYTES);
    if (tsr_ranges_overlap(x_range, y_range)) return false;
    a[m::Y] = y;
    a[m::Y_STRIDE] = r.args[sc::DST_STRIDE];
    return true;
}

bool tsr_fuse_mm_scale(const tsr_build_env & env, const ggml_tensor * mm, const ggml_tensor * scale,
                       uint32_t * a) {
    tsr_rec r;
    if (scale->src[0] != mm || !tsr_mm_epilogue_supported(env, opl::K_MUL_MAT_SCALE) ||
        !tsr_mm_scale_rec(env, scale, a, &r) || !tsr_mm_epilogue_dst(r, a))
        return false;
    a[opl::mm_scale::SCALE] = r.args[opl::scl::S];
    return true;
}

static bool tsr_is_tanh(const ggml_tensor * t) {
    return t && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_TANH;
}

static bool tsr_softcap_weight_work(const tsr_build_env & env, const uint32_t * a) {
    namespace m = opl::mm;
    tsr::plan::mm_tiling tile{};
    if (tsr::plan::mm_plan(env.cfg, a[m::FMT], a[m::N], a[m::K], a[m::T], &tile) != opl::RET_OK) return false;
    // Three stages cost about two ideal issue cycles per owner row and
    // token. Prefer a sufficiently long weight stream to overlap that work.
    // This is an admission heuristic, not a timing estimate.
    return tsr_row_beats(a[m::FMT], a[m::K]) >= 32u * tile.t_tile;
}

bool tsr_fuse_mm_softcap(const tsr_build_env & env, const ggml_tensor * mm, const ggml_tensor * first,
                         const ggml_tensor * last, uint32_t * a) {
    const ggml_tensor * tanh = last->src[0];
    tsr_rec r0, r1;
    if (first->src[0] != mm || !tsr_is_tanh(tanh) || tanh->src[0] != first ||
        !tsr_mm_epilogue_supported(env, opl::K_MUL_MAT_SOFTCAP) || !tsr_softcap_weight_work(env, a) ||
        !tsr_mm_scale_rec(env, first, a, &r0) || !tsr_mm_scale_rec(env, last, a, &r1) ||
        !tsr_mm_epilogue_dst(r1, a))
        return false;
    a[opl::mm_scale::SCALE] = r0.args[opl::scl::S];
    a[opl::mm_scale::POST_SCALE] = r1.args[opl::scl::S];
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
static bool tsr_can_hoist(const ggml_cgraph * cg, int i, int k, const ggml_tensor * other, const ggml_tensor * out,
                           int skip0 = -1, int skip1 = -1) {
    const tsr_vrange ob = other ? tsr_bytes_of(other) : tsr_vrange{}, rb = tsr_bytes_of(out);
    if ((other && ob.lo == 0) || rb.lo == 0) {
        return false;
    }
    for (int j = i + 1; j < k; ++j) {
        if (j == skip0 || j == skip1) continue;
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

static int tsr_fusion_user(const ggml_cgraph * cg, int i) {
    for (int j = i + 1; j < cg->n_nodes; ++j) {
        for (const ggml_tensor * src : cg->nodes[j]->src)
            if (src == cg->nodes[i]) return j;
    }
    return -1;
}

tsr_fusion tsr_build_fusion(const tsr_build_env & env, const ggml_cgraph * cg) {
    tsr_fusion f;
    const int n = cg->n_nodes;
    f.with.assign(n, -1);
    f.into.assign(n, -1);
    f.softcap_mid.assign(n, -1);
    f.softcap_tail.assign(n, -1);
    f.conv_concat.assign(n, -1);
    f.conv_cpy.assign(n, -1);
    f.conv_silu.assign(n, -1);
    for (int i = 0; i < n; ++i) {            // SSM_CONV groups (tsr_lower_conv_group)
        const ggml_tensor * node = cg->nodes[i];
        if (node->op != GGML_OP_SSM_CONV || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !tsr_kind_in_image(env.kinds, opl::K_SSM_CONV)) {
            continue;
        }
        tsr_conv_parts p;
        if (!tsr_conv_parts_of(node, &p, nullptr)) {
            continue;
        }
        int ic = -1, icpy = -1, isilu = -1;
        for (int j = i - 1; j >= 0 && ic < 0; --j) {
            if (cg->nodes[j] == p.concat) ic = j;
        }
        for (int j = ic + 1; j < n && ic >= 0; ++j) {
            const ggml_tensor * u = cg->nodes[j];
            if (u->op == GGML_OP_CPY && u->src[0] != nullptr && u->src[0]->view_src == p.concat) {
                if (icpy >= 0) {
                    icpy = -2;                 // two state copies: not the pinned graph
                    break;
                }
                icpy = j;
            }
        }
        if (ic < 0 || icpy < 0 || icpy > i || !ggml_node_has_n_uses(cg, ic, 2) || !tsr_conv_state_cpy_ok(cg->nodes[icpy], nullptr) ||
            (cg->nodes[icpy]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        // the state CPY moves down to the SSM_CONV: no node in between touches the cache run it writes
        const tsr_vrange rb = tsr_bytes_of(cg->nodes[icpy]);
        bool clash = rb.lo == 0;
        for (int j = icpy + 1; j < i && !clash; ++j) {
            const ggml_tensor * q = cg->nodes[j];
            if ((q->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || tsr_node_is_noop(q)) continue;
            clash = tsr_vr_overlap(tsr_bytes_of(q), rb);
            for (int s = 0; s < GGML_MAX_SRC && !clash; ++s) {
                clash = q->src[s] != nullptr && tsr_vr_overlap(tsr_bytes_of(q->src[s]), rb);
            }
        }
        if (clash) continue;
        if (ggml_node_has_n_uses(cg, i, 1)) {
            const int k = tsr_fusion_user(cg, i);
            const ggml_tensor * u = k >= 0 ? cg->nodes[k] : nullptr;
            if (u != nullptr && (u->flags & GGML_TENSOR_FLAG_COMPUTE) && u->op == GGML_OP_UNARY &&
                ggml_get_unary_op(u) == GGML_UNARY_OP_SILU && u->src[0] == node && f.into[k] < 0 && tsr_can_hoist(cg, i, k, nullptr, u)) {
                isilu = k;
            }
        }
        f.conv_concat[i] = ic;
        f.conv_cpy[i] = icpy;
        f.conv_silu[i] = isilu;
        f.into[ic] = f.into[icpy] = i;
        if (isilu >= 0) f.into[isilu] = i;
        f.n_pairs++;
    }
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * node = cg->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || f.into[i] >= 0 ||
            (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_RMS_NORM) || !ggml_node_has_n_uses(cg, i, 1)) {
            continue;
        }
        const int k = tsr_fusion_user(cg, i);
        if (k < 0) {
            continue;    // its user is in another graph (another split, an observed node's next call)
        }
        const ggml_tensor * u = cg->nodes[k];
        if ((u->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || f.into[k] >= 0 || f.with[k] >= 0) {
            continue;
        }
        tsr_tloc l;
        if (node->op == GGML_OP_MUL_MAT && u->op == GGML_OP_SCALE) {
            if (u->src[0] != node || !tsr_can_hoist(cg, i, k, nullptr, u)) continue;
            if (tsr_mm_epilogue_supported(env, opl::K_MUL_MAT_SOFTCAP) && ggml_node_has_n_uses(cg, k, 1)) {
                const int mid = tsr_fusion_user(cg, k);
                if (mid >= 0 && tsr_is_tanh(cg->nodes[mid]) && cg->nodes[mid]->src[0] == u &&
                    (cg->nodes[mid]->flags & GGML_TENSOR_FLAG_COMPUTE) && f.into[mid] < 0 && f.with[mid] < 0 &&
                    ggml_node_has_n_uses(cg, mid, 1)) {
                    const int tail = tsr_fusion_user(cg, mid);
                    if (tail >= 0 && cg->nodes[tail]->op == GGML_OP_SCALE && cg->nodes[tail]->src[0] == cg->nodes[mid] &&
                        (cg->nodes[tail]->flags & GGML_TENSOR_FLAG_COMPUTE) && f.into[tail] < 0 && f.with[tail] < 0 &&
                        tsr_can_hoist(cg, i, mid, nullptr, cg->nodes[mid], k) &&
                        tsr_can_hoist(cg, i, tail, nullptr, cg->nodes[tail], k, mid)) {
                        f.softcap_mid[i] = mid;
                        f.softcap_tail[i] = tail;
                        f.into[mid] = f.into[tail] = i;
                    }
                }
            }
            if (f.softcap_tail[i] < 0 && !tsr_mm_epilogue_supported(env, opl::K_MUL_MAT_SCALE)) continue;
        } else if (node->op == GGML_OP_MUL_MAT) {
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

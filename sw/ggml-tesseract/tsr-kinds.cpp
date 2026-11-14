// tsr-kinds.cpp — see tsr-kinds.h
#include "tsr-kinds.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "tsr_ref.h"

namespace opl = tsr::opl;

namespace {

bool kfail(std::string * why, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
bool kfail(std::string * why, const char * fmt, ...) {
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

uint32_t bits(float f) { return tsr_f32_bits(f); }

// Fills *out from the words and checks it; an empty record on refusal.
bool finish(const tsr_kenv & env, tsr_rec * out, std::string * why) {
    if (!tsr_k_check(env, *out, why)) {
        *out = tsr_rec();
        return false;
    }
    return true;
}

bool rows_ok(const char * what, const tsr_k_rows & r, uint32_t rows, uint32_t row_lines, std::string * why) {
    if (r.h == opl::HANDLE_NONE) {
        return kfail(why, "%s: no handle", what);
    }
    if (rows > 1 && r.stride < row_lines) {
        return kfail(why, "%s: row stride %u lines is below the row's %u lines", what, r.stride, row_lines);
    }
    return true;
}

}  // namespace

uint64_t tsr_kinds_mask(const tsr_versions & v) {
    uint64_t m = 0;
    for (uint32_t k = 0; k < opl::N_KINDS && k < 64u; ++k) {
        if (v.kind_rev[k] != 0) {
            m |= 1ull << k;
        }
    }
    return m;
}

tsr_kenv tsr_kenv_make(const tsr_config & cfg, uint32_t pcap, uint64_t kinds, const char * image) {
    tsr_kenv e;
    e.cfg = tsr_plan_cfg(cfg);
    e.cfg.num_ch = cfg.num_ch;
    e.cfg.pcap = pcap;
    e.cfg.vl = cfg.vl;
    e.cfg.caps = cfg.caps;
    e.pl = tsr::plan::plan_limits_of(e.cfg);
    e.num_ch = cfg.num_ch;
    e.att = { cfg.num_ch, cfg.t_max, pcap, cfg.k_max };
    e.kinds = kinds;
    e.image = image != nullptr ? image : "";
    return e;
}

bool tsr_k_check(const tsr_kenv & env, const tsr_rec & r, std::string * why) {
    if (!env.has(r.kind)) {
        return kfail(why, "the booted image (%s) lacks the %s kind (or JOIN)", env.image.c_str(), opl::kind_name(r.kind));
    }
    if (r.kind == opl::K_MUL_MAT) {
        if (r.n_args != opl::mm::NARGS) {
            return kfail(why, "MUL_MAT record: %u words", r.n_args);
        }
        tsr::plan::mm_tiling t{};
        const uint32_t st = tsr::plan::mm_plan(tsr::plan::mm_limits{ env.pl.k_max, env.pl.t_max, env.pl.budget },
                                               r.args[opl::mm::FMT], r.args[opl::mm::N], r.args[opl::mm::K], r.args[opl::mm::T], &t);
        if (st != opl::RET_OK) {
            return kfail(why, "MUL_MAT record: the firmware's planner refuses it (%s)",
                         st == opl::RET_SCRATCH ? "SCRATCH: one token does not fit the SPM ring" : "BAD_RECORD");
        }
        return true;
    }
    const uint32_t e = opl::check_args(r.kind, r.args, r.n_args, tsr::plan::PRE_MAX_LINES);
    if (e != opl::RET_OK) {
        return kfail(why, "%s record: argument rules (opl::check_args) refuse it", opl::kind_name(r.kind));
    }
    tsr::plan::kind_plan kp{};
    if (!tsr::plan::plan_kind(env.pl, r.kind, r.args, r.n_args, &kp)) {
        return kfail(why, "%s record: the firmware's planner refuses it (%s)", opl::kind_name(r.kind),
                     kp.status == opl::RET_SCRATCH ? "SCRATCH: one tile does not fit the SPM ring" : "BAD_RECORD");
    }
    return true;
}

bool tsr_k_mm_deferrable(const tsr_kenv & env, const tsr_rec & mm) {
    if (mm.kind != opl::K_MUL_MAT || !env.has(opl::K_MUL_MAT_BEGIN) || !env.has(opl::K_MUL_MAT_END)) {
        return false;
    }
    tsr::plan::mm_tiling t{};
    return tsr::plan::mm_plan(tsr::plan::mm_limits{ env.pl.k_max, env.pl.t_max, env.pl.budget }, mm.args[opl::mm::FMT],
                              mm.args[opl::mm::N], mm.args[opl::mm::K], mm.args[opl::mm::T], &t) == opl::RET_OK &&
           t.n_tiles == 1u;
}

uint32_t tsr_k_emit(tsr_opl_builder & b, tsr_rec & r, int32_t tag, tsr_fanew_map * fanew, bool defer) {
    if (r.kind == opl::K_MUL_MAT) {
        return b.mul_mat(r.args, tag, opl::K_MUL_MAT, defer);
    }
    static const tsr_fanew_map none;
    if (r.kind == opl::K_SET_ROWS_KV && fanew != nullptr) {
        tsr_note_kv_write(r, fanew);
    }
    return tsr_encode(b, r, tag, fanew != nullptr ? *fanew : none);
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT
// ---------------------------------------------------------------------------------------------
bool tsr_k_mul_mat(const tsr_kenv & env, const tsr_kd_mm & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    namespace mm = opl::mm;
    if (!tsr_fmt_valid(d.fmt) || d.N == 0 || d.T == 0 || d.K == 0 || d.K % tsr_fmt_block_size(d.fmt) != 0) {
        return kfail(why, "MUL_MAT: format %u, N %u, K %u, T %u", d.fmt, d.N, d.K, d.T);
    }
    if (!rows_ok("MUL_MAT x", d.x, d.T, d.K / 16u, why) || !rows_ok("MUL_MAT y", d.y, d.T, tsr_k_lines(d.N), why)) {
        return false;
    }
    const bool has_add = d.add.h != opl::HANDLE_NONE;
    if (has_add && d.acc_dst) {
        return kfail(why, "MUL_MAT: an addend and ACC_DST together");
    }
    if (has_add && !rows_ok("MUL_MAT addend", d.add, d.add.stride == 0 ? 1u : d.T, tsr_k_lines(d.N), why)) {
        return false;
    }
    out->kind = opl::K_MUL_MAT;
    out->n_args = mm::NARGS;
    uint32_t * a = out->args;
    a[mm::X] = d.x.h;
    a[mm::X_STRIDE] = d.x.stride;
    a[mm::Y] = d.y.h;
    a[mm::Y_STRIDE] = d.y.stride;
    a[mm::W_OFF] = d.w_off;
    a[mm::FMT] = d.fmt;
    a[mm::N] = d.N;
    a[mm::K] = d.K;
    a[mm::T] = d.T;
    a[mm::ADD] = has_add ? d.add.h : opl::HANDLE_NONE;
    a[mm::ADD_STRIDE] = has_add ? d.add.stride : 0u;
    a[mm::FLAGS] = d.acc_dst ? mm::FLAG_ACC_DST : 0u;
    return finish(env, out, why);
}

// ---------------------------------------------------------------------------------------------
// ADD, MUL, SCALE
// ---------------------------------------------------------------------------------------------
bool tsr_k_add(const tsr_kenv & env, const tsr_kd_ew & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.kind != opl::K_ADD && d.kind != opl::K_MUL) {
        return kfail(why, "ADD / MUL encoder given kind %u", d.kind);
    }
    if (d.n == 0 || d.rows == 0 || d.b_period == 0) {
        return kfail(why, "%s: n %u, rows %u, b period %u", opl::kind_name(d.kind), d.n, d.rows, d.b_period);
    }
    const uint32_t L = tsr_k_lines(d.n);
    if (!rows_ok("a", d.a, d.rows, L, why) || !rows_ok("dst", d.dst, d.rows, L, why)) {
        return false;
    }
    if (d.b.h == opl::HANDLE_NONE) {
        return kfail(why, "%s: no b", opl::kind_name(d.kind));
    }
    uint32_t bstride = 0;
    if (!d.b_scalar && d.b_period > 1) {
        if (!rows_ok("b", d.b, d.b_period, L, why)) {
            return false;
        }
        bstride = d.b.stride;
    }
    out->kind = d.kind;
    out->n_args = opl::ew::NARGS;
    const opl::ew_args w{ d.dst.h, d.dst.stride, d.a.h, d.a.stride, d.b.h, bstride, d.n, d.rows, d.b_period,
                          d.b_scalar ? opl::ew::FLAG_B_SCALAR : 0u };
    opl::put_args(w, out->args);
    return finish(env, out, why);
}

bool tsr_k_scale(const tsr_kenv & env, const tsr_kd_scale & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.n == 0 || d.rows == 0) {
        return kfail(why, "SCALE: n %u, rows %u", d.n, d.rows);
    }
    const uint32_t L = tsr_k_lines(d.n);
    if (!rows_ok("src", d.src, d.rows, L, why) || !rows_ok("dst", d.dst, d.rows, L, why)) {
        return false;
    }
    out->kind = opl::K_SCALE;
    out->n_args = opl::scl::NARGS;
    const opl::scale_args w{ d.dst.h, d.dst.stride, d.src.h, d.src.stride, d.n, d.rows, bits(d.s), bits(d.b) };
    opl::put_args(w, out->args);
    return finish(env, out, why);
}

// ---------------------------------------------------------------------------------------------
// RMS_NORM, GLU, UNARY
// ---------------------------------------------------------------------------------------------
bool tsr_k_rms_norm(const tsr_kenv & env, const tsr_kd_rms & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.n == 0 || d.rows == 0) {
        return kfail(why, "RMS_NORM: n %u, rows %u", d.n, d.rows);
    }
    const uint32_t L = tsr_k_lines(d.n);
    if (!rows_ok("src", d.src, d.rows, L, why) || !rows_ok("dst", d.dst, d.rows, L, why)) {
        return false;
    }
    const bool has_w = d.weight != opl::HANDLE_NONE;
    if (has_w && d.w_period == 0) {
        return kfail(why, "RMS_NORM: a weight with period 0");
    }
    if (has_w && d.w_period > 1 && (d.n * 4u) % 64u != 0u) {
        return kfail(why, "RMS_NORM: per-row weights must be dense rows of whole lines (n %% 16 != 0)");
    }
    out->kind = opl::K_RMS_NORM;
    out->n_args = opl::rms::NARGS;
    // inv_n = 1 / n correctly rounded (tsr_fdiv), SW §7.2
    const opl::rms_norm_args w{ d.dst.h, d.dst.stride, d.src.h, d.src.stride, d.n, d.rows, bits(d.eps),
                                bits(tsr_fdiv(1.0f, (float) d.n)), d.weight, has_w ? d.w_period : 1u };
    opl::put_args(w, out->args);
    return finish(env, out, why);
}

bool tsr_k_glu(const tsr_kenv & env, const tsr_kd_glu & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.n == 0 || d.rows == 0) {
        return kfail(why, "GLU: n %u, rows %u", d.n, d.rows);
    }
    const uint32_t L = tsr_k_lines(d.n);
    if (!rows_ok("gate", d.gate, d.rows, L, why) || !rows_ok("up", d.up, d.rows, L, why) || !rows_ok("dst", d.dst, d.rows, L, why)) {
        return false;
    }
    out->kind = opl::K_GLU;
    out->n_args = opl::glu::NARGS;
    const opl::glu_args w{ d.dst.h, d.dst.stride, d.gate.h, d.gate.stride, d.up.h, d.up.stride, d.n, d.rows, d.variant };
    opl::put_args(w, out->args);
    return finish(env, out, why);
}

bool tsr_k_unary(const tsr_kenv & env, const tsr_kd_unary & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.n == 0 || d.rows == 0) {
        return kfail(why, "UNARY: n %u, rows %u", d.n, d.rows);
    }
    const uint32_t L = tsr_k_lines(d.n);
    if (!rows_ok("src", d.src, d.rows, L, why) || !rows_ok("dst", d.dst, d.rows, L, why)) {
        return false;
    }
    out->kind = opl::K_UNARY;
    out->n_args = opl::unary::NARGS;
    const opl::unary_args w{ d.dst.h, d.dst.stride, d.src.h, d.src.stride, d.n, d.rows, d.func };
    opl::put_args(w, out->args);
    return finish(env, out, why);
}

// ---------------------------------------------------------------------------------------------
// ROPE
// ---------------------------------------------------------------------------------------------
bool tsr_k_rope(const tsr_kenv & env, const tsr_kd_rope & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.n_heads == 0 || d.head_dim == 0 || d.T == 0 || d.n_dims == 0 || d.n_dims > d.head_dim) {
        return kfail(why, "ROPE: %u heads of %u, %u rotated dims, T %u", d.n_heads, d.head_dim, d.n_dims, d.T);
    }
    if (d.pos_src == opl::HANDLE_NONE) {
        return kfail(why, "ROPE: no positions");
    }
    const uint32_t H = (uint32_t) (((uint64_t) d.n_heads * d.head_dim + 15u) / 16u);
    if (!rows_ok("src", d.src, d.T, H, why) || !rows_ok("dst", d.dst, d.T, H, why)) {
        return false;
    }
    const bool atab = d.table != opl::HANDLE_NONE;
    out->kind = opl::K_ROPE;
    out->n_args = opl::rope::NARGS;
    // ATAB: the angles come from the table, TS is unused. Without a table: TS = powf(freq_base, -2 / n_dims) with the host's
    // libm, as ggml-cpu's theta_scale (SW §7.2).
    const uint32_t ts = atab ? 0u : bits(powf(d.freq_base, -2.0f / (float) d.n_dims));
    const uint32_t mode = (d.neox ? opl::rope::MODE_NEOX : 0u) | (atab ? opl::rope::MODE_ATAB : 0u);
    const opl::rope_args w{ d.dst.h, d.dst.stride, d.src.h, d.src.stride, 0u, d.table, ts, bits(d.attn_factor), bits(1.0f),
                            d.head_dim, d.n_dims, d.n_heads, d.T, mode };
    opl::put_args(w, out->args);
    out->pre.push_back({ opl::rope::POS, d.pos_src, opl::PRE_POS_BYTES * d.T });
    return finish(env, out, why);
}

// ---------------------------------------------------------------------------------------------
// CPY
// ---------------------------------------------------------------------------------------------
bool tsr_k_cpy(const tsr_kenv & env, const tsr_kd_cpy & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.row_lines == 0 || d.rows == 0) {
        return kfail(why, "CPY: %u lines, %u rows", d.row_lines, d.rows);
    }
    if (!rows_ok("src", d.src, d.rows, d.row_lines, why) || !rows_ok("dst", d.dst, d.rows, d.row_lines, why)) {
        return false;
    }
    out->kind = opl::K_CPY;
    out->n_args = opl::cpy::NARGS;
    const opl::cpy_args w{ d.dst.h, d.dst.stride, d.src.h, d.src.stride, d.row_lines, d.rows };
    opl::put_args(w, out->args);
    return finish(env, out, why);
}

// ---------------------------------------------------------------------------------------------
// KV planes: SET_ROWS_KV, FLASH_ATTN
// ---------------------------------------------------------------------------------------------
bool tsr_k_kv_geom(uint32_t E, uint32_t kv_size, uint32_t num_ch, tsr_k_kv * out, std::string * why) {
    *out = tsr_k_kv();
    if (E == 0 || E % 64u != 0 || E / 64u > 511u) {
        return kfail(why, "KV plane: E = %u values per cell is not a multiple of 64 up to 511 * 64", E);
    }
    if (kv_size == 0 || num_ch == 0 || num_ch > TSR_NUM_CH_MAX) {
        return kfail(why, "KV plane: %u cells over %u channels", kv_size, num_ch);
    }
    out->E = E;
    out->C = num_ch;
    out->LG = tsr_kv_lg_min(kv_size, num_ch);
    if (out->LG > 0xFFFFu) {     // the GEOM word holds LG in 16 bits
        return kfail(why, "KV plane: %u cells over %u channels does not fit (LG %u above 65535)", kv_size, num_ch, out->LG);
    }
    const tsr_kv_geom g = { E, num_ch, out->LG };
    out->bytes = tsr_kv_plane_beats(&g) * TSR_BEAT_BYTES;
    return true;
}

bool tsr_k_set_rows_kv(const tsr_kenv & env, const tsr_kd_set_rows & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if (d.n_rows == 0 || d.kv.E == 0 || d.kv.LG == 0 || d.which > opl::setrows::WHICH_V) {
        return kfail(why, "SET_ROWS_KV: %u rows, E %u, LG %u, plane kind %u", d.n_rows, d.kv.E, d.kv.LG, d.which);
    }
    if (d.kv.C != env.num_ch) {
        return kfail(why, "SET_ROWS_KV: the plane geometry has %u channels, the device %u", d.kv.C, env.num_ch);
    }
    if (d.idx_src == opl::HANDLE_NONE || d.plane % 64u != 0) {
        return kfail(why, "SET_ROWS_KV: no cell indices, or the plane is not 4 KiB aligned (beat %u)", d.plane);
    }
    if (!rows_ok("src", d.src, d.n_rows, tsr_k_lines(d.kv.E), why)) {
        return false;
    }
    out->kind = opl::K_SET_ROWS_KV;
    out->n_args = opl::setrows::NARGS;
    // words 0, 1 and 11 are informational (the plane's handle on channel 0, a row stride of 0, the stream), 8 to 10 the plane
    const opl::set_rows_kv_args w{ tsr_dram_handle(0, (uint64_t) d.plane * TSR_BEAT_BYTES), 0u, d.src.h, d.src.stride, 0u,
                                   d.n_rows, d.kv.E, d.which, d.plane, d.kv.geom_word(), 0u, 0u };
    opl::put_args(w, out->args);
    out->pre.push_back({ opl::setrows::IDX, d.idx_src, opl::PRE_KVIDX_BYTES * d.n_rows });
    out->kv_plane = d.plane;
    out->kv_idx_src = d.idx_src;
    out->kv_idx_n = d.n_rows;
    return finish(env, out, why);
}

bool tsr_k_fa_mask_build(const tsr_kenv & env, uint32_t E, uint32_t D, uint32_t G, float scale, uint32_t n_q, uint32_t n_kv,
                         uint32_t f32_h, const uint16_t * mask, size_t mask_stride, uint32_t pos0, tsr_k_fa_mask * out,
                         std::string * why) {
    *out = tsr_k_fa_mask();
    if (n_q == 0 || n_kv == 0 || E == 0 || D == 0 || G == 0) {
        return kfail(why, "FLASH_ATTN mask: %u rows, %u cells, E %u, D %u, G %u", n_q, n_kv, E, D, G);
    }
    if (!tsr::plan::attn::scale_ok(bits(scale))) {
        return kfail(why, "FLASH_ATTN scale %g is not finite, normal and positive", (double) scale);
    }
    std::vector<uint16_t> causal;
    if (mask == nullptr) {
        if ((uint64_t) pos0 + n_q > n_kv) {
            return kfail(why, "FLASH_ATTN causal mask: positions %u + %u rows exceed %u cells", pos0, n_q, n_kv);
        }
        causal.assign((size_t) n_q * n_kv, 0xFC00u);
        for (uint32_t q = 0; q < n_q; ++q) {
            for (uint32_t j = 0; j <= pos0 + q; ++j) {
                causal[(size_t) q * n_kv + j] = 0u;
            }
        }
        mask = causal.data();
        mask_stride = n_kv;
    }
    out->n_q = n_q;
    out->n_kv = n_kv;
    for (uint32_t q = 0; q < n_q && out->binary; ++q) {
        for (uint32_t j = 0; j < n_kv; ++j) {
            const uint16_t v = mask[(size_t) q * mask_stride + j];
            if (v != 0x0000u && v != 0xFC00u) {
                out->binary = false;
                break;
            }
        }
    }
    out->msum.assign(tsr_msum_words(n_q), 0u);
    const uint32_t flags = tsr_msum_build_f16(mask, mask_stride, n_kv, n_q, f32_h, out->msum.data());
    bool general = (flags & TSR_MSUM_GENERAL) != 0u;
    if (!general) {
        // Plan every query tile as the FLASH_ATTN kernel will (tsr_fa_plan, the firmware's source): a GENERAL chunk reads the
        // f32 copy; a refusal that depends on the data (AT §6.6 rules 5 and 6) promotes the summary to GENERAL (every tile
        // then takes the path that never refuses for mask data) as the backend's mask producer does (tsr-msum.cpp).
        tsr_msum ms;
        if (tsr_msum_view(out->msum.data(), out->msum.size(), &ms) != 0) {
            return kfail(why, "FLASH_ATTN mask: bad summary image");
        }
        uint32_t GS = 0, Tq = 0;
        tsr_fa_pass_rule(&env.att, G, &GS, &Tq);
        std::vector<tsr_fa_chunk> chunks(64);
        bool refused = false;
        for (uint32_t q0 = 0; q0 < n_q && Tq != 0 && !refused && !general; q0 += Tq) {
            const uint32_t tq = std::min(Tq, n_q - q0);
            long n = tsr_fa_plan(&env.att, E, D, &ms, q0, tq, nullptr, 0, chunks.data(), chunks.size());
            if (n > (long) chunks.size()) {
                chunks.resize((size_t) n);
                n = tsr_fa_plan(&env.att, E, D, &ms, q0, tq, nullptr, 0, chunks.data(), chunks.size());
            }
            if (n == -3) {
                refused = true;
            } else if (n > 0) {
                for (long k = 0; k < n; ++k) {
                    general = general || chunks[(size_t) k].general != 0u;
                }
            } else if (n < 0) {
                return kfail(why, "FLASH_ATTN: the chunk planner refuses the shape (E %u, D %u, G %u): %ld", E, D, G, n);
            }
        }
        if (refused) {
            out->msum[3] |= TSR_MSUM_GENERAL;
            out->msum[4] = f32_h;
            general = true;
        }
    }
    out->need_f32 = general;
    if (general) {
        if (f32_h == opl::HANDLE_NONE) {
            return kfail(why, "FLASH_ATTN mask: this mask needs the f32 copy and no place for it was given");
        }
        out->f32.assign((size_t) n_q * 16u * tsr_k_lines(n_kv), 0.0f);
        tsr_msum_mask_f32(mask, mask_stride, n_kv, n_q, scale, out->f32.data());
    }
    return true;
}

bool tsr_k_flash_attn(const tsr_kenv & env, const tsr_kd_flash_attn & d, tsr_rec * out, std::string * why) {
    *out = tsr_rec();
    if ((d.D != 64 && d.D != 128 && d.D != 256 && d.D != 512) || d.n_head_kv == 0 || d.n_head % d.n_head_kv != 0 ||
        d.T == 0 || d.n_kv == 0) {
        return kfail(why, "FLASH_ATTN shape: D %u (64, 128, 256 or 512), %u heads over %u KV heads, T %u, %u cells", d.D, d.n_head,
                     d.n_head_kv, d.T, d.n_kv);
    }
    if (d.kv.E != d.n_head_kv * d.D || d.kv.C != env.num_ch || d.n_kv > d.kv.cells()) {
        return kfail(why, "FLASH_ATTN: the planes (E %u, %u channels, %llu cells) do not match n_head_kv x D = %u, %u channels, "
                     "%u cells", d.kv.E, d.kv.C, (unsigned long long) d.kv.cells(), d.n_head_kv * d.D, env.num_ch, d.n_kv);
    }
    if (d.k_plane % 64u != 0 || d.v_plane % 64u != 0 || d.msum_h == opl::HANDLE_NONE) {
        return kfail(why, "FLASH_ATTN: planes not 4 KiB aligned or no mask summary");
    }
    // The summary and the new-cell indices of this record alone must fit the preload area (SW-43): T <= 680.
    const uint64_t pre_lines = (opl::msum_bytes(d.T) + opl::PRE_ALIGN - 1) / opl::PRE_ALIGN +
                               ((uint64_t) opl::PRE_KVIDX_BYTES * d.T + opl::PRE_ALIGN - 1) / opl::PRE_ALIGN;
    if (pre_lines > opl::PRE_MAX_LINES) {
        return kfail(why, "FLASH_ATTN of %u tokens: its mask summary and new-cell indices need %llu preload lines (at most %u: "
                     "T <= 680)", d.T, (unsigned long long) pre_lines, opl::PRE_MAX_LINES);
    }
    const uint32_t token_lines = d.n_head * d.D / 16u;
    if (!rows_ok("q", d.q, d.T, token_lines, why) || !rows_ok("dst", d.dst, d.T, token_lines, why)) {
        return false;
    }
    out->kind = opl::K_FLASH_ATTN;
    out->n_args = opl::fa::NARGS;
    const opl::flash_attn_args w{ d.dst.h, d.dst.stride, d.q.h, d.D / 16u, d.q.stride,
                                  tsr_dram_handle(0, (uint64_t) d.k_plane * TSR_BEAT_BYTES),
                                  tsr_dram_handle(0, (uint64_t) d.v_plane * TSR_BEAT_BYTES), d.n_kv, d.D, d.n_head, d.n_head_kv,
                                  d.T, 0u, d.mask_h, bits(d.scale), 0u, d.k_plane, d.v_plane, d.kv.geom_word(), d.D,
                                  d.n_head / d.n_head_kv, 0u };
    opl::put_args(w, out->args);
    out->pre.push_back({ opl::fa::MSUM, d.msum_h, (uint32_t) opl::msum_bytes(d.T) });
    out->kv_plane = d.k_plane;
    if (!finish(env, out, why)) {
        return false;
    }
    // also with FA_NEW set (tsr_encode adds it when a SET_ROWS_KV wrote the plane earlier in the list)
    tsr_rec probe = *out;
    probe.args[opl::fa::FLAGS] = tsr::plan::attn::fa_flags(0u, d.T);
    if (!tsr_k_check(env, probe, why)) {
        *out = tsr_rec();
        return false;
    }
    return true;
}

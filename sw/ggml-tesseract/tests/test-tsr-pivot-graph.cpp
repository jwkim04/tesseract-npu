// test-tsr-pivot-graph.cpp — Tesseract Pivot's ggml adapter (tsr-pivot-graph.{h,cpp}; ctest ggml_tesseract_pivot_graph).
// No device and no data: graphs of ggml nodes on shape-only tensors, built in the node order llama.cpp's graph builders give
// (the DFS post-order of their ggml_build_forward_expand calls: docs/design/07-graph-scheduler.md §6), with the shapes of
// Gemma 4 26B-A4B (a MoE layer with a dense FFN beside the experts) and Qwen3.8 27B (Gated DeltaNet layers, a full attention
// layer, a dense FFN).
//
//   - units: every group the builder fuses lies inside one unit (tsr_build_fusion on the original graph, on fake addresses)
//     and the builder takes the same groups from the permuted graph; no-op nodes and the span of a group with members apart
//   - the moves: the dense FFN products leave the router chain and the expert products behind them, the z product goes in front
//     of the GATED_DELTA_NET, the K and V products come next to Q
//   - the order constraints: the cache planes (SET_ROWS before FLASH_ATTN), the state caches, the SSM_CONV's state CPYs
//   - the node validator refuses hand-broken orders (a consumer before its producer, a FLASH_ATTN before its SET_ROWS, a node
//     inside the router chain, a CPY after its SSM_CONV, a permutation that is none) and the restore after a refusal
//   - the same permutation and the same unit costs for 1, 2, 8 and 64 tokens, the Gated DeltaNet state gather costed the same for
//     every batch (sequences, snapshot slots, extra states), and a second run that changes nothing (the fixed point)
//   - a split whose first node is a MUL_MAT_ID keeps it first (llama.cpp's expert-copy callback reads the first node)
//   - the structure key: one value for every batch of a graph (tokens, sequences, snapshot slots, extra states, rows of the state
//     zeroing) and for graphs that differ in the COMPUTE flag alone, another for another graph (nodes, weight shape and type, op,
//     unary op, link, view base, use count, type, the op of a tensor outside the graph and which links reach the same one)
//   - the memo: a later graph of a structure gets the first graph's node order without a search (also an order the search
//     would not give, also the original order, also when a node of it is not computed), a permuted graph is left alone when it
//     comes again, a remembered order that the graph's own analysis refuses is a conflict and leaves the nodes alone, a full
//     memo leaves a new structure alone and still serves the ones it knows, and a conflict stops the process
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ggml.h"
// struct ggml_cgraph and ggml_node_has_n_uses; the header's static helpers are not all used here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "ggml-impl.h"
#pragma GCC diagnostic pop

#include "tsr-build.h"
#include "tsr-pivot-graph.h"
#include "tsr/plan.hpp"

static int g_fail = 0;
#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);     \
            g_fail++;                                                                    \
        }                                                                                \
    } while (0)

static bool g_verbose = false;

// ---------------------------------------------------------------------------------------------
// graphs
// ---------------------------------------------------------------------------------------------
struct gctx {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf = nullptr;
    gctx() {
        ggml_init_params ip = { ggml_tensor_overhead() * 20000 + ggml_graph_overhead_custom(16384, false), nullptr, true };
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, 16384, false);
    }
    ~gctx() { ggml_free(ctx); }
    ggml_tensor * w(ggml_type type, int64_t K, int64_t N, const char * name) {       // a weight leaf [K, N]
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, K, N);
        ggml_set_name(t, name);
        return t;
    }
    ggml_tensor * f32(int64_t n0, int64_t n1, const char * name) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name);
        return t;
    }
    ggml_tensor * named(ggml_tensor * t, const char * name, int il = -1) {
        if (il >= 0) ggml_format_name(t, "%s-%d", name, il); else ggml_set_name(t, name);
        return t;
    }
    void expand(ggml_tensor * t) { ggml_build_forward_expand(gf, t); }
};

static ggml_tensor * norm_mul(gctx & g, ggml_tensor * x, ggml_tensor * w, const char * name, int il) {
    ggml_tensor * n = ggml_rms_norm(g.ctx, x, 1e-6f);
    return g.named(ggml_mul(g.ctx, n, w), name, il);
}

static int node_index(const ggml_cgraph * cg, const ggml_tensor * t) {
    for (int i = 0; i < cg->n_nodes; ++i) {
        if (cg->nodes[i] == t) return i;
    }
    return -1;
}

// ---- Gemma 4 26B-A4B ---------------------------------------------------------------------------------------------
struct gemma_dims {
    int64_t E = 2816, H = 16, HKV = 8, D = 256, FF = 2112, X = 128, K = 8, FX = 704, KV = 512, V = 4096;
};

struct gemma_layer_nodes {
    ggml_tensor *q, *k, *v, *qmm, *kmm, *vmm, *set_k, *set_v, *fa, *o, *attn_out;
    ggml_tensor *dense_norm, *dense_gate, *dense_up, *dense_down, *geglu_dense;
    ggml_tensor *logits, *soft_max, *div, *mmid_gate_up, *mmid_down, *scale_mul, *weighted, *moe_out, *combined, *out;
};

// One Gemma 4 layer as gemma4.cpp builds it (the statements, with its ggml_build_forward_expand calls: llm_graph_context::build_attn's
// expansions of q, k, v, the cache writes, build_moe_ffn's weights / experts / views / adds, and the layer's end).
static gemma_layer_nodes gemma_layer(gctx & g, const gemma_dims & d, ggml_tensor * inp, int il, int64_t T, ggml_tensor * pos,
                                     ggml_tensor * kidx, ggml_tensor * vidx, ggml_tensor * mask) {
    gemma_layer_nodes n = {};
    ggml_context * c = g.ctx;
    auto W = [&](int64_t K, int64_t N, const char * nm) { return g.w(GGML_TYPE_Q8_0, K, N, nm); };
    auto F = [&](int64_t n0, const char * nm) { return g.f32(n0, 1, nm); };
    // attention
    ggml_tensor * an = norm_mul(g, inp, F(d.E, "attn_norm.w"), "attn_norm", il);
    ggml_tensor * q = ggml_mul_mat(c, W(d.E, d.H * d.D, "attn_q.weight"), an);
    g.named(q, "Qcur", il);
    n.qmm = q;
    q = ggml_reshape_3d(c, q, d.D, d.H, T);
    q = norm_mul(g, q, F(d.D, "q_norm.w"), "Qcur_normed", il);
    q = ggml_rope_ext(c, q, pos, nullptr, (int) d.D, 2, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    g.named(q, "Qcur_pos", il);
    ggml_tensor * k = ggml_mul_mat(c, W(d.E, d.HKV * d.D, "attn_k.weight"), an);
    g.named(k, "Kcur", il);
    n.kmm = k;
    k = ggml_reshape_3d(c, k, d.D, d.HKV, T);
    k = norm_mul(g, k, F(d.D, "k_norm.w"), "Kcur_normed", il);
    k = ggml_rope_ext(c, k, pos, nullptr, (int) d.D, 2, 0, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    g.named(k, "Kcur_pos", il);
    ggml_tensor * v = ggml_mul_mat(c, W(d.E, d.HKV * d.D, "attn_v.weight"), an);
    g.named(v, "Vcur", il);
    n.vmm = v;
    v = ggml_reshape_3d(c, v, d.D, d.HKV, T);
    v = g.named(ggml_rms_norm(c, v, 1e-6f), "Vcur_normed", il);
    n.q = q; n.k = k; n.v = v;
    g.expand(q);
    g.expand(k);
    g.expand(v);
    ggml_tensor * ck = g.w(GGML_TYPE_F16, d.HKV * d.D, d.KV, "cache_k_l");
    ggml_tensor * cv = g.w(GGML_TYPE_F16, d.HKV * d.D, d.KV, "cache_v_l");
    ggml_format_name(ck, "cache_k_l%d", il);
    ggml_format_name(cv, "cache_v_l%d", il);
    ggml_tensor * ckv = ggml_view_2d(c, ck, ck->ne[0], ck->ne[1], ck->nb[1], 0);
    ggml_tensor * cvv = ggml_view_2d(c, cv, cv->ne[0], cv->ne[1], cv->nb[1], 0);
    n.set_k = ggml_set_rows(c, ckv, ggml_reshape_2d(c, ggml_view_3d(c, k, d.D, d.HKV, T, k->nb[1], k->nb[2], 0), d.D * d.HKV, T), kidx);
    n.set_v = ggml_set_rows(c, cvv, ggml_reshape_2d(c, ggml_view_3d(c, v, d.D, d.HKV, T, v->nb[1], v->nb[2], 0), d.D * d.HKV, T), vidx);
    g.expand(n.set_k);
    g.expand(n.set_v);
    const int64_t n_kv = d.KV;
    ggml_tensor * qp = ggml_permute(c, ggml_view_3d(c, q, d.D, d.H, T, q->nb[1], q->nb[2], 0), 0, 2, 1, 3);
    ggml_tensor * kc = ggml_permute(c, ggml_view_3d(c, ck, d.D, d.HKV, n_kv, d.D * ck->nb[0], ck->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * vc = ggml_permute(c, ggml_view_3d(c, cv, d.D, d.HKV, n_kv, d.D * cv->nb[0], cv->nb[1], 0), 0, 2, 1, 3);
    n.fa = ggml_flash_attn_ext(c, qp, kc, vc, mask, 1.0f, 0.0f, 0.0f);
    g.named(n.fa, "fa", il);
    ggml_tensor * fr = ggml_reshape_2d(c, n.fa, d.H * d.D, T);
    n.o = ggml_mul_mat(c, W(d.H * d.D, d.E, "attn_output.weight"), fr);
    g.named(n.o, "attn_o", il);
    ggml_tensor * on = norm_mul(g, n.o, F(d.E, "attn_post_norm.w"), "attn_post_norm", il);
    n.attn_out = g.named(ggml_add(c, on, inp), "attn_out", il);
    // router
    ggml_tensor * tmp = ggml_rms_norm(c, n.attn_out, 1e-6f);
    tmp = ggml_scale(c, tmp, 1.0f / std::sqrt((float) d.E));
    tmp = ggml_mul(c, tmp, F(d.E, "ffn_gate_inp.scale"));
    n.logits = g.named(ggml_mul_mat(c, g.w(GGML_TYPE_F32, d.E, d.X, "ffn_gate_inp.weight"), tmp), "ffn_moe_logits", il);
    ggml_tensor * probs = g.named(ggml_soft_max(c, n.logits), "ffn_moe_probs", il);
    n.soft_max = probs;
    ggml_tensor * sel = ggml_argsort_top_k(c, probs, (int) d.K);
    g.named(sel, "ffn_moe_topk", il);
    ggml_tensor * pr = ggml_reshape_3d(c, probs, 1, d.X, T);
    ggml_tensor * wt = ggml_get_rows(c, pr, sel);
    g.named(wt, "ffn_moe_weights", il);
    wt = ggml_reshape_2d(c, wt, d.K, T);
    ggml_tensor * sum = ggml_sum_rows(c, wt);
    sum = ggml_clamp(c, sum, 6.103515625e-5f, INFINITY);
    wt = ggml_div(c, wt, sum);
    n.div = g.named(wt, "ffn_moe_weights_norm", il);
    wt = ggml_reshape_3d(c, wt, 1, d.K, T);
    g.expand(wt);
    // experts
    ggml_tensor * cur = norm_mul(g, n.attn_out, F(d.E, "ffn_pre_norm_2.w"), "ffn_norm_2", il);
    cur = ggml_reshape_3d(c, cur, d.E, 1, T);
    // the expert tensors are 3D [K, N, X]
    ggml_tensor * e3 = ggml_new_tensor_3d(c, GGML_TYPE_Q8_0, d.E, 2 * d.FX, d.X);
    ggml_set_name(e3, "ffn_gate_up_exps.weight");
    ggml_tensor * gu = ggml_mul_mat_id(c, e3, cur, sel);
    n.mmid_gate_up = g.named(gu, "ffn_moe_gate_up", il);
    ggml_tensor * gate = ggml_view_3d(c, gu, d.FX, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
    ggml_tensor * up = ggml_view_3d(c, gu, d.FX, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], d.FX * gu->nb[0]);
    cur = g.named(ggml_geglu_split(c, gate, up), "ffn_moe_geglu", il);
    ggml_tensor * e3d = ggml_new_tensor_3d(c, GGML_TYPE_Q8_0, d.FX, d.E, d.X);
    ggml_set_name(e3d, "ffn_down_exps.weight");
    ggml_tensor * ex = ggml_mul_mat_id(c, e3d, cur, sel);
    n.mmid_down = ex;
    ggml_tensor * sc = g.f32(d.X, 1, "ffn_down_exps.scale");
    sc = ggml_reshape_3d(c, sc, 1, d.X, 1);
    sc = ggml_repeat_4d(c, sc, 1, d.X, T, 1);
    sc = ggml_get_rows(c, sc, sel);
    ex = ggml_mul(c, ex, sc);
    n.scale_mul = g.named(ex, "ffn_moe_down_scaled", il);
    ex = g.named(ggml_mul(c, ex, wt), "ffn_moe_weighted", il);
    n.weighted = ex;
    g.expand(ex);
    ggml_tensor * moe = nullptr;
    for (int64_t i = 0; i < d.K; ++i) {
        ggml_tensor * vv = ggml_view_2d(c, ex, d.E, T, ex->nb[2], i * ex->nb[1]);
        g.expand(vv);
        if (moe == nullptr) { moe = vv; continue; }
        moe = ggml_add(c, moe, vv);
        g.expand(moe);
    }
    n.moe_out = g.named(moe, "ffn_moe_out", il);
    // dense (reached by the DFS only from the combine)
    ggml_tensor * dn = norm_mul(g, n.attn_out, F(d.E, "ffn_norm.w"), "ffn_norm_1", il);
    n.dense_norm = dn;
    ggml_tensor * dup = ggml_mul_mat(c, W(d.E, d.FF, "ffn_up.weight"), dn);
    g.named(dup, "ffn_up", il);
    ggml_tensor * dgate = ggml_mul_mat(c, W(d.E, d.FF, "ffn_gate.weight"), dn);
    g.named(dgate, "ffn_gate", il);
    n.dense_up = dup; n.dense_gate = dgate;
    n.geglu_dense = g.named(ggml_geglu_split(c, dgate, dup), "ffn_geglu", il);
    n.dense_down = ggml_mul_mat(c, W(d.FF, d.E, "ffn_down.weight"), n.geglu_dense);
    ggml_tensor * mlp = norm_mul(g, n.dense_down, F(d.E, "ffn_post_norm_1.w"), "ffn_mlp", il);
    ggml_tensor * moen = norm_mul(g, n.moe_out, F(d.E, "ffn_post_norm_2.w"), "ffn_moe", il);
    n.combined = g.named(ggml_add(c, mlp, moen), "ffn_moe_combined", il);
    ggml_tensor * o2 = norm_mul(g, n.combined, F(d.E, "ffn_post_norm.w"), "ffn_post_norm", il);
    o2 = ggml_add(c, o2, n.attn_out);
    o2 = ggml_mul(c, o2, F(1, "layer_output_scale.w"));
    n.out = g.named(o2, "l_out", il);
    g.expand(n.out);
    return n;
}

struct gemma_graph {
    gctx g;
    std::vector<gemma_layer_nodes> L;
    gemma_dims d;
};

static void build_gemma(gemma_graph & gg, int layers, int64_t T) {
    gctx & g = gg.g;
    const gemma_dims & d = gg.d;
    ggml_tensor * inp = g.f32(d.E, T, "inp_embd");
    ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "inp_pos");
    ggml_tensor * kidx = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, T);
    ggml_set_name(kidx, "attn_inp_k_idxs");
    ggml_tensor * vidx = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, T);
    ggml_set_name(vidx, "attn_inp_v_idxs");
    ggml_tensor * mask = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, d.KV, std::max<int64_t>(T, 32));
    ggml_set_name(mask, "attn_inp_kq_mask");
    ggml_tensor * cur = ggml_scale(g.ctx, inp, 1.0f);
    g.expand(cur);
    for (int il = 0; il < layers; ++il) {
        gg.L.push_back(gemma_layer(g, d, cur, il, T, pos, kidx, vidx, mask));
        cur = gg.L.back().out;
    }
    ggml_tensor * fin = norm_mul(g, cur, g.f32(d.E, 1, "output_norm.w"), "result_norm", -1);
    ggml_tensor * lg = ggml_mul_mat(g.ctx, g.w(GGML_TYPE_Q8_0, d.E, d.V, "output.weight"), fin);
    g.named(lg, "result_output");
    ggml_set_output(lg);
    g.expand(lg);
}

// ---- Qwen3.8 27B ---------------------------------------------------------------------------------------------------
struct qwen_dims {
    int64_t E = 5120, S = 128, HK = 16, HV = 48, DC = 4, FF = 17408, HQ = 24, HKV = 4, D = 256, KV = 512;
    int64_t zero_rows = 1;       // the cells build_rs zeroes: 1 for a new sequence, 0 (an empty view) when the sequence goes on
    int64_t C() const { return 2 * S * HK + S * HV; }       // the conv channels: q, k and v of the recurrence
};

struct qwen_gdn_nodes {
    ggml_tensor *qkv, *alpha, *beta, *z, *conv, *silu, *gdn, *gather, *cpy_state, *cpy_conv, *scale_zero_s, *scale_zero_r, *out, *attn_res,
                *ffn_gate, *ffn_up, *ffn_down, *l_out;
};

static qwen_gdn_nodes qwen_gdn_layer(gctx & g, const qwen_dims & d, ggml_tensor * inp, int il, int64_t T, ggml_tensor * rs_idx) {
    qwen_gdn_nodes n = {};
    ggml_context * c = g.ctx;
    auto W = [&](int64_t K, int64_t N, const char * nm) { return g.w(GGML_TYPE_Q8_0, K, N, nm); };
    auto F = [&](int64_t n0, const char * nm) { return g.f32(n0, 1, nm); };
    ggml_tensor * an = norm_mul(g, inp, F(d.E, "attn_norm.w"), "attn_norm", il);
    g.expand(an);
    // the conv state: zero the cell, gather it
    const int64_t C = d.C(), cells = 4, conv_size = (d.DC - 1) * C;
    ggml_tensor * cache_r = g.f32(conv_size, cells, "cache_r_l");
    ggml_format_name(cache_r, "cache_r_l%d", il);
    ggml_tensor * cell_r = ggml_view_2d(c, cache_r, conv_size, d.zero_rows, cache_r->nb[1], 0);
    n.scale_zero_r = ggml_scale_inplace(c, cell_r, 0.0f);
    g.expand(n.scale_zero_r);
    ggml_tensor * gr = ggml_get_rows(c, ggml_reshape_2d(c, cache_r, conv_size, cells), rs_idx);
    ggml_tensor * conv_states = ggml_reshape_3d(c, ggml_view_2d(c, gr, conv_size, 1, gr->nb[1], 0), d.DC - 1, C, 1);
    n.qkv = ggml_mul_mat(c, W(d.E, C, "attn_qkv.weight"), an);
    g.named(n.qkv, "linear_attn_qkv_mixed", il);
    ggml_tensor * xt = ggml_transpose(c, ggml_reshape_3d(c, n.qkv, C, T, 1));
    ggml_tensor * cat = ggml_concat(c, conv_states, xt, 0);
    g.named(cat, "conv_input", il);
    ggml_tensor * last = ggml_view_3d(c, cat, d.DC - 1, C, 1, cat->nb[1], cat->nb[2], (size_t) (cat->ne[0] - (d.DC - 1)) * 4);
    ggml_tensor * cell_r2 = ggml_view_1d(c, cache_r, conv_size, 0);
    n.cpy_conv = ggml_cpy(c, last, cell_r2);
    g.expand(n.cpy_conv);
    // the recurrent state: zero, gather
    const int64_t SD = d.S * d.S * d.HV;
    ggml_tensor * cache_s = g.f32(SD, cells, "cache_s_l");
    ggml_format_name(cache_s, "cache_s_l%d", il);
    ggml_tensor * cell_s = ggml_view_2d(c, cache_s, SD, d.zero_rows, cache_s->nb[1], 0);
    n.scale_zero_s = ggml_scale_inplace(c, cell_s, 0.0f);
    g.expand(n.scale_zero_s);
    n.gather = ggml_get_rows(c, ggml_reshape_2d(c, cache_s, SD, cells), rs_idx);
    ggml_tensor * taps = F(d.DC, "ssm_conv1d.w");
    taps = g.f32(d.DC, C, "ssm_conv1d.weight");
    n.conv = ggml_ssm_conv(c, cat, taps);
    g.named(n.conv, "conv_output_raw", il);
    n.silu = ggml_silu(c, n.conv);
    g.named(n.silu, "conv_output_silu", il);
    ggml_tensor * qv = ggml_view_4d(c, n.silu, d.S, d.HK, T, 1, d.S * 4, d.S * d.HK * 4, d.S * d.HK * T * 4, 0);
    ggml_tensor * kv = ggml_view_4d(c, n.silu, d.S, d.HK, T, 1, d.S * 4, d.S * d.HK * 4, d.S * d.HK * T * 4, (size_t) (d.S * d.HK * 4));
    ggml_tensor * vv = ggml_view_4d(c, n.silu, d.S, d.HV, T, 1, d.S * 4, d.S * d.HV * 4, d.S * d.HV * T * 4, (size_t) (2 * d.S * d.HK * 4));
    qv = ggml_scale(c, ggml_rms_norm(c, qv, 1e-6f), 1.0f / std::sqrt((float) d.S));
    kv = ggml_scale(c, ggml_rms_norm(c, kv, 1e-6f), 1.0f);
    n.alpha = ggml_mul_mat(c, W(d.E, d.HV, "ssm_alpha.weight"), an);
    g.named(n.alpha, "alpha", il);
    ggml_tensor * gate = ggml_mul(c, ggml_softplus(c, ggml_add(c, ggml_reshape_3d(c, n.alpha, d.HV, T, 1), g.f32(d.HV, 1, "ssm_dt.bias"))),
                                  g.f32(d.HV, 1, "ssm_a"));
    n.beta = ggml_mul_mat(c, W(d.E, d.HV, "ssm_beta.weight"), an);
    g.named(n.beta, "beta", il);
    ggml_tensor * bsig = ggml_sigmoid(c, ggml_reshape_3d(c, n.beta, d.HV, T, 1));
    ggml_tensor * st4 = ggml_reshape_4d(c, ggml_view_2d(c, n.gather, SD, 1, n.gather->nb[1], 0), d.S, d.S, d.HV, 1);
    gate = ggml_reshape_4d(c, gate, 1, d.HV, T, 1);
    bsig = ggml_reshape_4d(c, bsig, 1, d.HV, T, 1);
    n.gdn = ggml_gated_delta_net(c, qv, kv, vv, gate, bsig, st4, 1);
    g.named(n.gdn, "gdn", il);
    ggml_tensor * ns_view = ggml_view_4d(c, n.gdn, d.S, d.S, d.HV, 1, d.S * 4, d.S * d.S * 4, d.S * d.S * d.HV * 4, (size_t) (d.S * d.HV * T * 4));
    ggml_tensor * dst = ggml_view_2d(c, cache_s, SD, 1, cache_s->nb[1], 0);
    n.cpy_state = ggml_cpy(c, ns_view, dst);
    g.expand(n.cpy_state);
    ggml_tensor * att = ggml_view_4d(c, n.gdn, d.S, d.HV, T, 1, d.S * 4, d.S * d.HV * 4, d.S * d.HV * T * 4, 0);
    ggml_tensor * nrm = norm_mul(g, att, F(d.S, "ssm_norm.w"), "attn_out_norm", il);
    n.z = ggml_mul_mat(c, W(d.E, d.S * d.HV, "attn_gate.weight"), an);     // reached only from the final chain: after the recurrence
    g.named(n.z, "z", il);
    ggml_tensor * zs = ggml_silu(c, ggml_reshape_3d(c, n.z, d.S, d.HV, T));
    ggml_tensor * gated = ggml_mul(c, nrm, zs);
    ggml_tensor * fo = ggml_reshape_2d(c, gated, d.S * d.HV, T);
    n.out = ggml_mul_mat(c, W(d.S * d.HV, d.E, "ssm_out.weight"), fo);
    g.named(n.out, "linear_attn_out", il);
    n.attn_res = g.named(ggml_add(c, n.out, inp), "attn_residual", il);
    ggml_tensor * pn = norm_mul(g, n.attn_res, F(d.E, "post_attention_norm.w"), "attn_post_norm", il);
    n.ffn_gate = ggml_mul_mat(c, W(d.E, d.FF, "ffn_gate.weight"), pn);
    g.named(n.ffn_gate, "ffn_gate", il);
    n.ffn_up = ggml_mul_mat(c, W(d.E, d.FF, "ffn_up.weight"), pn);
    g.named(n.ffn_up, "ffn_up", il);
    ggml_tensor * sw = ggml_swiglu_split(c, n.ffn_gate, n.ffn_up);
    n.ffn_down = ggml_mul_mat(c, W(d.FF, d.E, "ffn_down.weight"), sw);
    g.named(n.ffn_down, "ffn_out", il);
    n.l_out = g.named(ggml_add(c, n.ffn_down, n.attn_res), "l_out", il);
    g.expand(n.l_out);
    return n;
}

struct qwen_graph {
    gctx g;
    qwen_dims d;
    std::vector<qwen_gdn_nodes> L;
};

static void build_qwen(qwen_graph & qg, int layers, int64_t T) {
    gctx & g = qg.g;
    ggml_tensor * inp = g.f32(qg.d.E, T, "inp_embd");
    ggml_set_input(inp);
    ggml_tensor * rs_idx = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 1);
    ggml_set_name(rs_idx, "rs_s_copy");
    ggml_tensor * cur = ggml_scale(g.ctx, inp, 1.0f);
    g.expand(cur);
    for (int il = 0; il < layers; ++il) {
        qg.L.push_back(qwen_gdn_layer(g, qg.d, cur, il, T, rs_idx));
        cur = qg.L.back().l_out;
    }
    ggml_tensor * fin = norm_mul(g, cur, g.f32(qg.d.E, 1, "output_norm.w"), "result_norm", -1);
    ggml_tensor * lg = ggml_mul_mat(g.ctx, g.w(GGML_TYPE_Q8_0, qg.d.E, 4096, "output.weight"), fin);
    g.named(lg, "result_output");
    ggml_set_output(lg);
    g.expand(lg);
}

// ---------------------------------------------------------------------------------------------
// the builder's view of a graph: fake addresses, then tsr_build_fusion
// ---------------------------------------------------------------------------------------------
static const uint64_t FAKE = 0x1000000000ull;

// Addresses in the graph's node order: every node and every leaf its own bytes (no reuse), a view inside its base.
static void place(ggml_tensor * t, uint64_t * next) {
    if (t == nullptr || t->data != nullptr) return;
    if (t->view_src != nullptr) {
        place(t->view_src, next);
        t->data = (char *) t->view_src->data + t->view_offs;
        return;
    }
    t->data = (void *) (uintptr_t) (FAKE + *next);
    *next += (ggml_nbytes(t) + 4095) / 4096 * 4096 + 4096;
}
static void place_graph(ggml_cgraph * cg) {
    uint64_t next = 0;
    for (int i = 0; i < cg->n_nodes; ++i) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) place(cg->nodes[i]->src[s], &next);
        place(cg->nodes[i], &next);
    }
}
static void unplace_graph(ggml_cgraph * cg) {
    for (int i = 0; i < cg->n_nodes; ++i) {
        cg->nodes[i]->data = nullptr;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (cg->nodes[i]->src[s] != nullptr) cg->nodes[i]->src[s]->data = nullptr;
        }
    }
}

static tsr_build_env fake_env(bool gdn_cells) {
    tsr_build_env e;
    e.cfg = tsr::plan::U200_K4;
    e.pl = tsr::plan::plan_limits_of(e.cfg);
    e.num_ch = 4;
    e.loc = [](const ggml_tensor * t, tsr_tloc * l) {
        const uint64_t a = (uint64_t) (uintptr_t) t->data;
        if (t->data == nullptr || a < FAKE) return false;
        l->ch = 1;
        l->off = a - FAKE;
        return true;
    };
    if (gdn_cells) {
        e.read_i32 = [](const ggml_tensor * t, std::vector<int32_t> * v) {
            v->assign((size_t) t->ne[0], 0);           // the identity: cell 0 for one sequence
            return true;
        };
    }
    return e;
}

// The fusion decisions of a graph as pairs (member, head) of tensors: what the builder absorbs into which record.
using fused_set = std::set<std::pair<const ggml_tensor *, const ggml_tensor *>>;
static fused_set fusion_of(ggml_cgraph * cg, tsr_fusion * out = nullptr) {
    place_graph(cg);
    const tsr_fusion f = tsr_build_fusion(fake_env(true), cg);
    fused_set s;
    for (int k = 0; k < cg->n_nodes; ++k) {
        if (f.into[k] >= 0) s.insert({ cg->nodes[k], cg->nodes[f.into[k]] });
    }
    for (int i = 0; i < cg->n_nodes; ++i) {
        if (f.gdn_cell[i].ids != nullptr) {
            s.insert({ f.gdn_cell[i].ids, cg->nodes[i] });
        }
    }
    if (out != nullptr) *out = f;
    unplace_graph(cg);
    return s;
}

// ---------------------------------------------------------------------------------------------
// helpers on a result
// ---------------------------------------------------------------------------------------------
static tsr_pivot_dev card_dev() {
    tsr_pivot_dev d;
    d.num_ch = 4;
    d.dual = true;
    return d;
}

static std::vector<ggml_tensor *> nodes_of(const ggml_cgraph * cg) {
    return std::vector<ggml_tensor *>(cg->nodes, cg->nodes + cg->n_nodes);
}

static void print_moves(const tsr_pivot_report & rep) {
    for (const auto & m : rep.moves) printf("  move %s\n", m.c_str());
    printf("  %u nodes, %u units, %zu moves, %u passes, model %lld -> %lld cycles, %.1f us\n", rep.n_nodes, rep.n_units, rep.moves.size(),
           rep.passes, (long long) rep.cycles_before, (long long) rep.cycles_after, (double) rep.ns * 1e-3);
}

static void print_order(const ggml_cgraph * cg) {
    for (int i = 0; i < cg->n_nodes; ++i) {
        const ggml_tensor * t = cg->nodes[i];
        printf("   %3d %-14s %s\n", i, ggml_op_name(t->op), t->name);
    }
}

// the unit of a node of the original graph
static int unit_of(const tsr_pivot_graph & g, const std::vector<ggml_tensor *> & orig, const ggml_tensor * t) {
    for (size_t i = 0; i < orig.size(); ++i) {
        if (orig[i] == t) return g.unit_of[i];
    }
    return -1;
}
static bool has_pred(const tsr_pivot_graph & g, int u, int p) {
    const auto & v = g.units[(size_t) u].preds;
    return std::find(v.begin(), v.end(), (uint32_t) p) != v.end();
}

// ---------------------------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------------------------

// The units of a Gemma layer and a Qwen layer: classes, groups, the no-ops, and every group the builder fuses inside one unit.
static void test_units() {
    gemma_graph gg;
    build_gemma(gg, 2, 1);
    const std::vector<ggml_tensor *> orig = nodes_of(gg.g.gf);
    tsr_pivot_graph g;
    CHECK(tsr_pivot_analyze(gg.g.gf, card_dev(), &g));
    CHECK(g.units.size() < orig.size());
    for (const gemma_layer_nodes & n : gg.L) {
        const int u_gate = unit_of(g, orig, n.dense_gate), u_norm = unit_of(g, orig, n.dense_norm);
        CHECK(g.units[(size_t) u_gate].cls == TSR_PIVOT_MVM && g.units[(size_t) u_gate].defer);
        CHECK(u_norm != u_gate && g.units[(size_t) u_norm].cls == TSR_PIVOT_GLUE);
        CHECK(unit_of(g, orig, n.dense_norm->src[0]) == u_norm);               // RMS_NORM + MUL: one unit
        // the router chain, SOFT_MAX to DIV, is one BLOCK unit of 6,500 + 96 x 128 cycles
        const int u_router = unit_of(g, orig, n.soft_max);
        CHECK(unit_of(g, orig, n.div) == u_router && g.units[(size_t) u_router].cls == TSR_PIVOT_BLOCK);
        CHECK(g.units[(size_t) u_router].body == 6500 + 96 * 128);
        // the MUL_MAT_ID with its scale chain: one MVM_ID unit
        const int u_down = unit_of(g, orig, n.mmid_down);
        CHECK(unit_of(g, orig, n.scale_mul) == u_down && g.units[(size_t) u_down].cls == TSR_PIVOT_MVM_ID);
        CHECK(g.units[(size_t) unit_of(g, orig, n.mmid_gate_up)].cls == TSR_PIVOT_MVM_ID);
        CHECK(g.units[(size_t) unit_of(g, orig, n.fa)].cls == TSR_PIVOT_ATTN);
        // a weight product: ld 64 + K / 16, back 110 + N / 16, MVM time from the weight bytes per channel at 1.41 beats per cycle
        const tsr_pivot_unit & q = g.units[(size_t) unit_of(g, orig, n.dense_up)];
        CHECK(q.ld == 64 + 2816 / 16 && q.back == 110 + 2112 / 16 && q.qact == 16 * (2816 / 32) + 44);
        const int64_t beats = ((2112 + 3) / 4) * tsr::plan::row_beats(tsr::plan::FMT_Q8_0, 2816);
        CHECK(q.mvm == beats * 100 / 141);
        CHECK(q.issue == TSR_PIVOT_ISSUE_MUL_MAT && q.issue_open == TSR_PIVOT_ISSUE_BEGIN && q.issue_close == TSR_PIVOT_ISSUE_END);
        // data dependencies and the order constraints of the cache planes: the FLASH_ATTN follows both SET_ROWS units
        const int u_fa = unit_of(g, orig, n.fa);
        CHECK(has_pred(g, u_fa, unit_of(g, orig, n.set_k)) && has_pred(g, u_fa, unit_of(g, orig, n.set_v)));
    }
    // no-op nodes join the unit of the next compute node; the last run joins the unit before it
    for (size_t i = 0; i < orig.size(); ++i) {
        CHECK(g.unit_of[i] >= 0 && g.unit_of[i] < (int) g.units.size());
        if (i > 0) CHECK(g.unit_of[i] >= g.unit_of[i - 1]);                        // units are intervals of the original order
    }
    // every group the builder fuses lies inside one unit
    {
        tsr_fusion f;
        const fused_set before = fusion_of(gg.g.gf, &f);
        int checked = 0, n_moe = 0;
        for (int k = 0; k < gg.g.gf->n_nodes; ++k) {
            if (f.into[k] < 0) continue;
            CHECK(g.unit_of[(size_t) k] == g.unit_of[(size_t) f.into[k]]);
            checked++;
            n_moe += gg.g.gf->nodes[f.into[k]]->op == GGML_OP_SOFT_MAX || gg.g.gf->nodes[f.into[k]]->op == GGML_OP_MUL_MAT_ID;
        }
        for (int i = 0; i < gg.g.gf->n_nodes; ++i) {
            if (f.with[i] >= 0) CHECK(g.unit_of[(size_t) i] == g.unit_of[(size_t) f.with[i]]);
        }
        CHECK(checked >= 2 * (6 + 3 + 4) && n_moe >= 2 * 8);                   // pairs, route and scale members: not vacuous
    }
    qwen_graph qg;
    build_qwen(qg, 2, 1);
    const std::vector<ggml_tensor *> qorig = nodes_of(qg.g.gf);
    tsr_pivot_graph h;
    CHECK(tsr_pivot_analyze(qg.g.gf, card_dev(), &h));
    for (const qwen_gdn_nodes & n : qg.L) {
        const int u_conv = unit_of(h, qorig, n.conv);
        CHECK(unit_of(h, qorig, n.silu) == u_conv && h.units[(size_t) u_conv].cls == TSR_PIVOT_GLUE);   // SSM_CONV + SILU
        const int u_gdn = unit_of(h, qorig, n.gdn);
        CHECK(h.units[(size_t) u_gdn].cls == TSR_PIVOT_BLOCK && h.units[(size_t) u_gdn].body == 3300 + 48 * 3040);
        // the state gather and the state CPY are absorbed by the record: units of no cost
        const int u_g = unit_of(h, qorig, n.gather), u_c = unit_of(h, qorig, n.cpy_state);
        CHECK(h.units[(size_t) u_g].body == 0 && h.units[(size_t) u_g].issue == 0 && h.units[(size_t) u_c].body == 0 && h.units[(size_t) u_c].issue == 0);
        for (const ggml_tensor * z : { n.scale_zero_s, n.scale_zero_r }) {              // zeroing a cell: free, rows or no rows
            const tsr_pivot_unit & zu = h.units[(size_t) unit_of(h, qorig, z)];
            CHECK(zu.cls == TSR_PIVOT_GLUE && zu.body == 0 && zu.issue == 0);
        }
        // the state caches keep their order: zero the cell, gather, the recurrence, the copy back
        CHECK(has_pred(h, u_g, unit_of(h, qorig, n.scale_zero_s)) && has_pred(h, u_gdn, u_g) && has_pred(h, u_c, u_gdn));
        // the SSM_CONV's state CPY comes first, no source says so
        CHECK(has_pred(h, u_conv, unit_of(h, qorig, n.cpy_conv)));
        CHECK(h.units[(size_t) unit_of(h, qorig, n.z)].cls == TSR_PIVOT_MVM);
    }
    {
        tsr_fusion f;
        fusion_of(qg.g.gf, &f);
        int conv_groups = 0, cells = 0;
        for (int i = 0; i < qg.g.gf->n_nodes; ++i) {
            if (f.conv_silu[i] >= 0) {
                CHECK(h.unit_of[(size_t) i] == h.unit_of[(size_t) f.conv_silu[i]]);
                conv_groups++;
            }
            cells += f.gdn_cell[i].ids != nullptr;
        }
        CHECK(conv_groups == 2 && cells == 2);
    }
}

// Units from a few hand-built graphs: no-ops, the span of a group whose members are apart, a leading and a trailing no-op.
static void test_unit_shapes() {
    gctx g;
    ggml_context * c = g.ctx;
    ggml_tensor * x = g.f32(896, 1, "x");
    ggml_tensor * lead = ggml_view_2d(c, x, 896, 1, x->nb[1], 0);                 // a no-op in front of everything
    ggml_tensor * n0 = ggml_scale(c, lead, 2.0f);
    ggml_tensor * wq = g.w(GGML_TYPE_Q8_0, 896, 896, "wq");
    ggml_tensor * mid = ggml_view_2d(c, x, 896, 1, x->nb[1], 0);                  // a no-op between two units: it joins the next one
    ggml_tensor * s2 = ggml_scale(c, mid, 4.0f);
    ggml_tensor * mm = ggml_mul_mat(c, wq, n0);
    ggml_tensor * other = ggml_scale(c, x, 3.0f);                                 // independent, between the product and its ADD
    ggml_tensor * bias = g.f32(896, 1, "bias");
    ggml_tensor * add = ggml_add(c, mm, bias);
    ggml_tensor * tail = ggml_reshape_2d(c, add, 896, 1);                          // a no-op behind everything
    g.expand(n0);
    g.expand(s2);
    g.expand(mm);
    g.expand(other);
    g.expand(add);
    g.expand(tail);
    const std::vector<ggml_tensor *> orig = nodes_of(g.gf);
    tsr_pivot_graph u;
    CHECK(tsr_pivot_analyze(g.gf, card_dev(), &u));
    // lead (a view) joins n0 and mid joins s2, the next compute node; mm, other and add are one unit (the span between a MUL_MAT and its
    // fused ADD); tail joins the unit before it
    CHECK(unit_of(u, orig, lead) == unit_of(u, orig, n0));
    CHECK(unit_of(u, orig, mm) == unit_of(u, orig, add) && unit_of(u, orig, other) == unit_of(u, orig, mm));
    CHECK(unit_of(u, orig, tail) == unit_of(u, orig, add));
    CHECK(unit_of(u, orig, mid) == unit_of(u, orig, s2) && unit_of(u, orig, mid) != unit_of(u, orig, n0));
    CHECK(u.units.size() == 3 && u.units[2].cls == TSR_PIVOT_MVM);
    CHECK(u.units[2].preds.size() == 1 && u.units[2].preds[0] == 0);
    // a graph with no compute node is left alone
    gctx e;
    ggml_tensor * a = e.f32(8, 1, "a");
    ggml_tensor * v = ggml_view_2d(e.ctx, a, 8, 1, a->nb[1], 0);
    e.gf->nodes[e.gf->n_nodes++] = v;
    tsr_pivot_graph none;
    CHECK(!tsr_pivot_analyze(e.gf, card_dev(), &none));
    tsr_pivot_report rep;
    tsr_pivot_optimize(e.gf, card_dev(), tsr_pivot_params(), &rep);
    CHECK(!rep.ran && !rep.applied && e.gf->nodes[0] == v);
}

// An in-place op: the reader of the original value stays in front of the write.
static void test_in_place() {
    gctx g;
    ggml_context * c = g.ctx;
    ggml_tensor * x = g.f32(896, 1, "x"), * y = g.f32(896, 1, "y");
    ggml_tensor * a = ggml_add(c, x, y);
    ggml_tensor * r1 = ggml_scale(c, a, 2.0f);                                    // reads a
    ggml_tensor * w = ggml_scale_inplace(c, a, 3.0f);                             // overwrites a: after r1, though no source says so
    ggml_tensor * r2 = ggml_add(c, r1, w);
    g.expand(a);
    g.expand(r1);
    g.expand(w);
    g.expand(r2);
    const std::vector<ggml_tensor *> orig = nodes_of(g.gf);
    tsr_pivot_graph u;
    CHECK(tsr_pivot_analyze(g.gf, card_dev(), &u));
    CHECK(has_pred(u, unit_of(u, orig, w), unit_of(u, orig, r1)));
    std::vector<int> bad = { 0, 2, 1, 3 };                                        // a, w, r1, r2
    std::string why;
    CHECK(!tsr_pivot_check_order(orig, u, bad, &why) && why.find("order constraint") != std::string::npos);
    std::vector<int> ok = { 0, 1, 2, 3 };
    CHECK(tsr_pivot_check_order(orig, u, ok, &why));
}

// The moves of a Gemma layer: the dense FFN products leave the router chain and the expert products behind them, K and V next to Q.
static void test_moves_gemma(int64_t T) {
    gemma_graph gg;
    build_gemma(gg, 3, T);
    const std::vector<ggml_tensor *> orig = nodes_of(gg.g.gf);
    const fused_set fused_before = fusion_of(gg.g.gf);
    tsr_pivot_report rep;
    tsr_pivot_optimize(gg.g.gf, card_dev(), tsr_pivot_params(), &rep);
    CHECK(rep.ran && rep.applied && !rep.refused && !rep.moves.empty());
    CHECK(rep.cycles_after < rep.cycles_before);
    if (g_verbose && T == 1) { print_moves(rep); print_order(gg.g.gf); }
    const ggml_cgraph * cg = gg.g.gf;
    for (const gemma_layer_nodes & n : gg.L) {
        CHECK(node_index(cg, n.dense_gate) < node_index(cg, n.soft_max));            // the dense gate product runs under the router
        CHECK(node_index(cg, n.dense_up) > node_index(cg, n.mmid_down));              // the up product under the expert outputs' combine
        CHECK(node_index(cg, n.dense_up) < node_index(cg, n.combined));
        CHECK(node_index(cg, n.kmm) < node_index(cg, n.q));                         // the K and V products come in front of Q's norm and rope chain's end
        CHECK(node_index(cg, n.vmm) < node_index(cg, n.k));
        CHECK(node_index(cg, n.set_k) < node_index(cg, n.fa) && node_index(cg, n.set_v) < node_index(cg, n.fa));
        // the router chain is contiguous in compute nodes: no compute node between its first and its last
        int lo = node_index(cg, n.soft_max), hi = node_index(cg, n.div);
        for (int i = lo; i <= hi; ++i) {
            const ggml_tensor * t = cg->nodes[i];
            CHECK(tsr_node_is_noop(t) || t->op == GGML_OP_SOFT_MAX || t->op == GGML_OP_ARGSORT || t->op == GGML_OP_GET_ROWS ||
                  t->op == GGML_OP_SUM_ROWS || t->op == GGML_OP_CLAMP || t->op == GGML_OP_DIV);
        }
    }
    // the builder takes the same groups from the permuted graph
    CHECK(fusion_of(gg.g.gf) == fused_before);
    // the order is a permutation that the node validator accepts
    CHECK(tsr_pivot_check_order(orig, rep.graph, rep.node_order));
}

static void test_moves_qwen(int64_t T) {
    qwen_graph qg;
    build_qwen(qg, 3, T);
    const std::vector<ggml_tensor *> orig = nodes_of(qg.g.gf);
    const fused_set fused_before = fusion_of(qg.g.gf);
    tsr_pivot_report rep;
    tsr_pivot_optimize(qg.g.gf, card_dev(), tsr_pivot_params(), &rep);
    CHECK(rep.ran && rep.applied && !rep.refused && rep.cycles_after < rep.cycles_before);
    if (g_verbose && T == 1) { print_moves(rep); print_order(qg.g.gf); }
    const ggml_cgraph * cg = qg.g.gf;
    for (const qwen_gdn_nodes & n : qg.L) {
        CHECK(node_index(cg, n.z) < node_index(cg, n.gdn));                         // the z product runs under the recurrence
        CHECK(node_index(cg, n.scale_zero_s) < node_index(cg, n.gather) && node_index(cg, n.gather) < node_index(cg, n.gdn) &&
              node_index(cg, n.gdn) < node_index(cg, n.cpy_state));
        CHECK(node_index(cg, n.scale_zero_r) < node_index(cg, n.cpy_conv) && node_index(cg, n.cpy_conv) < node_index(cg, n.conv));
        CHECK(node_index(cg, n.conv) + 1 == node_index(cg, n.silu) ||
              (node_index(cg, n.conv) < node_index(cg, n.silu) && node_index(cg, n.silu) - node_index(cg, n.conv) <= 1));
        CHECK(node_index(cg, n.alpha) < node_index(cg, n.gdn) && node_index(cg, n.beta) < node_index(cg, n.gdn));
    }
    CHECK(fusion_of(qg.g.gf) == fused_before);
    CHECK(tsr_pivot_check_order(orig, rep.graph, rep.node_order));
}

// The node validator refuses orders the search never makes.
static void test_validator() {
    gemma_graph gg;
    build_gemma(gg, 2, 1);
    const std::vector<ggml_tensor *> orig = nodes_of(gg.g.gf);
    tsr_pivot_graph g;
    CHECK(tsr_pivot_analyze(gg.g.gf, card_dev(), &g));
    const int n = (int) orig.size();
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i) id[i] = i;
    std::string why;
    CHECK(tsr_pivot_check_order(orig, g, id, &why));
    const gemma_layer_nodes & l = gg.L[0];
    const int i_set_k = node_index(gg.g.gf, l.set_k), i_fa = node_index(gg.g.gf, l.fa);
    // a consumer before its producer: the two nodes after the attention output swap
    {
        std::vector<int> o = id;
        const int a = node_index(gg.g.gf, l.attn_out);
        std::swap(o[a], o[a + 1]);
        CHECK(!tsr_pivot_check_order(orig, g, o, &why) && why.find("consumer") != std::string::npos);
    }
    // a FLASH_ATTN before its SET_ROWS (no source links them): the SET_ROWS of K goes behind the FLASH_ATTN
    {
        std::vector<int> o = id;
        o.erase(o.begin() + i_set_k);
        o.insert(std::find(o.begin(), o.end(), i_fa) + 1, i_set_k);
        CHECK(!tsr_pivot_check_order(orig, g, o, &why) && why.find("order constraint") != std::string::npos);
    }
    // a node inside the router chain: the dense norm (two nodes) goes between the ARGSORT and the view of the top-k
    {
        const int i_norm_mul = node_index(gg.g.gf, l.dense_norm), i_norm = i_norm_mul - 1;
        const int i_argsort = node_index(gg.g.gf, l.soft_max) + 2;
        std::vector<int> o;
        for (int i = 0; i < n; ++i) {
            if (i != i_norm && i != i_norm_mul) o.push_back(id[i]);
            if (i == i_argsort) { o.push_back(i_norm); o.push_back(i_norm_mul); }
        }
        CHECK((int) o.size() == n);
        CHECK(!tsr_pivot_check_order(orig, g, o, &why) && why.find("inside the group") != std::string::npos);
    }
    // not a permutation
    {
        std::vector<int> o = id;
        o[3] = o[4];
        CHECK(!tsr_pivot_check_order(orig, g, o, &why) && why.find("permutation") != std::string::npos);
        o.pop_back();
        CHECK(!tsr_pivot_check_order(orig, g, o, &why) && why.find("permutation") != std::string::npos);
    }
    // a refused order puts the original back: a unit order that runs a unit before its predecessor
    {
        const std::vector<ggml_tensor *> before = nodes_of(gg.g.gf);
        tsr_pivot_report rep;
        std::function<void(std::vector<uint32_t> *)> tamper = [&](std::vector<uint32_t> * o) {
            // put the unit of the dense down product in front of its GEGLU unit
            const uint32_t u_down = (uint32_t) g.unit_of[(size_t) node_index(gg.g.gf, l.dense_down)];
            auto a = std::find(o->begin(), o->end(), u_down), b = std::find(o->begin(), o->end(), u_down - 1);
            std::iter_swap(a, b);
        };
        tsr_pivot_optimize(gg.g.gf, card_dev(), tsr_pivot_params(), &rep, &tamper);
        CHECK(rep.refused && !rep.applied && !rep.why.empty() && nodes_of(gg.g.gf) == before);
    }
    // a CPY behind its SSM_CONV (the builder takes the CPYs down to the SSM_CONV: they have to come first)
    {
        qwen_graph qg;
        build_qwen(qg, 1, 1);
        const std::vector<ggml_tensor *> qo = nodes_of(qg.g.gf);
        tsr_pivot_graph h;
        CHECK(tsr_pivot_analyze(qg.g.gf, card_dev(), &h));
        const qwen_gdn_nodes & q = qg.L[0];
        const int qn = (int) qo.size();
        std::vector<int> o(qn);
        for (int i = 0; i < qn; ++i) o[i] = i;
        CHECK(tsr_pivot_check_order(qo, h, o, &why));
        const int i_cpy = node_index(qg.g.gf, q.cpy_conv), i_silu = node_index(qg.g.gf, q.silu);
        o.erase(o.begin() + i_cpy);
        o.insert(std::find(o.begin(), o.end(), i_silu) + 1, i_cpy);
        CHECK(!tsr_pivot_check_order(qo, h, o, &why) && why.find("order constraint") != std::string::npos);
    }
}

static bool units_equal(const std::vector<tsr_pivot_unit> & a, const std::vector<tsr_pivot_unit> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const tsr_pivot_unit &x = a[i], &y = b[i];
        if (x.cls != y.cls || x.preds != y.preds || x.issue != y.issue || x.issue_open != y.issue_open || x.issue_close != y.issue_close ||
            x.body != y.body || x.ld != y.ld || x.qact != y.qact || x.ids != y.ids || x.front != y.front || x.mvm != y.mvm ||
            x.tail != y.tail || x.back != y.back || x.defer != y.defer) {
            fprintf(stderr, "units differ at %zu\n", i);
            return false;
        }
    }
    return true;
}

// A Gated DeltaNet layer's state group as llama.cpp's build_rs lays it out (one GET_ROWS for the main and the extra states, views of
// it, a CPY of the extra states and a CPY of the new states into the snapshot slots), for a token count T, sequences n_seqs, K - 1
// snapshot slots and n_extra extra states. Whether tsr_gdn_state_shape matches depends on these; the units must not.
static void gdn_state_graph(gctx & g, int64_t T, int64_t ns, int64_t K, int64_t n_extra, ggml_tensor ** gather_out) {
    ggml_context * c = g.ctx;
    const int64_t S = 32, H = 4, D = S * S * H, cells = 16, mem = 4;
    ggml_tensor * cache = g.f32(D, cells * K, "cache_s_l0");
    ggml_tensor * x = g.f32(S * H, T * ns, "x");
    ggml_tensor * ids_all = ggml_new_tensor_1d(c, GGML_TYPE_I32, ns + n_extra);
    ggml_tensor * gather = ggml_get_rows(c, ggml_reshape_2d(c, cache, D, cells * K), ids_all);
    ggml_tensor * st4 = ggml_reshape_4d(c, ggml_view_2d(c, gather, D, ns, gather->nb[1], 0), S, S, H, ns);
    ggml_tensor * extra = ggml_view_2d(c, gather, D, n_extra, gather->nb[1], ns * gather->nb[1]);
    ggml_tensor * cpy_extra = ggml_cpy(c, extra, ggml_view_2d(c, cache, D, n_extra, cache->nb[1], (size_t) (ns + 2) * cache->nb[1]));
    ggml_tensor * q = ggml_reshape_4d(c, ggml_scale(c, x, 1.0f), S, H, T, ns);
    ggml_tensor * k = ggml_reshape_4d(c, ggml_scale(c, x, 2.0f), S, H, T, ns);
    ggml_tensor * v = ggml_reshape_4d(c, ggml_scale(c, x, 3.0f), S, H, T, ns);
    ggml_tensor * gg = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, H, T, ns);
    ggml_tensor * bb = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, H, T, ns);
    ggml_tensor * gdn = ggml_gated_delta_net(c, q, k, v, gg, bb, st4, K);
    const int64_t att = S * H * T * ns, nw = T < K ? T : K;
    ggml_tensor * src = ggml_view_3d(c, gdn, D, ns, nw, ggml_row_size(gdn->type, D), ggml_row_size(gdn->type, D * ns), att * 4);
    ggml_tensor * dst = ggml_view_3d(c, cache, D, ns, nw, cache->nb[1], (size_t) mem * cache->nb[1], 0);
    ggml_tensor * cpy = ggml_cpy(c, src, dst);
    ggml_tensor * out = ggml_scale(c, ggml_view_2d(c, gdn, S * H, T * ns, gdn->nb[1], 0), 1.0f);
    g.expand(ggml_scale_inplace(c, ggml_view_2d(c, cache, D, 0, cache->nb[1], 0), 0.0f));
    g.expand(gather);
    g.expand(cpy_extra);
    g.expand(cpy);
    g.expand(out);
    *gather_out = gather;
}

// The cost of a unit is a function of the graph's structure, not of the batch: for the batches that make tsr_gdn_state_shape match
// (one sequence, no extra states, T or K that give one snapshot slot) and those that make it fail (several sequences with
// snapshots, extra states), the graphs have the same nodes, so the units must be the same, the gather's included.
static void test_gdn_state_cost_independent_of_batch() {
    struct cfg { int64_t T, ns, K, extra; };
    const cfg cfgs[] = { { 1, 1, 4, 0 }, { 4, 1, 4, 0 }, { 16, 1, 4, 0 }, { 1, 2, 4, 0 }, { 4, 2, 4, 0 }, { 16, 2, 4, 0 }, { 1, 1, 1, 0 }, { 1, 1, 1, 1 },
                         { 8, 1, 1, 1 }, { 4, 2, 1, 1 } };
    std::vector<tsr_pivot_unit> ref;
    int ref_nodes = -1;
    bool matched = false, unmatched = false;
    for (const cfg & k : cfgs) {
        gctx g;
        ggml_tensor * gather = nullptr;
        gdn_state_graph(g, k.T, k.ns, k.K, k.extra, &gather);
        tsr_pivot_graph pg;
        CHECK(tsr_pivot_analyze(g.gf, card_dev(), &pg));
        int gdn_i = -1;
        for (int i = 0; i < g.gf->n_nodes; ++i) if (g.gf->nodes[i]->op == GGML_OP_GATED_DELTA_NET) gdn_i = i;
        tsr_gdn_shape sh;
        (tsr_gdn_state_shape(g.gf, gdn_i, &sh) ? matched : unmatched) = true;
        const tsr_pivot_unit & gu = pg.units[(size_t) pg.unit_of[(size_t) node_index(g.gf, gather)]];
        CHECK(gu.cls == TSR_PIVOT_GLUE && gu.body == 0 && gu.issue == 0);
        if (ref_nodes < 0) {
            ref = pg.units;
            ref_nodes = g.gf->n_nodes;
        } else {
            CHECK(g.gf->n_nodes == ref_nodes);
            CHECK(units_equal(pg.units, ref));
        }
    }
    CHECK(matched && unmatched);                                  // the sweep covers both outcomes of the matcher
}

// llama.cpp's expert-copy callback reads the first node of a split: a split that starts with a MUL_MAT_ID keeps it first, whatever the
// search would gain from a unit in front of it.
static void test_first_mul_mat_id_stays() {
    for (int first_is_mmid = 0; first_is_mmid < 2; ++first_is_mmid) {
        gctx g;
        ggml_context * c = g.ctx;
        ggml_tensor * experts = ggml_new_tensor_3d(c, GGML_TYPE_Q8_0, 256, 64, 8);
        ggml_set_name(experts, "ffn_gate_exps.weight");
        ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 2, 1);
        ggml_tensor * xe = ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, 1, 1);
        ggml_tensor * x = g.f32(256, 1, "x");
        ggml_tensor * mmid = ggml_mul_mat_id(c, experts, xe, ids);
        ggml_tensor * y = ggml_mul_mat(c, g.w(GGML_TYPE_Q8_0, 256, 256, "w"), x);       // independent of the expert product
        ggml_tensor * glue = ggml_scale(c, ggml_scale(c, ggml_scale(c, mmid, 2.0f), 3.0f), 4.0f);
        if (first_is_mmid) { g.expand(mmid); g.expand(glue); g.expand(y); } else { g.expand(y); g.expand(mmid); g.expand(glue); }
        const std::vector<ggml_tensor *> orig = nodes_of(g.gf);
        tsr_pivot_graph pg;
        CHECK(tsr_pivot_analyze(g.gf, card_dev(), &pg));
        CHECK(pg.pin_first == (first_is_mmid == 1));
        if (first_is_mmid) {
            for (size_t u = 1; u < pg.units.size(); ++u) {
                CHECK(std::find(pg.units[u].preds.begin(), pg.units[u].preds.end(), 0u) != pg.units[u].preds.end());
            }
            std::vector<int> o(orig.size());
            for (size_t i = 0; i < o.size(); ++i) o[i] = (int) i;
            std::string why;
            CHECK(tsr_pivot_check_order(orig, pg, o, &why));
            // the independent product in front of it: a valid order by sources, refused for the first node
            const int iy = node_index(g.gf, y);
            o.erase(o.begin() + iy);
            o.insert(o.begin(), iy);
            CHECK(!tsr_pivot_check_order(orig, pg, o, &why) && why.find("first node") != std::string::npos);
        }
        tsr_pivot_report rep;
        tsr_pivot_optimize(g.gf, card_dev(), tsr_pivot_params(), &rep);
        CHECK(!rep.refused);
        if (first_is_mmid) CHECK(g.gf->nodes[0] == mmid);
    }
}

// The permutation is a function of the structure: the same for 1, 2, 8 and 64 tokens. And a second run changes nothing.
static void test_token_independence() {
    std::vector<int> ref_g, ref_q;
    std::vector<tsr_pivot_unit> ref_gu, ref_qu;
    for (int64_t T : { 1, 2, 8, 64 }) {
        {
            gemma_graph gg;
            build_gemma(gg, 3, T);
            tsr_pivot_report rep;
            tsr_pivot_optimize(gg.g.gf, card_dev(), tsr_pivot_params(), &rep);
            CHECK(rep.applied);
            if (ref_g.empty()) ref_g = rep.node_order; else CHECK(rep.node_order == ref_g);
            // the costs themselves, not only the order they lead to: a cost that grows with the batch could change nothing in this
            // graph and still change the order of another one
            if (ref_gu.empty()) ref_gu = rep.graph.units; else CHECK(units_equal(rep.graph.units, ref_gu));
        }
        {
            qwen_graph qg;
            build_qwen(qg, 3, T);
            tsr_pivot_report rep;
            tsr_pivot_optimize(qg.g.gf, card_dev(), tsr_pivot_params(), &rep);
            CHECK(rep.applied);
            if (ref_q.empty()) ref_q = rep.node_order; else CHECK(rep.node_order == ref_q);
            if (ref_qu.empty()) ref_qu = rep.graph.units; else CHECK(units_equal(rep.graph.units, ref_qu));
        }
    }
    // whether the zeroing SCALEs of the state caches have rows depends on the batch, not on the graph: the same permutation
    {
        std::vector<int> ref;
        std::vector<std::string> ref_moves;
        uint32_t ref_units = 0;
        for (int64_t T : { 1, 7 }) {
            for (int64_t zero : { 0, 1 }) {
                qwen_graph qg;
                qg.d.zero_rows = zero;
                build_qwen(qg, 3, T);
                tsr_pivot_report rep;
                tsr_pivot_optimize(qg.g.gf, card_dev(), tsr_pivot_params(), &rep);
                CHECK(rep.applied);
                const bool is_empty = ggml_is_empty(qg.L[0].scale_zero_s);
                CHECK(is_empty == (zero == 0));
                if (ref.empty()) {
                    ref = rep.node_order;
                    ref_moves = rep.moves;
                    ref_units = rep.n_units;
                } else {
                    CHECK(rep.node_order == ref && rep.moves == ref_moves && rep.n_units == ref_units);   // the units, not only the nodes
                }
            }
        }
    }
    {
        gemma_graph gg;
        build_gemma(gg, 3, 1);
        tsr_pivot_report r1, r2;
        tsr_pivot_optimize(gg.g.gf, card_dev(), tsr_pivot_params(), &r1);
        const std::vector<ggml_tensor *> once = nodes_of(gg.g.gf);
        tsr_pivot_optimize(gg.g.gf, card_dev(), tsr_pivot_params(), &r2);
        CHECK(r1.applied && !r2.applied && r2.moves.empty() && nodes_of(gg.g.gf) == once);
        // the same device, the same graph: the same result
        gemma_graph g2;
        build_gemma(g2, 3, 1);
        tsr_pivot_report r3;
        tsr_pivot_optimize(g2.g.gf, card_dev(), tsr_pivot_params(), &r3);
        CHECK(r3.node_order == r1.node_order && r3.cycles_after == r1.cycles_after);
    }
    {
        qwen_graph qg;
        build_qwen(qg, 3, 1);
        tsr_pivot_report r1, r2;
        tsr_pivot_optimize(qg.g.gf, card_dev(), tsr_pivot_params(), &r1);
        const std::vector<ggml_tensor *> once = nodes_of(qg.g.gf);
        tsr_pivot_optimize(qg.g.gf, card_dev(), tsr_pivot_params(), &r2);
        CHECK(r1.applied && !r2.applied && r2.moves.empty() && nodes_of(qg.g.gf) == once);
    }
}

// The structure key: the same for every batch of one graph, another for another graph.
static void test_structure_key() {
    // 1, 2, 8 and 64 tokens, and the state zeroing with and without rows
    uint64_t kg = 0, kq = 0;
    for (int64_t T : { 1, 2, 8, 64 }) {
        gemma_graph gg;
        build_gemma(gg, 3, T);
        const uint64_t k = tsr_pivot_structure_key(gg.g.gf);
        if (kg == 0) kg = k; else CHECK(k == kg);
        for (int64_t zero : { 0, 1 }) {
            qwen_graph qg;
            qg.d.zero_rows = zero;
            build_qwen(qg, 3, T);
            const uint64_t q = tsr_pivot_structure_key(qg.g.gf);
            if (kq == 0) kq = q; else CHECK(q == kq);
        }
    }
    CHECK(kg != 0 && kq != 0 && kg != kq);
    // the batches of a Gated DeltaNet state group (tokens, sequences, snapshot slots, extra states): one structure
    {
        struct cfg { int64_t T, ns, K, extra; };
        const cfg cfgs[] = { { 1, 1, 4, 0 }, { 16, 1, 4, 0 }, { 4, 2, 4, 0 }, { 1, 1, 1, 0 }, { 1, 1, 1, 1 }, { 4, 2, 1, 1 } };
        uint64_t ks = 0;
        for (const cfg & c : cfgs) {
            gctx g;
            ggml_tensor * gather = nullptr;
            gdn_state_graph(g, c.T, c.ns, c.K, c.extra, &gather);
            const uint64_t k = tsr_pivot_structure_key(g.gf);
            if (ks == 0) ks = k; else CHECK(k == ks);
        }
    }
    // more nodes
    {
        gemma_graph a, b;
        build_gemma(a, 3, 1);
        build_gemma(b, 2, 1);
        CHECK(tsr_pivot_structure_key(a.g.gf) != tsr_pivot_structure_key(b.g.gf));
    }
    // One small graph and what the key must see in it. y = MUL_MAT(w, x), z = SCALE(y), s = SCALE(y), out = ADD(z, s), e = SCALE(x)
    // and one node behind them that the key's graph (a view without the last node) does not hold. Variants: 1 another weight
    // shape, 2 another weight type, 3 another op, 4 another link, 5 a node without the COMPUTE flag (the same key), 6 a reader of
    // z outside the view (another use count), 7 and 8 a view that starts behind y, whose op is then all the key sees of it
    // (MUL_MAT, SQR), 9 e reads another leaf than y does (which links reach one tensor), 10 and 11 e is a cast of x to F16 and to
    // F32 (the type), 12 and 13 z is a GELU and a SILU of y (the unary op).
    auto small = [](int variant) -> uint64_t {
        gctx g;
        ggml_context * c = g.ctx;
        const int64_t N = variant == 1 ? 512 : 256;
        ggml_tensor * x = g.f32(256, 1, "x");
        ggml_tensor * x2 = g.f32(256, 1, "x2");
        ggml_tensor * xn = g.f32(N, 1, "xn");
        ggml_tensor * w = g.w(variant == 2 ? GGML_TYPE_Q4_K : GGML_TYPE_Q8_0, 256, N, "w");
        ggml_tensor * y = variant == 8 ? ggml_sqr(c, xn) : ggml_mul_mat(c, w, x);
        ggml_tensor * z = variant == 3 ? ggml_sqr(c, y) : variant == 12 ? ggml_gelu(c, y) : variant == 13 ? ggml_silu(c, y)
                                                                                          : ggml_scale(c, y, 2.0f);
        ggml_tensor * s2 = ggml_scale(c, y, 3.0f);
        ggml_tensor * out = variant == 4 ? ggml_add(c, z, y) : ggml_add(c, z, s2);
        ggml_tensor * e = variant == 10 ? ggml_cast(c, x, GGML_TYPE_F16) : variant == 11 ? ggml_cast(c, x, GGML_TYPE_F32)
                                                                                         : ggml_scale(c, variant == 9 ? x2 : x, 4.0f);
        ggml_tensor * last = ggml_scale(c, variant == 6 ? z : xn, 5.0f);
        g.expand(z);
        g.expand(s2);
        g.expand(out);
        g.expand(e);
        g.expand(last);
        if (variant == 5) z->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        CHECK(g.gf->n_nodes == 6 && g.gf->nodes[0] == y && g.gf->nodes[4] == e && g.gf->nodes[5] == last);
        // a view as ggml_graph_view makes it for a split: a run of the nodes, with the use counts of the whole graph
        ggml_cgraph v = *g.gf;
        const int i0 = (variant == 7 || variant == 8) ? 1 : 0;
        v.nodes = g.gf->nodes + i0;
        v.n_nodes = 5 - i0;
        return tsr_pivot_structure_key(&v);
    };
    const uint64_t base = small(0);
    CHECK(small(0) == base);                                      // other addresses, the same key
    for (int variant : { 1, 2, 3, 4, 6, 9, 12, 13 }) {
        CHECK(small(variant) != base);
    }
    // The COMPUTE flag is not in the key: llama.cpp selects branches by it, and ggml-alloc plans a node with or without it.
    CHECK(small(5) == base);
    CHECK(small(7) != base && small(7) != small(8));
    CHECK(small(10) != small(11));
    CHECK(small(12) != small(13));
    // The view base. b = SCALE(VIEW(VIEW(leaf))), and the inner view lies in front of the key's graph: the outer view's source is
    // then a view outside the graph in both cases, and only its base says which leaf it is (the one a reads, or another).
    auto viewed = [](bool other) -> uint64_t {
        gctx g;
        ggml_context * c = g.ctx;
        ggml_tensor * x = g.f32(256, 1, "x");
        ggml_tensor * x2 = g.f32(256, 1, "x2");
        ggml_tensor * inner = ggml_view_1d(c, other ? x2 : x, 128, 0);
        ggml_tensor * outer = ggml_view_1d(c, inner, 64, 0);
        ggml_tensor * a = ggml_scale(c, x, 2.0f);
        ggml_tensor * b = ggml_scale(c, outer, 3.0f);
        g.expand(inner);
        g.expand(a);
        g.expand(b);
        CHECK(g.gf->n_nodes == 4 && g.gf->nodes[0] == inner && g.gf->nodes[1] == a && g.gf->nodes[2] == outer);
        CHECK(outer->view_src == (other ? x2 : x) && outer->src[0] == inner);
        ggml_cgraph v = *g.gf;
        v.nodes = g.gf->nodes + 1;
        v.n_nodes = 3;
        return tsr_pivot_structure_key(&v);
    };
    CHECK(viewed(false) == viewed(false) && viewed(false) != viewed(true));
}

// The memo: the first graph of a structure is searched, every later one gets its node order without a search.
static void test_memo() {
    const auto in_order = [](const ggml_cgraph * cg, const std::vector<ggml_tensor *> & orig, const std::vector<int> & order) {
        bool same = (int) order.size() == cg->n_nodes;
        for (size_t p = 0; same && p < order.size(); ++p) same = cg->nodes[p] == orig[(size_t) order[p]];
        return same;
    };
    // a graph of another batch gets the order of the first graph, and no search runs for it
    {
        tsr_pivot_memo memo;
        gemma_graph a, b;
        build_gemma(a, 3, 64);
        build_gemma(b, 3, 1);
        const std::vector<ggml_tensor *> orig = nodes_of(b.g.gf);
        tsr_pivot_report ra, rb;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, nullptr, &memo);
        CHECK(ra.applied && !ra.memo_hit && !ra.moves.empty() && ra.key == tsr_pivot_structure_key(b.g.gf));
        CHECK(memo.orders.size() == 2);                           // the structure, and the permuted graph as a structure that stays
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        CHECK(rb.memo_hit && rb.applied && !rb.conflict && !rb.refused && rb.key == ra.key);
        CHECK(rb.moves.empty() && rb.passes == 0 && rb.cycles_before == 0 && rb.unit_order.empty());
        CHECK(rb.node_order == ra.node_order && in_order(b.g.gf, orig, ra.node_order));
        CHECK(memo.orders.size() == 2);
        // the permuted graph again (a scheduler that reserves a graph and then computes it): it stays, without a search
        const std::vector<ggml_tensor *> once = nodes_of(a.g.gf);
        tsr_pivot_report r2;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &r2, nullptr, &memo);
        CHECK(r2.memo_hit && !r2.applied && !r2.ran && !r2.conflict && r2.key != ra.key && nodes_of(a.g.gf) == once);
        CHECK(memo.orders.size() == 2);
    }
    // The memo decides, not a new search. The first graph keeps its order (the test hook puts the identity back), so the second one
    // keeps its order too, although a search would move its nodes.
    {
        tsr_pivot_memo memo;
        const std::function<void(std::vector<uint32_t> *)> keep = [](std::vector<uint32_t> * o) {
            for (size_t k = 0; k < o->size(); ++k) (*o)[k] = (uint32_t) k;
        };
        gemma_graph a, b, c;
        build_gemma(a, 3, 1);
        build_gemma(b, 3, 8);
        build_gemma(c, 3, 8);
        const std::vector<ggml_tensor *> orig = nodes_of(b.g.gf);
        tsr_pivot_report ra, rb, rc;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, &keep, &memo);
        CHECK(!ra.applied && !ra.refused && memo.orders.size() == 1);
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        CHECK(rb.memo_hit && !rb.applied && !rb.ran && !rb.conflict && nodes_of(b.g.gf) == orig);
        tsr_pivot_optimize(c.g.gf, card_dev(), tsr_pivot_params(), &rc);
        CHECK(rc.applied);
    }
    // The same with an order that is not the search's: two neighbours of the first graph's unit order that do not depend on each
    // other change places. The second graph gets exactly that node order.
    {
        tsr_pivot_memo memo;
        tsr_pivot_report ra, rb, rp;
        bool swapped = false;
        const std::function<void(std::vector<uint32_t> *)> swap = [&](std::vector<uint32_t> * o) {
            for (size_t k = 0; k + 1 < o->size() && !swapped; ++k) {
                const std::vector<uint32_t> & pb = ra.graph.units[(*o)[k + 1]].preds;
                if (std::find(pb.begin(), pb.end(), (*o)[k]) == pb.end()) {
                    std::swap((*o)[k], (*o)[k + 1]);
                    swapped = true;
                }
            }
        };
        gemma_graph a, b, plain;
        build_gemma(a, 3, 1);
        build_gemma(b, 3, 8);
        build_gemma(plain, 3, 1);
        const std::vector<ggml_tensor *> orig = nodes_of(b.g.gf);
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, &swap, &memo);
        CHECK(swapped && ra.applied && !ra.refused);
        tsr_pivot_optimize(plain.g.gf, card_dev(), tsr_pivot_params(), &rp);
        CHECK(rp.applied && rp.node_order != ra.node_order);
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        CHECK(rb.memo_hit && rb.applied && rb.node_order == ra.node_order && in_order(b.g.gf, orig, ra.node_order));
    }
    // A first graph whose order the validator refuses keeps its order, and so does every later graph of the structure.
    {
        tsr_pivot_memo memo;
        const std::function<void(std::vector<uint32_t> *)> reverse = [](std::vector<uint32_t> * o) { std::reverse(o->begin(), o->end()); };
        qwen_graph a, b;
        build_qwen(a, 3, 1);
        build_qwen(b, 3, 7);
        const std::vector<ggml_tensor *> orig_a = nodes_of(a.g.gf), orig_b = nodes_of(b.g.gf);
        tsr_pivot_report ra, rb;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, &reverse, &memo);
        CHECK(ra.refused && !ra.applied && !ra.conflict && nodes_of(a.g.gf) == orig_a && memo.orders.size() == 1);
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        CHECK(rb.memo_hit && !rb.applied && !rb.refused && !rb.conflict && nodes_of(b.g.gf) == orig_b);
    }
    // A conflict: the remembered order does not fit the graph. Consumers before producers, a node missing, and an order that is
    // valid by the sources but not by the graph's own analysis (the first node of this graph is a MUL_MAT_ID and must stay).
    {
        tsr_pivot_memo memo;
        gemma_graph a, b, c;
        build_gemma(a, 3, 1);
        build_gemma(b, 3, 1);
        build_gemma(c, 3, 1);
        tsr_pivot_report ra, rb, rc;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, nullptr, &memo);
        CHECK(ra.applied);
        const uint64_t mk = [&]() {
            for (const auto & e : memo.orders) if (!e.second.empty()) return e.first;
            return (uint64_t) 0;
        }();
        std::vector<int> & o = memo.orders[mk];
        CHECK(o == ra.node_order);
        std::reverse(o.begin(), o.end());
        const std::vector<ggml_tensor *> orig_b = nodes_of(b.g.gf), orig_c = nodes_of(c.g.gf);
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        CHECK(rb.memo_hit && rb.conflict && rb.refused && !rb.applied && rb.why.find("producer") != std::string::npos);
        CHECK(nodes_of(b.g.gf) == orig_b);
        o = ra.node_order;
        o.pop_back();
        tsr_pivot_optimize(c.g.gf, card_dev(), tsr_pivot_params(), &rc, nullptr, &memo);
        CHECK(rc.memo_hit && rc.conflict && !rc.applied && rc.why.find("not a permutation") != std::string::npos && nodes_of(c.g.gf) == orig_c);
    }
    {
        tsr_pivot_memo memo;
        tsr_pivot_report r1, r2;
        int iy = -1;
        for (int round = 0; round < 2; ++round) {
            gctx g;
            ggml_context * c = g.ctx;
            ggml_tensor * experts = ggml_new_tensor_3d(c, GGML_TYPE_Q8_0, 256, 64, 8);
            ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 2, 1);
            ggml_tensor * xe = ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, 1, 1);
            ggml_tensor * mmid = ggml_mul_mat_id(c, experts, xe, ids);
            ggml_tensor * y = ggml_mul_mat(c, g.w(GGML_TYPE_Q8_0, 256, 256, "w"), g.f32(256, 1, "x"));
            g.expand(mmid);
            g.expand(ggml_scale(c, mmid, 2.0f));
            g.expand(y);
            const std::vector<ggml_tensor *> orig = nodes_of(g.gf);
            if (round == 0) {
                tsr_pivot_optimize(g.gf, card_dev(), tsr_pivot_params(), &r1, nullptr, &memo);
                CHECK(!r1.memo_hit && !r1.refused && memo.orders.size() == 1 && g.gf->nodes[0] == mmid);
                iy = node_index(g.gf, y);
                // the independent product in front of the MUL_MAT_ID: producers first, but not what this graph allows
                std::vector<int> bad(orig.size());
                for (size_t i = 0; i < bad.size(); ++i) bad[i] = (int) i;
                bad.erase(bad.begin() + iy);
                bad.insert(bad.begin(), iy);
                memo.orders.begin()->second = bad;
            } else {
                tsr_pivot_optimize(g.gf, card_dev(), tsr_pivot_params(), &r2, nullptr, &memo);
                CHECK(r2.memo_hit && r2.conflict && !r2.applied && r2.why.find("first node") != std::string::npos);
                CHECK(nodes_of(g.gf) == orig);
            }
        }
        CHECK(iy > 0);
    }
    // a full memo: a new structure is not searched and keeps its order, and the memo does not grow
    {
        tsr_pivot_memo memo;
        for (uint64_t k = 0; k < tsr_pivot_memo::MAX_ENTRIES; ++k) memo.orders[k * 0x9e3779b97f4a7c15ull + 1] = {};
        gemma_graph a;
        build_gemma(a, 3, 1);
        const std::vector<ggml_tensor *> orig = nodes_of(a.g.gf);
        tsr_pivot_report ra;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, nullptr, &memo);
        CHECK(ra.memo_full && !ra.memo_hit && !ra.ran && !ra.applied && nodes_of(a.g.gf) == orig);
        CHECK(memo.orders.size() == tsr_pivot_memo::MAX_ENTRIES);
    }
    // a full memo still serves the structures it knows
    {
        tsr_pivot_memo memo;
        gemma_graph a, b;
        build_gemma(a, 3, 64);
        build_gemma(b, 3, 1);
        tsr_pivot_report ra, rb;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, nullptr, &memo);
        CHECK(ra.applied);
        for (uint64_t k = 0; memo.orders.size() < tsr_pivot_memo::MAX_ENTRIES; ++k) {
            memo.orders.emplace(k * 0x9e3779b97f4a7c15ull + 1, std::vector<int>());
        }
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        CHECK(rb.memo_hit && rb.applied && !rb.memo_full && rb.node_order == ra.node_order);
        CHECK(memo.orders.size() == tsr_pivot_memo::MAX_ENTRIES);
    }
    // Graphs that differ in the COMPUTE flag alone (the branches llama.cpp selects with ggml_build_forward_select) share a plan in
    // ggml-alloc and a key here: the one with a node that is not computed gets the order of the first, checked by its own analysis.
    {
        tsr_pivot_memo memo;
        gemma_graph a, b;
        build_gemma(a, 3, 1);
        build_gemma(b, 3, 1);
        b.g.gf->nodes[0]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        const std::vector<ggml_tensor *> orig = nodes_of(b.g.gf);
        tsr_pivot_report ra, rb;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, nullptr, &memo);
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        CHECK(ra.applied && rb.key == ra.key && rb.memo_hit && rb.applied && !rb.conflict && rb.node_order == ra.node_order);
        CHECK(rb.n_units + 1 == ra.n_units && in_order(b.g.gf, orig, ra.node_order));
    }
    // the device and the search parameters belong to the key: another device, another search
    {
        tsr_pivot_memo memo;
        tsr_pivot_dev single = card_dev();
        single.dual = false;
        tsr_pivot_params wide;
        wide.window = 32;
        gemma_graph a, b, c;
        build_gemma(a, 3, 1);
        build_gemma(b, 3, 1);
        build_gemma(c, 3, 1);
        tsr_pivot_report ra, rb, rc;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, nullptr, &memo);
        tsr_pivot_optimize(b.g.gf, single, tsr_pivot_params(), &rb, nullptr, &memo);
        tsr_pivot_optimize(c.g.gf, card_dev(), wide, &rc, nullptr, &memo);
        CHECK(!ra.memo_hit && !rb.memo_hit && !rc.memo_hit && rb.ran && rc.ran && rb.key == ra.key && rc.key == ra.key);
    }
}

// A conflict stops the process (tsr_pivot_stop_on_conflict). A child process runs into one: the remembered order of its second
// graph is the first graph's order reversed.
static void test_conflict_stops() {
    tsr_pivot_report none;
    tsr_pivot_stop_on_conflict(none);                             // no conflict: it returns
    int fds[2];
    if (pipe(fds) != 0) {
        CHECK(!"pipe");
        return;
    }
    fflush(nullptr);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], 2);
        setenv("GGML_NO_BACKTRACE", "1", 1);
        const struct rlimit no_core = { 0, 0 };
        setrlimit(RLIMIT_CORE, &no_core);
        tsr_pivot_memo memo;
        gemma_graph a, b;
        build_gemma(a, 3, 1);
        build_gemma(b, 3, 1);
        tsr_pivot_report ra, rb;
        tsr_pivot_optimize(a.g.gf, card_dev(), tsr_pivot_params(), &ra, nullptr, &memo);
        for (auto & e : memo.orders) std::reverse(e.second.begin(), e.second.end());
        tsr_pivot_optimize(b.g.gf, card_dev(), tsr_pivot_params(), &rb, nullptr, &memo);
        tsr_pivot_stop_on_conflict(rb);
        _exit(0);                                                 // not reached
    }
    close(fds[1]);
    std::string msg;
    char buf[512];
    for (ssize_t n; (n = read(fds[0], buf, sizeof(buf))) > 0;) msg.append(buf, (size_t) n);
    close(fds[0]);
    int status = 0;
    CHECK(pid > 0 && waitpid(pid, &status, 0) == pid);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    CHECK(msg.find("Tesseract Pivot") != std::string::npos && msg.find("producer") != std::string::npos &&
          msg.find("TSR_PIVOT=0") != std::string::npos);
}

// The cost is the device's: a single weight stream and another channel count change the units.
static void test_costs_follow_the_device() {
    gemma_graph gg;
    build_gemma(gg, 1, 1);
    tsr_pivot_dev single = card_dev();
    single.dual = false;
    tsr_pivot_graph a, b;
    CHECK(tsr_pivot_analyze(gg.g.gf, card_dev(), &a) && tsr_pivot_analyze(gg.g.gf, single, &b));
    const std::vector<ggml_tensor *> orig = nodes_of(gg.g.gf);
    const int u = unit_of(a, orig, gg.L[0].dense_up);
    const int64_t beats = ((2112 + 3) / 4) * tsr::plan::row_beats(tsr::plan::FMT_Q8_0, 2816);
    CHECK(b.units[(size_t) u].mvm == beats && a.units[(size_t) u].mvm == beats * 100 / 141);
    tsr_pivot_dev c2 = card_dev();
    c2.num_ch = 2;
    tsr_pivot_graph c;
    CHECK(tsr_pivot_analyze(gg.g.gf, c2, &c));
    CHECK(c.units[(size_t) u].mvm > 1.9 * a.units[(size_t) u].mvm);
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) g_verbose = true;
    }
    test_unit_shapes();
    test_in_place();
    test_units();
    test_moves_gemma(1);
    test_moves_gemma(64);
    test_moves_qwen(1);
    test_moves_qwen(64);
    test_validator();
    test_token_independence();
    test_gdn_state_cost_independent_of_batch();
    test_first_mul_mat_id_stays();
    test_structure_key();
    test_memo();
    test_conflict_stops();
    test_costs_follow_the_device();
    if (g_fail != 0) {
        fprintf(stderr, "test-tsr-pivot-graph: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test-tsr-pivot-graph: ok\n");
    return 0;
}

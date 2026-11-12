// tsr-build.h — the M7 part of the op-list builder of the ggml-tesseract backend
// (docs/design/06-software-verification.md "SW" §9.3 M7: builder passes 1-6, supports_op (M7), mask
// summaries, KV buffers; §8 R1-R10 one launch per decode token; §7.2 OPL kinds and their argument
// words; docs/design/03-attention-kv.md "AT" §2.6 the supports_op contract, §12 the FA flags word).
//
// ggml-aware and device-free: the backend (ggml-tesseract.cpp) supplies where tensors live through
// tsr_build_env's callbacks, the unit tests (tests/test-tsr-backend-units.cpp) supply fake ones.
//
//   tsr_lower_m7        a ggml node of an M7 kind -> one OPL record (kind, argument words, preload
//                       references, the GET_ROWS source extent): pass 2 (each tensor resolved to a
//                       handle and a row stride in lines) and the encoding words of pass 6. The same
//                       function serves supports_op in "probe" mode (no device locations: handles 0,
//                       alignment from the tensor's address or view offset) followed by the kind's
//                       planner (tsr/plan.hpp plan_kind, SW-20), so the host refuses exactly what
//                       the firmware cannot plan and what the builder cannot encode.
//   tsr_build_supports  supports_op for every op but MUL_MAT (whose rules stay the backend's); a kind
//                       the booted image does not provide is refused (env.kinds, SW §6.9).
//   tsr_build_fusion    pass 3: MUL_MAT + ADD of a bias or the residual (through the MUL_MAT
//                       addend, SW §7.4) and RMS_NORM + MUL by a weight (the fused weight word).
//   tsr_encode          passes 4-6 for one record: preload entries (deduplicated per list, the
//                       FA_NEW word of FLASH_ATTN), DRAM ranges (tsr_opl_builder::kind_ranges), RAW /
//                       WAR / WAW dependencies with transitive reduction and JOINs (tsr_opl_builder).
//   tsr_graph_key       the identity of a graph for the per-graph cache (R7): nodes, flags, shapes,
//                       strides, addresses and op parameters, not the tensors' values, so a decode
//                       token's list is reused while its inputs change (positions, KV indices and
//                       masks reach the device through the preload table).
//
// Pass 1 (drop NONE / VIEW / RESHAPE / PERMUTE / TRANSPOSE and nodes without GGML_TENSOR_FLAG_COMPUTE)
// is tsr_node_is_noop / the backend's loop.
//
// Decisions (recorded in SW §9.3 / §13):
//   - supports_op does not test GGML_TENSOR_FLAG_COMPUTE (SW §8 R1 / SW-33 say it should):
//     test-backend-ops asks supports_op before ggml_build_forward_expand sets the flag
//     (tests/test-backend-ops.cpp:1415-1421), so the rule would refuse every op there. The nodes
//     the rule is for are refused anyway (DUP, a SET_ROWS whose destination is not a KV cache, the
//     token-embedding GET_ROWS of a quantized table) and graph_compute skips flagless nodes; a split
//     made only of flagless nodes costs no launch.
//   - Fusion places the fused record at the MUL_MAT's / RMS_NORM's position when the second node
//     is its only user (ggml use count 1 over the whole graph, not a view, not an output: ggml's own
//     ggml_node_has_n_uses rule, so a consumer in another split or an observed node is never
//     fused away), its other operand is not written and its result is not read or written by a
//     node in between (byte ranges of the tensors' addresses), and every operand is addressable in
//     place. Bit-identical to the unfused device pair: ACC adds y = addend + W·x in one f32
//     addition (commutative, SW §7.4), and RMS_NORM's VOP.MULCB multiplies (x·s)·w as the separate
//     MULC then MUL (checked by ggml_tesseract_opl_launch).
//   - FA_NEW (SW §7.2 word 21, AT §12): the FLASH_ATTN record names the indices of the SET_ROWS_KV
//     that wrote its K plane earlier in the same graph_compute (the same preload entry when both are
//     in one list; added to the FA's list otherwise), so the chunk order, hence the output bits, do
//     not depend on where the builder cut its lists. supports_op therefore refuses a FLASH_ATTN of
//     more than 680 tokens: its MSUM and the indices must fit one fresh list's 256 preload lines.
//     The FLASH_ATTN is not a ggml consumer of the K indices, so ggml-alloc may reuse their bytes
//     after the last SET_ROWS_KV that reads them. When a cut separates the two, the FA's list
//     preloads them from DRAM at its start, which is right only while no node between them has
//     written those bytes. In a layer with its own KV only the V SET_ROWS_KV lies between the
//     two. A layer that shares KV (Gemma 3n and 4 E-series) reads a plane that an earlier layer
//     wrote, with the rest of the layers' nodes between, and those nodes' results can lie on the
//     indices. The backend then copies the indices to a device area of its own right after the
//     SET_ROWS_KV (tsr_lower_kidx_snapshot) and every FLASH_ATTN in a later list preloads FA_NEW
//     from the copy; one in the list of the copy reads the original, which is intact at the start
//     of that list. The backend aborts naming the nodes only when it cannot take the copy.
//   - FA rule-5 refusals (AT §6.6 rule 5: data-dependent, invisible to supports_op): the mask
//     producer (tsr-msum.h) plans every query tile of every registered reader and promotes
//     the MSUM to GENERAL with the f32 copy when a tile would be refused; the builder registers each
//     FLASH_ATTN as a reader (env.fa_mask -> tsr_msum_store::attach) on every build and every cached
//     reuse, so the executor never refuses a tile for mask data.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"

#include "tsr-kvbuf.h"
#include "tsr-msum.h"
#include "tsr-opl.h"

// An expert weight tensor as MUL_MAT_ID addresses it (SW-60): the device format, the W_OFF of expert 0 and the
// per-channel stride of one expert (both in beats), and the number of experts. Probe mode passes none.
struct tsr_moe_w {
    uint32_t fmt = 0;
    uint32_t w_off = 0;
    uint32_t exp_stride = 0;
    uint32_t n_expert = 0;
};

// Where a tensor's first byte lives on the device.
struct tsr_tloc {
    uint32_t ch = 0;
    uint64_t off = 0;     // channel byte offset
};

struct tsr_build_env {
    tsr::plan::dev_cfg     cfg = {};     // the device (tsr_plan_cfg)
    tsr::plan::plan_limits pl = {};      // plan_limits_of(cfg): what the firmware plans against
    uint32_t num_ch = 1;
    // The op list kinds the booted image provides (SW §6.9: bit k set iff its .tsr_kinds has kind k;
    // TSR_KINDS narrows it, tsr_kinds_parse) and the image named in the reason of a refusal
    // ("Tesseract Basis 1.0.0, image <id>"). A record of another kind (or any record when JOIN is
    // missing) is refused while lowering, in probe and build mode.
    uint64_t kinds = ~0ull;
    std::string image;

    // Build mode: where the device addresses the tensor's bytes in place (a compute buffer's window
    // in IO or the compute heap; a raw tensor of a weight buffer in AUX or the heap). False when it
    // cannot (host memory, a device-format weight, not allocated).
    std::function<bool(const ggml_tensor *, tsr_tloc *)> loc;
    // Build mode: the plane of a view of a placed KV-cache tensor (tsr_kv_view_of).
    std::function<bool(const ggml_tensor *, tsr_kv_view *)> kv_view;
    // Build mode: a FLASH_ATTN's mask words (tsr_fa_mask_ref, i.e. tsr_msum_store::attach /
    // nomask); valid = false: no summary.
    std::function<tsr_msum_ref(const ggml_tensor *)> fa_mask;

    // Probe and build mode: t (or the tensor it views) has a buffer, and it is a Tesseract buffer. An allocated tensor of
    // another backend (a CPU recurrent-state cache with -nkvo) keeps the ops that read or write it in place off the device.
    std::function<bool(const ggml_tensor *)> on_device;
    // Probe mode (supports_op): whether the bytes of t can be read / written as raw rows by the device
    // (false: a device-format weight, a KV cache); t may be unallocated.
    std::function<bool(const ggml_tensor *)> raw_ok;
    // Probe mode: t (or the tensor it views) is a KV-cache tensor in a Tesseract buffer that is or will
    // be a KV buffer (tsr_kv_tensor_supported); its shape (tsr_kv_identify) in *out.
    std::function<bool(const ggml_tensor *, tsr_kv_shape *)> kv_probe;
    // Probe mode: a FLASH_ATTN's mask can be summarized (tsr_msum_mask_supported); no mask is
    // always fine (nomask).
    std::function<bool(const ggml_tensor *)> fa_mask_probe;
    // Probe mode: t is a weight MUL_MAT_ID can address: the type loads (a device format or convertible), the
    // buffer holds weights or raw bytes the backend converts lazily (the rules of MUL_MAT's src0).
    std::function<bool(const ggml_tensor *)> moe_w_probe;
    // Build mode: the device scratch area of at least `bytes` bytes (64-byte aligned, in device memory the kinds address in
    // place; the same area for every call, grown when needed, never part of a graph's tensors). Only the records of one
    // op sequence use it at a time: its users write it, read it back and are done before the next user starts (SW-76).
    // False when the device has no room for it.
    std::function<bool(uint64_t bytes, tsr_tloc *)> scratch;
    // Build mode: the values of an I32 graph input as the next launch will see them (the backend reads them from its
    // buffer). False when they cannot be read. A graph whose plan depends on the values (the recurrent-state gather of a
    // GATED_DELTA_NET, tsr_gdn_cell) is built only with it, and the backend checks the values again before it reuses
    // the plan.
    std::function<bool(const ggml_tensor *, std::vector<int32_t> *)> read_i32;
    // Whether tsr_build_fusion may put a GATED_DELTA_NET on its state cache cell (tsr_gdn_cell). TSR_GDN_CELL=0 turns it off
    // (the three passes of the pinned graph: the same results, for a comparison).
    bool gdn_cell = true;
};

// Pass 1: nodes that cost nothing on the device.
bool tsr_node_is_noop(const ggml_tensor * node);

// One preload reference of a record: argument word `word` holds the preload byte offset of `bytes`
// bytes from DRAM handle `src` (SW §7.2 "Preload entry").
struct tsr_pre_ref {
    uint32_t word;
    uint32_t src;
    uint32_t bytes;
};

// A lowered record (before encoding).
struct tsr_rec {
    uint32_t kind = tsr::opl::K_NOP;
    uint32_t n_args = 0;
    uint32_t args[tsr::opl::fa::NARGS] = {};   // the longest kind (FLASH_ATTN, 22 words)
    std::vector<tsr_pre_ref> pre;
    uint64_t aux = 0;                          // kind_ranges' aux: GET_ROWS source rows
    // SET_ROWS_KV: its plane and the indices it writes (the FA_NEW of a FLASH_ATTN reading the plane).
    // FLASH_ATTN: its K plane; FA_NEW is resolved by tsr_encode.
    uint32_t kv_plane = 0;
    uint32_t kv_idx_src = 0, kv_idx_n = 0;
    tsr_msum_ref mask;                         // FLASH_ATTN: the mask words used
    // Further records of the same node (one per token of an RMS_NORM whose source tokens are not spaced like its
    // heads, tsr_lower_rms); run in order after this one. Probe mode fills none: all share the first one's plan.
    std::vector<tsr_rec> more;
};

// The FA_NEW source of each KV plane written in this graph_compute: K plane base beat -> (handle of
// the I64 indices, count). Filled by tsr_note_kv_write from the SET_ROWS_KV records in program order.
using tsr_fanew_map = std::map<uint32_t, std::pair<uint32_t, uint32_t>>;
void tsr_note_kv_write(const tsr_rec & r, tsr_fanew_map * m);

// The K plane (base beat, as tsr_rec::kv_plane) that the FLASH_ATTN node reads; false when its K is not a view of a placed
// KV plane. Build mode (env.kv_view).
bool tsr_fa_k_plane(const tsr_build_env & env, const ggml_tensor * fa, uint32_t * plane);

// FA_NEW across a cut when the K indices were overwritten (the assumption of FA_NEW below): the CPY record that copies the n I64
// indices at DRAM handle `src` (ceil(8 n / 64) lines, one row) to `dst`, a 64-byte-aligned area the graphs do not use. The
// backend puts it right after the SET_ROWS_KV and lets later lists preload FA_NEW's entry from `dst`. False with *why when the
// image has no CPY kind or the planner refuses the copy.
bool tsr_lower_kidx_snapshot(const tsr_build_env & env, uint32_t src, uint32_t n, uint32_t dst, tsr_rec * out, std::string * why);

// Pass 2 / 6: the record of an M7 node (every op of the SW §9.3 table but MUL_MAT; the no-ops have
// none). probe: supports_op's question (handles 0, preload offsets 0, the KV shape from kv_probe, no
// side effects); else the device's words (env.loc / kv_view / fa_mask). fused: the MUL that
// tsr_build_fusion folded into this RMS_NORM (its weight and its result), or the state CPY of a
// GATED_DELTA_NET in place (tsr_gdn_cell: its destination is the cell the record reads and writes), else nullptr. False with
// *why when the node cannot be expressed as a record of its kind.
bool tsr_lower_m7(const tsr_build_env & env, bool probe, const ggml_tensor * node, const ggml_tensor * fused,
                  tsr_rec * out, std::string * why);

// The records of a GATED_DELTA_NET node with its state group (tsr_gdn_cell): the node's record, or for K > 1 state snapshots the
// chain of min(T, K) records (SW-83), in out and out->more. The CPY writes the cache's groups: in_place says whether the first
// record also reads its state from the cell the CPY writes (the ids are the identity), else it reads the state tensor.
bool tsr_lower_gdn_group(const tsr_build_env & env, const ggml_tensor * node, const ggml_tensor * cpy, bool in_place, tsr_rec * out,
                         std::string * why);

// The MUL_MAT half of a fused MUL_MAT + ADD (pass 3): sets args' Y / Y_STRIDE to the ADD's result and
// ADD / ADD_STRIDE to its other operand (a bias row: stride 0; the residual: its row stride).
// False with *why if they cannot be addressed (the record is then left as it was).
bool tsr_fuse_mm_add(const tsr_build_env & env, const ggml_tensor * mm, const ggml_tensor * add, uint32_t * mm_args,
                     std::string * why);

bool tsr_fuse_mm_scale(const tsr_build_env & env, const ggml_tensor * mm, const ggml_tensor * scale,
                       uint32_t * mm_args);
bool tsr_fuse_mm_softcap(const tsr_build_env & env, const ggml_tensor * mm, const ggml_tensor * first,
                         const ggml_tensor * last, uint32_t * mm_args);

// supports_op for every op but MUL_MAT (SW §9.3 table; AT §2.6): lowers in probe mode and asks the
// kind's planner (plan_kind with env.pl), then the 64-byte rule (every operand's base offset and row
// stride a multiple of 64 bytes, and a result that is a view has rows of whole lines; checked while
// lowering). The no-ops are supported; DUP and a SET_ROWS that does not write a KV cache are not (R1).
bool tsr_build_supports(const tsr_build_env & env, const ggml_tensor * op, std::string * why);

// MoE (SW-60 to SW-65; docs/spec.md §13). llama.cpp's build_moe_ffn emits, for softmax gating with a normalized top-k:
//     ffn_moe_probs  SOFT_MAX(logits)                 [n_expert, T]
//     ffn_moe_argsort  ARGSORT(probs, DESC), ffn_moe_topk  VIEW [k, T]
//     ffn_moe_weights  GET_ROWS(RESHAPE(probs) [1, n, T], topk) [1, k, T]   (+ RESHAPE [k, T])
//     SUM_ROWS, CLAMP(6.1035e-5, inf), DIV            [k, T] (+ RESHAPE [1, k, T])
// one MOE_ROUTE record absorbs the six nodes (the group of the SOFT_MAX): the ids are written into the first line of
// each row of the ARGSORT tensor, the weights into the DIV tensor (for more than 64 tokens through a device scratch and two
// CPY records, SW-76). And, after a MUL_MAT_ID with a per-expert output
// scale (Gemma 4's ffn_down_exps.scale):
//     RESHAPE(scale) [1, E], REPEAT [1, E, T], GET_ROWS(repeat, topk) [1, k, T], MUL(MUL_MAT_ID, gathered)
// the MUL_MAT_ID record takes the scale table (y = FM(dot, scale[id])) and a CPY record moves its result into the MUL's
// tensor; REPEAT, GET_ROWS and MUL are absorbed. A node of either chain that the fusion did not absorb has no device kind.
struct tsr_moe_route {
    int softmax = -1, argsort = -1, get_rows = -1, sum_rows = -1, clamp = -1, div = -1;
    const ggml_tensor * topk = nullptr;      // the VIEW [k, T] of the ARGSORT tensor (ids of the experts)
};
struct tsr_moe_scale {
    int mmid = -1, repeat = -1, get_rows = -1, mul = -1;
    const ggml_tensor * table = nullptr;     // the F32 tensor of n_expert elements (a leaf: the weight, or the scheduler's copy of its reshaped view)
};

// The SSM_CONV record of a Gated DeltaNet layer (SW §7.10): the SSM_CONV node with its CONCAT absorbed, the state CPY
// (nullptr in probe mode) and an optional SILU. False with *why if the group cannot be expressed as one record.
// slots (SW-83, build mode): every state CPY of the group when the graph keeps rollback snapshots, `cpy` being the one of slot 0
// (the final state): the record writes slot 0 itself and CPY records (r->more) copy the window rows of the other slots into
// the cache's groups.
bool tsr_lower_conv_group(const tsr_build_env & env, bool probe, const ggml_tensor * conv, const ggml_tensor * silu,
                          const ggml_tensor * cpy, tsr_rec * r, std::string * why,
                          const std::vector<const ggml_tensor *> * slots = nullptr);

// The window of an SSM_CONV group, at its CONCAT: two CPY records (the conv state rows, the token rows) into the CONCAT's
// result buffer (SW §7.10; ggml-alloc frees the CONCAT's inputs after it, the SSM_CONV record reads the window).
bool tsr_lower_conv_window(const tsr_build_env & env, bool probe, const ggml_tensor * concat, tsr_rec * recs, uint32_t * n,
                           std::string * why);

// Pass 3: fusion pairs of a graph (decisions above). with[i] = k: node i's record absorbs node k
// (MUL_MAT + ADD or RMS_NORM + MUL); into[k] = i. Pairs need build-mode locations (env.loc).
// route_of[i] / scale_of[i]: the index into routes / scales of the group headed by node i (SOFT_MAX / MUL_MAT_ID), else -1;
// into[k] = the head for every absorbed member of a group.
// The state cell of a GATED_DELTA_NET (SW-82). llama.cpp's build_rs gathers the layer's state rows from the cache by the
// ids of `rs_s_copy` (GET_ROWS), the recurrence reads that copy and writes a new state into its own result, and a CPY
// moves the result into the cache cell of the sequence (kv_head + j). When the ids say that every sequence keeps its cell
// (ids[j] = cell0 + j, the CPY's destination) and nothing else reads the copy or the result's state rows, the gather and the
// CPY change nothing the graph can see: the record reads the cell and writes the new state back into it (STATE_IN =
// STATE_OUT = the cell), one pass over the state instead of three. The group is made for the GATED_DELTA_NET node when the
// graph has the gather and the CPY in the shape above (`gather`, `cpy` >= 0; `ids` the index input, `n` its length,
// `cell0` the cell of the CPY's first row) and `in_place` when the ids of this launch are the identity; the plan keeps
// the ids tensor and the decision and is rebuilt when they disagree (tsr_plan_state_holds).
struct tsr_gdn_cell {
    int32_t gather = -1, cpy = -1;
    const ggml_tensor * ids = nullptr;
    uint32_t n = 0;
    int32_t cell0 = 0;
    bool in_place = false;
    // SW-83: the snapshot slots the CPY writes (min(T, K); the CPY is then a 3D view of the cache, one group per slot), and
    // whether the record absorbs the CPY (in place, or more than one slot: the record writes the cache's groups itself and the
    // gather, if the ids are not the identity, still runs and the record reads its copy)
    uint32_t slots = 1;
    bool absorb_cpy = false;
};

struct tsr_fusion {
    std::vector<int32_t> with, into, softcap_mid, softcap_tail;
    std::vector<int32_t> route_of, scale_of;
    std::vector<tsr_moe_route> routes;
    std::vector<tsr_moe_scale> scales;
    // SSM_CONV groups (tsr_lower_conv_group): for the SSM_CONV node i, the CONCAT, the state CPY (both earlier in the
    // graph) and the SILU (later) that its record absorbs; -1 for no group (silu: the group without SILU)
    std::vector<int32_t> conv_concat, conv_cpy, conv_silu;
    // SW-83: every state CPY of an SSM_CONV group (conv_cpy[i] is the one of slot 0, the final state)
    std::vector<std::vector<int32_t>> conv_cpys;
    // GATED_DELTA_NET nodes with their state gather and state CPY (tsr_gdn_cell): by node index
    std::vector<tsr_gdn_cell> gdn_cell;
    uint32_t n_pairs = 0;
};
tsr_fusion tsr_build_fusion(const tsr_build_env & env, const ggml_cgraph * cg);

// The structure of the fusion groups, the part of their conditions that does not depend on the device, the addresses or the
// token count. tsr_build_fusion applies it and adds what the device and the addresses decide; Tesseract Pivot
// (tsr-pivot-graph.h) takes the same groups as the units it must keep together, so the two cannot disagree on a group's shape.
// A node that matches may still not fuse (placement, a node written in between): the unit then holds nodes the builder lowers
// separately, which costs nothing.
//   tsr_fusion_user     the first node after i that lists node i as a source, or -1 (a user in another graph)
//   tsr_fuse_pair_shape u, the first user of the MUL_MAT or RMS_NORM `node`, is the SCALE or the ADD (a bias or the residual of
//                       the MUL_MAT's shape) that its record absorbs, or the MUL by a weight that an RMS_NORM record absorbs
//   tsr_softcap_shape   behind the SCALE k of a MUL_MAT: the TANH and the closing SCALE of a softcap, each the only user of the
//                       one before (*mid, *tail)
//   tsr_moe_route_match the routing group headed by the SOFT_MAX i (group above), tsr_moe_scale_match the scale group of the
//                       MUL_MAT_ID i
//   tsr_conv_group_shape the SSM_CONV node i with its CONCAT, state CPYs and SILU (the SILU is -1 when there is none)
//   tsr_gdn_state_shape the GATED_DELTA_NET node i with its state gather and state CPY
struct tsr_conv_shape {
    int concat = -1, cpy = -1, silu = -1;
    std::vector<int32_t> cpys;            // every state CPY (cpy is the one of the final state)
};
struct tsr_gdn_shape {
    int gather = -1, cpy = -1;
    uint32_t slots = 1, n_seq = 0;
    int32_t cell0 = 0;
    const ggml_tensor * ids = nullptr;    // the gather's index input
};
int  tsr_fusion_user(const ggml_cgraph * cg, int i);
bool tsr_fuse_pair_shape(const ggml_tensor * node, const ggml_tensor * u);
bool tsr_softcap_shape(const ggml_cgraph * cg, int k, int * mid, int * tail);
bool tsr_moe_route_match(const ggml_cgraph * cg, int i, tsr_moe_route * out);
bool tsr_moe_scale_match(const ggml_cgraph * cg, int i, tsr_moe_scale * out);
bool tsr_conv_group_shape(const ggml_cgraph * cg, int i, tsr_conv_shape * out);
bool tsr_gdn_state_shape(const ggml_cgraph * cg, int i, tsr_gdn_shape * out);


// MoE lowering. tsr_lower_moe_route: the records of a router group, recs[0 .. *n) (at most TSR_MOE_ROUTE_RECS): for up to
// 64 tokens (one tile) the one MOE_ROUTE record, which writes the ids and the weights where the graph has them; for more
// tokens (SW-76) the MOE_ROUTE record writes them to the device scratch (env.scratch) and two CPY records move them into
// the ARGSORT and DIV tensors, because the allocator may have put those where the logits of a later tile still are;
// tsr_lower_mul_mat_id: the MUL_MAT_ID record of a node, with the scale group's table when sc != nullptr (the record then
// writes the MUL_MAT_ID's own tensor); w: the resolved weights (nullptr in probe mode). False with *why if a record cannot
// be expressed or the firmware's planner refuses it.
constexpr uint32_t TSR_MOE_ROUTE_RECS = 3;
bool tsr_lower_moe_route(const tsr_build_env & env, const ggml_cgraph * cg, const tsr_moe_route & g, tsr_rec * recs, uint32_t * n,
                         std::string * why);
bool tsr_lower_mul_mat_id(const tsr_build_env & env, bool probe, const ggml_tensor * node, const tsr_moe_scale * sc,
                          const tsr_moe_w * w, tsr_rec * out, std::string * why);
// The CPY record that moves the scaled MUL_MAT_ID result into the MUL's tensor (src: the MUL_MAT_ID's tensor).
bool tsr_lower_moe_scale_copy(const tsr_build_env & env, const ggml_tensor * mmid, const ggml_tensor * mul, tsr_rec * out,
                              std::string * why);


// Passes 4-6 for one record: its preload entries (deduplicated; FA_NEW from fanew for a FLASH_ATTN),
// their offsets into its words, its DRAM ranges and its dependencies (b.add). Returns the op index,
// tsr_opl_builder::NO_ROOM (the record, its JOINs or its preload entries do not fit this list:
// nothing was added; run the list and encode into a new one) or FAILED (writer error).
uint32_t tsr_encode(tsr_opl_builder & b, tsr_rec & r, int32_t tag, const tsr_fanew_map & fanew);

// The identity of a graph for the per-graph cache (R7): every node's address, op, flags, type,
// shape, strides, data address, op parameters and sources (address, data, type, shape, strides),
// mixed into a 64-bit hash with `salt` (the backend's placement epoch). Values are not part of it.
uint64_t tsr_graph_key(const ggml_cgraph * cg, uint64_t salt);

// Name of a kind for logs and statistics ("MUL_MAT", "FLASH_ATTN", ...): tsr/opl.hpp kind_name.
const char * tsr_kind_name(uint32_t kind);

// The booted image provides kind (and JOIN, which every list may need): env.kinds (SW §6.9).
inline bool tsr_kind_in_image(uint64_t kinds, uint32_t kind) {
    return kind < tsr::opl::N_KINDS && ((kinds >> kind) & 1u) != 0 && ((kinds >> tsr::opl::K_JOIN) & 1u) != 0;
}

// TSR_KINDS (SW-14, SW-49): the kinds supports_op may route, as a mask the backend intersects with the
// booted image's. A comma-separated list read from left to right: "all" (every kind), "m6" (NOP, JOIN and
// MUL_MAT), "NAME" or "+NAME" adds a kind, "-NAME" removes one (names as tsr/opl.hpp kind_name, any
// case). The list starts from every kind if its first word removes one, else from NOP and JOIN, which
// cannot be removed. Unset or empty: every kind. Examples: "m6", "-FLASH_ATTN,-SET_ROWS_KV",
// "all,-RMS_NORM", "m6,+RMS_NORM". False with *why on an unknown word.
bool tsr_kinds_parse(const char * spec, uint64_t * mask, std::string * why);

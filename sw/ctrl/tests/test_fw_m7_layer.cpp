// test_fw_m7_layer.cpp — fw_m7_layer_*: one whole decoder layer as one OPL list on the RISC-V image
// (docs/design/06-software-verification.md "SW" §7.2, §7.5 last paragraph, §7.8,
// §10.2). The list is what the backend builds for a layer (tile
// ops of every M7 kind in one launch, the executor's deps, scratch reuse across kinds, overlap of
// the MVM with the VU / AT kinds), launched through the `graph` kernel.
//
//   test_fw_m7_layer IMAGE suite GROUP [--scheds N] [--threads N] [--only NAME]
//       Every case of the group (qwen05b: Qwen2.5-0.5B layer 0 decode T = 1 and a prefill tile T = 8;
//       gemma12b: a Gemma 4 12B SWA layer, decode and prefill tile) under the fixed schedule and
//       N - 1 random ones (default 8) with TSR_SIM_RACE fatal: DONE OK / RET 0; every record's
//       output rows and both KV planes equal, bit for bit, to the host composition of the per-kind
//       references applied in op order to a shadow DRAM (sw/ctrl/ref: mulmat_ref, rmsnorm_ref,
//       rope_ref, elem_ref, kv_ref, fattn_ref); every written DRAM beat inside an output or a plane;
//       no race report; the SW §4.7 observables and DONE r0-r6 equal under every schedule.
//       Weights: Qwen2.5-0.5B's layer 0 from models/qwen2.5-0.5b-instruct-q8_0.gguf (a minimal
//       GGUF v3 reader below; the inputs are the dequantized token embeddings of fixed token ids),
//       converted to the device layout by the tsr_ref layout functions (tsr_row_location +
//       tsr_repack_row); without the file, random Q8_0 weights in the same shapes (printed). Gemma
//       4 12B: random weights and inputs in the shapes of its SWA layers (n_embd 3840, 16 / 8 heads
//       of 256, n_ff 15,360, window 1,024). KV history: random rows in every cell before the new
//       tokens' cells (QACT A32 + KVST, the device image); the new tokens' cells are SET_ROWS_KV's.
//       k4min: the device of the reduced card image tsr_u200_k4 (4 channels, T_MAX 8, CAPS 0): the seven MUL_MAT
//       records of a layer only (M6; build_mm), decode and a T = 8 prefill tile, against mulmat_ref; `dump` writes
//       them as RTL scenarios (tb/core/test_core.py test_core_m6_k4min).
//   test_fw_m7_layer IMAGE ggml
//       (TSR_WITH_LLAMA) each case's device output (fixed schedule) against ggml-cpu computing the
//       same layer as llama.cpp's graph does (qwen2: src/models/qwen2.cpp; gemma4: src/models/
//       gemma4.cpp, SWA layer) on the same weights, inputs, positions and q8_0 KV cache (the planes
//       read back with tsr_kv_get_bytes), FLASH_ATTN_EXT with the F16 mask: NMSE, max |diff| and the
//       row-softmax KL of the layer output and of the attention output, twice: with the CPU's own new
//       K / V cells (quantized by ggml's reference quantizer; their codes against the device's) and
//       with the device's cache (the arithmetic after it, gates near the measurements).
//   test_fw_m7_layer IMAGE perf
//       Measurements (SW §7.8; fixed schedule, the Ibex table of TSR_SIM_RV_CORE (default maxperf),
//       U200 queue depths and deep queues, the backend's list flags (no MARK)): the 0.5B decode layer
//       as a list of 1 and of 3 layers (shared weights, a KV plane pair per layer); per record kind
//       the RISC-V cycles of work per record (fw_prof.h: spin-free busy cycles, attributed per record
//       between the executor's RT_INFO writes); RISC-V work vs the units' time per layer; MVM / VU /
//       DMA busy and issue-idle (the unit idle while its next op was not pushed yet); push -> DONE;
//       the same with a near-ideal issuer (F_RV = 4 F_CORE): the share of the layer the RISC-V's
//       speed costs; at F_RV 166.7 MHz and 250 MHz.
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image is missing.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "../ref/elem_ref.hpp"
#include "../ref/fw_sim_host.hpp"
#include "../ref/kvfa_ref.hpp"
#include "../ref/mulmat_ref.hpp"
#include "../ref/rmsnorm_ref.hpp"
#include "../ref/rope_ref.hpp"
#include "fw_prof.h"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_race.h"

#if defined(TSR_M7L_GGML)
#include "ggml-cpu.h"
#include "ggml.h"
#endif

#ifndef TSR_M7L_MODELS
#define TSR_M7L_MODELS "models"
#endif

namespace {

namespace opl = tsr::opl;
constexpr uint64_t LINE = 64u;
constexpr double F_CORE = 276.3;   // MHz (U200 core clock)

const char * g_image = nullptr;
uint32_t g_threads = 2;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t h_ch(uint32_t h) { return h >> 28; }
uint64_t h_beat(uint32_t h) { return h & 0x0FFFFFFFu; }
uint32_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
float b2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// =================================================================================================
// A minimal GGUF v3 reader (tensor table, a few scalar keys; the data are read on demand)
// =================================================================================================
struct gguf_tensor {
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    uint64_t off = 0;   // absolute file offset of the data
};
class gguf_file {
public:
    ~gguf_file() { if (f_) fclose(f_); }
    bool open(const char * path) {
        f_ = fopen(path, "rb");
        if (!f_) return false;
        uint32_t magic = 0, ver = 0;
        uint64_t nt = 0, nkv = 0;
        if (!rd(&magic, 4) || magic != 0x46554747u || !rd(&ver, 4) || ver < 2u || !rd(&nt, 8) || !rd(&nkv, 8)) return false;
        uint64_t align = 32;
        for (uint64_t i = 0; i < nkv; ++i) {
            std::string key;
            uint32_t ty = 0;
            if (!rd_str(&key) || !rd(&ty, 4)) return false;
            double v = 0;
            if (!rd_val(ty, &v)) return false;
            num_[key] = v;
            if (key == "general.alignment") align = uint64_t(v);
        }
        std::vector<std::pair<std::string, gguf_tensor>> ts;
        for (uint64_t i = 0; i < nt; ++i) {
            std::string name;
            uint32_t nd = 0;
            if (!rd_str(&name) || !rd(&nd, 4) || nd > 4u) return false;
            gguf_tensor t;
            t.ne.resize(nd);
            for (uint32_t d = 0; d < nd; ++d)
                if (!rd(&t.ne[d], 8)) return false;
            if (!rd(&t.type, 4) || !rd(&t.off, 8)) return false;
            ts.emplace_back(name, t);
        }
        long pos = ftell(f_);
        const uint64_t data = (uint64_t(pos) + align - 1u) / align * align;
        for (auto & kv : ts) {
            kv.second.off += data;
            t_[kv.first] = kv.second;
        }
        return true;
    }
    const gguf_tensor * tensor(const std::string & n) const {
        auto it = t_.find(n);
        return it == t_.end() ? nullptr : &it->second;
    }
    bool read(const gguf_tensor & t, void * dst, size_t bytes) {
        return fseeko(f_, off_t(t.off), SEEK_SET) == 0 && fread(dst, 1, bytes, f_) == bytes;
    }
    double num(const std::string & k, double dflt) const {
        auto it = num_.find(k);
        return it == num_.end() ? dflt : it->second;
    }

private:
    bool rd(void * p, size_t n) { return fread(p, 1, n, f_) == n; }
    bool rd_str(std::string * s) {
        uint64_t n = 0;
        if (!rd(&n, 8) || n > (1u << 20)) return false;
        s->resize(size_t(n));
        return n == 0 || rd(s->data(), size_t(n));
    }
    // One value of type ty (numbers as double; strings and arrays skipped).
    bool rd_val(uint32_t ty, double * v) {
        static const uint8_t sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (ty == 8u) {
            std::string s;
            return rd_str(&s);
        }
        if (ty == 9u) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!rd(&et, 4) || !rd(&n, 8)) return false;
            for (uint64_t i = 0; i < n; ++i) {
                double x;
                if (!rd_val(et, &x)) return false;
            }
            return true;
        }
        if (ty > 12u) return false;
        uint8_t b[8] = {};
        if (!rd(b, sz[ty])) return false;
        switch (ty) {
            case 0: *v = b[0]; break;
            case 1: *v = int8_t(b[0]); break;
            case 2: { uint16_t x; memcpy(&x, b, 2); *v = x; break; }
            case 3: { int16_t x; memcpy(&x, b, 2); *v = x; break; }
            case 4: { uint32_t x; memcpy(&x, b, 4); *v = x; break; }
            case 5: { int32_t x; memcpy(&x, b, 4); *v = x; break; }
            case 6: { float x; memcpy(&x, b, 4); *v = x; break; }
            case 7: *v = b[0]; break;
            case 10: { uint64_t x; memcpy(&x, b, 8); *v = double(x); break; }
            case 11: { int64_t x; memcpy(&x, b, 8); *v = double(x); break; }
            case 12: { double x; memcpy(&x, b, 8); *v = x; break; }
        }
        return true;
    }
    FILE * f_ = nullptr;
    std::map<std::string, gguf_tensor> t_;
    std::map<std::string, double> num_;
};
constexpr uint32_t GGUF_F32 = 0u, GGUF_Q8_0 = 8u;

// =================================================================================================
// Models and cases
// =================================================================================================
struct model {
    const char * name;
    uint32_t n_embd, n_head, n_kv_head, D, n_ff;
    float eps, rope_base, attn_scale;
    bool bias;       // q / k / v bias addends (Qwen2)
    bool gemma;      // per-head q / k norms (weight) and v norm (none), post norms, ADD residuals, out_scale
    uint32_t glu;    // opl::glu variant
    uint32_t n_swa;  // 0: causal
};
// Qwen2.5-0.5B (GGUF qwen2.*: 896, 14 / 2 heads of 64, n_ff 4864, eps 1e-6, rope base 1e6, NEOX)
const model QWEN05B = {"qwen2.5-0.5b", 896, 14, 2, 64, 4864, 1e-6f, 1e6f, 0.125f, true, false, opl::glu::SWIGLU, 0};
// Gemma 4 12B SWA layers (GGUF gemma4.*: 3840, 16 / 8 heads of 256 (key_length_swa), n_ff 15,360, eps 1e-6,
// rope base swa 1e4, NEOX, window 1,024, attention scale 1.0, layer_output_scale)
const model GEMMA12B = {"gemma4-12b-swa", 3840, 16, 8, 256, 15360, 1e-6f, 1e4f, 1.0f, false, true, opl::glu::GEGLU16, 1024};

// a decoder layer in the smallest shapes that still use every M7 kind (Qwen2 structure: bias
// addends, SWIGLU; 128 / 2 heads of 64 / 1 KV head, n_ff 256). The RTL core bench (tb/core, test_core_m7)
// runs it on a 2-channel core with T_MAX 2 and 1 MiB of SPM (`dump` mode below).
const model TINY = {"tiny", 128, 2, 1, 64, 256, 1e-6f, 1e6f, 0.125f, true, false, opl::glu::SWIGLU, 0};

struct lcase {
    const char * name;
    const char * group;
    const model * m;
    uint32_t T, n_kv, pos0;
    bool gguf;        // Qwen: weights / inputs from the GGUF file when present
    bool tiny = false;   // the RTL bench's reduced device (2 channels, T_MAX 2, 1 MiB SPM, CAPS VU | ATT)
    bool mm = false;     // M6: MUL_MAT records only, on the device of the reduced card image (build_mm)
};
const std::vector<lcase> & cases() {
    static const std::vector<lcase> v = {
        {"qwen05b-decode", "qwen05b", &QWEN05B, 1, 256, 200, true},
        {"qwen05b-prefill", "qwen05b", &QWEN05B, 8, 256, 120, true},
        {"gemma12b-swa-decode", "gemma12b", &GEMMA12B, 1, 2048, 1500, false},
        {"gemma12b-swa-prefill", "gemma12b", &GEMMA12B, 8, 2048, 1100, false},
        {"tiny-decode", "tiny", &TINY, 1, 64, 40, false, true},
        {"tiny-prefill", "tiny", &TINY, 2, 64, 30, false, true},
        // tsr_u200_k4 (docs/platform-u200.md §8.7): 4 channels, T_MAX 8, CAPS 0, MUL_MAT records only (M6)
        {"k4min-mm-decode", "k4min", &TINY, 1, 64, 40, false, false, true},
        {"k4min-mm-prefill", "k4min", &TINY, 8, 64, 30, false, false, true},
    };
    return v;
}
const lcase & find_case(const char * n) {
    for (const auto & c : cases())
        if (!strcmp(c.name, n)) return c;
    fprintf(stderr, "no case %s\n", n);
    exit(2);
}

// =================================================================================================
// The layer's data and list
// =================================================================================================
struct region {
    uint32_t ch;
    uint64_t beat;
    std::vector<uint8_t> data;
};
struct rec {
    uint32_t kind;
    std::vector<uint32_t> a;
    uint32_t op;
    std::string what;
};
struct out_rows {      // a record's dst rows (row r at h + r * stride lines, `lines` lines each)
    std::string what;
    uint32_t h, rows, stride, lines;
};
// One layer's weights as host data (ggml blocks / f32), shared by the device image and ggml.
struct weights_t {
    std::vector<uint8_t> wq, wk, wv, wo, wg, wu, wd;                       // Q8_0 blocks, row n = output n
    std::vector<float> attn_norm, ffn_norm, bq, bk, bv, q_norm, k_norm, post_attn_norm, post_ffw_norm;
    float out_scale = 1.0f;
};
struct layer_io {      // per layer: the handles the ggml comparison and the checks read back
    uint32_t x, y, att;
    uint32_t kB, vB;
};
struct layer_data {
    const lcase * c = nullptr;
    const model * m = nullptr;
    tsr_config cfg{};
    uint32_t C = 0, E = 0, LG = 0, G = 0;
    uint64_t plane_beats = 0;
    bool real = false;                         // GGUF weights
    std::vector<region> in;                    // the host's inputs (written before the launch)
    std::vector<rec> recs;                     // op order, JOINs excluded
    std::vector<out_rows> outs;
    std::vector<layer_io> lio;
    std::vector<uint8_t> blob, pre;
    uint32_t opl_h = 0, pre_pos = 0, pre_idx = 0, pre_msum = 0, mask_h = opl::HANDLE_NONE;
    // host copies for the ggml comparison
    weights_t w;
    std::vector<float> x;                      // [T][n_embd] layer 0 input
    std::vector<uint16_t> mask16;              // [T][n_kv]
    std::vector<std::vector<uint8_t>> kc0, vc0;   // per layer: the ggml q8_0 cache images before the list
    // expected (the references on the shadow DRAM)
    std::vector<std::vector<uint8_t>> want;    // per out_rows: the rows' bytes
    std::vector<std::vector<uint8_t>> want_planes;   // per layer, K then V, per channel
    tsr_fa_stats fa_st{};
};

uint32_t lines_of(uint32_t n) { return (n + 15u) / 16u; }

// Q8_0 blocks of an N x K tensor with values ~ N(0, sigma^2): d = amax/127 per block (a block of
// 32 normal values), q = round(v / d) (what a quantized model looks like).
std::vector<uint8_t> q8_tensor(rng_t * r, uint32_t N, uint32_t K, double sigma) {
    const uint32_t nb = K / 32u;
    std::vector<uint8_t> out(size_t(N) * nb * 34u);
    float v[32];
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < nb; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                v[i] = float(rng_normal(r) * sigma);
                amax = std::max(amax, std::fabs(v[i]));
            }
            const float d = amax / 127.0f;
            const uint16_t dh = tsr_f32_to_fp16(d);
            uint8_t * blk = out.data() + (size_t(n) * nb + b) * 34u;
            blk[0] = uint8_t(dh);
            blk[1] = uint8_t(dh >> 8);
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            for (int i = 0; i < 32; ++i) blk[2 + i] = uint8_t(int8_t(std::lrintf(std::clamp(v[i] * id, -127.0f, 127.0f))));
        }
    return out;
}
std::vector<float> f32_vec(rng_t * r, size_t n, double mean, double sigma) {
    std::vector<float> v(n);
    for (auto & x : v) x = float(mean + rng_normal(r) * sigma);
    return v;
}

// The case's weights: Qwen2.5-0.5B layer 0 from the GGUF file (if present and c.gguf), else random.
bool load_qwen_gguf(const model & m, weights_t & w, std::vector<float> & x, uint32_t T) {
    gguf_file g;
    const std::string path = std::string(TSR_M7L_MODELS) + "/qwen2.5-0.5b-instruct-q8_0.gguf";
    if (!g.open(path.c_str())) return false;
    if (g.num("qwen2.embedding_length", 0) != m.n_embd || g.num("qwen2.feed_forward_length", 0) != m.n_ff ||
        g.num("qwen2.attention.head_count", 0) != m.n_head || g.num("qwen2.attention.head_count_kv", 0) != m.n_kv_head ||
        float(g.num("qwen2.rope.freq_base", 0)) != m.rope_base ||
        float(g.num("qwen2.attention.layer_norm_rms_epsilon", 0)) != m.eps) {
        CHECK(0, "%s: hparams differ from the model table", path.c_str());
        return false;
    }
    auto q8 = [&](const char * n, uint32_t N, uint32_t K, std::vector<uint8_t> & out) {
        const gguf_tensor * t = g.tensor(n);
        if (!t || t->type != GGUF_Q8_0 || t->ne.size() != 2 || t->ne[0] != K || t->ne[1] != N) return false;
        out.resize(size_t(N) * (K / 32u) * 34u);
        return g.read(*t, out.data(), out.size());
    };
    auto f32 = [&](const char * n, uint32_t len, std::vector<float> & out) {
        const gguf_tensor * t = g.tensor(n);
        if (!t || t->type != GGUF_F32 || t->ne.size() != 1 || t->ne[0] != len) return false;
        out.resize(len);
        return g.read(*t, out.data(), size_t(len) * 4u);
    };
    const uint32_t E = m.n_kv_head * m.D, Q = m.n_head * m.D;
    bool ok = q8("blk.0.attn_q.weight", Q, m.n_embd, w.wq) && q8("blk.0.attn_k.weight", E, m.n_embd, w.wk) &&
              q8("blk.0.attn_v.weight", E, m.n_embd, w.wv) && q8("blk.0.attn_output.weight", m.n_embd, Q, w.wo) &&
              q8("blk.0.ffn_gate.weight", m.n_ff, m.n_embd, w.wg) && q8("blk.0.ffn_up.weight", m.n_ff, m.n_embd, w.wu) &&
              q8("blk.0.ffn_down.weight", m.n_embd, m.n_ff, w.wd) && f32("blk.0.attn_norm.weight", m.n_embd, w.attn_norm) &&
              f32("blk.0.ffn_norm.weight", m.n_embd, w.ffn_norm) && f32("blk.0.attn_q.bias", Q, w.bq) &&
              f32("blk.0.attn_k.bias", E, w.bk) && f32("blk.0.attn_v.bias", E, w.bv);
    // Inputs: the token embeddings (token_embd Q8_0, dequantized as ggml does: d * q) of fixed ids.
    const gguf_tensor * te = g.tensor("token_embd.weight");
    ok = ok && te && te->type == GGUF_Q8_0 && te->ne.size() == 2 && te->ne[0] == m.n_embd;
    if (!ok) {
        CHECK(0, "%s: a layer-0 tensor is missing or has another type / shape", path.c_str());
        return false;
    }
    static const uint32_t ids[8] = {9707, 11, 1879, 0, 525, 3555, 13, 151645};   // "Hello", ",", " world", ...
    const size_t rb = size_t(m.n_embd / 32u) * 34u;
    std::vector<uint8_t> row(rb);
    x.resize(size_t(T) * m.n_embd);
    for (uint32_t t = 0; t < T; ++t) {
        gguf_tensor tt = *te;
        tt.off += uint64_t(ids[t % 8u]) * rb;
        if (!g.read(tt, row.data(), rb)) return false;
        for (uint32_t b = 0; b < m.n_embd / 32u; ++b) {
            const uint8_t * blk = row.data() + size_t(b) * 34u;
            const float d = tsr_fp16_to_f32(uint16_t(blk[0] | blk[1] << 8));
            for (uint32_t i = 0; i < 32u; ++i) x[size_t(t) * m.n_embd + 32u * b + i] = d * float(int8_t(blk[2 + i]));
        }
    }
    return true;
}

void random_weights(const model & m, rng_t * r, weights_t & w, std::vector<float> & x, uint32_t T) {
    const uint32_t E = m.n_kv_head * m.D, Q = m.n_head * m.D;
    // sigma 1/sqrt(K): outputs of the order of the inputs
    w.wq = q8_tensor(r, Q, m.n_embd, 1.0 / std::sqrt(double(m.n_embd)));
    w.wk = q8_tensor(r, E, m.n_embd, 1.0 / std::sqrt(double(m.n_embd)));
    w.wv = q8_tensor(r, E, m.n_embd, 1.0 / std::sqrt(double(m.n_embd)));
    w.wo = q8_tensor(r, m.n_embd, Q, 1.0 / std::sqrt(double(Q)));
    w.wg = q8_tensor(r, m.n_ff, m.n_embd, 1.0 / std::sqrt(double(m.n_embd)));
    w.wu = q8_tensor(r, m.n_ff, m.n_embd, 1.0 / std::sqrt(double(m.n_embd)));
    w.wd = q8_tensor(r, m.n_embd, m.n_ff, 1.0 / std::sqrt(double(m.n_ff)));
    w.attn_norm = f32_vec(r, m.n_embd, 1.0, 0.1);
    w.ffn_norm = f32_vec(r, m.n_embd, 1.0, 0.1);
    if (m.bias) {
        w.bq = f32_vec(r, Q, 0.0, 0.5);
        w.bk = f32_vec(r, E, 0.0, 0.5);
        w.bv = f32_vec(r, E, 0.0, 0.5);
    }
    if (m.gemma) {
        w.q_norm = f32_vec(r, m.D, 1.0, 0.1);
        w.k_norm = f32_vec(r, m.D, 1.0, 0.1);
        w.post_attn_norm = f32_vec(r, m.n_embd, 1.0, 0.1);
        w.post_ffw_norm = f32_vec(r, m.n_embd, 1.0, 0.1);
        w.out_scale = 0.6171875f;
    }
    x = f32_vec(r, size_t(T) * m.n_embd, 0.0, m.gemma ? 4.0 : 0.05);
}

void build_mm(const lcase & c, layer_data & L, bool want_ref);

// Builds the case: data on a shadow simulator's DRAM (inputs captured as regions), the list
// (n_layers layers: the same weights, a KV plane pair per layer, layer l + 1 reads layer l's
// output), the preload area's image, then the expected outputs (the references in op order).
// want_ref = false: no reference (perf lists).
void build(const lcase & c, layer_data & L, uint32_t n_layers = 1, bool mark = false, bool want_ref = true) {
    if (c.mm) {
        build_mm(c, L, want_ref);
        return;
    }
    const model & m = *c.m;
    L.c = &c;
    L.m = &m;
    L.cfg = TSR_CONFIG_U200;
    L.cfg.t_max = 8;
    L.cfg.ch_bytes = 512ull << 20;
    L.cfg.caps = TSR_SIM_CAPS_IMPLEMENTED;
    if (c.tiny) {   // the device of the RTL bench (tb/core/core_tb.py M7_CFG): CAPS VU | ATT as the RTL builds them
        L.cfg.num_ch = 2;
        L.cfg.t_max = 2;
        L.cfg.k_max = 4096;
        L.cfg.spm_bytes = 1024u * 1024u;   // the graph kernel's OPL area alone is 256 KiB + 16 argument lines
        L.cfg.addr_beat_w = 24;
        L.cfg.caps = TSR_CAPS_VU | TSR_CAPS_ATT;
    }
    L.C = L.cfg.num_ch;
    L.E = m.n_kv_head * m.D;
    L.G = m.n_head / m.n_kv_head;
    L.LG = tsr_kv_lg_min(c.n_kv, L.C);
    const tsr_kv_geom geo = {L.E, L.C, L.LG};
    L.plane_beats = tsr_kv_plane_beats(&geo);
    const uint32_t T = c.T, D = m.D, Qd = m.n_head * D, C = L.C;
    uint64_t seed = 0x6d376c61u;
    for (const char * p = c.name; *p; ++p) seed = seed * 131u + uint8_t(*p);
    rng_t r = {seed};
    L.real = c.gguf && load_qwen_gguf(m, L.w, L.x, T);
    if (!L.real) random_weights(m, &r, L.w, L.x, T);

    tsr_sim * sh = tsr_sim_create(&L.cfg, nullptr);
    CHECK(sh != nullptr, "shadow simulator");
    if (sh == nullptr) return;
    uint64_t io[4] = {0x10000u, 0x10000u, 0x10000u, 0x10000u};   // 4 MiB: activations, small inputs
    uint64_t sym = 0x100000u;                                    // 64 MiB: weights and planes (every channel)
    uint32_t rr_ch = 0;
    auto keep = [&](uint32_t ch, uint64_t beat, uint64_t beats) {
        region rg{ch, beat, std::vector<uint8_t>(size_t(beats * LINE))};
        memcpy(rg.data.data(), tsr_sim_dram(sh, ch) + beat * LINE, rg.data.size());
        L.in.push_back(std::move(rg));
    };
    // n_rows rows of `stride` lines on the next channel (round robin); vals: f32 rows to write (input).
    auto rows = [&](uint32_t n_rows, uint32_t stride, const float * vals = nullptr, uint32_t n = 0) -> uint32_t {
        const uint32_t ch = rr_ch++ % C;
        const uint64_t beat = io[ch];
        io[ch] += (uint64_t(n_rows) * stride + 63u) & ~63ull;
        if (vals) {
            for (uint32_t t = 0; t < n_rows; ++t)
                memcpy(tsr_sim_dram(sh, ch) + (beat + uint64_t(t) * stride) * LINE, vals + size_t(t) * n, size_t(n) * 4u);
            keep(ch, beat, uint64_t(n_rows) * stride);
        }
        return H(ch, beat);
    };
    auto bytes_in = [&](const void * p, size_t n) -> uint32_t {
        const uint32_t ch = rr_ch++ % C;
        const uint64_t beat = io[ch];
        const uint64_t beats = (n + LINE - 1u) / LINE;
        io[ch] += (beats + 63u) & ~63ull;
        memcpy(tsr_sim_dram(sh, ch) + beat * LINE, p, n);
        keep(ch, beat, beats);
        return H(ch, beat);
    };
    auto weights = [&](const std::vector<uint8_t> & blocks, uint32_t N, uint32_t K) -> uint32_t {
        const uint64_t w_off = sym;
        const size_t rb = size_t(K / 32u) * 34u;
        for (uint32_t n = 0; n < N; ++n) {
            uint32_t ch;
            uint64_t off;
            tsr_row_location(TSR_FMT_Q8_0, N, K, C, n, &ch, &off);
            tsr_repack_row(TSR_FMT_Q8_0, blocks.data() + size_t(n) * rb, K, tsr_sim_dram(sh, ch) + (w_off + off) * LINE);
        }
        const uint64_t beats = tsr_tensor_ch_beats(TSR_FMT_Q8_0, N, K, C);
        for (uint32_t ch = 0; ch < C; ++ch) keep(ch, w_off, beats);
        sym += (beats + 63u) & ~63ull;
        return uint32_t(w_off);
    };
    auto vec = [&](const std::vector<float> & v) -> uint32_t { return rows(1, lines_of(uint32_t(v.size())), v.data(), uint32_t(v.size())); };

    // --- weights (shared by every layer) ---
    const uint32_t wq = weights(L.w.wq, Qd, m.n_embd), wk = weights(L.w.wk, L.E, m.n_embd), wv = weights(L.w.wv, L.E, m.n_embd);
    const uint32_t wo = weights(L.w.wo, m.n_embd, Qd), wg = weights(L.w.wg, m.n_ff, m.n_embd), wu = weights(L.w.wu, m.n_ff, m.n_embd);
    const uint32_t wd = weights(L.w.wd, m.n_embd, m.n_ff);
    const uint32_t h_attn_norm = vec(L.w.attn_norm), h_ffn_norm = vec(L.w.ffn_norm);
    uint32_t h_bq = opl::HANDLE_NONE, h_bk = opl::HANDLE_NONE, h_bv = opl::HANDLE_NONE;
    if (m.bias) { h_bq = vec(L.w.bq); h_bk = vec(L.w.bk); h_bv = vec(L.w.bv); }
    uint32_t h_qn = 0, h_kn = 0, h_pan = 0, h_pfn = 0, h_osc = 0;
    if (m.gemma) {
        h_qn = vec(L.w.q_norm);
        h_kn = vec(L.w.k_norm);
        h_pan = vec(L.w.post_attn_norm);
        h_pfn = vec(L.w.post_ffw_norm);
        const std::vector<float> s1(1, L.w.out_scale);
        h_osc = vec(s1);
    }

    // --- positions, KV indices, mask, MSUM (one set for every layer) ---
    std::vector<int32_t> pos(T);
    std::vector<uint8_t> idx(size_t(T) * 8u, 0);
    for (uint32_t t = 0; t < T; ++t) {
        pos[t] = int32_t(c.pos0 + t);
        const uint32_t cell = c.pos0 + t;
        memcpy(idx.data() + 8u * t, &cell, 4);
    }
    L.mask16.assign(size_t(T) * c.n_kv, 0xFC00u);
    for (uint32_t t = 0; t < T; ++t) {
        const int64_t pq = int64_t(c.pos0) + t;
        for (uint32_t j = 0; j < c.n_kv; ++j) {
            const int64_t p = j;   // cell j holds position j (or nothing when j >= pos0 + T)
            bool vis = p < int64_t(c.pos0) + int64_t(T) && p <= pq;
            if (m.n_swa != 0u && pq - p >= int64_t(m.n_swa)) vis = false;
            if (vis) L.mask16[size_t(t) * c.n_kv + j] = 0x0000u;
        }
    }
    const uint32_t mask_h_alloc = rows(T, lines_of(c.n_kv));   // the f32 copy's place (written if needed)
    std::vector<uint32_t> msum(tsr_msum_words(T));
    const uint32_t mflags = tsr_msum_build_f16(L.mask16.data(), c.n_kv, c.n_kv, T, mask_h_alloc, msum.data());
    bool need_f32 = (mflags & TSR_MSUM_GENERAL) != 0u;
    const tsr_att_cfg acfg = {L.cfg.num_ch, L.cfg.t_max, tsr_qa_lines(TSR_QA_A256, L.cfg.k_max) * 16u, L.cfg.k_max};
    if (!need_f32) {   // a tile the planner sends down the GENERAL path needs the copy too
        tsr_msum ms;
        tsr_msum_view(msum.data(), msum.size(), &ms);
        uint32_t GS = 0, Tq = 0;
        tsr_fa_pass_rule(&acfg, L.G, &GS, &Tq);
        for (uint32_t q0 = 0; q0 < T && !need_f32; q0 += Tq) {
            const uint32_t tq = std::min(Tq, T - q0);
            std::vector<tsr_fa_chunk> chs(4096);
            const long n = tsr_fa_plan(&acfg, L.E, D, &ms, q0, tq, nullptr, 0, chs.data(), chs.size());
            for (long i = 0; i < n && size_t(i) < chs.size(); ++i) need_f32 = need_f32 || chs[size_t(i)].general;
        }
    }
    if (need_f32) {
        std::vector<float> mf(size_t(T) * 16u * lines_of(c.n_kv));
        tsr_msum_mask_f32(L.mask16.data(), c.n_kv, c.n_kv, T, m.attn_scale, mf.data());
        memcpy(tsr_sim_dram(sh, h_ch(mask_h_alloc)) + h_beat(mask_h_alloc) * LINE, mf.data(), mf.size() * 4u);
        keep(h_ch(mask_h_alloc), h_beat(mask_h_alloc), uint64_t(T) * lines_of(c.n_kv));
        L.mask_h = mask_h_alloc;
    }
    const uint32_t pos_h = bytes_in(pos.data(), pos.size() * 4u);
    const uint32_t idx_h = bytes_in(idx.data(), idx.size());
    const uint32_t msum_h = bytes_in(msum.data(), msum.size() * 4u);

    opl::writer w(opl::F_STOP_ON_ERROR | (mark ? opl::F_MARK : 0u), 0x4d374c00u + n_layers);
    L.pre_pos = w.pre_positions(pos_h, T);
    L.pre_idx = w.pre_kv_idx(idx_h, T);
    L.pre_msum = w.pre_msum(msum_h, T);
    auto addr = [&](uint32_t kind, const uint32_t * a, uint32_t n, std::initializer_list<uint32_t> deps, const char * what,
                    uint32_t dst, uint32_t dst_rows, uint32_t dst_stride, uint32_t dst_lines) -> uint32_t {
        std::vector<uint32_t> d;
        for (uint32_t v : deps)
            if (v != opl::DEP_NONE) d.push_back(v);
        const uint32_t op = w.add(kind, a, n, d.data(), uint32_t(d.size()));
        L.recs.push_back(rec{kind, std::vector<uint32_t>(a, a + n), op, what});
        if (dst != opl::HANDLE_NONE) L.outs.push_back(out_rows{what, dst, dst_rows, dst_stride, dst_lines});
        return op;
    };
    auto mmul = [&](const char * what, uint32_t x, uint32_t xs, uint32_t y, uint32_t ys, uint32_t woff, uint32_t N, uint32_t K,
                    uint32_t add, uint32_t add_s, std::initializer_list<uint32_t> deps) {
        const uint32_t a[12] = {x, xs, y, ys, woff, TSR_FMT_Q8_0, N, K, T, add, add_s, 0u};
        return addr(opl::K_MUL_MAT, a, 12, deps, what, y, T, ys, lines_of(N));
    };
    auto rms = [&](const char * what, uint32_t dst, uint32_t src, uint32_t stride, uint32_t n, uint32_t nrows, uint32_t wgt,
                   std::initializer_list<uint32_t> deps) {
        const opl::rms_norm_args a{dst, stride, src, stride, n, nrows, f2b(m.eps), f2b(1.0f / float(n)), wgt, 1u};
        uint32_t wds[10];
        opl::put_args(a, wds);
        return addr(opl::K_RMS_NORM, wds, 10, deps, what, dst, nrows, stride, lines_of(n));
    };
    auto rope = [&](const char * what, uint32_t dst, uint32_t src, uint32_t stride, uint32_t n_heads, std::initializer_list<uint32_t> deps) {
        const opl::rope_args a{dst, stride, src, stride, L.pre_pos, opl::HANDLE_NONE, f2b(powf(m.rope_base, -2.0f / float(D))),
                               f2b(1.0f), f2b(1.0f), D, D, n_heads, T, opl::rope::MODE_NEOX};
        uint32_t wds[14];
        opl::put_args(a, wds);
        return addr(opl::K_ROPE, wds, 14, deps, what, dst, T, stride, n_heads * D / 16u);
    };
    auto ew = [&](uint32_t kind, const char * what, uint32_t dst, uint32_t a_h, uint32_t b_h, uint32_t b_stride, uint32_t n,
                  uint32_t p, uint32_t flags, std::initializer_list<uint32_t> deps) {
        const uint32_t s = lines_of(n);
        const opl::ew_args a{dst, s, a_h, s, b_h, b_stride, n, T, p, flags};
        uint32_t wds[10];
        opl::put_args(a, wds);
        return addr(kind, wds, 10, deps, what, dst, T, s, s);
    };

    const uint32_t xs = lines_of(m.n_embd), qs = lines_of(Qd), ks = lines_of(L.E), fs = lines_of(m.n_ff), hs = D / 16u;
    const uint32_t geom = opl::geom_word(L.E / 64u, L.LG);
    uint32_t X = rows(T, xs, L.x.data(), m.n_embd);
    uint32_t prev = opl::DEP_NONE;
    const size_t cb = size_t(c.n_kv) * (L.E / 32u) * 34u;
    for (uint32_t l = 0; l < n_layers; ++l) {
        layer_io io_l{};
        io_l.x = X;
        // KV planes of this layer with random history in the cells before the new tokens' (QACT A32 +
        // KVST: the device image); the ggml cache images read back from them.
        io_l.kB = uint32_t(sym);
        io_l.vB = uint32_t(sym + ((L.plane_beats + 63u) & ~63ull));
        sym = io_l.vB + ((L.plane_beats + 63u) & ~63ull);
        {
            std::vector<uint8_t *> kp(C), vp(C);
            for (uint32_t i = 0; i < C; ++i) {
                kp[i] = tsr_sim_dram(sh, i) + uint64_t(io_l.kB) * LINE;
                vp[i] = tsr_sim_dram(sh, i) + uint64_t(io_l.vB) * LINE;
            }
            std::vector<uint8_t> qa(size_t(tsr_qa_lines(TSR_QA_A32, L.E)) * LINE);
            std::vector<float> h(L.E);
            const double ksig = m.gemma ? 1.0 : 2.0;
            for (uint32_t j = 0; j < c.pos0; ++j)
                for (uint32_t kv = 0; kv < 2u; ++kv) {
                    for (auto & v : h) v = float(rng_normal(&r) * ksig);
                    tsr_qact(TSR_QA_A32, h.data(), L.E, qa.data());
                    tsr_ref_kvst(kv, &geo, kv ? vp.data() : kp.data(), j, 1u, qa.data(), 0u, nullptr);
                }
            for (uint32_t ch = 0; ch < C; ++ch) {
                keep(ch, io_l.kB, L.plane_beats);
                keep(ch, io_l.vB, L.plane_beats);
            }
            std::vector<uint8_t> kc(cb), vc(cb);
            tsr_kv_get_bytes(TSR_KV_K, &geo, kp.data(), kc.data(), 0, cb);
            tsr_kv_get_bytes(TSR_KV_V, &geo, vp.data(), vc.data(), 0, cb);
            L.kc0.push_back(std::move(kc));
            L.vc0.push_back(std::move(vc));
        }
        const uint32_t Xn = rows(T, xs), Q = rows(T, qs), K = rows(T, ks), V = rows(T, ks), Qr = rows(T, qs), Kr = rows(T, ks);
        const uint32_t Att = rows(T, qs), G = rows(T, fs), U = rows(T, fs), GL = rows(T, fs), Y = rows(T, xs);
        const uint32_t r1 = rms("attn_norm", Xn, X, xs, m.n_embd, T, h_attn_norm, {prev});
        const uint32_t q = mmul("Qcur", Xn, xs, Q, qs, wq, Qd, m.n_embd, h_bq, 0, {r1});
        const uint32_t k = mmul("Kcur", Xn, xs, K, ks, wk, L.E, m.n_embd, h_bk, 0, {r1});
        const uint32_t v = mmul("Vcur", Xn, xs, V, ks, wv, L.E, m.n_embd, h_bv, 0, {r1});
        uint32_t qsrc = Q, ksrc = K, vsrc = V, dq = q, dk = k, dv = v;
        if (m.gemma) {   // per-head norms: rows = heads x T, n = D (VU §8.1)
            const uint32_t Qn = rows(T, qs), Kn = rows(T, ks), Vn = rows(T, ks);
            dq = rms("Qcur_normed", Qn, Q, hs, D, m.n_head * T, h_qn, {q});
            dk = rms("Kcur_normed", Kn, K, hs, D, m.n_kv_head * T, h_kn, {k});
            dv = rms("Vcur_normed", Vn, V, hs, D, m.n_kv_head * T, opl::HANDLE_NONE, {v});
            qsrc = Qn;
            ksrc = Kn;
            vsrc = Vn;
        }
        const uint32_t rq = rope("Qcur_pos", Qr, qsrc, qs, m.n_head, {dq});
        const uint32_t rk = rope("Kcur_pos", Kr, ksrc, ks, m.n_kv_head, {dk});
        const opl::set_rows_kv_args sk{H(0, io_l.kB), 0u, Kr, ks, L.pre_idx, T, L.E, 0u, io_l.kB, geom, 0u, 0u};
        const opl::set_rows_kv_args sv{H(0, io_l.vB), 0u, vsrc, ks, L.pre_idx, T, L.E, 1u, io_l.vB, geom, 0u, 0u};
        uint32_t wds[22];
        opl::put_args(sk, wds);
        const uint32_t ok_ = addr(opl::K_SET_ROWS_KV, wds, 12, {rk}, "set_rows K", opl::HANDLE_NONE, 0, 0, 0);
        opl::put_args(sv, wds);
        const uint32_t ov_ = addr(opl::K_SET_ROWS_KV, wds, 12, {dv}, "set_rows V", opl::HANDLE_NONE, 0, 0, 0);
        const opl::flash_attn_args fa{Att, qs, Qr, hs, qs, H(0, io_l.kB), H(0, io_l.vB), c.n_kv, D, m.n_head, m.n_kv_head, T,
                                      L.pre_msum, L.mask_h, f2b(m.attn_scale), 0u, io_l.kB, io_l.vB, geom, D, L.G,
                                      tsr::plan::attn::fa_flags(L.pre_idx, T)};
        opl::put_args(fa, wds);
        const uint32_t f = addr(opl::K_FLASH_ATTN, wds, 22, {rq, ok_, ov_}, "attn (FLASH_ATTN)", Att, T, qs, qs);
        io_l.att = Att;
        if (!m.gemma) {
            const uint32_t Hh = rows(T, xs), Hn = rows(T, xs);
            const uint32_t o = mmul("attn_out + inpSA", Att, qs, Hh, xs, wo, m.n_embd, Qd, X, xs, {f});
            const uint32_t r2 = rms("ffn_norm", Hn, Hh, xs, m.n_embd, T, h_ffn_norm, {o});
            const uint32_t g = mmul("ffn_gate", Hn, xs, G, fs, wg, m.n_ff, m.n_embd, opl::HANDLE_NONE, 0, {r2});
            const uint32_t u = mmul("ffn_up", Hn, xs, U, fs, wu, m.n_ff, m.n_embd, opl::HANDLE_NONE, 0, {r2});
            const opl::glu_args ga{GL, fs, G, fs, U, fs, m.n_ff, T, m.glu};
            opl::put_args(ga, wds);
            const uint32_t gl = addr(opl::K_GLU, wds, 9, {g, u}, "ffn_swiglu", GL, T, fs, fs);
            prev = mmul("ffn_out + ffn_inp", GL, fs, Y, xs, wd, m.n_embd, m.n_ff, Hh, xs, {gl});
        } else {
            const uint32_t O = rows(T, xs), On = rows(T, xs), A = rows(T, xs), An = rows(T, xs), F = rows(T, xs), Fn = rows(T, xs),
                           R = rows(T, xs);
            const uint32_t o = mmul("attn_out", Att, qs, O, xs, wo, m.n_embd, Qd, opl::HANDLE_NONE, 0, {f});
            const uint32_t on = rms("attn_post_norm", On, O, xs, m.n_embd, T, h_pan, {o});
            const uint32_t a = ew(opl::K_ADD, "attn_out + inpL", A, On, X, xs, m.n_embd, T, 0u, {on});
            const uint32_t an = rms("ffn_norm", An, A, xs, m.n_embd, T, h_ffn_norm, {a});
            const uint32_t g = mmul("ffn_gate", An, xs, G, fs, wg, m.n_ff, m.n_embd, opl::HANDLE_NONE, 0, {an});
            const uint32_t u = mmul("ffn_up", An, xs, U, fs, wu, m.n_ff, m.n_embd, opl::HANDLE_NONE, 0, {an});
            const opl::glu_args ga{GL, fs, G, fs, U, fs, m.n_ff, T, m.glu};
            opl::put_args(ga, wds);
            const uint32_t gl = addr(opl::K_GLU, wds, 9, {g, u}, "ffn_geglu", GL, T, fs, fs);
            const uint32_t dd = mmul("ffn_out", GL, fs, F, xs, wd, m.n_embd, m.n_ff, opl::HANDLE_NONE, 0, {gl});
            const uint32_t fn = rms("ffn_post_norm", Fn, F, xs, m.n_embd, T, h_pfn, {dd});
            const uint32_t rs = ew(opl::K_ADD, "ffn_post_norm + attn_out", R, Fn, A, xs, m.n_embd, T, 0u, {fn, a});
            prev = ew(opl::K_MUL, "out_scaled", Y, R, h_osc, 0u, m.n_embd, 1u, opl::ew::FLAG_B_SCALAR, {rs});
        }
        io_l.y = Y;
        L.lio.push_back(io_l);
        X = Y;
    }
    L.opl_h = H(0, 0x8000u);   // ch0 2 MiB
    L.blob = w.finish();
    CHECK(w.ok() && !L.blob.empty(), "%s: OPL writer: %s", c.name, w.error());
    CHECK(L.blob.size() <= 0x8000u * LINE, "%s: the list overlaps the activations", c.name);
    for (const rec & x : L.recs) {   // the host reader's structural rules (SW §7.2)
        const uint32_t e = opl::check_args(x.kind, x.a.data(), uint32_t(x.a.size()), w.pre_lines());
        CHECK(e == opl::RET_OK, "%s: %s: check_args %u", c.name, x.what.c_str(), e);
    }
    L.pre.assign(size_t(w.pre_lines()) * LINE, 0);
    memcpy(L.pre.data() + L.pre_pos, pos.data(), pos.size() * 4u);
    memcpy(L.pre.data() + L.pre_idx, idx.data(), idx.size());
    memcpy(L.pre.data() + L.pre_msum, msum.data(), msum.size() * 4u);

    if (want_ref) {
        // TSR_M7L_MUTATE=1 (a check of the test itself): the reference sees attn_norm's weight 0 with
        // a mantissa bit flipped, the device the original; the first norm's output must then differ.
        if (getenv("TSR_M7L_MUTATE")) {
            uint32_t * p = reinterpret_cast<uint32_t *>(tsr_sim_dram(sh, h_ch(h_attn_norm)) + h_beat(h_attn_norm) * LINE);
            *p ^= 0x00400000u;
        }
        // The references in op order on the shadow DRAM.
        std::vector<uint8_t *> chp(C);
        for (uint32_t i = 0; i < C; ++i) chp[i] = tsr_sim_dram(sh, i);
        const tsr::ref::dram_view view{chp.data(), C, L.cfg.ch_bytes};
        for (const rec & x : L.recs) {
            int rc = 0;
            const uint32_t * a = x.a.data();
            switch (x.kind) {
                case opl::K_MUL_MAT: rc = tsr::ref::mulmat_ref(view, a, 8); break;
                case opl::K_RMS_NORM: rc = tsr::ref::rmsnorm_ref(view, a); break;
                case opl::K_ROPE: rc = tsr::ref::rope_ref(view, a, L.pre.data(), uint32_t(L.pre.size())); break;
                case opl::K_SET_ROWS_KV: {
                    std::vector<uint8_t *> pl(C);
                    for (uint32_t i = 0; i < C; ++i) pl[i] = tsr_sim_dram(sh, i) + uint64_t(a[opl::setrows::PLANE]) * LINE;
                    rc = tsr::ref::set_rows_kv_ref(view, a, L.pre.data(), L.pre.size(), pl.data(), C);
                    break;
                }
                case opl::K_FLASH_ATTN: {
                    std::vector<const uint8_t *> kp(C), vp(C);
                    for (uint32_t i = 0; i < C; ++i) {
                        kp[i] = tsr_sim_dram(sh, i) + uint64_t(a[opl::fa::K_PLANE]) * LINE;
                        vp[i] = tsr_sim_dram(sh, i) + uint64_t(a[opl::fa::V_PLANE]) * LINE;
                    }
                    rc = tsr::ref::flash_attn_ref(view, a, L.pre.data(), L.pre.size(), kp.data(), vp.data(), acfg, &L.fa_st);
                    break;
                }
                default: rc = tsr::ref::elem_ref(view, x.kind, a, uint32_t(x.a.size()), L.pre.data(), L.pre.size()); break;
            }
            CHECK(rc == 0, "%s: reference of %s (op %u): %d", c.name, x.what.c_str(), x.op, rc);
        }
        for (const out_rows & o : L.outs) {
            std::vector<uint8_t> b(size_t(o.rows) * o.lines * LINE);
            for (uint32_t i = 0; i < o.rows; ++i)
                memcpy(b.data() + size_t(i) * o.lines * LINE, tsr_sim_dram(sh, h_ch(o.h)) + (h_beat(o.h) + uint64_t(i) * o.stride) * LINE,
                       size_t(o.lines) * LINE);
            L.want.push_back(std::move(b));
        }
        for (const layer_io & x : L.lio)
            for (uint32_t kv = 0; kv < 2u; ++kv)
                for (uint32_t ch = 0; ch < C; ++ch) {
                    const uint8_t * p = tsr_sim_dram(sh, ch) + uint64_t(kv ? x.vB : x.kB) * LINE;
                    L.want_planes.emplace_back(p, p + L.plane_beats * LINE);
                }
    }
    tsr_sim_destroy(sh);
}

// The M6 list on the device of the reduced card image (tsr_u200_k4: 4 channels, T_MAX 8, CAPS 0, 1 MiB of SPM
// for the graph kernel's list area): the seven MUL_MAT records of a decoder layer (Q / K / V with their bias addends,
// the attention output projection with the residual add, gate, up and down with the residual add) on random Q8_0
// weights, with the same shapes and dependencies as the layer, and nothing else. No KV planes, no preload area, no
// norm, RoPE, attention or GLU: the host keeps those on the CPU when CAPS is 0. The references are mulmat_ref in op
// order on the shadow DRAM, as in build().
void build_mm(const lcase & c, layer_data & L, bool want_ref) {
    const model & m = *c.m;
    L.c = &c;
    L.m = &m;
    L.cfg = TSR_CONFIG_U200;
    L.cfg.num_ch = 4;
    L.cfg.t_max = 8;
    L.cfg.k_max = 4096;
    L.cfg.spm_bytes = 1024u * 1024u;
    L.cfg.ch_bytes = 512ull << 20;
    L.cfg.addr_beat_w = 24;
    L.cfg.caps = 0;
    L.C = L.cfg.num_ch;
    L.E = m.n_kv_head * m.D;
    L.G = m.n_head / m.n_kv_head;
    const uint32_t T = c.T, D = m.D, Qd = m.n_head * D, C = L.C;
    uint64_t seed = 0x6d366d6du;
    for (const char * p = c.name; *p; ++p) seed = seed * 131u + uint8_t(*p);
    rng_t r = {seed};
    random_weights(m, &r, L.w, L.x, T);
    tsr_sim * sh = tsr_sim_create(&L.cfg, nullptr);
    CHECK(sh != nullptr, "shadow simulator");
    if (sh == nullptr) return;
    uint64_t io[4] = {0x10000u, 0x10000u, 0x10000u, 0x10000u};
    uint64_t sym = 0x100000u;
    uint32_t rr_ch = 0;
    auto keep = [&](uint32_t ch, uint64_t beat, uint64_t beats) {
        region rg{ch, beat, std::vector<uint8_t>(size_t(beats * LINE))};
        memcpy(rg.data.data(), tsr_sim_dram(sh, ch) + beat * LINE, rg.data.size());
        L.in.push_back(std::move(rg));
    };
    auto rows = [&](uint32_t n_rows, uint32_t stride, const float * vals = nullptr, uint32_t n = 0) -> uint32_t {
        const uint32_t ch = rr_ch++ % C;
        const uint64_t beat = io[ch];
        io[ch] += (uint64_t(n_rows) * stride + 63u) & ~63ull;
        if (vals) {
            for (uint32_t t = 0; t < n_rows; ++t)
                memcpy(tsr_sim_dram(sh, ch) + (beat + uint64_t(t) * stride) * LINE, vals + size_t(t) * n, size_t(n) * 4u);
            keep(ch, beat, uint64_t(n_rows) * stride);
        }
        return H(ch, beat);
    };
    auto weights = [&](const std::vector<uint8_t> & blocks, uint32_t N, uint32_t K) -> uint32_t {
        const uint64_t w_off = sym;
        const size_t rb = size_t(K / 32u) * 34u;
        for (uint32_t n = 0; n < N; ++n) {
            uint32_t ch;
            uint64_t off;
            tsr_row_location(TSR_FMT_Q8_0, N, K, C, n, &ch, &off);
            tsr_repack_row(TSR_FMT_Q8_0, blocks.data() + size_t(n) * rb, K, tsr_sim_dram(sh, ch) + (w_off + off) * LINE);
        }
        const uint64_t beats = tsr_tensor_ch_beats(TSR_FMT_Q8_0, N, K, C);
        for (uint32_t ch = 0; ch < C; ++ch) keep(ch, w_off, beats);
        sym += (beats + 63u) & ~63ull;
        return uint32_t(w_off);
    };
    auto vec = [&](const std::vector<float> & v) -> uint32_t { return rows(1, lines_of(uint32_t(v.size())), v.data(), uint32_t(v.size())); };
    const uint32_t wq = weights(L.w.wq, Qd, m.n_embd), wk = weights(L.w.wk, L.E, m.n_embd), wv = weights(L.w.wv, L.E, m.n_embd);
    const uint32_t wo = weights(L.w.wo, m.n_embd, Qd), wg = weights(L.w.wg, m.n_ff, m.n_embd), wu = weights(L.w.wu, m.n_ff, m.n_embd);
    const uint32_t wd = weights(L.w.wd, m.n_embd, m.n_ff);
    const uint32_t h_bq = vec(L.w.bq), h_bk = vec(L.w.bk), h_bv = vec(L.w.bv);

    opl::writer w(opl::F_STOP_ON_ERROR, 0x4d364d00u);
    auto mmul = [&](const char * what, uint32_t x, uint32_t xs, uint32_t y, uint32_t ys, uint32_t woff, uint32_t N, uint32_t K,
                    uint32_t add, uint32_t add_s, std::initializer_list<uint32_t> deps) {
        const uint32_t a[12] = {x, xs, y, ys, woff, TSR_FMT_Q8_0, N, K, T, add, add_s, 0u};
        std::vector<uint32_t> d;
        for (uint32_t v : deps)
            if (v != opl::DEP_NONE) d.push_back(v);
        const uint32_t op = w.add(opl::K_MUL_MAT, a, 12, d.data(), uint32_t(d.size()));
        L.recs.push_back(rec{opl::K_MUL_MAT, std::vector<uint32_t>(a, a + 12), op, what});
        L.outs.push_back(out_rows{what, y, T, ys, lines_of(N)});
        return op;
    };
    const uint32_t xs = lines_of(m.n_embd), qs = lines_of(Qd), ks = lines_of(L.E), fs = lines_of(m.n_ff);
    const uint32_t X = rows(T, xs, L.x.data(), m.n_embd);
    const uint32_t Q = rows(T, qs), K = rows(T, ks), V = rows(T, ks), Hh = rows(T, xs), G = rows(T, fs), U = rows(T, fs),
                   Y = rows(T, xs);
    const uint32_t q = mmul("Qcur", X, xs, Q, qs, wq, Qd, m.n_embd, h_bq, 0, {});
    mmul("Kcur", X, xs, K, ks, wk, L.E, m.n_embd, h_bk, 0, {});
    mmul("Vcur", X, xs, V, ks, wv, L.E, m.n_embd, h_bv, 0, {});
    const uint32_t o = mmul("attn_out + inpSA", Q, qs, Hh, xs, wo, m.n_embd, Qd, X, xs, {q});
    const uint32_t g = mmul("ffn_gate", Hh, xs, G, fs, wg, m.n_ff, m.n_embd, opl::HANDLE_NONE, 0, {o});
    const uint32_t u = mmul("ffn_up", Hh, xs, U, fs, wu, m.n_ff, m.n_embd, opl::HANDLE_NONE, 0, {o});
    mmul("ffn_out + ffn_inp", G, fs, Y, xs, wd, m.n_embd, m.n_ff, Hh, xs, {g, u});
    L.opl_h = H(0, 0x8000u);
    L.blob = w.finish();
    CHECK(w.ok() && !L.blob.empty(), "%s: OPL writer: %s", c.name, w.error());
    CHECK(L.blob.size() <= 0x8000u * LINE, "%s: the list overlaps the activations", c.name);
    for (const rec & x : L.recs) {
        const uint32_t e = opl::check_args(x.kind, x.a.data(), uint32_t(x.a.size()), w.pre_lines());
        CHECK(e == opl::RET_OK, "%s: %s: check_args %u", c.name, x.what.c_str(), e);
    }
    L.pre.assign(size_t(w.pre_lines()) * LINE, 0);
    if (want_ref) {
        std::vector<uint8_t *> chp(C);
        for (uint32_t i = 0; i < C; ++i) chp[i] = tsr_sim_dram(sh, i);
        const tsr::ref::dram_view view{chp.data(), C, L.cfg.ch_bytes};
        for (const rec & x : L.recs) {
            const int rc = tsr::ref::mulmat_ref(view, x.a.data(), 8);
            CHECK(rc == 0, "%s: reference of %s (op %u): %d", c.name, x.what.c_str(), x.op, rc);
        }
        for (const out_rows & o_ : L.outs) {
            std::vector<uint8_t> b(size_t(o_.rows) * o_.lines * LINE);
            for (uint32_t i = 0; i < o_.rows; ++i)
                memcpy(b.data() + size_t(i) * o_.lines * LINE,
                       tsr_sim_dram(sh, h_ch(o_.h)) + (h_beat(o_.h) + uint64_t(i) * o_.stride) * LINE, size_t(o_.lines) * LINE);
            L.want.push_back(std::move(b));
        }
    }
    tsr_sim_destroy(sh);
}

void fill(tsr_sim * sim, const layer_data & L, const std::vector<uint8_t> * blob = nullptr) {
    for (const region & rg : L.in) tsr_sim_dram_write(sim, rg.ch, rg.beat * LINE, rg.data.data(), rg.data.size());
    const std::vector<uint8_t> & b = blob ? *blob : L.blob;
    tsr_sim_dram_write(sim, h_ch(L.opl_h), h_beat(L.opl_h) * LINE, b.data(), b.size());
}

// After a run: every record's output rows and the planes bit for bit; every written DRAM beat
// inside an output or a plane.
bool check_outputs(tsr_sim * sim, const layer_data & L, const char * tag, const obs * o) {
    bool ok = true;
    for (size_t i = 0; i < L.outs.size() && ok; ++i) {
        const out_rows & x = L.outs[i];
        for (uint32_t row = 0; row < x.rows && ok; ++row) {
            const uint8_t * got = tsr_sim_dram(sim, h_ch(x.h)) + (h_beat(x.h) + uint64_t(row) * x.stride) * LINE;
            const uint8_t * want = L.want[i].data() + size_t(row) * x.lines * LINE;
            if (memcmp(got, want, size_t(x.lines) * LINE) != 0) {
                size_t b = 0;
                while (got[b] == want[b]) ++b;
                uint32_t g = 0, wv = 0;
                memcpy(&g, got + (b & ~size_t(3)), 4);
                memcpy(&wv, want + (b & ~size_t(3)), 4);
                CHECK(0, "%s [%s]: %s row %u differs at word %zu: 0x%08x (%g), expected 0x%08x (%g)", L.c->name, tag, x.what.c_str(), row,
                      b / 4u, g, double(b2f(g)), wv, double(b2f(wv)));
                ok = false;
            }
        }
    }
    size_t pi = 0;
    for (const layer_io & x : L.lio)
        for (uint32_t kv = 0; kv < 2u; ++kv)
            for (uint32_t ch = 0; ch < L.C; ++ch, ++pi) {
                const uint8_t * got = tsr_sim_dram(sim, ch) + uint64_t(kv ? x.vB : x.kB) * LINE;
                if (memcmp(got, L.want_planes[pi].data(), L.want_planes[pi].size()) != 0) {
                    CHECK(0, "%s [%s]: %c plane ch%u differs", L.c->name, tag, kv ? 'V' : 'K', ch);
                    ok = false;
                }
            }
    if (o != nullptr) {
        for (uint32_t i = 0; i < o->n_dram; ++i) {
            const ival_t & d = o->dram_wr[i];
            bool in = false;
            for (const layer_io & x : L.lio)
                in = in || (d.lo >= x.kB && d.hi <= x.vB + L.plane_beats);
            for (const out_rows & x : L.outs) {
                if (in || d.ch != h_ch(x.h)) continue;
                bool all = true;
                const uint64_t b0 = h_beat(x.h);
                for (uint64_t b = d.lo; b < d.hi && all; ++b) {
                    const uint64_t r = b >= b0 ? (b - b0) / x.stride : ~0ull;
                    all = r < x.rows && b - b0 - r * x.stride < x.lines;
                }
                in = all;
            }
            if (!in) {
                CHECK(0, "%s [%s]: a write outside the outputs and planes: ch%u beats [%llu, %llu)", L.c->name, tag, d.ch,
                      (unsigned long long)d.lo, (unsigned long long)d.hi);
                ok = false;
                break;
            }
        }
    }
    return ok;
}

struct run_obs {
    obs o;
    uint32_t done[7];
    uint64_t cycles = 0;
};

bool run_case(const layer_data & L, const char * sched, run_obs * out) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 2;
    so.threads = g_threads;
    tsr_sim * sim = tsr_sim_create_opts(&L.cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", L.c->name);
    if (sim == nullptr) return false;
    fill(sim, L);
    obs_attach(&out->o, sim);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    CHECK(ok, "%s [%s]: boot", L.c->name, sched);
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    CHECK(!ok || gi >= 0, "%s: no graph kernel", L.c->name);
    if (ok && gi >= 0) {
        const uint32_t a[7] = {L.opl_h, uint32_t(L.blob.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0};
        ok = fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s [%s]: launch", L.c->name, sched);
    }
    if (ok) {
        CHECK(TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0,
              "%s [%s]: DONE 0x%08x RET 0x%08x (err ticket 0x%08x info 0x%08x w0 0x%08x pidx %u)", L.c->name, sched, h.done[0],
              h.done[2], h.done[3], h.done[4], h.done[5], h.done[6]);
        for (int i = 0; i < 7; ++i) out->done[i] = h.done[i];
        out->cycles = h.cycles;
        obs_finish(&out->o, sim);
        ok = check_outputs(sim, L, sched, &out->o) && ok;
        const tsr_race_report * rr = nullptr;
        const size_t reports = tsr_sim_race_reports(sim, &rr);
        CHECK(reports == 0, "%s [%s]: %zu race reports", L.c->name, sched, reports);
        const uint64_t stall = tsr_sim_perf(sim, TSR_PERF_PUSH_STALL(L.cfg.num_ch));
        CHECK(stall == 0u, "%s [%s]: PERF PUSH_STALL %llu (CC §7.3)", L.c->name, sched, (unsigned long long)stall);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

void run_scheds(const lcase & c, uint32_t n_sched) {
    layer_data L;
    build(c, L);
    uint32_t n_kinds[64] = {};
    for (const rec & x : L.recs) ++n_kinds[x.kind & 63u];
    printf("%s (%s, %s weights): T %u, n_kv %u, pos %u; %zu records (%u MUL_MAT, %u RMS_NORM, %u ROPE, %u SET_ROWS_KV, %u FLASH_ATTN, "
           "%u GLU, %u ADD, %u MUL) + JOINs = %u ops, list %zu B, preload %zu B; reference FA: %u tiles, %u chunks (%u GENERAL)\n",
           c.name, c.m->name, L.real ? "GGUF layer-0" : "random", c.T, c.n_kv, c.pos0, L.recs.size(), n_kinds[opl::K_MUL_MAT],
           n_kinds[opl::K_RMS_NORM], n_kinds[opl::K_ROPE], n_kinds[opl::K_SET_ROWS_KV], n_kinds[opl::K_FLASH_ATTN], n_kinds[opl::K_GLU],
           n_kinds[opl::K_ADD], n_kinds[opl::K_MUL], L.recs.empty() ? 0u : L.recs.back().op + 1u, L.blob.size(), L.pre.size(),
           L.fa_st.tiles, L.fa_st.chunks, L.fa_st.general_chunks);
    if (c.gguf && !L.real) printf("  note: %s/qwen2.5-0.5b-instruct-q8_0.gguf not readable: random weights instead\n", TSR_M7L_MODELS);
    std::vector<run_obs> runs(n_sched);
    for (uint32_t s = 0; s < n_sched; ++s) {
        char sched[64];
        if (s == 0) snprintf(sched, sizeof(sched), "fixed");
        else snprintf(sched, sizeof(sched), "random:%u", 1000u + s);
        const bool ok = run_case(L, sched, &runs[s]);
        printf("  [%-11s] %s: push -> DONE %llu core cycles (%.1f us), %llu RV instructions\n", sched, ok ? "ok  " : "FAIL",
               (unsigned long long)runs[s].cycles, double(runs[s].cycles) / F_CORE, (unsigned long long)runs[s].o.instret);
        if (s > 0) {
            char what[128];
            snprintf(what, sizeof(what), "%s: %s vs fixed", c.name, sched);
            obs_equal(&runs[0].o, &runs[s].o, what);
            CHECK(memcmp(runs[0].done, runs[s].done, sizeof(runs[0].done)) == 0, "%s: DONE r0-r6 differ", what);
        }
    }
    for (auto & x : runs) obs_free(&x.o);
}

// =================================================================================================
// dump: a case as a scenario for the RTL core bench (tb/core/test_core.py test_core_m7). The case is run
// once on the simulator (fixed schedule, checked as in `suite`); the directory gets
//   meta.txt     cfg NUM_CH T_MAX K_MAX SPM_BYTES CH_BYTES ADDR_BEAT_W CAPS | launch KERNEL A0 A1 A2 | in CH BEAT
//                BYTES OFFSET (a DRAM region before the launch, in data.bin) | out CH BEAT ROWS STRIDE LINES OFFSET
//                (a record's output rows, the simulator's bytes in expect.bin) | plane CH BEAT BYTES OFFSET (a KV
//                plane after the launch) | csr OFFSET VALUE (a host register after the launch) | id OFFSET VALUE (an
//                identification register) | done W0..W6
//                dbg HEAD WPTR DROPS PERF61 PERF62 | hist P V0..V13 (DB §5.6 history entry P, the DBG_HIST_* registers) |
//                trw BEAT LINES (the trace ring, DB §8.3: 32-byte records, in trw.bin)
//   data.bin     the regions the host writes before the launch (inputs, weights, KV history, the list)
//   trw.bin      the trace records the launch wrote (history and the trace writer are on during it)
//   expect.bin   the simulator's bytes of every output and plane after the launch
// =================================================================================================
int dump_case(const lcase & c, const char * dir) {
    layer_data L;
    build(c, L);
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = 2;
    so.threads = g_threads;
    // tsr_u200_k4 has no VU queue (QD_VU 0 without the vector unit); every other device keeps the U200 depths
    const uint8_t depth_novu[TSR_NUM_UNITS] = {TSR_QDEPTH_U200_DMA, TSR_QDEPTH_U200_QACT, TSR_QDEPTH_U200_MVM, 0,
                                               TSR_QDEPTH_U200_ATT, TSR_QDEPTH_U200_SYNC};
    if (c.mm) so.q_depth = depth_novu;
    tsr_sim * sim = tsr_sim_create_opts(&L.cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", c.name);
    if (sim == nullptr) return 1;
    fill(sim, L);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    CHECK(ok && gi >= 0, "%s: boot / graph kernel", c.name);
    uint32_t a[7] = {L.opl_h, uint32_t(L.blob.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0};
    // DB: the dispatch history is on by default; the trace writer writes a 64-line ring in the ch0 IO region below the
    // list during the launch (the RTL bench configures the same: the DB block changes no result)
    const uint32_t trw_beat = 0x6000u, trw_lines = 64u;
    // (the reduced image has neither trace writer nor history: its scenario has no dbg / hist / trw lines)
    if (!c.mm) {
        tsr_sim_csr_write(sim, TSR_CSR_DBG_CFG, tsr_sim_csr_read(sim, TSR_CSR_DBG_CFG) | TSR_DBG_CFG_TRW_EN);
        tsr_sim_csr_write(sim, TSR_CSR_DBG_TRW_BASE, trw_beat);
        tsr_sim_csr_write(sim, TSR_CSR_DBG_TRW_LINES, trw_lines);
    }
    if (ok && gi >= 0) ok = fwh_launch(&h, uint32_t(gi), a, 0) == 0;
    CHECK(ok, "%s: launch", c.name);
    CHECK(ok && check_outputs(sim, L, "dump", nullptr), "%s: outputs", c.name);
    if (!ok || gi < 0) return 1;
    if (!c.mm) tsr_sim_csr_write(sim, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    tsr_sim_advance(sim, 5000);
    const std::string d = dir;
    FILE * fm = fopen((d + "/meta.txt").c_str(), "w");
    FILE * fd = fopen((d + "/data.bin").c_str(), "wb");
    FILE * fe = fopen((d + "/expect.bin").c_str(), "wb");
    CHECK(fm && fd && fe, "dump: cannot write into %s", dir);
    if (!fm || !fd || !fe) return 1;
    fprintf(fm, "cfg %u %u %u %u %llu %u %u\n", L.cfg.num_ch, L.cfg.t_max, L.cfg.k_max, L.cfg.spm_bytes,
            (unsigned long long)L.cfg.ch_bytes, L.cfg.addr_beat_w, L.cfg.caps);
    fprintf(fm, "launch %d %u %u %u\n", gi, a[0], a[1], a[2]);
    size_t off = 0;
    for (const region & rg : L.in) {
        fprintf(fm, "in %u %llu %zu %zu\n", rg.ch, (unsigned long long)rg.beat, rg.data.size(), off);
        fwrite(rg.data.data(), 1, rg.data.size(), fd);
        off += rg.data.size();
    }
    fprintf(fm, "in %u %llu %zu %zu\n", h_ch(L.opl_h), (unsigned long long)h_beat(L.opl_h), L.blob.size(), off);
    fwrite(L.blob.data(), 1, L.blob.size(), fd);
    size_t eo = 0;
    for (const out_rows & x : L.outs) {
        fprintf(fm, "out %u %llu %u %u %u %zu\n", h_ch(x.h), (unsigned long long)h_beat(x.h), x.rows, x.stride, x.lines, eo);
        for (uint32_t row = 0; row < x.rows; ++row) {
            const uint8_t * p = tsr_sim_dram(sim, h_ch(x.h)) + (h_beat(x.h) + uint64_t(row) * x.stride) * LINE;
            fwrite(p, 1, size_t(x.lines) * LINE, fe);
            eo += size_t(x.lines) * LINE;
        }
    }
    for (const layer_io & x : L.lio)
        for (uint32_t kv = 0; kv < 2u; ++kv)
            for (uint32_t ch = 0; ch < L.C; ++ch) {
                const uint64_t beat = kv ? x.vB : x.kB;
                fprintf(fm, "plane %u %llu %llu %zu\n", ch, (unsigned long long)beat, (unsigned long long)(L.plane_beats * LINE), eo);
                fwrite(tsr_sim_dram(sim, ch) + beat * LINE, 1, size_t(L.plane_beats * LINE), fe);
                eo += size_t(L.plane_beats * LINE);
            }
    // host registers after the launch: per-unit Q_DONE / Q_ISSUED / Q_ERR_CODE (the VU's and the AT ops' unit
    // routing), Q_PIDX
    for (uint32_t u = 0; u < 6u; ++u) {
        for (uint32_t o : {uint32_t(TSR_CSR_Q_DONE_BASE) + 4u * u, uint32_t(TSR_CSR_Q_ISSUED_BASE) + 4u * u,
                           uint32_t(TSR_CSR_Q_ERR_CODE_BASE) + 16u * u})
            fprintf(fm, "csr %u %u\n", o, tsr_sim_csr_read(sim, o));
    }
    fprintf(fm, "csr %u %u\n", uint32_t(TSR_CSR_Q_PIDX), tsr_sim_csr_read(sim, TSR_CSR_Q_PIDX));
    // the s-register bank (SREG0-15, VU §5.10): what the layer's RED / SOP ops left in it
    for (uint32_t i = 0; i < 16u; ++i) fprintf(fm, "csr %u %u\n", uint32_t(TSR_CSR_SREG_BASE) + 4u * i, tsr_sim_csr_read(sim, TSR_CSR_SREG_BASE + 4u * i));
    // PERF event counters (EX §6; the time counters depend on the schedule): RETIRED, PUSHES, LAUNCHES, VU_LINES,
    // KVST_RMW, as the host reads them (LO half)
    for (uint32_t i : {uint32_t(TSR_PERF_RETIRED), uint32_t(TSR_PERF_EXT(L.cfg.num_ch, TSR_PERF_XOFF_PUSHES)),
                       uint32_t(TSR_PERF_EXT(L.cfg.num_ch, TSR_PERF_XOFF_LAUNCHES)),
                       uint32_t(TSR_PERF_EXT(L.cfg.num_ch, TSR_PERF_XOFF_VU_LINES)),
                       uint32_t(TSR_PERF_EXT(L.cfg.num_ch, TSR_PERF_XOFF_KVST_RMW))})
        fprintf(fm, "csr %u %u\n", uint32_t(TSR_CSR_PERF_LO_BASE) + 8u * i, tsr_sim_csr_read(sim, TSR_CSR_PERF_LO_BASE + 8u * i));
    for (uint32_t o : {uint32_t(TSR_CSR_ID), uint32_t(TSR_CSR_VERSION), uint32_t(TSR_CSR_CFG0), uint32_t(TSR_CSR_CFG1),
                       uint32_t(TSR_CSR_CFG2), uint32_t(TSR_CSR_CAPS), uint32_t(TSR_CSR_CFG3), uint32_t(TSR_CSR_CTL_ID),
                       uint32_t(TSR_CSR_RV_CFG), uint32_t(TSR_CSR_Q_DEPTH), uint32_t(TSR_CSR_LQ_CFG),
                       uint32_t(TSR_CSR_PERF_INFO)})
        fprintf(fm, "id %u %u\n", o, tsr_sim_csr_read(sim, o));
    const uint32_t hist_head = c.mm ? 0u : tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_HEAD);
    const uint32_t wptr = c.mm ? 0u : tsr_sim_csr_read(sim, TSR_CSR_DBG_TRW_WPTR);
    if (!c.mm)
        fprintf(fm, "dbg %u %u %u %llu %llu\n", hist_head, wptr, tsr_sim_csr_read(sim, TSR_CSR_DBG_TRW_DROPS),
                (unsigned long long)tsr_sim_perf(sim, TSR_PERF_TRW_RECORDS), (unsigned long long)tsr_sim_perf(sim, TSR_PERF_TRW_DROPS));
    for (uint32_t p = 0; p < hist_head; ++p) {
        tsr_sim_csr_write(sim, TSR_CSR_DBG_HIST_IDX, p);
        fprintf(fm, "hist %u", p);
        for (uint32_t o : {uint32_t(TSR_CSR_DBG_HIST_INFO), uint32_t(TSR_CSR_DBG_HIST_PIDX), uint32_t(TSR_CSR_DBG_HIST_TICKET),
                           uint32_t(TSR_CSR_DBG_HIST_DEP0), uint32_t(TSR_CSR_DBG_HIST_DEP1), uint32_t(TSR_CSR_DBG_HIST_PC)})
            fprintf(fm, " %u", tsr_sim_csr_read(sim, o));
        for (uint32_t k = 0; k < 8u; ++k) fprintf(fm, " %u", tsr_sim_csr_read(sim, TSR_CSR_DBG_HIST_W_BASE + 4u * k));
        fprintf(fm, "\n");
    }
    CHECK(c.mm || (wptr > 0 && wptr <= 2u * trw_lines), "dump: %u trace records (a wrapped ring is not compared)", wptr);
    if (!c.mm) fprintf(fm, "trw %u %u\n", trw_beat, trw_lines);
    {
        FILE * ft = fopen((d + "/trw.bin").c_str(), "wb");
        CHECK(ft != nullptr, "dump: cannot write trw.bin");
        if (ft) {
            fwrite(tsr_sim_dram(sim, 0) + uint64_t(trw_beat) * LINE, 32u, wptr, ft);
            fclose(ft);
        }
    }
    fprintf(fm, "done");
    for (int i = 0; i < 7; ++i) fprintf(fm, " %u", h.done[i]);
    fprintf(fm, "\n");
    fclose(fm);
    fclose(fd);
    fclose(fe);
    printf("%s: scenario in %s: %zu input bytes, %zu expected bytes, list %zu B, %zu records, push -> DONE %llu core cycles\n",
           c.name, dir, off + L.blob.size(), eo, L.blob.size(), L.recs.size(), (unsigned long long)h.cycles);
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return 0;
}

// =================================================================================================
// ggml: the same layer on ggml-cpu
// =================================================================================================
#if defined(TSR_M7L_GGML)
struct cmp_t {
    double nmse = 0, max_abs = 0, kl_max = 0;
};
cmp_t compare(const float * dev, const float * cpu, size_t rows, size_t n) {
    cmp_t r;
    double se = 0.0, sc = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        const float * a = dev + i * n, * b = cpu + i * n;
        double ma = -INFINITY, mb = -INFINITY;
        for (size_t d = 0; d < n; ++d) {
            const double e = std::fabs(double(a[d]) - double(b[d]));
            se += e * e;
            sc += double(b[d]) * double(b[d]);
            r.max_abs = std::max(r.max_abs, e);
            ma = std::max(ma, double(a[d]));
            mb = std::max(mb, double(b[d]));
        }
        double za = 0.0, zb = 0.0;
        for (size_t d = 0; d < n; ++d) { za += std::exp(double(a[d]) - ma); zb += std::exp(double(b[d]) - mb); }
        double kl = 0.0;
        for (size_t d = 0; d < n; ++d) {
            const double pb = std::exp(double(b[d]) - mb) / zb, pa = std::exp(double(a[d]) - ma) / za;
            if (pb > 0.0) kl += pb * std::log(pb / pa);
        }
        r.kl_max = std::max(r.kl_max, kl);
    }
    r.nmse = sc > 0.0 ? se / sc : se;
    return r;
}

// ggml-cpu threads (the test's ctest PROCESSORS, 4; TSR_M7L_GGML_THREADS overrides).
int ggml_threads() {
    const char * e = getenv("TSR_M7L_GGML_THREADS");
    const int n = e && e[0] ? atoi(e) : 4;
    return n > 0 ? n : 4;
}

// The layer as llama.cpp's graph builds it (qwen2.cpp; gemma4.cpp, an SWA layer), on ggml-cpu, with
// the case's weights, input, positions, KV history and mask. out: [T][n_embd]; att: [T][n_head D].
// Two graphs: the new tokens' K / V rows first; then, with kdev / vdev, the
// device's cache images (the new cells as QACT A32 + KVST wrote them) are what FLASH_ATTN_EXT reads,
// else the host quantizes the CPU's rows into their cells with ggml's reference quantizer
// (quantize_row_q8_0_ref, the from_float_ref type trait: QACT A32's) and returns the cache images in
// kcpu / vcpu. The first form isolates the device's arithmetic after the cache (a near-floor gate);
// the second also checks the K / V wiring, but a K / V value at a q8_0 rounding boundary (the MUL_MATs
// sum in another order) can land on the neighbouring code: one such V code made the 0.5B prefill
// tile's output NMSE 4.9e-6 against 1e-13 elsewhere (a per-token diagnosis).
bool cpu_layer(const layer_data & L, std::vector<float> & out, std::vector<float> & att,
               const std::vector<uint8_t> * kdev = nullptr, const std::vector<uint8_t> * vdev = nullptr,
               std::vector<uint8_t> * kcpu = nullptr, std::vector<uint8_t> * vcpu = nullptr) {
    const model & m = *L.m;
    const lcase & c = *L.c;
    const uint32_t T = c.T, D = m.D, Qd = m.n_head * D, E = L.E;
    size_t wbytes = L.w.wq.size() + L.w.wk.size() + L.w.wv.size() + L.w.wo.size() + L.w.wg.size() + L.w.wu.size() + L.w.wd.size();
    const size_t act = size_t(T) * (8u * m.n_embd + 12u * Qd + 6u * m.n_ff) * 4u + 2u * L.kc0[0].size();
    struct ggml_init_params ip = {wbytes + 4u * act + (256u << 20), nullptr, false};
    struct ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) return false;
    auto q8 = [&](const std::vector<uint8_t> & b, uint32_t K, uint32_t N) {
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        memcpy(t->data, b.data(), b.size());
        return t;
    };
    auto f1 = [&](const std::vector<float> & v) {
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, int64_t(v.size()));
        memcpy(t->data, v.data(), v.size() * 4u);
        return t;
    };
    auto norm = [&](struct ggml_tensor * x, struct ggml_tensor * wgt) {
        struct ggml_tensor * y = ggml_rms_norm(ctx, x, m.eps);
        return wgt ? ggml_mul(ctx, y, wgt) : y;
    };
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.n_embd, T);
    memcpy(x->data, L.x.data(), L.x.size() * 4u);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    struct ggml_tensor * kidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    for (uint32_t t = 0; t < T; ++t) {
        static_cast<int32_t *>(pos->data)[t] = int32_t(c.pos0 + t);
        static_cast<int64_t *>(kidx->data)[t] = int64_t(c.pos0 + t);
    }
    struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, c.n_kv);
    struct ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, c.n_kv);
    memcpy(kc->data, L.kc0[0].data(), L.kc0[0].size());
    memcpy(vc->data, L.vc0[0].data(), L.vc0[0].size());
    struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, T, 1, 1);
    memcpy(mask->data, L.mask16.data(), L.mask16.size() * 2u);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    struct ggml_tensor * cur = norm(x, f1(L.w.attn_norm));
    struct ggml_tensor * Qc = ggml_mul_mat(ctx, q8(L.w.wq, m.n_embd, Qd), cur);
    struct ggml_tensor * Kc = ggml_mul_mat(ctx, q8(L.w.wk, m.n_embd, E), cur);
    struct ggml_tensor * Vc = ggml_mul_mat(ctx, q8(L.w.wv, m.n_embd, E), cur);
    if (m.bias) {
        Qc = ggml_add(ctx, Qc, f1(L.w.bq));
        Kc = ggml_add(ctx, Kc, f1(L.w.bk));
        Vc = ggml_add(ctx, Vc, f1(L.w.bv));
    }
    Qc = ggml_reshape_3d(ctx, Qc, D, m.n_head, T);
    Kc = ggml_reshape_3d(ctx, Kc, D, m.n_kv_head, T);
    Vc = ggml_reshape_3d(ctx, Vc, D, m.n_kv_head, T);
    if (m.gemma) {
        Qc = norm(Qc, f1(L.w.q_norm));
        Kc = norm(Kc, f1(L.w.k_norm));
        Vc = norm(Vc, nullptr);
    }
    Qc = ggml_rope_ext(ctx, Qc, pos, nullptr, int(D), GGML_ROPE_TYPE_NEOX, 32768, m.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kc = ggml_rope_ext(ctx, Kc, pos, nullptr, int(D), GGML_ROPE_TYPE_NEOX, 32768, m.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    struct ggml_tensor * Kf = ggml_cont(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, Kc), E, T));
    struct ggml_tensor * Vf = ggml_cont(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, Vc), E, T));
    {
        struct ggml_cgraph * g1 = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(g1, Kf);
        ggml_build_forward_expand(g1, Vf);
        if (ggml_graph_compute_with_ctx(ctx, g1, ggml_threads()) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            return false;
        }
        if (kdev != nullptr && vdev != nullptr) {
            memcpy(kc->data, kdev->data(), std::min(kdev->size(), size_t(ggml_nbytes(kc))));
            memcpy(vc->data, vdev->data(), std::min(vdev->size(), size_t(ggml_nbytes(vc))));
        } else {
            const ggml_from_float_t qref = ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref;
            for (uint32_t t = 0; t < T; ++t) {
                const int64_t cell = static_cast<const int64_t *>(kidx->data)[t];
                qref(static_cast<const float *>(Kf->data) + size_t(t) * E, static_cast<uint8_t *>(kc->data) + size_t(cell) * kc->nb[1], E);
                qref(static_cast<const float *>(Vf->data) + size_t(t) * E, static_cast<uint8_t *>(vc->data) + size_t(cell) * vc->nb[1], E);
            }
            if (kcpu) kcpu->assign(static_cast<const uint8_t *>(kc->data), static_cast<const uint8_t *>(kc->data) + ggml_nbytes(kc));
            if (vcpu) vcpu->assign(static_cast<const uint8_t *>(vc->data), static_cast<const uint8_t *>(vc->data) + ggml_nbytes(vc));
        }
    }
    struct ggml_tensor * kv = ggml_view_4d(ctx, kc, D, c.n_kv, m.n_kv_head, 1, kc->nb[1], ggml_row_size(GGML_TYPE_Q8_0, D),
                                           kc->nb[1] * c.n_kv, 0);
    struct ggml_tensor * vv = ggml_view_4d(ctx, vc, D, c.n_kv, m.n_kv_head, 1, vc->nb[1], ggml_row_size(GGML_TYPE_Q8_0, D),
                                           vc->nb[1] * c.n_kv, 0);
    struct ggml_tensor * q = ggml_permute(ctx, Qc, 0, 2, 1, 3);
    struct ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kv, vv, mask, m.attn_scale, 0.0f, 0.0f);
    ggml_prec_set_acc(fa, GGML_PREC_F32);   // as build_attn_mha (src/llama-graph.cpp)
    struct ggml_tensor * a = ggml_reshape_2d(ctx, fa, Qd, T);
    struct ggml_tensor * y;
    if (!m.gemma) {
        cur = ggml_add(ctx, ggml_mul_mat(ctx, q8(L.w.wo, Qd, m.n_embd), a), x);
        struct ggml_tensor * ffn_inp = cur;
        cur = norm(ffn_inp, f1(L.w.ffn_norm));
        struct ggml_tensor * g = ggml_mul_mat(ctx, q8(L.w.wg, m.n_embd, m.n_ff), cur);
        struct ggml_tensor * u = ggml_mul_mat(ctx, q8(L.w.wu, m.n_embd, m.n_ff), cur);
        cur = ggml_swiglu_split(ctx, g, u);
        y = ggml_add(ctx, ggml_mul_mat(ctx, q8(L.w.wd, m.n_ff, m.n_embd), cur), ffn_inp);
    } else {
        cur = norm(ggml_mul_mat(ctx, q8(L.w.wo, Qd, m.n_embd), a), f1(L.w.post_attn_norm));
        struct ggml_tensor * attn_out = ggml_add(ctx, cur, x);
        cur = norm(attn_out, f1(L.w.ffn_norm));
        struct ggml_tensor * g = ggml_mul_mat(ctx, q8(L.w.wg, m.n_embd, m.n_ff), cur);
        struct ggml_tensor * u = ggml_mul_mat(ctx, q8(L.w.wu, m.n_embd, m.n_ff), cur);
        cur = ggml_geglu_split(ctx, g, u);
        cur = norm(ggml_mul_mat(ctx, q8(L.w.wd, m.n_ff, m.n_embd), cur), f1(L.w.post_ffw_norm));
        cur = ggml_add(ctx, cur, attn_out);
        const std::vector<float> s1(1, L.w.out_scale);
        y = ggml_mul(ctx, cur, f1(s1));
    }
    ggml_build_forward_expand(gf, y);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, ggml_threads());
    const bool ok = st == GGML_STATUS_SUCCESS;
    if (ok) {
        out.assign(size_t(T) * m.n_embd, 0.0f);
        memcpy(out.data(), y->data, out.size() * 4u);
        att.assign(size_t(T) * Qd, 0.0f);
        memcpy(att.data(), a->data, att.size() * 4u);
    }
    ggml_free(ctx);
    return ok;
}

void run_ggml() {
    ggml_cpu_init();
    printf("  %-22s %-9s %10s %10s %10s   %10s %10s %10s\n", "case", "weights", "out NMSE", "max|d|", "KL_max", "attn NMSE", "max|d|",
           "KL_max");
    for (const lcase & c : cases()) {
        if (c.mm) continue;   // MUL_MAT-only M6 lists have no layer to compare with ggml (their reference is mulmat_ref)
        layer_data L;
        build(c, L);
        run_obs ro;
        const bool ok = run_case(L, "fixed", &ro);
        obs_free(&ro.o);
        if (!ok) continue;
        // the device's rows = the reference's (checked by run_case): read them from want
        const model & m = *L.m;
        const uint32_t T = c.T, Qd = m.n_head * m.D;
        std::vector<float> dev_y(size_t(T) * m.n_embd), dev_a(size_t(T) * Qd), cpu_y, cpu_a;
        for (size_t i = 0; i < L.outs.size(); ++i) {
            const out_rows & o = L.outs[i];
            std::vector<float> * dst = o.h == L.lio[0].y ? &dev_y : o.h == L.lio[0].att ? &dev_a : nullptr;
            if (!dst) continue;
            const uint32_t n = o.h == L.lio[0].y ? m.n_embd : Qd;
            for (uint32_t t = 0; t < T; ++t) memcpy(dst->data() + size_t(t) * n, L.want[i].data() + size_t(t) * o.lines * LINE, size_t(n) * 4u);
        }
        // The device's cache images of the layer (want_planes: K then V, per channel; = the device's
        // planes, checked bit for bit by run_case).
        const tsr_kv_geom geo = {L.E, L.C, L.LG};
        const size_t cb = size_t(c.n_kv) * (L.E / 32u) * 34u;
        std::vector<uint8_t> kdev(cb), vdev(cb), kcpu, vcpu;
        {
            std::vector<const uint8_t *> kp(L.C), vp(L.C);
            for (uint32_t ch = 0; ch < L.C; ++ch) {
                kp[ch] = L.want_planes[ch].data();
                vp[ch] = L.want_planes[L.C + ch].data();
            }
            tsr_kv_get_bytes(TSR_KV_K, &geo, kp.data(), kdev.data(), 0, cb);
            tsr_kv_get_bytes(TSR_KV_V, &geo, vp.data(), vdev.data(), 0, cb);
        }
        std::vector<float> cpu_y2, cpu_a2;
        const bool cok = cpu_layer(L, cpu_y, cpu_a, nullptr, nullptr, &kcpu, &vcpu) && cpu_layer(L, cpu_y2, cpu_a2, &kdev, &vdev);
        CHECK(cok, "%s: ggml compute failed", c.name);
        if (!cok) continue;
        // The new cells: the CPU's K / V codes against the device's (q8_0 blocks of 34 bytes: fp16 d, 32
        // int8). A wiring error moves most codes; a rounding-boundary value moves one code by 1.
        uint32_t n_codes = 0, n_diff = 0, n_far = 0;
        for (uint32_t t = 0; t < T; ++t) {
            const size_t row = size_t(c.pos0 + t) * (L.E / 32u) * 34u;
            for (int kv = 0; kv < 2; ++kv) {
                const std::vector<uint8_t> & a = kv ? vcpu : kcpu, & b = kv ? vdev : kdev;
                for (uint32_t blk = 0; blk < L.E / 32u; ++blk) {
                    const size_t o = row + size_t(blk) * 34u;
                    for (uint32_t i = 0; i < 32u; ++i, ++n_codes) {
                        const int qa = int8_t(a[o + 2u + i]), qb = int8_t(b[o + 2u + i]);
                        n_diff += qa != qb;
                        n_far += std::abs(qa - qb) > 1;
                    }
                    n_far += memcmp(&a[o], &b[o], 2) != 0 && std::abs(int(tsr_fp16_to_f32(uint16_t(a[o] | a[o + 1] << 8)) /
                                                                      tsr_fp16_to_f32(uint16_t(b[o] | b[o + 1] << 8)) * 1000.0f) - 1000) > 1;
                }
            }
        }
        const cmp_t ry = compare(dev_y.data(), cpu_y.data(), T, m.n_embd);
        const cmp_t ra = compare(dev_a.data(), cpu_a.data(), size_t(T) * m.n_head, m.D);
        const cmp_t ry2 = compare(dev_y.data(), cpu_y2.data(), T, m.n_embd);
        const cmp_t ra2 = compare(dev_a.data(), cpu_a2.data(), size_t(T) * m.n_head, m.D);
        printf("  %-22s %-9s %10.2e %10.2e %10.2e   %10.2e %10.2e %10.2e   new K/V codes %u of %u differ (by 1)\n", c.name,
               L.real ? "GGUF" : "random", ry.nmse, ry.max_abs, ry.kl_max, ra.nmse, ra.max_abs, ra.kl_max, n_diff, n_codes);
        printf("  %-22s %-9s %10.2e %10.2e %10.2e   %10.2e %10.2e %10.2e   (the device's cache)\n", "", "", ry2.nmse,
               ry2.max_abs, ry2.kl_max, ra2.nmse, ra2.max_abs, ra2.kl_max);
        // Bounds (test-backend-ops' 5e-4 was 100x above every measurement). The
        // remaining differences are rounding-boundary values: ggml-cpu quantizes the activations of its
        // MUL_MATs with the x86 quantize_row_q8_0 (127/amax, round-half-even; QACT A32 is the reference
        // quantizer: 1/d, roundf) and sums the block products in another order, so a value on a q8_0
        // boundary lands on the neighbouring code now and then (the 0.5B prefill tile: 3 of 2,048 new
        // K / V codes; with the device's cache its output still differs at NMSE 1.6e-6 through the
        // activations of wo / gate / up / down), and its RMS_NORM sums in double (VU §6.2: f32;
        // Gemma 4: four norms per layer). Measured: output NMSE <= 4.9e-6 (CPU cache) /
        // 1.6e-6 (device cache), attention <= 8.2e-10 / 5.2e-13. Gates: output 2e-5, attention 1e-8
        // (CPU cache) / 1e-10 (device cache), new codes all within one step and at most 1 in 256 moved.
        CHECK(ry.nmse <= 2e-5, "%s: layer output NMSE %.3e above 2e-5", c.name, ry.nmse);
        CHECK(ra.nmse <= 1e-8, "%s: attention output NMSE %.3e above 1e-8", c.name, ra.nmse);
        CHECK(n_far == 0u && n_diff * 256u <= n_codes, "%s: new K / V cells: %u of %u codes differ (%u by more than one step)",
              c.name, n_diff, n_codes, n_far);
        CHECK(ry2.nmse <= 2e-5, "%s: layer output NMSE %.3e with the device's cache above 2e-5", c.name, ry2.nmse);
        CHECK(ra2.nmse <= 1e-10, "%s: attention output NMSE %.3e with the device's cache above 1e-10", c.name, ra2.nmse);
    }
}
#endif

// =================================================================================================
// perf
// =================================================================================================
double g_f_rv = 166.7;

struct op_ev {
    uint32_t unit = 0, w0 = 0;
    uint32_t rec = ~0u;                 // the OPL record whose handler pushed it (~0u: the executor's own)
    uint64_t push = 0, start = 0, done = 0;
};
struct perf_ctx {
    tsr_rv * rv = nullptr;
    uint64_t last = 0, work = 0, work_at_push = 0;
    uint32_t last_w0 = 0;
    uint32_t cur_op = ~0u;
    uint64_t op_mark = 0;
    std::vector<double> op_work;        // per op index: work between its RT_INFO write and the next one
    fwprof::spin_filter spin;
    fwprof::prof * prof = nullptr;
    std::map<uint32_t, op_ev> ops;      // by push index
};
int perf_retire(void * ctx, const tsr_rv_retire * r) {
    perf_ctx * p = static_cast<perf_ctx *>(ctx);
    const uint64_t now = tsr_rv_cycles(p->rv);
    const uint64_t d = now - p->last;
    p->last = now;
    const bool spin = p->spin.step(r);
    if (d <= 256u) {
        if (!spin) p->work += d;
        if (p->prof) p->prof->add(r, d, spin);
    }
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_W(0)) p->last_w0 = r->mem_val;
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_PUSH && (p->last_w0 & 0xFFu) != TSR_OP_FENCE)
        p->work_at_push = p->work;
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_RT_INFO) {   // the executor's op boundary
        if (p->cur_op != ~0u && p->cur_op < p->op_work.size()) p->op_work[p->cur_op] += double(p->work - p->op_mark);
        p->cur_op = r->mem_val;
        p->op_mark = p->work;
    }
    return 0;
}
void perf_q(void * ctx, tsr_sim *, const tsr_q_event * ev) {
    perf_ctx * p = static_cast<perf_ctx *>(ctx);
    if (ev->kind == TSR_QEV_PUSH) {
        op_ev & o = p->ops[ev->pidx];
        o.unit = ev->unit;
        o.w0 = ev->w ? ev->w[0] : 0u;
        o.push = ev->time;
        o.rec = p->cur_op;
    } else if (ev->kind == TSR_QEV_START || ev->kind == TSR_QEV_DONE) {
        auto it = p->ops.find(ev->pidx);
        if (it == p->ops.end()) return;
        if (ev->kind == TSR_QEV_START) it->second.start = ev->time;
        else it->second.done = ev->time;
    }
}

struct unit_stat {
    uint64_t busy = 0, idle_issue = 0;
    uint32_t n = 0;
    std::map<uint32_t, uint64_t> idle_by_kind;   // idle_issue by the kind of the record whose op was late
};
struct perf_run {
    double work = 0;                    // RV cycles of work up to the last op push
    uint64_t launch = 0;                // core cycles push -> DONE
    uint64_t span = 0;                  // first op start -> last op done (core cycles)
    unit_stat u[8];
    std::vector<double> op_work;
    std::vector<uint32_t> op_kind;
};
// One launch (after a warm-up launch of the same list) on a fresh simulator, fixed schedule.
// qmode: 0 U200 depths, 1 every queue 31 deep, 2 the U200 depths with a 31-deep DMA queue.
perf_run perf_launch(const layer_data & L, int qmode, double ratio, fwprof::prof * prof = nullptr) {
    perf_run pr;
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.threads = g_threads;
    static const uint8_t deep_q[6] = {31, 31, 31, 31, 0, 31}, dma_q[6] = {31, 4, 4, 16, 0, 4};
    if (qmode == 1) so.q_depth = deep_q;
    if (qmode == 2) so.q_depth = dma_q;
    so.rv_ratio = ratio;
    tsr_sim * sim = tsr_sim_create_opts(&L.cfg, &so);
    fill(sim, L);
    fwh h;
    if (fwh_open(&h, sim, g_image) != 0) {
        CHECK(0, "perf: boot");
        tsr_sim_destroy(sim);
        return pr;
    }
    if (ratio == 0.0) g_f_rv = tsr_sim_rv_ratio(sim) * F_CORE;
    const int gi = fwh_kernel(&h, "graph");
    const uint32_t a[7] = {L.opl_h, uint32_t(L.blob.size()), 0, 0, 0, 0, 0};
    // Warm-up (the steady state of a decode loop: kernel.hpp's opcode -> unit table filled); the
    // second launch rewrites the same rows and cells.
    CHECK(gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0 && fwh_ret(&h) == 0, "perf: warm-up launch");
    // op index -> kind from the blob
    {
        const uint8_t * b = L.blob.data();
        uint32_t hw[16];
        memcpy(hw, b, 64);
        const uint32_t n_ops = hw[opl::hw::N_OPS], ops_off = hw[opl::hw::OPS_OFF];
        size_t at = ops_off;
        for (uint32_t i = 0; i < n_ops && at + 4u <= L.blob.size(); ++i) {
            uint32_t w0;
            memcpy(&w0, b + at, 4);
            pr.op_kind.push_back(opl::w0_kind(w0));
            at += size_t(opl::w0_slots(w0)) * 64u;
        }
    }
    perf_ctx p;
    p.rv = tsr_rv_agent_rv(tsr_sim_rv_agent(sim));
    p.last = tsr_rv_cycles(p.rv);
    p.prof = prof;
    p.op_work.assign(pr.op_kind.size(), 0.0);
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), perf_retire, &p);
    const int lid = tsr_sim_add_listener(sim, perf_q, &p);
    const bool ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0;
    CHECK(ok && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0, "perf: DONE 0x%08x RET 0x%08x", h.done[0], h.done[2]);
    // the last op's work runs to the last push
    if (p.cur_op != ~0u && p.cur_op < p.op_work.size()) p.op_work[p.cur_op] += double(p.work_at_push - p.op_mark);
    pr.launch = h.cycles;
    pr.work = double(p.work_at_push);
    pr.op_work = p.op_work;
    uint64_t lo = ~0ull, hi = 0;
    uint64_t prev_done[8] = {};
    for (const auto & kv : p.ops) {
        const op_ev & o = kv.second;
        // ops of the launch that ran (the executor's own loads included; join_all's FENCE and SYNC
        // join NOPs, which wait rather than work, excluded)
        if ((o.w0 & 0xFFu) == TSR_OP_FENCE || o.unit >= 8u || o.unit == TSR_UNIT_SYNC || o.start == 0u || o.done < o.start) continue;
        lo = std::min(lo, o.start);
        hi = std::max(hi, o.done);
        unit_stat & u = pr.u[o.unit];
        u.busy += o.done - o.start;
        ++u.n;
        if (prev_done[o.unit] != 0 && o.push > prev_done[o.unit]) {
            const uint64_t gap = std::min(o.start, o.push) - prev_done[o.unit];
            u.idle_issue += gap;
            u.idle_by_kind[o.rec < pr.op_kind.size() ? pr.op_kind[o.rec] : 0xFFu] += gap;
        }
        prev_done[o.unit] = o.done;
    }
    pr.span = hi > lo ? hi - lo : 0;
    const uint64_t stall = tsr_sim_perf(sim, TSR_PERF_PUSH_STALL(tsr_sim_config(sim)->num_ch));
    CHECK(stall == 0u, "perf: PERF PUSH_STALL %llu (CC §7.3)", (unsigned long long)stall);
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), nullptr, nullptr);
    tsr_sim_remove_listener(sim, lid);
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return pr;
}

const char * kind_name(uint32_t k) {
    switch (k) {
        case opl::K_JOIN: return "JOIN";
        case opl::K_MUL_MAT: return "MUL_MAT";
        case opl::K_ADD: return "ADD";
        case opl::K_MUL: return "MUL";
        case opl::K_SCALE: return "SCALE";
        case opl::K_RMS_NORM: return "RMS_NORM";
        case opl::K_GLU: return "GLU";
        case opl::K_ROPE: return "ROPE";
        case opl::K_GET_ROWS: return "GET_ROWS";
        case opl::K_SET_ROWS_KV: return "SET_ROWS_KV";
        case opl::K_FLASH_ATTN: return "FLASH_ATTN";
        case opl::K_CPY: return "CPY";
        case opl::K_UNARY: return "UNARY";
        default: return "?";
    }
}

void perf_case(const char * name, uint32_t n_big) {
    const lcase & c = find_case(name);
    layer_data L1, Ln;
    build(c, L1, 1, false, false);
    build(c, Ln, n_big, false, false);
    const char * core = getenv("TSR_SIM_RV_CORE");
    printf("fw_m7_layer_perf: %s (%s weights), %u vs 1 layers, Ibex %s, backend list flags (no MARK)\n", name, L1.real ? "GGUF" : "random",
           n_big, core && core[0] ? core : tsr_rv_core_name(TSR_RV_CORE_DEFAULT));
    struct cfg_t { const char * tag; int qmode; double ratio; };
    const cfg_t cfgs[] = {{"166.7 MHz, U200 queues", 0, 0.0}, {"166.7 MHz, deep queues", 1, 0.0},
                          {"166.7 MHz, U200 queues but a 31-deep DMA queue", 2, 0.0},
                          {"250 MHz, U200 queues", 0, 250.0 / F_CORE}};
    for (const cfg_t & cf : cfgs) {
        fwprof::prof prof[2];
        const bool syms = cf.qmode == 1 && prof[0].load_symbols(g_image) && prof[1].load_symbols(g_image);
        const perf_run r1 = perf_launch(L1, cf.qmode, cf.ratio, syms ? &prof[0] : nullptr);
        const perf_run rn = perf_launch(Ln, cf.qmode, cf.ratio, syms ? &prof[1] : nullptr);
        const perf_run ri = perf_launch(Ln, cf.qmode, 4.0);
        const double f_rv = cf.ratio > 0.0 ? cf.ratio * F_CORE : g_f_rv;
        const double per_layer_work = (rn.work - r1.work) / double(n_big - 1u);
        const double per_layer_dev = double(rn.launch - r1.launch) / double(n_big - 1u);
        printf("  [%s] F_RV %.1f MHz\n", cf.tag, f_rv);
        printf("    1 layer: RISC-V work %.0f RV cycles = %.1f us; push -> DONE %llu core cycles = %.1f us; units span %.1f us\n", r1.work,
               r1.work / f_rv, (unsigned long long)r1.launch, double(r1.launch) / F_CORE, double(r1.span) / F_CORE);
        printf("    %u layers: RISC-V work %.0f RV cycles = %.1f us; push -> DONE %.1f us; near-ideal issuer %.1f us: the RISC-V's "
               "speed costs %+.2f %% of the launch\n", n_big, rn.work, rn.work / f_rv, double(rn.launch) / F_CORE,
               double(ri.launch) / F_CORE, 100.0 * (double(rn.launch) - double(ri.launch)) / double(ri.launch));
        printf("    per added layer: RISC-V work %.0f RV cycles = %.1f us against %.1f us of device time (push -> DONE)\n",
               per_layer_work, per_layer_work / f_rv, per_layer_dev / F_CORE);
        static const char * const un[6] = {"DMA", "QACT", "MVM", "VU", "AT", "SYNC"};
        for (uint32_t u = 0; u < 5u; ++u) {
            const unit_stat & s = rn.u[u];
            if (s.n == 0u) continue;
            printf("    %-4s %4u ops: busy %5.1f %% of the span, idle waiting for a push %5.2f %% of the span (near-ideal issuer %5.2f %%)",
                   un[u], s.n, 100.0 * double(s.busy) / double(rn.span), 100.0 * double(s.idle_issue) / double(rn.span),
                   100.0 * double(ri.u[u].idle_issue) / double(ri.span));
            if (s.idle_issue != 0u) {
                printf("; late pushes by record kind:");
                for (const auto & kv : s.idle_by_kind)
                    printf(" %s %.1f us", kv.first == 0xFFu ? "executor" : kind_name(kv.first), double(kv.second) / F_CORE);
            }
            printf("\n");
        }
        // per kind: RISC-V cycles of work per record (all layers of the n-layer list)
        std::map<uint32_t, std::pair<double, uint32_t>> pk;
        for (size_t i = 0; i < rn.op_work.size() && i < rn.op_kind.size(); ++i) {
            auto & e = pk[rn.op_kind[i]];
            e.first += rn.op_work[i];
            ++e.second;
        }
        printf("    RISC-V cycles of work per record (spin-free, executor included):");
        for (const auto & kv : pk) printf(" %s %.0f (x%u)", kind_name(kv.first), kv.second.first / kv.second.second, kv.second.second);
        printf("\n");
        if (syms) {
            fwprof::prof::diff(prof[1], prof[0]).report(stdout, "    profile per added layer", double(n_big - 1u), 24, (std::string(".layer.") + name).c_str());
        }
    }
}

void run_perf() {
    perf_case("qwen05b-decode", 3);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE suite GROUP|ggml|perf|dump CASE DIR [--scheds N] [--threads N] [--only NAME]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    struct stat st;
    if (stat(g_image, &st) != 0) {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    const std::string mode = argv[2];
    std::string group;
    int i = 3;
    if (mode == "suite" && argc > 3) group = argv[i++];
    uint32_t n_sched = 8;
    const char * only = nullptr;
    for (; i < argc; ++i) {
        if (!strcmp(argv[i], "--scheds") && i + 1 < argc) n_sched = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) g_threads = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    }
    if (mode == "suite") {
        uint32_t n = 0;
        for (const auto & c : cases()) {
            if (group != c.group || (only && strcmp(only, c.name) != 0)) continue;
            run_scheds(c, n_sched);
            ++n;
        }
        CHECK(n > 0, "no case in group %s", group.c_str());
    } else if (mode == "ggml") {
#if defined(TSR_M7L_GGML)
        run_ggml();
#else
        printf("%s: built without ggml (TSR_WITH_LLAMA off): skipped\n", argv[0]);
        return 77;
#endif
    } else if (mode == "perf") {
        run_perf();
    } else if (mode == "dump") {
        // test_fw_m7_layer IMAGE dump CASE DIR
        if (argc < 5) {
            fprintf(stderr, "usage: %s IMAGE dump CASE DIR\n", argv[0]);
            return 2;
        }
        return dump_case(find_case(argv[3]), argv[4]) | test_finish("fw_m7_layer dump");
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    return test_finish(("fw_m7_layer " + mode + (group.empty() ? "" : " " + group)).c_str());
}

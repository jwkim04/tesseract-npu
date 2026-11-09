// tsr-numcheck — one-prompt numeric dump of every ggml graph node (docs/numcheck.md).
//
// Numeric verification with as few model runs as possible: one short prompt, decoded in ONE
// prefill batch with logits for every position, three runs per model, each its own process:
//   cpu    ggml CPU backend only, input = token_embd rows ("embd", bit-exact with the token path)
//   floor  CPU only, input = token_embd rows with every element nudged by 1 ulp (klcheck's
//          embd_nudge, sequence 0, position = token index, seed --seed)
//   dev    Tesseract device (TSR_DEVICE=sim, or --device mock|vkernel|xrt): every layer and the
//          output head offloaded, KV cache and attention on the host in M6 (offload_kqv = false),
//          on the device in M7 (--phase m7), input "embd"
// All three use the embd input path, so the three graphs are identical node for node. Settings
// (SW-22, docs/design/06-software-verification.md §9.5, §10.4), the same in every run: flash
// attention on, q8_0 K and V cache, LLAMA_ATTN_ROT_DISABLE=1 (set by the tool), offload_kqv by
// phase; recorded in meta.json "info" and checked equal across the runs by selfcheck. The device's
// kinds follow the phase: --phase m6 sets TSR_KINDS=m6, --phase m7 keeps the caller's TSR_KINDS
// (meta.json info "TSR_KINDS"). The per-node observation below makes the scheduler compute one
// node per graph_compute, i.e. one device launch per observed node: numcheck needs no device mode
// of its own (SW-27). Each run observes the graph through llama_context_params.cb_eval and writes
// a run directory:
//
//   <run_dir>/meta.json     format "tsr-numcheck" version 2 (fields below; version 1 = the
//                           FA-off settings, still read), written last
//   <run_dir>/logits.f32    f32 [T][n_vocab]
//   <run_dir>/t/NNNNNN.f32  node i's output, f32, ggml element order (ne0 fastest); non-contiguous
//                           tensors are gathered element by element; F16/BF16/I32 converted to f32,
//                           quantized outputs dequantized with ggml's to_float
//   <run_dir>/t/NNNNNN.src1.f32  src1 (the activation x) of a weight MUL_MAT, same layout
//
// meta.json: { "format", "version", "model" (absolute GGUF path), "run" (cpu|floor|dev),
//   "device" (none|sim|mock|vkernel|xrt), "input" (embd|embd+1ulp; "tokens" only for the self-check run),
//   "seed", "n_tokens", "tokens", "n_vocab", "logits_file", "versions" (dev runs: the Tesseract Core
//   capability, CAPS, the booted image's Tesseract Basis version, Basis ABI, id, kind table signature
//   and every kind's numerics revision, SW §6.9: {"core": "1.0", "caps": N, "basis": "1.0.0", "abi":
//   1, "image": "<16 hex>", "kinds_sig": "0xfa6f", "kinds": {"MUL_MAT": 1, ...}}; null otherwise),
//   "info" {settings, timings, counts;
//   settings: "phase" (m6|m7), "flash_attn" (on|off), "type_k", "type_v" (ggml type names),
//   "offload_kqv" (bool), "attn_rot_disable" (bool: LLAMA_ATTN_ROT_DISABLE was 1 for the context)},
//   "nodes": [ {"i", "name", "op" (ggml_op_name), "ne" [4], "file", "backend", "weight",
//               "weight_type", "src1_file", "src1_ne"} ] }
//   i         index of the node among ALL computed non-view nodes of the graph in graph order.
//             Skipped: pure view ops (NONE / VIEW / RESHAPE / PERMUTE / TRANSPOSE) and nodes
//             without GGML_TENSOR_FLAG_COMPUTE, which no backend computes (llama.cpp builds all
//             three input-embedding branches with ggml_build_forward_select and only the selected
//             one is computed: for embd input the token GET_ROWS and the mixed-batch GET_ROWS / DUP /
//             SET_ROWS stay in the graph with stale buffers). Without --nodes every such node is
//             dumped and i = 0, 1, 2, ...; with --nodes the dumped subset keeps these indices
//             (gaps), so filtered and unfiltered runs stay comparable.
//   backend   ggml_backend_dev_name of the device whose buffer holds the node's output, which is
//             the backend that computed it ("CPU" or "Tesseract0"; equal to ggml_backend_name of
//             those backends). The scheduler places a node's output in its backend's buffer type,
//             or assigns a node whose output is pre-allocated (KV-cache writes) to the backend of
//             that buffer, so the buffer identifies the computing backend.
//   weight    src0 name if the node is a MUL_MAT / MUL_MAT_ID whose src0 is a leaf whose name (or,
//             for a scheduler copy "<backend>#<name>#<copy>", the inner name) is a tensor of the
//             GGUF (any split); else null. weight_type = ggml_type_name(src0->type) (the GGUF type).
//   src1_file only for weight MUL_MATs in --mm-layers (default all): the src1 tensor the
//             computing backend actually read (on the device run: the scheduler's input copy in
//             the Tesseract buffer), read with ggml_backend_tensor_get after the node ran.
//
// Observation: cb_eval asks for every non-view node, so the scheduler computes the graph one node
// at a time and synchronizes after each; `--no-observe` sets no callback (logits only) for the
// check that observing does not change device numerics. `all` runs everything as child processes
// and then `selfcheck`.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include "tsr/opl.hpp"        // kind_name (SW §6.9)
#include "tsr_transport.h"    // tsr_versions (the backend's hook "ggml_backend_tesseract_versions")

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------------------------
// utilities (after klcheck.cpp)

[[noreturn]] static void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "tsr-numcheck: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static long long parse_int(const char * opt, const char * v, long long lo, long long hi) {
    char * end = nullptr;
    errno = 0;
    const long long r = strtoll(v, &end, 0);
    if (errno || end == v || *end || r < lo || r > hi) die("%s: \"%s\" is not an integer in [%lld, %lld]", opt, v, lo, hi);
    return r;
}

static uint64_t parse_u64(const char * opt, const char * v) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long r = strtoull(v, &end, 0);
    if (errno || end == v || *end || v[0] == '-') die("%s: \"%s\" is not an unsigned integer", opt, v);
    return (uint64_t) r;
}

static void mkdir_p(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) die("mkdir %s: %s", cur.c_str(), strerror(errno));
        }
        if (i < path.size()) cur += path[i];
    }
}

static int default_threads() {
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n >= 2 ? (int) (n / 2) : 1;   // klcheck's default (llama.cpp: physical cores)
}

static std::string abs_path(const std::string & p) {
    char buf[PATH_MAX];
    return realpath(p.c_str(), buf) ? std::string(buf) : p;
}

static void write_file(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) die("%s: %s", path.c_str(), strerror(errno));
    if (bytes && fwrite(data, 1, bytes, f) != bytes) die("%s: write failed: %s", path.c_str(), strerror(errno));
    if (fclose(f) != 0) die("%s: write failed: %s", path.c_str(), strerror(errno));
}

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
    fclose(f);
    return true;
}

static int64_t file_size(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (int64_t) st.st_size : -1;
}

// ---------------------------------------------------------------------------------------------
// the 1-ulp input perturbation — copied from klcheck.cpp (embd_nudge_key / embd_nudge), so the
// floor here is the same perturbation as klcheck's floor for sequence 0.

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static uint64_t embd_nudge_key(uint64_t seed, uint32_t seq, uint32_t pos) {
    return splitmix64(splitmix64(splitmix64(seed) ^ seq) ^ ((uint64_t) pos << 1));
}

static float embd_nudge(float x, uint64_t key, uint32_t j) {
    return nextafterf(x, (splitmix64(key + j) >> 63) ? -INFINITY : INFINITY);
}

// ---------------------------------------------------------------------------------------------
// GGUF access: split files, tensor names, token_embd rows (after klcheck's embd_table)

static int gguf_split_count(gguf_context * g) {
    const int64_t kc = gguf_find_key(g, "split.count");
    if (kc < 0) return 1;
    switch (gguf_get_kv_type(g, kc)) {
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, kc);
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(g, kc);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, kc);
        default:               return 1;
    }
}

// All files of a (possibly split) model, given any path llama.cpp accepts (the first split).
static std::vector<std::string> model_files(const std::string & model_path) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(model_path.c_str(), gp);
    if (!g) die("%s: cannot read GGUF", model_path.c_str());
    const int n_split = gguf_split_count(g);
    gguf_free(g);
    std::vector<std::string> out = {model_path};
    char prefix[4096];
    if (n_split > 1 && llama_split_prefix(prefix, sizeof(prefix), model_path.c_str(), 0, n_split) > 0) {
        for (int i = 1; i < n_split; ++i) {
            char p[4096 + 32];
            llama_split_path(p, sizeof(p), prefix, i, n_split);
            out.emplace_back(p);
        }
    }
    return out;
}

// name -> type of every tensor in every split
static std::unordered_map<std::string, ggml_type> gguf_tensor_types(const std::vector<std::string> & files) {
    std::unordered_map<std::string, ggml_type> m;
    for (const std::string & f : files) {
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(f.c_str(), gp);
        if (!g) die("%s: cannot read GGUF", f.c_str());
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) m[gguf_get_tensor_name(g, i)] = gguf_get_tensor_type(g, i);
        gguf_free(g);
    }
    return m;
}

// The token-row scale of the model graph (klcheck.cpp's token_row_scale, the same rule): llama.cpp
// multiplies token rows only by tok_scale (`inp_scale_rows`, src/llama-graph.cpp:2766-2783, :104-112),
// sqrtf(n_embd) for Gemma 3 / 3n / 4 and the Gemma embedding models; the embd rows fed here carry it
// so that the embd path stays the token path (and the "tokens" self-check run bit-identical).
static float token_row_scale(gguf_context * g) {
    const int64_t ka = gguf_find_key(g, "general.architecture");
    if (ka < 0 || gguf_get_kv_type(g, ka) != GGUF_TYPE_STRING) return 1.0f;
    const std::string arch = gguf_get_val_str(g, ka);
    static const char * const scaled[] = {"gemma3", "gemma3n", "gemma4", "gemma-embedding", "gemma-embedding2"};
    if (std::find_if(std::begin(scaled), std::end(scaled), [&](const char * a) { return arch == a; }) == std::end(scaled)) {
        return 1.0f;
    }
    const int64_t ke = gguf_find_key(g, (arch + ".embedding_length").c_str());
    if (ke < 0) die("%s: no %s.embedding_length", arch.c_str(), arch.c_str());
    const int64_t n_embd = gguf_get_kv_type(g, ke) == GGUF_TYPE_UINT64 ? (int64_t) gguf_get_val_u64(g, ke)
                                                                        : (int64_t) gguf_get_val_u32(g, ke);
    return sqrtf((float) n_embd);
}

struct embd_table {
    std::string     file;
    int             fd       = -1;
    uint64_t        data_off = 0;
    ggml_type       type     = GGML_TYPE_F32;
    int64_t         n_embd   = 0, n_rows = 0;
    size_t          row_size = 0;
    ggml_to_float_t to_float = nullptr;
    float           tok_scale = 1.0f;       // token_row_scale() (first split's metadata)

    ~embd_table() { if (fd >= 0) close(fd); }

    void open_model(const std::vector<std::string> & files) {
        for (const std::string & path : files) {
            gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
            gguf_context * g = gguf_init_from_file(path.c_str(), gp);
            if (!g) die("%s: cannot read GGUF", path.c_str());
            if (&path == &files[0]) tok_scale = token_row_scale(g);
            const int64_t t = gguf_find_tensor(g, "token_embd.weight");
            if (t >= 0) {
                const int64_t * ne = gguf_get_tensor_ne(g, t);
                type     = gguf_get_tensor_type(g, t);
                n_embd   = ne[0];
                n_rows   = ne[1];
                row_size = ggml_row_size(type, n_embd);
                data_off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, t);
                to_float = ggml_get_type_traits(type)->to_float;   // NULL for F32: rows are copied
                if (!to_float && type != GGML_TYPE_F32) die("token_embd.weight: type %s has no to_float", ggml_type_name(type));
                fd = ::open(path.c_str(), O_RDONLY);
                if (fd < 0) die("%s: %s", path.c_str(), strerror(errno));
                file = path;
            }
            gguf_free(g);
            if (t >= 0) return;
        }
        die("%s: token_embd.weight not found", files[0].c_str());
    }

    // the row the CPU backend's GET_ROWS produces for `tok` (ggml to_float of the GGUF bytes), times
    // the token-row scale (one f32 multiply, as ggml's MUL by inp_scale_rows)
    void row(int32_t tok, float * dst) const {
        if (tok < 0 || tok >= n_rows) die("token %d outside token_embd (%" PRId64 " rows)", tok, n_rows);
        std::vector<uint8_t> raw(row_size);
        if (pread(fd, raw.data(), row_size, (off_t) (data_off + (uint64_t) tok * row_size)) != (ssize_t) row_size) {
            die("token_embd: short read");
        }
        if (to_float) to_float(raw.data(), dst, n_embd);
        else memcpy(dst, raw.data(), sizeof(float) * n_embd);
        if (tok_scale != 1.0f) {
            for (int64_t j = 0; j < n_embd; ++j) dst[j] *= tok_scale;
        }
    }
};

// ---------------------------------------------------------------------------------------------
// tensor read-back

static bool op_is_view(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

// Reads t (any buffer, also device buffers) into out as f32 in ggml element order (ne0 fastest),
// gathering non-contiguous tensors element by element.
static void gather_f32(const ggml_tensor * t, std::vector<uint8_t> & raw, std::vector<float> & out) {
    const int64_t n = ggml_nelements(t);
    out.resize((size_t) n);
    if (n == 0) return;
    const size_t nbytes = ggml_nbytes(t);
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
        return;
    }
    raw.resize(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const size_t  nb0 = t->nb[0], nb1 = t->nb[1], nb2 = t->nb[2], nb3 = t->nb[3];
    size_t k = 0;
    if (ggml_is_quantized(t->type)) {
        const ggml_to_float_t to_float = ggml_get_type_traits(t->type)->to_float;
        if (!to_float) die("%s: type %s has no to_float", t->name, ggml_type_name(t->type));
        if (nb0 != ggml_type_size(t->type) || ne0 % ggml_blck_size(t->type) != 0) {
            die("%s: quantized tensor with non-contiguous rows", t->name);
        }
        for (int64_t i3 = 0; i3 < ne3; ++i3)
            for (int64_t i2 = 0; i2 < ne2; ++i2)
                for (int64_t i1 = 0; i1 < ne1; ++i1, k += (size_t) ne0) {
                    to_float(raw.data() + i1 * nb1 + i2 * nb2 + i3 * nb3, out.data() + k, ne0);
                }
        return;
    }
    for (int64_t i3 = 0; i3 < ne3; ++i3)
        for (int64_t i2 = 0; i2 < ne2; ++i2)
            for (int64_t i1 = 0; i1 < ne1; ++i1)
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const uint8_t * p = raw.data() + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;
                    float v;
                    switch (t->type) {
                        case GGML_TYPE_F32:  memcpy(&v, p, 4); break;
                        case GGML_TYPE_F16:  { ggml_fp16_t h; memcpy(&h, p, 2); v = ggml_fp16_to_fp32(h); } break;
                        case GGML_TYPE_BF16: { ggml_bf16_t h; memcpy(&h, p, 2); v = ggml_bf16_to_fp32(h); } break;
                        case GGML_TYPE_I32:  { int32_t x; memcpy(&x, p, 4); v = (float) x; } break;
                        case GGML_TYPE_I16:  { int16_t x; memcpy(&x, p, 2); v = (float) x; } break;
                        case GGML_TYPE_I8:   { int8_t x; memcpy(&x, p, 1); v = (float) x; } break;
                        default: die("%s: cannot convert type %s to f32", t->name, ggml_type_name(t->type));
                    }
                    out[k++] = v;
                }
}

// The backend that computed t: the device owning its output buffer (see the header comment).
static std::string backend_of(const ggml_tensor * t) {
    ggml_backend_buffer_t b = t->view_src ? t->view_src->buffer : t->buffer;
    if (!b) return "?";
    ggml_backend_buffer_type_t bt = ggml_backend_buffer_get_type(b);
    ggml_backend_dev_t d = ggml_backend_buft_get_device(bt);
    return d ? ggml_backend_dev_name(d) : ggml_backend_buft_name(bt);
}

// "<backend>#<name>#<copy>" (a scheduler input copy) -> "<name>"; any other name unchanged
static std::string strip_sched_copy(const std::string & n) {
    const size_t a = n.find('#'), b = n.rfind('#');
    if (a == std::string::npos || b <= a) return n;
    const std::string tail = n.substr(b + 1);
    if (tail.empty() || tail.find_first_not_of("0123456789") != std::string::npos) return n;
    return n.substr(a + 1, b - a - 1);
}

// layer of a weight name: N for "blk.N.*", -1 otherwise (output head, token_embd, ...)
static int weight_layer(const std::string & w) {
    int l = -1;
    if (sscanf(w.c_str(), "blk.%d.", &l) == 1) return l;
    return -1;
}

// ---------------------------------------------------------------------------------------------
// the observer (cb_eval)

struct mm_select {
    bool          all = true, output = false;
    std::set<int> layers;
    std::string   spec;   // as given

    // "all", "none", or a comma list of N, A-B and "output" (non-blk weights: the output head)
    void parse(const std::string & s) {
        spec = s;
        all = false;
        output = false;
        layers.clear();
        if (s == "all") { all = true; return; }
        if (s == "none") return;
        size_t p = 0;
        while (p <= s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) q = s.size();
            const std::string item = s.substr(p, q - p);
            if (item == "output" || item == "out" || item == "head") {
                output = true;
            } else if (!item.empty()) {
                const size_t d = item.find('-');
                if (d == std::string::npos) {
                    layers.insert((int) parse_int("--mm-layers", item.c_str(), 0, 1 << 20));
                } else {
                    const int a = (int) parse_int("--mm-layers", item.substr(0, d).c_str(), 0, 1 << 20);
                    const int b = (int) parse_int("--mm-layers", item.substr(d + 1).c_str(), 0, 1 << 20);
                    for (int l = a; l <= b; ++l) layers.insert(l);
                }
            }
            p = q + 1;
        }
    }
    bool selected(const std::string & weight) const {
        if (all) return true;
        const int l = weight_layer(weight);
        return l < 0 ? output : layers.count(l) > 0;
    }
};

struct observer {
    bool        active = false;
    std::string dir;
    bool        use_re = false;
    std::regex  re;
    mm_select   mm;
    std::unordered_map<std::string, ggml_type> gguf;   // GGUF tensor name -> type

    int64_t             n_seen = 0;    // computed non-view nodes seen (ask = true)
    int64_t             n_not_computed = 0;   // non-view nodes without GGML_TENSOR_FLAG_COMPUTE
    int64_t             cur_i  = -1;
    const ggml_tensor * cur_t  = nullptr;
    json                nodes  = json::array();
    uint64_t            bytes  = 0;
    int64_t             n_weight_mm = 0, n_src1 = 0, n_type_mismatch = 0;
    std::vector<uint8_t> raw;
    std::vector<float>   buf;

    // src0 of a MUL_MAT that is a model weight: a leaf with a GGUF tensor name
    bool weight_of(const ggml_tensor * t, std::string * name, ggml_type * type) const {
        if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) return false;
        const ggml_tensor * w = t->src[0];
        if (!w || w->op != GGML_OP_NONE || w->view_src) return false;
        const std::string n = strip_sched_copy(w->name);
        if (!gguf.count(n)) return false;
        *name = n;
        *type = w->type;
        return true;
    }

    void dump(const ggml_tensor * t, int64_t i) {
        char rel[64];
        snprintf(rel, sizeof(rel), "t/%06" PRId64 ".f32", i);
        gather_f32(t, raw, buf);
        write_file(dir + "/" + rel, buf.data(), buf.size() * sizeof(float));
        bytes += buf.size() * sizeof(float);

        json e = json::object();
        e["i"]       = i;
        e["name"]    = t->name;
        e["op"]      = ggml_op_name(t->op);
        e["ne"]      = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        e["file"]    = rel;
        e["backend"] = backend_of(t);
        std::string wname;
        ggml_type   wtype = GGML_TYPE_F32;
        if (weight_of(t, &wname, &wtype)) {
            n_weight_mm++;
            if (gguf.at(wname) != wtype) n_type_mismatch++;
            e["weight"]      = wname;
            e["weight_type"] = ggml_type_name(wtype);
            if (mm.selected(wname)) {
                const ggml_tensor * x = t->src[1];
                char rel1[64];
                snprintf(rel1, sizeof(rel1), "t/%06" PRId64 ".src1.f32", i);
                gather_f32(x, raw, buf);
                write_file(dir + "/" + rel1, buf.data(), buf.size() * sizeof(float));
                bytes += buf.size() * sizeof(float);
                n_src1++;
                e["src1_file"] = rel1;
                e["src1_ne"]   = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
            } else {
                e["src1_file"] = nullptr;
                e["src1_ne"]   = nullptr;
            }
        } else {
            e["weight"]      = nullptr;
            e["weight_type"] = nullptr;
            e["src1_file"]   = nullptr;
            e["src1_ne"]     = nullptr;
        }
        nodes.push_back(std::move(e));
    }
};

// ggml_backend_sched_eval_callback: ask = true once per node before it is computed (true = "I want
// this node": the scheduler computes up to it, synchronizes and calls again with ask = false).
static bool observe_cb(ggml_tensor * t, bool ask, void * ud) {
    observer & o = *(observer *) ud;
    if (ask) {
        if (!o.active || op_is_view(t->op)) return false;
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {   // not selected (ggml_build_forward_select)
            o.n_not_computed++;
            return false;
        }
        const int64_t i = o.n_seen++;
        const bool want = !o.use_re || std::regex_search(t->name, o.re);
        if (want) {
            o.cur_i = i;
            o.cur_t = t;
        }
        return want;
    }
    if (t != o.cur_t) die("internal: cb_eval returned %s without a matching ask", t->name);
    o.dump(t, o.cur_i);
    o.cur_t = nullptr;
    return true;
}

// ---------------------------------------------------------------------------------------------
// tsr-numcheck run

static int g_verbose = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    static ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (g_verbose || last == GGML_LOG_LEVEL_ERROR || last == GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

// One fixed English sentence (55 tokens with the Qwen2.5 tokenizer), truncated to T tokens.
static const char * const k_prompt =
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg "
    "combined movable metal type, an oil-based ink and a wooden screw press, so that a small "
    "workshop could produce hundreds of identical books in the time a scribe needed to copy one.";

struct run_opts {
    std::string model, out, run, device = "sim", input, nodes_re, mm_layers, prompt = k_prompt;
    int         T = 0, threads = default_threads();
    uint64_t    seed = 1;
    bool        observe = true;
    // settings (SW-22; klcheck's): FA on, q8_0 / q8_0 KV, rotation off; offload_kqv by phase
    std::string phase = "m6";
    int         fa = 1, offload_kqv = -1, attn_rot_disable = 1;   // offload_kqv -1: the phase's value
    ggml_type   type_k = GGML_TYPE_Q8_0, type_v = GGML_TYPE_Q8_0;
    bool        offload() const { return offload_kqv >= 0 ? offload_kqv != 0 : phase == "m7"; }
};

static int parse_on_off(const char * opt, const std::string & v) {
    if (v == "on" || v == "1") return 1;
    if (v == "off" || v == "0") return 0;
    die("%s: on or off", opt);
}

static ggml_type parse_cache_type(const char * opt, const std::string & v) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && v == n && ggml_get_type_traits((ggml_type) t)->blck_size > 0) return (ggml_type) t;
    }
    die("%s: unknown ggml type \"%s\" (e.g. q8_0, f16, f32)", opt, v.c_str());
}

static void usage_run() {
    fprintf(stderr,
        "usage: tsr-numcheck run -m MODEL -o RUN_DIR --run cpu|floor|dev [options]\n"
        "  --device KIND       device of the dev run: sim (default), mock, vkernel, xrt (cpu/floor never open one)\n"
        "  -T N                prompt tokens (default: 16 for models > 8 B parameters, else 32)\n"
        "  --seed S            floor: 1-ulp perturbation seed (default 1)\n"
        "  --nodes REGEX       dump only nodes whose name matches (std::regex search; default all)\n"
        "  --mm-layers LIST    src1 dumps of weight MUL_MATs only for these layers: N,A-B,output | all | none\n"
        "  --no-observe        no cb_eval: logits only (observation-invariance check)\n"
        "  --input tokens|embd override the input path (tokens: only for the embd-identity check)\n"
        "  --prompt TEXT       prompt (default: a built-in English sentence)\n"
        "  -t N                CPU threads (default online CPUs / 2, as klcheck)\n"
        "  -v                  llama.cpp log\n"
        "  settings (SW-22, every run of a set must use the same; recorded in meta.json):\n"
        "  --phase m6|m7       M6 (default): KV cache and attention on the host; M7: on the device\n"
        "                      (m6 sets TSR_KINDS=m6; m7 keeps the caller's TSR_KINDS, unset: every kind)\n"
        "  --offload-kqv on|off  override the phase's offload_kqv\n"
        "  --fa on|off  -ctk TYPE  -ctv TYPE  --attn-rot on|off   (defaults: on, q8_0, q8_0, off)\n");
}

static void parse_run(int argc, char ** argv, run_opts & o) {
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * {
            if (i + 1 >= argc) die("%s needs a value", a.c_str());
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")   o.model = val();
        else if (a == "-o" || a == "--out")     o.out = val();
        else if (a == "--run")                  o.run = val();
        else if (a == "--device")               o.device = val();
        else if (a == "-T")                     o.T = (int) parse_int("-T", val(), 1, 4096);
        else if (a == "--seed")                 o.seed = parse_u64("--seed", val());
        else if (a == "--nodes")                o.nodes_re = val();
        else if (a == "--mm-layers")            o.mm_layers = val();
        else if (a == "--no-observe")           o.observe = false;
        else if (a == "--input")                o.input = val();
        else if (a == "--prompt")               o.prompt = val();
        else if (a == "-t" || a == "--threads") o.threads = (int) parse_int("-t", val(), 1, 1024);
        else if (a == "-v" || a == "--verbose") g_verbose = 1;
        else if (a == "--phase")                o.phase = val();
        else if (a == "--fa")                   o.fa = parse_on_off("--fa", val());
        else if (a == "--offload-kqv")          o.offload_kqv = parse_on_off("--offload-kqv", val());
        else if (a == "--attn-rot")             o.attn_rot_disable = !parse_on_off("--attn-rot", val());
        else if (a == "-ctk" || a == "--cache-type-k") o.type_k = parse_cache_type("-ctk", val());
        else if (a == "-ctv" || a == "--cache-type-v") o.type_v = parse_cache_type("-ctv", val());
        else if (a == "-h" || a == "--help")    { usage_run(); exit(0); }
        else { usage_run(); die("run: unknown option %s", a.c_str()); }
    }
    if (o.model.empty() || o.out.empty() || o.run.empty()) { usage_run(); die("run: -m, -o and --run are required"); }
    if (o.run != "cpu" && o.run != "floor" && o.run != "dev") die("--run: cpu, floor or dev");
    if (o.phase != "m6" && o.phase != "m7") die("--phase: m6 or m7");
    if (o.run == "dev" && o.device != "sim" && o.device != "mock" && o.device != "vkernel" && o.device != "xrt") {
        die("--device: sim, mock, vkernel or xrt");
    }
    const std::string def = o.run == "floor" ? "embd+1ulp" : "embd";
    if (o.input.empty()) o.input = def;
    if (o.input == "embd" && o.run == "floor") o.input = def;       // the floor is always nudged
    if (o.input != def && !(o.input == "tokens" && o.run != "floor")) {
        die("--input %s is not valid for --run %s (tokens|embd for cpu/dev)", o.input.c_str(), o.run.c_str());
    }
}

static ggml_backend_dev_t find_tesseract_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "Tesseract", 9) == 0) return d;
    }
    return nullptr;
}

static void clear_run_dir(const std::string & dir) {
    mkdir_p(dir + "/t");
    unlink((dir + "/meta.json").c_str());
    unlink((dir + "/logits.f32").c_str());
    DIR * d = opendir((dir + "/t").c_str());
    if (!d) die("%s/t: %s", dir.c_str(), strerror(errno));
    while (dirent * e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".f32") == 0) unlink((dir + "/t/" + n).c_str());
    }
    closedir(d);
}

static std::string detok(const llama_vocab * vocab, const std::vector<int32_t> & toks) {
    std::string out;
    char buf[256];
    for (int32_t t : toks) {
        const int k = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (k > 0) out.append(buf, k);
    }
    return out;
}

static int cmd_run(int argc, char ** argv) {
    run_opts o;
    parse_run(argc, argv, o);
    const bool dev = o.run == "dev";

    // The device is chosen through the environment before ggml's backend registry is created.
    setenv("TSR_DEVICE", dev ? o.device.c_str() : "none", 1);
    if (!getenv("TSR_SIM_THREADS")) setenv("TSR_SIM_THREADS", "4", 1);   // shared host (docs/numcheck.md)
    // llama.cpp's Hadamard KV rotation off in every run (SW-16, SW-22; read at context creation)
    if (o.attn_rot_disable) setenv("LLAMA_ATTN_ROT_DISABLE", "1", 1);
    else                    unsetenv("LLAMA_ATTN_ROT_DISABLE");
    // the phase's device kinds, as klcheck: M6 MUL_MAT only (SW-14); M7 keeps the caller's TSR_KINDS
    // (a kind list such as all,-RMS_NORM, sw/ggml-tesseract/README.md; unset: every kind, SW-49).
    // The value the run used is recorded in meta.json info "TSR_KINDS".
    if (o.phase == "m6") setenv("TSR_KINDS", "m6,+MUL_MAT_ID", 1);

    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    const std::string model_abs = abs_path(o.model);
    const std::vector<std::string> files = model_files(o.model);

    // model: exactly klcheck's load_model for the device / cpu phases
    const double t0 = now_s();
    llama_model_params mp = llama_model_default_params();
    static ggml_backend_dev_t devs[2] = {nullptr, nullptr};
    if (!dev) {
        mp.n_gpu_layers = 0;                    // devices = {}: nothing is offloaded
    } else {
        devs[0] = find_tesseract_device();
        if (!devs[0]) die("no Tesseract device registered (TSR_DEVICE=%s)", getenv("TSR_DEVICE"));
        mp.n_gpu_layers = 999;                  // -ngl 99: every layer and the output head
    }
    mp.devices         = devs;
    mp.use_extra_bufts = false;                 // CPU weight repack off (klcheck default)
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) die("%s: failed to load the model", o.model.c_str());
    const double t_load = now_s() - t0;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd_inp(model);
    const int n_layer = llama_model_n_layer(model);
    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));

    // prompt -> T tokens (klcheck's tokenizer call: add_special, parse_special)
    int T = o.T > 0 ? o.T : (llama_model_n_params(model) > 8000000000ull ? 16 : 32);
    std::vector<int32_t> toks;
    {
        const int n = -llama_tokenize(vocab, o.prompt.c_str(), (int32_t) o.prompt.size(), nullptr, 0, true, true);
        toks.resize(std::max(n, 0));
        if (n <= 0 || llama_tokenize(vocab, o.prompt.c_str(), (int32_t) o.prompt.size(), toks.data(), n, true, true) != n) {
            die("tokenization of the prompt failed");
        }
        if ((int) toks.size() < T) die("the prompt has %zu tokens, -T %d needs more", toks.size(), T);
        toks.resize(T);
    }

    embd_table embd;
    if (o.input != "tokens") {
        embd.open_model(files);
        if (embd.n_embd != n_embd) die("token_embd.weight has %" PRId64 " columns, n_embd %d", embd.n_embd, n_embd);
    }

    observer obs;
    obs.dir = o.out;
    if (!o.nodes_re.empty()) {
        obs.use_re = true;
        try { obs.re = std::regex(o.nodes_re, std::regex::ECMAScript); }
        catch (const std::regex_error & e) { die("--nodes: bad regex \"%s\": %s", o.nodes_re.c_str(), e.what()); }
    }
    if (!o.mm_layers.empty()) obs.mm.parse(o.mm_layers);
    if (o.observe) obs.gguf = gguf_tensor_types(files);
    clear_run_dir(o.out);

    // context: klcheck's create_context with n_ctx_needed = T and every position an output
    llama_context_params cp = llama_context_default_params();
    const int n_ctx = std::max(256, (T + 255) / 256 * 256);
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) std::min(2048, n_ctx);
    cp.n_ubatch        = (uint32_t) std::min<int>(512, cp.n_batch);
    cp.n_seq_max       = 1;
    cp.n_outputs_max   = (uint32_t) std::min<int>(T, cp.n_batch);
    cp.n_threads       = o.threads;
    cp.n_threads_batch = o.threads;
    cp.flash_attn_type = o.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;   // the same in every run
    cp.type_k          = o.type_k;                         // -ctk q8_0 (SW-22)
    cp.type_v          = o.type_v;                         // -ctv q8_0 (a quantized V needs FA)
    cp.offload_kqv     = o.offload();                      // M6: false (-nkvo, attention on the host); M7: true
    cp.no_perf         = false;
    cp.cb_eval           = o.observe ? observe_cb : nullptr;
    cp.cb_eval_user_data = o.observe ? &obs : nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("failed to create the llama context");
    if (T > (int) cp.n_ubatch) die("-T %d exceeds n_ubatch %u (one prefill ubatch is required)", T, cp.n_ubatch);

    // one batch: T tokens at positions 0..T-1 of sequence 0, logits for every position
    llama_batch b = llama_batch_init(T, o.input == "tokens" ? 0 : n_embd, 1);
    b.n_tokens = T;
    for (int i = 0; i < T; ++i) {
        if (o.input == "tokens") {
            b.token[i] = toks[i];
        } else {
            float * dst = b.embd + (size_t) i * n_embd;
            embd.row(toks[i], dst);
            if (o.input == "embd+1ulp") {
                const uint64_t key = embd_nudge_key(o.seed, 0, (uint32_t) i);
                for (int j = 0; j < n_embd; ++j) dst[j] = embd_nudge(dst[j], key, (uint32_t) j);
            }
        }
        b.pos[i]       = i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    const std::string tag = "[" + o.run + (o.input == "tokens" ? "+tokens" : "") + (o.observe ? "" : ",no-observe") + "]";
    fprintf(stderr, "tsr-numcheck %s: %s, %d layers, n_vocab %d, T %d, input %s, model load %.2f s\n", tag.c_str(), desc,
            n_layer, n_vocab, T, o.input.c_str(), t_load);

    obs.active = true;
    const double t1 = now_s();
    const int ret = llama_decode(ctx, b);
    if (ret != 0) die("llama_decode failed (%d)", ret);
    llama_synchronize(ctx);
    const double t_decode = now_s() - t1;
    obs.active = false;
    if (obs.cur_t) die("internal: node %s was asked for but never delivered", obs.cur_t->name);

    std::vector<float> logits((size_t) T * n_vocab);
    uint64_t nonfinite = 0;
    for (int i = 0; i < T; ++i) {
        const float * lg = llama_get_logits_ith(ctx, i);
        if (!lg) die("no logits for position %d", i);
        memcpy(logits.data() + (size_t) i * n_vocab, lg, sizeof(float) * n_vocab);
        for (int k = 0; k < n_vocab; ++k) nonfinite += !std::isfinite(lg[k]);
    }
    write_file(o.out + "/logits.f32", logits.data(), logits.size() * sizeof(float));
    if (nonfinite) fprintf(stderr, "tsr-numcheck %s: WARNING: %" PRIu64 " non-finite logits\n", tag.c_str(), nonfinite);
    if (obs.n_type_mismatch) {
        fprintf(stderr, "tsr-numcheck %s: WARNING: %" PRId64 " weights have a ggml type different from the GGUF\n", tag.c_str(),
                obs.n_type_mismatch);
    }

    auto envs = [](const char * k) -> json { const char * v = getenv(k); return v ? json(v) : json(nullptr); };
    json m = json::object();
    m["format"]      = "tsr-numcheck";
    m["version"]     = 2;
    m["model"]       = model_abs;
    m["run"]         = o.run;
    m["device"]      = dev ? o.device : "none";
    m["input"]       = o.input;
    m["seed"]        = o.seed;
    m["n_tokens"]    = T;
    m["tokens"]      = toks;
    m["n_vocab"]     = n_vocab;
    m["logits_file"] = "logits.f32";
    m["versions"]    = nullptr;
    if (dev) {   // SW §6.9: what the dev run ran on, every kind with its numerics revision
        typedef bool (*versions_fn)(ggml_backend_dev_t, tsr_versions *);
        ggml_backend_reg_t reg = devs[0] ? ggml_backend_dev_backend_reg(devs[0]) : nullptr;
        versions_fn fn = reg ? (versions_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_versions") : nullptr;
        tsr_versions v;
        if (fn && fn(devs[0], &v)) {
            char b[32];
            json jv = json::object();
            snprintf(b, sizeof(b), "%u.%u", (v.core >> 8) & 0xFFu, v.core & 0xFFu);
            jv["core"] = b;
            jv["caps"] = v.caps;
            snprintf(b, sizeof(b), "%u.%u.%u", (v.basis >> 16) & 0xFFu, (v.basis >> 8) & 0xFFu, v.basis & 0xFFu);
            jv["basis"] = b;
            jv["abi"] = v.abi;
            snprintf(b, sizeof(b), "%016" PRIx64, v.image_id);
            jv["image"] = b;
            snprintf(b, sizeof(b), "0x%04x", v.kinds_sig);
            jv["kinds_sig"] = b;
            json kinds = json::object();
            for (uint32_t k = 0; k < 64; ++k)
                if (v.kind_rev[k]) kinds[tsr::opl::kind_name(k)] = v.kind_rev[k];
            jv["kinds"] = kinds;
            m["versions"] = jv;
        }
    }
    json info = json::object();
    info["model_desc"]      = desc;
    info["model_files"]     = files;
    info["n_params"]        = llama_model_n_params(model);
    info["n_layer"]         = n_layer;
    info["n_embd"]          = n_embd;
    info["token_embd_type"] = o.input != "tokens" ? json(ggml_type_name(embd.type)) : json(nullptr);
    info["token_row_scale"] = o.input != "tokens" ? json((double) embd.tok_scale) : json(nullptr);   // Gemma 3/4: sqrt(n_embd)
    info["prompt"]          = o.prompt;
    info["prompt_used"]     = detok(vocab, toks);
    info["n_ctx"]           = cp.n_ctx;
    info["n_batch"]         = cp.n_batch;
    info["n_ubatch"]        = cp.n_ubatch;
    info["n_threads"]       = o.threads;
    info["n_gpu_layers"]    = mp.n_gpu_layers;
    info["phase"]           = o.phase;
    info["flash_attn"]      = o.fa ? "on" : "off";
    info["type_k"]          = ggml_type_name(o.type_k);
    info["type_v"]          = ggml_type_name(o.type_v);
    info["offload_kqv"]     = o.offload();
    {
        const char * e = getenv("LLAMA_ATTN_ROT_DISABLE");
        info["attn_rot_disable"] = e != nullptr && atoi(e) != 0;
    }
    info["use_extra_bufts"] = false;
    info["observe"]         = o.observe;
    info["nodes_regex"]     = o.nodes_re.empty() ? json(nullptr) : json(o.nodes_re);
    info["mm_layers"]       = o.mm_layers.empty() ? json("all") : json(o.mm_layers);
    info["n_graph_nodes"]   = obs.n_seen;
    info["n_not_computed"]  = obs.n_not_computed;
    info["n_dumped"]        = obs.nodes.size();
    info["n_weight_mm"]     = obs.n_weight_mm;
    info["n_src1_dumped"]   = obs.n_src1;
    info["bytes_dumped"]    = obs.bytes;
    info["nonfinite_logits"] = nonfinite;
    info["t_load_s"]        = t_load;
    info["t_decode_s"]      = t_decode;
    info["TSR_DEVICE"]      = envs("TSR_DEVICE");
    info["TSR_KINDS"]       = envs("TSR_KINDS");      // m6, the caller's list (m7), or null: every kind
    info["LLAMA_ATTN_ROT_DISABLE"] = envs("LLAMA_ATTN_ROT_DISABLE");
    info["TSR_SIM_THREADS"] = envs("TSR_SIM_THREADS");
    info["TSR_SIM_TMAX"]    = envs("TSR_SIM_TMAX");
    info["TSR_SIM_DRAM_DIR"] = envs("TSR_SIM_DRAM_DIR");
    info["created_unix"]    = (uint64_t) time(nullptr);
    info["backend_field"]   = "ggml_backend_dev_name of the device owning the node's output buffer = the computing backend";
    m["info"]  = info;
    m["nodes"] = obs.nodes;
    const std::string js = m.dump(1);
    const std::string tmp = o.out + "/meta.json.tmp";
    write_file(tmp, js.data(), js.size());
    if (rename(tmp.c_str(), (o.out + "/meta.json").c_str()) != 0) die("rename %s: %s", tmp.c_str(), strerror(errno));

    fprintf(stderr, "tsr-numcheck %s: decode %.2f s, %zu nodes dumped (%" PRId64 " in the graph), %" PRId64
            " weight MUL_MATs (%" PRId64 " with src1), %.1f MiB -> %s\n", tag.c_str(), t_decode, obs.nodes.size(), obs.n_seen,
            obs.n_weight_mm, obs.n_src1, obs.bytes / 1048576.0, o.out.c_str());
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// tsr-numcheck selfcheck ROOT — consistency of the run directories (no model needed)

static int g_fail = 0;

static void check(bool ok, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("  [%s] ", ok ? " ok " : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (!ok) g_fail++;
}

struct run_dir {
    std::string        dir;
    json               meta;
    std::vector<float> logits;
    bool               present = false;
};

static bool load_run(const std::string & dir, run_dir & r) {
    r.dir = dir;
    std::vector<uint8_t> raw;
    if (!read_file(dir + "/meta.json", raw)) return false;
    try {
        r.meta = json::parse(raw.begin(), raw.end());
    } catch (const std::exception & e) {
        die("%s/meta.json: %s", dir.c_str(), e.what());
    }
    const int ver = r.meta.value("format", "") == "tsr-numcheck" ? r.meta.value("version", 0) : 0;
    if (ver != 1 && ver != 2) die("%s: not a tsr-numcheck version 1 / 2 run", dir.c_str());
    if (!read_file(dir + "/" + r.meta["logits_file"].get<std::string>(), raw)) die("%s: no logits file", dir.c_str());
    r.logits.resize(raw.size() / 4);
    memcpy(r.logits.data(), raw.data(), r.logits.size() * 4);
    r.present = true;
    return true;
}

static int64_t prod_ne(const json & ne) {
    int64_t n = 1;
    for (const auto & v : ne) n *= v.get<int64_t>();
    return n;
}

// false (and nothing read) unless both logits files hold exactly T x V floats of a's header
static bool logits_diff(const run_dir & a, const run_dir & b, double * maxdiff, int * ident_rows, int * top1) {
    const int64_t T = a.meta["n_tokens"], V = a.meta["n_vocab"];
    *maxdiff = 0;
    *ident_rows = 0;
    *top1 = 0;
    if (T <= 0 || V <= 0 || a.logits.size() != (size_t) (T * V) || b.logits.size() != (size_t) (T * V)) {
        *maxdiff = NAN;
        *ident_rows = -1;
        *top1 = -1;
        return false;
    }
    for (int64_t t = 0; t < T; ++t) {
        const float * x = a.logits.data() + t * V;
        const float * y = b.logits.data() + t * V;
        *ident_rows += memcmp(x, y, sizeof(float) * V) == 0;
        *top1 += std::max_element(x, x + V) - x == std::max_element(y, y + V) - y;
        for (int64_t k = 0; k < V; ++k) *maxdiff = std::max(*maxdiff, (double) std::fabs(x[k] - y[k]));
    }
    return true;
}

static int cmd_selfcheck(int argc, char ** argv) {
    if (argc != 1) die("usage: tsr-numcheck selfcheck ROOT   (ROOT/{cpu,floor,dev}[, dev-noobs, cpu-tok])");
    const std::string root = argv[0];
    run_dir cpu, flr, dev, noobs, tok;
    if (!load_run(root + "/cpu", cpu) || !load_run(root + "/floor", flr) || !load_run(root + "/dev", dev)) {
        die("%s: cpu, floor and dev run directories (with meta.json) are required", root.c_str());
    }
    load_run(root + "/dev-noobs", noobs);
    load_run(root + "/cpu-tok", tok);
    printf("tsr-numcheck selfcheck %s\n", root.c_str());

    // 1. run identities and common header fields
    check(cpu.meta["run"] == "cpu" && cpu.meta["input"] == "embd" && cpu.meta["device"] == "none", "cpu: run cpu, input embd, device none");
    check(flr.meta["run"] == "floor" && flr.meta["input"] == "embd+1ulp" && flr.meta["device"] == "none",
          "floor: run floor, input embd+1ulp, device none");
    check(dev.meta["run"] == "dev" && dev.meta["input"] == "embd" && dev.meta["device"] != "none", "dev: run dev, input embd, device %s",
          dev.meta["device"].get<std::string>().c_str());
    {   // SW §6.9: the dev run recorded what it ran on (CPU runs: null)
        const json & v = dev.meta.contains("versions") ? dev.meta["versions"] : json(nullptr);
        check(v.is_object() && v.contains("core") && v.contains("basis") && v.contains("kinds") && v["kinds"].contains("MUL_MAT") &&
                  cpu.meta.contains("versions") && cpu.meta["versions"].is_null(),
              "dev: versions recorded (%s)", v.dump().c_str());
    }
    std::vector<run_dir *> all = {&cpu, &flr, &dev};
    if (noobs.present) all.push_back(&noobs);
    if (tok.present) all.push_back(&tok);
    bool same_hdr = true;
    for (run_dir * r : all) {
        for (const char * k : {"model", "n_tokens", "tokens", "n_vocab"}) same_hdr = same_hdr && r->meta[k] == cpu.meta[k];
        same_hdr = same_hdr && r->logits.size() == (size_t) cpu.meta["n_tokens"].get<int64_t>() * cpu.meta["n_vocab"].get<int64_t>();
    }
    check(same_hdr, "all %zu runs: same model, tokens (T = %d), n_vocab (%d), logits files [T][V]", all.size(),
          cpu.meta["n_tokens"].get<int>(), cpu.meta["n_vocab"].get<int>());
    check(cpu.meta["seed"] == flr.meta["seed"], "floor seed %" PRIu64, flr.meta["seed"].get<uint64_t>());
    {   // the same settings in every run (SW-22): one comparison, one attention / KV configuration
        auto settings = [](const run_dir & r) {
            const json & i = r.meta.contains("info") ? r.meta["info"] : json::object();
            json v = json::object();
            for (const char * k : {"phase", "flash_attn", "type_k", "type_v", "offload_kqv", "attn_rot_disable"})
                v[k] = i.contains(k) ? i[k] : json(nullptr);
            return v;
        };
        const json ref = settings(cpu);
        std::string bad;
        for (run_dir * r : all)
            if (settings(*r) != ref) bad += (bad.empty() ? "" : ", ") + r->meta["run"].get<std::string>() + " " + settings(*r).dump();
        check(bad.empty(), "all %zu runs: the same settings %s%s", all.size(), ref.dump().c_str(),
              bad.empty() ? "" : ("; differ: " + bad).c_str());
    }
    for (run_dir * r : all) {
        bool finite = true;
        for (float v : r->logits) finite = finite && std::isfinite(v);
        check(finite, "%s: all logits finite", r->dir.c_str());
    }

    // 2. node lists of cpu / floor / dev agree index by index
    const json & nc = cpu.meta["nodes"];
    for (run_dir * r : {&flr, &dev}) {
        const json & nr = r->meta["nodes"];
        size_t bad = 0, first_bad = SIZE_MAX;
        for (size_t k = 0; k < std::min(nc.size(), nr.size()); ++k) {
            const json & a = nc[k], & b = nr[k];
            const bool same = a["i"] == b["i"] && a["name"] == b["name"] && a["op"] == b["op"] && a["ne"] == b["ne"] &&
                              a["weight"] == b["weight"] && a["weight_type"] == b["weight_type"] && a["src1_ne"] == b["src1_ne"];
            if (!same) { bad++; if (first_bad == SIZE_MAX) first_bad = k; }
        }
        check(nc.size() == nr.size() && bad == 0, "node list cpu vs %s: %zu vs %zu nodes, %zu differ (i, name, op, ne, weight, src1_ne)%s",
              r->meta["run"].get<std::string>().c_str(), nc.size(), nr.size(), bad,
              first_bad == SIZE_MAX ? "" : (", first at " + std::to_string(first_bad) + ": " + nc[first_bad]["name"].get<std::string>() +
                                            " vs " + nr[first_bad]["name"].get<std::string>()).c_str());
    }
    check(!nc.empty(), "cpu run dumped %zu nodes", nc.size());

    // 3. dump files present with the right sizes
    for (run_dir * r : {&cpu, &flr, &dev}) {
        size_t bad = 0, n = 0;
        for (const json & e : r->meta["nodes"]) {
            n++;
            if (file_size(r->dir + "/" + e["file"].get<std::string>()) != 4 * prod_ne(e["ne"])) bad++;
            if (!e["src1_file"].is_null()) {
                n++;
                if (file_size(r->dir + "/" + e["src1_file"].get<std::string>()) != 4 * prod_ne(e["src1_ne"])) bad++;
            }
        }
        check(bad == 0, "%s: %zu dump files, all present with 4 x prod(ne) bytes (%zu bad)", r->meta["run"].get<std::string>().c_str(), n, bad);
    }

    // 4. backends: cpu / floor entirely on the CPU; on dev every weight MUL_MAT on Tesseract
    for (run_dir * r : {&cpu, &flr}) {
        size_t not_cpu = 0;
        for (const json & e : r->meta["nodes"]) not_cpu += e["backend"] != "CPU";
        check(not_cpu == 0, "%s: every node computed by CPU (%zu are not)", r->meta["run"].get<std::string>().c_str(), not_cpu);
    }
    {
        size_t wmm = 0, wmm_tsr = 0, mm_other = 0, other_tsr = 0, cpu_nodes = 0;
        std::string first_bad;
        std::map<std::string, size_t> wtypes;
        for (const json & e : dev.meta["nodes"]) {
            const std::string be = e["backend"];
            const bool tsr = be.rfind("Tesseract", 0) == 0;
            if (!e["weight"].is_null()) {
                wmm++;
                wtypes[e["weight_type"].get<std::string>()]++;
                if (tsr) wmm_tsr++; else if (first_bad.empty()) first_bad = e["name"].get<std::string>() + " on " + be;
            } else {
                if (e["op"] == "MUL_MAT") mm_other++;
                if (tsr) other_tsr++;
            }
            cpu_nodes += be == "CPU";
        }
        std::string types;
        for (const auto & kv : wtypes) types += (types.empty() ? "" : ", ") + kv.first + " " + std::to_string(kv.second);
        check(wmm > 0 && wmm_tsr == wmm, "dev: %zu/%zu weight MUL_MATs computed by Tesseract (%s)%s", wmm_tsr, wmm, types.c_str(),
              first_bad.empty() ? "" : (", first not: " + first_bad).c_str());
        const bool kv_dev = dev.meta.contains("info") && dev.meta["info"].value("offload_kqv", false);
        printf("         dev: %zu other MUL_MATs (%s), %zu non-weight nodes on Tesseract, %zu nodes on CPU\n", mm_other,
               kv_dev ? "M7: attention and KV on the device" : "attention, KV on the host", other_tsr, cpu_nodes);
        size_t src1 = 0;
        for (const json & e : dev.meta["nodes"]) src1 += !e["src1_file"].is_null();
        printf("         dev: %zu weight MUL_MATs with src1 dumps\n", src1);
    }

    // 5. observation does not change device numerics: dev logits == dev-noobs logits bit for bit
    if (noobs.present) {
        check(noobs.meta["run"] == "dev" && noobs.meta["info"]["observe"] == false && noobs.meta["input"] == "embd",
              "dev-noobs: dev run, input embd, no observation");
        double md; int ident, top1;
        const bool sized = logits_diff(dev, noobs, &md, &ident, &top1);
        check(sized && dev.logits.size() == noobs.logits.size() && memcmp(dev.logits.data(), noobs.logits.data(), dev.logits.size() * 4) == 0,
              "dev logits == dev-noobs logits bit-exactly (%d/%d rows identical, max |d| %.3g)", ident, cpu.meta["n_tokens"].get<int>(), md);
    } else {
        printf("  [skip] dev-noobs absent: observation invariance not checked\n");
    }
    // 6. embd input path == token input path on the CPU (and observed == unobserved on the CPU)
    if (tok.present) {
        check(tok.meta["run"] == "cpu" && tok.meta["input"] == "tokens", "cpu-tok: cpu run, input tokens");
        double md; int ident, top1;
        const bool sized = logits_diff(cpu, tok, &md, &ident, &top1);
        check(sized && cpu.logits.size() == tok.logits.size() && memcmp(cpu.logits.data(), tok.logits.data(), cpu.logits.size() * 4) == 0,
              "cpu (embd, observed) logits == cpu-tok (tokens, %s) logits bit-exactly (%d/%d rows identical, max |d| %.3g)",
              tok.meta["info"]["observe"] == true ? "observed" : "not observed", ident, cpu.meta["n_tokens"].get<int>(), md);
    } else {
        printf("  [skip] cpu-tok absent: embd-path identity not checked\n");
    }
    // 7. the floor moved and the device is not the CPU (sanity), with the logit distances
    {
        double md_f, md_d;
        int id_f, id_d, t1_f, t1_d;
        const bool sized_f = logits_diff(cpu, flr, &md_f, &id_f, &t1_f);
        const bool sized_d = logits_diff(cpu, dev, &md_d, &id_d, &t1_d);
        const int T = cpu.meta["n_tokens"];
        if (!sized_f || !sized_d) printf("         logits of floor or dev do not hold T x V floats: not compared\n");
        check(sized_f && id_f < T, "floor logits differ from cpu: %d/%d rows bit-identical, max |dlogit| %.4g, top-1 %d/%d", id_f, T, md_f, t1_f, T);
        printf("         dev vs cpu: %d/%d rows bit-identical, max |dlogit| %.4g, top-1 %d/%d\n", id_d, T, md_d, t1_d, T);
        size_t same_f = 0, same_d = 0, n = std::min({nc.size(), flr.meta["nodes"].size(), dev.meta["nodes"].size()});
        std::vector<uint8_t> a, b;
        for (size_t k = 0; k < n; ++k) {
            const std::string f = nc[k]["file"];
            read_file(cpu.dir + "/" + f, a);
            read_file(flr.dir + "/" + f, b);
            same_f += a == b;
            read_file(dev.dir + "/" + f, b);
            same_d += a == b;
        }
        printf("         bit-identical node dumps vs cpu: floor %zu/%zu, dev %zu/%zu\n", same_f, n, same_d, n);
    }
    printf("tsr-numcheck selfcheck: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------
// tsr-numcheck all — every run as a child process (one model in memory at a time), then selfcheck

struct child_stat {
    int    rc = 0;
    double wall_s = 0;
    long   maxrss_kb = 0;
};

static child_stat run_child(const std::vector<std::string> & args) {
    std::string line;
    for (const std::string & a : args) line += (line.empty() ? "" : " ") + a;
    fprintf(stderr, "tsr-numcheck all: $ %s\n", line.c_str());
    fflush(stderr);
    fflush(stdout);
    std::vector<char *> av;
    for (const std::string & a : args) av.push_back(const_cast<char *>(a.c_str()));
    av.push_back(nullptr);
    child_stat cs;
    const double t0 = now_s();
    const pid_t pid = fork();
    if (pid < 0) die("fork: %s", strerror(errno));
    if (pid == 0) {
        execv("/proc/self/exe", av.data());
        fprintf(stderr, "tsr-numcheck: exec: %s\n", strerror(errno));
        _exit(127);
    }
    int st = 0;
    struct rusage ru;
    if (wait4(pid, &st, 0, &ru) < 0) die("wait4: %s", strerror(errno));
    cs.wall_s    = now_s() - t0;
    cs.maxrss_kb = ru.ru_maxrss;
    cs.rc        = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    return cs;
}

static int cmd_all(int argc, char ** argv, const char * self) {
    std::string root, checks = "cheap";
    std::vector<std::string> common, dev_only;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die("%s needs a value", a.c_str());
            return argv[++i];
        };
        if (a == "-o" || a == "--out")    root = val();
        else if (a == "--checks")         checks = val();
        else if (a == "--device")         { dev_only.push_back(a); dev_only.push_back(val()); }
        else if (a == "-m" || a == "--model" || a == "-T" || a == "--seed" || a == "--nodes" || a == "--mm-layers" ||
                 a == "--prompt" || a == "-t" || a == "--threads" || a == "--phase" || a == "--fa" || a == "--offload-kqv" ||
                 a == "--attn-rot" || a == "-ctk" || a == "--cache-type-k" || a == "-ctv" || a == "--cache-type-v") {
            common.push_back(a); common.push_back(val());   // the settings go to every run (SW-22)
        }
        else if (a == "-v")               common.push_back(a);
        else die("all: unknown option %s\nusage: tsr-numcheck all -m MODEL [-o ROOT] [--device sim] [-T N] [--seed S] [--nodes RE] "
                 "[--mm-layers L] [--prompt TEXT] [-t N] [--phase m6|m7] [--checks none|cheap|full]", a.c_str());
    }
    if (checks != "none" && checks != "cheap" && checks != "full") die("--checks: none, cheap or full");
    // without -o: a private temporary directory, removed afterwards (ctest)
    const bool tmp_root = root.empty();
    if (tmp_root) {
        const char * t = getenv("TMPDIR");
        std::string tmpl = std::string(t && *t ? t : "/tmp") + "/tsr-numcheck-XXXXXX";
        std::vector<char> b(tmpl.begin(), tmpl.end());
        b.push_back(0);
        if (!mkdtemp(b.data())) die("mkdtemp %s: %s", tmpl.c_str(), strerror(errno));
        root = b.data();
    }
    mkdir_p(root);

    struct job { std::string name; std::vector<std::string> args; };
    std::vector<job> jobs = {
        {"cpu",   {"--run", "cpu"}},
        {"floor", {"--run", "floor"}},
        {"dev",   {"--run", "dev"}},
    };
    if (checks != "none") jobs.push_back({"cpu-tok", {"--run", "cpu", "--input", "tokens", "--no-observe"}});
    if (checks == "full") jobs.push_back({"dev-noobs", {"--run", "dev", "--no-observe"}});
    // selfcheck reads ROOT/cpu-tok and ROOT/dev-noobs whenever they exist: remove runs of an earlier `all`
    // that this --checks level does not redo, so they are never checked against the new runs
    for (const char * opt : {"cpu-tok", "dev-noobs"}) {
        bool redo = false;
        for (const job & j : jobs) redo = redo || j.name == opt;
        const std::string d = root + "/" + opt;
        struct stat sb;
        if (redo || stat(d.c_str(), &sb) != 0) continue;
        if (stat((d + "/meta.json").c_str(), &sb) == 0) {
            clear_run_dir(d);
            rmdir((d + "/t").c_str());
            rmdir(d.c_str());
            fprintf(stderr, "tsr-numcheck all: removed the stale run %s (not redone with --checks %s)\n", d.c_str(), checks.c_str());
        }
    }

    json stats = json::object();
    for (job & j : jobs) {
        std::vector<std::string> a = {self, "run", "-o", root + "/" + j.name};
        a.insert(a.end(), j.args.begin(), j.args.end());
        a.insert(a.end(), common.begin(), common.end());
        if (j.name.rfind("dev", 0) == 0) a.insert(a.end(), dev_only.begin(), dev_only.end());
        const child_stat cs = run_child(a);
        fprintf(stderr, "tsr-numcheck all: %s: exit %d, wall %.1f s, peak RSS %.0f MiB\n", j.name.c_str(), cs.rc, cs.wall_s,
                cs.maxrss_kb / 1024.0);
        stats[j.name] = {{"exit", cs.rc}, {"wall_s", cs.wall_s}, {"maxrss_kb", cs.maxrss_kb}};
        const std::string js = stats.dump(1);
        write_file(root + "/runs.json", js.data(), js.size());
        if (cs.rc) die("run %s failed (exit %d)", j.name.c_str(), cs.rc);
    }
    const child_stat cs = run_child({self, "selfcheck", root});
    if (tmp_root) {
        nftw(root.c_str(), [](const char * p, const struct stat *, int, struct FTW *) { return remove(p); }, 16, FTW_DEPTH | FTW_PHYS);
    }
    return cs.rc;
}


// ---------------------------------------------------------------------------------------------
// tsr-numcheck selftest — read-back and parsing helpers (no model). The gather of views
// (permuted, transposed, strided, offset) and of F16 / BF16 / Q8_0 / I32 tensors is checked
// against ggml's own CPU CONT of the same view, an independent implementation of the element order.

static int cmd_selftest() {
    int fails = 0;
    auto st = [&](bool ok, const char * what) {
        if (!ok) { fprintf(stderr, "selftest FAIL: %s\n", what); fails++; }
    };
    setenv("TSR_DEVICE", "none", 1);
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) die("selftest: no CPU device");
    ggml_backend_t be = ggml_backend_dev_init(cpu, nullptr);
    ggml_init_params ip = { /*.mem_size =*/ 256 * ggml_tensor_overhead() + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr,
                            /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 4, 3, 2);
    ggml_tensor * h   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 3, 2);
    ggml_tensor * bf  = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 6, 5);
    ggml_tensor * q   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 3);
    ggml_tensor * i32 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 9);
    struct view_case { const char * what; ggml_tensor * v; ggml_tensor * c; };
    std::vector<view_case> vc;
    auto add = [&](const char * what, ggml_tensor * v) { vc.push_back({what, v, ggml_cont(ctx, v)}); };
    add("f32 permute(2,0,1,3)", ggml_permute(ctx, a, 2, 0, 1, 3));
    add("f32 permute(1,3,0,2)", ggml_permute(ctx, a, 1, 3, 0, 2));
    add("f32 transpose", ggml_transpose(ctx, ggml_view_2d(ctx, a, 5, 24, a->nb[1], 0)));
    add("f32 strided view_3d + offset", ggml_view_3d(ctx, a, 3, 2, 3, a->nb[1], a->nb[2], 2 * sizeof(float) + a->nb[1]));
    add("f16 permute(1,0,2,3)", ggml_permute(ctx, h, 1, 0, 2, 3));
    add("bf16 transpose", ggml_transpose(ctx, bf));
    ggml_tensor * qrows = ggml_view_2d(ctx, q, 64, 2, q->nb[1], q->nb[1]);   // rows 1..2 of a Q8_0 matrix
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (const view_case & c : vc) ggml_build_forward_expand(gf, c.c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(cpu));
    if (!buf) die("selftest: allocation failed");

    std::vector<float> av(ggml_nelements(a));
    for (size_t i = 0; i < av.size(); ++i) av[i] = (float) i * 0.5f - 7.0f;
    ggml_backend_tensor_set(a, av.data(), 0, ggml_nbytes(a));
    std::vector<ggml_fp16_t> hv(ggml_nelements(h));
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = ggml_fp32_to_fp16((float) i * 0.25f - 3.0f);
    ggml_backend_tensor_set(h, hv.data(), 0, ggml_nbytes(h));
    std::vector<ggml_bf16_t> bv(ggml_nelements(bf));
    for (size_t i = 0; i < bv.size(); ++i) bv[i] = ggml_fp32_to_bf16((float) i * 0.5f - 4.0f);
    ggml_backend_tensor_set(bf, bv.data(), 0, ggml_nbytes(bf));
    std::vector<float> qsrc(64 * 3);
    uint64_t rng = 7;
    for (float & x : qsrc) { rng = splitmix64(rng); x = (float) ((double) (rng >> 11) / 9007199254740992.0 * 4 - 2); }
    std::vector<uint8_t> qb(ggml_nbytes(q));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, qsrc.data(), qb.data(), 0, 3, 64, nullptr);
    ggml_backend_tensor_set(q, qb.data(), 0, qb.size());
    const int32_t iv[9] = {0, 1, -1, 7, -300, 16777216, -16777216, 42, 5};
    ggml_backend_tensor_set(i32, iv, 0, sizeof(iv));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) die("selftest: CPU graph compute failed");

    std::vector<uint8_t> raw;
    std::vector<float> got, want;
    for (const view_case & c : vc) {
        st(!ggml_is_contiguous(c.v) || strstr(c.what, "offset"), c.what);   // the cases are real gathers
        gather_f32(c.v, raw, got);
        gather_f32(c.c, raw, want);                                        // contiguous: plain conversion
        st(got.size() == (size_t) ggml_nelements(c.v) && got == want, c.what);
    }
    gather_f32(a, raw, got);
    st(got == av, "f32 contiguous");
    gather_f32(h, raw, got);
    bool ok = got.size() == hv.size();
    for (size_t i = 0; ok && i < hv.size(); ++i) ok = got[i] == (float) i * 0.25f - 3.0f;
    st(ok, "f16 -> f32");
    gather_f32(bf, raw, got);
    ok = got.size() == bv.size();
    for (size_t i = 0; ok && i < bv.size(); ++i) ok = got[i] == (float) i * 0.5f - 4.0f;
    st(ok, "bf16 -> f32");
    gather_f32(qrows, raw, got);
    want.assign(128, 0.0f);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(qb.data() + q->nb[1], want.data(), 128);
    st(got == want, "q8_0 rows 1..2 -> to_float");
    gather_f32(i32, raw, got);
    ok = got.size() == 9;
    for (int i = 0; ok && i < 9; ++i) ok = got[i] == (float) iv[i];
    st(ok, "i32 -> f32");
    st(backend_of(a) == "CPU" && backend_of(vc[0].v) == "CPU", "backend_of a CPU buffer is \"CPU\"");
    st(op_is_view(vc[0].v->op) && op_is_view(a->op) && !op_is_view(vc[0].c->op), "op_is_view");

    st(strip_sched_copy("Tesseract0#blk.0.attn_q.weight#0") == "blk.0.attn_q.weight", "strip copy 1");
    st(strip_sched_copy("CPU#attn_norm-0#1") == "attn_norm-0", "strip copy 2");
    st(strip_sched_copy("cache_k_l0 (view)") == "cache_k_l0 (view)", "strip copy 3");
    st(strip_sched_copy("a#b") == "a#b" && strip_sched_copy("x#y#z") == "x#y#z", "strip copy 4");
    st(weight_layer("blk.12.ffn_down.weight") == 12 && weight_layer("output.weight") == -1, "weight_layer");
    mm_select m;
    st(m.selected("blk.5.attn_q.weight") && m.selected("output.weight"), "mm default all");
    m.parse("0,2-4,output");
    st(m.selected("blk.0.x.weight") && !m.selected("blk.1.x.weight") && m.selected("blk.3.x.weight") &&
       !m.selected("blk.5.x.weight") && m.selected("output.weight"), "mm 0,2-4,output");
    m.parse("31");
    st(m.selected("blk.31.x.weight") && !m.selected("blk.3.x.weight") && !m.selected("output.weight"), "mm 31");
    m.parse("none");
    st(!m.selected("blk.0.x.weight") && !m.selected("output.weight"), "mm none");
    for (uint32_t j = 0; j < 4096; ++j) {
        const float x = (float) j * 0.01f - 20.0f;
        const float y = embd_nudge(x, embd_nudge_key(1, 0, j % 32), j);
        if (!(y == nextafterf(x, INFINITY) || y == nextafterf(x, -INFINITY))) { st(false, "embd_nudge one ulp"); break; }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    printf("tsr-numcheck selftest: %s (%d failures)\n", fails ? "FAIL" : "ok", fails);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------

static void usage() {
    fprintf(stderr,
        "tsr-numcheck — one-prompt node dump for numeric verification (docs/numcheck.md)\n"
        "  tsr-numcheck run -m MODEL -o RUN_DIR --run cpu|floor|dev [--device sim] [-T N] [--seed 1] [--phase m6|m7]\n"
        "                   [--nodes REGEX] [--mm-layers 0,1,31,output] [--no-observe] [--prompt TEXT] [-t N]\n"
        "  tsr-numcheck all -m MODEL [-o ROOT] [run options] [--checks none|cheap|full]   (no -o: temp dir, removed)\n"
        "                   ROOT/{cpu,floor,dev} (+ cpu-tok: cheap, + dev-noobs: full), then selfcheck\n"
        "  tsr-numcheck selfcheck ROOT          consistency of ROOT/{cpu,floor,dev[,dev-noobs,cpu-tok]}\n"
        "  tsr-numcheck selftest                read-back / parsing helpers, no model\n"
        "'tsr-numcheck run -h' lists the run options.\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string cmd = argv[1];
    if (cmd == "run")       return cmd_run(argc - 2, argv + 2);
    if (cmd == "all")       return cmd_all(argc - 2, argv + 2, argv[0]);
    if (cmd == "selfcheck") return cmd_selfcheck(argc - 2, argv + 2);
    if (cmd == "selftest")  return cmd_selftest();
    if (cmd == "-h" || cmd == "--help") { usage(); return 0; }
    usage();
    return 2;
}

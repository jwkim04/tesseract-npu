// klcheck — model-level KL check (spec "Verification strategy", "Model-level KL criterion").
// Usage: docs/klcheck.md.
//
// Method (after openTPU tools/validate.py, https://github.com/FeSens/openTPU):
//   1. device run   greedy-generate G tokens per prompt on the Tesseract device (all layers and the
//                   output head offloaded; KV cache and attention on the host in M6, on the device in
//                   M7) and record the full logits at the last `prompt-tail` prompt positions and at
//                   every generated position but the last;
//   2. CPU run      feed the same token sequence (prompt + device tokens) to the ggml CPU backend
//                   alone and record the logits at the same positions (same context per position);
//   3. floor        the CPU run again with its input embeddings nudged by one f32 ulp each
//                   (nextafterf in a seeded random direction, see embd_nudge()); `klcheck all`
//                   makes --floor-seeds N such runs (default 3) with seeds S..S+N-1 (--seed S,
//                   default 1), since one seed's floor can differ a lot from another's;
//   4. compare      KL(CPU || device) and KL(CPU || CPU-perturbed) per position, top-1 agreement.
//                   PASS: mean KL(CPU || device) <= kl-ratio x mean floor KL (3) and top-1 >= 80%.
//                   With several floor runs the floor KL is the mean over the runs of each run's
//                   mean KL; the report gives the ratio against each run and against that mean.
// Every run is its own process that writes a logits file (f32, small header), so the device model
// and the CPU model are never loaded at the same time (32B on a 14 GB machine). `klcheck all` runs
// the three phases as child processes and then compares.
//
// File format (little-endian; klc_header, then n_seq records):
//   record = klc_seq, int32 tokens[n_prompt + n_gen], f32 logits[n_rows][n_vocab]
//   row r holds the logits computed at token index row0 + r, i.e. the prediction of token
//   row0 + r + 1; row0 + n_rows = n_prompt + n_gen - 1. Generate mode: row0 = n_prompt - tail.
// The header's n_seq counts complete records only and is rewritten after each one (--resume).
// Settings (SW-22, docs/design/06-software-verification.md §9.5, §10.4): every phase runs with
// flash attention on, a q8_0 K and V cache and LLAMA_ATTN_ROT_DISABLE=1 (set by the tool);
// offload_kqv is false in M6 (--phase m6, the default) and true in M7 (--phase m7); all of it is
// recorded in the file header (version 2), and compare warns when two files differ. --phase m6 also
// sets TSR_KINDS=m6 (the device takes MUL_MAT only); --phase m7 keeps the caller's TSR_KINDS (unset:
// every kind of the booted image; a kind list such as all,-RMS_NORM, sw/ggml-tesseract/README.md).
// The header has no room for it; `klcheck all` records it in report.json ("tsr_kinds"). A device
// run records the versions it ran with (SW §6.9: the Tesseract Core capability, the Tesseract Basis
// version and the kind table signature of the booted image) in the header; compare prints them and
// report.json carries them ("versions", null when not recorded).

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "tsr_transport.h"   // tsr_versions (the backend's hook "ggml_backend_tesseract_versions")

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// ---------------------------------------------------------------------------------------------
// utilities

[[noreturn]] static void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "klcheck: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string basename_of(const std::string & p) {
    const size_t k = p.find_last_of('/');
    return k == std::string::npos ? p : p.substr(k + 1);
}

static void copy_str(char * dst, size_t n, const std::string & s) {
    memset(dst, 0, n);
    memcpy(dst, s.data(), std::min(n - 1, s.size()));
}

static std::string fixed_str(const char * s, size_t n) {
    return std::string(s, strnlen(s, n));
}

static long long parse_int(const char * opt, const char * v, long long lo, long long hi) {
    char * end = nullptr;
    errno = 0;
    const long long r = strtoll(v, &end, 0);
    if (errno || end == v || *end || r < lo || r > hi) {
        die("%s: \"%s\" is not an integer in [%lld, %lld]", opt, v, lo, hi);
    }
    return r;
}

static uint64_t parse_u64(const char * opt, const char * v) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long r = strtoull(v, &end, 0);
    if (errno || end == v || *end || v[0] == '-') die("%s: \"%s\" is not an unsigned integer", opt, v);
    return (uint64_t) r;
}

static double parse_double(const char * opt, const char * v) {
    char * end = nullptr;
    errno = 0;
    const double r = strtod(v, &end);
    if (errno || end == v || *end || !std::isfinite(r) || r < 0) die("%s: \"%s\" is not a number >= 0", opt, v);
    return r;
}

static int parse_on_off(const char * opt, const std::string & v) {
    if (v == "on" || v == "1") return 1;
    if (v == "off" || v == "0") return 0;
    die("%s: on or off", opt);
}

static int parse_phase(const std::string & v) {
    if (v == "m6" || v == "M6") return 6;           // KLC_PHASE_M6
    if (v == "m7" || v == "M7") return 7;           // KLC_PHASE_M7
    die("--phase: m6 or m7");
}

static bool file_exists(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

static void mkdir_p(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) {
                die("mkdir %s: %s", cur.c_str(), strerror(errno));
            }
        }
        if (i < path.size()) cur += path[i];
    }
}

static int default_threads() {
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n >= 2 ? (int) (n / 2) : 1;   // llama.cpp's default: physical cores (2-way SMT assumed)
}

// ---------------------------------------------------------------------------------------------
// the 1-ulp input perturbation (floor)
//
// Element j of the input embedding of token index `pos` in sequence `seq` becomes
// nextafterf(x, +inf) or nextafterf(x, -inf), the direction being the top bit of a splitmix64 hash
// of (seed, seq, pos, j). Every element moves by exactly one ulp (0 becomes the smallest subnormal
// of either sign); the pattern does not depend on batching, threads or resume.

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
// logits file

enum { KLC_MODE_GENERATE = 0, KLC_MODE_REPLAY = 1 };
enum { KLC_REPLAY_BATCH = 0, KLC_REPLAY_STEP = 1 };
enum { KLC_INPUT_TOKENS = 0, KLC_INPUT_EMBD = 1, KLC_INPUT_EMBD_ULP = 2 };
enum { KLC_FLAG_REPACK = 1u << 0, KLC_FLAG_CHAT = 1u << 1 };

static const char KLC_MAGIC[8] = {'T', 'S', 'R', 'K', 'L', 'C', '0', '1'};
// Version 2 (P3.3, 2026-10-30) records the baseline settings (SW-22, SW §9.5, §10.4) in five
// words of the former reserved area: K / V cache types, offload_kqv, LLAMA_ATTN_ROT_DISABLE and the
// phase. Version 1 files (FA-off settings: f16 KV on the host, no rotation setting) are still read; their
// fields read as KLC_PHASE_V1 with f16 / f16.
static const uint32_t KLC_VERSION = 2;
enum { KLC_PHASE_V1 = 0, KLC_PHASE_M6 = 6, KLC_PHASE_M7 = 7 };
// The SWA layers' KV cache (W3b, one more word of the reserved area, version unchanged): 0 = not
// recorded (files written before; klcheck then used llama_context_default_params' full-size SWA
// cache), 1 = full-size (--swa-full on), 2 = the SWA ring of pad(n_swa + n_ubatch, 256) cells
// (--swa-full off, the default: llama-cli's and llama-server's default, common.h swa_full = false;
// the M7 gate's long Gemma 4 prompts must wrap it, AT §13.6). Models without SWA layers ignore it.
enum { KLC_SWA_UNRECORDED = 0, KLC_SWA_FULL = 1, KLC_SWA_RING = 2 };

struct klc_header {
    char     magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t seq_bytes;
    uint32_t n_vocab;
    uint32_t n_seq;            // complete records in the file
    uint32_t n_seq_planned;
    uint32_t n_gen;            // G (generate mode; copied from the tokens file in replay mode)
    uint32_t prompt_tail;
    uint32_t mode;             // KLC_MODE_*
    uint32_t replay;           // KLC_REPLAY_* (generate mode decodes step by step)
    uint32_t input;            // KLC_INPUT_*
    uint32_t flash_attn;
    uint64_t perturb_seed;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_threads;
    uint32_t flags;            // KLC_FLAG_*
    uint64_t model_n_params;
    uint64_t model_size;
    double   t_load_s;
    double   t_prefill_s;      // sum over records
    double   t_gen_s;
    uint64_t n_prefill_tok;    // tokens decoded in multi-token batches
    uint64_t n_gen_tok;        // single-token decodes
    uint64_t created_unix;
    char     device[16];       // "cpu", "sim", "mock", "vkernel", "xrt"
    char     model_desc[128];
    char     model_file[128];
    char     tokens_from[64];  // replay: basename of the tokens file
    // version 2 (the baseline settings, SW-22); a version 1 file reads as f16 / f16, 0, 0, KLC_PHASE_V1
    int32_t  type_k;           // ggml_type of the K cache (GGML_TYPE_Q8_0 in version 2)
    int32_t  type_v;           // ggml_type of the V cache (GGML_TYPE_Q8_0 in version 2)
    uint32_t offload_kqv;      // 1: KV cache and attention on the device (M7); 0: on the host (M6, or a version 1 file)
    uint32_t attn_rot_disable; // 1: the run had LLAMA_ATTN_ROT_DISABLE=1 (llama.cpp's Hadamard KV rotation off)
    uint32_t phase;            // KLC_PHASE_M6 / _M7: which phase's settings; KLC_PHASE_V1 for a version 1 file
    uint32_t swa_cache;        // KLC_SWA_* (0 in files written before the field)
    // The versions a device run used (SW §6.9; the last reserved bytes, version unchanged; 0 = not
    // recorded: a CPU run or a file written before the fields)
    uint32_t basis;            // Tesseract Basis version of the booted image (major << 16 | minor << 8 | patch)
    uint16_t core;             // Tesseract Core capability (VERSION[15:0]: major << 8 | minor)
    uint16_t kinds_sig;        // kind table signature of the booted image (HELLO r4[31:16])
};
static_assert(sizeof(klc_header) == 512, "klc_header layout");

struct klc_seq {
    char     magic[4];         // "KSEQ"
    uint32_t index;            // sequence (prompt) index
    uint32_t n_prompt;
    uint32_t n_gen;
    uint32_t row0;
    uint32_t n_rows;
    double   t_prefill_s;
    double   t_gen_s;
    uint32_t n_eog;            // end-of-generation tokens among the generated ones (information)
    uint32_t reserved;
};
static_assert(sizeof(klc_seq) == 48, "klc_seq layout");

static const char KLC_SEQ_MAGIC[4] = {'K', 'S', 'E', 'Q'};

static uint64_t klc_payload_bytes(const klc_seq & s, uint32_t n_vocab) {
    return (uint64_t) (s.n_prompt + s.n_gen) * 4 + (uint64_t) s.n_rows * n_vocab * 4;
}

static const char * mode_name(uint32_t m) { return m == KLC_MODE_GENERATE ? "generate" : "replay"; }
static const char * replay_name(uint32_t r) { return r == KLC_REPLAY_BATCH ? "batch" : "step"; }
static const char * input_name(uint32_t i) {
    return i == KLC_INPUT_TOKENS ? "tokens" : i == KLC_INPUT_EMBD ? "embd" : "embd+1ulp";
}
static const char * type_name_or(int32_t t) {
    return t >= 0 && t < GGML_TYPE_COUNT && ggml_type_name((ggml_type) t) ? ggml_type_name((ggml_type) t) : "?";
}
static const char * phase_name(uint32_t p) { return p == KLC_PHASE_M6 ? "M6" : p == KLC_PHASE_M7 ? "M7" : p == KLC_PHASE_V1 ? "v1" : "?"; }
// The settings a file was written with (SW-22), one line.
static std::string settings_str(const klc_header & h) {
    char b[320];
    snprintf(b, sizeof(b), "%s settings: flash attn %s, KV %s/%s on the %s (offload_kqv %u), KV rotation %s%s",
             phase_name(h.phase), h.flash_attn ? "on" : "off", type_name_or(h.type_k), type_name_or(h.type_v),
             h.offload_kqv ? "device" : "host", h.offload_kqv, h.attn_rot_disable ? "off (LLAMA_ATTN_ROT_DISABLE=1)" : "not disabled",
             h.swa_cache == KLC_SWA_RING ? ", SWA cache ring" : h.swa_cache == KLC_SWA_FULL ? ", SWA cache full-size" : "");
    return b;
}
// a file without the field ran with the full-size SWA cache
static uint32_t swa_cache_of(const klc_header & h) { return h.swa_cache == KLC_SWA_UNRECORDED ? (uint32_t) KLC_SWA_FULL : h.swa_cache; }
// The versions a device run recorded (SW §6.9), one phrase: "Tesseract Core 1.0, Tesseract Basis 0.1.0
// (kind table 0xfa6f)", or "versions not recorded".
static std::string versions_str(const klc_header & h) {
    if (h.basis == 0 && h.core == 0) return "versions not recorded";
    char b[128];
    snprintf(b, sizeof(b), "Tesseract Core %u.%u, Tesseract Basis %u.%u.%u (kind table 0x%04x)", (h.core >> 8) & 0xFFu, h.core & 0xFFu,
             (h.basis >> 16) & 0xFFu, (h.basis >> 8) & 0xFFu, h.basis & 0xFFu, h.kinds_sig);
    return b;
}
// The TSR_KINDS a device run of the phase uses (cmd_run): "m6" for --phase m6, else the caller's
// value, or "unset" without one (the device takes every kind of the booted image; an empty value
// is the same as none).
static std::string tsr_kinds_of(int phase) {
    if (phase == KLC_PHASE_M6) return "m6";
    const char * e = getenv("TSR_KINDS");
    return e && *e ? e : "unset";
}
// The device kinds a run accepts (SW §9.2 TSR_DEVICE values with a device, and cpu).
static bool device_kind_ok(const std::string & d) {
    return d == "cpu" || d == "sim" || d == "mock" || d == "vkernel" || d == "xrt";
}
static ggml_type parse_cache_type(const char * opt, const std::string & v) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && v == n && ggml_get_type_traits((ggml_type) t)->blck_size > 0) return (ggml_type) t;
    }
    die("%s: unknown ggml type \"%s\" (e.g. q8_0, f16, f32)", opt, v.c_str());
}

static bool klc_read_header(FILE * f, const std::string & path, klc_header * h, std::string * err) {
    if (fseeko(f, 0, SEEK_SET) != 0 || fread(h, sizeof(*h), 1, f) != 1) {
        *err = path + ": cannot read the header";
        return false;
    }
    if (memcmp(h->magic, KLC_MAGIC, 8) != 0) {
        *err = path + ": not a klcheck logits file";
        return false;
    }
    if ((h->version != 1 && h->version != KLC_VERSION) || h->header_bytes != sizeof(klc_header) ||
        h->seq_bytes != sizeof(klc_seq)) {
        *err = path + ": unsupported klcheck file version";
        return false;
    }
    if (h->version == 1) {      // FA-off settings (docs/klcheck.md "Settings fixed in every phase")
        h->type_k = h->type_v = GGML_TYPE_F16;
        h->offload_kqv = 0;
        h->attn_rot_disable = 0;
        h->phase = KLC_PHASE_V1;
    }
    if (h->n_vocab == 0 || h->n_vocab > (1u << 24)) {
        *err = path + ": bad n_vocab";
        return false;
    }
    return true;
}

static bool klc_seq_valid(const klc_seq & s) {
    return memcmp(s.magic, KLC_SEQ_MAGIC, 4) == 0 && s.n_prompt >= 1 && s.n_gen >= 1 &&
           s.row0 < s.n_prompt && (uint64_t) s.row0 + s.n_rows == (uint64_t) s.n_prompt + s.n_gen - 1;
}

struct klc_reader {
    std::string          path;
    FILE *               f = nullptr;
    klc_header           h{};
    klc_seq              seq{};
    std::vector<int32_t> toks;
    uint32_t             rows_left = 0;
    uint32_t             n_read    = 0;   // records started

    ~klc_reader() { if (f) fclose(f); }

    void open(const std::string & p) {
        path = p;
        f    = fopen(p.c_str(), "rb");
        if (!f) die("%s: %s", p.c_str(), strerror(errno));
        std::string err;
        if (!klc_read_header(f, p, &h, &err)) die("%s", err.c_str());
        if (fseeko(f, sizeof(klc_header), SEEK_SET) != 0) die("%s: seek failed", p.c_str());
    }

    // Moves to the next record (skipping unread rows). False after the last complete record.
    bool next() {
        if (rows_left && fseeko(f, (off_t) rows_left * h.n_vocab * 4, SEEK_CUR) != 0) die("%s: seek failed", path.c_str());
        rows_left = 0;
        if (n_read >= h.n_seq) return false;
        if (fread(&seq, sizeof(seq), 1, f) != 1 || !klc_seq_valid(seq)) {
            die("%s: record %u is truncated or corrupt", path.c_str(), n_read);
        }
        toks.resize(seq.n_prompt + seq.n_gen);
        if (fread(toks.data(), 4, toks.size(), f) != toks.size()) die("%s: record %u is truncated", path.c_str(), n_read);
        rows_left = seq.n_rows;
        n_read++;
        return true;
    }

    void read_row(float * dst) {
        if (!rows_left) die("%s: internal: no row left", path.c_str());
        if (fread(dst, 4, h.n_vocab, f) != h.n_vocab) die("%s: record %u is truncated", path.c_str(), n_read - 1);
        rows_left--;
    }
};

struct klc_writer {
    std::string path;
    FILE *      f = nullptr;
    klc_header  h{};

    ~klc_writer() { if (f) fclose(f); }

    void create(const std::string & p, const klc_header & hdr) {
        path = p;
        h    = hdr;
        f    = fopen(p.c_str(), "w+b");
        if (!f) die("%s: %s", p.c_str(), strerror(errno));
        flush_header();
    }

    // Opens an existing file, keeps its first header.n_seq complete records, drops anything after
    // them (a record cut short by a crash). Returns the number of records kept.
    uint32_t resume(const std::string & p) {
        path = p;
        f    = fopen(p.c_str(), "r+b");
        if (!f) die("%s: %s", p.c_str(), strerror(errno));
        std::string err;
        if (!klc_read_header(f, p, &h, &err)) die("%s", err.c_str());
        off_t off = sizeof(klc_header);
        for (uint32_t i = 0; i < h.n_seq; ++i) {
            klc_seq s;
            if (fseeko(f, off, SEEK_SET) != 0 || fread(&s, sizeof(s), 1, f) != 1 || !klc_seq_valid(s)) {
                die("%s: record %u (of %u in the header) is corrupt", p.c_str(), i, h.n_seq);
            }
            off += sizeof(s) + (off_t) klc_payload_bytes(s, h.n_vocab);
        }
        struct stat st;
        if (fstat(fileno(f), &st) != 0) die("%s: %s", p.c_str(), strerror(errno));
        if (st.st_size < off) die("%s: shorter than its %u records", p.c_str(), h.n_seq);
        if (st.st_size > off && ftruncate(fileno(f), off) != 0) die("%s: truncate: %s", p.c_str(), strerror(errno));
        if (fseeko(f, off, SEEK_SET) != 0) die("%s: seek failed", p.c_str());
        return h.n_seq;
    }

    void flush_header() {
        const off_t end = ftello(f);
        if (fseeko(f, 0, SEEK_SET) != 0 || fwrite(&h, sizeof(h), 1, f) != 1) die("%s: write failed", path.c_str());
        if (end > (off_t) sizeof(h) && fseeko(f, end, SEEK_SET) != 0) die("%s: seek failed", path.c_str());
        if (fflush(f) != 0) die("%s: write failed: %s", path.c_str(), strerror(errno));
    }

    void write_seq(const klc_seq & s, const std::vector<int32_t> & toks, const float * rows) {
        if (!klc_seq_valid(s) || toks.size() != s.n_prompt + s.n_gen) die("internal: bad record");
        if (fseeko(f, 0, SEEK_END) != 0) die("%s: seek failed", path.c_str());
        if (fwrite(&s, sizeof(s), 1, f) != 1 || fwrite(toks.data(), 4, toks.size(), f) != toks.size() ||
            fwrite(rows, 4, (size_t) s.n_rows * h.n_vocab, f) != (size_t) s.n_rows * h.n_vocab) {
            die("%s: write failed: %s", path.c_str(), strerror(errno));
        }
        if (fflush(f) != 0) die("%s: write failed: %s", path.c_str(), strerror(errno));
        h.n_seq++;
        h.t_prefill_s += s.t_prefill_s;
        h.t_gen_s += s.t_gen_s;
        flush_header();
    }
};

// ---------------------------------------------------------------------------------------------
// prompts

static const char * const k_builtin_prompts[] = {
    "The history of the printing press begins",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain why the sky is blue in simple terms.",
    "대한민국의 수도는 서울이다. 서울은",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA:",
    "Here is a list of five common kitchen herbs:\n1.",
    "In a distant future, humanity discovered",
    "The following JSON describes a book:\n{\"title\": \"",
};

// One prompt per line; "\n", "\t" and "\\" are unescaped; empty lines and lines starting with '#'
// are skipped.
static std::vector<std::string> load_prompts(const std::string & file) {
    std::vector<std::string> out;
    if (file.empty()) {
        for (const char * p : k_builtin_prompts) out.emplace_back(p);
        return out;
    }
    FILE * f = fopen(file.c_str(), "rb");
    if (!f) die("%s: %s", file.c_str(), strerror(errno));
    std::string line;
    int c;
    auto flush = [&]() {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty() && line[0] != '#') {
            std::string s;
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] == '\\' && i + 1 < line.size()) {
                    const char e = line[++i];
                    s += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                } else {
                    s += line[i];
                }
            }
            out.push_back(s);
        }
        line.clear();
    };
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') flush(); else line += (char) c;
    }
    flush();
    fclose(f);
    if (out.empty()) die("%s: no prompts", file.c_str());
    return out;
}

// ---------------------------------------------------------------------------------------------
// token_embd rows straight from the GGUF (all splits), dequantized with ggml's to_float — the
// function the CPU backend's GET_ROWS uses, so `--input embd` reproduces the token path exactly.
// Rows are scaled by the token-row scale the model graph gives token rows only: llama.cpp's
// build_inp_embd(tok_embd, tok_scale) multiplies by `inp_scale_rows`, which is tok_scale for token
// rows and 1 for embd rows (src/llama-graph.cpp:2766-2783, :104-112). Gemma 3 / 3n / 4 and the
// Gemma embedding models pass sqrtf(n_embd) (src/models/gemma4.cpp:171); without the same factor
// here the embd and floor runs would feed unscaled rows (W3b: the Gemma 4 floor was 8 nats, top-1
// 0 %). The product is one f32 multiply, as ggml's MUL, so the embd path stays bit-identical.
// Gemma and Gemma 2 scale after the select, for both paths alike: no factor here.

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
    return sqrtf((float) n_embd);                   // the graph's sqrtf(n_embd) of an int64_t n_embd
}

struct embd_table {
    std::string         file;           // the GGUF (split) that holds token_embd.weight
    int                 fd       = -1;
    uint64_t            data_off = 0;   // file offset of row 0
    ggml_type           type     = GGML_TYPE_F32;
    int64_t             n_embd   = 0;
    int64_t             n_rows   = 0;
    size_t              row_size = 0;
    ggml_to_float_t     to_float = nullptr;
    float               tok_scale = 1.0f;   // token_row_scale() of the model (first split's metadata)
    std::vector<uint8_t> raw;
    std::unordered_map<int32_t, std::vector<float>> cache;

    ~embd_table() { if (fd >= 0) close(fd); }

    bool try_file(const std::string & path, int * split_count) {
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(path.c_str(), gp);
        if (!g) die("%s: cannot read GGUF", path.c_str());
        const int64_t kc = gguf_find_key(g, "split.count");
        if (split_count) {
            *split_count = 1;
            if (kc >= 0) {
                switch (gguf_get_kv_type(g, kc)) {
                    case GGUF_TYPE_UINT16: *split_count = gguf_get_val_u16(g, kc); break;
                    case GGUF_TYPE_UINT32: *split_count = (int) gguf_get_val_u32(g, kc); break;
                    case GGUF_TYPE_INT32:  *split_count = gguf_get_val_i32(g, kc); break;
                    default: break;
                }
            }
            tok_scale = token_row_scale(g);         // the model's metadata is in the first split
        }
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
        return t >= 0;
    }

    void open_model(const std::string & model_path) {
        int n_split = 1;
        if (try_file(model_path, &n_split)) return;
        char prefix[4096];
        if (n_split > 1 && llama_split_prefix(prefix, sizeof(prefix), model_path.c_str(), 0, n_split) > 0) {
            for (int i = 1; i < n_split; ++i) {
                char p[4096 + 32];
                llama_split_path(p, sizeof(p), prefix, i, n_split);
                if (try_file(p, nullptr)) return;
            }
        }
        die("%s: token_embd.weight not found", model_path.c_str());
    }

    const float * row(int32_t tok) {
        auto it = cache.find(tok);
        if (it != cache.end()) return it->second.data();
        if (tok < 0 || tok >= n_rows) die("token %d outside token_embd (%" PRId64 " rows)", tok, n_rows);
        raw.resize(row_size);
        const off_t off = (off_t) (data_off + (uint64_t) tok * row_size);
        if (pread(fd, raw.data(), row_size, off) != (ssize_t) row_size) die("token_embd: short read");
        std::vector<float> v((size_t) n_embd);
        if (to_float) to_float(raw.data(), v.data(), n_embd);
        else memcpy(v.data(), raw.data(), sizeof(float) * n_embd);
        if (tok_scale != 1.0f) {
            for (float & x : v) x *= tok_scale;
        }
        return cache.emplace(tok, std::move(v)).first->second.data();
    }
};

// ---------------------------------------------------------------------------------------------
// llama session

static int g_verbose = 0;

static void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    static ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (g_verbose || last == GGML_LOG_LEVEL_ERROR || last == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct run_opts {
    std::string model, device = "sim", out, tokens, prompts_file;
    int         n_gen = 64, n_prompts = -1, prompt_tail = 1;
    bool        chat = false, perturb = false, resume = false, repack = false;
    uint64_t    seed = 0;
    int         input = KLC_INPUT_TOKENS;
    int         replay = -1;                      // -1: step on a device (identity), batch on the CPU
    int         threads = default_threads(), n_ctx = 0, n_batch = 2048, n_ubatch = 512;
    // baseline settings (SW-22, SW §10.4): FA on, q8_0 / q8_0 KV, LLAMA_ATTN_ROT_DISABLE=1 in
    // every phase; KV cache and attention on the host in M6 (offload_kqv false), on the device in M7.
    int         fa = 1;
    int         phase = KLC_PHASE_M6;
    int         offload_kqv = -1;                 // -1: the phase's value; 0 / 1: --offload-kqv
    ggml_type   type_k = GGML_TYPE_Q8_0, type_v = GGML_TYPE_Q8_0;
    int         attn_rot_disable = 1;             // --attn-rot on: 0 (experiments only; recorded)
    int         swa_full = 0;                     // --swa-full on: llama_context_default_params' full-size SWA cache
    bool        offload() const { return offload_kqv >= 0 ? offload_kqv != 0 : phase == KLC_PHASE_M7; }
};

struct session {
    llama_model *        model = nullptr;
    llama_context *      ctx   = nullptr;
    const llama_vocab *  vocab = nullptr;
    int                  n_vocab = 0, n_embd = 0, n_batch = 0;
    int                  input = KLC_INPUT_TOKENS;
    uint64_t             seed  = 0;
    embd_table           embd;
    llama_batch          batch{};
    bool                 have_batch = false;
    bool                 mrope = false;        // M-RoPE / IMROPE model: an embedding batch carries 4 position streams
    uint64_t             n_prefill_tok = 0, n_gen_tok = 0;

    ~session() {
        if (have_batch) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

static ggml_backend_dev_t find_tesseract_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "Tesseract", 9) == 0) return d;
    }
    return nullptr;
}

static void load_model(session & s, const run_opts & o, double * t_load) {
    const double t0 = now_s();
    llama_model_params mp = llama_model_default_params();
    static ggml_backend_dev_t devs[2] = {nullptr, nullptr};
    if (o.device == "cpu") {
        mp.n_gpu_layers = 0;                    // devices = {} : nothing is offloaded
    } else {
        devs[0] = find_tesseract_device();
        if (!devs[0]) die("no Tesseract device registered (TSR_DEVICE=%s)", getenv("TSR_DEVICE"));
        mp.n_gpu_layers = 999;                  // -ngl 99: every layer and the output head
    }
    mp.devices         = devs;
    mp.use_extra_bufts = o.repack;
    s.model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!s.model) die("%s: failed to load the model", o.model.c_str());
    s.vocab   = llama_model_get_vocab(s.model);
    s.n_vocab = llama_vocab_n_tokens(s.vocab);
    s.n_embd  = llama_model_n_embd_inp(s.model);
    *t_load   = now_s() - t0;
}

static void create_context(session & s, const run_opts & o, int n_ctx_needed, int n_outputs_max) {
    llama_context_params cp = llama_context_default_params();
    const int n_ctx = o.n_ctx > 0 ? o.n_ctx : std::max(256, (n_ctx_needed + 255) / 256 * 256);
    if (n_ctx < n_ctx_needed) die("-c %d is smaller than the longest sequence (%d)", n_ctx, n_ctx_needed);
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) std::min(o.n_batch, n_ctx);
    cp.n_ubatch        = (uint32_t) std::min<int>(o.n_ubatch, cp.n_batch);
    cp.n_seq_max       = 1;
    cp.n_outputs_max   = (uint32_t) std::min<int>(std::max(1, n_outputs_max), cp.n_batch);
    cp.n_threads       = o.threads;
    cp.n_threads_batch = o.threads;
    cp.flash_attn_type = o.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k          = o.type_k;           // -ctk q8_0 (SW-22)
    cp.type_v          = o.type_v;           // -ctv q8_0 (needs FA: src/llama-context.cpp refuses a quantized V without it)
    cp.offload_kqv     = o.offload();        // M6: false (-nkvo, attention on the host); M7: true
    cp.swa_full        = o.swa_full != 0;    // default off: the SWA ring, as llama-cli / llama-server (KLC_SWA_*)
    cp.no_perf         = false;
    s.ctx = llama_init_from_model(s.model, cp);
    if (!s.ctx) die("failed to create the llama context");
    s.n_batch = (int) cp.n_batch;
    s.batch   = llama_batch_init(s.n_batch, s.input == KLC_INPUT_TOKENS ? 0 : s.n_embd, 1);
    s.have_batch = true;
    const enum llama_rope_type rt = llama_model_rope_type(s.model);
    s.mrope = rt == LLAMA_ROPE_TYPE_MROPE || rt == LLAMA_ROPE_TYPE_IMROPE;
    if (s.mrope && s.input != KLC_INPUT_TOKENS) {
        // llama_batch_init allocates one position per token; an embedding batch of an M-RoPE model has four per
        // token, section-major (src/llama-batch.cpp: pos[j * n_tokens + i]). The token path expands one to [p, p, p, 0].
        s.batch.pos = (llama_pos *) realloc(s.batch.pos, sizeof(llama_pos) * 4u * (size_t) s.n_batch);
        if (!s.batch.pos) die("out of memory (batch positions)");
    }
}

static std::vector<int32_t> tokenize(session & s, const std::string & text, bool chat) {
    std::string t = text;
    if (chat) {
        const char * tmpl = llama_model_chat_template(s.model, nullptr);
        if (!tmpl) die("--chat: the model has no chat template");
        llama_chat_message msg = {"user", text.c_str()};
        std::vector<char> buf(text.size() * 4 + 1024);
        int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
        if (n > (int) buf.size()) {
            buf.resize(n + 1);
            n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
        }
        if (n < 0) die("--chat: llama_chat_apply_template failed");
        t.assign(buf.data(), n);
    }
    int n = -llama_tokenize(s.vocab, t.c_str(), (int32_t) t.size(), nullptr, 0, true, true);
    std::vector<int32_t> toks(std::max(n, 0));
    if (n <= 0 || llama_tokenize(s.vocab, t.c_str(), (int32_t) t.size(), toks.data(), n, true, true) != n) {
        die("tokenization failed for prompt \"%s\"", text.c_str());
    }
    return toks;
}

static std::string detok(session & s, const int32_t * toks, size_t n) {
    std::string out;
    char buf[256];
    for (size_t i = 0; i < n; ++i) {
        const int k = llama_token_to_piece(s.vocab, toks[i], buf, sizeof(buf), 0, true);
        if (k > 0) out.append(buf, k);
    }
    return out;
}

static std::string one_line(const std::string & s, size_t max_len = 160) {
    std::string o;
    for (char c : s) {
        if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t"; else o += c;
    }
    if (o.size() > max_len) {
        size_t k = max_len;
        while (k > 0 && ((unsigned char) o[k] & 0xC0) == 0x80) k--;   // do not split UTF-8
        o = o.substr(0, k) + "...";
    }
    return o;
}

static int argmax_row(const float * x, int n, int * n_nonfinite) {
    int   best = -1;
    float bv   = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) { (*n_nonfinite)++; continue; }
        if (best < 0 || x[i] > bv) { best = i; bv = x[i]; }
    }
    return best < 0 ? 0 : best;
}

// Decodes token indices [i0, i1) of `toks` (positions = indices, sequence 0) in n_batch chunks.
// Token indices in [r0, r1) output logits, copied to rows + (i - row0) * n_vocab.
static void decode_span(session & s, uint32_t seq_index, const std::vector<int32_t> & toks, int i0, int i1,
                        int r0, int r1, int row0, float * rows) {
    for (int c0 = i0; c0 < i1; c0 += s.n_batch) {
        const int c1 = std::min(i1, c0 + s.n_batch);
        llama_batch & b = s.batch;
        b.n_tokens = 0;
        for (int i = c0; i < c1; ++i) {
            const int k = b.n_tokens++;
            if (s.input == KLC_INPUT_TOKENS) {
                b.token[k] = toks[i];
            } else {
                const float * e = s.embd.row(toks[i]);
                float * dst = b.embd + (size_t) k * s.n_embd;
                if (s.input == KLC_INPUT_EMBD_ULP) {
                    const uint64_t key = embd_nudge_key(s.seed, seq_index, (uint32_t) i);
                    for (int j = 0; j < s.n_embd; ++j) dst[j] = embd_nudge(e[j], key, (uint32_t) j);
                } else {
                    memcpy(dst, e, sizeof(float) * s.n_embd);
                }
            }
            b.pos[k]       = i;
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = 0;
            b.logits[k]    = (i >= r0 && i < r1) ? 1 : 0;
        }
        if (s.mrope && s.input != KLC_INPUT_TOKENS) {
            // the text positions of the token path, [p, p, p, 0], in the embedding batch's section-major layout
            const int n = b.n_tokens;
            for (int k = 0; k < n; ++k) {
                const llama_pos p = b.pos[k];
                b.pos[k] = p;
                b.pos[n + k] = p;
                b.pos[2 * n + k] = p;
                b.pos[3 * n + k] = 0;
            }
        }
        // -v: one marker line before and after each llama_decode, so that a log reader can assign
        // the backend's per-launch lines (TSR_DEBUG=1) and the scheduler's split listings
        // (GGML_SCHED_DEBUG) to a decode call (the tiny-model graph ctests, SW §10.2)
        if (g_verbose) fprintf(stderr, "klcheck: decode begin seq %u tokens [%d, %d) n_tokens %d\n", seq_index, c0, c1, c1 - c0);
        const int ret = llama_decode(s.ctx, b);
        if (ret != 0) die("llama_decode failed (%d) at token index %d", ret, c0);
        for (int i = std::max(c0, r0); i < std::min(c1, r1); ++i) {
            const float * lg = llama_get_logits_ith(s.ctx, i - c0);
            if (!lg) die("no logits for token index %d", i);
            memcpy(rows + (size_t) (i - row0) * s.n_vocab, lg, sizeof(float) * s.n_vocab);
        }
        llama_synchronize(s.ctx);
        if (g_verbose) fprintf(stderr, "klcheck: decode end\n");
        if (c1 - c0 > 1) s.n_prefill_tok += c1 - c0; else s.n_gen_tok++;
    }
}

// ---------------------------------------------------------------------------------------------
// klcheck run

static void usage_run() {
    fprintf(stderr,
        "usage: klcheck run -m MODEL --device sim|mock|vkernel|xrt|cpu --out FILE [options]\n"
        "  generate mode (no --tokens): greedy-generate -n G tokens per prompt, record logits\n"
        "  replay mode (--tokens FILE): feed the token sequences of FILE (any klcheck file),\n"
        "                               record logits at the same positions\n"
        "  --prompts FILE      one prompt per line (\\n escapes); default: 8 built-in prompts\n"
        "  --n-prompts N       use the first N prompts / sequences\n"
        "  -n G                tokens to generate per prompt (default 64)\n"
        "  --prompt-tail N     also record the last N prompt positions (default 1 = last prompt pos)\n"
        "  --chat              wrap prompts in the model's chat template\n"
        "  --perturb-ulp SEED  input = token_embd rows, every element nudged by 1 ulp (floor run)\n"
        "  --input tokens|embd input path without perturbation (embd: must equal tokens bit-exactly)\n"
        "  --replay batch|step replay: one decode per sequence or prompt + 1-token steps (default: batch on the\n"
        "                      CPU, step on a device, whose replay must repeat the generating run's graphs for\n"
        "                      compare --expect-identical, SW §10.4 M6 gate item 2)\n"
        "  --repack            CPU: allow repacked weight buffers (anonymous memory; off by default)\n"
        "  -t N  -c N  -b N  -ub N  --resume  -v\n"
        "  settings (SW-22; every phase of a comparison must use the same, all are recorded in the file):\n"
        "  --phase m6|m7       M6 (default): KV cache and attention on the host; M7: on the device (offload_kqv)\n"
        "  --offload-kqv on|off  override the phase's offload_kqv\n"
        "  --fa on|off         flash attention (default on)\n"
        "  -ctk TYPE -ctv TYPE K / V cache types (default q8_0 / q8_0)\n"
        "  --attn-rot on|off   llama.cpp's Hadamard KV rotation (default off: LLAMA_ATTN_ROT_DISABLE=1)\n"
        "  --swa-full on|off   full-size SWA KV cache (default off: the SWA ring of pad(n_swa + n_ubatch, 256) cells,\n"
        "                      as llama-cli / llama-server; recorded)\n");
}

static void parse_run(int argc, char ** argv, run_opts & o) {
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * {
            if (i + 1 >= argc) die("%s needs a value", a.c_str());
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")  o.model = val();
        else if (a == "--device")              o.device = val();
        else if (a == "--out" || a == "-o")    o.out = val();
        else if (a == "--tokens")              o.tokens = val();
        else if (a == "--prompts")             o.prompts_file = val();
        else if (a == "--n-prompts")           o.n_prompts = (int) parse_int("--n-prompts", val(), 1, 1 << 20);
        else if (a == "-n" || a == "--n-gen")  o.n_gen = (int) parse_int("-n", val(), 1, 1 << 16);
        else if (a == "--prompt-tail")         o.prompt_tail = (int) parse_int("--prompt-tail", val(), 1, 1 << 16);
        else if (a == "--chat")                o.chat = true;
        else if (a == "--perturb-ulp")         { o.perturb = true; o.seed = parse_u64("--perturb-ulp", val()); }
        else if (a == "--input") {
            const std::string v = val();
            if (v == "tokens") o.input = KLC_INPUT_TOKENS;
            else if (v == "embd") o.input = KLC_INPUT_EMBD;
            else die("--input: tokens or embd");
        }
        else if (a == "--replay") {
            const std::string v = val();
            if (v == "batch") o.replay = KLC_REPLAY_BATCH;
            else if (v == "step") o.replay = KLC_REPLAY_STEP;
            else die("--replay: batch or step");
        }
        else if (a == "--repack")              o.repack = true;
        else if (a == "-t" || a == "--threads") o.threads = (int) parse_int("-t", val(), 1, 1024);
        else if (a == "-c" || a == "--ctx-size") o.n_ctx = (int) parse_int("-c", val(), 1, 1 << 24);
        else if (a == "-b" || a == "--batch-size") o.n_batch = (int) parse_int("-b", val(), 1, 1 << 20);
        else if (a == "-ub" || a == "--ubatch-size") o.n_ubatch = (int) parse_int("-ub", val(), 1, 1 << 20);
        else if (a == "--fa")                  o.fa = parse_on_off("--fa", val());
        else if (a == "--offload-kqv")         o.offload_kqv = parse_on_off("--offload-kqv", val());
        else if (a == "--attn-rot")            o.attn_rot_disable = !parse_on_off("--attn-rot", val());
        else if (a == "--swa-full")            o.swa_full = parse_on_off("--swa-full", val());
        else if (a == "--phase")               o.phase = parse_phase(val());
        else if (a == "-ctk" || a == "--cache-type-k") o.type_k = parse_cache_type("-ctk", val());
        else if (a == "-ctv" || a == "--cache-type-v") o.type_v = parse_cache_type("-ctv", val());
        else if (a == "--resume")              o.resume = true;
        else if (a == "-v" || a == "--verbose") g_verbose = 1;
        else if (a == "-h" || a == "--help")   { usage_run(); exit(0); }
        else { usage_run(); die("run: unknown option %s", a.c_str()); }
    }
    if (o.model.empty() || o.out.empty()) { usage_run(); die("run: -m and --out are required"); }
    if (!device_kind_ok(o.device)) die("--device %s: sim, mock, vkernel, xrt or cpu", o.device.c_str());
    if (o.replay < 0) o.replay = o.device == "cpu" ? KLC_REPLAY_BATCH : KLC_REPLAY_STEP;
    if (o.perturb) o.input = KLC_INPUT_EMBD_ULP;
}

struct plan_seq {
    uint32_t             index;
    std::vector<int32_t> toks;        // replay: prompt + generated; generate: empty until tokenized
    uint32_t             n_prompt = 0, n_gen = 0, row0 = 0, n_rows = 0;
    std::string          prompt;      // generate mode
};

static int cmd_run(int argc, char ** argv) {
    run_opts o;
    parse_run(argc, argv, o);

    // The device is chosen through the environment before ggml's backend registry is created.
    if (o.device == "cpu") setenv("TSR_DEVICE", "none", 1);
    else                   setenv("TSR_DEVICE", o.device.c_str(), 1);
    // llama.cpp's Hadamard KV rotation (src/llama-kv-cache.cpp reads the variable when a context is
    // created): off for every phase, device and CPU alike (SW-16, SW-22).
    if (o.attn_rot_disable) setenv("LLAMA_ATTN_ROT_DISABLE", "1", 1);
    else                    unsetenv("LLAMA_ATTN_ROT_DISABLE");
    // The phase also picks the device's kinds (read by the backend's supports_op): M6 runs MUL_MAT
    // only (SW-14). M7 keeps the caller's TSR_KINDS, so a kind list (all,-RMS_NORM) reaches the
    // backend; unset, the device takes every kind of SW §9.3 (W3b integration, SW-49).
    if (o.phase == KLC_PHASE_M6) setenv("TSR_KINDS", "m6", 1);

    const bool replay = !o.tokens.empty();
    std::vector<plan_seq> plan;
    klc_header src{};
    if (replay) {
        klc_reader r;
        r.open(o.tokens);
        src = r.h;
        while (r.next()) {
            if (o.n_prompts > 0 && (int) plan.size() >= o.n_prompts) break;
            plan_seq p;
            p.index    = r.seq.index;
            p.toks     = r.toks;
            p.n_prompt = r.seq.n_prompt;
            p.n_gen    = r.seq.n_gen;
            p.row0     = r.seq.row0;
            p.n_rows   = r.seq.n_rows;
            plan.push_back(std::move(p));
        }
        if (plan.empty()) die("%s: no complete sequences", o.tokens.c_str());
    } else {
        const std::vector<std::string> prompts = load_prompts(o.prompts_file);
        const size_t n = o.n_prompts > 0 ? std::min<size_t>(o.n_prompts, prompts.size()) : prompts.size();
        for (size_t i = 0; i < n; ++i) {
            plan_seq p;
            p.index  = (uint32_t) i;
            p.prompt = prompts[i];
            plan.push_back(std::move(p));
        }
    }

    const std::string tag = "[" + o.device + (o.input == KLC_INPUT_EMBD_ULP ? "+ulp" : o.input == KLC_INPUT_EMBD ? "+embd" : "") + "]";

    // --resume: keep the complete records of an existing file (skip the model load if done)
    uint32_t done = 0;
    if (o.resume && file_exists(o.out)) {
        FILE * f = fopen(o.out.c_str(), "rb");
        klc_header h;
        std::string err;
        if (!f || !klc_read_header(f, o.out, &h, &err)) die("--resume: %s", err.empty() ? strerror(errno) : err.c_str());
        fclose(f);
        if (h.mode != (replay ? KLC_MODE_REPLAY : KLC_MODE_GENERATE) || h.input != (uint32_t) o.input ||
            h.perturb_seed != (o.perturb ? o.seed : 0) || fixed_str(h.device, sizeof(h.device)) != o.device ||
            h.flash_attn != (uint32_t) o.fa || (!replay && (h.n_gen != (uint32_t) o.n_gen || h.prompt_tail != (uint32_t) o.prompt_tail)) ||
            h.type_k != (int32_t) o.type_k || h.type_v != (int32_t) o.type_v || h.offload_kqv != (uint32_t) o.offload() ||
            h.attn_rot_disable != (uint32_t) o.attn_rot_disable || h.phase != (uint32_t) o.phase ||
            swa_cache_of(h) != (o.swa_full ? (uint32_t) KLC_SWA_FULL : (uint32_t) KLC_SWA_RING)) {
            die("--resume: %s was written with different options (%s)", o.out.c_str(), settings_str(h).c_str());
        }
        if (h.n_seq >= plan.size()) {
            fprintf(stderr, "klcheck %s: %s already has %u sequences, nothing to do\n", tag.c_str(), o.out.c_str(), h.n_seq);
            return 0;
        }
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    session s;
    s.input = o.input;
    s.seed  = o.seed;
    double t_load = 0;
    load_model(s, o, &t_load);
    if (s.input != KLC_INPUT_TOKENS) {
        s.embd.open_model(o.model);
        if (s.embd.n_embd != s.n_embd) {
            die("token_embd.weight [%" PRId64 ", %" PRId64 "] does not match n_embd %d", s.embd.n_embd, s.embd.n_rows, s.n_embd);
        }
    }
    if (replay && src.n_vocab != (uint32_t) s.n_vocab) {
        die("%s has n_vocab %u, the model %d", o.tokens.c_str(), src.n_vocab, s.n_vocab);
    }

    // tokenize (generate mode) and size the context
    int n_ctx_needed = 0, n_out_max = 1;
    for (plan_seq & p : plan) {
        if (!replay) {
            p.toks     = tokenize(s, p.prompt, o.chat);
            p.n_prompt = (uint32_t) p.toks.size();
            p.n_gen    = (uint32_t) o.n_gen;
            const uint32_t tail = std::min<uint32_t>(o.prompt_tail, p.n_prompt);
            p.row0   = p.n_prompt - tail;
            p.n_rows = tail + p.n_gen - 1;
            n_out_max = std::max<int>(n_out_max, tail);
        } else {
            n_out_max = std::max<int>(n_out_max, o.replay == KLC_REPLAY_BATCH ? (int) p.n_rows : (int) (p.n_prompt - std::min(p.row0, p.n_prompt)));
        }
        n_ctx_needed = std::max<int>(n_ctx_needed, p.n_prompt + p.n_gen);
    }
    create_context(s, o, n_ctx_needed, n_out_max);

    klc_header h{};
    memcpy(h.magic, KLC_MAGIC, 8);
    h.version        = KLC_VERSION;
    h.header_bytes   = sizeof(klc_header);
    h.seq_bytes      = sizeof(klc_seq);
    h.n_vocab        = (uint32_t) s.n_vocab;
    h.n_seq          = 0;
    h.n_seq_planned  = (uint32_t) plan.size();
    h.n_gen          = replay ? src.n_gen : (uint32_t) o.n_gen;
    h.prompt_tail    = replay ? src.prompt_tail : (uint32_t) o.prompt_tail;
    h.mode           = replay ? KLC_MODE_REPLAY : KLC_MODE_GENERATE;
    h.replay         = replay ? (uint32_t) o.replay : (uint32_t) KLC_REPLAY_STEP;
    h.input          = (uint32_t) o.input;
    h.flash_attn     = (uint32_t) o.fa;
    h.perturb_seed   = o.perturb ? o.seed : 0;
    h.n_layer        = (uint32_t) llama_model_n_layer(s.model);
    h.n_embd         = (uint32_t) s.n_embd;
    h.n_threads      = (uint32_t) o.threads;
    h.flags          = (o.repack ? KLC_FLAG_REPACK : 0) | (o.chat ? KLC_FLAG_CHAT : 0);
    h.model_n_params = llama_model_n_params(s.model);
    h.model_size     = llama_model_size(s.model);
    h.t_load_s       = t_load;
    h.created_unix   = (uint64_t) time(nullptr);
    copy_str(h.device, sizeof(h.device), o.device);
    char desc[256];
    llama_model_desc(s.model, desc, sizeof(desc));
    copy_str(h.model_desc, sizeof(h.model_desc), desc);
    copy_str(h.model_file, sizeof(h.model_file), basename_of(o.model));
    if (replay) copy_str(h.tokens_from, sizeof(h.tokens_from), basename_of(o.tokens));
    h.type_k           = (int32_t) o.type_k;
    h.type_v           = (int32_t) o.type_v;
    h.offload_kqv      = o.offload() ? 1u : 0u;
    h.phase            = (uint32_t) o.phase;
    h.swa_cache        = o.swa_full ? KLC_SWA_FULL : KLC_SWA_RING;
    if (o.device != "cpu") {    // the device and image the model loaded onto (SW §6.9)
        typedef bool (*versions_fn)(ggml_backend_dev_t, tsr_versions *);
        ggml_backend_dev_t dev = find_tesseract_device();
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        versions_fn fn = reg ? (versions_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_tesseract_versions") : nullptr;
        tsr_versions v;
        if (fn && fn(dev, &v)) {
            h.basis = v.basis;
            h.core = (uint16_t) v.core;
            h.kinds_sig = (uint16_t) v.kinds_sig;
        }
        fprintf(stderr, "klcheck [%s]: %s\n", o.device.c_str(), versions_str(h).c_str());
        const std::string kinds = tsr_kinds_of(o.phase);
        fprintf(stderr, "klcheck [%s]: TSR_KINDS %s\n", o.device.c_str(),
                kinds == "unset" ? "unset (every kind of the booted image)" : kinds.c_str());
    }
    {   // what the context saw (set above, before llama_init_from_model)
        const char * e = getenv("LLAMA_ATTN_ROT_DISABLE");
        h.attn_rot_disable = e != nullptr && atoi(e) != 0 ? 1u : 0u;
    }

    klc_writer w;
    if (o.resume && file_exists(o.out)) {
        done = w.resume(o.out);
        if (w.h.n_vocab != h.n_vocab || w.h.model_n_params != h.model_n_params) {
            die("--resume: %s was written for another model", o.out.c_str());
        }
        w.h.t_load_s = h.t_load_s;
        fprintf(stderr, "klcheck %s: resuming %s after %u sequences\n", tag.c_str(), o.out.c_str(), done);
    } else {
        w.create(o.out, h);
    }
    const uint64_t base_prefill_tok = w.h.n_prefill_tok, base_gen_tok = w.h.n_gen_tok;

    fprintf(stderr, "klcheck %s: %s (%s), n_vocab %d, %s mode, input %s, %zu sequences, model load %.2f s\n",
            tag.c_str(), desc, basename_of(o.model).c_str(), s.n_vocab, mode_name(h.mode), input_name(h.input),
            plan.size(), t_load);
    fprintf(stderr, "klcheck %s: %s\n", tag.c_str(), settings_str(w.h).c_str());

    llama_memory_t mem = llama_get_memory(s.ctx);
    std::vector<float> rows;
    for (size_t pi = done; pi < plan.size(); ++pi) {
        plan_seq & p = plan[pi];
        rows.assign((size_t) p.n_rows * s.n_vocab, 0.0f);
        llama_memory_clear(mem, true);
        klc_seq rec{};
        memcpy(rec.magic, KLC_SEQ_MAGIC, 4);
        rec.index    = p.index;
        rec.n_prompt = p.n_prompt;
        rec.n_gen    = p.n_gen;
        rec.row0     = p.row0;
        rec.n_rows   = p.n_rows;
        const int end = (int) (p.row0 + p.n_rows);   // = n_prompt + n_gen - 1
        int n_nonfinite = 0, agree = 0;
        double t0 = now_s();

        if (!replay) {
            std::vector<int32_t> & toks = p.toks;
            decode_span(s, p.index, toks, 0, (int) p.n_prompt, (int) p.row0, (int) p.n_prompt, (int) p.row0, rows.data());
            rec.t_prefill_s = now_s() - t0;
            t0 = now_s();
            for (uint32_t g = 0; g < p.n_gen; ++g) {
                const int i = (int) p.n_prompt - 1 + (int) g;     // token index whose logits pick token i + 1
                if (g > 0) decode_span(s, p.index, toks, i, i + 1, i, i + 1, (int) p.row0, rows.data());
                toks.push_back(argmax_row(rows.data() + (size_t) (i - (int) p.row0) * s.n_vocab, s.n_vocab, &n_nonfinite));
            }
            rec.t_gen_s = now_s() - t0;
        } else if (o.replay == KLC_REPLAY_BATCH) {
            decode_span(s, p.index, p.toks, 0, end, (int) p.row0, end, (int) p.row0, rows.data());
            rec.t_prefill_s = now_s() - t0;
        } else {
            decode_span(s, p.index, p.toks, 0, (int) p.n_prompt, (int) p.row0, std::min(end, (int) p.n_prompt), (int) p.row0, rows.data());
            rec.t_prefill_s = now_s() - t0;
            t0 = now_s();
            for (int i = (int) p.n_prompt; i < end; ++i) {
                decode_span(s, p.index, p.toks, i, i + 1, std::max(i, (int) p.row0), i + 1, (int) p.row0, rows.data());
            }
            rec.t_gen_s = now_s() - t0;
        }
        for (int i = std::max((int) p.row0, (int) p.n_prompt - 1); i < end; ++i) {
            agree += argmax_row(rows.data() + (size_t) (i - (int) p.row0) * s.n_vocab, s.n_vocab, &n_nonfinite) == p.toks[i + 1];
        }
        for (uint32_t g = 0; g < p.n_gen; ++g) rec.n_eog += llama_vocab_is_eog(s.vocab, p.toks[p.n_prompt + g]) ? 1 : 0;

        w.h.n_prefill_tok = base_prefill_tok + s.n_prefill_tok;
        w.h.n_gen_tok     = base_gen_tok + s.n_gen_tok;
        w.write_seq(rec, p.toks, rows.data());

        const double per_tok = p.n_gen > 1 && rec.t_gen_s > 0 ? rec.t_gen_s / (p.n_gen - 1) : 0;
        fprintf(stderr, "klcheck %s: seq %zu/%zu: %u prompt + %u gen tokens, %u rows, prefill %.2f s, gen %.2f s",
                tag.c_str(), pi + 1, plan.size(), p.n_prompt, p.n_gen, p.n_rows, rec.t_prefill_s, rec.t_gen_s);
        if (per_tok > 0) fprintf(stderr, " (%.1f ms/token)", per_tok * 1e3);
        if (replay) fprintf(stderr, ", argmax = recorded token %d/%u", agree, p.n_gen);
        if (rec.n_eog) fprintf(stderr, ", %u EOG", rec.n_eog);
        fprintf(stderr, "\n");
        if (n_nonfinite) fprintf(stderr, "klcheck %s: WARNING: %d non-finite logits in seq %zu\n", tag.c_str(), n_nonfinite, pi + 1);
        if (!replay) {
            fprintf(stderr, "    prompt: %s\n    output: %s\n",
                    one_line(detok(s, p.toks.data(), p.n_prompt)).c_str(),
                    one_line(detok(s, p.toks.data() + p.n_prompt, p.n_gen)).c_str());
        }
    }
    fprintf(stderr, "klcheck %s: wrote %s (%u sequences, prefill %.1f s, gen %.1f s, total decode %.1f s)\n",
            tag.c_str(), o.out.c_str(), w.h.n_seq, w.h.t_prefill_s, w.h.t_gen_s, w.h.t_prefill_s + w.h.t_gen_s);
    return 0;
}

// ---------------------------------------------------------------------------------------------
// klcheck compare

struct cmp_opts {
    std::string ref, dev, json;
    std::vector<std::string> floors;   // --floor, once per floor run (seed); the floor KL is the mean over them
    std::string tsr_kinds;             // --tsr-kinds: the device run's TSR_KINDS, for the report (klcheck all)
    double      kl_ratio = 3.0, min_top1 = 0.8, min_kl = 0.0;
    bool        expect_identical = false;
    bool        quiet = false;
};

struct row_stat {
    uint32_t seq, row;
    bool     gen;            // the row predicts a generated token (else a prompt token)
    double   kl_dev, kl_floor;            // kl_floor: the mean over the floor runs
    int      am_ref, am_dev, am_floor;    // am_floor: the first floor run's argmax
    int      top1_floor;                  // floor runs whose argmax equals the reference's
    double   maxdiff_dev, cos_dev;
    bool     ident_dev, ident_floor;      // ident_floor: the first floor run's
    std::vector<double> kl_floors;        // KL per floor run
};

struct agg {
    uint64_t n = 0, top1 = 0, ident = 0;
    double   sum_kl = 0, max_kl = 0, max_absdiff = 0, min_cos = 1;
    size_t   max_at = 0;
    std::vector<double> kls;
    void add(double kl, bool t1, bool id, double md, double cs, size_t at) {
        n++;
        top1 += t1;
        ident += id;
        sum_kl += kl;
        if (kl > max_kl || n == 1) { max_kl = kl; max_at = at; }
        max_absdiff = std::max(max_absdiff, md);
        min_cos     = std::min(min_cos, cs);
        kls.push_back(kl);
    }
    double mean() const { return n ? sum_kl / n : 0; }
    double t1() const { return n ? (double) top1 / n : 0; }
    double median() const {
        if (kls.empty()) return 0;
        std::vector<double> v = kls;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }
};

struct cmp_result {
    agg      dev, floor, dev_gen, floor_gen, dev_prompt, floor_prompt;   // floor*: the rows of every floor run
    std::vector<agg>      floor_run;      // per floor run, all rows
    std::vector<uint64_t> floor_seed;     // per floor run: the perturbation seed in its header
    std::vector<double>   floor_ratio;    // per floor run: mean KL(dev) / the run's mean KL
    std::vector<row_stat> rows;
    uint32_t n_seq = 0;
    bool     have_floor = false, complete = true, pass = false;
    uint64_t nonfinite = 0, gen_token_mismatch = 0;
    double   floor_kl = 0;                // the criterion's floor: the mean over the runs of each run's mean KL
    double   ratio = 0, cap = 0;          // ratio: mean KL(dev) / floor_kl
};

// log-softmax in double
static void log_softmax(const float * x, int n, std::vector<double> & out, uint64_t * nonfinite) {
    double mx = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) { (*nonfinite)++; continue; }
        mx = std::max(mx, (double) x[i]);
    }
    double sum = 0;
    for (int i = 0; i < n; ++i) if (std::isfinite(x[i])) sum += std::exp((double) x[i] - mx);
    const double lse = mx + std::log(sum);
    out.resize(n);
    for (int i = 0; i < n; ++i) out[i] = std::isfinite(x[i]) ? (double) x[i] - lse : -INFINITY;
}

// KL(P || Q) in nats, P = softmax(lp)
static double kl_div(const std::vector<double> & lp, const std::vector<double> & lq) {
    double kl = 0;
    for (size_t i = 0; i < lp.size(); ++i) {
        if (lp[i] == -INFINITY) continue;
        const double p = std::exp(lp[i]);
        if (p == 0) continue;
        kl += p * (lp[i] - lq[i]);
    }
    return kl;
}

static int argmax_d(const std::vector<double> & v) {
    return (int) (std::max_element(v.begin(), v.end()) - v.begin());
}

static void diff_stats(const float * a, const float * b, int n, double * maxdiff, double * cosv) {
    double md = 0, ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < n; ++i) {
        const double x = a[i], y = b[i];
        md = std::max(md, std::fabs(x - y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    *maxdiff = md;
    *cosv    = (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : (aa == bb ? 1.0 : 0.0);
}

static void check_pair(const klc_header & a, const std::string & pa, const klc_header & b, const std::string & pb) {
    if (a.n_vocab != b.n_vocab) die("%s and %s: n_vocab %u vs %u", pa.c_str(), pb.c_str(), a.n_vocab, b.n_vocab);
    if (a.model_n_params != b.model_n_params || fixed_str(a.model_desc, 128) != fixed_str(b.model_desc, 128)) {
        fprintf(stderr, "klcheck: WARNING: %s and %s were written with different models (%s / %s)\n", pa.c_str(),
                pb.c_str(), fixed_str(a.model_desc, 128).c_str(), fixed_str(b.model_desc, 128).c_str());
    }
    if (a.flash_attn != b.flash_attn) {
        fprintf(stderr, "klcheck: WARNING: %s and %s differ in flash attention\n", pa.c_str(), pb.c_str());
    }
    // SW-22: the device, CPU and floor runs of one comparison use the same KV type, FA and
    // rotation settings (offload_kqv differs legitimately between a device run and a CPU run only
    // in what it moves; it is reported, not checked).
    if (a.type_k != b.type_k || a.type_v != b.type_v || a.attn_rot_disable != b.attn_rot_disable || a.phase != b.phase ||
        swa_cache_of(a) != swa_cache_of(b)) {
        fprintf(stderr, "klcheck: WARNING: %s and %s were written with different settings:\n  %s\n  %s\n", pa.c_str(),
                pb.c_str(), settings_str(a).c_str(), settings_str(b).c_str());
    }
}

static void check_seq(const klc_reader & a, const klc_reader & b) {
    if (a.seq.index != b.seq.index || a.seq.row0 != b.seq.row0 || a.seq.n_rows != b.seq.n_rows ||
        a.seq.n_prompt != b.seq.n_prompt || a.toks != b.toks) {
        die("%s and %s: record %u has different tokens or positions (was %s replayed from %s?)", a.path.c_str(),
            b.path.c_str(), a.n_read - 1, b.path.c_str(), a.path.c_str());
    }
}

static std::string json_str(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char) c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += (char) c;
    }
    return o + "\"";
}

// a JSON number; null for inf / nan (a ratio against a floor of 0)
static std::string json_num(double v, const char * fmt) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    snprintf(b, sizeof(b), fmt, v);
    return b;
}

static int compare_files(const cmp_opts & o, cmp_result & R) {
    klc_reader ref, dev;
    ref.open(o.ref);
    dev.open(o.dev);
    check_pair(ref.h, o.ref, dev.h, o.dev);
    const size_t nf = o.floors.size();
    std::vector<klc_reader> flr(nf);
    R.have_floor = nf > 0;
    R.floor_run.assign(nf, agg());
    R.floor_seed.assign(nf, 0);
    R.floor_ratio.assign(nf, 0.0);
    for (size_t i = 0; i < nf; ++i) {
        flr[i].open(o.floors[i]);
        check_pair(ref.h, o.ref, flr[i].h, o.floors[i]);
        R.floor_seed[i] = flr[i].h.perturb_seed;
        for (size_t j = 0; j < i; ++j) {
            if (flr[j].h.input == flr[i].h.input && flr[j].h.perturb_seed == flr[i].h.perturb_seed) {
                // averaging one floor twice is never intended
                die("floor runs %s and %s have the same seed %" PRIu64 " (the same floor twice)",
                    o.floors[j].c_str(), o.floors[i].c_str(), flr[i].h.perturb_seed);
            }
        }
    }
    const uint32_t n_vocab = ref.h.n_vocab;
    uint32_t n_seq = std::min(ref.h.n_seq, dev.h.n_seq);
    uint32_t want  = std::max({ref.h.n_seq, dev.h.n_seq, dev.h.n_seq_planned});
    for (const klc_reader & f : flr) {
        n_seq = std::min(n_seq, f.h.n_seq);
        want  = std::max(want, f.h.n_seq);
    }
    R.complete = n_seq == want;
    if (!R.complete) {
        fprintf(stderr, "klcheck: WARNING: only %u of %u sequences are present in all files; comparing those\n", n_seq, want);
    }

    std::vector<float>    xr(n_vocab), xd(n_vocab), xf(n_vocab);
    std::vector<double>   lr, ld, lf;
    std::vector<uint32_t> seq_prompt;
    for (uint32_t si = 0; si < n_seq; ++si) {
        ref.next();
        dev.next();
        check_seq(dev, ref);
        for (klc_reader & f : flr) { f.next(); check_seq(dev, f); }
        const klc_seq & sq = dev.seq;
        seq_prompt.push_back(sq.n_prompt);
        for (uint32_t r = 0; r < sq.n_rows; ++r) {
            ref.read_row(xr.data());
            dev.read_row(xd.data());
            row_stat st{};
            st.seq = si;
            st.row = r;
            const uint32_t ti = sq.row0 + r;          // token index of the row; predicts ti + 1
            st.gen = ti + 1 >= sq.n_prompt;
            log_softmax(xr.data(), n_vocab, lr, &R.nonfinite);
            log_softmax(xd.data(), n_vocab, ld, &R.nonfinite);
            st.kl_dev    = kl_div(lr, ld);
            st.am_ref    = argmax_d(lr);
            st.am_dev    = argmax_d(ld);
            st.ident_dev = memcmp(xr.data(), xd.data(), sizeof(float) * n_vocab) == 0;
            diff_stats(xr.data(), xd.data(), n_vocab, &st.maxdiff_dev, &st.cos_dev);
            if (dev.h.mode == KLC_MODE_GENERATE && st.gen && st.am_dev != dev.toks[ti + 1]) R.gen_token_mismatch++;
            const size_t at = R.rows.size();
            R.dev.add(st.kl_dev, st.am_ref == st.am_dev, st.ident_dev, st.maxdiff_dev, st.cos_dev, at);
            (st.gen ? R.dev_gen : R.dev_prompt).add(st.kl_dev, st.am_ref == st.am_dev, st.ident_dev, st.maxdiff_dev, st.cos_dev, at);
            st.kl_floors.resize(nf);
            double sum_floor = 0;
            for (size_t i = 0; i < nf; ++i) {
                double md, cs;
                flr[i].read_row(xf.data());
                log_softmax(xf.data(), n_vocab, lf, &R.nonfinite);
                const double kl = kl_div(lr, lf);
                const int    am = argmax_d(lf);
                const bool   id = memcmp(xr.data(), xf.data(), sizeof(float) * n_vocab) == 0;
                diff_stats(xr.data(), xf.data(), n_vocab, &md, &cs);
                st.kl_floors[i] = kl;
                sum_floor += kl;
                st.top1_floor += am == st.am_ref;
                if (i == 0) {
                    st.am_floor    = am;
                    st.ident_floor = id;
                }
                R.floor.add(kl, am == st.am_ref, id, md, cs, at);
                (st.gen ? R.floor_gen : R.floor_prompt).add(kl, am == st.am_ref, id, md, cs, at);
                R.floor_run[i].add(kl, am == st.am_ref, id, md, cs, at);
            }
            if (nf) st.kl_floor = sum_floor / nf;
            R.rows.push_back(std::move(st));
        }
    }
    R.n_seq = n_seq;

    // The floor KL: the mean over the floor runs of each run's mean KL (one run: its mean KL). One
    // run's floor depends on its seed, so `klcheck all` makes several (--floor-seeds, cmd_all).
    double fl = 0;
    for (const agg & a : R.floor_run) fl += a.mean();
    if (nf) fl /= nf;
    auto ratio_to = [&](double f) { return f > 0 ? R.dev.mean() / f : (R.dev.mean() == 0 ? 0 : INFINITY); };
    for (size_t i = 0; i < nf; ++i) R.floor_ratio[i] = ratio_to(R.floor_run[i].mean());
    R.floor_kl = fl;
    R.cap   = std::max(o.min_kl, o.kl_ratio * fl);
    R.ratio = ratio_to(fl);
    bool pass = R.complete && R.nonfinite == 0 && R.dev.n > 0;
    if (o.expect_identical) {
        pass = pass && R.dev.ident == R.dev.n;
    } else {
        pass = pass && R.dev.t1() >= o.min_top1;
        if (R.have_floor) pass = pass && R.dev.mean() <= R.cap;
    }
    R.pass = pass;

    if (!o.quiet) {
        printf("klcheck compare: ref %s (%s, %s, input %s)\n", o.ref.c_str(), fixed_str(ref.h.device, 16).c_str(),
               mode_name(ref.h.mode), input_name(ref.h.input));
        printf("                 dev %s (%s, %s, input %s)\n", o.dev.c_str(), fixed_str(dev.h.device, 16).c_str(),
               mode_name(dev.h.mode), input_name(dev.h.input));
        for (size_t i = 0; i < nf; ++i) {
            printf("               floor %s (%s, input %s, seed %" PRIu64 ")\n", o.floors[i].c_str(),
                   fixed_str(flr[i].h.device, 16).c_str(), input_name(flr[i].h.input), flr[i].h.perturb_seed);
        }
        printf("model: %s, n_vocab %u, %u sequences, %zu positions (%" PRIu64 " prompt-tail, %" PRIu64 " generated)\n",
               fixed_str(dev.h.model_desc, 128).c_str(), n_vocab, n_seq, R.rows.size(), R.dev_prompt.n, R.dev_gen.n);
        printf("dev %s\n", settings_str(dev.h).c_str());
        printf("dev: %s, %s%s%s\n", fixed_str(dev.h.device, 16).c_str(), versions_str(dev.h).c_str(),
               o.tsr_kinds.empty() ? "" : ", TSR_KINDS ", o.tsr_kinds.c_str());
        printf("\n%-4s %6s %5s  %-11s %-11s %7s  %8s %8s  %s\n", "seq", "prompt", "rows", "KL(ref|dev)", "KL(floor)", "ratio",
               "top1 dev", "top1 flr", "first top-1 miss (row)");
        size_t k = 0;
        for (uint32_t si = 0; si < n_seq; ++si) {
            double sd = 0, sf = 0;
            uint32_t n = 0, td = 0, tf = 0;
            int first_miss = -1;
            for (; k < R.rows.size() && R.rows[k].seq == si; ++k, ++n) {
                const row_stat & rs = R.rows[k];
                sd += rs.kl_dev;
                sf += rs.kl_floor;
                td += rs.am_ref == rs.am_dev;
                tf += rs.top1_floor;
                if (rs.am_ref != rs.am_dev && first_miss < 0) first_miss = (int) rs.row;
            }
            const double md = n ? sd / n : 0, mf = n ? sf / n : 0;
            char cf[16] = "-", cr[16] = "-", ctf[16] = "-", cm[16] = "-";
            if (R.have_floor) {
                snprintf(cf, sizeof(cf), "%.4e", mf);
                snprintf(cr, sizeof(cr), "%.2f", mf > 0 ? md / mf : 0.0);
                snprintf(ctf, sizeof(ctf), "%.1f%%", n ? 100.0 * tf / ((double) n * nf) : 0.0);
            }
            if (first_miss >= 0) snprintf(cm, sizeof(cm), "%d", first_miss);
            printf("%-4u %6u %5u  %-11.4e %-11s %7s  %7.1f%% %8s  %s\n", si, seq_prompt[si], n, md, cf, cr,
                   n ? 100.0 * td / n : 0.0, ctf, cm);
        }
        auto line = [&](const char * name, const agg & a) {
            if (!a.n) return;
            const row_stat & m = R.rows[a.max_at];
            printf("  %-22s mean %.4e  median %.4e  max %.4e (seq %u row %u)  top-1 %6.2f%%  bit-identical %" PRIu64 "/%" PRIu64
                   "  max|dlogit| %.4g  min cos %.6f\n",
                   name, a.mean(), a.median(), a.max_kl, m.seq, m.row, 100.0 * a.t1(), a.ident, a.n, a.max_absdiff, a.min_cos);
        };
        printf("\nKL in nats per position, KL(ref || x); top-1 = argmax(x) == argmax(ref)\n");
        line("dev   (all)", R.dev);
        line("dev   (prompt tail)", R.dev_prompt);
        line("dev   (generated)", R.dev_gen);
        if (R.have_floor) {
            line("floor (all)", R.floor);
            line("floor (prompt tail)", R.floor_prompt);
            line("floor (generated)", R.floor_gen);
        }
        if (nf > 1) {   // the floor lines above pool the rows of every floor run
            printf("  floor runs: %zu, floor KL = the mean of their mean KL = %.4e\n", nf, fl);
            for (size_t i = 0; i < nf; ++i) {
                const agg & a = R.floor_run[i];
                printf("    seed %-6" PRIu64 " mean %.4e  median %.4e  max %.4e  top-1 %6.2f%%  dev ratio %.3f  %s\n", R.floor_seed[i],
                       a.mean(), a.median(), a.max_kl, 100.0 * a.t1(), R.floor_ratio[i], o.floors[i].c_str());
            }
        }
        if (R.nonfinite) printf("ERROR: %" PRIu64 " non-finite logits\n", R.nonfinite);
        if (R.gen_token_mismatch) {
            printf("WARNING: %" PRIu64 " generated tokens differ from the argmax of the device's own logits\n", R.gen_token_mismatch);
        }
        printf("\n");
        if (o.expect_identical) {
            printf("criterion: every row bit-identical: %" PRIu64 "/%" PRIu64 "\n", R.dev.ident, R.dev.n);
        } else {
            if (R.have_floor) {
                std::string runs, per;   // several floor runs: the floor is their mean; the ratio against each
                if (nf > 1) {
                    runs = ", the mean of " + std::to_string(nf) + " floor runs";
                    for (size_t i = 0; i < nf; ++i) {
                        char b[64];
                        snprintf(b, sizeof(b), "%s seed %" PRIu64 " %.3f", i ? "," : "; against", R.floor_seed[i], R.floor_ratio[i]);
                        per += b;
                    }
                }
                printf("criterion: mean KL %.4e <= %.4e (%.2f x floor %.4e%s%s): %s   [ratio %.3f%s]\n", R.dev.mean(), R.cap,
                       o.kl_ratio, fl, runs.c_str(), o.min_kl > 0 ? ", or the --min-kl cap" : "",
                       R.dev.mean() <= R.cap ? "ok" : "FAIL", R.ratio, per.c_str());
            }
            printf("criterion: top-1 agreement %.2f%% >= %.2f%%: %s\n", 100.0 * R.dev.t1(), 100.0 * o.min_top1,
                   R.dev.t1() >= o.min_top1 ? "ok" : "FAIL");
        }
        if (!R.complete) printf("criterion: all sequences present: FAIL (incomplete files)\n");
        printf("%s\n", R.pass ? "PASS" : "FAIL");
    }

    if (!o.json.empty()) {
        FILE * f = fopen(o.json.c_str(), "w");
        if (!f) die("%s: %s", o.json.c_str(), strerror(errno));
        auto jagg = [&](const char * name, const agg & a, bool last) {
            fprintf(f, "    %s: {\"n\": %" PRIu64 ", \"kl_mean\": %.9e, \"kl_median\": %.9e, \"kl_max\": %.9e, \"top1\": %.6f, "
                       "\"bit_identical\": %" PRIu64 ", \"max_abs_dlogit\": %.6g, \"min_cos\": %.9f}%s\n",
                    json_str(name).c_str(), a.n, a.mean(), a.median(), a.max_kl, a.t1(), a.ident, a.max_absdiff, a.min_cos,
                    last ? "" : ",");
        };
        // "floor": the first floor file, as before several floors; "floors": every floor run
        fprintf(f, "{\n  \"ref\": %s, \"dev\": %s, \"floor\": %s,\n", json_str(o.ref).c_str(), json_str(o.dev).c_str(),
                json_str(nf ? o.floors[0] : std::string()).c_str());
        fprintf(f, "  \"floors\": [");
        for (size_t i = 0; i < nf; ++i) {
            const agg & a = R.floor_run[i];
            fprintf(f, "%s\n    {\"file\": %s, \"seed\": %" PRIu64 ", \"n\": %" PRIu64 ", \"kl_mean\": %.9e, \"kl_median\": %.9e, "
                       "\"kl_max\": %.9e, \"top1\": %.6f, \"bit_identical\": %" PRIu64 ", \"ratio\": %s}",
                    i ? "," : "", json_str(o.floors[i]).c_str(), R.floor_seed[i], a.n, a.mean(), a.median(), a.max_kl, a.t1(), a.ident,
                    json_num(R.floor_ratio[i], "%.6f").c_str());
        }
        fprintf(f, "%s],\n", nf ? "\n  " : "");
        fprintf(f, "  \"model\": %s, \"n_vocab\": %u, \"n_seq\": %u, \"n_rows\": %zu,\n",
                json_str(fixed_str(dev.h.model_desc, 128)).c_str(), n_vocab, n_seq, R.rows.size());
        fprintf(f, "  \"settings\": {\"phase\": %s, \"flash_attn\": %s, \"type_k\": %s, \"type_v\": %s, \"offload_kqv\": %s, "
                   "\"attn_rot_disable\": %s, \"swa_cache\": %s},\n",
                json_str(phase_name(dev.h.phase)).c_str(), dev.h.flash_attn ? "true" : "false", json_str(type_name_or(dev.h.type_k)).c_str(),
                json_str(type_name_or(dev.h.type_v)).c_str(), dev.h.offload_kqv ? "true" : "false",
                dev.h.attn_rot_disable ? "true" : "false", json_str(swa_cache_of(dev.h) == KLC_SWA_RING ? "ring" : "full").c_str());
        fprintf(f, "  \"tsr_kinds\": %s,\n", o.tsr_kinds.empty() ? "null" : json_str(o.tsr_kinds).c_str());
        if (dev.h.basis == 0 && dev.h.core == 0) {
            fprintf(f, "  \"versions\": null,\n");
        } else {
            fprintf(f, "  \"versions\": {\"core\": \"%u.%u\", \"basis\": \"%u.%u.%u\", \"kinds_sig\": \"0x%04x\"},\n",
                    (dev.h.core >> 8) & 0xFFu, dev.h.core & 0xFFu, (dev.h.basis >> 16) & 0xFFu, (dev.h.basis >> 8) & 0xFFu,
                    dev.h.basis & 0xFFu, dev.h.kinds_sig);
        }
        fprintf(f, "  \"kl_ratio_limit\": %g, \"min_top1\": %g, \"min_kl\": %g, \"n_floor\": %zu, \"floor_kl\": %.9e, "
                   "\"ratio\": %s, \"cap\": %.9e,\n", o.kl_ratio, o.min_top1, o.min_kl, nf, fl, json_num(R.ratio, "%.6f").c_str(), R.cap);
        fprintf(f, "  \"complete\": %s, \"nonfinite\": %" PRIu64 ", \"pass\": %s,\n", R.complete ? "true" : "false", R.nonfinite,
                R.pass ? "true" : "false");
        fprintf(f, "  \"summary\": {\n");
        jagg("dev", R.dev, false);
        jagg("dev_prompt", R.dev_prompt, false);
        jagg("dev_gen", R.dev_gen, false);
        jagg("floor", R.floor, false);
        jagg("floor_prompt", R.floor_prompt, false);
        jagg("floor_gen", R.floor_gen, true);
        fprintf(f, "  },\n  \"rows\": [\n");
        for (size_t i = 0; i < R.rows.size(); ++i) {
            const row_stat & r = R.rows[i];
            std::string kls;     // several floor runs: kl_floor is their mean, kl_floors has each
            if (nf > 1) {
                kls = ", \"kl_floors\": [";
                for (size_t k = 0; k < nf; ++k) {
                    char b[32];
                    snprintf(b, sizeof(b), "%s%.6e", k ? ", " : "", r.kl_floors[k]);
                    kls += b;
                }
                kls += "]";
            }
            fprintf(f, "    {\"seq\": %u, \"row\": %u, \"gen\": %d, \"kl_dev\": %.6e, \"kl_floor\": %.6e%s, \"argmax_ref\": %d, "
                       "\"argmax_dev\": %d, \"argmax_floor\": %d, \"max_abs_dlogit\": %.5g, \"cos\": %.9f}%s\n",
                    r.seq, r.row, r.gen ? 1 : 0, r.kl_dev, r.kl_floor, kls.c_str(), r.am_ref, r.am_dev, r.am_floor, r.maxdiff_dev,
                    r.cos_dev, i + 1 < R.rows.size() ? "," : "");
        }
        fprintf(f, "  ]\n}\n");
        fclose(f);
    }
    return R.pass ? 0 : 1;
}

static int cmd_compare(int argc, char ** argv) {
    cmp_opts o;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * {
            if (i + 1 >= argc) die("%s needs a value", a.c_str());
            return argv[++i];
        };
        if      (a == "--ref")              o.ref = val();
        else if (a == "--dev")              o.dev = val();
        else if (a == "--floor")            o.floors.push_back(val());   // several: one per floor run
        else if (a == "--tsr-kinds")        o.tsr_kinds = val();
        else if (a == "--json")             o.json = val();
        else if (a == "--kl-ratio")         o.kl_ratio = parse_double("--kl-ratio", val());
        else if (a == "--min-top1")         o.min_top1 = parse_double("--min-top1", val());
        else if (a == "--min-kl")           o.min_kl = parse_double("--min-kl", val());
        else if (a == "--expect-identical") o.expect_identical = true;
        else if (a == "-q" || a == "--quiet") o.quiet = true;
        else die("compare: unknown option %s\nusage: klcheck compare --ref CPU --dev DEV [--floor FLOOR]... [--kl-ratio 3] "
                 "[--min-top1 0.8] [--min-kl 0] [--json FILE] [--tsr-kinds LIST] [--expect-identical] [-q]", a.c_str());
    }
    if (o.ref.empty() || o.dev.empty()) die("compare: --ref and --dev are required");
    if (o.floors.empty() && !o.expect_identical) {
        fprintf(stderr, "klcheck: no --floor: only the top-1 criterion is checked\n");
    }
    cmp_result R;
    return compare_files(o, R);
}

// ---------------------------------------------------------------------------------------------
// klcheck all — the three runs as child processes, then compare

static int run_child(const std::vector<std::string> & args) {
    std::string line;
    for (const std::string & a : args) line += (line.empty() ? "" : " ") + a;
    fprintf(stderr, "klcheck all: $ %s\n", line.c_str());
    fflush(stderr);
    std::vector<char *> av;
    for (const std::string & a : args) av.push_back(const_cast<char *>(a.c_str()));
    av.push_back(nullptr);
    const pid_t pid = fork();
    if (pid < 0) die("fork: %s", strerror(errno));
    if (pid == 0) {
        execv("/proc/self/exe", av.data());
        fprintf(stderr, "klcheck: exec: %s\n", strerror(errno));
        _exit(127);
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) die("waitpid: %s", strerror(errno));
    if (WIFEXITED(st)) return WEXITSTATUS(st);
    return 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
}

// The floor files of `klcheck all`: one run per seed, seed0 .. seed0 + n - 1. A single run keeps the
// name floor.klc.
static std::vector<std::string> all_floor_files(const std::string & dir, uint64_t seed0, int n) {
    if (n == 1) return {dir + "/floor.klc"};
    std::vector<std::string> v;
    for (int i = 0; i < n; ++i) v.push_back(dir + "/floor-seed" + std::to_string(seed0 + (uint64_t) i) + ".klc");
    return v;
}

static int cmd_all(int argc, char ** argv, const char * self) {
    std::string dir, device = "sim", model;
    uint64_t seed = 1;          // the first floor run's seed
    // --floor-seeds N: N floor runs with the seeds seed .. seed + N - 1; the floor KL is the mean over
    // them. Default 3 (decision, 2026-10-31): one run's floor depends on its seed. On Gemma 4 12B
    // Q4_K_M (8 prompts) the M7 ratio was 2.19, 1.39, 1.69 and 1.51 against seeds 1 to 4, and the
    // floor of one sequence ranged from 0.133 to 0.423. Each run is one more CPU replay.
    int n_floor = 3;
    int phase = KLC_PHASE_M6;   // --phase, for the report's TSR_KINDS (cmd_run's default)
    bool embd_check = true;
    std::vector<std::string> common, gen_only, cmp;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die("%s needs a value", a.c_str());
            return argv[++i];
        };
        if (a == "-o" || a == "--out-dir")      dir = val();
        else if (a == "--device")               device = val();
        else if (a == "--seed")                 seed = parse_u64("--seed", val().c_str());
        else if (a == "--floor-seeds")          n_floor = (int) parse_int("--floor-seeds", val().c_str(), 1, 64);
        else if (a == "--no-embd-check")        embd_check = false;
        else if (a == "-m" || a == "--model")   { model = val(); common.insert(common.end(), {"-m", model}); }
        else if (a == "--prompts" || a == "--n-prompts" || a == "-n" || a == "--prompt-tail") {
            gen_only.push_back(a); gen_only.push_back(val());
        }
        else if (a == "--chat")                 gen_only.push_back(a);
        else if (a == "-t" || a == "-c" || a == "-b" || a == "-ub" || a == "--fa" || a == "--replay" || a == "--phase" ||
                 a == "--offload-kqv" || a == "-ctk" || a == "--cache-type-k" || a == "-ctv" || a == "--cache-type-v" ||
                 a == "--attn-rot" || a == "--swa-full") {
            const std::string v = val();
            if (a == "--phase") phase = parse_phase(v);
            common.push_back(a); common.push_back(v);       // the baseline settings go to every phase (SW-22)
        }
        else if (a == "--repack" || a == "--resume" || a == "-v") common.push_back(a);
        else if (a == "--kl-ratio" || a == "--min-top1" || a == "--min-kl") { cmp.push_back(a); cmp.push_back(val()); }
        else die("all: unknown option %s\nusage: klcheck all -m MODEL [-o DIR] [--device sim] [--seed S] [--floor-seeds N] "
                 "[--no-embd-check] [run options] [--kl-ratio R] [--min-top1 P] [--min-kl K]", a.c_str());
    }
    if (model.empty()) die("all: -m is required");
    if (!device_kind_ok(device)) die("all: --device %s: sim, mock, vkernel, xrt or cpu", device.c_str());
    // without -o: a private temporary directory, removed afterwards (concurrent ctest runs)
    const bool tmp_dir = dir.empty();
    if (tmp_dir) {
        const char * t = getenv("TMPDIR");
        std::string tmpl = std::string(t && *t ? t : "/tmp") + "/klcheck-XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back(0);
        if (!mkdtemp(buf.data())) die("mkdtemp %s: %s", tmpl.c_str(), strerror(errno));
        dir = buf.data();
    }
    mkdir_p(dir);
    const std::string fdev = dir + "/dev.klc", fcpu = dir + "/cpu.klc";
    const std::vector<std::string> ffloors = all_floor_files(dir, seed, n_floor);
    const std::string fembd = dir + "/cpu-embd.klc", fjson = dir + "/report.json";
    const double t0 = now_s();

    // --replay applies to the CPU runs only; the device run generates step by step
    std::vector<std::string> common_dev;
    for (size_t i = 0; i < common.size(); ++i) {
        if (common[i] == "--replay") { ++i; continue; }
        common_dev.push_back(common[i]);
    }
    std::vector<std::string> a1 = {self, "run", "--device", device, "--out", fdev};
    a1.insert(a1.end(), common_dev.begin(), common_dev.end());
    a1.insert(a1.end(), gen_only.begin(), gen_only.end());
    int rc = run_child(a1);
    if (rc) die("device run failed (exit %d)", rc);
    const double t1 = now_s();

    std::vector<std::string> a2 = {self, "run", "--device", "cpu", "--tokens", fdev, "--out", fcpu};
    a2.insert(a2.end(), common.begin(), common.end());
    rc = run_child(a2);
    if (rc) die("CPU run failed (exit %d)", rc);
    const double t2 = now_s();

    for (int i = 0; i < n_floor; ++i) {
        const uint64_t s = seed + (uint64_t) i;
        std::vector<std::string> a3 = {self, "run", "--device", "cpu", "--tokens", fdev, "--perturb-ulp", std::to_string(s),
                                       "--out", ffloors[i]};
        a3.insert(a3.end(), common.begin(), common.end());
        rc = run_child(a3);
        if (rc) die("CPU floor run (seed %" PRIu64 ") failed (exit %d)", s, rc);
    }
    const double t3 = now_s();

    // The floor run feeds token_embd rows instead of token ids. Without the nudge that path must
    // reproduce the token path bit for bit, so the floor measures the 1-ulp nudge and nothing else.
    int rc_embd = 0;
    if (embd_check) {
        std::vector<std::string> a5 = {self, "run", "--device", "cpu", "--tokens", fdev, "--input", "embd", "--out", fembd};
        a5.insert(a5.end(), common.begin(), common.end());
        rc = run_child(a5);
        if (rc) die("CPU embd run failed (exit %d)", rc);
        rc_embd = run_child({self, "compare", "--ref", fcpu, "--dev", fembd, "--expect-identical", "-q"});
        fprintf(stderr, "klcheck all: embd input path vs token input path: %s\n",
                rc_embd == 0 ? "bit-identical (ok)" : "NOT bit-identical (FAIL: the floor is not a pure 1-ulp floor)");
    }
    const double t4 = now_s();

    std::vector<std::string> a4 = {self, "compare", "--ref", fcpu, "--dev", fdev, "--json", fjson};
    for (const std::string & f : ffloors) a4.insert(a4.end(), {"--floor", f});
    // the device run's kinds (cmd_run inherits this environment), recorded in report.json
    if (device != "cpu") a4.insert(a4.end(), {"--tsr-kinds", tsr_kinds_of(phase)});
    a4.insert(a4.end(), cmp.begin(), cmp.end());
    rc = run_child(a4);
    fprintf(stderr, "klcheck all: wall time device %.1f s, cpu %.1f s, floor %.1f s (%d run%s), embd check %.1f s, compare %.1f s\n",
            t1 - t0, t2 - t1, t3 - t2, n_floor, n_floor == 1 ? "" : "s", t4 - t3, now_s() - t4);
    if (tmp_dir) {
        for (const std::string & p : ffloors) unlink(p.c_str());
        for (const std::string & p : {fdev, fcpu, fembd, fjson}) unlink(p.c_str());
        rmdir(dir.c_str());
    } else {
        fprintf(stderr, "klcheck all: files in %s (report.json has every position)\n", dir.c_str());
    }
    if (rc == 0 && rc_embd != 0) rc = 1;
    return rc;
}

// ---------------------------------------------------------------------------------------------
// klcheck info

static int cmd_info(int argc, char ** argv) {
    std::string file, model;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) model = argv[++i];
        else if (file.empty() && a[0] != '-') file = a;
        else die("info: usage: klcheck info FILE [-m MODEL]");
    }
    if (file.empty()) die("info: usage: klcheck info FILE [-m MODEL]");
    klc_reader r;
    r.open(file);
    const klc_header & h = r.h;
    printf("%s: %s, device %s, %s mode", file.c_str(), fixed_str(h.model_desc, 128).c_str(), fixed_str(h.device, 16).c_str(),
           mode_name(h.mode));
    if (h.mode == KLC_MODE_REPLAY) printf(" (%s, from %s)", replay_name(h.replay), fixed_str(h.tokens_from, 64).c_str());
    printf(", input %s", input_name(h.input));
    if (h.input == KLC_INPUT_EMBD_ULP) printf(" seed %" PRIu64, h.perturb_seed);
    printf("\n  model %s, %" PRIu64 " params, %.2f GiB, n_layer %u, n_embd %u, n_vocab %u\n", fixed_str(h.model_file, 128).c_str(),
           h.model_n_params, h.model_size / 1073741824.0, h.n_layer, h.n_embd, h.n_vocab);
    printf("  %u/%u sequences, G %u, prompt tail %u, flash attn %s, threads %u, repack %s, chat %s\n", h.n_seq, h.n_seq_planned,
           h.n_gen, h.prompt_tail, h.flash_attn ? "on" : "off", h.n_threads, (h.flags & KLC_FLAG_REPACK) ? "on" : "off",
           (h.flags & KLC_FLAG_CHAT) ? "on" : "off");
    printf("  file version %u, %s\n", h.version, settings_str(h).c_str());
    printf("  %s\n", versions_str(h).c_str());
    printf("  load %.2f s, prefill %.2f s (%" PRIu64 " tok in batches), gen %.2f s (%" PRIu64 " single-token decodes%s)\n", h.t_load_s,
           h.t_prefill_s, h.n_prefill_tok, h.t_gen_s, h.n_gen_tok,
           h.n_gen_tok && h.t_gen_s > 0 ? (", " + std::to_string((int) (1e3 * h.t_gen_s / h.n_gen_tok)) + " ms each").c_str() : "");

    session s;
    if (!model.empty()) {
        setenv("TSR_DEVICE", "none", 1);
        llama_log_set(log_cb, nullptr);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        s.model = llama_model_load_from_file(model.c_str(), mp);
        if (!s.model) die("%s: failed to load the vocabulary", model.c_str());
        s.vocab = llama_model_get_vocab(s.model);
    }
    while (r.next()) {
        printf("  seq %u: %u prompt + %u gen tokens, rows %u from token %u, prefill %.2f s, gen %.2f s, %u EOG\n", r.seq.index,
               r.seq.n_prompt, r.seq.n_gen, r.seq.n_rows, r.seq.row0, r.seq.t_prefill_s, r.seq.t_gen_s, r.seq.n_eog);
        if (s.vocab) {
            printf("    prompt: %s\n    output: %s\n", one_line(detok(s, r.toks.data(), r.seq.n_prompt), 400).c_str(),
                   one_line(detok(s, r.toks.data() + r.seq.n_prompt, r.seq.n_gen), 400).c_str());
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// klcheck embd — diagnostic: the token_embd rows the --input embd / --perturb-ulp runs feed

static int cmd_embd(int argc, char ** argv) {
    if (argc < 2) die("embd: usage: klcheck embd MODEL TOKEN... [--perturb-ulp SEED --seq S --pos P]");
    embd_table t;
    t.open_model(argv[0]);
    bool perturb = false;
    uint64_t seed = 0;
    uint32_t seq = 0, pos = 0;
    std::vector<int32_t> toks;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--perturb-ulp" && i + 1 < argc) { perturb = true; seed = parse_u64("--perturb-ulp", argv[++i]); }
        else if (a == "--seq" && i + 1 < argc) seq = (uint32_t) parse_int("--seq", argv[++i], 0, UINT32_MAX);
        else if (a == "--pos" && i + 1 < argc) pos = (uint32_t) parse_int("--pos", argv[++i], 0, UINT32_MAX);
        else toks.push_back((int32_t) parse_int("TOKEN", argv[i], 0, INT32_MAX));
    }
    printf("token_embd.weight: %s [%" PRId64 ", %" PRId64 "] in %s, token-row scale %.9g\n", ggml_type_name(t.type), t.n_embd,
           t.n_rows, t.file.c_str(), t.tok_scale);
    for (int32_t tok : toks) {
        const float * r = t.row(tok);
        const uint64_t key = embd_nudge_key(seed, seq, pos);
        double sum = 0;
        for (int64_t j = 0; j < t.n_embd; ++j) sum += perturb ? embd_nudge(r[j], key, (uint32_t) j) : r[j];
        printf("token %d: sum %.9g, first:", tok, sum);
        for (int j = 0; j < 8 && j < t.n_embd; ++j) printf(" %.9g", perturb ? embd_nudge(r[j], key, (uint32_t) j) : r[j]);
        printf("\n");
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// klcheck selftest — file format, resume, perturbation and the compare arithmetic (no model)

static int g_fail = 0;
#define ST_CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "selftest FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

static klc_header st_header(uint32_t n_vocab, uint32_t planned, const char * dev, uint32_t mode) {
    klc_header h{};
    memcpy(h.magic, KLC_MAGIC, 8);
    h.version = KLC_VERSION;
    h.header_bytes = sizeof(klc_header);
    h.seq_bytes = sizeof(klc_seq);
    h.n_vocab = n_vocab;
    h.n_seq_planned = planned;
    h.n_gen = 5;
    h.prompt_tail = 2;
    h.mode = mode;
    copy_str(h.device, sizeof(h.device), dev);
    copy_str(h.model_desc, sizeof(h.model_desc), "selftest");
    h.flash_attn = 1;
    h.type_k = h.type_v = GGML_TYPE_Q8_0;
    h.attn_rot_disable = 1;
    h.phase = KLC_PHASE_M6;
    return h;
}

static int cmd_selftest(int argc, char ** argv) {
    std::string dir = argc > 0 ? argv[0] : "";
    if (dir.empty()) {
        const char * t = getenv("TMPDIR");
        dir = std::string(t ? t : "/tmp") + "/klcheck-selftest-" + std::to_string(getpid());
    }
    mkdir_p(dir);

    // 1. perturbation: exactly one ulp, both directions, deterministic, balanced
    {
        uint64_t up = 0, n = 0;
        const float xs[] = {1.0f, -1.0f, 0.0f, -0.0f, 3.1415926f, -2.5e-3f, 1e-30f, 65504.0f, 0.03125f};
        for (uint32_t pos = 0; pos < 64; ++pos) {
            const uint64_t key = embd_nudge_key(42, 3, pos);
            ST_CHECK(key == embd_nudge_key(42, 3, pos), "key not deterministic");
            ST_CHECK(key != embd_nudge_key(43, 3, pos) && key != embd_nudge_key(42, 4, pos), "key ignores seed/seq");
            for (uint32_t j = 0; j < 1024; ++j) {
                const float x = xs[j % 9] * (1.0f + j * 1e-3f);
                const float y = embd_nudge(x, key, j);
                ST_CHECK(y == nextafterf(x, INFINITY) || y == nextafterf(x, -INFINITY), "not one ulp: %a -> %a", x, y);
                ST_CHECK(y != x, "unchanged: %a", x);
                ST_CHECK(y == embd_nudge(x, key, j), "not deterministic");
                up += y > x;
                n++;
            }
        }
        const double frac = (double) up / n;
        ST_CHECK(frac > 0.48 && frac < 0.52, "direction not balanced: %.4f up", frac);
    }

    // 2. files: ref, dev = ref (+ noise in seq 1), floor = ref + smaller noise
    const uint32_t V = 257, NS = 3;
    const std::string fr = dir + "/ref.klc", fd = dir + "/dev.klc", ff = dir + "/floor.klc", fi = dir + "/ident.klc";
    std::vector<std::vector<int32_t>> toks(NS);
    std::vector<std::vector<float>> lref(NS), ldev(NS), lflr(NS);
    std::vector<klc_seq> seqs(NS);
    uint64_t rng = 12345;
    auto urand = [&]() { rng = splitmix64(rng); return (double) (rng >> 11) / 9007199254740992.0; };
    for (uint32_t s = 0; s < NS; ++s) {
        klc_seq q{};
        memcpy(q.magic, KLC_SEQ_MAGIC, 4);
        q.index = s;
        q.n_prompt = 4 + s;
        q.n_gen = 5;
        q.row0 = q.n_prompt - 2;
        q.n_rows = 2 + q.n_gen - 1;
        seqs[s] = q;
        for (uint32_t i = 0; i < q.n_prompt + q.n_gen; ++i) toks[s].push_back((int32_t) (urand() * V));
        for (uint32_t i = 0; i < q.n_rows * V; ++i) {
            const float x = (float) (urand() * 8 - 4);
            lref[s].push_back(x);
            ldev[s].push_back(s == 1 ? x + (float) (urand() - 0.5) * 0.2f : x);
            lflr[s].push_back(x + (float) (urand() - 0.5) * 0.02f);
        }
    }
    // two more floor runs for the several-floor compare (5b): seed 2 with less noise than floor.klc
    // (seed 1), seed 3 with more
    const std::string ff2 = dir + "/floor-seed2.klc", ff3 = dir + "/floor-seed3.klc";
    std::vector<std::vector<float>> lflr2(NS), lflr3(NS);
    for (uint32_t s = 0; s < NS; ++s) {
        for (size_t i = 0; i < lref[s].size(); ++i) {
            lflr2[s].push_back(lref[s][i] + (float) (urand() - 0.5) * 0.005f);
            lflr3[s].push_back(lref[s][i] + (float) (urand() - 0.5) * 0.06f);
        }
    }
    auto floor_header = [&](uint64_t seed) {
        klc_header h = st_header(V, NS, "cpu", KLC_MODE_REPLAY);
        h.input = KLC_INPUT_EMBD_ULP;
        h.perturb_seed = seed;
        return h;
    };
    {
        klc_writer a, b, c, c2, c3;
        a.create(fr, st_header(V, NS, "cpu", KLC_MODE_REPLAY));
        b.create(fd, st_header(V, NS, "sim", KLC_MODE_REPLAY));   // replay: no greedy-token consistency check
        c.create(ff, floor_header(1));
        c2.create(ff2, floor_header(2));
        c3.create(ff3, floor_header(3));
        for (uint32_t s = 0; s < NS; ++s) {
            a.write_seq(seqs[s], toks[s], lref[s].data());
            b.write_seq(seqs[s], toks[s], ldev[s].data());
            c.write_seq(seqs[s], toks[s], lflr[s].data());
            c2.write_seq(seqs[s], toks[s], lflr2[s].data());
            c3.write_seq(seqs[s], toks[s], lflr3[s].data());
        }
    }
    // 3. resume: two records, then a torn third record; resume keeps two and appends the third
    {
        klc_writer w;
        w.create(fi, st_header(V, NS, "cpu", KLC_MODE_REPLAY));
        w.write_seq(seqs[0], toks[0], lref[0].data());
        w.write_seq(seqs[1], toks[1], lref[1].data());
        fwrite(&seqs[2], sizeof(klc_seq), 1, w.f);
        fwrite(lref[2].data(), 4, 100, w.f);
        fflush(w.f);
    }
    {
        klc_writer w;
        const uint32_t kept = w.resume(fi);
        ST_CHECK(kept == 2, "resume kept %u records", kept);
        w.write_seq(seqs[2], toks[2], lref[2].data());
    }
    {
        klc_reader r;
        r.open(fi);
        uint32_t n = 0;
        std::vector<float> row(V);
        while (r.next()) {
            ST_CHECK(r.toks == toks[n], "tokens differ after resume");
            for (uint32_t k = 0; k < r.seq.n_rows; ++k) {
                r.read_row(row.data());
                ST_CHECK(memcmp(row.data(), lref[n].data() + (size_t) k * V, V * 4) == 0, "row differs after resume");
            }
            n++;
        }
        ST_CHECK(n == NS, "read %u records after resume", n);
    }

    // 4. compare: identical files
    {
        cmp_opts o;
        o.ref = fr; o.dev = fi; o.expect_identical = true; o.quiet = true;
        cmp_result R;
        ST_CHECK(compare_files(o, R) == 0, "identical files do not pass --expect-identical");
        ST_CHECK(R.dev.sum_kl == 0 && R.dev.ident == R.dev.n && R.dev.t1() == 1.0, "identical: kl %g", R.dev.sum_kl);
    }
    // 5. compare: KL against an independent long-double evaluation, per-row and aggregated
    {
        cmp_opts o;
        o.ref = fr; o.dev = fd; o.floors = {ff}; o.quiet = true; o.json = dir + "/report.json";
        cmp_result R;
        compare_files(o, R);
        ST_CHECK(R.rows.size() == (size_t) NS * seqs[0].n_rows, "rows %zu", R.rows.size());
        long double sum_d = 0, sum_f = 0;
        size_t k = 0;
        for (uint32_t s = 0; s < NS; ++s) {
            for (uint32_t r = 0; r < seqs[s].n_rows; ++r, ++k) {
                const float * x = lref[s].data() + (size_t) r * V;
                long double zr = 0, zd = 0, zf = 0;
                for (uint32_t i = 0; i < V; ++i) {
                    zr += expl((long double) x[i]);
                    zd += expl((long double) ldev[s][(size_t) r * V + i]);
                    zf += expl((long double) lflr[s][(size_t) r * V + i]);
                }
                long double kd = 0, kf = 0;
                for (uint32_t i = 0; i < V; ++i) {
                    const long double p = expl((long double) x[i]) / zr;
                    kd += p * (logl(p) - logl(expl((long double) ldev[s][(size_t) r * V + i]) / zd));
                    kf += p * (logl(p) - logl(expl((long double) lflr[s][(size_t) r * V + i]) / zf));
                }
                ST_CHECK(fabsl(kd - R.rows[k].kl_dev) <= 1e-12L + 1e-9L * kd, "row %zu kl_dev %.12Lg vs %.12g", k, kd, R.rows[k].kl_dev);
                ST_CHECK(fabsl(kf - R.rows[k].kl_floor) <= 1e-12L + 1e-9L * kf, "row %zu kl_floor %.12Lg vs %.12g", k, kf, R.rows[k].kl_floor);
                ST_CHECK((s == 1) == !R.rows[k].ident_dev, "row %zu identity flag", k);
                sum_d += kd;
                sum_f += kf;
            }
        }
        ST_CHECK(fabsl(sum_d / k - R.dev.mean()) <= 1e-12L, "mean kl_dev");
        ST_CHECK(fabsl(sum_f / k - R.floor.mean()) <= 1e-12L, "mean kl_floor");
        ST_CHECK(R.dev.ident == 2 * seqs[0].n_rows, "bit-identical rows %" PRIu64, R.dev.ident);
        const bool want = R.dev.mean() <= 3 * R.floor.mean() && R.dev.t1() >= 0.8;
        ST_CHECK(R.pass == want, "verdict %d, expected %d", R.pass, want);
        // a stricter / looser ratio flips the verdict as expected
        cmp_opts o2 = o;
        o2.json.clear();
        o2.kl_ratio = R.ratio * 1.01;
        o2.min_top1 = 0;
        cmp_result R2;
        ST_CHECK(compare_files(o2, R2) == 0, "ratio just above the measured one must pass");
        o2.kl_ratio = R.ratio * 0.99;
        cmp_result R3;
        ST_CHECK(compare_files(o2, R3) == 1, "ratio just below the measured one must fail");
        ST_CHECK(file_exists(o.json), "no json report");
    }
    // 5b. several floor runs (klcheck all --floor-seeds): each run's mean KL and ratio equal a compare
    //     against that run alone, the floor KL is the mean of the runs' mean KL, the verdict follows
    //     that mean, and the report lists every run; a single floor's report keeps its fields
    {
        const std::vector<std::string> fls = {ff, ff2, ff3};
        std::vector<cmp_result> one(fls.size());
        for (size_t i = 0; i < fls.size(); ++i) {
            cmp_opts o;
            o.ref = fr; o.dev = fd; o.floors = {fls[i]}; o.quiet = true;
            compare_files(o, one[i]);
        }
        cmp_opts o;
        o.ref = fr; o.dev = fd; o.floors = fls; o.quiet = true; o.json = dir + "/report3.json"; o.tsr_kinds = "all,-RMS_NORM";
        cmp_result R;
        compare_files(o, R);
        double sum = 0;
        for (size_t i = 0; i < fls.size(); ++i) {
            ST_CHECK(R.floor_seed[i] == i + 1, "floor run %zu: seed %" PRIu64, i, R.floor_seed[i]);
            ST_CHECK(R.floor_run[i].n == one[i].floor.n && R.floor_run[i].mean() == one[i].floor.mean(),
                     "floor run %zu: mean KL %.12g, alone %.12g", i, R.floor_run[i].mean(), one[i].floor.mean());
            ST_CHECK(R.floor_ratio[i] == one[i].ratio, "floor run %zu: ratio %.12g, alone %.12g", i, R.floor_ratio[i], one[i].ratio);
            sum += one[i].floor.mean();
        }
        const double mean = sum / fls.size();
        ST_CHECK(one[1].floor.mean() < one[0].floor.mean() && one[0].floor.mean() < one[2].floor.mean(), "floor noise order");
        ST_CHECK(R.floor_kl == mean && R.ratio == R.dev.mean() / mean, "floor KL %.12g (mean of the runs %.12g), ratio %.12g",
                 R.floor_kl, mean, R.ratio);
        ST_CHECK(R.floor.n == fls.size() * R.dev.n && fabs(R.floor.mean() - mean) <= 1e-12 * mean, "pooled floor rows: n %" PRIu64,
                 R.floor.n);
        bool rows_ok = R.rows.size() == one[0].rows.size();
        for (size_t k = 0; rows_ok && k < R.rows.size(); ++k) {
            const row_stat & r = R.rows[k];
            double s = 0;
            int t1 = 0;
            rows_ok = r.kl_floors.size() == fls.size() && r.am_floor == one[0].rows[k].am_floor;
            for (size_t i = 0; rows_ok && i < fls.size(); ++i) {
                rows_ok = r.kl_floors[i] == one[i].rows[k].kl_floor;
                s += r.kl_floors[i];
                t1 += one[i].rows[k].top1_floor;
            }
            rows_ok = rows_ok && r.kl_floor == s / fls.size() && r.top1_floor == t1;
            ST_CHECK(rows_ok, "row %zu: per-run floor KL", k);
        }
        // the verdict follows the mean: a limit just above the ratio to the mean passes, although it is
        // far below the ratio to the smallest floor (seed 2)
        ST_CHECK(R.floor_ratio[1] > 2 * R.ratio, "ratio to seed 2 %.4g vs to the mean %.4g", R.floor_ratio[1], R.ratio);
        cmp_opts o2 = o;
        o2.json.clear();
        o2.min_top1 = 0;
        o2.kl_ratio = R.ratio * 1.01;
        cmp_result R2;
        ST_CHECK(compare_files(o2, R2) == 0, "several floors: ratio just above the measured one must pass");
        o2.kl_ratio = R.ratio * 0.99;
        cmp_result R3;
        ST_CHECK(compare_files(o2, R3) == 1, "several floors: ratio just below the measured one must fail");
        // the command line takes --floor once per run
        char kr_pass[32], kr_fail[32];
        snprintf(kr_pass, sizeof(kr_pass), "%.17g", R.ratio * 1.01);
        snprintf(kr_fail, sizeof(kr_fail), "%.17g", R.ratio * 0.99);
        for (const char * kr : {kr_pass, kr_fail}) {
            std::vector<std::string> args = {"--ref", fr, "--dev", fd, "--floor", ff, "--floor", ff2, "--floor", ff3,
                                             "--kl-ratio", kr, "--min-top1", "0", "-q"};
            std::vector<char *> av;
            for (std::string & a : args) av.push_back(&a[0]);
            ST_CHECK(cmd_compare((int) av.size(), av.data()) == (kr == kr_pass ? 0 : 1), "compare --floor x3 --kl-ratio %s", kr);
        }
        // the reports: the several-floor one lists the runs and the kinds; the single-floor one of 5
        // has the same fields, no per-row kl_floors
        auto slurp = [](const std::string & p) {
            std::string t;
            FILE * f = fopen(p.c_str(), "rb");
            if (!f) return t;
            char b[4096];
            size_t n;
            while ((n = fread(b, 1, sizeof(b), f)) > 0) t.append(b, n);
            fclose(f);
            return t;
        };
        auto count = [](const std::string & t, const std::string & k) {
            size_t n = 0;
            for (size_t p = t.find(k); p != std::string::npos; p = t.find(k, p + 1)) n++;
            return n;
        };
        const std::string j3 = slurp(o.json), j1 = slurp(dir + "/report.json");
        ST_CHECK(count(j3, "\"floors\": [") == 1 && count(j3, "\"seed\": ") == 3 && count(j3, "\"n_floor\": 3,") == 1 &&
                 count(j3, "\"tsr_kinds\": \"all,-RMS_NORM\"") == 1 && count(j3, "\"kl_floors\": [") == R.rows.size() &&
                 count(j3, "\"floor\": " + json_str(ff)) == 1, "several-floor report.json fields");
        ST_CHECK(count(j1, "\"seed\": 1,") == 1 && count(j1, "\"n_floor\": 1,") == 1 && count(j1, "\"tsr_kinds\": null") == 1 &&
                 count(j1, "kl_floors") == 0 && count(j1, "\"kl_floor\": ") == R.rows.size(), "single-floor report.json fields");
        // a floor of KL 0 (the CPU run itself) against a device with KL > 0: the ratio is infinite and
        // is written as null, at the top level and in floors[] (JSON has no inf)
        {
            cmp_opts o0;
            o0.ref = fr; o0.dev = fd; o0.floors = {fr}; o0.quiet = true; o0.json = dir + "/report0.json";
            cmp_result R0;
            ST_CHECK(compare_files(o0, R0) == 1, "a floor of KL 0 must fail");
            ST_CHECK(R0.floor_kl == 0 && std::isinf(R0.ratio) && std::isinf(R0.floor_ratio[0]), "floor KL %g, ratio %g", R0.floor_kl, R0.ratio);
            const std::string j0 = slurp(o0.json);
            ST_CHECK(count(j0, "\"ratio\": null") == 2 && count(j0, "\"ratio\": inf") == 0 && count(j0, "\"ratio\": nan") == 0,
                     "infinite ratio in report.json");
        }
        // klcheck all's floor files
        ST_CHECK(all_floor_files("d", 1, 1) == std::vector<std::string>{"d/floor.klc"} &&
                 all_floor_files("d", 5, 3) == (std::vector<std::string>{"d/floor-seed5.klc", "d/floor-seed6.klc", "d/floor-seed7.klc"}),
                 "klcheck all floor file names");
    }
    // 6. incomplete files fail
    {
        klc_writer w;
        w.create(fi, st_header(V, NS, "cpu", KLC_MODE_REPLAY));
        w.write_seq(seqs[0], toks[0], lref[0].data());
    }
    {
        cmp_opts o;
        o.ref = fi; o.dev = fd; o.floors = {ff}; o.quiet = true; o.kl_ratio = 1e9; o.min_top1 = 0;
        cmp_result R;
        ST_CHECK(compare_files(o, R) == 1 && !R.complete, "incomplete reference must fail");
    }
    // 7. header versions: the settings survive a write / read; a version 1 file (reserved bytes
    //    zero) reads with the FA-off settings, and a resume check against the current options sees the difference
    {
        klc_header h = st_header(V, NS, "sim", KLC_MODE_GENERATE);
        h.phase = KLC_PHASE_M7;
        h.offload_kqv = 1;
        h.type_v = GGML_TYPE_F16;
        {
            klc_writer w;
            w.create(fi, h);
        }
        FILE * f = fopen(fi.c_str(), "rb");
        klc_header g;
        std::string err;
        ST_CHECK(f && klc_read_header(f, fi, &g, &err), "version 2 header read: %s", err.c_str());
        if (f) fclose(f);
        ST_CHECK(g.version == 2 && g.phase == KLC_PHASE_M7 && g.offload_kqv == 1 && g.type_k == GGML_TYPE_Q8_0 &&
                 g.type_v == GGML_TYPE_F16 && g.attn_rot_disable == 1 && g.flash_attn == 1, "settings round trip: %s",
                 settings_str(g).c_str());
        ST_CHECK(settings_str(g) == "M7 settings: flash attn on, KV q8_0/f16 on the device (offload_kqv 1), KV rotation off "
                                    "(LLAMA_ATTN_ROT_DISABLE=1)", "settings string: %s", settings_str(g).c_str());
        ST_CHECK(swa_cache_of(g) == KLC_SWA_FULL, "a header without the SWA field reads as the full-size SWA cache");
        ST_CHECK(g.basis == 0 && g.core == 0 && g.kinds_sig == 0 && versions_str(g) == "versions not recorded",
                 "a header without the version fields: %s", versions_str(g).c_str());
        h.swa_cache = KLC_SWA_RING;
        h.basis = 0x000100u;          // SW §6.9: the versions of a device run survive a write / read
        h.core = 0x0100u;
        h.kinds_sig = 0xFA6Fu;
        {
            klc_writer w;
            w.create(fi, h);
        }
        f = fopen(fi.c_str(), "rb");
        ST_CHECK(f && klc_read_header(f, fi, &g, &err), "version 2 header (SWA ring) read: %s", err.c_str());
        if (f) fclose(f);
        ST_CHECK(g.swa_cache == KLC_SWA_RING && settings_str(g) == "M7 settings: flash attn on, KV q8_0/f16 on the device "
                 "(offload_kqv 1), KV rotation off (LLAMA_ATTN_ROT_DISABLE=1), SWA cache ring", "SWA field round trip: %s",
                 settings_str(g).c_str());
        ST_CHECK(g.basis == 0x000100u && g.core == 0x0100u && g.kinds_sig == 0xFA6Fu &&
                 versions_str(g) == "Tesseract Core 1.0, Tesseract Basis 0.1.0 (kind table 0xfa6f)", "versions round trip: %s",
                 versions_str(g).c_str());
        klc_header v1 = st_header(V, NS, "sim", KLC_MODE_GENERATE);
        v1.version = 1;
        v1.flash_attn = 0;
        v1.type_k = v1.type_v = 0;
        v1.attn_rot_disable = v1.phase = 0;
        {
            klc_writer w;
            w.create(fi, v1);
            w.write_seq(seqs[0], toks[0], lref[0].data());
        }
        f = fopen(fi.c_str(), "rb");
        ST_CHECK(f && klc_read_header(f, fi, &g, &err), "version 1 header read: %s", err.c_str());
        if (f) fclose(f);
        ST_CHECK(g.version == 1 && g.phase == KLC_PHASE_V1 && g.type_k == GGML_TYPE_F16 && g.type_v == GGML_TYPE_F16 &&
                 g.offload_kqv == 0 && g.attn_rot_disable == 0, "version 1 file reads with the FA-off settings: %s", settings_str(g).c_str());
        klc_reader r;
        r.open(fi);
        ST_CHECK(r.next() && r.toks == toks[0], "version 1 file: record readable");
        klc_header v3 = v1;
        v3.version = 3;
        {
            klc_writer w;
            w.create(fi, v3);
        }
        f = fopen(fi.c_str(), "rb");
        ST_CHECK(f && !klc_read_header(f, fi, &g, &err), "version 3 refused");
        if (f) fclose(f);
        ST_CHECK(parse_cache_type("-ctk", "q8_0") == GGML_TYPE_Q8_0 && parse_cache_type("-ctv", "f16") == GGML_TYPE_F16,
                 "cache type names");
        ST_CHECK(device_kind_ok("mock") && device_kind_ok("vkernel") && !device_kind_ok("none"), "device kinds");
    }
    for (const std::string & p : {fr, fd, ff, ff2, ff3, fi, dir + "/report.json", dir + "/report3.json", dir + "/report0.json"}) unlink(p.c_str());
    rmdir(dir.c_str());
    printf("klcheck selftest: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------

static void usage() {
    fprintf(stderr,
        "klcheck — model-level KL check against the ggml CPU backend (docs/klcheck.md)\n"
        "  klcheck run -m MODEL --device sim|mock|vkernel|xrt|cpu [--tokens FILE] [--perturb-ulp SEED] --out FILE [...]\n"
        "  klcheck compare --ref CPU.klc --dev DEV.klc --floor FLOOR.klc [--floor FLOOR2.klc ...] [--json FILE] [...]\n"
        "  klcheck all -m MODEL [-o DIR] [--device sim] [-n G] [--n-prompts N] [--prompts FILE] [--phase m6|m7]\n"
        "              [--seed S] [--floor-seeds N (default 3)] [...]\n"
        "  klcheck info FILE [-m MODEL]\n"
        "  klcheck selftest [DIR]\n"
        "  klcheck embd MODEL TOKEN... [--perturb-ulp SEED --seq S --pos P]   (diagnostic)\n"
        "'klcheck run -h' lists the run options.\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string cmd = argv[1];
    if (cmd == "run")      return cmd_run(argc - 2, argv + 2);
    if (cmd == "compare")  return cmd_compare(argc - 2, argv + 2);
    if (cmd == "all")      return cmd_all(argc - 2, argv + 2, argv[0]);
    if (cmd == "info")     return cmd_info(argc - 2, argv + 2);
    if (cmd == "selftest") return cmd_selftest(argc - 2, argv + 2);
    if (cmd == "embd")     return cmd_embd(argc - 2, argv + 2);
    if (cmd == "-h" || cmd == "--help") { usage(); return 0; }
    usage();
    return 2;
}

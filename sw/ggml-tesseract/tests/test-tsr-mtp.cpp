// test-tsr-mtp.cpp — ctest ggml_tesseract_mtp_* (docs/design/06-software-verification.md §7.11, SW-83): the verification
// batch of multi-token prediction on the device. A context with n_rs_seq = K keeps K + 1 recurrent snapshots, so one decode of
// [id_last, d_0 .. d_{K-1}] returns the logits of K + 1 positions and leaves the Gated DeltaNet state and the convolution state
// of every one of the last K + 1 tokens in the cache's rollback groups. The test drives llama.cpp's own protocol (llama_decode
// of the batch with every output flagged, llama_memory_seq_rm of the rejected tail, the next decode reading the snapshot the
// removal selected) with drafts that are the true continuation up to a chosen position and a wrong token after it, so that
// every accepted length 0 .. K and every rollback runs, on a model that has no MTP head to draw drafts from.
//
// It checks, on the device (the simulator or the card, TSR_DEVICE):
//   - the logits of every row of a verification batch against the logits of plain decoding on the same device (one token per
//     decode; the device's numerics per token do not depend on T) and against the CPU backend (cosine, top-1);
//   - after a rollback, the next steps against the same reference: the rollback restored the state that plain decoding
//     would have had (for the recurrent state: bit for bit, which is what the logits of the following rows show);
//   - that the whole verification graph is one launch (TSR_STATS-free: the backend's counters hook).
//   test-tsr-mtp <model.gguf> [K] [steps]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);     \
            fprintf(stderr, __VA_ARGS__);                                         \
            fprintf(stderr, "\n");                                                \
            g_fail++;                                                             \
        }                                                                         \
    } while (0)

typedef bool (*counters_fn)(ggml_backend_dev_t, uint64_t *, size_t);

static llama_context * make_ctx(llama_model * m, bool device, uint32_t n_rs_seq) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx = 512;
    p.n_batch = 64;
    p.n_ubatch = 16;
    p.n_threads = 4;
    p.n_threads_batch = 4;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.type_k = GGML_TYPE_Q8_0;
    p.type_v = GGML_TYPE_Q8_0;
    p.offload_kqv = device;
    p.n_rs_seq = n_rs_seq;
    return llama_init_from_model(m, p);
}

// decode tokens at positions pos0 .. with the logits of every row
static int decode_rows(llama_context * ctx, const std::vector<llama_token> & toks, int pos0) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    b.n_tokens = (int) toks.size();
    for (int i = 0; i < b.n_tokens; ++i) {
        b.token[i] = toks[(size_t) i];
        b.pos[i] = pos0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static std::vector<float> row_of(llama_context * ctx, int i, int nv) {
    const float * l = llama_get_logits_ith(ctx, i);
    return std::vector<float>(l, l + nv);
}

static int argmax(const std::vector<float> & v) {
    return (int) (std::max_element(v.begin(), v.end()) - v.begin());
}

static double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / std::sqrt(na * nb);
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test-tsr-mtp <model.gguf> [K] [steps]\n");
        return 2;
    }
    const int K = argc > 2 ? atoi(argv[2]) : 3;
    const int steps = argc > 3 ? atoi(argv[3]) : 12;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * dev_model = llama_model_load_from_file(argv[1], mp);
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t only_cpu[2] = { cpu, nullptr };
    llama_model_params mc = llama_model_default_params();
    mc.n_gpu_layers = 0;
    mc.devices = only_cpu;
    llama_model * cpu_model = llama_model_load_from_file(argv[1], mc);
    if (dev_model == nullptr || cpu_model == nullptr) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 2;
    }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(cpu_model));
    ggml_backend_dev_t tdev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (strncmp(ggml_backend_dev_name(ggml_backend_dev_get(i)), "Tesseract", 9) == 0) {
            tdev = ggml_backend_dev_get(i);
        }
    }
    counters_fn counters = tdev ? (counters_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(tdev), "ggml_backend_tesseract_counters") : nullptr;
    auto launches = [&]() {
        uint64_t c[8] = {};
        return counters && counters(tdev, c, 8) ? c[6] : 0;
    };

    // prompt lengths: shorter than K + 1, one ubatch, several ubatches (n_ubatch 16)
    const int lens[] = { 3, 9, 37 };
    for (int len : lens) {
        std::vector<llama_token> prompt((size_t) len);
        for (int i = 0; i < len; ++i) {
            prompt[(size_t) i] = (llama_token) ((i * 7 + 3 + len) % std::max(nv - 4, 1) + 2);
        }
        // the reference: plain decoding on the device (one token per decode), the true continuation and every row's logits
        llama_context * ref = make_ctx(dev_model, true, 0);
        llama_context * cc = make_ctx(cpu_model, false, 0);
        llama_context * sp = make_ctx(dev_model, true, (uint32_t) K);
        CHECK(ref && cc && sp, "contexts");
        if (!ref || !cc || !sp) return 2;
        // a target without recurrent layers keeps no snapshots (llama.cpp clamps n_rs_seq to 0) and rolls back by KV removal
        const bool recurrent = llama_model_is_recurrent(dev_model) || llama_model_is_hybrid(dev_model);
        CHECK(llama_n_rs_seq(sp) == (recurrent ? (uint32_t) K : 0u), "n_rs_seq %u", llama_n_rs_seq(sp));
        std::vector<llama_token> seq = prompt;                       // prompt, then the plain continuation
        std::vector<std::vector<float>> ref_rows;                    // logits after seq[0 .. i] for i >= len - 1
        std::vector<std::vector<float>> cpu_rows;
        {
            std::vector<llama_token> head(prompt.begin(), prompt.end() - 1);
            if (!head.empty()) {
                CHECK(decode_rows(ref, head, 0) == 0 && decode_rows(cc, head, 0) == 0, "prefill");
            }
            int pos = (int) head.size();
            llama_token cur = prompt.back();
            for (int s = 0; s < steps * (K + 1) + 2; ++s) {
                CHECK(decode_rows(ref, { cur }, pos) == 0 && decode_rows(cc, { cur }, pos) == 0, "plain decode");
                ref_rows.push_back(row_of(ref, 0, nv));
                cpu_rows.push_back(row_of(cc, 0, nv));
                cur = (llama_token) argmax(ref_rows.back());
                seq.push_back(cur);
                ++pos;
            }
        }
        // the speculative context: the prompt but its last token in one go (n_batch 64, ubatches of 16: more than K + 1 tokens
        // in a ubatch, so the snapshots cover only the trailing K + 1), then verification batches
        {
            std::vector<llama_token> head(prompt.begin(), prompt.end() - 1);
            if (!head.empty()) CHECK(decode_rows(sp, head, 0) == 0, "prefill (n_rs_seq %d)", K);
        }
        int pos = len - 1;                    // position of id_last = seq[pos]
        int n_equal_rows = 0, n_rows = 0, tmiss = 0, hist[16] = {};
        double worst_cos_cpu = 1.0;
        const uint64_t l0 = launches();
        uint64_t n_verify = 0;
        for (int s = 0; s < steps && pos + K + 1 < (int) seq.size(); ++s) {
            const int nd = (s % 4 == 3 && K > 1) ? K - 1 : K;                      // a shorter draft now and then
            const int p = s % (nd + 1);                                            // accepted drafts: 0 .. nd
            std::vector<llama_token> batch = { seq[(size_t) pos] };
            for (int i = 0; i < nd; ++i) {
                llama_token d = seq[(size_t) (pos + 1 + i)];
                if (i >= p && i == p) d = (llama_token) ((d + 5 + s) % nv);        // the first wrong draft; the rest after it is as good as wrong
                else if (i > p) d = (llama_token) ((d + 11 + s) % nv);
                batch.push_back(d);
            }
            const uint64_t a = launches();
            CHECK(decode_rows(sp, batch, pos) == 0, "verify decode (step %d, T %zu)", s, batch.size());
            const uint64_t b = launches();
            n_verify++;
            if (counters && b - a != 1) {
                CHECK(false, "step %d: the verification took %llu launches, want 1", s, (unsigned long long) (b - a));
            }
            // rows 0 .. p are the logits of contexts that plain decoding had too: after seq[0 .. pos + i]
            int acc = 0;
            for (int i = 0; i <= nd; ++i) {
                const std::vector<float> r = row_of(sp, i, nv);
                if (i > p) break;                                                  // behind the first wrong draft: not a plain context
                const size_t ri = (size_t) (pos + i - (len - 1));
                CHECK(ri < ref_rows.size(), "reference row");
                if (ri >= ref_rows.size()) break;
                const bool eq = same_bits(r, ref_rows[ri]);
                n_equal_rows += eq ? 1 : 0;
                n_rows++;
                const double cs = cosine(r, ref_rows[ri]);
                worst_cos_cpu = std::min(worst_cos_cpu, cosine(r, cpu_rows[ri]));
                tmiss += argmax(r) != argmax(ref_rows[ri]) ? 1 : 0;
                CHECK(cs > 0.99999 || eq, "step %d row %d (T %zu, accepted %d): cosine %.8f against plain decoding", s, i, batch.size(), p, cs);
                acc = i;
            }
            hist[std::min(p, 15)]++;
            // the target accepts the drafts before the wrong one and takes its own token there; roll the rest back
            const int n_acc_tokens = p + 1;                                        // id_last and p drafts stay: positions pos .. pos + p
            llama_memory_seq_rm(llama_get_memory(sp), 0, pos + n_acc_tokens, -1);
            pos += n_acc_tokens;
            (void) acc;
        }
        (void) l0;
        printf("  prompt %2d tokens, K %d: %llu verify decodes, %d of %d rows bit-identical to plain decoding, top-1 misses %d, "
               "worst cosine against the CPU %.6f, accepted-length histogram", len, K, (unsigned long long) n_verify, n_equal_rows, n_rows,
               tmiss, worst_cos_cpu);
        for (int i = 0; i <= K; ++i) printf(" %d:%d", i, hist[i]);
        printf("\n");
        CHECK(tmiss == 0, "prompt %d: %d top-1 misses against plain decoding", len, tmiss);
        // the tiny Gemma 4 (soft-capped logits, Q8_0 on the device against the CPU's rounding) sits a little lower than the Qwen ones
        CHECK(worst_cos_cpu >= (recurrent ? 0.9995 : 0.995), "prompt %d: cosine %.6f against the CPU", len, worst_cos_cpu);
        llama_free(ref);
        llama_free(cc);
        llama_free(sp);
    }
    llama_model_free(dev_model);
    llama_model_free(cpu_model);
    llama_backend_free();
    if (g_fail) {
        fprintf(stderr, "test-tsr-mtp: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-mtp: OK\n");
    return 0;
}

// test-tsr-gdn-logits.cpp — ctest ggml_tesseract_gdn_logits (docs/design/06-software-verification.md §7.9
// "Gated DeltaNet", SW-60): the logits of a Gated DeltaNet model (llama.cpp's qwen35 graph: the conv state, the
// SSM_CONV group, the recurrent state in a device cache) on the device against the CPU, for short prompts and
// single-token steps. klcheck's KL criterion has a floor and a mean over 24 rows; a wrong hidden state of a
// random tiny model can hide inside it. This test checks the logits of every step directly: cosine and the
// largest difference against the logit spread. It exists because ggml-alloc frees the CONCAT's inputs after the
// CONCAT: an SSM_CONV record that read them later got zeros when a node in between reused their memory (found
// on the 2B model with prompts of one token; the tiny model's klcheck passed with the same defect).
//   test-tsr-gdn-logits <model.gguf> [steps]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static llama_context * make_ctx(llama_model * m, bool device) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx = 256;
    p.n_batch = 64;
    p.n_ubatch = 16;
    p.n_threads = 4;
    p.n_threads_batch = 4;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.type_k = GGML_TYPE_Q8_0;
    p.type_v = GGML_TYPE_Q8_0;
    p.offload_kqv = device;
    return llama_init_from_model(m, p);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test-tsr-gdn-logits <model.gguf> [steps]\n");
        return 2;
    }
    const int steps = argc > 2 ? atoi(argv[2]) : 3;
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
    // prompts of 1, 2, 5 and 17 tokens (17 crosses a ubatch of 16): the decode graphs of the first token have
    // the shapes (one token, a zero state) that the allocator packs tightest
    const int lens[] = { 1, 2, 5, 17 };
    for (int len : lens) {
        llama_context * dc = make_ctx(dev_model, true);
        llama_context * cc = make_ctx(cpu_model, false);
        CHECK(dc != nullptr && cc != nullptr, "context");
        if (dc == nullptr || cc == nullptr) {
            return 2;
        }
        std::vector<llama_token> toks(len);
        for (int i = 0; i < len; ++i) {
            toks[i] = (llama_token) ((i * 7 + 3 + len) % std::max(nv - 4, 1) + 2);
        }
        double worst_cos = 1.0, worst_rel = 0.0;
        int top1_miss = 0;
        for (int step = 0; step <= steps; ++step) {
            std::vector<llama_token> in = step == 0 ? toks : std::vector<llama_token>{ toks.back() };
            llama_batch b1 = llama_batch_get_one(in.data(), (int) in.size());
            llama_batch b2 = llama_batch_get_one(in.data(), (int) in.size());
            CHECK(llama_decode(dc, b1) == 0 && llama_decode(cc, b2) == 0, "decode (prompt %d, step %d)", len, step);
            const float * a = llama_get_logits_ith(dc, -1);
            const float * c = llama_get_logits_ith(cc, -1);
            double dot = 0, na = 0, nc = 0, maxd = 0, spread = 0;
            int ia = 0, ic = 0;
            for (int i = 0; i < nv; ++i) {
                dot += (double) a[i] * c[i];
                na += (double) a[i] * a[i];
                nc += (double) c[i] * c[i];
                maxd = std::max(maxd, std::fabs((double) a[i] - c[i]));
                spread = std::max(spread, std::fabs((double) c[i]));
                ia = a[i] > a[ia] ? i : ia;
                ic = c[i] > c[ic] ? i : ic;
            }
            const double cs = dot / std::sqrt(na * nc);
            worst_cos = std::min(worst_cos, cs);
            worst_rel = std::max(worst_rel, maxd / std::max(spread, 1e-9));
            top1_miss += ia != ic;
            toks.push_back(ic);
        }
        printf("  prompt %2d tokens, %d steps: worst cosine %.6f, largest logit difference %.3f%% of the logit range, top-1 misses %d\n",
               len, steps + 1, worst_cos, 100.0 * worst_rel, top1_miss);
        CHECK(worst_cos >= 0.9995, "prompt %d: cosine %.6f", len, worst_cos);
        CHECK(worst_rel <= 0.03, "prompt %d: largest logit difference %.3f of the range", len, worst_rel);
        llama_free(dc);
        llama_free(cc);
    }
    llama_model_free(dev_model);
    llama_model_free(cpu_model);
    llama_backend_free();
    if (g_fail) {
        fprintf(stderr, "test-tsr-gdn-logits: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-gdn-logits: OK\n");
    return 0;
}

// test-tsr-kvstate.cpp — ctest ggml_tesseract_kvstate (W3b P3.5; docs/v2/06-software-verification.md §9.3
// M7 "KV buffers": "set_tensor / get_tensor / memset_tensor / clear convert ... so llama.cpp's state save /
// load keeps working"; docs/v2/03-attention-kv.md §4.6 "state files are bit-identical to the CPU's"):
// llama.cpp's own state API with the KV cache in device KV planes (M7 settings: -fa on, -ctk / -ctv q8_0,
// offload_kqv, LLAMA_ATTN_ROT_DISABLE=1), on a tiny model (tests/gen_tiny_models.py, V2-SW-21).
//
//   1. Context A (device) decodes a prompt (several ubatches); its whole state is saved.
//   2. Context B (device) loads it: B's saved state equals A's byte for byte, and the next token's
//      logits of A and B are bit-identical (the planes B rebuilt from the state equal A's).
//   3. Context C (model and cache on the CPU) loads the device's state: its saved state equals it byte
//      for byte (the device's state file is a ggml q8_0 image, AT §4.6), and C's next-token logits
//      agree with A's at the model's device-vs-CPU level (reported).
//   4. Sequence state (llama_state_seq_get / set_data: cell ranges, i.e. partial row ranges of the
//      cache tensors): A's sequence 0 into a cleared B; the next token's logits of A and B are
//      bit-identical again.
//   test-tsr-kvstate <model.gguf>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

namespace {

llama_context * make_ctx(llama_model * m, bool kv_on_device) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx = 512;
    p.n_batch = 64;
    p.n_ubatch = 16;
    p.n_seq_max = 1;
    p.n_threads = 4;
    p.n_threads_batch = 4;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.type_k = GGML_TYPE_Q8_0;
    p.type_v = GGML_TYPE_Q8_0;
    p.offload_kqv = kv_on_device;
    return llama_init_from_model(m, p);
}

std::vector<uint8_t> state_of(llama_context * c) {
    std::vector<uint8_t> s(llama_state_get_size(c));
    const size_t n = llama_state_get_data(c, s.data(), s.size());
    s.resize(n);
    return s;
}

// Decode tokens (positions continue from the context's memory), return the last logits.
std::vector<float> decode(llama_context * c, const llama_model * m, std::vector<llama_token> toks) {
    llama_batch b = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    CHECK(llama_decode(c, b) == 0, "llama_decode of %zu tokens", toks.size());
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const float * l = llama_get_logits_ith(c, -1);
    return l ? std::vector<float>(l, l + nv) : std::vector<float>();
}

bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() && memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * mdev = llama_model_load_from_file(argv[1], mp);
    mp.n_gpu_layers = 0;
    llama_model * mcpu = llama_model_load_from_file(argv[1], mp);
    if (mdev == nullptr || mcpu == nullptr) {
        fprintf(stderr, "test-tsr-kvstate: cannot load %s\n", argv[1]);
        return 1;
    }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(mdev));
    std::vector<llama_token> prompt(40);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (llama_token) ((i * 7 + 3) % (size_t) nv);

    // 1. A decodes the prompt (3 ubatches of 16) and saves its state
    llama_context * A = make_ctx(mdev, true);
    llama_context * B = make_ctx(mdev, true);
    llama_context * C = make_ctx(mcpu, false);
    CHECK(A && B && C, "contexts");
    decode(A, mdev, prompt);
    const std::vector<uint8_t> sa = state_of(A);
    printf("state of the device context after %zu tokens: %zu bytes\n", prompt.size(), sa.size());

    // 2. B loads it: the same state back, the same next-token logits bit for bit
    CHECK(llama_state_set_data(B, sa.data(), sa.size()) == sa.size(), "B: state load");
    const std::vector<uint8_t> sb = state_of(B);
    CHECK(sb == sa, "B's saved state differs from the state it loaded (%zu vs %zu bytes)", sb.size(), sa.size());
    const llama_token x = 5 % nv;
    const std::vector<float> la = decode(A, mdev, { x }), lb = decode(B, mdev, { x });
    CHECK(bit_equal(la, lb), "A and B: next-token logits differ after the state load");

    // 3. C (CPU cache) loads the device's state file
    CHECK(llama_state_set_data(C, sa.data(), sa.size()) == sa.size(), "C: state load");
    const std::vector<uint8_t> sc = state_of(C);
    CHECK(sc == sa, "C's (CPU) saved state differs from the device's state it loaded");
    const std::vector<float> lc = decode(C, mcpu, { x });
    double num = 0, den = 0;
    int am = 0, cm = 0;
    for (int i = 0; i < nv && !la.empty() && !lc.empty(); ++i) {
        num += ((double) la[i] - lc[i]) * ((double) la[i] - lc[i]);
        den += (double) lc[i] * lc[i];
        if (la[i] > la[am]) am = i;
        if (lc[i] > lc[cm]) cm = i;
    }
    printf("device vs CPU next-token logits from the same state: NMSE %.3e, argmax %d / %d\n", den ? num / den : num, am, cm);
    CHECK(den > 0 && num / den < 1e-2, "device vs CPU logits from the same state: NMSE %.3e", den ? num / den : num);

    // 4. sequence state (cell ranges) into a cleared B
    std::vector<uint8_t> ss(llama_state_seq_get_size(A, 0));
    ss.resize(llama_state_seq_get_data(A, ss.data(), ss.size(), 0));
    llama_memory_clear(llama_get_memory(B), true);
    CHECK(llama_state_seq_set_data(B, ss.data(), ss.size(), 0) == ss.size(), "B: sequence state load");
    const llama_token y = 11 % nv;
    const std::vector<float> la2 = decode(A, mdev, { y }), lb2 = decode(B, mdev, { y });
    CHECK(bit_equal(la2, lb2), "A and B: logits differ after the sequence state load");
    printf("sequence state: %zu bytes, reloaded into a cleared context: logits %s\n", ss.size(),
           bit_equal(la2, lb2) ? "bit-identical" : "DIFFER");

    llama_free(C);
    llama_free(B);
    llama_free(A);
    llama_model_free(mcpu);
    llama_model_free(mdev);
    llama_backend_free();
    if (g_fail) {
        fprintf(stderr, "test-tsr-kvstate: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-kvstate: OK\n");
    return 0;
}

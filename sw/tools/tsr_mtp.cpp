// tsr-mtp — multi-token prediction (MTP) speculative decoding on the Tesseract device: the driver and checker of
// docs/design/06-software-verification.md §7.11 (SW-83 and the following).
//
// The loop is llama.cpp's examples/speculative-simple (the pinned tree's common_speculative with the MTP drafter): per step
//   draft    K tokens from the MTP head, one single-token decode of the draft context each,
//   verify   one decode of the target on [id_last, d_0 .. d_{K-1}] with the logits of every position,
//   process  the draft context catches up on the same K + 1 tokens (its KV cache and the hidden-state hand-over),
//   accept   the target's greedy choices are compared with the draft; a rejected tail is removed with llama_memory_seq_rm, which
//            the recurrent memory serves from the rollback snapshots (n_rs_seq = K).
// What this tool adds to the example:
//   - the device counters of the backend (hook "ggml_backend_tesseract_counters") around each phase, so a step's core cycles are
//     split into draft / verify / process, and the cycles per generated token come out of the same counters on the simulator and
//     on the card (TSR_DEVICE=sim | xrt);
//   - --mtp-mode plain: the same greedy loop with one token per decode and no drafting, the baseline whose token stream a
//     speculative run must reproduce;
//   - --mtp-force-reject: corrupts a draft token at a chosen position every step, so that the rollback of every length runs;
//   - several prompts (klcheck's built-in eight, or a file), the generated ids written out (--mtp-tokens-out), and a JSON summary.
// --mtp-mode all runs the plain baseline and the speculation for every K of --mtp-k-list (one model load, one process) and compares the
// streams. It takes llama.cpp's own options for the models and the speculation (-m, -md, --spec-type, --spec-draft-n-max, -c, -ngl, ...);
// the --mtp-* options below are removed from the command line before llama.cpp's parser sees it. The defaults are the M7
// settings of klcheck (SW-22): flash attention, q8_0 KV for the target and the draft, every layer on the device, the Hadamard
// rotation off.
//
//   tsr-mtp -m MODEL [-md MTP.gguf] [--spec-type draft-mtp --spec-draft-n-max K] [-n 64] [--mtp-device sim|mock|xrt]
//           [--mtp-mode spec|plain] [--mtp-prompts FILE] [--mtp-n-prompts N] [--mtp-tokens-out FILE] [--mtp-json FILE]
//           [--mtp-force-reject cycle|first|last|all] [-p PROMPT]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

[[noreturn]] void fail(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "tsr-mtp: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// klcheck's built-in prompts (sw/tools/klcheck.cpp k_builtin_prompts)
const char * const k_builtin_prompts[] = {
    "The history of the printing press begins",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain why the sky is blue in simple terms.",
    "\xEB\x8C\x80\xED\x95\x9C\xEB\xAF\xBC\xEA\xB5\xAD\xEC\x9D\x98 \xEC\x88\x98\xEB\x8F\x84\xEB\x8A\x94 \xEC\x84\x9C\xEC\x9A\xB8\xEC\x9D\xB4\xEB\x8B\xA4. \xEC\x84\x9C\xEC\x9A\xB8\xEC\x9D\x80",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA:",
    "Here is a list of five common kitchen herbs:\n1.",
    "In a distant future, humanity discovered",
    "The following JSON describes a book:\n{\"title\": \"",
};

std::vector<std::string> load_prompts(const std::string & file) {
    std::vector<std::string> out;
    if (file.empty()) {
        for (const char * p : k_builtin_prompts) out.emplace_back(p);
        return out;
    }
    FILE * f = fopen(file.c_str(), "rb");
    if (!f) fail("%s: %s", file.c_str(), strerror(errno));
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
    if (out.empty()) fail("%s: no prompts", file.c_str());
    return out;
}

// the backend's counters (ggml-tesseract.cpp tsr_hook_counters)
struct ctr {
    uint64_t v[8] = {};
};
typedef bool (*counters_fn)(ggml_backend_dev_t, uint64_t *, size_t);
ggml_backend_dev_t g_dev = nullptr;
counters_fn g_ctr_fn = nullptr;

ctr read_ctr() {
    ctr c;
    if (g_ctr_fn != nullptr) g_ctr_fn(g_dev, c.v, 8);
    return c;
}

enum { C_CYC = 0, C_MVM, C_VU, C_DMA, C_QACT, C_ATT, C_LAUNCH, C_GRAPHS };

struct phase {
    uint64_t cycles = 0, mvm = 0, vu = 0, dma = 0, launches = 0;
    double wall = 0;
    uint64_t n = 0;                     // how many times the phase ran
    void add(const ctr & a, const ctr & b, double w) {
        cycles += b.v[C_CYC] - a.v[C_CYC];
        mvm += b.v[C_MVM] - a.v[C_MVM];
        vu += b.v[C_VU] - a.v[C_VU];
        dma += b.v[C_DMA] - a.v[C_DMA];
        launches += b.v[C_LAUNCH] - a.v[C_LAUNCH];
        wall += w;
        n++;
    }
    void add(const phase & o) {
        cycles += o.cycles; mvm += o.mvm; vu += o.vu; dma += o.dma; launches += o.launches; wall += o.wall; n += o.n;
    }
};

struct result {
    std::vector<llama_token> ids;       // the generated tokens, exactly n_gen of them (or up to the end of generation)
    int steps = 0, drafted = 0, accepted = 0;
    std::vector<int> hist;              // steps that accepted 0, 1, .. K drafts
    std::vector<int> pos_drafted, pos_accepted;   // by draft position
    phase prefill, draft, verify, process, other, decode;   // decode: the whole loop, the sum of the above
    double t_decode = 0;
};

struct opts {
    std::string device = "sim", mode = "spec", prompts_file, tokens_out, json_out, force_reject, k_list;
    int n_prompts = -1, rounds = 1;
    bool have_prompt_arg = false;
};

// Remove the --mtp-* options from argv (they take a value), fill o.
void take_own_options(int & argc, char ** argv, opts & o) {
    int w = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) fail("%s needs a value", a.c_str());
            return argv[++i];
        };
        if (a == "--mtp-device") o.device = val();
        else if (a == "--mtp-mode") o.mode = val();
        else if (a == "--mtp-prompts") o.prompts_file = val();
        else if (a == "--mtp-n-prompts") o.n_prompts = atoi(val().c_str());
        else if (a == "--mtp-tokens-out") o.tokens_out = val();
        else if (a == "--mtp-json") o.json_out = val();
        else if (a == "--mtp-force-reject") o.force_reject = val();
        else if (a == "--mtp-k-list") o.k_list = val();
        else if (a == "--mtp-rounds") o.rounds = std::max(1, atoi(val().c_str()));
        else {
            if (a == "-p" || a == "--prompt" || a == "-f" || a == "--file") o.have_prompt_arg = true;
            argv[w++] = argv[i];
        }
    }
    argc = w;
}

bool has_arg(int argc, char ** argv, const char * name) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return true;
    }
    return false;
}

void print_phase(const char * name, const phase & p, uint64_t tokens) {
    if (p.n == 0) return;
    printf("  %-8s %4" PRIu64 " x, %5.1f launches each, %12" PRIu64 " cycles (%8.3f M per run, %9.3f M per generated token), "
           "VU %.3f M, DMA %.3f M, MVM %.3f M, wall %.2f s\n",
           name, p.n, (double) p.launches / (double) p.n, p.cycles, p.cycles / 1e6 / (double) p.n,
           tokens ? p.cycles / 1e6 / (double) tokens : 0.0, p.vu / 1e6, p.dma / 1e6, p.mvm / 1e6, p.wall);
}


// One configuration of the loop: plain decoding or MTP speculation with K drafts, over all prompts.
struct engine {
    common_params *  params = nullptr;
    llama_model *    model = nullptr;
    llama_context *  ctx_spec = nullptr;      // the target with rollback snapshots (spec runs)
    llama_context *  ctx_plain = nullptr;     // a target without (plain runs), created when needed
    llama_context *  ctx_dft = nullptr;
    int              n_gen = 64;
    std::string     force_reject;
};

llama_context * make_plain_context(engine & e) {
    llama_context_params cp = common_context_params_to_llama(*e.params);
    cp.n_rs_seq = 0;
    cp.n_outputs_max = 1;
    llama_context * c = llama_init_from_model(e.model, cp);
    if (c == nullptr) fail("cannot create the plain target context");
    return c;
}

std::string text_of(llama_context * ctx, const std::vector<llama_token> & ids) {
    std::string text;
    for (llama_token t : ids) text += common_token_to_piece(ctx, t);
    for (char & ch : text) if (ch == '\n') ch = ' ';
    return text;
}

// Run every prompt: spec = K drafts per step (K >= 1), else plain decoding. per_prompt gets the generated ids.
result run_set(engine & e, bool spec, int K, const std::vector<std::string> & prompts, std::vector<std::vector<llama_token>> * per_prompt) {
    common_params & params = *e.params;
    llama_context * ctx_tgt = spec ? e.ctx_spec : e.ctx_plain;
    llama_context * ctx_dft = spec ? e.ctx_dft : nullptr;
    const llama_vocab * vocab = llama_model_get_vocab(e.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
    llama_memory_t mem_dft = ctx_dft ? llama_get_memory(ctx_dft) : nullptr;
    const llama_seq_id seq = 0;
    const int n_batch = (int) llama_n_batch(ctx_tgt);
    const int n_ctx = (int) llama_n_ctx(ctx_tgt);
    if (spec) {
        params.speculative.draft.n_max = K;
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft;
    }
    result total;
    const int Kh = spec ? K : 0;
    total.hist.assign((size_t) Kh + 1, 0);
    total.pos_drafted.assign((size_t) std::max(Kh, 1), 0);
    total.pos_accepted.assign((size_t) std::max(Kh, 1), 0);
    int ip = 0;
    for (const std::string & prompt : prompts) {
        ++ip;
        result r;
        r.hist.assign((size_t) Kh + 1, 0);
        r.pos_drafted.assign((size_t) std::max(Kh, 1), 0);
        r.pos_accepted.assign((size_t) std::max(Kh, 1), 0);
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt, true, true);
        if (inp.size() < 2) fail("prompt %d has %zu tokens (need two)", ip, inp.size());
        if ((int) inp.size() + e.n_gen + Kh + 4 > n_ctx) fail("prompt %d (%zu tokens) and %d generated tokens do not fit -c %d", ip, inp.size(), e.n_gen, n_ctx);
        llama_memory_clear(mem_tgt, true);
        if (mem_dft) llama_memory_clear(mem_dft, true);
        common_sampler_ptr smpl(common_sampler_init(e.model, params.sampling));
        common_speculative_ptr sp;
        if (spec) {
            sp.reset(common_speculative_init(params.speculative, 1));
            if (!sp) fail("cannot initialize the speculative decoding");
        }
        {   // prefill all but the last prompt token (the last one opens the first step), in ubatches of n_batch
            const ctr c0 = read_ctr();
            const double t0 = now_s();
            for (size_t i0 = 0; i0 + 1 < inp.size(); i0 += (size_t) n_batch) {
                const size_t n = std::min<size_t>((size_t) n_batch, inp.size() - 1 - i0);
                common_batch b(ctx_tgt);
                for (size_t i = 0; i < n; ++i) b.add(inp[i0 + i], (llama_pos) (i0 + i), seq, false);
                if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) fail("prefill of the target failed");
                if (spec && !common_speculative_process(sp.get(), b)) fail("prefill of the draft failed");
            }
            r.prefill.add(c0, read_ctr(), now_s() - t0);
        }
        llama_token id_last = inp.back();
        llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
        int n_past = (int) inp.size() - 1;
        if (spec) common_speculative_begin(sp.get(), seq, prompt_tgt);
        common_batch batch_tgt(ctx_tgt);
        llama_tokens draft;
        bool eog = false;
        const double t_loop0 = now_s();
        int step = 0;
        while ((int) r.ids.size() < e.n_gen && !eog) {
            draft.clear();
            if (spec) {   // draft
                int n_draft_max = n_ctx - n_past - 2;
                n_draft_max = std::min(n_draft_max, e.n_gen - (int) r.ids.size() - 1);
                n_draft_max = std::max(n_draft_max, 0);
                common_speculative_get_draft_params(sp.get(), seq) = {
                    /* .drafting   = */ true,
                    /* .n_max      = */ n_draft_max,
                    /* .pos0       = */ n_past,
                    /* .id_last    = */ id_last,
                    /* .prompt     = */ &prompt_tgt,
                    /* .result     = */ &draft,
                };
                const ctr c0 = read_ctr();
                const double t0 = now_s();
                common_speculative_draft(sp.get());
                r.draft.add(c0, read_ctr(), now_s() - t0);
                if (mem_dft) llama_memory_seq_rm(mem_dft, seq, n_past, -1);
                if (!e.force_reject.empty() && !draft.empty()) {   // a forced rejection: corrupt one draft token, the position cycles
                    const int nd = (int) draft.size();
                    int p = -1;
                    if (e.force_reject == "cycle") p = step % (nd + 1);
                    else if (e.force_reject == "first") p = 0;
                    else if (e.force_reject == "last") p = nd - 1;
                    else if (e.force_reject == "all") p = 0;
                    else fail("--mtp-force-reject: cycle, first, last or all");
                    for (int i = p; i >= 0 && i < nd; ++i) {
                        draft[(size_t) i] = (draft[(size_t) i] + 7919 + step) % n_vocab;
                        if (e.force_reject != "all") break;
                    }
                }
            }
            batch_tgt.clear();   // verify: the target on [id_last, drafts]
            batch_tgt.add(id_last, n_past, seq, true);
            for (size_t i = 0; i < draft.size(); ++i) batch_tgt.add(draft[i], n_past + 1 + (int) i, seq, true);
            {
                const ctr c0 = read_ctr();
                const double t0 = now_s();
                if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0) fail("verify decode failed (step %d)", step);
                r.verify.add(c0, read_ctr(), now_s() - t0);
            }
            if (spec) {   // process: the draft context catches up
                const ctr c0 = read_ctr();
                const double t0 = now_s();
                if (!common_speculative_process(sp.get(), batch_tgt)) fail("process of the draft failed (step %d)", step);
                r.process.add(c0, read_ctr(), now_s() - t0);
            }
            const size_t n_draft = draft.size();
            std::vector<llama_token> ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
            if (ids.empty()) fail("no token accepted (step %d)", step);
            const int n_acc = (int) ids.size() - 1;
            if (spec) common_speculative_accept(sp.get(), seq, (uint16_t) n_acc);
            r.steps++;
            r.drafted += (int) n_draft;
            r.accepted += n_acc;
            if (spec) {
                r.hist[(size_t) std::min(n_acc, Kh)]++;
                for (size_t i = 0; i < n_draft && i < r.pos_drafted.size(); ++i) {
                    r.pos_drafted[i]++;
                    if ((int) i < n_acc) r.pos_accepted[i]++;
                }
            }
            n_past += (int) ids.size();
            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);
                id_last = ids[i];
                if ((int) r.ids.size() < e.n_gen) r.ids.push_back(id_last);
                if (llama_vocab_is_eog(vocab, id_last)) {
                    eog = true;
                    break;
                }
            }
            {   // roll the target and the draft back to the accepted position: the recurrent memory moves to the snapshot
                const ctr c0 = read_ctr();
                const double t0 = now_s();
                llama_memory_seq_rm(mem_tgt, seq, n_past, -1);
                if (mem_dft) llama_memory_seq_rm(mem_dft, seq, n_past, -1);
                r.other.add(c0, read_ctr(), now_s() - t0);
            }
            ++step;
        }
        r.t_decode = now_s() - t_loop0;
        r.decode.add(r.draft);
        r.decode.add(r.verify);
        r.decode.add(r.process);
        printf("  prompt %d: %zu prompt tokens, %zu generated, %d steps", ip, inp.size(), r.ids.size(), r.steps);
        if (spec) printf(", %d drafted, %d accepted (%.3f per step)", r.drafted, r.accepted, r.steps ? (double) r.accepted / r.steps : 0.0);
        printf("  | %.80s\n", text_of(ctx_tgt, r.ids).c_str());
        fflush(stdout);
        if (per_prompt) per_prompt->push_back(r.ids);
        total.ids.insert(total.ids.end(), r.ids.begin(), r.ids.end());
        total.steps += r.steps; total.drafted += r.drafted; total.accepted += r.accepted;
        for (size_t i = 0; i < r.hist.size(); ++i) total.hist[i] += r.hist[i];
        for (size_t i = 0; i < r.pos_drafted.size(); ++i) { total.pos_drafted[i] += r.pos_drafted[i]; total.pos_accepted[i] += r.pos_accepted[i]; }
        total.prefill.add(r.prefill); total.draft.add(r.draft); total.verify.add(r.verify); total.process.add(r.process);
        total.other.add(r.other); total.decode.add(r.decode); total.t_decode += r.t_decode;
    }
    return total;
}

struct summary {
    std::string tag;
    result r;
    double cyc_tok = 0, acc_step = 0;
    uint64_t tokens = 0;
    std::vector<double> round_tps;          // wall tok/s of the decode loop of each round
};

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

// The first round's result is reported; further rounds (--mtp-rounds) only add wall-clock samples (the same tokens)
result run_rounds(engine & e, bool spec, int K, const std::vector<std::string> & prompts, std::vector<std::vector<llama_token>> * ids,
                  int rounds, std::vector<double> * tps) {
    result first;
    for (int r = 0; r < rounds; ++r) {
        if (r > 0) printf("  round %d\n", r + 1);
        std::vector<std::vector<llama_token>> ids_r;
        result t = run_set(e, spec, K, prompts, r == 0 ? ids : &ids_r);
        tps->push_back(t.t_decode > 0 ? (double) t.ids.size() / t.t_decode : 0.0);
        if (r == 0) first = t;
        else if (ids_r.size() == ids->size()) {
            for (size_t i = 0; i < ids_r.size(); ++i)
                if (ids_r[i] != (*ids)[i]) printf("  WARNING: round %d prompt %zu generated other tokens than round 1\n", r + 1, i + 1);
        }
    }
    return first;
}

void report(const char * tag, bool spec, int K, size_t n_prompts, const result & t, summary * out) {
    const uint64_t tokens = t.ids.size();
    printf("\n== %s: %zu prompts, %" PRIu64 " generated tokens, %d steps ==\n", tag, n_prompts, tokens, t.steps);
    if (spec) {
        printf("  K = %d, drafted %d, accepted %d: %.3f accepted per step (%.3f tokens per step), %.1f %% of the drafts\n", K, t.drafted,
               t.accepted, t.steps ? (double) t.accepted / t.steps : 0.0, t.steps ? (double) t.accepted / t.steps + 1.0 : 0.0,
               t.drafted ? 100.0 * t.accepted / t.drafted : 0.0);
        printf("  steps by accepted drafts:");
        for (size_t i = 0; i < t.hist.size(); ++i) printf(" %zu:%d", i, t.hist[i]);
        printf("\n  accepted by draft position:");
        for (size_t i = 0; i < t.pos_drafted.size(); ++i)
            printf(" %zu: %d/%d (%.1f %%)", i, t.pos_accepted[i], t.pos_drafted[i], t.pos_drafted[i] ? 100.0 * t.pos_accepted[i] / t.pos_drafted[i] : 0.0);
        printf("\n");
    }
    print_phase("prefill", t.prefill, 0);
    print_phase("draft", t.draft, tokens);
    print_phase("verify", t.verify, tokens);
    print_phase("process", t.process, tokens);
    const double cyc_tok = tokens ? (double) t.decode.cycles / (double) tokens : 0.0;
    printf("  decode loop: %.3f M core cycles per generated token (draft %.3f + verify %.3f + process %.3f), %.2f launches per token,"
           " wall %.2f s = %.3f tok/s on this host\n",
           cyc_tok / 1e6, tokens ? t.draft.cycles / 1e6 / (double) tokens : 0.0, tokens ? t.verify.cycles / 1e6 / (double) tokens : 0.0,
           tokens ? t.process.cycles / 1e6 / (double) tokens : 0.0, tokens ? (double) t.decode.launches / (double) tokens : 0.0, t.t_decode,
           t.t_decode > 0 ? (double) tokens / t.t_decode : 0.0);
    for (double mhz : { 200.0, 250.0 })
        printf("  at %.0f MHz: %.3f tok/s from device cycles alone\n", mhz, cyc_tok > 0 ? mhz * 1e6 / cyc_tok : 0.0);
    if (out) {
        out->tag = tag;
        out->r = t;
        out->cyc_tok = cyc_tok;
        out->tokens = tokens;
        out->acc_step = t.steps ? (double) t.accepted / t.steps : 0.0;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    opts o;
    take_own_options(argc, argv, o);
    if (o.mode != "spec" && o.mode != "plain" && o.mode != "all") fail("--mtp-mode: spec, plain or all");

    // the M7 settings of klcheck (SW-22), unless the command line says otherwise
    std::vector<std::string> store;
    store.push_back(argv[0]);
    auto def = [&](const char * name, const char * v1, const char * v2 = nullptr) {
        if (!has_arg(argc, argv, name)) {
            store.push_back(name);
            if (v1) store.push_back(v1);
            if (v2) store.push_back(v2);
        }
    };
    def("-fa", "on");
    def("-ctk", "q8_0");
    def("-ctv", "q8_0");
    def("-ctkd", "q8_0");
    def("-ctvd", "q8_0");
    def("-ngl", "99");
    def("-ngld", "99");
    def("--temp", "0");
    def("-c", "4096");
    for (int i = 1; i < argc; ++i) store.push_back(argv[i]);
    if (o.mode != "plain" && !has_arg(argc, argv, "--spec-type")) {
        store.push_back("--spec-type");
        store.push_back("draft-mtp");
    }
    std::vector<char *> av;
    for (std::string & s : store) av.push_back(s.data());
    int ac = (int) av.size();

    setenv("TSR_DEVICE", o.device == "cpu" ? "none" : o.device.c_str(), 1);
    setenv("LLAMA_ATTN_ROT_DISABLE", "1", 0);

    common_params params;
    common_init();
    if (!common_params_parse(ac, av.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;
    if (params.n_predict < 0) params.n_predict = 64;
    const bool want_spec = o.mode != "plain", want_plain = o.mode != "spec";
    std::vector<int> ks;
    if (want_spec) {
        if (!o.k_list.empty()) {
            size_t i = 0;
            while (i < o.k_list.size()) {
                const size_t j = o.k_list.find(',', i);
                ks.push_back(atoi(o.k_list.substr(i, j == std::string::npos ? std::string::npos : j - i).c_str()));
                if (j == std::string::npos) break;
                i = j + 1;
            }
        } else {
            ks.push_back(params.speculative.draft.n_max);
        }
        int kmax = 0;
        for (int k : ks) { if (k < 1) fail("--mtp-k-list: K >= 1"); kmax = std::max(kmax, k); }
        params.speculative.draft.n_max = kmax;      // the target keeps kmax + 1 snapshots, every K <= kmax rolls back within them
    } else {
        params.speculative.types.clear();
    }

    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, want_spec ? common_speculative_n_max(&params.speculative) : 0);
    params.n_outputs_max = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    llama_backend_init();
    auto init_tgt = common_init_from_params(params);
    llama_model * model_tgt = init_tgt->model();
    llama_context * ctx_tgt = init_tgt->context();
    if (model_tgt == nullptr || ctx_tgt == nullptr) fail("cannot load the target model");

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "Tesseract", 9) == 0) {
            g_dev = d;
            g_ctr_fn = (counters_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(d), "ggml_backend_tesseract_counters");
            break;
        }
    }
    if (g_ctr_fn == nullptr) fprintf(stderr, "tsr-mtp: no Tesseract counters (device %s): cycles read 0\n", o.device.c_str());

    engine e;
    e.params = &params;
    e.model = model_tgt;
    e.n_gen = params.n_predict;
    e.force_reject = o.force_reject;
    common_speculative_init_result_ptr spec_init;
    if (want_spec) {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        e.ctx_spec = ctx_tgt;
        e.ctx_dft = spec_init->context();
        if (e.ctx_dft == nullptr) fail("cannot create the MTP draft context (--spec-type draft-mtp with the MTP head in the model or -md)");
        if (llama_n_rs_seq(ctx_tgt) < (uint32_t) params.speculative.draft.n_max)
            fail("the target keeps %u rollback snapshots, the draft is up to %d tokens", llama_n_rs_seq(ctx_tgt), params.speculative.draft.n_max);
    }
    if (want_plain) e.ctx_plain = want_spec ? make_plain_context(e) : ctx_tgt;

    std::vector<std::string> prompts;
    if (o.have_prompt_arg && o.prompts_file.empty() && !params.prompt.empty()) prompts.push_back(params.prompt);
    else prompts = load_prompts(o.prompts_file);
    if (o.n_prompts > 0 && (int) prompts.size() > o.n_prompts) prompts.resize((size_t) o.n_prompts);

    std::vector<summary> sums;
    std::vector<std::vector<llama_token>> plain_ids;
    auto write_tokens = [&](const std::string & tag, const std::vector<std::vector<llama_token>> & v) {
        if (o.tokens_out.empty()) return;
        FILE * f = fopen((o.tokens_out + "." + tag + ".tok").c_str(), "wb");
        if (!f) fail("%s: %s", o.tokens_out.c_str(), strerror(errno));
        for (const auto & ids : v) {
            for (size_t i = 0; i < ids.size(); ++i) fprintf(f, "%s%d", i ? " " : "", ids[i]);
            fprintf(f, "\n");
        }
        fclose(f);
    };
    if (want_plain) {
        printf("plain decoding\n");
        summary s;
        const result r = run_rounds(e, false, 0, prompts, &plain_ids, o.rounds, &s.round_tps);
        report("plain", false, 0, prompts.size(), r, &s);
        sums.push_back(s);
        write_tokens("plain", plain_ids);
    }
    if (want_spec) {
        for (int K : ks) {
            printf("\nMTP, K = %d\n", K);
            std::vector<std::vector<llama_token>> ids;
            summary s;
            const result r = run_rounds(e, true, K, prompts, &ids, o.rounds, &s.round_tps);
            char tag[32];
            snprintf(tag, sizeof(tag), "mtp K=%d", K);
            report(tag, true, K, prompts.size(), r, &s);
            // the token streams against plain decoding on the same device
            if (want_plain) {
                size_t same = 0, diverge_tokens = 0;
                for (size_t i = 0; i < ids.size() && i < plain_ids.size(); ++i) {
                    size_t j = 0;
                    while (j < ids[i].size() && j < plain_ids[i].size() && ids[i][j] == plain_ids[i][j]) ++j;
                    if (j == ids[i].size() && j == plain_ids[i].size()) ++same;
                    else printf("  prompt %zu: the stream diverges from plain decoding at generated token %zu (of %zu)\n", i + 1, j, plain_ids[i].size());
                    diverge_tokens += plain_ids[i].size() - std::min(j, plain_ids[i].size());
                }
                printf("  token streams against plain decoding: %zu of %zu prompts identical, %zu tokens after a first divergence\n", same, ids.size(), diverge_tokens);
            }
            sums.push_back(s);
            write_tokens("k" + std::to_string(K), ids);
        }
    }
    if (sums.size() > 1 || !o.json_out.empty()) {
        printf("\n== summary (device cycles per generated token; tok/s at the image's clock from the cycles) ==\n");
        printf("  %-10s %8s %10s %10s %10s %10s %10s %10s %12s %12s\n", "config", "tokens", "acc/step", "M cyc/tok", "draft", "verify", "process", "vs plain", "wall best", "wall median");
        const double base = sums.empty() || sums[0].tag != "plain" ? 0.0 : sums[0].cyc_tok;
        for (const summary & s : sums) {
            printf("  %-10s %8" PRIu64 " %10.3f %10.3f %10.3f %10.3f %10.3f %9.2fx %9.3f t/s %9.3f t/s\n", s.tag.c_str(), s.tokens, s.acc_step, s.cyc_tok / 1e6,
                   s.tokens ? s.r.draft.cycles / 1e6 / (double) s.tokens : 0.0, s.tokens ? s.r.verify.cycles / 1e6 / (double) s.tokens : 0.0,
                   s.tokens ? s.r.process.cycles / 1e6 / (double) s.tokens : 0.0, base > 0 && s.cyc_tok > 0 ? base / s.cyc_tok : 0.0,
                   s.round_tps.empty() ? 0.0 : *std::max_element(s.round_tps.begin(), s.round_tps.end()), median_of(s.round_tps));
        }
    }
    if (!o.json_out.empty()) {
        FILE * f = fopen(o.json_out.c_str(), "wb");
        if (!f) fail("%s: %s", o.json_out.c_str(), strerror(errno));
        fprintf(f, "{\"prompts\":%zu,\"runs\":[", prompts.size());
        for (size_t i = 0; i < sums.size(); ++i) {
            const summary & s = sums[i];
            fprintf(f, "%s{\"config\":\"%s\",\"tokens\":%" PRIu64 ",\"steps\":%d,\"drafted\":%d,\"accepted\":%d,\"accepted_per_step\":%.6f,"
                       "\"cycles_per_token\":%.1f,\"cycles\":{\"draft\":%" PRIu64 ",\"verify\":%" PRIu64 ",\"process\":%" PRIu64 ",\"prefill\":%" PRIu64 "},"
                       "\"launches\":{\"draft\":%" PRIu64 ",\"verify\":%" PRIu64 ",\"process\":%" PRIu64 "},\"wall_decode_s\":%.3f,\"hist\":[",
                    i ? "," : "", s.tag.c_str(), s.tokens, s.r.steps, s.r.drafted, s.r.accepted, s.acc_step, s.cyc_tok, s.r.draft.cycles,
                    s.r.verify.cycles, s.r.process.cycles, s.r.prefill.cycles, s.r.draft.launches, s.r.verify.launches, s.r.process.launches, s.r.t_decode);
            for (size_t j = 0; j < s.r.hist.size(); ++j) fprintf(f, "%s%d", j ? "," : "", s.r.hist[j]);
            fprintf(f, "]}");
        }
        fprintf(f, "]}\n");
        fclose(f);
    }
    if (e.ctx_plain != nullptr && e.ctx_plain != ctx_tgt) llama_free(e.ctx_plain);
    spec_init.reset();
    init_tgt.reset();
    llama_backend_free();
    return 0;
}

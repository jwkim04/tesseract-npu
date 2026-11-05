// test-tsr-versions.cpp — ctests ggml_tesseract_versions_* (docs/design/06-software-verification.md
// "SW" §6.9, §9.3): what the backend does with the versions of the device and the booted image. One
// mode per process (the device is opened once per process, with the TSR_IMAGE of that moment). The
// numbers it expects are derived from the sources that define them (tsr/version.h, the VERSION value
// of tsr_tables.h, tsr/opl.hpp kind_rev), so a release or a kind revision bump does not edit this file.
//
//   test-tsr-versions full             the build's tsr-fw.elf: the description names the Tesseract Core
//                                      capability and "Tesseract Basis <version> (ABI <n>, 14 kinds,
//                                      image <id>)"; the hook ggml_backend_tesseract_versions returns
//                                      the device's capability and the image's versions and 14 kinds;
//                                      ADD, RMS_NORM, ROPE and MUL_MAT are routed to the device, and an
//                                      ADD of two MUL_MAT results runs on it
//   test-tsr-versions subset IMAGE     TSR_IMAGE = IMAGE (fw_exec_ref: NOP, JOIN, MUL_MAT only):
//                                      MUL_MAT is routed, ADD / RMS_NORM / ROPE are not ("kind ADD is
//                                      not in the booted image", TSR_DEBUG=2), and a graph of two
//                                      MUL_MATs (on the device, results in its compute buffer) and
//                                      their sum runs the sum on the CPU, exactly
//   test-tsr-versions m6               TSR_KINDS=m6 on tsr-fw.elf: the same answers and graph as
//                                      subset
//   test-tsr-versions old IMAGE        TSR_IMAGE = IMAGE (fw_basis_old: the production image linked as
//                                      Tesseract Basis 0.0.5, which the transport boots): the device
//                                      is not opened ("older than this backend's <version>"), nothing
//                                      is routed, the hook returns false
// Exit 0 / 1.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <unistd.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-tesseract.h"

#include "tsr/opl.hpp"       // kind_rev, kinds_sig
#include "tsr/version.h"      // TSR_BASIS_VERSION_*, TSR_SDK_ABI
#include "tsr_tables.h"       // TSR_VERSION_VALUE
#include "tsr_transport.h"    // tsr_versions

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

// What f writes to stderr (the backend's log and TSR_DEBUG lines), also echoed there.
static std::string stderr_of(const std::function<void()> & f) {
    fflush(stderr);
    FILE * tmp = tmpfile();
    const int saved = dup(2);
    if (tmp == nullptr || saved < 0) {
        f();
        return "";
    }
    dup2(fileno(tmp), 2);
    f();
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    rewind(tmp);
    char buf[4096];
    size_t k;
    while ((k = fread(buf, 1, sizeof(buf), tmp)) > 0) out.append(buf, k);
    fclose(tmp);
    fputs(out.c_str(), stderr);
    return out;
}

// The numbers the build defines.
static std::string basis_str() { return TSR_BASIS_VERSION_STRING; }
static uint32_t core_cap() { return TSR_VERSION_VALUE & 0xFFFFu; }
static std::string core_str() { return std::to_string(core_cap() >> 8) + "." + std::to_string(core_cap() & 0xFFu); }
// The kind table (revisions by kind, 0 = absent) of an image that provides `kinds` at the revisions of tsr/opl.hpp.
struct kind_table {
    uint8_t rev[tsr::opl::N_KINDS] = {};
    uint32_t n = 0;
    uint32_t sig() const { return tsr::opl::kinds_sig(rev); }
};
static kind_table table_of(const std::vector<uint32_t> & kinds) {
    kind_table t;
    for (uint32_t k : kinds) {
        t.rev[k] = uint8_t(tsr::opl::kind_rev(k));
        t.n++;
    }
    return t;
}
static kind_table all_kinds() {
    std::vector<uint32_t> ks;
    for (uint32_t k = 0; k < tsr::opl::N_KINDS; ++k)
        if (tsr::opl::kind_rev(k) != 0u) ks.push_back(k);
    return table_of(ks);
}

typedef bool (*versions_fn)(ggml_backend_dev_t, tsr_versions *);

static ggml_backend_dev_t tsr_dev() {
    ggml_backend_reg_t reg = ggml_backend_tesseract_reg();
    return reg != nullptr && ggml_backend_reg_dev_count(reg) > 0 ? ggml_backend_reg_dev_get(reg, 0) : nullptr;
}

static versions_fn hook() {
    return (versions_fn) ggml_backend_reg_get_proc_address(ggml_backend_tesseract_reg(), "ggml_backend_tesseract_versions");
}

// The answers of supports_op for ADD, RMS_NORM, ROPE and MUL_MAT (unallocated F32 tensors, a Q8_0 weight).
struct answers {
    bool add, rms, rope, mm;
};
static answers ask(ggml_backend_dev_t dev) {
    ggml_init_params p = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
    ggml_tensor * add = ggml_add(ctx, a, b);
    ggml_tensor * rms = ggml_rms_norm(ctx, a, 1e-6f);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 2);        // head_dim 64, 4 heads, 2 tokens
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * rope = ggml_rope_ext(ctx, x, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 32768, 1e6f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 128);
    ggml_tensor * mm = ggml_mul_mat(ctx, w, a);
    answers r = { ggml_backend_dev_supports_op(dev, add), ggml_backend_dev_supports_op(dev, rms),
                  ggml_backend_dev_supports_op(dev, rope), ggml_backend_dev_supports_op(dev, mm) };
    ggml_free(ctx);
    return r;
}

// c = W1 x + W2 x scheduled over Tesseract0 and the CPU: the weights and x in a Tesseract buffer, the
// two MUL_MATs on the device (the image provides MUL_MAT), their results in the scheduler's compute
// buffer, i.e. operands the device could take for the ADD in place. Whether the ADD runs there is the
// image's kind table (on_device): without ADD it runs on the CPU, and either way c is exactly the f32
// sum of the two results. err gets supports_op's reasons (TSR_DEBUG=2) from the allocation.
static void add_graph(ggml_backend_t tsr, bool on_device, std::string * err) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    const int K = 256, N = 128, T = 4;   // rows of whole lines (the 64-byte rule of supports_op, SW §9.3)
    ggml_init_params pd = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * cd = ggml_init(pd);
    ggml_tensor * w1 = ggml_new_tensor_2d(cd, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * w2 = ggml_new_tensor_2d(cd, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(cd, GGML_TYPE_F32, K, T);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(cd, tsr);
    CHECK(buf != nullptr, "a Tesseract buffer for the weights and x");
    std::vector<float> wf(size_t(K) * N), vx(size_t(K) * T);
    std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0, K) * N);
    for (int m = 0; m < 2; ++m) {
        for (size_t i = 0; i < wf.size(); ++i) wf[i] = float(int((i * (37 + 2 * m)) % 251) - 125) * (1.0f / 128.0f);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, N, K, nullptr);
        ggml_backend_tensor_set(m ? w2 : w1, wq.data(), 0, wq.size());
    }
    for (size_t i = 0; i < vx.size(); ++i) vx[i] = float(int((i * 11) % 67) - 33) * 0.0625f;
    ggml_backend_tensor_set(x, vx.data(), 0, vx.size() * 4);
    ggml_init_params pg = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * cg = ggml_init(pg);
    ggml_tensor * m1 = ggml_mul_mat(cg, w1, x);
    ggml_tensor * m2 = ggml_mul_mat(cg, w2, x);
    ggml_tensor * c = ggml_add(cg, m1, m2);
    ggml_set_output(m1);
    ggml_set_output(m2);
    ggml_set_output(c);
    ggml_cgraph * gf = ggml_new_graph(cg);
    ggml_build_forward_expand(gf, c);
    ggml_backend_t backends[2] = { tsr, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    bool alloc_ok = false;
    *err = stderr_of([&] { alloc_ok = ggml_backend_sched_alloc_graph(sched, gf); });
    CHECK(alloc_ok, "allocate the graph");
    CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS, "compute the graph");
    std::vector<float> v1(size_t(N) * T), v2(v1.size()), vc(v1.size());
    ggml_backend_tensor_get(m1, v1.data(), 0, v1.size() * 4);
    ggml_backend_tensor_get(m2, v2.data(), 0, v2.size() * 4);
    ggml_backend_tensor_get(c, vc.data(), 0, vc.size() * 4);
    int bad = 0;
    for (size_t i = 0; i < vc.size(); ++i) bad += vc[i] != v1[i] + v2[i];
    CHECK(bad == 0, "graph: %d of %zu sums differ from the host's", bad, vc.size());
    ggml_backend_t mm_on = ggml_backend_sched_get_tensor_backend(sched, m1);
    ggml_backend_t add_on = ggml_backend_sched_get_tensor_backend(sched, c);
    CHECK(mm_on == tsr, "the MUL_MAT ran on %s", mm_on ? ggml_backend_name(mm_on) : "(none)");
    CHECK(add_on == (on_device ? tsr : cpu), "the ADD ran on %s, expected %s", add_on ? ggml_backend_name(add_on) : "(none)",
          on_device ? "Tesseract0" : "the CPU");
    printf("ADD of two MUL_MAT results on %s: %zu values %s\n", add_on ? ggml_backend_name(add_on) : "(none)", vc.size() - size_t(bad),
           bad ? "differ" : "exact");
    ggml_backend_sched_free(sched);
    ggml_free(cg);
    ggml_backend_buffer_free(buf);
    ggml_free(cd);
    ggml_backend_free(cpu);
}

static void scale_graph(ggml_backend_t tsr, bool local_cap, bool softcap = false, bool enough_work = true) {
    const uint32_t K = softcap && enough_work ? 8192u : 256u;
    constexpr uint32_t N = 80u, T = 5u;
    std::vector<float> w(size_t(K) * N), x(size_t(K) * T);
    std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q8_0, K) * N);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int(i % 251u) - 125) * 0.015625f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 59u) - 29) * 0.125f;
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), packed.data(), 0, N, K, nullptr);
    std::vector<float> result(N * T), observed(N * T);
    const int last_case = softcap ? 2 : 1;
    for (int escape = 0; escape <= last_case; ++escape) {
        ggml_context * ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        ggml_tensor * mm = ggml_mul_mat(ctx, W, X);
        ggml_tensor * y = ggml_scale(ctx, mm, softcap ? 1.0f / 30.0f : -0.25f);
        if (softcap) {
            y = ggml_tanh(ctx, y);
            if (escape == 2) ggml_set_output(y);
            y = ggml_scale(ctx, y, 30.0f);
        }
        if (escape == 1) ggml_set_output(mm);
        ggml_set_output(y);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, tsr);
        CHECK(buf != nullptr, "scale graph buffer");
        if (buf == nullptr) {
            ggml_free(ctx);
            return;
        }
        ggml_backend_tensor_set(W, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(X, x.data(), 0, x.size() * sizeof(float));
        const std::string log = stderr_of([&] {
            CHECK(ggml_backend_graph_compute(tsr, gf) == GGML_STATUS_SUCCESS, "scale graph compute");
        });
        const char * marker = softcap ? "MUL_MAT_SOFTCAP " : "MUL_MAT_SCALE ";
        const bool fused = log.find(marker) != std::string::npos;
        CHECK(fused == (local_cap && escape == 0), "scale graph mode, cap=%d escape=%d softcap=%d:\n%s",
              int(local_cap), escape, int(softcap), log.c_str());
        ggml_backend_tensor_get(y, (escape ? observed : result).data(), 0, result.size() * sizeof(float));
        if (escape)
            CHECK(memcmp(result.data(), observed.data(), result.size() * sizeof(float)) == 0,
                  "local and observed/legacy scale graph differ");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "full";
    const bool legacy_scale = mode == "legacy-scale";
    const bool m7 = mode == "full" || legacy_scale;
    if ((mode == "subset" || legacy_scale) && argc > 2) {
        setenv("TSR_IMAGE", argv[2], 1);
    } else if (mode == "m6") {
        setenv("TSR_KINDS", "m6", 1);
    } else if (mode == "old" && argc > 2) {
        setenv("TSR_IMAGE", argv[2], 1);
    } else if (mode != "full") {
        fprintf(stderr, "usage: test-tsr-versions full | subset IMAGE | m6 | old IMAGE\n");
        return 2;
    }
    setenv("TSR_DEBUG", "2", 1);                  // supports_op's reasons on stderr
    ggml_backend_dev_t dev = tsr_dev();
    CHECK(dev != nullptr, "the Tesseract0 device is registered");
    if (dev == nullptr) return 1;
    tsr_versions v;
    memset(&v, 0, sizeof(v));
    if (mode == "old") {
        const std::string err = stderr_of([&] { (void) ggml_backend_dev_description(dev); });
        CHECK(err.find("has Tesseract Basis 0.0.5, older than this backend's " + basis_str()) != std::string::npos, "message:\n%s",
              err.c_str());
        CHECK(strcmp(ggml_backend_dev_description(dev), "Tesseract Core (not opened)") == 0, "description: %s",
              ggml_backend_dev_description(dev));
        const answers r = ask(dev);
        CHECK(!r.add && !r.rms && !r.rope && !r.mm, "nothing routed to a refused device");
        CHECK(hook() != nullptr && !hook()(dev, &v), "the versions hook: no device");
        CHECK(ggml_backend_dev_init(dev, nullptr) == nullptr, "no backend on a refused device");
    } else {
        const char * desc = ggml_backend_dev_description(dev);
        printf("device: %s\n", desc);
        const std::string core = "Tesseract Core " + core_str() + " (";
        CHECK(strncmp(desc, core.c_str(), core.size()) == 0, "description: %s", desc);
        const bool full_image = mode != "subset";
        kind_table kt = full_image ? all_kinds() : table_of({ tsr::opl::K_NOP, tsr::opl::K_JOIN, tsr::opl::K_MUL_MAT });
        if (legacy_scale) {
            kt.rev[tsr::opl::K_MUL_MAT_SCALE] = 0;
            kt.rev[tsr::opl::K_MUL_MAT_SOFTCAP] = 0;
            kt.n -= 2;
        }
        const std::string want = "), Tesseract Basis " + basis_str() + " (ABI " + std::to_string(TSR_SDK_ABI) + ", " +
                                 std::to_string(kt.n) + " kinds, image ";
        CHECK(strstr(desc, want.c_str()) != nullptr, "description: %s", desc);
        CHECK(hook() != nullptr && hook()(dev, &v), "the versions hook");
        CHECK(v.core == core_cap() && v.basis == uint32_t(TSR_BASIS_VERSION) && v.abi == TSR_SDK_ABI && v.n_kinds == kt.n &&
                  v.kinds_sig == kt.sig() && memcmp(v.kind_rev, kt.rev, sizeof(kt.rev)) == 0,
              "versions: core 0x%04x basis 0x%06x abi %u kinds %u sig 0x%04x", v.core, v.basis, v.abi, v.n_kinds, v.kinds_sig);
        answers r;
        const std::string err = stderr_of([&] { r = ask(dev); });
        if (m7) {
            CHECK(r.add && r.rms && r.rope && r.mm, "full image: ADD %d RMS_NORM %d ROPE %d MUL_MAT %d", r.add, r.rms, r.rope, r.mm);
        } else {
            CHECK(!r.add && !r.rms && !r.rope && r.mm, "%s: ADD %d RMS_NORM %d ROPE %d MUL_MAT %d", mode.c_str(), r.add, r.rms,
                  r.rope, r.mm);
            if (mode == "subset")
                CHECK(err.find("kind ADD is not in the booted image (Tesseract Basis " + basis_str() + ", image ") != std::string::npos,
                      "the reason at TSR_DEBUG=2:\n%s", err.c_str());
        }
        ggml_backend_t tsr = ggml_backend_dev_init(dev, nullptr);
        CHECK(tsr != nullptr, "backend");
        if (tsr != nullptr) {
            std::string why;
            add_graph(tsr, m7, &why);
            if (!m7)   // the scheduler asked supports_op about the allocated ADD, and was told why not
                CHECK(why.find("kind ADD is not in the booted image") != std::string::npos, "the ADD's reason at TSR_DEBUG=2:\n%s",
                      why.c_str());
            if (m7) {
                scale_graph(tsr, (v.caps & TSR_CAPS_MVM_LOCAL_SCALE) != 0u &&
                                 v.kind_rev[tsr::opl::K_MUL_MAT_SCALE] != 0u);
                scale_graph(tsr, (v.caps & TSR_CAPS_MVM_LOCAL_SOFTCAP) != 0u &&
                                 v.kind_rev[tsr::opl::K_MUL_MAT_SOFTCAP] != 0u, true);
                scale_graph(tsr, false, true, false);
            }
            ggml_backend_free(tsr);
        }
    }
    printf("test-tsr-versions %s: %s\n", mode.c_str(), g_fail ? "FAILED" : "passed");
    return g_fail ? 1 : 0;
}

// test_fw_kvfa_rv.cpp — fw_kvfa_rv_*: the SET_ROWS_KV and FLASH_ATTN kinds (sw/ctrl/kernels/k_kv.cpp,
// k_fattn.cpp; W2b P2.7) in the RISC-V image on tsr_rv, launched through the `graph` executor
// (docs/design/06-software-verification.md "SW" §7.2, §7.5, §10.2 "fw_kernels_rv": per kind, 8 schedules,
// race fatal, outputs bit-exact against the references).
//
//   test_fw_kvfa_rv IMAGE suite GROUP [--scheds N] [--threads N] [--only NAME]
//       Every case of test_fw_kvfa.h's group (qwen05b, qwen32b, gemma_swa, gemma_glb, general, slow):
//       one list [SET_ROWS_KV K, SET_ROWS_KV V, FLASH_ATTN], launched under the fixed schedule and
//       N - 1 random ones (default 8) with TSR_SIM_RACE fatal: DONE OK / RET 0, both planes after
//       the list = sw/ctrl/ref/kv_ref.cpp's (tsr_qact + tsr_ref_kvst per row) byte for byte, the
//       output rows = fattn_ref.cpp's (tsr_ref_fa) bit for bit, every written beat inside the planes
//       or the dst rows, no race report, and the SW §4.7 observables (per-unit op streams, the
//       RISC-V's SPM stores, the final written DRAM and SPM, DONE r0-r6) equal under every schedule.
//   test_fw_kvfa_rv IMAGE kl
//       (TSR_WITH_LLAMA) the end-to-end check against ggml-cpu: cases run through the kernel (fixed
//       schedule; output = tsr_ref_fa bit for bit as above), the same q8_0 cache (the planes read back
//       with tsr_kv_get_bytes), Q and F16 mask through ggml_flash_attn_ext on the CPU (ref_at_ggml's
//       harness, sw/sim/tests/test_ref_at.c fa_run_cpu): NMSE <= 1e-9 and the softmax-KL of the output
//       rows <= 1e-12 (W2b fix: measured 1.1e-13 to 2.7e-12 / <= 9e-15 on these cases; tight, because
//       this is the only check of the shared chunk planner besides fw_kvfa_plan, kvfa_ref.hpp).
//   test_fw_kvfa_rv IMAGE perf
//       Measurements (SW §7.8, AT §7.4, §9.2; the default timing model, Ibex maxperf, or
//       TSR_SIM_RV_CORE; fixed schedule; every list launched twice on one simulator and the second,
//       warm launch measured): RISC-V work per SET_ROWS_KV decode record (fw_prof.h: busy cycles up
//       to the last op push, wait-loop iterations after their first test excluded; 8 vs 24 records,
//       deep queues and U200 depths), and per FLASH_ATTN record (the list with it against the list
//       without it) for Qwen2.5-32B decode at n_kv 4096 and the other model shapes, decode and
//       prefill: the record's work, its ops' span on the units, the MVM's idle time waiting for a
//       push, and the span with a near-ideal issuer (F_RV = 4 F_CORE): whether the RISC-V stays
//       ahead of the units (checked: MVM waits for a push <= 1 % of the span, span within 5 %). TSR_FW_PROF_DUMP
//       writes the per-PC histograms (fw_prof_lines.py).
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image is missing.
#undef _GNU_SOURCE
#include "test_fw_kvfa.h"

#include <sys/stat.h>

#include <array>
#include <map>

#include "../ref/fw_sim_host.hpp"
#include "fw_prof.h"
#include "tsr_race.h"

#if defined(TSR_KVFA_GGML)
#include "ggml-cpu.h"
#include "ggml.h"
#endif

namespace {

using namespace kvfa;

const char * g_image = nullptr;
uint32_t g_threads = 2;

const kcase & find_case(const char * name) {
    for (const auto & c : cases())
        if (strcmp(c.name, name) == 0) return c;
    fprintf(stderr, "no case %s\n", name);
    exit(2);
}

struct run_obs {
    obs o;
    uint32_t done[7];
};

// One launch of case k's list under `sched` on a fresh simulator. The first one also computes the
// expected results.
bool run_case(kdata & k, const char * sched, run_obs * out, bool quiet = false) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 2;
    so.threads = g_threads;
    tsr_sim * sim = tsr_sim_create_opts(&k.cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", k.c->name);
    if (sim == nullptr) return false;
    fill(sim, k);
    if (!k.have_expect) expect(sim, k);
    obs_attach(&out->o, sim);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    CHECK(ok, "%s [%s]: boot", k.c->name, sched);
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    CHECK(!ok || gi >= 0, "%s: no graph kernel in the image", k.c->name);
    if (ok && gi >= 0) {
        const uint32_t a[7] = { k.opl_h, uint32_t(k.blob.size()), tsr::opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
        ok = fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s [%s]: launch", k.c->name, sched);
    }
    if (ok) {
        CHECK(TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK &&
                  fwh_ret(&h) == 0,
              "%s [%s]: DONE 0x%08x RET 0x%08x (err ticket 0x%08x info 0x%08x w0 0x%08x pidx %u)", k.c->name, sched, h.done[0],
              h.done[2], h.done[3], h.done[4], h.done[5], h.done[6]);
        for (int i = 0; i < 7; ++i) out->done[i] = h.done[i];
        obs_finish(&out->o, sim);
        check_outputs(sim, k, sched, &out->o);
        const tsr_race_report * rr = nullptr;
        const size_t reports = tsr_sim_race_reports(sim, &rr);
        CHECK(reports == 0, "%s [%s]: %zu race reports", k.c->name, sched, reports);
        if (!quiet)
            printf("  %-26s [%-11s] T %2u n_kv %5u: chunks %u (general %u), %llu core cycles push -> DONE, %llu RV instructions\n",
                   k.c->name, sched, k.c->T, k.c->n_kv, k.st.chunks, k.st.general_chunks, (unsigned long long)h.cycles,
                   (unsigned long long)out->o.instret);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

void run_scheds(const kcase & c, uint32_t n_sched) {
    kdata k;
    build(c, k);
    std::vector<run_obs> runs(n_sched);
    bool ok = true;
    for (uint32_t s = 0; s < n_sched; ++s) {
        char sched[64];
        if (s == 0) snprintf(sched, sizeof(sched), "fixed");
        else snprintf(sched, sizeof(sched), "random:%u", 1000u + s);
        ok = run_case(k, sched, &runs[s]) && ok;
        if (s > 0 && ok) {
            char what[128];
            snprintf(what, sizeof(what), "%s: %s vs fixed", c.name, sched);
            obs_equal(&runs[0].o, &runs[s].o, what);
            CHECK(memcmp(runs[0].done, runs[s].done, sizeof(runs[0].done)) == 0, "%s: DONE r0-r6 differ", what);
        }
    }
    for (auto & r : runs) obs_free(&r.o);
}

// =================================================================================================
// kl: the kernel against ggml-cpu's FLASH_ATTN_EXT
// =================================================================================================
#if defined(TSR_KVFA_GGML)
struct fa_cmp { double nmse, max_abs, kl_max, kl_mean; };

fa_cmp compare(const float * dev, const float * cpu, size_t rows, uint32_t D) {
    fa_cmp r{0, 0, 0, 0};
    double se = 0.0, sc = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        const float * a = dev + i * D, * b = cpu + i * D;
        double ma = -INFINITY, mb = -INFINITY;
        for (uint32_t d = 0; d < D; ++d) {
            const double e = std::fabs(double(a[d]) - double(b[d]));
            se += e * e;
            sc += double(b[d]) * double(b[d]);
            r.max_abs = std::max(r.max_abs, e);
            ma = std::max(ma, double(a[d]));
            mb = std::max(mb, double(b[d]));
        }
        double za = 0.0, zb = 0.0;
        for (uint32_t d = 0; d < D; ++d) { za += std::exp(double(a[d]) - ma); zb += std::exp(double(b[d]) - mb); }
        double kl = 0.0;
        for (uint32_t d = 0; d < D; ++d) {
            const double pb = std::exp(double(b[d]) - mb) / zb, pa = std::exp(double(a[d]) - ma) / za;
            if (pb > 0.0) kl += pb * std::log(pb / pa);
        }
        r.kl_max = std::max(r.kl_max, kl);
        r.kl_mean += kl / double(rows);
    }
    r.nmse = sc > 0.0 ? se / sc : se;
    return r;
}

// ggml's CPU FA on the cache bytes kc / vc ([n_kv][E/32 blocks]), Q [T][n_head][D] and the F16 mask.
bool cpu_fa(const kdata & k, const std::vector<uint8_t> & kc, const std::vector<uint8_t> & vc, const std::vector<uint16_t> & m16,
            std::vector<float> & out) {
    const kcase & c = *k.c;
    const uint32_t D = c.D, E = k.E, n_kv = c.n_kv, T = c.T, n_head = k.n_head;
    const size_t cache_b = kc.size();
    const size_t mem = 2u * cache_b + 3u * size_t(T) * n_head * D * 4u + size_t(T) * n_kv * 2u + (64u << 20);
    struct ggml_init_params ip = { mem, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) return false;
    struct ggml_tensor * kt = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, n_kv);
    struct ggml_tensor * vt = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, n_kv);
    memcpy(kt->data, kc.data(), cache_b);
    memcpy(vt->data, vc.data(), cache_b);
    struct ggml_tensor * kv = ggml_view_4d(ctx, kt, D, n_kv, c.nkvh, 1, kt->nb[1], ggml_row_size(GGML_TYPE_Q8_0, D), kt->nb[1] * n_kv, 0);
    struct ggml_tensor * vv = ggml_view_4d(ctx, vt, D, n_kv, c.nkvh, 1, vt->nb[1], ggml_row_size(GGML_TYPE_Q8_0, D), vt->nb[1] * n_kv, 0);
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, n_head, 1);
    float * qd = static_cast<float *>(q->data);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t h = 0; h < n_head; ++h)
            memcpy(qd + (size_t(h) * T + t) * D, k.q.data() + (size_t(t) * n_head + h) * D, size_t(D) * 4u);
    struct ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, T, 1, 1);
    memcpy(m->data, m16.data(), m16.size() * 2u);
    struct ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kv, vv, m, c.scale, 0.0f, 0.0f);
    ggml_prec_set_acc(fa, GGML_PREC_F32);   // as build_attn_mha (src/llama-graph.cpp)
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 4);
    out.assign(size_t(T) * n_head * D, 0.0f);
    if (st == GGML_STATUS_SUCCESS) memcpy(out.data(), fa->data, out.size() * 4u);
    ggml_free(ctx);
    return st == GGML_STATUS_SUCCESS;
}

// The F16 mask of a case (test_fw_kvfa.h build's rules, recomputed).
std::vector<uint16_t> mask16(const kcase & c) {
    std::vector<int64_t> cpos(c.n_kv, -1);
    const int64_t npos = int64_t(c.pos0) + c.T;
    for (int64_t p = 0; p < npos; ++p) {
        if (c.mask == M_RING) cpos[size_t(p % c.n_kv)] = p;
        else if (p < int64_t(c.n_kv)) cpos[size_t(p)] = p;
    }
    std::vector<uint16_t> m(size_t(c.T) * c.n_kv);
    for (uint32_t t = 0; t < c.T; ++t) {
        const int64_t pq = int64_t(c.pos0) + t;
        for (uint32_t j = 0; j < c.n_kv; ++j) {
            const int64_t p = cpos[j];
            bool vis = p >= 0 && p <= pq;
            if ((c.mask == M_SWA || c.mask == M_RING) && pq - p >= int64_t(c.n_swa)) vis = false;
            if (c.mask == M_HOLES && j % 7u == 3u) vis = false;
            if (c.mask == M_EMPTYROW && t == 0) vis = false;
            uint16_t h = vis ? 0x0000 : 0xFC00;
            if (vis && c.mask == M_FINITE) h = tsr_f32_to_fp16(-0.25f * float((pq - p) % 13));
            m[size_t(t) * c.n_kv + j] = h;
        }
    }
    return m;
}

void run_kl() {
    ggml_cpu_init();
    static const char * const names[] = { "q32b-dec-4096", "q05b-pre-256", "g12swa-pre-4096-slide", "g12swa-ring-dec",
                                          "g12glb-dec-4096", "general-finite-pre-4096" };
    printf("  %-26s %9s %9s %9s %9s\n", "case", "NMSE", "max_abs", "KL_max", "KL_mean");
    for (const char * nm : names) {
        const kcase * c = nullptr;
        for (const auto & x : cases()) if (strcmp(x.name, nm) == 0) c = &x;
        if (c == nullptr) continue;
        kdata k;
        build(*c, k);
        // One launch (fixed schedule) with the bit-exact checks, then the planes read back.
        tsr_sim_opts so;
        memset(&so, 0, sizeof(so));
        so.sched = "fixed";
        so.threads = g_threads;
        tsr_sim * sim = tsr_sim_create_opts(&k.cfg, &so);
        fill(sim, k);
        expect(sim, k);
        fwh h;
        bool ok = fwh_open(&h, sim, g_image) == 0;
        const int gi = ok ? fwh_kernel(&h, "graph") : -1;
        const uint32_t a[7] = { k.opl_h, uint32_t(k.blob.size()), 0, 0, 0, 0, 0 };
        ok = ok && gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0 && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0;
        CHECK(ok, "%s: launch", c->name);
        if (ok) {
            check_outputs(sim, k, "kl", nullptr);
            const tsr_kv_geom g = { k.E, k.C, k.LG };
            std::vector<const uint8_t *> kp(k.C), vp(k.C);
            for (uint32_t i = 0; i < k.C; ++i) {
                kp[i] = tsr_sim_dram(sim, i) + uint64_t(k.kB) * LINE;
                vp[i] = tsr_sim_dram(sim, i) + uint64_t(k.vB) * LINE;
            }
            const size_t cb = size_t(c->n_kv) * (k.E / 32u) * 34u;
            std::vector<uint8_t> kc(cb), vc(cb);
            tsr_kv_get_bytes(TSR_KV_K, &g, kp.data(), kc.data(), 0, cb);
            tsr_kv_get_bytes(TSR_KV_V, &g, vp.data(), vc.data(), 0, cb);
            std::vector<float> dev(size_t(c->T) * k.K), cpu;
            for (uint32_t t = 0; t < c->T; ++t)
                memcpy(dev.data() + size_t(t) * k.K, at(sim, k.dst_h, uint64_t(t) * k.dst_stride), size_t(k.K) * 4u);
            const bool cok = cpu_fa(k, kc, vc, mask16(*c), cpu);
            CHECK(cok, "%s: ggml compute failed", c->name);
            if (cok) {
                const fa_cmp r = compare(dev.data(), cpu.data(), size_t(c->T) * k.n_head, c->D);
                printf("  %-26s %9.2e %9.2e %9.2e %9.2e\n", c->name, r.nmse, r.max_abs, r.kl_max, r.kl_mean);
                // W2b fix (review): the gates sit a few orders above the measured floor (NMSE 1.1e-13 to
                // 2.7e-12, KL_max <= 9e-15 on these cases), so a one-cell mask or chunk-bound error in the
                // shared planner (tsr_fa_plan.h, which the bit-exact references also use) fails here:
                // test-backend-ops' 5e-4 let a q05b / g12swa one-cell error pass (KL 2.4e-3 / 8.0e-7).
                CHECK(r.nmse <= 1e-9, "%s: NMSE %.3e above 1e-9", c->name, r.nmse);
                CHECK(r.kl_max <= 1e-12, "%s: softmax-KL of the output rows %.3e above 1e-12", c->name, r.kl_max);
            }
        }
        fwh_close(&h);
        tsr_sim_destroy(sim);
    }
}
#endif

// =================================================================================================
// perf
// =================================================================================================
constexpr double F_CORE = 276.3;   // MHz (U200 core clock)
double g_f_rv = 166.7;             // MHz (CC's F_RV; from the simulator's ratio)

struct op_ev {
    uint32_t unit = 0, w0 = 0;
    uint64_t push = 0, start = 0, done = 0;
};

struct perf_ctx {
    tsr_rv * rv = nullptr;
    uint64_t last = 0, work = 0, work_at_push = 0;
    uint32_t last_w0 = 0;
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
    if (d <= 256u) {                     // larger steps are sleeps / fast-forwarded waits (fw_m6_perf)
        if (!spin) p->work += d;
        if (p->prof) p->prof->add(r, d, spin);
    }
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_W(0)) p->last_w0 = r->mem_val;
    if (r->mem_write && r->mem_size == 4u && r->mem_addr == 0x20000000u + TSR_CW_ISSUE_PUSH && (p->last_w0 & 0xFFu) != TSR_OP_FENCE)
        p->work_at_push = p->work;       // the work ends with the last op push (join_all's FENCE excluded)
    return 0;
}

void perf_q(void * ctx, tsr_sim *, const tsr_q_event * ev) {
    perf_ctx * p = static_cast<perf_ctx *>(ctx);
    if (ev->kind == TSR_QEV_PUSH) {
        op_ev & o = p->ops[ev->pidx];
        o.unit = ev->unit;
        o.w0 = ev->w ? ev->w[0] : 0u;
        o.push = ev->time;
    } else if (ev->kind == TSR_QEV_START || ev->kind == TSR_QEV_DONE) {
        auto it = p->ops.find(ev->pidx);
        if (it == p->ops.end()) return;
        if (ev->kind == TSR_QEV_START) it->second.start = ev->time;
        else it->second.done = ev->time;
    }
}

struct perf_run {
    double work = 0;                     // RV cycles of work up to the last op push
    uint64_t launch = 0;                 // core cycles push -> DONE
    std::map<uint32_t, op_ev> ops;
};

// One launch of `blob` (with k's data) on a fresh simulator, fixed schedule, with the perf hooks.
// deep: every queue 31 deep (no credit waits); ratio > 0: F_RV / F_CORE.
perf_run perf_launch(const kdata & k, const std::vector<uint8_t> & blob, bool deep, double ratio, fwprof::prof * prof = nullptr) {
    perf_run pr;
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.threads = g_threads;
    static const uint8_t deep_q[6] = { 31, 31, 31, 31, 0, 31 };
    if (deep) so.q_depth = deep_q;
    so.rv_ratio = ratio;
    tsr_sim * sim = tsr_sim_create_opts(&k.cfg, &so);
    fill(sim, k);
    memcpy(at(sim, k.opl_h), blob.data(), blob.size());
    fwh h;
    if (fwh_open(&h, sim, g_image) != 0) {
        CHECK(0, "perf: boot");
        tsr_sim_destroy(sim);
        return pr;
    }
    if (ratio == 0.0) g_f_rv = tsr_sim_rv_ratio(sim) * F_CORE;
    // A warm-up launch first (the steady state of a decode loop: the issue path's opcode -> unit
    // table, kernel.hpp push_unit_slow, is filled at the first use of an opcode after boot), then
    // the measured one; the second rewrites the same cells with the same rows.
    const int gi = fwh_kernel(&h, "graph");
    const uint32_t a[7] = { k.opl_h, uint32_t(blob.size()), 0, 0, 0, 0, 0 };
    CHECK(gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0 && fwh_ret(&h) == 0, "perf: warm-up launch");
    perf_ctx p;
    p.rv = tsr_rv_agent_rv(tsr_sim_rv_agent(sim));
    p.last = tsr_rv_cycles(p.rv);
    p.prof = prof;
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), perf_retire, &p);
    const int lid = tsr_sim_add_listener(sim, perf_q, &p);
    const bool ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0;
    CHECK(ok && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0, "perf: DONE 0x%08x RET 0x%08x", h.done[0], h.done[2]);
    pr.launch = h.cycles;
    pr.work = double(p.work_at_push);
    pr.ops = p.ops;
    const uint64_t stall = tsr_sim_perf(sim, TSR_PERF_PUSH_STALL(tsr_sim_config(sim)->num_ch));
    CHECK(stall == 0u, "perf: PERF PUSH_STALL %llu: a push waited for a credit in hardware (CC §7.3)", (unsigned long long)stall);
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), nullptr, nullptr);
    tsr_sim_remove_listener(sim, lid);
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return pr;
}

// A list of n SET_ROWS_KV decode records (one row each, K and V planes alternating, cells 0 .. n-1;
// independent: no deps) on k's planes; k.idx is replaced by the n indices.
std::vector<uint8_t> setrows_list(kdata & k, uint32_t n) {
    k.idx.assign(size_t(n) * 8u, 0);
    for (uint32_t i = 0; i < n; ++i) memcpy(k.idx.data() + 8u * i, &i, 4);
    tsr::opl::writer w(tsr::opl::F_STOP_ON_ERROR, 0x5e7u);
    const uint32_t pi = w.add_preload(k.idx_h, uint32_t(k.idx.size()));
    for (uint32_t i = 0; i < n; ++i) {
        tsr::opl::set_rows_kv_args s = (i & 1u) ? k.sv : k.sk;
        s.idx = pi + 8u * i;
        s.n_rows = 1u;
        w.add_set_rows_kv(s);
    }
    return w.finish();
}

// One FLASH_ATTN record's cost: the list [SET_ROWS_KV K, SET_ROWS_KV V, FLASH_ATTN] against the same
// list without FLASH_ATTN (the RISC-V's work for the record), its ops' span on the units, and the
// span with a near-ideal issuer (F_RV = 4 F_CORE; fw_m6_perf's M7 projection method).
void fa_perf(const char * name, bool both_depths) {
    kdata k;
    build(find_case(name), k);
    {   // the reference's stats (chunks, tiles)
        tsr_sim * sim = tsr_sim_create(&k.cfg, nullptr);
        fill(sim, k);
        expect(sim, k);
        tsr_sim_destroy(sim);
    }
    kdata k0 = k;
    make_blob(k0, true, false);          // the same list without FLASH_ATTN
    for (int deep = both_depths ? 1 : 0; deep >= 0; --deep) {
        fwprof::prof prof[2];
        const bool want = deep || getenv("TSR_FW_PROF_DUMP") != nullptr;   // every case's profile with a dump
        const bool syms = want && prof[0].load_symbols(g_image) && prof[1].load_symbols(g_image);
        const perf_run r0 = perf_launch(k0, k0.blob, deep != 0, 0.0, syms ? &prof[0] : nullptr);
        const perf_run r1 = perf_launch(k, k.blob, deep != 0, 0.0, syms ? &prof[1] : nullptr);
        const perf_run ri = perf_launch(k, k.blob, deep != 0, 4.0);
        // The record's ops: the pushes beyond r0's (the executor's and SET_ROWS's come first), without
        // the final FENCE.
        struct span_t { uint64_t span, mvm_busy, vu_busy, mvm_wait, push_lo, push_hi; uint32_t n; };
        auto fa_span = [&](const perf_run & r) {
            span_t x{0, 0, 0, 0, ~0ull, 0, 0};
            const uint32_t skip = uint32_t(r0.ops.size()) - 1u;   // r0's pushes but its FENCE
            uint32_t i = 0;
            uint64_t lo = ~0ull, hi = 0, prev_mvm_done = 0;
            for (const auto & kv : r.ops) {
                const op_ev & o = kv.second;
                if (i++ < skip || (o.w0 & 0xFFu) == TSR_OP_FENCE) continue;
                ++x.n;
                lo = std::min(lo, o.start);
                hi = std::max(hi, o.done);
                x.push_lo = std::min(x.push_lo, o.push);
                x.push_hi = std::max(x.push_hi, o.push);
                if (o.unit == TSR_UNIT_MVM) {
                    x.mvm_busy += o.done - o.start;
                    // the MVM idled on issue: its previous op was done before this one was pushed
                    if (prev_mvm_done != 0 && o.push > prev_mvm_done) x.mvm_wait += std::min(o.start, o.push) - prev_mvm_done;
                    prev_mvm_done = o.done;
                }
                if (o.unit == TSR_UNIT_VU) x.vu_busy += o.done - o.start;
            }
            x.span = hi > lo ? hi - lo : 0;
            return x;
        };
        const span_t s1 = fa_span(r1), si = fa_span(ri);
        const double work = r1.work - r0.work;
        const double slow = 100.0 * (double(s1.span) - double(si.span)) / double(si.span);
        printf("fw_kvfa_perf: FLASH_ATTN %s (D %u, G %u, T %u, n_kv %u; %s queues; %u ops, %u chunks, %u tiles):\n", name,
               k.c->D, k.c->G, k.c->T, k.c->n_kv, deep ? "deep" : "U200", s1.n, k.st.chunks, k.st.tiles);
        printf("  RISC-V work for the record %.0f RV cycles = %.1f us at %.1f MHz (= %.0f core cycles); its pushes span %llu "
               "core cycles\n", work, work / g_f_rv, g_f_rv, work / g_f_rv * F_CORE, (unsigned long long)(s1.push_hi - s1.push_lo));
        printf("  the units: first op start -> last op done %llu core cycles = %.1f us; MVM busy %llu, VU busy %llu; MVM idle "
               "on issue %llu\n", (unsigned long long)s1.span, double(s1.span) / F_CORE, (unsigned long long)s1.mvm_busy,
               (unsigned long long)s1.vu_busy, (unsigned long long)s1.mvm_wait);
        printf("  near-ideal issuer (F_RV = 4 F_CORE): span %llu core cycles; the real issuer %+.2f %%\n",
               (unsigned long long)si.span, slow);
        if (syms) fwprof::prof::diff(prof[1], prof[0]).report(stdout, "  profile of the FLASH_ATTN record", 1.0, 12, (std::string(".fa.") + name).c_str());
        // The RISC-V stays ahead of the units (AT §7.4): the MVM waits for an op the RISC-V has not
        // pushed yet for at most 1 % of the span (at Ibex maxperf: never; Ibex small: between the
        // tiles of a 0.5B prefill record), and the op's span is within 5 % of a near-ideal issuer's
        // (the difference is the head: an isolated list exposes the work before LD Q's ops, a
        // decode layer hides it behind the Q / K / V projections).
        CHECK(100 * s1.mvm_wait <= s1.span, "perf: %s: the MVM idled on issue %llu core cycles", name,
              (unsigned long long)s1.mvm_wait);
        CHECK(slow <= 5.0, "perf: %s: the FA op is %.2f %% slower than with a near-ideal issuer", name, slow);
    }
}

void run_perf() {
    const char * core = getenv("TSR_SIM_RV_CORE");
    printf("fw_kvfa_perf: RISC-V timing model: Ibex %s\n", core && core[0] ? core : tsr_rv_core_name(TSR_RV_CORE_DEFAULT));
    // ---- SET_ROWS_KV decode records (Qwen2.5-32B rows, E = 1024) ----
    {
        kdata k;
        build(find_case("q32b-dec-256"), k);
        const uint32_t n1 = 8, n2 = 24;
        for (int deep = 1; deep >= 0; --deep) {
            fwprof::prof prof[2];
            const bool syms = prof[0].load_symbols(g_image) && prof[1].load_symbols(g_image);
            double w[2] = {};
            for (int j = 0; j < 2; ++j) {
                kdata kk = k;
                const std::vector<uint8_t> blob = setrows_list(kk, j ? n2 : n1);
                w[j] = perf_launch(kk, blob, deep != 0, 0.0, syms ? &prof[j] : nullptr).work;
            }
            const double per = (w[1] - w[0]) / double(n2 - n1);
            printf("fw_kvfa_perf: SET_ROWS_KV decode record (E 1024, %s queues): %.0f RV cycles of work = %.2f us at %.1f MHz\n",
                   deep ? "deep" : "U200", per, per / g_f_rv, g_f_rv);
            if (syms && deep) fwprof::prof::diff(prof[1], prof[0]).report(stdout, "fw_kvfa_perf: profile per SET_ROWS_KV record", double(n2 - n1), 12, ".setrows");
            CHECK(per <= 600.0, "perf: SET_ROWS_KV decode record %.0f RV cycles (regression ceiling 600)", per);
        }
    }
    // ---- FLASH_ATTN: Qwen2.5-32B decode at 4,096 cells (AT §9.2), then the other shapes ----
    fa_perf("q32b-dec-4096", true);
    for (const char * nm : { "q05b-dec-4096", "g12swa-dec-4096", "g12glb-dec-4096", "q05b-pre-4096", "q32b-pre-4096",
                             "g12swa-pre-4096-slide" })
        fa_perf(nm, false);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE suite GROUP|kl|perf [--scheds N] [--threads N] [--only NAME]\n", argv[0]);
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
    } else if (mode == "kl") {
#if defined(TSR_KVFA_GGML)
        run_kl();
#else
        printf("%s: built without ggml (TSR_WITH_LLAMA off): skipped\n", argv[0]);
        return 77;
#endif
    } else if (mode == "perf") {
        run_perf();
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    return test_finish(("fw_kvfa_rv " + mode + (group.empty() ? "" : " " + group)).c_str());
}

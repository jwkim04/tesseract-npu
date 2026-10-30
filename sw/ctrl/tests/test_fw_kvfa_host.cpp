// test_fw_kvfa_host.cpp — fw_kvfa_host: the SET_ROWS_KV and FLASH_ATTN kinds compiled for the host
// (docs/design/06-software-verification.md "SW" §6.8, SW-25: tsr_fw_host under ASan + UBSan), the
// `graph` executor run through tsr::host_run on the simulator's test hooks (tsr/backend_host.hpp).
// The reference stays the RV binary on tsr_rv (fw_kvfa_rv_*); this build adds the sanitizers.
//
//   1. a sample of test_fw_kvfa.h's cases (every shape and mask kind, decode and prefill), fixed
//      schedule, race fatal: RET 0, planes and output rows bit-exact against sw/ctrl/ref/kv_ref.cpp /
//      fattn_ref.cpp;
//   2. SET_ROWS_KV alone with rows the kernel splits (70 consecutive cells: runs of 64 + 6; strided
//      source rows; repeated and jumping indices: the later row wins), planes against kv_ref.cpp;
//   3. refused records (RET BAD_RECORD 0x8000_0004 / SCRATCH 0x8000_0006, and the refused op pushes
//      nothing: its push count = the empty list's): FLASH_ATTN softcap != 0, head_dim != D, n_head !=
//      G n_head_kv, a q head stride != D/16, reserved flag bits, a bad MSUM magic / n_q / n_kv, MSUM rows
//      beyond the preload area, a tile that
//      needs GENERAL without the f32 mask copy, a block larger than the ring; SET_ROWS_KV which 2,
//      plane kind 1, row elements != E, an index beyond the plane, an index >= 2^32, indices outside
//      the preload area.
// Exit 0 / 1.
#undef _GNU_SOURCE
#include "test_fw_kvfa.h"

#include "tsr/exec.hpp"
#include "tsr/rt.hpp"
#include "tsr_race.h"

extern "C" int32_t graph(const tsr_args * args);

namespace {

using namespace kvfa;

struct host_out {
    int32_t ret;
    uint32_t err, pushes;
};

// One launch of k's list on the bound simulator (host_run: the kernel, join_all, the first error).
host_out launch(tsr_sim * sim, const kdata & k) {
    memcpy(at(sim, k.opl_h), k.blob.data(), k.blob.size());
    tsr_args args{};
    args.a[0] = k.opl_h;
    args.a[1] = uint32_t(k.blob.size());
    args.a[2] = tsr::opl::LF_VERIFY_CRC;
    const tsr::host_result h = tsr::host_run(&graph, args, nullptr, 0u);
    if (tsr::err_units() != 0u) tsr::join(tsr::qclear());
    return host_out{ h.ret, h.err_code, h.pidx_end - h.pidx0 };
}

tsr_sim * make_sim(const tsr_config & c) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = 2;
    so.threads = 2;
    tsr_sim * sim = tsr_sim_create_opts(&c, &so);
    CHECK(sim != nullptr, "simulator");
    if (sim) tsr::host_bind(sim);
    tsr::host_spm_lines = c.spm_bytes / 64u;   // the host build's SPM size (rt.hpp; the RISC-V reads CFG1)
    return sim;
}
void drop_sim(tsr_sim * sim) {
    tsr::host_bind(nullptr);
    if (sim) tsr_sim_destroy(sim);
}

// 1. / 2.: a list whose results are compared with the references.
void run_good(kdata & k, const char * tag) {
    tsr_sim * sim = make_sim(k.cfg);
    if (!sim) return;
    fill(sim, k);
    expect(sim, k);
    const host_out r = launch(sim, k);
    CHECK(r.ret == 0 && r.err == 0u, "%s [%s]: RET 0x%08x, error %u", k.c->name, tag, uint32_t(r.ret), r.err);
    if (r.ret == 0) check_outputs(sim, k, tag, nullptr);
    const tsr_race_report * rr = nullptr;
    CHECK(tsr_sim_race_reports(sim, &rr) == 0u, "%s [%s]: race reports", k.c->name, tag);
    printf("  %-26s [%s] T %2u n_kv %5u: %u pushes, chunks %u (general %u)\n", k.c->name, tag, k.c->T, k.c->n_kv, r.pushes,
           k.st.chunks, k.st.general_chunks);
    drop_sim(sim);
}

const kcase & find(const char * name) {
    for (const auto & c : cases())
        if (strcmp(c.name, name) == 0) return c;
    fprintf(stderr, "no case %s\n", name);
    exit(2);
}

// 3.: a mutated list must end with RET `want` and the refused record must push nothing (base: the
// pushes of the same list without the refused record, `base_sets`: SET_ROWS records kept).
void run_bad(const char * what, kdata k, uint32_t want, bool sets, const tsr_config * cfg = nullptr) {
    const tsr_config c = cfg ? *cfg : k.cfg;
    // Base: the records before the refused one (none, or the two SET_ROWS), same preload.
    uint32_t base = 0;
    {
        kdata b = k;
        tsr_sim * sim = make_sim(c);
        if (!sim) return;
        fill(sim, b);
        if (sets) make_blob(b, true, false);
        else {   // an empty list: one NOP record
            tsr::opl::writer w(tsr::opl::F_STOP_ON_ERROR, 0x6b7666u);
            w.add_preload(b.idx_h, uint32_t(b.idx.size()));
            w.add_preload(b.msum_h, uint32_t(b.msum.size() * 4u));
            w.add(tsr::opl::K_NOP, {});
            b.blob = w.finish();
        }
        const host_out r = launch(sim, b);
        base = r.pushes;
        drop_sim(sim);
    }
    tsr_sim * sim = make_sim(c);
    if (!sim) return;
    fill(sim, k);
    const host_out r = launch(sim, k);
    CHECK(uint32_t(r.ret) == want && r.err == 0u, "%s: RET 0x%08x (expected 0x%08x), error %u", what, uint32_t(r.ret), want, r.err);
    CHECK(r.pushes == base, "%s: %u pushes, the list without the refused record %u", what, r.pushes, base);
    printf("  refused: %-52s RET 0x%08x, %u pushes (= %u)\n", what, uint32_t(r.ret), r.pushes, base);
    drop_sim(sim);
}

}  // namespace

int main() {
    // 1. A sample of the cases.
    static const char * const sample[] = { "q05b-dec-16", "q05b-pre-256", "q32b-dec-256", "q32b-pre-256", "g12swa-dec-256",
                                           "g12swa-pre-16", "g12swa-ring-dec", "g12swa-ring-pre-wrap", "g12glb-dec-256",
                                           "g12glb-pre-256", "general-finite-dec-256", "general-finite-g12swa",
                                           "general-holes-q05b", "empty-row-q05b" };
    for (const char * nm : sample) {
        kdata k;
        build(find(nm), k);
        run_good(k, "host");
    }
    // 2. SET_ROWS_KV alone: 70 consecutive cells, then repeats and jumps.
    {
        std::vector<uint32_t> cells;
        for (uint32_t j = 100; j < 170; ++j) cells.push_back(j);
        for (uint32_t j : { 5u, 6u, 5u, 255u, 0u, 1u, 2u, 3u, 16u, 31u, 32u }) cells.push_back(j);
        for (const char * nm : { "q05b-dec-256", "q32b-dec-256", "g12glb-dec-256" }) {
            kdata k;
            build_setrows(k, find(nm), cells, 3u);
            run_good(k, "setrows");
        }
    }
    // 3. Refused records.
    {
        namespace fa = tsr::opl::fa;
        const uint32_t BAD = tsr::opl::RET_BAD_RECORD, SCR = tsr::opl::RET_SCRATCH;
        kdata g;
        build(find("q32b-dec-256"), g);
        auto fa_bad = [&](const char * what, auto mut, uint32_t want = tsr::opl::RET_BAD_RECORD) {
            kdata k = g;
            mut(k);
            make_blob(k, false, true);
            run_bad(what, k, want, false);
        };
        fa_bad("FLASH_ATTN softcap != 0", [](kdata & k) { k.fa.softcap = 0x3F800000u; });
        fa_bad("FLASH_ATTN head_dim != D", [](kdata & k) { k.fa.head_dim = 64u; });
        fa_bad("FLASH_ATTN n_head != G n_head_kv", [](kdata & k) { k.fa.n_head = 41u; });
        fa_bad("FLASH_ATTN q head stride != D/16", [](kdata & k) { k.fa.q_head_stride = 9u; });
        fa_bad("FLASH_ATTN reserved flag bit", [](kdata & k) { k.fa.flags |= 2u; });
        fa_bad("FLASH_ATTN scale <= 0", [](kdata & k) { k.fa.scale = 0xBF800000u; });
        fa_bad("FLASH_ATTN MSUM magic", [](kdata & k) { k.msum[0] ^= 1u; });
        fa_bad("FLASH_ATTN MSUM n_q != n_tokens", [](kdata & k) { k.msum[1] = 2u; });
        fa_bad("FLASH_ATTN MSUM n_kv != n_kv", [](kdata & k) { k.msum[2] = k.fa.n_kv + 16u; });   // W2b fix (review)
        {
            // W2b fix (review): the MSUM's rows must lie inside the preload area. T = 3: header + rows =
            // 80 bytes; its preload entry cut to 64 (the header and two rows) ends the area one line
            // after the MSUM's start, so the third row lies beyond it (a check one row short accepts).
            kdata k;
            build(find("q32b-pre-256"), k);
            k.msum.resize(tsr::opl::MSUM_HDR_WORDS + tsr::opl::MSUM_ROW_WORDS * (k.c->T - 1u));
            make_blob(k, false, true);
            CHECK(k.pre.size() == k.pre_msum + 64u, "MSUM rows test: the area ends 64 bytes after the MSUM (%zu, %u)",
                  k.pre.size(), k.pre_msum);
            run_bad("FLASH_ATTN MSUM rows beyond the preload area", k, BAD, false);
        }
        fa_bad("FLASH_ATTN new-cell indices outside the area", [](kdata & k) { k.fa.flags = tsr::plan::attn::fa_flags(0u, 5000u); });
        fa_bad("FLASH_ATTN GENERAL MSUM without the f32 copy", [](kdata & k) { k.msum[3] = 1u; k.fa.mask = tsr::opl::HANDLE_NONE; });
        {
            kdata k;
            build(find("g12swa-ring-pre-wrap"), k);   // a tile needs GENERAL (two runs per row)
            k.fa.mask = tsr::opl::HANDLE_NONE;
            make_blob(k, false, true);
            run_bad("FLASH_ATTN GENERAL tile without the f32 copy", k, BAD, false);
        }
        {
            tsr_config small = g.cfg;
            small.spm_bytes = 512u << 10;            // budget 3,376 lines < the 32B block (18,948)
            kdata k = g;
            make_blob(k, false, true);
            run_bad("FLASH_ATTN block larger than the ring", k, SCR, false, &small);
        }
        auto kv_bad = [&](const char * what, auto mut) {
            kdata k = g;
            mut(k);
            make_blob(k, true, false);
            run_bad(what, k, BAD, false);   // the refused record is the first: base = the empty list
        };
        kv_bad("SET_ROWS_KV which 2", [](kdata & k) { k.sk.which = 2u; });
        kv_bad("SET_ROWS_KV plane kind 1", [](kdata & k) { k.sk.plane_kind = 1u; });
        kv_bad("SET_ROWS_KV row elements != E", [](kdata & k) { k.sk.row_elems = 1000u; });
        kv_bad("SET_ROWS_KV index beyond the plane", [](kdata & k) { const uint32_t c = 1024u; memcpy(k.idx.data(), &c, 4); });
        kv_bad("SET_ROWS_KV index >= 2^32", [](kdata & k) { k.idx[4] = 1u; });
        kv_bad("SET_ROWS_KV indices outside the area", [](kdata & k) { k.sk.n_rows = 9u; k.sk.idx = 64u * 2u; });
        kv_bad("SET_ROWS_KV E/64 = 0", [](kdata & k) { k.sk.geom = tsr::opl::geom_word(0u, 64u); k.sk.row_elems = 0u; });
    }
    return test_finish("fw_kvfa_host");
}

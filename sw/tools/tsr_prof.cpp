// tsr_prof.cpp — tsr-prof (docs/design/05-debug.md "DB" §8.3, §8.6, §8.7; docs/prof-dbg.md §3; docs/design/
// 06-software-verification.md "SW" §9.5): the trace-writer source `trw` and the PERF summary.
//
//   tsr-prof trw [--device KIND] [--image F] --launch K[:a0,a1,...] [--launch ...] [--repeat N] [--prof]
//                [--ring-mib M] [--stop-full] [--out trace.json] [--raw ring.bin] [--check] [--stats]
//       runs the launches with the trace writer on (the ring at the top of the ch0 IO region, EX §11,
//       TSR_PROF_TRW_MIB / --ring-mib, default 64 MiB), drains the ring and writes the Perfetto /
//       Chrome trace-event timeline (DB §8.6: process Tesseract0, a thread per unit with RUN slices
//       and "waiting" slices, the RISC-V thread with launch and node slices from the markers, stops
//       and RV halts as instants, per-channel read GB/s and idle % counter tracks from PERF per
//       launch). K is a kernel name of the image, or "ping". --check runs the DB §8.7 self-checks
//       (exit 1 on a failure); --stats prints the DB §8.6 summary (the TSR_STATS lines) of the run.
//   tsr-prof trw --from-raw ring.bin [--image F] [--core-mhz F] --out trace.json
//       the timeline of saved records (tsr-prof trw --raw, tsr-dbg's monitor trw dump, hang dumps).
//   tsr-prof stats [--device KIND] [--image F] --launch ... [--repeat N]
//       the DB §8.6 summary only (no trace writer).
//
// Exit status: 0, 1 (a launch failed or a check failed), 2 (usage).

#include "tsr_dbg_lib.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace tsr::dbg;

namespace {

void usage() {
    fprintf(stderr,
            "usage: tsr-prof trw [--device KIND] [--image F] --launch K[:a0,a1,..] [...] [--repeat N] [--prof] [--ring-mib M] [--stop-full]\n"
            "                    [--out trace.json] [--raw ring.bin] [--check] [--stats]\n"
            "       tsr-prof trw --from-raw ring.bin [--image F] [--core-mhz F] --out trace.json\n"
            "       tsr-prof stats [--device KIND] [--image F] --launch K[:a0,..] [...] [--repeat N]\n");
}

struct launch_spec {
    std::string kernel;
    std::vector<uint32_t> args;
};

bool parse_launch(const std::string & s, launch_spec * l) {
    const size_t c = s.find(':');
    l->kernel = s.substr(0, c);
    if (c == std::string::npos) return !l->kernel.empty();
    std::string rest = s.substr(c + 1);
    size_t i = 0;
    while (i <= rest.size()) {
        const size_t j = rest.find(',', i);
        const std::string a = rest.substr(i, j == std::string::npos ? std::string::npos : j - i);
        uint32_t v = 0;
        if (!parse_u32(a, &v)) return false;
        l->args.push_back(v);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return l->args.size() <= 7 && !l->kernel.empty();
}

uint64_t mtime(tsr_transport * t) {
    const uint32_t hi = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_HI);
    uint32_t lo = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_LO);
    const uint32_t hi2 = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_HI);
    if (hi2 != hi) lo = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_LO);
    return (uint64_t) hi2 << 32 | lo;
}

bool write_file(const std::string & path, const std::string & s) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) return false;
    fputs(s.c_str(), f);
    return fclose(f) == 0;
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    if (a.empty() || (a[0] != "trw" && a[0] != "stats")) {
        usage();
        return 2;
    }
    const bool trw_mode = a[0] == "trw";
    std::string device, image, out, raw, from_raw;
    std::vector<launch_spec> launches;
    uint32_t repeat = 1, ring_mib = 0;
    double core_mhz = 276.3;
    bool stop_full = false, check = false, stats = false, prof = false;
    for (size_t i = 1; i < a.size(); ++i) {
        auto val = [&]() -> std::string {
            if (i + 1 >= a.size()) {
                fprintf(stderr, "tsr-prof: %s needs a value\n", a[i].c_str());
                exit(2);
            }
            return a[++i];
        };
        const std::string & o = a[i];
        if (o == "--device") device = val();
        else if (o == "--image" || o == "--elf") image = val();
        else if (o == "--out") out = val();
        else if (o == "--raw") raw = val();
        else if (o == "--from-raw") from_raw = val();
        else if (o == "--launch") {
            launch_spec l;
            if (!parse_launch(val(), &l)) {
                fprintf(stderr, "tsr-prof: --launch %s: expected KERNEL[:a0,a1,...] (at most 7 arguments)\n", a[i].c_str());
                return 2;
            }
            launches.push_back(l);
        } else if (o == "--repeat") {
            if (!parse_u32(val(), &repeat) || repeat == 0) return usage(), 2;
        } else if (o == "--ring-mib") {
            if (!parse_u32(val(), &ring_mib)) return usage(), 2;
        } else if (o == "--core-mhz") {
            core_mhz = atof(val().c_str());
        } else if (o == "--stop-full") stop_full = true;
        else if (o == "--check") check = true;
        else if (o == "--stats") stats = true;
        else if (o == "--prof") prof = true;
        else {
            fprintf(stderr, "tsr-prof: unknown option %s\n", o.c_str());
            usage();
            return 2;
        }
    }
    image_info img;
    std::string why;
    if (!image.empty() && !img.load_path(image, &why)) {
        fprintf(stderr, "tsr-prof: %s\n", why.c_str());
        return 2;
    }

    // Saved records: the timeline only.
    if (!from_raw.empty()) {
        std::vector<trw_rec> recs;
        if (!trw_load_raw(from_raw, &recs, &why)) {
            fprintf(stderr, "tsr-prof: %s\n", why.c_str());
            return 1;
        }
        timeline_opts to;
        to.core_hz = core_mhz * 1e6;
        to.img = img.loaded() ? &img : nullptr;
        if (out.empty()) out = "trace.json";
        if (!write_file(out, trw_chrome_json(recs, to))) {
            fprintf(stderr, "tsr-prof: cannot write %s\n", out.c_str());
            return 1;
        }
        printf("tsr-prof: %zu records -> %s\n", recs.size(), out.c_str());
        return 0;
    }
    if (launches.empty()) {
        fprintf(stderr, "tsr-prof: nothing to run (--launch)\n");
        return 2;
    }
    if (!img.loaded()) {
        tsr_image * def = nullptr;
        if (tsr_image_open_default(&def) != 0 || !def) {
            fprintf(stderr, "tsr-prof: no firmware image (--image or TSR_IMAGE)\n");
            return 2;
        }
        img.load_path(tsr_image_path(def), &why);
        tsr_image_close(def);
    }
    tsr_transport * t = tsr_transport_open(device.empty() ? nullptr : device.c_str());
    if (!t) return 1;
    int status = 0;
    int rc = tsr_transport_boot(t, img.image());
    if (rc) {
        fprintf(stderr, "tsr-prof: boot failed: %s\n", tsr_transport_strerror(rc));
        tsr_transport_close(t);
        return 1;
    }
    const tsr_config * cfg = tsr_transport_config(t);
    trw_ring ring;
    if (trw_mode) {
        if (ring_mib == 0) {
            const char * e = getenv("TSR_PROF_TRW_MIB");
            ring_mib = e && e[0] ? (uint32_t) atoi(e) : TSR_TRW_RING_DEFAULT_MIB;
        }
        ring = trw_default_ring(ring_mib);
        ring.stop_full = stop_full;
        if (trw_start(t, ring, &why) != 0) {
            fprintf(stderr, "tsr-prof: %s\n", why.c_str());
            tsr_transport_close(t);
            return 1;
        }
    }
    tsr_perf_summary p0, p1;
    tsr_transport_perf_summary(t, &p0);
    timeline_opts to;
    to.core_hz = p0.core_hz;
    to.img = &img;
    to.num_ch = cfg->num_ch;
    for (uint32_t r = 0; r < repeat; ++r) {
        for (const launch_spec & ls : launches) {
            tsr_launch l;
            memset(&l, 0, sizeof(l));
            if (ls.kernel == "ping") {
                l.type = TSR_DESC_TYPE_PING;
            } else {
                l.type = TSR_DESC_TYPE_LAUNCH;
                const int k = img.kernel_index(ls.kernel);
                if (k < 0) {
                    fprintf(stderr, "tsr-prof: the image has no kernel %s\n", ls.kernel.c_str());
                    tsr_transport_close(t);
                    return 2;
                }
                l.kernel = (uint16_t) k;
                l.flags = prof ? 1u : 0u;
            }
            for (size_t k = 0; k < ls.args.size(); ++k) l.arg[k] = ls.args[k];
            tsr_perf_summary b, e;
            tsr_transport_perf_summary(t, &b);
            perf_window w;
            w.t_begin = mtime(t);
            tsr_record rec;
            rc = tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec);
            w.t_end = mtime(t);
            tsr_transport_perf_summary(t, &e);
            for (int k = 0; k < 64; ++k) w.perf[k] = e.perf[k] - b.perf[k];
            to.windows.push_back(w);
            tsr_result res;
            tsr_record_decode(&rec, &res);
            if (rc || res.status != 0) {
                fprintf(stderr, "tsr-prof: launch %s: %s, status %s, RET %d\n", ls.kernel.c_str(), tsr_transport_strerror(rc),
                        tsr_status_name(res.status), res.ret);
                status = 1;
            }
        }
    }
    tsr_transport_perf_summary(t, &p1);
    uint64_t pd[64];
    for (int k = 0; k < 64; ++k) pd[k] = p1.perf[k] - p0.perf[k];
    if (trw_mode) {
        std::vector<trw_rec> recs;
        uint64_t lost = 0;
        if (trw_drain(t, ring, &recs, &lost, &why) != 0) {
            fprintf(stderr, "tsr-prof: %s\n", why.c_str());
            status = 1;
        }
        trw_stop(t);
        const uint64_t drops = pd[TSR_PERF_TRW_DROPS] + lost;
        uint32_t ops = 0, marks = 0;
        for (const trw_rec & x : recs) {
            ops += x.kind == TSR_TRW_W0_KIND_OP;
            marks += x.kind == TSR_TRW_W0_KIND_MARK;
        }
        if (out.empty()) out = "trace.json";
        if (!write_file(out, trw_chrome_json(recs, to))) {
            fprintf(stderr, "tsr-prof: cannot write %s\n", out.c_str());
            status = 1;
        }
        if (!raw.empty()) trw_save_raw(recs, raw);
        printf("tsr-prof trw: %zu records (%u ops, %u markers), %" PRIu64 " dropped or overwritten; PERF TRW_RECORDS %" PRIu64
               "; timeline %s\n",
               recs.size(), ops, marks, drops, pd[TSR_PERF_TRW_RECORDS], out.c_str());
        if (check) {
            const trw_check_result cr = trw_check(recs, pd, cfg->num_ch, drops);
            for (const std::string & l : cr.lines) printf("tsr-prof check: %s\n", l.c_str());
            if (!cr.ok) status = 1;
        }
    }
    if (stats || !trw_mode) {
        tsr_perf_summary run = p1;
        for (int k = 0; k < 64; ++k) run.perf[k] = pd[k];
        run.launches = p1.launches - p0.launches;
        run.launch_wall_ns = p1.launch_wall_ns - p0.launch_wall_ns;
        run.log_dropped_words = p1.log_dropped_words - p0.log_dropped_words;
        char b[4096];
        tsr_perf_summary_str(&run, b, sizeof(b));
        printf("%s", b);
    }
    tsr_transport_close(t);
    return status;
}

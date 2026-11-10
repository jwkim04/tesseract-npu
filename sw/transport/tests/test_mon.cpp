// test_mon.cpp — ctest transport_mon: the live monitor record (tsr_mon.h, docs/prof-dbg.md §7, SW-87).
//
// Covers: the record of an attached transport (identity, configuration, image, kind table, the
// fields the owner sets), the file's name and mode, that the file appears whole, the counters after
// launches (launch count, op records, wall time, the launch in flight), a PERF read (totals equal
// the transport's own summary, the deltas between two reads), the seqlock under a writer that never
// stops (a reader must never see a torn page), the layout table, TSR_MON=0 (no file), the stale file
// of a dead pid left alone, and that closing the transport removes the file. The capacity record of a
// card (tsr-cap-<bdf>: written once, read back, replaced, refused for a bad address, left by the process).
//
// `test-tsr-mon layout` prints the layout table as JSON (the Python reader's test compares it).

#include "tsr_fw_paths.h"
#include "tsr_mon.h"
#include "tsr_transport.h"

#include <atomic>
#include <cinttypes>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        g_checks++;                                                               \
        if (!(cond)) {                                                            \
            g_fail++;                                                             \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);       \
            fprintf(stderr, __VA_ARGS__);                                         \
            fprintf(stderr, "\n");                                                \
        }                                                                         \
    } while (0)

static bool exists(const char * p) {
    struct stat st;
    return stat(p, &st) == 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "layout") == 0) {
        char buf[16384];
        tsr_mon_layout_json(buf, sizeof(buf));
        printf("%s\n", buf);
        return 0;
    }
    const std::string dir = argc > 1 ? argv[1] : "tsr_mon_test";
    mkdir(dir.c_str(), 0755);
    setenv("TSR_MON_DIR", dir.c_str(), 1);
    setenv("TSR_MON", "1", 1);
    setenv("TSR_MON_PERF_MS", "0", 1);          // PERF after every launch: the test reads it often
    setenv("TSR_DEVICE", "sim", 1);
    char path[320];
    tsr_mon_path(path, sizeof(path), (uint32_t) getpid());
    unlink(path);

    // a record of a dead pid is not ours to touch
    char stale[320];
    tsr_mon_path(stale, sizeof(stale), 2147483000u);
    if (FILE * f = fopen(stale, "wb")) {
        fputs("stale", f);
        fclose(f);
    }

    tsr_image * img = nullptr;
    tsr_transport * t = tsr_transport_open("sim");
    CHECK(t != nullptr, "open sim");
    CHECK(tsr_image_open(TSR_FW_DEFAULT_IMAGE, &img) == 0, "open tsr-fw.elf");
    CHECK(t != nullptr && img != nullptr && tsr_transport_boot(t, img) == 0, "boot");
    if (t == nullptr || img == nullptr) return 1;
    CHECK(!exists(path), "no record before the owner attaches");

    CHECK(tsr_mon_enabled() == 1, "TSR_MON on by default");
    tsr_mon_attach(t, "test");
    CHECK(exists(path), "the record exists after attach: %s", path);
    struct stat st;
    CHECK(stat(path, &st) == 0 && st.st_size == TSR_MON_BYTES && (st.st_mode & 0777) == 0644, "file size %lld mode %o",
          (long long) st.st_size, (unsigned) (st.st_mode & 0777));
    CHECK(!exists((std::string(path) + ".tmp").c_str()), "no temporary file left");
    CHECK(exists(stale), "the stale record of another pid is left alone");

    tsr_mon_rec r;
    CHECK(tsr_mon_read(path, &r) == 0, "read the record");
    CHECK(memcmp(r.magic, TSR_MON_MAGIC, 8) == 0 && r.layout == TSR_MON_LAYOUT && r.size == TSR_MON_BYTES, "header");
    CHECK(r.pid == (uint32_t) getpid() && r.uid == (uint32_t) getuid(), "pid %u uid %u", r.pid, r.uid);
    CHECK(r.proc_start_ticks != 0, "start ticks");
    CHECK(strcmp(r.transport, "sim") == 0 && strcmp(r.owner, "test") == 0 && (r.flags & TSR_MON_F_SIM) != 0, "transport %s owner %s",
          r.transport, r.owner);
    const tsr_config * cfg = tsr_transport_config(t);
    CHECK(r.num_ch == cfg->num_ch && r.t_max == cfg->t_max && r.caps == cfg->caps && r.ch_bytes == cfg->ch_bytes, "config");
    CHECK(r.mem_total == (uint64_t) cfg->num_ch * cfg->ch_bytes, "mem_total");
    tsr_versions v;
    CHECK(tsr_transport_versions(t, &v) == 0 && r.image_id == v.image_id && r.basis == v.basis && r.n_kinds == v.n_kinds &&
              memcmp(r.kind_rev, v.kind_rev, sizeof(r.kind_rev)) == 0 && r.core_ver == v.core, "versions");
    CHECK(r.image_id != 0 && strstr(r.image_path, "tsr-fw") != nullptr, "image %s", r.image_path);
    CHECK(strstr(r.cmdline, "test-tsr-mon") != nullptr, "cmdline '%s'", r.cmdline);
    CHECK(r.comm[0] != '\0', "comm");
    CHECK(r.perf_reads == 1 && r.core_hz > 1000000, "first PERF read at attach, core_hz %" PRIu64, r.core_hz);

    // what the owner sets
    tsr_mon_set_model("tiny.gguf", "Q4_K");
    tsr_mon_set_mode("M7", 0x1234, "NOP JOIN MUL_MAT");
    tsr_mon_set_mem(1ull << 30, 0);
    tsr_mon_add_tokens(3, 5, 4);
    tsr_mon_launch_begin();
    CHECK(tsr_mon_read(path, &r) == 0 && r.launch_begin_mono_ns != 0, "a launch in flight is visible");
    CHECK(r.launches == 0 && strcmp(r.model, "tiny.gguf") == 0 && strcmp(r.quant, "Q4_K") == 0 && strcmp(r.mode, "M7") == 0 &&
              r.kind_mask == 0x1234 && strcmp(r.kinds, "NOP JOIN MUL_MAT") == 0 && r.mem_used == (1ull << 30) &&
              r.tok_decode == 3 && r.tok_prefill == 5 && r.passes == 4, "owner fields");
    tsr_mon_launch_end(1500000, 12, 0);
    CHECK(tsr_mon_read(path, &r) == 0, "read");
    CHECK(r.launch_begin_mono_ns == 0 && r.launches == 1 && r.launch_ns == 1500000 && r.ops == 12 && r.last_launch_ns == 1500000,
          "after one launch: %" PRIu64 " launches", r.launches);
    CHECK(r.perf_reads == 2 && r.perf_launches == 1 && r.d_launches == 1 && r.d_wall_ns > 0, "PERF read after the launch");

    // a launch that did not answer is counted but PERF is not read
    tsr_mon_launch_begin();
    tsr_mon_launch_end(10, 1, -3);
    CHECK(tsr_mon_read(path, &r) == 0 && r.launches == 2 && r.perf_reads == 2, "a failed launch: no PERF read");

    // the totals are the transport's own summary
    tsr_perf_summary ps;
    CHECK(tsr_transport_perf_summary(t, &ps) == 0, "perf summary");
    tsr_mon_launch_begin();
    tsr_mon_launch_end(10, 0, 0);
    CHECK(tsr_mon_read(path, &r) == 0, "read");
    bool same = true;
    for (uint32_t i = 0; i < TSR_MON_PERF_SLOTS; ++i) same = same && r.perf[i] == ps.perf[i];
    CHECK(same, "perf[] equals tsr_transport_perf_summary");
    CHECK(r.dperf[TSR_PERF_CYCLES] == 0, "no cycles between two reads with no device launch");

    // the seqlock: a writer that never stops against a reader that must never see a torn page.
    // launch_ns = 7 launches and ops = 3 launches hold on every consistent page.
    tsr_mon_set_model("stress.gguf", "Q8_0");
    const uint64_t base_launches = r.launches;
    const uint64_t base_ns = r.launch_ns, base_ops = r.ops;
    std::atomic<bool> stop { false };
    std::atomic<uint64_t> n_written { 0 };
    std::thread w([&] {
        while (!stop.load()) {
            tsr_mon_launch_begin();
            tsr_mon_launch_end(7, 3, -1);
            n_written++;
        }
    });
    uint64_t reads = 0, torn = 0, last = base_launches;
    const double t_end = [] { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }() + 1.0;
    for (;;) {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        if (ts.tv_sec + ts.tv_nsec * 1e-9 > t_end) break;
        tsr_mon_rec x;
        if (tsr_mon_read(path, &x) != 0) { torn++; continue; }
        reads++;
        const uint64_t n = x.launches - base_launches;
        if (x.launch_ns - base_ns != 7 * n || x.ops - base_ops != 3 * n || x.launches < last) torn++;
        if (x.launch_begin_mono_ns != 0 && x.update_mono_ns == 0) torn++;
        last = x.launches;
    }
    stop = true;
    w.join();
    CHECK(torn == 0 && reads > 100, "seqlock: %" PRIu64 " reads, %" PRIu64 " torn, %" PRIu64 " launches written", reads, torn, n_written.load());

    // the cost of the hooks without a PERF read: a launch_begin / launch_end pair
    {
        tsr_mon_rec x;
        CHECK(tsr_mon_read(path, &x) == 0, "read");
        const uint64_t h0 = x.hook_ns, l0 = x.launches;
        const int n = 20000;
        timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int i = 0; i < n; ++i) {
            tsr_mon_launch_begin();
            tsr_mon_launch_end(7, 3, -1);
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        const double us = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / 1e3 / n;
        CHECK(tsr_mon_read(path, &x) == 0 && x.launches - l0 == (uint64_t) n, "launches counted");
        const double hook_us = (double) (x.hook_ns - h0) / 1e3 / n;
        printf("hooks: %.2f us per launch (%.2f us measured inside), PERF reads %u, %.1f us each\n", us, hook_us, x.perf_reads,
               x.perf_reads ? (double) x.perf_ns / 1e3 / x.perf_reads : 0.0);
        CHECK(us < 50.0, "a launch_begin / launch_end pair costs %.2f us", us);
        CHECK(x.perf_ns > 0 && x.hook_ns >= x.perf_ns, "perf_ns %llu hook_ns %llu", (unsigned long long) x.perf_ns, (unsigned long long) x.hook_ns);
    }

    // closing the transport removes the file
    tsr_image_close(img);
    tsr_transport_close(t);
    CHECK(!exists(path), "the record is removed when the transport closes");
    CHECK(exists(stale), "the stale record of another pid is still there");
    unlink(stale);

    // The capacity record of a card (tsr_mon_cap_*): what an owner open measured, for the view of an idle card.
    {
        const char * bdf = "0000:3b:00.1";
        char cp[320];
        tsr_mon_cap_path(cp, sizeof(cp), bdf);
        unlink(cp);
        uint32_t n = 0;
        uint64_t b = 0;
        CHECK(tsr_mon_cap_read(bdf, &n, &b) != 0, "no capacity record before the write");
        CHECK(tsr_mon_cap_write(bdf, 4, 8ull << 30) == 0 && exists(cp), "write the capacity record");
        CHECK(strstr(cp, "tsr-cap-0000:3b:00.1") != nullptr && strstr(cp, dir.c_str()) == cp, "its name %s", cp);
        struct stat cst;
        CHECK(stat(cp, &cst) == 0 && (cst.st_mode & 0777) == 0644, "mode %o", (unsigned) (cst.st_mode & 0777));
        CHECK(tsr_mon_cap_read(bdf, &n, &b) == 0 && n == 4 && b == (8ull << 30), "read it back: %u channels of %" PRIu64 " bytes", n, b);
        FILE * f = fopen(cp, "r");
        char txt[256] = {};
        if (f) {
            size_t got = fread(txt, 1, sizeof(txt) - 1, f);
            (void) got;
            fclose(f);
        }
        CHECK(strcmp(txt, "tsr-cap 1\nbdf 0000:3b:00.1\nnum_ch 4\nch_bytes 8589934592\n") == 0, "the file is four text lines: %s", txt);
        CHECK(tsr_mon_cap_write(bdf, 4, 16ull << 30) == 0 && tsr_mon_cap_read(bdf, &n, &b) == 0 && b == (16ull << 30), "a new write replaces it");
        CHECK(tsr_mon_cap_read("0000:3b:00.0", &n, &b) != 0, "another card has no record");
        CHECK(tsr_mon_cap_write("../x", 4, 1) != 0 && tsr_mon_cap_write("", 4, 1) != 0 && tsr_mon_cap_write(bdf, 0, 1) != 0 &&
              tsr_mon_cap_write(bdf, 4, 0) != 0 && tsr_mon_cap_read("a/b", &n, &b) != 0, "bad addresses and values refused");
        CHECK(!exists((std::string(cp) + "." + std::to_string(getpid()) + ".tmp").c_str()), "no temporary file left");
        // A file that is not a record of this card is not read.
        f = fopen(cp, "w");
        if (f) {
            fputs("tsr-cap 1\nbdf 0000:ff:00.0\nnum_ch 4\nch_bytes 8589934592\n", f);
            fclose(f);
        }
        CHECK(tsr_mon_cap_read(bdf, &n, &b) != 0, "a record of another address is refused");
        f = fopen(cp, "w");
        if (f) {
            fputs("tsr-cap 2\nbdf 0000:3b:00.1\nnum_ch 4\nch_bytes 8589934592\n", f);
            fclose(f);
        }
        CHECK(tsr_mon_cap_read(bdf, &n, &b) != 0, "an unknown version is refused");
        CHECK(tsr_mon_cap_write(bdf, 4, 8ull << 30) == 0 && exists(cp), "written again");
        unlink(cp);
    }
    rmdir(dir.c_str());

    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}

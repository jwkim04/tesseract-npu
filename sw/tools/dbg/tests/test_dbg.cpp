// test_dbg.cpp — libtsr_dbg on the sim kind (docs/design/05-debug.md "DB" §3-§8), with the
// fw_rt test image (rt_nop, rt_copy: LD then ST).
//
//   test-tsr-dbg lib <fw_rt.elf> <scratch>
//       the library end to end (dbg.cmake tools_dbg_lib lists the checks)
//   test-tsr-dbg embed-host <fw_rt.elf> <launches>
//       an owning process for the embedded service (TSR_GDB / TSR_GDB_WAIT from the environment):
//       boot, then <launches> rt_nop launches with the default (debug-aware) wait; prints one line
//       per answer. Linked with tsr_dbg_embed, so the service is registered before main.

#include "tsr_dbg_lib.hpp"

#include "tsr_dm.h"
#include "tsr_elf.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace tsr::dbg;

namespace {

int g_fail = 0, g_pass = 0;
#define CHECK(c, ...)                                                   \
    do {                                                                \
        if (c) {                                                        \
            g_pass++;                                                   \
        } else {                                                        \
            g_fail++;                                                   \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                 \
            printf(__VA_ARGS__);                                        \
            printf("\n");                                               \
        }                                                               \
    } while (0)

bool wait_halt(target & tg, tsr_transport * t, uint64_t ns = 500000000ull) {
    const uint64_t t0 = tsr_transport_now_ns(t);
    while (!tg.poll_halted()) {
        if (tsr_transport_now_ns(t) - t0 > ns) return false;
        tsr_transport_idle(t, 20000);
    }
    return true;
}

// Waits for the monitor's pending launches; returns their answer text.
std::string wait_answers(monitor & mon, tsr_transport * t) {
    std::string o;
    for (int i = 0; i < 20000 && !mon.pending.empty(); ++i) {
        o += mon.poll_launches();
        if (!mon.pending.empty()) tsr_transport_idle(t, 20000);
    }
    return o;
}

bool contains(const std::string & s, const std::string & w) { return s.find(w) != std::string::npos; }

uint32_t kind_at(const image_info & img, uint32_t a) {
    uint8_t b[2] = {};
    img.bytes_at(a, b, 2);
    return (b[0] & 3u) != 3u ? 2u : 4u;
}

int test_lib(const std::string & elf, const std::string & scratch) {
    tsr_transport * t = tsr_transport_open("sim");
    if (!t) {
        printf("FAIL: open sim\n");
        return 1;
    }
    image_info img;
    std::string why;
    CHECK(img.load_path(elf, &why), "image %s: %s", elf.c_str(), why.c_str());
    target tg(t, &img);
    monitor mon(tg, true);
    std::string out;

    // 1. Halt-on-boot (DB §3.8): the boot's HELLO wait waits while the hart is halted.
    int e = mon.boot_async(true, &why);
    CHECK(e == 0, "boot_async(halt): %s", why.c_str());
    CHECK(mon.boot_running(), "the boot waits for HELLO while the hart is halted at boot");
    e = tg.attach(&why);
    CHECK(e == 0, "attach: %s", why.c_str());
    stop_info si;
    CHECK(tg.stop(&si) == 0 && si.note == "halted at boot" && si.pc == 0x80u && si.dcsr_cause == 3,
          "the first stop: halted at boot at 0x80 (note \"%s\", pc 0x%08x, cause %u)", si.note.c_str(), si.pc, si.dcsr_cause);
    CHECK(!(tg.csr(TSR_CSR_DBG_CFG) & TSR_DBG_CFG_HALT_ON_BOOT), "HALT_ON_BOOT cleared after its one halt");
    // TSR_BOOT_MS (2 s) counts only while the hart runs: 3 s of device time halted at boot, then
    // some host time for the boot thread's polls, and the boot still waits.
    const uint64_t tb0 = tsr_transport_now_ns(t);
    tsr_transport_idle(t, 3000000000ull);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(tsr_transport_now_ns(t) - tb0 >= 3000000000ull && mon.boot_running(),
          "the boot still waits after %.1f s of device time halted at boot", (double) (tsr_transport_now_ns(t) - tb0) / 1e9);
    CHECK(tg.csr(TSR_CSR_CTRL) & TSR_CTRL_HOLD_ON_HALT, "HOLD_ON_HALT set at attach (DB §3.4)");

    // 2. Registers (DB §3.5, §4.4). DMI accesses against DB §3.10's table: g = 32 x (1 W + 2 R)
    // (x0 is not read), 64 B through SBA = 3 W + 17 R.
    uint32_t r[33];
    uint64_t acc0 = tg.dm().accesses;
    CHECK(tg.read_gprs(r) == 0 && r[0] == 0 && r[32] == 0x80u, "g: x0 0, pc 0x%08x", r[32]);
    const uint64_t acc_g = tg.dm().accesses - acc0;
    uint8_t m64[64];
    acc0 = tg.dm().accesses;
    CHECK(tg.read_mem(TSR_RV_DTCM_BASE, m64, 64) == 0, "64 B of DTCM");
    const uint64_t acc_m = tg.dm().accesses - acc0;
    printf("  DMI accesses: g %" PRIu64 " (DB §3.10: 33 x 3), 64-B SBA read %" PRIu64 " (3 W + 17 R = 20)\n", acc_g, acc_m);
    CHECK(acc_g == 96 && acc_m == 20, "DMI access counts");
    uint32_t misa = 0, hart = 1, v = 0;
    CHECK(tg.read_reg(65 + 0x301, &misa) == 0 && (misa >> 30) == 1 && (misa & (1u << 8)) && (misa & (1u << 12)) && (misa & (1u << 2)),
          "misa 0x%08x: RV32 I M C", misa);
    CHECK(tg.read_reg(65 + 0xF14, &hart) == 0 && hart == 0, "mhartid %u", hart);
    CHECK(tg.write_reg(5, 0x12345678u) == 0 && tg.read_reg(5, &v) == 0 && v == 0x12345678u, "write / read t0: 0x%08x", v);
    CHECK(tg.read_reg(65 + 0x7B0, &v) == E_POLICY, "dcsr is the server's (DB §4.4)");
    CHECK(tg.read_reg(4161, &v) == 0 && v == 3, "priv reads 3");

    // 3. Memory against the ELF (SBA, DB §3.6).
    uint32_t a_main = 0, a_copy = 0, a_nop = 0, a_dbg = 0, a_launch = 0, a_lh = 0;
    CHECK(img.symbol("tsr_rt_main", &a_main) && img.symbol("rt_copy", &a_copy) && img.symbol("rt_nop", &a_nop) &&
              img.symbol("__tsr_dbg", &a_dbg) && img.symbol("tsr_rt_launch", &a_launch) && img.symbol("__tsr_launch_halt", &a_lh),
          "symbols of fw_rt");
    uint8_t m[64], ref[64];
    CHECK(tg.read_mem(a_main, m, 16) == 0 && img.bytes_at(a_main, ref, 16) && !memcmp(m, ref, 16), "ITCM at tsr_rt_main = the ELF");
    CHECK(tg.read_mem(0x81, m, 7) == 0 && img.bytes_at(0x81, ref, 7) && !memcmp(m, ref, 7), "an unaligned ITCM read");
    uint32_t w[4] = {};
    CHECK(tg.read_mem(a_dbg, (uint8_t *) w, 16) == 0 && w[0] == TSR_DBG_INFO_MAGIC && w[3] == 1, "__tsr_dbg: magic 0x%08x, attached %u", w[0],
          w[3]);
    // the write policy (DB §3.6, §4.5)
    const uint8_t pat[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    CHECK(tg.write_mem(TSR_RV_CTL_BASE + 0x100, pat, 4) == E_POLICY, "control-window writes refused");
    CHECK(tg.write_mem(TSR_RV_CORE_BASE + 0x20, pat, 4) == E_POLICY, "core-window writes refused");
    CHECK(tg.write_mem(TSR_RV_DM_BASE + 0x380, pat, 4) == E_POLICY && tg.read_mem(TSR_RV_DM_BASE, m, 4) == E_POLICY, "the DM range refused");
    CHECK(tg.read_mem(TSR_RV_CORE_BASE + TSR_CSR_ID, (uint8_t *) &v, 4) == 0 && v == TSR_ID_MAGIC, "core-window read: ID 0x%08x", v);
    CHECK(tg.write_mem(a_dbg + 33, pat, 7) == 0 && tg.read_mem(a_dbg + 33, m, 7) == 0 && !memcmp(m, pat, 7), "an unaligned DTCM write");
    std::vector<uint8_t> line(64);
    for (int i = 0; i < 64; ++i) line[i] = (uint8_t) (0xA0 + i);
    CHECK(tg.write_mem(TSR_RV_SPM_BASE + 64 * 10, line.data(), 64) == 0 && tg.read_mem(TSR_RV_SPM_BASE + 64 * 10, m, 64) == 0 &&
              !memcmp(m, line.data(), 64), "SPM write while QQUIESCENT and halted");
    CHECK(mon.run("spm x/4x 10", &out) == 0 && contains(out, "0xa3a2a1a0"), "monitor spm: %s", out.c_str());

    // 4. A software breakpoint on a firmware symbol (DB §3.7), continue, hit.
    CHECK(tg.insert_bp(0, a_main, kind_at(img, a_main)) == 0, "Z0 at tsr_rt_main");
    CHECK(tg.read_mem(a_main, m, 2) == 0 && (m[0] | m[1] << 8) == (kind_at(img, a_main) == 2 ? 0x9002 : 0x0073), "the ebreak is in ITCM");
    CHECK(tg.resume() == 0 && wait_halt(tg, t), "continue: halts again");
    CHECK(tg.stop(&si) == 0 && si.reason == "swbreak" && si.pc == a_main, "stopped at the breakpoint: %s pc 0x%08x (want 0x%08x)",
          si.reason.c_str(), si.pc, a_main);
    CHECK(tg.remove_bp(0, a_main, kind_at(img, a_main)) == 0 && tg.read_mem(a_main, m, 16) == 0 && img.bytes_at(a_main, ref, 16) &&
              !memcmp(m, ref, 16), "z0 restores the code");
    w[0] = 0;
    CHECK(tg.read_mem(a_dbg + 8, (uint8_t *) w, 4) == 0 && w[0] == (uint32_t) img.image_id(), "crt0 set the build id 0x%08x", w[0]);
    // 5. Step (DB §3.4).
    CHECK(tg.step(&si) == 0 && si.dcsr_cause == 4 && si.pc != a_main && si.pc - a_main <= 4, "step: cause %u, pc 0x%08x", si.dcsr_cause, si.pc);
    // 6. Run the boot to HELLO.
    CHECK(tg.resume() == 0, "continue");
    CHECK(mon.boot_join(&why) == 0, "the boot completed: %s", why.c_str());

    // 7. Halt-on-launch (DB §3.8).
    CHECK(mon.run("launch-break rt_nop", &out) == 0, "launch-break: %s", out.c_str());
    out.clear();
    CHECK(mon.run("launch rt_nop", &out) == 0 && mon.pending.size() == 1, "launch rt_nop: %s", out.c_str());
    CHECK(wait_halt(tg, t), "the launch halts the hart");
    CHECK(tg.stop(&si) == 0 && si.pc == a_lh && contains(si.note, "launch rt_nop seq"), "launch halt: pc 0x%08x, \"%s\"", si.pc, si.note.c_str());
    CHECK(tg.read_reg(10, &v) == 0 && v == (uint32_t) img.kernel_index("rt_nop"), "a0 = the kernel index (%u)", v);
    CHECK(tg.resume() == 0 && contains(wait_answers(mon, t), "answered: OK"), "the launch completes after the halt");
    CHECK(mon.run("launch-break off", &out) == 0, "launch-break off");

    // 8. A hardware breakpoint (Ibex trigger, DB §3.7) on rt_copy.
    const uint32_t src = 0x100000, dst = 0x100100;   // ch0 beats in the IO region (EX §11)
    std::vector<uint8_t> data(128), back(128);
    for (int i = 0; i < 128; ++i) data[i] = (uint8_t) (i * 7 + 1);
    CHECK(tsr_transport_write(t, 0, (uint64_t) src * 64, data.data(), 128) == 0, "source data");
    CHECK(tg.interrupt() == 0 && wait_halt(tg, t) && tg.stop(&si) == 0 && si.signal == 2, "Ctrl-C: SIGINT (signal %d)", si.signal);
    CHECK(tg.insert_bp(1, a_copy, 2) == 0, "Z1 at rt_copy");
    out.clear();
    CHECK(mon.run("launch rt_copy 0x100000 0x100100 2 0", &out) == 0, "launch rt_copy: %s", out.c_str());
    CHECK(tg.resume() == 0 && wait_halt(tg, t) && tg.stop(&si) == 0 && si.reason == "hwbreak" && si.pc == a_copy,
          "hardware breakpoint: %s at 0x%08x (want 0x%08x)", si.reason.c_str(), si.pc, a_copy);
    CHECK(tg.remove_bp(1, a_copy, 2) == 0, "z1");

    // 9. A tile breakpoint that halts the hart (DB §5.2, §5.3, §4.7), QSTEP, the history.
    out.clear();
    CHECK(mon.run("tile break op ST halt", &out) == 0, "tile break: %s", out.c_str());
    CHECK(tg.resume() == 0 && wait_halt(tg, t) && tg.stop(&si) == 0, "the tile stop halts the hart");
    CHECK(contains(si.note, "tile stop at pidx") && contains(si.note, "ST ch0") && contains(si.note, "in rt_copy"),
          "classified as a tile stop with the op and its push line: \"%s\"", si.note.c_str());
    const uint32_t dbs = tg.csr(TSR_CSR_DBG_STATUS);
    CHECK((dbs & TSR_DBG_STATUS_STOP_ACTIVE) && (TSR_GET(dbs, TSR_DBG_STATUS_CAUSE) & TSR_DBG_CAUSE_BP), "DBG_STATUS 0x%08x: stop, cause BP", dbs);
    out.clear();
    CHECK(mon.run("hist 2", &out) == 0 && contains(out, "LD ch0") && contains(out, "ST ch0") && contains(out, "in rt_copy"),
          "hist: LD and ST pushed in rt_copy:\n%s", out.c_str());
    const std::vector<hist_entry> hs = hist_last(t, 2);
    CHECK(hs.size() == 2 && hist_status(t, hs[0]) == "done" && hist_status(t, hs[1]) == "queued", "the LD is done, the ST held: %s / %s",
          hs.size() == 2 ? hist_status(t, hs[0]).c_str() : "?", hs.size() == 2 ? hist_status(t, hs[1]).c_str() : "?");
    CHECK(hs.size() == 2 && contains(img.symbolize(hs[1].pc), "rt_copy"), "push PC in rt_copy: %s", hs.size() == 2 ? img.symbolize(hs[1].pc).c_str() : "");
    CHECK(hist_export(hs, scratch + "/hist_push.bin", &why), "hist export: %s", why.c_str());
    out.clear();
    CHECK(mon.run("units", &out) == 0 && contains(out, "DMA") && contains(out, "held by"), "units (QSTATUS):\n%s", out.c_str());
    const uint32_t done0 = tg.csr(TSR_CSR_Q_DONE(TSR_UNIT_DMA));
    out.clear();
    CHECK(mon.run("tile step", &out) == 0 && tg.csr(TSR_CSR_Q_DONE(TSR_UNIT_DMA)) == done0 + 1, "QSTEP ran the ST: %s", out.c_str());
    CHECK(mon.run("tile delete 0", &out) == 0 && mon.run("tile cont", &out) == 0, "tile delete / cont");
    CHECK(tg.resume() == 0 && contains(wait_answers(mon, t), "answered: OK"), "rt_copy answers OK");
    CHECK(tsr_transport_read(t, 0, (uint64_t) dst * 64, back.data(), 128) == 0 && back == data, "the copy happened");

    // 10. An RV watchpoint (DB §3.7) on tsr_rt_launch (written by every launch).
    CHECK(tg.interrupt() == 0 && wait_halt(tg, t) && tg.stop(&si) == 0, "halt");
    CHECK(tg.insert_bp(2, a_launch, 4) == 0, "Z2 on tsr_rt_launch");
    CHECK(mon.run("launch rt_nop", &out) == 0, "launch rt_nop");
    CHECK(tg.resume() == 0 && wait_halt(tg, t) && tg.stop(&si) == 0 && si.reason == "watch" && si.watch_addr >= a_launch &&
              si.watch_addr < a_launch + 4,
          "watchpoint: %s at 0x%08x", si.reason.c_str(), si.watch_addr);
    CHECK(tg.remove_bp(2, a_launch, 4) == 0, "z2");
    CHECK(tg.resume() == 0 && contains(wait_answers(mon, t), "answered: OK"), "rt_nop answers OK");

    // 11. The trace writer (DB §8.3) and its timeline (DB §8.6, §8.7).
    out.clear();
    CHECK(mon.run("trw start 1", &out) == 0, "trw start: %s", out.c_str());
    tsr_perf_summary p0, p1;
    tsr_transport_perf_summary(t, &p0);
    CHECK(mon.run("launch rt_copy 0x100000 0x100100 2 0", &out) == 0 && contains(wait_answers(mon, t), "answered: OK"), "a traced rt_copy");
    tsr_transport_perf_summary(t, &p1);
    const std::string js = scratch + "/trw.json", rawp = scratch + "/trw.bin";
    out.clear();
    CHECK(mon.run("trw dump " + js + " " + rawp, &out) == 0, "trw dump: %s", out.c_str());
    std::ifstream jf(js);
    std::stringstream ss;
    ss << jf.rdbuf();
    const std::string j = ss.str();
    CHECK(contains(j, "\"name\":\"LD\"") && contains(j, "\"name\":\"ST\"") && contains(j, "launch rt_copy seq") && contains(j, "Tesseract0"),
          "the timeline has the LD, the ST and the launch slice");
    std::vector<trw_rec> recs;
    CHECK(trw_load_raw(rawp, &recs, &why), "raw records: %s", why.c_str());
    uint64_t pd[64];
    for (int k = 0; k < 64; ++k) pd[k] = p1.perf[k] - p0.perf[k];
    const trw_check_result cr = trw_check(recs, pd, tsr_transport_config(t)->num_ch, 0);
    for (const std::string & l : cr.lines) printf("  trw check %s\n", l.c_str());
    CHECK(cr.ok, "the DB §8.7 self-checks");
    CHECK(mon.run("trw stop", &out) == 0, "trw stop");

    // 12. Probes and text commands.
    for (const char * c : { "status", "perf", "sreg", "crash", "tile show", "log", "help" }) {
        out.clear();
        CHECK(mon.run(c, &out) == 0 && !out.empty(), "monitor %s", c);
        if (std::string(c) == "status") CHECK(contains(out, "verdict (DB §7.2)"), "status verdict:\n%s", out.c_str());
        if (std::string(c) == "crash") CHECK(contains(out, "holds no crash"), "crash: %s", out.c_str());
    }
    out.clear();
    CHECK(mon.run("dump " + scratch + "/dump", &out) == 0, "dump: %s", out.c_str());
    std::ifstream sj(scratch + "/dump/state.json");
    CHECK(sj.good(), "dump/state.json");

    // 13. Detach (DB §4.2): the hart runs on, the device answers.
    CHECK(tg.interrupt() == 0 && wait_halt(tg, t) && tg.stop(&si) == 0, "halt before detach");
    CHECK(tg.detach(false) == 0 && !tg.attached(), "detach");
    CHECK(!(tg.csr(TSR_CSR_DBG_CFG) & (TSR_DBG_CFG_DMI_EN | TSR_DBG_CFG_RV_HALT_EN | TSR_DBG_CFG_STOP_HALT_RV)),
          "detach restores DBG_CFG (0x%08x)", tg.csr(TSR_CSR_DBG_CFG));
    tsr_launch l;
    memset(&l, 0, sizeof(l));
    l.type = TSR_DESC_TYPE_PING;
    l.arg[0] = 0xDB6A11u;
    tsr_record rec;
    CHECK(tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec) == 0 && rec.w[2] == 0xDB6A11u, "PING after the session");

    // 14. TCM writes while RV_RUN = 0 go through EX's LOAD_* (DB §3.6). The session's PERF summary
    // (DB §8.6, TSR_STATS) survives the SOFT_RESET that zeroes the counters (EX §8.5).
    tsr_perf_summary s0, s1;
    tsr_transport_perf_summary(t, &s0);
    CHECK(tsr_transport_soft_reset(t) == 0 && !(tg.csr(TSR_CSR_CTRL) & TSR_CTRL_RV_RUN), "SOFT_RESET: RV_RUN = 0");
    tsr_transport_perf_summary(t, &s1);
    CHECK(tg.csr(TSR_CSR_PERF_LO(TSR_PERF_RETIRED)) == 0 && s0.perf[TSR_PERF_RETIRED] > 0 &&
              s1.perf[TSR_PERF_RETIRED] == s0.perf[TSR_PERF_RETIRED] && s1.perf[TSR_PERF_CYCLES] == s0.perf[TSR_PERF_CYCLES],
          "PERF folded across SOFT_RESET: RETIRED %" PRIu64 " -> %" PRIu64, s0.perf[TSR_PERF_RETIRED], s1.perf[TSR_PERF_RETIRED]);
    char sb[4096];
    CHECK(tsr_perf_summary_str(&s1, sb, sizeof(sb)) > 0 && contains(sb, "DBG_STOP_CYCLES") && contains(sb, "ch0 "), "the DB §8.6 text");
    tsr_transport_dmi_enable(t, 1);
    tg.dm().activate();
    CHECK(tg.write_mem(a_dbg + 40, pat, 6) == 0 && tg.read_mem(a_dbg + 40, m, 6) == 0 && !memcmp(m, pat, 6), "DTCM write through LOAD_*");
    uint8_t c4[4] = { 0x13, 0, 0, 0 };   // nop's low byte into ITCM
    CHECK(tg.write_mem(0x40, c4, 1) == 0 && tg.read_mem(0x40, m, 1) == 0 && m[0] == 0x13, "ITCM byte write through LOAD_*");
    tsr_transport_dmi_enable(t, 0);

    tsr_transport_close(t);
    printf("%s: %d checks passed, %d failed\n", g_fail ? "FAIL" : "PASS", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

int embed_host(const std::string & elf, uint32_t n) {
    tsr_transport * t = tsr_transport_open("sim");
    if (!t) return 1;
    tsr_image * im = nullptr;
    if (tsr_image_open(elf.c_str(), &im) != 0) return 1;
    int rc = tsr_transport_boot(t, im);
    if (rc) {
        printf("boot failed: %s\n", tsr_transport_strerror(rc));
        return 1;
    }
    const int k = tsr_image_kernel(im, "rt_nop");
    int bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        tsr_launch l;
        memset(&l, 0, sizeof(l));
        l.type = TSR_DESC_TYPE_LAUNCH;
        l.kernel = (uint16_t) k;
        tsr_record rec;
        rc = tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec);
        tsr_result r;
        tsr_record_decode(&rec, &r);
        printf("embed-host: launch %u: rc %d status %s\n", i, rc, tsr_status_name(r.status));
        fflush(stdout);
        if (rc || r.status) bad++;
    }
    tsr_transport_close(t);
    tsr_image_close(im);
    printf("embed-host: %u launches, %d failed\n", n, bad);
    return bad ? 1 : 0;
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc >= 4 && !strcmp(argv[1], "lib")) {
        mkdir(argv[3], 0777);
        return test_lib(argv[2], argv[3]);
    }
    if (argc >= 4 && !strcmp(argv[1], "embed-host")) return embed_host(argv[2], (uint32_t) atoi(argv[3]));
    fprintf(stderr, "usage: test-tsr-dbg lib <fw_rt.elf> <scratch> | embed-host <fw_rt.elf> <launches>\n");
    return 2;
}

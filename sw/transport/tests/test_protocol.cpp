// test_protocol.cpp — ctest transport_protocol (docs/design/06-software-verification.md "SW" §10.2;
// docs/design/01-execution-model.md "EX" §15 item 2): the protocol layer, one binary for every kind it
// can run. sim and mock always; vkernel when TSR_VKERNEL_SERVER / TSR_VKERNEL_SOCKET name a two-clock
// server (P5.3; else reported as not run). Firmware: the fw_rt test image (sw/ctrl/tests/fw/
// rt_kernels.cpp: rt_args0-7, rt_nop, rt_ret, rt_copy, rt_tile_err, rt_stop, rt_cancel, rt_log,
// rt_trap) and the production image tsr-fw.elf.
//
// Per kind:
//   boot      cold, then warm (no ITCM write), a different image (cold), back (cold); a corrupted
//             ITCM word found by VERIFY on a warm restart (that segment reloaded); image-id,
//             kernel-count, Tesseract Basis version and kind table signature mismatches at HELLO;
//             .note.tsr requirements the device lacks (CAPS, Tesseract Core capability 1.1 and 2.0);
//             the mock's TCM write fault (the receive CRC matches, VERIFY does not: the boot fails);
//   versions  tsr_image_open on mutated copies of tsr-fw.elf (Basis ABI 2, no Basis version, a newer
//             Basis version, a kind declared twice); tsr_transport_versions / tsr_versions_str after
//             booting tsr-fw.elf (SW §6.9);
//   launches  PING (ARG0 echo), rt_args0-7 with seven arguments (the kernel sees nargs of them, its
//             event carries them, RET), KERNEL_ERR, TILE_ERR (first-error ticket / info / w0 / pidx),
//             rt_copy (DRAM out = in), BAD_DESC (unknown TYPE, a reserved w0 bit, KERNEL beyond the
//             table), 8 outstanding and 24 pipelined (the credit rule: no LQ_OVERFLOW, the LQ_W shadow:
//             an identical descriptor writes only LQ_PUSH on the mock; an LQ_W word written behind
//             the transport is rewritten by the next launch);
//   rings     forced LQ_OVERFLOW (pushes behind the transport while the RISC-V is in reset) and
//             RING_RD_BAD (LCQ_RD / LOG_RD beyond the write pointer), both refused and recovered;
//   events    kernel events while the host waits, in order, before the answer;
//   log       FMT / TEXT / EVENT / BLOB records decoded from the image (.tsr_logfmt, %s from
//             .rodata); a flood without BLOCK drops records ("dropped n"), with LOG_CFG.BLOCK none;
//   control   soft stop (STOPPED, RET), hard cancel of a held running launch and the queued ones
//             behind it (ABORTED), tsr_transport_soft_reset mid-launch (TSR_TE_LOST, then a warm boot);
//   debug     the debug-aware timeout: a hart halted through the DM model (DMI from the event
//             callback) does not time out while another thread waits 300 ms of host time before it
//             resumes the hart; DBG_CFG.SERIAL turns the timeout off, also when an earlier session
//             left it set (seen at open);
//   recovery  the ladder: a held launch recovered at level 2 (LQ_CANCEL, TSR_TE_TIMEOUT with the
//             ABORTED answer, a hang dump in TSR_DUMP_DIR); a RISC-V that ignores the cancel recovered
//             at level 3 (SOFT_RESET + warm restart, launches behind it TSR_TE_LOST);
//   TRAP      illegal instruction, panic, load fault, ebreak, a bad event type: the TRAP record
//             (mcause, mepc), the decoded report (mepc -> rt_trap and its line, equal to addr2line's),
//             the warm restart, the launches behind it lost; a trap between launches (the idle hart
//             halted, a PING queued, resumed at an address without code: TRAP r1 = the PING's seq,
//             but no answer: the PING is lost); TSR_GDB_WAIT keeps the device TRAPPED;
//   records   EX §4.8's minimal reads (r0, r2, r7 of an OK DONE / PONG) equal the full reads.
// Across kinds: the record stream (r0-r6 of every answer and event), the DRAM outputs and the log
// words of one fixed scenario are identical on sim (minimal and full reads) and mock.
// On sim (fixed schedule): the EX §4.11 numbers: cold and warm boot, empty-launch round trip,
// pipelined launch rate, CSR accesses per launch (printed).
//
//   test-tsr-protocol [--addr2line PATH] [--scratch DIR]
// Exit 0 / 1; 77 (skipped) when the firmware images are missing.

#include <atomic>

#include "tsr/opl.hpp"   // kind_rev, kinds_sig (SW §6.9)
#include "tsr/version.h"   // TSR_BASIS_VERSION, TSR_SDK_ABI
#include "tsr_dm.h"
#include "tsr_elf.h"
#include "tsr_fw_paths.h"
#include "tsr_image_int.h"
#include "tsr_kdev.h"
#include "tsr_kdev_mock.h"
#include "tsr_kdev_sim.h"
#include "tsr_logdec.h"
#include "tsr_sim_int.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace tsr;

static std::atomic<int> g_fail{ 0 }, g_checks{ 0 };
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

namespace {

std::string g_addr2line, g_scratch = ".";

// The numbers the build defines (tsr/version.h, tsr_tables.h VERSION, tsr/opl.hpp kind_rev): the version tests
// derive what they expect from them, so a release or a kind revision bump does not edit this file.
uint32_t core_cap() { return TSR_VERSION_VALUE & 0xFFFFu; }
std::string cap_str(uint32_t cap) { return std::to_string(cap >> 8) + "." + std::to_string(cap & 0xFFu); }
std::string basis_str(uint32_t v) {
    return std::to_string(TSR_BASIS_MAJOR_OF(v)) + "." + std::to_string(TSR_BASIS_MINOR_OF(v)) + "." + std::to_string(TSR_BASIS_PATCH_OF(v));
}
uint32_t n_defined_kinds() {
    uint32_t n = 0;
    for (uint32_t k = 0; k < opl::N_KINDS; ++k) n += opl::kind_rev(k) != 0u;
    return n;
}
// "NAME rN" of a kind, as tsr_versions_str lists it.
std::string kind_rev_str(uint32_t k) { return std::string(opl::kind_name(k)) + " r" + std::to_string(opl::kind_rev(k)); }

// What f writes to stderr (the transport's messages), also echoed there.
std::string stderr_of(const std::function<void()> & f) {
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
const char * const RT_IMAGE = TSR_FW_TEST_IMAGE_fw_rt;
const char * const FW_IMAGE = TSR_FW_DEFAULT_IMAGE;

constexpr uint64_t MS = 1000000ull;
uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | (uint32_t) (beat & 0x0FFFFFFFu); }
constexpr uint64_t IO_BEAT = TSR_REGION_IO_OFF / 64u;

uint64_t wall_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The environment the transport reads, cleared so that the caller's shell cannot change the test.
void clear_env() {
    for (const char * v : { "TSR_DEVICE", "TSR_IMAGE", "TSR_LAUNCH_TIMEOUT_MS", "TSR_ABORT_MS", "TSR_BOOT_MS", "TSR_RESET_MS", "TSR_LOG",
                            "TSR_LOG_LEVEL", "TSR_ATTACH", "TSR_STATS", "TSR_DUMP_DIR", "TSR_GDB_WAIT", "TSR_TILE_SERIAL",
                            "TSR_TILE_HIST", "TSR_GDB", "TSR_DEBUGGER", "TSR_TRACE", "TSR_SIM_SCHED", "TSR_SIM_RACE", "TSR_SIM_LOG",
                            "TSR_MOCK_INJECT", "TSR_MOCK_TCM_FAULT", "TSR_MOCK_HANG", "TSR_SIM_TMAX", "TSR_MOCK_TMAX", "TSR_SIM_VU" })
        unsetenv(v);
    setenv("TSR_SIM_THREADS", "2", 1);
}

std::string type_name(uint32_t r0) {
    char b[16];
    snprintf(b, sizeof(b), "0x%02x", r0 & 0xFFu);
    return b;
}

// One transport with the images and an event log.
struct session {
    std::string kind;
    tsr_transport * t = nullptr;
    tsr_image * rt = nullptr;   // fw_rt
    std::vector<tsr_record> events;
    std::function<void(const tsr_record &)> on_event;
    bool ok = false;

    static void cb(void * u, const tsr_record * r) {
        session * s = (session *) u;
        s->events.push_back(*r);
        if (s->on_event) s->on_event(*r);
    }

    explicit session(const std::string & k, std::unique_ptr<kdev> dev = nullptr) : kind(k) {
        t = dev ? transport_open_kdev(std::move(dev)) : tsr_transport_open(k.c_str());
        CHECK(t != nullptr, "%s: open", k.c_str());
        if (!t) return;
        CHECK(tsr_image_open(RT_IMAGE, &rt) == 0, "%s: open the fw_rt image", k.c_str());
        if (!rt) return;
        tsr_transport_set_event_cb(t, cb, this);
        ok = true;
    }
    ~session() {
        if (t) tsr_transport_close(t);
        tsr_image_close(rt);
    }
    uint32_t k(const char * name) const {
        const int i = tsr_image_kernel(rt, name);
        CHECK(i >= 0, "kernel %s in fw_rt", name);
        return (uint32_t) (i < 0 ? 0 : i);
    }
    static tsr_launch L(uint8_t type, uint16_t kernel, std::vector<uint32_t> a = {}, uint8_t flags = 0) {
        tsr_launch l{};
        l.type = type;
        l.flags = flags;
        l.kernel = kernel;
        for (size_t i = 0; i < a.size() && i < 7; ++i) l.arg[i] = a[i];
        return l;
    }
    int call(const tsr_launch & l, tsr_record * r, uint64_t tmo = TSR_TIMEOUT_DEFAULT) { return tsr_transport_call(t, &l, tmo, r); }
    int run(const char * kname, std::vector<uint32_t> a, tsr_record * r, uint64_t tmo = TSR_TIMEOUT_DEFAULT) {
        return call(L(TSR_DESC_TYPE_LAUNCH, (uint16_t) k(kname), a), r, tmo);
    }
    tsr_sim * sim() const { return transport_sim(t); }
};

uint32_t st_of(const tsr_record & r) { return TSR_GET(r.w[0], TSR_LCQ_R0_STATUS); }
uint32_t ty_of(const tsr_record & r) { return TSR_GET(r.w[0], TSR_LCQ_R0_TYPE); }

// ---------------------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------------------
// The first boot of a session: cold (TCMs zero after create).
void test_first_boot(session & s) {
    const char * K = s.kind.c_str();
    CHECK(tsr_transport_image(s.t) == nullptr, "%s: no image before boot", K);
    tsr_record r;
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0), &r) == TSR_TE_STATE, "%s: launch before boot refused", K);
    CHECK(tsr_transport_boot(s.t, s.rt) == 0, "%s: cold boot of fw_rt", K);
    const boot_info b = transport_boot_info(s.t);
    CHECK(!b.warm && b.itcm_loaded >= 1 && b.dtcm_loaded >= 1 && b.words_written > 1000, "%s: cold: %u ITCM + %u DTCM segments, %u words",
          K, b.itcm_loaded, b.dtcm_loaded, b.words_written);
    CHECK(tsr_transport_image(s.t) == s.rt, "%s: booted image", K);
    CHECK(TSR_GET(tsr_transport_csr_read(s.t, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_IDLE, "%s: RT_STATE IDLE", K);
    printf("%s: cold boot of fw_rt: %.3f ms %s time, %.3f ms host, %u words, HELLO after %u core cycles\n", K, (double) b.time_ns / 1e6,
           transport_kdev(s.t)->simulated() ? "device" : "host", (double) b.wall_ns / 1e6, b.words_written, b.hello_cycles);
}

void test_boot(session & s) {
    const char * K = s.kind.c_str();
    tsr_image * fw = nullptr;
    CHECK(tsr_image_open(FW_IMAGE, &fw) == 0, "%s: open tsr-fw.elf", K);
    tsr_record r;
    boot_info b;

    CHECK(tsr_transport_boot(s.t, s.rt) == 0, "%s: warm boot", K);
    b = transport_boot_info(s.t);
    CHECK(b.warm && b.itcm_loaded == 0 && b.dtcm_loaded >= 1, "%s: warm: %u ITCM written", K, b.itcm_loaded);
    printf("%s: warm boot of fw_rt: %.3f ms %s time, %u words\n", K, (double) b.time_ns / 1e6,
           transport_kdev(s.t)->simulated() ? "device" : "host", b.words_written);
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 0xABCD0001u }), &r) == 0 && ty_of(r) == TSR_LCQ_R0_TYPE_PONG &&
          r.w[2] == 0xABCD0001u, "%s: PING after the warm boot", K);

    // Another image (cold), back to fw_rt (cold: its ITCM differs).
    if (fw) {
        CHECK(tsr_transport_boot(s.t, fw) == 0 && !transport_boot_info(s.t).warm, "%s: tsr-fw.elf boots cold", K);
        CHECK(tsr_image_kernel(fw, "graph") == 0 && tsr_image_kernel(fw, "selftest") == 1 && tsr_image_kernel(fw, "replay") == 2,
              "%s: base kernels at 0 / 1 / 2 (EX §4.6)", K);
        CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 7 }), &r) == 0 && r.w[2] == 7, "%s: PING on tsr-fw.elf", K);
        CHECK(tsr_transport_boot(s.t, s.rt) == 0 && !transport_boot_info(s.t).warm, "%s: back to fw_rt: cold", K);
    }

    // A corrupted ITCM word: the warm restart's VERIFY finds it and reloads that segment (EX §3.5).
    if (tsr_sim * sim = s.sim()) {
        uint32_t ib = 0;
        uint8_t * itcm = tsr_sim_itcm(sim, &ib);
        itcm[0x200] ^= 0x10;
        CHECK(tsr_transport_boot(s.t, s.rt) == 0, "%s: boot after an ITCM word flip", K);
        b = transport_boot_info(s.t);
        CHECK(!b.warm && b.itcm_loaded == 1, "%s: VERIFY mismatch reloads exactly one ITCM segment (%u)", K, b.itcm_loaded);
        CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 9 }), &r) == 0 && r.w[2] == 9, "%s: PING after the reload", K);
    }

    // Image-id mismatch at HELLO (EX §3.4): the device runs another image than the host expects.
    const uint64_t id = tsr_image_id(s.rt);
    image_test_set_id(s.rt, id ^ 0x100);
    CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "%s: HELLO with another image id refused", K);
    image_test_set_id(s.rt, id);
    // A kernel count HELLO does not report (EX §3.4): the host's kernel table is not the device's.
    const uint32_t nk = image_n_kernels(s.rt);
    image_test_set_n_kernels(s.rt, nk + 1);
    CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "%s: HELLO with another kernel count refused", K);
    image_test_set_n_kernels(s.rt, nk);
    // .note.tsr requirements (EX §3.1, SW §6.9): a CAPS bit the device lacks; a Tesseract Core
    // capability the device does not have (a higher minor, another major).
    const image_note n0 = image_get_note(s.rt);
    image_note n = n0;
    n.caps_req = 0x00010000u;
    image_test_set_note(s.rt, n);
    CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "%s: CAPS requirement refused before loading", K);
    n = n0;
    const uint32_t minor_up = core_cap() + 1u, major_up = (core_cap() & 0xFF00u) + 0x0100u;
    n.cap_min = uint16_t(minor_up);
    image_test_set_note(s.rt, n);
    std::string err = stderr_of([&] { CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "%s: a higher capability minor refused", K); });
    CHECK(err.find(" Tesseract Core capability " + cap_str(minor_up) + " (device " + cap_str(core_cap()) + ");") != std::string::npos,
          "%s: capability message: %s", K, err.c_str());
    n.cap_min = uint16_t(major_up);
    image_test_set_note(s.rt, n);
    err = stderr_of([&] { CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "%s: another capability major refused", K); });
    CHECK(err.find(" Tesseract Core capability " + cap_str(major_up) + " (device " + cap_str(core_cap()) + ");") != std::string::npos,
          "%s: capability message: %s", K, err.c_str());
    // HELLO r6 / r4[31:16] (SW §6.9): the device reports another Basis version or kind table than the
    // host read from the image.
    n = n0;
    const uint32_t basis_up = TSR_BASIS_VERSION + 0x100u;   // the next minor
    n.basis_version = basis_up;
    image_test_set_note(s.rt, n);
    err = stderr_of([&] { CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "%s: HELLO with another Basis version refused", K); });
    CHECK(err.find(" Tesseract Basis " + basis_str(TSR_BASIS_VERSION) + ", expected " + basis_str(basis_up) + ";") != std::string::npos,
          "%s: Basis message: %s", K, err.c_str());
    image_test_set_note(s.rt, n0);
    uint8_t rev[64] = {};
    rev[0x10] = 2;                          // the host believes in MUL_MAT r2; fw_rt has no kinds
    image_test_set_kinds(s.rt, rev);
    err = stderr_of([&] { CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "%s: HELLO with another kind table refused", K); });
    CHECK(err.find(" kind table signature 0xb51b, expected 0x") != std::string::npos, "%s: kind table message: %s", K, err.c_str());
    std::memset(rev, 0, sizeof(rev));
    image_test_set_kinds(s.rt, rev);
    CHECK(tsr_transport_boot(s.t, s.rt) == 0, "%s: boot again with the true id, note and kind table", K);
    tsr_image_close(fw);
}

// ---------------------------------------------------------------------------------------------
// Launches, records, the credit rule
// ---------------------------------------------------------------------------------------------
// The fixed scenario of the cross-kind comparison: every answer and event, in order.
struct observed {
    std::vector<tsr_record> recs;     // answers and events in consumption order
    std::vector<uint8_t> dram;        // rt_copy's output
    std::vector<uint32_t> log;        // raw log words
};

void scenario(session & s, observed * ob) {
    const char * K = s.kind.c_str();
    tsr_record r;
    s.events.clear();
    std::vector<tsr_record> answers;
    auto keep = [&](int rc, const char * what) {
        CHECK(rc == 0, "%s: %s: rc %d", K, what, rc);
        for (const auto & e : s.events) ob->recs.push_back(e);
        s.events.clear();
        ob->recs.push_back(r);
    };
    keep(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 0x1234u }), &r), "PING");
    CHECK(ty_of(r) == TSR_LCQ_R0_TYPE_PONG && st_of(r) == 0 && r.w[2] == 0x1234u, "%s: PONG echoes ARG0", K);
    for (uint32_t k = 0; k <= 7; ++k) {
        std::vector<uint32_t> a(7);
        for (uint32_t i = 0; i < 7; ++i) a[i] = 0x1000u * (k + 1) + i + 1;
        char kn[16];
        snprintf(kn, sizeof(kn), "rt_args%u", k);
        const size_t e0 = ob->recs.size();
        int rc = s.run(kn, a, &r);
        CHECK(s.events.size() == 1, "%s: rt_args%u: one event before the answer (%zu)", K, k, s.events.size());
        if (s.events.size() == 1) {
            const tsr_record & e = s.events[0];
            const uint32_t info = (e.w[0] >> 8) & 0xFFFFu;
            CHECK(ty_of(e) == 0xC1u && (info & 0xFFu) == k && (info & 0x1000u) && e.w[1] == r.w[1], "%s: rt_args%u event r0 0x%08x", K, k,
                  e.w[0]);
            for (uint32_t i = 0; i < 6; ++i)
                CHECK(e.w[2 + i] == (i < k ? a[i] : 0u), "%s: rt_args%u: the kernel sees arg %u = 0x%x", K, k, i, e.w[2 + i]);
        }
        keep(rc, kn);
        (void) e0;
        CHECK(ty_of(r) == TSR_LCQ_R0_TYPE_DONE && st_of(r) == 0 && (int32_t) r.w[2] == (int32_t) ((k == 7 ? (a[6] & 0xFFFFu) : 0u) | k << 16),
              "%s: rt_args%u RET 0x%08x", K, k, r.w[2]);
    }
    keep(s.run("rt_ret", { (uint32_t) -5 }, &r), "rt_ret -5");
    CHECK(st_of(r) == TSR_LCQ_R0_STATUS_KERNEL_ERR && (int32_t) r.w[2] == -5 && r.w[3] == TSR_TICKET_NONE, "%s: KERNEL_ERR RET -5", K);
    keep(s.run("rt_ret", { 3 }, &r), "rt_ret 3");
    CHECK(st_of(r) == 0 && r.w[2] == 3, "%s: OK RET 3", K);
    keep(s.run("rt_tile_err", { H(0, 0x0FFFFFF0u), H(1, IO_BEAT + 4096), 32, 0 }, &r), "rt_tile_err");   // beyond 2^28 beats
    CHECK(st_of(r) == TSR_LCQ_R0_STATUS_TILE_ERR && TSR_GET(r.w[0], TSR_LCQ_R0_ERR_CODE) == TSR_ERR_DRAM_RANGE &&
          TSR_GET(r.w[3], TSR_TICKET_UNIT) == TSR_UNIT_DMA && r.w[4] == 0x0FFFFFF0u && (r.w[5] & 0xFFu) == TSR_OP_LD,
          "%s: TILE_ERR DRAM_RANGE: r0 0x%08x ticket 0x%08x info 0x%08x w0 0x%08x pidx %u", K, r.w[0], r.w[3], r.w[4], r.w[5], r.w[6]);
    // rt_copy: 8 lines ch0 -> ch1.
    std::vector<uint8_t> in(8 * 64), out(8 * 64);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (uint8_t) (i * 37 + 11);
    CHECK(tsr_transport_write(s.t, 0, (IO_BEAT + 100) * 64, in.data(), in.size()) == 0, "%s: write input", K);
    keep(s.run("rt_copy", { H(0, IO_BEAT + 100), H(1, IO_BEAT + 200), 8, 16 }, &r), "rt_copy");
    CHECK(st_of(r) == 0 && tsr_transport_read(s.t, 1, (IO_BEAT + 200) * 64, out.data(), out.size()) == 0 && out == in,
          "%s: rt_copy: DRAM output = input", K);
    ob->dram = out;
    // BAD_DESC (EX §4.3): unknown TYPE, a reserved w0 bit (flags bit 1 = w0[9]), KERNEL beyond the table.
    keep(s.call(session::L(7, 0), &r), "unknown TYPE");
    CHECK(st_of(r) == TSR_LCQ_R0_STATUS_BAD_DESC && r.w[3] == TSR_TICKET_NONE, "%s: BAD_DESC unknown TYPE", K);
    keep(s.call(session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) s.k("rt_nop"), {}, 0x02), &r), "reserved bit");
    CHECK(st_of(r) == TSR_LCQ_R0_STATUS_BAD_DESC, "%s: BAD_DESC reserved w0 bit", K);
    keep(s.call(session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) image_n_kernels(s.rt)), &r), "kernel beyond");
    CHECK(st_of(r) == TSR_LCQ_R0_STATUS_BAD_DESC, "%s: BAD_DESC KERNEL beyond the table", K);
    keep(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 0x5678u }), &r), "last PING");
    // The log words of the scenario (rt_args posts none; the runtime logs nothing at INFO on success).
    std::vector<uint32_t> w(4096);
    size_t n = 0;
    CHECK(tsr_transport_log_drain(s.t, w.data(), w.size(), &n) == 0, "%s: log drain", K);
    ob->log.assign(w.begin(), w.begin() + (long) n);
}

void test_pipeline(session & s) {
    const char * K = s.kind.c_str();
    // 8 outstanding, then 24 more pipelined: the credit rule (blocks only at LQ_DEPTH), no overflow.
    const uint32_t knop = s.k("rt_nop");
    mock_kdev * mk = dynamic_cast<mock_kdev *>(transport_kdev(s.t));
    std::vector<uint32_t> seqs;
    size_t log0 = mk ? mk->window_write_log.size() : 0;
    const uint64_t reads0 = mk ? mk->window_reads : 0;
    for (int i = 0; i < 32; ++i) {
        if (mk && i == (int) TSR_LQ_DEPTH)
            CHECK(mk->window_reads == reads0, "%s: %u launches without a device read (host-side credit, EX §4.8)", K, TSR_LQ_DEPTH);
        uint32_t q = 0;
        const tsr_launch l = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) knop);
        CHECK(tsr_transport_launch(s.t, &l, &q) == 0, "%s: launch %d", K, i);
        if (i > 0) CHECK(q == seqs.back() + 1, "%s: sequence numbers count up", K);
        seqs.push_back(q);
        if (mk && i == 1) {
            // The shadow (EX §4.8): an identical descriptor writes LQ_PUSH only.
            bool only_push = true;
            for (size_t j = log0; j < mk->window_write_log.size(); ++j)
                if (mk->window_write_log[j] != TSR_CSR_LQ_PUSH && mk->window_write_log[j] != TSR_CSR_LCQ_RD) only_push = false;
            CHECK(only_push, "%s: the second identical descriptor writes no LQ_W word", K);
        }
        if (mk && i == 0) log0 = mk->window_write_log.size();
    }
    for (uint32_t q : seqs) {
        tsr_record r;
        CHECK(tsr_transport_wait(s.t, q, TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == 0 && r.w[1] == q, "%s: answer of seq %u", K, q);
    }
    tsr_record r;
    CHECK(tsr_transport_wait(s.t, seqs[0], TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_INVAL, "%s: a second wait for the same seq refused", K);
    CHECK(tsr_transport_wait(s.t, seqs.back() + 100, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_INVAL, "%s: wait for a seq never pushed", K);
    const uint32_t stt = tsr_transport_csr_read(s.t, TSR_CSR_STATUS);
    CHECK(!(stt & (TSR_STATUS_LQ_OVERFLOW | TSR_STATUS_RING_RD_BAD)), "%s: no LQ_OVERFLOW / RING_RD_BAD (STATUS 0x%08x)", K, stt);
}

// Forced LQ_OVERFLOW and RING_RD_BAD (EX §4.2), behind the transport's back.
void test_rings(session & s) {
    const char * K = s.kind.c_str();
    tsr_record r;
    // RING_RD_BAD: LCQ_RD / LOG_RD beyond the write pointers are refused.
    const uint32_t lwr = tsr_transport_csr_read(s.t, TSR_CSR_LCQ_WR), lrd = tsr_transport_csr_read(s.t, TSR_CSR_LCQ_RD);
    CHECK(lwr == lrd, "%s: LCQ drained", K);
    CHECK(tsr_transport_csr_write(s.t, TSR_CSR_LCQ_RD, lwr + 1) == 0, "%s: write a bad LCQ_RD", K);
    uint32_t stt = tsr_transport_csr_read(s.t, TSR_CSR_STATUS);
    CHECK((stt & TSR_STATUS_RING_RD_BAD) && tsr_transport_csr_read(s.t, TSR_CSR_LCQ_RD) == lrd, "%s: RING_RD_BAD, LCQ_RD unchanged", K);
    const uint32_t gwr = tsr_transport_csr_read(s.t, TSR_CSR_LOG_WR);
    tsr_transport_csr_write(s.t, TSR_CSR_STATUS, TSR_STATUS_RING_RD_BAD);
    CHECK(tsr_transport_csr_write(s.t, TSR_CSR_LOG_RD, gwr + 5) == 0 &&
          (tsr_transport_csr_read(s.t, TSR_CSR_STATUS) & TSR_STATUS_RING_RD_BAD), "%s: RING_RD_BAD for LOG_RD", K);
    tsr_transport_csr_write(s.t, TSR_CSR_STATUS, TSR_STATUS_RING_RD_BAD);
    CHECK(!(tsr_transport_csr_read(s.t, TSR_CSR_STATUS) & TSR_STATUS_RING_RD_BAD), "%s: W1C clears RING_RD_BAD", K);
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 1 }), &r) == 0, "%s: PING after RING_RD_BAD", K);

    // LQ_OVERFLOW: with the RISC-V in reset nothing pops; LQ_DEPTH + 1 pushes, the last one dropped.
    CHECK(tsr_transport_soft_reset(s.t) == 0, "%s: soft reset", K);
    const uint32_t s0 = tsr_transport_csr_read(s.t, TSR_CSR_LQ_SEQ);
    for (uint32_t i = 0; i <= TSR_LQ_DEPTH; ++i) tsr_transport_csr_write(s.t, TSR_CSR_LQ_PUSH, 1);
    stt = tsr_transport_csr_read(s.t, TSR_CSR_STATUS);
    CHECK((stt & TSR_STATUS_LQ_OVERFLOW) && tsr_transport_csr_read(s.t, TSR_CSR_LQ_SEQ) - s0 == TSR_LQ_DEPTH,
          "%s: LQ_OVERFLOW, %u accepted", K, tsr_transport_csr_read(s.t, TSR_CSR_LQ_SEQ) - s0);
    tsr_transport_csr_write(s.t, TSR_CSR_STATUS, TSR_STATUS_LQ_OVERFLOW);
    // The transport's own credit never overflows: flush (SOFT_RESET: TAKEN := PUSHED) and boot.
    CHECK(tsr_transport_soft_reset(s.t) == 0 && tsr_transport_boot(s.t, s.rt) == 0, "%s: flush and warm boot", K);
    CHECK(tsr_transport_csr_read(s.t, TSR_CSR_LQ_TAKEN) == tsr_transport_csr_read(s.t, TSR_CSR_LQ_SEQ), "%s: TAKEN = PUSHED", K);
    uint32_t q = 0;
    const tsr_launch l = session::L(TSR_DESC_TYPE_PING, 0, { 2 });
    CHECK(tsr_transport_launch(s.t, &l, &q) == 0 && q == tsr_transport_csr_read(s.t, TSR_CSR_LQ_SEQ) - 1, "%s: seq follows LQ_SEQ", K);
    CHECK(tsr_transport_wait(s.t, q, TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[2] == 2, "%s: PING after the overflow", K);
}

// ---------------------------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------------------------
void test_log(session & s) {
    const char * K = s.kind.c_str();
    tsr_record r;
    std::vector<uint32_t> w(1 << 16);
    size_t n = 0;
    tsr_transport_log_drain(s.t, w.data(), w.size(), &n);   // empty it
    // Every record kind (rt_log mode 2) and 10 FMT records (mode 1), decoded from the image.
    CHECK(s.run("rt_log", { 3, 10, 0xBEEFu }, &r) == 0 && st_of(r) == 0 && r.w[2] == 10, "%s: rt_log", K);
    CHECK(tsr_transport_log_drain(s.t, w.data(), w.size(), &n) == 0 && n > 0, "%s: log words (%zu)", K, n);
    std::vector<std::string> lines;
    log_decoder d(s.rt);
    d.feed(w.data(), n, [&](const std::string & l) { lines.push_back(l); });
    CHECK(d.bad == 0 && d.pending_words() == 0, "%s: every record decodes (%" PRIu64 " bad)", K, d.bad);
    auto has = [&](const char * sub) {
        for (const auto & l : lines)
            if (l.find(sub) != std::string::npos) return true;
        return false;
    };
    CHECK(has("rt_log 0 of 10: 0xbeef") && has("rt_log 9 of 10: 0xbeef"), "%s: FMT records", K);
    CHECK(has("rt_log all -5 7 beef 00001234 1 t(0,9) 1122334455667788 rt-rodata-string"), "%s: FMT conversions incl. %%s from .rodata", K);
    CHECK(has("rt_log text line"), "%s: TEXT record", K);
    CHECK(has("event user 0x000abcd 0x0000beef 0x00000002"), "%s: EVENT record", K);
    CHECK(has("blob tag 0x0000b10b 0x00000001 0x00000002 0x00000003"), "%s: BLOB record", K);
    CHECK(has("rt_log warn 48879") && !has("rt_log debug"), "%s: WARN kept, DEBUG filtered (LOG_CFG.LEVEL 3)", K);
    if (!lines.empty()) printf("%s: log line: %s\n", K, lines.front().c_str());
    // A flood without BLOCK drops whole records ("dropped n", DB §6.1); with BLOCK none are lost.
    tsr_transport_stats st0, st1;
    tsr_transport_get_stats(s.t, &st0);
    CHECK(s.run("rt_log", { 1, 120, 1 }, &r) == 0, "%s: flood", K);
    CHECK(tsr_transport_log_drain(s.t, w.data(), w.size(), &n) == 0, "%s: drain", K);
    const size_t kept = n;
    CHECK(s.run("rt_log", { 1, 1, 9 }, &r) == 0, "%s: one record after the flood (carries \"dropped n\")", K);
    CHECK(tsr_transport_log_drain(s.t, w.data(), w.size(), &n) == 0, "%s: drain", K);
    tsr_transport_get_stats(s.t, &st1);
    const uint64_t dropped = st1.log_records_dropped - st0.log_records_dropped;
    printf("%s: flood of 120 FMT records without BLOCK: %" PRIu64 " reported dropped, %zu words kept\n", K, dropped, kept);
    CHECK(dropped > 0 && dropped < 120 && kept / 6 + dropped == 120, "%s: records dropped whole without BLOCK (%" PRIu64 " dropped, "
          "%zu words kept)", K, dropped, kept);
    const uint32_t cfg = tsr_transport_csr_read(s.t, TSR_CSR_LOG_CFG);
    tsr_transport_csr_write(s.t, TSR_CSR_LOG_CFG, cfg | TSR_LOG_CFG_BLOCK);
    CHECK(s.run("rt_log", { 1, 120, 2 }, &r) == 0, "%s: flood with BLOCK", K);
    CHECK(tsr_transport_log_drain(s.t, w.data(), w.size(), &n) == 0, "%s: drain", K);
    lines.clear();
    log_decoder d2(s.rt);
    d2.feed(w.data(), n, [&](const std::string & l) { lines.push_back(l); });
    uint32_t fmt = 0;
    for (const auto & l : lines) fmt += l.find("rt_log ") != std::string::npos && l.find(": 0x2") != std::string::npos;
    tsr_transport_get_stats(s.t, &st0);
    CHECK(fmt == 120 && st0.log_records_dropped == st1.log_records_dropped, "%s: BLOCK: all 120 records (%u), none dropped", K, fmt);
    tsr_transport_csr_write(s.t, TSR_CSR_LOG_CFG, cfg);
    // tsr_log_decode on raw words.
    FILE * f = tmpfile();
    CHECK(tsr_log_decode(s.rt, w.data(), n, f) == (long) d2.records, "%s: tsr_log_decode", K);
    if (f) fclose(f);
}

// ---------------------------------------------------------------------------------------------
// Cancel, stop, soft reset
// ---------------------------------------------------------------------------------------------
void test_control(session & s) {
    const char * K = s.kind.c_str();
    tsr_record r;
    // Soft stop (EX §4.9): STOPPED with the kernel's RET.
    uint32_t q = 0;
    s.on_event = [&](const tsr_record & e) {
        if (ty_of(e) == 0xC3u && ((e.w[0] >> 8) & 0xFFFFu) == 1) tsr_transport_stop(s.t, e.w[1]);
    };
    CHECK(s.run("rt_stop", { 0xFFFFFFF0u, 77 }, &r) == 0, "%s: rt_stop", K);
    CHECK(st_of(r) == TSR_LCQ_R0_STATUS_STOPPED && r.w[2] == 77, "%s: STOPPED RET 77 (r0 0x%08x RET %d)", K, r.w[0], (int32_t) r.w[2]);
    // Hard cancel of a running launch whose ops are held, and of the two queued behind it.
    const uint32_t kc = s.k("rt_cancel"), knop = s.k("rt_nop");
    std::vector<uint32_t> seqs(3);
    s.on_event = [&](const tsr_record & e) {
        if (ty_of(e) == 0xC3u && ((e.w[0] >> 8) & 0xFFFFu) == 2) tsr_transport_cancel(s.t, seqs[2]);
    };
    tsr_launch l = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) kc, { H(0, IO_BEAT + 100), H(1, IO_BEAT + 300), 4, 16 });
    CHECK(tsr_transport_launch(s.t, &l, &seqs[0]) == 0, "%s: rt_cancel", K);
    l = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) knop);
    CHECK(tsr_transport_launch(s.t, &l, &seqs[1]) == 0 && tsr_transport_launch(s.t, &l, &seqs[2]) == 0, "%s: 2 queued", K);
    CHECK(tsr_transport_wait(s.t, seqs[0], TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == TSR_LCQ_R0_STATUS_ABORTED &&
          TSR_GET(r.w[0], TSR_LCQ_R0_ERR_CODE) == TSR_ERR_ABORTED, "%s: running launch ABORTED (r0 0x%08x)", K, r.w[0]);
    for (int i = 1; i < 3; ++i)
        CHECK(tsr_transport_wait(s.t, seqs[i], TSR_TIMEOUT_DEFAULT, &r) == 0 && st_of(r) == TSR_LCQ_R0_STATUS_ABORTED &&
              TSR_GET(r.w[0], TSR_LCQ_R0_ERR_CODE) == 0, "%s: queued launch %d ABORTED without running (r0 0x%08x)", K, i, r.w[0]);
    s.on_event = nullptr;
    CHECK(s.run("rt_nop", {}, &r) == 0 && st_of(r) == 0, "%s: a later launch is not covered by the watermark", K);
    // SOFT_RESET mid-launch: the running and the queued launch are lost (EX §8.5).
    s.on_event = [&](const tsr_record & e) {
        if (ty_of(e) == 0xC3u) CHECK(tsr_transport_soft_reset(s.t) == 0, "%s: soft reset from the callback", K);
    };
    l = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) s.k("rt_stop"), { 0xFFFFFFF0u, 1 });
    CHECK(tsr_transport_launch(s.t, &l, &seqs[0]) == 0, "%s: rt_stop", K);
    l = session::L(TSR_DESC_TYPE_PING, 0);
    CHECK(tsr_transport_launch(s.t, &l, &seqs[1]) == 0, "%s: PING behind it", K);
    CHECK(tsr_transport_wait(s.t, seqs[0], TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_LOST, "%s: running launch lost", K);
    CHECK(tsr_transport_wait(s.t, seqs[1], TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_LOST, "%s: queued launch lost", K);
    s.on_event = nullptr;
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0), &r) == TSR_TE_STATE, "%s: not booted after the soft reset", K);
    CHECK(tsr_transport_boot(s.t, s.rt) == 0, "%s: boot", K);
    // A RISC-V reset mid-launch (EX §8.5: RV_RUN 1 -> 0 aborts, ANSWERED := TAKEN, the queued
    // descriptors stay), then RV_RUN 1 behind the transport: the runtime boots from the intact TCMs,
    // its HELLO tells the transport the running launch is lost, the queued PING is served.
    s.on_event = [&](const tsr_record & e) {
        if (ty_of(e) != 0xC3u) return;
        const uint32_t ctrl = tsr_transport_csr_read(s.t, TSR_CSR_CTRL);
        tsr_transport_csr_write(s.t, TSR_CSR_CTRL, ctrl & ~TSR_CTRL_RV_RUN);
        CHECK(tsr_transport_csr_read(s.t, TSR_CSR_LCQ_ANSWERED) == tsr_transport_csr_read(s.t, TSR_CSR_LQ_TAKEN) &&
              !(tsr_transport_csr_read(s.t, TSR_CSR_STATUS) & TSR_STATUS_ACTIVE), "%s: RV reset: ANSWERED := TAKEN", K);
        tsr_transport_csr_write(s.t, TSR_CSR_CTRL, ctrl | TSR_CTRL_RV_RUN);
    };
    l = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) s.k("rt_stop"), { 0xFFFFFFF0u, 1 });
    CHECK(tsr_transport_launch(s.t, &l, &seqs[0]) == 0, "%s: rt_stop", K);
    l = session::L(TSR_DESC_TYPE_PING, 0, { 0x77 });
    CHECK(tsr_transport_launch(s.t, &l, &seqs[1]) == 0, "%s: PING queued behind it", K);
    CHECK(tsr_transport_wait(s.t, seqs[1], TSR_TIMEOUT_DEFAULT, &r) == 0 && r.w[2] == 0x77, "%s: the queued PING is served after the restart", K);
    CHECK(tsr_transport_wait(s.t, seqs[0], TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_LOST, "%s: the running launch is lost", K);
    bool hello = false;
    for (const auto & e : s.events) hello = hello || ty_of(e) == TSR_LCQ_R0_TYPE_HELLO;
    CHECK(hello, "%s: the HELLO went to the event callback", K);
    s.on_event = nullptr;
    CHECK(tsr_transport_soft_reset(s.t) == 0, "%s: soft reset", K);
    CHECK(tsr_transport_boot(s.t, s.rt) == 0 && transport_boot_info(s.t).warm, "%s: warm boot after the soft reset", K);
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 3 }), &r) == 0 && r.w[2] == 3, "%s: PING", K);
    (void) q;
}

// ---------------------------------------------------------------------------------------------
// The debug-aware timeout (DB §4.1): a halted hart and serial mode stop the clock
// ---------------------------------------------------------------------------------------------
void dmi_write(tsr_transport * t, uint32_t reg, uint32_t v) { tsr_transport_csr_write(t, TSR_CSR_DMI(reg), v); }

void test_debug_timeout(session & s) {
    const char * K = s.kind.c_str();
    if (!(tsr_transport_csr_read(s.t, TSR_CSR_CTL_ID) & TSR_CTL_ID_DM)) {
        printf("%s: debug-aware timeout: not run (no debug module in this simulator build)\n", K);
        return;
    }
    tsr_record r;
    const uint32_t dbg_cfg = tsr_transport_csr_read(s.t, TSR_CSR_DBG_CFG);
    tsr_transport_csr_write(s.t, TSR_CSR_DBG_CFG, dbg_cfg | TSR_DBG_CFG_DMI_EN);
    dmi_write(s.t, TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_DMACTIVE);
    std::atomic<uint32_t> seq{ 0 };
    std::atomic<bool> halted{ false };
    s.on_event = [&](const tsr_record & e) {
        if (ty_of(e) != 0xC3u || halted) return;
        seq = e.w[1];
        dmi_write(s.t, TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_HALTREQ | TSR_DM_DMCONTROL_DMACTIVE);   // halt the hart
        halted = true;
    };
    // Another thread (as DB's in-process debug service would) resumes the hart after 300 ms of
    // host time and stops the kernel; meanwhile the waiting thread lets device time pass.
    std::atomic<bool> resumed{ false };
    std::thread helper([&] {
        while (!halted) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const uint32_t st = tsr_transport_csr_read(s.t, TSR_CSR_STATUS);
        CHECK(st & TSR_STATUS_RV_HALTED, "%s: the hart is halted while the host waits (STATUS 0x%08x)", K, st);
        // The stop watermark first: the resumed kernel returns at its next poll, so no unhalted
        // time can pass before the answer (the 2 ms timeout would count it).
        tsr_transport_stop(s.t, seq);
        dmi_write(s.t, TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_RESUMEREQ | TSR_DM_DMCONTROL_DMACTIVE);
        resumed = true;
    });
    transport_set_timeouts(s.t, 0, 0, 0, 0);
    tsr_sim * sim = s.sim();
    const uint64_t t0 = sim ? tsr_sim_now(sim) : 0;
    const recovery_log before = transport_recovery_log(s.t);
    const int rc = s.run("rt_stop", { 0xFFFFFFF0u, 55 }, &r, 2 * MS);   // 2 ms timeout
    helper.join();
    const uint64_t dt = sim ? tsr_sim_now(sim) - t0 : 0;
    CHECK(rc == 0 && st_of(r) == TSR_LCQ_R0_STATUS_STOPPED && r.w[2] == 55 && resumed,
          "%s: no timeout while halted: rc %d r0 0x%08x", K, rc, r.w[0]);
    CHECK(transport_recovery_log(s.t).timeouts == before.timeouts, "%s: the ladder was not entered", K);
    printf("%s: debug-aware timeout: %.1f ms of device time passed during a 2 ms timeout while the hart was halted for 300 ms of host "
           "time (none of it counted)\n", K, (double) dt * 1e3 / CORE_HZ_NOMINAL);
    if (sim) CHECK((double) dt / CORE_HZ_NOMINAL > 0.004, "%s: device time moved past the timeout (%.3f ms)", K, (double) dt * 1e3 / CORE_HZ_NOMINAL);
    // Serial mode (DBG_CFG.SERIAL): the timeout is off.
    halted = false;
    s.on_event = [&](const tsr_record & e) {
        if (ty_of(e) == 0xC3u) {
            seq = e.w[1];
            halted = true;
        }
    };
    tsr_transport_csr_write(s.t, TSR_CSR_DBG_CFG, dbg_cfg | TSR_DBG_CFG_DMI_EN | TSR_DBG_CFG_SERIAL);
    std::thread helper2([&] {
        while (!halted) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        tsr_transport_stop(s.t, seq);
    });
    const int rc2 = s.run("rt_stop", { 0xFFFFFFF0u, 56 }, &r, 1 * MS);
    helper2.join();
    CHECK(rc2 == 0 && st_of(r) == TSR_LCQ_R0_STATUS_STOPPED && r.w[2] == 56, "%s: no timeout in serial mode: rc %d", K, rc2);
    tsr_transport_csr_write(s.t, TSR_CSR_DBG_CFG, dbg_cfg);
    s.on_event = nullptr;
}

// ---------------------------------------------------------------------------------------------
// The recovery ladder (DB §7.3) and TRAP handling (DB §6.3)
// ---------------------------------------------------------------------------------------------
bool file_exists(const std::string & p) {
    struct stat sb;
    return stat(p.c_str(), &sb) == 0;
}

void test_ladder(session & s) {
    const char * K = s.kind.c_str();
    tsr_record r;
    transport_set_timeouts(s.t, 0, 5, 0, 0);   // TSR_ABORT_MS 5
    // Level 2: rt_cancel holds its ops until a cancel; the ladder's LQ_CANCEL answers it ABORTED.
    recovery_log b = transport_recovery_log(s.t);
    int rc = s.run("rt_cancel", { H(0, IO_BEAT + 100), H(1, IO_BEAT + 300), 4, 16 }, &r, 1 * MS);
    recovery_log a = transport_recovery_log(s.t);
    CHECK(rc == TSR_TE_TIMEOUT && st_of(r) == TSR_LCQ_R0_STATUS_ABORTED, "%s: level 2: TSR_TE_TIMEOUT with the ABORTED answer "
          "(rc %d, r0 0x%08x)", K, rc, r.w[0]);
    CHECK(a.timeouts == b.timeouts + 1 && a.level2 == b.level2 + 1 && a.level2_ok == b.level2_ok + 1 && a.level3 == b.level3,
          "%s: recovered at level 2", K);
    CHECK(a.dumps == b.dumps + 1 && file_exists(a.last_dump_dir + "/state.json") && file_exists(a.last_dump_dir + "/units.txt") &&
          file_exists(a.last_dump_dir + "/log.txt"), "%s: hang dump in %s", K, a.last_dump_dir.c_str());
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 4 }), &r) == 0 && r.w[2] == 4, "%s: PING after level 2", K);
    // Level 3: rt_stop ignores the cancel; the ladder soft-resets and restarts warm. The PING
    // queued behind it is lost.
    b = a;
    tsr_launch l = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) s.k("rt_stop"), { 0xFFFFFFF0u, 1 });
    uint32_t q0 = 0, q1 = 0;
    CHECK(tsr_transport_launch(s.t, &l, &q0) == 0, "%s: rt_stop", K);
    l = session::L(TSR_DESC_TYPE_PING, 0);
    CHECK(tsr_transport_launch(s.t, &l, &q1) == 0, "%s: PING behind", K);
    rc = tsr_transport_wait(s.t, q0, 1 * MS, &r);
    a = transport_recovery_log(s.t);
    CHECK(rc == TSR_TE_TIMEOUT && a.level2 == b.level2 + 1 && a.level2_ok == b.level2_ok && a.level3 == b.level3 + 1 &&
          a.restarts == b.restarts + 1, "%s: level 3: rc %d", K, rc);
    CHECK(tsr_transport_wait(s.t, q1, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_LOST, "%s: the queued PING is lost", K);
    CHECK(transport_boot_info(s.t).warm, "%s: the restart is warm", K);
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 5 }), &r) == 0 && r.w[2] == 5, "%s: PING after level 3", K);
    transport_set_timeouts(s.t, 60000, 10000, 0, 0);
}

std::string run_addr2line(const char * elf, uint32_t pc) {
    if (g_addr2line.empty() || access(g_addr2line.c_str(), X_OK) != 0) return std::string();
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "'%s' -e '%s' 0x%x 2>/dev/null", g_addr2line.c_str(), elf, pc);
    FILE * p = popen(cmd, "r");
    if (!p) return std::string();
    char buf[1024] = "";
    if (!fgets(buf, sizeof(buf), p)) buf[0] = '\0';
    pclose(p);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    const size_t d = s.find(" (discriminator");
    if (d != std::string::npos) s.resize(d);
    return s;
}

std::string base_line(const std::string & fl) {   // "dir/file.cpp:12" -> "file.cpp:12"
    const size_t sl = fl.rfind('/');
    return sl == std::string::npos ? fl : fl.substr(sl + 1);
}

void test_traps(session & s) {
    const char * K = s.kind.c_str();
    tsr_record r;
    uint32_t trap_insn = 0;
    tsr_elf_symbol(image_elf(s.rt), "rt_trap_insn", &trap_insn);
    struct tcase { uint32_t kind, mcause; const char * what; };
    const tcase cs[5] = { { 0, 2, "illegal instruction" }, { 1, TSR_PANIC_MCAUSE, "panic" }, { 2, 5, "load access fault" },
                          { 3, 3, "ebreak" }, { 4, TSR_PANIC_MCAUSE, "bad event type" } };
    for (const tcase & c : cs) {
        const recovery_log b = transport_recovery_log(s.t);
        tsr_launch l = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) s.k("rt_trap"), { c.kind, 0x0000CAFEu });
        uint32_t q0 = 0, q1 = 0;
        CHECK(tsr_transport_launch(s.t, &l, &q0) == 0, "%s: rt_trap %u", K, c.kind);
        l = session::L(TSR_DESC_TYPE_PING, 0);
        CHECK(tsr_transport_launch(s.t, &l, &q1) == 0, "%s: PING behind the trap", K);
        const int rc = tsr_transport_wait(s.t, q0, TSR_TIMEOUT_DEFAULT, &r);
        const recovery_log a = transport_recovery_log(s.t);
        CHECK(rc == 0 && ty_of(r) == TSR_LCQ_R0_TYPE_TRAP && st_of(r) == TSR_LCQ_R0_STATUS_TRAP && r.w[1] == q0 &&
              r.w[TSR_LCQ_TRAP_MCAUSE_WORD] == c.mcause && r.w[TSR_LCQ_TRAP_RT_INFO_WORD] == 0xCAFEu,
              "%s: %s: TRAP record (rc %d r0 0x%08x mcause %u RT_INFO 0x%x)", K, c.what, rc, r.w[0], r.w[2], r.w[6]);
        CHECK(a.traps == b.traps + 1 && a.restarts == b.restarts + 1, "%s: %s: handled and restarted", K, c.what);
        CHECK(tsr_transport_wait(s.t, q1, TSR_TIMEOUT_DEFAULT, &r) == TSR_TE_LOST, "%s: %s: the launch behind it is lost", K, c.what);
        const uint32_t mepc = 0;
        (void) mepc;
        if (c.kind == 0) {
            CHECK(a.last_trap.find("illegal instruction") != std::string::npos && a.last_trap.find("rt_trap") != std::string::npos &&
                  a.last_trap.find("rt_kernels.cpp:") != std::string::npos, "%s: the report names the cause, rt_trap and its line: %s", K,
                  a.last_trap.c_str());
            printf("%s: %s\n", K, a.last_trap.c_str());
        }
        if (c.kind == 1) CHECK(a.last_trap.find("panic code 0xdead0001") != std::string::npos, "%s: panic code: %s", K, a.last_trap.c_str());
        CHECK(tsr_transport_call(s.t, &l, TSR_TIMEOUT_DEFAULT, &r) == 0 && ty_of(r) == TSR_LCQ_R0_TYPE_PONG, "%s: %s: PING after the restart", K, c.what);
    }
    // mepc -> line, against addr2line (the pinned RISC-V binutils): the trap instruction and a few
    // other code addresses.
    if (trap_insn != 0) {
        const std::string ours = image_line(s.rt, trap_insn), theirs = run_addr2line(RT_IMAGE, trap_insn);
        if (theirs.empty()) {
            printf("%s: addr2line not available: line check against it not run\n", K);
        } else {
            CHECK(base_line(ours) == base_line(theirs), "%s: line of rt_trap_insn: ours %s, addr2line %s", K, ours.c_str(), theirs.c_str());
            uint32_t ok = 0, total = 0;
            for (const char * sym : { "rt_args3", "rt_copy", "rt_log", "rt_stop", "tsr_rt_main", "main" }) {
                uint32_t a = 0;
                if (tsr_elf_symbol(image_elf(s.rt), sym, &a) != 0) continue;
                for (uint32_t off = 0; off < 24; off += 2) {
                    const std::string o = base_line(image_line(s.rt, a + off)), t2 = base_line(run_addr2line(RT_IMAGE, a + off));
                    if (t2.empty() || t2.find("??") != std::string::npos) continue;
                    total++;
                    ok += o == t2;
                    if (o != t2) fprintf(stderr, "  line of 0x%x (%s+%u): ours %s, addr2line %s\n", a + off, sym, off, o.c_str(), t2.c_str());
                }
            }
            CHECK(total > 0 && ok == total, "%s: DWARF lines equal addr2line's at %u / %u addresses", K, ok, total);
            printf("%s: mepc -> line: rt_trap_insn at %s; %u / %u code addresses give addr2line's file:line\n", K, base_line(ours).c_str(),
                   ok, total);
        }
    }
}

// A trap between launches (EX §4.4: TRAP answers a launch only while one is ACTIVE). The hart is
// halted while idle (the DM model), a PING is queued behind it and not taken, and the hart resumed
// at an address without code: its TRAP record carries r1 = LQ_HEAD_SEQ = the PING's seq, but
// ANSWERED did not move past the PING, so the record is no answer: the restart loses the PING.
void test_trap_between(session & s) {
    const char * K = s.kind.c_str();
    if (!(tsr_transport_csr_read(s.t, TSR_CSR_CTL_ID) & TSR_CTL_ID_DM)) {
        printf("%s: trap between launches: not run (no debug module in this simulator build)\n", K);
        return;
    }
    tsr_record r;
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 1 }), &r) == 0, "%s: idle", K);
    const uint32_t dbg_cfg = tsr_transport_csr_read(s.t, TSR_CSR_DBG_CFG);
    tsr_transport_csr_write(s.t, TSR_CSR_DBG_CFG, dbg_cfg | TSR_DBG_CFG_DMI_EN);
    dmi_write(s.t, TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_DMACTIVE);
    dmi_write(s.t, TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_HALTREQ | TSR_DM_DMCONTROL_DMACTIVE);
    bool halted = false;
    for (int i = 0; i < 1000 && !halted; ++i) {
        halted = (tsr_transport_csr_read(s.t, TSR_CSR_STATUS) & TSR_STATUS_RV_HALTED) != 0;
        if (!halted) transport_kdev(s.t)->idle(0, 10000);
    }
    CHECK(halted, "%s: the idle hart halts", K);
    uint32_t q = 0;
    const tsr_launch ping = session::L(TSR_DESC_TYPE_PING, 0, { 0x99u });
    CHECK(tsr_transport_launch(s.t, &ping, &q) == 0, "%s: a PING queued behind the halted hart", K);
    CHECK(tsr_transport_csr_read(s.t, TSR_CSR_LQ_TAKEN) == q, "%s: the PING is not taken (TAKEN %u, seq %u)", K,
          tsr_transport_csr_read(s.t, TSR_CSR_LQ_TAKEN), q);
    dmi_write(s.t, TSR_DMI_DATA0, 0x0F000000u);   // dpc: no memory there
    dmi_write(s.t, TSR_DMI_COMMAND, (2u << TSR_DM_CMD_AARSIZE_LSB) | TSR_DM_CMD_TRANSFER | TSR_DM_CMD_WRITE | 0x7B1u);
    CHECK((tsr_transport_csr_read(s.t, TSR_CSR_DMI(TSR_DMI_ABSTRACTCS)) & TSR_DM_ABSTRACTCS_CMDERR) == 0, "%s: dpc written", K);
    const recovery_log b = transport_recovery_log(s.t);
    s.events.clear();
    dmi_write(s.t, TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_RESUMEREQ | TSR_DM_DMCONTROL_DMACTIVE);
    const int rc = tsr_transport_wait(s.t, q, TSR_TIMEOUT_DEFAULT, &r);
    const recovery_log a = transport_recovery_log(s.t);
    CHECK(rc == TSR_TE_LOST, "%s: trap between launches: the queued PING is lost, not answered by the TRAP (rc %d, r0 0x%08x)", K, rc,
          r.w[0]);
    CHECK(a.traps == b.traps + 1 && a.restarts == b.restarts + 1 && a.last_trap.find("mcause") != std::string::npos,
          "%s: trap between launches: reported and restarted (%s)", K, a.last_trap.c_str());
    tsr_transport_csr_write(s.t, TSR_CSR_DBG_CFG, dbg_cfg);
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 2 }), &r) == 0 && r.w[2] == 2, "%s: PING after the trap between launches", K);
}

// EX §4.8's LQ_W shadow follows raw CSR writes: an LQ_W word written behind the transport (a tool, a
// debugger) is rewritten by the next launch even if that launch's word equals the old shadow.
void test_shadow(session & s) {
    const char * K = s.kind.c_str();
    tsr_record r;
    const tsr_launch ping = session::L(TSR_DESC_TYPE_PING, 0, { 0x5A5A0001u });
    CHECK(s.call(ping, &r) == 0 && r.w[2] == 0x5A5A0001u, "%s: PING", K);
    CHECK(tsr_transport_csr_write(s.t, TSR_CSR_LQ_W(1), 0xDEAD0001u) == 0, "%s: LQ_W(1) written behind the transport", K);
    CHECK(s.call(ping, &r) == 0 && r.w[2] == 0x5A5A0001u, "%s: the same PING again sees its own ARG0 (r2 0x%08x)", K, r.w[2]);
}

// A debug configuration an earlier debug session left armed (DB keeps SERIAL, the stop sources and
// the TBPs across SOFT_RESET): a new open sees it, so its waits do not count serial mode as a hang.
void test_stale_debug_config() {
    tsr_transport * a = tsr_transport_open("sim");
    CHECK(a != nullptr, "sim: open");
    if (!a) return;
    if (tsr_transport_csr_read(a, TSR_CSR_DBG_ID) == 0) {
        printf("sim: stale debug configuration: not run (no DBG block)\n");
        tsr_transport_close(a);
        return;
    }
    const uint32_t c = tsr_transport_csr_read(a, TSR_CSR_DBG_CFG);
    tsr_transport_csr_write(a, TSR_CSR_DBG_CFG, c | TSR_DBG_CFG_SERIAL);
    session s("sim", transport_detach_kdev(a));   // process A exits, a new one opens the device
    if (!s.ok) return;
    CHECK(tsr_transport_csr_read(s.t, TSR_CSR_DBG_CFG) & TSR_DBG_CFG_SERIAL, "sim: SERIAL kept across the open's SOFT_RESET");
    CHECK(tsr_transport_boot(s.t, s.rt) == 0, "sim: boot");
    std::atomic<uint32_t> seq{ 0 };
    std::atomic<bool> started{ false };
    s.on_event = [&](const tsr_record & e) {
        if (ty_of(e) == 0xC3u) {
            seq = e.w[1];
            started = true;
        }
    };
    std::thread helper([&] {
        while (!started) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        tsr_transport_stop(s.t, seq);
    });
    tsr_record r;
    const int rc = s.run("rt_stop", { 0xFFFFFFF0u, 57 }, &r, 1 * MS);
    helper.join();
    CHECK(rc == 0 && st_of(r) == TSR_LCQ_R0_STATUS_STOPPED && r.w[2] == 57,
          "sim: no timeout in the serial mode an earlier session left (rc %d, r0 0x%08x)", rc, r.w[0]);
    s.on_event = nullptr;
    tsr_transport_csr_write(s.t, TSR_CSR_DBG_CFG, c);
}

void test_gdb_wait(const std::string & kind) {
    setenv("TSR_GDB_WAIT", "launch", 1);
    session s(kind);
    unsetenv("TSR_GDB_WAIT");
    if (!s.ok) return;
    const char * K = kind.c_str();
    tsr_record r;
    CHECK(tsr_transport_boot(s.t, s.rt) == 0, "%s: boot", K);
    CHECK(s.run("rt_trap", { 0, 1 }, &r) == 0 && ty_of(r) == TSR_LCQ_R0_TYPE_TRAP, "%s: TSR_GDB_WAIT: TRAP answer", K);
    CHECK(TSR_GET(tsr_transport_csr_read(s.t, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED,
          "%s: TSR_GDB_WAIT: the device stays TRAPPED", K);
    CHECK(transport_recovery_log(s.t).restarts == 0, "%s: no restart", K);
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0), &r) == TSR_TE_STATE, "%s: launches refused while TRAPPED", K);
    CHECK(tsr_transport_soft_reset(s.t) == 0 && tsr_transport_boot(s.t, s.rt) == 0, "%s: soft reset + boot", K);
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 6 }), &r) == 0 && r.w[2] == 6, "%s: PING", K);
}

// The mock's TCM write fault: the receive CRC matches, VERIFY does not, the boot fails (EX §3.3).
void test_tcm_fault() {
    tsr_image * img = nullptr;
    if (tsr_image_open(RT_IMAGE, &img) != 0) return;
    mock_options o;
    o.threads = 2;
    o.tcm_fault_addr = image_segments(img)[0].paddr + 0x100;
    o.tcm_fault_xor = 0x00010000u;
    std::unique_ptr<mock_kdev> m(new mock_kdev(o));
    session s("mock", std::move(m));
    if (s.ok) {
        CHECK(tsr_transport_boot(s.t, s.rt) == TSR_TE_IMAGE, "mock: VERIFY catches the TCM write fault");
        CHECK(transport_boot_info(s.t).words_written > 0, "mock: the words were written");
    }
    tsr_image_close(img);
}

// ---------------------------------------------------------------------------------------------
// EX §4.11 numbers on sim
// ---------------------------------------------------------------------------------------------
void measure(session & s) {
    const char * K = s.kind.c_str();
    tsr_sim * sim = s.sim();
    if (!sim) return;
    tsr_record r;
    tsr_image * fw = nullptr;
    if (tsr_image_open(FW_IMAGE, &fw) != 0 || tsr_transport_boot(s.t, fw) != 0) {
        CHECK(0, "%s: boot tsr-fw.elf for the measurements", K);
        tsr_image_close(fw);
        return;
    }
    const boot_info cold = transport_boot_info(s.t);
    CHECK(tsr_transport_boot(s.t, fw) == 0, "%s: warm boot of tsr-fw.elf", K);
    const boot_info warm = transport_boot_info(s.t);
    printf("%s: boot of tsr-fw.elf: cold %.3f ms device time (%u words, HELLO after %u core cycles), warm %.3f ms (%u words); "
           "on the card add ~%.1f ms of posted LOAD_DATA writes for the cold load (EX §3.3: 15-25 core cycles per word)\n", K,
           (double) cold.time_ns / 1e6, cold.words_written, cold.hello_cycles, (double) warm.time_ns / 1e6, warm.words_written,
           (double) cold.words_written * 20.0 / CORE_HZ_NOMINAL * 1e3);
    // An empty launch: `replay` with no records (RET 0 on success, SW §6.4: EX §4.8's two reads).
    const tsr_launch empty = session::L(TSR_DESC_TYPE_LAUNCH, (uint16_t) tsr_image_kernel(fw, "replay"), { 0, 0, 0 });
    const tsr_launch ping = session::L(TSR_DESC_TYPE_PING, 0);
    transport_set_full_records(s.t, true);
    uint64_t dur_e = 0, dur_p = 0;
    for (int i = 0; i < 16; ++i) {
        CHECK(s.call(empty, &r) == 0 && st_of(r) == 0 && r.w[2] == 0, "%s: empty replay launch OK", K);
        dur_e += r.w[7];
        CHECK(s.call(ping, &r) == 0, "%s: PING", K);
        dur_p += r.w[7];
    }
    transport_set_full_records(s.t, false);
    tsr_transport_stats a, b;
    const int N = 200;
    for (const tsr_launch * l : { &ping, &empty }) {
        tsr_transport_get_stats(s.t, &a);
        uint64_t t0 = tsr_sim_now(sim), w0 = wall_ns();
        for (int i = 0; i < N; ++i) CHECK(s.call(*l, &r) == 0 && st_of(r) == 0, "%s: round trip", K);
        const uint64_t dt = tsr_sim_now(sim) - t0, dw = wall_ns() - w0;
        tsr_transport_get_stats(s.t, &b);
        printf("%s: round trip of %s: %.0f core cycles = %.2f us device time (DURATION %.0f cycles); %.1f us host per launch on the "
               "simulator; %.1f CSR reads + %.1f writes per launch\n", K, l == &ping ? "a PING" : "an empty launch (replay, 0 records)",
               (double) dt / N, (double) dt / N / CORE_HZ_NOMINAL * 1e6, (double) (l == &ping ? dur_p : dur_e) / 16, (double) dw / N / 1e3,
               (double) (b.csr_reads - a.csr_reads) / N, (double) (b.csr_writes - a.csr_writes) / N);
    }
    // Pipelined: LQ_DEPTH outstanding.
    tsr_transport_get_stats(s.t, &a);
    const uint64_t t0 = tsr_sim_now(sim), w0 = wall_ns();
    const int P = 1000;
    std::vector<uint32_t> seqs;
    uint32_t done = 0;
    for (int i = 0; i < P; ++i) {
        uint32_t q = 0;
        if (tsr_transport_launch(s.t, &empty, &q) != 0) break;
        seqs.push_back(q);
        if (seqs.size() - done > TSR_LQ_DEPTH) {
            CHECK(tsr_transport_wait(s.t, seqs[done], TSR_TIMEOUT_DEFAULT, &r) == 0, "%s: pipelined wait", K);
            done++;
        }
    }
    for (; done < seqs.size(); ++done) tsr_transport_wait(s.t, seqs[done], TSR_TIMEOUT_DEFAULT, &r);
    const uint64_t dt = tsr_sim_now(sim) - t0, dw = wall_ns() - w0;
    tsr_transport_get_stats(s.t, &b);
    printf("%s: pipelined (%u outstanding): %d empty launches in %.0f core cycles: %.0f launches/s device time (%.2f us each), "
           "%.0f launches/s host; %.1f CSR reads + %.1f writes per launch\n", K, TSR_LQ_DEPTH, P, (double) dt,
           (double) P / ((double) dt / CORE_HZ_NOMINAL), (double) dt / P / CORE_HZ_NOMINAL * 1e6, (double) P / ((double) dw / 1e9),
           (double) (b.csr_reads - a.csr_reads) / P, (double) (b.csr_writes - a.csr_writes) / P);
    tsr_image_close(fw);
}

// SW §6.9: what tsr_image_open refuses (mutated copies of tsr-fw.elf), and the versions of a booted
// device.
void test_versions() {
    printf("== versions\n");
    char err[256];
    tsr_elf * e = nullptr;
    if (tsr_elf_read(FW_IMAGE, &e, err, sizeof(err)) != 0) {
        CHECK(0, "tsr-elf: %s", err);
        return;
    }
    size_t nb = 0;
    const uint8_t * b0 = tsr_elf_bytes(e, &nb);
    const std::vector<uint8_t> good(b0, b0 + nb);
    uint32_t a = 0, note_off = 0, kinds_off = 0, sz = 0, ty = 0;
    CHECK(tsr_elf_section(e, ".note.tsr", &a, &note_off, &sz, &ty) == 0 && tsr_elf_section(e, ".tsr_kinds", &a, &kinds_off, &sz, &ty) == 0,
          "tsr-fw.elf: .note.tsr and .tsr_kinds");
    uint8_t rev[64];
    CHECK(tsr_elf_kinds(e, rev, nullptr, 0) == int(n_defined_kinds()), "tsr-fw.elf declares the %u kinds of tsr/opl.hpp", n_defined_kinds());
    const uint32_t sig = tsr::opl::kinds_sig(rev);
    tsr_elf_free(e);
    const uint32_t desc = note_off + 16u;   // .note.tsr: 12-byte header, "TSR\0", then the descriptor
    auto put32 = [](std::vector<uint8_t> & b, uint32_t at, uint32_t v) { for (int i = 0; i < 4; ++i) b[at + i] = uint8_t(v >> 8 * i); };
    struct mut { const char * what; std::function<void(std::vector<uint8_t> &)> f; int rc; const char * msg; };
    const std::string abi_msg = ": Basis ABI " + std::to_string(TSR_SDK_ABI + 1u) + ", this host implements ABI " + std::to_string(TSR_SDK_ABI);
    const mut muts[] = {
        { "another Basis ABI", [&](std::vector<uint8_t> & b) { put32(b, desc, TSR_SDK_ABI + 1u); }, TSR_TE_IMAGE,
          abi_msg.c_str() },
        { "no Basis version", [&](std::vector<uint8_t> & b) { put32(b, desc + 24, 0); }, TSR_TE_IMAGE,
          ": no Tesseract Basis version in .note.tsr (built before versioning)" },
        { "a newer Basis minor with the same ABI", [&](std::vector<uint8_t> & b) { put32(b, desc + 24, TSR_BASIS_VERSION + 0x100u); }, 0, "" },
        { "kind 0x10 twice", [&](std::vector<uint8_t> & b) { b[kinds_off + 4] = b[kinds_off]; }, TSR_TE_IMAGE,
          ": malformed kind table .tsr_kinds (kind 0x" },
    };
    for (const mut & m : muts) {
        std::vector<uint8_t> b = good;
        m.f(b);
        const std::string p = g_scratch + "/versions.elf";
        FILE * f = fopen(p.c_str(), "wb");
        CHECK(f && fwrite(b.data(), 1, b.size(), f) == b.size(), "write %s", p.c_str());
        if (f) fclose(f);
        tsr_image * img = nullptr;
        int rc = 0;
        const std::string msg = stderr_of([&] { rc = tsr_image_open(p.c_str(), &img); });
        CHECK(rc == m.rc, "tsr_image_open with %s: rc %d, expected %d (%s)", m.what, rc, m.rc, msg.c_str());
        CHECK(msg.find(m.msg) != std::string::npos, "tsr_image_open with %s: message: %s", m.what, msg.c_str());
        if (img) {
            tsr_versions v;
            CHECK(tsr_image_versions(img, &v) == 0 && v.basis == TSR_BASIS_VERSION + 0x100u && v.core == 0 && v.abi == TSR_SDK_ABI,
                  "%s: image versions", m.what);
            tsr_image_close(img);
        }
    }
    // A booted device: the capability and CAPS of the device, the image's Basis version, ABI and
    // kinds as HELLO proved them.
    session s("sim");
    if (!s.ok) return;
    tsr_image * fw = nullptr;
    CHECK(tsr_image_open(FW_IMAGE, &fw) == 0 && tsr_transport_boot(s.t, fw) == 0, "boot tsr-fw.elf");
    tsr_versions v;
    CHECK(tsr_transport_versions(s.t, &v) == 0, "tsr_transport_versions");
    CHECK(v.core == core_cap() && v.caps == tsr_transport_config(s.t)->caps && v.core_min == core_cap() && v.basis == TSR_BASIS_VERSION &&
              v.basis == tsr_basis_version() && v.abi == TSR_SDK_ABI && v.n_kinds == n_defined_kinds() &&
              v.kind_rev[opl::K_FLASH_ATTN] == opl::kind_rev(opl::K_FLASH_ATTN) && v.kind_rev[0x20] == 0 && v.kinds_sig == sig &&
              v.image_id == tsr_image_id(fw),
          "versions: core 0x%04x basis 0x%06x abi %u kinds %u sig 0x%04x", v.core, v.basis, v.abi, v.n_kinds, v.kinds_sig);
    char str[512];
    const int len = tsr_versions_str(&v, 0, str, sizeof(str));
    const std::string core_pre = "Tesseract Core " + cap_str(core_cap()) + " (VU";
    const std::string basis_part = ", Tesseract Basis " + basis_str(TSR_BASIS_VERSION) + " (ABI " + std::to_string(TSR_SDK_ABI) + ", " +
                                   std::to_string(n_defined_kinds()) + " kinds)";
    CHECK(len > 0 && strstr(str, core_pre.c_str()) == str && strstr(str, basis_part.c_str()) != nullptr, "tsr_versions_str: %s", str);
    tsr_versions_str(&v, 1, str, sizeof(str));
    char sigs[32];
    snprintf(sigs, sizeof(sigs), ", kinds 0x%04x: ", sig);
    const std::string kinds_pre = std::string(sigs) + kind_rev_str(opl::K_NOP) + " " + kind_rev_str(opl::K_JOIN) + " " +
                                  kind_rev_str(opl::K_MUL_MAT) + " ";
    CHECK(strstr(str, kinds_pre.c_str()) != nullptr && strstr(str, (" " + kind_rev_str(opl::K_UNARY)).c_str()) != nullptr,
          "tsr_versions_str verbose: %s", str);
    printf("versions: %s\n", str);
    tsr_image_close(fw);
    // An image of an older Basis (fw_basis_old, 0.0.5 in its note and HELLO r6) boots: the Basis rule
    // is the backends' (ggml-tesseract, sw/rt), so captured traces of older images stay replayable.
    tsr_image * old = nullptr;
    CHECK(tsr_image_open(TSR_FW_TEST_IMAGE_fw_basis_old, &old) == 0 && tsr_transport_boot(s.t, old) == 0, "boot fw_basis_old");
    CHECK(tsr_transport_versions(s.t, &v) == 0 && v.basis == 0x000005u, "fw_basis_old: Tesseract Basis 0x%06x", v.basis);
    tsr_image_close(old);
}

void run_kind(const std::string & kind, observed * ob) {
    printf("== %s\n", kind.c_str());
    session s(kind);
    if (!s.ok) return;
    test_first_boot(s);
    scenario(s, ob);
    test_boot(s);
    test_pipeline(s);
    test_rings(s);
    test_log(s);
    test_control(s);
    test_debug_timeout(s);
    test_traps(s);
    test_trap_between(s);
    test_shadow(s);
    tsr_record r;
    CHECK(s.call(session::L(TSR_DESC_TYPE_PING, 0, { 8 }), &r) == 0, "%s: PING at the end", kind.c_str());
    if (kind == "sim") measure(s);
    tsr_transport_stats st;
    tsr_transport_get_stats(s.t, &st);
    printf("%s: %" PRIu64 " launches, %" PRIu64 " CSR reads, %" PRIu64 " writes, %" PRIu64 " log words\n", kind.c_str(), st.launches,
           st.csr_reads, st.csr_writes, st.log_words);
}

void ladder_kind(const std::string & kind) {
    const std::string dd = g_scratch + "/dumps-" + kind;
    setenv("TSR_DUMP_DIR", dd.c_str(), 1);
    session s(kind);
    unsetenv("TSR_DUMP_DIR");
    if (!s.ok) return;
    CHECK(tsr_transport_boot(s.t, s.rt) == 0, "%s: boot", kind.c_str());
    test_ladder(s);
}

bool same_r06(const tsr_record & a, const tsr_record & b) {
    for (int i = 0; i < 7; ++i)
        if (a.w[i] != b.w[i]) return false;
    return true;
}

void compare(const char * what, const observed & a, const observed & b) {
    CHECK(a.recs.size() == b.recs.size(), "%s: %zu vs %zu records", what, a.recs.size(), b.recs.size());
    for (size_t i = 0; i < a.recs.size() && i < b.recs.size(); ++i)
        CHECK(same_r06(a.recs[i], b.recs[i]), "%s: record %zu (type %s) r0-r6 identical", what, i, type_name(a.recs[i].w[0]).c_str());
    CHECK(a.dram == b.dram, "%s: DRAM identical", what);
    CHECK(a.log == b.log, "%s: log words identical (%zu vs %zu)", what, a.log.size(), b.log.size());
}

} // namespace

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--addr2line") && i + 1 < argc) g_addr2line = argv[++i];
        else if (!strcmp(argv[i], "--scratch") && i + 1 < argc) g_scratch = argv[++i];
    }
    if (access(RT_IMAGE, R_OK) != 0 || access(FW_IMAGE, R_OK) != 0) {
        printf("transport_protocol: skipped: firmware images %s / %s not built\n", RT_IMAGE, FW_IMAGE);
        return 77;
    }
    if (system(("mkdir -p '" + g_scratch + "'").c_str()) != 0) return 2;
    clear_env();
    observed sim_min, sim_full, mock_min;
    run_kind("sim", &sim_min);
    run_kind("mock", &mock_min);
    {   // the scenario again on sim with full record reads
        printf("== sim, full record reads\n");
        session s("sim");
        if (s.ok && tsr_transport_boot(s.t, s.rt) == 0) {
            transport_set_full_records(s.t, true);
            scenario(s, &sim_full);
        }
    }
    compare("sim minimal vs full reads", sim_min, sim_full);
    compare("sim vs mock", sim_min, mock_min);
    test_versions();
    test_gdb_wait("sim");
    test_stale_debug_config();
    test_tcm_fault();
    ladder_kind("sim");
    ladder_kind("mock");
    // vkernel: only with a two-clock server (P5.3).
    if ((getenv("TSR_VKERNEL_SERVER") && getenv("TSR_VKERNEL_SERVER")[0]) || (getenv("TSR_VKERNEL_SOCKET") && getenv("TSR_VKERNEL_SOCKET")[0])) {
        observed vk;
        run_kind("vkernel", &vk);
        compare("sim vs vkernel", sim_min, vk);
    } else {
        printf("== vkernel: not run (no Verilated kernel server until P5.3; set TSR_VKERNEL_SERVER)\n");
    }
    printf("transport_protocol: %d checks, %d failed\n", g_checks.load(), g_fail.load());
    return g_fail.load() ? 1 : 0;
}

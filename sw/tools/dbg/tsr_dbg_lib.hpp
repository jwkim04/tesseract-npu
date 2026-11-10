// tsr_dbg_lib.hpp — libtsr_dbg: the debug service of docs/design/05-debug.md "DB" §3-§9
// over the transport API only (sw/transport/tsr_transport.h, DB §4.1, §9.2), so it runs on every kind
// (sim, mock, vkernel, xrt). docs/design/06-software-verification.md "SW" §9.5.
//
//   dmi       the DMI driver (DB §3.2, §3.5, §3.6): Access Register commands, the program buffer
//             (fence.i), halt / resume / ndmreset, SBA blocks with the DB §3.6 sequences
//   target    the RISC-V debug target (DB §3.4-§3.9, §4.2, §4.4-§4.7): attach / detach, registers
//             (x0-x31, pc = dpc, the CSRs of §4.4, priv), memory routed per §4.5 with §3.6's write
//             policy (TCMs through SBA, or EX's LOAD_* while RV_RUN = 0; windows read-only; SPM
//             writes only in a stopped tile state), software breakpoints (c.ebreak / ebreak in
//             ITCM), hardware breakpoints (Ibex triggers 0-1), RV watchpoints (DBG_RVWPj) and SPM
//             watchpoints (plus a tile breakpoint, §4.6), stop classification (§4.7: breakpoints,
//             __tsr_launch_halt, __tsr_trap_halt with the mret view, watchpoints, tile stops, halt
//             at boot, Ctrl-C), the firmware ELF's symbols and DWARF lines
//   monitor   the monitor command set (DB §4.8, the same in the tsr-dbg CLI, §9.3): reset, load,
//             launch-break, hold, status (§7.2 triage), log, crash, crash-view, tile (§5), hist
//             (§5.6), units / qstatus (§5.5), perf, spm, sreg, mem, dump (§7.4), recover (§7.3),
//             spm-write, trw (§8.3 capture), launch / ping (the standalone server's launches), rv
//   rsp       the GDB remote serial protocol server (DB §4.2, §4.3, §4.9): TCP, Unix socket or
//             stdio; all-stop, one thread; qSupported, qXfer features / memory-map, g G p P m M X,
//             c s C S vCont, Z0-Z4, qRcmd, qSymbol, QStartNoAckMode, D, k, Ctrl-C
//   trw       the trace writer (DB §8.3): ring set-up in DRAM, capture, decode (sw/sim's
//             tsr_trw_decode), the Chrome / Perfetto JSON timeline and the DB §8.7 self-checks
//   embed     the embedded service (DB-17, §9.5): TSR_GDB=tcp:<port>|unix:<path>, TSR_GDB_WAIT=
//             boot|launch, TSR_DEBUGGER=sock:<path>, through tsr_transport_set_debug_service
//
// Errors are DB §4.9's codes: E01 DMI timeout, E1x abstract command (x = cmderr), E2x SBA (x =
// sberror, 8 = sbbusyerror), E04 policy, E05 target running, E06 malformed, E07 transport.
// Threads: one target per transport; its calls are serialized by the caller (the RSP thread);
// every device access goes through the transport's mutex.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tsr_transport.h"

struct tsr_elf;

namespace tsr::dbg {

// DB §4.9
enum : int {
    E_OK        = 0,
    E_DMI       = 0x01,   // DMI timeout (no clk_rv)
    E_POLICY    = 0x04,   // refused: D15 window, unsafe SPM write, no free trigger / TBP / watchpoint
    E_RUNNING   = 0x05,   // the hart runs (or does not halt)
    E_MALFORMED = 0x06,
    E_TRANSPORT = 0x07,
    E_CMD       = 0x10,   // | cmderr
    E_SBA       = 0x20,   // | sberror; 0x28 sbbusyerror
};
std::string err_str(int e);   // "E21 (SBA failed: sberror 1)"

// ---------------------------------------------------------------------------------------------
// DMI driver (DB §3.5, §3.6)
// ---------------------------------------------------------------------------------------------
class dmi {
public:
    explicit dmi(tsr_transport * t) : t_(t) {}
    tsr_transport * transport() const { return t_; }

    int read(uint32_t addr7, uint32_t * v);
    int write(uint32_t addr7, uint32_t v);
    // Polls until pred(dmstatus) or timeout (the transport's time base): between polls the device
    // time advances (tsr_transport_idle). E_RUNNING on a timeout (E_DMI if STATUS.DMI_TIMEOUT).
    int poll_dmstatus(uint32_t mask, uint64_t timeout_ns, uint32_t * dmstatus = nullptr);
    int activate();                          // dmcontrol = dmactive
    int halt(uint64_t timeout_ns);           // haltreq, allhalted, haltreq cleared
    int resume(uint64_t timeout_ns);         // resumereq, allresumeack
    int set_haltreq(bool on);                // the level alone (Ctrl-C: poll elsewhere)
    // Access Register (aarsize 2): regno 0x1000 + i GPR, else CSR. E_CMD | cmderr on failure
    // (cmderr cleared), E_RUNNING when busy never drops.
    int reg_read(uint32_t regno, uint32_t * v);
    int reg_write(uint32_t regno, uint32_t v);
    int exec_progbuf(const uint32_t * w, unsigned n);   // postexec only; the words then ebreak
    int fence_i();
    // SBA (DB §3.6): word blocks at a word-aligned address; 8 / 16-bit single accesses. E_SBA | x.
    int sba_read32(uint32_t a, uint32_t * w, size_t n);
    int sba_write32(uint32_t a, const uint32_t * w, size_t n);
    int sba_write16(uint32_t a, uint16_t v);
    int sba_write8(uint32_t a, uint8_t v);

    uint64_t accesses = 0;                   // DMI reads + writes (statistics)

private:
    int abstract_wait();
    int sba_check();
    tsr_transport * t_;
};

// ---------------------------------------------------------------------------------------------
// The firmware image: symbols, lines, the debug contract's addresses (DB §6.3)
// ---------------------------------------------------------------------------------------------
class image_info {
public:
    image_info() = default;
    ~image_info();
    image_info(const image_info &) = delete;
    image_info & operator=(const image_info &) = delete;
    // From a booted transport image (preferred: symbolization through the transport's DWARF
    // reader) or an ELF path. false with a message.
    bool load(const tsr_image * img, std::string * why);
    bool load_path(const std::string & path, std::string * why);
    bool loaded() const { return elf_ != nullptr; }
    const tsr_image * image() const { return img_; }
    const std::string & path() const { return path_; }
    bool symbol(const std::string & name, uint32_t * v) const;
    std::string symbolize(uint32_t pc) const;   // "func+0x1c (file.cpp:12)" or 0x...
    std::string line(uint32_t pc) const;        // "file.cpp:12" or ""
    std::string where(uint32_t pc) const;       // "file.cpp:12 in func+0x1c" (DB §5.6: the push's line and function)
    int kernel_index(const std::string & name) const;   // -1 if absent
    std::string kernel_name(uint32_t index) const;
    uint32_t n_kernels() const;
    uint64_t image_id() const;
    // The bytes the image loads at [a, a + n) (PT_LOAD file data), false outside them.
    bool bytes_at(uint32_t a, uint8_t * out, size_t n) const;
    // qSymbol: values GDB gave for the contract symbols when no ELF is loaded.
    std::map<std::string, uint32_t> given;

private:
    const tsr_elf *   elf_ = nullptr;   // image_elf(img_)
    const tsr_image * img_ = nullptr;
    tsr_image *       own_ = nullptr;   // opened by load_path (tsr_image_open)
    std::string path_;
};

// ---------------------------------------------------------------------------------------------
// The RISC-V debug target (DB §3-§4)
// ---------------------------------------------------------------------------------------------
struct stop_info {
    int         signal = 5;      // 5 SIGTRAP, 2 SIGINT, 4 SIGILL, 6 SIGABRT, 7 SIGBUS, 11 SIGSEGV
    std::string reason;          // "swbreak", "hwbreak", "watch", "rwatch", "awatch", ""
    uint32_t    watch_addr = 0;
    std::string note;            // the O packet (DB §4.7): "launch rt_nop seq 3", "halted at boot", ...
    uint32_t    pc = 0;
    uint32_t    dcsr_cause = 0;
};

class target {
public:
    struct options {
        bool     hold_on_halt = true;        // CTRL.HOLD_ON_HALT while attached (monitor hold)
        bool     spm_write_unsafe = false;   // monitor spm-write unsafe
        uint64_t halt_timeout_ns = 5000000;  // DB §3.4: the server waits 5 ms
    };
    target(tsr_transport * t, image_info * img, const options & o);
    target(tsr_transport * t, image_info * img) : target(t, img, options()) {}
    ~target();

    tsr_transport * transport() const { return t_; }
    dmi & dm() { return dm_; }
    image_info * image() { return img_; }
    options & opts() { return o_; }

    // DB §4.2 ATTACH: DMI_EN, RV_HALT_EN, dmactive, the identity checks, halt, dcsr.ebreakm /
    // stopcount, HOLD_ON_HALT, STOP_HALT_RV, __tsr_dbg.attached when the build id matches. The
    // hart may already be halted (halt-on-boot): then it stays. 0, else an E code (why filled).
    int attach(std::string * why);
    // DB §4.2 DETACH: breakpoints, triggers, watchpoints and the session's tile breakpoints
    // removed, tile stops released, ebreakm off, __tsr_dbg.attached = 0, RV_HALT_EN off,
    // HOLD_ON_HALT restored, the hart resumed unless leave_halted, DMI_EN off.
    int detach(bool leave_halted);
    bool attached() const { return attached_; }
    bool halted() const { return halted_; }
    bool degraded() const { return degraded_; }   // STATUS.DMI_TIMEOUT seen: core-domain reads only
    const std::vector<std::string> & attach_notes() const { return notes_; }

    // Run control. resume / start_step leave the hart running (halted() false); poll_halted
    // checks for a stop without DMI (STATUS.RV_HALTED), then stop() classifies it (DB §4.7).
    int resume();
    int start_step();
    int interrupt();                 // Ctrl-C: haltreq (the stop is T02)
    bool poll_halted();
    int stop(stop_info * si);        // the hart is halted: classify, expedite, step cleanup
    int step(stop_info * si, uint64_t timeout_ns = 5000000);   // synchronous single step
    void note_boot_halt() { boot_halt_pending_ = true; }      // HALT_ON_BOOT requested (the next stop)
    int rv_sample(std::string * text);                         // DB §6.2 RV_SAMPLE, decoded

    // Registers (DB §4.4 numbering): 0-31 x0-x31, 32 pc (dpc), 65 + csr, 4161 priv.
    int read_reg(uint32_t regnum, uint32_t * v);
    int write_reg(uint32_t regnum, uint32_t v);
    int read_gprs(uint32_t v[33]);   // x0..x31, pc (the 'g' packet)
    bool crash_view = false;         // monitor crash-view on: GPRs / pc from __tsr_crash

    // Memory (DB §4.5, §3.6).
    int read_mem(uint32_t a, uint8_t * b, size_t n);
    int write_mem(uint32_t a, const uint8_t * b, size_t n);

    // Breakpoints and watchpoints (DB §4.6): type 0 software (kind 2 / 4), 1 hardware, 2 write,
    // 3 read, 4 access watchpoint (kind = length).
    int insert_bp(int type, uint32_t addr, uint32_t kind);
    int remove_bp(int type, uint32_t addr, uint32_t kind);

    // DB §4.4, §4.5: the target description and the memory map (sizes from RV_CFG and CFG1).
    std::string target_xml() const;
    std::string memory_map_xml() const;
    uint32_t itcm_bytes() const { return itcm_bytes_; }
    uint32_t dtcm_bytes() const { return dtcm_bytes_; }
    uint32_t spm_bytes() const { return spm_bytes_; }

    // The contract symbols (ELF, else qSymbol values; DB §6.3).
    bool sym(const char * name, uint32_t * v) const;
    // __tsr_dbg words (SBA): attached / launch_break (DB §6.3, §3.8).
    int set_dbg_word(uint32_t word, uint32_t v);
    int get_dbg_word(uint32_t word, uint32_t * v);

    uint32_t csr(uint32_t off) { return tsr_transport_csr_read(t_, off); }
    void csr_write(uint32_t off, uint32_t v) { tsr_transport_csr_write(t_, off, v); }
    std::string last_error;          // text of the last E reply (also on stderr)

private:
    int fail(int e, const std::string & why);
    int ensure_halted();
    int check_dmi_timeout(int e);
    int raw_write(uint32_t a, const uint8_t * b, size_t n);   // routing without the policy
    int load_write(uint32_t a, const uint8_t * b, size_t n);  // EX LOAD_* (RV_RUN = 0)
    int prepare_resume();
    int read_crash_reg(uint32_t regnum, uint32_t * v);
    bool in_itcm(uint32_t a, size_t n) const { return a < itcm_bytes_ && n <= itcm_bytes_ - a; }
    bool in_dtcm(uint32_t a, size_t n) const;
    int rvwp_apply();

    struct swbp { uint32_t addr, kind; uint8_t orig[4]; };
    struct hwbp { bool used = false; uint32_t addr = 0; };
    struct wp { bool used = false; int type = 0; uint32_t addr = 0, len = 0; int tbp = -1; };

    tsr_transport * t_;
    image_info *    img_;
    options         o_;
    dmi             dm_;
    bool attached_ = false, halted_ = false, degraded_ = false, stepping_ = false;
    bool itcm_dirty_ = false, interrupted_ = false, boot_halt_pending_ = false;
    bool skip_ebreak_ = false;       // dpc at a firmware ebreak (launch halt, others): step past it
    uint32_t saved_ctrl_hold_ = 0, saved_dbg_cfg_ = 0;
    uint32_t itcm_bytes_ = 0, dtcm_bytes_ = 0, spm_bytes_ = 0;
    std::map<uint32_t, swbp> sw_;
    hwbp hw_[2];
    wp   wp_[2];
    uint32_t session_tbps_ = 0;      // tile breakpoints this session set (cleared at detach)
    std::vector<std::string> notes_;
    friend class monitor;
};

// ---------------------------------------------------------------------------------------------
// Tile-level debug and probes (DB §5, §7), over the CSR window
// ---------------------------------------------------------------------------------------------
const char * unit_name(uint32_t u);          // DMA QACT MVM VU ATT SYNC
const char * op_name(uint32_t opcode);       // "LD", "MVM", ... "op 0x05"
std::string  op_disasm(const uint32_t w[8]); // one line: name, sub / flags, the operand words

struct hist_entry {
    bool present = false;
    uint32_t info = 0, unit = 0, code = 0, rec = 0, tbps = 0, stop_point = 0;
    uint32_t pidx = 0, ticket = 0, dep[2] = { 0, 0 }, time = 0, pc = 0;
    uint32_t w[8] = {};
};
void        hist_read(tsr_transport * t, uint32_t pidx, hist_entry * e);   // 1 W + 17 R
std::string hist_status(tsr_transport * t, const hist_entry & e);          // done / running / queued (+ code)
std::string hist_line(tsr_transport * t, const hist_entry & e, const image_info * img);
// The last n recorded entries (DBG_HIST_HEAD backward), oldest first.
std::vector<hist_entry> hist_last(tsr_transport * t, uint32_t n);
// SW §5.2 push.bin records of the given entries (DB §5.6 --export).
bool hist_export(const std::vector<hist_entry> & es, const std::string & path, std::string * why);

std::string tile_show(tsr_transport * t);
std::string units_text(tsr_transport * t);          // QSTATUS: EX Q_* and DB's unit probes
std::string perf_text(tsr_transport * t);
std::string sreg_text(tsr_transport * t);
std::string status_text(tsr_transport * t, const image_info * img);   // EX / DBG decode + §7.2 verdict
// QSTEP n entries (DB §5.2): waits for DRAINED, then STEP_BUSY = 0 of each. 0 or an E code.
int tile_step(tsr_transport * t, uint32_t n, uint64_t timeout_ns, std::string * out);
// tile break spec (DB §9.3): returns the TBP index or -1 (why filled).
int tile_break(tsr_transport * t, const std::vector<std::string> & args, const image_info * img, std::string * why);

// ---------------------------------------------------------------------------------------------
// Trace writer (DB §8.3) and the timeline (DB §8.6)
// ---------------------------------------------------------------------------------------------
struct trw_ring {
    uint32_t ch = TSR_TRW_CH;
    uint64_t off = 0;           // byte offset in the channel, 1 KiB aligned (16 beats)
    uint32_t lines = 0;         // 64-B lines, a power of two >= 16
    bool     stop_full = false; // TRW_STOP_FULL: drop when full instead of overwriting
    uint32_t mask = TSR_CSR_DBG_TRW_MASK_RESET;
};
struct trw_rec {
    uint32_t w[8];
    uint32_t kind, unit, code, ran, rec, tbp, opcode, w0_hi;
    uint32_t a1, a2;
    uint64_t time;
    uint32_t run, qwait, hwait;
};
void trw_decode(const uint8_t raw[32], trw_rec * r);
// The default ring: the top of the ch0 IO region (EX §11), `mib` MiB (TSR_PROF_TRW_MIB).
trw_ring trw_default_ring(uint32_t mib);
int  trw_start(tsr_transport * t, const trw_ring & r, std::string * why);   // RPTR := WPTR, TRW_EN
int  trw_stop(tsr_transport * t);
// TRW_FLUSH, then the records [RPTR, WPTR) from DRAM (the last `capacity` when the ring wrapped
// in overwrite mode), RPTR := WPTR. *lost = records overwritten before they were read.
int  trw_drain(tsr_transport * t, const trw_ring & r, std::vector<trw_rec> * out, uint64_t * lost, std::string * why);
bool trw_save_raw(const std::vector<trw_rec> & recs, const std::string & path);
bool trw_load_raw(const std::string & path, std::vector<trw_rec> * out, std::string * why);

// PERF over one launch (its deltas) and its MTIME span: one sample of every channel counter track.
struct perf_window {
    uint64_t t_begin = 0, t_end = 0;        // MTIME (core cycles)
    uint64_t perf[64] = {};
};
struct timeline_opts {
    double core_hz = 276.3e6;               // MTIME core cycles -> µs
    const image_info * img = nullptr;       // kernel names of LAUNCH markers
    uint32_t num_ch = 0;
    std::vector<perf_window> windows;       // channel counter tracks (GB/s, idle %), one sample per launch
};
// Perfetto / Chrome trace-event JSON (DB §8.6): process "Tesseract0", one thread per unit
// (slices = RUN, args QWAIT / HWAIT, a preceding "waiting" slice), a RISC-V thread (launch and
// node slices from markers, other markers as instants), stops and RV halts as instants, DROP
// records as instants with their count, channel counter tracks from PERF.
std::string trw_chrome_json(const std::vector<trw_rec> & recs, const timeline_opts & o);

// DB §8.7 self-checks on the records of a launch window against the PERF deltas of that window.
struct trw_check_result {
    bool ok = true;
    std::vector<std::string> lines;   // one per check, "ok: ..." / "FAIL: ..."
};
trw_check_result trw_check(const std::vector<trw_rec> & recs, const uint64_t perf_delta[64], uint32_t num_ch, uint64_t trw_drops);

// ---------------------------------------------------------------------------------------------
// Monitor commands (DB §4.8, §9.3)
// ---------------------------------------------------------------------------------------------
class monitor {
public:
    // standalone: the session owns the device (launch / ping, load, reset are allowed).
    monitor(target & tg, bool standalone) : tg_(tg), standalone_(standalone) {}
    // Runs one command line; output appended to *out (lines). 0, or an E code (message in *out).
    int run(const std::string & line, std::string * out);
    // Launches pushed by `launch` / `ping` and not answered yet (the server polls them while the
    // hart runs and reports their answers).
    std::vector<uint32_t> pending;
    std::vector<tsr_launch> deferred;   // pushed while a boot waits for HELLO: launched after it
    std::string poll_launches();     // answers that arrived, as text lines ("" if none); pushes the deferred
    // Asynchronous boot (--load / --halt-on-boot / monitor load): runs tsr_transport_boot on a
    // helper thread (the HELLO wait is debug-aware), so the session serves GDB meanwhile.
    int  boot_async(bool halt_on_boot, std::string * why);
    bool boot_running() const;
    int  boot_join(std::string * why);   // waits; the boot's rc
    ~monitor();

    trw_ring trw;                     // the ring of `trw start`
    bool trw_on = false;

private:
    int cmd_tile(const std::vector<std::string> & a, std::string * out);
    int cmd_rv(const std::vector<std::string> & a, std::string * out);
    int cmd_trw(const std::vector<std::string> & a, std::string * out);
    int cmd_launch(const std::vector<std::string> & a, bool ping, std::string * out);
    int cmd_reset(bool halt, std::string * out);
    int cmd_dump(const std::string & dir, std::string * out);
    int cmd_recover(int level, std::string * out);
    int cmd_mem(const std::vector<std::string> & a, bool spm, std::string * out);
    target & tg_;
    bool standalone_;
    struct boot_state;
    std::shared_ptr<boot_state> boot_;
};

// Splits a command line at blanks (no quoting).
std::vector<std::string> split_words(const std::string & s);
// Parses an unsigned number (decimal, 0x hex). false if malformed.
bool parse_u32(const std::string & s, uint32_t * v);

// ---------------------------------------------------------------------------------------------
// GDB RSP server (DB §4.2, §4.3)
// ---------------------------------------------------------------------------------------------
struct server_options {
    std::string listen = "tcp:0";   // tcp:<port> (0: any free port), unix:<path>, stdio
    std::string port_file;          // the bound TCP port written here (tests)
    bool        detach_halted = false;
    bool        sessions_forever = false;   // embedded: accept again after a session ends
    bool        verbose = false;            // packets on stderr
    // Embedded service: the session's device work runs under this mutex (shared with the CLI
    // socket), and tick() runs under it at every poll (deferred work of the boot hooks).
    std::mutex *          serial_mu = nullptr;
    std::function<void()> tick;
};
class rsp_server {
public:
    rsp_server(target & tg, monitor & mon, const server_options & o);
    ~rsp_server();
    int  open_listener(std::string * why);   // binds; prints "listening on ..."
    int  port() const { return port_; }
    // Accepts one connection (or uses stdio) and runs the session until D, k or EOF. 0 or < 0.
    int  serve(std::string * why);
    void request_stop();                     // another thread: end the session and serve()
    bool kill_requested() const { return killed_; }

private:
    struct impl;
    std::unique_ptr<impl> p_;
    target & tg_;
    monitor & mon_;
    server_options o_;
    int lfd_ = -1, port_ = 0;
    bool killed_ = false;
};

// ---------------------------------------------------------------------------------------------
// Embedded service (DB-17, §9.5)
// ---------------------------------------------------------------------------------------------
// Registers the service with the transport (tsr_transport_set_debug_service). Idempotent.
void embed_install();

} // namespace tsr::dbg

extern "C" void tsr_dbg_embed_install(void);

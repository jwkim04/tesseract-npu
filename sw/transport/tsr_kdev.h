// tsr_kdev.h — the device-access layer (kdev) and the protocol layer on top of it
// (docs/design/01-execution-model.md "EX" §10.2; docs/design/06-software-verification.md "SW" §9.1;
// docs/design/05-debug.md "DB" §4.1, §7.3, §9.2).
//
// Every kind has two layers:
//   kdev      raw access to one Tesseract kernel: 32-bit s_axi_control registers (the adapter's
//             and the core CSR window at 0x1000 + off), per-channel device memory, and the time
//             base the protocol counts in. Implementations: "sim" (the simulator in-process, core
//             CSR port only, no adapter; tsr_kdev_sim.cpp), "mock" (the sim kdev inside the
//             ap_ctrl_none adapter register model, with fault injection; tsr_kdev_mock.cpp),
//             "vkernel" (the Verilated kernel in a server process; tsr_kdev_vkernel.cpp), "xrt"
//             (xrt::ip plus per-channel buffers; tsr_transport_xrt.cpp), and kinds registered at
//             run time (register_kdev).
//   protocol  one implementation for every kind (tsr_kdev.cpp, SW-18): open (identity,
//             ADAPTER_STATUS, SOFT_RESET, bind), boot (EX §3.3-§3.5: requirements, VERIFY, cold or
//             warm load, HELLO), launch (credit, shadowed LQ_W writes, LQ_PUSH), wait (LCQ
//             consumption, events, the poll policy, the log, the debug-aware timeout and DB's
//             recovery ladder, TRAP handling), cancel, stop, soft reset, log drain.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tsr_transport.h"

struct tsr_sim;

namespace tsr {

// ---------------------------------------------------------------------------
// Kernel contract: s_axi_control map (EX §9.2; byte offsets, 32-bit registers, 13-bit space).
// 0x000-0x03C are reserved and read 0.
// ---------------------------------------------------------------------------
enum : uint32_t {
    KREG_CH_BASE0        = TSR_SAXI_CH_BASE_LO_BASE,   // ch c base (u64, 4 KiB aligned): lo at 0x040 + 8c, hi at +4
    KREG_CHB_BASE0       = 0x100,                      // args chb<c> of the read-only second ports at 0x100 + 8c: no register behind them
    KREG_ADAPTER_CTRL    = TSR_SAXI_ADAPTER_CTRL,      // bit0 MEM_EN, bit1 DUAL_DIS
    KREG_ADAPTER_STATUS  = TSR_SAXI_ADAPTER_STATUS,    // R / W1C sticky
    KREG_ADAPTER_VERSION = TSR_SAXI_ADAPTER_VERSION,   // {16'h0001, 8'h00, NUM_CH}
    KREG_CSR_WINDOW      = TSR_SAXI_CORE_BASE,         // core CSR off at 0x1000 + off, off < 0x1000
    KREG_SPACE           = 0x2000,                     // s_axi_control range the kernel declares (kernel.xml)
};

// ADAPTER_STATUS (EX §9.2): sticky error bits 0-5 + bits [31:16] saturating count of error responses.
enum : uint32_t {
    AST_RRESP       = TSR_ADAPTER_STATUS_RRESP_ERR,
    AST_BRESP       = TSR_ADAPTER_STATUS_BRESP_ERR,
    AST_ORPHAN      = TSR_ADAPTER_STATUS_ORPHAN_RESP,
    AST_BASE_ALIGN  = TSR_ADAPTER_STATUS_BASE_MISALIGNED,   // a base not 4 KiB aligned at the MEM_EN edge
    AST_REQ_HELD    = TSR_ADAPTER_STATUS_REQ_HELD,          // a core memory request held while MEM_EN = 0
    AST_BASE_LOCKED = TSR_ADAPTER_STATUS_BASE_LOCKED,       // a base write while MEM_EN = 1 (ignored)
    AST_PLAT_ERR    = AST_RRESP | AST_BRESP | AST_ORPHAN,   // these and a count != 0 drive plat_err_i (EX §9.3)
};
inline uint32_t adapter_status_count(uint32_t v) { return v >> 16; }
inline bool adapter_plat_err(uint32_t v) { return (v & AST_PLAT_ERR) != 0 || adapter_status_count(v) != 0; }
std::string adapter_status_str(uint32_t v);   // "RRESP error, 3 error responses" etc.; "ok" for 0

// Kernel CH_BYTES of the U200 kernels (contract): the core range-checks DRAM operands against
// this, independent of the size of the buffer bound to the channel.
constexpr uint64_t KERNEL_CH_BYTES = 16ull << 30;
constexpr uint32_t KERNEL_MAX_CH   = TSR_SAXI_CH_BASE_LO_N;   // 16 channel-base slots (0x040..0x0BC)

// Nominal core clock of the release image (U200 k4, docs/platform-u200.md §9: core 200 MHz, RISC-V
// 100 MHz, RV_FREQ ratio 0.5): converts core cycles (DURATION, the simulator's MTIME) into the
// transport's time base. Only the poll policy and the simulator kinds' timeouts use it; no result
// depends on it. On xrt, tsr_perf_summary replaces it with the clock the platform reports and, once
// the session is long enough, with the clock fitted from DBG_MTIME (kdev::core_clock_hz). Tools that
// report device time use that value and fall back to this.
constexpr double CORE_HZ_NOMINAL = 200e6;

// ---------------------------------------------------------------------------
// kdev: one kernel instance
// ---------------------------------------------------------------------------
class kdev {
public:
    virtual ~kdev() = default;
    virtual const char * name() const = 0;               // transport name ("xrt", "mock", ...)
    virtual uint32_t num_ch() const = 0;                 // m_axi_ch<c> ports with a buffer bound
    virtual uint64_t ch_bytes(uint32_t ch) const = 0;    // usable bytes of the buffer of channel ch (real memory)
    virtual uint64_t ch_addr(uint32_t ch) const = 0;     // its device address (what ch<c>_base gets)
    // The size of the buffer bound to channel ch. It can be more than ch_bytes(ch): an Alveo shell
    // reports 16 GB per DDR bank in its memory topology, and a buffer of that size on a card with
    // 8 GiB DIMMs wraps (offset + 8 GiB reads back what was written at offset). Default: ch_bytes.
    virtual uint64_t buf_bytes(uint32_t ch) const { return ch_bytes(ch); }
    // Finds how much of each buffer is real memory and lowers ch_bytes() to it. The protocol layer
    // calls it once at open, before it reads ch_bytes(). probe true (owner mode): measures the
    // capacity by writing and restoring probe patterns (detect_ch_capacity). false (TSR_ATTACH,
    // which must not write): writes nothing, uses what an earlier owner recorded, else keeps
    // ch_bytes(). 0 on success, else a message was printed. Default: the buffers are exact (the
    // simulated kinds).
    virtual int detect_capacity(bool probe) { (void) probe; return 0; }
    virtual std::string describe() const { return std::string(); }   // one line per item, for tools
    // false: the core CSR port only, no adapter registers (the "sim" kind, EX §10.2): the protocol
    // skips ADAPTER_STATUS and the bind; reg_* outside the CSR window fail.
    virtual bool has_adapter() const { return true; }

    // s_axi_control access: off a multiple of 4 below KREG_SPACE. 0 on success (message on stderr
    // otherwise).
    virtual int reg_read(uint32_t off, uint32_t * val) = 0;
    virtual int reg_write(uint32_t off, uint32_t val) = 0;

    // Channel memory, byte offsets inside the channel buffer, any alignment and size; the caller
    // has checked off + n <= ch_bytes(ch). 0 on success.
    virtual int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) = 0;
    virtual int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) = 0;
    // mem_read for bytes that may never have been written. Memory with ECC (the DDR4 of an Alveo
    // shell) answers a read with an error where a cell holds no valid ECC word, and it does so until
    // the cell is written. A failure here is taken as that answer: nothing is printed and the kdev
    // stays as it was (no change of the transfer path). Default: mem_read.
    virtual int mem_try_read(uint32_t ch, uint64_t off, void * dst, size_t n) { return mem_read(ch, off, dst, n); }

    // The time base of the protocol's timeouts and poll policy, in ns (monotonic). Default: the
    // host clock. The simulator kinds count device time (MTIME at CORE_HZ_NOMINAL).
    virtual uint64_t now_ns();
    // SW §9.1 idle(), EX §10.2's advance hint: called by the wait loops between polls (with the
    // transport mutex released) for about interval_ns of the time base. Default: sleep. sim:
    // tsr_sim_advance; vkernel: run the server's clocks. poll_iter counts the polls of this wait.
    virtual void idle(uint32_t poll_iter, uint64_t interval_ns);
    // The in-process simulator behind the kind (sim, mock), else nullptr: tests, the selftest's
    // shadow, the DMI helpers of DB §9.2.
    virtual tsr_sim * sim() { return nullptr; }
    // true for the simulated kinds (sim, mock, vkernel): now_ns() is device time and idle() drives
    // the model, so the protocol calls it with its mutex held; false (xrt): host time, and idle()
    // sleeps with the mutex released.
    virtual bool simulated() const { return false; }
    // The core clock of the loaded image in Hz as the platform reports it (xrt: clock_freqs of the
    // card in sysfs), 0 if it does not know. tsr_perf_summary takes it before a fit from DBG_MTIME
    // is possible, and CORE_HZ_NOMINAL when it is 0.
    virtual double core_clock_hz() const { return 0.0; }
};

// ---------------------------------------------------------------------------
// Chunked / aligned transfers: turns byte-granular requests into raw transfers that obey
// the rules of a DMA path (device offset, size and host address multiples of `align`, at most
// `max_xfer` bytes each). Unaligned edges are read-modify-written through an aligned bounce
// buffer of max_xfer bytes (host RAM use is bounded by it, never by the channel size).
// ---------------------------------------------------------------------------
struct dma_rules {
    size_t align         = 1;           // power of two, 1..4096
    size_t max_xfer      = 64u << 20;   // multiple of align
    bool   always_bounce = false;       // copy through the bounce buffer even when aligned
};

class dma_raw {
public:
    virtual ~dma_raw() = default;
    virtual int raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) = 0;
    virtual int raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) = 0;
};

class dma_engine {
public:
    dma_engine(dma_raw & raw, const dma_rules & rules);
    ~dma_engine();
    dma_engine(const dma_engine &) = delete;
    dma_engine & operator=(const dma_engine &) = delete;

    int write(uint32_t ch, uint64_t off, const void * src, size_t n);
    int read(uint32_t ch, uint64_t off, void * dst, size_t n);
    const dma_rules & rules() const { return r_; }

    // Statistics (tests, selftest).
    uint64_t raw_calls     = 0;
    uint64_t bounce_copies = 0;   // pieces that went through the bounce buffer
    uint64_t rmw_reads     = 0;   // edge units read for read-modify-write

private:
    uint8_t * bounce();
    dma_raw & raw_;
    dma_rules r_;
    uint8_t * buf_ = nullptr;
};

// Valid dma_rules? Empty string if so, else the reason.
std::string dma_rules_check(const dma_rules & r);

// ---------------------------------------------------------------------------
// Channel capacity: how much of a buffer is real memory
// ---------------------------------------------------------------------------
// Accessors of one channel buffer: byte offset, host pointer, size. 0 on success.
using ch_write_fn = std::function<int(uint64_t off, const void * src, size_t n)>;
using ch_read_fn  = std::function<int(uint64_t off, void * dst, size_t n)>;

// The capacity of a channel buffer of `size` bytes whose memory may be smaller than the buffer and
// wrap (alias). The capacity is a power of two (a DIMM address range). The test: from the largest
// power of two P below the current size down, write two distinct 64-byte patterns at A and at
// A + P (A = size - P - 64, so the pair ends at the last block of the range) and read both back.
// If A + P returns the pattern of A, memory repeats every P bytes, the capacity is P, and the
// test goes on inside it. The first P that does not wrap ends it, and the capacity is the size
// then. Every byte it touches is saved first and restored afterward, so the buffer's content is
// unchanged. A pair that does not read back as written (either pattern lost) counts as a wrap:
// the result can only be smaller than the truth, never larger. It never goes below min_bytes
// (the smallest size still tested). 0 on success with *bytes set, -1 if an access failed (a
// message on stderr, *bytes untouched).
// try_rd, when given, saves the blocks instead of rd: a read of bytes that may never have been
// written (kdev::mem_try_read). A block it cannot read is taken as never written. The test writes
// it like any other, and it holds zeros afterward. *blank, when given, gets the offsets of those
// blocks inside the capacity found, in ascending order (on success only). Without try_rd a block
// that cannot be read fails the probe.
int detect_ch_capacity(uint64_t size, uint64_t min_bytes, const ch_write_fn & wr, const ch_read_fn & rd, uint64_t * bytes,
                       const ch_read_fn & try_rd = nullptr, std::vector<uint64_t> * blank = nullptr);

// ---------------------------------------------------------------------------
// The protocol layer as a transport
// ---------------------------------------------------------------------------
// Opens the protocol layer on dev (takes ownership): identity, ADAPTER_STATUS, SOFT_RESET and bind
// (unless TSR_ATTACH=1), exactly as tsr_transport_open. NULL on failure (message on stderr).
tsr_transport * transport_open_kdev(std::unique_ptr<kdev> dev);
// Closes the protocol layer without any device access (no log drain, no reset) and returns its
// kdev: what a crashed process leaves behind (tests of the stale core at open, warm restarts
// across "processes").
std::unique_ptr<kdev> transport_detach_kdev(tsr_transport * t);

// The kdev behind a transport (every kind).
kdev * transport_kdev(tsr_transport * t);
// The in-process simulator behind a transport (sim, mock), else nullptr.
tsr_sim * transport_sim(tsr_transport * t);

// ADAPTER_STATUS as the protocol layer saw it. It reads the register at open and clears it (W1C)
// when it is not 0, so reading the register afterward shows only what happened since; tools
// check this record instead (all zero without an adapter). last_* are from the last time a launch
// answered PLAT_ERR or STATUS.PLAT_ERR was seen (read, reported and cleared then, EX §8.6).
struct adapter_log {
    uint32_t at_open   = 0;   // found at open: left by an earlier process (cleared then)
    uint32_t plat_errs = 0;   // PLAT_ERR events handled since open
    uint32_t last      = 0;   // ADAPTER_STATUS at the last one
};
adapter_log transport_adapter_log(tsr_transport * t);

// Read one 64-bit PERF_* counter through the window (LO first: the core snapshots on LO).
uint64_t transport_perf(tsr_transport * t, uint32_t idx);

// What the protocol layer did, for tests and tools (tsr-selftest).
struct boot_info {
    bool     warm = false;          // no ITCM segment had to be written (EX §3.5)
    uint32_t itcm_loaded = 0;       // ITCM segments cold-loaded (VERIFY mismatch)
    uint32_t dtcm_loaded = 0;       // DTCM segments (always reloaded)
    uint32_t words_written = 0;     // LOAD_DATA writes
    uint64_t time_ns = 0;           // the transport's time base: from the start of boot to HELLO
    uint64_t wall_ns = 0;           // host time, the same span
    uint32_t hello_cycles = 0;      // HELLO r7: core cycles from crt0's entry to the post
};
boot_info transport_boot_info(tsr_transport * t);

// Recovery and diagnosis counters (DB §7.3), since open.
struct recovery_log {
    uint32_t timeouts = 0;          // waits that timed out (ladder entered)
    uint32_t level2 = 0;            // LQ_CANCEL issued by the ladder
    uint32_t level2_ok = 0;         // ... and every outstanding launch answered within TSR_ABORT_MS
    uint32_t level3 = 0;            // SOFT_RESET + warm restart by the ladder
    uint32_t traps = 0;             // TRAP records handled
    uint32_t restarts = 0;          // warm restarts after traps and level 3
    uint32_t dumps = 0;             // hang dumps written (TSR_DUMP_DIR)
    uint32_t lost = 0;              // launches flushed without an answer
    bool     failed = false;        // the device is FAILED (card reset needed, no clk_rv, ...)
    std::string last_trap;          // decoded report of the last TRAP record
    std::string last_dump_dir;      // directory of the last hang dump
};
recovery_log transport_recovery_log(tsr_transport * t);

// Test hooks. full_records: read all 8 words of every LCQ record instead of EX §4.8's minimal
// reads (an OK DONE / PONG answering the oldest launch: r0, plus r2 unless the kernel's RET is 0 on
// success; r7 DURATION is not read); the protocol tests compare both. timeouts override the
// environment's TSR_*_MS for this transport (ms; 0 keeps the value). wm_refresh: launches after
// which a valid LQ_CANCEL / LQ_STOP watermark is rewritten below the oldest unanswered launch
// (default 2^30, EX §4.9: it would cover new launches 2^31 on), so tests see the rewrite.
void transport_set_full_records(tsr_transport * t, bool on);
void transport_set_timeouts(tsr_transport * t, uint64_t launch_ms, uint64_t abort_ms, uint64_t boot_ms, uint64_t reset_ms);
void transport_set_wm_refresh(tsr_transport * t, uint32_t launches);

// Extra kdev kinds for TSR_DEVICE / tsr_transport_open. The factory returns nullptr on failure
// after printing why. Registering a built-in kind ("sim", "xrt", "mock", "vkernel") is refused.
using kdev_factory = std::function<std::unique_ptr<kdev>()>;
bool register_kdev(const std::string & kind, kdev_factory f);

// Built-in kdevs (each reads its own environment, see the .cpp files).
std::unique_ptr<kdev> open_sim_kdev();       // tsr_kdev_sim.cpp
std::unique_ptr<kdev> open_mock_kdev();      // tsr_kdev_mock.cpp
std::unique_ptr<kdev> open_vkernel_kdev();   // tsr_kdev_vkernel.cpp
std::unique_ptr<kdev> open_xrt_kdev();       // nullptr + message when built without XRT

// Defaults of the xrt kdev's environment that depend on XRT's emulation mode (docs/xrt-host.md
// §9). With XCL_EMULATION_MODE=hw_emu: TSR_XRT_CH_GIB=3, TSR_XRT_DMA=staged and
// TSR_XRT_DMA_ALIGN=4096, each one only if the environment leaves it unset or empty (the Vitis
// 2024.2 hw_emu simulator crashes on a channel buffer of 4 GiB or more, and drops the last page
// of a DMA that starts off a 4 KiB boundary); none in any other mode. open_xrt_kdev applies them
// and names them in one line on stderr. `env` is a getenv-like lookup (empty: getenv). In
// tsr_kdev.cpp, so that it is tested without XRT.
struct env_default {
    const char * name;
    const char * value;
};
std::vector<env_default> xrt_env_defaults(const std::function<const char *(const char *)> & env = {});

// Environment helpers shared by the kdevs: unsigned decimal value in [lo, hi]. 0 if unset or
// empty, 1 if set and valid (*out), -1 if malformed (message "tsr_transport(<who>): ..." printed).
int env_u64(const char * who, const char * name, uint64_t lo, uint64_t hi, uint64_t * out);

} // namespace tsr

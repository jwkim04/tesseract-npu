// tsr_kdev.h — the device-access layer (kdev) and the kernel-protocol transport on top of it
// (docs/xrt-host.md; kernel contract: docs/xrt-host.md §2, M3 part C).
//
// Every hardware-like transport has two layers:
//   kdev      raw access to one Tesseract kernel: 32-bit s_axi_control registers, per-channel
//             device memory, start / wait. Implementations: "xrt" (Alveo through XRT,
//             tsr_transport_xrt.cpp), "mock" (adapter register model around the ISA simulator,
//             tsr_kdev_mock.cpp), "vkernel" (the Verilated kernel of tb/platform in a server
//             process, tsr_kdev_vkernel.cpp), and kinds registered at run time (register_kdev).
//   protocol  the kernel contract on top of any kdev (tsr_kdev.cpp): configuration from the CSR
//             window, exec = args + start + wait with a timeout (then CTRL.ABORT through the
//             window), a stale completion detected and waited out (settle), STATUS / ERR_* from
//             the window, ADAPTER_STATUS after every run (adapter_log), range checks.
//             It is an ordinary tsr_transport, so the backend and the tools use it unchanged.
// Not thread-safe: one caller at a time (the backend serialises on its device mutex).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tsr_transport.h"

namespace tsr {

// ---------------------------------------------------------------------------
// Kernel contract: s_axi_control map (byte offsets, 32-bit registers, 13-bit address space)
// ---------------------------------------------------------------------------
enum : uint32_t {
    KREG_AP_CTRL         = 0x000,  // bit0 ap_start, bit1 ap_done (COR), bit2 ap_idle, bit3 ap_ready (COR)
    KREG_GIE             = 0x004,
    KREG_IER             = 0x008,
    KREG_ISR             = 0x00C,  // toggle on write
    KREG_CMD_ADDR_LO     = 0x010,  // cmd_addr (u64, ch0 beat address)
    KREG_CMD_ADDR_HI     = 0x014,
    KREG_CMD_COUNT       = 0x018,
    KREG_ARG0            = 0x020,  // arg i at 0x020 + 4 i, i < 8
    KREG_CH_BASE0        = 0x040,  // ch c base (u64, 4 KiB aligned): lo at 0x040 + 8 c, hi at +4
    KREG_ADAPTER_STATUS  = 0x0F0,  // R / W1C sticky
    KREG_ADAPTER_VERSION = 0x0F4,
    KREG_CSR_WINDOW      = 0x1000, // core CSR off at 0x1000 + off, off < 0x1000
    KREG_SPACE           = 0x2000, // s_axi_control range the kernel must declare (kernel.xml)
};

enum : uint32_t {
    AP_START        = 1u << 0,
    AP_DONE         = 1u << 1,
    AP_IDLE         = 1u << 2,
    AP_READY        = 1u << 3,
    AP_AUTO_RESTART = 1u << 7,  // not supported: reads 0, writes ignored
};

// ADAPTER_STATUS: sticky error bits + bits[31:16] saturating count of error responses.
enum : uint32_t {
    AST_RRESP      = 1u << 0,  // a read response with RRESP != OKAY
    AST_BRESP      = 1u << 1,  // a write response with BRESP != OKAY
    AST_ORPHAN     = 1u << 2,  // a response with no outstanding request for its id
    AST_BASE_ALIGN = 1u << 3,  // a ch<c>_base not 4 KiB aligned at ap_start
};
inline uint32_t adapter_status_count(uint32_t v) { return v >> 16; }
std::string adapter_status_str(uint32_t v);   // "RRESP error, 3 error responses" etc.; "ok" for 0

// Kernel CH_BYTES of the U200 kernels (contract): the core range-checks DRAM operands against
// this, independent of the size of the buffer bound to the channel.
constexpr uint64_t KERNEL_CH_BYTES = 16ull << 30;
constexpr uint32_t KERNEL_MAX_CH   = (KREG_ADAPTER_STATUS - KREG_CH_BASE0) / 8;   // 22 >= TSR_NUM_CH_MAX

// Arguments of one run (the adapter latches them at ap_start).
struct kargs {
    uint64_t cmd_addr  = 0;   // ch0 beat address of the command buffer
    uint32_t cmd_count = 0;
    uint32_t arg[8]    = {};
};

enum class kwait { done, timeout, error };

// ---------------------------------------------------------------------------
// kdev: one kernel instance
// ---------------------------------------------------------------------------
class kdev {
public:
    virtual ~kdev() = default;
    virtual const char * name() const = 0;               // transport name ("xrt", "mock", ...)
    virtual uint32_t num_ch() const = 0;                 // m_axi_ch<c> ports with a buffer bound
    virtual uint64_t ch_bytes(uint32_t ch) const = 0;    // usable bytes of the buffer of channel ch
    virtual uint64_t ch_addr(uint32_t ch) const = 0;     // its device address (what ch<c>_base holds)
    virtual std::string describe() const { return std::string(); }   // one line per item, for tools

    // s_axi_control access: off a multiple of 4 below KREG_SPACE. 0 on success (message on
    // stderr otherwise). The protocol layer uses only ADAPTER_STATUS / ADAPTER_VERSION and the
    // CSR window: AP_CTRL, GIE / IER / ISR and the argument registers belong to start(), wait()
    // and settle() (under XRT the scheduler owns them, and ap_done is clear-on-read).
    virtual int reg_read(uint32_t off, uint32_t * val) = 0;
    virtual int reg_write(uint32_t off, uint32_t val) = 0;

    // Channel memory, byte offsets inside the channel buffer, any alignment and size; the
    // caller has checked off + n <= ch_bytes(ch). 0 on success.
    virtual int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) = 0;
    virtual int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) = 0;

    // Start one run with these arguments (every ch<c>_base = ch_addr(c)), then wait for ap_done.
    // timeout_ms 0 waits forever. After kwait::timeout the run is still outstanding: the caller
    // stops the core (CTRL.ABORT through the window) and calls wait() again.
    virtual int   start(const kargs & a) = 0;
    virtual kwait wait(uint32_t timeout_ms) = 0;

    // Read AP_CTRL while no run of start() / wait() is outstanding (the protocol layer calls it at
    // open and after every completed wait()). ap_done / ap_ready are clear-on-read and accepting a
    // start does not clear them (tsr_xrt_ctrl), so a completion nobody consumed — a run started
    // by an earlier process, e.g. with TSR_XRT_CTRL=regs, that ended after it died, or a run
    // stopped by the open-time CTRL.ABORT — would otherwise make the next xrt::run complete at
    // once: the XRT scheduler starts a compute unit without reading AP_CTRL first and completes
    // the command on the first read that shows ap_done (xocl cu_hls.c, ERT). *ap gets the value
    // (ap_idle = 0 after a completed wait() means that completion was such a stale one).
    virtual int settle(uint32_t * ap) { return reg_read(KREG_AP_CTRL, ap); }
};

// Register-level ap_ctrl_hs control (UG1393) for kdevs that own AP_CTRL (mock, vkernel, and
// xrt with TSR_XRT_CTRL=regs). regs_start reads AP_CTRL (clearing a stale ap_done), requires
// ap_idle, writes cmd_addr / cmd_count / arg0..7 / every ch<c>_base, then sets ap_start.
// regs_wait polls AP_CTRL until ap_done (consuming it), sleeping between polls (backoff to 100 us).
// kds_start / kds_wait do what the XRT scheduler does for an ap_ctrl_hs compute unit (xocl
// cu_hls.c / ERT): write cmd_addr .. every ch<c>_base and AP_CTRL = ap_start without reading
// AP_CTRL first, then complete on the first AP_CTRL read with ap_done; the protocol tests run
// the xrt::run path through them on the mock (they are not used with hardware).
int   regs_start(kdev & d, const kargs & a);
kwait regs_wait(kdev & d, uint32_t timeout_ms);
int   kds_start(kdev & d, const kargs & a);
kwait kds_wait(kdev & d, uint32_t timeout_ms);

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
// The protocol layer as a transport
// ---------------------------------------------------------------------------
// Wrap a kdev into a tsr_transport (takes ownership). Reads the configuration through the CSR
// window and fails (NULL, message on stderr) if ID is not "TSR1", the ISA major differs,
// CFG0.NUM_CH differs from dev->num_ch(), a channel buffer is not 4 KiB aligned, or the core
// is BUSY and does not stop after CTRL.ABORT. Environment (read here, for every kdev kind):
//   TSR_XRT_TIMEOUT_MS  exec timeout, default 60000, 0 = none; on timeout the core is stopped
//                       with CTRL.ABORT and PC / STATUS are reported
//   TSR_XRT_ABORT_MS    how long to wait for the core to stop after ABORT, default 10000
tsr_transport * transport_open_kdev(std::unique_ptr<kdev> dev);

// The kdev behind a transport opened from one (xrt, mock, registered kinds); nullptr for "sim".
kdev * transport_kdev(tsr_transport * t);

// ADAPTER_STATUS as the protocol layer saw it. It reads the register at open and after every
// run and clears it (W1C) when it is not 0, so reading the register afterwards shows only what
// happened since; tools check this record instead (all zero for "sim").
struct adapter_log {
    uint32_t at_open  = 0;   // found at open: left by an earlier process (cleared then)
    uint32_t last_run = 0;   // after the most recent exec (0 = clean, also when that exec failed early)
    uint32_t sticky   = 0;   // OR of the error bits [15:0] of every run since open (not at_open)
    uint64_t runs     = 0;   // execs that ended with ADAPTER_STATUS != 0
    uint64_t count    = 0;   // sum of their error-response counts (bits [31:16])
};
adapter_log transport_adapter_log(tsr_transport * t);

// Read one 64-bit PERF_* counter through the window (LO first: the core snapshots on LO).
uint64_t transport_perf(tsr_transport * t, uint32_t idx);

// Extra kdev kinds for TSR_DEVICE / tsr_transport_open. The factory returns nullptr on failure
// after printing why. Registering a built-in kind ("sim", "xrt", "mock", "vkernel") is refused.
using kdev_factory = std::function<std::unique_ptr<kdev>()>;
bool register_kdev(const std::string & kind, kdev_factory f);

// Built-in kdevs (each reads its own environment, see the .cpp files and docs/xrt-host.md).
std::unique_ptr<kdev> open_mock_kdev();
std::unique_ptr<kdev> open_vkernel_kdev();   // Verilated kernel server (tsr_kdev_vkernel.cpp)
std::unique_ptr<kdev> open_xrt_kdev();   // nullptr + message when built without XRT

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

} // namespace tsr

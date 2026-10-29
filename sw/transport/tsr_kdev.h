// tsr_kdev.h — the device-access layer (kdev) and the protocol layer on top of it
// (docs/xrt-host.md; v2: docs/v2/01-execution-model.md §9-§10).
//
// Every hardware-like transport has two layers:
//   kdev      raw access to one Tesseract kernel: 32-bit s_axi_control registers and
//             per-channel device memory. Implementations: "xrt" (Alveo through XRT,
//             tsr_transport_xrt.cpp), "mock" (adapter register model around the ISA simulator,
//             tsr_kdev_mock.cpp), "vkernel" (the Verilated kernel of tb/platform in a server
//             process, tsr_kdev_vkernel.cpp), and kinds registered at run time (register_kdev).
//   protocol  the contract on top of any kdev (tsr_kdev.cpp): identity and configuration from
//             the CSR window, ADAPTER_STATUS left at open (adapter_log), range checks, transfers.
//             It is an ordinary tsr_transport, so the backend and the tools use it unchanged.
// v1's run protocol (ap_ctrl_hs start / wait / settle, the argument registers, exec with a
// timeout and CTRL.ABORT) is deleted (X0, EX §10.2); the v2 protocol layer (bind, load and
// VERIFY, boot, launch, wait, cancel, stop, reset, log over every kdev) arrives with P3.1.
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
// Kernel contract: s_axi_control map (byte offsets, 32-bit registers, 13-bit address space).
// 0x000-0x03C (v1's AP_CTRL, GIE, IER, ISR, cmd_addr, cmd_count, arg0-7) are deleted (EX §9.2).
// ---------------------------------------------------------------------------
enum : uint32_t {
    KREG_CH_BASE0        = 0x040,  // ch c base (u64, 4 KiB aligned): lo at 0x040 + 8 c, hi at +4
    KREG_ADAPTER_STATUS  = 0x0F0,  // R / W1C sticky
    KREG_ADAPTER_VERSION = 0x0F4,
    KREG_CSR_WINDOW      = 0x1000, // core CSR off at 0x1000 + off, off < 0x1000
    KREG_SPACE           = 0x2000, // s_axi_control range the kernel must declare (kernel.xml)
};

// ADAPTER_STATUS: sticky error bits + bits[31:16] saturating count of error responses.
enum : uint32_t {
    AST_RRESP      = 1u << 0,  // a read response with RRESP != OKAY
    AST_BRESP      = 1u << 1,  // a write response with BRESP != OKAY
    AST_ORPHAN     = 1u << 2,  // a response with no outstanding request for its id
    AST_BASE_ALIGN = 1u << 3,  // a ch<c>_base not 4 KiB aligned when the bases were latched
};
inline uint32_t adapter_status_count(uint32_t v) { return v >> 16; }
std::string adapter_status_str(uint32_t v);   // "RRESP error, 3 error responses" etc.; "ok" for 0

// Kernel CH_BYTES of the U200 kernels (contract): the core range-checks DRAM operands against
// this, independent of the size of the buffer bound to the channel.
constexpr uint64_t KERNEL_CH_BYTES = 16ull << 30;
constexpr uint32_t KERNEL_MAX_CH   = (KREG_ADAPTER_STATUS - KREG_CH_BASE0) / 8;   // 22 >= TSR_NUM_CH_MAX

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
    // stderr otherwise). The protocol layer uses ADAPTER_STATUS / ADAPTER_VERSION and the CSR
    // window.
    virtual int reg_read(uint32_t off, uint32_t * val) = 0;
    virtual int reg_write(uint32_t off, uint32_t val) = 0;

    // Channel memory, byte offsets inside the channel buffer, any alignment and size; the
    // caller has checked off + n <= ch_bytes(ch). 0 on success.
    virtual int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) = 0;
    virtual int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) = 0;
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
// The protocol layer as a transport
// ---------------------------------------------------------------------------
// Wrap a kdev into a tsr_transport (takes ownership). Reads the configuration through the CSR
// window and fails (NULL, message on stderr) if ID is not "TSR1", the ISA major differs (a v1
// bitstream reports major 1), CFG0.NUM_CH differs from dev->num_ch(), or a channel buffer is not
// 4 KiB aligned.
tsr_transport * transport_open_kdev(std::unique_ptr<kdev> dev);

// The kdev behind a transport opened from one (xrt, mock, registered kinds); nullptr for "sim".
kdev * transport_kdev(tsr_transport * t);

// ADAPTER_STATUS as the protocol layer saw it. It reads the register at open and clears it (W1C)
// when it is not 0, so reading the register afterwards shows only what happened since; tools
// check this record instead (all zero for "sim"). The per-launch record returns with the v2
// launch path (P3.1).
struct adapter_log {
    uint32_t at_open  = 0;   // found at open: left by an earlier process (cleared then)
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

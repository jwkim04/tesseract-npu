// tsr_kdev_mock.h — the "mock" kdev: a model of the platform adapter's s_axi_control semantics
// (kernel contract, docs/xrt-host.md §2) around the ISA simulator as the core, with device
// memory reached through the dma_engine under configurable DMA rules, and fault injection.
// It exists to test the protocol layer and tsr_xrt_selftest without hardware (TSR_DEVICE=mock).
//
// What it models: AP_CTRL (ap_start accepted when idle and cleared, ap_done / ap_ready clear on
// read and NOT cleared by accepting a start, as tsr_xrt_ctrl, ap_idle, auto_restart reads 0),
// GIE / IER bit0 / ISR bit0 (toggle on a write of 1), the argument registers,
// ap_start latching the bases and writing CMD_ADDR / CMD_COUNT / ARG0..7, STATUS W1C DONE|ERR,
// CTRL.START into the core, ADAPTER_STATUS (W1C; a ch<c>_base that is not 4 KiB aligned sets
// bit3 and completes the handshake without starting the core, as tsr_xrt_ctrl does; an aligned
// ch<c>_base that is not the bound buffer's address is reported as an RRESP error, like a DECERR
// on hardware), ADAPTER_VERSION ({16'h0001, 8'h00, NUM_CH} as in tsr_xrt_ctrl) and the CSR
// window. ADAPTER_STATUS W1C as the RTL with all strobes set: a 1 in bit k < 4 clears bit k, a 1
// anywhere in [31:16] clears the whole count. The core runs to completion inside ap_start unless
// a hang is injected. start() / wait() use regs_start / regs_wait, or with ctrl_kds the XRT
// scheduler's sequence (kds_start / kds_wait). Not modelled: AXI timing, interrupts as signals,
// an ap_start queued while running, reset.
#pragma once

#include "tsr_kdev.h"
#include "tsr_sim.h"

#include <vector>

namespace tsr {

struct mock_options {
    uint32_t  num_ch   = 4;
    uint32_t  t_max    = 1;
    uint64_t  ch_bytes = 16ull << 30;
    dma_rules dma;                       // rules the mock's raw DMA enforces (violations fail the transfer)
    uint32_t  adapter_version = 0;       // 0: {16'h0001, 8'h00, NUM_CH} like tsr_xrt_ctrl
    bool      ctrl_kds = false;          // start / wait like the XRT scheduler (kds_start / kds_wait)
};

enum class mock_hang {
    none,          // the core runs to completion inside ap_start
    until_abort,   // the core stays BUSY until CTRL.ABORT is written through the window
    forever,       // the core stays BUSY and ignores ABORT (a stuck memory transaction)
    reads,         // the core finishes by itself after hang_reads more register reads (a long run
                   // that is still going when a stale ap_done is seen); ABORT stops it earlier
};

class mock_kdev final : public kdev, private dma_raw {
public:
    explicit mock_kdev(const mock_options & o);
    ~mock_kdev() override;
    bool ok() const { return sim_ != nullptr; }

    const char * name() const override { return "mock"; }
    uint32_t num_ch() const override { return o_.num_ch; }
    uint64_t ch_bytes(uint32_t) const override { return o_.ch_bytes; }
    uint64_t ch_addr(uint32_t ch) const override { return addr_[ch]; }
    int reg_read(uint32_t off, uint32_t * val) override;
    int reg_write(uint32_t off, uint32_t val) override;
    int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    int start(const kargs & a) override;
    kwait wait(uint32_t timeout_ms) override;
    int settle(uint32_t * ap) override;

    // --- test hooks ---
    tsr_sim *    sim() { return sim_; }
    dma_engine & dma() { return dma_; }
    void set_ch_addr(uint32_t ch, uint64_t a) { addr_[ch] = a; }
    mock_hang    hang = mock_hang::none;           // applies to the next ap_start
    uint32_t     hang_reads = 0;                   // mock_hang::reads: register reads until the core finishes
    uint32_t     inject_status = 0;                // OR-ed into ADAPTER_STATUS when the next run ends
    std::vector<std::pair<uint64_t, uint32_t>> inject_at;   // (n, v): OR v into ADAPTER_STATUS when run n
                                                   // (1-based count of ended runs) ends
    uint32_t     id_override = 0;                  // non-zero: the window's ID reads this
    uint32_t     done_after_polls = 0;             // ap_done becomes visible only after this many AP_CTRL reads
    // Observations.
    uint64_t starts = 0;                           // accepted ap_start
    uint32_t last_cmd_lo = 0, last_cmd_hi = 0, last_count = 0, last_args[8] = {};
    std::vector<uint64_t> last_bases;
    uint64_t ctl_access_outside = 0;               // AP_CTRL / GIE / IER / ISR / argument register accesses
                                                   // made outside start() / wait() / settle() (protocol
                                                   // layer bug)
    uint64_t runs_ended = 0;                       // ended runs (core finished, or start refused)
    uint64_t settles = 0;                          // settle() calls
    uint64_t window_reads = 0, window_writes = 0;
    std::vector<uint32_t> window_write_log;        // offsets of window writes (core CSR offsets)

private:
    int raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    int raw_check(const char * what, uint32_t ch, uint64_t off, const void * p, size_t n) const;
    void accept_start();
    void core_finished();
    void run_ended();
    bool is_ctl_reg(uint32_t off) const;

    mock_options o_;
    tsr_sim *    sim_ = nullptr;
    dma_engine   dma_;
    std::vector<uint64_t> addr_;
    // adapter registers
    bool     running_ = false, done_ = false, ready_ = false;
    uint32_t gie_ = 0, ier_ = 0, isr_ = 0, ast_ = 0;
    uint32_t cmd_lo_ = 0, cmd_hi_ = 0, count_ = 0, args_[8] = {};
    std::vector<uint32_t> base_;   // 2 per channel
    mock_hang run_hang_ = mock_hang::none;
    uint32_t reads_left_ = 0;      // mock_hang::reads
    uint32_t polls_left_ = 0;
    int      in_ctl_ = 0;          // inside start() / wait()
};

} // namespace tsr

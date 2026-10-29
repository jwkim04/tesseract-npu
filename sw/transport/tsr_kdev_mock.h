// tsr_kdev_mock.h — the "mock" kdev: a model of the platform adapter's s_axi_control registers
// (docs/xrt-host.md §2, v2: docs/v2/01-execution-model.md §9.2) around the ISA simulator as the
// core, with device memory reached through the dma_engine under configurable DMA rules, and fault
// injection. It exists to test the protocol layer and tsr_xrt_selftest without hardware
// (TSR_DEVICE=mock).
//
// What it models: the channel base registers (read / write storage), ADAPTER_STATUS (W1C as the
// RTL with all strobes set: a 1 in bit k < 4 clears bit k, a 1 anywhere in [31:16] clears the
// whole count), ADAPTER_VERSION ({16'h0001, 8'h00, NUM_CH} as in tsr_xrt_ctrl) and the CSR window
// into the simulator. v1's ap_ctrl_hs model (AP_CTRL, GIE / IER / ISR, the argument registers,
// ap_start running the core, hangs) is deleted with the run protocol (X0); the v2 adapter model
// (ADAPTER_CTRL.MEM_EN, base locking, held requests, EX §9.2) arrives with P3.1. Not modelled:
// AXI timing, reset.
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

    // --- test hooks ---
    tsr_sim *    sim() { return sim_; }
    dma_engine & dma() { return dma_; }
    void set_ch_addr(uint32_t ch, uint64_t a) { addr_[ch] = a; }
    void         inject_status(uint32_t v);        // OR v into ADAPTER_STATUS (count bits saturate at 0xFFFF)
    uint32_t     id_override = 0;                  // non-zero: the window's ID reads this
    // Observations.
    uint64_t window_reads = 0, window_writes = 0;
    std::vector<uint32_t> window_write_log;        // offsets of window writes (core CSR offsets)
    uint64_t unmapped_accesses = 0;                // register accesses below 0x040 or to other unmapped offsets

private:
    int raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    int raw_check(const char * what, uint32_t ch, uint64_t off, const void * p, size_t n) const;

    mock_options o_;
    tsr_sim *    sim_ = nullptr;
    dma_engine   dma_;
    std::vector<uint64_t> addr_;
    // adapter registers
    uint32_t ast_ = 0;
    std::vector<uint32_t> base_;   // 2 per channel
};

} // namespace tsr

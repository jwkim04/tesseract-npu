// tsr_kdev_mock.h — the "mock" kdev: the sim kdev (tsr_kdev_sim.h) inside a model of the
// ap_ctrl_none platform adapter's s_axi_control registers (docs/design/01-execution-model.md "EX" §9.2;
// docs/design/06-software-verification.md "SW" §9.1), with device memory reached through the
// dma_engine under configurable DMA rules, and fault injection. It tests the protocol layer and
// the tools on the hardware register path without hardware (TSR_DEVICE=mock).
//
// Adapter model (EX §9.2): ch<c>_base lo / hi (RW while MEM_EN = 0; a write while MEM_EN = 1 is
// ignored and sets BASE_LOCKED); ADAPTER_CTRL.MEM_EN (rising edge: every base 4 KiB aligned ->
// latch them and set MEM_EN, else stay 0 and set BASE_MISALIGNED; while MEM_EN = 0 a core memory
// request is held and sets REQ_HELD: the model then holds the whole simulator while an op is
// queued or running, an approximation, since the hardware holds only the memory requests);
// ADAPTER_CTRL.DUAL_DIS (bit 1, reset 0) is stored as written (the model has no MVM unit behind it); ADAPTER_STATUS (W1C as the RTL with all strobes set: a 1 in bit k < 6 clears bit k, a 1 anywhere
// in [31:16] clears the whole count); ADAPTER_VERSION {16'h0001, 8'h00, NUM_CH}; plat_err_i =
// ADAPTER_STATUS bits 0-2 or a count != 0 (EX §9.3), shown in STATUS.PLAT_ERR (and in the
// RISC-V's LQ_FLAGS when the simulator models the input, tsr_sim_set_plat_err); the core CSR
// window at 0x1000. The latched bases must equal the channel buffers' device addresses, else the
// core would address the wrong memory: the model then refuses to move (bind_errors counts it).
// Not modeled: AXI timing, the reset scheme.
//
// Fault injection (TSR_MOCK_INJECT and friends, tsr_kdev_mock.cpp): adapter errors at open or at
// the N-th LQ_PUSH, a TCM word that does not hold what was written (VERIFY catches it, EX §3.3), a
// unit that hangs at the N-th LQ_PUSH (UNIT_HUNG, SOFT_RESET never completes: EX §8.6).
#pragma once

#include "tsr_kdev_sim.h"

#include <memory>
#include <vector>

namespace tsr {

struct mock_options {
    uint32_t  num_ch   = 4;
    uint32_t  t_max    = 1;
    uint64_t  ch_bytes = 16ull << 30;
    unsigned  threads  = 1;
    bool      no_vu    = false;          // TSR_SIM_VU=0: the core of tsr_u200_k4 (no VU, no AT, CAPS 0)
    dma_rules dma;                       // rules the mock's raw DMA enforces (violations fail the transfer)
    uint32_t  adapter_version = 0;       // 0: {16'h0001, 8'h00, NUM_CH} (ADAPTER_MAJOR)
    // Faults (0 = none):
    uint32_t  tcm_fault_addr = 0, tcm_fault_xor = 0;   // after every LOAD_DATA write to this TCM word
    uint32_t  hang_at_push = 0;                        // the unit MVM hangs at this LQ_PUSH (1-based)
};

class mock_kdev final : public kdev, private dma_raw {
public:
    explicit mock_kdev(const mock_options & o);
    ~mock_kdev() override;
    bool ok() const { return core_ && core_->ok(); }

    const char * name() const override { return "mock"; }
    uint32_t num_ch() const override { return o_.num_ch; }
    uint64_t ch_bytes(uint32_t) const override { return o_.ch_bytes; }
    uint64_t ch_addr(uint32_t ch) const override { return addr_[ch]; }
    std::string describe() const override;
    int reg_read(uint32_t off, uint32_t * val) override;
    int reg_write(uint32_t off, uint32_t val) override;
    int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    uint64_t now_ns() override { return core_->now_ns(); }
    void idle(uint32_t poll_iter, uint64_t interval_ns) override { core_->idle(poll_iter, interval_ns); }
    tsr_sim * sim() override { return core_->sim(); }
    bool simulated() const override { return true; }

    // --- test hooks ---
    sim_kdev &   core() { return *core_; }
    dma_engine & dma() { return dma_; }
    void         set_ch_addr(uint32_t ch, uint64_t a) { addr_[ch] = a; }
    void         inject_status(uint32_t v);         // OR v into ADAPTER_STATUS (count bits saturate at 0xFFFF)
    void         inject_at_push(uint32_t n, uint32_t v) { push_inj_.push_back({ n, v }); }
    uint32_t     id_override = 0;                   // non-zero: the window's ID reads this
    uint32_t     version_override = 0;              // non-zero: the window's VERSION reads this
    uint32_t     ctl_id_override = UINT32_MAX;      // not UINT32_MAX: CTL_ID reads this (0: no control core)
    uint32_t     rv_freq_override = UINT32_MAX;     // not UINT32_MAX: RV_FREQ reads this (0: no clk_rv)
    bool         sr_stuck = false;                  // SOFT_RESET never completes (SR_BUSY stays 1)
    // Observations.
    uint64_t window_reads = 0, window_writes = 0;
    std::vector<uint32_t> window_write_log;        // offsets of window writes (core CSR offsets)
    uint64_t unmapped_accesses = 0;                // register accesses below 0x040 or to other unmapped offsets
    uint64_t pushes = 0;                           // LQ_PUSH writes
    uint64_t bind_errors = 0;                      // latched bases that differ from the buffers
    bool     mem_en() const { return mem_en_; }
    bool     dual_dis() const { return dual_dis_; }     // ADAPTER_CTRL.DUAL_DIS as last written
    uint64_t latched_base(uint32_t ch) const { return latched_[ch]; }
    bool     hung() const { return hung_; }

private:
    int raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    int raw_check(const char * what, uint32_t ch, uint64_t off, const void * p, size_t n) const;
    bool may_advance();
    void plat_err_update();
    uint32_t window_read(uint32_t c);

    mock_options o_;
    std::unique_ptr<sim_kdev> core_;
    dma_engine   dma_;
    std::vector<uint64_t> addr_;
    // adapter registers
    uint32_t ast_ = 0;
    std::vector<uint32_t> base_;      // 2 per channel
    std::vector<uint64_t> latched_;
    bool     mem_en_ = false;
    bool     dual_dis_ = false;
    bool     hung_ = false;
    bool     sr_busy_ = false;        // a SOFT_RESET that cannot complete (hung_, sr_stuck)
    uint32_t load_addr_ = 0;          // LOAD_ADDR as the host wrote it, advanced per LOAD_DATA (TCM fault)
    std::vector<std::pair<uint64_t, uint32_t>> push_inj_;
};

} // namespace tsr

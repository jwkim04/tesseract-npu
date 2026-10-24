// kernel_dut.h — the XRT kernel as a fastreplay DUT (fr_common.h "A DUT class provides"): the
// Verilated tsr_xrt_kernel (or a packaged top behind its kwrap_* wrapper; model class Vkernel)
// driven like XRT drives a Vitis ap_ctrl_hs kernel, with AXI4 slave memories behind m_axi.
// Included by tb/platform/kernel_bench.cpp after fr_config.h, fr_common.h and Vkernel.h.
//
// Host side (XrtHost, an AXI4-Lite master like the XDMA shell's control path): one transaction at
// a time; AWVALID / WVALID / ARVALID raised after independent random delays and held until their
// handshake, BREADY / RREADY early or late at random (dut axil=fast: no delays, always ready).
// It checks the slave: B only after both AW and W, R only after AR, BVALID / RVALID held until the
// handshake with stable RDATA, BRESP / RRESP OKAY, no response without a request.
// Run protocol (start()): like XRT, write the register map 0x010 .. the last arg in order
// (cmd_addr, cmd_count, the reserved word 0x01C, arg0..7, ch<c>_base lo / hi), GIE / IER, then
// AP_CTRL = 1. Completion: by interrupt (`dut done=irq`) or by polling AP_CTRL every `poll` cycles
// (`done=poll`, XRT's polling mode; `done=rand` picks per run). After the run (post_run): AP_CTRL
// (idle, start clear; done / ready: still set when the interrupt reported the run, already
// cleared by the poll read otherwise; clear on read), ISR / interrupt toggle and clear, and
// ADAPTER_STATUS == the errors this run injected (then cleared, W1C).
// Memory side (AxiSlave): per channel the AXI4 port is translated back to the core's memory port
// semantics (beat = (AxADDR - base) / 64, len = AxLEN + 1, id) and served by fr::MemModel
// (memmodel.h: random AR / AW / W ready, latency, reordering and beat interleaving between IDs,
// in order per ID, one B per AW, the regimes and every trace check). Channel bases: random 4 KiB
// aligned device addresses per run (dut base=rand; base=zero, or base=0xHEX for every channel),
// including carries across bit 32 and addresses near 2^64. Optional error injection:
// --mem p_rresp_err=P / p_bresp_err=P (SLVERR / DECERR on that fraction of R beats / B
// responses; the data is still correct). inject_stale() presents R beats / B responses nobody
// asked for (a pre-reset interconnect) while the kernel is idle.
// AXI4 checker (AxiCheck, every channel, every cycle): AR / AW / W VALID held until READY with a
// stable payload, AxSIZE 6, AxBURST INCR, AxCACHE 0011, AxPROT / AxLOCK / AxQOS 0, AxADDR 64-byte
// aligned and inside [base, base + channel size), AxLEN <= 63, no burst across a 4 KiB boundary,
// WSTRB all ones (WLAST position: memmodel P4), no AR / AW / W VALID while ap_rst_n has been low
// for more than RST_GRACE cycles, nor in the first cycle after it rises.
// hw_reset(): ap_rst_n low for `rst` cycles (default 20) with the memory dropping every
// transaction (the Vitis case: the interconnect is reset with the kernel), then 8 cycles.
#pragma once

#include <deque>
#include <functional>

namespace {

constexpr unsigned KB_AXIL_WIN = 0x1000;       // core CSR window
constexpr unsigned KR_AP_CTRL = 0x000, KR_GIE = 0x004, KR_IER = 0x008, KR_ISR = 0x00C;
constexpr unsigned KR_CMD_LO = 0x010, KR_CMD_HI = 0x014, KR_CMD_CNT = 0x018, KR_ARG0 = 0x020, KR_BASE0 = 0x040;
constexpr unsigned KR_ASTATUS = 0x0F0, KR_AVERSION = 0x0F4;
constexpr unsigned AP_START = 1, AP_DONE = 2, AP_IDLE = 4, AP_READY = 8;
constexpr unsigned RST_GRACE = 3;              // reset reaches the AXI-side flops after 3 edges
// Shortest ap_rst_n pulse the directed `reset` test applies: the kernel's minimum max(3, RST_PIPE + 1)
// (docs/platform-u200.md §5; tsr_xrt_kernel checks it in simulation) for every RST_PIPE <= 3, i.e.
// the U200 tops (RST_PIPE 3) and the kernel-body configurations of the tests.
constexpr unsigned RST_MIN = 4;

struct KOpts {
    std::string done = "rand";     // irq | poll | rand
    std::string base = "rand";     // rand | zero | 0xHEX
    bool axil_fast = false;        // AXI4-Lite master without random delays
    unsigned poll = 16;            // cycles between AP_CTRL polls
    unsigned rst = 20;             // ap_rst_n low cycles in hw_reset
    double p_rresp_err = 0, p_bresp_err = 0;
    uint64_t perf_lo = 0, perf_hi = 0;     // allowed (cycles - PERF_CYCLES) range; 0, 0 = default
};

// ---------------------------------------------------------------------------------------------
// AXI4-Lite master (XRT-like host): a write lane and a read lane. serial (default, like XRT):
// one transaction at a time in push order; !serial: the lanes run concurrently (a PCIe bridge may
// present a read while a posted write is still pending), each in its own push order.
// ---------------------------------------------------------------------------------------------
struct LiteTx {
    bool we = false;
    uint32_t addr = 0, wdata = 0, rdata = 0;
    uint8_t wstrb = 0xF;
    bool done = false;
    std::function<void(LiteTx&)> cb;          // called at completion (polls)
};

class XrtHost {
  public:
    std::vector<std::string>* errs = nullptr;
    fr::Rng rng;
    bool fast = false;
    bool serial = true;

    void reset() {
        q_.clear();
        w_ = Lane();
        r_ = Lane();
        b_held_ = r_held_ = false;
    }
    bool idle() const { return q_.empty() && !w_.t && !r_.t; }
    std::shared_ptr<LiteTx> push(bool we, uint32_t addr, uint32_t wdata, uint8_t wstrb = 0xF,
                                 std::function<void(LiteTx&)> cb = nullptr) {
        auto t = std::make_shared<LiteTx>();
        t->we = we;
        t->addr = addr;
        t->wdata = wdata;
        t->wstrb = wstrb;
        t->cb = std::move(cb);
        q_.push_back(t);
        return t;
    }

    // Sampled slave outputs of this cycle -> handshakes at the coming edge, outputs for the next
    // cycle. in_reset: everything idle.
    struct In {
        bool awready, wready, bvalid, arready, rvalid;
        unsigned bresp, rresp;
        uint32_t rdata;
    };
    // quiet: the slave's reset has had time to arrive (B / R VALID must be low)
    void edge(const In& s, bool in_reset, bool quiet, uint64_t cyc) {
        if (in_reset) {
            if (quiet && (s.bvalid || s.rvalid)) err(cyc, "B / R VALID during reset");
            reset();
            return;
        }
        // ---- slave checks on this cycle's values ----
        if (s.bvalid) {
            if (!w_.t || !(w_.a_done && w_.w_done)) err(cyc, "BVALID without a completed write request (AW and W)");
            if (s.bresp != 0) err(cyc, strf("BRESP %u", s.bresp));
        }
        if (b_held_ && !s.bvalid) err(cyc, "BVALID dropped before BREADY");
        if (s.rvalid) {
            if (!r_.t || !r_.a_done) err(cyc, "RVALID without a read request");
            if (s.rresp != 0) err(cyc, strf("RRESP %u", s.rresp));
            if (r_held_ && s.rdata != r_data_) err(cyc, "RDATA changed while RVALID was held");
        }
        if (r_held_ && !s.rvalid) err(cyc, "RVALID dropped before RREADY");
        // ---- handshakes at this edge ----
        const bool aw_hs = w_.av && s.awready, w_hs = w_.wv && s.wready, ar_hs = r_.av && s.arready;
        const bool b_hs = w_.rdy && s.bvalid, r_hs = r_.rdy && s.rvalid;
        b_held_ = s.bvalid && !w_.rdy;
        r_held_ = s.rvalid && !r_.rdy;
        r_data_ = s.rdata;
        if (aw_hs) w_.av = false, w_.a_done = true;
        if (w_hs) w_.wv = false, w_.w_done = true;
        if (ar_hs) r_.av = false, r_.a_done = true;
        if (w_.t && b_hs) finish(w_, 0);
        if (r_.t && r_hs) finish(r_, s.rdata);
        // ---- outputs for the next cycle ----
        if (serial) {
            if (!w_.t && !r_.t && !q_.empty()) take(q_.front()->we ? w_ : r_);
        } else {
            for (auto& l : {&w_, &r_}) {
                if (l->t) continue;
                for (auto it = q_.begin(); it != q_.end(); ++it) {
                    if ((*it)->we == (l == &w_)) {
                        auto t = *it;
                        q_.erase(it);
                        q_.push_front(t);
                        take(*l);
                        break;
                    }
                }
            }
        }
        lane_out(w_, true);
        lane_out(r_, false);
    }

    // values to drive for the next cycle
    bool awvalid() const { return w_.av; }
    bool wvalid() const { return w_.wv; }
    bool arvalid() const { return r_.av; }
    bool bready() const { return w_.rdy; }
    bool rready() const { return r_.rdy; }
    const LiteTx* wtx() const { return w_.t.get(); }
    const LiteTx* rtx() const { return r_.t.get(); }

  private:
    struct Lane {
        std::shared_ptr<LiteTx> t;
        bool av = false, wv = false, rdy = false;     // AWVALID / ARVALID, WVALID, BREADY / RREADY
        bool a_done = false, w_done = false;
        bool early = false;                           // response ready raised with the request
        unsigned d_a = 0, d_w = 0, d_r = 0;
    };
    std::deque<std::shared_ptr<LiteTx>> q_;
    Lane w_, r_;
    bool b_held_ = false, r_held_ = false;
    uint32_t r_data_ = 0;

    void take(Lane& l) {
        l.t = q_.front();
        q_.pop_front();
        l.av = l.wv = l.rdy = false;
        l.a_done = l.w_done = false;
        l.d_a = fast ? 0 : (unsigned)rng.range(0, 3);
        l.d_w = fast ? 0 : (unsigned)rng.range(0, 3);
        l.d_r = fast ? 0 : (unsigned)rng.range(0, 4);
        l.early = fast || (rng.next() & 1);
    }
    void lane_out(Lane& l, bool wr) {
        if (!l.t) return;
        if (!l.a_done && !l.av) { if (l.d_a) --l.d_a; else l.av = true; }
        if (wr && !l.w_done && !l.wv) { if (l.d_w) --l.d_w; else l.wv = true; }
        if (l.early) l.rdy = true;
        else if (l.a_done && (!wr || l.w_done)) { if (l.d_r) --l.d_r; else l.rdy = true; }
    }
    void finish(Lane& l, uint32_t rdata) {
        auto t = l.t;
        l = Lane();
        t->rdata = rdata;
        t->done = true;
        if (t->cb) t->cb(*t);
    }
    void err(uint64_t cyc, const std::string& s) {
        if (errs && errs->size() < 16) errs->push_back(strf("axi-lite cycle %llu: %s", (unsigned long long)cyc, s.c_str()));
    }
};

// ---------------------------------------------------------------------------------------------
// The kernel DUT
// ---------------------------------------------------------------------------------------------
class KernelDut {
  public:
    static const char* kind() { return "kernel"; }
    static bool mem_key(const std::string& kv) {
        return kv.rfind("p_rresp_err=", 0) == 0 || kv.rfind("p_bresp_err=", 0) == 0;
    }

    KOpts ko;
    XrtHost host;
    std::vector<std::string> errs;           // AXI / AXI-Lite / adapter findings (go into the next result)
    // per run: completion mode, injected error counts, stale responses injected
    bool irq_mode = false;
    bool manual = false;                     // directed tests detect completion themselves
    uint64_t inj_r = 0, inj_b = 0, inj_stale = 0;
    uint64_t base[NUM_CH] = {0};

    KernelDut(const Options& opt, int argc, char** argv) {
        for (const auto& kv : opt.dut_args) parse_kv(kv);
        for (const auto& kv : opt.mem_overrides) {
            if (kv.rfind("p_rresp_err=", 0) == 0) ko.p_rresp_err = atof(kv.c_str() + 12);
            if (kv.rfind("p_bresp_err=", 0) == 0) ko.p_bresp_err = atof(kv.c_str() + 12);
        }
        ctx_ = std::make_unique<VerilatedContext>();
        ctx_->commandArgs(argc, argv);
        ctx_->fatalOnError(false);
        ctx_->randReset(2);
        ctx_->randSeed((int)((opt.seed * 2654435761u) & 0x7FFFFFFF) | 1);
#if VM_TRACE
        if (!opt.vcd.empty()) ctx_->traceEverOn(true);
#endif
        top_ = std::make_unique<Vkernel>(ctx_.get(), "kernel");
#if VM_TRACE
        if (!opt.vcd.empty()) {
            vcd_ = std::make_unique<VerilatedVcdC>();
            top_->trace(vcd_.get(), 99);
            vcd_->open(opt.vcd.c_str());
        }
#else
        if (!opt.vcd.empty()) fprintf(stderr, "kernel bench: --vcd needs a build with run.py --waves; ignored\n");
#endif
        fr::MemModel::Config mc;
        mc.num_ch = NUM_CH;
        mc.ch_bytes = FR_CH_BYTES;
        const uint64_t lim_a = FR_CH_BYTES / 64, lim_b = 1ull << AW;
        mc.limit = std::min(lim_a, lim_b);
        mc.rd_outstanding = FR_RD_OUTSTANDING;
        mc.has_mvm = FR_HAS_MVM != 0;
        mem_.configure(mc);
        host.errs = &errs;
        host.fast = ko.axil_fast;
        host.rng.seed(opt.seed ^ 0xA5A5A5A5ull);
        rng_.seed(opt.seed ^ 0x5A5A1234ull);
        top_->ap_clk = 0;
        top_->ap_rst_n = 0;
        drive();
    }
    ~KernelDut() {
#if VM_TRACE
        if (vcd_) vcd_->close();
#endif
        if (top_) top_->final();
    }

    uint64_t cycles() const { return cyc_; }
    fr::MemModel& mem() { return mem_; }
    bool running() const { return running_; }
    bool finished() const { return finished_; }
    uint64_t start_cyc() const { return start_cyc_; }
    uint64_t done_cyc() const { return done_cyc_; }
    std::vector<std::string>& done_errs() { return done_errs_; }
    bool irq() const { return top_->__SYM__interrupt; }
    uint64_t completions() const { return n_done_; }
    // end of a run detected by a directed test: quiescence checks as on_done()
    void end_run() {
        if (running_) on_done();
    }

    // ---- reset ----
    void hw_reset() { hw_reset_n(ko.rst); }
    // ap_rst_n low for rst_cycles; the memory drops every transaction (Vitis: the interconnect is
    // reset with the kernel), the host forgets its transactions.
    void hw_reset_n(unsigned rst_cycles) {
        top_->ap_rst_n = 0;
        in_reset_ = true;
        rst_low_ = 0;
        running_ = false;
        poll_busy_ = false;
        mem_.set_active(false);
        for (unsigned i = 0; i < rst_cycles; ++i) step();
        top_->ap_rst_n = 1;
        in_reset_ = false;
        for (int i = 0; i < 8; ++i) step();
    }

    std::string check_constants() {
        std::string e = check_core_constants(*this);
        auto chk = [&](uint32_t a, uint32_t want, const char* n) {
            const uint32_t got = reg(a);
            if (got != want) e += strf("%s = 0x%08x, expected 0x%08x; ", n, got, want);
        };
        chk(KR_AVERSION, 0x00010000u | NUM_CH, "ADAPTER_VERSION");
        chk(KR_AP_CTRL, AP_IDLE, "AP_CTRL after reset");
        chk(KR_ASTATUS, 0, "ADAPTER_STATUS after reset");
        for (const auto& x : errs) e += x + "; ";
        errs.clear();
        return e;
    }

    // ---- register access through the AXI4-Lite master ----
    uint32_t lite(bool we, uint32_t addr, uint32_t wdata, uint8_t wstrb, bool* ok) {
        auto t = host.push(we, addr, wdata, wstrb);
        for (uint32_t n = 0; !t->done; ++n) {
            if (n >= 4 * CSR_TIMEOUT || in_reset_) {
                if (ok) *ok = false;
                errs.push_back(strf("AXI4-Lite %s 0x%x timed out", we ? "write" : "read", addr));
                return 0;
            }
            step();
        }
        return t->rdata;
    }
    uint32_t reg(uint32_t addr, bool* ok = nullptr) { return lite(false, addr, 0, 0xF, ok); }
    void wreg(uint32_t addr, uint32_t v, bool* ok = nullptr, uint8_t strb = 0xF) { lite(true, addr, v, strb, ok); }
    // core CSR through the window
    uint32_t csr(uint32_t addr, bool we, uint32_t wdata, bool* ok) {
        return lite(we, KB_AXIL_WIN + addr, wdata, 0xF, ok);
    }

    // ---- per trace ----
    void prepare(const fr::Regime& reg_, uint64_t seed) {
        mem_.prepare(reg_, seed);
        rng_.seed(seed ^ 0xC0FFEEull);
        host.rng.seed(seed ^ 0xBADC0DEull);
        irq_mode = ko.done == "irq" ? true : ko.done == "poll" ? false : (rng_.next() & 1) != 0;
        for (unsigned c = 0; c < NUM_CH; ++c) base[c] = pick_base();
        inj_r = inj_b = inj_stale = 0;
        for (auto& s : sl_) s = SlaveCh();
    }

    // XRT-like start: the register map 0x010 .. last arg, GIE / IER, AP_CTRL = 1.
    void start(const tsr_trace& tr, bool* ok) { start_raw(tr.cmd_addr_beat, tr.cmd_count, tr.args, ok); }
    void start_raw(uint64_t cmd_addr, uint32_t count, const uint32_t* args, bool* ok, bool queue_only = false) {
        wreg(KR_GIE, irq_mode ? 1 : 0, ok);
        wreg(KR_IER, irq_mode ? 1 : 0, ok);
        wreg(KR_CMD_LO, (uint32_t)cmd_addr, ok);
        wreg(KR_CMD_HI, (uint32_t)(cmd_addr >> 32), ok);
        wreg(KR_CMD_CNT, count, ok);
        wreg(0x01C, 0, ok);
        for (unsigned i = 0; i < 8; ++i) wreg(KR_ARG0 + 4 * i, args ? args[i] : 0, ok);
        for (unsigned c = 0; c < NUM_CH; ++c) {
            wreg(KR_BASE0 + 8 * c, (uint32_t)base[c], ok);
            wreg(KR_BASE0 + 8 * c + 4, (uint32_t)(base[c] >> 32), ok);
        }
        if (queue_only) return;
        finished_ = false;
        bool o = true;
        wreg(KR_AP_CTRL, AP_START, &o);
        if (!o) {
            *ok = false;
            return;
        }
        running_ = true;
        finished_ = false;
        start_cyc_ = cyc_;
        next_poll_ = cyc_ + ko.poll;
        mem_.set_active(true);
    }

    // PERF_CYCLES counts BUSY cycles; the kernel-level cycles (AP_CTRL write .. completion seen by
    // the host) add the start sequence and the completion latency (docs/platform-u200.md §3).
    void perf_window(uint64_t cycles, uint64_t* lo, uint64_t* hi) const {
        // measured: 77..80 cycles with the interrupt, + up to the polling interval and one AXI4-Lite
        // read when polling (start sequence 13 CSR writes ~ 6 cycles each, completion 4 cycles)
        const uint64_t olo = ko.perf_lo ? ko.perf_lo : 70;
        const uint64_t ohi = ko.perf_hi ? ko.perf_hi : (irq_mode ? 90 : 90 + ko.poll + 20);
        *lo = cycles > ohi ? cycles - ohi : 0;
        *hi = cycles > olo ? cycles - olo : 0;
    }

    void post_run(const tsr_trace&, RunResult& r, bool timed_out) {
        bool ok = true;
        if (!timed_out) {
            const uint32_t c1 = reg(KR_AP_CTRL, &ok), c2 = reg(KR_AP_CTRL, &ok);
            const uint32_t want1 = irq_mode ? (AP_IDLE | AP_DONE | AP_READY) : AP_IDLE;
            if (c1 != want1) r.err(strf("AP_CTRL after the run 0x%x, expected 0x%x (%s)", c1, want1, irq_mode ? "irq" : "poll"));
            if (c2 != AP_IDLE) r.err(strf("AP_CTRL second read 0x%x, expected 0x%x (clear on read)", c2, AP_IDLE));
            const uint32_t isr = reg(KR_ISR, &ok);
            if (isr != (irq_mode ? 1u : 0u)) r.err(strf("ISR = %u after the run (%s)", isr, irq_mode ? "irq" : "poll"));
            if (irq() != irq_mode) r.err(strf("interrupt = %d after the run (%s)", (int)irq(), irq_mode ? "irq" : "poll"));
            if (irq_mode) {
                wreg(KR_ISR, 1, &ok);       // toggle on write: clears the done status
                step();
                step();
                if (reg(KR_ISR, &ok) != 0 || irq()) r.err("ISR / interrupt not cleared by writing ISR = 1");
            }
        }
        // ADAPTER_STATUS == injected errors of this run, then W1C
        const uint32_t as = reg(KR_ASTATUS, &ok);
        const uint64_t n = std::min<uint64_t>(inj_r + inj_b, 0xFFFF);
        const uint32_t want = (inj_r ? 1u : 0u) | (inj_b ? 2u : 0u) | (inj_stale ? 4u : 0u) | ((uint32_t)n << 16);
        if (as != want)
            r.err(strf("ADAPTER_STATUS 0x%08x, expected 0x%08x (injected: %llu RRESP, %llu BRESP errors, %llu stale)", as,
                       want, (unsigned long long)inj_r, (unsigned long long)inj_b, (unsigned long long)inj_stale));
        if (as) {
            wreg(KR_ASTATUS, 0xFFFF000Fu, &ok);
            if (reg(KR_ASTATUS, &ok) != 0) r.err("ADAPTER_STATUS not cleared by W1C");
        }
        if (!ok) r.err("AXI4-Lite access timed out after the run");
        for (const auto& e : errs) r.err(e);
        errs.clear();
        std::string b = ", \"kernel\": {\"done\": ";
        b += irq_mode ? "\"irq\"" : "\"poll\"";
        b += strf(", \"overhead\": %lld, \"rresp_err\": %llu, \"bresp_err\": %llu, \"bases\": [",
                  (long long)r.cycles - (long long)r.perf_cycles, (unsigned long long)inj_r, (unsigned long long)inj_b);
        for (unsigned c = 0; c < NUM_CH; ++c) b += strf("%s\"0x%llx\"", c ? ", " : "", (unsigned long long)base[c]);
        r.extra_json = b + "]}";
    }

    // ---- stale responses (pre-reset interconnect): only while the kernel and memory are idle ----
    void inject_stale(unsigned ch, bool read, unsigned id, unsigned beats) {
        if (read) {
            for (unsigned i = 0; i < beats; ++i) sl_[ch].stale_r.push_back({id, i + 1 == beats});
        } else {
            sl_[ch].stale_b.push_back(id);
        }
    }
    bool stale_pending() const {
        for (const auto& s : sl_)
            if (!s.stale_r.empty() || !s.stale_b.empty() || s.cur_stale_r || s.cur_stale_b) return true;
        return false;
    }

    // ---- one clock cycle ----
    void step() {
        top_->ap_clk = 0;
        top_->eval();
#if VM_TRACE
        if (vcd_) vcd_->dump(ctx_->time());
#endif
        ctx_->timeInc(5);
        sample();
        check_axi();
        if (running_ && irq_mode && !manual && irq()) on_done();
        XrtHost::In li;
        li.awready = top_->s_axi_control_awready;
        li.wready = top_->s_axi_control_wready;
        li.bvalid = top_->s_axi_control_bvalid;
        li.arready = top_->s_axi_control_arready;
        li.rvalid = top_->s_axi_control_rvalid;
        li.bresp = top_->s_axi_control_bresp;
        li.rresp = top_->s_axi_control_rresp;
        li.rdata = top_->s_axi_control_rdata;
        host.edge(li, in_reset_, in_reset_ && rst_low_ >= RST_GRACE, cyc_);
        if (in_reset_) {
            ++rst_low_;
            mem_.reset_transactions();
            for (auto& s : sl_) s.cur_r = s.cur_b = false, s.r_resp = s.b_resp = 0;
        } else {
            mem_.edge(co_);
        }
        slave_edge();
        top_->ap_clk = 1;
        top_->eval();
#if VM_TRACE
        if (vcd_) vcd_->dump(ctx_->time());
#endif
        ctx_->timeInc(5);
        ++cyc_;
        was_reset_ = in_reset_;
        // polling (XRT polling mode): AP_CTRL read every ko.poll cycles while a run is in progress
        if (running_ && !irq_mode && !manual && !poll_busy_ && cyc_ >= next_poll_) {
            poll_busy_ = true;
            host.push(false, KR_AP_CTRL, 0, 0xF, [this](LiteTx& t) {
                poll_busy_ = false;
                next_poll_ = cyc_ + ko.poll;
                if (!running_) return;
                if (t.rdata & AP_DONE) on_done();
            });
        }
        if (!running_ && !poll_busy_) next_poll_ = cyc_;
        drive();
    }

  private:
    struct SlaveCh {
        bool cur_r = false, cur_b = false;           // a memory-model beat / response is presented
        unsigned r_resp = 0, b_resp = 0;
        std::deque<std::pair<unsigned, bool>> stale_r;   // (id, last)
        std::deque<unsigned> stale_b;
        bool cur_stale_r = false, cur_stale_b = false;
        // checker state: last cycle's AR / AW / W presentation without a handshake
        bool ar_pend = false, aw_pend = false, w_pend = false;
        uint64_t ar_addr = 0, aw_addr = 0;
        unsigned ar_misc = 0, aw_misc = 0;
        uint64_t w_strb = 0;
        bool w_last = false;
        uint8_t w_data[64];
    };
    struct AxiOut {
        bool arvalid = false, awvalid = false, wvalid = false, rready = false, bready = false;
        uint64_t araddr = 0, awaddr = 0, wstrb = 0;
        unsigned arlen = 0, arsize = 0, arburst = 0, arlock = 0, arcache = 0, arprot = 0, arqos = 0, arid = 0;
        unsigned awlen = 0, awsize = 0, awburst = 0, awlock = 0, awcache = 0, awprot = 0, awqos = 0, awid = 0;
        bool wlast = false;
        const uint8_t* wdata = nullptr;
    };

    std::unique_ptr<VerilatedContext> ctx_;
    std::unique_ptr<Vkernel> top_;
#if VM_TRACE
    std::unique_ptr<VerilatedVcdC> vcd_;
#endif
    fr::MemModel mem_;
    fr::Rng rng_;
    AxiOut ao_[NUM_CH];
    ChOut co_[NUM_CH];
    SlaveCh sl_[NUM_CH];
    // driven slave values of the current cycle (for handshake detection)
    bool drv_rvalid_[NUM_CH] = {false}, drv_bvalid_[NUM_CH] = {false};
    uint64_t cyc_ = 0;
    bool in_reset_ = true, was_reset_ = true;
    uint64_t rst_low_ = 0;
    bool running_ = false, finished_ = false, poll_busy_ = false;
    uint64_t start_cyc_ = 0, done_cyc_ = 0, next_poll_ = 0, n_done_ = 0;
    std::vector<std::string> done_errs_;

    void parse_kv(const std::string& kv) {
        const size_t eq = kv.find('=');
        const std::string k = kv.substr(0, eq), v = eq == std::string::npos ? "" : kv.substr(eq + 1);
        if (k == "done") ko.done = v;
        else if (k == "base") ko.base = v;
        else if (k == "axil") ko.axil_fast = v == "fast";
        else if (k == "poll") ko.poll = (unsigned)strtoul(v.c_str(), nullptr, 0);
        else if (k == "rst") ko.rst = (unsigned)strtoul(v.c_str(), nullptr, 0);
        else if (k == "perf_lo") ko.perf_lo = strtoull(v.c_str(), nullptr, 0);
        else if (k == "perf_hi") ko.perf_hi = strtoull(v.c_str(), nullptr, 0);
        else {
            fprintf(stderr, "kernel bench: unknown --dut option %s\n", kv.c_str());
            exit(2);
        }
        if (ko.done != "irq" && ko.done != "poll" && ko.done != "rand") {
            fprintf(stderr, "kernel bench: done=irq|poll|rand\n");
            exit(2);
        }
    }

    // A 4 KiB aligned device address with the whole channel buffer below 2^64.
    uint64_t pick_base() {
        if (ko.base == "zero") return 0;
        if (ko.base != "rand") return strtoull(ko.base.c_str(), nullptr, 0);
        const uint64_t span = FR_CH_BYTES;                  // buffer size
        const uint64_t top = ~0ull - span + 1;              // highest legal base + 1 (before alignment)
        uint64_t b = 0;
        switch (rng_.range(0, 4)) {
        case 0: b = 0; break;
        case 1: b = rng_.range(0, 1ull << 36); break;                                   // low addresses
        case 2: b = (1ull << 32) - rng_.range(0, std::min<uint64_t>(span, 1ull << 31)); break;   // straddles bit 32
        case 3: b = top - 4096 * rng_.range(0, 1 << 20); break;                         // near 2^64
        default: b = rng_.range(0, top); break;
        }
        b &= ~0xFFFull;
        if (b > top - 4096) b = (top - 4096) & ~0xFFFull;
        return b;
    }

    void on_done() {
        running_ = false;
        finished_ = true;
        done_cyc_ = cyc_;
        ++n_done_;
        // DONE => quiescent (rtl-m2 §4 M7) seen at the AXI side: nothing offered, nothing in flight.
        for (unsigned c = 0; c < NUM_CH; ++c) {
            if (ao_[c].arvalid || ao_[c].awvalid || ao_[c].wvalid)
                done_errs_.push_back(strf("ch%u: AXI request / write data offered when the run completed", c));
        }
        if (!mem_.idle()) done_errs_.push_back("memory traffic in flight at completion: " + mem_.busy_desc());
        mem_.set_active(false);
    }

    void sample() {
        for (unsigned c = 0; c < NUM_CH; ++c) {
            AxiOut& o = ao_[c];
            o.arvalid = getb(top_->m_axi_arvalid, c, 1);
            o.araddr = getb(top_->m_axi_araddr, 64 * c, 64);
            o.arlen = (unsigned)getb(top_->m_axi_arlen, 8 * c, 8);
            o.arsize = (unsigned)getb(top_->m_axi_arsize, 3 * c, 3);
            o.arburst = (unsigned)getb(top_->m_axi_arburst, 2 * c, 2);
            o.arlock = (unsigned)getb(top_->m_axi_arlock, c, 1);
            o.arcache = (unsigned)getb(top_->m_axi_arcache, 4 * c, 4);
            o.arprot = (unsigned)getb(top_->m_axi_arprot, 3 * c, 3);
            o.arqos = (unsigned)getb(top_->m_axi_arqos, 4 * c, 4);
            o.arid = (unsigned)getb(top_->m_axi_arid, 4 * c, 4);
            o.awvalid = getb(top_->m_axi_awvalid, c, 1);
            o.awaddr = getb(top_->m_axi_awaddr, 64 * c, 64);
            o.awlen = (unsigned)getb(top_->m_axi_awlen, 8 * c, 8);
            o.awsize = (unsigned)getb(top_->m_axi_awsize, 3 * c, 3);
            o.awburst = (unsigned)getb(top_->m_axi_awburst, 2 * c, 2);
            o.awlock = (unsigned)getb(top_->m_axi_awlock, c, 1);
            o.awcache = (unsigned)getb(top_->m_axi_awcache, 4 * c, 4);
            o.awprot = (unsigned)getb(top_->m_axi_awprot, 3 * c, 3);
            o.awqos = (unsigned)getb(top_->m_axi_awqos, 4 * c, 4);
            o.awid = (unsigned)getb(top_->m_axi_awid, 4 * c, 4);
            o.wvalid = getb(top_->m_axi_wvalid, c, 1);
            o.wstrb = getb(top_->m_axi_wstrb, 64 * c, 64);
            o.wlast = getb(top_->m_axi_wlast, c, 1);
            o.wdata = reinterpret_cast<const uint8_t*>(wide_words_c(top_->m_axi_wdata) + 16 * c);
            o.rready = getb(top_->m_axi_rready, c, 1);
            o.bready = getb(top_->m_axi_bready, c, 1);
            // the core's memory-port view for the memory model
            ChOut& m = co_[c];
            m.rd_req_valid = o.arvalid;
            m.rd_req_addr = to_beat(c, o.araddr);
            m.rd_req_len = o.arlen + 1;
            m.rd_req_id = o.arid;
            m.rd_rsp_ready = o.rready && !sl_[c].cur_stale_r;
            m.wr_req_valid = o.awvalid;
            m.wr_req_addr = to_beat(c, o.awaddr);
            m.wr_req_len = o.awlen + 1;
            m.wr_req_id = o.awid;
            m.wr_dat_valid = o.wvalid;
            m.wr_dat_data = o.wdata;
            m.wr_dat_last = o.wlast;
            m.wr_rsp_ready = o.bready && !sl_[c].cur_stale_b;
        }
    }

    uint64_t to_beat(unsigned c, uint64_t a) const {
        if (a < base[c]) return ~0ull >> 8;                // below the buffer: beyond any limit
        return (a - base[c]) >> 6;
    }

    void axi_err(unsigned c, const std::string& s) {
        if (errs.size() < 24) errs.push_back(strf("axi ch%u cycle %llu: %s", c, (unsigned long long)cyc_, s.c_str()));
    }

    void check_req(unsigned c, const char* k, uint64_t a, unsigned len, unsigned size, unsigned burst, unsigned lock,
                   unsigned cache, unsigned prot, unsigned qos) {
        const uint64_t lim = std::min<uint64_t>(FR_CH_BYTES, 64ull << AW);
        if (a & 63) axi_err(c, strf("%s addr 0x%llx not 64-byte aligned", k, (unsigned long long)a));
        if (size != 6) axi_err(c, strf("%sSIZE %u", k, size));
        if (burst != 1) axi_err(c, strf("%sBURST %u", k, burst));
        if (lock != 0 || prot != 0 || qos != 0) axi_err(c, strf("%sLOCK/PROT/QOS %u/%u/%u", k, lock, prot, qos));
        if (cache != 3) axi_err(c, strf("%sCACHE 0x%x", k, cache));
        if (len > 63) axi_err(c, strf("%sLEN %u > 63", k, len));
        if ((a & 0xFFF) + 64ull * (len + 1) > 0x1000)
            axi_err(c, strf("%s burst 0x%llx len %u crosses a 4 KiB boundary", k, (unsigned long long)a, len + 1));
        if (a < base[c] || a - base[c] + 64ull * (len + 1) > lim)
            axi_err(c, strf("%s burst 0x%llx len %u outside the buffer [0x%llx, +0x%llx)", k, (unsigned long long)a,
                            len + 1, (unsigned long long)base[c], (unsigned long long)lim));
    }

    void check_axi() {
        const bool rst_quiet = (in_reset_ && rst_low_ >= RST_GRACE) || (!in_reset_ && was_reset_);
        for (unsigned c = 0; c < NUM_CH; ++c) {
            const AxiOut& o = ao_[c];
            SlaveCh& s = sl_[c];
            if (rst_quiet && (o.arvalid || o.awvalid || o.wvalid))
                axi_err(c, in_reset_ ? "AR / AW / W VALID during reset" : "AR / AW / W VALID in the first cycle after reset");
            const unsigned ar_misc = o.arlen | o.arsize << 8 | o.arburst << 12 | o.arlock << 14 | o.arcache << 16 |
                                     o.arprot << 20 | o.arqos << 24 | o.arid << 28;
            const unsigned aw_misc = o.awlen | o.awsize << 8 | o.awburst << 12 | o.awlock << 14 | o.awcache << 16 |
                                     o.awprot << 20 | o.awqos << 24 | o.awid << 28;
            if (!in_reset_) {
                if (s.ar_pend && (!o.arvalid || o.araddr != s.ar_addr || ar_misc != s.ar_misc))
                    axi_err(c, "ARVALID dropped or AR payload changed before ARREADY");
                if (s.aw_pend && (!o.awvalid || o.awaddr != s.aw_addr || aw_misc != s.aw_misc))
                    axi_err(c, "AWVALID dropped or AW payload changed before AWREADY");
                if (s.w_pend && (!o.wvalid || o.wstrb != s.w_strb || o.wlast != s.w_last || memcmp(o.wdata, s.w_data, 64)))
                    axi_err(c, "WVALID dropped or W payload changed before WREADY");
            }
            if (o.arvalid && !in_reset_) check_req(c, "AR", o.araddr, o.arlen, o.arsize, o.arburst, o.arlock, o.arcache, o.arprot, o.arqos);
            if (o.awvalid && !in_reset_) check_req(c, "AW", o.awaddr, o.awlen, o.awsize, o.awburst, o.awlock, o.awcache, o.awprot, o.awqos);
            if (o.wvalid && !in_reset_ && o.wstrb != ~0ull) axi_err(c, strf("WSTRB 0x%llx", (unsigned long long)o.wstrb));
            // pending = offered and not accepted at the coming edge (readiness as driven this cycle)
            const ChIn& in = mem_.in_[c];
            s.ar_pend = o.arvalid && !in.rd_req_ready && !in_reset_;
            s.aw_pend = o.awvalid && !in.wr_req_ready && !in_reset_;
            s.w_pend = o.wvalid && !in.wr_dat_ready && !in_reset_;
            s.ar_addr = o.araddr;
            s.ar_misc = ar_misc;
            s.aw_addr = o.awaddr;
            s.aw_misc = aw_misc;
            s.w_strb = o.wstrb;
            s.w_last = o.wlast;
            if (o.wvalid) memcpy(s.w_data, o.wdata, 64);
        }
    }

    // After the memory model's edge: response codes (error injection), stale responses.
    void slave_edge() {
        for (unsigned c = 0; c < NUM_CH; ++c) {
            SlaveCh& s = sl_[c];
            const ChIn& in = mem_.in_[c];
            const AxiOut& o = ao_[c];
            // handshakes of the values driven in this cycle
            const bool r_hs = drv_rvalid_[c] && o.rready, b_hs = drv_bvalid_[c] && o.bready;
            if (in_reset_) {
                s.cur_stale_r = s.cur_stale_b = false;
                continue;
            }
            if (r_hs) {
                if (s.cur_stale_r) {
                    s.cur_stale_r = false;
                    s.stale_r.pop_front();
                    ++inj_stale;
                } else {
                    if (s.r_resp) ++inj_r;
                    s.cur_r = false;
                }
            }
            if (b_hs) {
                if (s.cur_stale_b) {
                    s.cur_stale_b = false;
                    s.stale_b.pop_front();
                    ++inj_stale;
                } else {
                    if (s.b_resp) ++inj_b;
                    s.cur_b = false;
                }
            }
            if (in.rd_rsp_valid && !s.cur_r) {
                s.cur_r = true;
                s.r_resp = (ko.p_rresp_err > 0 && rng_.chance(fr::prob_thr(ko.p_rresp_err))) ? (unsigned)rng_.range(2, 3) : 0;
            }
            if (in.wr_rsp_valid && !s.cur_b) {
                s.cur_b = true;
                s.b_resp = (ko.p_bresp_err > 0 && rng_.chance(fr::prob_thr(ko.p_bresp_err))) ? (unsigned)rng_.range(2, 3) : 0;
            }
            const bool quiet = mem_.idle() && !mem_.active();
            if (!s.cur_stale_r && !s.stale_r.empty() && !in.rd_rsp_valid && quiet) s.cur_stale_r = true;
            if (!s.cur_stale_b && !s.stale_b.empty() && !in.wr_rsp_valid && quiet) s.cur_stale_b = true;
        }
    }

    void drive() {
        uint64_t arr = 0, awr = 0, wr = 0, rv = 0, rl = 0, bv = 0;
        for (unsigned c = 0; c < NUM_CH; ++c) {
            const ChIn& in = mem_.in_[c];
            SlaveCh& s = sl_[c];
            const bool q = in_reset_;
            arr |= (uint64_t)(in.rd_req_ready && !q) << c;
            awr |= (uint64_t)(in.wr_req_ready && !q) << c;
            wr |= (uint64_t)(in.wr_dat_ready && !q) << c;
            if (s.cur_stale_r) {
                static const uint8_t junk[64] = {0xDE, 0xAD, 0xBE, 0xEF};
                memcpy(wide_words(top_->m_axi_rdata) + 16 * c, junk, 64);
                setb(top_->m_axi_rid, 4 * c, 4, s.stale_r.front().first);
                setb(top_->m_axi_rresp, 2 * c, 2, 0);
                rv |= 1ull << c;
                rl |= (uint64_t)s.stale_r.front().second << c;
                drv_rvalid_[c] = true;
            } else {
                if (in.rd_rsp_new) memcpy(wide_words(top_->m_axi_rdata) + 16 * c, in.rd_rsp_data, 64);
                if (in.rd_rsp_valid) {
                    setb(top_->m_axi_rid, 4 * c, 4, in.rd_rsp_id);
                    setb(top_->m_axi_rresp, 2 * c, 2, s.r_resp);
                }
                rv |= (uint64_t)(in.rd_rsp_valid && !q) << c;
                rl |= (uint64_t)(in.rd_rsp_last && !q) << c;
                drv_rvalid_[c] = in.rd_rsp_valid && !q;
            }
            if (s.cur_stale_b) {
                setb(top_->m_axi_bid, 4 * c, 4, s.stale_b.front());
                setb(top_->m_axi_bresp, 2 * c, 2, 0);
                bv |= 1ull << c;
                drv_bvalid_[c] = true;
            } else {
                if (in.wr_rsp_valid) {
                    setb(top_->m_axi_bid, 4 * c, 4, in.wr_rsp_id);
                    setb(top_->m_axi_bresp, 2 * c, 2, s.b_resp);
                }
                bv |= (uint64_t)(in.wr_rsp_valid && !q) << c;
                drv_bvalid_[c] = in.wr_rsp_valid && !q;
            }
        }
        setb(top_->m_axi_arready, 0, NUM_CH, arr);
        setb(top_->m_axi_awready, 0, NUM_CH, awr);
        setb(top_->m_axi_wready, 0, NUM_CH, wr);
        setb(top_->m_axi_rvalid, 0, NUM_CH, rv);
        setb(top_->m_axi_rlast, 0, NUM_CH, rl);
        setb(top_->m_axi_bvalid, 0, NUM_CH, bv);
        // AXI4-Lite master
        top_->s_axi_control_awvalid = host.awvalid();
        top_->s_axi_control_wvalid = host.wvalid();
        top_->s_axi_control_arvalid = host.arvalid();
        top_->s_axi_control_bready = host.bready();
        top_->s_axi_control_rready = host.rready();
        if (const LiteTx* t = host.wtx()) {
            top_->s_axi_control_awaddr = t->addr & 0x1FFF;
            top_->s_axi_control_wdata = t->wdata;
            top_->s_axi_control_wstrb = t->wstrb;
        }
        if (const LiteTx* t = host.rtx()) top_->s_axi_control_araddr = t->addr & 0x1FFF;
    }
};

}  // namespace

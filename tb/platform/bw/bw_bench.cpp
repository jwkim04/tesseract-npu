// bw_bench.cpp — Verilator C++ bench of the M0 DMA bandwidth kernel (tsr_bw_kernel, or the packaged
// top tsr_bw_k4 behind tb/platform/bw/sv/kwrap_tsr_bw_k4.sv; model class Vbw). Built and run by
// tb/platform/bw/run_bw.py (docs/bandwidth-kernel.md §5).
//
//   bw_bench --test all|NAME,... [--regime all|ideal|fast|rand|slow|ordered,...] [--seed N]
//            [--runs N] [--out FILE.jsonl] [--stop-on-fail] [-v]
//
// The kernel is driven like XRT drives an ap_ctrl_hs kernel: an AXI4-Lite master (the adapter
// bench's XRT-like host: random VALID delays, early / late BREADY / RREADY, checks of the slave)
// writes the whole argument map 0x010 .. 0x0BC, GIE / IER, AP_CTRL = 1, and the run completes by
// interrupt or by polling AP_CTRL every 16 cycles (random per run). Each m_axi port is served by
// the adapter's memory model fr::MemModel (tb/fastreplay/memmodel.h: random AR / AW / W ready,
// latency, reordering and beat interleaving between IDs, in order per ID, one B per AW, stall
// phases, write-order modes, the regimes, protocol checks P1-P6) behind an AXI4 translation (beat
// = (AxADDR - buffer address) / 64; the 16 AXI IDs are mapped onto the model's read / write ids,
// which keeps per-ID order and is stricter than AXI), with optional SLVERR / DECERR injection and
// stale responses. The extra "ideal" regime has latency 0 and every
// READY / VALID at 1.
// AXI4 checker (every channel, every cycle): VALID held with a
// stable payload until READY, AxSIZE 6, INCR, AxCACHE 0011, AxPROT / AxLOCK / AxQOS 0, addresses
// 64-byte aligned inside the channel buffer, AxLEN <= 63, no 4 KiB crossing, WSTRB all ones, WLAST
// exactly at the end of each burst (P4), no VALID during reset (after 3 cycles) nor in the first
// cycle after it. Plus the bandwidth kernel's own rules: every AR / AW equals the next burst of
// the reference burst sequence (address, length, ID; tsr_bw_regs.h), bursts in flight per
// direction <= the configured outstanding limit, no traffic on a disabled or refused channel.
// Per run: every result register against the reference — status, bytes, burst counts, read
// checksum (tsr_bw_regs.h over the bytes the model held), error-response counts == injected,
// cycles >= beats, first-response latency <= cycles <= TOTAL_CYCLES — the written region bit-exact
// against the pattern (tsr_bw_regs.h) and written exactly once in full, no write outside it, no
// read outside the read region, AP_CTRL / ISR / interrupt after the run, quiescence at completion,
// no RTL assertion. In the ideal regime with 64 outstanding bursts: cycles - beats <= the overhead
// bound (IDEAL_OVH).
//
// Tests:
//   regmap   reset values, read-back of every argument register with byte strobes, reserved and
//            unmapped offsets, BW_VERSION / BW_INFO, result block = 0 after reset, GIE / IER / ISR
//            and the interrupt pin, auto_restart ignored
//   single   each channel alone: read, write, read + write; a few burst / outstanding / ID settings
//   all4     the channels together: same mode on all, then mixed modes
//   combos   the combination classes: bursts {1,2,3,7,16,31,32,33,63,64} x outstanding
//            {1,2,3,5,16,63,64} x id_log2 {0..4} x modes, random channel subsets, region offsets
//            (4 KiB aligned or not) and sizes (64 B .. 256 KiB), --runs N random runs per regime
//   ideal    the "cycles ~ bytes / 64 + small overhead" check and the measured numbers: read /
//            write / read + write, 1 and 4 channels, bursts 1 / 4 / 16 / 64, outstanding 64; then
//            outstanding 1 .. 32 at burst 64 and 1 (latency-bound throughput)
//   cfgerr   every invalid configuration on one channel while another channel runs
//   errors   SLVERR / DECERR on R and B (counts, saturation not reached), stale R / B while idle
//   apctrl   back-to-back runs, results cleared at a start, the argument snapshot (registers
//            rewritten during a run, and right after AP_CTRL = 1 by a fast host), ap_start while busy
//            (queued), all channels off
//   reset    ap_rst_n in the middle of traffic (3, 4, 20 cycles): registers at reset values, no
//            AXI traffic, then a clean run
//   big      16 MiB read + write on every channel concurrently (fast regime)
// Exit status: number of failed runs / tests.

#include <verilated.h>

#include "Vbw.h"
#if VM_TRACE
#include <verilated_vcd_c.h>
#endif

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "memmodel.h"
#include "tsr_bw_regs.h"

#ifndef VL_USER_STOP
#error "build with -DVL_USER_STOP (run_bw.py does): RTL $error/$stop must not abort the bench"
#endif
#ifndef BW_NUM_CH
#define BW_NUM_CH 4
#endif
#ifndef BW_AW
#define BW_AW 28
#endif
#ifndef BW_SLICES
#define BW_SLICES 2
#endif
#ifndef BW_DESC
#define BW_DESC "tsr_bw_kernel"
#endif

// ---------------------------------------------------------------------------------------------
// RTL $stop / $error / failed assertions: recorded, the run continues (and fails)
// ---------------------------------------------------------------------------------------------
static std::vector<std::string> g_rtl_errors;
static uint64_t g_rtl_error_count = 0;

void vl_stop(const char* filename, int linenum, const char* hier) VL_MT_UNSAFE {
    ++g_rtl_error_count;
    if (g_rtl_errors.size() < 8) {
        char b[600];
        snprintf(b, sizeof b, "RTL $stop/$error at %s:%d%s%s (message in the process log)", filename ? filename : "?",
                 linenum, hier && *hier ? " in " : "", hier ? hier : "");
        g_rtl_errors.push_back(b);
    }
    fflush(stdout);
}

namespace {

using fr::ChIn;
using fr::ChOut;
using namespace tsr_bw;

constexpr unsigned NCH = BW_NUM_CH;
constexpr unsigned AW = BW_AW;
constexpr uint64_t CH_BYTES = 64ull << AW;     // the buffer bound to each channel (the beat range)
constexpr unsigned RST_GRACE = 3;              // reset reaches the AXI-side flops after 3 edges
constexpr uint64_t IDEAL_OVH = 4 + 4 * BW_SLICES;   // ideal regime: cycles - beats bound (measured 7 / 8 at 2 slices)
static_assert(NCH >= 1 && NCH <= 4, "NCH");
static_assert(AW >= 20, "the bench places regions in the low 64 MiB (ADDR_BEAT_W >= 20)");

std::string strf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char* fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    return b;
}

std::string jesc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if ((unsigned char)c < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

// ---- bit access on Verilator port types ----
inline uint64_t mask64(unsigned w) { return w >= 64 ? ~0ull : ((1ull << w) - 1); }
template <typename T>
inline uint64_t getb(const T& v, unsigned lsb, unsigned w) {
    if constexpr (VlIsVlWide<T>::value) {
        const unsigned wi = lsb >> 5, sh = lsb & 31;
        unsigned __int128 acc = 0;
        for (unsigned k = 0; k < 3 && wi + k < T::Words; ++k) acc |= (unsigned __int128)v.at(wi + k) << (32 * k);
        return (uint64_t)(acc >> sh) & mask64(w);
    } else {
        return lsb >= 64 ? 0 : ((uint64_t)v >> lsb) & mask64(w);
    }
}
template <typename T>
inline void setb(T& v, unsigned lsb, unsigned w, uint64_t val) {
    if constexpr (VlIsVlWide<T>::value) {
        unsigned i = 0;
        while (i < w) {
            const unsigned bit = lsb + i, wi = bit >> 5, sh = bit & 31;
            const unsigned n = std::min(32u - sh, w - i);
            const uint32_t m = (n == 32 ? 0xFFFFFFFFu : ((1u << n) - 1)) << sh;
            v.at(wi) = (v.at(wi) & ~m) | (((uint32_t)(val >> i) << sh) & m);
            i += n;
        }
    } else {
        const uint64_t m = mask64(w) << lsb;
        v = (T)(((uint64_t)v & ~m) | ((val << lsb) & m));
    }
}
template <typename T>
inline uint32_t* wide_words(T& v) {
    return v.data();
}

// ---------------------------------------------------------------------------------------------
// AXI4-Lite master (XRT-like host; serial lanes only)
// ---------------------------------------------------------------------------------------------
struct LiteTx {
    bool we = false;
    uint32_t addr = 0, wdata = 0, rdata = 0;
    uint8_t wstrb = 0xF;
    bool done = false;
    std::function<void(LiteTx&)> cb;
};

class LiteHost {
  public:
    std::vector<std::string>* errs = nullptr;
    fr::Rng rng;
    bool fast = false;

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
    struct In {
        bool awready, wready, bvalid, arready, rvalid;
        unsigned bresp, rresp;
        uint32_t rdata;
    };
    void edge(const In& s, bool in_reset, bool quiet, uint64_t cyc) {
        if (in_reset) {
            if (quiet && (s.bvalid || s.rvalid)) err(cyc, "B / R VALID during reset");
            reset();
            return;
        }
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
        if (!w_.t && !r_.t && !q_.empty()) take(q_.front()->we ? w_ : r_);
        lane_out(w_, true);
        lane_out(r_, false);
    }
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
        bool av = false, wv = false, rdy = false;
        bool a_done = false, w_done = false;
        bool early = false;
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
// Run configuration, results, reference burst sequence
// ---------------------------------------------------------------------------------------------
struct ChCfg {
    bool en = false, rd = false, wr = false;
    unsigned b = 64, mo = 16, idl = 0;
    uint64_t bytes = 0, rd_off = 0, wr_off = 0;
    bool raw = false;           // use raw_ctrl as the ctrl word (invalid configurations)
    uint32_t raw_ctrl = 0;
    uint64_t base_override = 0; // != 0: this buffer address (misaligned-base test)
    bool bad = false;           // expected: CFG_ERR
    uint32_t ctrl() const { return raw ? raw_ctrl : ctrl_word(en, rd, wr, b, mo, idl); }
    bool active() const { return en && !bad; }
    bool rd_on() const { return active() && rd && bytes; }
    bool wr_on() const { return active() && wr && bytes; }
};
struct RunCfg {
    uint32_t seed = 1;
    ChCfg ch[NCH];
    std::string regime = "rand";
    int done_mode = -1;         // -1 random, 0 poll, 1 irq
    double p_rresp_err = 0, p_bresp_err = 0;
    bool check_ideal = false;   // ideal regime: cycles - beats <= IDEAL_OVH (needs outstanding 64)
};
struct ChRes {
    uint32_t status = 0, errs = 0, rd_lat = 0, wr_lat = 0, rd_bursts = 0, wr_bursts = 0;
    uint64_t rd_cyc = 0, wr_cyc = 0, rd_bytes = 0, wr_bytes = 0, csum = 0;
};

struct Seq {     // the kernel's burst sequence of one region (tsr_bw_regs.h burst_len)
    uint64_t beat = 0, end = 0, k = 0;
    unsigned b = 1, mask = 0;
    bool next(uint64_t* a, unsigned* len, unsigned* id) {
        if (beat >= end) return false;
        *a = beat;
        *len = burst_len(beat, end - beat, b);
        *id = (unsigned)(k & mask);
        beat += *len;
        ++k;
        return true;
    }
};

fr::Regime regime_of(const std::string& name) {
    fr::Regime r;
    if (name == "ideal") {
        fr::Regime::by_name("fast", &r);
        r.name = "ideal";
        r.rd_lat_lo = r.rd_lat_hi = 0;
        r.wr_lat_lo = r.wr_lat_hi = 0;
        return r;
    }
    if (!fr::Regime::by_name(name, &r)) {
        fprintf(stderr, "bw_bench: unknown regime %s\n", name.c_str());
        exit(2);
    }
    return r;
}

// ---------------------------------------------------------------------------------------------
// The bench
// ---------------------------------------------------------------------------------------------
class Bench {
  public:
    std::vector<std::string> errs;     // findings of the current run / test
    uint64_t checks = 0;               // comparisons made by check_results (all runs)
    LiteHost host;
    fr::Rng rng;
    bool verbose = false;
    uint64_t base[NCH] = {0};

    Bench(uint64_t seed, int argc, char** argv) {
        ctx_ = std::make_unique<VerilatedContext>();
        ctx_->commandArgs(argc, argv);
        ctx_->fatalOnError(false);
        ctx_->randReset(2);
        ctx_->randSeed((int)((seed * 2654435761u) & 0x7FFFFFFF) | 1);
        top_ = std::make_unique<Vbw>(ctx_.get(), "bw");
        fr::MemModel::Config mc;
        mc.num_ch = NCH;
        mc.ch_bytes = CH_BYTES;
        mc.limit = CH_BYTES / 64;
        mc.rd_outstanding = 64;
        mc.has_mvm = true;            // read ids 4..11 (DMA + WS sources) are legal on every channel
        mem_.configure(mc);
        host.errs = &errs;
        host.rng.seed(seed ^ 0xA5A5A5A5ull);
        rng.seed(seed ^ 0x5A5A1234ull);
        fr::Regime r;
        fr::Regime::by_name("fast", &r);
        mem_.prepare(r, seed);
        top_->ap_clk = 0;
        top_->ap_rst_n = 0;
        drive();
        hw_reset(20);
    }
    ~Bench() {
        if (top_) top_->final();
    }

    uint64_t cycles() const { return cyc_; }
    bool irq() const { return top_->__SYM__interrupt; }

    // ---- reset: ap_rst_n low for n cycles; the memory drops every transaction ----
    void hw_reset(unsigned n) {
        top_->ap_rst_n = 0;
        in_reset_ = true;
        rst_low_ = 0;
        running_ = false;
        poll_busy_ = false;
        mem_.set_active(false);
        for (unsigned i = 0; i < n; ++i) step();
        top_->ap_rst_n = 1;
        in_reset_ = false;
        for (unsigned c = 0; c < NCH; ++c) tr_[c] = Track();
        for (int i = 0; i < 8; ++i) step();
    }
    void set_reset(bool low) {   // for mid-run resets driven by a test
        if (low) {
            top_->ap_rst_n = 0;
            in_reset_ = true;
            rst_low_ = 0;
            running_ = false;
            poll_busy_ = false;
            mem_.set_active(false);
        } else {
            top_->ap_rst_n = 1;
            in_reset_ = false;
            for (unsigned c = 0; c < NCH; ++c) tr_[c] = Track();
        }
    }

    // ---- register access ----
    uint32_t lite(bool we, uint32_t addr, uint32_t wdata, uint8_t wstrb, bool* ok) {
        auto t = host.push(we, addr, wdata, wstrb);
        for (uint32_t n = 0; !t->done; ++n) {
            if (n >= 2000 || in_reset_) {
                if (ok) *ok = false;
                errs.push_back(strf("AXI4-Lite %s 0x%x timed out", we ? "write" : "read", addr));
                return 0;
            }
            step();
        }
        return t->rdata;
    }
    uint32_t reg(uint32_t a, bool* ok = nullptr) { return lite(false, a, 0, 0xF, ok); }
    void wreg(uint32_t a, uint32_t v, bool* ok = nullptr, uint8_t strb = 0xF) { lite(true, a, v, strb, ok); }
    uint64_t reg64(uint32_t a) {
        const uint64_t lo = reg(a);
        return lo | (uint64_t)reg(a + 4) << 32;
    }

    // ---- one clock cycle ----
    void step() {
        top_->ap_clk = 0;
        top_->eval();
        ctx_->timeInc(5);
        sample();
        check_axi();
        if (running_ && irq_mode_ && irq()) on_done();
        LiteHost::In li;
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
            for (auto& s : sl_) s.cur_r = s.cur_b = s.cur_stale_r = s.cur_stale_b = false, s.r_resp = s.b_resp = 0;
        } else {
            mem_.edge(co_);
        }
        slave_edge();
        top_->ap_clk = 1;
        top_->eval();
        ctx_->timeInc(5);
        ++cyc_;
        was_reset_ = in_reset_;
        if (running_ && !irq_mode_ && !poll_busy_ && cyc_ >= next_poll_) {
            poll_busy_ = true;
            host.push(false, AP_CTRL, 0, 0xF, [this](LiteTx& t) {
                poll_busy_ = false;
                next_poll_ = cyc_ + 16;
                if (!running_) return;
                if (t.rdata & AP_DONE) on_done();
            });
        }
        drive();
    }

    // ---- a complete run: memory set-up, XRT-like start, completion, every check ----
    // Returns the per-channel results (out) and TOTAL_CYCLES; findings go to errs.
    bool run(const RunCfg& rc, ChRes out[NCH], uint64_t* total, const char* tag) {
        errs.clear();
        const size_t rtl0 = g_rtl_errors.size();
        const uint64_t rtl_n0 = g_rtl_error_count;
        cfg_ = rc;
        mem_.prepare(regime_of(rc.regime), rc.seed ^ 0x77ull);
        p_rresp_ = rc.p_rresp_err;
        p_bresp_ = rc.p_bresp_err;
        inj_r_.assign(NCH, 0);
        inj_b_.assign(NCH, 0);
        inj_stale_ = 0;
        irq_mode_ = rc.done_mode < 0 ? (rng.next() & 1) != 0 : rc.done_mode == 1;
        for (unsigned c = 0; c < NCH; ++c) {
            const ChCfg& cc = rc.ch[c];
            base[c] = cc.base_override ? cc.base_override : pick_base();
            tr_[c] = Track();
            tr_[c].cfg = cc;
            const uint64_t nb = cc.bytes / 64;
            tr_[c].rseq.b = tr_[c].wseq.b = cc.b;
            tr_[c].rseq.mask = tr_[c].wseq.mask = (1u << std::min(cc.idl, 4u)) - 1;
            if (cc.rd_on()) {
                tr_[c].rseq.beat = cc.rd_off / 64;
                tr_[c].rseq.end = cc.rd_off / 64 + nb;
                // read region data: random bytes or another pattern; reference checksum
                std::vector<uint8_t> d(cc.bytes);
                if (rng.next() & 1) {
                    for (size_t i = 0; i < d.size(); i += 8) {
                        const uint64_t v = rng.next();
                        memcpy(&d[i], &v, 8);
                    }
                } else {
                    const uint32_t s2 = (uint32_t)rng.next();
                    for (uint64_t o = 0; o < cc.bytes; o += 64) {
                        uint64_t w[8];
                        pattern_beat(s2, c, cc.rd_off + o, w);
                        memcpy(&d[o], w, 64);
                    }
                }
                mem_.load(c, cc.rd_off, d.data(), d.size());
                mem_.allow_beats(c, cc.rd_off / 64, cc.rd_off / 64 + nb);
                uint64_t h = 0;
                for (uint64_t o = 0; o < cc.bytes; o += 64) h += beat_hash_bytes(&d[o]);
                tr_[c].ref_csum = h;
            }
            if (cc.wr_on()) {
                tr_[c].wseq.beat = cc.wr_off / 64;
                tr_[c].wseq.end = cc.wr_off / 64 + nb;
                mem_.expect_beats(c, cc.wr_off / 64, cc.wr_off / 64 + nb);
            }
        }
        // idle kernel, no stale ap_done / ISR from an earlier (directed) run
        bool ok = true;
        const uint32_t pre = reg(AP_CTRL, &ok);
        if (pre != AP_IDLE && pre != (AP_IDLE | AP_DONE | AP_READY)) errs.push_back(strf("AP_CTRL 0x%x before the start", pre));
        if (reg(ISR, &ok) & 1) wreg(ISR, 1, &ok);
        // XRT-like start: the whole argument map 0x010 .. 0x0BC, GIE / IER, AP_CTRL = 1
        write_args(rc, &ok);
        wreg(GIE, irq_mode_ ? 1 : 0, &ok);
        wreg(IER, irq_mode_ ? 1 : 0, &ok);
        mem_.set_active(true);
        running_ = true;
        finished_ = false;
        start_cyc_ = cyc_;
        next_poll_ = cyc_ + 16;
        wreg(AP_CTRL, AP_START, &ok);
        uint64_t limit = 20000;
        for (unsigned c = 0; c < NCH; ++c) {
            const ChCfg& cc = rc.ch[c];
            const uint64_t nb = cc.bytes / 64;
            const uint64_t per = (rc.regime == "slow" ? 12 : 4) * (cc.rd_on() + cc.wr_on());
            limit += nb * per + 300 * burst_count(0, nb, std::max(1u, cc.b)) / std::max(1u, cc.mo);
        }
        while (!finished_ && cyc_ - start_cyc_ < limit) step();
        if (!finished_) {
            errs.push_back(strf("no completion within %llu cycles", (unsigned long long)limit));
            running_ = false;
            mem_.set_active(false);
        }
        // AP_CTRL / ISR / interrupt after the run
        if (finished_) {
            const uint32_t c1 = reg(AP_CTRL, &ok), c2 = reg(AP_CTRL, &ok);
            const uint32_t want1 = irq_mode_ ? (AP_IDLE | AP_DONE | AP_READY) : AP_IDLE;
            if (c1 != want1) errs.push_back(strf("AP_CTRL after the run 0x%x, expected 0x%x", c1, want1));
            if (c2 != AP_IDLE) errs.push_back(strf("AP_CTRL second read 0x%x, expected 0x%x (clear on read)", c2, AP_IDLE));
            const uint32_t isr = reg(ISR, &ok);
            if (isr != (irq_mode_ ? 1u : 0u)) errs.push_back(strf("ISR = %u after the run", isr));
            if (irq() != irq_mode_) errs.push_back(strf("interrupt = %d after the run", (int)irq()));
            if (irq_mode_) {
                wreg(ISR, 1, &ok);
                step();
                step();
                if (reg(ISR, &ok) != 0 || irq()) errs.push_back("ISR / interrupt not cleared by writing ISR = 1");
            }
        }
        read_results(out, total);
        check_results(rc, out, *total);
        if (!ok) errs.push_back("AXI4-Lite access failed");
        for (const auto& e : mem_.errors()) errs.push_back(e);
        if (g_rtl_error_count != rtl_n0) {
            for (size_t i = rtl0; i < g_rtl_errors.size(); ++i) errs.push_back(g_rtl_errors[i]);
            errs.push_back(strf("%llu RTL assertion(s)", (unsigned long long)(g_rtl_error_count - rtl_n0)));
        }
        if (verbose || !errs.empty()) {
            printf("%s %s seed %u regime %s %s: %s\n", errs.empty() ? "ok  " : "FAIL", tag, rc.seed, rc.regime.c_str(),
                   irq_mode_ ? "irq" : "poll", describe(rc).c_str());
            for (unsigned c = 0; c < NCH; ++c) {
                if (!rc.ch[c].en) continue;
                printf("      ch%u status 0x%x rd %llu B %llu cyc (lat %u, %u bursts, out max %u) wr %llu B %llu cyc (lat %u, %u bursts, out max %u)\n",
                       c, out[c].status, (unsigned long long)out[c].rd_bytes, (unsigned long long)out[c].rd_cyc, out[c].rd_lat,
                       out[c].rd_bursts, tr_[c].rd_max, (unsigned long long)out[c].wr_bytes, (unsigned long long)out[c].wr_cyc,
                       out[c].wr_lat, out[c].wr_bursts, tr_[c].wr_max);
            }
            for (const auto& e : errs) printf("      %s\n", e.c_str());
        }
        return errs.empty();
    }

    std::string describe(const RunCfg& rc) const {
        std::string s;
        for (unsigned c = 0; c < NCH; ++c) {
            const ChCfg& cc = rc.ch[c];
            if (!cc.en && !cc.raw) continue;
            s += strf("[ch%u %s%s B%u mo%u id%u %llu B rd@0x%llx wr@0x%llx%s] ", c, cc.rd ? "r" : "", cc.wr ? "w" : "", cc.b, cc.mo,
                      cc.idl, (unsigned long long)cc.bytes, (unsigned long long)cc.rd_off, (unsigned long long)cc.wr_off,
                      cc.bad ? " bad" : "");
        }
        return s.empty() ? "(no channel)" : s;
    }

    void write_args(const RunCfg& rc, bool* ok) {
        std::vector<uint32_t> img((CH_BUF0 + 8 * NCH - SEED) / 4, 0);
        auto put = [&](uint32_t off, uint32_t v) { img[(off - SEED) / 4] = v; };
        put(SEED, rc.seed);
        for (unsigned c = 0; c < NCH; ++c) {
            const ChCfg& cc = rc.ch[c];
            put(cfg_reg(c, CFG_CTRL), cc.ctrl());
            put(cfg_reg(c, CFG_BYTES), (uint32_t)cc.bytes);
            put(cfg_reg(c, CFG_BYTES) + 4, (uint32_t)(cc.bytes >> 32));
            put(cfg_reg(c, CFG_RD_OFF), (uint32_t)cc.rd_off);
            put(cfg_reg(c, CFG_RD_OFF) + 4, (uint32_t)(cc.rd_off >> 32));
            put(cfg_reg(c, CFG_WR_OFF), (uint32_t)cc.wr_off);
            put(cfg_reg(c, CFG_WR_OFF) + 4, (uint32_t)(cc.wr_off >> 32));
            put(buf_reg(c), (uint32_t)base[c]);
            put(buf_reg(c) + 4, (uint32_t)(base[c] >> 32));
        }
        for (size_t i = 0; i < img.size(); ++i) wreg(SEED + 4 * (uint32_t)i, img[i], ok);
    }

    void read_results(ChRes out[NCH], uint64_t* total) {
        for (unsigned c = 0; c < NCH; ++c) {
            ChRes& r = out[c];
            r.status = reg(res_reg(c, R_STATUS));
            r.errs = reg(res_reg(c, R_ERRS));
            r.rd_cyc = reg64(res_reg(c, R_RD_CYCLES));
            r.wr_cyc = reg64(res_reg(c, R_WR_CYCLES));
            r.rd_bytes = reg64(res_reg(c, R_RD_BYTES));
            r.wr_bytes = reg64(res_reg(c, R_WR_BYTES));
            r.csum = reg64(res_reg(c, R_CHECKSUM));
            r.rd_lat = reg(res_reg(c, R_RD_LAT));
            r.wr_lat = reg(res_reg(c, R_WR_LAT));
            r.rd_bursts = reg(res_reg(c, R_RD_BURSTS));
            r.wr_bursts = reg(res_reg(c, R_WR_BURSTS));
        }
        *total = reg64(TOTAL_CYCLES);
    }

    void check_results(const RunCfg& rc, const ChRes out[NCH], uint64_t total) {
        for (unsigned c = 0; c < NCH; ++c) {
            const ChCfg& cc = rc.ch[c];
            const ChRes& r = out[c];
            const Track& t = tr_[c];
            auto bad = [&](const std::string& s) { errs.push_back(strf("ch%u: %s", c, s.c_str())); };
            auto ck = [&](bool fail) { ++checks; return fail; };
            const uint64_t nb = cc.bytes / 64;
            const uint64_t inj_r = inj_r_[c], inj_b = inj_b_[c];
            uint32_t want_st = ST_DONE;
            if (cc.en && cc.bad) want_st |= ST_CFG_ERR;
            if (cc.active()) want_st |= ST_RAN;
            if (inj_r) want_st |= ST_RRESP;
            if (inj_b) want_st |= ST_BRESP;
            if (ck((r.status & ~(uint32_t)ST_UNEXPECTED) != want_st))
                bad(strf("status 0x%x, expected 0x%x", r.status, want_st));
            if (ck(r.status & ST_UNEXPECTED)) bad("status UNEXPECTED (a response without a request)");
            const uint32_t want_errs = (uint32_t)std::min<uint64_t>(inj_b, 0xFFFF) << 16 | (uint32_t)std::min<uint64_t>(inj_r, 0xFFFF);
            if (ck(r.errs != want_errs)) bad(strf("errs 0x%08x, expected 0x%08x (injected)", r.errs, want_errs));
            // read direction
            if (cc.rd_on()) {
                if (ck(r.rd_bytes != cc.bytes)) bad(strf("rd_bytes %llu, expected %llu", (unsigned long long)r.rd_bytes, (unsigned long long)cc.bytes));
                if (ck(r.csum != t.ref_csum)) bad(strf("checksum 0x%016llx, expected 0x%016llx", (unsigned long long)r.csum, (unsigned long long)t.ref_csum));
                const uint64_t nbu = burst_count(cc.rd_off / 64, nb, cc.b);
                if (ck(r.rd_bursts != nbu)) bad(strf("rd_bursts %u, expected %llu", r.rd_bursts, (unsigned long long)nbu));
                if (ck(r.rd_cyc < nb)) bad(strf("rd_cycles %llu < %llu beats", (unsigned long long)r.rd_cyc, (unsigned long long)nb));
                if (ck(r.rd_lat < 1 || r.rd_lat > r.rd_cyc)) bad(strf("rd_lat %u outside [1, rd_cycles %llu]", r.rd_lat, (unsigned long long)r.rd_cyc));
                if (ck(r.rd_cyc > total)) bad(strf("rd_cycles %llu > TOTAL_CYCLES %llu", (unsigned long long)r.rd_cyc, (unsigned long long)total));
                const auto& st = mem_.stats(c);
                if (ck(st.rd_beats != nb)) bad(strf("memory saw %llu read beats, expected %llu", (unsigned long long)st.rd_beats, (unsigned long long)nb));
                if (ck(st.bad_reads)) bad(strf("%llu reads outside the read region (first beat 0x%llx)", (unsigned long long)st.bad_reads, (unsigned long long)st.first_bad_read));
                if (ck(t.rseq.beat != t.rseq.end)) bad("read burst sequence incomplete");
                if (ck(t.rd_max > cc.mo)) bad(strf("%u read bursts in flight > outstanding %u", t.rd_max, cc.mo));
                if (rc.check_ideal && cc.mo == 64) {
                    // full rate from the first beat on: last beat = first + beats - 1
                    if (ck(r.rd_cyc > nb + IDEAL_OVH))
                        bad(strf("ideal memory: rd_cycles %llu > beats %llu + %llu", (unsigned long long)r.rd_cyc, (unsigned long long)nb, (unsigned long long)IDEAL_OVH));
                    if (ck(r.rd_cyc != r.rd_lat + nb - 1))
                        bad(strf("ideal memory: rd_cycles %llu != rd_lat %u + beats %llu - 1 (not one beat per cycle)", (unsigned long long)r.rd_cyc, r.rd_lat, (unsigned long long)nb));
                }
            } else {
                if (ck(r.rd_bytes || r.rd_cyc || r.csum || r.rd_lat || r.rd_bursts)) bad("read results not 0 without reads");
                if (ck(t.ar_hs)) bad(strf("%llu AR handshakes without reads", (unsigned long long)t.ar_hs));
            }
            // write direction
            if (cc.wr_on()) {
                if (ck(r.wr_bytes != cc.bytes)) bad(strf("wr_bytes %llu, expected %llu", (unsigned long long)r.wr_bytes, (unsigned long long)cc.bytes));
                const uint64_t nbu = burst_count(cc.wr_off / 64, nb, cc.b);
                if (ck(r.wr_bursts != nbu)) bad(strf("wr_bursts %u, expected %llu", r.wr_bursts, (unsigned long long)nbu));
                if (ck(r.wr_cyc < nb)) bad(strf("wr_cycles %llu < %llu beats", (unsigned long long)r.wr_cyc, (unsigned long long)nb));
                if (ck(r.wr_lat < 1 || r.wr_lat > r.wr_cyc)) bad(strf("wr_lat %u outside [1, wr_cycles %llu]", r.wr_lat, (unsigned long long)r.wr_cyc));
                if (ck(r.wr_cyc > total)) bad(strf("wr_cycles %llu > TOTAL_CYCLES %llu", (unsigned long long)r.wr_cyc, (unsigned long long)total));
                const auto& st = mem_.stats(c);
                if (ck(st.wr_beats != nb)) bad(strf("memory saw %llu write beats, expected %llu", (unsigned long long)st.wr_beats, (unsigned long long)nb));
                if (ck(st.extra_writes)) bad(strf("%llu writes outside the write region (first beat 0x%llx)", (unsigned long long)st.extra_writes, (unsigned long long)st.first_extra_write));
                if (ck(t.wseq.beat != t.wseq.end)) bad("write burst sequence incomplete");
                if (ck(t.wr_max > cc.mo)) bad(strf("%u write bursts in flight > outstanding %u", t.wr_max, cc.mo));
                if (rc.check_ideal && cc.mo == 64) {
                    // full rate: one B per burst right after its last W beat; last B = first B + beats - first burst
                    const unsigned l0 = burst_len(cc.wr_off / 64, nb, cc.b);
                    if (ck(r.wr_cyc > nb + IDEAL_OVH))
                        bad(strf("ideal memory: wr_cycles %llu > beats %llu + %llu", (unsigned long long)r.wr_cyc, (unsigned long long)nb, (unsigned long long)IDEAL_OVH));
                    if (ck(r.wr_cyc != r.wr_lat + nb - l0))
                        bad(strf("ideal memory: wr_cycles %llu != wr_lat %u + beats %llu - first burst %u", (unsigned long long)r.wr_cyc, r.wr_lat, (unsigned long long)nb, l0));
                }
                // the written region, bit-exact, written in full
                uint64_t nerr = 0, first = 0, unwritten = 0;
                const uint8_t* m = mem_.mem(c);
                for (uint64_t k = 0; k < nb; ++k) {
                    const uint64_t beat = cc.wr_off / 64 + k;
                    if (!mem_.written(c, beat)) {
                        ++unwritten;
                        continue;
                    }
                    uint64_t w[8];
                    pattern_beat(rc.seed, c, beat * 64, w);
                    ++checks;
                    if (memcmp(m + beat * 64, w, 64) != 0 && nerr++ == 0) first = beat;
                }
                if (ck(unwritten)) bad(strf("%llu beats of the write region not written", (unsigned long long)unwritten));
                if (ck(nerr)) bad(strf("%llu written beats differ from the pattern (first beat 0x%llx)", (unsigned long long)nerr, (unsigned long long)first));
            } else {
                if (ck(r.wr_bytes || r.wr_cyc || r.wr_lat || r.wr_bursts)) bad("write results not 0 without writes");
                if (ck(t.aw_hs || t.w_hs)) bad(strf("%llu AW / %llu W handshakes without writes", (unsigned long long)t.aw_hs, (unsigned long long)t.w_hs));
            }
        }
    }

    // stale responses while idle (a pre-reset interconnect): ids with nothing outstanding
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
    uint64_t stale_count() const { return inj_stale_; }
    bool any_axi_valid() const {
        for (unsigned c = 0; c < NCH; ++c)
            if (ao_[c].arvalid || ao_[c].awvalid || ao_[c].wvalid) return true;
        return false;
    }
    uint64_t handshakes(unsigned c) const { return tr_[c].ar_hs + tr_[c].aw_hs + tr_[c].w_hs; }
    // start without waiting (apctrl / reset tests): the run state is tracked as in run()
    void start_async(const RunCfg& rc, bool* ok) {
        cfg_ = rc;
        irq_mode_ = false;
        for (unsigned c = 0; c < NCH; ++c) {
            tr_[c] = Track();
            tr_[c].cfg = rc.ch[c];
            tr_[c].unchecked = true;
            base[c] = rc.ch[c].base_override ? rc.ch[c].base_override : pick_base();
        }
        write_args(rc, ok);
        wreg(GIE, 0, ok);
        wreg(IER, 0, ok);
        mem_.set_active(true);
        wreg(AP_CTRL, AP_START, ok);
    }
    void prepare_mem(const std::string& regime, uint64_t seed) {
        mem_.prepare(regime_of(regime), seed);
        for (unsigned c = 0; c < NCH; ++c) {
            mem_.allow_beats(c, 0, std::min<uint64_t>(CH_BYTES / 64, 1ull << 22));
            mem_.expect_beats(c, 0, std::min<uint64_t>(CH_BYTES / 64, 1ull << 22));
        }
    }
    fr::MemModel& mem() { return mem_; }
    void stop_tracking() {
        running_ = false;
        mem_.set_active(false);
    }

  private:
    struct Track {
        ChCfg cfg;
        Seq rseq, wseq;
        uint64_t ref_csum = 0;
        int64_t rd_out = 0, wr_out = 0;
        unsigned rd_max = 0, wr_max = 0;
        uint64_t ar_hs = 0, aw_hs = 0, w_hs = 0;
        bool unchecked = false;        // no reference sequence (async starts)
        std::deque<unsigned> rq[16], wq[16];   // AXI ids per model id, in request order
    };
    struct SlaveCh {
        bool cur_r = false, cur_b = false;
        unsigned r_resp = 0, b_resp = 0;
        std::deque<std::pair<unsigned, bool>> stale_r;
        std::deque<unsigned> stale_b;
        bool cur_stale_r = false, cur_stale_b = false;
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
    std::unique_ptr<Vbw> top_;
    fr::MemModel mem_;
    RunCfg cfg_;
    Track tr_[NCH];
    AxiOut ao_[NCH];
    ChOut co_[NCH];
    SlaveCh sl_[NCH];
    bool drv_rvalid_[NCH] = {false}, drv_bvalid_[NCH] = {false};
    unsigned drv_rid_[NCH] = {0}, drv_rmid_[NCH] = {0}, drv_bmid_[NCH] = {0};
    bool drv_rlast_[NCH] = {false};
    double p_rresp_ = 0, p_bresp_ = 0;
    std::vector<uint64_t> inj_r_ = std::vector<uint64_t>(NCH, 0), inj_b_ = std::vector<uint64_t>(NCH, 0);
    uint64_t inj_stale_ = 0;
    uint64_t cyc_ = 0;
    bool in_reset_ = true, was_reset_ = true;
    uint64_t rst_low_ = 0;
    bool running_ = false, finished_ = false, poll_busy_ = false, irq_mode_ = false;
    uint64_t start_cyc_ = 0, next_poll_ = 0;

    static unsigned rmid(unsigned id) { return 4 + (id & 7); }   // model read ids: DMA 4..7, WS 8..11
    static unsigned wmid(unsigned id) { return 4 + (id & 3); }   // model write ids: DMA 4..7

    // A 4 KiB aligned device address with the whole buffer below 2^64.
    uint64_t pick_base() {
        const uint64_t span = CH_BYTES;
        const uint64_t top = ~0ull - span + 1;
        uint64_t b = 0;
        switch (rng.range(0, 4)) {
        case 0: b = 0; break;
        case 1: b = rng.range(0, 1ull << 36); break;
        case 2: b = (1ull << 32) - rng.range(0, std::min<uint64_t>(span, 1ull << 31)); break;
        case 3: b = top - 4096 * rng.range(0, 1 << 20); break;
        default: b = rng.range(0, top); break;
        }
        b &= ~0xFFFull;
        if (b > top - 4096) b = (top - 4096) & ~0xFFFull;
        return b;
    }

    void on_done() {
        running_ = false;
        finished_ = true;
        for (unsigned c = 0; c < NCH; ++c) {
            if (ao_[c].arvalid || ao_[c].awvalid || ao_[c].wvalid)
                errs.push_back(strf("ch%u: AXI request / write data offered when the run completed", c));
            if (tr_[c].rd_out || tr_[c].wr_out)
                errs.push_back(strf("ch%u: %lld read / %lld write bursts in flight at completion", c, (long long)tr_[c].rd_out,
                                    (long long)tr_[c].wr_out));
        }
        if (!mem_.idle()) errs.push_back("memory traffic in flight at completion: " + mem_.busy_desc());
        mem_.set_active(false);
    }

    void sample() {
        for (unsigned c = 0; c < NCH; ++c) {
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
            o.wdata = reinterpret_cast<const uint8_t*>(top_->m_axi_wdata.data() + 16 * c);
            o.rready = getb(top_->m_axi_rready, c, 1);
            o.bready = getb(top_->m_axi_bready, c, 1);
            ChOut& m = co_[c];
            m.rd_req_valid = o.arvalid;
            m.rd_req_addr = to_beat(c, o.araddr);
            m.rd_req_len = o.arlen + 1;
            m.rd_req_id = rmid(o.arid);
            m.rd_rsp_ready = o.rready && !sl_[c].cur_stale_r;
            m.wr_req_valid = o.awvalid;
            m.wr_req_addr = to_beat(c, o.awaddr);
            m.wr_req_len = o.awlen + 1;
            m.wr_req_id = wmid(o.awid);
            m.wr_dat_valid = o.wvalid;
            m.wr_dat_data = o.wdata;
            m.wr_dat_last = o.wlast;
            m.wr_rsp_ready = o.bready && !sl_[c].cur_stale_b;
        }
    }

    uint64_t to_beat(unsigned c, uint64_t a) const {
        if (a < base[c]) return ~0ull >> 8;
        return (a - base[c]) >> 6;
    }

    void axi_err(unsigned c, const std::string& s) {
        if (errs.size() < 24) errs.push_back(strf("axi ch%u cycle %llu: %s", c, (unsigned long long)cyc_, s.c_str()));
    }

    void check_req(unsigned c, const char* k, uint64_t a, unsigned len, unsigned size, unsigned burst, unsigned lock,
                   unsigned cache, unsigned prot, unsigned qos) {
        checks += 8;
        if (a & 63) axi_err(c, strf("%s addr 0x%llx not 64-byte aligned", k, (unsigned long long)a));
        if (size != 6) axi_err(c, strf("%sSIZE %u", k, size));
        if (burst != 1) axi_err(c, strf("%sBURST %u", k, burst));
        if (lock != 0 || prot != 0 || qos != 0) axi_err(c, strf("%sLOCK/PROT/QOS %u/%u/%u", k, lock, prot, qos));
        if (cache != 3) axi_err(c, strf("%sCACHE 0x%x", k, cache));
        if (len > 63) axi_err(c, strf("%sLEN %u > 63", k, len));
        if ((a & 0xFFF) + 64ull * (len + 1) > 0x1000)
            axi_err(c, strf("%s burst 0x%llx len %u crosses a 4 KiB boundary", k, (unsigned long long)a, len + 1));
        if (a < base[c] || a - base[c] + 64ull * (len + 1) > CH_BYTES)
            axi_err(c, strf("%s burst 0x%llx len %u outside the buffer [0x%llx, +0x%llx)", k, (unsigned long long)a, len + 1,
                            (unsigned long long)base[c], (unsigned long long)CH_BYTES));
    }

    // Expected next burst of a direction; disabled / refused / async: none expected (or unchecked).
    void check_seq(unsigned c, bool rd, uint64_t addr, unsigned len1, unsigned id) {
        Track& t = tr_[c];
        if (t.unchecked) return;
        checks += 3;
        Seq& s = rd ? t.rseq : t.wseq;
        uint64_t a = 0;
        unsigned l = 0, i = 0;
        if (!s.next(&a, &l, &i)) {
            axi_err(c, strf("%s burst 0x%llx len %u beyond the region (no burst expected)", rd ? "AR" : "AW", (unsigned long long)addr, len1));
            return;
        }
        if (addr != base[c] + a * 64 || len1 != l || id != i)
            axi_err(c, strf("%s burst #%llu: addr 0x%llx len %u id %u, expected 0x%llx len %u id %u", rd ? "AR" : "AW",
                            (unsigned long long)(s.k - 1), (unsigned long long)addr, len1, id,
                            (unsigned long long)(base[c] + a * 64), l, i));
    }

    void check_axi() {
        const bool rst_quiet = (in_reset_ && rst_low_ >= RST_GRACE) || (!in_reset_ && was_reset_);
        for (unsigned c = 0; c < NCH; ++c) {
            const AxiOut& o = ao_[c];
            SlaveCh& s = sl_[c];
            Track& t = tr_[c];
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
            const ChIn& in = mem_.in_[c];
            // handshakes at the coming edge: sequence, IDs for the response mapping, in-flight counts
            if (!in_reset_) {
                if (o.arvalid && in.rd_req_ready) {
                    check_seq(c, true, o.araddr, o.arlen + 1, o.arid);
                    t.rq[rmid(o.arid)].push_back(o.arid);
                    ++t.ar_hs;
                    if (++t.rd_out > (int64_t)t.rd_max) t.rd_max = (unsigned)t.rd_out;
                    if (!t.unchecked && !t.cfg.rd_on()) axi_err(c, "read request on a channel without reads");
                }
                if (o.awvalid && in.wr_req_ready) {
                    check_seq(c, false, o.awaddr, o.awlen + 1, o.awid);
                    t.wq[wmid(o.awid)].push_back(o.awid);
                    ++t.aw_hs;
                    if (++t.wr_out > (int64_t)t.wr_max) t.wr_max = (unsigned)t.wr_out;
                    if (!t.unchecked && !t.cfg.wr_on()) axi_err(c, "write request on a channel without writes");
                }
                if (o.wvalid && in.wr_dat_ready) ++t.w_hs;
            }
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

    // After the memory model's edge: handshakes of the driven responses, response codes, stale.
    void slave_edge() {
        for (unsigned c = 0; c < NCH; ++c) {
            SlaveCh& s = sl_[c];
            Track& t = tr_[c];
            const ChIn& in = mem_.in_[c];
            const AxiOut& o = ao_[c];
            const bool r_hs = drv_rvalid_[c] && o.rready, b_hs = drv_bvalid_[c] && o.bready;
            if (in_reset_) {
                s.cur_stale_r = s.cur_stale_b = false;
                continue;
            }
            if (r_hs) {
                if (s.cur_stale_r) {
                    s.cur_stale_r = false;
                    s.stale_r.pop_front();
                    ++inj_stale_;
                } else {
                    if (s.r_resp) ++inj_r_[c];
                    s.cur_r = false;
                    if (drv_rlast_[c]) {
                        auto& q = t.rq[drv_rmid_[c]];
                        if (!q.empty()) q.pop_front();
                        --t.rd_out;
                    }
                }
            }
            if (b_hs) {
                if (s.cur_stale_b) {
                    s.cur_stale_b = false;
                    s.stale_b.pop_front();
                    ++inj_stale_;
                } else {
                    if (s.b_resp) ++inj_b_[c];
                    s.cur_b = false;
                    auto& q = t.wq[drv_bmid_[c]];
                    if (!q.empty()) q.pop_front();
                    --t.wr_out;
                }
            }
            if (in.rd_rsp_valid && !s.cur_r) {
                s.cur_r = true;
                s.r_resp = (p_rresp_ > 0 && rng.chance(fr::prob_thr(p_rresp_))) ? (unsigned)rng.range(2, 3) : 0;
            }
            if (in.wr_rsp_valid && !s.cur_b) {
                s.cur_b = true;
                s.b_resp = (p_bresp_ > 0 && rng.chance(fr::prob_thr(p_bresp_))) ? (unsigned)rng.range(2, 3) : 0;
            }
            const bool quiet = mem_.idle() && !mem_.active();
            if (!s.cur_stale_r && !s.stale_r.empty() && !in.rd_rsp_valid && quiet) s.cur_stale_r = true;
            if (!s.cur_stale_b && !s.stale_b.empty() && !in.wr_rsp_valid && quiet) s.cur_stale_b = true;
        }
    }

    void drive() {
        uint64_t arr = 0, awr = 0, wr = 0, rv = 0, rl = 0, bv = 0;
        for (unsigned c = 0; c < NCH; ++c) {
            const ChIn& in = mem_.in_[c];
            SlaveCh& s = sl_[c];
            Track& t = tr_[c];
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
                drv_rlast_[c] = s.stale_r.front().second;
            } else {
                if (in.rd_rsp_new) memcpy(wide_words(top_->m_axi_rdata) + 16 * c, in.rd_rsp_data, 64);
                if (in.rd_rsp_valid && !q) {
                    const auto& rq = t.rq[in.rd_rsp_id & 15];
                    const unsigned id = rq.empty() ? 0xF : rq.front();
                    if (rq.empty()) axi_err(c, strf("model read response for id %u with no AXI burst recorded", in.rd_rsp_id));
                    setb(top_->m_axi_rid, 4 * c, 4, id);
                    setb(top_->m_axi_rresp, 2 * c, 2, s.r_resp);
                    drv_rid_[c] = id;
                    drv_rmid_[c] = in.rd_rsp_id & 15;
                }
                rv |= (uint64_t)(in.rd_rsp_valid && !q) << c;
                rl |= (uint64_t)(in.rd_rsp_last && !q) << c;
                drv_rvalid_[c] = in.rd_rsp_valid && !q;
                drv_rlast_[c] = in.rd_rsp_last;
            }
            if (s.cur_stale_b) {
                setb(top_->m_axi_bid, 4 * c, 4, s.stale_b.front());
                setb(top_->m_axi_bresp, 2 * c, 2, 0);
                bv |= 1ull << c;
                drv_bvalid_[c] = true;
            } else {
                if (in.wr_rsp_valid && !q) {
                    const auto& wq = t.wq[in.wr_rsp_id & 15];
                    const unsigned id = wq.empty() ? 0xF : wq.front();
                    if (wq.empty()) axi_err(c, strf("model write response for id %u with no AXI burst recorded", in.wr_rsp_id));
                    setb(top_->m_axi_bid, 4 * c, 4, id);
                    setb(top_->m_axi_bresp, 2 * c, 2, s.b_resp);
                    drv_bmid_[c] = in.wr_rsp_id & 15;
                }
                bv |= (uint64_t)(in.wr_rsp_valid && !q) << c;
                drv_bvalid_[c] = in.wr_rsp_valid && !q;
            }
        }
        setb(top_->m_axi_arready, 0, NCH, arr);
        setb(top_->m_axi_awready, 0, NCH, awr);
        setb(top_->m_axi_wready, 0, NCH, wr);
        setb(top_->m_axi_rvalid, 0, NCH, rv);
        setb(top_->m_axi_rlast, 0, NCH, rl);
        setb(top_->m_axi_bvalid, 0, NCH, bv);
        top_->s_axi_control_awvalid = host.awvalid();
        top_->s_axi_control_wvalid = host.wvalid();
        top_->s_axi_control_arvalid = host.arvalid();
        top_->s_axi_control_bready = host.bready();
        top_->s_axi_control_rready = host.rready();
        if (const LiteTx* t = host.wtx()) {
            top_->s_axi_control_awaddr = t->addr & 0xFFF;
            top_->s_axi_control_wdata = t->wdata;
            top_->s_axi_control_wstrb = t->wstrb;
        }
        if (const LiteTx* t = host.rtx()) top_->s_axi_control_araddr = t->addr & 0xFFF;
    }
};

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------
struct Opts {
    std::vector<std::string> tests, regimes;
    uint64_t seed = 1;
    unsigned runs = 40;
    std::string out;
    bool verbose = false;
    bool stop_on_fail = false;   // exit at the first failed run (mutation campaigns)
};

struct TestResult {
    std::string name, regime;
    uint64_t runs = 0, failed = 0, checks = 0;
    std::vector<std::string> errs;
    std::string extra;     // JSON fragment (measurements)
};

const std::vector<std::string> ALL_TESTS = {"regmap", "single", "all4", "combos", "ideal", "cfgerr", "errors", "apctrl", "reset", "big"};
const std::vector<std::string> ALL_REGIMES = {"ideal", "fast", "rand", "slow", "ordered"};

// region helpers: read regions inside the low min(64 MiB, half the buffer), write regions disjoint
constexpr uint64_t SPAN = std::min<uint64_t>(64ull << 20, CH_BYTES / 2);
uint64_t pick_off(fr::Rng& r, uint64_t bytes, bool aligned4k) {
    uint64_t o = r.range(0, (SPAN - bytes) / 64) * 64;
    if (aligned4k) o &= ~0xFFFull;
    return o;
}
// a write offset that does not overlap [ro, ro + bytes) and ends inside the buffer
uint64_t fit_wr(uint64_t ro, uint64_t wo, uint64_t bytes) {
    wo &= ~63ull;
    if (wo + bytes > CH_BYTES || (wo < ro + bytes && wo + bytes > ro)) wo = ro >= bytes ? ro - bytes : ro + bytes;
    return wo;
}

ChCfg mk(bool rd, bool wr, unsigned b, unsigned mo, unsigned idl, uint64_t bytes, uint64_t rd_off, uint64_t wr_off) {
    ChCfg c;
    c.en = true;
    c.rd = rd;
    c.wr = wr;
    c.b = b;
    c.mo = mo;
    c.idl = idl;
    c.bytes = bytes;
    c.rd_off = rd_off;
    c.wr_off = wr_off;
    return c;
}

struct Runner {
    Bench& b;
    Opts& o;
    TestResult* tr = nullptr;
    uint32_t seed_ctr = 1;

    bool go(RunCfg rc, const char* tag, ChRes* res_out = nullptr, uint64_t* total_out = nullptr) {
        ChRes res[NCH];
        uint64_t total = 0;
        rc.seed = (uint32_t)(o.seed * 7919 + seed_ctr++ * 104729);
        const uint64_t ck0 = b.checks;
        const bool ok = b.run(rc, res, &total, tag);
        tr->checks += b.checks - ck0;
        ++tr->runs;
        if (!ok && o.stop_on_fail) {
            printf("FAIL %s (stop on fail): %s\n", tag, b.errs.empty() ? "?" : b.errs[0].c_str());
            exit(1);
        }
        if (!ok) {
            ++tr->failed;
            if (tr->errs.size() < 12) tr->errs.push_back(strf("%s: %s: %s", tag, b.describe(rc).c_str(), b.errs.empty() ? "?" : b.errs[0].c_str()));
        }
        if (res_out) memcpy(res_out, res, sizeof res);
        if (total_out) *total_out = total;
        return ok;
    }
};

void t_regmap(Bench& b, TestResult& tr) {
    auto eq = [&](uint64_t got, uint64_t want, const std::string& what) {
        ++tr.checks;
        if (got != want && tr.errs.size() < 20) tr.errs.push_back(strf("%s = 0x%llx, expected 0x%llx", what.c_str(), (unsigned long long)got, (unsigned long long)want));
    };
    b.errs.clear();
    b.hw_reset(20);
    eq(b.reg(AP_CTRL), AP_IDLE, "AP_CTRL after reset");
    eq(b.reg(GIE), 0, "GIE");
    eq(b.reg(IER), 0, "IER");
    eq(b.reg(ISR), 0, "ISR");
    eq(b.reg(BW_VERSION), (VERSION_MAGIC << 16) | (VERSION_MAJOR << 8) | NCH, "BW_VERSION");
    eq(b.reg(BW_INFO), (AW << 8) | BW_SLICES, "BW_INFO (ADDR_BEAT_W, AXI_SLICES)");
    eq(b.reg64(TOTAL_CYCLES), 0, "TOTAL_CYCLES after reset");
    for (unsigned c = 0; c < NCH; ++c)
        for (uint32_t w = 0; w < RES_STRIDE; w += 4) eq(b.reg(res_reg(c, w)), 0, strf("ch%u result +0x%02x after reset", c, w));
    // every argument register: reset value 0, read-back, byte strobes
    std::vector<uint32_t> args = {SEED};
    for (unsigned c = 0; c < NCH; ++c) {
        for (uint32_t f : {CFG_CTRL, CFG_BYTES, CFG_BYTES + 4, CFG_RD_OFF, CFG_RD_OFF + 4, CFG_WR_OFF, CFG_WR_OFF + 4}) args.push_back(cfg_reg(c, f));
        args.push_back(buf_reg(c));
        args.push_back(buf_reg(c) + 4);
    }
    for (uint32_t a : args) eq(b.reg(a), 0, strf("arg 0x%03x after reset", a));
    for (uint32_t a : args) {
        const uint32_t v = (uint32_t)b.rng.next();
        b.wreg(a, v);
        eq(b.reg(a), v, strf("arg 0x%03x read-back", a));
        const uint8_t strb = (uint8_t)b.rng.range(1, 14);
        const uint32_t v2 = (uint32_t)b.rng.next();
        b.wreg(a, v2, nullptr, strb);
        uint32_t want = v;
        for (unsigned k = 0; k < 4; ++k)
            if (strb >> k & 1) want = (want & ~(0xFFu << 8 * k)) | (v2 & (0xFFu << 8 * k));
        eq(b.reg(a), want, strf("arg 0x%03x after a write with strobe 0x%x", a, strb));
    }
    // reserved / unmapped / read-only offsets: read 0 (or their constant), writes ignored
    std::vector<uint32_t> rsv = {0x014, 0x018, 0x01C, 0x0C0, 0x0E0, 0x0EC, 0x200, 0x204, 0x7FC, 0xFFC};
    for (unsigned c = 0; c < NCH; ++c) rsv.push_back(cfg_reg(c, 4));
    for (unsigned c = NCH; c < 4; ++c) {
        rsv.push_back(cfg_reg(c, CFG_CTRL));
        rsv.push_back(buf_reg(c));
        rsv.push_back(res_reg(c, R_STATUS));
    }
    for (uint32_t a : rsv) {
        b.wreg(a, 0xFFFFFFFFu);
        eq(b.reg(a), 0, strf("reserved 0x%03x", a));
    }
    b.wreg(BW_VERSION, 0);
    eq(b.reg(BW_VERSION), (VERSION_MAGIC << 16) | (VERSION_MAJOR << 8) | NCH, "BW_VERSION after a write");
    b.wreg(res_reg(0, R_CHECKSUM), 0x12345678);
    eq(b.reg(res_reg(0, R_CHECKSUM)), 0, "result register after a write");
    // GIE / IER / ISR, interrupt pin
    b.wreg(GIE, 0xFFFFFFFF);
    eq(b.reg(GIE), 1, "GIE");
    b.wreg(IER, 0xFFFFFFFF);
    eq(b.reg(IER), 1, "IER (bit0 only)");
    eq(b.irq(), 0, "interrupt with ISR 0");
    b.wreg(ISR, 1);
    eq(b.reg(ISR), 1, "ISR toggled to 1");
    b.step();
    b.step();
    eq(b.irq(), 1, "interrupt = GIE & IER & ISR");
    b.wreg(GIE, 0);
    b.step();
    b.step();
    eq(b.irq(), 0, "interrupt with GIE 0");
    b.wreg(ISR, 1);
    eq(b.reg(ISR), 0, "ISR toggled back");
    b.wreg(IER, 0);
    // auto_restart: reads 0, a write of bit7 alone starts nothing
    b.wreg(AP_CTRL, 0x80);
    eq(b.reg(AP_CTRL), AP_IDLE, "AP_CTRL after writing auto_restart");
    for (int i = 0; i < 50; ++i) b.step();
    eq(b.reg(AP_CTRL), AP_IDLE, "AP_CTRL 50 cycles later (no run)");
    for (const auto& e : b.errs) tr.errs.push_back(e);
    tr.runs = 1;
    tr.failed = tr.errs.empty() ? 0 : 1;
}

void t_single(Runner& R, const std::string& reg) {
    struct P { unsigned b, mo, idl; };
    const P ps[] = {{64, 16, 0}, {16, 4, 2}, {1, 8, 4}, {37, 3, 1}};
    for (unsigned c = 0; c < NCH; ++c) {
        for (int mode = 1; mode <= 3; ++mode) {
            for (const P& p : ps) {
                RunCfg rc;
                rc.regime = reg;
                rc.check_ideal = reg == "ideal";
                const uint64_t bytes = 64 * R.b.rng.range(1, 600);
                const uint64_t ro = pick_off(R.b.rng, bytes, R.b.rng.next() & 1);
                const uint64_t wo = fit_wr(ro, ro + bytes + 64 * R.b.rng.range(0, 100), bytes);
                rc.ch[c] = mk(mode & 1, mode & 2, p.b, p.mo, p.idl, bytes, ro, wo);
                R.go(rc, strf("single ch%u", c).c_str());
            }
        }
    }
}

void t_all4(Runner& R, const std::string& reg) {
    for (int mode = 1; mode <= 4; ++mode) {
        for (int k = 0; k < 3; ++k) {
            RunCfg rc;
            rc.regime = reg;
            rc.check_ideal = reg == "ideal";
            for (unsigned c = 0; c < NCH; ++c) {
                const int m = mode == 4 ? (int)R.b.rng.range(1, 3) : mode;
                const uint64_t bytes = 64 * R.b.rng.range(64, 1500);
                const uint64_t ro = pick_off(R.b.rng, bytes, k == 0);
                rc.ch[c] = mk(m & 1, m & 2, (unsigned)R.b.rng.range(1, 64), (unsigned)R.b.rng.range(1, 64), (unsigned)R.b.rng.range(0, 4),
                              bytes, ro, fit_wr(ro, ro + bytes + 4096, bytes));
            }
            R.go(rc, mode == 4 ? "all4 mixed" : "all4");
        }
    }
}

void t_combos(Runner& R, const std::string& reg, unsigned runs) {
    static const unsigned BS[] = {1, 2, 3, 7, 16, 31, 32, 33, 63, 64};
    static const unsigned MO[] = {1, 2, 3, 5, 16, 63, 64};
    fr::Rng& r = R.b.rng;
    // every burst x outstanding class once (one channel each, rotating), then random runs
    unsigned k = 0;
    for (unsigned bi = 0; bi < 10; ++bi) {
        for (unsigned mi = 0; mi < 7; mi += (reg == "slow" ? 2 : 1)) {
            RunCfg rc;
            rc.regime = reg;
            rc.check_ideal = reg == "ideal";
            const unsigned c = k++ % NCH;
            const int m = (int)(k % 3) + 1;
            const uint64_t bytes = 64 * r.range(1, 400);
            const uint64_t ro = pick_off(r, bytes, (k & 3) == 0);
            rc.ch[c] = mk(m & 1, m & 2, BS[bi], MO[mi], (unsigned)(k % 5), bytes, ro, fit_wr(ro, ro + bytes + 64 * r.range(0, 70), bytes));
            R.go(rc, "combos class");
        }
    }
    for (unsigned n = 0; n < runs; ++n) {
        RunCfg rc;
        rc.regime = reg;
        rc.check_ideal = reg == "ideal";
        for (unsigned c = 0; c < NCH; ++c) {
            if (r.range(0, 3) == 0) continue;     // channel off
            const int m = (int)r.range(1, 3);
            const uint64_t bytes = 64 * (r.range(0, 4) == 0 ? r.range(1, 8) : r.range(1, 4096));
            const uint64_t ro = pick_off(r, bytes, r.range(0, 3) == 0);
            const uint64_t wo = fit_wr(ro, (r.next() & 1) ? (ro + bytes + 64 * r.range(0, 200)) : pick_off(r, bytes, false), bytes);
            rc.ch[c] = mk(m & 1, m & 2, BS[r.range(0, 9)], MO[r.range(0, 6)], (unsigned)r.range(0, 4), bytes, ro, wo);
        }
        R.go(rc, "combos random");
    }
}

// ideal memory: measured cycles vs beats; returns the JSON fragment of the measurements
std::string t_ideal(Runner& R) {
    std::string js = "\"ideal\": [";
    bool first = true;
    auto rec = [&](const char* what, unsigned nch, int mode, unsigned b, unsigned mo, uint64_t bytes, const ChRes* res, uint64_t total) {
        const uint64_t nb = bytes / 64;
        for (unsigned c = 0; c < nch; ++c) {
            const ChRes& r = res[c];
            js += strf("%s{\"what\": \"%s\", \"ch\": %u, \"nch\": %u, \"mode\": \"%s\", \"burst\": %u, \"outstanding\": %u, \"beats\": %llu, "
                       "\"rd_cycles\": %llu, \"wr_cycles\": %llu, \"rd_lat\": %u, \"wr_lat\": %u, \"total\": %llu}",
                       first ? "" : ", ", what, c, nch, mode == 1 ? "r" : mode == 2 ? "w" : "rw", b, mo, (unsigned long long)nb,
                       (unsigned long long)r.rd_cyc, (unsigned long long)r.wr_cyc, r.rd_lat, r.wr_lat, (unsigned long long)total);
            first = false;
        }
    };
    const uint64_t bytes = 64 * 8192;   // 512 KiB per direction and channel
    for (unsigned nch : {1u, NCH}) {
        for (int mode = 1; mode <= 3; ++mode) {
            for (unsigned b : {1u, 4u, 16u, 64u}) {
                RunCfg rc;
                rc.regime = "ideal";
                rc.check_ideal = true;
                for (unsigned c = 0; c < nch; ++c) rc.ch[c] = mk(mode & 1, mode & 2, b, 64, 0, bytes, 0, SPAN);
                ChRes res[NCH];
                uint64_t total = 0;
                R.go(rc, "ideal", res, &total);
                rec("rate", nch, mode, b, 64, bytes, res, total);
            }
        }
    }
    // latency-bound: outstanding 1 .. 32 (read and write separately, one channel)
    for (int mode = 1; mode <= 2; ++mode) {
        for (unsigned b : {64u, 1u}) {
            for (unsigned mo : {1u, 2u, 4u, 8u, 16u, 32u}) {
                RunCfg rc;
                rc.regime = "ideal";
                const uint64_t by = 64 * (b == 1 ? 1024 : 4096);
                rc.ch[0] = mk(mode == 1, mode == 2, b, mo, 0, by, 0, SPAN);
                ChRes res[NCH];
                uint64_t total = 0;
                R.go(rc, "ideal outstanding", res, &total);
                rec("outstanding", 1, mode, b, mo, by, res, total);
            }
        }
    }
    return js + "]";
}

void t_cfgerr(Runner& R) {
    struct Bad { const char* what; std::function<void(ChCfg&)> f; };
    const std::vector<Bad> bads = {
        {"mode 0", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, false, false, 16, 8, 0); }},
        {"burst 0", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, true, false, 0, 8, 0); }},
        {"burst 65", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, true, true, 65, 8, 0); }},
        {"burst 127", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, false, true, 127, 8, 0); }},
        {"outstanding 0", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, true, false, 16, 0, 0); }},
        {"outstanding 65", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, false, true, 16, 65, 0); }},
        {"id_log2 5", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, true, false, 16, 8, 5); }},
        {"id_log2 7", [](ChCfg& c) { c.raw = true; c.raw_ctrl = ctrl_word(true, true, true, 16, 8, 7); }},
        {"bytes % 64", [](ChCfg& c) { c.bytes += 13; }},
        {"rd_off % 64", [](ChCfg& c) { c.rd = true; c.rd_off += 32; }},
        {"wr_off % 64", [](ChCfg& c) { c.wr = true; c.wr_off += 8; }},
        {"base not 4 KiB aligned", [](ChCfg& c) { c.base_override = 0x10000040ull; }},
        {"read region beyond the limit", [](ChCfg& c) { c.rd = true; c.rd_off = CH_BYTES - c.bytes + 64; }},
        {"write region beyond the limit", [](ChCfg& c) { c.wr = true; c.wr_off = CH_BYTES; }},
        {"bytes beyond the limit", [](ChCfg& c) { c.bytes = CH_BYTES + 64; c.rd_off = 0; }},
        {"bytes 2^63", [](ChCfg& c) { c.bytes = 1ull << 63; }},
        {"offset 2^64 - 64", [](ChCfg& c) { c.rd = true; c.rd_off = ~63ull; }},
    };
    unsigned k = 0;
    for (const Bad& bd : bads) {
        RunCfg rc;
        rc.regime = "fast";
        const unsigned c = k % NCH, o = (k + 1) % NCH;
        ++k;
        rc.ch[c] = mk(true, false, 16, 8, 0, 64 * 100, 4096, 1 << 20);
        bd.f(rc.ch[c]);
        rc.ch[c].bad = true;
        if (NCH > 1) rc.ch[o] = mk(true, true, 64, 16, 1, 64 * 300, 8192, 2 << 20);
        R.go(rc, (std::string("cfgerr ") + bd.what).c_str());
    }
    // the limit itself is legal: a region ending exactly at 2^(ADDR_BEAT_W + 6)
    RunCfg rc;
    rc.regime = "fast";
    rc.ch[0] = mk(true, true, 64, 16, 0, 64 * 128, CH_BYTES - 64 * 128, CH_BYTES - 64 * 256);
    R.go(rc, "cfgerr region ending at the limit (legal)");
    // bytes = 0 with the offsets exactly at the limit: a legal empty region (it ends at the limit),
    // no traffic; one offset beyond the limit is refused. (Regression: the offset's beat index
    // wrapped to 0 and the generators walked the whole 2^ADDR_BEAT_W-beat range.)
    {
        const struct { bool rd, wr; uint64_t ro, wo; bool bad; const char* what; } zs[] = {
            {true, true, CH_BYTES, CH_BYTES, false, "cfgerr bytes 0 at the limit, read + write (legal, empty)"},
            {true, false, CH_BYTES, 0, false, "cfgerr bytes 0 at the limit, read (legal, empty)"},
            {false, true, 0, CH_BYTES, false, "cfgerr bytes 0 at the limit, write (legal, empty)"},
            {true, true, CH_BYTES, CH_BYTES + 64, true, "cfgerr bytes 0, write offset beyond the limit"},
        };
        for (const auto& z : zs) {
            RunCfg r0;
            r0.regime = "fast";
            const unsigned c = NCH > 1 ? 1 : 0;
            r0.ch[c] = mk(z.rd, z.wr, 64, 16, 0, 0, z.ro, z.wo);
            r0.ch[c].bad = z.bad;
            if (NCH > 1) r0.ch[0] = mk(true, true, 16, 8, 1, 64 * 200, 4096, 1 << 20);
            R.go(r0, z.what);
        }
    }
    // bytes = 0 and a disabled channel with garbage fields: no error, no traffic
    RunCfg rz;
    rz.regime = "fast";
    rz.ch[0] = mk(true, true, 64, 16, 0, 0, 0, 0);
    if (NCH > 1) {
        rz.ch[1].raw = true;
        rz.ch[1].raw_ctrl = 0xFFFFFFFEu;   // enable bit 0 clear
        rz.ch[1].bytes = 13;
    }
    R.go(rz, "cfgerr bytes 0 / disabled channel");
}

void t_errors(Runner& R) {
    for (int k = 0; k < 6; ++k) {
        RunCfg rc;
        rc.regime = k & 1 ? "rand" : "fast";
        rc.p_rresp_err = k < 2 ? 0.05 : k < 4 ? 0.0 : 0.5;
        rc.p_bresp_err = k < 2 ? 0.0 : 0.3;
        for (unsigned c = 0; c < NCH; ++c) rc.ch[c] = mk(true, true, (unsigned)R.b.rng.range(1, 64), 16, 2, 64 * 500, 0, 1 << 20);
        R.go(rc, "errors injected");
    }
    // stale responses while idle: dropped by the bridge, flagged (status UNEXPECTED) until the next start
    Bench& b = R.b;
    TestResult& tr = *R.tr;
    for (unsigned c = 0; c < NCH; ++c) {
        b.errs.clear();
        b.inject_stale(c, true, 3, 2);
        b.inject_stale(c, false, 9, 1);
        for (int i = 0; i < 400 && b.stale_pending(); ++i) b.step();
        for (int i = 0; i < 20; ++i) b.step();
        ++tr.checks;
        const uint32_t st = b.reg(res_reg(c, R_STATUS));
        if (!(st & ST_UNEXPECTED) || b.stale_pending()) tr.errs.push_back(strf("ch%u: stale responses not flagged (status 0x%x)", c, st));
        if (b.any_axi_valid()) tr.errs.push_back(strf("ch%u: AXI traffic after stale responses", c));
        for (const auto& e : b.errs) tr.errs.push_back(e);
        // the next run clears it (and must see none of the stale data)
        RunCfg rc;
        rc.regime = "fast";
        rc.ch[c] = mk(true, true, 8, 8, 2, 64 * 64, 0, 4096 * 4);
        R.go(rc, "errors after stale");
    }
    if (!tr.errs.empty()) ++tr.failed;
}

void t_apctrl(Runner& R) {
    Bench& b = R.b;
    TestResult& tr = *R.tr;
    // back-to-back runs, both completion modes: results of a run never leak into the next
    for (int k = 0; k < 6; ++k) {
        RunCfg rc;
        rc.regime = "fast";
        rc.done_mode = k & 1;
        rc.ch[k % NCH] = mk(k % 3 != 1, k % 3 != 0, 64, 32, 0, 64 * (k + 1) * 50, 0, 1 << 20);
        R.go(rc, "apctrl back-to-back");
    }
    // all channels off: completes, every status DONE only
    RunCfg off;
    off.regime = "fast";
    R.go(off, "apctrl all channels off");
    // argument snapshot: rewriting the registers during a run changes nothing; ap_start while busy
    b.errs.clear();
    b.prepare_mem("fast", 99);
    RunCfg rc;
    rc.regime = "fast";
    for (unsigned c = 0; c < NCH; ++c) rc.ch[c] = mk(false, true, 64, 16, 0, 64 * 3000, 0, 64 * 4096);
    rc.seed = 0xC0FFEE;
    bool ok = true;
    b.start_async(rc, &ok);
    for (int i = 0; i < 100; ++i) b.step();
    ++tr.checks;
    const uint32_t c1 = b.reg(AP_CTRL);
    if (c1 & AP_IDLE) tr.errs.push_back(strf("AP_CTRL 0x%x during a run", c1));
    // rewrite the arguments: bytes x2, other seed, write offset moved
    for (unsigned c = 0; c < NCH; ++c) {
        b.wreg(cfg_reg(c, CFG_BYTES), 64 * 6000);
        b.wreg(cfg_reg(c, CFG_WR_OFF), 64 * 8192);
    }
    b.wreg(SEED, 0xBAD);
    b.wreg(AP_CTRL, AP_START);      // queued
    ++tr.checks;
    if ((b.reg(AP_CTRL) & (AP_START | AP_IDLE)) != AP_START) tr.errs.push_back("ap_start written during a run not pending");
    // wait for the first run (poll: ap_done), then the queued one
    int done_runs = 0;
    for (uint64_t n = 0; n < 200000 && done_runs < 2; ++n) {
        if ((n & 15) == 0) {
            if (b.reg(AP_CTRL) & AP_DONE) ++done_runs;
        } else {
            b.step();
        }
    }
    b.stop_tracking();
    ++tr.checks;
    if (done_runs != 2) tr.errs.push_back(strf("%d completions of the queued pair, expected 2", done_runs));
    // memory: run 1 wrote [4096, 4096 + 3000) beats with seed 0xC0FFEE, run 2 [8192, 8192 + 6000) with 0xBAD
    for (unsigned c = 0; c < NCH; ++c) {
        const uint8_t* m = b.mem().mem(c);
        uint64_t bad1 = 0, bad2 = 0;
        for (uint64_t k = 0; k < 3000; ++k) {
            uint64_t w[8];
            pattern_beat(0xC0FFEE, c, (4096 + k) * 64, w);
            if (memcmp(m + (4096 + k) * 64, w, 64)) ++bad1;
        }
        for (uint64_t k = 0; k < 6000; ++k) {
            uint64_t w[8];
            pattern_beat(0xBAD, c, (8192 + k) * 64, w);
            if (memcmp(m + (8192 + k) * 64, w, 64)) ++bad2;
        }
        ++tr.checks;
        if (bad1 || bad2) tr.errs.push_back(strf("ch%u: snapshot / queued run: %llu / %llu beats differ", c, (unsigned long long)bad1, (unsigned long long)bad2));
        ++tr.checks;
        const uint64_t wb = b.reg64(res_reg(c, R_WR_BYTES));
        if (wb != 64ull * 6000) tr.errs.push_back(strf("ch%u: queued run wr_bytes %llu, expected %llu", c, (unsigned long long)wb, 64ull * 6000));
    }
    for (const auto& e : b.errs) tr.errs.push_back(e);
    for (const auto& e : b.mem().errors()) tr.errs.push_back(e);
    // race: a fast host rewrites channel 0's ctrl, bytes and the seed right after AP_CTRL = 1, before
    // the engines have latched the start (reachable with CTRL_PIPE >= 3): the run must use the
    // snapshot taken at the accept
    b.errs.clear();
    b.prepare_mem("fast", 77);
    b.host.fast = true;
    RunCfg rr;
    rr.regime = "fast";
    rr.ch[0] = mk(false, true, 64, 16, 0, 64 * 700, 0, 64 * 16384);
    rr.seed = 0x5EED;
    bool ok2 = true;
    b.start_async(rr, &ok2);
    b.wreg(cfg_reg(0, CFG_CTRL), ctrl_word(true, false, true, 64, 16, 0) ^ 1u);   // enable bit cleared
    b.wreg(cfg_reg(0, CFG_BYTES), 64 * 100);
    b.wreg(SEED, 0x0BAD);
    b.host.fast = false;
    bool done = false;
    for (uint64_t n = 0; n < 100000 && !done; ++n) {
        if ((n & 15) == 0) done = (b.reg(AP_CTRL) & AP_DONE) != 0;
        else b.step();
    }
    b.stop_tracking();
    ++tr.checks;
    if (!done) tr.errs.push_back("race: no completion");
    {
        const uint8_t* m = b.mem().mem(0);
        uint64_t bad = 0;
        for (uint64_t k = 0; k < 700; ++k) {
            uint64_t w[8];
            pattern_beat(0x5EED, 0, (16384 + k) * 64, w);
            if (memcmp(m + (16384 + k) * 64, w, 64)) ++bad;
        }
        ++tr.checks;
        if (bad) tr.errs.push_back(strf("race: %llu of 700 beats not written with the snapshot (registers rewritten after AP_CTRL won)", (unsigned long long)bad));
        ++tr.checks;
        const uint64_t wb = b.reg64(res_reg(0, R_WR_BYTES));
        if (wb != 64ull * 700) tr.errs.push_back(strf("race: wr_bytes %llu, expected %llu", (unsigned long long)wb, 64ull * 700));
    }
    for (const auto& e : b.errs) tr.errs.push_back(e);
    if (!tr.errs.empty()) ++tr.failed;
}

void t_reset(Runner& R) {
    Bench& b = R.b;
    TestResult& tr = *R.tr;
    for (unsigned rl : {3u, 4u, 20u}) {
        for (int k = 0; k < 3; ++k) {
            b.errs.clear();
            b.prepare_mem("rand", 1000 + rl * 7 + k);
            RunCfg rc;
            rc.regime = "rand";
            for (unsigned c = 0; c < NCH; ++c) rc.ch[c] = mk(true, true, (unsigned)b.rng.range(1, 64), 16, 4, 64 * 2000, 0, 1 << 20);
            bool ok = true;
            b.start_async(rc, &ok);
            const unsigned wait = (unsigned)b.rng.range(10, 2000);
            for (unsigned i = 0; i < wait; ++i) b.step();
            b.set_reset(true);
            for (unsigned i = 0; i < rl; ++i) b.step();
            b.set_reset(false);
            for (int i = 0; i < 8; ++i) b.step();
            ++tr.checks;
            if (b.any_axi_valid()) tr.errs.push_back(strf("reset %u: AXI VALID after the reset", rl));
            const uint32_t a = b.reg(AP_CTRL);
            if (a != AP_IDLE) tr.errs.push_back(strf("reset %u: AP_CTRL 0x%x", rl, a));
            for (unsigned c = 0; c < NCH; ++c) {
                ++tr.checks;
                if (b.reg(res_reg(c, R_STATUS)) != 0 || b.reg(cfg_reg(c, CFG_CTRL)) != 0)
                    tr.errs.push_back(strf("reset %u: ch%u status / ctrl not 0", rl, c));
            }
            for (int i = 0; i < 300; ++i) b.step();
            if (b.any_axi_valid()) tr.errs.push_back(strf("reset %u: AXI traffic after the reset", rl));
            for (const auto& e : b.errs) tr.errs.push_back(strf("reset %u: %s", rl, e.c_str()));
            // then a clean run
            RunCfg c2;
            c2.regime = "rand";
            for (unsigned c = 0; c < NCH; ++c) c2.ch[c] = mk(true, true, 32, 8, 1, 64 * 300, 0, 1 << 20);
            R.go(c2, "reset then run");
        }
    }
    if (!tr.errs.empty()) ++tr.failed;
}

void t_big(Runner& R) {
    RunCfg rc;
    rc.regime = "fast";
    const uint64_t big = std::min<uint64_t>(16ull << 20, SPAN);
    for (unsigned c = 0; c < NCH; ++c) rc.ch[c] = mk(true, true, 64, 32, 0, big, 0, SPAN);
    ChRes res[NCH];
    uint64_t total = 0;
    R.go(rc, "big 16 MiB rw x all channels", res, &total);
    R.tr->extra = strf("\"big\": {\"beats\": %llu, \"total\": %llu, \"rd_cycles\": [", (unsigned long long)(big / 64),
                       (unsigned long long)total);
    for (unsigned c = 0; c < NCH; ++c) R.tr->extra += strf("%s%llu", c ? ", " : "", (unsigned long long)res[c].rd_cyc);
    R.tr->extra += "], \"wr_cycles\": [";
    for (unsigned c = 0; c < NCH; ++c) R.tr->extra += strf("%s%llu", c ? ", " : "", (unsigned long long)res[c].wr_cyc);
    R.tr->extra += "]}";
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = s.find(',', p);
        const std::string t = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (!t.empty()) v.push_back(t);
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "bw_bench: %s needs a value\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--test") o.tests = split(val());
        else if (a == "--regime") o.regimes = split(val());
        else if (a == "--seed") o.seed = strtoull(val().c_str(), nullptr, 0);
        else if (a == "--runs") o.runs = (unsigned)strtoul(val().c_str(), nullptr, 0);
        else if (a == "--out") o.out = val();
        else if (a == "-v") o.verbose = true;
        else if (a == "--stop-on-fail") o.stop_on_fail = true;
        else if (a.rfind("+verilator", 0) == 0) continue;
        else {
            fprintf(stderr, "bw_bench: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (o.tests.empty() || (o.tests.size() == 1 && o.tests[0] == "all")) o.tests = ALL_TESTS;
    if (o.regimes.empty() || (o.regimes.size() == 1 && o.regimes[0] == "all")) o.regimes = ALL_REGIMES;
    for (const auto& t : o.tests)
        if (std::find(ALL_TESTS.begin(), ALL_TESTS.end(), t) == ALL_TESTS.end()) {
            fprintf(stderr, "bw_bench: unknown test %s\n", t.c_str());
            return 2;
        }
    FILE* jf = o.out.empty() ? nullptr : fopen(o.out.c_str(), "w");
    Bench b(o.seed, argc, argv);
    b.verbose = o.verbose;
    printf("bw_bench: %s, NUM_CH %u, ADDR_BEAT_W %u, seed %llu\n", BW_DESC, NCH, AW, (unsigned long long)o.seed);
    int failed = 0;
    for (const auto& t : o.tests) {
        // regime-independent tests run once
        const bool per_regime = t == "single" || t == "all4" || t == "combos";
        std::vector<std::string> regs = per_regime ? o.regimes : std::vector<std::string>{"-"};
        for (const auto& rg : regs) {
            TestResult tr;
            tr.name = t;
            tr.regime = rg;
            Runner R{b, o, &tr};
            const uint64_t c0 = b.cycles();
            if (t == "regmap") t_regmap(b, tr);
            else if (t == "single") t_single(R, rg);
            else if (t == "all4") t_all4(R, rg);
            else if (t == "combos") t_combos(R, rg, o.runs);
            else if (t == "ideal") tr.extra = t_ideal(R);
            else if (t == "cfgerr") t_cfgerr(R);
            else if (t == "errors") t_errors(R);
            else if (t == "apctrl") t_apctrl(R);
            else if (t == "reset") t_reset(R);
            else if (t == "big") t_big(R);
            const bool ok = tr.failed == 0 && tr.errs.empty();
            if (!ok) ++failed;
            if (!ok && o.stop_on_fail) {
                printf("FAIL %s %s (stop on fail): %s\n", t.c_str(), rg.c_str(), tr.errs.empty() ? "?" : tr.errs[0].c_str());
                return 1;
            }
            printf("%s %-7s %-8s runs %llu failed %llu checks %llu cycles %llu\n", ok ? "PASS" : "FAIL", t.c_str(), rg.c_str(),
                   (unsigned long long)tr.runs, (unsigned long long)tr.failed, (unsigned long long)tr.checks,
                   (unsigned long long)(b.cycles() - c0));
            for (const auto& e : tr.errs) printf("     %s\n", e.c_str());
            if (jf) {
                fprintf(jf, "{\"test\": \"%s\", \"regime\": \"%s\", \"ok\": %s, \"runs\": %llu, \"failed\": %llu, \"checks\": %llu, \"cycles\": %llu, \"errs\": [",
                        t.c_str(), rg.c_str(), ok ? "true" : "false", (unsigned long long)tr.runs, (unsigned long long)tr.failed,
                        (unsigned long long)tr.checks, (unsigned long long)(b.cycles() - c0));
                for (size_t i = 0; i < tr.errs.size(); ++i) fprintf(jf, "%s\"%s\"", i ? ", " : "", jesc(tr.errs[i]).c_str());
                fprintf(jf, "]%s%s}\n", tr.extra.empty() ? "" : ", ", tr.extra.c_str());
                fflush(jf);
            }
            fflush(stdout);
        }
    }
    if (jf) fclose(jf);
    printf("bw_bench: %d failed test group(s)\n", failed);
    return failed;
}

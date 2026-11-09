// tsr_bw.cpp — DMA bandwidth tool for the Alveo U200 (docs/bandwidth-kernel.md §6): drives
// the bandwidth kernel tsr_bw_k4 (rtl/platform/alveo_xrt/bw) through the XRT native API and measures
// per-channel and 4-channel concurrent DDR4 bandwidth (read, write, read + write) over transfer
// sizes and burst lengths, checking every byte it can: the read checksum of every read run
// against a host-side reference, the written region of every write run (read back over PCIe and
// compared with the kernel's pattern; sampled or in full), the kernel's status, error-response
// counts, byte and burst counts. Optionally host <-> card DMA bandwidth per channel (--pcie).
//
//   tsr_bw [--xclbin FILE] [--device N|BDF] [--kernel NAME] [--sizes LIST] [--bursts LIST]
//          [--outstanding N] [--ids N] [--modes r,w,rw] [--channels LIST|none] [--no-all]
//          [--reps N] [--sweep-outstanding] [--pcie] [--verify sample|full|none] [--clock-mhz F]
//          [--ch-mib N] [--csv FILE] [--seed N] [--timeout-ms N] [--mock] [--dry-run] [-v]
//
// Defaults: $TSR_XCLBIN, device $TSR_XRT_DEVICE or 0, kernel tsr_bw_k4, sizes 1M,16M,256M,1G,
// bursts 1,4,16,64, outstanding 32, IDs 1, modes r,w,rw, every channel alone and all together,
// 3 repetitions (the median is reported), sampled write verification, the kernel clock from the
// xclbin's clock topology (DATA clock; 300 MHz if absent). Buffers: one device-only buffer per
// channel in the bank its m_axi port is bound to, 2 x the largest size (read region at offset 0,
// write region at offset max size), unless --ch-mib.
// Kernel bandwidth = bytes / (cycles / f_clk), cycles counted by the kernel from its first request
// to its last response (docs/bandwidth-kernel.md §3), so launch overheads are excluded; the wall-clock rate
// (xrt::run start .. wait) is printed next to it.
// --mock runs everything against a C++ model of the kernel's documented behaviour (no XRT device):
// a smoke test of the tool itself; TSR_BW_MOCK_FAULT=csum|write|status|bytes makes the model
// misbehave so that the checks can be seen to fire. --dry-run only opens the device (xclbin, kernel,
// arguments, buffers) and completes one empty run (also on XRT's no-op shim: tb/platform/bw/noop).
// Exit status: 0 = every run passed its checks, 1 = a check failed, 2 = setup error.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <experimental/xrt_xclbin.h>
#include <xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "../../rtl/platform/alveo_xrt/bw/tsr_bw_regs.h"

namespace {

using namespace tsr_bw;

constexpr uint64_t KiB = 1ull << 10, MiB = 1ull << 20, GiB = 1ull << 30;
constexpr uint64_t DMA_CHUNK = 64 * MiB;     // largest single bo.write / bo.read (docs/xrt-host.md §4.2)

std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char * f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

std::string human(uint64_t b) {
    if (b >= GiB && b % GiB == 0) return fmt("%" PRIu64 "G", b / GiB);
    if (b >= MiB && b % MiB == 0) return fmt("%" PRIu64 "M", b / MiB);
    if (b >= KiB && b % KiB == 0) return fmt("%" PRIu64 "K", b / KiB);
    return fmt("%" PRIu64, b);
}

uint64_t parse_size(const std::string & s) {
    char * e = nullptr;
    errno = 0;
    const unsigned long long v = strtoull(s.c_str(), &e, 0);
    if (errno || e == s.c_str()) throw std::runtime_error("bad size '" + s + "'");
    uint64_t m = 1;
    if (*e == 'K' || *e == 'k') m = KiB, ++e;
    else if (*e == 'M' || *e == 'm') m = MiB, ++e;
    else if (*e == 'G' || *e == 'g') m = GiB, ++e;
    if (*e) throw std::runtime_error("bad size '" + s + "'");
    return v * m;
}

std::vector<std::string> split(const std::string & s) {
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

// ---------------------------------------------------------------------------------------------
// One run of the kernel: request and results
// ---------------------------------------------------------------------------------------------
struct ChReq {
    bool en = false, rd = false, wr = false;
    unsigned burst = 64, outstanding = 32, id_log2 = 0;
    uint64_t bytes = 0, rd_off = 0, wr_off = 0;
};
struct RunReq {
    uint32_t seed = 1;
    ChReq ch[MAX_CH];
};
struct ChRes {
    uint32_t status = 0, errs = 0, rd_lat = 0, wr_lat = 0, rd_bursts = 0, wr_bursts = 0;
    uint64_t rd_cycles = 0, wr_cycles = 0, rd_bytes = 0, wr_bytes = 0, checksum = 0;
};
struct RunRes {
    ChRes ch[MAX_CH];
    uint64_t total_cycles = 0;
    double wall_s = 0;
};

// ---------------------------------------------------------------------------------------------
// Device access
// ---------------------------------------------------------------------------------------------
class BwDev {
public:
    virtual ~BwDev() = default;
    virtual unsigned num_ch() const = 0;
    virtual uint64_t ch_bytes() const = 0;
    virtual uint64_t ch_addr(unsigned c) const = 0;
    virtual double clock_mhz() const = 0;
    virtual bool clock_known() const = 0;
    virtual std::string describe() const = 0;
    virtual void exec(const RunReq & r, uint32_t timeout_ms, double * wall_s) = 0;   // start + wait (throws)
    virtual uint32_t reg(uint32_t off) = 0;
    virtual void mem_write(unsigned c, uint64_t off, const void * src, size_t n) = 0;
    virtual void mem_read(unsigned c, uint64_t off, void * dst, size_t n) = 0;

    uint64_t reg64(uint32_t off) { return (uint64_t) reg(off) | (uint64_t) reg(off + 4) << 32; }
    RunRes results() {
        RunRes r;
        for (unsigned c = 0; c < num_ch(); ++c) {
            ChRes & x = r.ch[c];
            x.status = reg(res_reg(c, R_STATUS));
            x.errs = reg(res_reg(c, R_ERRS));
            x.rd_cycles = reg64(res_reg(c, R_RD_CYCLES));
            x.wr_cycles = reg64(res_reg(c, R_WR_CYCLES));
            x.rd_bytes = reg64(res_reg(c, R_RD_BYTES));
            x.wr_bytes = reg64(res_reg(c, R_WR_BYTES));
            x.checksum = reg64(res_reg(c, R_CHECKSUM));
            x.rd_lat = reg(res_reg(c, R_RD_LAT));
            x.wr_lat = reg(res_reg(c, R_WR_LAT));
            x.rd_bursts = reg(res_reg(c, R_RD_BURSTS));
            x.wr_bursts = reg(res_reg(c, R_WR_BURSTS));
        }
        r.total_cycles = reg64(TOTAL_CYCLES);
        return r;
    }
};

// ---- XRT: the kernel through xrt::run, results through read_register (exclusive CU access) ----
class XrtDev final : public BwDev {
public:
    XrtDev(const std::string & xclbin, const std::string & devs, const std::string & kname, uint64_t ch_bytes, double clock_override) {
        dev_ = devs.find_first_not_of("0123456789") == std::string::npos ? xrt::device((unsigned) std::stoul(devs))
                                                                          : xrt::device(devs);
        xb_ = xrt::xclbin(xclbin);
        xrt::xclbin::kernel xk;
        std::string klist;
        for (const auto & k : xb_.get_kernels()) {
            klist += (klist.empty() ? "" : ", ") + k.get_name();
            if (k.get_name() == kname) xk = k;
        }
        if (!xk) throw std::runtime_error(fmt("%s has no kernel %s (kernels: %s)", xclbin.c_str(), kname.c_str(), klist.c_str()));
        const auto cus = xk.get_cus();
        if (cus.size() != 1) throw std::runtime_error(fmt("kernel %s has %zu compute units, expected 1", kname.c_str(), cus.size()));
        if (cus[0].get_control_type() != xrt::xclbin::ip::control_type::hs)
            throw std::runtime_error(fmt("kernel %s is not ap_ctrl_hs", kname.c_str()));
        if (cus[0].get_size() < REG_RANGE)
            throw std::runtime_error(fmt("kernel %s: s_axi_control range 0x%zx < 0x%x (kernel_bw_k4.xml)", kname.c_str(),
                                         (size_t) cus[0].get_size(), REG_RANGE));
        // arguments by name, offsets checked against the register map (docs/bandwidth-kernel.md §2)
        auto find = [&](const std::string & n, uint64_t off, uint64_t sz) {
            for (const auto & a : xk.get_args()) {
                if (a.get_name() != n) continue;
                if (a.get_offset() != off || a.get_size() != sz)
                    throw std::runtime_error(fmt("kernel %s: argument %s at 0x%" PRIx64 " (%" PRIu64 " bytes), expected 0x%" PRIx64
                                                 " (%" PRIu64 ")", kname.c_str(), n.c_str(), (uint64_t) a.get_offset(),
                                                 (uint64_t) a.get_size(), off, sz));
                return (int) a.get_index();
            }
            throw std::runtime_error(fmt("kernel %s has no argument %s", kname.c_str(), n.c_str()));
        };
        a_seed_ = find("seed", SEED, 4);
        for (unsigned c = 0; c < MAX_CH; ++c) {
            a_ctrl_[c] = find(fmt("ctrl%u", c), cfg_reg(c, CFG_CTRL), 4);
            a_bytes_[c] = find(fmt("bytes%u", c), cfg_reg(c, CFG_BYTES), 8);
            a_rdo_[c] = find(fmt("rd_off%u", c), cfg_reg(c, CFG_RD_OFF), 8);
            a_wro_[c] = find(fmt("wr_off%u", c), cfg_reg(c, CFG_WR_OFF), 8);
            a_buf_[c] = find(fmt("ch%u", c), buf_reg(c), 8);
        }
        // kernel clock: the DATA clock of the xclbin's clock topology (what XRT programs at load)
        clock_ = 300.0;
        try {
            const auto * t = xb_.get_axlf_section<const clock_freq_topology *>(CLOCK_FREQ_TOPOLOGY);
            for (int i = 0; t && i < t->m_count; ++i) {
                if (t->m_clock_freq[i].m_type == CT_DATA && t->m_clock_freq[i].m_freq_Mhz) {
                    clock_ = t->m_clock_freq[i].m_freq_Mhz;
                    clock_known_ = true;
                    clock_name_ = t->m_clock_freq[i].m_name;
                    break;
                }
            }
        } catch (const std::exception &) {
        }
        if (clock_override > 0) {
            clock_ = clock_override;
            clock_known_ = true;
            clock_name_ = "--clock-mhz";
        }
        // XRT's no-op shim (XCL_EMULATION_MODE=noop, tb/platform/bw/noop) implements only the
        // hw_context flow (register_xclbin) and no register access; the Alveo shim runs load_xclbin.
        const char * em = getenv("XCL_EMULATION_MODE");
        noop_ = em != nullptr && strcmp(em, "noop") == 0;
        uuid_ = noop_ ? dev_.register_xclbin(xb_) : dev_.load_xclbin(xb_);
        krn_ = xrt::kernel(dev_, uuid_, kname, xrt::kernel::cu_access_mode::exclusive);
        uint32_t ver = 0, inf = 0;
        if (!noop_) {
            ver = reg(BW_VERSION);
            inf = reg(BW_INFO);
            if (ver >> 16 != VERSION_MAGIC || (ver >> 8 & 0xFF) != VERSION_MAJOR || (ver & 0xFF) != MAX_CH)
                throw std::runtime_error(fmt("BW_VERSION 0x%08x: not tsr_bw_k4 v%u with %u channels", ver, VERSION_MAJOR, MAX_CH));
            // Accepting a start does not clear ap_done (only an AP_CTRL read does) and xrt::run never
            // reads AP_CTRL before a start: a completion an earlier process did not wait for would
            // complete our first run at once. Consume it (and wait out a run still going).
            if (!settle_idle(10000))
                throw std::runtime_error("the kernel is still busy with a run of an earlier process after 10 s (it cannot be "
                                         "aborted: wait, or reset the card with xbutil reset)");
        }
        const auto mems = xb_.get_mems();
        for (unsigned c = 0; c < MAX_CH; ++c) {
            const int grp = krn_.group_id(a_buf_[c]);
            std::string tag = fmt("group %d", grp);
            uint64_t bank = 0;
            for (const auto & m : mems) {
                if (m.get_index() == (grp & 0xFFFF)) {
                    tag = m.get_tag();
                    bank = m.get_size_kb() * KiB;
                }
            }
            if (bank && ch_bytes > bank)
                throw std::runtime_error(fmt("channel %u: %s of %.2f GiB is smaller than the buffer (%.2f GiB)", c, tag.c_str(),
                                             (double) bank / GiB, (double) ch_bytes / GiB));
            bo_.emplace_back(dev_, ch_bytes, xrt::bo::flags::device_only, (xrt::memory_group) grp);
            tags_.push_back(tag);
        }
        ch_bytes_ = ch_bytes;
        run_ = xrt::run(krn_);
        for (unsigned c = 0; c < MAX_CH; ++c) run_.set_arg(a_buf_[c], bo_[c]);
        std::string devname = "?", bdf = "?";
        try {
            devname = dev_.get_info<xrt::info::device::name>();
            bdf = dev_.get_info<xrt::info::device::bdf>();
        } catch (const std::exception &) {
        }
        info_ = fmt("device %s (%s), xclbin %s (uuid %s), kernel %s, BW_VERSION 0x%08x, BW_INFO 0x%08x%s\n", devname.c_str(), bdf.c_str(),
                    xclbin.c_str(), uuid_.to_string().c_str(), kname.c_str(), ver, inf,
                    noop_ ? " (no-op shim: no register access)" : "");
        info_ += fmt("kernel clock %.1f MHz (%s)\n", clock_, clock_known_ ? clock_name_.c_str() : "not in the xclbin: assumed");
        for (unsigned c = 0; c < MAX_CH; ++c)
            info_ += fmt("ch%u: %s, %.2f GiB device-only buffer at 0x%" PRIx64 "\n", c, tags_[c].c_str(), (double) ch_bytes / GiB,
                         (uint64_t) bo_[c].address());
    }
    unsigned num_ch() const override { return MAX_CH; }
    uint64_t ch_bytes() const override { return ch_bytes_; }
    uint64_t ch_addr(unsigned c) const override { return bo_[c].address(); }
    double clock_mhz() const override { return clock_; }
    bool clock_known() const override { return clock_known_; }
    std::string describe() const override { return info_; }

    void exec(const RunReq & r, uint32_t timeout_ms, double * wall_s) override {
        run_.set_arg(a_seed_, (uint32_t) r.seed);
        for (unsigned c = 0; c < MAX_CH; ++c) {
            const ChReq & q = r.ch[c];
            run_.set_arg(a_ctrl_[c], ctrl_word(q.en, q.rd, q.wr, q.burst, q.outstanding, q.id_log2));
            run_.set_arg(a_bytes_[c], (uint64_t) q.bytes);
            run_.set_arg(a_rdo_[c], (uint64_t) q.rd_off);
            run_.set_arg(a_wro_[c], (uint64_t) q.wr_off);
        }
        const auto t0 = std::chrono::steady_clock::now();
        run_.start();
        const ert_cmd_state st = run_.wait(std::chrono::milliseconds(timeout_ms));
        *wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (st == ERT_CMD_STATE_TIMEOUT)
            throw std::runtime_error(fmt("no ap_done within %u ms (the bandwidth kernel cannot be aborted: reset the card, xbutil reset)",
                                         timeout_ms));
        if (st != ERT_CMD_STATE_COMPLETED) throw std::runtime_error(fmt("xrt::run ended in state %d (ert_cmd_state)", (int) st));
        // The completion must be this run's: then the kernel is idle. If not, it was a stale
        // ap_done; wait for this run (its ap_done is consumed by the reads).
        if (!noop_ && !settle_idle(0)) {
            fprintf(stderr, "tsr_bw: completion reported while the kernel is still running: a stale ap_done; waiting for the run\n");
            const double spent_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (!settle_idle(timeout_ms > spent_ms ? (uint32_t) (timeout_ms - spent_ms) : 1u))
                throw std::runtime_error(fmt("no end of the run within %u ms (the bandwidth kernel cannot be aborted: reset the card, "
                                             "xbutil reset)", timeout_ms));
            *wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    // read_register is deprecated in XRT 2.18 in favour of xrt::ip, which cannot share a compute
    // unit with xrt::run; it is the supported path for an exclusive kernel (docs/xrt-host.md §4.1).
    uint32_t reg(uint32_t off) override { return krn_.read_register(off); }
    // Read AP_CTRL (no run outstanding) until ap_idle with no start pending, for at most ms (0: one
    // read). Every read consumes ap_done / ap_ready.
    bool settle_idle(uint32_t ms) {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            const uint32_t ap = krn_.read_register(0x000);
            if ((ap & 0x4u) && !(ap & 0x1u)) return true;
            if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() >= ms) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
#pragma GCC diagnostic pop

    void mem_write(unsigned c, uint64_t off, const void * src, size_t n) override {
        for (size_t d = 0; d < n; d += DMA_CHUNK) bo_[c].write((const uint8_t *) src + d, std::min<size_t>(DMA_CHUNK, n - d), off + d);
    }
    void mem_read(unsigned c, uint64_t off, void * dst, size_t n) override {
        for (size_t d = 0; d < n; d += DMA_CHUNK) bo_[c].read((uint8_t *) dst + d, std::min<size_t>(DMA_CHUNK, n - d), off + d);
    }

private:
    xrt::device dev_;
    xrt::xclbin xb_;
    xrt::uuid uuid_;
    xrt::kernel krn_;
    xrt::run run_;
    std::vector<xrt::bo> bo_;
    std::vector<std::string> tags_;
    int a_seed_ = -1, a_ctrl_[MAX_CH], a_bytes_[MAX_CH], a_rdo_[MAX_CH], a_wro_[MAX_CH], a_buf_[MAX_CH];
    uint64_t ch_bytes_ = 0;
    bool noop_ = false;
    double clock_ = 300.0;
    bool clock_known_ = false;
    std::string clock_name_, info_;
};

// ---- mock: the kernel's documented behaviour (docs/bandwidth-kernel.md §2, §3) in C++, no device ----
class MockDev final : public BwDev {
public:
    explicit MockDev(uint64_t ch_bytes) : ch_bytes_(ch_bytes) {
        for (unsigned c = 0; c < MAX_CH; ++c) mem_.emplace_back(ch_bytes, 0);
        const char * f = getenv("TSR_BW_MOCK_FAULT");
        fault_ = f ? f : "";
        regs_.assign(REG_RANGE / 4, 0);
        regs_[BW_VERSION / 4] = VERSION_MAGIC << 16 | VERSION_MAJOR << 8 | MAX_CH;
        regs_[BW_INFO / 4] = 28u << 8 | 2u;
    }
    unsigned num_ch() const override { return MAX_CH; }
    uint64_t ch_bytes() const override { return ch_bytes_; }
    uint64_t ch_addr(unsigned c) const override { return 0x4000000000ull + 0x400000000ull * c; }
    double clock_mhz() const override { return 300.0; }
    bool clock_known() const override { return false; }
    std::string describe() const override {
        return fmt("mock device (C++ model of tsr_bw_k4, no hardware), %.0f MiB per channel%s\n", (double) ch_bytes_ / MiB,
                   fault_.empty() ? "" : (", fault " + fault_).c_str());
    }
    void exec(const RunReq & r, uint32_t, double * wall_s) override {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t total = 16;
        for (unsigned c = 0; c < MAX_CH; ++c) {
            const ChReq & q = r.ch[c];
            ChRes x;
            x.status = ST_DONE;
            const uint64_t lim = 1ull << 34;
            const bool bad = q.en && ((!q.rd && !q.wr) || q.burst < 1 || q.burst > 64 || q.outstanding < 1 || q.outstanding > 64 ||
                                      q.id_log2 > 4 || q.bytes % 64 || (q.rd && q.rd_off % 64) || (q.wr && q.wr_off % 64) ||
                                      (q.rd && q.rd_off + q.bytes > std::min(lim, ch_bytes_)) ||
                                      (q.wr && q.wr_off + q.bytes > std::min(lim, ch_bytes_)));
            if (q.en && bad) x.status |= ST_CFG_ERR;
            if (q.en && !bad) {
                x.status |= ST_RAN;
                const uint64_t nb = q.bytes / 64;
                if (q.rd && nb) {
                    for (uint64_t o = 0; o < q.bytes; o += 64) x.checksum += beat_hash_bytes(&mem_[c][q.rd_off + o]);
                    x.rd_bytes = q.bytes;
                    x.rd_bursts = (uint32_t) burst_count(q.rd_off / 64, nb, q.burst);
                    x.rd_lat = 40;
                    x.rd_cycles = cycles(nb, x.rd_bursts, q.outstanding, x.rd_lat);
                }
                if (q.wr && nb) {
                    for (uint64_t o = 0; o < q.bytes; o += 64) {
                        uint64_t w[8];
                        pattern_beat(r.seed, c, q.wr_off + o, w);
                        if (fault_ == "write" && o == q.bytes / 2) w[3] ^= 1;
                        memcpy(&mem_[c][q.wr_off + o], w, 64);
                    }
                    x.wr_bytes = q.bytes;
                    x.wr_bursts = (uint32_t) burst_count(q.wr_off / 64, nb, q.burst);
                    x.wr_lat = 40 + std::min<unsigned>(q.burst, (unsigned) nb);
                    x.wr_cycles = cycles(nb, x.wr_bursts, q.outstanding, x.wr_lat);
                }
                if (fault_ == "csum") x.checksum ^= 1ull << 17;
                if (fault_ == "status") x.status |= ST_BRESP, x.errs = 1u << 16;
                if (fault_ == "bytes" && x.rd_bytes) x.rd_bytes -= 64;
            }
            total = std::max(total, std::max(x.rd_cycles, x.wr_cycles) + 16);
            put(c, x);
        }
        regs_[TOTAL_CYCLES / 4] = (uint32_t) total;
        regs_[TOTAL_CYCLES / 4 + 1] = (uint32_t) (total >> 32);
        *wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() + (double) total / 300e6 + 30e-6;
    }
    uint32_t reg(uint32_t off) override { return off < REG_RANGE ? regs_[off / 4] : 0; }
    void mem_write(unsigned c, uint64_t off, const void * src, size_t n) override { memcpy(&mem_[c][off], src, n); }
    void mem_read(unsigned c, uint64_t off, void * dst, size_t n) override { memcpy(dst, &mem_[c][off], n); }

private:
    uint64_t ch_bytes_;
    std::vector<std::vector<uint8_t>> mem_;
    std::vector<uint32_t> regs_;
    std::string fault_;

    // a DDR4-like model: 85 % of a beat per cycle when enough is in flight, latency-bound otherwise
    static uint64_t cycles(uint64_t nb, uint64_t bursts, unsigned mo, uint32_t lat) {
        const double per_burst = (double) lat / mo;
        const double c = std::max((double) nb / 0.85, (double) bursts * per_burst);
        return (uint64_t) c + lat;
    }
    void put(unsigned c, const ChRes & x) {
        auto w32 = [&](uint32_t f, uint32_t v) { regs_[res_reg(c, f) / 4] = v; };
        auto w64 = [&](uint32_t f, uint64_t v) { w32(f, (uint32_t) v), w32(f + 4, (uint32_t) (v >> 32)); };
        w32(R_STATUS, x.status);
        w32(R_ERRS, x.errs);
        w64(R_RD_CYCLES, x.rd_cycles);
        w64(R_WR_CYCLES, x.wr_cycles);
        w64(R_RD_BYTES, x.rd_bytes);
        w64(R_WR_BYTES, x.wr_bytes);
        w64(R_CHECKSUM, x.checksum);
        w32(R_RD_LAT, x.rd_lat);
        w32(R_WR_LAT, x.wr_lat);
        w32(R_RD_BURSTS, x.rd_bursts);
        w32(R_WR_BURSTS, x.wr_bursts);
    }
};

// ---------------------------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------------------------
struct Opts {
    std::string xclbin, device = "0", kernel = "tsr_bw_k4", csv, verify = "sample";
    std::vector<uint64_t> sizes = {1 * MiB, 16 * MiB, 256 * MiB, 1 * GiB};
    std::vector<unsigned> bursts = {1, 4, 16, 64};
    std::vector<std::string> modes = {"r", "w", "rw"};
    std::vector<unsigned> channels = {0, 1, 2, 3};
    unsigned outstanding = 32, id_log2 = 0, reps = 3;
    bool all = true, sweep_out = false, pcie = false, mock = false, verbose = false, dry_run = false;
    double clock_mhz = 0;
    uint64_t ch_bytes = 0, seed = 1;
    uint32_t timeout_ms = 60000;
};

struct Tool {
    Opts o;
    BwDev & d;
    FILE * csv = nullptr;
    uint64_t max_size = 0;
    uint32_t fill_seed = 0;
    std::vector<std::vector<uint64_t>> ref_prefix;   // [c][i]: checksum of the first sizes[i] bytes of the read region
    std::mt19937_64 rng;
    unsigned runs = 0, failed = 0;

    Tool(const Opts & opts, BwDev & dev) : o(opts), d(dev), rng(opts.seed) {}

    double gbps(uint64_t bytes, uint64_t cycles) const {
        return cycles ? (double) bytes / ((double) cycles / (d.clock_mhz() * 1e6)) / 1e9 : 0.0;
    }

    // Checks of one run's results against the request; returns "" or the findings.
    std::string check(const RunReq & q, const RunRes & r, const std::vector<uint64_t> & ref_csum) {
        std::string e;
        for (unsigned c = 0; c < d.num_ch(); ++c) {
            const ChReq & x = q.ch[c];
            const ChRes & y = r.ch[c];
            uint32_t want = ST_DONE | (x.en ? ST_RAN : 0);
            if (y.status != want) e += fmt("ch%u status 0x%x (expected 0x%x: %s%s%s%s); ", c, y.status, want,
                                           y.status & ST_CFG_ERR ? "config refused " : "", y.status & ST_RRESP ? "RRESP error " : "",
                                           y.status & ST_BRESP ? "BRESP error " : "", y.status & ST_UNEXPECTED ? "unexpected response" : "");
            if (y.errs) e += fmt("ch%u %u RRESP / %u BRESP error responses; ", c, y.errs & 0xFFFF, y.errs >> 16);
            if (!x.en) continue;
            const uint64_t nb = x.bytes / 64;
            if (x.rd) {
                if (y.rd_bytes != x.bytes) e += fmt("ch%u read %" PRIu64 " bytes, expected %" PRIu64 "; ", c, y.rd_bytes, x.bytes);
                if (y.rd_bursts != burst_count(x.rd_off / 64, nb, x.burst))
                    e += fmt("ch%u %u read bursts, expected %" PRIu64 "; ", c, y.rd_bursts, burst_count(x.rd_off / 64, nb, x.burst));
                if (y.checksum != ref_csum[c])
                    e += fmt("ch%u read checksum 0x%016" PRIx64 ", expected 0x%016" PRIx64 "; ", c, y.checksum, ref_csum[c]);
                if (y.rd_cycles < nb) e += fmt("ch%u rd_cycles %" PRIu64 " < %" PRIu64 " beats; ", c, y.rd_cycles, nb);
            }
            if (x.wr) {
                if (y.wr_bytes != x.bytes) e += fmt("ch%u wrote %" PRIu64 " bytes, expected %" PRIu64 "; ", c, y.wr_bytes, x.bytes);
                if (y.wr_bursts != burst_count(x.wr_off / 64, nb, x.burst))
                    e += fmt("ch%u %u write bursts, expected %" PRIu64 "; ", c, y.wr_bursts, burst_count(x.wr_off / 64, nb, x.burst));
                if (y.wr_cycles < nb) e += fmt("ch%u wr_cycles %" PRIu64 " < %" PRIu64 " beats; ", c, y.wr_cycles, nb);
                e += verify_written(c, x.wr_off, x.bytes, q.seed);
            }
        }
        return e;
    }

    // The written region against the pattern: in full, or sampled (first / last 4 MiB, 8 x 4 MiB inside).
    std::string verify_written(unsigned c, uint64_t off, uint64_t bytes, uint32_t seed) {
        if (o.verify == "none" || bytes == 0) return "";
        std::vector<std::pair<uint64_t, uint64_t>> parts;     // (offset in region, length)
        const uint64_t piece = 4 * MiB;
        if (o.verify == "full" || bytes <= 16 * piece) {
            parts.push_back({0, bytes});
        } else {
            parts.push_back({0, piece});
            parts.push_back({bytes - piece, piece});
            for (int i = 0; i < 8; ++i) parts.push_back({(rng() % ((bytes - piece) / 64)) * 64, piece});
        }
        std::vector<uint8_t> buf;
        uint64_t bad = 0, first = 0, checked = 0;
        for (const auto & p : parts) {
            for (uint64_t done = 0; done < p.second; done += 64 * MiB) {
                const uint64_t n = std::min<uint64_t>(64 * MiB, p.second - done);
                buf.resize(n);
                d.mem_read(c, off + p.first + done, buf.data(), n);
                for (uint64_t b = 0; b < n; b += 64) {
                    uint64_t w[8];
                    pattern_beat(seed, c, off + p.first + done + b, w);
                    if (memcmp(&buf[b], w, 64) != 0 && bad++ == 0) first = off + p.first + done + b;
                }
                checked += n;
            }
        }
        if (o.verbose) printf("    ch%u: verified %.1f MiB of the written region\n", c, (double) checked / MiB);
        return bad ? fmt("ch%u %" PRIu64 " written beats differ from the pattern (first at offset 0x%" PRIx64 "); ", c, bad, first) : "";
    }

    RunRes exec(const RunReq & q) {
        RunRes r;
        d.exec(q, o.timeout_ms, &r.wall_s);
        const double w = r.wall_s;
        r = d.results();
        r.wall_s = w;
        return r;
    }

    // One measurement point: `reps` runs, median by the slowest channel's cycles.
    void point(const char * mode, const std::vector<unsigned> & chans, uint64_t size, unsigned burst, unsigned out) {
        const bool rd = strchr(mode, 'r') != nullptr, wr = strchr(mode, 'w') != nullptr;
        const size_t si = std::find(o.sizes.begin(), o.sizes.end(), size) - o.sizes.begin();
        if (si >= o.sizes.size()) throw std::logic_error("point(): size not in --sizes");
        std::vector<RunRes> rs;
        std::string errs;
        auto add_err = [&](const std::string & e) {     // the same finding in several repetitions: once
            if (errs.find(e) == std::string::npos) errs += e;
        };
        for (unsigned k = 0; k < o.reps; ++k) {
            RunReq q;
            q.seed = (uint32_t) rng() | 1;
            std::vector<uint64_t> ref(d.num_ch(), 0);
            for (unsigned c : chans) {
                q.ch[c] = ChReq{true, rd, wr, burst, out, o.id_log2, size, 0, max_size};
                ref[c] = rd ? ref_prefix[c][si] : 0;
            }
            RunRes r;
            try {
                r = exec(q);
            } catch (const std::exception & ex) {
                add_err(std::string(ex.what()) + "; ");
                ++runs;
                ++failed;
                break;
            }
            ++runs;
            const std::string e = check(q, r, ref);
            if (!e.empty()) {
                ++failed;
                for (size_t p = 0, n; p < e.size(); p = n + 2) {
                    n = e.find("; ", p);
                    if (n == std::string::npos) n = e.size() - 2;
                    add_err(e.substr(p, n + 2 - p));
                }
            }
            rs.push_back(r);
        }
        if (rs.empty()) {
            printf("%-3s %-8s %6s %3u %3u  run failed: %s\n", mode, chans.size() > 1 ? "all" : fmt("ch%u", chans[0]).c_str(),
                   human(size).c_str(), burst, out, errs.c_str());
            return;
        }
        auto worst = [&](const RunRes & r) {
            uint64_t m = 0;
            for (unsigned c : chans) m = std::max({m, r.ch[c].rd_cycles, r.ch[c].wr_cycles});
            return m;
        };
        std::sort(rs.begin(), rs.end(), [&](const RunRes & a, const RunRes & b) { return worst(a) < worst(b); });
        const RunRes & r = rs[rs.size() / 2];
        double agg_rd = 0, agg_wr = 0;
        uint64_t rd_b = 0, wr_b = 0, rd_c = 0, wr_c = 0;
        std::string per;
        for (unsigned c : chans) {
            const ChRes & y = r.ch[c];
            rd_b += y.rd_bytes;
            wr_b += y.wr_bytes;
            rd_c = std::max(rd_c, y.rd_cycles);
            wr_c = std::max(wr_c, y.wr_cycles);
            per += fmt(" %5.2f/%-5.2f", gbps(y.rd_bytes, y.rd_cycles), gbps(y.wr_bytes, y.wr_cycles));
        }
        agg_rd = gbps(rd_b, rd_c);
        agg_wr = gbps(wr_b, wr_c);
        const double wall = (double) (rd_b + wr_b) / r.wall_s / 1e9;
        printf("%-3s %-8s %6s %3u %3u  %6.2f %6.2f %6.2f  %6.1f %%  %7.2f  %6u %6u %s %s\n", mode,
               chans.size() > 1 ? "all" : fmt("ch%u", chans[0]).c_str(), human(size).c_str(), burst, out, agg_rd, agg_wr, agg_rd + agg_wr,
               100.0 * (agg_rd + agg_wr) / (chans.size() * 64.0 * d.clock_mhz() * 1e6 / 1e9 * ((rd && wr) ? 2 : 1)), wall,
               r.ch[chans[0]].rd_lat, r.ch[chans[0]].wr_lat, errs.empty() ? "ok" : "FAIL", o.verbose ? per.c_str() : "");
        if (!errs.empty()) printf("    %s\n", errs.c_str());
        if (csv) {
            for (const RunRes & x : rs) {
                for (unsigned c : chans) {
                    const ChRes & y = x.ch[c];
                    fprintf(csv, "%s,%s,%u,%" PRIu64 ",%u,%u,%u,%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,%u,%" PRIu64
                                 ",%.9f,%.4f,%.4f,%s\n",
                            mode, chans.size() > 1 ? "all" : "one", c, size, burst, out, o.id_log2, d.clock_mhz(), y.rd_bytes,
                            y.rd_cycles, y.wr_bytes, y.wr_cycles, y.rd_lat, y.wr_lat, y.rd_bursts, y.wr_bursts, x.total_cycles, x.wall_s,
                            gbps(y.rd_bytes, y.rd_cycles), gbps(y.wr_bytes, y.wr_cycles), errs.empty() ? "ok" : "fail");
                }
            }
            fflush(csv);
        }
        fflush(stdout);
    }

    // Read regions: filled by the kernel (all channels, largest size, fill_seed); reference checksums.
    bool fill() {
        fill_seed = (uint32_t) rng() | 1;
        RunReq q;
        q.seed = fill_seed;
        for (unsigned c = 0; c < d.num_ch(); ++c) q.ch[c] = ChReq{true, false, true, 64, 32, 0, max_size, 0, 0};
        RunRes r;
        try {
            r = exec(q);
        } catch (const std::exception & e) {
            printf("fill run failed: %s\n", e.what());
            return false;
        }
        const std::string e = check(q, r, std::vector<uint64_t>(d.num_ch(), 0));
        ++runs;
        printf("fill: %s of the pattern (seed 0x%08x) on every channel at once: write %.2f GB/s per channel%s%s\n",
               human(max_size).c_str(), fill_seed, gbps(r.ch[0].wr_bytes, r.ch[0].wr_cycles), e.empty() ? ", verified" : " FAIL: ",
               e.c_str());
        if (!e.empty()) ++failed;
        // reference checksums at every size (prefix sums), one thread per channel
        ref_prefix.assign(d.num_ch(), std::vector<uint64_t>(o.sizes.size(), 0));
        std::vector<std::thread> th;
        for (unsigned c = 0; c < d.num_ch(); ++c) {
            th.emplace_back([this, c]() {
                uint64_t h = 0, off = 0;
                for (size_t i = 0; i < o.sizes.size(); ++i) {
                    for (; off < o.sizes[i]; off += 64) {
                        uint64_t w[8];
                        pattern_beat(fill_seed, c, off, w);
                        h += beat_hash(w);
                    }
                    ref_prefix[c][i] = h;
                }
            });
        }
        for (auto & t : th) t.join();
        return e.empty();
    }

    // Data path self-check independent of the kernel's pattern: host-written random data read by
    // the kernel (checksum), then a kernel-written region read by the host (pattern).
    void selfcheck() {
        const uint64_t n = std::min<uint64_t>(4 * MiB, max_size);
        for (unsigned c = 0; c < d.num_ch(); ++c) {
            std::vector<uint8_t> buf(n);
            for (uint64_t i = 0; i < n; i += 8) {
                const uint64_t v = rng();
                memcpy(&buf[i], &v, 8);
            }
            d.mem_write(c, 0, buf.data(), n);
            uint64_t h = 0;
            for (uint64_t i = 0; i < n; i += 64) h += beat_hash_bytes(&buf[i]);
            RunReq q;
            q.seed = (uint32_t) rng() | 1;
            q.ch[c] = ChReq{true, true, true, 64, 32, 0, n, 0, max_size};
            std::string e;
            try {
                const RunRes r = exec(q);
                std::vector<uint64_t> ref(d.num_ch(), 0);
                ref[c] = h;
                e = check(q, r, ref);
            } catch (const std::exception & ex) {
                e = ex.what();
            }
            ++runs;
            if (!e.empty()) ++failed;
            printf("selfcheck ch%u: host-written %s read by the kernel + kernel-written %s read by the host: %s%s\n", c, human(n).c_str(),
                   human(n).c_str(), e.empty() ? "ok" : "FAIL: ", e.c_str());
        }
    }

    void pcie() {
        const uint64_t n = std::min<uint64_t>(256 * MiB, d.ch_bytes());
        std::vector<uint8_t> buf(n, 0x5A);
        printf("host <-> card DMA (xrt::bo write / read of %s, wall clock):\n", human(n).c_str());
        for (unsigned c = 0; c < d.num_ch(); ++c) {
            auto t0 = std::chrono::steady_clock::now();
            d.mem_write(c, 0, buf.data(), n);
            const double tw = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            t0 = std::chrono::steady_clock::now();
            d.mem_read(c, 0, buf.data(), n);
            const double tr = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("  ch%u: host -> card %.2f GB/s, card -> host %.2f GB/s\n", c, n / tw / 1e9, n / tr / 1e9);
        }
    }

    int main_run() {
        max_size = *std::max_element(o.sizes.begin(), o.sizes.end());
        printf("%s", d.describe().c_str());
        if (!d.clock_known())
            printf("note: kernel clock assumed %.0f MHz (no clock topology in the xclbin, no --clock-mhz)\n", d.clock_mhz());
        printf("theoretical per channel: 64 B x %.1f MHz = %.2f GB/s per direction\n", d.clock_mhz(), 64.0 * d.clock_mhz() / 1e3);
        selfcheck();
        if (o.pcie) pcie();          // before the fill: it overwrites the start of every buffer
        if (!fill()) printf("warning: the read checks below use the fill pattern, which failed its own check\n");
        printf("\n%-3s %-8s %6s %3s %3s  %6s %6s %6s  %8s  %7s  %6s %6s %s\n", "mod", "channels", "size", "bur", "out", "rdGB/s", "wrGB/s",
               "sumGB/s", "of peak", "wallGB/s", "rd_lat", "wr_lat", "check");
        for (const auto & m : o.modes) {
            for (uint64_t s : o.sizes) {
                for (unsigned b : o.bursts) {
                    for (unsigned c : o.channels) point(m.c_str(), {c}, s, b, o.outstanding);
                    if (o.all) {
                        std::vector<unsigned> all;
                        for (unsigned c = 0; c < d.num_ch(); ++c) all.push_back(c);
                        point(m.c_str(), all, s, b, o.outstanding);
                    }
                }
            }
        }
        if (o.sweep_out) {
            // the largest listed size up to 16 MiB (reference checksums exist for listed sizes only)
            uint64_t s = o.sizes.front();
            for (uint64_t z : o.sizes)
                if (z <= 16 * MiB) s = z;
            printf("\noutstanding sweep (%s, channel 0 alone and all channels):\n", human(s).c_str());
            for (const auto & m : o.modes) {
                if (m == "rw") continue;
                for (unsigned b : {64u, 1u}) {
                    for (unsigned out : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
                        point(m.c_str(), {0}, s, b, out);
                        if (o.all) point(m.c_str(), {0, 1, 2, 3}, s, b, out);
                    }
                }
            }
        }
        printf("\n%u runs, %u failed their checks\n", runs, failed);
        return failed ? 1 : 0;
    }
};

void usage() {
    printf("usage: tsr_bw [--xclbin FILE] [--device N|BDF] [--kernel NAME] [--sizes LIST] [--bursts LIST]\n"
           "              [--outstanding N] [--ids N] [--modes r,w,rw] [--channels LIST|none] [--no-all] [--reps N]\n"
           "              [--sweep-outstanding] [--pcie] [--verify sample|full|none] [--clock-mhz F] [--ch-mib N]\n"
           "              [--csv FILE] [--seed N] [--timeout-ms N] [--mock] [--dry-run] [-v]\n"
           "  (header of sw/tools/tsr_bw.cpp and docs/bandwidth-kernel.md section 6)\n");
}

}  // namespace

int main(int argc, char ** argv) {
    Opts o;
    if (const char * x = getenv("TSR_XCLBIN")) o.xclbin = x;
    if (const char * x = getenv("TSR_XRT_DEVICE")) o.device = x;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
                return argv[++i];
            };
            if (a == "--xclbin") o.xclbin = val();
            else if (a == "--device") o.device = val();
            else if (a == "--kernel") o.kernel = val();
            else if (a == "--sizes") {
                o.sizes.clear();
                for (const auto & s : split(val())) o.sizes.push_back(parse_size(s));
            } else if (a == "--bursts") {
                o.bursts.clear();
                for (const auto & s : split(val())) o.bursts.push_back((unsigned) std::stoul(s));
            } else if (a == "--outstanding") o.outstanding = (unsigned) std::stoul(val());
            else if (a == "--ids") {
                const unsigned n = (unsigned) std::stoul(val());
                if (n == 0 || n > 16 || (n & (n - 1))) throw std::runtime_error("--ids must be 1, 2, 4, 8 or 16");
                o.id_log2 = (unsigned) __builtin_ctz(n);
            } else if (a == "--modes") o.modes = split(val());
            else if (a == "--channels") {
                const std::string v = val();
                o.channels.clear();
                if (v != "none")
                    for (const auto & s : split(v)) o.channels.push_back((unsigned) std::stoul(s));
            } else if (a == "--no-all") o.all = false;
            else if (a == "--reps") o.reps = (unsigned) std::stoul(val());
            else if (a == "--sweep-outstanding") o.sweep_out = true;
            else if (a == "--pcie") o.pcie = true;
            else if (a == "--verify") o.verify = val();
            else if (a == "--clock-mhz") o.clock_mhz = std::stod(val());
            else if (a == "--ch-mib") o.ch_bytes = std::stoull(val()) * MiB;
            else if (a == "--csv") o.csv = val();
            else if (a == "--seed") o.seed = std::stoull(val());
            else if (a == "--timeout-ms") o.timeout_ms = (uint32_t) std::stoul(val());
            else if (a == "--mock") o.mock = true;
            else if (a == "--dry-run") o.dry_run = true;
            else if (a == "-v") o.verbose = true;
            else if (a == "-h" || a == "--help") {
                usage();
                return 0;
            } else {
                throw std::runtime_error("unknown option " + a);
            }
        }
        for (const auto & m : o.modes)
            if (m != "r" && m != "w" && m != "rw") throw std::runtime_error("--modes: r, w, rw");
        for (unsigned b : o.bursts)
            if (b < 1 || b > 64) throw std::runtime_error("--bursts: 1..64");
        for (unsigned c : o.channels)
            if (c >= MAX_CH) throw std::runtime_error("--channels: 0..3");
        for (uint64_t s : o.sizes)
            if (s == 0 || s % 64) throw std::runtime_error("--sizes: multiples of 64 bytes");
        if (o.outstanding < 1 || o.outstanding > 64) throw std::runtime_error("--outstanding: 1..64");
        if (o.verify != "sample" && o.verify != "full" && o.verify != "none") throw std::runtime_error("--verify: sample, full, none");
        if (o.reps < 1) throw std::runtime_error("--reps: >= 1");
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_bw: %s\n", e.what());
        usage();
        return 2;
    }
    std::sort(o.sizes.begin(), o.sizes.end());
    const uint64_t max_size = o.sizes.back();
    const uint64_t ch_bytes = o.ch_bytes ? o.ch_bytes : 2 * max_size;
    if (ch_bytes < 2 * max_size || ch_bytes > 16 * GiB) {
        fprintf(stderr, "tsr_bw: the buffer per channel (%s) must hold 2 x the largest size and be at most 16 GiB\n",
                human(ch_bytes).c_str());
        return 2;
    }
    std::unique_ptr<BwDev> dev;
    try {
        if (o.mock) {
            dev.reset(new MockDev(ch_bytes));
        } else {
            if (o.xclbin.empty()) throw std::runtime_error("no xclbin: --xclbin FILE or TSR_XCLBIN (hw/u200/bw/build_bw_xclbin.sh builds it)");
            dev.reset(new XrtDev(o.xclbin, o.device, o.kernel, ch_bytes, o.clock_mhz));
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_bw: %s\n", e.what());
        return 2;
    }
    if (o.dry_run) {
        // open, buffers, one run with every argument written (all channels disabled), no checks
        try {
            printf("%s", dev->describe().c_str());
            RunReq q;
            q.seed = 1;
            for (unsigned c = 0; c < MAX_CH; ++c) q.ch[c] = ChReq{false, true, true, 64, 32, 0, 0, 0, 0};
            double w = 0;
            dev->exec(q, o.timeout_ms, &w);
            printf("dry run: kernel opened, %u buffers of %s, one empty run started and completed (%.1f us)\n", dev->num_ch(),
                   human(dev->ch_bytes()).c_str(), w * 1e6);
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "tsr_bw: dry run: %s\n", e.what());
            return 1;
        }
    }
    Tool t(o, *dev);
    if (!o.csv.empty()) {
        t.csv = fopen(o.csv.c_str(), "w");
        if (!t.csv) {
            fprintf(stderr, "tsr_bw: cannot write %s\n", o.csv.c_str());
            return 2;
        }
        fprintf(t.csv, "mode,set,ch,bytes,burst,outstanding,id_log2,clock_mhz,rd_bytes,rd_cycles,wr_bytes,wr_cycles,rd_lat,wr_lat,"
                       "rd_bursts,wr_bursts,total_cycles,wall_s,rd_gbps,wr_gbps,check\n");
    }
    int rc = 2;
    try {
        rc = t.main_run();
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_bw: %s\n", e.what());
    }
    if (t.csv) fclose(t.csv);
    return rc;
}

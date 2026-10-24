// tsr_transport_xrt.cpp — the "xrt" kdev: one Tesseract kernel (tsr_u200_k1 / tsr_u200_k4) on an
// Alveo card through XRT 2.18 (docs/xrt-host.md). Built only with -DTSR_WITH_XRT=ON; the
// protocol on top (config, exec, timeout -> ABORT, ADAPTER_STATUS) is tsr_kdev.cpp.
//
// Environment (read at open; a malformed value makes the open fail):
//   TSR_XCLBIN             the xclbin to load (required; loading the one already on the card is cheap)
//   TSR_XRT_DEVICE         device index (default 0) or PCIe BDF such as 0000:65:00.1
//   TSR_XRT_KERNEL         kernel name (default: tsr_u200_k4, else tsr_u200_k1, whichever the xclbin has)
//   TSR_XRT_CH_GIB         channel buffer size in GiB, 1..16 (default: the bank size, at most 16 GiB;
//                          shrunk in 256 MiB steps while the allocation fails, down to 1 GiB; with
//                          m2m the channel's staging buffer is allocated first, in the same bank)
//   TSR_XRT_CTRL           run   (default) xrt::run: the XRT scheduler writes the arguments, starts the CU
//                                and catches ap_done
//                          regs  the transport writes the argument registers and AP_CTRL itself and
//                                polls ap_done (the sequence the mock and vkernel kdevs use)
//   TSR_XRT_DMA            direct (default) device-only buffers, xrt::bo::write / read = the xocl
//                                 unmanaged-DMA ioctls from the caller's memory
//                          staged the same ioctls through an aligned bounce buffer (TSR_XRT_DMA_CHUNK_MIB)
//                          m2m    a host-backed staging buffer per channel + xrt::bo::copy (M2M / KDMA)
//   TSR_XRT_DMA_CHUNK_MIB  largest single DMA in MiB, 1..1024 (default 64)
//   TSR_XRT_DMA_ALIGN      alignment imposed on DMA, power of two 1..4096 (default 1; 64 for staged / m2m)
// With XCL_EMULATION_MODE=hw_emu the defaults are TSR_XRT_CH_GIB=3, TSR_XRT_DMA=staged and
// TSR_XRT_DMA_ALIGN=4096 for each of them left unset (xrt_env_defaults, docs/xrt-host.md §9; one
// line on stderr names those applied).
// A failed direct transfer is retried once through the staged path; if that succeeds the
// transport stays on staged (one warning).

#include "tsr_kdev.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace tsr {

namespace {

constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t GiB = 1ull << 30;

enum class ctrl_mode { run, regs };
enum class dma_mode { direct, staged, m2m };

const char * dma_mode_name(dma_mode m) {
    return m == dma_mode::direct ? "direct" : (m == dma_mode::staged ? "staged" : "m2m");
}

std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char * f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

// The environment's value (unset or empty: none), else the emulation-mode default in `emu`
// (xrt_env_defaults), else nullptr.
const char * env_str(const char * name, const std::vector<env_default> & emu = {}) {
    const char * v = getenv(name);
    if (v != nullptr && v[0] != '\0') return v;
    for (const env_default & d : emu) {
        if (strcmp(d.name, name) == 0) return d.value;
    }
    return nullptr;
}

// Unsigned decimal environment value in [lo, hi]; dflt if unset; throws if malformed.
uint64_t env_u64(const char * name, uint64_t lo, uint64_t hi, uint64_t dflt, bool * set = nullptr,
                 const std::vector<env_default> & emu = {}) {
    const char * v = env_str(name, emu);
    if (set) *set = v != nullptr;
    if (v == nullptr) return dflt;
    char * end = nullptr;
    errno = 0;
    const unsigned long long r = strtoull(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || v[0] == '-' || r < lo || r > hi) {
        throw std::runtime_error(fmt("%s=\"%s\" is not an integer in [%" PRIu64 ", %" PRIu64 "]", name, v, lo, hi));
    }
    return r;
}

class xrt_kdev final : public kdev, private dma_raw {
public:
    xrt_kdev();
    ~xrt_kdev() override = default;

    const char * name() const override { return "xrt"; }
    uint32_t num_ch() const override { return (uint32_t) bo_.size(); }
    uint64_t ch_bytes(uint32_t ch) const override { return size_[ch]; }
    uint64_t ch_addr(uint32_t ch) const override { return addr_[ch]; }
    std::string describe() const override { return info_; }
    int reg_read(uint32_t off, uint32_t * val) override;
    int reg_write(uint32_t off, uint32_t val) override;
    int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    int start(const kargs & a) override;
    kwait wait(uint32_t timeout_ms) override;

private:
    int raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    xrt::bo & stage(uint32_t ch);
    bool fall_back(const char * what);

    xrt::device  dev_;
    xrt::xclbin  xb_;
    xrt::uuid    uuid_;
    xrt::kernel  krn_;
    xrt::run     run_;
    std::string  kname_, info_;
    int          a_cmd_addr_ = -1, a_cmd_count_ = -1, a_arg_[8];
    std::vector<int>      a_ch_;
    std::vector<int>      grp_;
    std::vector<xrt::bo>  bo_, stage_;
    std::vector<uint64_t> size_, addr_;
    ctrl_mode ctrl_ = ctrl_mode::run;
    dma_mode  dma_  = dma_mode::direct;
    dma_rules staged_rules_;
    std::unique_ptr<dma_engine> eng_;
    bool fell_back_ = false;
};

xrt_kdev::xrt_kdev() {
    for (int & a : a_arg_) a = -1;
    const char * xclbin = env_str("TSR_XCLBIN");
    if (xclbin == nullptr) throw std::runtime_error("TSR_XCLBIN is not set (path of the Tesseract xclbin)");
    const char * ctl = env_str("TSR_XRT_CTRL");
    if (ctl == nullptr || strcmp(ctl, "run") == 0) ctrl_ = ctrl_mode::run;
    else if (strcmp(ctl, "regs") == 0) ctrl_ = ctrl_mode::regs;
    else throw std::runtime_error(fmt("TSR_XRT_CTRL=\"%s\": must be run or regs", ctl));
    // Emulation-mode defaults (hw_emu: docs/xrt-host.md §9), only for variables left unset.
    const std::vector<env_default> emu = xrt_env_defaults();
    if (!emu.empty()) {
        std::string l;
        for (const env_default & d : emu) l += fmt(" %s=%s", d.name, d.value);
        fprintf(stderr, "tsr_transport(xrt): XCL_EMULATION_MODE=hw_emu, defaults applied:%s (simulator defects, "
                        "docs/xrt-host.md §9; set a variable to override)\n", l.c_str());
    }
    bool gib_emu = false;
    for (const env_default & d : emu) gib_emu = gib_emu || strcmp(d.name, "TSR_XRT_CH_GIB") == 0;
    const char * gib_note = gib_emu ? " (the hw_emu default)" : "";
    const char * dm = env_str("TSR_XRT_DMA", emu);
    if (dm == nullptr || strcmp(dm, "direct") == 0) dma_ = dma_mode::direct;
    else if (strcmp(dm, "staged") == 0) dma_ = dma_mode::staged;
    else if (strcmp(dm, "m2m") == 0) dma_ = dma_mode::m2m;
    else throw std::runtime_error(fmt("TSR_XRT_DMA=\"%s\": must be direct, staged or m2m", dm));
    const uint64_t chunk = env_u64("TSR_XRT_DMA_CHUNK_MIB", 1, 1024, 64) * MiB;
    const uint64_t align = env_u64("TSR_XRT_DMA_ALIGN", 1, 4096, dma_ == dma_mode::direct ? 1 : 64, nullptr, emu);
    bool gib_set = false;
    const uint64_t gib = env_u64("TSR_XRT_CH_GIB", 1, 16, 16, &gib_set, emu);

    // Device and xclbin.
    const std::string devs = env_str("TSR_XRT_DEVICE") ? env_str("TSR_XRT_DEVICE") : "0";
    if (devs.find_first_not_of("0123456789") == std::string::npos) {
        dev_ = xrt::device((unsigned) std::stoul(devs));
    } else {
        dev_ = xrt::device(devs);
    }
    xb_ = xrt::xclbin(std::string(xclbin));
    std::vector<std::string> knames;
    for (const auto & k : xb_.get_kernels()) knames.push_back(k.get_name());
    if (const char * kn = env_str("TSR_XRT_KERNEL")) {
        kname_ = kn;
    } else {
        for (const char * cand : { "tsr_u200_k4", "tsr_u200_k1" }) {
            for (const auto & k : knames) {
                if (kname_.empty() && k == cand) kname_ = k;
            }
        }
    }
    std::string klist;
    for (const auto & k : knames) klist += (klist.empty() ? "" : ", ") + k;
    xrt::xclbin::kernel xk;
    for (const auto & k : xb_.get_kernels()) {
        if (k.get_name() == kname_) xk = k;
    }
    if (kname_.empty() || !xk) {
        throw std::runtime_error(fmt("%s has no kernel %s (kernels: %s)", xclbin,
                                     kname_.empty() ? "tsr_u200_k4 / tsr_u200_k1" : kname_.c_str(), klist.c_str()));
    }
    const auto cus = xk.get_cus();
    if (cus.size() != 1) {
        throw std::runtime_error(fmt("kernel %s has %zu compute units; the transport drives exactly one", kname_.c_str(), cus.size()));
    }
    if (cus[0].get_control_type() != xrt::xclbin::ip::control_type::hs) {
        throw std::runtime_error(fmt("kernel %s is not ap_ctrl_hs", kname_.c_str()));
    }
    if (cus[0].get_size() < KREG_SPACE) {
        throw std::runtime_error(fmt("kernel %s declares an s_axi_control range of 0x%zx; the contract needs 0x%x "
                                     "(the core CSR window is at 0x%x; kernel.xml port range)",
                                     kname_.c_str(), (size_t) cus[0].get_size(), KREG_SPACE, KREG_CSR_WINDOW));
    }

    // Arguments, identified by their register offset (kernel contract); names are free.
    std::vector<int> ch_by_slot(KERNEL_MAX_CH, -1);
    for (const auto & a : xk.get_args()) {
        const uint64_t off = a.get_offset(), sz = a.get_size();
        const int idx = (int) a.get_index();
        if (off == KREG_CMD_ADDR_LO && sz == 8) {
            a_cmd_addr_ = idx;
        } else if (off == KREG_CMD_COUNT && sz == 4) {
            a_cmd_count_ = idx;
        } else if (off >= KREG_ARG0 && off < KREG_ARG0 + 32 && off % 4 == 0 && sz == 4) {
            a_arg_[(off - KREG_ARG0) / 4] = idx;
        } else if (off >= KREG_CH_BASE0 && off < KREG_ADAPTER_STATUS && (off - KREG_CH_BASE0) % 8 == 0 && sz == 8 &&
                   !a.get_mems().empty()) {
            ch_by_slot[(off - KREG_CH_BASE0) / 8] = idx;
        } else {
            throw std::runtime_error(fmt("kernel %s: argument %s (offset 0x%" PRIx64 ", %" PRIu64 " bytes) is not in the "
                                         "kernel contract", kname_.c_str(), a.get_name().c_str(), off, sz));
        }
    }
    bool args_ok = a_cmd_addr_ >= 0 && a_cmd_count_ >= 0;
    for (int a : a_arg_) args_ok = args_ok && a >= 0;
    for (uint32_t c = 0; c < KERNEL_MAX_CH; ++c) {
        if (ch_by_slot[c] >= 0 && c != a_ch_.size()) args_ok = false;   // channel slots must be contiguous from 0
        if (ch_by_slot[c] >= 0) a_ch_.push_back(ch_by_slot[c]);
    }
    if (!args_ok || a_ch_.empty() || a_ch_.size() > TSR_NUM_CH_MAX) {
        throw std::runtime_error(fmt("kernel %s: arguments do not match the kernel contract (cmd_addr u64 at 0x10, cmd_count "
                                     "at 0x18, arg0..7 at 0x20..0x3C, ch0..ch(C-1) buffers at 0x40 + 8c)", kname_.c_str()));
    }

    // The Alveo shim (xocl) of XRT 2.18 runs the legacy flow: load_xclbin programs the card and
    // its register_xclbin is a no-op. XRT's no-op shim (XCL_EMULATION_MODE=noop, used by the
    // transport_xrt_noop ctest) implements only the hw_context flow, i.e. register_xclbin.
    const char * em = getenv("XCL_EMULATION_MODE");
    uuid_ = (em != nullptr && strcmp(em, "noop") == 0) ? dev_.register_xclbin(xb_) : dev_.load_xclbin(xb_);
    krn_ = xrt::kernel(dev_, uuid_, kname_, xrt::kernel::cu_access_mode::exclusive);

    // One device-only buffer per channel in the bank its m_axi port is connected to.
    const auto mems = xb_.get_mems();
    for (size_t c = 0; c < a_ch_.size(); ++c) {
        const int grp = krn_.group_id(a_ch_[c]);   // xcl_bo_flags: bank [15:0], slot [23:16]
        uint64_t bank = KERNEL_CH_BYTES;
        std::string tag = fmt("group %d", grp);
        for (const auto & m : mems) {
            if (m.get_index() == (grp & 0xFFFF)) {
                bank = m.get_size_kb() * 1024ull;
                tag = m.get_tag();
            }
        }
        uint64_t sz = gib_set ? gib * GiB : std::min(bank, KERNEL_CH_BYTES);
        if (gib_set && sz > bank) {
            throw std::runtime_error(fmt("TSR_XRT_CH_GIB=%" PRIu64 "%s exceeds memory %s of channel %zu (%.2f GiB)", gib, gib_note,
                                         tag.c_str(), c, (double) bank / GiB));
        }
        // m2m: the staging buffer (host pages + device memory in this bank) first, so that the
        // default full-bank channel buffer shrinks around it instead of leaving it no room.
        xrt::bo stg;
        if (dma_ == dma_mode::m2m) {
            try {
                stg = xrt::bo(dev_, (size_t) chunk, xrt::bo::flags::normal, (xrt::memory_group) grp);
            } catch (const std::exception & e) {
                throw std::runtime_error(fmt("cannot allocate the %" PRIu64 " MiB m2m staging buffer on %s for channel %zu: %s",
                                             chunk / MiB, tag.c_str(), c, e.what()));
            }
        }
        xrt::bo bo;
        for (;;) {
            try {
                bo = xrt::bo(dev_, sz, xrt::bo::flags::device_only, (xrt::memory_group) grp);
                break;
            } catch (const std::exception & e) {
                if (gib_set || sz <= GiB) {
                    throw std::runtime_error(fmt("cannot allocate %.2f GiB device-only on %s for channel %zu: %s%s", (double) sz / GiB,
                                                 tag.c_str(), c, e.what(), gib_set ? gib_note : " (set TSR_XRT_CH_GIB)"));
                }
                sz -= 256 * MiB;
            }
        }
        grp_.push_back(grp);
        size_.push_back(sz);
        addr_.push_back(bo.address());
        bo_.push_back(std::move(bo));
        stage_.push_back(std::move(stg));
    }

    run_ = xrt::run(krn_);
    for (size_t c = 0; c < bo_.size(); ++c) run_.set_arg(a_ch_[c], bo_[c]);

    dma_rules r;
    r.align = (size_t) align;
    r.max_xfer = (size_t) chunk;
    r.always_bounce = dma_ == dma_mode::staged;
    staged_rules_ = r;
    staged_rules_.always_bounce = true;
    if (staged_rules_.align < 64) staged_rules_.align = 64;
    std::string bad = dma_rules_check(r);
    if (bad.empty()) bad = dma_rules_check(staged_rules_);
    if (!bad.empty()) throw std::runtime_error("TSR_XRT_DMA_*: " + bad);
    eng_.reset(new dma_engine(*this, r));

    std::string devname = "?", bdf = "?";
    try {
        devname = dev_.get_info<xrt::info::device::name>();
        bdf = dev_.get_info<xrt::info::device::bdf>();
    } catch (const std::exception &) {
    }
    info_ = fmt("device %s (%s), xclbin %s (uuid %s), kernel %s, control %s\n", devname.c_str(), bdf.c_str(), xclbin,
                uuid_.to_string().c_str(), kname_.c_str(), ctrl_ == ctrl_mode::run ? "xrt::run" : "registers");
    info_ += fmt("DMA %s, chunk %" PRIu64 " MiB, align %" PRIu64 "%s\n", dma_mode_name(dma_), chunk / MiB, align,
                 dma_ == dma_mode::m2m ? ", a staging buffer of one chunk per channel (host + its bank)" : "");
    for (size_t c = 0; c < bo_.size(); ++c) {
        info_ += fmt("ch%zu: arg %d, memory group %d, %.2f GiB device-only at 0x%" PRIx64 "\n", c, a_ch_[c], grp_[c],
                     (double) size_[c] / GiB, addr_[c]);
    }
}

// read_register / write_register are deprecated in XRT 2.18 in favour of xrt::ip, which cannot be
// combined with xrt::run on the same compute unit; they remain the supported way for an
// exclusive-access kernel (xrt_kernel.cpp get_cuidx_or_error).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
int xrt_kdev::reg_read(uint32_t off, uint32_t * val) {
    try {
        *val = krn_.read_register(off);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): register read at 0x%x failed: %s\n", off, e.what());
        return -1;
    }
}

int xrt_kdev::reg_write(uint32_t off, uint32_t val) {
    try {
        krn_.write_register(off, val);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): register write at 0x%x failed: %s\n", off, e.what());
        return -1;
    }
}
#pragma GCC diagnostic pop

int xrt_kdev::start(const kargs & a) {
    if (ctrl_ == ctrl_mode::regs) return regs_start(*this, a);
    try {
        run_.set_arg(a_cmd_addr_, (uint64_t) a.cmd_addr);
        run_.set_arg(a_cmd_count_, (uint32_t) a.cmd_count);
        for (int i = 0; i < 8; ++i) run_.set_arg(a_arg_[i], (uint32_t) a.arg[i]);
        run_.start();
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): xrt::run start failed: %s\n", e.what());
        return -1;
    }
}

kwait xrt_kdev::wait(uint32_t timeout_ms) {
    if (ctrl_ == ctrl_mode::regs) return regs_wait(*this, timeout_ms);
    try {
        const ert_cmd_state st = run_.wait(std::chrono::milliseconds(timeout_ms));
        if (st == ERT_CMD_STATE_COMPLETED) return kwait::done;
        if (st == ERT_CMD_STATE_TIMEOUT) return kwait::timeout;
        fprintf(stderr, "tsr_transport(xrt): xrt::run ended in state %d (ert_cmd_state)\n", (int) st);
        return kwait::error;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): xrt::run wait failed: %s\n", e.what());
        return kwait::error;
    }
}

xrt::bo & xrt_kdev::stage(uint32_t ch) { return stage_[ch]; }   // allocated at open (m2m only)

int xrt_kdev::raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) {
    try {
        if (dma_ == dma_mode::m2m) {
            xrt::bo & s = stage(ch);
            s.write(src, n, 0);
            s.sync(XCL_BO_SYNC_BO_TO_DEVICE, n, 0);
            bo_[ch].copy(s, n, 0, off);
        } else {
            bo_[ch].write(src, n, off);
        }
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): %s DMA to ch%u + 0x%" PRIx64 " (%zu bytes) failed: %s\n", dma_mode_name(dma_), ch, off, n,
                e.what());
        return -1;
    }
}

int xrt_kdev::raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) {
    try {
        if (dma_ == dma_mode::m2m) {
            xrt::bo & s = stage(ch);
            s.copy(bo_[ch], n, off, 0);
            s.sync(XCL_BO_SYNC_BO_FROM_DEVICE, n, 0);
            s.read(dst, n, 0);
        } else {
            bo_[ch].read(dst, n, off);
        }
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): %s DMA from ch%u + 0x%" PRIx64 " (%zu bytes) failed: %s\n", dma_mode_name(dma_), ch, off,
                n, e.what());
        return -1;
    }
}

// After a failed direct transfer: switch to the staged path once (returns true if switched).
bool xrt_kdev::fall_back(const char * what) {
    if (dma_ != dma_mode::direct || fell_back_) return false;
    fell_back_ = true;
    dma_ = dma_mode::staged;
    eng_.reset(new dma_engine(*this, staged_rules_));
    fprintf(stderr, "tsr_transport(xrt): direct DMA %s failed; retrying through an aligned %zu-byte bounce buffer (TSR_XRT_DMA=staged)\n",
            what, staged_rules_.max_xfer);
    return true;
}

int xrt_kdev::mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) {
    int r = eng_->write(ch, off, src, n);
    if (r != 0 && fall_back("write")) r = eng_->write(ch, off, src, n);
    return r;
}

int xrt_kdev::mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) {
    int r = eng_->read(ch, off, dst, n);
    if (r != 0 && fall_back("read")) r = eng_->read(ch, off, dst, n);
    return r;
}

} // namespace

std::unique_ptr<kdev> open_xrt_kdev() {
    try {
        return std::unique_ptr<kdev>(new xrt_kdev());
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): %s\n", e.what());
        return nullptr;
    }
}

} // namespace tsr

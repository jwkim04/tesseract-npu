// tsr_transport_xrt.cpp — the "xrt" kdev: one Tesseract kernel (tsr_u200_k1 / tsr_u200_k4) on an
// Alveo card through XRT 2.18 (docs/xrt-host.md; docs/design/01-execution-model.md "EX" §9.1,
// docs/design/06-software-verification.md "SW" §9.1). Built only with -DTSR_WITH_XRT=ON; the protocol
// on top (identity, bind, load, boot, launch, wait, recovery) is tsr_kdev.cpp.
//
// The kernel is ap_ctrl_none (EX §9.1) and is opened through xrt::ip, which XRT opens
// with exclusive access (EX §2.4): register access is xrt::ip::read_register / write_register (an
// mmap'ed MMIO access); no xrt::kernel and no xrt::run exist, so a halted RISC-V cannot break
// XRT's scheduler. EX §9.1's fallback form (ap_ctrl_hs with interrupt="false", never started) opens
// the same way. The only kernel arguments are the channel buffers ch<c> at 0x040 + 8c; a kernel
// with other arguments (cmd_addr / cmd_count / arg0-7) is refused at open. The buffer of channel c is a device-only
// xrt::bo in the memory bank of arg ch<c> from xrt::xclbin::arg::get_mems() (EX §9.1; *confirm* on
// the card that this equals the group id of the kernel argument). idle() sleeps (the
// default): the protocol's poll policy and timeouts count host time.
//
// Environment (read at open; a malformed value makes the open fail):
//   TSR_XCLBIN             the xclbin to load (required; loading the one already on the card is cheap)
//   TSR_XRT_DEVICE         device index (default 0) or PCIe BDF such as 0000:65:00.1
//   TSR_XRT_KERNEL         kernel name (default: tsr_u200_k4, else tsr_u200_k1, whichever the xclbin has)
//   TSR_XRT_CH_GIB         channel buffer size in GiB, 1..16 (default: the bank size, at most 16 GiB;
//                          shrunk in 256 MiB steps while the allocation fails, down to 1 GiB; with
//                          m2m the channel's staging buffer is allocated first, in the same bank).
//                          The memory behind a buffer can be smaller than the bank the xclbin names
//                          (the lab U200 has 8 GiB DIMMs under a 16 GB topology, and addresses wrap
//                          at 8 GiB): an owner open measures it (detect_capacity, docs/xrt-host.md
//                          §4.4) and uses the measured size as the channel size. A value above the
//                          measured capacity is refused.
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
#include "tsr_mon.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <experimental/xrt_ip.h>
#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

namespace tsr {

namespace {

constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t GiB = 1ull << 30;

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
    uint64_t buf_bytes(uint32_t ch) const override { return buf_[ch]; }
    uint64_t ch_addr(uint32_t ch) const override { return addr_[ch]; }
    int detect_capacity(bool probe) override;
    std::string describe() const override;
    int reg_read(uint32_t off, uint32_t * val) override;
    int reg_write(uint32_t off, uint32_t val) override;
    int mem_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int mem_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;

private:
    int raw_write(uint32_t ch, uint64_t off, const void * src, size_t n) override;
    int raw_read(uint32_t ch, uint64_t off, void * dst, size_t n) override;
    xrt::bo & stage(uint32_t ch);
    bool fall_back(const char * what);

    xrt::device  dev_;
    xrt::xclbin  xb_;
    xrt::uuid    uuid_;
    xrt::ip      ip_;
    std::string  kname_, cuname_, info_;
    std::vector<int>      a_ch_;
    std::vector<int>      grp_;
    std::vector<xrt::bo>  bo_, stage_;
    std::vector<uint64_t> size_, addr_;   // size_: usable bytes (real memory), at most buf_
    std::vector<uint64_t> buf_;           // the size of the device-only buffer of each channel
    uint64_t gib_override_ = 0;           // TSR_XRT_CH_GIB (0: unset)
    bool whole_ = true;                   // every buffer has the full size the topology gives (no override, no shrink)
    std::string bdf_;                     // PCIe address of the card ("" if unknown)
    std::string cap_how_;                 // how the usable size was found: "", "measured", "recorded"
    dma_mode  dma_  = dma_mode::direct;
    dma_rules staged_rules_;
    std::unique_ptr<dma_engine> eng_;
    bool fell_back_ = false;
};

xrt_kdev::xrt_kdev() {
    const char * xclbin = env_str("TSR_XCLBIN");
    if (xclbin == nullptr) throw std::runtime_error("TSR_XCLBIN is not set (path of the Tesseract xclbin)");
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
    if (gib_set) gib_override_ = gib;

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
    if (cus[0].get_size() < KREG_SPACE) {
        throw std::runtime_error(fmt("kernel %s declares an s_axi_control range of 0x%zx; the contract needs 0x%x "
                                     "(the core CSR window is at 0x%x; kernel.xml port range)",
                                     kname_.c_str(), (size_t) cus[0].get_size(), KREG_SPACE, KREG_CSR_WINDOW));
    }

    // EX §9.1: ap_ctrl_none (the contract), or its fallback ap_ctrl_hs without an interrupt,
    // never started. Both open through xrt::ip; a kernel with an interrupt is not the contract.
    const auto ctl = cus[0].get_control_type();
    const char * ctl_name = ctl == xrt::xclbin::ip::control_type::none ? "ap_ctrl_none" :
                            (ctl == xrt::xclbin::ip::control_type::hs ? "ap_ctrl_hs (EX §9.1 fallback)" : nullptr);
    if (ctl_name == nullptr) {
        throw std::runtime_error(fmt("kernel %s: control protocol %d is not the kernel contract (ap_ctrl_none, or the EX §9.1 "
                                     "fallback ap_ctrl_hs without an interrupt)", kname_.c_str(), (int) ctl));
    }
    cuname_ = cus[0].get_name();

    // Arguments, identified by their register offset (kernel contract, EX §9.1); names are free.
    // The channel buffers are the only arguments: cmd_addr / cmd_count / arg0-7 are refused.
    std::vector<int> ch_by_slot(KERNEL_MAX_CH, -1);
    for (const auto & a : xk.get_args()) {
        const uint64_t off = a.get_offset(), sz = a.get_size();
        const int idx = (int) a.get_index();
        if (off >= KREG_CH_BASE0 && off < KREG_ADAPTER_STATUS && (off - KREG_CH_BASE0) % 8 == 0 && sz == 8 &&
            !a.get_mems().empty()) {
            ch_by_slot[(off - KREG_CH_BASE0) / 8] = idx;
        } else {
            throw std::runtime_error(fmt("kernel %s: argument %s (offset 0x%" PRIx64 ", %" PRIu64 " bytes) is not in the "
                                         "kernel contract (an older xclbin without the control core?)", kname_.c_str(),
                                         a.get_name().c_str(), off, sz));
        }
    }
    bool args_ok = true;
    for (uint32_t c = 0; c < KERNEL_MAX_CH; ++c) {
        if (ch_by_slot[c] >= 0 && c != a_ch_.size()) args_ok = false;   // channel slots must be contiguous from 0
        if (ch_by_slot[c] >= 0) a_ch_.push_back(ch_by_slot[c]);
    }
    if (!args_ok || a_ch_.empty() || a_ch_.size() > TSR_NUM_CH_MAX) {
        throw std::runtime_error(fmt("kernel %s: arguments do not match the kernel contract (ch0..ch(C-1) buffers at "
                                     "0x40 + 8c)", kname_.c_str()));
    }

    // The Alveo shim (xocl) of XRT 2.18 runs the legacy flow: load_xclbin programs the card and
    // its register_xclbin is a no-op. XRT's no-op shim (XCL_EMULATION_MODE=noop, used by the
    // transport_xrt_noop ctest) implements only the hw_context flow, i.e. register_xclbin.
    const char * em = getenv("XCL_EMULATION_MODE");
    uuid_ = (em != nullptr && strcmp(em, "noop") == 0) ? dev_.register_xclbin(xb_) : dev_.load_xclbin(xb_);
    // Exclusive access to the compute unit (EX §2.4): a second process cannot open it.
    try {
        ip_ = xrt::ip(dev_, uuid_, cuname_);
    } catch (const std::exception & e) {
        throw std::runtime_error(fmt("cannot open compute unit %s through xrt::ip (exclusive; is another process using the "
                                     "card?): %s", cuname_.c_str(), e.what()));
    }

    // One device-only buffer per channel in the bank its m_axi port is connected to: the memory
    // index of arg ch<c> (xrt::xclbin::arg::get_mems(), EX §9.1).
    const auto mems = xb_.get_mems();
    std::vector<xrt::xclbin::arg> kargs = xk.get_args();
    for (size_t c = 0; c < a_ch_.size(); ++c) {
        xrt::xclbin::mem m;
        for (const auto & a : kargs) {
            if ((int) a.get_index() == a_ch_[c] && !a.get_mems().empty()) m = a.get_mems().front();
        }
        if (!m) throw std::runtime_error(fmt("kernel %s: arg ch%zu has no memory connection", kname_.c_str(), c));
        const int grp = m.get_index();
        const uint64_t bank = m.get_size_kb() * 1024ull;
        const std::string tag = m.get_tag();
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
        if (gib_set || sz != std::min(bank, KERNEL_CH_BYTES)) whole_ = false;
        grp_.push_back(grp);
        size_.push_back(sz);
        buf_.push_back(sz);
        addr_.push_back(bo.address());
        bo_.push_back(std::move(bo));
        stage_.push_back(std::move(stg));
    }

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
        bdf_ = bdf;
    } catch (const std::exception &) {
    }
    info_ = fmt("device %s (%s), xclbin %s (uuid %s), kernel %s, compute unit %s through xrt::ip (exclusive), %s\n",
                devname.c_str(), bdf.c_str(), xclbin, uuid_.to_string().c_str(), kname_.c_str(), cuname_.c_str(), ctl_name);
    info_ += fmt("DMA %s, chunk %" PRIu64 " MiB, align %" PRIu64 "%s\n", dma_mode_name(dma_), chunk / MiB, align,
                 dma_ == dma_mode::m2m ? ", a staging buffer of one chunk per channel (host + its bank)" : "");
}

// The header lines of the open, then one line per channel: the buffer, and how much of it is memory.
std::string xrt_kdev::describe() const {
    std::string d = info_;
    for (size_t c = 0; c < bo_.size(); ++c) {
        d += fmt("ch%zu: arg %d, memory group %d, %.2f GiB device-only at 0x%" PRIx64, c, a_ch_[c], grp_[c], (double) buf_[c] / GiB, addr_[c]);
        if (size_[c] != buf_[c]) {
            d += fmt(", memory holds %.2f GiB (%s; addresses above it wrap)", (double) size_[c] / GiB, cap_how_.c_str());
        } else if (!cap_how_.empty()) {
            d += fmt(", all of it is memory (%s)", cap_how_.c_str());
        }
        d += "\n";
    }
    return d;
}

// The memory behind a buffer can be smaller than the buffer (the xclbin topology says 16 GB per
// bank, the lab U200 has 8 GiB DIMMs, and an address above 8 GiB then reads and writes the cell
// 8 GiB below). Owner mode measures it with detect_ch_capacity and makes it the channel size, so
// that no heap, KV cache or tensor is placed above it. TSR_ATTACH writes nothing: it takes the
// value an earlier owner recorded for this card (tsr_mon_cap_write), else keeps the buffer size.
int xrt_kdev::detect_capacity(bool probe) {
    constexpr uint64_t MIN_CAP = 1 * GiB;   // smaller sizes are not tested (the channel buffer is at least 1 GiB)
    if (bo_.empty()) return 0;
    if (!probe) {
        uint32_t n = 0;
        uint64_t b = 0;
        if (!bdf_.empty() && tsr_mon_cap_read(bdf_.c_str(), &n, &b) == 0 && n == bo_.size() && b >= MIN_CAP) {
            for (size_t c = 0; c < bo_.size(); ++c) size_[c] = std::min(size_[c], b);
            cap_how_ = "recorded by an earlier open";
        }
        return 0;
    }
    uint64_t smallest = UINT64_MAX;
    for (size_t c = 0; c < bo_.size(); ++c) {
        const uint32_t ch = (uint32_t) c;
        uint64_t cap = 0;
        const int rc = detect_ch_capacity(
            buf_[c], MIN_CAP, [this, ch](uint64_t off, const void * p, size_t n) { return mem_write(ch, off, p, n); },
            [this, ch](uint64_t off, void * p, size_t n) { return mem_read(ch, off, p, n); }, &cap);
        if (rc != 0) {
            fprintf(stderr, "tsr_transport(xrt): cannot measure the capacity of channel %zu\n", c);
            return -1;
        }
        if (gib_override_ != 0 && cap < buf_[c]) {
            fprintf(stderr, "tsr_transport(xrt): TSR_XRT_CH_GIB=%" PRIu64 " is more than channel %zu holds: its memory is %.2f GiB (measured at open; "
                            "the xclbin memory topology says %.2f GiB). Use TSR_XRT_CH_GIB=%" PRIu64 " or less, or leave it unset\n",
                    gib_override_, c, (double) cap / GiB, (double) buf_[c] / GiB, std::max<uint64_t>(cap / GiB, 1));
            return -1;
        }
        size_[c] = cap;
        smallest = std::min(smallest, cap);
    }
    cap_how_ = "measured";
    const char * dbg = getenv("TSR_DEBUG");
    if (dbg != nullptr && atoi(dbg) >= 1) {
        for (size_t c = 0; c < bo_.size(); ++c) {
            fprintf(stderr, "tsr_transport(xrt): ch%zu: buffer %.2f GiB, memory %.2f GiB (measured)\n", c, (double) buf_[c] / GiB,
                    (double) size_[c] / GiB);
        }
    }
    // The card's capacity, for the view of an idle card and for the attach open. Only when the buffers had the full topology
    // size: with TSR_XRT_CH_GIB or a shrunk buffer the measurement cannot see past the buffer.
    if (whole_ && !bdf_.empty()) (void) tsr_mon_cap_write(bdf_.c_str(), (uint32_t) bo_.size(), smallest);
    return 0;
}

// EX §9.1: xrt::ip register access (offsets inside the compute unit's s_axi_control range).
int xrt_kdev::reg_read(uint32_t off, uint32_t * val) {
    try {
        *val = ip_.read_register(off);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): register read at 0x%x failed: %s\n", off, e.what());
        return -1;
    }
}

int xrt_kdev::reg_write(uint32_t off, uint32_t val) {
    try {
        ip_.write_register(off, val);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "tsr_transport(xrt): register write at 0x%x failed: %s\n", off, e.what());
        return -1;
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

// dmi.cpp — libtsr_dbg: the DMI driver (docs/design/05-debug.md "DB" §3.2, §3.5, §3.6, §3.10) and the
// firmware image helper (symbols, DWARF lines, the DB §6.3 contract addresses).
//
// Every access is a core CSR access of the DMI window (tsr_transport_dmi_read / _write, 0xC00 + 4r),
// the same on every kind. Polls (abstract commands, halt, resume) let device time pass between reads
// with tsr_transport_idle: on the simulated kinds that advances the model (a halt request takes effect
// at the hart's next instruction boundary on the time base, sw/sim/include/tsr_dm.h), on xrt it sleeps.

#include "tsr_dbg_lib.hpp"

#include "tsr_elf.h"
#include "tsr_image_int.h"
#include "tsr_dm.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace tsr::dbg {

std::string err_str(int e) {
    char b[96];
    if (e == E_OK) return "OK";
    if ((e & 0xF0) == E_CMD) snprintf(b, sizeof(b), "E%02x (abstract command failed: cmderr %d)", e, e & 7);
    else if ((e & 0xF0) == E_SBA) snprintf(b, sizeof(b), "E%02x (SBA failed: %s)", e, (e & 0xF) == 8 ? "sbbusyerror" : "sberror");
    else if (e == E_DMI) snprintf(b, sizeof(b), "E01 (DMI timeout: no clk_rv, STATUS.DMI_TIMEOUT)");
    else if (e == E_POLICY) snprintf(b, sizeof(b), "E04 (refused by policy)");
    else if (e == E_RUNNING) snprintf(b, sizeof(b), "E05 (target running)");
    else if (e == E_MALFORMED) snprintf(b, sizeof(b), "E06 (malformed)");
    else snprintf(b, sizeof(b), "E%02x (transport error)", e & 0xFF);
    return b;
}

// ---------------------------------------------------------------------------------------------
// DMI
// ---------------------------------------------------------------------------------------------
int dmi::read(uint32_t a, uint32_t * v) {
    accesses++;
    return tsr_transport_dmi_read(t_, a, v) == 0 ? E_OK : E_TRANSPORT;
}

int dmi::write(uint32_t a, uint32_t v) {
    accesses++;
    return tsr_transport_dmi_write(t_, a, v) == 0 ? E_OK : E_TRANSPORT;
}

// A poll that did not complete: STATUS.DMI_TIMEOUT (sticky) names a missing clk_rv (E01).
static int timeout_code(tsr_transport * t) {
    return (tsr_transport_csr_read(t, TSR_CSR_STATUS) & TSR_STATUS_DMI_TIMEOUT) ? E_DMI : E_RUNNING;
}

int dmi::poll_dmstatus(uint32_t mask, uint64_t timeout_ns, uint32_t * out) {
    const uint64_t t0 = tsr_transport_now_ns(t_);
    uint64_t step = 1000;   // 1 µs, doubling to 1 ms of the time base
    for (;;) {
        uint32_t s = 0;
        int e = read(TSR_DMI_DMSTATUS, &s);
        if (e) return e;
        if (out) *out = s;
        if ((s & mask) == mask) return E_OK;
        if (tsr_transport_now_ns(t_) - t0 >= timeout_ns) return timeout_code(t_);
        tsr_transport_idle(t_, step);
        step = std::min<uint64_t>(step * 2, 1000000);
    }
}

int dmi::activate() {
    int e = write(TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_DMACTIVE);
    if (e) return e;
    uint32_t c = 0;
    if ((e = read(TSR_DMI_DMCONTROL, &c)) != 0) return e;
    return (c & TSR_DM_DMCONTROL_DMACTIVE) ? E_OK : E_DMI;
}

int dmi::set_haltreq(bool on) {
    return write(TSR_DMI_DMCONTROL, (on ? TSR_DM_DMCONTROL_HALTREQ : 0u) | TSR_DM_DMCONTROL_DMACTIVE);
}

// DB §3.5: W dmcontrol = 0x8000_0001, poll allhalted, W dmcontrol = 0x0000_0001.
int dmi::halt(uint64_t timeout_ns) {
    int e = set_haltreq(true);
    if (e) return e;
    const int r = poll_dmstatus(TSR_DM_DMSTATUS_ALLHALTED, timeout_ns);
    e = set_haltreq(false);
    return r ? r : e;
}

// DB §3.5: W dmcontrol = 0x4000_0001, poll allresumeack, W dmcontrol = 0x0000_0001.
int dmi::resume(uint64_t timeout_ns) {
    int e = write(TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_RESUMEREQ | TSR_DM_DMCONTROL_DMACTIVE);
    if (e) return e;
    const int r = poll_dmstatus(TSR_DM_DMSTATUS_ALLRESUMEACK, timeout_ns);
    e = write(TSR_DMI_DMCONTROL, TSR_DM_DMCONTROL_DMACTIVE);
    return r ? r : e;
}

// R abstractcs until busy = 0; cmderr != 0: cleared (W 0x700) and E_CMD | cmderr.
int dmi::abstract_wait() {
    const uint64_t t0 = tsr_transport_now_ns(t_);
    uint64_t step = 1000;
    for (;;) {
        uint32_t acs = 0;
        int e = read(TSR_DMI_ABSTRACTCS, &acs);
        if (e) return e;
        if (!(acs & TSR_DM_ABSTRACTCS_BUSY)) {
            const uint32_t err = (acs & TSR_DM_ABSTRACTCS_CMDERR) >> TSR_DM_ABSTRACTCS_CMDERR_LSB;
            if (err == 0) return E_OK;
            write(TSR_DMI_ABSTRACTCS, TSR_DM_ABSTRACTCS_CMDERR);
            return E_CMD | (int) err;
        }
        if (tsr_transport_now_ns(t_) - t0 >= 5000000) return timeout_code(t_);
        tsr_transport_idle(t_, step);
        step = std::min<uint64_t>(step * 2, 100000);
    }
}

// DB §3.5: read W command = 0x0022_0000 + regno; R abstractcs; R data0.
int dmi::reg_read(uint32_t regno, uint32_t * v) {
    int e = write(TSR_DMI_COMMAND, (2u << TSR_DM_CMD_AARSIZE_LSB) | TSR_DM_CMD_TRANSFER | (regno & 0xFFFFu));
    if (e) return e;
    if ((e = abstract_wait()) != 0) return e;
    return read(TSR_DMI_DATA0, v);
}

// DB §3.5: W data0; W command = 0x0023_0000 + regno; R abstractcs.
int dmi::reg_write(uint32_t regno, uint32_t v) {
    int e = write(TSR_DMI_DATA0, v);
    if (e) return e;
    if ((e = write(TSR_DMI_COMMAND, (2u << TSR_DM_CMD_AARSIZE_LSB) | TSR_DM_CMD_TRANSFER | TSR_DM_CMD_WRITE | (regno & 0xFFFFu))) != 0)
        return e;
    return abstract_wait();
}

int dmi::exec_progbuf(const uint32_t * w, unsigned n) {
    if (n + 1 > TSR_DM_PROGBUF_SIZE) return E_MALFORMED;
    int e;
    for (unsigned i = 0; i < n; ++i)
        if ((e = write(TSR_DMI_PROGBUF0 + i, w[i])) != 0) return e;
    if ((e = write(TSR_DMI_PROGBUF0 + n, 0x00100073u)) != 0) return e;   // ebreak
    if ((e = write(TSR_DMI_COMMAND, TSR_DM_CMD_POSTEXEC)) != 0) return e;   // 0x0004_0000: postexec only
    return abstract_wait();
}

// DB §3.5: progbuf0 = fence.i (0x0000_100F), progbuf1 = ebreak.
int dmi::fence_i() {
    const uint32_t w = 0x0000100Fu;
    return exec_progbuf(&w, 1);
}

// R sbcs: sberror / sbbusyerror (W1C on the next sbcs write).
int dmi::sba_check() {
    uint32_t cs = 0;
    int e = read(TSR_DMI_SBCS, &cs);
    if (e) return e;
    if (cs & TSR_DM_SBCS_SBBUSYERROR) {
        write(TSR_DMI_SBCS, TSR_DM_SBCS_SBBUSYERROR | TSR_DM_SBCS_SBERROR);
        return E_SBA | 8;
    }
    const uint32_t err = (cs & TSR_DM_SBCS_SBERROR) >> TSR_DM_SBCS_SBERROR_LSB;
    if (err) {
        write(TSR_DMI_SBCS, TSR_DM_SBCS_SBERROR);
        return E_SBA | (int) err;
    }
    return E_OK;
}

// DB §3.6 read n words: W sbcs = 0x0055_F000; W sbaddress0 = a; n - 1 x R sbdata0; W sbcs =
// 0x0045_7000; R sbdata0 (the last word); R sbcs. On sbbusyerror the block is retried word by word.
int dmi::sba_read32(uint32_t a, uint32_t * w, size_t n) {
    if (n == 0) return E_OK;
    if (a & 3u) return E_SBA | TSR_DM_SBERR_ALIGN;
    int e;
    if ((e = write(TSR_DMI_SBCS, 0x0055F000u)) != 0) return e;
    if ((e = write(TSR_DMI_SBADDRESS0, a)) != 0) return e;
    for (size_t i = 0; i + 1 < n; ++i)
        if ((e = read(TSR_DMI_SBDATA0, &w[i])) != 0) return e;
    if ((e = write(TSR_DMI_SBCS, 0x00457000u)) != 0) return e;
    if ((e = read(TSR_DMI_SBDATA0, &w[n - 1])) != 0) return e;
    e = sba_check();
    if (e == (E_SBA | 8) && n > 1) {
        for (size_t i = 0; i < n; ++i)
            if ((e = sba_read32(a + 4 * (uint32_t) i, &w[i], 1)) != 0) return e;
        return E_OK;
    }
    return e;
}

// DB §3.6 write n words: W sbcs = 0x0045_7000; W sbaddress0 = a; n x W sbdata0; R sbcs.
int dmi::sba_write32(uint32_t a, const uint32_t * w, size_t n) {
    if (n == 0) return E_OK;
    if (a & 3u) return E_SBA | TSR_DM_SBERR_ALIGN;
    int e;
    if ((e = write(TSR_DMI_SBCS, 0x00457000u)) != 0) return e;
    if ((e = write(TSR_DMI_SBADDRESS0, a)) != 0) return e;
    for (size_t i = 0; i < n; ++i)
        if ((e = write(TSR_DMI_SBDATA0, w[i])) != 0) return e;
    e = sba_check();
    if (e == (E_SBA | 8) && n > 1) {
        for (size_t i = 0; i < n; ++i)
            if ((e = sba_write32(a + 4 * (uint32_t) i, &w[i], 1)) != 0) return e;
        return E_OK;
    }
    return e;
}

// DB §3.6 16-bit write: W sbcs = 0x0042_7000; W sbaddress0; W sbdata0; R sbcs.
int dmi::sba_write16(uint32_t a, uint16_t v) {
    int e;
    if ((e = write(TSR_DMI_SBCS, 0x00427000u)) != 0) return e;
    if ((e = write(TSR_DMI_SBADDRESS0, a)) != 0) return e;
    if ((e = write(TSR_DMI_SBDATA0, v)) != 0) return e;
    return sba_check();
}

// The 8-bit form of the same sequence (sbaccess 0).
int dmi::sba_write8(uint32_t a, uint8_t v) {
    int e;
    if ((e = write(TSR_DMI_SBCS, 0x00407000u)) != 0) return e;
    if ((e = write(TSR_DMI_SBADDRESS0, a)) != 0) return e;
    if ((e = write(TSR_DMI_SBDATA0, v)) != 0) return e;
    return sba_check();
}

// ---------------------------------------------------------------------------------------------
// The firmware image
// ---------------------------------------------------------------------------------------------
image_info::~image_info() {
    if (own_) tsr_image_close(own_);
}

bool image_info::load(const tsr_image * img, std::string * why) {
    if (img == nullptr) {
        if (why) *why = "no image";
        return false;
    }
    img_ = img;
    elf_ = tsr::image_elf(img);
    path_ = tsr_image_path(img);
    return true;
}

bool image_info::load_path(const std::string & path, std::string * why) {
    tsr_image * im = nullptr;
    if (tsr_image_open(path.c_str(), &im) != 0 || im == nullptr) {
        if (why) *why = "cannot open the firmware image " + path + " (tsr_image_open)";
        return false;
    }
    if (own_) tsr_image_close(own_);
    own_ = im;
    return load(im, why);
}

bool image_info::symbol(const std::string & name, uint32_t * v) const {
    if (elf_ && tsr_elf_symbol(elf_, name.c_str(), v) == 0) return true;
    auto it = given.find(name);
    if (it != given.end()) {
        *v = it->second;
        return true;
    }
    return false;
}

std::string image_info::symbolize(uint32_t pc) const {
    if (img_) return tsr::image_symbolize(img_, pc);
    char b[16];
    snprintf(b, sizeof(b), "0x%08x", pc);
    return b;
}

std::string image_info::line(uint32_t pc) const { return img_ ? tsr::image_line(img_, pc) : std::string(); }

// image_symbolize gives "0x... func+0x1c (file:line)"; the function is its second word.
std::string image_info::where(uint32_t pc) const {
    const std::string s = symbolize(pc), l = line(pc);
    std::string fn;
    const size_t a = s.find(' ');
    if (a != std::string::npos) fn = s.substr(a + 1, s.find(' ', a + 1) == std::string::npos ? std::string::npos : s.find(' ', a + 1) - a - 1);
    if (!fn.empty() && fn[0] == '(') fn.clear();
    if (l.empty()) return fn.empty() ? s.substr(0, a) : fn;
    return fn.empty() ? l : l + " in " + fn;
}

int image_info::kernel_index(const std::string & name) const { return img_ ? tsr_image_kernel(img_, name.c_str()) : -1; }

std::string image_info::kernel_name(uint32_t index) const {
    const char * n = img_ ? tsr::image_kernel_name(img_, index) : "";
    return n ? n : "";
}

uint32_t image_info::n_kernels() const { return img_ ? tsr::image_n_kernels(img_) : 0; }

uint64_t image_info::image_id() const { return img_ ? tsr_image_id(img_) : 0; }

bool image_info::bytes_at(uint32_t a, uint8_t * out, size_t n) const {
    if (!elf_) return false;
    for (uint32_t i = 0; i < tsr_elf_nseg(elf_); ++i) {
        const tsr_elf_seg * s = tsr_elf_segment(elf_, i);
        if (a >= s->paddr && (uint64_t) a + n <= (uint64_t) s->paddr + s->filesz) {
            memcpy(out, s->data + (a - s->paddr), n);
            return true;
        }
    }
    return false;
}

} // namespace tsr::dbg

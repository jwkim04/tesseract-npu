// target.cpp — libtsr_dbg: the RISC-V debug target (docs/design/05-debug.md "DB" §3.3-§3.9, §4.2,
// §4.4-§4.7, §6.3): attach / detach, run control, registers, memory routing and policy, software and
// hardware breakpoints, RV and SPM watchpoints, stop classification.

#include "tsr_dbg_lib.hpp"

#include "tsr_dm.h"
#include "tsr_logdec.h"   // tsr_mcause_name
#include "tsr_rv.h"       // dcsr.cause values

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace tsr::dbg {

namespace {

constexpr uint32_t CSR_MSTATUS = 0x300, CSR_MEPC = 0x341, CSR_MCAUSE = 0x342, CSR_MTVAL = 0x343;
constexpr uint32_t CSR_TSELECT = 0x7A0, CSR_TDATA1 = 0x7A1, CSR_TDATA2 = 0x7A2, CSR_DCSR = 0x7B0, CSR_DPC = 0x7B1;
constexpr uint32_t DCSR_EBREAKM = 1u << 15, DCSR_STOPCOUNT = 1u << 10, DCSR_STEP = 1u << 2;
constexpr uint32_t MSTATUS_MIE = 1u << 3, MSTATUS_MPIE = 1u << 7;
constexpr uint32_t TDATA1_EXEC = 0x28001044u;   // DB §3.7: type 2, dmode, action = debug mode, m, execute
constexpr uint32_t REG_PC = 32, REG_CSR0 = 65, REG_PRIV = 4161;
constexpr uint32_t DM_LO = TSR_RV_DM_BASE, DM_HI = TSR_RV_DM_BASE + 0x1000u;
constexpr uint32_t WIN_LO = TSR_RV_CTL_BASE, WIN_HI = TSR_RV_CORE_BASE + 0x1000u;   // control + core window
constexpr uint32_t BOOT_PC = TSR_RV_ITCM_BASE + 0x80u;                               // RV_BOOT (0) + 0x80

bool overlaps(uint32_t a, size_t n, uint32_t lo, uint32_t hi) { return n != 0 && a < hi && (uint64_t) a + n > lo; }

// DB §4.4: the CSRs the target description lists (regnum = 65 + number).
struct csr_desc { const char * name; uint32_t num; };
const csr_desc k_csrs[] = {
    { "mstatus", 0x300 }, { "misa", 0x301 }, { "mie", 0x304 }, { "mtvec", 0x305 }, { "mcountinhibit", 0x320 },
    { "mscratch", 0x340 }, { "mepc", 0x341 }, { "mcause", 0x342 }, { "mtval", 0x343 }, { "mip", 0x344 },
    { "mcycle", 0xB00 }, { "minstret", 0xB02 }, { "mcycleh", 0xB80 }, { "minstreth", 0xB82 },
    { "mvendorid", 0xF11 }, { "marchid", 0xF12 }, { "mimpid", 0xF13 }, { "mhartid", 0xF14 },
};
const char * const k_gpr[32] = { "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "fp", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                                 "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6" };

// The CSRs the server owns (DB §4.4: not exposed).
bool server_csr(uint32_t c) { return c == CSR_DCSR || c == CSR_DPC || c == 0x7B2 || c == 0x7B3 || (c >= CSR_TSELECT && c <= 0x7A3); }

} // namespace

target::target(tsr_transport * t, image_info * img, const options & o) : t_(t), img_(img), o_(o), dm_(t) {
    const uint32_t rv_cfg = csr(TSR_CSR_RV_CFG);
    itcm_bytes_ = TSR_GET(rv_cfg, TSR_RV_CFG_ITCM_KIB) * 1024u;
    dtcm_bytes_ = TSR_GET(rv_cfg, TSR_RV_CFG_DTCM_KIB) * 1024u;
    spm_bytes_ = csr(TSR_CSR_CFG1) * 64u;
}

target::~target() = default;

int target::fail(int e, const std::string & why) {
    last_error = err_str(e) + ": " + why;
    fprintf(stderr, "tsr-dbg: %s\n", last_error.c_str());
    return e;
}

bool target::in_dtcm(uint32_t a, size_t n) const {
    return a >= TSR_RV_DTCM_BASE && a - TSR_RV_DTCM_BASE < dtcm_bytes_ && n <= dtcm_bytes_ - (a - TSR_RV_DTCM_BASE);
}

bool target::sym(const char * name, uint32_t * v) const { return img_ != nullptr && img_->symbol(name, v); }

int target::check_dmi_timeout(int e) {
    if (e == E_DMI) degraded_ = true;
    return e;
}

int target::ensure_halted() {
    if (!attached_) return fail(E_POLICY, "not attached");
    if (!halted_) return fail(E_RUNNING, "the hart is running");
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// __tsr_dbg (DB §6.3)
// ---------------------------------------------------------------------------------------------
int target::get_dbg_word(uint32_t word, uint32_t * v) {
    uint32_t a = 0;
    if (!sym("__tsr_dbg", &a)) return fail(E_POLICY, "no __tsr_dbg symbol (load the firmware ELF, or GDB's qSymbol)");
    return dm_.sba_read32(a + 4 * word, v, 1);
}

int target::set_dbg_word(uint32_t word, uint32_t v) {
    uint32_t a = 0;
    if (!sym("__tsr_dbg", &a)) return fail(E_POLICY, "no __tsr_dbg symbol (load the firmware ELF, or GDB's qSymbol)");
    uint8_t b[4] = { (uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24) };
    return raw_write(a + 4 * word, b, 4);
}

// ---------------------------------------------------------------------------------------------
// DB §4.2 ATTACH / DETACH
// ---------------------------------------------------------------------------------------------
int target::attach(std::string * why) {
    notes_.clear();
    auto bad = [&](int e, const std::string & w) {
        if (why) *why = w;
        return fail(e, w);
    };
    // An image built without the DB block (HAS_DB 0: the maximum-clock card image) has no DMI window, so no debugger attaches.
    if (csr(TSR_CSR_DBG_ID) == 0)
        return bad(E_POLICY, "DBG_ID = 0: this image has no DB block (HAS_DB 0), so no DMI window and no debugger (DB §3.2, §10.1)");
    const uint32_t ctl = csr(TSR_CSR_CTL_ID);
    if (!(ctl & TSR_CTL_ID_DM)) return bad(E_POLICY, "CTL_ID.DM = 0: the core has no debug module (DB §3.1)");
    if (!(csr(TSR_CSR_STATUS) & TSR_STATUS_RV_CLK_OK)) return bad(E_DMI, "STATUS.RV_CLK_OK = 0: no clk_rv, the DM cannot answer");
    saved_dbg_cfg_ = csr(TSR_CSR_DBG_CFG);
    csr_write(TSR_CSR_DBG_CFG, saved_dbg_cfg_ | TSR_DBG_CFG_DMI_EN | TSR_DBG_CFG_RV_HALT_EN);
    tsr_transport_set_debugger(t_, 1);
    int e = dm_.activate();
    if (e) return bad(check_dmi_timeout(e), "dmcontrol.dmactive does not read back");
    uint32_t ds = 0, acs = 0, sbcs = 0;
    if ((e = dm_.read(TSR_DMI_DMSTATUS, &ds)) || (e = dm_.read(TSR_DMI_ABSTRACTCS, &acs)) || (e = dm_.read(TSR_DMI_SBCS, &sbcs)))
        return bad(e, "DMI read failed");
    char b[160];
    if ((ds & TSR_DM_DMSTATUS_VERSION) != 2) {
        snprintf(b, sizeof(b), "dmstatus 0x%08x: version %u, expected 2 (Debug Spec 0.13)", ds, ds & 0xFu);
        return bad(E_DMI, b);
    }
    if ((acs >> TSR_DM_ABSTRACTCS_PROGBUFSIZE_LSB & 0x1Fu) < 2) {
        snprintf(b, sizeof(b), "abstractcs 0x%08x: progbufsize < 2 (fence.i needs 2)", acs);
        return bad(E_DMI, b);
    }
    if ((sbcs >> TSR_DM_SBCS_SBVERSION_LSB) != 1 || !(sbcs & 4u)) {
        snprintf(b, sizeof(b), "sbcs 0x%08x: SBA without sbversion 1 / sbaccess32 (DB §3.6)", sbcs);
        return bad(E_DMI, b);
    }
    if (ds & TSR_DM_DMSTATUS_ALLUNAVAIL) {
        // Held in reset (RV_RUN = 0): the haltreq level stays, so the hart halts before its first
        // instruction when released (tsr_dm.h); registers answer cmderr 4 meanwhile.
        dm_.set_haltreq(true);
        notes_.push_back("the hart is held in reset (RV_RUN = 0): it halts at its first instruction when released "
                         "(boot an image, or monitor load / reset halt)");
        boot_halt_pending_ = true;
    } else if (!(ds & TSR_DM_DMSTATUS_ALLHALTED)) {
        e = dm_.halt(o_.halt_timeout_ns);
        if (e) {
            std::string s;
            rv_sample(&s);
            return bad(check_dmi_timeout(e), "the hart does not halt within 5 ms (DB §3.4): " + s);
        }
    }
    halted_ = true;
    attached_ = true;
    interrupted_ = false;
    // dcsr.ebreakm = 1, stopcount = 1 (if writable)
    uint32_t dcsr = 0;
    if (!(ds & TSR_DM_DMSTATUS_ALLUNAVAIL) && dm_.reg_read(CSR_DCSR, &dcsr) == 0) {
        dm_.reg_write(CSR_DCSR, (dcsr | DCSR_EBREAKM | DCSR_STOPCOUNT) & ~DCSR_STEP);
        uint32_t back = 0;
        dm_.reg_read(CSR_DCSR, &back);
        if (!(back & DCSR_EBREAKM)) notes_.push_back("dcsr.ebreakm did not stick: software breakpoints trap instead");
        if (!(back & DCSR_STOPCOUNT)) notes_.push_back("dcsr.stopcount reads 0 (Ibex): mcycle / minstret count in debug mode");
    }
    // HOLD_ON_HALT (DB §3.4), STOP_HALT_RV (DB §4.2)
    const uint32_t ctrl = csr(TSR_CSR_CTRL);
    saved_ctrl_hold_ = ctrl & TSR_CTRL_HOLD_ON_HALT;
    const uint32_t levels = ctrl & (TSR_CTRL_RV_RUN | TSR_CTRL_HOST_HOLD | TSR_CTRL_HOLD_ON_HALT | TSR_CTRL_IRQ_EN);
    csr_write(TSR_CSR_CTRL, o_.hold_on_halt ? (levels | TSR_CTRL_HOLD_ON_HALT) : (levels & ~TSR_CTRL_HOLD_ON_HALT));
    csr_write(TSR_CSR_DBG_CFG, csr(TSR_CSR_DBG_CFG) | TSR_DBG_CFG_STOP_HALT_RV);
    if (csr(TSR_CSR_DBG_CFG) & TSR_DBG_CFG_HALT_ON_BOOT) boot_halt_pending_ = true;
    // __tsr_dbg.attached when the build id matches (DB §4.2, §6.3); 0 before crt0 ran (halt-on-boot).
    uint32_t a = 0;
    if (!sym("__tsr_dbg", &a)) {
        notes_.push_back("no __tsr_dbg symbol: launch-break and the trap halt need the firmware ELF (--elf, or qSymbol)");
    } else {
        uint32_t w[3] = {};
        e = dm_.sba_read32(a, w, 3);
        const uint32_t id = img_ ? (uint32_t) img_->image_id() : 0;
        if (e) {
            notes_.push_back("__tsr_dbg not readable: " + err_str(e));
        } else if (w[0] != TSR_DBG_INFO_MAGIC) {
            snprintf(b, sizeof(b), "__tsr_dbg magic 0x%08x (expected \"TDBG\"): the device runs another image", w[0]);
            notes_.push_back(b);
        } else if (w[2] != 0 && id != 0 && w[2] != id) {
            snprintf(b, sizeof(b), "__tsr_dbg build id 0x%08x, the ELF's is 0x%08x: the device runs another image; attached not set", w[2], id);
            notes_.push_back(b);
        } else {
            set_dbg_word(TSR_DBG_INFO_ATTACHED_WORD, 1);
        }
    }
    for (const std::string & n : notes_) fprintf(stderr, "tsr-dbg: %s\n", n.c_str());
    return E_OK;
}

int target::detach(bool leave_halted) {
    if (!attached_) return E_OK;
    if (!halted_) {
        dm_.halt(o_.halt_timeout_ns);
        halted_ = true;
        stepping_ = false;
    }
    for (auto & kv : sw_) raw_write(kv.second.addr, kv.second.orig, kv.second.kind);
    sw_.clear();
    for (uint32_t i = 0; i < 2; ++i)
        if (hw_[i].used) {
            dm_.reg_write(CSR_TSELECT, i);
            dm_.reg_write(CSR_TDATA1, 0);
            hw_[i] = hwbp();
        }
    bool wps = false;
    for (uint32_t j = 0; j < 2; ++j)
        if (wp_[j].used) {
            csr_write(TSR_CSR_DBG_RVWP_CTL(j), 0);
            wp_[j] = wp();
            wps = true;
        }
    if (wps) rvwp_apply();
    for (uint32_t i = 0; i < TSR_DBG_NTBP; ++i)
        if (session_tbps_ & (1u << i)) csr_write(TSR_CSR_DBG_TBP_CTL(i), 0);
    session_tbps_ = 0;
    csr_write(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RELEASE);
    uint32_t dcsr = 0;
    if (dm_.reg_read(CSR_DCSR, &dcsr) == 0) dm_.reg_write(CSR_DCSR, dcsr & ~(DCSR_EBREAKM | DCSR_STEP));
    uint32_t a = 0;
    if (sym("__tsr_dbg", &a)) {
        set_dbg_word(TSR_DBG_INFO_ATTACHED_WORD, 0);
        set_dbg_word(TSR_DBG_INFO_LAUNCH_BREAK_WORD, 0);
    }
    const uint32_t ctrl = csr(TSR_CSR_CTRL);
    const uint32_t levels = ctrl & (TSR_CTRL_RV_RUN | TSR_CTRL_HOST_HOLD | TSR_CTRL_IRQ_EN);
    csr_write(TSR_CSR_CTRL, levels | saved_ctrl_hold_);
    // DB §4.2: RV_HALT_EN = 0 (and the halt sources the session armed); DMI_EN = 0 last.
    csr_write(TSR_CSR_DBG_CFG, csr(TSR_CSR_DBG_CFG) & ~(TSR_DBG_CFG_RV_HALT_EN | TSR_DBG_CFG_STOP_HALT_RV | TSR_DBG_CFG_HALT_ON_BOOT));
    int e = E_OK;
    if (!leave_halted) {
        prepare_resume();
        e = dm_.resume(o_.halt_timeout_ns);
        halted_ = false;
    }
    csr_write(TSR_CSR_DBG_CFG, csr(TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_DMI_EN);
    tsr_transport_set_debugger(t_, 0);
    attached_ = false;
    return e;
}

// ---------------------------------------------------------------------------------------------
// Run control
// ---------------------------------------------------------------------------------------------
// Before a resume: fence.i when ITCM was written (DB §3.4), and past a firmware ebreak (the launch
// halt, an ebreak of the code) that would otherwise stop again at once.
int target::prepare_resume() {
    int e = E_OK;
    if (itcm_dirty_) {
        e = dm_.fence_i();
        if (e) fail(e, "fence.i from the program buffer failed");
        itcm_dirty_ = false;
    }
    if (skip_ebreak_) {
        uint32_t dpc = 0;
        if (dm_.reg_read(CSR_DPC, &dpc) == 0) {
            uint8_t h[2] = {};
            read_mem(dpc, h, 2);
            const uint32_t len = ((h[0] | h[1] << 8) == 0x9002u) ? 2u : 4u;   // c.ebreak or ebreak
            dm_.reg_write(CSR_DPC, dpc + len);
        }
        skip_ebreak_ = false;
    }
    return e;
}

int target::resume() {
    int e = ensure_halted();
    if (e) return e;
    prepare_resume();
    e = dm_.resume(o_.halt_timeout_ns);
    if (e) return fail(check_dmi_timeout(e), "resumereq: no resume ack");
    halted_ = false;
    return E_OK;
}

int target::start_step() {
    int e = ensure_halted();
    if (e) return e;
    uint32_t dcsr = 0;
    if ((e = dm_.reg_read(CSR_DCSR, &dcsr)) != 0) return fail(e, "dcsr read");
    if ((e = dm_.reg_write(CSR_DCSR, dcsr | DCSR_STEP)) != 0) return fail(e, "dcsr.step");
    prepare_resume();
    stepping_ = true;
    e = dm_.resume(o_.halt_timeout_ns);
    if (e) return fail(check_dmi_timeout(e), "resumereq: no resume ack");
    halted_ = false;
    return E_OK;
}

int target::interrupt() {
    if (!attached_) return E_POLICY;
    interrupted_ = true;
    return dm_.set_haltreq(true);
}

bool target::poll_halted() {
    if (halted_) return true;
    const uint32_t s = csr(TSR_CSR_STATUS);   // on the simulated kinds this read runs the model (EX §10.2)
    if (s & TSR_STATUS_DMI_TIMEOUT) degraded_ = true;
    uint32_t ds = 0;
    if (dm_.read(TSR_DMI_DMSTATUS, &ds) != 0) return false;
    return (ds & TSR_DM_DMSTATUS_ALLHALTED) != 0;
}

int target::step(stop_info * si, uint64_t timeout_ns) {
    int e = start_step();
    if (e) return e;
    const uint64_t t0 = tsr_transport_now_ns(t_);
    uint64_t dt = 1000;
    while (!poll_halted()) {
        if (tsr_transport_now_ns(t_) - t0 >= timeout_ns) return E_RUNNING;
        tsr_transport_idle(t_, dt);
        dt = std::min<uint64_t>(dt * 2, 100000);
    }
    return stop(si);
}

int target::rv_sample(std::string * text) {
    csr_write(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RV_SAMPLE);
    const uint32_t st = csr(TSR_CSR_DBG_RV_STATE), pc = csr(TSR_CSR_DBG_RV_PC), npc = csr(TSR_CSR_DBG_RV_NPC),
                   daddr = csr(TSR_CSR_DBG_RV_DADDR), epc = csr(TSR_CSR_DBG_RV_EPC), eaddr = csr(TSR_CSR_DBG_RV_EADDR);
    char b[320];
    snprintf(b, sizeof(b), "RV sample %u%s: pc %s, npc 0x%08x, last data address 0x%08x, epc 0x%08x, eaddr 0x%08x, state 0x%08x%s%s%s",
             TSR_GET(st, TSR_DBG_RV_STATE_SAMPLE_SEQ), (st & TSR_DBG_RV_STATE_SAMPLE_VALID) ? "" : " (not valid)",
             img_ ? img_->symbolize(pc).c_str() : "?", npc, daddr, epc, eaddr, st, (st & TSR_DBG_RV_STATE_HALTED) ? " halted" : "",
             (st & TSR_DBG_RV_STATE_SLEEP) ? " sleep" : "", (st & TSR_DBG_RV_STATE_IN_RESET) ? " in-reset" : "");
    if (text) *text = b;
    return (st & TSR_DBG_RV_STATE_SAMPLE_VALID) ? E_OK : E_DMI;
}

// DB §4.7: the hart is halted; find out why.
int target::stop(stop_info * si) {
    *si = stop_info();
    halted_ = true;
    uint32_t dcsr = 0, dpc = 0;
    int e = dm_.reg_read(CSR_DCSR, &dcsr);
    if (!e) e = dm_.reg_read(CSR_DPC, &dpc);
    if (stepping_) {
        if (!e) dm_.reg_write(CSR_DCSR, dcsr & ~DCSR_STEP);
        stepping_ = false;
    }
    if (interrupted_) dm_.set_haltreq(false);
    if (e) {
        interrupted_ = false;
        return fail(check_dmi_timeout(e), "dcsr / dpc not readable");
    }
    const uint32_t cause = (dcsr >> 6) & 7u;
    si->dcsr_cause = cause;
    si->pc = dpc;
    skip_ebreak_ = false;
    char b[512];
    uint32_t launch_halt = 0, trap_halt = 0;
    const bool have_lh = sym("__tsr_launch_halt", &launch_halt), have_th = sym("__tsr_trap_halt", &trap_halt);
    const uint32_t dbs = csr(TSR_CSR_DBG_STATUS);
    if (cause == TSR_RV_DBG_CAUSE_EBREAK) {
        if (sw_.count(dpc)) {
            si->reason = "swbreak";
        } else if (have_lh && dpc == launch_halt) {
            uint32_t k = 0, s = 0;
            dm_.reg_read(TSR_DM_REGNO_GPR(10), &k);
            dm_.reg_read(TSR_DM_REGNO_GPR(11), &s);
            const std::string kn = img_ ? img_->kernel_name(k) : std::string();
            snprintf(b, sizeof(b), "launch %s seq %u (kernel %u; a2 = the tsr_args; continue to enter the kernel)",
                     kn.empty() ? "?" : kn.c_str(), s, k);
            si->note = b;
            skip_ebreak_ = true;
        } else if (have_th && dpc == trap_halt) {
            // The crash view (DB §4.7): dpc <- mepc, and the effect of mret on mstatus.
            uint32_t mepc = 0, mcause = 0, mtval = 0, mstatus = 0;
            dm_.reg_read(CSR_MEPC, &mepc);
            dm_.reg_read(CSR_MCAUSE, &mcause);
            dm_.reg_read(CSR_MTVAL, &mtval);
            dm_.reg_read(CSR_MSTATUS, &mstatus);
            dm_.reg_write(CSR_DPC, mepc);
            mstatus = (mstatus & ~MSTATUS_MIE) | ((mstatus & MSTATUS_MPIE) ? MSTATUS_MIE : 0u) | MSTATUS_MPIE;
            dm_.reg_write(CSR_MSTATUS, mstatus);
            const uint32_t c = mcause & 0x7FFFFFFFu;
            if (mcause & 0x80000000u) si->signal = 5;
            else if (c == 0 || c == 4 || c == 6) si->signal = 7;
            else if (c == 1 || c == 5 || c == 7) si->signal = 11;
            else if (c == 2) si->signal = 4;
            else if (c == 24) si->signal = 6;
            else si->signal = 5;
            snprintf(b, sizeof(b), "trap: %s (mcause %u) at %s, mtval 0x%08x (the faulting context; DB §4.7)", tsr_mcause_name(mcause),
                     mcause, img_ ? img_->symbolize(mepc).c_str() : "?", mtval);
            si->note = b;
            si->pc = mepc;
        } else {
            si->note = "ebreak in the firmware at " + (img_ ? img_->symbolize(dpc) : std::string("?"));
            skip_ebreak_ = true;
        }
    } else if (cause == TSR_RV_DBG_CAUSE_TRIGGER) {
        si->reason = "hwbreak";
    } else if (cause == TSR_RV_DBG_CAUSE_HALTREQ) {
        const uint32_t hits = TSR_GET(dbs, TSR_DBG_STATUS_RVWP_HIT);
        const uint32_t bp_hit = TSR_GET(dbs, TSR_DBG_STATUS_BP_HIT);
        int wj = -1;
        for (uint32_t j = 0; j < 2 && wj < 0; ++j)
            if ((hits & (1u << j)) && wp_[j].used) wj = (int) j;
        for (uint32_t j = 0; j < 2 && wj < 0; ++j)
            if (wp_[j].used && wp_[j].tbp >= 0 && (dbs & TSR_DBG_STATUS_STOP_ACTIVE) && (bp_hit & (1u << wp_[j].tbp))) wj = (int) j;
        if (wj >= 0) {
            const wp & w = wp_[wj];
            si->reason = w.type == 2 ? "watch" : w.type == 3 ? "rwatch" : "awatch";
            si->watch_addr = (hits & (1u << wj)) ? csr(TSR_CSR_DBG_RV_WPADDR) : w.addr;
            if (hits) csr_write(TSR_CSR_DBG_STATUS, TSR_PUT(TSR_DBG_STATUS_RVWP_HIT, hits));   // W1C
        }
        if (dbs & TSR_DBG_STATUS_STOP_ACTIVE) {
            const uint32_t p = csr(TSR_CSR_DBG_STOP_PIDX);
            const uint32_t c = TSR_GET(dbs, TSR_DBG_STATUS_CAUSE);
            std::string cs;
            static const char * const cn[5] = { "NOW", "BP", "STEP", "ERR", "HUNG" };
            for (uint32_t i = 0; i < 5; ++i)
                if (c & (1u << i)) cs += cs.empty() ? cn[i] : std::string("|") + cn[i];
            hist_entry h;
            hist_read(t_, p, &h);
            if (h.present) {
                // The push's file:line (the innermost inlined line) and its function (DB §4.7, §5.6).
                const std::string where = img_ ? img_->where(h.pc) : std::string("?");
                snprintf(b, sizeof(b), "tile stop at pidx %u (%s): %s, pushed at %s", p, cs.empty() ? "?" : cs.c_str(), op_disasm(h.w).c_str(),
                         where.c_str());
            } else {
                snprintf(b, sizeof(b), "tile stop at pidx %u (%s): not pushed yet (DBG_NEXT_PIDX %u)", p, cs.empty() ? "?" : cs.c_str(),
                         csr(TSR_CSR_DBG_NEXT_PIDX));
            }
            si->note = b;
        } else if (boot_halt_pending_ && dpc == BOOT_PC) {
            si->note = "halted at boot";
        } else if (interrupted_ && si->reason.empty()) {
            si->signal = 2;
        }
    }
    if (boot_halt_pending_ && dpc == BOOT_PC) {
        // One request, one halt (DB §3.3): HALT_ON_BOOT is not left armed for later resets.
        csr_write(TSR_CSR_DBG_CFG, csr(TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_HALT_ON_BOOT);
        // crt0 keeps what a debugger writes before it runs (DB §3.8): attached for the new boot.
        if (sym("__tsr_dbg", &launch_halt)) set_dbg_word(TSR_DBG_INFO_ATTACHED_WORD, 1);
        boot_halt_pending_ = false;
    }
    interrupted_ = false;
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// Registers (DB §4.4)
// ---------------------------------------------------------------------------------------------
int target::read_crash_reg(uint32_t regnum, uint32_t * v) {
    uint32_t a = 0;
    if (!sym("__tsr_crash", &a)) return fail(E_POLICY, "crash-view: no __tsr_crash symbol");
    if (regnum == 0) {
        *v = 0;
        return E_OK;
    }
    const uint32_t word = regnum == REG_PC ? TSR_CRASH_MEPC_WORD : TSR_CRASH_X1_WORD + regnum - 1;
    return dm_.sba_read32(a + 4 * word, v, 1);
}

int target::read_reg(uint32_t regnum, uint32_t * v) {
    int e = ensure_halted();
    if (e) return e;
    if (crash_view && regnum <= REG_PC) return read_crash_reg(regnum, v);
    if (regnum == 0) {
        *v = 0;
        return E_OK;
    }
    if (regnum < 32) e = dm_.reg_read(TSR_DM_REGNO_GPR(regnum), v);
    else if (regnum == REG_PC) e = dm_.reg_read(CSR_DPC, v);
    else if (regnum == REG_PRIV) {
        *v = 3;   // M only (DB §4.4)
        return E_OK;
    } else if (regnum >= REG_CSR0 && regnum < REG_CSR0 + 4096) {
        const uint32_t c = regnum - REG_CSR0;
        if (server_csr(c)) return fail(E_POLICY, "dcsr, dpc, dscratch, tselect, tdata are the server's (DB §4.4)");
        e = dm_.reg_read(c, v);
    } else {
        return fail(E_MALFORMED, "no register " + std::to_string(regnum));
    }
    return e ? check_dmi_timeout(e) : E_OK;
}

int target::write_reg(uint32_t regnum, uint32_t v) {
    int e = ensure_halted();
    if (e) return e;
    if (crash_view && regnum <= REG_PC) return fail(E_POLICY, "crash-view is on: the saved trap registers are read-only");
    if (regnum == 0 || regnum == REG_PRIV) return E_OK;
    if (regnum < 32) {
        e = dm_.reg_write(TSR_DM_REGNO_GPR(regnum), v);
    } else if (regnum == REG_PC) {
        uint32_t dpc = 0;
        dm_.reg_read(CSR_DPC, &dpc);
        if (dpc != v) {
            fprintf(stderr, "tsr-dbg: pc moved from 0x%08x to 0x%08x: an inferior call that issues tile ops corrupts the interrupted "
                    "push's staging (DB §3.9); unsupported\n", dpc, v);
            skip_ebreak_ = false;
        }
        e = dm_.reg_write(CSR_DPC, v);
    } else if (regnum >= REG_CSR0 && regnum < REG_CSR0 + 4096) {
        const uint32_t c = regnum - REG_CSR0;
        if (server_csr(c)) return fail(E_POLICY, "dcsr, dpc, dscratch, tselect, tdata are the server's (DB §4.4)");
        e = dm_.reg_write(c, v);
    } else {
        return fail(E_MALFORMED, "no register " + std::to_string(regnum));
    }
    return e ? check_dmi_timeout(e) : E_OK;
}

int target::read_gprs(uint32_t v[33]) {
    for (uint32_t i = 0; i <= REG_PC; ++i) {
        const int e = read_reg(i, &v[i]);
        if (e) return e;
    }
    return E_OK;
}

// ---------------------------------------------------------------------------------------------
// Memory (DB §4.5 routing, §3.6 policy)
// ---------------------------------------------------------------------------------------------
int target::read_mem(uint32_t a, uint8_t * b, size_t n) {
    if (n == 0) return E_OK;
    if ((uint64_t) a + n > 0x100000000ull) return fail(E_MALFORMED, "range wraps");
    if (overlaps(a, n, DM_LO, DM_HI)) return fail(E_POLICY, "the debug module range is not readable through SBA (DB §3.6)");
    const uint32_t a0 = a & ~3u;
    const uint64_t end = ((uint64_t) a + n + 3) & ~3ull;
    std::vector<uint32_t> w;
    for (uint64_t p = a0; p < end;) {
        const size_t k = (size_t) std::min<uint64_t>((end - p) / 4, 256);
        w.resize(k);
        const int e = dm_.sba_read32((uint32_t) p, w.data(), k);
        if (e) return check_dmi_timeout(e);
        for (size_t i = 0; i < k; ++i)
            for (uint32_t j = 0; j < 4; ++j) {
                const uint64_t ba = p + 4 * i + j;
                if (ba >= a && ba < (uint64_t) a + n) b[ba - a] = (uint8_t) (w[i] >> (8 * j));
            }
        p += 4 * k;
    }
    return E_OK;
}

// EX's loader while RV_RUN = 0 (DB §3.6: SBA TCM writes are refused then): read-modify-write of
// each touched word through LOAD_ADDR / LOAD_DATA.
int target::load_write(uint32_t a, const uint8_t * b, size_t n) {
    const uint32_t a0 = a & ~3u;
    for (uint64_t p = a0; p < (uint64_t) a + n; p += 4) {
        csr_write(TSR_CSR_LOAD_ADDR, (uint32_t) p);
        uint32_t w = csr(TSR_CSR_LOAD_DATA);
        for (uint32_t j = 0; j < 4; ++j) {
            const uint64_t ba = p + j;
            if (ba >= a && ba < (uint64_t) a + n) w = (w & ~(0xFFu << (8 * j))) | (uint32_t) b[ba - a] << (8 * j);
        }
        csr_write(TSR_CSR_LOAD_ADDR, (uint32_t) p);
        csr_write(TSR_CSR_LOAD_DATA, w);
    }
    if (csr(TSR_CSR_STATUS) & TSR_STATUS_LOAD_REFUSED) {
        csr_write(TSR_CSR_STATUS, TSR_STATUS_LOAD_REFUSED);
        return fail(E_POLICY, "LOAD_* refused (RV_RUN = 1 or SR_BUSY)");
    }
    return E_OK;
}

// Bytes to the bus: unaligned head and tail as byte accesses, the middle as one word block.
int target::raw_write(uint32_t a, const uint8_t * b, size_t n) {
    if (n == 0) return E_OK;
    if ((in_itcm(a, n) || in_dtcm(a, n)) && !(csr(TSR_CSR_CTRL) & TSR_CTRL_RV_RUN)) return load_write(a, b, n);
    if (in_itcm(a, n)) itcm_dirty_ = true;
    int e;
    size_t i = 0;
    while (i < n && ((a + i) & 3u)) {
        if ((e = dm_.sba_write8(a + (uint32_t) i, b[i])) != 0) return check_dmi_timeout(e);
        ++i;
    }
    const size_t nw = (n - i) / 4;
    if (nw) {
        std::vector<uint32_t> w(nw);
        for (size_t k = 0; k < nw; ++k)
            w[k] = (uint32_t) b[i + 4 * k] | (uint32_t) b[i + 4 * k + 1] << 8 | (uint32_t) b[i + 4 * k + 2] << 16 |
                   (uint32_t) b[i + 4 * k + 3] << 24;
        if ((e = dm_.sba_write32(a + (uint32_t) i, w.data(), nw)) != 0) return check_dmi_timeout(e);
        i += 4 * nw;
    }
    for (; i < n; ++i)
        if ((e = dm_.sba_write8(a + (uint32_t) i, b[i])) != 0) return check_dmi_timeout(e);
    return E_OK;
}

int target::write_mem(uint32_t a, const uint8_t * b, size_t n) {
    if (n == 0) return E_OK;
    if ((uint64_t) a + n > 0x100000000ull) return fail(E_MALFORMED, "range wraps");
    if (overlaps(a, n, DM_LO, DM_HI)) return fail(E_POLICY, "the debug module range is not writable (DB §3.6)");
    if (overlaps(a, n, WIN_LO, WIN_HI))
        return fail(E_POLICY, "the control and core windows are read-only for the debugger (D15, DB §3.6, §3.9)");
    const bool spm = overlaps(a, n, TSR_RV_SPM_BASE, TSR_RV_SPM_BASE + spm_bytes_) ||
                     overlaps(a, n, TSR_RV_SPM_LB_BASE, TSR_RV_SPM_LB_BASE + spm_bytes_);
    if (spm && !o_.spm_write_unsafe) {
        const uint32_t dbs = csr(TSR_CSR_DBG_STATUS), st = csr(TSR_CSR_STATUS);
        const bool stopped = (dbs & TSR_DBG_STATUS_DRAINED) || ((st & TSR_STATUS_QQUIESCENT) && halted_);
        if (!stopped)
            return fail(E_POLICY, "SPM writes only in a stopped tile state (DBG_STATUS.DRAINED, or QQUIESCENT with the hart halted; "
                                  "monitor spm-write unsafe overrides, DB §3.6)");
    }
    return raw_write(a, b, n);
}

// ---------------------------------------------------------------------------------------------
// Breakpoints and watchpoints (DB §3.7, §4.6)
// ---------------------------------------------------------------------------------------------
int target::rvwp_apply() {
    csr_write(TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_RVWP_APPLY);
    const uint32_t dbs = csr(TSR_CSR_DBG_STATUS);
    if (dbs & TSR_DBG_STATUS_RVWP_TIMEOUT) {
        csr_write(TSR_CSR_DBG_STATUS, TSR_DBG_STATUS_RVWP_TIMEOUT);
        return fail(E_DMI, "RVWP_APPLY timed out (no clk_rv, DB §3.7)");
    }
    return E_OK;
}

int target::insert_bp(int type, uint32_t addr, uint32_t kind) {
    if (!attached_) return fail(E_POLICY, "not attached");
    int e;
    if (type == 0) {
        if (kind != 2 && kind != 4) return fail(E_MALFORMED, "software breakpoint kind must be 2 or 4");
        if (!in_itcm(addr, kind)) return fail(E_POLICY, "software breakpoints only in ITCM (code); use a hardware breakpoint");
        if (sw_.count(addr)) return E_OK;
        swbp s{ addr, kind, {} };
        if ((e = read_mem(addr, s.orig, kind)) != 0) return e;
        const uint8_t c_ebreak[2] = { 0x02, 0x90 }, ebreak[4] = { 0x73, 0x00, 0x10, 0x00 };
        if ((e = raw_write(addr, kind == 2 ? c_ebreak : ebreak, kind)) != 0) return e;
        sw_[addr] = s;
        return E_OK;
    }
    if (type == 1) {
        for (uint32_t i = 0; i < 2; ++i)
            if (hw_[i].used && hw_[i].addr == addr) return E_OK;
        for (uint32_t i = 0; i < 2; ++i) {
            if (hw_[i].used) continue;
            if ((e = ensure_halted()) != 0) return e;
            uint32_t d1 = 0, d2 = 0;
            if ((e = dm_.reg_write(CSR_TSELECT, i)) || (e = dm_.reg_write(CSR_TDATA2, addr)) || (e = dm_.reg_write(CSR_TDATA1, TDATA1_EXEC)) ||
                (e = dm_.reg_read(CSR_TDATA1, &d1)) || (e = dm_.reg_read(CSR_TDATA2, &d2)))
                return fail(e, "trigger write");
            if (!(d1 & 4u) || d2 != addr) {
                char b[96];
                snprintf(b, sizeof(b), "trigger %u reads back tdata1 0x%08x tdata2 0x%08x", i, d1, d2);
                return fail(E_POLICY, b);
            }
            hw_[i] = { true, addr };
            return E_OK;
        }
        return fail(E_POLICY, "both hardware triggers are in use (DB §4.6)");
    }
    if (type >= 2 && type <= 4) {
        if (kind == 0) return fail(E_MALFORMED, "watchpoint length 0");
        int j = -1;
        for (int k = 0; k < 2 && j < 0; ++k)
            if (!wp_[k].used) j = k;
        if (j < 0) return fail(E_POLICY, "both RV watchpoints are in use (DB §3.7)");
        // SPM: a tile breakpoint of kind SPM on the lines too (DB §4.6), so tile ops that write
        // (read) them stop the target after they complete.
        int tb = -1;
        uint32_t spm_base = 0;
        if (overlaps(addr, kind, TSR_RV_SPM_BASE, TSR_RV_SPM_BASE + spm_bytes_)) spm_base = TSR_RV_SPM_BASE;
        else if (overlaps(addr, kind, TSR_RV_SPM_LB_BASE, TSR_RV_SPM_LB_BASE + spm_bytes_)) spm_base = TSR_RV_SPM_LB_BASE;
        if (spm_base) {
            for (uint32_t i = 0; i < TSR_DBG_NTBP && tb < 0; ++i)
                if (!(csr(TSR_CSR_DBG_TBP_CTL(i)) & TSR_DBG_TBP_CTL_EN)) tb = (int) i;
            if (tb < 0) return fail(E_POLICY, "an SPM watchpoint needs a free tile breakpoint (DB §4.6)");
            const uint32_t lo = std::max(addr, spm_base) - spm_base, hi = std::min<uint64_t>((uint64_t) addr + kind, spm_base + spm_bytes_) - spm_base;
            csr_write(TSR_CSR_DBG_TBP_A(tb), lo / 64);
            csr_write(TSR_CSR_DBG_TBP_B(tb), (hi - 1) / 64 + 1);
            uint32_t rw = type == 2 ? TSR_DBG_TBP_CTL_W : type == 3 ? TSR_DBG_TBP_CTL_R : (TSR_DBG_TBP_CTL_R | TSR_DBG_TBP_CTL_W);
            csr_write(TSR_CSR_DBG_TBP_HITS(tb), 0);
            csr_write(TSR_CSR_DBG_TBP_CTL(tb), TSR_DBG_TBP_CTL_EN | TSR_PUT(TSR_DBG_TBP_CTL_KIND, TSR_DBG_TBP_CTL_KIND_SPM) | rw |
                                                   TSR_DBG_TBP_CTL_AFTER | TSR_DBG_TBP_CTL_HALT_RV);
            session_tbps_ |= 1u << tb;
        }
        const uint32_t ctl = TSR_DBG_RVWP_CTL_EN | TSR_DBG_RVWP_CTL_HALT |
                             (type == 2 ? TSR_DBG_RVWP_CTL_STORE : type == 3 ? TSR_DBG_RVWP_CTL_LOAD : (TSR_DBG_RVWP_CTL_LOAD | TSR_DBG_RVWP_CTL_STORE));
        csr_write(TSR_CSR_DBG_RVWP_LO(j), addr);
        csr_write(TSR_CSR_DBG_RVWP_HI(j), addr + kind);
        csr_write(TSR_CSR_DBG_RVWP_HITS(j), 0);
        csr_write(TSR_CSR_DBG_RVWP_CTL(j), ctl);
        if ((e = rvwp_apply()) != 0) {
            csr_write(TSR_CSR_DBG_RVWP_CTL(j), 0);
            return e;
        }
        wp_[j] = { true, type, addr, kind, tb };
        return E_OK;
    }
    return fail(E_MALFORMED, "breakpoint type " + std::to_string(type));
}

int target::remove_bp(int type, uint32_t addr, uint32_t kind) {
    if (!attached_) return fail(E_POLICY, "not attached");
    if (type == 0) {
        auto it = sw_.find(addr);
        if (it == sw_.end()) return E_OK;
        const int e = raw_write(addr, it->second.orig, it->second.kind);
        sw_.erase(it);
        return e;
    }
    if (type == 1) {
        for (uint32_t i = 0; i < 2; ++i)
            if (hw_[i].used && hw_[i].addr == addr) {
                int e = ensure_halted();
                if (e) return e;
                dm_.reg_write(CSR_TSELECT, i);
                e = dm_.reg_write(CSR_TDATA1, 0);
                hw_[i] = hwbp();
                return e;
            }
        return E_OK;
    }
    if (type >= 2 && type <= 4) {
        for (uint32_t j = 0; j < 2; ++j)
            if (wp_[j].used && wp_[j].addr == addr && wp_[j].len == kind && wp_[j].type == type) {
                csr_write(TSR_CSR_DBG_RVWP_CTL(j), 0);
                const int e = rvwp_apply();
                if (wp_[j].tbp >= 0) {
                    csr_write(TSR_CSR_DBG_TBP_CTL(wp_[j].tbp), 0);
                    session_tbps_ &= ~(1u << wp_[j].tbp);
                }
                wp_[j] = wp();
                return e;
            }
        return E_OK;
    }
    return fail(E_MALFORMED, "breakpoint type " + std::to_string(type));
}

// ---------------------------------------------------------------------------------------------
// DB §4.4, §4.5
// ---------------------------------------------------------------------------------------------
std::string target::target_xml() const {
    std::string s = "<?xml version=\"1.0\"?>\n<!DOCTYPE target SYSTEM \"gdb-target.dtd\">\n<target version=\"1.0\">\n"
                    "  <architecture>riscv:rv32</architecture>\n  <feature name=\"org.gnu.gdb.riscv.cpu\">\n";
    char b[160];
    for (uint32_t i = 0; i < 32; ++i) {
        snprintf(b, sizeof(b), "    <reg name=\"%s\" bitsize=\"32\" type=\"%s\" regnum=\"%u\"/>\n", k_gpr[i],
                 i == 1 ? "code_ptr" : (i == 2 || i == 8) ? "data_ptr" : "int", i);
        s += b;
    }
    s += "    <reg name=\"pc\" bitsize=\"32\" type=\"code_ptr\" regnum=\"32\"/>\n  </feature>\n  <feature name=\"org.gnu.gdb.riscv.csr\">\n";
    for (const csr_desc & c : k_csrs) {
        snprintf(b, sizeof(b), "    <reg name=\"%s\" bitsize=\"32\" type=\"int\" regnum=\"%u\" group=\"csr\"/>\n", c.name, REG_CSR0 + c.num);
        s += b;
    }
    s += "  </feature>\n  <feature name=\"org.gnu.gdb.riscv.virtual\">\n"
         "    <reg name=\"priv\" bitsize=\"32\" type=\"int\" regnum=\"4161\" group=\"general\"/>\n  </feature>\n</target>\n";
    return s;
}

std::string target::memory_map_xml() const {
    char b[640];
    snprintf(b, sizeof(b),
             "<?xml version=\"1.0\"?>\n<!DOCTYPE memory-map PUBLIC \"+//IDN gnu.org//DTD GDB Memory Map V1.0//EN\" "
             "\"http://sourceware.org/gdb/gdb-memory-map.dtd\">\n<memory-map>\n"
             "  <memory type=\"ram\" start=\"0x%08x\" length=\"0x%x\"/>\n  <memory type=\"ram\" start=\"0x%08x\" length=\"0x%x\"/>\n"
             "  <memory type=\"ram\" start=\"0x%08x\" length=\"0x2000\"/>\n  <memory type=\"ram\" start=\"0x%08x\" length=\"0x%x\"/>\n"
             "  <memory type=\"ram\" start=\"0x%08x\" length=\"0x%x\"/>\n</memory-map>\n",
             TSR_RV_ITCM_BASE, itcm_bytes_, TSR_RV_DTCM_BASE, dtcm_bytes_, TSR_RV_CTL_BASE, TSR_RV_SPM_BASE, spm_bytes_, TSR_RV_SPM_LB_BASE,
             spm_bytes_);
    return b;
}

} // namespace tsr::dbg

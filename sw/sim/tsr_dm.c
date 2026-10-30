// tsr_dm.c — debug-module model of the simulator (P1.12, V2-DB-18): DB §9.1's DMI profile on the
// built-in hart (tsr_rv) and its bus (tsr_bus). The register-level contract, the model decisions
// and the timing rules are in include/tsr_dm.h.
//
// Normative sources: docs/v2/05-debug.md (DB) §3.1-§3.8, §9.1, §14 item 1; docs/v2/04-control-
// core.md (CC) §3.2 (dm_top configuration, RV_HALTED), §4.3 (SBA refusals), §12.2 (ndmreset);
// the RISC-V Debug Specification 0.13.2 (§3 Debug Module) for everything the profile leaves to
// it; riscv-dbg (pulp-platform) for its implementation choices (memory layout, hartinfo,
// hartsello width), marked *confirm* where recalled.
//
// Structure: the DM keeps dmcontrol / abstract-command / SBA state; data0-1 and progbuf0-7 live
// in the bus's DM memory at riscv-dbg's addresses, so a program-buffer load / store of data0
// reaches the same word a DMI access does. The hart's debug hook (tsr_rv_set_debug_hook) tells
// the DM about entries, parks (program buffer done or faulted), resumes and resets; the DM
// forwards them to DB's model (tsr_dm_set_event_hook). Everything runs on the simulator's thread,
// at the current MTIME (no host time, no unordered state): deterministic.

#include "tsr_dm.h"

#include "tsr_bus.h"
#include "tsr_isa.h"
#include "tsr_rv.h"
#include "tsr_sim_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PB_SYNC_INSNS 1000u          // program-buffer instructions run inside the command write
#define PB_CHUNK      50u
#define EBREAK_INSN   0x00100073u
#define SBCS_RW       (TSR_DM_SBCS_SBREADONADDR | TSR_DM_SBCS_SBACCESS | TSR_DM_SBCS_SBAUTOINCREMENT | \
                       TSR_DM_SBCS_SBREADONDATA)
#define SBCS_RESET    (2u << TSR_DM_SBCS_SBACCESS_LSB)   // sbaccess 2 (32-bit), Debug Spec reset value

struct tsr_dm {
    tsr_sim *      sim;
    tsr_rv_agent * agent;
    tsr_rv *       rv;
    tsr_bus *      bus;
    uint8_t *      mem;          // the bus's DM memory, TSR_RV_DM_SIZE bytes at TSR_RV_DM_BASE

    // dmcontrol
    int      dmactive;
    int      haltreq;            // hart 0's halt request (a level)
    int      ndmreset;
    uint32_t hartsel;            // hartsello[0]
    int      ext_req;            // DB's dbg_req_rv (tsr_dm_set_ext_haltreq)

    // hart status kept by the DM
    int      resumeack;
    int      havereset;
    uint64_t seen_gen;           // boot generation havereset has accounted for
    uint64_t reset_gen;          // boot generation the hart was last reset for (debug event RESET)

    // abstract commands
    uint32_t cmderr;
    int      busy;               // a command's program buffer still runs (on the time base)

    // system bus access
    uint32_t sbcs;               // the R/W fields (SBCS_RW)
    uint32_t sberror;
    int      sbbusyerror;
    uint32_t sbaddress, sbdata;

    tsr_dm_event_fn ev_fn;
    void *   ev_ctx;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static uint32_t mem_rd(const tsr_dm * dm, uint32_t addr) {
    const uint8_t * p = dm->mem + (addr - TSR_RV_DM_BASE);
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void mem_wr(tsr_dm * dm, uint32_t addr, uint32_t v) {
    uint8_t * p = dm->mem + (addr - TSR_RV_DM_BASE);
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

// The hart is held in reset: CTRL.RV_RUN = 0, ndmreset, or a boot the time base has not taken
// yet (tsr_sim_rv_boot_gen moved, the agent resets the hart at its next step).
static int in_reset(const tsr_dm * dm) {
    return !tsr_sim_rv_run(dm->sim) || dm->ndmreset || tsr_sim_rv_boot_gen(dm->sim) != dm->reset_gen;
}
// RV_HALTED (DB §3.1): in debug mode, parked or running the program buffer.
static int hart_halted(const tsr_dm * dm) { return !in_reset(dm) && tsr_rv_debug_mode(dm->rv); }
// Halted in the park loop: abstract commands and resume act only then.
static int hart_parked(const tsr_dm * dm) { return !in_reset(dm) && tsr_rv_halted(dm->rv) && !dm->busy; }

// Ibex debug_req_i = the DM's request OR DB's dbg_req_rv (DB §3.3).
static void apply_req(tsr_dm * dm) { tsr_rv_set_debug_req(dm->rv, dm->haltreq || dm->ext_req); }

// The hart left debug mode outside the agent's step (a resume): STATUS.RV_HALTED now, and the
// agent stepped at the next time point (a parked agent waits for an event).
static void hart_changed(tsr_dm * dm) {
    if (!in_reset(dm)) tsr_rv_agent_sync(dm->agent);
    tsr_sim_rv_kick(dm->sim);
}

static void end_command(tsr_dm * dm, uint32_t err) {
    dm->busy = 0;
    if (err != TSR_DM_CMDERR_NONE && dm->cmderr == TSR_DM_CMDERR_NONE) dm->cmderr = err;
}

// Lazy bookkeeping at every DMI access: a boot generation the DM has not seen sets havereset; a
// command whose hart went into reset ends (cmderr 4).
static void dm_sync(tsr_dm * dm) {
    const uint64_t gen = tsr_sim_rv_boot_gen(dm->sim);
    if (gen != dm->seen_gen) {
        dm->seen_gen = gen;
        dm->havereset = 1;
    }
    if (dm->busy && in_reset(dm)) end_command(dm, TSR_DM_CMDERR_HALTRESUME);
}

static void busy_access(tsr_dm * dm) {
    if (dm->cmderr == TSR_DM_CMDERR_NONE) dm->cmderr = TSR_DM_CMDERR_BUSY;
}

// Reset values of the DM's registers (create, dmactive 1 -> 0). The hart status bits (resumeack,
// havereset) are the hart's, not the DM's.
static void dm_regs_reset(tsr_dm * dm) {
    dm->haltreq = 0;
    dm->hartsel = 0;
    dm->cmderr = TSR_DM_CMDERR_NONE;
    dm->busy = 0;
    for (uint32_t i = 0; i < TSR_DM_DATA_COUNT; ++i) mem_wr(dm, TSR_DM_DATA_ADDR + 4u * i, 0);
    for (uint32_t i = 0; i < TSR_DM_PROGBUF_SIZE; ++i) mem_wr(dm, TSR_DM_PROGBUF_ADDR + 4u * i, 0);
    dm->sbcs = SBCS_RESET;
    dm->sberror = 0;
    dm->sbbusyerror = 0;
    dm->sbaddress = 0;
    dm->sbdata = 0;
}

// ---------------------------------------------------------------------------
// The hart's debug events (tsr_rv.h)
// ---------------------------------------------------------------------------
static void dm_debug_ev(void * ctx, uint32_t ev, uint32_t arg) {
    tsr_dm * dm = (tsr_dm *)ctx;
    switch (ev) {
    case TSR_RV_DBG_EV_RESET:
        // The time base restarts the hart for the current boot generation (CC §12.2).
        dm->reset_gen = tsr_sim_rv_boot_gen(dm->sim);
        if (dm->reset_gen != dm->seen_gen) {
            dm->seen_gen = dm->reset_gen;
            dm->havereset = 1;
        }
        if (dm->busy) end_command(dm, TSR_DM_CMDERR_HALTRESUME);
        break;
    case TSR_RV_DBG_EV_PARK:          // program buffer done (arg 0, ebreak) or faulted (arg 1)
        if (dm->busy) end_command(dm, arg ? TSR_DM_CMDERR_EXCEPTION : TSR_DM_CMDERR_NONE);
        break;
    case TSR_RV_DBG_EV_RESUME:
        dm->resumeack = 1;
        if (dm->busy) end_command(dm, TSR_DM_CMDERR_EXCEPTION);   // a dret in the program buffer
        break;
    default:                          // ENTER: the state is the hart's (tsr_rv_debug_mode)
        break;
    }
    if (dm->ev_fn) dm->ev_fn(dm->ev_ctx, ev, arg);
}

// ---------------------------------------------------------------------------
// Abstract commands (Debug Spec 0.13.2 §3.6.1.1 Access Register; DB §3.5)
// ---------------------------------------------------------------------------
// One register transfer between data0 and the halted hart: 0 or a cmderr value.
static uint32_t transfer(tsr_dm * dm, uint32_t regno, int write) {
    if (regno < 0x1000u) {                                   // CSR, as executed in debug mode
        if (write) return tsr_rv_csr_write(dm->rv, regno, mem_rd(dm, TSR_DM_DATA_ADDR)) ? TSR_DM_CMDERR_EXCEPTION : 0u;
        uint32_t v = 0;
        if (tsr_rv_csr_read(dm->rv, regno, &v)) return TSR_DM_CMDERR_EXCEPTION;
        mem_wr(dm, TSR_DM_DATA_ADDR, v);
        return 0;
    }
    if (regno < 0x1020u) {                                   // GPR x0..x31
        if (write) tsr_rv_set_reg(dm->rv, regno - 0x1000u, mem_rd(dm, TSR_DM_DATA_ADDR));
        else mem_wr(dm, TSR_DM_DATA_ADDR, tsr_rv_reg(dm->rv, regno - 0x1000u));
        return 0;
    }
    if (regno < 0x1040u) return TSR_DM_CMDERR_EXCEPTION;     // FPRs: riscv-dbg's fmv traps (no F)
    return TSR_DM_CMDERR_NOTSUP;
}

// postexec: the program buffer runs in debug mode from progbuf0 until it parks (ebreak, or an
// exception: PARK event). PB_SYNC_INSNS instructions run here, at the current time; a program
// that is still running then, or waits on a store, continues on the time base (busy stays 1).
static void run_progbuf(tsr_dm * dm) {
    dm->busy = 1;
    tsr_rv_debug_exec(dm->rv, TSR_DM_PROGBUF_ADDR);
    for (uint32_t n = 0; n < PB_SYNC_INSNS && dm->busy; n += PB_CHUNK) {
        const uint32_t r = tsr_rv_run(dm->rv, PB_CHUNK, UINT64_MAX);
        if (r == TSR_RV_STOP_WAIT || r == TSR_RV_STOP_YIELD) break;
    }
}

static void command_write(tsr_dm * dm, uint32_t v) {
    if (dm->busy) {
        busy_access(dm);
        return;
    }
    if (dm->cmderr != TSR_DM_CMDERR_NONE) return;             // no command starts until cleared
    if ((v >> TSR_DM_CMD_TYPE_LSB) != 0u) {                   // Quick Access, Access Memory
        dm->cmderr = TSR_DM_CMDERR_NOTSUP;
        return;
    }
    if (dm->hartsel != 0u || !hart_parked(dm)) {
        dm->cmderr = TSR_DM_CMDERR_HALTRESUME;
        return;
    }
    if (v & TSR_DM_CMD_TRANSFER) {
        if (((v >> TSR_DM_CMD_AARSIZE_LSB) & 7u) != 2u) {    // only 32-bit accesses (RV32)
            dm->cmderr = TSR_DM_CMDERR_NOTSUP;
            return;
        }
        const uint32_t e = transfer(dm, v & 0xFFFFu, (v & TSR_DM_CMD_WRITE) != 0u);
        if (e != 0u) {
            dm->cmderr = e;
            return;
        }
        // aarpostincrement would change regno for the next abstractauto execution; the model has
        // no abstractauto and command reads 0, so it has no visible effect.
    }
    if (v & TSR_DM_CMD_POSTEXEC) {
        run_progbuf(dm);
        if (dm->busy) tsr_sim_rv_kick(dm->sim);          // the program continues on the time base
    }
}

// ---------------------------------------------------------------------------
// System bus access (Debug Spec 0.13.2 §3.10, DB §3.6, CC §4.3)
// ---------------------------------------------------------------------------
static void sba_emit(tsr_dm * dm, uint32_t kind, uint32_t addr, uint32_t size, uint32_t val) {
    if (!tsr_sim_acc_listened(dm->sim)) return;
    tsr_sim_acc a;
    memset(&a, 0, sizeof(a));
    a.kind = kind;
    a.addr = addr;
    a.size = size;
    a.len = size;
    a.val = val;
    tsr_sim_acc_emit(dm->sim, &a);
}

// The access size in bytes, or 0 with sberror set (unsupported size, misaligned address).
static uint32_t sba_size(tsr_dm * dm) {
    const uint32_t acc = (dm->sbcs & TSR_DM_SBCS_SBACCESS) >> TSR_DM_SBCS_SBACCESS_LSB;
    if (acc > 2u) {
        dm->sberror = TSR_DM_SBERR_SIZE;
        return 0;
    }
    const uint32_t size = 1u << acc;
    if (dm->sbaddress & (size - 1u)) {
        dm->sberror = TSR_DM_SBERR_ALIGN;
        return 0;
    }
    return size;
}

static void sba_read(tsr_dm * dm) {
    const uint32_t size = sba_size(dm);
    if (size == 0u) return;
    uint32_t v = 0;
    if (tsr_bus_sba_read(dm->bus, dm->sbaddress, size, &v) != 0) {
        dm->sberror = TSR_DM_SBERR_BADADDR;
        return;
    }
    dm->sbdata = v;                                          // the low 8 x size bits, zero-extended
    sba_emit(dm, TSR_ACC_SBA_RD, dm->sbaddress, size, v);
    if (dm->sbcs & TSR_DM_SBCS_SBAUTOINCREMENT) dm->sbaddress += size;
}

static void sba_write(tsr_dm * dm) {
    const uint32_t size = sba_size(dm);
    if (size == 0u) return;
    const uint32_t v = size == 4u ? dm->sbdata : dm->sbdata & ((1u << (8u * size)) - 1u);
    if (tsr_bus_sba_write(dm->bus, dm->sbaddress, size, v) != 0) {
        dm->sberror = TSR_DM_SBERR_BADADDR;
        return;
    }
    sba_emit(dm, TSR_ACC_SBA_WR, dm->sbaddress, size, v);
    if (dm->sbcs & TSR_DM_SBCS_SBAUTOINCREMENT) dm->sbaddress += size;
}

static int sba_blocked(const tsr_dm * dm) { return dm->sberror != 0u || dm->sbbusyerror; }

static uint32_t sbcs_read(const tsr_dm * dm) {
    return (1u << TSR_DM_SBCS_SBVERSION_LSB) | (dm->sbbusyerror ? TSR_DM_SBCS_SBBUSYERROR : 0u) | dm->sbcs |
           (dm->sberror << TSR_DM_SBCS_SBERROR_LSB) | (32u << TSR_DM_SBCS_SBASIZE_LSB) | TSR_DM_SBCS_ACCESS_8_16_32;
}

// ---------------------------------------------------------------------------
// dmcontrol / dmstatus
// ---------------------------------------------------------------------------
static void dm_deactivate(tsr_dm * dm) {
    dm->dmactive = 0;
    if (dm->ndmreset) {
        dm->ndmreset = 0;
        tsr_sim_rv_ndmreset(dm->sim, 0);
    }
    dm_regs_reset(dm);
    apply_req(dm);
    tsr_sim_rv_kick(dm->sim);
}

static void dmcontrol_write(tsr_dm * dm, uint32_t v) {
    if (!dm->dmactive) {                                     // only dmactive can be written
        if (v & TSR_DM_DMCONTROL_DMACTIVE) dm->dmactive = 1;
        return;
    }
    if (!(v & TSR_DM_DMCONTROL_DMACTIVE)) {
        dm_deactivate(dm);
        return;
    }
    const int req0 = dm->haltreq;
    dm->hartsel = (v >> TSR_DM_DMCONTROL_HARTSELLO_LSB) & 1u;
    const int sel0 = dm->hartsel == 0u;
    if (sel0) dm->haltreq = (v & TSR_DM_DMCONTROL_HALTREQ) != 0u;
    if (sel0 && (v & TSR_DM_DMCONTROL_ACKHAVERESET)) dm->havereset = 0;
    const int nd = (v & TSR_DM_DMCONTROL_NDMRESET) != 0u;
    if (nd != dm->ndmreset) {
        dm->ndmreset = nd;
        tsr_sim_rv_ndmreset(dm->sim, nd);                    // CC §12.2 reset sequence / release
    }
    apply_req(dm);
    if (sel0 && (v & TSR_DM_DMCONTROL_RESUMEREQ) && !(v & TSR_DM_DMCONTROL_HALTREQ) && hart_parked(dm)) {
        dm->resumeack = 0;
        tsr_rv_debug_resume(dm->rv);                         // as dret: dpc, dcsr.prv, dcsr.step
        hart_changed(dm);
    } else if (dm->haltreq != req0) {
        tsr_sim_rv_kick(dm->sim);                            // a halt request also wakes WFI
    }
}

static uint32_t dmcontrol_read(const tsr_dm * dm) {
    return ((dm->hartsel == 0u && dm->haltreq) ? TSR_DM_DMCONTROL_HALTREQ : 0u) |
           (dm->hartsel << TSR_DM_DMCONTROL_HARTSELLO_LSB) | (dm->ndmreset ? TSR_DM_DMCONTROL_NDMRESET : 0u) |
           (dm->dmactive ? TSR_DM_DMCONTROL_DMACTIVE : 0u);
}

static uint32_t dmstatus_read(const tsr_dm * dm) {
    uint32_t v = 2u | TSR_DM_DMSTATUS_AUTHENTICATED;
    if (dm->hartsel != 0u) return v | TSR_DM_DMSTATUS_ANYNONEXISTENT | TSR_DM_DMSTATUS_ALLNONEXISTENT;
    if (in_reset(dm))           v |= TSR_DM_DMSTATUS_ANYUNAVAIL | TSR_DM_DMSTATUS_ALLUNAVAIL;
    else if (hart_halted(dm))   v |= TSR_DM_DMSTATUS_ANYHALTED | TSR_DM_DMSTATUS_ALLHALTED;
    else                        v |= TSR_DM_DMSTATUS_ANYRUNNING | TSR_DM_DMSTATUS_ALLRUNNING;
    if (dm->resumeack) v |= TSR_DM_DMSTATUS_ANYRESUMEACK | TSR_DM_DMSTATUS_ALLRESUMEACK;
    if (dm->havereset) v |= TSR_DM_DMSTATUS_ANYHAVERESET | TSR_DM_DMSTATUS_ALLHAVERESET;
    return v;
}

// ---------------------------------------------------------------------------
// DMI
// ---------------------------------------------------------------------------
static int is_progbuf(uint32_t a) { return a >= TSR_DMI_PROGBUF0 && a < TSR_DMI_PROGBUF0 + TSR_DM_PROGBUF_SIZE; }
static int is_data(uint32_t a) { return a >= TSR_DMI_DATA0 && a < TSR_DMI_DATA0 + TSR_DM_DATA_COUNT; }
static uint32_t buf_addr(uint32_t a) {
    return is_data(a) ? TSR_DM_DATA_ADDR + 4u * (a - TSR_DMI_DATA0) : TSR_DM_PROGBUF_ADDR + 4u * (a - TSR_DMI_PROGBUF0);
}

uint32_t tsr_dm_dmi_read(tsr_dm * dm, uint32_t addr7) {
    if (dm == NULL) return 0;
    dm_sync(dm);
    const uint32_t a = addr7 & 0x7Fu;
    if (is_data(a) || is_progbuf(a)) {
        if (dm->busy) busy_access(dm);
        return mem_rd(dm, buf_addr(a));
    }
    switch (a) {
    case TSR_DMI_DMCONTROL:  return dmcontrol_read(dm);
    case TSR_DMI_DMSTATUS:   return dmstatus_read(dm);
    case TSR_DMI_HARTINFO:   return TSR_DM_HARTINFO;
    case TSR_DMI_ABSTRACTCS:
        return (TSR_DM_PROGBUF_SIZE << TSR_DM_ABSTRACTCS_PROGBUFSIZE_LSB) | (dm->busy ? TSR_DM_ABSTRACTCS_BUSY : 0u) |
               (dm->cmderr << TSR_DM_ABSTRACTCS_CMDERR_LSB) | TSR_DM_DATA_COUNT;
    case TSR_DMI_SBCS:       return sbcs_read(dm);
    case TSR_DMI_SBADDRESS0: return dm->sbaddress;
    case TSR_DMI_SBDATA0: {
        const uint32_t v = dm->sbdata;
        if (dm->dmactive && !sba_blocked(dm) && (dm->sbcs & TSR_DM_SBCS_SBREADONDATA)) sba_read(dm);
        return v;
    }
    default:                 return 0;                       // command (W) and registers outside the profile
    }
}

void tsr_dm_dmi_write(tsr_dm * dm, uint32_t addr7, uint32_t val) {
    if (dm == NULL) return;
    dm_sync(dm);
    const uint32_t a = addr7 & 0x7Fu;
    if (a == TSR_DMI_DMCONTROL) {
        dmcontrol_write(dm, val);
        return;
    }
    if (!dm->dmactive) return;
    if (is_data(a) || is_progbuf(a)) {
        if (dm->busy) busy_access(dm);
        else mem_wr(dm, buf_addr(a), val);
        return;
    }
    switch (a) {
    case TSR_DMI_ABSTRACTCS:
        if (dm->busy) busy_access(dm);
        else dm->cmderr &= ~((val & TSR_DM_ABSTRACTCS_CMDERR) >> TSR_DM_ABSTRACTCS_CMDERR_LSB);   // W1C
        return;
    case TSR_DMI_COMMAND:
        command_write(dm, val);
        return;
    case TSR_DMI_SBCS:
        dm->sbcs = val & SBCS_RW;
        if (val & TSR_DM_SBCS_SBBUSYERROR) dm->sbbusyerror = 0;                                   // W1C
        dm->sberror &= ~((val & TSR_DM_SBCS_SBERROR) >> TSR_DM_SBCS_SBERROR_LSB);                // W1C
        return;
    case TSR_DMI_SBADDRESS0:
        dm->sbaddress = val;
        if (!sba_blocked(dm) && (dm->sbcs & TSR_DM_SBCS_SBREADONADDR)) sba_read(dm);
        return;
    case TSR_DMI_SBDATA0:
        if (sba_blocked(dm)) return;
        dm->sbdata = val;
        sba_write(dm);
        return;
    default:
        return;                                              // read-only and absent registers
    }
}

// ---------------------------------------------------------------------------
// DB's request, RV_HALTED, event forwarding
// ---------------------------------------------------------------------------
void tsr_dm_set_ext_haltreq(tsr_dm * dm, int level) {
    if (dm == NULL || dm->ext_req == (level != 0)) return;
    dm->ext_req = level != 0;
    apply_req(dm);
    tsr_sim_rv_kick(dm->sim);
}

int tsr_dm_rv_halted(const tsr_dm * dm) { return dm != NULL && hart_halted(dm); }

void tsr_dm_set_event_hook(tsr_dm * dm, tsr_dm_event_fn fn, void * ctx) {
    if (dm == NULL) return;
    dm->ev_fn = fn;
    dm->ev_ctx = ctx;
}

// ---------------------------------------------------------------------------
// Optional-model entry points (tsr_sim_int.h §5)
// ---------------------------------------------------------------------------
tsr_dm * tsr_sim_dm(tsr_sim * sim) {
    void ** slot = sim ? tsr_sim_model_slot(sim, TSR_MODEL_DM) : NULL;
    return slot ? (tsr_dm *)*slot : NULL;
}

void tsr_dm_attach(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_DM);
    tsr_rv_agent * a = tsr_sim_rv_agent(sim);
    if (slot == NULL || *slot != NULL || a == NULL) return;   // no RISC-V model: no DM
    tsr_bus * bus = tsr_rv_agent_bus(a);
    uint8_t * mem = tsr_bus_mem(bus, TSR_RV_DM_BASE, TSR_RV_DM_SIZE);
    if (mem == NULL) return;
    tsr_dm * dm = (tsr_dm *)calloc(1, sizeof(*dm));
    if (dm == NULL) {
        fprintf(stderr, "tsr_dm: out of memory\n");
        abort();
    }
    dm->sim = sim;
    dm->agent = a;
    dm->rv = tsr_rv_agent_rv(a);
    dm->bus = bus;
    dm->mem = mem;
    // riscv-dbg's memory as far as the model uses it: data0-1 and progbuf0-7 (dm_regs_reset),
    // ebreak at the halt address (a program that jumps there parks again), 0 elsewhere (an
    // illegal instruction: an exception, cmderr 3).
    memset(mem, 0, TSR_RV_DM_SIZE);
    mem_wr(dm, TSR_RV_DM_HALT_ADDR, EBREAK_INSN);
    dm_regs_reset(dm);
    dm->havereset = 1;                                       // hardware reset (riscv-dbg: all ones)
    dm->seen_gen = dm->reset_gen = tsr_sim_rv_boot_gen(sim);
    tsr_rv_set_debug_hook(dm->rv, dm_debug_ev, dm);
    *slot = dm;
}

void tsr_dm_detach(tsr_sim * sim) {
    void ** slot = tsr_sim_model_slot(sim, TSR_MODEL_DM);
    if (slot == NULL || *slot == NULL) return;
    tsr_dm * dm = (tsr_dm *)*slot;
    tsr_rv_set_debug_hook(dm->rv, NULL, NULL);
    tsr_rv_set_debug_req(dm->rv, 0);
    if (dm->ndmreset) tsr_sim_rv_ndmreset(sim, 0);
    free(dm);
    *slot = NULL;
}

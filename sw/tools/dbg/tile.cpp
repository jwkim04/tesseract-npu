// tile.cpp — libtsr_dbg: tile-level debug and probes over the core CSR window (docs/design/05-debug.md
// "DB" §5.2-§5.7, §7.1-§7.2, §10.1; docs/design/01-execution-model.md "EX" §5.4, §6): the dispatch
// history, unit probes (QSTATUS), PERF, the s registers, the stop logic (stop, break, step, serial),
// and the status triage. Every read here is a core-domain register (DB §7.1: always answerable).

#include "tsr_dbg_lib.hpp"

#include "tsr_logdec.h"
#include "tsr_trace.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <strings.h>

namespace tsr::dbg {

namespace {
std::string strf(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char * f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}
float f32_of(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
} // namespace

const char * unit_name(uint32_t u) {
    static const char * const n[8] = { "DMA", "QACT", "MVM", "VU", "ATT", "SYNC", "u6", "u7" };
    return u < 8 ? n[u] : (u == TSR_UNIT_MARKER ? "MARK" : "?");
}

const char * op_name(uint32_t op) {
    switch (op & 0xFFu) {
    case TSR_OP_NOP: return "NOP";       case TSR_OP_LD: return "LD";         case TSR_OP_ST: return "ST";
    case TSR_OP_DQROW: return "DQROW";   case TSR_OP_KVST: return "KVST";     case TSR_OP_QACT: return "QACT";
    case TSR_OP_MVM: return "MVM";       case TSR_OP_ATTQK: return "ATTQK";   case TSR_OP_ATTPV: return "ATTPV";
    case TSR_OP_VOP: return "VOP";       case TSR_OP_RED: return "RED";       case TSR_OP_ROPE: return "ROPE";
    case TSR_OP_TOPK: return "TOPK";     case TSR_OP_ASMAX: return "ASMAX";   case TSR_OP_AMASK: return "AMASK";
    case TSR_OP_AMERGE: return "AMERGE"; case TSR_OP_ANORM: return "ANORM";   case TSR_OP_VMASK: return "VMASK";
    case TSR_OP_SOP: return "SOP";       case TSR_OP_FENCE: return "FENCE";
    case TSR_OP_GDN: return "GDN";
    default: return nullptr;
    }
}

// One line per op: the name, sub / flags / x16, and the operand words in the forms the spec names
// for the common ops (LD / ST: ch, SPM line, beat, lines; QACT, MVM: lines, K, T); other ops show
// their words raw.
std::string op_disasm(const uint32_t w[8]) {
    const uint32_t op = tsr_w0_opcode(w[0]), sub = tsr_w0_sub(w[0]), fl = tsr_w0_flags(w[0]), x16 = tsr_w0_x16(w[0]);
    const char * n = op_name(op);
    std::string s = n ? n : strf("op 0x%02x", op);
    switch (op) {
    case TSR_OP_LD:
        return s + strf(" ch%u spm[%u] <- beat 0x%x, %u lines", sub, w[1], w[2], w[3]);
    case TSR_OP_ST:
        return s + strf(" ch%u spm[%u] -> beat 0x%x, %u lines", sub, w[1], w[2], w[3]);
    case TSR_OP_QACT:
        return s + strf(".%u qa[%u] <- f32[%u], K %u, T %u, stride %u", sub, w[1], w[2], w[3], w[4], w[5]);
    case TSR_OP_MVM:
        return s + strf(".%u%s dst[%u] x[%u] w 0x%08x rows %u K %u T %u", sub, (fl & 1u) ? " ACC" : "", w[1], w[2], w[3], w[4],
                        w[5] & 0xFFFFu, (w[5] >> 16) & 0xFu);
    case TSR_OP_NOP:
    case TSR_OP_FENCE:
        return s;
    default:
        return s + strf(".%u fl %u x16 0x%04x w1-7 %08x %08x %08x %08x %08x %08x %08x", sub, fl, x16, w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    }
}

// ---------------------------------------------------------------------------------------------
// Dispatch history (DB §5.6)
// ---------------------------------------------------------------------------------------------
void hist_read(tsr_transport * t, uint32_t pidx, hist_entry * e) {
    *e = hist_entry();
    tsr_transport_csr_write(t, TSR_CSR_DBG_HIST_IDX, pidx);
    e->info = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_INFO);
    e->present = (e->info & TSR_DBG_HIST_INFO_PRESENT) != 0;
    e->unit = TSR_GET(e->info, TSR_DBG_HIST_INFO_UNIT);
    e->code = TSR_GET(e->info, TSR_DBG_HIST_INFO_CODE);
    e->rec = TSR_GET(e->info, TSR_DBG_HIST_INFO_REC);
    e->tbps = TSR_GET(e->info, TSR_DBG_HIST_INFO_TBPS);
    e->stop_point = TSR_GET(e->info, TSR_DBG_HIST_INFO_STOP);
    e->pidx = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_PIDX);
    if (!e->present) e->pidx = pidx;   // a slot that is not PRESENT reads 0 (DB §5.6): keep the asked index
    e->ticket = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_TICKET);
    e->dep[0] = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_DEP0);
    e->dep[1] = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_DEP1);
    e->time = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_TIME);
    e->pc = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_PC);
    for (uint32_t i = 0; i < 8; ++i) e->w[i] = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_W(i));
}

// DB §5.6: done iff (Q_DONE[u] - seq) mod 2^28 in [1, 2^27); running iff it is DBG_Uu_OLDEST with
// ops accepted; else queued. Codes from the first error record when it names this entry.
std::string hist_status(tsr_transport * t, const hist_entry & e) {
    if (!e.present) return "absent";
    if (e.code != 0) return strf("refused %u (checker)", e.code);
    const uint32_t u = e.ticket >> 28;
    if (u >= TSR_NUM_UNITS) return "no ticket";
    const uint32_t seq = e.ticket & TSR_TICKET_SEQ;
    const uint32_t done = tsr_transport_csr_read(t, TSR_CSR_Q_DONE(u));
    const uint32_t d = (done - seq) & 0x0FFFFFFFu;
    if (d >= 1 && d < 0x08000000u) {
        const uint32_t eseq = tsr_transport_csr_read(t, TSR_CSR_Q_ERR_SEQ(u)), ecode = tsr_transport_csr_read(t, TSR_CSR_Q_ERR_CODE(u)) & 0xFFu;
        if (ecode != 0 && (eseq & TSR_TICKET_SEQ) == seq) return strf("done ERR %u", ecode);
        return "done";
    }
    const uint32_t oldest = tsr_transport_csr_read(t, TSR_CSR_DBG_U_OLDEST(u));
    const uint32_t st = tsr_transport_csr_read(t, TSR_CSR_DBG_U_STATE(u));
    if (oldest == e.ticket && TSR_GET(st, TSR_DBG_U_STATE_ACCEPTED) != 0) return "running";
    return "queued";
}

std::string hist_line(tsr_transport * t, const hist_entry & e, const image_info * img) {
    if (!e.present) return strf("pidx %-6u (not in the ring)", e.pidx);
    auto dep = [](uint32_t d) { return (d >> 28) == 15u ? std::string("-") : strf("%s:%u", unit_name(d >> 28), d & TSR_TICKET_SEQ); };
    std::string where;
    if (img) {
        // DB §5.6: file:line of the push (the innermost inlined line) and the function it is in.
        where = img->where(e.pc);
    } else {
        where = strf("0x%08x", e.pc);
    }
    // The ring has no OP / CTRL bit (DB §5.6): a ticketed CTRL entry is a QCLEAR, w0[3:0] = 4 (CC §8.1
    // FIFO layout) on SYNC with checker code 0; the retired opcode 0x04 would be refused (ILLEGAL_OP).
    bool all0 = true;
    for (int i = 1; i < 8; ++i) all0 = all0 && e.w[i] == 0;
    const bool qclear = e.unit == TSR_UNIT_SYNC && e.w[0] == 4u && e.code == 0 && all0;
    return strf("pidx %-6u %-4s:%-6u %-11s deps %s %s  t %u  %s%s%s  @ %s", e.pidx, unit_name(e.ticket >> 28), e.ticket & TSR_TICKET_SEQ,
                hist_status(t, e).c_str(), dep(e.dep[0]).c_str(), dep(e.dep[1]).c_str(), e.time, qclear ? "QCLEAR" : op_disasm(e.w).c_str(),
                e.stop_point ? "  [stop point]" : "", e.tbps ? strf("  [TBP 0x%x]", e.tbps).c_str() : "", where.c_str());
}

std::vector<hist_entry> hist_last(tsr_transport * t, uint32_t n) {
    std::vector<hist_entry> out;
    const uint32_t head = tsr_transport_csr_read(t, TSR_CSR_DBG_HIST_HEAD);
    const uint32_t depth = 1u << TSR_GET(tsr_transport_csr_read(t, TSR_CSR_DBG_ID), TSR_DBG_ID_LOG2_HIST);
    n = std::min(n, depth);
    for (uint32_t k = n; k >= 1; --k) {
        hist_entry e;
        hist_read(t, head - k, &e);
        if (e.present) out.push_back(e);
    }
    return out;
}

// SW §5.2 push.bin: 48-byte records in push order; deps made relative to the window's first
// ticket of each unit (a dependency on an entry before the window becomes "none": it completed
// before the window starts).
bool hist_export(const std::vector<hist_entry> & es, const std::string & path, std::string * why) {
    uint32_t base[TSR_NUM_UNITS];
    bool have[TSR_NUM_UNITS] = {};
    for (const hist_entry & e : es) {
        const uint32_t u = e.ticket >> 28;
        if (u < TSR_NUM_UNITS && !have[u]) {
            base[u] = ((e.ticket & TSR_TICKET_SEQ) - 1u) & TSR_TICKET_SEQ;
            have[u] = true;
        }
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        if (why) *why = "cannot write " + path;
        return false;
    }
    uint32_t dropped = 0;
    for (const hist_entry & e : es) {
        tsr_trace_push p;
        memset(&p, 0, sizeof(p));
        memcpy(p.w, e.w, sizeof(p.w));
        for (int i = 0; i < 2; ++i) {
            const uint32_t u = e.dep[i] >> 28;
            if (u >= TSR_NUM_UNITS) {
                p.dep[i] = TSR_TICKET_NONE;
            } else if (!have[u] || ((e.dep[i] - base[u]) & TSR_TICKET_SEQ) >= 0x08000000u || ((e.dep[i] - base[u]) & TSR_TICKET_SEQ) == 0) {
                p.dep[i] = TSR_TICKET_NONE;
                dropped++;
            } else {
                p.dep[i] = tsr_trace_rel_ticket(e.dep[i], base);
            }
        }
        p.push = 0xFF00u | (e.rec ? 1u : 0u);
        p.kind = TSR_TRACE_PUSH_OP;
        fwrite(&p, sizeof(p), 1, f);
    }
    fclose(f);
    if (why) *why = strf("%zu records, %u dependencies on entries before the window dropped", es.size(), dropped);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Probes (DB §5.5, EX §5.4, §6)
// ---------------------------------------------------------------------------------------------
std::string units_text(tsr_transport * t) {
    std::string o;
    const uint32_t qd = tsr_transport_csr_read(t, TSR_CSR_Q_DEPTH);
    static const char * const head[8] = { "EMPTY", "LOAD", "WAIT_DEP", "WAIT_SNAP", "HELD", "WAIT_UNIT", "ISSUE", "?" };
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const uint32_t depth = (qd >> (5 * u)) & 31u;
        const uint32_t q = tsr_transport_csr_read(t, TSR_CSR_Q_STATE(u));
        const uint32_t ds = tsr_transport_csr_read(t, TSR_CSR_DBG_U_STATE(u));
        std::string fl;
        if (q & TSR_Q_STATE_RUNNING) fl += " RUNNING";
        if (q & TSR_Q_STATE_DEP_WAIT) fl += " DEP_WAIT";
        if (q & TSR_Q_STATE_HELD) fl += " HELD";
        if (q & TSR_Q_STATE_ERR) fl += " ERR";
        if (q & TSR_Q_STATE_HUNG) fl += " HUNG";
        const uint32_t hs = TSR_GET(ds, TSR_DBG_U_STATE_HOLD);
        std::string holds;
        if (hs & TSR_HOLD_SRC_HOST) holds += " host";
        if (hs & TSR_HOLD_SRC_RV) holds += " rv";
        if (hs & TSR_HOLD_SRC_HALT) holds += " halt";
        if (hs & TSR_HOLD_SRC_DB) holds += " db";
        o += strf("%-4s depth %2u entries %2u%s; done %u issued %u; head %s%s%s, accepted %u, outstanding %u; oldest %s pidx %u age %u",
                  unit_name(u), depth, q & TSR_Q_STATE_ENTRIES, depth ? fl.c_str() : " (absent)", tsr_transport_csr_read(t, TSR_CSR_Q_DONE(u)),
                  tsr_transport_csr_read(t, TSR_CSR_Q_ISSUED(u)), head[TSR_GET(ds, TSR_DBG_U_STATE_HEAD) & 7u],
                  holds.empty() ? "" : " held by", holds.c_str(), TSR_GET(ds, TSR_DBG_U_STATE_ACCEPTED), TSR_GET(ds, TSR_DBG_U_STATE_OUTSTANDING),
                  [&] {
                      const uint32_t ot = tsr_transport_csr_read(t, TSR_CSR_DBG_U_OLDEST(u));
                      return ot == TSR_TICKET_NONE ? std::string("none") : strf("%s:%u", unit_name(ot >> 28), ot & TSR_TICKET_SEQ);
                  }().c_str(),
                  tsr_transport_csr_read(t, TSR_CSR_DBG_U_OPIDX(u)), tsr_transport_csr_read(t, TSR_CSR_DBG_U_AGE(u)));
        const uint32_t ec = tsr_transport_csr_read(t, TSR_CSR_Q_ERR_CODE(u)) & 0xFFu;
        if (ec)
            o += strf("; first error code %u seq %u info 0x%08x w0 0x%08x pidx %u", ec, tsr_transport_csr_read(t, TSR_CSR_Q_ERR_SEQ(u)),
                      tsr_transport_csr_read(t, TSR_CSR_Q_ERR_INFO(u)), tsr_transport_csr_read(t, TSR_CSR_Q_ERR_W0(u)),
                      tsr_transport_csr_read(t, TSR_CSR_Q_ERR_PIDX(u)));
        o += "\n";
    }
    const uint32_t qs = tsr_transport_csr_read(t, TSR_CSR_DBG_QSTATE);
    o += strf("issue FIFO %u, Q_PIDX %u, dispatcher state %u, outstanding %u; read bursts out 0x%08x 0x%08x 0x%08x 0x%08x, write 0x%08x 0x%08x "
              "0x%08x 0x%08x\n",
              tsr_transport_csr_read(t, TSR_CSR_Q_FIFO), tsr_transport_csr_read(t, TSR_CSR_Q_PIDX), TSR_GET(qs, TSR_DBG_QSTATE_DISP),
              TSR_GET(qs, TSR_DBG_QSTATE_OUTSTANDING), tsr_transport_csr_read(t, TSR_CSR_DBG_RDOUT(0)), tsr_transport_csr_read(t, TSR_CSR_DBG_RDOUT(1)),
              tsr_transport_csr_read(t, TSR_CSR_DBG_RDOUT(2)), tsr_transport_csr_read(t, TSR_CSR_DBG_RDOUT(3)),
              tsr_transport_csr_read(t, TSR_CSR_DBG_WROUT(0)), tsr_transport_csr_read(t, TSR_CSR_DBG_WROUT(1)),
              tsr_transport_csr_read(t, TSR_CSR_DBG_WROUT(2)), tsr_transport_csr_read(t, TSR_CSR_DBG_WROUT(3)));
    return o;
}

std::string perf_text(tsr_transport * t) {
    const tsr_config * c = tsr_transport_config(t);
    const uint32_t C = c->num_ch, X = TSR_PERF_X(C);
    const uint32_t n = std::min<uint32_t>(TSR_GET(tsr_transport_csr_read(t, TSR_CSR_PERF_INFO), TSR_PERF_INFO_COUNT), TSR_PERF_SLOTS);
    static const char * const ext[] = { "DMA_CYCLES", "QACT_CYCLES", "ATT_CYCLES", "MVM_WAIT", "MVM_EMPTY", "HOLD_CYCLES", "PUSHES",
                                        "PUSH_STALL", "BRIDGE_ACC", "BRIDGE_WAIT", "REC_DROPPED", "LAUNCHES", "RV_WFI", "VU_SPM_WAIT",
                                        "VU_HAZARD", "VU_LINES", "ATT_QK_CYCLES", "ATT_PV_CYCLES", "ATT_PATH_STALL", "KVST_RMW",
                                        "SPM_RD_WAIT", "SPM_WR_WAIT" };
    std::string o;
    auto rd64 = [&](uint32_t i) {
        const uint32_t lo = tsr_transport_csr_read(t, TSR_CSR_PERF_LO(i));
        return (uint64_t) tsr_transport_csr_read(t, TSR_CSR_PERF_HI(i)) << 32 | lo;
    };
    for (uint32_t i = 0; i < TSR_PERF_SLOTS; ++i) {
        if (i >= n && i < TSR_PERF_TRW_RECORDS) continue;
        std::string nm;
        if (i == 0) nm = "CYCLES";
        else if (i == 1) nm = "RETIRED";
        else if (i == 2) nm = "MVM_CYCLES";
        else if (i == 3) nm = "VU_CYCLES";
        else if (i == 4) continue;
        else if (i < 5 + C) nm = strf("CH_RD_BEATS[%u]", i - 5);
        else if (i < X) nm = strf("CH_WAIT[%u]", i - 5 - C);
        else if (i - X < sizeof(ext) / sizeof(ext[0])) nm = ext[i - X];
        else if (i == TSR_PERF_TRW_RECORDS) nm = "TRW_RECORDS";
        else if (i == TSR_PERF_TRW_DROPS) nm = "TRW_DROPS";
        else if (i == TSR_PERF_DBG_STOP_CYCLES) nm = "DBG_STOP_CYCLES";
        else continue;
        o += strf("PERF_%-2u %-16s %" PRIu64 "\n", i, nm.c_str(), rd64(i));
    }
    return o;
}

std::string sreg_text(tsr_transport * t) {
    std::string o;
    for (uint32_t i = 0; i < TSR_CSR_SREG_N; ++i) {
        const uint32_t v = tsr_transport_csr_read(t, TSR_CSR_SREG(i));
        o += strf("s%-2u 0x%08x %.9g\n", i, v, (double) f32_of(v));
    }
    return o;
}

// ---------------------------------------------------------------------------------------------
// Stop logic (DB §5.2-§5.4)
// ---------------------------------------------------------------------------------------------
std::string tile_show(tsr_transport * t) {
    const uint32_t s = tsr_transport_csr_read(t, TSR_CSR_DBG_STATUS), c = tsr_transport_csr_read(t, TSR_CSR_DBG_CFG);
    std::string o;
    std::string cause;
    static const char * const cn[5] = { "NOW", "BP", "STEP", "ERR", "HUNG" };
    for (uint32_t i = 0; i < 5; ++i)
        if (TSR_GET(s, TSR_DBG_STATUS_CAUSE) & (1u << i)) cause += cause.empty() ? cn[i] : std::string("|") + cn[i];
    o += strf("DBG_STATUS 0x%08x: %s%s%s%s; T %u, S %s%u, next pidx %u, step pidx %u, match queue %u\n", s,
              (s & TSR_DBG_STATUS_STOP_ACTIVE) ? "stop active" : "no stop", (s & TSR_DBG_STATUS_DRAINED) ? ", DRAINED" : "",
              (s & TSR_DBG_STATUS_STEP_BUSY) ? ", STEP_BUSY" : "", cause.empty() ? "" : (" (" + cause + ")").c_str(),
              tsr_transport_csr_read(t, TSR_CSR_DBG_STOP_PIDX), (s & TSR_DBG_STATUS_S_VALID) ? "" : "invalid ",
              tsr_transport_csr_read(t, TSR_CSR_DBG_STOP_AT), tsr_transport_csr_read(t, TSR_CSR_DBG_NEXT_PIDX),
              tsr_transport_csr_read(t, TSR_CSR_DBG_STEP_PIDX), tsr_transport_csr_read(t, TSR_CSR_DBG_MATCHQ) & TSR_DBG_MATCHQ_PENDING);
    o += strf("DBG_CFG 0x%08x:%s%s%s%s%s%s%s%s%s%s%s\n", c, (c & TSR_DBG_CFG_DMI_EN) ? " DMI_EN" : "", (c & TSR_DBG_CFG_SERIAL) ? " SERIAL" : "",
              (c & TSR_DBG_CFG_ERR_STOP) ? " ERR_STOP" : "", (c & TSR_DBG_CFG_STOP_HALT_RV) ? " STOP_HALT_RV" : "",
              (c & TSR_DBG_CFG_HIST_EN) ? " HIST_EN" : "", (c & TSR_DBG_CFG_HIST_FREEZE_ERR) ? " HIST_FREEZE_ERR" : "",
              (c & TSR_DBG_CFG_HIST_FREEZE) ? " HIST_FREEZE" : "", (c & TSR_DBG_CFG_HUNG_STOP) ? " HUNG_STOP" : "",
              (c & TSR_DBG_CFG_HALT_ON_BOOT) ? " HALT_ON_BOOT" : "", (c & TSR_DBG_CFG_RV_HALT_EN) ? " RV_HALT_EN" : "",
              (c & TSR_DBG_CFG_TRW_EN) ? " TRW_EN" : "");
    static const char * const kinds[8] = { "pidx", "op", "spm", "dram", "ticket", "k5", "k6", "k7" };
    for (uint32_t i = 0; i < TSR_DBG_NTBP; ++i) {
        const uint32_t ctl = tsr_transport_csr_read(t, TSR_CSR_DBG_TBP_CTL(i));
        if (!(ctl & TSR_DBG_TBP_CTL_EN)) continue;
        o += strf("TBP%u %s A 0x%08x B 0x%08x%s%s%s%s%s%s hits %u\n", i, kinds[TSR_GET(ctl, TSR_DBG_TBP_CTL_KIND)],
                  tsr_transport_csr_read(t, TSR_CSR_DBG_TBP_A(i)), tsr_transport_csr_read(t, TSR_CSR_DBG_TBP_B(i)),
                  (ctl & TSR_DBG_TBP_CTL_R) ? " r" : "", (ctl & TSR_DBG_TBP_CTL_W) ? " w" : "", (ctl & TSR_DBG_TBP_CTL_AFTER) ? " after" : "",
                  (ctl & TSR_DBG_TBP_CTL_HALT_RV) ? " halt" : "", (ctl & TSR_DBG_TBP_CTL_COUNT_ONLY) ? " count" : "",
                  TSR_GET(ctl, TSR_DBG_TBP_CTL_CHMASK) ? strf(" chmask 0x%x", TSR_GET(ctl, TSR_DBG_TBP_CTL_CHMASK)).c_str() : "",
                  tsr_transport_csr_read(t, TSR_CSR_DBG_TBP_HITS(i)));
    }
    return o;
}

// QSTEP (DB §5.2): with a stop active, waits for DRAINED, then each QSTEP until STEP_BUSY = 0.
int tile_step(tsr_transport * t, uint32_t n, uint64_t timeout_ns, std::string * out) {
    if (n == 0) n = 1;
    auto wait_for = [&](uint32_t mask, bool set) {
        const uint64_t t0 = tsr_transport_now_ns(t);
        uint64_t dt = 1000;
        for (;;) {
            const uint32_t s = tsr_transport_csr_read(t, TSR_CSR_DBG_STATUS);
            if (((s & mask) != 0) == set) return true;
            if (tsr_transport_now_ns(t) - t0 >= timeout_ns) return false;
            tsr_transport_idle(t, dt);
            dt = std::min<uint64_t>(dt * 2, 1000000);
        }
    };
    const uint32_t s0 = tsr_transport_csr_read(t, TSR_CSR_DBG_STATUS);
    if ((s0 & TSR_DBG_STATUS_STOP_ACTIVE) && !wait_for(TSR_DBG_STATUS_DRAINED, true)) {
        if (out) *out += "tile step: the stop does not drain (DRAINED stays 0): an entry below T is still running or held\n";
        return E_RUNNING;
    }
    tsr_transport_csr_write(t, TSR_CSR_DBG_STEP_N, n);
    tsr_transport_csr_write(t, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_QSTEP);
    // STEP_BUSY rises at once for a QSTEP with a stop (or through the debug-halt hold) and falls
    // when the stepped entries complete; with neither the QSTEP acts as STOP_NOW.
    if (!wait_for(TSR_DBG_STATUS_STEP_BUSY, false)) {
        if (out) *out += "tile step: STEP_BUSY stays 1 (the stepped entry did not complete)\n";
        return E_RUNNING;
    }
    if (out)
        *out += strf("tile step: X = pidx %u; now T %u, next pidx %u\n", tsr_transport_csr_read(t, TSR_CSR_DBG_STEP_PIDX),
                     tsr_transport_csr_read(t, TSR_CSR_DBG_STOP_PIDX), tsr_transport_csr_read(t, TSR_CSR_DBG_NEXT_PIDX));
    return E_OK;
}

// tile break <pidx P[-Q] | op NAME | spm A B [r|w|rw] | dram CH A B [r|w] | ticket T> [after] [halt] [count]
int tile_break(tsr_transport * t, const std::vector<std::string> & a, const image_info *, std::string * why) {
    auto bad = [&](const std::string & w) {
        if (why) *why = w;
        return -1;
    };
    if (a.empty()) return bad("tile break: what? (pidx P[-Q] | op NAME | spm A B [r|w|rw] | dram CH A B [r|w] | ticket T)");
    int idx = -1;
    for (uint32_t i = 0; i < TSR_DBG_NTBP && idx < 0; ++i)
        if (!(tsr_transport_csr_read(t, TSR_CSR_DBG_TBP_CTL(i)) & TSR_DBG_TBP_CTL_EN)) idx = (int) i;
    if (idx < 0) return bad("tile break: all four tile breakpoints are in use (tile delete N)");
    uint32_t kind = 0, A = 0, B = 0, ctl = 0;
    size_t k = 1;
    const std::string & w = a[0];
    if (w == "pidx" && a.size() > 1) {
        kind = TSR_DBG_TBP_CTL_KIND_PIDX;
        const std::string & r = a[1];
        const size_t dash = r.find('-');
        if (!parse_u32(r.substr(0, dash), &A)) return bad("tile break pidx: bad number");
        B = A + 1;
        if (dash != std::string::npos && !parse_u32(r.substr(dash + 1), &B)) return bad("tile break pidx: bad range");
        if (dash != std::string::npos) B += 1;   // P-Q inclusive
        k = 2;
    } else if (w == "op" && a.size() > 1) {
        kind = TSR_DBG_TBP_CTL_KIND_OPCODE;
        uint32_t opc = 0xFFFFFFFFu;
        for (uint32_t o = 0; o < 0x80; ++o)
            if (op_name(o) && strcasecmp(op_name(o), a[1].c_str()) == 0) opc = o;
        if (opc == 0xFFFFFFFFu && !parse_u32(a[1], &opc)) return bad("tile break op: unknown opcode " + a[1]);
        A = TSR_PUT(TSR_TBP_OPC_A_OPC, opc) | TSR_PUT(TSR_TBP_OPC_A_OPC_MASK, 0xFFu);
        B = 0;   // any unit
        k = 2;
    } else if (w == "spm" && a.size() > 2) {
        kind = TSR_DBG_TBP_CTL_KIND_SPM;
        if (!parse_u32(a[1], &A) || !parse_u32(a[2], &B)) return bad("tile break spm: A B (lines)");
        k = 3;
        ctl |= TSR_DBG_TBP_CTL_R | TSR_DBG_TBP_CTL_W;
    } else if (w == "dram" && a.size() > 3) {
        kind = TSR_DBG_TBP_CTL_KIND_DRAM;
        uint32_t ch = 0;
        if (!parse_u32(a[1], &ch) || ch >= 16 || !parse_u32(a[2], &A) || !parse_u32(a[3], &B)) return bad("tile break dram: CH A B (beats)");
        ctl |= TSR_PUT(TSR_DBG_TBP_CTL_CHMASK, 1u << ch) | TSR_DBG_TBP_CTL_R | TSR_DBG_TBP_CTL_W;
        k = 4;
    } else if (w == "ticket" && a.size() > 1) {
        kind = TSR_DBG_TBP_CTL_KIND_TICKET;
        if (!parse_u32(a[1], &A)) return bad("tile break ticket: T");
        k = 2;
    } else {
        return bad("tile break: unknown kind " + w);
    }
    for (; k < a.size(); ++k) {
        const std::string & o = a[k];
        if (o == "r") ctl = (ctl & ~TSR_DBG_TBP_CTL_W) | TSR_DBG_TBP_CTL_R;
        else if (o == "w") ctl = (ctl & ~TSR_DBG_TBP_CTL_R) | TSR_DBG_TBP_CTL_W;
        else if (o == "rw") ctl |= TSR_DBG_TBP_CTL_R | TSR_DBG_TBP_CTL_W;
        else if (o == "after") ctl |= TSR_DBG_TBP_CTL_AFTER;
        else if (o == "halt") ctl |= TSR_DBG_TBP_CTL_HALT_RV;
        else if (o == "count") ctl |= TSR_DBG_TBP_CTL_COUNT_ONLY;
        else return bad("tile break: unknown option " + o);
    }
    tsr_transport_csr_write(t, TSR_CSR_DBG_TBP_A(idx), A);
    tsr_transport_csr_write(t, TSR_CSR_DBG_TBP_B(idx), B);
    tsr_transport_csr_write(t, TSR_CSR_DBG_TBP_HITS(idx), 0);
    tsr_transport_csr_write(t, TSR_CSR_DBG_TBP_CTL(idx), ctl | TSR_DBG_TBP_CTL_EN | TSR_PUT(TSR_DBG_TBP_CTL_KIND, kind));
    return idx;
}

// ---------------------------------------------------------------------------------------------
// Status and the DB §7.2 triage
// ---------------------------------------------------------------------------------------------
std::string status_text(tsr_transport * t, const image_info * img) {
    const uint32_t s = tsr_transport_csr_read(t, TSR_CSR_STATUS);
    const uint32_t rt = tsr_transport_csr_read(t, TSR_CSR_RT_STATE), info = tsr_transport_csr_read(t, TSR_CSR_RT_INFO);
    const uint32_t ph = TSR_GET(rt, TSR_RT_STATE_PHASE), k = TSR_GET(rt, TSR_RT_STATE_KERNEL);
    std::string o;
    static const char * const sb[28] = { "RV_RUNNING", "RV_HALTED", "RV_SLEEP", "RV_CLK_OK", "RV_RESET_DONE", "ACTIVE", "QIDLE",
                                         "QQUIESCENT", "QERR", "SR_BUSY", "LCQ_PENDING", "LOG_PENDING", "PLAT_ERR", "HOLD_HOST",
                                         "HOLD_RV", "HOLD_HALT", "LOAD_REFUSED", "LOAD_TIMEOUT", "DMI_TIMEOUT", "QCLEAR_REFUSED",
                                         "PUSH_TIMEOUT", "BRIDGE_TIMEOUT", "LQ_OVERFLOW", "RING_RD_BAD", "LOG_DROP", "UNIT_HUNG",
                                         "SR_RV_TIMEOUT", "RV_RUN_REFUSED" };
    std::string bits;
    for (uint32_t i = 0; i < 28; ++i)
        if (s & (1u << i)) bits += std::string(" ") + sb[i];
    o += strf("STATUS 0x%08x:%s\n", s, bits.c_str());
    std::string kn = img ? img->kernel_name(k) : std::string();
    o += strf("RT_STATE 0x%08x (%s, kernel %u%s%s%s), RT_INFO 0x%08x; CTRL 0x%08x; LQ pushed %u taken %u answered %u; LCQ wr %u rd %u; "
              "LOG wr %u rd %u dropped %u\n",
              rt, rt_phase_name(ph), k, kn.empty() ? "" : " ", kn.c_str(), "", info, tsr_transport_csr_read(t, TSR_CSR_CTRL),
              tsr_transport_csr_read(t, TSR_CSR_LQ_SEQ), tsr_transport_csr_read(t, TSR_CSR_LQ_TAKEN),
              tsr_transport_csr_read(t, TSR_CSR_LCQ_ANSWERED), tsr_transport_csr_read(t, TSR_CSR_LCQ_WR),
              tsr_transport_csr_read(t, TSR_CSR_LCQ_RD), tsr_transport_csr_read(t, TSR_CSR_LOG_WR), tsr_transport_csr_read(t, TSR_CSR_LOG_RD),
              tsr_transport_csr_read(t, TSR_CSR_LOG_DROPPED));
    const uint32_t has_dbg = tsr_transport_csr_read(t, TSR_CSR_DBG_ID);
    if (has_dbg) o += tile_show(t);
    // DB §7.2: the first rows that apply.
    std::string verdict;
    uint32_t running = 0, hung = 0, held = 0, depw = 0;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
        const uint32_t q = tsr_transport_csr_read(t, TSR_CSR_Q_STATE(u));
        if (q & TSR_Q_STATE_RUNNING) running |= 1u << u;
        if (q & TSR_Q_STATE_HUNG) hung |= 1u << u;
        if (q & TSR_Q_STATE_HELD) held |= 1u << u;
        if (q & TSR_Q_STATE_DEP_WAIT) depw |= 1u << u;
    }
    if (s == 0xFFFFFFFFu) verdict = "reads return 0xFFFFFFFF: the AXI firewall tripped or the card is lost (card reset)";
    else if (s & TSR_STATUS_UNIT_HUNG) verdict = strf("a unit is stuck without progress (units 0x%02x HUNG): card reset (EX §8.6)", hung);
    else if (!(s & TSR_STATUS_RV_CLK_OK)) verdict = "clk_rv absent (RV_CLK_OK = 0): platform / clock";
    else if (s & TSR_STATUS_RV_HALTED) verdict = "the RISC-V is halted: stopped by a debugger, a breakpoint or the trap halt (attach GDB)";
    else if (ph == TSR_RT_STATE_PHASE_TRAPPED) verdict = "the firmware trapped (RT_STATE TRAPPED): monitor crash, crash-view";
    else if (!(s & TSR_STATUS_RV_RUNNING)) verdict = "the RISC-V is not running (held in reset: boot an image)";
    else if (held) verdict = strf("queue heads held (units 0x%02x): a host hold, the RISC-V's QHOLD, HOLD_ON_HALT or a DB stop (units)", held);
    else if (depw) verdict = strf("heads waiting for dependencies (units 0x%02x): follow the chain in the history (hist)", depw);
    else if (running) verdict = strf("units running (0x%02x): compare the probes twice (units, perf)", running);
    else if (s & TSR_STATUS_RV_SLEEP) verdict = "the RISC-V sleeps in WFI, waiting for a ticket or a descriptor";
    else verdict = "no fault visible in the core-domain registers (RV samples: rv sample -n 5)";
    o += "verdict (DB §7.2): " + verdict + "\n";
    return o;
}

} // namespace tsr::dbg

// trw.cpp — libtsr_dbg: the trace writer (docs/design/05-debug.md "DB" §8.3, §8.4, §8.6, §8.7, §10.1;
// P3.8): the DRAM ring set-up and drain over the transport, the record decoder, the Perfetto /
// Chrome trace-event timeline and the self-checks.
//
// Ring: DBG_TRW_BASE {ch[31:28], beat[27:0]} (16-beat aligned), DBG_TRW_LINES 64-B lines (a power
// of two >= 16), record k at byte beat * 64 + 32 (k mod 2 LINES) of the channel; DBG_TRW_WPTR counts
// records written (write response returned), DBG_TRW_RPTR the host's. Without TRW_STOP_FULL the
// ring overwrites (flight recorder): a drain that finds more than the capacity reads the newest
// capacity records and reports the rest lost.

#include "tsr_dbg_lib.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace tsr::dbg {

namespace {
uint32_t rd32(const uint8_t * p) { return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24; }
std::string strf(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char * f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}
} // namespace

void trw_decode(const uint8_t raw[32], trw_rec * r) {
    memset(r, 0, sizeof(*r));
    for (int i = 0; i < 8; ++i) r->w[i] = rd32(raw + 4 * i);
    const uint32_t w0 = r->w[0];
    r->kind = TSR_GET(w0, TSR_TRW_W0_KIND);
    r->unit = TSR_GET(w0, TSR_TRW_W0_UNIT);
    r->code = TSR_GET(w0, TSR_TRW_W0_CODE);
    r->ran = TSR_GET(w0, TSR_TRW_W0_RAN);
    r->rec = TSR_GET(w0, TSR_TRW_W0_REC);
    r->tbp = TSR_GET(w0, TSR_TRW_W0_TBP);
    r->opcode = TSR_GET(w0, TSR_TRW_W0_OPCODE);
    r->w0_hi = TSR_GET(w0, TSR_TRW_W0_W0_HI);
    r->a1 = r->w[1];
    r->a2 = r->w[2];
    r->time = (uint64_t) r->w[4] << 32 | r->w[3];
    r->run = r->w[5];
    r->qwait = r->w[6];
    r->hwait = r->w[7];
}

trw_ring trw_default_ring(uint32_t mib) {
    trw_ring r;
    if (mib == 0) mib = TSR_TRW_RING_DEFAULT_MIB;
    mib = std::min<uint32_t>(mib, (uint32_t) (TSR_REGION_IO_SIZE >> 20) / 2);
    uint32_t lines = 16;
    while ((uint64_t) lines * 2 * 64 <= (uint64_t) mib << 20) lines *= 2;
    r.ch = TSR_TRW_CH;
    r.lines = lines;
    r.off = TSR_REGION_IO_OFF + TSR_REGION_IO_SIZE - (uint64_t) lines * 64;   // the top of the ch0 IO region (EX §11)
    return r;
}

int trw_start(tsr_transport * t, const trw_ring & r, std::string * why) {
    auto bad = [&](const std::string & w) {
        if (why) *why = w;
        return E_POLICY;
    };
    const uint32_t id = tsr_transport_csr_read(t, TSR_CSR_DBG_ID);
    if (!(id & TSR_DBG_ID_TRW)) return bad("DBG_ID.TRW = 0: the core has no trace writer (DB §10.3 HAS_TRW)");
    if (r.lines < 16 || (r.lines & (r.lines - 1))) return bad("the ring needs a power-of-two number of lines >= 16");
    if (r.off & 1023u) return bad("the ring base must be 16-beat (1 KiB) aligned");
    const tsr_config * c = tsr_transport_config(t);
    if (r.ch >= c->num_ch || r.off + (uint64_t) r.lines * 64 > c->ch_bytes) return bad("the ring lies outside the channel");
    uint32_t cfg = tsr_transport_csr_read(t, TSR_CSR_DBG_CFG);
    tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, cfg & ~TSR_DBG_CFG_TRW_EN);
    tsr_transport_csr_write(t, TSR_CSR_DBG_TRW_BASE, TSR_PUT(TSR_HANDLE_CH, r.ch) | (uint32_t) (r.off / 64));
    tsr_transport_csr_write(t, TSR_CSR_DBG_TRW_LINES, r.lines);
    tsr_transport_csr_write(t, TSR_CSR_DBG_TRW_MASK, r.mask);
    if (tsr_transport_csr_read(t, TSR_CSR_DBG_TRW_LINES) != r.lines) return bad("DBG_TRW_LINES did not take the ring size (PTR_ERR)");
    tsr_transport_csr_write(t, TSR_CSR_DBG_TRW_RPTR, tsr_transport_csr_read(t, TSR_CSR_DBG_TRW_WPTR));
    tsr_transport_csr_write(t, TSR_CSR_DBG_STATUS, TSR_DBG_STATUS_TRW_DROP | TSR_DBG_STATUS_PTR_ERR);   // W1C
    cfg = (cfg & ~TSR_DBG_CFG_TRW_STOP_FULL) | TSR_DBG_CFG_TRW_EN | (r.stop_full ? TSR_DBG_CFG_TRW_STOP_FULL : 0u);
    tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, cfg);
    return E_OK;
}

int trw_stop(tsr_transport * t) {
    tsr_transport_csr_write(t, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    tsr_transport_csr_write(t, TSR_CSR_DBG_CFG, tsr_transport_csr_read(t, TSR_CSR_DBG_CFG) & ~TSR_DBG_CFG_TRW_EN);
    return E_OK;
}

int trw_drain(tsr_transport * t, const trw_ring & r, std::vector<trw_rec> * out, uint64_t * lost, std::string * why) {
    if (lost) *lost = 0;
    tsr_transport_csr_write(t, TSR_CSR_DBG_CTRL, TSR_DBG_CTRL_TRW_FLUSH);
    // WPTR counts records whose write response returned: wait until it stops moving.
    uint32_t wp = tsr_transport_csr_read(t, TSR_CSR_DBG_TRW_WPTR);
    for (int i = 0; i < 16; ++i) {
        tsr_transport_idle(t, 10000);
        const uint32_t w2 = tsr_transport_csr_read(t, TSR_CSR_DBG_TRW_WPTR);
        if (w2 == wp) break;
        wp = w2;
    }
    const uint32_t rp = tsr_transport_csr_read(t, TSR_CSR_DBG_TRW_RPTR);
    const uint64_t cap = 2ull * r.lines;
    uint64_t n = (uint32_t) (wp - rp);
    uint32_t first = rp;
    if (n > cap) {
        if (lost) *lost = n - cap;
        first = wp - (uint32_t) cap;
        n = cap;
    }
    std::vector<uint8_t> buf((size_t) n * 32);
    for (uint64_t done = 0; done < n;) {
        const uint64_t k = (first + done) % cap;
        const uint64_t run = std::min(n - done, cap - k);
        if (tsr_transport_read(t, r.ch, r.off + k * 32, buf.data() + done * 32, (size_t) run * 32) != 0) {
            if (why) *why = "reading the trace ring failed";
            return E_TRANSPORT;
        }
        done += run;
    }
    for (uint64_t i = 0; i < n; ++i) {
        trw_rec rec;
        trw_decode(buf.data() + i * 32, &rec);
        if (rec.kind == TSR_TRW_W0_KIND_PAD) continue;
        out->push_back(rec);
    }
    tsr_transport_csr_write(t, TSR_CSR_DBG_TRW_RPTR, wp);
    return E_OK;
}

bool trw_save_raw(const std::vector<trw_rec> & recs, const std::string & path) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    for (const trw_rec & r : recs) {
        uint8_t b[32];
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) b[4 * i + j] = (uint8_t) (r.w[i] >> (8 * j));
        fwrite(b, 32, 1, f);
    }
    return fclose(f) == 0;
}

bool trw_load_raw(const std::string & path, std::vector<trw_rec> * out, std::string * why) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        if (why) *why = "cannot open " + path;
        return false;
    }
    uint8_t b[32];
    size_t got;
    while ((got = fread(b, 1, 32, f)) == 32) {
        trw_rec r;
        trw_decode(b, &r);
        if (r.kind != TSR_TRW_W0_KIND_PAD) out->push_back(r);
    }
    fclose(f);
    if (got != 0) {
        if (why) *why = path + ": not a whole number of 32-byte records";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Timeline (DB §8.6): trace-event JSON as Perfetto and chrome://tracing read it.
// ---------------------------------------------------------------------------------------------
namespace {
constexpr int TID_RV = 16, TID_DBG = 17, TID_WAIT0 = 8;

std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += c;
        else if ((unsigned char) c < 0x20) o += strf("\\u%04x", (unsigned) (unsigned char) c);
        else o += c;
    }
    return o + "\"";
}
} // namespace

std::string trw_chrome_json(const std::vector<trw_rec> & recs, const timeline_opts & o) {
    const double us = 1e6 / (o.core_hz > 0 ? o.core_hz : 276.3e6);
    std::vector<std::string> ev;
    auto meta = [&](int tid, const std::string & name) {
        ev.push_back(strf("{\"ph\":\"M\",\"pid\":0,\"tid\":%d,\"name\":\"thread_name\",\"args\":{\"name\":%s}}", tid, jstr(name).c_str()));
    };
    ev.push_back("{\"ph\":\"M\",\"pid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"Tesseract0\"}}");
    uint32_t units_seen = 0;
    for (const trw_rec & r : recs)
        if (r.kind == TSR_TRW_W0_KIND_OP && r.unit < 8) units_seen |= 1u << r.unit;
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u)
        if (units_seen & (1u << u)) {
            meta((int) u, unit_name(u));
            meta(TID_WAIT0 + (int) u, std::string(unit_name(u)) + " waiting");
        }
    meta(TID_RV, "RISC-V");
    meta(TID_DBG, "debug");
    // Markers: LAUNCH / NODE begin-end pairs become slices; others instants.
    std::vector<const trw_rec *> open_launch, open_node;
    for (const trw_rec & r : recs) {
        switch (r.kind) {
        case TSR_TRW_W0_KIND_OP: {
            const char * on = op_name(r.opcode);
            const std::string name = on ? on : strf("op 0x%02x", r.opcode);
            const uint64_t t_start = r.ran ? r.time - r.run : r.time;
            const uint64_t t_head = r.ran ? t_start - r.hwait : r.time - r.hwait;
            const uint64_t t_disp = t_head - r.qwait;
            const std::string args = strf("{\"pidx\":%u,\"ticket\":\"%s:%u\",\"code\":%u,\"ran\":%u,\"qwait\":%u,\"hwait\":%u,\"run\":%u%s}", r.a1,
                                          unit_name(r.a2 >> 28), r.a2 & TSR_TICKET_SEQ, r.code, r.ran, r.qwait, r.hwait, r.run,
                                          r.tbp ? ",\"tbp\":1" : "");
            if (r.ran)
                ev.push_back(strf("{\"ph\":\"X\",\"pid\":0,\"tid\":%u,\"name\":%s,\"ts\":%.4f,\"dur\":%.4f,\"args\":%s}", r.unit, jstr(name).c_str(),
                                  (double) t_start * us, (double) r.run * us, args.c_str()));
            else
                ev.push_back(strf("{\"ph\":\"i\",\"s\":\"t\",\"pid\":0,\"tid\":%u,\"name\":%s,\"ts\":%.4f,\"args\":%s}", r.unit,
                                  jstr(name + strf(" (not run, code %u)", r.code)).c_str(), (double) r.time * us, args.c_str()));
            if (r.qwait + r.hwait > 0)
                ev.push_back(strf("{\"ph\":\"X\",\"pid\":0,\"tid\":%u,\"name\":%s,\"ts\":%.4f,\"dur\":%.4f,\"args\":{\"pidx\":%u,\"qwait\":%u,\"hwait\":%u}}",
                                  TID_WAIT0 + r.unit, jstr("waiting " + name).c_str(), (double) t_disp * us,
                                  (double) (t_start - t_disp) * us, r.a1, r.qwait, r.hwait));
            break;
        }
        case TSR_TRW_W0_KIND_MARK: {
            const uint32_t v = r.a2, cls = TSR_GET(v, TSR_MARK_CLASS), pl = TSR_GET(v, TSR_MARK_PAYLOAD);
            std::string name;
            if (cls == TSR_MARK_CLASS_LAUNCH_BEGIN || cls == TSR_MARK_CLASS_LAUNCH_END) {
                const uint32_t k = TSR_GET(v, TSR_MARK_LAUNCH_KERNEL), s = TSR_GET(v, TSR_MARK_LAUNCH_SEQ);
                const std::string kn = o.img ? o.img->kernel_name(k) : std::string();
                name = strf("launch %s seq %u", kn.empty() ? strf("kernel %u", k).c_str() : kn.c_str(), s);
                ev.push_back(strf("{\"ph\":\"%s\",\"pid\":0,\"tid\":%d,\"name\":%s,\"ts\":%.4f,\"args\":{\"next_pidx\":%u}}",
                                  cls == TSR_MARK_CLASS_LAUNCH_BEGIN ? "B" : "E", TID_RV, jstr(name).c_str(), (double) r.time * us, r.a1));
            } else if (cls == TSR_MARK_CLASS_NODE_BEGIN || cls == TSR_MARK_CLASS_NODE_END) {
                name = strf("node %u", pl);
                ev.push_back(strf("{\"ph\":\"%s\",\"pid\":0,\"tid\":%d,\"name\":%s,\"ts\":%.4f,\"args\":{\"next_pidx\":%u}}",
                                  cls == TSR_MARK_CLASS_NODE_BEGIN ? "B" : "E", TID_RV, jstr(name).c_str(), (double) r.time * us, r.a1));
            } else {
                static const char * const cn[16] = { "user", "", "", "", "", "layer", "phase", "rsvd7", "rsvd8", "rsvd9", "rsvd10", "rsvd11",
                                                     "rsvd12", "rsvd13", "rsvd14", "rsvd15" };
                name = strf("%s %u", cn[cls], pl);
                ev.push_back(strf("{\"ph\":\"i\",\"s\":\"t\",\"pid\":0,\"tid\":%d,\"name\":%s,\"ts\":%.4f,\"args\":{\"value\":\"0x%08x\"}}", TID_RV,
                                  jstr(name).c_str(), (double) r.time * us, v));
            }
            break;
        }
        case TSR_TRW_W0_KIND_STOP:
            ev.push_back(strf("{\"ph\":\"i\",\"s\":\"p\",\"pid\":0,\"tid\":%d,\"name\":\"%s T %u\",\"ts\":%.4f,\"args\":{\"cause\":%u}}", TID_DBG,
                              r.run ? "tile stop" : "tile release", r.a1, (double) r.time * us, r.a2));
            break;
        case TSR_TRW_W0_KIND_RVHALT:
            ev.push_back(strf("{\"ph\":\"i\",\"s\":\"p\",\"pid\":0,\"tid\":%d,\"name\":\"%s\",\"ts\":%.4f}", TID_DBG,
                              r.a2 ? "RISC-V halted" : "RISC-V resumed", (double) r.time * us));
            break;
        case TSR_TRW_W0_KIND_DROP:
            ev.push_back(strf("{\"ph\":\"i\",\"s\":\"g\",\"pid\":0,\"tid\":%d,\"name\":\"%u records dropped\",\"ts\":%.4f}", TID_DBG, r.a1,
                              (double) r.time * us));
            break;
        default:
            break;
        }
    }
    // Channel counter tracks (DB §8.6): GB/s and idle % per launch window from PERF.
    const uint32_t C = o.num_ch;
    for (const perf_window & w : o.windows) {
        const double cyc = (double) w.perf[TSR_PERF_CYCLES];
        const double dev_s = cyc / (o.core_hz > 0 ? o.core_hz : 276.3e6);
        for (uint32_t c = 0; c < C && c < 16; ++c) {
            const double beats = (double) w.perf[TSR_PERF_CH_RD_BEATS(c)], wait = (double) w.perf[TSR_PERF_CH_WAIT(C, c)];
            const double gbs = dev_s > 0 ? beats * 64.0 / dev_s / 1e9 : 0.0, idle = cyc > 0 ? 100.0 * (cyc - beats - wait) / cyc : 0.0;
            ev.push_back(strf("{\"ph\":\"C\",\"pid\":0,\"name\":\"ch%u read\",\"ts\":%.4f,\"args\":{\"GB/s\":%.4f,\"idle %%\":%.2f}}", c,
                              (double) w.t_begin * us, gbs, idle));
            ev.push_back(strf("{\"ph\":\"C\",\"pid\":0,\"name\":\"ch%u read\",\"ts\":%.4f,\"args\":{\"GB/s\":0,\"idle %%\":100}}", c,
                              (double) w.t_end * us));
        }
    }
    std::string s = "{\"displayTimeUnit\":\"ns\",\"otherData\":{\"source\":\"tsr-prof trw (DB §8.3, §8.6)\"},\"traceEvents\":[\n";
    for (size_t i = 0; i < ev.size(); ++i) s += ev[i] + (i + 1 < ev.size() ? ",\n" : "\n");
    s += "]}\n";
    return s;
}

// ---------------------------------------------------------------------------------------------
// DB §8.7 self-checks
// ---------------------------------------------------------------------------------------------
trw_check_result trw_check(const std::vector<trw_rec> & recs, const uint64_t pd[64], uint32_t C, uint64_t trw_drops) {
    trw_check_result r;
    uint64_t ops = 0, run[8] = {};
    std::vector<std::pair<uint64_t, uint64_t>> launches;   // begin, end
    uint64_t lb = 0;
    bool in_launch = false, in_order = true;
    uint32_t last_pidx = 0;
    bool have_last = false;
    for (const trw_rec & x : recs) {
        if (x.kind == TSR_TRW_W0_KIND_OP) {
            ops++;
            if (x.unit < 8) run[x.unit] += x.run;
            if (have_last && (uint32_t) (x.a1 - last_pidx) >= 0x80000000u) in_order = false;
            last_pidx = x.a1;
            have_last = true;
        } else if (x.kind == TSR_TRW_W0_KIND_MARK) {
            const uint32_t cls = TSR_GET(x.a2, TSR_MARK_CLASS);
            if (cls == TSR_MARK_CLASS_LAUNCH_BEGIN) {
                lb = x.time;
                in_launch = true;
            } else if (cls == TSR_MARK_CLASS_LAUNCH_END && in_launch) {
                launches.emplace_back(lb, x.time);
                in_launch = false;
            }
        }
    }
    auto line = [&](bool ok, const std::string & s) {
        r.lines.push_back((ok ? "ok: " : "FAIL: ") + s);
        if (!ok) r.ok = false;
    };
    if (trw_drops == 0) line(ops == pd[TSR_PERF_RETIRED], strf("RETIRED %" PRIu64 " = OP records %" PRIu64, pd[TSR_PERF_RETIRED], ops));
    else r.lines.push_back(strf("skipped: RETIRED vs OP records (%" PRIu64 " records dropped)", trw_drops));
    const uint32_t X = TSR_PERF_X(C);
    struct { uint32_t unit; uint32_t slot; const char * name; } busy[] = {
        { TSR_UNIT_DMA, X + TSR_PERF_XOFF_DMA_CYCLES, "DMA_CYCLES" }, { TSR_UNIT_QACT, X + TSR_PERF_XOFF_QACT_CYCLES, "QACT_CYCLES" },
        { TSR_UNIT_VU, TSR_PERF_VU_CYCLES, "VU_CYCLES" }, { TSR_UNIT_ATT, X + TSR_PERF_XOFF_ATT_CYCLES, "ATT_CYCLES" } };
    if (trw_drops == 0) {
        for (const auto & b : busy)
            line(run[b.unit] == pd[b.slot], strf("%s: sum of RUN %" PRIu64 " = delta %s %" PRIu64, unit_name(b.unit), run[b.unit], b.name, pd[b.slot]));
        line(run[TSR_UNIT_MVM] >= pd[TSR_PERF_MVM_CYCLES],
             strf("MVM: sum of RUN %" PRIu64 " >= delta MVM_CYCLES %" PRIu64 " (overlap)", run[TSR_UNIT_MVM], pd[TSR_PERF_MVM_CYCLES]));
    }
    const uint64_t limit = pd[TSR_PERF_CYCLES] + pd[TSR_PERF_DBG_STOP_CYCLES];
    uint64_t span = 0;
    for (const auto & l : launches) span += l.second - l.first;
    if (!launches.empty())
        line(span <= limit, strf("launch windows (LAUNCH_END - LAUNCH_BEGIN) %" PRIu64 " <= delta CYCLES + DBG_STOP_CYCLES %" PRIu64 " over %zu launch%s",
                                 span, limit, launches.size(), launches.size() == 1 ? "" : "es"));
    r.lines.push_back(std::string(in_order ? "info: " : "info: not ") + "in push order (required only in serial mode)");
    r.lines.push_back("not checked: per-channel read beats (needs the LD / MVM / VU / AT beat formulas per record, DB §8.7)");
    return r;
}

} // namespace tsr::dbg

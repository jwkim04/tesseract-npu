// selftest.cpp — self-test of the fastreplay memory model (memmodel.h) without Verilator: a
// scripted "DUT" drives the port values; legal traffic must pass with the right data, ordering
// and tracking, and every protocol violation the model asserts (P1..P6) must be reported.
//
//   g++ -std=c++17 -O1 -I tb/fastreplay tb/fastreplay/selftest.cpp -o selftest && ./selftest
// (tb/fastreplay/test_fastreplay.py test_memmodel_selftest builds and runs it.)
// Exit status: number of failed cases.

#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "memmodel.h"

using fr::ChIn;
using fr::ChOut;
using fr::MemModel;
using fr::Regime;

namespace {

int g_fail = 0, g_cases = 0;

void expect(bool cond, const std::string& what) {
    if (!cond) {
        ++g_fail;
        printf("  FAIL: %s\n", what.c_str());
    }
}

// Deterministic regime: everything ready / valid, latency lat.
Regime ideal(uint64_t lat = 0) {
    Regime r;
    Regime::by_name("fast", &r);
    r.rd_lat_lo = r.rd_lat_hi = lat;
    r.wr_lat_lo = r.wr_lat_hi = lat;
    return r;
}

struct Fix {
    MemModel m;
    std::vector<ChOut> o;
    explicit Fix(const Regime& r, unsigned ch = 2, unsigned rdo = 4, bool has_mvm = false, uint64_t seed = 1) {
        MemModel::Config c;
        c.num_ch = ch;
        c.ch_bytes = 1ull << 20;          // 16384 beats
        c.limit = 8160;                   // below the channel size, not 64-aligned (ADDR_BEAT_W 5: 32)
        c.rd_outstanding = rdo;
        c.has_mvm = has_mvm;
        m.configure(c);
        m.prepare(r, seed);
        o.assign(ch, ChOut());
        // beat b of channel ch holds bytes (b * 7 + j + ch) & 0xFF
        std::vector<uint8_t> buf(64 * 1024);
        for (unsigned k = 0; k < ch; ++k) {
            for (size_t j = 0; j < buf.size(); ++j) buf[j] = (uint8_t)((j / 64) * 7 + (j % 64) + k);
            m.load(k, 0, buf.data(), buf.size());
            m.allow_beats(k, 0, 512);
        }
        m.set_active(true);
        edge();   // the model chooses its first outputs (the bench always runs cycles before a request)
    }
    void edge() { m.edge(o.data()); }
    void clear() {
        for (auto& x : o) x = ChOut();
    }
};

void run_case(const char* name, const std::function<void()>& f) {
    ++g_cases;
    const int before = g_fail;
    f();
    printf("%s %s\n", g_fail == before ? "ok  " : "FAIL", name);
}

uint8_t pat(uint64_t beat, unsigned j, unsigned ch) { return (uint8_t)(beat * 7 + j + ch); }

// Issue one read request on ch (held until accepted), then accept every beat; returns beats.
std::vector<std::vector<uint8_t>> read_burst(Fix& x, unsigned ch, uint64_t addr, unsigned len, unsigned id,
                                             std::vector<bool>* lasts = nullptr, unsigned max_cycles = 500) {
    std::vector<std::vector<uint8_t>> got;
    bool req = true;
    for (unsigned cyc = 0; cyc < max_cycles && (req || got.size() < std::max(len, 1u)); ++cyc) {
        ChOut& o = x.o[ch];
        o.rd_req_valid = req;
        o.rd_req_addr = addr;
        o.rd_req_len = len;
        o.rd_req_id = id;
        o.rd_rsp_ready = true;
        const ChIn in = x.m.in_[ch];   // values presented in this cycle
        const bool acc = req && in.rd_req_ready;
        if (in.rd_rsp_valid) {
            got.emplace_back(in.rd_rsp_data, in.rd_rsp_data + 64);
            if (lasts) lasts->push_back(in.rd_rsp_last);
        }
        x.edge();
        if (acc) req = false;
        x.o[ch].rd_req_valid = false;
    }
    x.clear();
    return got;
}

// Write a burst on ch: request and data offered independently (M5), then take the response.
bool write_burst(Fix& x, unsigned ch, uint64_t addr, unsigned len, unsigned id, uint8_t fill, int bad_last_beat = -1,
                 unsigned max_cycles = 500) {
    bool req = true, rsp = false;
    unsigned sent = 0;
    uint8_t data[64];
    for (unsigned cyc = 0; cyc < max_cycles && !rsp; ++cyc) {
        ChOut& o = x.o[ch];
        memset(data, fill + sent, 64);
        o.wr_req_valid = req;
        o.wr_req_addr = addr;
        o.wr_req_len = len;
        o.wr_req_id = id;
        o.wr_dat_valid = sent < len;
        o.wr_dat_data = data;
        o.wr_dat_last = (int)sent == bad_last_beat ? !(sent == len - 1) : (sent == len - 1);
        o.wr_rsp_ready = true;
        const ChIn in = x.m.in_[ch];
        const bool acc_req = req && in.wr_req_ready, acc_dat = o.wr_dat_valid && in.wr_dat_ready;
        if (in.wr_rsp_valid) rsp = true;
        x.edge();
        if (acc_req) req = false;
        if (acc_dat) ++sent;
    }
    x.clear();
    return rsp;
}

}  // namespace

int main() {
    // ---------------------------------------------------------------- legal traffic
    run_case("legal read burst: data, last, source count, idle afterwards", [] {
        Fix x(ideal(3));
        std::vector<bool> lasts;
        auto got = read_burst(x, 1, 100, 5, 0x4, &lasts);
        expect(got.size() == 5, "5 beats received");
        bool data_ok = got.size() == 5;
        for (unsigned b = 0; b < got.size(); ++b)
            for (unsigned j = 0; j < 64; ++j) data_ok &= got[b][j] == pat(100 + b, j, 1);
        expect(data_ok, "beat data equals the loaded memory");
        expect(lasts.size() == 5 && lasts[4] && !lasts[0] && !lasts[3], "rd_rsp_last on the last beat only");
        expect(x.m.stats(1).src_beats[fr::SRC_DMA] == 5 && x.m.stats(1).rd_beats == 5, "source DMA beats counted");
        expect(x.m.n_errors() == 0, "no protocol error");
        expect(x.m.idle(), "idle after the burst");
        expect(x.m.stats(1).bad_reads == 0, "reads inside the allowed beats");
    });
    run_case("legal write burst: memory, written set, one response, expect tracking", [] {
        Fix x(ideal(2));
        x.m.expect_beats(0, 200, 204);
        expect(write_burst(x, 0, 200, 4, 0x4, 0x30), "write response received");
        expect(x.m.mem(0)[200 * 64] == 0x30 && x.m.mem(0)[203 * 64 + 63] == 0x33, "data written in order");
        expect(x.m.written(0, 200) && x.m.written(0, 203) && !x.m.written(0, 204), "written set");
        expect(x.m.n_errors() == 0 && x.m.stats(0).extra_writes == 0, "no error");
        expect(x.m.idle(), "idle after the response");
    });
    run_case("read of a beat written earlier by the run is allowed; uninitialised read is counted", [] {
        Fix x(ideal(1));
        x.m.expect_beats(0, 1000, 1001);
        write_burst(x, 0, 1000, 1, 0x4, 0x55);
        read_burst(x, 0, 1000, 1, 0x4);
        expect(x.m.stats(0).bad_reads == 0, "written-then-read beat is fine");
        read_burst(x, 0, 2000, 2, 0x4);
        expect(x.m.stats(0).bad_reads == 2 && x.m.stats(0).first_bad_read == 2000, "2 uninitialised beats reported");
    });
    run_case("write outside the expect beats is counted", [] {
        Fix x(ideal(1));
        x.m.expect_beats(0, 300, 301);
        write_burst(x, 0, 300, 2, 0x4, 1);
        expect(x.m.stats(0).extra_writes == 1 && x.m.stats(0).first_extra_write == 301, "beat 301 is extra");
    });
    run_case("same-id read bursts return in request order; poison when no beat is presented", [] {
        Regime r = ideal(0);
        Fix x(r);
        // two requests back to back with id 4, then receive 2 + 3 beats
        std::vector<uint64_t> beats_seen;
        unsigned reqs = 0;
        for (unsigned cyc = 0; cyc < 200 && beats_seen.size() < 5; ++cyc) {
            ChOut& o = x.o[0];
            o.rd_req_valid = reqs < 2;
            o.rd_req_addr = reqs == 0 ? 10 : 20;
            o.rd_req_len = reqs == 0 ? 2 : 3;
            o.rd_req_id = 0x4;
            o.rd_rsp_ready = true;
            const ChIn in = x.m.in_[0];
            if (in.rd_rsp_valid) beats_seen.push_back(in.rd_rsp_data[0]);
            const bool acc = o.rd_req_valid && in.rd_req_ready;
            x.edge();
            if (acc) ++reqs;
        }
        expect(beats_seen.size() == 5 && beats_seen[0] == pat(10, 0, 0) && beats_seen[1] == pat(11, 0, 0) &&
                   beats_seen[2] == pat(20, 0, 0) && beats_seen[4] == pat(22, 0, 0),
               "beats in request order");
        x.clear();
        x.edge();
        const ChIn in = x.m.in_[0];
        expect(!in.rd_rsp_valid && in.rd_rsp_data[0] == 0xE5 && in.rd_rsp_data[63] == 0xE5, "poisoned data lines");
    });
    run_case("cross-id interleave happens in rand, never inside a started burst in ordered", [] {
        auto interleaves = [](const char* regime, uint64_t seed) {
            Regime r;
            Regime::by_name(regime, &r);
            r.p_rd_req_ready = 1.0;
            Fix x(r, 2, 16, false, seed);
            unsigned reqs = 0, got = 0, switches = 0;
            int last_src = -1;
            for (unsigned cyc = 0; cyc < 4000 && got < 64; ++cyc) {
                ChOut& o = x.o[0];
                o.rd_req_valid = reqs < 2;
                o.rd_req_addr = reqs == 0 ? 64 : 128;
                o.rd_req_len = 32;
                o.rd_req_id = reqs == 0 ? 0x0 : 0x4;   // FETCH, DMA
                o.rd_rsp_ready = true;
                const ChIn in = x.m.in_[0];
                if (in.rd_rsp_valid) {
                    ++got;
                    const int src = (int)(in.rd_rsp_id >> 2);
                    if (last_src >= 0 && src != last_src) ++switches;
                    last_src = src;
                }
                const bool acc = o.rd_req_valid && in.rd_req_ready;
                x.edge();
                if (acc) ++reqs;
            }
            return switches;
        };
        unsigned sw_rand = 0, sw_ord = 0;
        for (uint64_t s = 1; s <= 20; ++s) {
            sw_rand += interleaves("rand", s);
            sw_ord += interleaves("ordered", s) > 1;
        }
        expect(sw_rand > 20, "rand interleaves beats of different ids");
        expect(sw_ord == 0, "ordered: at most one switch (no interleave once a burst started)");
    });
    // ---------------------------------------------------------------- violations
    run_case("P1 burst length 0 and 65", [] {
        Fix x(ideal());
        read_burst(x, 0, 0, 0, 0x4, nullptr, 20);
        expect(x.m.n_errors() >= 1, "len 0 reported");
        Fix y(ideal());
        read_burst(y, 0, 0, 65, 0x4, nullptr, 200);
        expect(y.m.n_errors() >= 1, "len 65 reported");
    });
    run_case("P1 read burst crossing a 4 KiB block", [] {
        Fix x(ideal());
        read_burst(x, 0, 60, 8, 0x4);
        expect(x.m.n_errors() >= 1, "crossing reported");
    });
    run_case("P1 burst beyond the DRAM limit (read and write)", [] {
        Fix x(ideal());
        read_burst(x, 0, 8160 - 2, 2, 0x4);
        expect(x.m.n_errors() == 0, "burst ending exactly at the limit is legal");
        read_burst(x, 0, 8160 - 2, 4, 0x4);        // inside its 4 KiB block, beyond the limit
        expect(x.m.n_errors() == 1, "read beyond the limit reported");
        Fix y(ideal());
        write_burst(y, 1, 8160 - 1, 2, 0x4, 0);
        expect(y.m.n_errors() == 1, "write beyond the limit reported");
    });
    run_case("P2 read requests beyond RD_OUTSTANDING (summed over sources)", [] {
        Fix x(ideal(50), 2, 2);
        for (unsigned i = 0; i < 3; ++i) {     // three single-cycle requests, no response taken
            x.o[0].rd_req_valid = true;
            x.o[0].rd_req_addr = 64 * i;
            x.o[0].rd_req_len = 4;
            x.o[0].rd_req_id = i == 1 ? 0x0 : 0x4;
            x.o[0].rd_rsp_ready = false;
            x.edge();
        }
        expect(x.m.n_errors() == 1, "third request reported");
        Fix y(ideal(50), 2, 3);
        for (unsigned i = 0; i < 3; ++i) {
            y.o[0].rd_req_valid = true;
            y.o[0].rd_req_addr = 64 * i;
            y.o[0].rd_req_len = 4;
            y.o[0].rd_req_id = 0x4;
            y.edge();
        }
        expect(y.m.n_errors() == 0, "three requests with RD_OUTSTANDING 3 are legal");
    });
    run_case("P3 rd_req payload change and valid drop before the handshake", [] {
        Regime r = ideal();
        r.p_rd_req_ready = 0.0;
        Fix x(r);
        x.o[0].rd_req_valid = true, x.o[0].rd_req_addr = 5, x.o[0].rd_req_len = 1, x.o[0].rd_req_id = 4;
        x.edge();
        x.o[0].rd_req_addr = 6;
        x.edge();
        expect(x.m.n_errors() == 1, "payload change reported");
        x.o[0].rd_req_valid = false;
        x.edge();
        expect(x.m.n_errors() == 2, "valid drop reported");
    });
    run_case("P3 wr_req / wr_dat payload change before the handshake", [] {
        Regime r = ideal();
        r.p_wr_req_ready = 0.0;
        r.p_wr_dat_ready = 0.0;
        Fix x(r);
        uint8_t d[64] = {1};
        x.o[0].wr_req_valid = true, x.o[0].wr_req_addr = 7, x.o[0].wr_req_len = 1, x.o[0].wr_req_id = 4;
        x.o[0].wr_dat_valid = true, x.o[0].wr_dat_data = d, x.o[0].wr_dat_last = true;
        x.edge();
        x.o[0].wr_req_len = 2;
        d[10] = 9;
        x.edge();
        expect(x.m.n_errors() == 2, "both changes reported");
    });
    run_case("P4 wr_dat_last on the wrong beat", [] {
        Fix x(ideal());
        x.m.expect_beats(0, 0, 64);
        write_burst(x, 0, 8, 4, 0x4, 0, 1);
        expect(x.m.n_errors() >= 1, "early last reported");
        Fix y(ideal());
        y.m.expect_beats(0, 0, 64);
        write_burst(y, 0, 8, 4, 0x4, 0, 3);
        expect(y.m.n_errors() >= 1, "missing last reported");
    });
    run_case("P5 ids: FETCH on channel 1, WS without MVM, source 3, write not from DMA", [] {
        Fix x(ideal());
        read_burst(x, 1, 0, 1, 0x0);
        expect(x.m.n_errors() == 1, "fetch on ch1");
        read_burst(x, 0, 0, 1, 0x0);
        expect(x.m.n_errors() == 1, "fetch on ch0 is legal");
        read_burst(x, 0, 0, 1, 0x8);
        expect(x.m.n_errors() == 2, "WS without has_mvm");
        read_burst(x, 0, 0, 1, 0xC);
        expect(x.m.n_errors() == 3, "source 3");
        x.m.expect_beats(0, 0, 8);
        write_burst(x, 0, 0, 1, 0x8, 0);
        expect(x.m.n_errors() == 4, "write from WS");
        Fix y(ideal(), 2, 4, true);
        read_burst(y, 1, 0, 2, 0x8);
        expect(y.m.n_errors() == 0 && y.m.stats(1).src_beats[fr::SRC_WS] == 2, "WS with has_mvm counted");
    });
    run_case("P6 request offered while no run is active", [] {
        Fix x(ideal());
        x.m.set_active(false);
        read_burst(x, 0, 0, 1, 0x4);
        expect(x.m.n_errors() >= 1, "reported");
    });
    run_case("M5: data_first never accepts a request without data; req_first never data without request", [] {
        Regime r = ideal();
        r.wr_order = fr::WR_DATA_FIRST;
        Fix x(r);
        unsigned acc = 0;
        for (unsigned i = 0; i < 100; ++i) {
            x.o[0].wr_req_valid = true, x.o[0].wr_req_addr = 0, x.o[0].wr_req_len = 2, x.o[0].wr_req_id = 4;
            acc += x.m.in_[0].wr_req_ready;
            x.edge();
        }
        expect(acc == 0, "request waits for data");
        Regime q = ideal();
        q.wr_order = fr::WR_REQ_FIRST;
        Fix y(q);
        uint8_t d[64] = {0};
        acc = 0;
        for (unsigned i = 0; i < 100; ++i) {
            y.o[0].wr_dat_valid = true, y.o[0].wr_dat_data = d, y.o[0].wr_dat_last = false;
            acc += y.m.in_[0].wr_dat_ready;
            y.edge();
        }
        expect(acc == 0, "data waits for its request");
    });
    run_case("hardware reset drops transactions; idle() sees reads in flight", [] {
        Fix x(ideal(30));
        x.o[0].rd_req_valid = true, x.o[0].rd_req_addr = 0, x.o[0].rd_req_len = 4, x.o[0].rd_req_id = 4;
        x.edge();
        x.clear();
        expect(!x.m.idle(), "busy with a read in flight");
        x.m.reset_transactions();
        expect(x.m.idle(), "idle after reset");
    });
    printf("selftest: %d cases, %d check(s) failed\n", g_cases, g_fail);
    return g_fail;
}

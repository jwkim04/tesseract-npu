// memmodel.h — cycle-level C++ model of the tsr_core memory ports (docs/rtl-core.md §4), the fast
// counterpart of tb/common/mem_model.py (MemModel) for the C++ benches (tb/platform/bw; the
// tile-only replay of tb/fastreplay, docs/design SW §10.3).
//
// One MemModel serves NUM_CH channels. It knows nothing about Verilator: the testbench samples
// the DUT outputs of every channel into a ChOut before the rising edge, calls edge(), lets the DUT
// take the edge, and drives the ChIn values (in_[c]) into the DUT. edge() first performs the
// handshakes of that edge with the values both sides presented (transfer = valid && ready at the
// edge, rule R1), then chooses the outputs for the next cycle — the falling-edge sample /
// rising-edge drive discipline of the Python model.
//
// Behavior (seeded per channel; Regime): random request acceptance (backpressure on rd_req,
// wr_req, wr_dat), random latency per burst, random gaps in rd_rsp / wr_rsp valid, read data in
// order within one id and interleaved at beat granularity across ids (or no interleave once a
// burst has started: "ordered"), write responses in order per id and reordered across ids, random
// stall phases; write ordering modes per channel: any / req_first (data beats only for requests
// already accepted) / data_first (a request only after one of its data beats) / random (a mode
// per channel every 50..400 cycles). req_first / data_first turn a core that waits for one
// write handshake before offering the other (rule M5) into a hang.
//
// Protocol assertions (every violation is recorded in errors(); the testbench fails the trace):
//   P1  read / write burst length 1..64, inside one 64-beat (4 KiB) block, below the DRAM limit
//   P2  at most rd_outstanding read requests outstanding per channel, summed over all sources
//       (request handshake .. handshake of its last beat)
//   P3  valid stays high and the payload stable until the handshake (rd_req, wr_req, wr_dat)
//   P4  write data: exactly len beats per request, wr_dat_last on (and only on) each burst's last
//       beat (data beats follow request order; full-beat writes: the port has no strobes)
//   P5  ids: read source (id[3:2]) AUX (0, the former fetch source) only on channel 0, WS only
//       with has_mvm, never 3; write source DMA only
//   P6  no request / write data offered while no run is active (set_active; DONE => quiescent)
// Tracking for the trace checks: per channel read beats per source (id[3:2]), reads of beats
// that were neither initialized (allowed bitmap: init regions + command beats) nor written
// earlier by the run, writes outside the expect bitmap, written bitmap, max outstanding.
// Memory: a flat MAP_NORESERVE mapping of ch_bytes per channel (zero until loaded, pages are
// only materialized where touched), bitmaps likewise; released by release().
#pragma once

#include <sys/mman.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace fr {

// ---------------------------------------------------------------------------------------------
// Random numbers: xoshiro256** seeded by splitmix64 (fast, reproducible across platforms)
// ---------------------------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Rng {
    uint64_t s[4] = {1, 2, 3, 4};
    void seed(uint64_t v) {
        for (auto& w : s) w = splitmix64(v);
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    // uniform integer in [lo, hi]
    inline uint64_t range(uint64_t lo, uint64_t hi) {
        if (hi <= lo) return lo;
        return lo + (uint64_t)(((unsigned __int128)next() * (hi - lo + 1)) >> 64);
    }
    // true with probability thr / 2^32 (thr = 2^32: always, 0: never)
    inline bool chance(uint64_t thr) {
        if (thr >= (1ull << 32)) return true;
        if (thr == 0) return false;
        return (next() >> 32) < thr;
    }
};

static inline uint64_t prob_thr(double p) {
    if (p <= 0.0) return 0;
    if (p >= 1.0) return 1ull << 32;
    return (uint64_t)(p * 4294967296.0);
}

// ---------------------------------------------------------------------------------------------
// Regimes (the REGIMES of the core bench, on top of the MemModel defaults)
// ---------------------------------------------------------------------------------------------
enum WrOrder { WR_ANY = 0, WR_REQ_FIRST = 1, WR_DATA_FIRST = 2, WR_RANDOM = 3 };

struct Regime {
    std::string name = "rand";
    uint64_t rd_lat_lo = 1, rd_lat_hi = 40, wr_lat_lo = 1, wr_lat_hi = 40;
    double p_rd_req_ready = 0.7, p_rd_rsp_valid = 0.8, p_wr_req_ready = 0.7, p_wr_dat_ready = 0.7,
           p_wr_rsp_valid = 0.8, p_stall = 0.01;
    uint64_t stall_lo = 1, stall_hi = 30;
    bool interleave = true;
    int wr_order = WR_RANDOM;

    static const char* const* names() {
        static const char* const n[] = {"rand", "fast", "slow", "ordered", nullptr};
        return n;
    }
    // Returns false for an unknown name.
    static bool by_name(const std::string& n, Regime* r) {
        *r = Regime();
        r->name = n;
        if (n == "rand") return true;
        if (n == "fast") {
            r->rd_lat_lo = 0, r->rd_lat_hi = 2, r->wr_lat_lo = 0, r->wr_lat_hi = 2;
            r->p_rd_req_ready = r->p_rd_rsp_valid = r->p_wr_req_ready = r->p_wr_dat_ready = r->p_wr_rsp_valid = 1.0;
            r->p_stall = 0.0;
            r->wr_order = WR_ANY;
            return true;
        }
        if (n == "slow") {
            r->rd_lat_lo = 20, r->rd_lat_hi = 160, r->wr_lat_lo = 10, r->wr_lat_hi = 120;
            r->p_rd_req_ready = 0.3, r->p_rd_rsp_valid = 0.45, r->p_wr_req_ready = 0.3, r->p_wr_dat_ready = 0.45;
            r->p_wr_rsp_valid = 0.4, r->p_stall = 0.02, r->stall_lo = 5, r->stall_hi = 60;
            return true;
        }
        if (n == "ordered") {
            r->interleave = false;
            r->wr_order = WR_REQ_FIRST;
            r->p_rd_rsp_valid = 0.9;
            return true;
        }
        return false;
    }
    // Override one field: key=value (lat ranges as lo:hi). Returns an error message or "".
    std::string set(const std::string& kv) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) return "expected key=value: " + kv;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        auto range = [&](uint64_t* lo, uint64_t* hi) -> std::string {
            const size_t c = v.find(':');
            char* e = nullptr;
            *lo = strtoull(v.c_str(), &e, 0);
            *hi = c == std::string::npos ? *lo : strtoull(v.c_str() + c + 1, &e, 0);
            return *hi < *lo ? "bad range " + v : "";
        };
        if (k == "rd_lat") return range(&rd_lat_lo, &rd_lat_hi);
        if (k == "wr_lat") return range(&wr_lat_lo, &wr_lat_hi);
        if (k == "stall_len") return range(&stall_lo, &stall_hi);
        if (k == "interleave") { interleave = atoi(v.c_str()) != 0; return ""; }
        if (k == "wr_order") {
            if (v == "any") wr_order = WR_ANY;
            else if (v == "req_first") wr_order = WR_REQ_FIRST;
            else if (v == "data_first") wr_order = WR_DATA_FIRST;
            else if (v == "random") wr_order = WR_RANDOM;
            else return "bad wr_order " + v;
            return "";
        }
        double* p = nullptr;
        if (k == "p_rd_req_ready") p = &p_rd_req_ready;
        else if (k == "p_rd_rsp_valid") p = &p_rd_rsp_valid;
        else if (k == "p_wr_req_ready") p = &p_wr_req_ready;
        else if (k == "p_wr_dat_ready") p = &p_wr_dat_ready;
        else if (k == "p_wr_rsp_valid") p = &p_wr_rsp_valid;
        else if (k == "p_stall") p = &p_stall;
        if (p == nullptr) return "unknown memory parameter " + k;
        *p = atof(v.c_str());
        return "";
    }
};

// ---------------------------------------------------------------------------------------------
// Port values of one channel
// ---------------------------------------------------------------------------------------------
struct ChOut {                 // DUT -> memory, sampled before the edge
    bool rd_req_valid = false;
    uint64_t rd_req_addr = 0;
    unsigned rd_req_len = 0, rd_req_id = 0;
    bool rd_rsp_ready = false;
    bool wr_req_valid = false;
    uint64_t wr_req_addr = 0;
    unsigned wr_req_len = 0, wr_req_id = 0;
    bool wr_dat_valid = false;
    const uint8_t* wr_dat_data = nullptr;   // 64 bytes, valid until the next DUT evaluation
    bool wr_dat_last = false;
    bool wr_rsp_ready = false;
};

struct ChIn {                  // memory -> DUT, for the coming cycle
    bool rd_req_ready = false;
    bool rd_rsp_valid = false, rd_rsp_last = false;
    unsigned rd_rsp_id = 0;
    uint8_t rd_rsp_data[64] = {0};
    bool rd_rsp_new = false;   // a new beat was put on rd_rsp_data (the testbench copies it)
    bool wr_req_ready = false, wr_dat_ready = false;
    bool wr_rsp_valid = false;
    unsigned wr_rsp_id = 0;
};

enum { SRC_AUX = 0, SRC_DMA = 1, SRC_WS = 2, SRC_RSVD = 3 };

// ---------------------------------------------------------------------------------------------
// Bitmaps over beats (lazily backed)
// ---------------------------------------------------------------------------------------------
static inline void* vm_alloc(size_t n) {
    void* p = mmap(nullptr, n ? n : 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "memmodel: mmap of %zu bytes failed\n", n);
        abort();
    }
    return p;
}
static inline void vm_free(void* p, size_t n) {
    if (p != nullptr) munmap(p, n ? n : 4096);
}

struct Bitmap {
    uint64_t* w = nullptr;
    size_t bytes = 0;
    uint64_t nbits = 0;
    void alloc(uint64_t n) {
        free_();
        nbits = n;
        bytes = (size_t)(((n + 63) / 64) * 8);
        w = (uint64_t*)vm_alloc(bytes);
    }
    void free_() {
        vm_free(w, bytes);
        w = nullptr;
        bytes = 0;
        nbits = 0;
    }
    inline bool get(uint64_t i) const { return i < nbits && ((w[i >> 6] >> (i & 63)) & 1); }
    inline bool set(uint64_t i) {  // returns the previous value
        if (i >= nbits) return true;
        const uint64_t m = 1ull << (i & 63);
        const bool old = (w[i >> 6] & m) != 0;
        w[i >> 6] |= m;
        return old;
    }
    void set_range(uint64_t lo, uint64_t hi) {
        for (uint64_t i = lo; i < hi && i < nbits; ++i) set(i);
    }
};

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------
class MemModel {
  public:
    struct Config {
        unsigned num_ch = 4;
        uint64_t ch_bytes = 0;        // bytes per channel (host loads may use all of it)
        uint64_t limit = 0;           // DRAM limit in beats: min(ch_bytes / 64, 2^addr_beat_w)
        unsigned rd_outstanding = 16; // the RTL's RD_OUTSTANDING
        bool has_mvm = false;
    };

    struct Stats {
        uint64_t rd_bursts = 0, rd_beats = 0, wr_bursts = 0, wr_beats = 0;
        uint64_t src_beats[4] = {0, 0, 0, 0};
        unsigned max_outstanding = 0;
        uint64_t bad_reads = 0, first_bad_read = 0;       // beats neither allowed nor written before
        uint64_t extra_writes = 0, first_extra_write = 0; // written beats outside the expect bitmap
    };

    std::vector<ChIn> in_;     // current model outputs (drive these)

    MemModel() = default;
    MemModel(const MemModel&) = delete;
    MemModel& operator=(const MemModel&) = delete;
    ~MemModel() { release(); }

    void configure(const Config& c) {
        release();
        cfg_ = c;
        ch_.clear();
        ch_.resize(c.num_ch);
        in_.assign(c.num_ch, ChIn());
        for (unsigned i = 0; i < c.num_ch; ++i) ch_[i].idx = i;
        errors_.clear();
        n_errors_ = 0;
    }
    const Config& config() const { return cfg_; }

    // Fresh memory for a trace (zero), regime and seed; no transaction in flight.
    void prepare(const Regime& r, uint64_t seed) {
        release();
        reg_ = r;
        thr_rd_req_ = prob_thr(r.p_rd_req_ready);
        thr_rd_rsp_ = prob_thr(r.p_rd_rsp_valid);
        thr_wr_req_ = prob_thr(r.p_wr_req_ready);
        thr_wr_dat_ = prob_thr(r.p_wr_dat_ready);
        thr_wr_rsp_ = prob_thr(r.p_wr_rsp_valid);
        thr_stall_ = prob_thr(r.p_stall);
        const uint64_t beats = cfg_.ch_bytes / 64;
        for (auto& c : ch_) {
            c.mem = (uint8_t*)vm_alloc(cfg_.ch_bytes);
            c.written.alloc(beats);
            c.allowed.alloc(beats);
            c.expect.alloc(beats);
            uint64_t s = seed * 1000003ull ^ (c.idx * 7919ull + 17);
            c.rng.seed(splitmix64(s));
            c.stats = Stats();
            c.mode = WR_ANY;
            c.mode_left = 0;
            c.stall = 0;
        }
        reset_transactions();
        errors_.clear();
        n_errors_ = 0;
        active_ = false;
    }

    void release() {
        for (auto& c : ch_) {
            vm_free(c.mem, cfg_.ch_bytes);
            c.mem = nullptr;
            c.written.free_();
            c.allowed.free_();
            c.expect.free_();
        }
    }

    // Host-side load (not a write of the run). Clipped at the channel end.
    void load(unsigned ch, uint64_t byte_off, const uint8_t* data, uint64_t size) {
        if (ch >= ch_.size() || byte_off >= cfg_.ch_bytes) return;
        if (size > cfg_.ch_bytes - byte_off) size = cfg_.ch_bytes - byte_off;
        memcpy(ch_[ch].mem + byte_off, data, size);
    }
    void allow_beats(unsigned ch, uint64_t lo, uint64_t hi) { ch_[ch].allowed.set_range(lo, hi); }
    void expect_beats(unsigned ch, uint64_t lo, uint64_t hi) { ch_[ch].expect.set_range(lo, hi); }
    const uint8_t* mem(unsigned ch) const { return ch_[ch].mem; }
    bool written(unsigned ch, uint64_t beat) const { return ch_[ch].written.get(beat); }
    const Stats& stats(unsigned ch) const { return ch_[ch].stats; }

    // Runs: requests / write data are legal only while active (P6).
    void set_active(bool a) { active_ = a; }
    bool active() const { return active_; }

    // Hardware reset of the core: every transaction in flight is dropped, outputs low.
    void reset_transactions() {
        for (auto& c : ch_) {
            c.rd.clear();
            c.outstanding = 0;
            c.rsp_valid = false;
            c.rsp_poison = false;
            c.rsp_seq = 0;
            c.wreqs.clear();
            c.wdata.clear();
            c.wrsp.clear();
            c.wrsp_valid = false;
            c.wrsp_seq = 0;
            c.pend_rd_req = c.pend_wr_req = c.pend_wr_dat = false;
            c.rd_req_ready = c.wr_req_ready = c.wr_dat_ready = false;
        }
        for (auto& i : in_) i = ChIn();
    }

    // No transaction in flight on any channel (accepted read bursts, write requests, data beats,
    // write responses pending or presented).
    bool idle() const {
        for (const auto& c : ch_)
            if (!c.rd.empty() || !c.wreqs.empty() || !c.wdata.empty() || !c.wrsp.empty() || c.wrsp_valid) return false;
        return true;
    }
    std::string busy_desc() const {
        std::string s;
        char b[256];
        for (const auto& c : ch_) {
            if (c.rd.empty() && c.wreqs.empty() && c.wdata.empty() && c.wrsp.empty() && !c.wrsp_valid) continue;
            snprintf(b, sizeof b, "ch%u: %zu rd bursts, %zu wr reqs, %zu wr beats, %zu wr rsps; ", c.idx, c.rd.size(),
                     c.wreqs.size(), c.wdata.size(), c.wrsp.size());
            s += b;
        }
        return s;
    }

    const std::vector<std::string>& errors() const { return errors_; }
    uint64_t n_errors() const { return n_errors_; }
    uint64_t cycle() const { return cycle_; }

    // One rising edge: handshakes with the values presented in this cycle, then the outputs for
    // the next cycle (in_). o[c] for every channel.
    void edge(const ChOut* o) {
        ++cycle_;
        for (auto& c : ch_) update(c, o[c.idx]);
        for (auto& c : ch_) choose(c);
    }

    void fail(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        ++n_errors_;
        if (errors_.size() >= 16) return;
        char b[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(b, sizeof b, fmt, ap);
        va_end(ap);
        char c[600];
        snprintf(c, sizeof c, "mem cycle %llu: %s", (unsigned long long)cycle_, b);
        errors_.push_back(c);
    }

  private:
    struct RdBurst {
        unsigned id;
        uint64_t addr;
        unsigned len, sent;
        uint64_t due, seq;
    };
    struct WReq {
        unsigned id;
        uint64_t addr;
        unsigned len;
    };
    struct WBeat {
        uint8_t d[64];
        bool last;
    };
    struct WRsp {
        unsigned id;
        uint64_t due, seq;
    };
    struct Chan {
        unsigned idx = 0;
        Rng rng;
        uint8_t* mem = nullptr;
        Bitmap written, allowed, expect;
        std::vector<RdBurst> rd;      // accepted, unfinished, in acceptance order
        unsigned outstanding = 0;
        bool rsp_valid = false;
        bool rsp_poison = false;      // data lines still show the last accepted beat
        uint64_t rsp_seq = 0;         // burst whose beat is presented
        std::deque<WReq> wreqs;       // accepted, not yet complete
        std::deque<WBeat> wdata;      // accepted data beats not yet committed
        std::vector<WRsp> wrsp;       // pending write responses
        bool wrsp_valid = false;
        uint64_t wrsp_seq = 0;
        bool rd_req_ready = false, wr_req_ready = false, wr_dat_ready = false;
        bool pend_rd_req = false, pend_wr_req = false, pend_wr_dat = false;
        uint64_t prq_addr = 0, pwq_addr = 0;
        unsigned prq_len = 0, prq_id = 0, pwq_len = 0, pwq_id = 0;
        uint8_t pwd_data[64];
        bool pwd_last = false;
        uint64_t stall = 0;
        int mode = WR_ANY;
        int64_t mode_left = 0;
        Stats stats;
    };

    Config cfg_;
    Regime reg_;
    uint64_t thr_rd_req_ = 0, thr_rd_rsp_ = 0, thr_wr_req_ = 0, thr_wr_dat_ = 0, thr_wr_rsp_ = 0, thr_stall_ = 0;
    std::vector<Chan> ch_;
    std::vector<std::string> errors_;
    uint64_t n_errors_ = 0;
    uint64_t cycle_ = 0, seq_ = 0;
    bool active_ = false;

    void check_burst(const char* kind, unsigned c, uint64_t addr, unsigned len) {
        if (len < 1 || len > 64) fail("ch%u %s burst length %u not in 1..64 (addr 0x%llx)", c, kind, len, (unsigned long long)addr);
        if ((addr % 64) + len > 64)
            fail("ch%u %s burst [0x%llx, +%u) crosses a 4 KiB boundary", c, kind, (unsigned long long)addr, len);
        if (addr + len > cfg_.limit)
            fail("ch%u %s burst [0x%llx, +%u) beyond the DRAM limit 0x%llx", c, kind, (unsigned long long)addr, len,
                 (unsigned long long)cfg_.limit);
    }

    void update(Chan& ch, const ChOut& o) {
        const unsigned c = ch.idx;
        if (!active_ && (o.rd_req_valid || o.wr_req_valid || o.wr_dat_valid))
            fail("ch%u: %s offered while no run is active", c,
                 o.rd_req_valid ? "read request" : o.wr_req_valid ? "write request" : "write data");
        // ---- read request ----
        if (o.rd_req_valid) {
            if (ch.pend_rd_req && (o.rd_req_addr != ch.prq_addr || o.rd_req_len != ch.prq_len || o.rd_req_id != ch.prq_id))
                fail("ch%u rd_req payload changed before the handshake: (0x%llx,%u,%u) -> (0x%llx,%u,%u)", c,
                     (unsigned long long)ch.prq_addr, ch.prq_len, ch.prq_id, (unsigned long long)o.rd_req_addr,
                     o.rd_req_len, o.rd_req_id);
            if (ch.rd_req_ready) {
                ch.pend_rd_req = false;
                check_burst("read", c, o.rd_req_addr, o.rd_req_len);
                if (ch.outstanding >= cfg_.rd_outstanding)
                    fail("ch%u read request beyond RD_OUTSTANDING=%u", c, cfg_.rd_outstanding);
                const unsigned src = (o.rd_req_id >> 2) & 3;
                if (src == SRC_RSVD || (src == SRC_AUX && c != 0) || (src == SRC_WS && !cfg_.has_mvm))
                    fail("ch%u read request with id 0x%x (source %u not allowed here)", c, o.rd_req_id, src);
                RdBurst b;
                b.id = o.rd_req_id;
                b.addr = o.rd_req_addr;
                b.len = o.rd_req_len ? o.rd_req_len : 1;
                b.sent = 0;
                b.due = cycle_ + ch.rng.range(reg_.rd_lat_lo, reg_.rd_lat_hi);
                b.seq = ++seq_;
                ch.rd.push_back(b);
                ++ch.outstanding;
                if (ch.outstanding > ch.stats.max_outstanding) ch.stats.max_outstanding = ch.outstanding;
                ++ch.stats.rd_bursts;
            } else {
                ch.pend_rd_req = true;
                ch.prq_addr = o.rd_req_addr;
                ch.prq_len = o.rd_req_len;
                ch.prq_id = o.rd_req_id;
            }
        } else if (ch.pend_rd_req) {
            fail("ch%u rd_req_valid dropped before the handshake", c);
            ch.pend_rd_req = false;
        }
        // ---- read response beat ----
        if (ch.rsp_valid && o.rd_rsp_ready) {
            size_t k = 0;
            while (k < ch.rd.size() && ch.rd[k].seq != ch.rsp_seq) ++k;
            if (k == ch.rd.size()) abort();   // model invariant: the presented burst is in flight
            RdBurst& b = ch.rd[k];
            const uint64_t beat = b.addr + b.sent;
            if (beat < cfg_.limit && !ch.written.get(beat) && !ch.allowed.get(beat)) {
                if (ch.stats.bad_reads++ == 0) ch.stats.first_bad_read = beat;
            }
            ++b.sent;
            ++ch.stats.rd_beats;
            ++ch.stats.src_beats[(b.id >> 2) & 3];
            if (b.sent == b.len) {
                ch.rd.erase(ch.rd.begin() + (long)k);
                --ch.outstanding;
            }
            ch.rsp_valid = false;
            ch.rsp_poison = true;   // choose(): poison the data lines unless a new beat follows
        }
        // ---- write request ----
        if (o.wr_req_valid) {
            if (ch.pend_wr_req && (o.wr_req_addr != ch.pwq_addr || o.wr_req_len != ch.pwq_len || o.wr_req_id != ch.pwq_id))
                fail("ch%u wr_req payload changed before the handshake", c);
            if (ch.wr_req_ready) {
                ch.pend_wr_req = false;
                check_burst("write", c, o.wr_req_addr, o.wr_req_len);
                if (((o.wr_req_id >> 2) & 3) != SRC_DMA) fail("ch%u write request with id 0x%x (not source DMA)", c, o.wr_req_id);
                ch.wreqs.push_back(WReq{o.wr_req_id, o.wr_req_addr, o.wr_req_len ? o.wr_req_len : 1});
                ++ch.stats.wr_bursts;
            } else {
                ch.pend_wr_req = true;
                ch.pwq_addr = o.wr_req_addr;
                ch.pwq_len = o.wr_req_len;
                ch.pwq_id = o.wr_req_id;
            }
        } else if (ch.pend_wr_req) {
            fail("ch%u wr_req_valid dropped before the handshake", c);
            ch.pend_wr_req = false;
        }
        // ---- write data ----
        if (o.wr_dat_valid) {
            if (ch.pend_wr_dat && (o.wr_dat_last != ch.pwd_last || memcmp(o.wr_dat_data, ch.pwd_data, 64) != 0))
                fail("ch%u wr_dat payload changed before the handshake", c);
            if (ch.wr_dat_ready) {
                ch.pend_wr_dat = false;
                ch.wdata.emplace_back();
                memcpy(ch.wdata.back().d, o.wr_dat_data, 64);
                ch.wdata.back().last = o.wr_dat_last;
                ++ch.stats.wr_beats;
            } else {
                ch.pend_wr_dat = true;
                memcpy(ch.pwd_data, o.wr_dat_data, 64);
                ch.pwd_last = o.wr_dat_last;
            }
        } else if (ch.pend_wr_dat) {
            fail("ch%u wr_dat_valid dropped before the handshake", c);
            ch.pend_wr_dat = false;
        }
        // ---- commit complete write bursts ----
        while (!ch.wreqs.empty() && ch.wdata.size() >= ch.wreqs.front().len) {
            const WReq w = ch.wreqs.front();
            ch.wreqs.pop_front();
            for (unsigned i = 0; i < w.len; ++i) {
                const WBeat& d = ch.wdata.front();
                if (d.last != (i == w.len - 1))
                    fail("ch%u write burst at 0x%llx len %u: wr_dat_last=%d on beat %u", c, (unsigned long long)w.addr, w.len,
                         (int)d.last, i);
                const uint64_t beat = w.addr + i;
                if (beat < cfg_.limit) {
                    memcpy(ch.mem + beat * 64, d.d, 64);
                    ch.written.set(beat);
                    if (!ch.expect.get(beat) && ch.stats.extra_writes++ == 0) ch.stats.first_extra_write = beat;
                }
                ch.wdata.pop_front();
            }
            ch.wrsp.push_back(WRsp{w.id, cycle_ + ch.rng.range(reg_.wr_lat_lo, reg_.wr_lat_hi), ++seq_});
        }
        // ---- write response ----
        if (ch.wrsp_valid && o.wr_rsp_ready) {
            for (size_t k = 0; k < ch.wrsp.size(); ++k) {
                if (ch.wrsp[k].seq == ch.wrsp_seq) {
                    ch.wrsp.erase(ch.wrsp.begin() + (long)k);
                    break;
                }
            }
            ch.wrsp_valid = false;
        }
    }

    // Index of a random eligible item: the oldest pending item of each id whose latency has
    // elapsed (same-id order kept). started_only: prefer read bursts that already sent a beat.
    template <typename T>
    long pick(Chan& ch, const std::vector<T>& items, bool prefer_started) {
        long cand[32];
        int n = 0, n_started = 0;
        long started[32];
        unsigned seen = 0;   // ids are 4 bits
        for (size_t k = 0; k < items.size(); ++k) {
            const unsigned id = items[k].id & 15;
            if (seen & (1u << id)) continue;
            seen |= 1u << id;
            if (items[k].due <= cycle_ && n < 32) {
                cand[n++] = (long)k;
                if (prefer_started && is_started(items[k])) started[n_started++] = (long)k;
            }
        }
        if (n_started) return started[ch.rng.range(0, (uint64_t)n_started - 1)];
        if (n == 0) return -1;
        return cand[n == 1 ? 0 : ch.rng.range(0, (uint64_t)n - 1)];
    }
    static bool is_started(const RdBurst& b) { return b.sent > 0; }
    static bool is_started(const WRsp&) { return false; }

    void choose(Chan& ch) {
        ChIn& in = in_[ch.idx];
        in.rd_rsp_new = false;
        // stall phases: everything quiet for a while
        if (ch.stall > 0) --ch.stall;
        else if (thr_stall_ && ch.rng.chance(thr_stall_)) ch.stall = ch.rng.range(reg_.stall_lo, reg_.stall_hi);
        const bool stalled = ch.stall > 0;
        // write ordering mode
        if (reg_.wr_order == WR_RANDOM) {
            if (ch.mode_left <= 0) {
                ch.mode = (int)ch.rng.range(0, 2);
                ch.mode_left = (int64_t)ch.rng.range(50, 400);
            }
            --ch.mode_left;
        } else {
            ch.mode = reg_.wr_order;
        }
        ch.rd_req_ready = !stalled && ch.rng.chance(thr_rd_req_);
        int64_t need = 0;
        for (const auto& w : ch.wreqs) need += w.len;
        need -= (int64_t)ch.wdata.size();
        if (ch.mode == WR_REQ_FIRST) {
            ch.wr_req_ready = !stalled && ch.rng.chance(thr_wr_req_);
            ch.wr_dat_ready = !stalled && need > 0 && ch.rng.chance(thr_wr_dat_);
        } else if (ch.mode == WR_DATA_FIRST) {
            ch.wr_req_ready = !stalled && need < 0 && ch.rng.chance(thr_wr_req_);
            ch.wr_dat_ready = !stalled && ch.rng.chance(thr_wr_dat_);
        } else {
            ch.wr_req_ready = !stalled && ch.rng.chance(thr_wr_req_);
            ch.wr_dat_ready = !stalled && ch.rng.chance(thr_wr_dat_);
        }
        // read response: hold the presented beat until accepted
        if (!ch.rsp_valid && !stalled && !ch.rd.empty() && ch.rng.chance(thr_rd_rsp_)) {
            const long k = pick(ch, ch.rd, !reg_.interleave);
            if (k >= 0) {
                const RdBurst& b = ch.rd[(size_t)k];
                const uint64_t beat = b.addr + b.sent;
                ch.rsp_valid = true;
                ch.rsp_seq = b.seq;
                if (beat < cfg_.limit) memcpy(in.rd_rsp_data, ch.mem + beat * 64, 64);
                else memset(in.rd_rsp_data, 0, 64);
                in.rd_rsp_new = true;
                in.rd_rsp_id = b.id;
                in.rd_rsp_last = b.sent == b.len - 1;
            }
        }
        // write response
        if (!ch.wrsp_valid && !stalled && !ch.wrsp.empty() && ch.rng.chance(thr_wr_rsp_)) {
            const long k = pick(ch, ch.wrsp, false);
            if (k >= 0) {
                ch.wrsp_valid = true;
                ch.wrsp_seq = ch.wrsp[(size_t)k].seq;
                in.wr_rsp_id = ch.wrsp[(size_t)k].id;
            }
        }
        if (ch.rsp_poison && !ch.rsp_valid) {
            // No beat presented: drive a poison pattern instead of the stale beat, so that a core
            // that takes data without rd_rsp_valid gets wrong data (the Python model drives 0).
            memset(in.rd_rsp_data, 0xE5, 64);
            in.rd_rsp_new = true;
        }
        ch.rsp_poison = false;
        in.rd_req_ready = ch.rd_req_ready;
        in.rd_rsp_valid = ch.rsp_valid;
        if (!ch.rsp_valid) in.rd_rsp_last = false;
        in.wr_req_ready = ch.wr_req_ready;
        in.wr_dat_ready = ch.wr_dat_ready;
        in.wr_rsp_valid = ch.wrsp_valid;
    }
};

}  // namespace fr

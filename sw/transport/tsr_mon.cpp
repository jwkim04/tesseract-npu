// tsr_mon.cpp — the publisher of the live monitor record (tsr_mon.h, docs/prof-dbg.md §7, SW-87).
#include "tsr_mon.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

uint64_t clock_ns(clockid_t id) {
    timespec ts;
    clock_gettime(id, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

// Copies at most n - 1 bytes and terminates.
void put_str(char * dst, size_t n, const char * src) {
    if (n == 0) return;
    size_t i = 0;
    if (src != nullptr) {
        for (; i + 1 < n && src[i] != '\0'; ++i) dst[i] = src[i];
    }
    memset(dst + i, 0, n - i);
}

std::string read_file(const char * path, size_t max = 1u << 20) {
    std::string s;
    FILE * f = fopen(path, "rb");
    if (f == nullptr) return s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0 && s.size() < max) s.append(buf, n);
    fclose(f);
    return s;
}

// Field 22 of /proc/self/stat (starttime in clock ticks); the comm field may hold blanks and ')'.
uint64_t proc_start_ticks() {
    const std::string s = read_file("/proc/self/stat", 8192);
    const size_t rp = s.rfind(')');
    if (rp == std::string::npos) return 0;
    // after ") " field 3 is the state; starttime is field 22, the 20th token after the comm
    const char * p = s.c_str() + rp + 1;
    for (int field = 3; field < 22; ++field) {
        while (*p == ' ') ++p;
        while (*p != ' ' && *p != '\0') ++p;
    }
    while (*p == ' ') ++p;
    return strtoull(p, nullptr, 10);
}

const char * base_name(const char * p) {
    const char * b = strrchr(p, '/');
    return b != nullptr ? b + 1 : p;
}

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// The model's file name: TSR_MON_MODEL, else the argument of -m / --model / -hf / --hf-repo, else
// the first mapped file that ends in .gguf (llama.cpp maps the model).
std::string find_model() {
    if (const char * e = getenv("TSR_MON_MODEL"); e != nullptr && e[0] != '\0') return e;
    const std::string cl = read_file("/proc/self/cmdline", 1u << 16);
    std::vector<std::string> args;
    for (size_t i = 0; i < cl.size();) {
        const size_t z = cl.find('\0', i);
        args.push_back(cl.substr(i, z == std::string::npos ? std::string::npos : z - i));
        if (z == std::string::npos) break;
        i = z + 1;
    }
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string & a = args[i];
        if ((a == "-m" || a == "--model" || a == "-hf" || a == "--hf-repo" || a == "--model-url") && i + 1 < args.size()) {
            return base_name(args[i + 1].c_str());
        }
        if (a.compare(0, 8, "--model=") == 0) return base_name(a.c_str() + 8);
    }
    const std::string maps = read_file("/proc/self/maps", 1u << 22);
    size_t pos = 0;
    while (pos < maps.size()) {
        size_t e = maps.find('\n', pos);
        if (e == std::string::npos) e = maps.size();
        std::string line = maps.substr(pos, e - pos);
        pos = e + 1;
        const size_t slash = line.find('/');
        if (slash == std::string::npos) continue;
        std::string path = line.substr(slash);
        if (ends_with(path, " (deleted)")) path.resize(path.size() - 10);
        if (ends_with(path, ".gguf")) return base_name(path.c_str());
    }
    return std::string();
}

struct mon_state {
    std::mutex   mu;
    std::atomic<bool> attached { false };   // read without the lock by the launch hooks (one flag test when off)
    bool         failed = false;
    tsr_transport * tp = nullptr;
    int          fd = -1;
    uint8_t *    map = nullptr;
    char         path[320] = {};
    uint32_t     pid = 0;
    tsr_mon_rec  cur;
    uint64_t     perf_interval_ns = 100ull * 1000000ull;
    uint64_t     last_perf_mono = 0;
    uint64_t     last_publish_mono = 0;
    uint64_t     prev_perf[TSR_MON_PERF_SLOTS] = {};
    uint64_t     prev_perf_launches = 0;
    uint32_t     model_tries = 0;
    bool         exit_hook = false;
};

mon_state & st() {
    static mon_state * s = new mon_state();   // intentionally leaked: the atexit hook runs after static destructors
    return *s;
}

// The seqlock store: seq odd, the page, seq even. The page is stored as relaxed 64-bit words so the
// race with a reader is a defined one; the fences order the words against the two seq stores.
void publish_locked(mon_state & s) {
    if (s.map == nullptr) return;
    tsr_mon_rec & c = s.cur;
    c.update_real_ns = clock_ns(CLOCK_REALTIME);
    c.update_mono_ns = clock_ns(CLOCK_MONOTONIC);
    s.last_publish_mono = c.update_mono_ns;
    uint64_t * words = reinterpret_cast<uint64_t *>(s.map);
    const uint64_t * src = reinterpret_cast<const uint64_t *>(&c);
    const size_t nw = (sizeof(tsr_mon_rec) + 7) / 8;
    const uint64_t seq = __atomic_load_n(&words[2], __ATOMIC_RELAXED);
    __atomic_store_n(&words[2], seq + 1, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    for (size_t i = 0; i < nw; ++i) {
        if (i == 2) continue;                       // seq
        __atomic_store_n(&words[i], src[i], __ATOMIC_RELAXED);
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&words[2], seq + 2, __ATOMIC_RELAXED);
}

// TSR_MON_KEEP=1 leaves the record behind at exit, marked TSR_MON_F_EXITING (a post-mortem look at the
// last counters; the tests read it after the run). tsr-top ignores it as the record of a dead pid.
bool keep_file() {
    const char * e = getenv("TSR_MON_KEEP");
    return e != nullptr && e[0] != '\0' && atoi(e) != 0;
}

void remove_file_locked(mon_state & s) {
    if (s.pid != 0 && (uint32_t) getpid() == s.pid) {
        if (s.attached && s.map != nullptr) {
            s.cur.flags |= TSR_MON_F_EXITING;
            publish_locked(s);
        }
        if (s.path[0] != '\0' && !keep_file()) unlink(s.path);
    }
    if (s.map != nullptr) {
        munmap(s.map, TSR_MON_BYTES);
        s.map = nullptr;
    }
    if (s.fd >= 0) {
        close(s.fd);
        s.fd = -1;
    }
    s.attached = false;
    s.path[0] = '\0';
}

void at_exit_hook() {
    mon_state & s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.attached) remove_file_locked(s);
}

// The PERF counters, the clocks. Called with the lock held, never inside a launch.
void read_perf_locked(mon_state & s) {
    tsr_mon_rec & c = s.cur;
    tsr_perf_summary ps;
    const uint64_t t_in = clock_ns(CLOCK_MONOTONIC);
    if (tsr_transport_perf_summary(s.tp, &ps) != 0) return;
    const uint64_t now = clock_ns(CLOCK_MONOTONIC);
    const bool first = c.perf_reads == 0;
    for (uint32_t i = 0; i < TSR_MON_PERF_SLOTS; ++i) {
        c.dperf[i] = first ? 0 : ps.perf[i] - s.prev_perf[i];
        c.perf[i] = ps.perf[i];
        s.prev_perf[i] = ps.perf[i];
    }
    c.d_launches = first ? 0 : c.launches - s.prev_perf_launches;
    c.d_wall_ns = first ? 0 : now - c.perf_mono_ns;
    s.prev_perf_launches = c.launches;
    c.perf_launches = c.launches;
    c.perf_mono_ns = now;
    c.perf_reads++;
    c.core_hz = (uint64_t) (ps.core_hz > 0 ? ps.core_hz : 0);
    if (ps.core_hz_fitted) c.flags |= TSR_MON_F_FITTED; else c.flags &= ~TSR_MON_F_FITTED;
    const uint32_t rv = tsr_transport_csr_read(s.tp, TSR_CSR_RV_FREQ);
    if (rv != 0xFFFFFFFFu) {
        c.rv_freq_raw = rv;
        c.rv_hz = (uint64_t) ((double) c.core_hz * (double) rv / 65536.0);
    }
    s.last_perf_mono = now;
    c.perf_ns += clock_ns(CLOCK_MONOTONIC) - t_in;
}

} // namespace

extern "C" int tsr_mon_enabled(void) {
    static const bool on = [] {
        const char * e = getenv("TSR_MON");
        return e == nullptr || e[0] == '\0' || atoi(e) != 0;
    }();
    return on ? 1 : 0;
}

extern "C" void tsr_mon_path(char * path, size_t n, uint32_t pid) {
    const char * dir = getenv("TSR_MON_DIR");
    if (dir == nullptr || dir[0] == '\0') dir = "/dev/shm";
    snprintf(path, n, "%s/tsr-mon-%u", dir, pid);
}

namespace {

// A PCIe address such as 0000:3b:00.1 and nothing else (it becomes part of a file name).
bool cap_bdf_ok(const char * bdf) {
    if (bdf == nullptr || bdf[0] == '\0' || strlen(bdf) > 32) return false;
    for (const char * p = bdf; *p != '\0'; ++p) {
        if (!((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f') || (*p >= 'A' && *p <= 'F') || *p == ':' || *p == '.')) return false;
    }
    return true;
}

std::string cap_text(const char * bdf, uint32_t num_ch, uint64_t ch_bytes) {
    char b[160];
    snprintf(b, sizeof(b), "tsr-cap 1\nbdf %s\nnum_ch %u\nch_bytes %" PRIu64 "\n", bdf, num_ch, ch_bytes);
    return b;
}

} // namespace

extern "C" void tsr_mon_cap_path(char * path, size_t n, const char * bdf) {
    const char * dir = getenv("TSR_MON_DIR");
    if (dir == nullptr || dir[0] == '\0') dir = "/dev/shm";
    snprintf(path, n, "%s/tsr-cap-%s", dir, bdf != nullptr ? bdf : "");
}

extern "C" int tsr_mon_cap_write(const char * bdf, uint32_t num_ch, uint64_t ch_bytes) {
    if (!cap_bdf_ok(bdf) || num_ch == 0 || ch_bytes == 0) return -1;
    char path[512], tmp[560];
    tsr_mon_cap_path(path, sizeof(path), bdf);
    const std::string want = cap_text(bdf, num_ch, ch_bytes);
    if (read_file(path, 4096) == want) return 0;
    snprintf(tmp, sizeof(tmp), "%s.%u.tmp", path, (unsigned) getpid());
    const int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) return -1;
    bool ok = write(fd, want.data(), want.size()) == (ssize_t) want.size();
    ok = fchmod(fd, 0644) == 0 && ok;                     // the umask must not hide the file from other users
    ok = close(fd) == 0 && ok;
    if (!ok || rename(tmp, path) != 0) {                  // a file of another user in a sticky directory stays
        unlink(tmp);
        return -1;
    }
    return 0;
}

extern "C" int tsr_mon_cap_read(const char * bdf, uint32_t * num_ch, uint64_t * ch_bytes) {
    if (!cap_bdf_ok(bdf)) return -1;
    char path[512];
    tsr_mon_cap_path(path, sizeof(path), bdf);
    const std::string s = read_file(path, 4096);
    unsigned n = 0;
    unsigned long long b = 0;
    char who[40] = {};
    if (s.compare(0, 10, "tsr-cap 1\n") != 0) return -1;
    const char * p = s.c_str() + 10;
    if (sscanf(p, "bdf %39s\nnum_ch %u\nch_bytes %llu\n", who, &n, &b) != 3) return -1;
    if (strcmp(who, bdf) != 0 || n == 0 || b == 0) return -1;
    *num_ch = n;
    *ch_bytes = b;
    return 0;
}

extern "C" void tsr_mon_attach(tsr_transport * t, const char * owner) {
    if (t == nullptr || !tsr_mon_enabled()) return;
    mon_state & s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.attached || s.failed) return;
    s.failed = true;                                   // until the file exists
    memset(&s.cur, 0, sizeof(s.cur));
    tsr_mon_rec & c = s.cur;
    memcpy(c.magic, TSR_MON_MAGIC, 8);
    c.layout = TSR_MON_LAYOUT;
    c.size = TSR_MON_BYTES;
    c.used = (uint32_t) sizeof(tsr_mon_rec);
    s.pid = (uint32_t) getpid();
    c.pid = s.pid;
    c.uid = (uint32_t) getuid();
    c.proc_start_ticks = proc_start_ticks();
    c.start_real_ns = clock_ns(CLOCK_REALTIME);
    put_str(c.transport, sizeof(c.transport), tsr_transport_name(t));
    if (strcmp(c.transport, "sim") == 0 || strcmp(c.transport, "mock") == 0 || strcmp(c.transport, "vkernel") == 0) {
        c.flags |= TSR_MON_F_SIM;
    }
    put_str(c.owner, sizeof(c.owner), owner != nullptr ? owner : "");
    put_str(c.mode, sizeof(c.mode), "?");
    const tsr_config * cfg = tsr_transport_config(t);
    if (cfg != nullptr) {
        c.num_ch = cfg->num_ch;
        c.t_max = cfg->t_max;
        c.lanes = cfg->lanes;
        c.spm_bytes = cfg->spm_bytes;
        c.k_max = cfg->k_max;
        c.caps = cfg->caps;
        c.ch_bytes = cfg->ch_bytes;
        c.mem_total = (uint64_t) cfg->num_ch * cfg->ch_bytes;
        c.perf_x = TSR_PERF_X(cfg->num_ch);
    }
    tsr_versions v;
    if (tsr_transport_versions(t, &v) == 0) {
        c.core_ver = v.core;
        c.core_min = v.core_min;
        c.basis = v.basis;
        c.abi = v.abi;
        c.n_kinds = v.n_kinds;
        c.kinds_sig = v.kinds_sig;
        c.image_id = v.image_id;
        memcpy(c.kind_rev, v.kind_rev, std::min<size_t>(sizeof(c.kind_rev), sizeof(v.kind_rev)));
    }
    if (const tsr_image * img = tsr_transport_image(t); img != nullptr) {
        put_str(c.image_path, sizeof(c.image_path), tsr_image_path(img));
    }
    put_str(c.xclbin, sizeof(c.xclbin), getenv("TSR_XCLBIN"));
    char info[256] = {};
    tsr_transport_describe(t, info, sizeof(info));
    if (char * nl = strchr(info, '\n')) *nl = '\0';
    put_str(c.device_info, sizeof(c.device_info), info);
    std::string cl = read_file("/proc/self/cmdline", 1u << 16);
    for (char & ch : cl) {
        if (ch == '\0') ch = ' ';
    }
    while (!cl.empty() && cl.back() == ' ') cl.pop_back();
    put_str(c.cmdline, sizeof(c.cmdline), cl.c_str());
    std::string comm = read_file("/proc/self/comm", 64);
    while (!comm.empty() && (comm.back() == '\n' || comm.back() == ' ')) comm.pop_back();
    put_str(c.comm, sizeof(c.comm), comm.c_str());

    if (const char * e = getenv("TSR_MON_PERF_MS"); e != nullptr && e[0] != '\0') {
        char * end = nullptr;
        const long long ms = strtoll(e, &end, 10);
        if (end != e && ms >= 0) s.perf_interval_ns = (uint64_t) ms * 1000000ull;
    }

    // The file appears whole or not at all: write a hidden temporary file, then rename it.
    tsr_mon_path(s.path, sizeof(s.path), s.pid);
    char tmp[sizeof(s.path) + 16];
    snprintf(tmp, sizeof(tmp), "%s.tmp", s.path);
    const int fd = open(tmp, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0 || ftruncate(fd, TSR_MON_BYTES) != 0) {
        if (fd >= 0) { close(fd); unlink(tmp); }
        s.path[0] = '\0';
        return;
    }
    void * m = mmap(nullptr, TSR_MON_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        close(fd);
        unlink(tmp);
        s.path[0] = '\0';
        return;
    }
    s.fd = fd;
    s.map = (uint8_t *) m;
    s.tp = t;
    publish_locked(s);                                 // a complete page before the file has its name
    if (rename(tmp, s.path) != 0) {
        munmap(m, TSR_MON_BYTES);
        close(fd);
        unlink(tmp);
        s.map = nullptr;
        s.fd = -1;
        s.path[0] = '\0';
        return;
    }
    s.failed = false;
    s.attached = true;
    if (!s.exit_hook) {
        s.exit_hook = true;
        atexit(at_exit_hook);
    }
    read_perf_locked(s);
    publish_locked(s);
}

extern "C" void tsr_mon_detach(tsr_transport * t) {
    mon_state & s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.attached || s.tp != t) return;
    remove_file_locked(s);
    s.tp = nullptr;
}

extern "C" void tsr_mon_launch_begin(void) {
    mon_state & s = st();
    if (!s.attached) return;                           // one atomic flag read: attach happens before any launch
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.attached) return;
    const uint64_t t_in = clock_ns(CLOCK_MONOTONIC);
    s.cur.launch_begin_mono_ns = t_in;
    publish_locked(s);
    s.cur.hook_ns += clock_ns(CLOCK_MONOTONIC) - t_in;
}

extern "C" void tsr_mon_launch_end(uint64_t wall_ns, uint32_t n_ops, int rc) {
    mon_state & s = st();
    if (!s.attached) return;
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.attached) return;
    tsr_mon_rec & c = s.cur;
    const uint64_t t_in = clock_ns(CLOCK_MONOTONIC);
    c.launch_begin_mono_ns = 0;
    c.launches++;
    c.launch_ns += wall_ns;
    c.ops += n_ops;
    c.last_launch_ns = wall_ns;
    if (c.model[0] == '\0' && s.model_tries < 8 && (c.launches == 1 || (c.launches & 15) == 0)) {
        s.model_tries++;
        const std::string m = find_model();
        if (!m.empty()) put_str(c.model, sizeof(c.model), m.c_str());
    }
    const uint64_t now = clock_ns(CLOCK_MONOTONIC);
    if (rc == 0 && now - s.last_perf_mono >= s.perf_interval_ns) read_perf_locked(s);
    publish_locked(s);
    c.hook_ns += clock_ns(CLOCK_MONOTONIC) - t_in;
}

extern "C" void tsr_mon_set_model(const char * model, const char * quant) {
    mon_state & s = st();
    if (!s.attached) return;
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.attached) return;
    if (model != nullptr && model[0] != '\0') put_str(s.cur.model, sizeof(s.cur.model), model);
    if (quant != nullptr && quant[0] != '\0') put_str(s.cur.quant, sizeof(s.cur.quant), quant);
}

extern "C" void tsr_mon_set_mode(const char * mode, uint64_t kind_mask, const char * kind_names) {
    mon_state & s = st();
    if (!s.attached) return;
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.attached) return;
    if (mode != nullptr && mode[0] != '\0') put_str(s.cur.mode, sizeof(s.cur.mode), mode);
    s.cur.kind_mask = kind_mask;
    if (kind_names != nullptr) put_str(s.cur.kinds, sizeof(s.cur.kinds), kind_names);
    publish_locked(s);
}

extern "C" void tsr_mon_set_mem(uint64_t used, uint64_t total) {
    mon_state & s = st();
    if (!s.attached) return;
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.attached) return;
    s.cur.mem_used = used;
    if (total != 0) s.cur.mem_total = total;
    // memory moves while the weights are placed, before the first launch: let the monitor see it (at most every 100 ms)
    if (clock_ns(CLOCK_MONOTONIC) - s.last_publish_mono >= 100000000ull) publish_locked(s);
}

extern "C" void tsr_mon_add_tokens(uint32_t decode, uint32_t prefill, uint32_t passes) {
    mon_state & s = st();
    if (!s.attached) return;
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.attached) return;
    s.cur.tok_decode += decode;
    s.cur.tok_prefill += prefill;
    s.cur.passes += passes;
}

// ---------------------------------------------------------------------------------------------
// Reader helpers
// ---------------------------------------------------------------------------------------------
extern "C" int tsr_mon_read(const char * path, tsr_mon_rec * out) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    int rc = -3;
    for (int tries = 0; tries < 100000; ++tries) {
        uint8_t page[TSR_MON_BYTES];
        if (pread(fd, page, sizeof(page), 0) != (ssize_t) sizeof(page)) { rc = -1; break; }
        uint64_t seq2 = 0;
        if (pread(fd, &seq2, sizeof(seq2), 16) != (ssize_t) sizeof(seq2)) { rc = -1; break; }
        uint64_t seq1;
        memcpy(&seq1, page + 16, 8);
        if ((seq1 & 1u) != 0 || seq1 != seq2) continue;
        if (memcmp(page, TSR_MON_MAGIC, 8) != 0) { rc = -1; break; }
        uint32_t layout;
        memcpy(&layout, page + 8, 4);
        if (layout != TSR_MON_LAYOUT) { rc = -2; break; }
        memcpy(out, page, sizeof(*out));
        rc = 0;
        break;
    }
    close(fd);
    return rc;
}

#define TSR_MON_FIELDS(X)                                                                                   \
    X(magic, char, 8) X(layout, uint32_t, 1) X(size, uint32_t, 1) X(seq, uint64_t, 1) X(flags, uint32_t, 1) \
    X(used, uint32_t, 1) X(pid, uint32_t, 1) X(uid, uint32_t, 1) X(proc_start_ticks, uint64_t, 1)           \
    X(start_real_ns, uint64_t, 1) X(update_real_ns, uint64_t, 1) X(update_mono_ns, uint64_t, 1)             \
    X(perf_mono_ns, uint64_t, 1) X(launches, uint64_t, 1) X(launch_ns, uint64_t, 1) X(ops, uint64_t, 1)     \
    X(last_launch_ns, uint64_t, 1) X(launch_begin_mono_ns, uint64_t, 1) X(perf_launches, uint64_t, 1)       \
    X(d_launches, uint64_t, 1) X(d_wall_ns, uint64_t, 1) X(tok_decode, uint64_t, 1)                         \
    X(tok_prefill, uint64_t, 1) X(passes, uint64_t, 1) X(perf_ns, uint64_t, 1) X(hook_ns, uint64_t, 1) X(mem_used, uint64_t, 1) X(mem_total, uint64_t, 1)   \
    X(core_hz, uint64_t, 1) X(rv_hz, uint64_t, 1) X(rv_freq_raw, uint32_t, 1) X(perf_reads, uint32_t, 1)    \
    X(num_ch, uint32_t, 1) X(t_max, uint32_t, 1) X(lanes, uint32_t, 1) X(spm_bytes, uint32_t, 1)            \
    X(k_max, uint32_t, 1) X(caps, uint32_t, 1) X(core_ver, uint32_t, 1) X(core_min, uint32_t, 1)            \
    X(basis, uint32_t, 1) X(abi, uint32_t, 1) X(n_kinds, uint32_t, 1) X(kinds_sig, uint32_t, 1)             \
    X(ch_bytes, uint64_t, 1) X(image_id, uint64_t, 1) X(kind_mask, uint64_t, 1) X(perf_x, uint32_t, 1)      \
    X(reserved1, uint32_t, 1) X(kind_rev, uint8_t, TSR_MON_KINDS) X(perf, uint64_t, TSR_MON_PERF_SLOTS)     \
    X(dperf, uint64_t, TSR_MON_PERF_SLOTS) X(transport, char, 16) X(owner, char, 16) X(mode, char, 16)      \
    X(quant, char, 24) X(model, char, 128) X(image_path, char, 192) X(xclbin, char, 192)                    \
    X(device_info, char, 192) X(kinds, char, 320) X(cmdline, char, 384) X(comm, char, 16)

extern "C" int tsr_mon_layout_json(char * buf, size_t n) {
    std::string o;
    char b[160];
    snprintf(b, sizeof(b), "{\"layout\":%u,\"size\":%u,\"used\":%zu,\"fields\":[", TSR_MON_LAYOUT, TSR_MON_BYTES, sizeof(tsr_mon_rec));
    o += b;
    bool first = true;
#define X(name, type, count)                                                                                \
    snprintf(b, sizeof(b), "%s{\"name\":\"%s\",\"off\":%zu,\"elem\":%zu,\"count\":%u}", first ? "" : ",", #name, \
             offsetof(tsr_mon_rec, name), sizeof(type), (unsigned) (count));                                 \
    static_assert(sizeof(((tsr_mon_rec *) nullptr)->name) == sizeof(type) * (count), "layout table: " #name);\
    o += b;                                                                                                 \
    first = false;
    TSR_MON_FIELDS(X)
#undef X
    o += "]}";
    if (buf != nullptr && n != 0) snprintf(buf, n, "%s", o.c_str());
    return (int) o.size();
}

// tsr-mon-probe — what tsr-top shows for a card that no process holds (docs/prof-dbg.md §7, SW-87).
//
// Opens the transport in attach mode (TSR_ATTACH=1: no SOFT_RESET, no bind, no boot, as `tsr-dbg
// --attach`), reads the identity, the clocks and the PERF counters, closes it and prints one JSON
// object. The card is held for about --settle-ms (default 300: the core clock is fitted from
// DBG_MTIME against host time over that span). It never launches anything and writes only what an
// attach open writes (ADAPTER_STATUS is cleared at open, EX §5).
//
// tsr-top runs it only when it was asked to (--probe), when no tsr-mon publisher is alive, when no
// process has the card's render node open and when the xclbin on the card is the one named by
// TSR_XCLBIN (a different one would be loaded by the open). The xrt transport opens the compute unit
// exclusively, so a process that starts while this runs fails to open the card for that span.
//
//   tsr-mon-probe [--device KIND] [--settle-ms N]        (xrt needs TSR_XCLBIN; TSR_XRT_DEVICE selects the card)
// Exit: 0 on success, 3 when the device cannot be opened (the JSON says why).
#include "tsr_transport.h"

#include <inttypes.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static uint64_t mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static uint64_t csr64(tsr_transport * t, uint32_t lo, uint32_t hi_off) {
    const uint32_t l = tsr_transport_csr_read(t, lo);   // the LO read snapshots a PERF counter for the HI read
    const uint32_t h = tsr_transport_csr_read(t, hi_off);
    return (uint64_t) h << 32 | l;
}

static uint64_t perf(tsr_transport * t, uint32_t i) { return csr64(t, TSR_CSR_PERF_LO(i), TSR_CSR_PERF_HI(i)); }

static uint64_t mtime(tsr_transport * t) {
    const uint32_t hi = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_HI);
    uint32_t lo = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_LO);
    const uint32_t hi2 = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_HI);
    if (hi2 != hi) lo = tsr_transport_csr_read(t, TSR_CSR_DBG_MTIME_LO);
    return (uint64_t) hi2 << 32 | lo;
}

static std::string jstr(const char * s) {
    std::string o = "\"";
    for (; *s != '\0'; ++s) {
        if (*s == '"' || *s == '\\') { o += '\\'; o += *s; }
        else if ((unsigned char) *s < 0x20) o += ' ';
        else o += *s;
    }
    return o + "\"";
}

int main(int argc, char ** argv) {
    const char * kind = "xrt";
    unsigned settle_ms = 300;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) kind = argv[++i];
        else if (strcmp(argv[i], "--settle-ms") == 0 && i + 1 < argc) settle_ms = (unsigned) atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: tsr-mon-probe [--device KIND] [--settle-ms N]\n");
            return 2;
        }
    }
    setenv("TSR_ATTACH", "1", 1);
    setenv("TSR_MON", "0", 1);
    const uint64_t w0 = mono_ns();
    tsr_transport * t = tsr_transport_open(kind);
    if (t == nullptr) {
        printf("{\"ok\":false,\"error\":\"the transport could not be opened (is the card held by another process?)\"}\n");
        return 3;
    }
    const uint64_t t_open = mono_ns();
    const tsr_config * c = tsr_transport_config(t);
    tsr_versions v;
    tsr_transport_versions(t, &v);
    const uint32_t perf_info = tsr_transport_csr_read(t, TSR_CSR_PERF_INFO);
    const uint32_t C = c->num_ch;
    const uint32_t X = TSR_PERF_X(C);
    const bool has_dbg = tsr_transport_csr_read(t, TSR_CSR_DBG_ID) != 0 && tsr_transport_csr_read(t, TSR_CSR_DBG_ID) != 0xFFFFFFFFu;
    uint64_t mt0 = 0, mt1 = 0, wm0 = 0, wm1 = 0;
    if (has_dbg) {
        wm0 = mono_ns();
        mt0 = mtime(t);
    }
    const uint64_t cycles = perf(t, TSR_PERF_CYCLES), mvm = perf(t, TSR_PERF_MVM_CYCLES), retired = perf(t, TSR_PERF_RETIRED);
    const uint64_t launches = X + TSR_PERF_XOFF_LAUNCHES < 64 ? perf(t, X + TSR_PERF_XOFF_LAUNCHES) : 0;
    uint64_t beats[TSR_NUM_CH_MAX] = {};
    for (uint32_t ch = 0; ch < C && ch < TSR_NUM_CH_MAX; ++ch) beats[ch] = perf(t, TSR_PERF_CH_RD_BEATS(ch));
    const uint32_t rt_state = tsr_transport_csr_read(t, TSR_CSR_RT_STATE);
    const uint32_t status = tsr_transport_csr_read(t, TSR_CSR_STATUS);
    const uint32_t rv_freq = tsr_transport_csr_read(t, TSR_CSR_RV_FREQ);
    if (has_dbg) {
        const uint64_t spent = (mono_ns() - wm0) / 1000000ull;
        if (spent < settle_ms) usleep((useconds_t) ((settle_ms - spent) * 1000));
        wm1 = mono_ns();
        mt1 = mtime(t);
    }
    char info[512] = {};
    tsr_transport_describe(t, info, sizeof(info));
    if (char * nl = strchr(info, '\n')) *nl = '\0';
    const std::string name = tsr_transport_name(t);     // the pointer is the transport's: copy it before the close
    const double core_hz = has_dbg && wm1 > wm0 && mt1 > mt0 ? (double) (mt1 - mt0) * 1e9 / (double) (wm1 - wm0) : 0.0;
    tsr_transport_close(t);                               // the card is released here
    const uint64_t t_close = mono_ns();

    printf("{\"ok\":true,\"transport\":%s,\"device_info\":%s,\"held_ms\":%.1f,\"open_ms\":%.1f,", jstr(name.c_str()).c_str(), jstr(info).c_str(),
           (double) (t_close - w0) / 1e6, (double) (t_open - w0) / 1e6);
    printf("\"num_ch\":%u,\"t_max\":%u,\"lanes\":%u,\"spm_bytes\":%u,\"caps\":%u,\"core_ver\":%u,\"ch_bytes\":%" PRIu64 ",",
           c->num_ch, c->t_max, c->lanes, c->spm_bytes, c->caps, v.core, c->ch_bytes);
    printf("\"core_hz\":%.0f,\"rv_freq_raw\":%u,\"rv_hz\":%.0f,\"perf_info\":%u,\"rt_state\":%u,\"status\":%u,", core_hz, rv_freq,
           core_hz > 0 ? core_hz * rv_freq / 65536.0 : 0.0, perf_info, rt_state, status);
    printf("\"perf_cycles\":%" PRIu64 ",\"perf_mvm\":%" PRIu64 ",\"perf_retired\":%" PRIu64 ",\"launches\":%" PRIu64 ",\"rd_beats\":[",
           cycles, mvm, retired, launches);
    for (uint32_t ch = 0; ch < C && ch < TSR_NUM_CH_MAX; ++ch) printf("%s%" PRIu64, ch ? "," : "", beats[ch]);
    printf("]}\n");
    return 0;
}

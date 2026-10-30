// tsr_devrun_cfg.cpp — tsr_tgt_cfg_of_transport (tsr_devrun.h): the device configuration a shadow
// simulator must have (SW §7.6). C++ only because the inner simulator of the sim / mock kinds is
// reached through the protocol layer's C++ interface (tsr::transport_sim, tsr_kdev.h).
#include "tsr_devrun.h"
#include "tsr_kdev.h"

#include <cstring>

extern "C" void tsr_tgt_cfg_of_transport(tsr_transport * t, tsr_tgt_cfg * c) {
    std::memset(c, 0, sizeof(*c));
    // sim / mock: the inner simulator's exact configuration (its CH_BYTES and ADDR_BEAT_W are the DRAM
    // range limit the core checks against); a hardware kind: the CSRs, with the kernel contract's
    // CH_BYTES (the core range-checks against it whatever buffer is bound, tsr_kdev.h).
    if (tsr_sim * s = tsr::transport_sim(t)) {
        c->core = *tsr_sim_config(s);
    } else {
        c->core = *tsr_transport_config(t);
        c->core.ch_bytes = tsr::KERNEL_CH_BYTES;
        c->core.addr_beat_w = 28;
    }
    const uint32_t qd = tsr_transport_csr_read(t, TSR_CSR_Q_DEPTH);
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) c->q_depth[u] = (uint8_t) ((qd >> (5 * u)) & 0x1Fu);
    const uint32_t rv = tsr_transport_csr_read(t, TSR_CSR_RV_CFG);
    c->itcm_bytes = (rv & 0xFFFFu) * 1024u;
    c->dtcm_bytes = (rv >> 16) * 1024u;
    const uint32_t f = tsr_transport_csr_read(t, TSR_CSR_RV_FREQ);
    c->rv_ratio = f != 0 && f != 0xFFFFFFFFu ? (double) f / 65536.0 : 0.0;
}

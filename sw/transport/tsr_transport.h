// tsr_transport.h — host <-> device transport (spec "ggml 백엔드 통합", 전송 계층).
//
// The only target-dependent part of the software stack. Implementations:
//   "sim"  — in-process ISA simulator (sw/transport/tsr_transport_sim.c)
//   "xrt"  — Alveo via XRT: the protocol layer (tsr_kdev.cpp) over the XRT kdev
//            (tsr_transport_xrt.cpp, built only with -DTSR_WITH_XRT=ON)
//   "mock" — the same protocol layer over a model of the platform adapter around the ISA
//            simulator (tsr_kdev_mock.cpp): tests the hardware path without hardware
//   "vkernel" — the same protocol layer over the Verilated kernel (RTL) in a server process
//            (tsr_kdev_vkernel.cpp, tb/platform/vkernel; TSR_VKERNEL_*)
//   other kdev kinds registered at run time (tsr::register_kdev, tsr_kdev.h)
// Selected with TSR_DEVICE (default sim). Region allocation policy lives in the backend; the
// transport opens the device, reads its configuration and moves bytes. The v1 run interface
// (one host-started command buffer per exec) is deleted (X0, D14): the host only launches
// RISC-V kernels, through the v2 API of docs/v2/01-execution-model.md §10 (image, boot,
// launch, wait, cancel, stop, soft reset, log), which arrives with P3.1. docs/xrt-host.md has
// the environment of the xrt / mock transports (TSR_XCLBIN, TSR_XRT_*, TSR_MOCK_*).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_transport tsr_transport;

// Open by kind ("sim", "xrt", "mock", "vkernel", registered kinds); NULL means getenv("TSR_DEVICE") or "sim".
// Returns NULL on failure (message on stderr).
// sim honours: TSR_SIM_DRAM_DIR (file-backed DRAM), TSR_SIM_TMAX (override T_MAX),
//              TSR_SIM_CH_GIB (per-channel size, default 16),
//              TSR_SIM_THREADS (MVM worker threads, default min(online CPUs, 8)),
//              TSR_SIM_LOG ("stderr" or a file: per-op log of the simulator).
//              A malformed value makes the open fail (see tsr_transport_sim.c).
tsr_transport * tsr_transport_open(const char * kind);
void            tsr_transport_close(tsr_transport * t);
const char *    tsr_transport_name(const tsr_transport * t);

// Configuration as read from the CFG*/CAPS CSRs (+ ch_bytes, which the CSRs don't carry).
const tsr_config * tsr_transport_config(const tsr_transport * t);

// Host access to channel DRAM. Offsets/sizes in bytes, any alignment. 0 on success.
int tsr_transport_write(tsr_transport * t, uint32_t ch, uint64_t off, const void * src, size_t size);
int tsr_transport_read (tsr_transport * t, uint32_t ch, uint64_t off, void * dst, size_t size);

// Raw CSR access (perf counters, debug).
uint32_t tsr_transport_csr_read(tsr_transport * t, uint32_t off);

#ifdef __cplusplus
}
#endif

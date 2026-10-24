// tsr_transport.h — host <-> device transport (spec "ggml 백엔드 통합", 전송 계층).
//
// The only target-dependent part of the software stack. Implementations:
//   "sim"  — in-process ISA simulator (sw/transport/tsr_transport_sim.c)
//   "xrt"  — Alveo via XRT: the kernel-protocol layer (tsr_kdev.cpp) over the XRT kdev
//            (tsr_transport_xrt.cpp, built only with -DTSR_WITH_XRT=ON)
//   "mock" — the same protocol layer over a model of the platform adapter around the ISA
//            simulator (tsr_kdev_mock.cpp): tests the hardware path without hardware
//   "vkernel" — the same protocol layer over the Verilated kernel (RTL) in a server process
//            (tsr_kdev_vkernel.cpp, tb/platform/vkernel; TSR_VKERNEL_*)
//   other kdev kinds registered at run time (tsr::register_kdev, tsr_kdev.h)
// Selected with TSR_DEVICE (default sim). Region allocation policy lives in the backend; the
// transport only moves bytes and runs programs. docs/xrt-host.md has the kernel contract and
// the environment of the xrt / mock transports (TSR_XCLBIN, TSR_XRT_*, TSR_MOCK_*).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_transport tsr_transport;

typedef struct tsr_exec_result {
    uint32_t status;      // final STATUS (DONE or ERR set)
    uint32_t err_code;    // tsr_err
    uint32_t err_pc;
    uint32_t err_info;
} tsr_exec_result;

// Open by kind ("sim", "xrt", "mock", "vkernel", registered kinds); NULL means getenv("TSR_DEVICE") or "sim".
// Returns NULL on failure (message on stderr).
// sim honours: TSR_SIM_DRAM_DIR (file-backed DRAM), TSR_SIM_TMAX (override T_MAX),
//              TSR_SIM_CH_GIB (per-channel size, default 16), TSR_TRACE (dump each exec as a
//              trace directory <TSR_TRACE>/<seq 6 digits>/, first free seq, never overwrites),
//              TSR_SIM_THREADS (MVM worker threads, default min(online CPUs, 8)),
//              TSR_SIM_LOG ("stderr" or a file: per-instruction log).
//              A malformed value makes the open fail (see tsr_transport_sim.c).
// Status convention: an exec that faults ends with STATUS = DONE | ERR (spec 구현 명확화).
tsr_transport * tsr_transport_open(const char * kind);
void            tsr_transport_close(tsr_transport * t);
const char *    tsr_transport_name(const tsr_transport * t);

// Configuration as read from the CFG*/CAPS CSRs (+ ch_bytes, which the CSRs don't carry).
const tsr_config * tsr_transport_config(const tsr_transport * t);

// Host access to channel DRAM. Offsets/sizes in bytes, any alignment. 0 on success.
int tsr_transport_write(tsr_transport * t, uint32_t ch, uint64_t off, const void * src, size_t size);
int tsr_transport_read (tsr_transport * t, uint32_t ch, uint64_t off, void * dst, size_t size);

// Run a program already written to ch0: CMD_ADDR = cmd_addr_beat, CMD_COUNT, ARG0..7
// (args may be NULL). Blocks until DONE or ERR. Returns 0 if DONE without ERR. Kernel-protocol
// transports also return non-zero when the run timed out (the core is then stopped with
// CTRL.ABORT: ERR_CODE = ABORTED) or the platform adapter reported an error (ADAPTER_STATUS).
int tsr_transport_exec(tsr_transport * t, uint64_t cmd_addr_beat, uint32_t cmd_count,
                       const uint32_t args[8], tsr_exec_result * res);

// Raw CSR access (perf counters, debug).
uint32_t tsr_transport_csr_read(tsr_transport * t, uint32_t off);

#ifdef __cplusplus
}
#endif

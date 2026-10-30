// tsr_vkernel_proto.h — wire protocol between the "vkernel" kdev (tsr_kdev_vkernel.cpp, in the
// host process) and the Verilated kernel server (tb/platform/vkernel; the v1 server is deleted in
// X0, the two-clock v2 server is P5.3). docs/platform-u200.md §13.
//
// One stream connection (a socketpair to a server the client spawned, or a Unix socket), one
// request at a time, native byte order (both ends run on the same host). Every request is a
// vk_req, followed by `c` payload bytes for VK_MEM_WR; every response is a vk_rsp followed by
// `len` payload bytes. `viol` in every response is the server's running count of verification
// findings (AXI4 / AXI4-Lite protocol checks, memory-model rules, RTL assertions, quiescence at
// completion, cycle budget): the client fetches the messages with VK_MSGS and fails the
// operation as soon as the count grows, so a protocol bug cannot pass silently.
//
// Version 2 (X0, docs/v2/01-execution-model.md §12): VK_WAIT (6), the scheduler's side of an
// ap_ctrl_hs run, is deleted with the run protocol; a version-1 server refuses the HELLO.
// Version 3 (P3.1, EX §10.2): the kernel is ap_ctrl_none and its clocks run only when the client
// lets device time pass: VK_ADVANCE (10), the "advance" hint of the kdev layer (kdev::idle), and
// VK_TIME (11), the device time the protocol's timeouts count. The two-clock v2 server that
// implements them is P5.3 (docs/v2/06-software-verification.md §11); until then no server exists
// and TSR_DEVICE=vkernel stops at the open with a message naming P5.3.
#pragma once

#include <stdint.h>

#define TSR_VK_PROTO_VERSION 3u

enum tsr_vk_op {
    VK_HELLO  = 1,   // a = protocol version -> val = NUM_CH, x = channel bytes; payload =
                     //   u64 base[NUM_CH] (device address of each channel buffer), then a
                     //   NUL-terminated description (build configuration, regime, seed)
    VK_REG_RD = 2,   // a = s_axi_control byte offset -> val (one AXI4-Lite read through the RTL slave)
    VK_REG_WR = 3,   // a = offset, b = value (one AXI4-Lite write, WSTRB 0xF)
    VK_MEM_WR = 4,   // a = channel, b = byte offset, c = bytes (payload follows; host DMA, no cycles)
    VK_MEM_RD = 5,   // a = channel, b = byte offset, c = bytes -> payload
                     // 6: deleted (v1 VK_WAIT), never reused
    VK_STATS  = 7,   // -> payload: JSON summary of the session so far (NUL-terminated)
    VK_MSGS   = 8,   // -> payload: findings not fetched yet, one per line (NUL-terminated)
    VK_BYE    = 9,   // the server prints its summary and exits
    VK_ADVANCE = 10, // b = core cycles (kernel clock): run both clocks for at most that long, ending
                     //   early when LCQ_WR, LOG_WR or STATUS.SR_BUSY / RV_HALTED change -> val = 1 if
                     //   it ended early, x = device time in core cycles after the step
    VK_TIME   = 11,  // -> x = device time in core cycles (kernel clock) since the server started
};

typedef struct tsr_vk_req {
    uint32_t op;
    uint32_t a;
    uint64_t b;
    uint64_t c;
} tsr_vk_req;

typedef struct tsr_vk_rsp {
    int32_t  rc;      // 0 ok, < 0 failed
    uint32_t val;
    uint64_t x;
    uint32_t viol;    // findings so far (cumulative)
    uint32_t len;     // payload bytes that follow
} tsr_vk_rsp;

// Largest VK_MEM_WR / VK_MEM_RD payload (the client splits larger transfers).
#define TSR_VK_MAX_XFER (64u << 20)

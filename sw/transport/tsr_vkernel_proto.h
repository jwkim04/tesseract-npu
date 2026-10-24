// tsr_vkernel_proto.h — wire protocol between the "vkernel" kdev (tsr_kdev_vkernel.cpp, in the
// host process) and the Verilated kernel server (tb/platform/vkernel/vkernel_server.cpp).
// docs/platform-u200.md §13.
//
// One stream connection (a socketpair to a server the client spawned, or a Unix socket), one
// request at a time, native byte order (both ends run on the same host). Every request is a
// vk_req, followed by `c` payload bytes for VK_MEM_WR; every response is a vk_rsp followed by
// `len` payload bytes. `viol` in every response is the server's running count of verification
// findings (AXI4 / AXI4-Lite protocol checks, memory-model rules, RTL assertions, quiescence at
// completion, cycle budget): the client fetches the messages with VK_MSGS and fails the
// operation as soon as the count grows, so a protocol bug cannot pass silently.
#pragma once

#include <stdint.h>

#define TSR_VK_PROTO_VERSION 1u

enum tsr_vk_op {
    VK_HELLO  = 1,   // a = protocol version -> val = NUM_CH, x = channel bytes; payload =
                     //   u64 base[NUM_CH] (device address of each channel buffer), then a
                     //   NUL-terminated description (build configuration, regime, seed)
    VK_REG_RD = 2,   // a = s_axi_control byte offset -> val (one AXI4-Lite read through the RTL slave)
    VK_REG_WR = 3,   // a = offset, b = value (one AXI4-Lite write, WSTRB 0xF)
    VK_MEM_WR = 4,   // a = channel, b = byte offset, c = bytes (payload follows; host DMA, no cycles)
    VK_MEM_RD = 5,   // a = channel, b = byte offset, c = bytes -> payload
    VK_WAIT   = 6,   // a = VK_WAIT_*, b = poll interval in cycles (VK_WAIT_POLL), c = timeout in ms
                     //   (0 = none) -> rc VK_DONE / VK_TIMEOUT / VK_ERROR
    VK_STATS  = 7,   // -> payload: JSON summary of the session so far (NUL-terminated)
    VK_MSGS   = 8,   // -> payload: findings not fetched yet, one per line (NUL-terminated)
    VK_BYE    = 9,   // the server prints its summary and exits
};

enum tsr_vk_wait {
    VK_WAIT_POLL = 1,   // the XRT scheduler's polling: an AP_CTRL read every b cycles through the
                        // AXI4-Lite master, complete on the first read that shows ap_done
    VK_WAIT_IRQ  = 2,   // interrupt mode: run until `interrupt` is high, then (as the scheduler's
                        // interrupt handler) ISR := toggle and an AP_CTRL read that must show ap_done
};

enum { VK_DONE = 0, VK_TIMEOUT = 1, VK_ERROR = -1 };

typedef struct tsr_vk_req {
    uint32_t op;
    uint32_t a;
    uint64_t b;
    uint64_t c;
} tsr_vk_req;

typedef struct tsr_vk_rsp {
    int32_t  rc;      // 0 ok (VK_WAIT: VK_DONE / VK_TIMEOUT), < 0 failed
    uint32_t val;
    uint64_t x;
    uint32_t viol;    // findings so far (cumulative)
    uint32_t len;     // payload bytes that follow
} tsr_vk_rsp;

// Largest VK_MEM_WR / VK_MEM_RD payload (the client splits larger transfers).
#define TSR_VK_MAX_XFER (64u << 20)

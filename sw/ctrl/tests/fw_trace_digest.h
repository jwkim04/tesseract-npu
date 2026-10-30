// fw_trace_digest.h — a 64-bit digest of a captured launch trace (sw/sim/include/tsr_trace.h): the
// push records in push order (the eight words, both deps as relative tickets, the ISSUE_PUSH value,
// the record kind: ops, QCTRL and TRACE_MARK entries), the DONE fields (status, RET, error code,
// relative error ticket and push index, ERR_INFO, ERR_W0) and the expect regions (every DRAM range
// the launch wrote, with its content at the answer). FNV-1a 64 over little-endian words.
//
// Used to pin the executor's push stream to W2a's (docs/design/06-software-verification.md "SW" §7.3,
// §7.4; executor optimisation review, 2026-10-30): fw_exec_rv's streamed lists and fw_m6_e2e's graph
// launches compare the digests of their captured traces with golden values taken from the W2a
// images (92f0b17). A deliberate change of what the executor or a kind pushes changes them; it is a
// decision (record it in SW §7.3 / §7.4 and update the values from the test's output).
#pragma once

#include <stdint.h>

#include "tsr_trace.h"

static inline uint64_t fw_digest_u32(uint64_t h, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        h ^= (v >> (8 * i)) & 0xFFu;
        h *= 1099511628211ull;
    }
    return h;
}

static inline uint64_t fw_trace_digest(const tsr_trace2 * t) {
    uint64_t h = 1469598103934665603ull;
    h = fw_digest_u32(h, (uint32_t)t->n_push);
    for (size_t i = 0; i < t->n_push; ++i) {
        const tsr_trace_push * p = &t->push[i];
        for (int k = 0; k < 8; ++k) h = fw_digest_u32(h, p->w[k]);
        h = fw_digest_u32(h, p->dep[0]);
        h = fw_digest_u32(h, p->dep[1]);
        h = fw_digest_u32(h, p->push);
        h = fw_digest_u32(h, p->kind);
    }
    const tsr_trace_done * d = &t->done;
    h = fw_digest_u32(h, d->status);
    h = fw_digest_u32(h, (uint32_t)d->ret);
    h = fw_digest_u32(h, d->err_code);
    h = fw_digest_u32(h, d->err_ticket_rel);
    h = fw_digest_u32(h, d->err_pidx_rel);
    h = fw_digest_u32(h, d->err_info);
    h = fw_digest_u32(h, d->err_w0);
    h = fw_digest_u32(h, (uint32_t)t->mem.n_expect);
    for (size_t i = 0; i < t->mem.n_expect; ++i) {
        const tsr_region * r = &t->mem.expect[i];
        h = fw_digest_u32(h, r->ch);
        h = fw_digest_u32(h, (uint32_t)r->byte_off);
        h = fw_digest_u32(h, (uint32_t)(r->byte_off >> 32));
        h = fw_digest_u32(h, (uint32_t)r->size);
        const uint8_t * b = (const uint8_t *)r->data;
        for (uint64_t k = 0; k < r->size; ++k) h = (h ^ b[k]) * 1099511628211ull;
    }
    return h;
}

// tsr/exec.hpp — the graph executor's kind-handler interface (docs/v2/06-software-verification.md
// "SW" §7.3, §7.4, §7.5; V2-SW-12).
//
// The `graph` kernel (sw/ctrl/kernels/graph.cpp) walks an OPL v1 list (tsr/opl.hpp) and, for each
// op record, calls the handler of the record's kind with the record's argument words, the deps of
// its predecessors and a context. A handler:
//   - allocates its SPM scratch from ctx.ring (spm_ring::alloc; its writes also wait for the deps
//     alloc returns), plans its tiles with tsr/plan.hpp (the host's supports_op uses the same
//     functions, V2-SW-20), and pushes its tile ops with issue(op, deps, ctx.push_flags | ...);
//     every op it pushes depends (directly or through others of the op) on `d`;
//   - returns ONE ticket T whose completion implies that every tile op of the op is done (the last
//     op depends on the others, or a join2 provides that; I1, docs/ctrl-riscv.md:544-550), and
//     releases its scratch blocks with release(block, T) before returning;
//   - never joins itself (no software wait inside the op stream: SW §7.3 "Overlap"; the ring's
//     own software joins, chosen by count, are the exception), never reads a timing-dependent
//     register (SW §4.7, V2-SW-30), keeps no state across launches (V2-SW-4);
//   - may read the preload area with the RISC-V (the executor joined its loads before op 0);
//   - on arguments it cannot run (bad fmt, a shape the ring cannot hold) pushes nothing more, sets
//     ctx.ret = opl::RET_BAD_RECORD or opl::RET_SCRATCH and returns ticket{} (ops it pushed before
//     stay joinable; the executor stops, joins all and returns that code, SW §7.3).
// Kinds without a handler in this image are BAD_RECORD.
//
// Kind numbers and argument words: tsr/opl.hpp (SW §7.2). NOP and JOIN are built into graph.cpp;
// MUL_MAT is kind_mulmat (sw/ctrl/kernels/k_mulmat.cpp); the M7 kinds come with W2b (P2.6, P2.7)
// and are added to the table in graph.cpp (kind_handler) as they land.
#pragma once

#include <stdint.h>

#include "tsr/kernel.hpp"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"

namespace tsr {

// Per-op context the executor hands to a kind handler (one object per launch, fields refreshed
// per op).
struct kind_ctx {
    spm_ring * ring;                   // the launch's scratch ring: SPM lines [0, layout.ring_lines)
    const volatile uint8_t * preload;  // the preload area through the SPM window (nullptr: none)
    uint32_t pre_lines;                // its size in lines (header pre_lines)
    spm pre;                           // its first SPM line (same area as `preload`)
    const plan::dev_cfg * cfg;         // device parameters (CFG0 / CFG1 / CFG2), read once per launch
    uint32_t push_flags;               // flags every issue() of the handler must OR in: ISSUE_PUSH
                                       // bit0 REC when the list has F_REC or the launch is PROF
    uint32_t op_index;                 // index of the op being executed (logs, panics)
    uint32_t ret;                      // 0; a failing handler sets opl::RET_SCRATCH / RET_BAD_RECORD
};

// A kind handler (SW §7.3 step 2.4). args = the record's n_args argument words (a DTCM copy),
// kflags = word0 [23:16], d = after(ticket[dep0], ticket[dep1]).
using kind_fn = ticket (*)(const uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);

// MUL_MAT (kind 0x10, SW §7.4): the kernels agent's handler (sw/ctrl/kernels/k_mulmat.cpp).
ticket kind_mulmat(const uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);

// The executor's kind table: the handler of `kind`, or nullptr (BAD_RECORD).
kind_fn kind_handler(uint32_t kind);

// Test images only (tsr_fw_image extras, sw/ctrl/tests/fw/exec_*.cpp; host test drivers): a
// handler for a kind, consulted BEFORE the built-in table. kinds 0x30-0x3F are reserved for test
// kinds (inside the M8 range 0x20-0x3F, never used by a production list). Returns true and fills
// *out to take the kind (nargs: the exact argument count a record of that kind must carry; for
// a kind of SW §7.2 it must equal opl::kind_nargs). The production image does not define it
// (weak reference: nullptr).
struct kind_entry {
    kind_fn  fn;
    uint32_t nargs;
};
}  // namespace tsr
extern "C" bool tsr_exec_test_kind(uint32_t kind, tsr::kind_entry * out) __attribute__((weak));

namespace tsr {
// Executor state for the debugger and the tests (DTCM, rebuilt at every launch, V2-SW-4): the
// done ticket and the first push index of each op (SW §7.3 "DTCM tables") and a summary of the
// last launch (opl::exec_state, defined in opl.hpp so host tools can read it from the DTCM).
// Symbols: __tsr_exec_ticket, __tsr_exec_pidx, __tsr_exec_state.
using exec_state = opl::exec_state;
}  // namespace tsr
extern "C" uint32_t __tsr_exec_ticket[tsr::opl::MAX_OPS];
extern "C" uint32_t __tsr_exec_pidx[tsr::opl::MAX_OPS];
extern "C" tsr::exec_state __tsr_exec_state;

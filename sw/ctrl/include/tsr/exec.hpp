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
//   - may read the preload area with the RISC-V (the executor joined its loads before op 0;
//     kind_ctx::pre_ok / pre_lb / pre_line below), or hand its lines to a unit as an operand;
//   - on arguments it cannot run (bad fmt, a shape the ring cannot hold) pushes nothing more, sets
//     ctx.ret = opl::RET_BAD_RECORD or opl::RET_SCRATCH and returns ticket{} (ops it pushed before
//     stay joinable; the executor stops, joins all and returns that code, SW §7.3).
// Kinds without a handler in this image are BAD_RECORD.
//
// Kind numbers and argument words: tsr/opl.hpp (SW §7.2). NOP and JOIN are built into graph.cpp;
// MUL_MAT is kind_mulmat (sw/ctrl/kernels/k_mulmat.cpp); the M7 kinds (W2b, P2.6 / P2.7) are
// kind_add ... kind_unary below, in graph.cpp's kind table, defined by the kernel packages'
// sources (sw/ctrl/kernels/k_*.cpp). sw/ctrl/kernels/zz_kind_stubs.cpp defines each of them weakly
// (BAD_RECORD: "not in this image"), so an image without a kind's source links and refuses its
// records; a strong definition replaces the stub (RISC-V image: every object is linked; host
// archive tsr_fw_host: the stubs' member comes last, so a kernel member defining the handler is
// the one the linker extracts, see that file).
//
// Short kinds (SW §7.3 "Hot path", V2-SW-35, V2-SW-36, V2-SW-37; W2b's performance requirement:
// <= 350 RV cycles of work per decode record): a leaf fast entry (kind_<name>_fast below, the
// executor's first call; no calls in its common path), its argument words read in place into locals
// before the first push, the plan by shifts and compares (plan.hpp, no divide for one tile), the
// one-tile block (kind_ctx::one), one ISSUE_CREDIT read covering the record's pushes;
// sw/ctrl/kernels/k_fast.hpp has these pieces. The general case (more tiles, a record the fast entry
// does not take) is the handler's, out of line, as kind_mulmat's.
#pragma once

#include <stdint.h>

#include "tsr/kernel.hpp"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"

namespace tsr {

// Per-op context the executor hands to a kind handler (one object per launch; op_index is
// refreshed per op).
struct kind_ctx {
    spm_ring * ring;                   // the launch's scratch ring: SPM lines [0, layout.ring_lines)
    const volatile uint8_t * preload;  // the preload area through the SPM window (nullptr: none)
    uint32_t pre_lines;                // its size in lines (header pre_lines)
    spm pre;                           // its first SPM line (same area as `preload`)
    const plan::dev_cfg * cfg;         // device parameters (CFG0-CFG3, CAPS), read once per launch
    plan::mm_limits mml;               // plan::mm_limits_of(*cfg), computed once per launch (a caller
                                       // that builds a kind_ctx by hand fills it the same way)
    uint32_t push_flags;               // flags every issue() of the handler must OR in: ISSUE_PUSH
                                       // bit0 REC when the list has F_REC or the launch is PROF
    uint32_t op_index;                 // index of the op being executed (logs, panics)
    uint32_t ret;                      // 0 at launch start; a failing handler sets opl::RET_SCRATCH /
                                       // RET_BAD_RECORD (the launch then ends, so it is never reset)
    plan::plan_limits pl;              // plan::plan_limits_of(*cfg), computed once per launch like mml (the
                                       // M7 planners' limits; last, so the fields above keep their offsets)
    // The VU s registers as this launch's pushes leave them (W2b; VU §4.4: s[i] is written only by RED
    // with TO_S and by SOP, read only by VU ops, all in VU queue order). Bit i of sreg_valid: the
    // last push of this launch that writes s[i] is an SOP.MOVI of sreg_val[i], so every VU op pushed
    // after it reads that value and a kind needing it there pushes no MOVI (k_fast.hpp sreg_movi;
    // RMS_NORM's 1/n and eps, ROPE's TS / M / freq_scale repeat across a list's records). 0 at launch
    // start (the bank survives launches in hardware, but a kernel never relies on it, V2-SW-4).
    // Depends only on the pushes (SW §4.7). Rule: every push of a kind that writes s[i] other than
    // through sreg_movi (RED TO_S, SOP other than STS) clears bit i (sreg_clobber) in the same
    // record, before any later record can push.
    uint32_t sreg_valid;
    uint32_t sreg_val[16];
    // The one-tile block (plan.hpp one_block_lines; W2b integration): the SPM lines every single-tile
    // record of a DMA-bracketed kind uses instead of a ring block (k_fast.hpp alloc_fast / one_block);
    // one_lines = 0: none (every record then takes the ring). Set once per launch like `pl`.
    spm one;
    uint32_t one_lines;

    // --- helpers of the kind handlers (W2b) -------------------------------------------------------
    // A failing handler's exit: ctx.ret = code (opl::RET_BAD_RECORD / RET_SCRATCH), no ticket.
    TSR_HOT ticket fail(uint32_t code) {
        ret = code;
        return ticket{};
    }
    // Preload references (SW §7.2; the record's "preload byte offset" words): n elements of
    // 2^esz_log2 bytes at byte offset off lie inside this list's preload area and off & align_mask
    // == 0 (opl::PRE_*_ALIGN - 1). No 64-bit arithmetic, no overflow (the area is <= 16 KiB). The same
    // rule as opl::check_args (the host reader's check).
    TSR_HOT bool pre_ok(uint32_t off, uint32_t n, uint32_t esz_log2, uint32_t align_mask) const {
        const uint32_t pb = pre_lines << 6;
        return (off & align_mask) == 0u && off <= pb && n <= ((pb - off) >> esz_log2);
    }
    // The SPM line of a line-aligned preload reference (a unit operand, e.g. ROPE's positions).
    TSR_HOT spm pre_line(uint32_t off) const { return spm{pre.line + (off >> 6)}; }
    // The preload words at byte offset off through the line-buffered SPM alias (CC §10.5: one bridge
    // read per line; any push invalidates the buffer, so read a line's words before the next push).
    // Safe without a join: the executor joined the preload loads before op 0, and nothing writes the
    // area during the launch.
    TSR_HOT const volatile uint32_t * pre_lb(uint32_t off) const { return spm_lb_ptr<uint32_t>(pre, off); }
};

// A kind handler (SW §7.3 step 2.4). d = after(ticket[dep0], ticket[dep1]); kflags = word0
// [23:16]; n_args = the record's argument count, which the executor has already checked to be the
// kind's exact count (opl::kind_nargs, or kind_entry::nargs for a test kind).
// args = the record's argument words where the executor found them: normally the record itself in
// the OPL area through the line-buffered SPM alias (CC §10.5; no copy), else a DTCM copy (a record
// whose lines wrap the streamed OPL area, or a host test's array). The words stay valid for the
// whole call (the executor loads no new list chunk before the handler returns), but every
// control-window access (a push) invalidates the line buffer: a handler reads the words it needs
// into locals before its first push (one line fill), as kind_mulmat does.
using kind_fn = ticket (*)(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);

// MUL_MAT (kind 0x10, SW §7.4): sw/ctrl/kernels/k_mulmat.cpp.
ticket kind_mulmat(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);

// The M7 kinds (SW §7.2, §7.5; argument words opl.hpp ew / scl / rms / glu / rope / getrows /
// setrows / fa / cpy / unary). Defined by the W2b kernel sources; weak BAD_RECORD stubs in
// sw/ctrl/kernels/zz_kind_stubs.cpp (header comment).
// 0x11 ADD, 0x12 MUL, 0x13 SCALE, 0x14 RMS_NORM, 0x15 GLU, 0x16 ROPE, 0x17 GET_ROWS, 0x18 SET_ROWS_KV,
// 0x19 FLASH_ATTN, 0x1A CPY, 0x1B UNARY:
ticket kind_add(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_mul(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_scale(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_rms_norm(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_glu(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_rope(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_get_rows(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_set_rows_kv(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_flash_attn(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_cpy(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
ticket kind_unary(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);

// The decode fast paths of the short M7 kinds (W2b integration; SW §7.3 "M7 kinds in the executor"),
// the executor's first call for a record of these kinds (graph.cpp's kind table): args, kflags and d
// as for the kind_fn (n_args already checked by the executor). A fast function either takes the
// record (pushes its tile ops, returns the op's ticket) or returns KIND_NOT_FAST having changed
// nothing (no push, no state), and the executor then calls the kind's handler (kind_fn), whose
// general path runs it. The common path of a fast function calls nothing (a leaf: no callee-saved
// registers to save, the executor's state stays in its own callee-saved registers), so a short
// record costs the executor one indirect call, as MUL_MAT's mulmat_t1_fast (SW §7.3 "Hot path"). Each
// kind's handler begins with the same function, so the two entries push the same ops (host tests call
// the handlers). zz_kind_stubs.cpp defines each one weakly as KIND_NOT_FAST.
constexpr uint32_t KIND_NOT_FAST = 0xE000'0001u;   // unit 14: never a ticket (CC §7.4); = kf / mmk NOT_FAST
using kind_fast_fn = uint32_t (*)(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_add_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_mul_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_scale_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_rms_norm_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_glu_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_rope_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_get_rows_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_set_rows_kv_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_cpy_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t kind_unary_fast(const volatile uint32_t * args, uint32_t kflags, deps d, kind_ctx & ctx);

// The executor's kind table (graph.cpp, .rodata): the handler of `kind`, or nullptr (BAD_RECORD).
kind_fn kind_handler(uint32_t kind);

// Test images only (tsr_fw_image extras, sw/ctrl/tests/fw/exec_*.cpp; host test drivers): a
// handler for a kind that has no built-in handler (NOP, JOIN and MUL_MAT are built in). Kinds
// 0x30-0x3F are reserved for test kinds (inside the M8 range 0x20-0x3F, never used by a production
// list). Returns true and fills *out to take the kind (nargs: the exact argument count a record of
// that kind must carry). The production image does not define it (weak reference: nullptr); the
// executor consults it only for a kind its own table does not have, so production records never
// pay for it.
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

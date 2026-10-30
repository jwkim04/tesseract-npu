// tsr/kernel.hpp — the tile/VU header library of the firmware SDK (docs/v2/06-software-
// verification.md "SW" §6.7; base docs/ctrl-riscv.md §8.2 without CALL). Freestanding C++20
// (SW §6.1: no libstdc++, no heap, no exceptions / RTTI, no float, V2-SW-8), header-only.
//
// What is here (namespace tsr):
//   - types: spm, dram, handle() / from_handle(), wref, qa, op (mmio.hpp), ticket, deps, after();
//   - constexpr op builders LD, ST, QACT, MVM, NOP, FENCE (the VU / AT builders are generated:
//     vu.hpp, at.hpp, included here); field checks below;
//   - the issue path (CC §5.1, §7): issue() with the credit check, done / join / join_irq,
//     join2 (SYNC NOP), join_all (FENCE + join), error / first_error, qclear, pidx_now, the other
//     QCTRL entries, TRACE_MARK, stop_requested (EX §4.9), the SPM windows (spm_ptr);
//   - group (latest ticket per unit, folded to <= 2 deps), spm_ring (scratch ring with release
//     tickets).
// The hardware access goes through namespace tsr::hw, which backend_dev.hpp (RISC-V: MMIO, CSRs)
// or backend_host.hpp (TSR_FW_HOST: the simulator through its test hooks, SW §6.8) provides with
// the same semantics, so a kernel compiled for the host pushes the same op stream (words, push
// values, deps) as on tsr_rv.
//
// Tickets and deps (CC §7.5, §8.3, §9). Ticket = {unit[31:28], seq[27:0]}; unit 15 = none (always
// done). issue() returns the ticket the hardware assigned (ISSUE_TICKET, read right after the
// push): exact for every opcode, including one the issue path refuses (SYNC, ILLEGAL_OP).
// done(t) iff (DONE_SEQ[u] - s) mod 2^28 in [1, 2^27) (wrap-safe). Same-unit tickets are ordered:
// done(later) implies done(earlier) (I1: completion at the head, in queue order); that holds for
// the SYNC unit too, so a SYNC ticket subsumes every earlier SYNC ticket. An op takes at most two
// hardware deps (ISSUE_DEP0 / DEP1). after() folds same-unit pairs; issue() pushes the deps it
// is given unchanged (the replay kernel relies on that).
//
// Determinism (SW §4.7, V2-SW-30): nothing here lets a timing-dependent value (DONE_SEQ, credits,
// QSTATUS, MTIME, ...) reach pushed words, deps or SPM stores. Such values only end wait loops:
// issue() spins on the credit, join() on DONE_SEQ, spm_ring joins in software chosen by count,
// group folds by unit order. first_error() / error() / err_units() read the error records
// (ERR_CODE / ERR_SEQ / ...), which are exact once the tickets concerned are done (I2); called
// earlier they are a snapshot (timing-dependent). err_units_poll() (QSTATUS) is never exact.
// The library keeps no state that changes what is pushed across launches (V2-SW-4): spm_ring is
// rebuilt by init() per launch; the only DTCM caches hold hardware constants (ISSUE_UNITS, CAPS
// and the opcode -> unit table derived from them).
//
// Not reentrant (single hart, docs/ctrl-riscv.md §3.2): never call issue() from an interrupt
// handler (SW §6.5). Logging and panic (TSR_LOG, tsr::panic) are log.hpp's (SW §6.6), included
// here when present.
#ifndef TSR_KERNEL_HPP
#define TSR_KERNEL_HPP

#include <stddef.h>
#include <stdint.h>

#include "traits.hpp"
#include "mmio.hpp"
#include "vu.hpp"
#include "at.hpp"

#if __has_include("log.hpp")
#include "log.hpp"
#define TSR_KERNEL_HAVE_LOG 1
#endif
#if __has_include("rt.hpp")
#include "rt.hpp"                             // tsr_args, rt_stop_requested (the runtime's soft-stop poll)
#define TSR_KERNEL_HAVE_RT 1
#endif

namespace tsr {

// The issue path is inlined at every call site (builders, issue, the backend's push): with the op
// in registers a push is the credit read and a handful of stores (CC §7.6). Measured on tsr_rv
// (TSR_SIM_RV_RATIO 1, the model's CPI; loop overhead included): 36 RISC-V cycles per NOP push,
// ~100 per 8-word VOP push; ~190 per LD push when -Os built the op on the stack and called issue().
#define TSR_HOT [[gnu::always_inline]] inline

// -----------------------------------------------------------------------------------------------
// Types (docs/ctrl-riscv.md §8.2)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t LINE_BYTES = 64u;          // SPM line = DRAM beat (spec BEAT_BITS 512)
constexpr uint32_t T_MAX_ARCH = 8u;           // spec 구성 파라미터: T_MAX upper bound
constexpr uint32_t SPM_NONE = 0xFFFF'FFFFu;   // spm.line of "no block" (spm_ring::alloc failure)

struct spm {                                  // SPM line address
    uint32_t line;
};
constexpr spm operator+(spm s, uint32_t lines) { return spm{s.line + lines}; }
constexpr bool operator==(spm a, spm b) { return a.line == b.line; }
constexpr bool valid(spm s) { return s.line != SPM_NONE; }

struct dram {                                 // DRAM channel + beat (64-byte units)
    uint8_t ch;
    uint32_t beat;
};
constexpr dram operator+(dram d, uint32_t beats) { return dram{d.ch, d.beat + beats}; }
// ABI form of a DRAM address: {ch[31:28], beat[27:0]} (SW §6.5, docs/ctrl-riscv.md:1094-1097).
TSR_HOT constexpr uint32_t handle(dram d) { return (uint32_t(d.ch) << 28) | (d.beat & 0x0FFF'FFFFu); }
TSR_HOT constexpr dram from_handle(uint32_t h) { return dram{uint8_t(h >> 28), h & 0x0FFF'FFFFu}; }

struct wref {                                 // symmetric-allocated weight tensor
    uint32_t w_off;                           // beat offset on every channel (MVM W_OFF)
    uint8_t fmt;                              // FMT_Q8_0 / FMT_Q4_K
};
enum class qa : uint8_t { a32 = QA_A32, a256 = QA_A256 };
constexpr qa qa_for_fmt(uint32_t fmt) { return fmt == FMT_Q8_0 ? qa::a32 : qa::a256; }   // Q8_0 -> A32, Q4_K -> A256
constexpr qa qa_for(wref w) { return qa_for_fmt(w.fmt); }
constexpr uint32_t f32_lines(uint32_t n) { return (4u * n + 63u) / 64u; }   // lines of n f32 values
// (QA_LINES, ROW_BEATS and the block sizes are plan.hpp's: tsr::plan::qa_lines, row_beats, ...)

// Ticket {unit[31:28], seq[27:0]} (CC §7.5); the default value is none (unit 15, always done).
struct ticket {
    uint32_t v = TICKET_NONE;
    constexpr bool none() const { return (v >> 28) == UNIT_NONE; }
    constexpr uint32_t unit() const { return v >> 28; }
    constexpr uint32_t seq() const { return v & TICKET_SEQ; }
};
constexpr bool operator==(ticket a, ticket b) { return a.v == b.v; }
constexpr ticket make_ticket(uint32_t unit, uint32_t seq) { return ticket{(unit << 28) | (seq & TICKET_SEQ)}; }
// Same unit: a was pushed no earlier than b ((a - b) mod 2^28 < 2^27; outstanding tickets of a unit
// span less than 2^27).
TSR_HOT constexpr bool same_or_after(ticket a, ticket b) { return ((a.v - b.v) & TICKET_SEQ) < 0x0800'0000u; }
static_assert(same_or_after(ticket{0x2000'0001u}, ticket{0x2FFF'FFFFu}) && !same_or_after(ticket{0x2FFF'FFFFu}, ticket{0x2000'0001u}),
              "same_or_after across the 2^28 wrap");
// The later of two tickets of the same unit (either may be none: the other is returned).
TSR_HOT constexpr ticket later(ticket a, ticket b) {
    if (a.none()) return b;
    if (b.none()) return a;
    return same_or_after(a, b) ? a : b;
}

// The two hardware dependencies of a push (ISSUE_DEP0 / DEP1).
struct deps {
    ticket a, b;
    constexpr deps() = default;
    constexpr deps(ticket x, ticket y = ticket{}) : a(x), b(y) {}
};
// deps on a and b: a none is dropped (the other moves to the first slot), two tickets of the same
// unit fold to the later one (it implies the earlier, I1).
TSR_HOT constexpr deps after(ticket a = {}, ticket b = {}) {
    if (a.none()) return deps{b};
    if (b.none() || a == b) return deps{a};
    if (a.unit() == b.unit()) return deps{later(a, b)};
    return deps{a, b};
}

// ISSUE_PUSH flag bits a push may carry (issue()'s flags; [15:8] WMASK comes from op::used).
constexpr uint32_t PUSH_REC = ISSUE_PUSH_REC;  // completion record wanted (DB §8.2)

// -----------------------------------------------------------------------------------------------
// Op builders: LD, ST, QACT, MVM, NOP, FENCE (spec ISA, EX §7.2; the VU / AT ones are generated)
//
// Field checks match the checker (tsr_sim.c chk_ldst / chk_qact / chk_mvm, CC §8.2):
//   - in a constant expression (constexpr op x = MVM(...); static_assert) every rule the checker
//     applies whatever the configuration is a compile error: a field that does not fit its width,
//     QACT kind >= 2 (ILLEGAL_OP), fmt >= 2 (BAD_FMT), K = 0 or not a multiple of the block size
//     (K_RANGE), T = 0 or T > 8 (T_RANGE). Configuration-dependent rules (NUM_CH, K_MAX, T_MAX,
//     SPM and DRAM ranges) are the checker's alone;
//   - op::used is the encoding's <OP>_WORDS (the words that hold a field; a field word that is 0 is
//     written and sent as 0: the same push content without a test per word), as in vu.hpp / at.hpp;
//     the op is built in place (no copies on the issue path);
//   - at run time the builders encode what they are given and the checker reports the code.
//     A value that does not fit its field is never masked into a different, legal value: MVM K
//     > 0xFFFF saturates to 0xFFFF (K_RANGE, as for the true K), T > 15 to 15 (T_RANGE), fmt > 15
//     to 15 (BAD_FMT); an LD / ST channel >= 16 or a QACT kind > 15 sets the reserved w0 bit 12
//     (ILLEGAL_OP, as the checker gives for any channel >= NUM_CH or kind >= 2).
// -----------------------------------------------------------------------------------------------
namespace detail {
// Deliberately not constexpr: reaching one of these during constant evaluation is a compile error
// naming the broken rule. At run time they do nothing.
// (op_field_does_not_fit is mmio.hpp's, shared with the generated builders.)
inline void op_illegal_qact_kind() {}
inline void op_bad_fmt() {}
inline void op_k_range() {}
inline void op_t_range() {}
constexpr uint32_t POISON_W0 = 1u << 12;      // reserved in LD / ST / QACT w0 (LD_RSV_W0 bit 12)
}  // namespace detail

TSR_HOT constexpr op LD(spm dst, dram src, uint32_t lines) {
    op o{};
    o.w[0] = OP_LD | ((uint32_t(src.ch) & 0xFu) << LD_CH_LSB);
    if (src.ch >= NUM_CH_MAX) {
        detail::op_field_does_not_fit();
        o.w[0] |= detail::POISON_W0;
    }
    o.w[1] = dst.line;
    o.w[2] = src.beat;
    o.w[3] = lines;
    o.used = uint8_t(LD_WORDS);
    return o;
}

TSR_HOT constexpr op ST(spm src, dram dst, uint32_t lines) {
    op o{};
    o.w[0] = OP_ST | ((uint32_t(dst.ch) & 0xFu) << ST_CH_LSB);
    if (dst.ch >= NUM_CH_MAX) {
        detail::op_field_does_not_fit();
        o.w[0] |= detail::POISON_W0;
    }
    o.w[1] = src.line;
    o.w[2] = dst.beat;
    o.w[3] = lines;
    o.used = uint8_t(ST_WORDS);
    return o;
}

// T f32 vectors of K values at src (src_stride lines apart; 0 = ceil(4K/64)) -> T QA images at
// dst, QA_LINES(kind, K) lines apart.
TSR_HOT constexpr op QACT(qa kind, spm dst, spm src, uint32_t k, uint32_t t, uint32_t src_stride = 0) {
    op o{};
    const uint32_t kd = uint32_t(kind);
    o.w[0] = OP_QACT | ((kd & 0xFu) << QACT_KIND_LSB);
    if (kd > 0xFu) {
        detail::op_field_does_not_fit();
        o.w[0] |= detail::POISON_W0;
    }
    if (kd > QA_A256) detail::op_illegal_qact_kind();
    if (k == 0u || k % (kd == QA_A256 ? 256u : 32u) != 0u) detail::op_k_range();
    if (t == 0u || t > T_MAX_ARCH) detail::op_t_range();
    o.w[1] = dst.line;
    o.w[2] = src.line;
    o.w[3] = k;
    o.w[4] = t;
    o.w[5] = src_stride;
    o.used = uint8_t(QACT_WORDS);
    return o;
}

// dst (T vectors of N f32, dst_stride lines apart; 0 = ceil(4N/64)) = W (N x K, fmt, W_OFF beats
// on every channel) . x (T QA images); acc: dst += W . x.
TSR_HOT constexpr op MVM(spm dst, spm x, wref w, uint32_t n, uint32_t k, uint32_t t, uint32_t dst_stride = 0,
                 bool acc = false) {
    op o{};
    uint32_t fmt = w.fmt;
    if (fmt > 0xFu) {
        detail::op_field_does_not_fit();
        fmt = 0xFu;
    }
    if (fmt > FMT_Q4_K) detail::op_bad_fmt();
    uint32_t kf = k;
    if (kf > 0xFFFFu) {
        detail::op_field_does_not_fit();
        kf = 0xFFFFu;
    }
    if (k == 0u || k % (fmt == FMT_Q8_0 ? 32u : 256u) != 0u) detail::op_k_range();
    uint32_t tf = t;
    if (tf > 0xFu) {
        detail::op_field_does_not_fit();
        tf = 0xFu;
    }
    if (t == 0u || t > T_MAX_ARCH) detail::op_t_range();
    o.w[0] = OP_MVM | (fmt << MVM_FMT_LSB) | (acc ? MVM_ACC : 0u);
    o.w[1] = dst.line;
    o.w[2] = x.line;
    o.w[3] = w.w_off;
    o.w[4] = n;
    o.w[5] = (kf << MVM_K_LSB) | (tf << MVM_T_LSB);
    o.w[6] = dst_stride;
    o.used = uint8_t(MVM_WORDS);
    return o;
}

TSR_HOT constexpr op NOP() { op o{}; o.w[0] = OP_NOP; o.used = uint8_t(NOP_WORDS); return o; }         // SYNC join point
TSR_HOT constexpr op FENCE() { op o{}; o.w[0] = OP_FENCE; o.used = uint8_t(FENCE_WORDS); return o; }   // SYNC barrier

// Encodings against the generated layouts (spec ISA; tsr_isa.h tsr_enc_*).
static_assert(LD(spm{5}, dram{2, 7}, 3).w[0] == (OP_LD | (2u << LD_CH_LSB)) && LD(spm{5}, dram{2, 7}, 3).used == 0x0Fu);
static_assert(ST(spm{5}, dram{1, 0}, 1).w[2] == 0u && ST(spm{5}, dram{1, 0}, 1).used == 0x0Fu);   // used = ST_WORDS
static_assert(QACT(qa::a256, spm{1}, spm{2}, 512, 2).w[0] == (OP_QACT | (1u << QACT_KIND_LSB)));
static_assert(MVM(spm{0}, spm{8}, wref{100, FMT_Q4_K}, 64, 256, 2, 0, true).w[0] ==
              (OP_MVM | (FMT_Q4_K << MVM_FMT_LSB) | MVM_ACC));
static_assert(MVM(spm{0}, spm{8}, wref{100, FMT_Q8_0}, 64, 4096, 3).w[5] == (4096u | (3u << MVM_T_LSB)));
static_assert(NOP().w[0] == OP_NOP && NOP().used == 1u && FENCE().w[0] == OP_FENCE);

}  // namespace tsr

// -----------------------------------------------------------------------------------------------
// Backend: namespace tsr::hw (control window, core window, push, QCTRL, wait step, mie / WFI, SPM
// windows, soft-stop query)
// -----------------------------------------------------------------------------------------------
#if defined(TSR_FW_HOST)
#include "backend_host.hpp"
#else
#include "backend_dev.hpp"
#endif

namespace tsr {

#if !defined(TSR_KERNEL_HAVE_LOG)
// tsr::panic (SW §6.6, DB §6.3): trap with mcause 24, mtval = code. Defined by the runtime
// (log.hpp / rt/), declared here until log.hpp exists.
[[noreturn]] void panic(uint32_t code, uint32_t a0 = 0, uint32_t a1 = 0, uint32_t a2 = 0);
#endif

// Panic code of this library (mtval; a violated firmware invariant, never a data error).
constexpr uint32_t PANIC_RING_RELEASE = 0x5250'0001u;   // spm_ring::release of a block it does not hold

// -----------------------------------------------------------------------------------------------
// Issue path (CC §5.1, §7)
// -----------------------------------------------------------------------------------------------
namespace detail {
// Hardware constants the credit check needs, read once (ISSUE_UNITS, CAPS through the core
// window); bit 31 = valid. The host backend clears them (and g_unit_tab) in host_bind().
inline uint32_t g_hw_units = 0;
inline uint32_t g_hw_caps = 0;

inline void hw_info_load() {
    g_hw_caps = hw::core_read(CSR_CAPS);
    g_hw_units = hw::cw_read(CW_ISSUE_UNITS) | 0x8000'0000u;
}

// The unit an op is pushed to (CC §7.4): UNIT_OF(opcode) when the issue path accepts the opcode
// (legal, CAPS bits present, unit built), else SYNC (it completes ILLEGAL_OP there). LD, ST, QACT,
// MVM, NOP and FENCE need no CAPS and their units (DMA, QACT, MVM, SYNC) are always built: the fast
// path. Every other opcode is looked up in g_unit_tab (unit + 1 per opcode, 0 = not known yet),
// filled from the cached ISSUE_UNITS / CAPS on its first use (push_unit_slow): one DTCM load.
inline uint8_t g_unit_tab[128] = {};
[[gnu::noinline]] inline uint32_t push_unit_slow(uint32_t opc) {
    uint32_t u = UNIT_SYNC;
    if (op_legal(opc)) {
        const uint32_t need = op_caps(opc);
        if (!(g_hw_units & 0x8000'0000u)) hw_info_load();
        if ((g_hw_caps & need) == need && ((g_hw_units >> op_unit(opc)) & 1u)) u = op_unit(opc);
    }
    g_unit_tab[opc] = uint8_t(u + 1u);
    return u;
}
TSR_HOT uint32_t push_unit(uint32_t w0) {
    const uint32_t opc = w0 & 0xFFu;
    switch (opc) {
    case OP_LD: case OP_ST: return UNIT_DMA;
    case OP_QACT: return UNIT_QACT;
    case OP_MVM: return UNIT_MVM;
    case OP_NOP: case OP_FENCE: return UNIT_SYNC;
    default: {
        if (opc >= 128u) return UNIT_SYNC;   // no legal opcode above FENCE (0x7F)
        const uint32_t v = g_unit_tab[opc];
        return v != 0u ? v - 1u : push_unit_slow(opc);
    }
    }
}

TSR_HOT uint32_t credit_of(uint32_t credits, uint32_t u) { return (credits >> (CREDIT_FIELD_W * u)) & 0xFu; }
}  // namespace detail

// Credit of unit u now (CC §5.1; timing-dependent: only for wait loops and diagnostics).
inline uint32_t credit(uint32_t u) { return detail::credit_of(hw::cw_read(CW_ISSUE_CREDIT), u); }

// Pushes one op: spins until its unit has a credit (CC §7.3: software never pushes without one;
// the hardware wait is a safety net), stages the words of op::used (w0 always), the deps that are
// not none, then ISSUE_PUSH = used << 8 | flags (flags: PUSH_REC; bits [15:8] of flags are
// ignored), and returns the ticket the hardware assigned (ISSUE_TICKET). Waits only for a credit.
namespace detail {
// The ticket a push returns is never none (ISSUE_TICKET carries the unit the entry went to, CC
// §7.1, 0..5). Telling the compiler so lets a later push drop its "dep is none" test for it.
TSR_HOT ticket pushed(uint32_t v) {
#if !defined(TSR_FW_HOST)
    if ((v >> 28) == UNIT_NONE) __builtin_unreachable();
#endif
    return ticket{v};
}
}  // namespace detail

TSR_HOT ticket issue(const op & o, deps d = {}, uint32_t flags = 0) {
    const uint32_t u = detail::push_unit(o.w[0]);
    while (detail::credit_of(hw::cw_read(CW_ISSUE_CREDIT), u) == 0u) hw::pause();
    const uint32_t push = (uint32_t(o.used) << ISSUE_PUSH_WMASK_LSB) | (flags & ~ISSUE_PUSH_WMASK);
    return detail::pushed(hw::push(o, d.a.v, d.b.v, push));
}

// A batch of pushes covered by one ISSUE_CREDIT read (CC §7.3: software checks the credit before
// pushing; issue() reads it before every push). Field u of ISSUE_CREDIT is min(free slots of queue
// u, free issue-FIFO entries, 15) (CC §5.1). Between two reads only this hart's pushes take credits
// away: a push to u takes one slot of queue u and one FIFO entry (so it may lower every unit's
// field by one); completions only add. A batch of n_u pushes to each unit u, N in total, therefore
// needs no further read when, at the read, field u >= n_u for every u and N <= the free FIFO
// entries, which every field bounds from below: credits_cover() takes the larger of the DMA and VU
// fields for that (VU's 16-deep queue is mostly empty, so its field is usually the FIFO's). The
// batch then pushes with issue_nc. The value is timing-dependent: it only chooses between pushing
// at once and issue()'s per-push wait, which push the same words, deps and flags (SW §4.7,
// V2-SW-30).
inline uint32_t credits() { return hw::cw_read(CW_ISSUE_CREDIT); }
TSR_HOT bool credits_cover(uint32_t c, uint32_t n_total) {
    const uint32_t a = detail::credit_of(c, UNIT_DMA), b = detail::credit_of(c, UNIT_VU);
    return (a > b ? a : b) >= n_total;
}
TSR_HOT bool has_credits(uint32_t c, uint32_t u, uint32_t n) { return detail::credit_of(c, u) >= n; }
TSR_HOT ticket issue_nc(const op & o, deps d = {}, uint32_t flags = 0) {
    const uint32_t push = (uint32_t(o.used) << ISSUE_PUSH_WMASK_LSB) | (flags & ~ISSUE_PUSH_WMASK);
    return detail::pushed(hw::push(o, d.a.v, d.b.v, push));
}
// issue (CHK) or issue_nc (!CHK), for code that pushes the same ops on both paths.
template <bool CHK> TSR_HOT ticket issue_c(const op & o, deps d = {}, uint32_t flags = 0) {
    if constexpr (CHK) return issue(o, d, flags);
    else return issue_nc(o, d, flags);
}

// Ticket (u, s) is done iff (D[u] - s) mod 2^28 is in [1, 2^27) (CC §7.5): wrap-safe.
constexpr bool seq_done(uint32_t done_seq, uint32_t seq) { return ((done_seq - seq) & TICKET_SEQ) - 1u < 0x07FF'FFFFu; }
static_assert(seq_done(1, 0) && !seq_done(0, 0) && !seq_done(5, 9), "done: D passed s / not yet");
static_assert(seq_done(0, 0x0FFF'FFFFu) && seq_done(3, 0x0FFF'FFFEu), "done across the 2^28 wrap");
static_assert(seq_done(0x07FF'FFFFu, 0) && !seq_done(0x0800'0000u, 0), "half range: [1, 2^27)");

// done(t): seq_done(DONE_SEQ[u], seq); none is always done. A unit that is not built reads
// DONE_SEQ 0. Timing-dependent: use it to end waits, never to choose what to push.
TSR_HOT bool done(ticket t) {
    if (t.none()) return true;
    return seq_done(hw::cw_read(CW_DONE_SEQ(t.unit())), t.seq());
}

// Spins until t is done (DONE_SEQ reads; the simulator fast-forwards such spins, SW §4.4).
TSR_HOT void join(ticket t) {
    while (!done(t)) hw::pause();
}

// Sleeps until t is done: WAIT_TICKET = t, irq_fast[0] enabled in mie, WFI while not done
// (mstatus.MIE stays 0: WFI wakes on a pending enabled line without taking it, CC §12.3); then
// WAIT_TICKET is disarmed and mie restored.
inline void join_irq(ticket t) {
    if (done(t)) return;
    hw::cw_write(CW_WAIT_TICKET, t.v);
    const uint32_t old = hw::mie_set(1u << (16u + IRQ_FAST_WAIT_TICKET));
    while (!done(t)) hw::wfi();
    hw::mie_write(old);
    hw::cw_write(CW_WAIT_TICKET, TICKET_NONE);
}

// A SYNC NOP over a and b: one ticket whose completion implies both (I1).
inline ticket join2(ticket a, ticket b) { return issue(NOP(), deps{a, b}); }

// FENCE (completes when every entry pushed before it, on every unit, is done), then join it.
inline ticket fence() { return issue(FENCE()); }
inline void join_all() { join(fence()); }

// The error code of a done ticket (CC §9.5, EX §8.1): 0 if it succeeded; its own code if it is
// the entry that put its unit into error state; ABORTED for a later entry of that unit (the
// hardware keeps only the first record per unit: a later entry that was itself poisoned, e.g.
// BAD_DEP, completes with its own code, visible only in completion records, DB §8.2). Valid once
// done(t) (I2: the record is visible before DONE_SEQ passes it); undefined for tickets pushed
// before the last qclear().
inline uint32_t error(ticket t) {
    if (t.none()) return 0u;
    const uint32_t code = hw::cw_read(CW_ERR_CODE(t.unit())) & 0xFFu;
    if (code == 0u) return 0u;
    const uint32_t x = (t.seq() - hw::cw_read(CW_ERR_SEQ(t.unit()))) & TICKET_SEQ;
    if (x >= 0x0800'0000u) return 0u;             // completed before the unit's first error
    return x == 0u ? code : ERR_ABORTED;
}

// The first error record (EX §4.4 r3-r6): the one with the smallest push index (ERR_PIDX) over
// the units in error state (ERR_CODE != 0, as the runtime's R7).
struct err_rec {
    uint32_t code;      // ERR_CODE, 0 = none
    ticket t;           // {unit, ERR_SEQ}
    uint32_t info;      // ERR_INFO
    uint32_t w0;        // ERR_W0
    uint32_t pidx;      // ERR_PIDX
};
// Bit u = unit u is in error state, from the error records (ERR_CODE(u) != 0, one control-window
// read per unit): exact for every error of an entry covered by a done ticket (I2: the record is
// visible before DONE_SEQ passes it), so it may decide RET and records after a join, as the
// runtime's R7 does. (fix W2a: QSTATUS[13:8] is a separately synchronised level, CC §11.2, that
// I2 does not order with DONE_SEQ; SW §4.7 lists QSTATUS as timing-dependent.)
inline uint32_t err_units() {
    uint32_t m = 0;
    for (uint32_t u = 0; u < NUM_UNITS; ++u)
        if ((hw::cw_read(CW_ERR_CODE(u)) & 0xFFu) != 0u) m |= 1u << u;
    return m;
}
// QSTATUS[13:8] (one read): timing-dependent (SW §4.7). Only for polls whose outcome may depend on
// timing, such as the executor's STOP_ON_ERROR decision (SW §7.3 step 2.6); never for RET or records.
inline uint32_t err_units_poll() { return field_get(hw::cw_read(CW_QSTATUS), QSTATUS_ERR, QSTATUS_ERR_LSB); }
inline bool first_error(err_rec * r) {
    const uint32_t ref = hw::cw_read(CW_ISSUE_PIDX);   // every error has a push index below it
    bool found = false;
    uint32_t best_age = 0, best_u = 0, best_pidx = 0, best_code = 0;
    for (uint32_t u = 0; u < NUM_UNITS; ++u) {
        const uint32_t code = hw::cw_read(CW_ERR_CODE(u)) & 0xFFu;
        if (code == 0u) continue;
        const uint32_t p = hw::cw_read(CW_ERR_PIDX(u));
        const uint32_t age = ref - p;                  // larger = pushed earlier (wrap-safe)
        if (!found || age > best_age) {
            found = true;
            best_age = age;
            best_u = u;
            best_pidx = p;
            best_code = code;
        }
    }
    if (r != nullptr) {
        if (!found) {
            *r = err_rec{0u, ticket{}, 0u, 0u, 0u};
        } else {
            r->code = best_code;
            r->t = make_ticket(best_u, hw::cw_read(CW_ERR_SEQ(best_u)));
            r->info = hw::cw_read(CW_ERR_INFO(best_u));
            r->w0 = hw::cw_read(CW_ERR_W0(best_u));
            r->pidx = best_pidx;
        }
    }
    return found;
}
// The code of the first error, 0 if none.
inline uint32_t first_error() {
    err_rec r{};
    return first_error(&r) ? r.code : 0u;
}

// QCTRL entries (CC §5.1, §9.5), in push order through the issue FIFO. qclear() returns its SYNC
// ticket (done = every earlier entry done, error states and records cleared); a QCLEAR takes a
// SYNC credit like a push.
inline ticket qclear() {
    while (credit(UNIT_SYNC) == 0u) hw::pause();
    return ticket{hw::qctrl(QCTRL_QCLEAR)};
}
inline void qabort() { hw::qctrl(QCTRL_QABORT); }
inline void qhold(bool on) { hw::qctrl(on ? QCTRL_QHOLD_SET : QCTRL_QHOLD_CLR); }

// ISSUE_PIDX (CC §5.1): the push index the next ticketed entry gets. Exact and timing-independent.
inline uint32_t pidx_now() { return hw::cw_read(CW_ISSUE_PIDX); }
// ISSUED_SEQ[u]: the sequence the next push to unit u gets (exact, timing-independent).
inline uint32_t issued_seq(uint32_t u) { return hw::cw_read(CW_ISSUED_SEQ(u)) & TICKET_SEQ; }

// QSTATUS.QIDLE (timing-dependent).
inline bool queues_idle() { return (hw::cw_read(CW_QSTATUS) & QSTATUS_QIDLE) != 0u; }

// TRACE_MARK (DB §8.4): a marker entry through the issue FIFO (no ticket, no push index).
inline void trace_mark(uint32_t value) { hw::cw_write(CW_TRACE_MARK, value); }

// EX §4.9: a soft stop covers the running launch (STOP_VALID and (LQ_STOP - seq) mod 2^32 < 2^31).
// Kernels poll it at safe points and return early (status STOPPED). Timing-dependent by nature.
// With the runtime (rt.hpp) it is rt_stop_requested(), which also remembers a true answer for R7
// (STOPPED); hw::stop_requested() is the bare register test.
#if defined(TSR_KERNEL_HAVE_RT)
inline bool stop_requested() { return rt_stop_requested(); }
#else
inline bool stop_requested() { return hw::stop_requested(); }
#endif

// Core window (CC §5.3): a word of the host CSR map, e.g. CSR_SREG(i) after a RED / SOP join.
inline uint32_t core_csr(uint32_t csr_off) { return hw::core_read(csr_off); }

// SPM windows (CC §4.1): exact (0x4000_0000 + 64 line + off) and the line-buffered alias
// (0x5000_0000, CC §10.5: loads may hit the one-line buffer; any control-window access, store or
// interrupt edge invalidates it).
template <class T> inline volatile T * spm_ptr(spm l, uint32_t byte_off = 0) {
    return reinterpret_cast<volatile T *>(hw::spm_addr(LINE_BYTES * l.line + byte_off));
}
template <class T> inline volatile T * spm_lb_ptr(spm l, uint32_t byte_off = 0) {
    return reinterpret_cast<volatile T *>(hw::spm_lb_addr(LINE_BYTES * l.line + byte_off));
}

// -----------------------------------------------------------------------------------------------
// group: the latest ticket per unit, folded to <= 2 deps
// -----------------------------------------------------------------------------------------------
// add() keeps, per unit, the later ticket (done(later) implies done(earlier), I1). as_deps():
// with tickets on at most two units they are the deps; otherwise SYNC NOPs (join2) fold them:
// any NOP subsumes every earlier SYNC entry (SYNC queue order), so a SYNC ticket already in the
// group is dropped, and the k other tickets need ceil((k - 1) / 2) NOPs, paired in unit order
// (0, 1), (2, 3), ...: the deps are then {the ticket left over, if any, the last NOP}. The group
// is replaced by the folded tickets, so a second as_deps() pushes nothing and returns the same
// deps (non-SYNC tickets in unit order, the SYNC ticket last). Deterministic: depends only on the
// tickets added.
struct group {
    ticket t[16];                                 // index = unit id

    constexpr void add(ticket x) {
        if (!x.none()) t[x.unit()] = later(t[x.unit()], x);
    }
    constexpr void add(deps d) {
        add(d.a);
        add(d.b);
    }
    constexpr void clear() {
        for (auto & x : t) x = ticket{};
    }
    constexpr uint32_t count() const {
        uint32_t n = 0;
        for (const auto & x : t) n += x.none() ? 0u : 1u;
        return n;
    }
    constexpr bool empty() const { return count() == 0u; }

    deps as_deps() {
        ticket list[16];
        uint32_t k = 0;
        bool sync = false;
        for (uint32_t u = 0; u < 16u; ++u) {
            if (t[u].none()) continue;
            if (u == UNIT_SYNC) sync = true;
            else list[k++] = t[u];
        }
        if (k + (sync ? 1u : 0u) <= 2u) {
            ticket d[2];
            uint32_t n = 0;
            for (uint32_t i = 0; i < k; ++i) d[n++] = list[i];
            if (sync) d[n++] = t[UNIT_SYNC];
            return deps{d[0], d[1]};
        }
        ticket last{};
        uint32_t i = 0;
        const uint32_t nops = k / 2u;                 // = ceil((k - 1) / 2); k >= 2 here
        for (uint32_t j = 0; j < nops; ++j) {
            const ticket a = list[i++];
            const ticket b = i < k ? list[i++] : ticket{};
            last = join2(a, b);
        }
        const ticket rest = i < k ? list[i] : ticket{};
        clear();
        add(last);
        add(rest);
        return rest.none() ? deps{last} : deps{rest, last};   // the order of the unfolded case: SYNC last
    }

    // One ticket whose completion implies every ticket of the group (a join2 when two remain).
    ticket as_ticket() {
        const deps d = as_deps();
        if (d.b.none()) return d.a;
        if (d.a.none()) return d.b;
        const ticket j = join2(d.a, d.b);
        clear();
        add(j);
        return j;
    }

    void join() {
        for (const auto & x : t) tsr::join(x);
    }
};

// -----------------------------------------------------------------------------------------------
// spm_ring: SPM scratch ring with release tickets (SW §6.7; the executor's scratch, SW §7.3)
// -----------------------------------------------------------------------------------------------
// The ring is [first, first + lines). alloc() hands out contiguous blocks that never wrap: a
// block that does not fit before the end starts again at `first` (the skipped tail stays unused
// this lap). parity 0 / 1 asks for an even / odd first line (the SPM's even / odd banks, CC §10.1;
// one padding line when needed), 2 any. Each block is recorded (in allocation order, i.e. ring
// order) until a later allocation reuses its lines; release(block, after) declares it reusable
// once `after` is done.
//
// alloc() returns the deps that must precede any write to the new block: the release tickets of
// the older blocks it overlaps (a partly covered block is shrunk and stays recorded with its
// ticket), folded per unit to the later ticket (I1); when more than two units remain, the oldest
// (the unit whose last overlapped block is the oldest) are joined in software until two remain.
// Blocks of the skipped tail are joined in software too (their lines are not reused now, so
// nothing else would carry their tickets). With must_wait = nullptr every such ticket is joined.
// When the record table is full the oldest released block is joined and forgotten. All of it
// depends only on the sequence of alloc / release calls and their tickets, never on done() (SW
// §4.7): the pushed deps are schedule-independent; joins only wait.
//
// Failure (no state change): lines = 0 or larger than the ring (with padding), parity > 2, or a
// block that would have to be reused, or forgotten, before it was released. alloc() then returns
// spm{SPM_NONE} (valid() is false) and *must_wait = {}.
//
// Cost (executor optimisation, SW §7.3 "Hot path"): O(1) amortized. A record is 4 words in a
// power-of-two table (no modulo), the fold keeps a unit bit mask and per-unit slots that are
// written before they are read (no zero-fill of 16-unit arrays), the common case (nothing
// overlapped, or only blocks released by at most two units) is inline and takes no software join
// except the table-full one; alloc_dma_first is the decode path's variant. release() searches from
// the newest record (the block just used: one probe).
//
// Tickets are only meaningful until the next qclear() (a dep below the clear point is BAD_DEP),
// so a ring is init()'d per launch (the runtime's R9 QCLEAR separates launches; V2-SW-4).
class spm_ring {
public:
    // Power of two (table index masks). A full table makes alloc join its oldest (released) block,
    // which bounds how far the executor can run ahead of the units in blocks: 128 is ~10x what the
    // U200 queue depths allow (DMA 8 entries: about 3 ops per MUL_MAT), 2 KiB of DTCM.
    static constexpr uint32_t MAX_BLOCKS = 128u;

#if defined(TSR_FW_HOST)
    // Host builds only (SW §6.8 tests): called with every ticket the ring joined in software, in
    // order, after the join (fw_hdr_host compares the sequence with W2a's algorithm). The RISC-V
    // build has no hook.
    using join_hook_fn = void (*)(const spm_ring * ring, uint32_t ticket);
    static inline join_hook_fn join_hook = nullptr;
#endif

    void init(uint32_t first_line, uint32_t lines) {
        first_ = first_line;
        end_ = first_line + lines;
        head_ = first_line;
        tail_ = 0;
        n_ = 0;
        joins_ = 0;
        slow_ = 0;
    }

    // The common case inline (no wrap, at most two units among the overlapped blocks' release
    // tickets, must_wait given): no call, no software join except the table-full one; every other
    // case (refusals included) is alloc_slow, the full algorithm, with the same result.
    TSR_HOT spm alloc(uint32_t lines, uint32_t parity, deps * must_wait) {
        const uint32_t head = head_;
        const uint32_t s = head + pad(head, parity);
        if (must_wait == nullptr || lines == 0u || parity > 2u || lines > end_ - first_ || head > end_ || s > end_ ||
            end_ - s < lines)
            return alloc_slow(lines, parity, must_wait);
        const uint32_t hi = s + lines;
        const uint32_t n = n_;
        uint32_t i = 0;
        uint32_t ta = TICKET_NONE, tb = TICKET_NONE;   // the later ticket of (at most) two units
        bool shrink = false;
        for (; i < n; ++i) {
            const rec & r = at(i);
            if (r.lo < head || r.lo >= hi) break;
            const uint32_t v = r.rel;
            if (v == REL_HELD) return alloc_slow(lines, parity, must_wait);   // refused there
            const uint32_t u = v >> 28;
            if (u != UNIT_NONE) {
                if ((ta >> 28) == UNIT_NONE || (ta >> 28) == u) {
                    if ((ta >> 28) == UNIT_NONE || !same_or_after(ticket{ta}, ticket{v})) ta = v;
                } else if ((tb >> 28) == UNIT_NONE || (tb >> 28) == u) {
                    if ((tb >> 28) == UNIT_NONE || !same_or_after(ticket{tb}, ticket{v})) tb = v;
                } else {
                    return alloc_slow(lines, parity, must_wait);   // a third unit: the age rule
                }
            }
            if (r.hi > hi) {                           // partly covered: shrink, keep its ticket
                shrink = true;
                break;
            }
        }
        const bool drop = n - i >= MAX_BLOCKS;         // table full: forget the oldest left
        if (drop && at(i).rel == REL_HELD) return alloc_slow(lines, parity, must_wait);
        // Commit (as alloc_slow: covered records go, a partly covered one shrinks, then the drop).
        tail_ = (tail_ + i) & (MAX_BLOCKS - 1u);
        uint32_t nn = n - i;
        if (shrink) at(0).lo = hi;
        if (drop) {
            const uint32_t v = at(0).rel;
            if ((v >> 28) != UNIT_NONE) ring_join(v);
            tail_ = (tail_ + 1u) & (MAX_BLOCKS - 1u);
            --nn;
        }
        if ((tb >> 28) != UNIT_NONE && (tb >> 28) < (ta >> 28)) {   // lowest unit first
            const uint32_t x = ta;
            ta = tb;
            tb = x;
        }
        *must_wait = deps{ticket{ta}, ticket{tb}};
        rec & r = at(nn);
        r.lo = head;
        r.hi = hi;
        r.start = s;
        r.rel = REL_HELD;
        n_ = nn + 1u;
        head_ = hi;
        return spm{s};
    }

    // alloc(lines, 2, &w) for a block whose first write is a DMA op (LD), in the case where w would
    // hold DMA tickets only, which the DMA queue order implies for that op: the same block and
    // records as alloc, nothing to wait for. In every other case (a wrap, a block released with a
    // ticket of another unit, a held block in the way, a refusal) it changes nothing and returns
    // spm{SPM_NONE}: the caller then uses alloc(). A leaf (no call): the executor's decode path.
    // *rel (when the block is valid) is the new record's release slot: release_at(*rel, after) is
    // release(block, after) without the search, valid until the next alloc.
    TSR_HOT spm alloc_dma_first(uint32_t lines, uint32_t ** rel) {
        const uint32_t s = head_;
        if (lines == 0u || s > end_ || end_ - s < lines) return spm{SPM_NONE};
        const uint32_t hi = s + lines;
        const uint32_t n = n_;
        uint32_t i = 0;
        bool shrink = false;
        for (; i < n; ++i) {
            const rec & r = at(i);
            if (r.lo < s || r.lo >= hi) break;
            const uint32_t u = r.rel >> 28;            // REL_HELD (unit 14) included
            if (u != UNIT_DMA && u != UNIT_NONE) return spm{SPM_NONE};
            if (r.hi > hi) {
                shrink = true;
                break;
            }
        }
        const bool drop = n - i >= MAX_BLOCKS;
        if (drop && at(i).rel == REL_HELD) return spm{SPM_NONE};
        tail_ = (tail_ + i) & (MAX_BLOCKS - 1u);
        uint32_t nn = n - i;
        if (shrink) at(0).lo = hi;
        if (drop) {
            const uint32_t v = at(0).rel;
            if ((v >> 28) != UNIT_NONE) ring_join(v);
            tail_ = (tail_ + 1u) & (MAX_BLOCKS - 1u);
            --nn;
        }
        rec & r = at(nn);
        r.lo = s;
        r.hi = hi;
        r.start = s;
        r.rel = REL_HELD;
        *rel = &r.rel;
        n_ = nn + 1u;
        head_ = hi;
        return spm{s};
    }
    TSR_HOT static void release_at(uint32_t * rel, ticket after) { *rel = after.v; }

    // The full algorithm (wraps, refusals, more than two units, must_wait = nullptr).
    [[gnu::noinline]] spm alloc_slow(uint32_t lines, uint32_t parity, deps * must_wait) {
        ++slow_;
        if (must_wait != nullptr) *must_wait = deps{};
        if (lines == 0u || parity > 2u || lines > end_ - first_) return spm{SPM_NONE};
        const uint32_t head = head_;
        uint32_t lo = head;
        uint32_t s = head + pad(head, parity);
        bool wrap = false;
        if (head > end_ || s > end_ || end_ - s < lines) {
            wrap = true;
            lo = first_;
            s = first_ + pad(first_, parity);
            if (s > end_ || end_ - s < lines) return spm{SPM_NONE};
        }
        const uint32_t hi = s + lines;                 // the new record owns [lo, hi) (padding included)
        // Plan without changing anything: i walks from the oldest record.
        const uint32_t n = n_;
        uint32_t i = 0;
        if (wrap) {                                    // records of the previous lap in the skipped tail
            while (i < n && at(i).lo >= head) {
                if (at(i).rel == REL_HELD) return spm{SPM_NONE};
                ++i;
            }
        }
        const uint32_t n_skip = i;
        bool shrink = false;
        while (i < n && at(i).lo >= lo && at(i).lo < hi) {
            if (at(i).rel == REL_HELD) return spm{SPM_NONE};
            if (at(i).hi > hi) {                       // partly covered: shrink, keep its ticket
                shrink = true;
                break;
            }
            ++i;
        }
        const uint32_t n_gone = i;                     // skipped + fully covered records
        const bool drop = n - n_gone >= MAX_BLOCKS;    // table full: forget the oldest left
        if (drop && at(n_gone).rel == REL_HELD) return spm{SPM_NONE};
        // Commit.
        for (uint32_t j = 0; j < n_skip; ++j) sw_join(at(j).rel);
        uint32_t mask = 0;                             // units with a ticket to wait for
        uint32_t tk[16], age[16];                      // per unit: the later ticket, its last record
        const uint32_t n_fold = shrink ? i + 1u : i;
        for (uint32_t j = n_skip; j < n_fold; ++j) {
            const uint32_t v = at(j).rel;
            const uint32_t u = v >> 28;
            if (u == UNIT_NONE) continue;
            const uint32_t bit = 1u << u;
            if (!(mask & bit) || !same_or_after(ticket{tk[u]}, ticket{v})) tk[u] = v;
            age[u] = j;                                // records come oldest first
            mask |= bit;
        }
        tail_ = (tail_ + n_gone) & (MAX_BLOCKS - 1u);
        n_ = n - n_gone;
        if (shrink) at(0).lo = hi;
        if (drop) {
            sw_join(at(0).rel);
            tail_ = (tail_ + 1u) & (MAX_BLOCKS - 1u);
            --n_;
        }
        deps d{};
        if (mask != 0u) d = take(mask, tk, age, must_wait == nullptr);
        if (must_wait != nullptr) *must_wait = d;
        rec & r = at(n_);
        r.lo = lo;
        r.hi = hi;
        r.start = s;
        r.rel = REL_HELD;
        ++n_;
        head_ = hi;
        return spm{s};
    }

    // The block (as returned by alloc) becomes reusable once `after` is done (none: at once).
    // Releasing a block the ring does not hold, or twice, is a firmware bug: panic.
    void release(spm block, ticket after) {
        for (uint32_t j = n_; j-- > 0u;) {
            rec & r = at(j);
            if (r.start == block.line && r.rel == REL_HELD) {
                r.rel = after.v;
                return;
            }
        }
        release_bad(block);
    }

    uint32_t first() const { return first_; }
    uint32_t lines() const { return end_ - first_; }
    uint32_t head() const { return head_; }          // where the next block starts (before padding)
    uint32_t blocks() const { return n_; }           // recorded blocks
    uint32_t joins() const { return joins_; }        // tickets joined in software so far
    uint32_t slow_allocs() const { return slow_; }   // allocations by the full algorithm (alloc_slow)

private:
    // A record: lines owned [lo, hi) (padding included), the block's first line `start` (alloc's
    // return value, the key of release()), and its release ticket; REL_HELD (unit 14, which no
    // ticket carries: CC §7.4 marker records) until release(). Trivial, so that a static ring is
    // zero-initialised (.bss).
    static constexpr uint32_t REL_HELD = 0xE000'0000u;
    struct rec {
        uint32_t lo, hi, start, rel;
    };

    // The per-unit tickets of an allocation folded to <= 2 deps (lowest unit first); while more
    // than two units remain (or all of them, join_all_of_them), the unit whose last overlapped
    // block is the oldest is joined in software.
    [[gnu::noinline]] deps take(uint32_t mask, const uint32_t * tk, const uint32_t * age, bool join_all_of_them) {
        for (;;) {
            const uint32_t m2 = mask & (mask - 1u);    // mask without its lowest unit
            if (mask == 0u || (!join_all_of_them && (m2 & (m2 - 1u)) == 0u)) break;
            uint32_t oldest = 16u;
            for (uint32_t u = 0, m = mask; m != 0u; ++u, m >>= 1)
                if ((m & 1u) && (oldest == 16u || age[u] < age[oldest])) oldest = u;
            sw_join(tk[oldest]);
            mask &= ~(1u << oldest);
        }
        ticket d[2];
        uint32_t k = 0;
        for (uint32_t u = 0, m = mask; m != 0u; ++u, m >>= 1)
            if (m & 1u) d[k++] = ticket{tk[u]};
        return deps{d[0], d[1]};
    }

    [[noreturn, gnu::noinline, gnu::cold]] void release_bad(spm block) { panic(PANIC_RING_RELEASE, block.line, n_, head_); }
    static uint32_t pad(uint32_t line, uint32_t parity) { return (parity < 2u && (line & 1u) != parity) ? 1u : 0u; }
    rec & at(uint32_t i) { return ring_[(tail_ + i) & (MAX_BLOCKS - 1u)]; }
    // Every software join of the ring (counted; a host build reports the joined ticket to join_hook
    // once the join returned).
    TSR_HOT void ring_join(uint32_t t) {
        ++joins_;
        join(ticket{t});
#if defined(TSR_FW_HOST)
        if (join_hook != nullptr) join_hook(this, t);
#endif
    }
    void sw_join(uint32_t t) {
        if ((t >> 28) == UNIT_NONE) return;
        ring_join(t);
    }

    uint32_t first_ = 0, end_ = 0, head_ = 0;
    uint32_t tail_ = 0, n_ = 0;
    uint32_t joins_ = 0, slow_ = 0;
    rec ring_[MAX_BLOCKS] = {};
};

}  // namespace tsr

#endif  // TSR_KERNEL_HPP

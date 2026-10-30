// runtime.cpp — the launch dispatcher (docs/v2/01-execution-model.md "EX" §4.7 R1-R10, normative;
// docs/v2/06-software-verification.md "SW" §6.3-§6.5; EX §4.3-§4.5, §4.9; docs/v2/05-debug.md
// "DB" §3.8, §8.2, §8.4).
//
//   R1  Boot (after crt0.S): CFG0 / CFG1 through the core window, kernel table check, qclear() and a
//       join on it, RT_STATE := IDLE, HELLO as an event (r1 TAKEN, r2 / r3 image id from .tsr_id,
//       r4 SDK ABI | runtime version << 16, r5 kernel count, r7 boot core cycles: MTIME from
//       crt0's entry to the post). Descriptors pushed before HELLO stay queued.
//   R2  Take: WFI on irq_fast[2] (LQ_AVAIL != 0 and not ACTIVE); copy LQ_HEAD_W0..7 and
//       LQ_HEAD_SEQ into DTCM (tsr_rt_launch); LQ_POP; t0 = MTIME (DURATION).
//   R4  Validate (before R3: EX §4.4 precedence BAD_DESC > ABORTED, see below): unknown TYPE,
//       reserved bits w0[15:9], PING with KERNEL != 0, KERNEL beyond the table, or an argument-block
//       handle with ch >= NUM_CH (CFG0) or beat + argblk_lines > 2^28 -> DONE / BAD_DESC. Args
//       beyond the kernel's nargs are ignored (the kernel sees 0).
//   R3  Cancelled or stopped before start: CANCEL_VALID and LQ_CANCEL covers the sequence, or
//       STOP_VALID and LQ_STOP covers it ((s - q) mod 2^32 < 2^31, EX §4.9) -> DONE / ABORTED,
//       ERR_CODE 0, without running. PING -> PONG (r2 = ARG0).
//   R5  Run: RT_STATE := KERNEL | index; PROF -> REC_CFG.ALL (restored after R6) and
//       tsr_args.flags PROF (NODE markers, SW §7.3); the argument block: one LD of argblk_lines
//       lines from ARG0 into the top 16 SPM lines and one join; a failing LD (a block beyond the
//       channel: DRAM_RANGE) answers BAD_DESC with the LD's error in r3-r6, an ABORTED one
//       (a cancel) ABORTED; launch_pidx = ISSUE_PIDX, base_u = ISSUED_SEQ[u]; DB's launch hook
//       tsr_rt_launch_hook(kernel, seq, args) (DB §3.8); TRACE_MARK LAUNCH_BEGIN; the kernel;
//       TRACE_MARK LAUNCH_END (DB §8.4: around every kernel).
//   R6  join_all() (FENCE, then join).
//   R7  First error = the smallest ERR_PIDX (relative to the launch's first push) over the units
//       whose ERR_CODE != 0 (valid after the join: I2); PLAT_ERR from LQ_FLAGS; STATUS by EX §4.4's
//       precedence ABORTED > TILE_ERR > PLAT_ERR > STOPPED > KERNEL_ERR > OK (TRAP and BAD_DESC
//       are decided elsewhere); r3-r6 = ticket {u, ERR_SEQ}, ERR_INFO, ERR_W0, ERR_PIDX (r3 =
//       0xF000_0000 without an error); ERR_CODE = the first error's code whatever the status. STOPPED
//       = the kernel saw tsr::stop_requested() true (it honoured the soft stop; a kernel that never
//       polls runs to the end and reports its own status, EX §4.9). RET = the kernel's return value.
//   R8  WFI on irq_fast[3] while LCQ_FREE = 0; post with ANSWER = 1; r7 = DURATION (core cycles
//       from the pop to the post, saturating at 2^32 - 1).
//   R9  If a unit is in error state (ERR_CODE != 0): qclear() and a join.
//   R10 RT_STATE := IDLE; debug builds (no NDEBUG) check QSTATUS.QIDLE (a bounded wait: the RISC-V
//       copy may lag; tsr::panic(PANIC_RT_NOT_IDLE) if it never rises); back to R2.
//
// Hot loops of the launch path (head copy, arguments, base_u, LCQ staging) are unrolled: -Os keeps
// them as loops otherwise, which costs about a third of an empty launch (EX §4.11).
//
// Decisions (P2.2): R4 is checked before R3 so that a descriptor that is both invalid and
// cancelled answers BAD_DESC, as EX §4.4's precedence says (EX §4.7 lists R3 first); a cancelled
// PING answers DONE / ABORTED. Every value that decides pushed words or records is a launch input
// or a timing-free register (ISSUE_PIDX, ISSUED_SEQ, ERR_* after the join, the watermarks the host
// wrote); DONE_SEQ, LQ_AVAIL, LCQ_FREE and MTIME only end waits or feed r7 (SW §4.7).
#include "rt_int.hpp"
#include "tsr/kernel.hpp"
#include "tsr/log.hpp"

using namespace tsr;
using tsr::rt::cw;

extern "C" {
extern const tsr_kernel_desc __tsr_kernels_start[];
extern const tsr_kernel_desc __tsr_kernels_end[];

// .tsr_id: 8 bytes, zero in the linker output; tsr-elf writes the image id (EX §3.2). Read through
// volatile: the value in DTCM is the stamped one, not the compiler's 0.
const volatile uint32_t __tsr_image_id[2] __attribute__((section(".tsr_id"), used, aligned(8))) = { 0u, 0u };

[[noreturn]] void tsr_rt_main(uint32_t boot_mtime);
void tsr_rt_launch_hook(uint32_t kernel, uint32_t seq, const tsr_args * args);
}

// The running launch (tsr::launch(), the trap handler, the debugger); C name tsr_rt_launch.
launch_info tsr::tsr_rt_launch;

namespace {

uint32_t g_n_kernels;           // entries of .tsr_kernels
uint32_t g_spm_lines;           // CFG1
uint32_t g_num_ch;              // CFG0.NUM_CH

constexpr uint32_t SEQ_MASK = 0x0FFFFFFFu;

struct first_err {
    uint32_t code, ticket, info, w0, pidx;
};

// R7: the error of the smallest push index (relative to pidx0) over the units in error state.
bool find_first_error(uint32_t pidx0, first_err & e) {
    bool found = false;
    uint32_t best = 0;
    for (uint32_t u = 0; u < NUM_UNITS; ++u) {
        const uint32_t code = cw(CW_ERR_CODE(u)) & 0xFFu;
        if (code == 0u) continue;
        const uint32_t pidx = cw(CW_ERR_PIDX(u));
        const uint32_t rel = pidx - pidx0;
        if (found && rel >= best) continue;
        found = true;
        best = rel;
        e.code = code;
        e.ticket = (u << 28) | (cw(CW_ERR_SEQ(u)) & SEQ_MASK);
        e.info = cw(CW_ERR_INFO(u));
        e.w0 = cw(CW_ERR_W0(u));
        e.pidx = pidx;
    }
    return found;
}

// R9: a unit is in error state. After R6 the only way an error can appear without R7 having seen
// one is a QABORT (a cancel landing after R7), and a QABORT puts every built unit, the SYNC unit
// included, into error state (CC §9.5): one ERR_CODE read decides when R7 found nothing.
bool any_unit_in_error(bool r7_found) { return r7_found || (cw(CW_ERR_CODE(UNIT_SYNC)) & 0xFFu) != 0u; }

inline bool covered(uint32_t watermark, uint32_t q) { return (watermark - q) < SEQ_HALF; }

inline uint32_t launch_mark(uint32_t cls, uint32_t kernel, uint32_t seq) {
    return (cls << MARK_CLASS_LSB) | ((kernel << MARK_LAUNCH_KERNEL_LSB) & MARK_LAUNCH_KERNEL) |
           (seq & MARK_LAUNCH_SEQ);
}

inline uint32_t r0_of(uint32_t type, uint32_t status, uint32_t err_code, uint32_t seq) {
    return type | (status << LCQ_R0_STATUS_LSB) | ((err_code & 0xFFu) << LCQ_R0_ERR_CODE_LSB) |
           ((seq & 0xFFu) << LCQ_R0_SEQ_LSB);
}

void qclear_join() { join(qclear()); }

// R3-R7 for the popped descriptor; fills r0..r6 (r1 already holds SEQ, r3 TICKET_NONE, the rest 0).
// Returns true if R7 (or the staging LD) found a unit in error state.
bool dispatch(uint32_t r[8]) {
    launch_info & L = tsr_rt_launch;
    const uint32_t w0 = L.desc[0];
    const uint32_t type = w0 & DESC_TYPE, kernel = w0 >> DESC_KERNEL_LSB;
    const tsr_kernel_desc * kd = nullptr;

    // R4: validation (BAD_DESC before ABORTED, EX §4.4).
    bool bad = (type != DESC_TYPE_LAUNCH && type != DESC_TYPE_PING) || (w0 & 0x0000FE00u) != 0u ||
               (type == DESC_TYPE_PING && kernel != 0u) || (type == DESC_TYPE_LAUNCH && kernel >= g_n_kernels);
    if (!bad && type == DESC_TYPE_LAUNCH) {
        kd = &__tsr_kernels_start[kernel];
        if (kd->flags & TSR_KFLAG_ARGBLK) {
            const uint32_t h = L.desc[1], ch = h >> 28, beat = h & SEQ_MASK;
            bad = ch >= g_num_ch || beat + kd->argblk_lines > 0x10000000u;
        }
    }
    if (bad) {
        TSR_LOG_SRC(TSR_LOG_WARN, LOG_HDR_SOURCE_RUNTIME, "BAD_DESC seq %u w0 0x%08x (%u kernels)", L.seq, w0,
                    g_n_kernels);
        r[0] = r0_of(LCQ_R0_TYPE_DONE, LCQ_R0_STATUS_BAD_DESC, 0u, L.seq);
        return false;
    }

    // R3: cancelled or stopped before start.
    const uint32_t fl = cw(CW_LQ_FLAGS);
    if (((fl & LQ_FLAGS_CANCEL_VALID) && covered(cw(CW_LQ_CANCEL), L.seq)) ||
        ((fl & LQ_FLAGS_STOP_VALID) && covered(cw(CW_LQ_STOP), L.seq))) {
        r[0] = r0_of(LCQ_R0_TYPE_DONE, LCQ_R0_STATUS_ABORTED, 0u, L.seq);
        return false;
    }

    if (type == DESC_TYPE_PING) {                        // R4: PING -> PONG
        r[0] = r0_of(LCQ_R0_TYPE_PONG, 0u, 0u, L.seq);
        r[2] = L.desc[1];
        r[3] = 0u;                                       // EX §4.4: PONG r3-r6 = 0 (fix W2a)
        return false;
    }

    // R5
    rt::set_state(RT_STATE_PHASE_KERNEL, kernel);
    L.kernel = kernel;
    L.flags = (w0 & DESC_PROF) ? TSR_ARGS_FLAG_PROF : 0u;
    L.stop_seen = 0;
    const uint32_t rec_cfg = (L.flags & TSR_ARGS_FLAG_PROF) ? cw(CW_REC_CFG) : 0u;
    if (L.flags & TSR_ARGS_FLAG_PROF) cw(CW_REC_CFG) = rec_cfg | REC_CFG_ALL;

    tsr_args args;
#pragma GCC unroll 7
    for (uint32_t i = 0; i < 7u; ++i) args.a[i] = i < kd->nargs ? L.desc[1 + i] : 0u;
    args.blk = nullptr;
    args.seq = L.seq;
    args.flags = L.flags;

    const uint32_t pidx_base = cw(CW_ISSUE_PIDX);        // errors of this launch have pidx >= this
    first_err e = { 0u, TICKET_NONE, 0u, 0u, 0u };
    if (kd->flags & TSR_KFLAG_ARGBLK) {
        const uint32_t line = g_spm_lines - TSR_ARG_LINES;
        const ticket t = issue(LD(spm{ line }, from_handle(L.desc[1]), kd->argblk_lines));
        join(t);
        if (find_first_error(pidx_base, e)) {
            const uint32_t status = e.code == ERR_ABORTED ? LCQ_R0_STATUS_ABORTED : LCQ_R0_STATUS_BAD_DESC;
            if (status == LCQ_R0_STATUS_BAD_DESC)
                TSR_LOG_SRC(TSR_LOG_WARN, LOG_HDR_SOURCE_RUNTIME, "BAD_DESC seq %u: argument block 0x%08x error %u",
                            L.seq, L.desc[1], e.code);
            if (L.flags & TSR_ARGS_FLAG_PROF) cw(CW_REC_CFG) = rec_cfg;
            r[0] = r0_of(LCQ_R0_TYPE_DONE, status, e.code, L.seq);
            r[3] = e.ticket;
            r[4] = e.info;
            r[5] = e.w0;
            r[6] = e.pidx;
            return true;
        }
        args.blk = reinterpret_cast<const volatile uint8_t *>(uintptr_t(RV_SPM_BASE + 64u * line));
    }
    L.pidx0 = cw(CW_ISSUE_PIDX);
#pragma GCC unroll 6
    for (uint32_t u = 0; u < NUM_UNITS; ++u) L.base[u] = cw(CW_ISSUED_SEQ(u)) & SEQ_MASK;

    tsr_rt_launch_hook(kernel, L.seq, &args);
    trace_mark(launch_mark(MARK_CLASS_LAUNCH_BEGIN, kernel, L.seq));
    const int32_t ret = kd->entry(&args);
    trace_mark(launch_mark(MARK_CLASS_LAUNCH_END, kernel, L.seq));

    // R6
    rt::set_state(RT_STATE_PHASE_JOIN, kernel);
    join_all();
    if (L.flags & TSR_ARGS_FLAG_PROF) cw(CW_REC_CFG) = rec_cfg;

    // R7
    uint32_t status = LCQ_R0_STATUS_OK;
    const bool found = find_first_error(pidx_base, e);
    if (found) {
        status = e.code == ERR_ABORTED ? LCQ_R0_STATUS_ABORTED : LCQ_R0_STATUS_TILE_ERR;
        r[3] = e.ticket;
        r[4] = e.info;
        r[5] = e.w0;
        r[6] = e.pidx;
    } else if (cw(CW_LQ_FLAGS) & LQ_FLAGS_PLAT_ERR) {
        status = LCQ_R0_STATUS_PLAT_ERR;
    } else if (L.stop_seen) {
        status = LCQ_R0_STATUS_STOPPED;
    } else if (ret < 0) {
        status = LCQ_R0_STATUS_KERNEL_ERR;
    }
    r[0] = r0_of(LCQ_R0_TYPE_DONE, status, e.code, L.seq);
    r[2] = uint32_t(ret);
    return found;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// LCQ posts (EX §4.5) and kernel events (EX §4.4)
// ---------------------------------------------------------------------------------------------
void tsr::rt::post(const uint32_t r[8], uint32_t answer) {
    if (cw(CW_LCQ_FREE) == 0u) {                         // R8: irq_fast[3] while LCQ_FREE != 0
        const uint32_t old = mie_swap(mie_fast(IRQ_FAST_LCQ));
        while (cw(CW_LCQ_FREE) == 0u) wfi();
        mie_swap(old);
    }
#pragma GCC unroll 8
    for (uint32_t k = 0; k < 8u; ++k) cw(CW_LCQ_W(k)) = r[k];
    cw(CW_LCQ_POST) = answer ? LCQ_POST_ANSWER : 0u;
}

void tsr::post_event(uint32_t type, uint32_t info, const uint32_t d[6]) {
    if (type < LCQ_R0_TYPE_EVENT_FIRST || type > LCQ_R0_TYPE_EVENT_LAST) panic(PANIC_RT_EVENT_TYPE, type);
    const uint32_t seq = tsr_rt_launch.seq;
    const uint32_t r[8] = { type | ((info & 0xFFFFu) << 8) | ((seq & 0xFFu) << LCQ_R0_SEQ_LSB), seq,
                            d[0], d[1], d[2], d[3], d[4], d[5] };
    rt::post(r, 0u);
}

bool tsr::rt_stop_requested() {
    if (!(cw(CW_LQ_FLAGS) & LQ_FLAGS_STOP_VALID)) return false;
    if (!covered(cw(CW_LQ_STOP), tsr_rt_launch.seq)) return false;
    tsr_rt_launch.stop_seen = 1;
    return true;
}

uint32_t tsr::spm_lines() { return g_spm_lines; }

// ---------------------------------------------------------------------------------------------
// DB §3.8 halt-on-launch: ebreak at __tsr_launch_halt when a debugger is attached and
// launch_break is 0xFFFF_FFFF (all) or kernel + 1. The hart stops with a0 = kernel index,
// a1 = SEQ, a2 = the tsr_args pointer (its a[0..6] are the arguments). Not inlined or cloned: the
// label exists once. Cost without a debugger: two DTCM loads and a branch.
// ---------------------------------------------------------------------------------------------
extern "C" __attribute__((noinline, noclone, used)) void tsr_rt_launch_hook(uint32_t kernel, uint32_t seq,
                                                                             const tsr_args * args) {
    if (__tsr_dbg[DBG_INFO_ATTACHED_WORD] == 0u) return;
    const uint32_t lb = __tsr_dbg[DBG_INFO_LAUNCH_BREAK_WORD];
    if (lb != 0xFFFFFFFFu && lb != kernel + 1u) return;
    register uint32_t a0 __asm__("a0") = kernel;
    register uint32_t a1 __asm__("a1") = seq;
    register const tsr_args * a2 __asm__("a2") = args;
    __asm__ volatile(".option push\n\t.option norvc\n\t.globl __tsr_launch_halt\n__tsr_launch_halt:\n\tebreak\n\t"
                     ".option pop"
                     :
                     : "r"(a0), "r"(a1), "r"(a2)
                     : "memory");
}

// ---------------------------------------------------------------------------------------------
// R1-R10
// ---------------------------------------------------------------------------------------------
extern "C" [[noreturn]] void tsr_rt_main(uint32_t boot_mtime) {
    launch_info & L = tsr_rt_launch;
    // R1
    rt::set_state(RT_STATE_PHASE_BOOT);
    g_num_ch = rt::core(CSR_CFG0) & 0xFFu;
    g_spm_lines = rt::core(CSR_CFG1);
    g_n_kernels = uint32_t(__tsr_kernels_end - __tsr_kernels_start);
    for (uint32_t k = 0; k < g_n_kernels; ++k) {
        const tsr_kernel_desc & d = __tsr_kernels_start[k];
        const bool blk = d.flags & TSR_KFLAG_ARGBLK;
        if (d.entry == nullptr || d.nargs > 7u || (d.flags & ~TSR_KFLAG_ARGBLK) ||
            (blk ? (d.argblk_lines < 1u || d.argblk_lines > TSR_ARG_LINES) : d.argblk_lines != 0u))
            panic(PANIC_RT_TABLE, k, d.nargs | uint32_t(d.flags) << 8 | uint32_t(d.argblk_lines) << 16);
    }
    qclear_join();
    rt::set_state(RT_STATE_PHASE_IDLE);
    {
        const uint32_t hello[8] = { r0_of(LCQ_R0_TYPE_HELLO, 0u, 0u, 0u), cw(CW_LQ_HEAD_SEQ), __tsr_image_id[0],
                                    __tsr_image_id[1], TSR_SDK_ABI | (TSR_RT_VERSION << HELLO_R4_RT_VERSION_LSB),
                                    g_n_kernels, 0u, cw(CW_MTIME_LO) - boot_mtime };
        rt::post(hello, 0u);
    }

    for (;;) {
        // R2
        if (cw(CW_LQ_AVAIL) == 0u) {
            const uint32_t old = rt::mie_swap(rt::mie_fast(IRQ_FAST_LQ));
            while (cw(CW_LQ_AVAIL) == 0u) rt::wfi();
            rt::mie_swap(old);
        }
#pragma GCC unroll 8
        for (uint32_t k = 0; k < 8u; ++k) L.desc[k] = cw(CW_LQ_HEAD_W(k));
        L.seq = cw(CW_LQ_HEAD_SEQ);
        L.kernel = 0;
        L.flags = 0;
        L.stop_seen = 0;
        cw(CW_LQ_POP) = 1u;
        const uint64_t t0 = rt::mtime64();
        L.active = 1;
        rt::set_state(RT_STATE_PHASE_DISPATCH, L.desc[0] >> DESC_KERNEL_LSB);

        uint32_t r[8];                                   // filled word by word (no memset call)
        r[1] = L.seq;
        r[2] = 0u;
        r[3] = TICKET_NONE;
        r[4] = r[5] = r[6] = 0u;
        const bool err = dispatch(r);

        // R8
        rt::set_state(RT_STATE_PHASE_ANSWER, L.kernel);
        const uint64_t dt = rt::mtime64() - t0;
        r[7] = dt > 0xFFFFFFFFull ? 0xFFFFFFFFu : uint32_t(dt);
        rt::post(r, 1u);
        L.active = 0;

        // R9
        if (any_unit_in_error(err)) qclear_join();

        // R10
        rt::set_state(RT_STATE_PHASE_IDLE);
#ifndef NDEBUG
        {
            uint32_t n = 0;
            while (!(cw(CW_QSTATUS) & QSTATUS_QIDLE))
                if (++n == 100000u) panic(PANIC_RT_NOT_IDLE, cw(CW_QSTATUS));
        }
#endif
    }
}

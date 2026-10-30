// rt_kernels.cpp — test kernels of the fw_runtime image (sw/ctrl/tests/runtime.cmake, P2.2;
// docs/design/06-software-verification.md "SW" §6.3-§6.6, §10.2). Each exercises one runtime step of
// EX §4.7 (docs/design/01-execution-model.md) or one SDK service; sw/ctrl/tests/test_fw_runtime.c
// launches them on the simulator and checks every record, log word and output. They obey the
// launch-stateless and timing rules (SW §4.6-§4.7): nothing read from a timing-dependent register
// reaches a pushed word, a dependency or a record (the poll loops only end waits).
//
// Kernel events (EX §4.4, types 0xC0-0xFF; r0[23:8] = info):
//   0xC1 ARGS     rt_args<k>: info = k | blk != nullptr << 8 | flags << 9 | (seq == launch().seq) << 12,
//                 r2..r7 = a[0..5] (RET = a[6][15:0] | k << 16)
//   0xC2 BLK      rt_blk1 / rt_blk16: info = argblk_lines, r2 = blk address, r3 = FNV-1a 32 of the
//                 block's bytes, r4 / r5 = its first / last word, r6 / r7 = a[1] / a[2]
//   0xC3 STARTED  rt_stop / rt_cancel: the host may now write the watermark; rt_cancel: r2 = the
//                 first push index, r3 / r4 = the LD / ST tickets
//   0xC4 TILE     rt_tile_err: r2 = the first push index, r3 / r4 = the LD / ST tickets
//   0xC5 PROF     rt_prof: r2 = REC_CFG, r3 = args flags, r4 = (pidx0 == ISSUE_PIDX at entry),
//                 r5 = (base[u] == ISSUED_SEQ[u] at entry for every unit)
#include "tsr/kernel.hpp"
#include "tsr/log.hpp"
#include "tsr/rt.hpp"

using namespace tsr;

namespace {

constexpr uint32_t EV_ARGS = 0xC1u, EV_BLK = 0xC2u, EV_STARTED = 0xC3u, EV_TILE = 0xC4u, EV_PROF = 0xC5u;

int32_t args_common(const tsr_args * a, uint32_t k) {
    const uint32_t info = k | (a->blk != nullptr ? 0x100u : 0u) | (a->flags << 9) |
                          (a->seq == launch().seq ? 0x1000u : 0u);
    event(EV_ARGS, info, a->a[0], a->a[1], a->a[2], a->a[3], a->a[4], a->a[5]);
    return int32_t((a->a[6] & 0xFFFFu) | (k << 16));
}

int32_t blk_common(const tsr_args * a, uint32_t lines) {
    const volatile uint8_t * b = a->blk;
    if (b == nullptr) return -1;
#ifdef RT_MUTANT_STALE
    // fw_runtime_stale (runtime.cmake): reads the whole argument range, 15 lines more than the
    // runtime staged for rt_blk1; the race detector must report SPM_STALE (SW §4.6, SW-4).
    const uint32_t rd_lines = 16u;
#else
    const uint32_t rd_lines = lines;
#endif
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < 64u * rd_lines; ++i) h = (h ^ b[i]) * 16777619u;
    const volatile uint32_t * w = reinterpret_cast<const volatile uint32_t *>(b);
    event(EV_BLK, lines, uint32_t(uintptr_t(b)), h, w[0], w[16u * lines - 1u], a->a[1], a->a[2]);
    return 0;
}

const char k_rodata_str[] = "rt-rodata-string";

}  // namespace

// ---- 0..7 arguments (EX §4.3: args beyond nargs are ignored: the kernel sees 0) ----
#define RT_ARGS_KERNEL(k)                                                                         \
    TSR_KERNEL(rt_args##k, k, 0, 0);                                                              \
    extern "C" int32_t rt_args##k(const tsr_args * a) { return args_common(a, k); }
RT_ARGS_KERNEL(0)
RT_ARGS_KERNEL(1)
RT_ARGS_KERNEL(2)
RT_ARGS_KERNEL(3)
RT_ARGS_KERNEL(4)
RT_ARGS_KERNEL(5)
RT_ARGS_KERNEL(6)
RT_ARGS_KERNEL(7)

// ---- argument blocks (SW §6.2 ARGBLK, R5): 1 and 16 lines ----
TSR_KERNEL(rt_blk1, 3, TSR_KFLAG_ARGBLK, 1);
extern "C" int32_t rt_blk1(const tsr_args * a) { return blk_common(a, 1); }
TSR_KERNEL(rt_blk16, 7, TSR_KFLAG_ARGBLK, 16);
extern "C" int32_t rt_blk16(const tsr_args * a) { return blk_common(a, 16); }

// ---- the empty kernel (launch cost, the debugger's launch break) ----
TSR_KERNEL(rt_nop, 0, 0, 0);
extern "C" int32_t rt_nop(const tsr_args *) { return 0; }

// ---- RET (KERNEL_ERR when negative) ----
TSR_KERNEL(rt_ret, 1, 0, 0);
extern "C" int32_t rt_ret(const tsr_args * a) { return int32_t(a->a[0]); }

// ---- a0 src handle, a1 dst handle, a2 lines, a3 SPM line: LD then ST ----
TSR_KERNEL(rt_copy, 4, 0, 0);
extern "C" int32_t rt_copy(const tsr_args * a) {
    const ticket t = issue(LD(spm{a->a[3]}, from_handle(a->a[0]), a->a[2]));
    issue(ST(spm{a->a[3]}, from_handle(a->a[1]), a->a[2]), after(t));
    return 0;
}

// ---- a tile error (R7 TILE_ERR, R9 clear): rt_copy with a source beyond the channel; RET < 0
// (TILE_ERR has precedence over KERNEL_ERR, EX §4.4) ----
TSR_KERNEL(rt_tile_err, 4, 0, 0);
extern "C" int32_t rt_tile_err(const tsr_args * a) {
    const uint32_t p0 = pidx_now();
    const ticket t1 = issue(LD(spm{a->a[3]}, from_handle(a->a[0]), a->a[2]));
    const ticket t2 = issue(ST(spm{a->a[3]}, from_handle(a->a[1]), a->a[2]), after(t1));
    event(EV_TILE, 0, p0, t1.v, t2.v);
    return -9;
}

// ---- soft stop during a long kernel (EX §4.9): polls stop_requested() at most a0 times, pushes
// nothing; RET = a1 when stopped (STOPPED has precedence over KERNEL_ERR, EX §4.4), else -2 ----
TSR_KERNEL(rt_stop, 2, 0, 0);
extern "C" int32_t rt_stop(const tsr_args * a) {
    event(EV_STARTED, 1);
    for (uint32_t i = 0; i < a->a[0]; ++i)
        if (stop_requested()) return int32_t(a->a[1]);
    return -2;
}

// ---- hard cancel during a long kernel (EX §4.9): the RISC-V hold keeps its ops queued until the
// host's LQ_CANCEL aborts them (they then complete ABORTED in order, CC §9.6) ----
TSR_KERNEL(rt_cancel, 4, 0, 0);
extern "C" int32_t rt_cancel(const tsr_args * a) {
    qhold(true);
    const uint32_t p0 = pidx_now();
    const ticket t1 = issue(LD(spm{a->a[3]}, from_handle(a->a[0]), a->a[2]));
    const ticket t2 = issue(ST(spm{a->a[3]}, from_handle(a->a[1]), a->a[2]), after(t1));
    event(EV_STARTED, 2, p0, t1.v, t2.v);
    join(t2);
    qhold(false);
    return 0;
}

// ---- PROF (R5: REC_CFG.ALL for the launch, tsr_args flags) ----
TSR_KERNEL(rt_prof, 0, 0, 0);
extern "C" int32_t rt_prof(const tsr_args * a) {
    uint32_t base_ok = 1;
    for (uint32_t u = 0; u < NUM_UNITS; ++u) base_ok &= launch().base[u] == issued_seq(u) ? 1u : 0u;
    event(EV_PROF, 0, hw::cw_read(CW_REC_CFG), a->flags, launch().pidx0 == pidx_now() ? 1u : 0u, base_ok);
    return 0;
}

// ---- logging (SW §6.6): a0 mode (bit0: a1 FMT records "rt_log %u of %u: 0x%x"; bit1: one record
// of every kind and the level filters), a2 a value ----
TSR_KERNEL(rt_log, 3, 0, 0);
extern "C" int32_t rt_log(const tsr_args * a) {
    const uint32_t mode = a->a[0], n = a->a[1], v = a->a[2];
    if (mode & 1u)
        for (uint32_t i = 0; i < n; ++i) TSR_LOG(TSR_LOG_INFO, "rt_log %u of %u: 0x%x", i, n, v);
    if (mode & 2u) {
        TSR_LOG(TSR_LOG_INFO, "rt_log all %d %u %x %08x %f %t %llx %s", int32_t(-5), 7u, v, 0x1234u, 0x3F800000u,
                make_ticket(UNIT_DMA, 9u), uint64_t(0x1122334455667788ull), k_rodata_str);
        log_text(TSR_LOG_INFO, LOG_HDR_SOURCE_KERNEL, "rt_log text line");
        const uint32_t ev[2] = { v, 2u };
        log_event(TSR_LOG_INFO, LOG_HDR_SOURCE_KERNEL, 0x0000ABCDu, ev, 2u);
        const uint32_t bl[3] = { 1u, 2u, 3u };
        log_blob(TSR_LOG_INFO, LOG_HDR_SOURCE_KERNEL, 0x0000B10Bu, bl, 3u);
        TSR_LOG(TSR_LOG_DEBUG, "rt_log debug %u", v);        // above TSR_LOG_MAX_LEVEL in a release build
        TSR_LOG(TSR_LOG_WARN, "rt_log warn %u", v);
    }
    return int32_t(n);
}

// ---- traps (DB §6.3): a0 0 illegal instruction at rt_trap_insn, 1 tsr::panic(0xDEAD0001, 1, 2, 3),
// 2 load access fault at 0x3000_0000, 3 ebreak, 4 a kernel event of type 0x7F (the runtime
// panics: PANIC_RT_EVENT_TYPE); a1 = RT_INFO before the trap ----
TSR_KERNEL(rt_trap, 2, 0, 0);
extern "C" int32_t rt_trap(const tsr_args * a) {
    set_info(a->a[1]);
    switch (a->a[0]) {
    case 0:
        __asm__ volatile(".globl rt_trap_insn\nrt_trap_insn:\n\t.word 0" : : : "memory");
        break;
    case 1:
        panic(0xDEAD0001u, 1u, 2u, 3u);
    case 2:
        (void)*reinterpret_cast<const volatile uint32_t *>(uintptr_t(0x30000000u));
        break;
    case 4:
        event(0x7Fu, 0);
        break;
    default:
        __asm__ volatile("ebreak" : : : "memory");
        break;
    }
    return -1;
}

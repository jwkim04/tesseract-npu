// trap.cpp — the runtime debug contract and the trap handler's C side (docs/design/05-debug.md "DB"
// §3.8, §6.1, §6.3; docs/design/06-software-verification.md "SW" §6.6; docs/design/01-execution-model.md
// "EX" §4.4, §4.7). The assembly entry (register frame, nesting, the debugger path at
// __tsr_trap_halt) is in crt0.S.
//
// __tsr_dbg (.tsr_dbg, loaded, 64 B; DB §6.3): 0 magic "TDBG", 1 version 1, 2 build id (the image
// id [31:0], EX §3.2: what the GDB server compares with its ELF), 3 attached (written by the
// debugger), 4 launch_break (0 off, 0xFFFF_FFFF all, else kernel index + 1), 5 trap_count,
// 6-15 reserved. It is a loaded section: the host's image load gives attached = launch_break = 0,
// and crt0 keeps what a debugger wrote while the hart was halted at boot (DB §3.8 halt-on-boot);
// crt0 sets only magic, version and build id, and zeroes trap_count.
//
// __tsr_crash (.tsr_crash, not loaded, zeroed by crt0, 192 B; DB §6.3): 0 magic "TRAP" (written by
// the trap handler: a record without it holds no crash), 1 version 1, 2 mcause, 3 mepc, 4 mtval,
// 5 mstatus, 6-7 MTIME lo / hi, 8 launch SEQ, 9 kernel index (0xFFFF_FFFF outside a kernel),
// 10 RT_STATE, 11 RT_INFO, 12-42 x1..x31, 43 nested, 44-47 reserved.
//
// tsr_rt_trap (no debugger attached; DB §6.3 steps 3-4):
//   fill __tsr_crash; push the TRAP log record (LEVEL FATAL, SOURCE 2 trap handler, TYPE TRAP,
//   payload MTIME + words 0..43 of __tsr_crash; never waits for log space, even with
//   LOG_CFG.BLOCK); post EX's TRAP record (type 0x81, STATUS 4: r1 SEQ of the running launch when
//   ACTIVE, else TAKEN; r2 mcause, r3 mepc, r4 mtval, r5 RT_STATE at the trap, r6 RT_INFO,
//   r7 MTIME[31:0]) with ANSWER = 1 if ACTIVE (LQ_FLAGS.ACTIVE), waiting for LCQ_FREE like R8;
//   RT_STATE := TRAPPED (kernel index kept in [31:16]); WFI loop with mie = 0.
#include "rt_int.hpp"
#include "tsr/log.hpp"

using namespace tsr;
using tsr::rt::cw;

extern "C" {
volatile uint32_t __tsr_dbg[DBG_INFO_NWORDS] __attribute__((section(".tsr_dbg"), used, aligned(4))) = {
    DBG_INFO_MAGIC, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
volatile uint32_t __tsr_crash[CRASH_NWORDS] __attribute__((section(".tsr_crash"), used, aligned(4)));

void tsr_rt_crt0_init();
[[noreturn]] void tsr_rt_trap(const uint32_t * frame);
}

// SW §6.3 crt0 step 2 (called from crt0.S after .bss, .tsr_crash and .init_array).
extern "C" void tsr_rt_crt0_init() {
    __tsr_dbg[DBG_INFO_MAGIC_WORD] = DBG_INFO_MAGIC;
    __tsr_dbg[DBG_INFO_VERSION_WORD] = 1u;
    __tsr_dbg[DBG_INFO_BUILD_ID_WORD] = __tsr_image_id[0];
    __tsr_dbg[DBG_INFO_TRAP_COUNT_WORD] = 0u;
    // __tsr_crash is zero (crt0): no magic until a trap fills it.
}

extern "C" [[noreturn]] void tsr_rt_trap(const uint32_t * frame) {
    uint32_t mcause, mepc, mtval, mstatus;
    __asm__ volatile("csrr %0, mcause" : "=r"(mcause));
    __asm__ volatile("csrr %0, mepc" : "=r"(mepc));
    __asm__ volatile("csrr %0, mtval" : "=r"(mtval));
    __asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
    const uint64_t now = rt::mtime64();
    const uint32_t rt_state = cw(CW_RT_STATE), rt_info = cw(CW_RT_INFO);
    const uint32_t phase = rt_state & RT_STATE_PHASE;
    const launch_info & L = tsr_rt_launch;
    const int in_kernel = L.active && (phase == RT_STATE_PHASE_KERNEL || phase == RT_STATE_PHASE_JOIN);

    volatile uint32_t * c = __tsr_crash;
    c[CRASH_VERSION_WORD] = 1u;
    c[CRASH_MCAUSE_WORD] = mcause;
    c[CRASH_MEPC_WORD] = mepc;
    c[CRASH_MTVAL_WORD] = mtval;
    c[CRASH_MSTATUS_WORD] = mstatus;
    c[CRASH_MTIME_LO_WORD] = uint32_t(now);
    c[CRASH_MTIME_HI_WORD] = uint32_t(now >> 32);
    c[CRASH_SEQ_WORD] = L.seq;
    c[CRASH_KERNEL_WORD] = in_kernel ? L.kernel : 0xFFFFFFFFu;
    c[CRASH_RT_STATE_WORD] = rt_state;
    c[CRASH_RT_INFO_WORD] = rt_info;
    for (uint32_t i = 1; i < 32u; ++i) c[CRASH_X1_WORD + i - 1u] = frame[i];
    c[CRASH_MAGIC_WORD] = CRASH_MAGIC;                  // last: the record is complete

    rt::log_write(LOG_HDR_LEVEL_FATAL, LOG_HDR_SOURCE_TRAP, LOG_HDR_TYPE_TRAP, c, CRASH_LOG_WORDS, 0);

    const uint32_t active = (cw(CW_LQ_FLAGS) & LQ_FLAGS_ACTIVE) ? 1u : 0u;
    const uint32_t seq = active ? L.seq : cw(CW_LQ_HEAD_SEQ);
    const uint32_t r[8] = {
        LCQ_R0_TYPE_TRAP | (LCQ_R0_STATUS_TRAP << LCQ_R0_STATUS_LSB) | ((seq & 0xFFu) << LCQ_R0_SEQ_LSB),
        seq, mcause, mepc, mtval, rt_state, rt_info, uint32_t(rt::mtime64()) };
    rt::post(r, active);

    rt::set_state(RT_STATE_PHASE_TRAPPED, rt_state >> RT_STATE_KERNEL_LSB);
    rt::mie_swap(0u);
    for (;;) rt::wfi();
}

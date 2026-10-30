/* sw/ctrl/rt/crt0.S — start-up, vector table and trap entry of the Tesseract firmware
 * (docs/v2/06-software-verification.md "SW" §6.2, §6.3, §6.6; docs/v2/05-debug.md "DB" §6.3;
 * docs/v2/01-execution-model.md "EX" §3.4, §4.7; docs/v2/04-control-core.md "CC" §3.2, §12.3).
 *
 * Vector table (.vectors at RV_BOOT = 0, sw/ctrl/rt/link.ld): 32 slots of 4 bytes. Ibex's mtvec is
 * vectored only (CC §3.2): exceptions enter slot 0, interrupt i slot i, NMI slot 31. The runtime
 * never enables interrupts (mstatus.MIE = 0; it waits in WFI with mie selecting the wake-up lines,
 * CC §12.3), so every slot jumps to the trap entry: an interrupt taken anyway is an unexpected
 * trap (mcause bit 31) and is recorded like any other.
 *
 * Reset entry _start at RV_BOOT + 0x80 (EX §3.4), SW §6.3 crt0:
 *   1. RT_STATE := BOOT; MTIME_LO at entry (boot core cycles, HELLO r7); gp, sp (8 KiB stack at the
 *      top of DTCM), mscratch := the top of the 512 B trap stack, mtvec := vectored, base
 *      __tsr_vectors (= RV_BOOT); mie := 0, mstatus.MIE := 0; zero [__zero_start, __zero_end)
 *      (.bss and .tsr_crash); run .init_array;
 *   2. tsr_rt_crt0_init: initialise __tsr_dbg and __tsr_crash (trap.cpp, DB §6.3);
 *   then the runtime, tsr_rt_main(boot MTIME) (runtime.cpp, EX §4.7 R1-R10), which never returns.
 *
 * Trap entry (DB §6.3, SW §6.6), synchronous exceptions and tsr::panic (mcause 24):
 *   1. swap sp with mscratch (the trap stack), save x1..x31 in a 128-byte frame (word i = x_i; the
 *      frame's word 0 is unused), __tsr_dbg.trap_count += 1, __tsr_crash.nested += 1; nested > 1
 *      (or a trap taken at __tsr_trap_halt itself: the debugger did not set dcsr.ebreakm) -> 5;
 *   2. __tsr_dbg.attached != 0: nested -= 1, restore every register (sp from the frame, mscratch
 *      back to the trap stack) and execute ebreak at __tsr_trap_halt; the GDB server sets dpc :=
 *      mepc and emulates mret (DB §4.7). A resume at __tsr_trap_halt + 4 executes mret itself;
 *   3./4. else tsr_rt_trap(frame) (trap.cpp): fill __tsr_crash, TRAP log record, EX's TRAP record,
 *      RT_STATE := TRAPPED, WFI loop with mie = 0 (never returns);
 *   5. nested trap: ebreak if attached, else a WFI loop (mie = 0).
 * The runtime does not unwind (EX §4.7): a trap can leave .data inconsistent; the host recovers
 * with SOFT_RESET and a warm restart (EX §3.5).
 *
 * tsr_rt_panic_trap(code) (tsr::panic, log.cpp): mtval := code, mcause := 24, mepc :=
 * __tsr_panic_point, then the trap entry. GDB shows SIGABRT at __tsr_panic_point (DB §4.7).
 */
#define CTL_BASE        0x20000000      /* control window (CC §5) */
#define CW_RT_STATE     0x290           /* EX §4.5 */
#define CW_MTIME_LO     0x300           /* CC §5.2 */
#define RT_PHASE_BOOT   1
#define DBG_ATTACHED    (3 * 4)         /* __tsr_dbg word 3 (DB §6.3) */
#define DBG_TRAP_COUNT  (5 * 4)         /* __tsr_dbg word 5 */
#define CRASH_NESTED    (43 * 4)        /* __tsr_crash word 43 */
#define FRAME           128             /* x1..x31 at 4 * i */
#define PANIC_MCAUSE    24              /* EX §4.7, DB §6.3 */

/* ---------------------------------------------------------------------------------------------
 * Vector table and reset entry (ITCM, RV_BOOT)
 * --------------------------------------------------------------------------------------------- */
    .section .vectors, "ax", @progbits
    .option push
    .option norvc
    .globl  __tsr_vectors
__tsr_vectors:
    .rept 32
    j       __tsr_trap_entry
    .endr

    .org    0x80
    .globl  _start
    .type   _start, @function
_start:
    li      t0, CTL_BASE
    li      t1, RT_PHASE_BOOT
    sw      t1, CW_RT_STATE(t0)         /* RT_STATE := BOOT (SW §6.3 crt0 step 2) */
    lw      s0, CW_MTIME_LO(t0)         /* MTIME at entry: HELLO r7 = boot core cycles */
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop
    la      sp, __stack_top
    la      t0, __trap_stack_top
    csrw    mscratch, t0
    la      t0, __tsr_vectors
    ori     t0, t0, 1                   /* MODE 1: vectored */
    csrw    mtvec, t0
    csrw    mie, zero
    csrci   mstatus, 8                  /* MIE := 0 */
    /* zero .bss and .tsr_crash (word aligned, link.ld) */
    la      t0, __zero_start
    la      t1, __zero_end
1:  bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
    /* static constructors (.init_array; the SDK itself has none) */
2:  la      s1, __init_array_start
    la      s2, __init_array_end
3:  bgeu    s1, s2, 4f
    lw      t0, 0(s1)
    jalr    t0
    addi    s1, s1, 4
    j       3b
4:  call    tsr_rt_crt0_init
    mv      a0, s0
    call    tsr_rt_main                 /* never returns */
5:  wfi
    j       5b
    .size   _start, . - _start
    .option pop

/* ---------------------------------------------------------------------------------------------
 * Trap entry (DB §6.3)
 * --------------------------------------------------------------------------------------------- */
    .text
    .option push
    .option norvc
    .p2align 2
    .globl  __tsr_trap_entry
    .type   __tsr_trap_entry, @function
__tsr_trap_entry:
    csrrw   sp, mscratch, sp            /* sp = trap stack, mscratch = the interrupted sp */
    addi    sp, sp, -FRAME
    sw      x1,   4(sp)
    sw      x3,  12(sp)
    sw      x4,  16(sp)
    sw      x5,  20(sp)
    sw      x6,  24(sp)
    sw      x7,  28(sp)
    sw      x8,  32(sp)
    sw      x9,  36(sp)
    sw      x10, 40(sp)
    sw      x11, 44(sp)
    sw      x12, 48(sp)
    sw      x13, 52(sp)
    sw      x14, 56(sp)
    sw      x15, 60(sp)
    sw      x16, 64(sp)
    sw      x17, 68(sp)
    sw      x18, 72(sp)
    sw      x19, 76(sp)
    sw      x20, 80(sp)
    sw      x21, 84(sp)
    sw      x22, 88(sp)
    sw      x23, 92(sp)
    sw      x24, 96(sp)
    sw      x25, 100(sp)
    sw      x26, 104(sp)
    sw      x27, 108(sp)
    sw      x28, 112(sp)
    sw      x29, 116(sp)
    sw      x30, 120(sp)
    sw      x31, 124(sp)
    csrr    t0, mscratch
    sw      t0,   8(sp)                 /* x2 = the interrupted sp */
    /* __tsr_dbg.trap_count += 1 */
    la      t1, __tsr_dbg
    lw      t2, DBG_TRAP_COUNT(t1)
    addi    t2, t2, 1
    sw      t2, DBG_TRAP_COUNT(t1)
    /* __tsr_crash.nested += 1 */
    la      t3, __tsr_crash
    lw      t2, CRASH_NESTED(t3)
    addi    t2, t2, 1
    sw      t2, CRASH_NESTED(t3)
    li      t4, 1
    bne     t2, t4, .Lnested
    csrr    t4, mepc                    /* a trap of the ebreak at __tsr_trap_halt (no ebreakm) */
    la      t5, __tsr_trap_halt
    beq     t4, t5, .Lnested
    lw      t4, DBG_ATTACHED(t1)
    bnez    t4, .Lattached
    /* 3./4. no debugger: the C handler on the trap stack (below the frame) */
    mv      a0, sp
    call    tsr_rt_trap                 /* never returns */
    j       .Lpark

.Lattached:
    /* 2. nested := 0 again, restore everything, stop in the debugger at __tsr_trap_halt */
    sw      zero, CRASH_NESTED(t3)
    lw      x1,   4(sp)
    lw      x3,  12(sp)
    lw      x4,  16(sp)
    lw      x5,  20(sp)
    lw      x6,  24(sp)
    lw      x7,  28(sp)
    lw      x8,  32(sp)
    lw      x9,  36(sp)
    lw      x10, 40(sp)
    lw      x11, 44(sp)
    lw      x12, 48(sp)
    lw      x13, 52(sp)
    lw      x14, 56(sp)
    lw      x15, 60(sp)
    lw      x16, 64(sp)
    lw      x17, 68(sp)
    lw      x18, 72(sp)
    lw      x19, 76(sp)
    lw      x20, 80(sp)
    lw      x21, 84(sp)
    lw      x22, 88(sp)
    lw      x23, 92(sp)
    lw      x24, 96(sp)
    lw      x25, 100(sp)
    lw      x26, 104(sp)
    lw      x27, 108(sp)
    lw      x28, 112(sp)
    lw      x29, 116(sp)
    lw      x30, 120(sp)
    lw      x31, 124(sp)
    addi    sp, sp, FRAME
    csrrw   sp, mscratch, sp            /* sp = the interrupted sp, mscratch = trap stack top */
    .globl  __tsr_trap_halt
__tsr_trap_halt:
    ebreak
    mret

.Lnested:
    /* 5. a trap inside the trap handler: stop for the debugger, else park */
    lw      t4, DBG_ATTACHED(t1)
    beqz    t4, .Lpark
    ebreak
.Lpark:
    csrw    mie, zero
1:  wfi
    j       1b
    .size   __tsr_trap_entry, . - __tsr_trap_entry

/* ---------------------------------------------------------------------------------------------
 * tsr_rt_panic_trap(code): enter the trap handler as an exception with mcause 24, mtval = code
 * --------------------------------------------------------------------------------------------- */
    .p2align 2
    .globl  tsr_rt_panic_trap
    .type   tsr_rt_panic_trap, @function
tsr_rt_panic_trap:
    csrw    mtval, a0
    li      t0, PANIC_MCAUSE
    csrw    mcause, t0
    la      t0, __tsr_panic_point
    csrw    mepc, t0
    csrci   mstatus, 8                  /* MIE := 0 (already 0 in this runtime) */
    j       __tsr_trap_entry
    .globl  __tsr_panic_point
__tsr_panic_point:
    j       __tsr_panic_point
    .size   tsr_rt_panic_trap, . - tsr_rt_panic_trap
    .option pop

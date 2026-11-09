/* crt0.S - start-up of the sim_rv_issue test firmware (sw/sim/tests/test_sim_rv_issue.c).
 *
 * Image layout of SW §6.2 in miniature: 32 vector slots at RV_BOOT = 0 (mtvec is vectored only,
 * CC §3.2), the reset entry at +0x80 (EX §3.4), stack at the top of DTCM. Not the SDK's crt0
 * (SW §6.3): no kernel table, image id or crash record. Every trap and interrupt goes to
 * __trap, which mirrors mcause / mepc to RT_INFO / LCQ staging, sets RT_STATE = TRAPPED (EX §4.5;
 * the host fails the test on it) and parks. The firmware itself never enables interrupts
 * (mstatus.MIE = 0); it sleeps in WFI with mie selecting the wake-up lines (SW §4.2).
 */
    .section .vectors, "ax", @progbits
    .option push
    .option norvc
    .globl __vectors
__vectors:
    .rept 32
    j       __trap
    .endr

    .org    0x80
    .globl  _start
_start:
    la      sp, __stack_top
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
2:  la      t0, __vectors
    ori     t0, t0, 1
    csrw    mtvec, t0
    call    main
3:  wfi
    j       3b

__trap:
    li      t1, 0x20000000          /* control window (CC §5) */
    csrr    t0, mcause
    sw      t0, 0x294(t1)           /* RT_INFO = mcause */
    csrr    t0, mepc
    sw      t0, 0x240(t1)           /* LCQ_W0 = mepc (debug aid, never posted) */
    li      t0, 7
    sw      t0, 0x290(t1)           /* RT_STATE = TRAPPED */
4:  wfi
    j       4b
    .option pop

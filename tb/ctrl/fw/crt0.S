/* crt0.S - start-up and trap handler of the tb/ctrl test firmware (fw_ctrl.c).
 *
 * The image layout of SW §6.2 in miniature (as sw/sim/tests/rv_issue/crt0.S): 32 vector slots at
 * RV_BOOT = 0 (mtvec vectored, CC §3.2), the reset entry at +0x80 (EX §3.4), stack at the top of
 * DTCM. The FPGA register file is not reset (Ibex RegFileFPGA), so _start writes every register
 * first: the RTL's RVFI stream and tsr_rv then agree from the first instruction on.
 *
 * Trap handler: the firmware provokes access faults on purpose (CC §4.1, §4.3, §5). Every
 * exception is logged into trap_log[] (mcause, mepc, mtval; trap_n counts them) and execution
 * continues after the faulting instruction (its length is read back from ITCM: an ITCM data
 * read, CC §4.1), or at ra for an instruction access fault. Interrupts are never enabled
 * (mstatus.MIE = 0; WFI wakes on mie).
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
    li      x1, 0
    li      x2, 0
    li      x3, 0
    li      x4, 0
    li      x5, 0
    li      x6, 0
    li      x7, 0
    li      x8, 0
    li      x9, 0
    li      x10, 0
    li      x11, 0
    li      x12, 0
    li      x13, 0
    li      x14, 0
    li      x15, 0
    li      x16, 0
    li      x17, 0
    li      x18, 0
    li      x19, 0
    li      x20, 0
    li      x21, 0
    li      x22, 0
    li      x23, 0
    li      x24, 0
    li      x25, 0
    li      x26, 0
    li      x27, 0
    li      x28, 0
    li      x29, 0
    li      x30, 0
    li      x31, 0
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
    .globl  __fw_end
__fw_end:
    j       __fw_end

__trap:
    csrw    mscratch, t0
    la      t0, trap_save
    sw      t1, 0(t0)
    sw      t2, 4(t0)
    lw      t1, trap_n
    li      t2, 16
    bgeu    t1, t2, 5f              /* log full: count only */
    slli    t2, t1, 4
    la      t0, trap_log
    add     t0, t0, t2
    csrr    t2, mcause
    sw      t2, 0(t0)
    csrr    t2, mepc
    sw      t2, 4(t0)
    csrr    t2, mtval
    sw      t2, 8(t0)
5:  addi    t1, t1, 1
    la      t0, trap_n
    sw      t1, 0(t0)
    csrr    t2, mcause
    li      t0, 1
    bne     t2, t0, 7f
    csrw    mepc, ra                /* instruction access fault: return to ra (fw_ctrl.c) */
    j       8f
7:  csrr    t1, mepc
    lhu     t2, 0(t1)               /* ITCM data read of the faulting instruction */
    andi    t2, t2, 3
    li      t0, 3
    addi    t1, t1, 2
    bne     t2, t0, 6f
    addi    t1, t1, 2
6:  csrw    mepc, t1
8:  la      t0, trap_save
    lw      t1, 0(t0)
    lw      t2, 4(t0)
    csrr    t0, mscratch
    mret
    .option pop

    .section .bss
    .balign 4
trap_save:
    .space  8

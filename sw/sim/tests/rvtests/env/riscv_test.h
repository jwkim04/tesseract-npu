// sw/sim/tests/rvtests/env/riscv_test.h - riscv-tests "p" environment for the tsr_rv test bus
// profile (docs/design/06-software-verification.md §4.2).
//
// Put this directory before riscv-tests' env/p on the include path: everything not redefined
// here (pass/fail, TESTNUM, the data section with tohost/fromhost, RVTEST_RV32U/M ...) comes from
// the pinned upstream header through #include_next.
//
// Differences from upstream env/p, both forced by the profile (CC §3.2, SW §4.2):
//  1. mtvec is vectored only, base 256-byte aligned (Ibex): trap_vector sits on a 256-byte
//     boundary and is installed as trap_vector | 1 (exceptions enter at the base in vectored mode,
//     so the handler is the same as upstream's). Upstream points mtvec at 4-byte aligned labels,
//     which such an mtvec cannot hold.
//  2. No probing of CSRs the profile does not have: upstream's INIT_RNMI (mnstatus), INIT_SATP
//     (satp), INIT_PMP (pmpaddr0 / pmpcfg0) and DELEGATE_NO_TRAPS (medeleg / mideleg) guard each
//     access with a trap to the next 4-byte label. The profile has M and U modes (Ibex), no
//     S-mode, PMP, RNMI or delegation; Spike (profile "test") still implements the PMP CSRs as
//     read-only zero while tsr_rv traps, so the probes would only make the two diverge. mie is
//     still cleared. As upstream, mstatus = 0 before the final mret, so rv32u* tests run in
//     U-mode on both models and rv32m* tests (init sets MPP = M) in M-mode.
// The rest of RVTEST_CODE_BEGIN is upstream's, line for line.

#ifndef TSR_RVTEST_ENV_H
#define TSR_RVTEST_ENV_H

#include_next "riscv_test.h"

#undef RVTEST_CODE_BEGIN
#define RVTEST_CODE_BEGIN                                               \
        .section .text.init;                                            \
        .align  6;                                                      \
        .weak stvec_handler;                                            \
        .weak mtvec_handler;                                            \
        .globl _start;                                                  \
_start:                                                                 \
        /* reset vector */                                              \
        j reset_vector;                                                 \
        .align 8;                                                       \
trap_vector:                                                            \
        /* test whether the test came from pass/fail */                 \
        csrr t5, mcause;                                                \
        li t6, CAUSE_USER_ECALL;                                        \
        beq t5, t6, write_tohost;                                       \
        li t6, CAUSE_SUPERVISOR_ECALL;                                  \
        beq t5, t6, write_tohost;                                       \
        li t6, CAUSE_MACHINE_ECALL;                                     \
        beq t5, t6, write_tohost;                                       \
        /* if an mtvec_handler is defined, jump to it */                \
        la t5, mtvec_handler;                                           \
        beqz t5, 1f;                                                    \
        jr t5;                                                          \
        /* was it an interrupt or an exception? */                      \
  1:    csrr t5, mcause;                                                \
        bgez t5, handle_exception;                                      \
        INTERRUPT_HANDLER;                                              \
handle_exception:                                                       \
        /* we don't know how to handle whatever the exception was */    \
  other_exception:                                                      \
        /* some unhandlable exception occurred */                       \
  1:    ori TESTNUM, TESTNUM, 1337;                                     \
  write_tohost:                                                         \
        sw TESTNUM, tohost, t5;                                         \
        sw zero, tohost + 4, t5;                                        \
        j write_tohost;                                                 \
reset_vector:                                                           \
        INIT_XREG;                                                      \
        RISCV_MULTICORE_DISABLE;                                        \
        csrwi mie, 0;                                                   \
        li TESTNUM, 0;                                                  \
        la t0, trap_vector;                                             \
        ori t0, t0, 1;                                                  \
        csrw mtvec, t0;                                                 \
        CHECK_XLEN;                                                     \
        /* if an stvec_handler is defined, delegate exceptions to it */ \
        la t0, stvec_handler;                                           \
        beqz t0, 1f;                                                    \
        csrw stvec, t0;                                                 \
        li t0, (1 << CAUSE_LOAD_PAGE_FAULT) |                           \
               (1 << CAUSE_STORE_PAGE_FAULT) |                          \
               (1 << CAUSE_FETCH_PAGE_FAULT) |                          \
               (1 << CAUSE_MISALIGNED_FETCH) |                          \
               (1 << CAUSE_USER_ECALL) |                                \
               (1 << CAUSE_BREAKPOINT);                                 \
        csrw medeleg, t0;                                               \
1:      csrwi mstatus, 0;                                               \
        init;                                                           \
        EXTRA_INIT;                                                     \
        EXTRA_INIT_TIMER;                                               \
        la t0, 1f;                                                      \
        csrw mepc, t0;                                                  \
        csrr a0, mhartid;                                               \
        mret;                                                           \
1:

#endif

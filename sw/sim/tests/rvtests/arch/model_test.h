// sw/sim/tests/rvtests/arch/model_test.h - riscv-arch-test 3.10.0 target macros for the tsr_rv
// test bus profile and Spike (docs/design/06-software-verification.md §4.2, §10.1; P1.11).
//
// Based on the suite's riscof-plugins/rv32/spike_simple/env/model_test.h: HTIF halt through
// tohost, signature between begin_signature and end_signature (dumped by Spike with
// +signature=FILE +signature-granularity=4 and by the tsr_rv runner with --signature FILE).
// Profile-specific: RVMODEL_MTVEC_ALIGN 8. mtvec is vectored only with a 256-byte aligned base
// (CC §3.2, SW §4.2), so arch_test.h must place its trap trampoline on a 256-byte boundary for
// its "is mtvec writable" probe to succeed on both models (it keeps mtvec's MODE bits).
// RVMODEL_BOOT (run at rvtest_entry_point, before the trap prolog) puts mtvec in vectored mode
// on both models: Spike resets it to 0 (direct), tsr_rv like Ibex to base | 1. The prolog keeps
// the MODE bits it reads and its handler stores (trap address - mtvec) in the signature, so
// without this the ebreak / ecall / cebreak signatures would differ in that one bit. Only
// exceptions are taken (they go to the base in both modes).

#ifndef _COMPLIANCE_MODEL_H
#define _COMPLIANCE_MODEL_H

#define ALIGNMENT 2
#define RVMODEL_MTVEC_ALIGN 8

#define RVMODEL_DATA_SECTION                                            \
        .pushsection .tohost,"aw",@progbits;                            \
        .align 8; .global tohost; tohost: .dword 0;                     \
        .align 8; .global fromhost; fromhost: .dword 0;                 \
        .popsection;

// Write 1 to tohost (pass / end of test); the result is the signature, not the exit code.
#define RVMODEL_HALT                                                    \
        li x1, 1;                                                       \
1:      sw x1, tohost, t2;                                              \
        j 1b;

#define RVMODEL_BOOT                                                    \
        csrwi mtvec, 1;

// mtval of an EBREAK is implementation-defined (zero or the pc); Ibex and tsr_rv write 0, Spike
// the pc. BREAKPOINT therefore leaves the suite's default mask of pc-relative tvals ("if xTVAL is
// set to zero for some cause, then the corresponding bit in SET_REL_TVAL_MSK should be cleared",
// arch_test.h); with the bit clear the handler records 0 for that cause on both models, instead of
// aborting the test when tval is outside the code, data and signature ranges (privilege/ebreak,
// C/cebreak-01). CAUSE_* come from encoding.h through arch_test.h.
#define SET_REL_TVAL_MSK ((1<<CAUSE_MISALIGNED_FETCH | 1<<CAUSE_FETCH_ACCESS |                      \
                           1<<CAUSE_MISALIGNED_LOAD  | 1<<CAUSE_LOAD_ACCESS  |                      \
                           1<<CAUSE_MISALIGNED_STORE | 1<<CAUSE_STORE_ACCESS |                      \
                           1<<CAUSE_FETCH_PAGE_FAULT | 1<<CAUSE_LOAD_PAGE_FAULT |                   \
                           1<<CAUSE_STORE_PAGE_FAULT) & 0xFFFFFFFF)

#define RVMODEL_DATA_BEGIN                                              \
        RVMODEL_DATA_SECTION                                            \
        .align ALIGNMENT;                                               \
        .global begin_signature; begin_signature:

#define RVMODEL_DATA_END                                                \
        .global end_signature; end_signature:

#define RVMODEL_IO_INIT
#define RVMODEL_IO_WRITE_STR(_R, _STR)
#define RVMODEL_IO_CHECK()
#define RVMODEL_IO_ASSERT_GPR_EQ(_S, _R, _I)
#define RVMODEL_IO_ASSERT_SFPR_EQ(_F, _R, _I)
#define RVMODEL_IO_ASSERT_DFPR_EQ(_D, _R, _I)

// No interrupt tests are built for the profile (privilege subset: ebreak, ecall, misalign-*).
#define RVMODEL_SET_MSW_INT
#define RVMODEL_CLEAR_MSW_INT
#define RVMODEL_CLEAR_MTIMER_INT
#define RVMODEL_CLEAR_MEXT_INT

#endif // _COMPLIANCE_MODEL_H

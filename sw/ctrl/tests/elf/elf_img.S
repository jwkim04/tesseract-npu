/* elf_img.S — toolchain-built images for the fw_elf tests (sw/ctrl/tests/build.cmake; P2.1).
 *
 * Not a runtime: a fixed image that uses every section of sw/ctrl/rt/link.ld (SW §6.2), linked
 * with that script and sw/ctrl/cmake/tsr_note.S exactly as the firmware images are, so the tests
 * see real linker output (sorted kernel table, the note's link-time values, .tsr_logfmt outside
 * every PT_LOAD) that does not change when the runtime or the kernels change. Variants (-D):
 *   NO_BASE     only the kernels k_alpha and k_beta (no graph / selftest / replay)
 *   BIG_TEXT    50 KiB more code: over SW §7.8's 48 KiB ITCM budget
 *   BAD_START   _start at +0x84                 -> link.ld ASSERT (reset entry)
 *   BAD_KT      a 4-byte .tsr_kernels fragment  -> link.ld ASSERT (whole entries)
 *   BAD_DBG     a 32-byte .tsr_dbg              -> link.ld ASSERT (DB §6.3 size)
 *   BIG_BSS     a 60 KiB .bss                   -> DTCM overflow into the stacks
 * Kernel hashes are FNV-1a 32 of the names (computed outside: graph 0xb9780647, selftest
 * 0x9692f9c3, replay 0xd0500bd2, k_alpha 0x18133c51, k_beta 0x59a73f25).
 */
    .section .vectors, "ax", @progbits
    .option push
    .option norvc
    .rept 32
    j       trap
    .endr
#ifdef BAD_START
    .org    0x84
#else
    .org    0x80
#endif
    .globl  _start
_start:
    la      gp, __global_pointer$
    la      sp, __stack_top
    la      t0, __trap_stack_top
    csrw    mscratch, t0
    la      a0, fmt_hello
    la      a1, msg
    la      a2, __tsr_image_id
    lw      a3, 0(a2)
1:  wfi
    j       1b
trap:
    j       trap
    .option pop

    .text
    .globl  graph, selftest, replay, k_alpha, k_beta
    .p2align 1
graph:      li a0, 0
            ret
selftest:   li a0, 1
            ret
replay:     li a0, 2
            ret
k_alpha:    li a0, 3
            ret
k_beta:     li a0, 4
            ret
#ifdef BIG_TEXT
    .zero   50 * 1024
#endif

/* Kernel table entries (SW §6.2): name_hash, entry, nargs, flags, argblk_lines, name[24]; each in
   its own ".tsr_kernels.<rank>.<name>" input section, emitted here in reverse table order so the
   linker's SORT_BY_NAME has to order them. */
    .macro  KDESC rank, name, hash, nargs, flags, lines
    .section .tsr_kernels.\rank\().\name, "aw"
    .p2align 2
    .word   \hash
    .word   \name
    .byte   \nargs, \flags
    .half   \lines
9:  .ascii  "\name"
    .zero   24 - (. - 9b)
    .endm

    KDESC   5, k_beta,   0x59a73f25, 0, 0, 0
    KDESC   5, k_alpha,  0x18133c51, 3, 1, 2
#ifndef NO_BASE
    KDESC   2, replay,   0xd0500bd2, 3, 0, 0
    KDESC   1, selftest, 0x9692f9c3, 5, 0, 0
    KDESC   0, graph,    0xb9780647, 3, 0, 0
#endif
#ifdef BAD_KT
    .section .tsr_kernels.9.junk, "aw"
    .word   0
#endif

    .section .tsr_id, "aw"
    .p2align 3
    .globl  __tsr_image_id
__tsr_image_id:
    .word   0, 0

    .section .tsr_dbg, "aw"
    .p2align 2
    .globl  __tsr_dbg
__tsr_dbg:
    .word   0x47424454, 1
#ifdef BAD_DBG
    .zero   24
#else
    .zero   56
#endif

    .section .tsr_crash, "aw", @nobits
    .p2align 2
    .globl  __tsr_crash
__tsr_crash:
    .zero   192

/* Log format strings (SW §6.6): not allocated, at 0xF000_0000. */
    .section .tsr_logfmt, "a"
fmt_hello:
    .asciz  "hello %d"
fmt_two:
    .asciz  "x=%08x\ty=%u"

    .section .rodata
msg:
    .asciz  "a loaded string"

    .data
    .p2align 2
    .word   1, 2, 3

    .section .sdata, "aw"
    .p2align 2
    .word   0x5DA7A

    .bss
    .p2align 2
#ifdef BIG_BSS
    .zero   60 * 1024
#else
    .zero   1024
#endif

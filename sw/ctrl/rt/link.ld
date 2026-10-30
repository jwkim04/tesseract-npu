/* sw/ctrl/rt/link.ld — linker script of the Tesseract firmware image (SW §6.2, EX §3.1, CC §4.1;
 * P2.1). Used for tsr-fw.elf and every test image (sw/ctrl/CMakeLists.txt).
 *
 * Memory map (CC §4.1): ITCM at 0x0000_0000, DTCM at 0x0010_0000. Their sizes are the build's
 * ITCM_BYTES / DTCM_BYTES (what RV_CFG reports; U200: 128 KiB / 64 KiB), passed by the build as
 * --defsym __tsr_itcm_bytes / __tsr_dtcm_bytes; the defaults below are the U200 values. The
 * control-core version the image needs (__tsr_ctl_id_min, CTL_ID[7:0]) comes the same way.
 *
 * Sections (SW §6.2):
 *   ITCM, loaded   .vectors  32 vector slots x 4 B at RV_BOOT = 0, reset entry _start at +0x80
 *                  .text     code
 *   DTCM, loaded   .tsr_kernels  kernel table, sorted by index (below)
 *                  .rodata, .data (incl. .init_array, small data); loaded directly by the host,
 *                            crt0 copies nothing
 *                  .tsr_id   the 8-byte image id (EX §3.2), written by tsr-elf after linking
 *                  .tsr_dbg  DB's debugger flags __tsr_dbg (DB §6.3, 64 B)
 *   DTCM, not loaded (crt0 zeroes [__zero_start, __zero_end) = .bss then .tsr_crash)
 *                  .bss      incl. small bss
 *                  .tsr_crash  DB's crash record __tsr_crash (DB §6.3, 192 B)
 *                  .stack    512 B trap stack [__trap_stack_bottom, __trap_stack_top), then the 8 KiB
 *                            stack [__stack_bottom, __stack_top) ending at the top of DTCM
 *   no memory      .tsr_logfmt  log format strings at VMA 0xF000_0000 (TSR_LOGFMT_VMA), kept in the
 *                            file but not allocated: no PT_LOAD, no TCM, not in the image id (SW §6.6)
 *                  .note.tsr  ELF note "TSR" type 1 (sw/ctrl/cmake/tsr_note.S; values below)
 *
 * Kernel table: TSR_KERNEL (tsr/rt.hpp) emits one entry per kernel into an input section named
 * ".tsr_kernels.<rank>.<name>"; the table is those sections sorted by name, so (rank, name) fixes
 * the kernel index (the position in the table, EX §3.1): rank 0 graph, 1 selftest, 2 replay, 5 every
 * other kernel (tsr/rt.hpp). Plain ".tsr_kernels" input sections follow, in link order. tsr-elf
 * validates the table, including this order, and writes the image id.
 *
 * Symbols for crt0 / the runtime: _start (crt0's, checked), __global_pointer$,
 * __zero_start / __zero_end (word aligned), __bss_start / __bss_end, __tsr_crash_start /
 * __tsr_crash_end, __init_array_start / __init_array_end, __trap_stack_bottom / __trap_stack_top,
 * __stack_bottom / __stack_top (16-byte aligned), __tsr_kernels_start / __tsr_kernels_end,
 * __tsr_n_kernels, __itcm_end, __dtcm_load_end, __tsr_logfmt_start / __tsr_logfmt_end.
 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    ITCM (rx) : ORIGIN = 0x00000000, LENGTH = DEFINED(__tsr_itcm_bytes) ? __tsr_itcm_bytes : 128K
    DTCM (rw) : ORIGIN = 0x00100000, LENGTH = DEFINED(__tsr_dtcm_bytes) ? __tsr_dtcm_bytes : 64K
}

/* Sizes of SW §6.2 / §6.5 / DB §6.3. */
__tsr_stack_bytes      = 8K;
__tsr_trap_stack_bytes = 512;
__tsr_kdesc_bytes      = 36;       /* sizeof(tsr_kernel_desc), SW §6.2 */
__tsr_dbg_bytes        = 64;
__tsr_crash_bytes      = 192;
/* .note.tsr values that do not come from the layout. */
__tsr_sdk_abi          = 1;
PROVIDE(__tsr_ctl_id_min = 2);     /* CTL_ID[7:0] of the control core (EX §5.1) */
PROVIDE(__tsr_caps_req   = 0);     /* an image needing CAPS bits defines this absolute symbol */

SECTIONS
{
    /* ---------------- ITCM ---------------- */
    .vectors ORIGIN(ITCM) : {
        KEEP(*(.vectors))
    } > ITCM

    .text : {
        *(.text.unlikely .text.*_unlikely .text.unlikely.*)
        *(.text.startup .text.startup.*)
        *(.text.hot .text.hot.*)
        *(.text .text.* .gnu.linkonce.t.*)
        . = ALIGN(4);
    } > ITCM
    __itcm_end = .;

    /* ---------------- DTCM, loaded ---------------- */
    .tsr_kernels ORIGIN(DTCM) : {
        __tsr_kernels_start = .;
        KEEP(*(SORT_BY_NAME(.tsr_kernels.*)))
        KEEP(*(.tsr_kernels))
        __tsr_kernels_end = .;
    } > DTCM

    .rodata : {
        . = ALIGN(4);
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2)
        *(.srodata .srodata.*)
        . = ALIGN(4);
    } > DTCM

    .data : {
        . = ALIGN(4);
        __init_array_start = .;
        KEEP(*(.preinit_array))
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
        KEEP(*(.init_array .ctors))
        __init_array_end = .;
        *(.data .data.* .gnu.linkonce.d.*)
        . = ALIGN(8);
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.* .gnu.linkonce.s.*)
        . = ALIGN(4);
    } > DTCM

    .tsr_id : {
        . = ALIGN(8);
        __tsr_id = .;
        KEEP(*(.tsr_id))
        . = __tsr_id + 8;
    } > DTCM

    .tsr_dbg : {
        . = ALIGN(4);
        KEEP(*(.tsr_dbg))
    } > DTCM
    __dtcm_load_end = .;

    /* ---------------- DTCM, not loaded (zeroed by crt0) ---------------- */
    .bss (NOLOAD) : {
        . = ALIGN(4);
        __zero_start = .;
        __bss_start = .;
        *(.sbss .sbss.* .gnu.linkonce.sb.*)
        *(.scommon)
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > DTCM

    .tsr_crash (NOLOAD) : {
        . = ALIGN(4);
        __tsr_crash_start = .;
        KEEP(*(.tsr_crash))
        . = ALIGN(4);
        __tsr_crash_end = .;
        __zero_end = .;
    } > DTCM

    /* 512 B trap stack, then the 8 KiB stack ending at the top of DTCM (SW §6.2). */
    .stack ORIGIN(DTCM) + LENGTH(DTCM) - __tsr_stack_bytes - __tsr_trap_stack_bytes (NOLOAD) : {
        __trap_stack_bottom = .;
        . += __tsr_trap_stack_bytes;
        __trap_stack_top = .;
        __stack_bottom = .;
        . += __tsr_stack_bytes;
        __stack_top = .;
    } > DTCM

    /* ---------------- not in memory ---------------- */
    .tsr_logfmt 0xF0000000 (INFO) : {
        __tsr_logfmt_start = .;
        KEEP(*(.tsr_logfmt .tsr_logfmt.*))
        __tsr_logfmt_end = .;
    }

    .note.tsr 0 (INFO) : { KEEP(*(.note.tsr)) }

    /* Non-allocated sections of the default script (DWARF for GDB, D18). */
    .comment         0 (INFO) : { *(.comment); LINKER_VERSION; }
    .riscv.attributes 0 : { KEEP(*(.riscv.attributes)) }
    .debug           0 : { *(.debug) }
    .line            0 : { *(.line) }
    .debug_srcinfo   0 : { *(.debug_srcinfo) }
    .debug_sfnames   0 : { *(.debug_sfnames) }
    .debug_aranges   0 : { *(.debug_aranges) }
    .debug_pubnames  0 : { *(.debug_pubnames) }
    .debug_info      0 : { *(.debug_info .gnu.linkonce.wi.*) }
    .debug_abbrev    0 : { *(.debug_abbrev) }
    .debug_line      0 : { *(.debug_line .debug_line.* .debug_line_end) }
    .debug_frame     0 : { *(.debug_frame) }
    .debug_str       0 : { *(.debug_str) }
    .debug_loc       0 : { *(.debug_loc) }
    .debug_macinfo   0 : { *(.debug_macinfo) }
    .debug_pubtypes  0 : { *(.debug_pubtypes) }
    .debug_ranges    0 : { *(.debug_ranges) }
    .debug_addr      0 : { *(.debug_addr) }
    .debug_line_str  0 : { *(.debug_line_str) }
    .debug_loclists  0 : { *(.debug_loclists) }
    .debug_macro     0 : { *(.debug_macro) }
    .debug_names     0 : { *(.debug_names) }
    .debug_rnglists  0 : { *(.debug_rnglists) }
    .debug_str_offsets 0 : { *(.debug_str_offsets) }
    .debug_sup       0 : { *(.debug_sup) }

    /* Relocation input sections: a static link applies them and emits none (an empty output
       section is dropped); listed so that --orphan-handling=warn stays quiet. */
    .rela.dyn : { *(.rela.*) } > ITCM

    /* No unwinding, no exceptions, no destructors at exit (SW §6.1 flags, SW-8). */
    /DISCARD/ : {
        *(.eh_frame .eh_frame_hdr .gcc_except_table .gcc_except_table.*)
        *(.fini_array .fini_array.* .dtors .dtors.*)
        *(.note.GNU-stack) *(.gnu_debuglink) *(.gnu.lto_*)
    }
}

/* .note.tsr values (EX §3.1). */
__tsr_itcm_min  = __itcm_end - ORIGIN(ITCM);
__tsr_dtcm_min  = __stack_top - ORIGIN(DTCM);
__tsr_n_kernels = (__tsr_kernels_end - __tsr_kernels_start) / __tsr_kdesc_bytes;

/* Link-time checks. Region overflows are reported by the linker itself; the SW §7.8 budget
   (ITCM <= 48 KiB, DTCM <= 56 KiB) is tsr-elf's (a warning, and the fw_elf ctest). */
ASSERT(_start == ORIGIN(ITCM) + 0x80, "link.ld: the reset entry _start must be at RV_BOOT + 0x80 (SW §6.2, EX §3.4)")
ASSERT(SIZEOF(.vectors) >= 0x80, "link.ld: .vectors must hold 32 vector slots of 4 bytes (SW §6.2)")
ASSERT((__tsr_kernels_end - __tsr_kernels_start) % __tsr_kdesc_bytes == 0,
       "link.ld: .tsr_kernels is not a whole number of kernel table entries (SW §6.2)")
ASSERT(SIZEOF(.tsr_id) == 8, "link.ld: .tsr_id must be 8 bytes (EX §3.2)")
ASSERT(SIZEOF(.tsr_dbg) == 0 || SIZEOF(.tsr_dbg) == __tsr_dbg_bytes, "link.ld: .tsr_dbg must be 64 bytes (DB §6.3)")
ASSERT(SIZEOF(.tsr_crash) == 0 || SIZEOF(.tsr_crash) == __tsr_crash_bytes, "link.ld: .tsr_crash must be 192 bytes (DB §6.3)")
ASSERT(__zero_end <= __trap_stack_bottom, "link.ld: DTCM overflow: .bss / .tsr_crash run into the stacks")
ASSERT(__stack_top == ORIGIN(DTCM) + LENGTH(DTCM), "link.ld: the stack must end at the top of DTCM (SW §6.2)")

/* link.ld - sim_rv_issue test firmware: the CC §4.1 / SW §6.2 memory map (ITCM at 0x0000_0000,
   DTCM at 0x0010_0000, U200 sizes). .rodata / .data are loaded into DTCM by the host loader
   (EX §3.3); .bss is not loaded, crt0 zeroes it. */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 128K
    DTCM (rw)  : ORIGIN = 0x00100000, LENGTH = 64K
}

SECTIONS
{
    .vectors : { KEEP(*(.vectors)) } > ITCM
    .text    : { *(.text .text.*) } > ITCM
    .rodata  : { . = ALIGN(4); *(.rodata .rodata.* .srodata .srodata.*) } > DTCM
    .data    : { . = ALIGN(4); *(.data .data.* .sdata .sdata.*) . = ALIGN(4); } > DTCM
    .bss (NOLOAD) : {
        . = ALIGN(4);
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > DTCM
    __stack_top = ORIGIN(DTCM) + LENGTH(DTCM);
    /DISCARD/ : { *(.comment) *(.note.*) *(.eh_frame*) }
}

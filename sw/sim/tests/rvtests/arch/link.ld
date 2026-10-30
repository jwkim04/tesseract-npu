/* sw/sim/tests/rvtests/arch/link.ld - riscv-arch-test programs on the test bus profile
   (RAM at 0x8000_0000, docs/design/06-software-verification.md §4.2); the layout of the suite's
   riscof-plugins/rv32/spike_simple/env/link.ld. */
OUTPUT_ARCH( "riscv" )
ENTRY(rvtest_entry_point)

SECTIONS
{
  . = 0x80000000;
  .text.init : { *(.text.init) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
  . = ALIGN(0x1000);
  .text : { *(.text) }
  . = ALIGN(0x1000);
  .data : { *(.data) }
  .data.string : { *(.data.string) }
  .bss : { *(.bss) }
  _end = .;
}

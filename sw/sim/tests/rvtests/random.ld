/* sw/sim/tests/rvtests/random.ld - random lockstep programs (rv_random.py) on the test bus
   profile: RAM at 0x8000_0000 (docs/design/06-software-verification.md §4.2). */
OUTPUT_ARCH( "riscv" )
ENTRY(_start)

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
  .bss : { *(.bss) }
  _end = .;
}

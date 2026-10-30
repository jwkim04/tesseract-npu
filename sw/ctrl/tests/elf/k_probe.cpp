// k_probe.cpp — an extra kernel for the fw_elf tests of tsr_fw_image() (sw/ctrl/tests/build.cmake;
// P2.1): built into the test images fw_elf_probe (production sources + this file) and
// fw_elf_probe_nok (NO_KERNELS: the runtime + this file, with DEFINES TSR_PROBE_RET=7). The tests
// check that the entry lands in the table after the base kernels (rank 5, tsr/rt.hpp) with the
// declared nargs / flags / argblk_lines, and that the image boots with HELLO's kernel count.
#include "tsr/rt.hpp"

#ifndef TSR_PROBE_RET
#define TSR_PROBE_RET 0
#endif

TSR_KERNEL(k_fw_elf_probe, 2, TSR_KFLAG_ARGBLK, 4);

extern "C" int32_t k_fw_elf_probe(const tsr_args * args) {
    return int32_t(args->a[1]) + TSR_PROBE_RET;
}

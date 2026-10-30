// Example kernel: y = a + b over n floats (the "Custom kernels" example of README.md).
// Built into the test image fw_example_vadd and checked bit-exactly by tests/test_fw_example_vadd.cpp.
#include "tsr/kernel.hpp"
#include "tsr/rt.hpp"

using namespace tsr;

// y = a + b over n floats. Args: a0 a, a1 b, a2 y (DRAM handles), a3 n (a multiple of 16).
TSR_KERNEL(vadd, 4, 0, 0);
extern "C" int32_t vadd(const tsr_args * args) {
    const uint32_t n = args->a[3], lines = n / 16;            // 16 floats per 64-byte SPM line
    const spm A{0}, B{lines};                                 // two buffers in the scratchpad
    const ticket la = issue(LD(A, from_handle(args->a[0]), lines));            // DMA
    const ticket lb = issue(LD(B, from_handle(args->a[1]), lines));            // DMA
    const ticket ad = issue(VOP({.func = VOP_FUNC_ADD, .d = A.line, .a = A.line,
                                 .b = B.line, .m = n, .rows = 1}), after(la, lb));  // vector unit
    issue(ST(A, from_handle(args->a[2]), lines), after(ad));                    // DMA
    return 0;                                                 // the runtime joins every op
}

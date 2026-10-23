// tsr_mvm_dbl.cpp — DPI-C shim for the tb/mvm test doubles (tb/mvm/hdl/dbl/tsr_mvm_back.sv):
// the fp operations of docs/rtl-m3.md §7.1 taken from libtsr_ref (sw/sim/tsr_ref.c, the bit-exact
// definition shared with the ISA simulator). f32 values cross the DPI boundary as 32-bit patterns
// (SV int); no arithmetic happens here, so NaN payloads reach tsr_ref.c unchanged.
// Linked against a private copy of build/dev/sw/sim/libtsr_ref.a (tb/mvm/test_mvm.py).
#include <cstdint>
#include <cstring>

#include "tsr_ref.h"

namespace {
inline float as_f(int u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline int as_u(float f) {
    int u;
    std::memcpy(&u, &f, 4);
    return u;
}
}  // namespace

extern "C" {
int tb_fadd(int a, int b) { return as_u(tsr_fadd(as_f(a), as_f(b))); }
int tb_fsub(int a, int b) { return as_u(tsr_fsub(as_f(a), as_f(b))); }
int tb_fmul(int a, int b) { return as_u(tsr_fmul(as_f(a), as_f(b))); }
int tb_i32_to_f32(int v) { return as_u(tsr_i32_to_f32(static_cast<int32_t>(v))); }
}

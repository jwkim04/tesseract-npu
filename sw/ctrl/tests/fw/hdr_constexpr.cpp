// hdr_constexpr.cpp — compile-time field checks of the op builders (P2.3; SW §6.7), run by the
// ctest fw_hdr_constexpr (sw/ctrl/tests/hdr.cmake, script mode): case 0 must compile, every other
// case must fail to compile, because in a constant expression a builder reaches a non-constexpr
// tsr::detail function for a field that does not fit or a rule the checker applies whatever the
// configuration (kernel.hpp, mmio.hpp op_put / op_poison_if).
#include "tsr/kernel.hpp"

using namespace tsr;

#if CASE == 0
constexpr op k_ld = LD(spm{1}, dram{15, 3}, 4);
constexpr op k_qa = QACT(qa::a256, spm{1}, spm{2}, 512, 8);
constexpr op k_mv = MVM(spm{1}, spm{2}, wref{3, FMT_Q4_K}, 9, 0xFF00, 8, 0, true);
constexpr op k_vo = VOP({.func = VOP_FUNC_COPY, .d = 0xFFFFFF, .a = 2, .m = 16, .rows = 0xFFFF});
constexpr op k_qk = ATTQK({.g_m1 = 63, .dst = 1, .x = 2, .base = 3, .e64 = 2, .n_groups = 4095});
static_assert(k_ld.w[0] == (OP_LD | (15u << 8)) && k_qa.w[4] == 8u && (k_mv.w[5] & 0xFFFFu) == 0xFF00u);
static_assert(k_vo.w[1] == 0xFFFFFFu && (k_vo.w[0] & 0x1000u) == 0u && (k_qk.w[0] & 0x8000u) == 0u);
int main() { return 0; }
#elif CASE == 1
constexpr op x = LD(spm{0}, dram{16, 0}, 1);                       // channel does not fit 4 bits
#elif CASE == 2
constexpr op x = ST(spm{0}, dram{255, 0}, 1);                      // channel does not fit 4 bits
#elif CASE == 3
constexpr op x = QACT(qa::a32, spm{0}, spm{1}, 48, 1);             // K_RANGE: not a multiple of 32
#elif CASE == 4
constexpr op x = QACT(qa::a32, spm{0}, spm{1}, 64, 0);             // T_RANGE: T = 0
#elif CASE == 5
constexpr op x = QACT(qa(2), spm{0}, spm{1}, 64, 1);               // ILLEGAL_OP: kind 2
#elif CASE == 6
constexpr op x = MVM(spm{0}, spm{1}, wref{0, 2}, 16, 64, 1);       // BAD_FMT: fmt 2
#elif CASE == 7
constexpr op x = MVM(spm{0}, spm{1}, wref{0, 0}, 16, 0x10000, 1);  // K does not fit 16 bits
#elif CASE == 8
constexpr op x = MVM(spm{0}, spm{1}, wref{0, 0}, 16, 64, 9);       // T_RANGE: T > 8
#elif CASE == 9
constexpr op x = VOP({.func = VOP_FUNC_COPY, .d = 0x1000000, .a = 2, .m = 16, .rows = 1});   // d does not fit 24 bits
#elif CASE == 10
constexpr op x = ATTQK({.g_m1 = 64, .dst = 1, .x = 2, .base = 3, .e64 = 2, .n_groups = 1});  // G - 1 does not fit 6 bits
#elif CASE == 11
constexpr op x = SOP({.sub = SOP_SUB_MOVI, .sd = 16, .imm = 1});   // sd does not fit 4 bits
#endif

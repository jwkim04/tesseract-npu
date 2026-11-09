// sw/tables/hdrcheck.cpp — build-time check that the generated headers compile together and agree
// (EX-30). Compiled as an object library in every build (C++20, the SDK's dialect, SW §6.1):
// the firmware headers sw/ctrl/include/tsr/{mmio,vu,at}.hpp and the C header tsr_tables.h in one
// translation unit, with constexpr spot checks across the two emitters. Nothing is linked.
#include "tsr/at.hpp"
#include "tsr/mmio.hpp"
#include "tsr/vu.hpp"
#include "tsr_tables.h"

static_assert(tsr::VERSION_VALUE == TSR_VERSION_VALUE && tsr::ISA_MAJOR == TSR_ISA_MAJOR, "VERSION");
static_assert(tsr::HELLO_R6_MAJOR == TSR_HELLO_R6_MAJOR && tsr::HELLO_R6_MINOR_LSB == TSR_HELLO_R6_MINOR_LSB &&
              tsr::HELLO_R4_KINDS_SIG_LSB == TSR_HELLO_R4_KINDS_SIG_LSB && tsr::LCQ_HELLO_BASIS_WORD == TSR_LCQ_HELLO_BASIS_WORD,
              "HELLO r4 / r6 (SW §6.9)");
static_assert(tsr::OP_MVM == TSR_OP_MVM && tsr::OP_ATTQK == TSR_OP_ATTQK, "opcodes");
static_assert(tsr::ERR_ILLEGAL_OP == TSR_ERR_ILLEGAL_OP, "error codes");
static_assert(tsr::CSR_PERF_LO_BASE == TSR_CSR_PERF_LO_BASE && tsr::CW_ISSUE_PIDX == TSR_CW_ISSUE_PIDX, "maps");
static_assert(tsr::VOP_FUNC_EXPSUB == TSR_VOP_FUNC_EXPSUB && tsr::VOP_RSV_W3 == TSR_VOP_RSV_W3, "VOP fields");

// A builder evaluated at compile time: VOP MULC, d 100, a 200, 896 elements, 1 row (VU §5.2).
constexpr tsr::op k_vop = tsr::VOP({.func = tsr::VOP_FUNC_MULC, .d = 100, .a = 200, .m = 896, .rows = 1});
static_assert(k_vop.w[0] == (TSR_OP_VOP | (TSR_VOP_FUNC_MULC << TSR_VOP_FUNC_LSB)), "VOP w0");
static_assert(k_vop.w[1] == 100 && k_vop.w[2] == 200 && k_vop.w[5] == 896 && k_vop.w[6] == 1, "VOP words");
// the generated builders set op::used to the encoding's field words (VOP_WORDS: w0..w7).
static_assert(k_vop.used == tsr::VOP_WORDS, "VOP used words (VOP_WORDS)");

// tsr_transport_xrt.cpp — the "xrt" transport (Alveo via XRT). M2+.
//
// Stub: built only with -DTSR_WITH_XRT=ON. The real implementation will map the CSR port to
// the kernel's s_axi_control (spec "XRT 어댑터"), move bytes with xrt::bo per DDR bank and
// poll ap_done. Until then opening it fails loudly instead of pretending to work.

#include <cstdio>

#include "tsr_transport_impl.h"

extern "C" void * tsr_transport_xrt_open(const tsr_transport_ops ** ops) {
    (void) ops;
    std::fprintf(stderr, "tsr_transport(xrt): not implemented yet (milestone M2)\n");
    return nullptr;
}

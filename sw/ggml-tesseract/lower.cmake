# lower.cmake — tsr_lower: the ggml-free part of the ggml-tesseract backend as a static library,
# linked by the backend (ggml-tesseract, see sources.cmake) and by the runtime sw/rt (tsr_rt).
#
#   tsr-lower.{h,cpp}    MUL_MAT shape checks and SPM tiling (weight placement, staging; the
#                        firmware tiles with plan.hpp, SW-20)
#   tsr-opl.{h,cpp}      the op-list path (docs/design/06-software-verification.md §9.3, §9.4): the op list format
#                        builder with DRAM-range hazards and JOINs, the OPL cache; tsr-opl-dev.cpp:
#                        the `graph` kernel on a transport (boot, launch, wait, error text)
#   tsr-alloc.{h,cpp}    first-fit range allocator (IO windows, AUX, symmetric WEIGHT heap, OPL cache)
#   tsr-wlayout.{h,cpp}  device-format row bytes, row repack + symmetric scatter / gather
#
# Included by the top-level CMakeLists after sw/transport and before llama.cpp (the backend's stub
# only checks tsr_ref and tsr_transport; targets are linked by name, so the order is for
# readability). Static, PIC (it is linked into libggml-tesseract.so). The firmware SDK's freestanding
# headers tsr/opl.hpp (op list format + host writer) and tsr/plan.hpp (the shared MUL_MAT planner) come from
# sw/ctrl/include.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

add_library(tsr_lower STATIC
    ${CMAKE_CURRENT_LIST_DIR}/tsr-alloc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsr-lower.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsr-opl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsr-opl-dev.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsr-wlayout.cpp
    # KV-cache buffers (tsr-kvbuf, AT §4.6) and the mask-summary producer (tsr-msum, AT §7.3)
    ${CMAKE_CURRENT_LIST_DIR}/tsr-kvbuf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsr-msum.cpp)
set_target_properties(tsr_lower PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_lower PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
target_include_directories(tsr_lower PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${TESSERACT_ROOT}/sw/ctrl/include)
target_link_libraries(tsr_lower PUBLIC tsr_ref tsr_transport)

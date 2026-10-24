# lower.cmake — tsr_lower: the ggml-free part of the ggml-tesseract backend as a static library,
# linked by the backend (ggml-tesseract, see sources.cmake) and by the runtime sw/rt (tsr_rt).
#
#   tsr-lower.{h,cpp}    MUL_MAT lowering (SPM tiling, LD / QACT / MVM / ST), disassembly,
#                        program cache
#   tsr-alloc.{h,cpp}    first-fit range allocator (IO windows, AUX, symmetric WEIGHT heap)
#   tsr-wlayout.{h,cpp}  device-format row bytes, row repack + symmetric scatter / gather
#
# Included by the top-level CMakeLists before llama.cpp (the backend's stub only checks tsr_ref and
# tsr_transport; targets are linked by name, so the order is for readability). Static, PIC (it is
# linked into libggml-tesseract.so).

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

add_library(tsr_lower STATIC
    ${CMAKE_CURRENT_LIST_DIR}/tsr-alloc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsr-lower.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tsr-wlayout.cpp)
set_target_properties(tsr_lower PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_lower PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
target_include_directories(tsr_lower PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(tsr_lower PUBLIC tsr_ref)

# sources.cmake — included by llama.cpp's ggml/src/ggml-tesseract/CMakeLists.txt (added by
# third_party/patches/0001-ggml-tesseract.patch). Keep the source list here so that adding a
# file does not require regenerating the llama.cpp patch.

set(GGML_TESSERACT_DIR ${CMAKE_CURRENT_LIST_DIR})

# tsr-alloc, tsr-lower, tsr-opl, tsr-wlayout, tsr-kvbuf and tsr-msum (no ggml) are the static
# library tsr_lower (lower.cmake), shared with the runtime sw/rt; only the ggml-dependent sources are
# listed here. tsr-build (the M7 records of ggml nodes, supports_op M7, fusion, the graph key;
# docs/design/06-software-verification.md §9.3) is ggml-aware but device-free, so the unit tests compile
# it too (CMakeLists.txt, test-tsr-backend-units). tsr-pivot-graph (Tesseract Pivot's ggml adapter over the core in tsr_lower) is
# ggml-aware and device-free as well (CMakeLists.txt, test-tsr-pivot-graph).
set(GGML_TESSERACT_SOURCES
    ${GGML_TESSERACT_DIR}/ggml-tesseract.cpp
    ${GGML_TESSERACT_DIR}/tsr-build.cpp
    ${GGML_TESSERACT_DIR}/tsr-build.h
    ${GGML_TESSERACT_DIR}/tsr-pivot-graph.cpp
    ${GGML_TESSERACT_DIR}/tsr-pivot-graph.h
    ${GGML_TESSERACT_DIR}/tsr-prof.cpp
    ${GGML_TESSERACT_DIR}/tsr-prof.h
    ${GGML_TESSERACT_DIR}/tsr-weights.cpp
    ${GGML_TESSERACT_DIR}/tsr-weights.h
)

set(GGML_TESSERACT_INCLUDE_DIRS
    ${GGML_TESSERACT_DIR}
    ${TESSERACT_ROOT}/sw/sim/include
    ${TESSERACT_ROOT}/sw/transport
    ${TESSERACT_ROOT}/sw/ctrl/include     # tsr/opl.hpp, tsr/plan.hpp (firmware SDK, freestanding)
)

# The stub creates ggml-tesseract after including this file and links tsr_ref / tsr_transport;
# tsr_lower is linked once the target exists (end of the stub's directory), so the llama.cpp patch
# does not have to change.
if (NOT TARGET tsr_lower)
    message(FATAL_ERROR "GGML_TESSERACT: target tsr_lower is not defined (sw/ggml-tesseract/lower.cmake, "
                        "included by the Tesseract top-level CMakeLists.txt before llama.cpp)")
endif()
cmake_language(DEFER CALL target_link_libraries ggml-tesseract PRIVATE tsr_lower)
# DB §4.1, §9.5: the backend is an owning process, so TSR_GDB / TSR_DEBUGGER start libtsr_dbg's embedded
# service (sw/tools/dbg/dbg.cmake: the OBJECT library registers it before main)
if (TARGET tsr_dbg_embed)
    cmake_language(DEFER CALL target_sources ggml-tesseract PRIVATE $<TARGET_OBJECTS:tsr_dbg_embed>)
    cmake_language(DEFER CALL target_link_libraries ggml-tesseract PRIVATE tsr_dbg)
endif()

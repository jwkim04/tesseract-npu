# sources.cmake — included by llama.cpp's ggml/src/ggml-tesseract/CMakeLists.txt (added by
# third_party/patches/0001-ggml-tesseract.patch). Keep the source list here so that adding a
# file does not require regenerating the llama.cpp patch.

set(GGML_TESSERACT_DIR ${CMAKE_CURRENT_LIST_DIR})

# tsr-alloc, tsr-lower and tsr-wlayout (no ggml) are the static library tsr_lower
# (lower.cmake), shared with the runtime sw/rt; only the ggml-dependent sources are listed here.
set(GGML_TESSERACT_SOURCES
    ${GGML_TESSERACT_DIR}/ggml-tesseract.cpp
    ${GGML_TESSERACT_DIR}/tsr-weights.cpp
    ${GGML_TESSERACT_DIR}/tsr-weights.h
)

set(GGML_TESSERACT_INCLUDE_DIRS
    ${GGML_TESSERACT_DIR}
    ${TESSERACT_ROOT}/sw/sim/include
    ${TESSERACT_ROOT}/sw/transport
)

# The stub creates ggml-tesseract after including this file and links tsr_ref / tsr_transport;
# tsr_lower is linked once the target exists (end of the stub's directory), so the llama.cpp patch
# does not have to change.
if (NOT TARGET tsr_lower)
    message(FATAL_ERROR "GGML_TESSERACT: target tsr_lower is not defined (sw/ggml-tesseract/lower.cmake, "
                        "included by the Tesseract top-level CMakeLists.txt before llama.cpp)")
endif()
cmake_language(DEFER CALL target_link_libraries ggml-tesseract PRIVATE tsr_lower)

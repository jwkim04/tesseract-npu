# sources.cmake — included by llama.cpp's ggml/src/ggml-tesseract/CMakeLists.txt (added by
# third_party/patches/0001-ggml-tesseract.patch). Keep the source list here so that adding a
# file does not require regenerating the llama.cpp patch.

set(GGML_TESSERACT_DIR ${CMAKE_CURRENT_LIST_DIR})

set(GGML_TESSERACT_SOURCES
    ${GGML_TESSERACT_DIR}/ggml-tesseract.cpp
    ${GGML_TESSERACT_DIR}/tsr-alloc.cpp
    ${GGML_TESSERACT_DIR}/tsr-alloc.h
    ${GGML_TESSERACT_DIR}/tsr-lower.cpp
    ${GGML_TESSERACT_DIR}/tsr-lower.h
    ${GGML_TESSERACT_DIR}/tsr-weights.cpp
    ${GGML_TESSERACT_DIR}/tsr-weights.h
)

set(GGML_TESSERACT_INCLUDE_DIRS
    ${GGML_TESSERACT_DIR}
    ${TESSERACT_ROOT}/sw/sim/include
    ${TESSERACT_ROOT}/sw/transport
)

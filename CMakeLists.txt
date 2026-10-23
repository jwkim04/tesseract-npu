cmake_minimum_required(VERSION 3.20)
project(tesseract C CXX)

# One build tree: the pinned llama.cpp (with the ggml-tesseract backend enabled via
# third_party/patches) plus Tesseract's own libraries, tools and tests.
#   cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release && cmake --build build/dev -j4

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TESSERACT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

# Spec 수치 규칙: separately rounded f32 ops, no fused multiply-add, no fast-math.
set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)

option(TSR_WITH_LLAMA "Build pinned llama.cpp with the ggml-tesseract backend" ON)
option(TSR_WITH_XRT   "Build the XRT transport (needs XRT)" OFF)

enable_testing()

add_subdirectory(sw/sim)         # tsr_ref, tsr_sim, tsr_trace (+ unit tests)
add_subdirectory(sw/transport)   # tsr_transport (sim, optional xrt)
add_subdirectory(sw/tools)       # trace runner, test vector generator, klcheck

if (TSR_WITH_LLAMA)
    set(GGML_TESSERACT ON CACHE BOOL "" FORCE)
    # As a subproject llama.cpp defaults common/tests/tools to OFF. M1 needs test-backend-ops
    # (tests, which require common), llama-completion / llama-perplexity (tools) and llama-cli
    # (built from the server sources).
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/llama.cpp)
    add_subdirectory(sw/ggml-tesseract)   # backend tests; the library is built inside ggml (patch 0001)
endif()

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
option(TSR_WITH_TORCH "Build the PyTorch extension sw/torch (_tesseract_rt; needs TSR_WITH_LLAMA and torch in TSR_TORCH_PYTHON)" OFF)

enable_testing()

add_subdirectory(sw/tables)      # v2 table source: --check / self-test ctests, generated-header compile check
add_subdirectory(sw/sim)         # tsr_ref, tsr_sim (+ unit tests)
add_subdirectory(sw/ctrl/host)   # firmware SDK, host side: tsr-elf, the tsr_fw ExternalProject
                                 # (RISC-V image tsr-fw.elf, SW §6.1), host build of the kernels, fw_* tests
add_subdirectory(sw/transport)   # tsr_transport (sim, mock, vkernel, optional xrt)
include(sw/ggml-tesseract/lower.cmake)   # tsr_lower: SPM tiling, allocator, weight layout (no ggml);
                                         # linked by ggml-tesseract and by the runtime sw/rt
add_subdirectory(sw/tools)       # selftest, klcheck, numcheck, tsr_bw

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
    add_subdirectory(sw/rt)               # tsr_rt: ggml-free runtime (weights + linear); uses ggml-base
                                          # for the load-time Q8_0 conversion (quantize_row_q8_0_ref)
    if (TSR_WITH_TORCH)
        add_subdirectory(sw/torch)        # _tesseract_rt (torch extension binding tsr_rt) + pytest ctests
    endif()
elseif (TSR_WITH_TORCH)
    message(FATAL_ERROR "TSR_WITH_TORCH needs TSR_WITH_LLAMA=ON (tsr_rt links ggml-base of the pinned llama.cpp)")
endif()

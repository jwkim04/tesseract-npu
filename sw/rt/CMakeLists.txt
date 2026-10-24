# sw/rt — tsr_rt: the ggml-free Tesseract runtime (weight upload, linear y = x W^T), the host stack
# of the PyTorch backend. See tsr_rt.h.
#
#   tsr_rt         static, PIC: tsr_rt.cpp + the ggml backend's load-time conversion
#                  (sw/ggml-tesseract/tsr-weights.cpp: f32 / f16 / bf16 -> quantize_row_q8_0_ref);
#                  links tsr_lower (lowering, allocator, weight layout: shared with ggml-tesseract),
#                  tsr_transport and ggml-base (pinned llama.cpp: quantize_row_q8_0_ref, to_float)
#   tsr_rt_shared  shared, OUTPUT_NAME tsr_rt -> libtsr_rt.so: everything above but ggml-base inside
#                  one library, for the torch extension (links libggml-base.so)
#
# Added by the top-level CMakeLists after llama.cpp (needs the ggml-base target).

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()
if (NOT TARGET ggml-base OR NOT TARGET tsr_lower)
    message(FATAL_ERROR "sw/rt needs the ggml-base target (pinned llama.cpp) and tsr_lower; add it after llama.cpp")
endif()

find_package(Threads REQUIRED)

set(TSR_RT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tsr_rt.cpp
    ${TESSERACT_ROOT}/sw/ggml-tesseract/tsr-weights.cpp)

function(tsr_rt_target tgt)
    set_target_properties(${tgt} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
    target_include_directories(${tgt}
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)   # ggml-quants.h (quantize_row_q8_0_ref)
    target_link_libraries(${tgt} PUBLIC tsr_ref PRIVATE tsr_lower tsr_transport ggml-base Threads::Threads)
endfunction()

add_library(tsr_rt STATIC ${TSR_RT_SOURCES})
tsr_rt_target(tsr_rt)

add_library(tsr_rt_shared SHARED ${TSR_RT_SOURCES})
tsr_rt_target(tsr_rt_shared)
set_target_properties(tsr_rt_shared PROPERTIES OUTPUT_NAME tsr_rt
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Tests (prefix rt_): device vs an independent model (ggml conversion + libtsr_ref) bit for bit,
# linear_ref, read_blocks, N / K / T / IO splits, program cache, counters, threads.
# ---------------------------------------------------------------------------
add_executable(test-tsr-rt tests/test_tsr_rt.cpp)
set_target_properties(test-tsr-rt PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(test-tsr-rt PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
target_include_directories(test-tsr-rt PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)
target_link_libraries(test-tsr-rt PRIVATE tsr_rt tsr_ref ggml-base Threads::Threads)

add_test(NAME rt_sim COMMAND test-tsr-rt)
set_tests_properties(rt_sim PROPERTIES ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_THREADS=4")
# T_MAX 8: T tiles inside one program, and a natural N split (an N = 130000 output head does not
# fit 8 tokens in the SPM unsplit)
add_test(NAME rt_sim_tmax8 COMMAND test-tsr-rt --big)
set_tests_properties(rt_sim_tmax8 PROPERTIES ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_TMAX=8;TSR_SIM_THREADS=4")
# the kernel protocol layer on the mock kdev, DMA rules that force bounce / read-modify-write
add_test(NAME rt_mock COMMAND test-tsr-rt)
set_tests_properties(rt_mock PROPERTIES ENVIRONMENT "TSR_DEVICE=mock;TSR_MOCK_DMA_ALIGN=64;TSR_MOCK_DMA_CHUNK_KIB=64;TSR_SIM_THREADS=4")
add_test(NAME rt_mock_tmax8 COMMAND test-tsr-rt)
set_tests_properties(rt_mock_tmax8 PROPERTIES ENVIRONMENT
    "TSR_DEVICE=mock;TSR_MOCK_TMAX=8;TSR_MOCK_NUM_CH=3;TSR_MOCK_DMA_ALIGN=4096;TSR_MOCK_DMA_CHUNK_KIB=12;TSR_SIM_THREADS=4")

# Verilated U200 kernel (RTL) through the vkernel kdev: a few shapes on k4t1. The server is built
# on first use by tb/platform/vkernel/vkernel.py (minutes, Verilator), so this test is opt-in.
option(TSR_RT_VKERNEL_TEST "Add the rt_vkernel_k4t1 ctest (builds the k4t1 vkernel server on first use)" OFF)
if (TSR_RT_VKERNEL_TEST)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME rt_vkernel_k4t1
             COMMAND ${Python3_EXECUTABLE} ${TESSERACT_ROOT}/tb/platform/vkernel/vkernel.py run --config k4t1
                     -- $<TARGET_FILE:test-tsr-rt> --quick)
endif()

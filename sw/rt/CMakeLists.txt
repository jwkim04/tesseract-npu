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
    # DB §4.1, §9.5: tsr_rt is an owning process's library: TSR_GDB / TSR_DEBUGGER start libtsr_dbg's embedded
    # service (the OBJECT library registers it before main)
    # (a static library hands the object file to its consumers' link lines: an archive member with only a
    # static initializer would not be pulled in otherwise)
    if (TARGET tsr_dbg_embed)
        get_target_property(_type ${tgt} TYPE)
        if (_type STREQUAL "STATIC_LIBRARY")
            target_link_libraries(${tgt} PUBLIC $<TARGET_OBJECTS:tsr_dbg_embed> tsr_dbg)
        else()
            target_sources(${tgt} PRIVATE $<TARGET_OBJECTS:tsr_dbg_embed>)
            target_link_libraries(${tgt} PRIVATE tsr_dbg)
        endif()
    endif()
endfunction()

add_library(tsr_rt STATIC ${TSR_RT_SOURCES})
tsr_rt_target(tsr_rt)

add_library(tsr_rt_shared SHARED ${TSR_RT_SOURCES})
tsr_rt_target(tsr_rt_shared)
set_target_properties(tsr_rt_shared PROPERTIES OUTPUT_NAME tsr_rt
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Tests (prefix rt_): device vs an independent model (ggml conversion + libtsr_ref) bit for bit,
# linear_ref, read_blocks, N / K / T / IO splits, op-list cache, counters, threads
# (docs/design/06-software-verification.md §10.2; the simulator's race detector fatal, SW §4.6).
# ---------------------------------------------------------------------------
add_executable(test-tsr-rt tests/test_tsr_rt.cpp)
set_target_properties(test-tsr-rt PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(test-tsr-rt PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
target_include_directories(test-tsr-rt PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)
target_link_libraries(test-tsr-rt PRIVATE tsr_rt tsr_ref ggml-base Threads::Threads)

add_test(NAME rt_sim COMMAND test-tsr-rt)
set_tests_properties(rt_sim PROPERTIES ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_THREADS=4;TSR_SIM_RACE=fatal")
# T_MAX 8: T tiles inside one record, and a natural N split (an N = 130000 output head does not
# fit 8 tokens in the SPM unsplit)
add_test(NAME rt_sim_tmax8 COMMAND test-tsr-rt --big)
set_tests_properties(rt_sim_tmax8 PROPERTIES ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_TMAX=8;TSR_SIM_THREADS=4;TSR_SIM_RACE=fatal")
# T_MAX 4: the core of the U200 K4 image (docs/platform-u200.md): T tiles of 4 tokens inside one record
add_test(NAME rt_sim_tmax4 COMMAND test-tsr-rt --big)
set_tests_properties(rt_sim_tmax4 PROPERTIES ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_TMAX=4;TSR_SIM_THREADS=4;TSR_SIM_RACE=fatal")
# the kernel protocol layer on the mock kdev, DMA rules that force bounce / read-modify-write
# TSR_GDB in an owning process (tsr_rt links libtsr_dbg's embedded service): the service starts (no "no embedded debug
# service" note), the run is unchanged
add_test(NAME rt_sim_gdb_embed COMMAND test-tsr-rt --quick)
set_tests_properties(rt_sim_gdb_embed PROPERTIES
    ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_THREADS=4;TSR_GDB=tcp:0;TSR_GDB_PORT_FILE=${CMAKE_CURRENT_BINARY_DIR}/rt_gdb_port.txt"
    FAIL_REGULAR_EXPRESSION "no embedded debug service")
add_test(NAME rt_mock COMMAND test-tsr-rt)
set_tests_properties(rt_mock PROPERTIES ENVIRONMENT
    "TSR_DEVICE=mock;TSR_MOCK_DMA_ALIGN=64;TSR_MOCK_DMA_CHUNK_KIB=64;TSR_SIM_THREADS=4;TSR_SIM_RACE=fatal")
add_test(NAME rt_mock_tmax8 COMMAND test-tsr-rt)
set_tests_properties(rt_mock_tmax8 PROPERTIES ENVIRONMENT
    "TSR_DEVICE=mock;TSR_MOCK_TMAX=8;TSR_MOCK_NUM_CH=3;TSR_MOCK_DMA_ALIGN=4096;TSR_MOCK_DMA_CHUNK_KIB=12;TSR_SIM_THREADS=4;TSR_SIM_RACE=fatal")

# The image rules of tsr_rt (SW §6.9, as the ggml backend's): an image of an older Tesseract Basis
# (fw_basis_old, 0.0.5: the production image linked as such) and an image without the MUL_MAT kind
# (fw_no_mulmat: the production image with its TSR_KIND(MUL_MAT) left out) are refused at open, with the reason.
add_test(NAME rt_sim_basis_old COMMAND test-tsr-rt --quick)
set_tests_properties(rt_sim_basis_old PROPERTIES
    ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_THREADS=4;TSR_IMAGE=${TSR_FW_TEST_ELF_fw_basis_old}"
    PASS_REGULAR_EXPRESSION "older than this library's ")
add_test(NAME rt_sim_no_mulmat COMMAND test-tsr-rt --quick)
set_tests_properties(rt_sim_no_mulmat PROPERTIES
    ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_THREADS=4;TSR_IMAGE=${TSR_FW_TEST_ELF_fw_no_mulmat}"
    PASS_REGULAR_EXPRESSION "does not provide the MUL_MAT kind")

# Verilated U200 kernel (RTL) through the vkernel kdev: a few shapes on k4t1. Opt-in; until the
# two-clock vkernel server exists (docs/design/06-software-verification.md §9.1) the
# transport has no vkernel device to run on, so the test is reported as skipped (exit 77) when the
# server is absent.
option(TSR_RT_VKERNEL_TEST "Add the rt_vkernel_k4t1 ctest (needs the k4t1 vkernel server)" OFF)
if (TSR_RT_VKERNEL_TEST)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_tsr_vk ${TESSERACT_ROOT}/tb/platform/vkernel/vkernel.py)
    add_test(NAME rt_vkernel_k4t1
             COMMAND sh -c "if [ ! -f '${_tsr_vk}' ]; then echo 'rt_vkernel_k4t1: no vkernel server: skipped'; exit 77; fi; exec '${Python3_EXECUTABLE}' '${_tsr_vk}' run --config k4t1 -- '$<TARGET_FILE:test-tsr-rt>' --quick")
    set_tests_properties(rt_vkernel_k4t1 PROPERTIES SKIP_RETURN_CODE 77)
endif()

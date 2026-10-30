# sw/sim/tests — unit tests for the reference library (prefix ref_).
# Simulator / trace tests (prefix sim_ / trace_) are to be added below by their owners.
#
# Run: ctest --test-dir <build> -R '^ref_' --output-on-failure

function(tsr_ref_test name src)
    add_executable(${name} ${src})
    tsr_c_target(${name})
    target_link_libraries(${name} PRIVATE tsr_ref m)
    add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

# Tests that need nothing but tsr_ref.
tsr_ref_test(ref_layout test_ref_layout.c)          # ROW_BEATS, repack/unrepack, allocation, QA lines
tsr_ref_test(ref_fp     test_ref_fp.c)              # fp helpers (incl. exhaustive int32 -> f32)

# Cross-checks against ggml-base at the llama.cpp pin. ggml-base is defined later in the
# top-level project (third_party/llama.cpp); target names resolve at generate time.
if (TSR_WITH_LLAMA)
    set(TSR_GGML_SRC_DIR ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)   # ggml-quants.h, ggml-common.h

    add_executable(ref_ggml test_ref_ggml.c)
    tsr_c_target(ref_ggml)
    target_include_directories(ref_ggml SYSTEM PRIVATE ${TSR_GGML_SRC_DIR})
    target_link_libraries(ref_ggml PRIVATE tsr_ref ggml-base m)
    add_test(NAME ref_ggml_fp16    COMMAND ref_ggml fp16)     # exhaustive fp16 <-> f32
    add_test(NAME ref_ggml_qact    COMMAND ref_ggml qact)     # A32/A256 vs quantize_row_q8_{0,K}_ref
    add_test(NAME ref_ggml_dequant COMMAND ref_ggml dequant)  # vs dequantize_row_q8_0 / q4_K

    add_executable(ref_dot test_ref_dot.c)
    tsr_c_target(ref_dot)
    target_include_directories(ref_dot SYSTEM PRIVATE ${TSR_GGML_SRC_DIR})
    target_link_libraries(ref_dot PRIVATE tsr_ref ggml-base m)
    add_test(NAME ref_dot COMMAND ref_dot)                    # spec formula bit-exact + double ref
endif()

# Python ctypes smoke test on libtsr_ref.so (the interface cocotb uses).
find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_test(NAME ref_ctypes
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_ref_ctypes.py
                     $<TARGET_FILE:tsr_ref_shared>)
endif()

# ---------------------------------------------------------------------------
# Simulator / capture / transport tests (owner: M1 step 3/4; v2 core: P1.2 / P1.6). Prefix sim_.
# Run: ctest --test-dir <build> -R '^sim_' --output-on-failure
# sim_basic / sim_mvm / sim_capture test the units through the tsr_sim_op test hook (op-level,
# outside the queues); sim_csr tests the host CSR port (EX §5) and the RISC-V side of the rings;
# sim_queue tests the queue model (dispatcher, checker order, tickets, deps, SYNC, errors) through
# the control window. The v1 trace round trip returns with trace v2 (P1.10); sim_transport is
# rewritten for the launch path in P3.1.
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

function(tsr_sim_test name src)
    add_executable(${name} ${src})
    tsr_c_target(${name})
    target_include_directories(${name} PRIVATE ${TSR_SIM_INT_DIR})   # tsr_sim_int.h
    target_link_libraries(${name} PRIVATE ${ARGN} Threads::Threads m)
endfunction()

set(TSR_SIM_TEST_SCRATCH ${CMAKE_CURRENT_BINARY_DIR}/scratch)

tsr_sim_test(sim_basic test_sim_basic.c tsr_sim tsr_ref)        # CSRs, LD/ST, errors, reserved fields incl. x16
add_test(NAME sim_basic COMMAND sim_basic)

tsr_sim_test(sim_mvm test_sim_mvm.c tsr_sim tsr_ref)            # LD/QACT/MVM/ST op sequences vs direct tsr_ref
add_test(NAME sim_mvm COMMAND sim_mvm 150)

tsr_sim_test(sim_capture test_sim_capture.c tsr_sim tsr_ref)    # access capture (init / expect regions)
add_test(NAME sim_capture COMMAND sim_capture)

tsr_sim_test(sim_csr test_sim_csr.c tsr_sim tsr_ref)            # host CSR port, loader, LQ / LCQ / log, SOFT_RESET, PERF
add_test(NAME sim_csr COMMAND sim_csr)

tsr_sim_test(sim_queue test_sim_queue.c tsr_sim tsr_ref)        # issue port, checker order, deps, SYNC, errors, credits
add_test(NAME sim_queue COMMAND sim_queue)

# The transport target is defined in sw/transport (added after sw/sim); names resolve at generate time.
tsr_sim_test(sim_transport test_sim_transport.c tsr_transport tsr_sim tsr_ref)   # sim transport + TSR_* env
add_test(NAME sim_transport COMMAND sim_transport ${TSR_SIM_TEST_SCRATCH})

# UBSan builds of the capture and queue tests (sources compiled in, -fno-sanitize-recover): keep
# the simulator free of undefined behaviour such as qsort(NULL, 0) on a channel without captured
# reads. Skipped when the toolchain cannot link -fsanitize=undefined.
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")
set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
check_c_source_compiles("int main(void) { return 0; }" TSR_HAVE_UBSAN)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LINK_OPTIONS)
if (TSR_HAVE_UBSAN)
    foreach(_t capture queue)
        add_executable(sim_${_t}_ubsan test_sim_${_t}.c ${TSR_REF_SOURCES} ${TSR_SIM_SOURCES})
        tsr_c_target(sim_${_t}_ubsan)
        target_include_directories(sim_${_t}_ubsan PRIVATE ${TSR_SIM_INCLUDE_DIR} ${TSR_SIM_INT_DIR})
        if (TSR_SIM_DEFS)
            target_compile_definitions(sim_${_t}_ubsan PRIVATE ${TSR_SIM_DEFS})
        endif()
        target_compile_options(sim_${_t}_ubsan PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
        target_link_options(sim_${_t}_ubsan PRIVATE -fsanitize=undefined)
        target_link_libraries(sim_${_t}_ubsan PRIVATE Threads::Threads m)
        add_test(NAME sim_${_t}_ubsan COMMAND sim_${_t}_ubsan)
    endforeach()
endif()

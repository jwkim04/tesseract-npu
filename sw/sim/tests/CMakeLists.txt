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
if ("${TSR_SIM_DEFS}" MATCHES "TSR_SIM_HAVE_RACE")               # P1.9: the same with the race detector fatal
    add_test(NAME sim_queue_race COMMAND sim_queue)
    set_tests_properties(sim_queue_race PROPERTIES ENVIRONMENT TSR_SIM_RACE=fatal TIMEOUT 120)
endif()

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

# ---------------------------------------------------------------------------
# P1.8 scheduler (docs/v2/06-software-verification.md §4.4, §4.7, §10.2; sw/sim/tsr_sched.c)
#   sim_sched_unit         xoshiro128** vectors, TSR_SIM_SCHED parsing, options and environment,
#                          listener list and push events, tie-breaking, random durations and
#                          crossings, the CDC lag of the RISC-V-side copies, access events, TSR_SIM_LOG,
#                          ndmreset (stub op handlers; no toolchain needed)
#   sim_sched_determinism  SW §4.7 under fixed + 7 random schedules: the sim_rv_issue firmware at two
#                          RV / core ratios (test_sim_rvhost.h), random tile programs with correct
#                          deps, a racy program that some seed exposes, spin fast-forward on a long
#                          MVM against the run without it. The firmware part reports "Skipped" (77)
#                          only if the RISC-V compiler was missing at configure time.
# ---------------------------------------------------------------------------
tsr_sim_test(sim_sched_unit test_sim_sched_unit.c tsr_sim tsr_ref)
add_test(NAME sim_sched_unit COMMAND sim_sched_unit ${TSR_SIM_TEST_SCRATCH})
set_tests_properties(sim_sched_unit PROPERTIES TIMEOUT 120)

if (TARGET tsr_rv)                                     # the simulator has its RISC-V model (rv.cmake)
    tsr_sim_test(sim_sched_determinism test_sim_sched.c tsr_sim tsr_ref)
    if (TARGET rv_issue_fw)
        add_dependencies(sim_sched_determinism rv_issue_fw)
    endif()
    add_test(NAME sim_sched_determinism COMMAND sim_sched_determinism ${_rv_fw_elf})
    set_tests_properties(sim_sched_determinism PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    # With the race detector built in: the same runs under TSR_SIM_RACE=fatal (the valid programs
    # of parts 1, 2 and 4 are race-free; part 3 turns the detector off for its racy program).
    if ("${TSR_SIM_DEFS}" MATCHES "TSR_SIM_HAVE_RACE")
        add_test(NAME sim_sched_determinism_race COMMAND sim_sched_determinism ${_rv_fw_elf})
        set_tests_properties(sim_sched_determinism_race PROPERTIES ENVIRONMENT TSR_SIM_RACE=fatal
                             SKIP_RETURN_CODE 77 TIMEOUT 300)
    endif()
endif()
# UBSan builds of both (the simulator sources compiled in, as sim_queue_ubsan).
if (TSR_HAVE_UBSAN)
    set(_sched_ubsan sched_unit)
    if (TARGET tsr_rv)
        list(APPEND _sched_ubsan sched)
    endif()
    foreach(_t ${_sched_ubsan})
        set(_n sim_${_t}_ubsan)
        if (_t STREQUAL "sched")
            set(_n sim_sched_determinism_ubsan)
        endif()
        add_executable(${_n} test_sim_${_t}.c ${TSR_REF_SOURCES} ${TSR_SIM_SOURCES})
        tsr_c_target(${_n})
        target_include_directories(${_n} PRIVATE ${TSR_SIM_INCLUDE_DIR} ${TSR_SIM_INT_DIR})
        if (TSR_SIM_DEFS)
            target_compile_definitions(${_n} PRIVATE ${TSR_SIM_DEFS})
        endif()
        target_compile_options(${_n} PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
        target_link_options(${_n} PRIVATE -fsanitize=undefined)
        target_link_libraries(${_n} PRIVATE Threads::Threads m)
    endforeach()
    add_test(NAME sim_sched_unit_ubsan COMMAND sim_sched_unit_ubsan ${TSR_SIM_TEST_SCRATCH})
    set_tests_properties(sim_sched_unit_ubsan PROPERTIES TIMEOUT 300)
    if (TARGET sim_sched_determinism_ubsan)
        if (TARGET rv_issue_fw)
            add_dependencies(sim_sched_determinism_ubsan rv_issue_fw)
        endif()
        add_test(NAME sim_sched_determinism_ubsan COMMAND sim_sched_determinism_ubsan ${_rv_fw_elf})
        set_tests_properties(sim_sched_determinism_ubsan PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    endif()
endif()

# ---------------------------------------------------------------------------
# W1b integration (docs/v2/README.md §10 W1): sim_w1b_e2e — the sim_rv_issue firmware with every W1b
# model attached (DM, DBG, RACE fatal, TRACE capture): fixed + 7 random schedules with equal SW §4.7
# observables and captured traces, the captured launch traces replayed on fresh simulators under 3
# schedules, and a GDB-style DMI session through host CSR 0xC00 + 4r (halt mid-launch, registers
# and SPM / DTCM through SBA, 5 single steps, a hardware breakpoint at the kernel's first push, resume
# to completion) whose run ends with the undisturbed run's observables (DB §5.8, §9.1). Added only
# when the library has the RISC-V model and all four W1b models; "Skipped" (77) without the
# firmware (no RISC-V compiler at configure time).
# ---------------------------------------------------------------------------
if (TARGET tsr_rv AND "${TSR_SIM_DEFS}" MATCHES "TSR_SIM_HAVE_DM" AND "${TSR_SIM_DEFS}" MATCHES "TSR_SIM_HAVE_DBG"
    AND "${TSR_SIM_DEFS}" MATCHES "TSR_SIM_HAVE_RACE" AND "${TSR_SIM_DEFS}" MATCHES "TSR_SIM_HAVE_TRACE")
    tsr_sim_test(sim_w1b_e2e test_sim_w1b_e2e.c tsr_sim tsr_ref)
    if (TARGET rv_issue_fw)
        add_dependencies(sim_w1b_e2e rv_issue_fw)
    endif()
    add_test(NAME sim_w1b_e2e COMMAND sim_w1b_e2e ${_rv_fw_elf} ${TSR_SIM_TEST_SCRATCH})
    set_tests_properties(sim_w1b_e2e PROPERTIES ENVIRONMENT TSR_SIM_RACE=fatal SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()

# Tests whose assertions include the fixed schedule's timing and that leave tsr_sim_opts.sched
# unset: an exported TSR_SIM_SCHED (SW §4.9) must not change them (W1b fix; the sw/sim ones are
# pinned in sw/sim/CMakeLists.txt).
foreach(_t sim_csr sim_queue sim_queue_race sim_queue_ubsan)
    if (TEST ${_t})
        set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT "TSR_SIM_SCHED=")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# RISC-V cycle model (docs/v2/06-software-verification.md §4.2; docs/v2/04-control-core.md §3.2,
# V2-CC-34): rv_timing runs test_rv_isa's cycle-table checks alone, for both Ibex configurations
# ("maxperf", the default, and "small"): per-class costs, the load-use and 32-bit-target penalties,
# the CSR pipeline flush, the class statistics, the crossing latencies at three clock ratios.
# (rv_unit runs the same checks among all the others; test_rv_isa is defined by sw/sim/rv.cmake.)
# ---------------------------------------------------------------------------
if (TARGET test_rv_isa)
    add_test(NAME rv_timing COMMAND test_rv_isa --timing)
endif()

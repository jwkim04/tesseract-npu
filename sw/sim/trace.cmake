# sw/sim/trace.cmake — trace format in the simulator (docs/design/06-software-verification.md
# §5, SW-5): tsr_trace.c (read / write / strict validation of trace directories, the EX §3.2
# image id, the TRACE model that captures one `launch` trace per launch for TSR_TRACE and
# tsr_sim_capture_launch / _take, and replay on a fresh simulator with the SW §5.4 comparisons;
# API in include/tsr_trace.h).
#
# Included by sw/sim/CMakeLists.txt with the other optional snippets, after tsr_sim /
# tsr_sim_shared exist. It adds tsr_trace.c to both targets and to TSR_SIM_SOURCES;
# sw/sim/CMakeLists.txt then defines TSR_SIM_HAVE_TRACE, so tsr_sim_create_opts attaches the model
# (tsr_trace_attach). The tests are defined at the end of the directory (cmake_language DEFER),
# when TSR_SIM_SOURCES / TSR_SIM_DEFS and rv.cmake's firmware target are complete.
#
# Tests (run: ctest --test-dir build/dev -R '^(sim_trace|trace_v2)' --output-on-failure):
#   sim_trace           tests/test_trace.c: SHA-256 / EX §3.2 image ids (known answers, .tsr_id
#                       zeroing), field-by-field write / read round trips of both kinds, every
#                       validation error on read, ops.load placement, relative tickets, tile-only
#                       runs (tsr_trace2_run_ops) and replays with detected corruptions
#   sim_trace_ubsan     the same, built with -fsanitize=undefined (sources compiled in)
#   trace_v2_roundtrip  tests/test_trace_roundtrip.c: the sim_rv_issue firmware's launches captured
#                       (tsr_sim_capture_launch and TSR_TRACE), written as a set, read back and
#                       replayed on freshly booted simulators under the fixed and 2 random
#                       schedules; a corrupted expect byte, push record and tickets field are each
#                       detected. Skipped (77) without the RISC-V compiler (no fw_issue.elf)

set(TSR_TRACE_SOURCES ${CMAKE_CURRENT_LIST_DIR}/tsr_trace.c)

foreach(_tgt tsr_sim tsr_sim_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_TRACE_SOURCES})
    endif()
endforeach()
list(APPEND TSR_SIM_SOURCES ${TSR_TRACE_SOURCES})

function(_tsr_trace_tests)
    set(_tr_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set(_tr_scratch ${CMAKE_CURRENT_BINARY_DIR}/tests/scratch)
    find_package(Threads REQUIRED)

    add_executable(sim_trace ${_tr_tests}/test_trace.c)
    tsr_c_target(sim_trace)
    target_include_directories(sim_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${_tr_tests})
    target_link_libraries(sim_trace PRIVATE tsr_sim tsr_ref Threads::Threads m)
    add_test(NAME sim_trace COMMAND sim_trace ${_tr_scratch})
    set_tests_properties(sim_trace PROPERTIES TIMEOUT 120)

    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
    check_c_source_compiles("int main(void) { return 0; }" TSR_TRACE_HAVE_UBSAN)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
    if (TSR_TRACE_HAVE_UBSAN)
        add_executable(sim_trace_ubsan ${_tr_tests}/test_trace.c ${TSR_REF_SOURCES} ${TSR_SIM_SOURCES})
        tsr_c_target(sim_trace_ubsan)
        target_include_directories(sim_trace_ubsan PRIVATE ${TSR_SIM_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${_tr_tests})
        set(_defs ${TSR_SIM_DEFS})
        if (NOT "TSR_SIM_HAVE_TRACE=1" IN_LIST _defs)
            list(APPEND _defs TSR_SIM_HAVE_TRACE=1)
        endif()
        target_compile_definitions(sim_trace_ubsan PRIVATE ${_defs})
        target_compile_options(sim_trace_ubsan PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
        target_link_options(sim_trace_ubsan PRIVATE -fsanitize=undefined)
        target_link_libraries(sim_trace_ubsan PRIVATE Threads::Threads m)
        add_test(NAME sim_trace_ubsan COMMAND sim_trace_ubsan ${_tr_scratch})
        set_tests_properties(sim_trace_ubsan PROPERTIES TIMEOUT 300)
    endif()

    # The sim_rv_issue firmware (rv.cmake builds it when the RISC-V compiler exists).
    set(_tr_fw ${CMAKE_CURRENT_BINARY_DIR}/rv_issue/fw_issue.elf)
    if (NOT EXISTS ${_tr_tests}/test_trace_roundtrip.c)
        return()
    endif()
    add_executable(trace_v2_roundtrip ${_tr_tests}/test_trace_roundtrip.c)
    tsr_c_target(trace_v2_roundtrip)
    target_include_directories(trace_v2_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${_tr_tests})
    target_link_libraries(trace_v2_roundtrip PRIVATE tsr_sim tsr_ref Threads::Threads m)
    if (TARGET rv_issue_fw)
        add_dependencies(trace_v2_roundtrip rv_issue_fw)
    endif()
    add_test(NAME trace_v2_roundtrip COMMAND trace_v2_roundtrip ${_tr_fw} ${_tr_scratch})
    set_tests_properties(trace_v2_roundtrip PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endfunction()
cmake_language(DEFER CALL _tsr_trace_tests)

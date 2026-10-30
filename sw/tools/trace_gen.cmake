# sw/tools/trace_gen.cmake — tsr-trace-gen v2 (P1.10; docs/v2/06-software-verification.md §5.5,
# §5.6, §10.2), included by sw/tools/CMakeLists.txt when present.
#   tsr-trace-gen             random tile programs as trace v2 `ops` traces (--ops gen|m2|m3|vu|at|ctrl)
#   tools_trace_gen_replay    a small set per --ops mode with a fixed seed, generated twice (byte-identical),
#                             each trace replayed under the fixed and two random schedules (the
#                             generator's --check-sched, and tsr-trace-replay when it is built)
# The trace sets themselves come from traces/{m2,gen,m3,vu,at,ctrl}/regen.sh.

add_executable(tsr-trace-gen ${CMAKE_CURRENT_SOURCE_DIR}/tsr_trace_gen.c)
set_target_properties(tsr-trace-gen PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(tsr-trace-gen PRIVATE ${TSR_FP_FLAGS}
    -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wdouble-promotion)
# tsr_sim_int.h (simulator-internal): the checker classes give every op's operand extents, from
# which the generator derives its dependencies, and the queue listener checks each completion.
target_include_directories(tsr-trace-gen PRIVATE ${TESSERACT_ROOT}/sw/sim)
target_link_libraries(tsr-trace-gen PRIVATE tsr_sim tsr_ref m)

find_program(TSR_BASH_TRACE_GEN bash)
if (TSR_BASH_TRACE_GEN)
    if (TARGET tsr-trace-replay)
        set(_tsr_replay $<TARGET_FILE:tsr-trace-replay>)
    else()
        set(_tsr_replay "")
    endif()
    add_test(NAME tools_trace_gen_replay
             COMMAND ${TSR_BASH_TRACE_GEN} ${CMAKE_CURRENT_SOURCE_DIR}/test_trace_gen.sh
                     $<TARGET_FILE:tsr-trace-gen> ${CMAKE_CURRENT_BINARY_DIR}/trace_gen_test "${_tsr_replay}")
    # The race detector checks every run (fatal: a race aborts), when the library has it (P1.9).
    set(_tsr_gen_env "TSR_SIM_THREADS=1")
    if (EXISTS ${TESSERACT_ROOT}/sw/sim/race.cmake AND NOT "race" IN_LIST TSR_SIM_SKIP_SNIPPETS)
        list(APPEND _tsr_gen_env "TSR_SIM_RACE=fatal")
    endif()
    set_tests_properties(tools_trace_gen_replay PROPERTIES TIMEOUT 120 ENVIRONMENT "${_tsr_gen_env}")
endif()

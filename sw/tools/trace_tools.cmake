# sw/tools/trace_tools.cmake — trace tools (docs/design/06-software-verification.md §5.4,
# §5.6, §9.5), included by sw/tools/CMakeLists.txt when present (before trace_gen.cmake, whose test
# uses tsr-trace-replay).
#   tsr-trace-replay   replays trace directories or sets on fresh simulators (tsr_trace.h):
#                      tsr-trace-replay [-q | -v] [--sched S]... [--threads N] [--max-cycles N]
#                      [--tmax N] PATH...; summary line, exit 0 / 1 (mismatch) / 2 (unreadable);
#                      tsr-trace-replay --device KIND [--shadow] [--replay-image ELF] [--serial] PATH...
#                      replays through a transport (tsr_devrun.h; linked by sw/tools/CMakeLists.txt)
#   tsr-trace-conv     cmd.bin traces -> kind-ops traces, serial dependency chain (SW §5.6):
#                      tsr-trace-conv [-q] [--replay] [--sched S]... [--limit N] [--pick LIST]
#                      -o OUT IN...
# Tests:
#   trace_conv_model       10 traces of traces/model (both models, Q8_0 and Q4_K MVMs, single and
#                          two-MVM splits) converted into the build tree and replayed under the
#                          fixed schedule and a random one: the units' results equal the traces'
#                          expected results bit for bit. Skipped (77) when traces/model is not on disk
#                          (a recorded set this tree cannot regenerate)
#   trace_conv_model_full  (LABELS slow) all 290 traces converted into the build tree and replayed
# The converted set itself is traces/model-v1conv (traces/model-v1conv/regen.sh).

# ---------------------------------------------------------------------------
# Script mode: cmake -DTSR_TRACE_CONV_TEST=1 -DCONV=<tsr-trace-conv> -DIN=<traces/model>
#              -DOUT=<dir> [-DPICK=<list>] [-DSCHED=<list>] -P trace_tools.cmake
# ---------------------------------------------------------------------------
if (TSR_TRACE_CONV_TEST)
    if (NOT EXISTS "${IN}")
        message("trace_conv_model: ${IN} not on disk (a recorded set this tree cannot regenerate); skipped")
        cmake_language(EXIT 77)
    endif()
    file(REMOVE_RECURSE "${OUT}")
    set(_args -q --replay)
    foreach(_s ${SCHED})
        list(APPEND _args --sched ${_s})
    endforeach()
    if (PICK)
        list(APPEND _args --pick "${PICK}")
    endif()
    execute_process(COMMAND "${CONV}" ${_args} -o "${OUT}" "${IN}" RESULT_VARIABLE _r)
    if (NOT _r EQUAL 0)
        message(FATAL_ERROR "trace_conv_model: tsr-trace-conv failed (${_r})")
    endif()
    file(REMOVE_RECURSE "${OUT}")
    return()
endif()

foreach(_tool replay conv)
    add_executable(tsr-trace-${_tool} ${CMAKE_CURRENT_SOURCE_DIR}/tsr_trace_${_tool}.c)
    set_target_properties(tsr-trace-${_tool} PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(tsr-trace-${_tool} PRIVATE ${TSR_FP_FLAGS}
        -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wdouble-promotion)
    target_link_libraries(tsr-trace-${_tool} PRIVATE tsr_sim tsr_ref m)
endforeach()

set(_tsr_model_sample
    "qwen05b-q8_0/000000,qwen05b-q8_0/000001,qwen05b-q8_0/000003,qwen05b-q8_0/000004,qwen05b-q8_0/000005,qwen05b-q4_k_m-requant/000001,qwen05b-q4_k_m-requant/000002,qwen05b-q4_k_m-requant/000006,qwen05b-q4_k_m-requant/000022,qwen05b-q4_k_m-requant/000023")
add_test(NAME trace_conv_model
         COMMAND ${CMAKE_COMMAND} -DTSR_TRACE_CONV_TEST=1 -DCONV=$<TARGET_FILE:tsr-trace-conv>
                 -DIN=${TESSERACT_ROOT}/traces/model -DOUT=${CMAKE_CURRENT_BINARY_DIR}/trace_conv_model
                 "-DPICK=${_tsr_model_sample}" "-DSCHED=fixed;random:1"
                 -P ${CMAKE_CURRENT_LIST_FILE})
set_tests_properties(trace_conv_model PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")
add_test(NAME trace_conv_model_full
         COMMAND ${CMAKE_COMMAND} -DTSR_TRACE_CONV_TEST=1 -DCONV=$<TARGET_FILE:tsr-trace-conv>
                 -DIN=${TESSERACT_ROOT}/traces/model -DOUT=${CMAKE_CURRENT_BINARY_DIR}/trace_conv_model_full
                 "-DSCHED=fixed" -P ${CMAKE_CURRENT_LIST_FILE})
set_tests_properties(trace_conv_model_full PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 LABELS slow
                     ENVIRONMENT "TSR_SIM_THREADS=4")

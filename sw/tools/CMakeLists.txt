# sw/tools — trace tools (spec 검증 전략, 트레이스 형식):
#   tsr-trace-replay  replay one or many trace directories on the ISA simulator; exit code = failures
#   tsr-trace-gen     random but valid M1 programs with random weights as trace directories (RTL regression)
#   klcheck           model-level KL check vs the ggml CPU backend (spec 검증 전략, 모델 단위 KL 기준;
#                     docs/klcheck.md). Links llama, so it is only built when the llama targets exist.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

function(tsr_tool name src)
    add_executable(${name} ${src})
    set_target_properties(${name} PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(${name} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes)
    target_link_libraries(${name} PRIVATE tsr_trace tsr_sim tsr_ref m)
endfunction()

tsr_tool(tsr-trace-replay tsr_trace_replay.c)
tsr_tool(tsr-trace-gen    tsr_trace_gen.c)

# Tests: generate traces (T_MAX 1 and 8, with error endings, 1..4 channels), replay them all,
# then corrupt one expect byte and check that replay reports exactly one failure.
find_program(TSR_BASH bash)
if (TSR_BASH)
    add_test(NAME tools_trace_gen_replay
             COMMAND ${TSR_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/test_tools.sh
                     $<TARGET_FILE:tsr-trace-gen> $<TARGET_FILE:tsr-trace-replay>
                     ${CMAKE_CURRENT_BINARY_DIR}/tools_test_traces)
endif()

# Regenerate the checked-in RTL regression set under traces/gen (not part of the default build):
#   cmake --build <build> --target tsr-traces
add_custom_target(tsr-traces
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${TESSERACT_ROOT}/traces/gen
    COMMAND tsr-trace-gen -q -o ${TESSERACT_ROOT}/traces/gen/tmax1 -n 24 -s 1 --max-k 2048 --max-n 96 --errors 15
    COMMAND tsr-trace-gen -q -o ${TESSERACT_ROOT}/traces/gen/tmax8 -n 8 -s 2 --t-max 8 --max-k 2048 --max-n 96
    COMMAND tsr-trace-replay -q ${TESSERACT_ROOT}/traces/gen
    COMMENT "Generating RTL regression traces into traces/gen"
    VERBATIM)

# klcheck links llama / ggml, which the top-level CMakeLists adds after this directory
# (TSR_WITH_LLAMA). The target is therefore created by a call deferred to the end of the top-level
# directory, where `if (TARGET llama)` sees the final set of targets.
function(tsr_add_klcheck)
    if (NOT TARGET llama OR NOT TARGET ggml)
        message(STATUS "klcheck: llama targets not available, klcheck is not built")
        return()
    endif()
    add_executable(klcheck ${TESSERACT_ROOT}/sw/tools/klcheck.cpp)
    set_target_properties(klcheck PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(klcheck PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    target_link_libraries(klcheck PRIVATE llama ggml)
    if (TARGET ggml-base)
        target_link_libraries(klcheck PRIVATE ggml-base)          # gguf_*, ggml_get_type_traits
    endif()
    # No model needed: file format, resume, the 1-ulp perturbation and the KL arithmetic.
    add_test(NAME tools_klcheck_selftest
             COMMAND klcheck selftest ${CMAKE_BINARY_DIR}/sw/tools/klcheck_selftest)
    # The model-level criterion itself on Qwen2.5-0.5B Q8_0 (only if the model is present): device
    # (sim) vs CPU vs 1-ulp floor, 4 built-in prompts x 32 tokens, plus the embd-path identity
    # check. ~15-30 s. Output goes to a private temporary directory.
    set(_tsr_kl_model ${TESSERACT_ROOT}/models/qwen2.5-0.5b-instruct-q8_0.gguf)
    if (EXISTS ${_tsr_kl_model})
        add_test(NAME tools_klcheck_qwen05b
                 COMMAND klcheck all -m ${_tsr_kl_model} --device sim --n-prompts 4 -n 32)
        set_tests_properties(tools_klcheck_qwen05b PROPERTIES TIMEOUT 900 LABELS model)
    endif()
endfunction()
cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_klcheck)

# tsr-numcheck (docs/numcheck.md): one-prompt numeric dump of every graph node for three runs
# (cpu, floor = cpu + 1-ulp input, dev = Tesseract), compared by numcheck.py. Links llama like
# klcheck, so it is created by a deferred call as well.
function(tsr_add_numcheck)
    if (NOT TARGET llama OR NOT TARGET ggml)
        message(STATUS "numcheck: llama targets not available, tsr-numcheck is not built")
        return()
    endif()
    add_executable(tsr-numcheck ${TESSERACT_ROOT}/sw/tools/numcheck.cpp)
    set_target_properties(tsr-numcheck PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(tsr-numcheck PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    target_link_libraries(tsr-numcheck PRIVATE llama ggml)
    target_include_directories(tsr-numcheck PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/vendor)   # nlohmann/json.hpp
    if (TARGET ggml-base)
        target_link_libraries(tsr-numcheck PRIVATE ggml-base)     # gguf_*, ggml_get_type_traits, fp16
    endif()
    # No model: gather of views / F16 / BF16 / Q8_0 / I32 against ggml's CPU CONT, name and option parsing.
    add_test(NAME tools_numcheck_dump_selftest COMMAND tsr-numcheck selftest)
    # Qwen2.5-0.5B Q8_0 (if present): cpu, floor, dev (sim), cpu-tok and dev-noobs runs, then selfcheck
    # (identical node lists, observation invariance, embd == token path, weight MUL_MATs on
    # Tesseract). ~10 s. Output goes to a private temporary directory.
    set(_tsr_nc_model ${TESSERACT_ROOT}/models/qwen2.5-0.5b-instruct-q8_0.gguf)
    if (EXISTS ${_tsr_nc_model})
        add_test(NAME tools_numcheck_dump_qwen05b
                 COMMAND tsr-numcheck all -m ${_tsr_nc_model} -T 32 --checks full)
        set_tests_properties(tools_numcheck_dump_qwen05b PROPERTIES TIMEOUT 600 LABELS model ENVIRONMENT TSR_SIM_THREADS=4)
    endif()
endfunction()
cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_numcheck)

# numcheck_helper (numcheck.py: ggml conversions + libtsr_ref B1 loop), see numcheck_helper.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/numcheck_helper.cmake)

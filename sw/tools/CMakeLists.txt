# sw/tools — tools (spec 검증 전략):
#   klcheck           model-level KL check vs the ggml CPU backend (spec 검증 전략, 모델 단위 KL 기준;
#                     docs/klcheck.md). Links llama, so it is only built when the llama targets exist.
#   tsr-numcheck      per-node numeric dump (docs/numcheck.md); links llama as well
#   tsr_xrt_selftest  card bring-up checklist over any transport (docs/xrt-host.md)
#   tsr_bw            M0 bandwidth tool (XRT builds, once tsr_bw.cpp exists)
# X0 deleted the v1 trace tools tsr-trace-gen / tsr-trace-replay (cmd.bin programs), their
# tools_trace_gen_replay ctest and the tsr-traces target; tsr-trace-gen / -replay / -conv return
# for trace v2 in P1.10 (docs/v2/06-software-verification.md §5, §9.5), the generator from
# v1-m4:sw/tools/tsr_trace_gen.c and the converter's v1 reader from v1-m4:sw/sim/tsr_trace.c
# (EX §12 "X0 as done"). The v1 binaries are in build/v1 (traces/v1_tools.sh).

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

# tsr_xrt_selftest (docs/xrt-host.md): card bring-up checklist on any transport (xrt on the card;
# mock here, without hardware). Built in every configuration so the mock checks always run.
# X0 reduced it to identity and memory (no host-written programs, D15); it becomes tsr-selftest
# with the `selftest` kernel in P3.3.
add_executable(tsr_xrt_selftest tsr_xrt_selftest.cpp)
set_target_properties(tsr_xrt_selftest PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(tsr_xrt_selftest PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_link_libraries(tsr_xrt_selftest PRIVATE tsr_transport)
# The checklist on the adapter register model: 4 channels T_MAX 1 with DMA rules that force the
# bounce / read-modify-write path; T_MAX 8 (the tsr_u200_k4 configuration) with 64-byte DMA
# alignment; 1 channel (tsr_u200_k1). The trace replays (traces/m2, traces/gen), the run-fault
# test (tools_xrt_selftest_mock_faults) and the M3 replays (tools_xrt_selftest_mock_m3, _m3_tmax8)
# are removed in X0 (SW §10.2): the first two return as tools_selftest_mock* in P3.3, the M3 replays
# as tools_selftest_mock_traces over the regenerated v2 m3 set.
add_test(NAME tools_xrt_selftest_mock COMMAND tsr_xrt_selftest --device mock --mem-mib 8)
set_tests_properties(tools_xrt_selftest_mock PROPERTIES
    ENVIRONMENT "TSR_MOCK_DMA_ALIGN=4096;TSR_MOCK_DMA_CHUNK_KIB=1024;TSR_SIM_THREADS=4" TIMEOUT 600)
add_test(NAME tools_xrt_selftest_mock_tmax8 COMMAND tsr_xrt_selftest --device mock --mem-mib 8)
set_tests_properties(tools_xrt_selftest_mock_tmax8 PROPERTIES
    ENVIRONMENT "TSR_MOCK_TMAX=8;TSR_MOCK_DMA_ALIGN=64;TSR_SIM_THREADS=4" TIMEOUT 600)
add_test(NAME tools_xrt_selftest_mock_1ch COMMAND tsr_xrt_selftest --device mock --mem-mib 0)
set_tests_properties(tools_xrt_selftest_mock_1ch PROPERTIES ENVIRONMENT "TSR_MOCK_NUM_CH=1;TSR_SIM_THREADS=4")

# tsr_bw: M0 host <-> card and kernel bandwidth tool (XRT only; the source is added by the M0
# work, so the target exists only once the file does).
if (TSR_WITH_XRT AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tsr_bw.cpp)
    add_executable(tsr_bw tsr_bw.cpp)
    set_target_properties(tsr_bw PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(tsr_bw PRIVATE -Wall -Wextra)
    target_link_libraries(tsr_bw PRIVATE tsr_xrt tsr_transport)
endif()

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
    # tools_klcheck_qwen05b (the criterion on Qwen2.5-0.5B Q8_0, device = sim) is removed in X0:
    # no op runs on the device until the op-list path, so its "device" run would be a CPU run. It
    # returns with the v2 settings (FA on, q8_0 KV, rotation off) at P3.4 (SW §10.2, §10.4).
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
    # tools_numcheck_dump_qwen05b (Qwen2.5-0.5B Q8_0: cpu, floor, dev (sim) runs and a selfcheck that
    # requires the weight MUL_MATs on Tesseract) is removed in X0 until the op-list path; it returns
    # with the v2 settings at P3.4 (SW §10.2).
endfunction()
cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_numcheck)

# numcheck_helper (numcheck.py: ggml conversions + libtsr_ref B1 loop), see numcheck_helper.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/numcheck_helper.cmake)

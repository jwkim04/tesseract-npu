# sw/tools — tools (docs/design/06-software-verification.md "SW" §9.5, §10.2):
#   tsr-selftest      card bring-up checklist over any transport (tsr_selftest.cpp):
#                     identity, memory, boot, PING latency / rate, the `selftest` kernel with the in-process
#                     shadow simulator, trace replays through the device, DMI smoke, log and status
#   tsr-log           decode raw device log words with a firmware ELF (tsr_log.cpp, DB §6.1)
#   tsr_devrun        launch targets (a transport or the shadow simulator) and trace replays through
#                     them (tsr_devrun.h): tsr-selftest and tsr-trace-replay --device
#   tsr-trace-gen / -replay / -conv   trace tools (trace_gen.cmake, trace_tools.cmake; P1.10)
#   klcheck           model-level KL check vs the ggml CPU backend (docs/klcheck.md). Links llama, so it is
#                     only built when the llama targets exist
#   tsr-numcheck      per-node numeric dump (docs/numcheck.md); links llama as well
#   tsr-mtp           multi-token prediction driver and checker (tsr_mtp.cpp, SW §7.11): llama.cpp's MTP drafter loop with
#                     the device counters, plain / speculative modes; links llama-common
#   tsr_bw            M0 bandwidth tool (XRT builds, once tsr_bw.cpp exists)
#   tsr_dbg, tsr-dbg, tsr-prof   libtsr_dbg (GDB RSP server over the DMI window, tile-level debug, the
#                     trace writer) and its tools (dbg/dbg.cmake; P3.7, P3.8)
# klcheck and numcheck run with the SW-22 settings (SW-22, SW §10.4): FA on, q8_0 / q8_0 KV,
# LLAMA_ATTN_ROT_DISABLE=1, offload_kqv false in M6 (--phase m6, the default) and true in M7.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()
find_program(TSR_BASH_TOOLS bash)

# The versions the selftest tests expect (SW §6.9): TSR_BASIS_VERSION_STRING from tsr/version.h and
# TSR_CAP_STRING from tsr_tables.h VERSION, as the firmware links get them.
include(${TESSERACT_ROOT}/sw/ctrl/cmake/tsr_version.cmake)
tsr_read_versions(${TESSERACT_ROOT})

# The trace library is part of tsr_sim only with sw/sim/trace.cmake (TSR_SIM_SKIP_SNIPPETS may
# leave it out); the tools below that read traces or run the shadow need it.
set(_tsr_have_trace FALSE)
if (EXISTS ${TESSERACT_ROOT}/sw/sim/trace.cmake AND NOT "trace" IN_LIST TSR_SIM_SKIP_SNIPPETS)
    set(_tsr_have_trace TRUE)
endif()

# ---------------------------------------------------------------------------
# tsr_devrun (P3.3): launch targets and trace replays through a transport (tsr_devrun.h). C11 with
# one C++ file (the inner simulator of the sim / mock kinds is reached through tsr_kdev.h).
# Internal headers: sw/sim/tsr_sim_int.h (the checker classes: trace fit test, the replay
# kernel's SPM buffer) and sw/ctrl/ref/fw_sim_host.hpp (the shadow's host loader).
# ---------------------------------------------------------------------------
if (_tsr_have_trace)
    add_library(tsr_devrun STATIC tsr_devrun.c tsr_devrun_cfg.cpp)
    set_target_properties(tsr_devrun PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(tsr_devrun PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow
        $<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes -Wmissing-prototypes>)
    target_include_directories(tsr_devrun PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/ctrl/ref)
    target_link_libraries(tsr_devrun PUBLIC tsr_transport tsr_sim tsr_ref)
endif()

# P1.10 trace tools, each snippet owned by its package: trace_tools.cmake (tsr-trace-replay,
# tsr-trace-conv) and trace_gen.cmake (tsr-trace-gen). Both link the trace library and are left
# out without it, and each one by its own name in TSR_SIM_SKIP_SNIPPETS.
if (_tsr_have_trace)
    foreach(_snip trace_tools trace_gen)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_snip}.cmake AND NOT _snip IN_LIST TSR_SIM_SKIP_SNIPPETS)
            include(${CMAKE_CURRENT_SOURCE_DIR}/${_snip}.cmake)
        endif()
    endforeach()
endif()
# P3.3: tsr-trace-replay --device KIND (replays through a transport, tsr_devrun.h).
if (TARGET tsr-trace-replay AND TARGET tsr_devrun)
    target_link_libraries(tsr-trace-replay PRIVATE tsr_devrun)
    set_target_properties(tsr-trace-replay PROPERTIES LINKER_LANGUAGE CXX)
endif()

# ---------------------------------------------------------------------------
# tsr-log (SW §9.5, DB §6.1): the transport's decoder (sw/transport/tsr_logdec.h) over saved words.
# ---------------------------------------------------------------------------
add_executable(tsr-log tsr_log.cpp)
set_target_properties(tsr-log PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(tsr-log PRIVATE -Wall -Wextra -Wshadow)
target_link_libraries(tsr-log PRIVATE tsr_transport tsr_elf tsr_fw_paths)

# ---------------------------------------------------------------------------
# tsr-selftest (SW §9.5, §7.6, SW-28): the card checklist; mock / sim here, xrt on the card.
# ---------------------------------------------------------------------------
if (TARGET tsr_devrun)
    add_executable(tsr-selftest tsr_selftest.cpp)
    set_target_properties(tsr-selftest PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(tsr-selftest PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
    target_include_directories(tsr-selftest PRIVATE ${TESSERACT_ROOT}/sw/ctrl/ref)   # selftest_abi.hpp
    target_link_libraries(tsr-selftest PRIVATE tsr_devrun tsr_transport tsr_elf tsr_sim tsr_ref tsr_dbg)

    # The checklist on the adapter register model (SW §10.2: was tools_xrt_selftest_mock*), each with
    # the shadow, the DMI smoke and the log written for tsr-log: 4 channels T_MAX 1 with DMA rules that
    # force the bounce / read-modify-write path; T_MAX 8 (the tsr_u200_k4 configuration) with 64-byte
    # DMA alignment; 1 channel (tsr_u200_k1).
    set(_st_common --device mock --image ${TSR_FW_ELF} --debug --rate-n 200)
    add_test(NAME tools_selftest_mock
             COMMAND tsr-selftest ${_st_common} --mem-mib 8 --log-out ${CMAKE_CURRENT_BINARY_DIR}/selftest_mock.log)
    set_tests_properties(tools_selftest_mock PROPERTIES
        ENVIRONMENT "TSR_MOCK_DMA_ALIGN=4096;TSR_MOCK_DMA_CHUNK_KIB=1024;TSR_SIM_THREADS=4" TIMEOUT 600)
    add_test(NAME tools_selftest_mock_tmax8 COMMAND tsr-selftest ${_st_common} --mem-mib 8)
    set_tests_properties(tools_selftest_mock_tmax8 PROPERTIES
        ENVIRONMENT "TSR_MOCK_TMAX=8;TSR_MOCK_DMA_ALIGN=64;TSR_SIM_THREADS=4" TIMEOUT 600)
    add_test(NAME tools_selftest_mock_1ch COMMAND tsr-selftest ${_st_common} --mem-mib 0)
    set_tests_properties(tools_selftest_mock_1ch PROPERTIES ENVIRONMENT "TSR_MOCK_NUM_CH=1;TSR_SIM_THREADS=4" TIMEOUT 600)
    if (TSR_BASH_TOOLS)
        # The checklist on the core of the reduced first card image tsr_u200_k4 (no VU, no attention: CAPS 0,
        # no VU queue, T_MAX 8; docs/platform-u200.md §8.7), on sim and mock (test_selftest_k4min.sh).
        add_test(NAME tools_selftest_sim_k4min
                 COMMAND ${TSR_BASH_TOOLS} ${CMAKE_CURRENT_SOURCE_DIR}/test_selftest_k4min.sh
                         $<TARGET_FILE:tsr-selftest> ${TSR_FW_ELF} ${CMAKE_CURRENT_BINARY_DIR}/selftest_k4min)
        set_tests_properties(tools_selftest_sim_k4min PROPERTIES TIMEOUT 600)
        # The selftest must fail on what the protocol reports (test_selftest_faults.sh): an AXI error
        # during a launch (PLAT_ERR), ADAPTER_STATUS left by an earlier process, a TCM write fault
        # (VERIFY), a hung unit (recovery ladder, "card reset needed"); and pass without them.
        add_test(NAME tools_selftest_mock_faults
                 COMMAND ${TSR_BASH_TOOLS} ${CMAKE_CURRENT_SOURCE_DIR}/test_selftest_faults.sh
                         $<TARGET_FILE:tsr-selftest> ${TSR_FW_ELF} ${CMAKE_CURRENT_BINARY_DIR}/selftest_faults
                         ${TSR_BASIS_VERSION_STRING} ${TSR_CAP_STRING} $<TARGET_FILE:tsr-elf>)
        set_tests_properties(tools_selftest_mock_faults PROPERTIES TIMEOUT 600 ENVIRONMENT "TSR_SIM_THREADS=2")
        # Replays of the trace sets through mock with the shadow (test_selftest_traces.sh): every trace of traces/m3, vu, at, ctrl and model-v1conv (SERIAL)
        # that fits the device, at T_MAX 1 (the U200 core) and T_MAX 8 (k4). The sets are not in
        # git (traces/*/regen.sh): skipped (77) when none is on disk. 690 traces in about 2 s.
        add_test(NAME tools_selftest_mock_traces
                 COMMAND ${TSR_BASH_TOOLS} ${CMAKE_CURRENT_SOURCE_DIR}/test_selftest_traces.sh
                         $<TARGET_FILE:tsr-selftest> ${TSR_FW_ELF} ${TESSERACT_ROOT}/traces
                         ${TSR_BASIS_VERSION_STRING} ${TSR_CAP_STRING})
        set_tests_properties(tools_selftest_mock_traces PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900
                             ENVIRONMENT "TSR_SIM_THREADS=4")
        add_test(NAME tools_selftest_mock_traces_tmax8
                 COMMAND ${TSR_BASH_TOOLS} ${CMAKE_CURRENT_SOURCE_DIR}/test_selftest_traces.sh
                         $<TARGET_FILE:tsr-selftest> ${TSR_FW_ELF} ${TESSERACT_ROOT}/traces
                         ${TSR_BASIS_VERSION_STRING} ${TSR_CAP_STRING})
        set_tests_properties(tools_selftest_mock_traces_tmax8 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900
                             ENVIRONMENT "TSR_MOCK_TMAX=8;TSR_MOCK_DMA_ALIGN=64;TSR_SIM_THREADS=4")
        # tsr-log end to end: the selftest kernel's id 9 on sim, its log words saved and decoded with
        # the ELF; then the decoder's own checks (every record type, malformed records, resync).
        add_test(NAME tools_log_selftest
                 COMMAND ${TSR_BASH_TOOLS} ${CMAKE_CURRENT_SOURCE_DIR}/test_log.sh
                         $<TARGET_FILE:tsr-log> $<TARGET_FILE:tsr-selftest> ${TSR_FW_ELF} ${CMAKE_CURRENT_BINARY_DIR}/log_test)
        set_tests_properties(tools_log_selftest PROPERTIES TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")
        # tsr-trace-replay --device: a small generated set (m3, vu, at, ctrl; errors on) through sim with
        # the shadow and through mock; a corrupted expect byte must fail; kind launch traces captured on
        # sim (TSR_TRACE) while tsr-selftest runs, replayed as full launches through mock.
        if (TARGET tsr-trace-gen AND TARGET tsr-trace-replay)
            add_test(NAME tools_trace_replay_device
                     COMMAND ${TSR_BASH_TOOLS} ${CMAKE_CURRENT_SOURCE_DIR}/test_trace_replay_device.sh
                             $<TARGET_FILE:tsr-trace-gen> $<TARGET_FILE:tsr-trace-replay> ${TSR_FW_ELF}
                             ${CMAKE_CURRENT_BINARY_DIR}/trace_replay_device $<TARGET_FILE:tsr-selftest>)
            set_tests_properties(tools_trace_replay_device PROPERTIES TIMEOUT 600 ENVIRONMENT "TSR_SIM_THREADS=2")
        endif()
    endif()
endif()

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
    # tsr_transport.h (the struct tsr_versions of the backend's versions hook, SW §6.9), no library
    target_include_directories(klcheck PRIVATE ${TESSERACT_ROOT}/sw/transport ${TESSERACT_ROOT}/sw/sim/include)
    target_link_libraries(klcheck PRIVATE llama ggml)
    if (TARGET ggml-base)
        target_link_libraries(klcheck PRIVATE ggml-base)          # gguf_*, ggml_get_type_traits
    endif()
    # No model needed: file format (format version 2 header with the settings, version 1 files still read), resume, the
    # 1-ulp perturbation and the KL arithmetic.
    add_test(NAME tools_klcheck_selftest
             COMMAND klcheck selftest ${CMAKE_BINARY_DIR}/sw/tools/klcheck_selftest)
    # The criterion on Qwen2.5-0.5B Q8_0 with the M6 settings (SW §10.4: FA on, q8_0 KV on the host,
    # rotation off; the MUL_MATs on the device through the graph kernel): device (sim) vs CPU vs the
    # 1-ulp floor, 4 built-in prompts x 32 tokens, plus the embd-path identity check. ~25 s. Registered
    # only if the model is present (models/ is not in git).
    set(_tsr_model ${TESSERACT_ROOT}/models/qwen2.5-0.5b-instruct-q8_0.gguf)
    if (EXISTS ${_tsr_model})
        add_test(NAME tools_klcheck_qwen05b
                 COMMAND klcheck all -m ${_tsr_model} --device sim --phase m6 --n-prompts 4 -n 32)
        set_tests_properties(tools_klcheck_qwen05b PROPERTIES TIMEOUT 900 LABELS model ENVIRONMENT "TSR_SIM_THREADS=4")
        # The M6 gate's identity check (SW §10.4 item 2) across two transports: generate on sim,
        # replay the tokens on mock, every logits row bit-identical (test_klcheck_identity.sh, ~10 s).
        if (TSR_BASH_TOOLS)
            add_test(NAME tools_klcheck_identity_qwen05b
                     COMMAND ${TSR_BASH_TOOLS} ${TESSERACT_ROOT}/sw/tools/test_klcheck_identity.sh $<TARGET_FILE:klcheck>
                             ${_tsr_model} ${CMAKE_BINARY_DIR}/sw/tools/klcheck_identity)
            set_tests_properties(tools_klcheck_identity_qwen05b PROPERTIES TIMEOUT 900 LABELS model ENVIRONMENT "TSR_SIM_THREADS=4")
            # TSR_KINDS per phase (SW-49): --phase m6 sets m6 over the caller's value, --phase m7 keeps the
            # caller's list (or none: every kind); the backend's kinds and klcheck's record agree (~3 s).
            add_test(NAME tools_klcheck_kinds_qwen05b
                     COMMAND ${TSR_BASH_TOOLS} ${TESSERACT_ROOT}/sw/tools/test_tsr_kinds.sh klcheck $<TARGET_FILE:klcheck>
                             ${_tsr_model} ${CMAKE_BINARY_DIR}/sw/tools/klcheck_kinds)
            set_tests_properties(tools_klcheck_kinds_qwen05b PROPERTIES TIMEOUT 300 LABELS model ENVIRONMENT "TSR_SIM_THREADS=4")
        endif()
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
    # tsr_transport.h (tsr_versions of the backend's versions hook) and tsr/opl.hpp (kind names), no library
    target_include_directories(tsr-numcheck PRIVATE ${TESSERACT_ROOT}/sw/transport ${TESSERACT_ROOT}/sw/sim/include
                               ${TESSERACT_ROOT}/sw/ctrl/include)
    target_link_libraries(tsr-numcheck PRIVATE llama ggml)
    target_include_directories(tsr-numcheck PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/vendor)   # nlohmann/json.hpp
    if (TARGET ggml-base)
        target_link_libraries(tsr-numcheck PRIVATE ggml-base)     # gguf_*, ggml_get_type_traits, fp16
    endif()
    # No model: gather of views / F16 / BF16 / Q8_0 / I32 against ggml's CPU CONT, name and option parsing.
    add_test(NAME tools_numcheck_dump_selftest COMMAND tsr-numcheck selftest)
    # Qwen2.5-0.5B Q8_0 with the M6 settings: cpu, floor, dev (sim), cpu-tok and dev-noobs runs, then
    # selfcheck (identical node lists, the same settings in every run, observation invariance, embd ==
    # token path, weight MUL_MATs on Tesseract). The per-node observation makes every observed
    # node its own launch (SW-27). ~10 s.
    set(_tsr_model ${TESSERACT_ROOT}/models/qwen2.5-0.5b-instruct-q8_0.gguf)
    if (EXISTS ${_tsr_model})
        add_test(NAME tools_numcheck_dump_qwen05b
                 COMMAND tsr-numcheck all -m ${_tsr_model} -T 32 --phase m6 --checks full)
        set_tests_properties(tools_numcheck_dump_qwen05b PROPERTIES TIMEOUT 600 LABELS model ENVIRONMENT TSR_SIM_THREADS=4)
        # TSR_KINDS per phase as for klcheck (test_tsr_kinds.sh); meta.json info.TSR_KINDS records it (~3 s).
        if (TSR_BASH_TOOLS)
            add_test(NAME tools_numcheck_kinds_qwen05b
                     COMMAND ${TSR_BASH_TOOLS} ${TESSERACT_ROOT}/sw/tools/test_tsr_kinds.sh numcheck $<TARGET_FILE:tsr-numcheck>
                             ${_tsr_model} ${CMAKE_BINARY_DIR}/sw/tools/numcheck_kinds)
            set_tests_properties(tools_numcheck_kinds_qwen05b PROPERTIES TIMEOUT 300 LABELS model ENVIRONMENT "TSR_SIM_THREADS=4")
        endif()
    endif()
endfunction()
cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_numcheck)

# tsr-mtp (docs/design/06-software-verification.md §7.11, SW-83): the multi-token prediction driver and checker. It
# is llama.cpp's speculative-simple loop with the device counters and the forced rejections, so it links llama-common
# (the pinned tree's common_speculative); created by a deferred call like klcheck.
function(tsr_add_mtp)
    if (NOT TARGET llama OR NOT TARGET ggml OR NOT TARGET llama-common)
        message(STATUS "tsr-mtp: llama targets not available, tsr-mtp is not built")
        return()
    endif()
    add_executable(tsr-mtp ${TESSERACT_ROOT}/sw/tools/tsr_mtp.cpp)
    set_target_properties(tsr-mtp PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(tsr-mtp PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    target_link_libraries(tsr-mtp PRIVATE llama-common llama ggml)
endfunction()
cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_mtp)

# numcheck_helper (numcheck.py: ggml conversions + libtsr_ref B1 loop), see numcheck_helper.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/numcheck_helper.cmake)

# P3.7 / P3.8: libtsr_dbg, tsr-dbg, tsr-prof and their tests (dbg/dbg.cmake).
include(${CMAKE_CURRENT_SOURCE_DIR}/dbg/dbg.cmake)

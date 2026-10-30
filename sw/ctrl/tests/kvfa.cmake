# sw/ctrl/tests/kvfa.cmake — W2b P2.7 tests of the SET_ROWS_KV and FLASH_ATTN kinds (sw/ctrl/kernels/
# k_kv.cpp, k_fattn.cpp; docs/design/06-software-verification.md "SW" §7.5, §10.2; docs/design/03-attention-
# kv.md "AT" §6.6, §13). Included by sw/ctrl/host/CMakeLists.txt (after kernels.cmake: it reuses the
# sanitizer build of the host kernels). Every test has label fw; long ones add slow.
#
#   fw_kvfa_plan           the FA chunk planner shared by the firmware and tsr_ref_fa (sw/sim/include/
#                          tsr_fa_plan.h) against W1a's planner (the oracle in the test): random and
#                          adversarial mask summaries, every chunk list equal (test_fw_kvfa_plan.cpp)
#   fw_kvfa_rv_<group>     the RISC-V image on tsr_rv, groups qwen05b, qwen32b, gemma_swa, gemma_glb,
#                          general (test_fw_kvfa.h cases): [SET_ROWS_KV K, SET_ROWS_KV V, FLASH_ATTN]
#                          per case, fixed + 7 random schedules, race fatal, planes and outputs bit-exact
#                          against sw/ctrl/ref/kv_ref.cpp / fattn_ref.cpp, SW §4.7 observables equal
#   fw_kvfa_rv_slow        (slow) the 16K-cell cases
#   fw_kvfa_kl             (TSR_WITH_LLAMA) the kernel's output against ggml-cpu's FLASH_ATTN_EXT on the
#                          same q8_0 cache: NMSE and softmax-KL (AT §6.7)
#   fw_kvfa_perf           measurements: RISC-V work per SET_ROWS_KV decode record and per 32B decode
#                          FLASH_ATTN at 4K cells against the units' time (SW §7.8, AT §7.4, §9.2)
#   fw_kvfa_host           the kernels compiled for the host (SW §6.8) under ASan + UBSan: the same
#                          lists through the host build of the executor, outputs against the references,
#                          and the refused records (BAD_RECORD / SCRATCH, nothing pushed)

set(_kv_dir ${TSR_CTRL_DIR}/tests)
set(_kv_inc ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/include
            ${TSR_FW_SIM_TESTS_DIR})

function(_tsr_kv_cxx tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_kv_inc})
endfunction()

# The host references of the two kinds (tests only; the backend's tests may link them too).
add_library(tsr_fw_kvfa_ref STATIC ${TSR_CTRL_DIR}/ref/kv_ref.cpp ${TSR_CTRL_DIR}/ref/fattn_ref.cpp)
set_target_properties(tsr_fw_kvfa_ref PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_fw_kvfa_ref PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_fw_kvfa_ref PUBLIC ${TSR_CTRL_DIR}/ref ${TSR_CTRL_INCLUDE_DIR} ${TESSERACT_ROOT}/sw/sim/include)
target_link_libraries(tsr_fw_kvfa_ref PUBLIC tsr_ref)

# --- the shared chunk planner against W1a's -------------------------------------------------------
add_executable(test_fw_kvfa_plan ${_kv_dir}/test_fw_kvfa_plan.cpp)
_tsr_kv_cxx(test_fw_kvfa_plan)
target_link_libraries(test_fw_kvfa_plan PRIVATE tsr_ref m)
add_test(NAME fw_kvfa_plan COMMAND test_fw_kvfa_plan)
set_tests_properties(fw_kvfa_plan PROPERTIES LABELS fw TIMEOUT 300)

# --- RISC-V image on tsr_rv -----------------------------------------------------------------------
add_executable(test_fw_kvfa_rv ${_kv_dir}/test_fw_kvfa_rv.cpp)
_tsr_kv_cxx(test_fw_kvfa_rv)
target_link_libraries(test_fw_kvfa_rv PRIVATE tsr_fw_kvfa_ref tsr_sim tsr_ref Threads::Threads m)
if (TSR_WITH_LLAMA)
    target_compile_definitions(test_fw_kvfa_rv PRIVATE TSR_KVFA_GGML=1)
    target_include_directories(test_fw_kvfa_rv SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    target_link_libraries(test_fw_kvfa_rv PRIVATE ggml-cpu ggml-base)
endif()
add_dependencies(test_fw_kvfa_rv tsr_fw)
foreach(_g qwen05b qwen32b gemma_swa gemma_glb general)
    add_test(NAME fw_kvfa_rv_${_g} COMMAND test_fw_kvfa_rv ${TSR_FW_ELF} suite ${_g})
    set_tests_properties(fw_kvfa_rv_${_g} PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77 PROCESSORS 2)
endforeach()
add_test(NAME fw_kvfa_rv_slow COMMAND test_fw_kvfa_rv ${TSR_FW_ELF} suite slow)
set_tests_properties(fw_kvfa_rv_slow PROPERTIES LABELS "fw;slow" TIMEOUT 7200 SKIP_RETURN_CODE 77 PROCESSORS 2)
if (TSR_WITH_LLAMA)
    add_test(NAME fw_kvfa_kl COMMAND test_fw_kvfa_rv ${TSR_FW_ELF} kl)
    set_tests_properties(fw_kvfa_kl PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77 PROCESSORS 4)
endif()
add_test(NAME fw_kvfa_perf COMMAND test_fw_kvfa_rv ${TSR_FW_ELF} perf)
set_tests_properties(fw_kvfa_perf PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77)

# --- host build of the kernels under ASan + UBSan (SW §6.8) ---------------------------------------
if (NOT TARGET tsr_fw_host_kern_san)
    tsr_fw_host_lib(tsr_fw_host_kern_san SANITIZE)
endif()
add_executable(test_fw_kvfa_host ${_kv_dir}/test_fw_kvfa_host.cpp ${TSR_CTRL_DIR}/ref/kv_ref.cpp
               ${TSR_CTRL_DIR}/ref/fattn_ref.cpp)
_tsr_kv_cxx(test_fw_kvfa_host)
target_link_libraries(test_fw_kvfa_host PRIVATE tsr_fw_host_kern_san)
add_test(NAME fw_kvfa_host COMMAND test_fw_kvfa_host)
set_tests_properties(fw_kvfa_host PROPERTIES LABELS fw TIMEOUT 600
                     ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")

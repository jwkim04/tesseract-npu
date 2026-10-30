# sw/ctrl/tests/elem.cmake — W2b P2.6 tests of the element-wise and row kinds ADD, MUL, SCALE, GLU,
# UNARY, CPY, GET_ROWS (docs/design/06-software-verification.md "SW" §7.2, §7.5, §10.2). Included by
# sw/ctrl/host/CMakeLists.txt (main project, after kernels.cmake). Every test has label fw.
#
#   fw_kernels_elem_rv        the kinds in tsr-fw.elf on tsr_rv through the graph executor: decode and
#                             prefill shapes of Qwen2.5-0.5B / 32B and Gemma 4 12B, every operand-b
#                             form (broadcast, rows, periods, the scalar forms), strided rows, tile
#                             boundaries on a small SPM, chunked copies, NaN / inf / subnormal inputs;
#                             fixed + 7 random schedules with TSR_SIM_RACE fatal, every dst row bit-exact
#                             against sw/ctrl/ref/elem_ref.cpp, SW §4.7 observables equal, written DRAM =
#                             the dst rows (test_fw_elem_rv.cpp suite)
#   fw_kernels_elem_rv_shapes (slow, ~30 s) the full model shapes: prefill ubatch 512 of the three
#                             models (SWIGLU 27648 x 512, GEGLU16 15360 x 512, ...), the 262,144-logit
#                             softcap; 8 schedules, race fatal
#   fw_kernels_elem_rv_errors refused records (BAD_RECORD, SCRATCH) and GET_ROWS ids the device reports
#                             (TILE_ERR DRAM_RANGE)
#   fw_kernels_elem_perf      RISC-V cycles of work per decode record of each kind (fw_prof.h), deep
#                             queues, Ibex maxperf; regression ceiling 600, target 350 reported
#   fw_kernels_elem_host      the kinds compiled for the host (SW §6.8) under ASan + UBSan: the planners
#                             against their rules, every decode fast path against the general path
#                             push for push on twin simulators, refused records push nothing, results
#                             against elem_ref (test_fw_elem_host.cpp)
#   fw_kernels_elem_ggml      (TSR_WITH_LLAMA) the reference's numerics against the pinned ggml-cpu ops
#                             on the same inputs: per kind the exact-match fraction, the ulp and
#                             absolute error figures, the known deltas of VU §6.9

if (NOT TARGET tsr_fw_ref)
    return()                       # kernels.cmake (P2.5) provides tsr_fw_ref (mulmat_ref, dram_view)
endif()
set(_el_dir ${TSR_CTRL_DIR}/tests)
set(_el_inc ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/include
            ${TSR_FW_SIM_TESTS_DIR})

function(_tsr_el_cxx tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_el_inc})
endfunction()

# The host reference of the element-wise and row kinds (tests and the ggml comparison).
add_library(tsr_fw_elem_ref STATIC ${TSR_CTRL_DIR}/ref/elem_ref.cpp)
set_target_properties(tsr_fw_elem_ref PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_fw_elem_ref PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_fw_elem_ref PUBLIC ${TSR_CTRL_DIR}/ref ${TSR_CTRL_INCLUDE_DIR})
target_link_libraries(tsr_fw_elem_ref PUBLIC tsr_ref)

# --- RISC-V image on tsr_rv (the reference, SW §6.8) ---------------------------------------------
add_executable(test_fw_elem_rv ${_el_dir}/test_fw_elem_rv.cpp)
_tsr_el_cxx(test_fw_elem_rv)
target_link_libraries(test_fw_elem_rv PRIVATE tsr_fw_elem_ref tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_elem_rv tsr_fw)
add_test(NAME fw_kernels_elem_rv COMMAND test_fw_elem_rv ${TSR_FW_ELF} suite)
set_tests_properties(fw_kernels_elem_rv PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_elem_rv_shapes COMMAND test_fw_elem_rv ${TSR_FW_ELF} shapes)
set_tests_properties(fw_kernels_elem_rv_shapes PROPERTIES LABELS "fw;slow" TIMEOUT 3600 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_elem_rv_errors COMMAND test_fw_elem_rv ${TSR_FW_ELF} errors)
set_tests_properties(fw_kernels_elem_rv_errors PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_elem_perf COMMAND test_fw_elem_rv ${TSR_FW_ELF} perf)
set_tests_properties(fw_kernels_elem_perf PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)

# --- host build of the kernels under ASan + UBSan (SW §6.8) --------------------------------------
if (NOT TARGET tsr_fw_host_kern_san)
    tsr_fw_host_lib(tsr_fw_host_kern_san SANITIZE)
endif()
add_executable(test_fw_elem_host ${_el_dir}/test_fw_elem_host.cpp ${TSR_CTRL_DIR}/ref/elem_ref.cpp)
_tsr_el_cxx(test_fw_elem_host)
target_link_libraries(test_fw_elem_host PRIVATE tsr_fw_host_kern_san)
add_test(NAME fw_kernels_elem_host COMMAND test_fw_elem_host)
set_tests_properties(fw_kernels_elem_host PROPERTIES LABELS fw TIMEOUT 300
                     ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")

# --- the reference against the pinned ggml-cpu (KL / ulp level, VU §6.9) --------------------------
# ggml-cpu is defined later in the top-level project (third_party/llama.cpp); target names resolve at
# generate time (as sw/sim/ref_math.cmake's ref_math_gelu16).
if (TSR_WITH_LLAMA)
    add_executable(test_fw_elem_ggml ${_el_dir}/test_fw_elem_ggml.cpp)
    _tsr_el_cxx(test_fw_elem_ggml)
    target_include_directories(test_fw_elem_ggml SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    target_link_libraries(test_fw_elem_ggml PRIVATE tsr_fw_elem_ref tsr_ref ggml-cpu ggml-base Threads::Threads m)
    add_test(NAME fw_kernels_elem_ggml COMMAND test_fw_elem_ggml)
    set_tests_properties(fw_kernels_elem_ggml PROPERTIES LABELS fw TIMEOUT 300)
endif()

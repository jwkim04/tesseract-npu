# sw/ctrl/tests/kernels.cmake — P2.5 tests of the M6 kernels: the MUL_MAT kind, `selftest` and
# `replay` (docs/v2/06-software-verification.md "SW" §7.4, §7.6, §7.7, §10.2). Included by
# sw/ctrl/host/CMakeLists.txt (main project). Every test has label fw; long sweeps add slow.
#
#   fw_kernels_rv          MUL_MAT through the graph executor of tsr-fw.elf on tsr_rv: three op lists
#                          (shapes, addends, ACC_DST, strides, deps, plan.hpp tile boundaries, the sweep
#                          Q8_0 / Q4_K x T = 1 .. T_MAX x addend forms), fixed + 7 random schedules with
#                          TSR_SIM_RACE fatal, y bit-exact against sw/ctrl/ref/mulmat_ref.cpp, SW §4.7
#                          observables equal (test_fw_kernels_rv.cpp)
#   fw_kernels_rv_mutant_race, fw_kernels_rv_mutant_v1   negative tests: images with a mutated MUL_MAT
#                          (QACT without its LD dep; QACT src stride 0) must be caught by the race
#                          detector / the v1 stream comparison
#   fw_kernels_rv_v1conv   the v1 lowering check: a sample of 10 traces/model-v1conv traces regrouped into
#                          MUL_MAT ops, outputs = v1's expect bit for bit, push streams = v1's modulo the
#                          v2 forms (skipped without the set on disk)
#   fw_kernels_rv_v1conv_all  (slow) all 290 converted traces
#   fw_kernels_rv_shapes   (slow) Qwen2.5-0.5B / 32B and Gemma 4 12B shapes incl. the vocab projections
#   fw_kernels_host        the kernels compiled for the host (SW §6.8) under ASan + UBSan: kind_mulmat
#                          with its own ring against the reference, replay and selftest through host_run
#   fw_selftest_sim        every selftest id (SW §7.6) on tsr_rv: device-side checks, outputs against
#                          the tsr_ref references, PING
#   fw_replay_sim          the replay kernel route of trace replay (tsr_trace2_replay_sim with
#                          replay_image): a fixed sample of traces/m2, gen, vu, at, ctrl and (SERIAL)
#                          model-v1conv, each equal to the tile-only route
#   fw_replay_sim_all      (slow) the whole sets

set(_k_dir ${TSR_CTRL_DIR}/tests)
set(_k_inc ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/include
           ${TSR_FW_SIM_TESTS_DIR})
set(_k_traces ${TESSERACT_ROOT}/traces)

function(_tsr_k_cxx tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_k_inc})
endfunction()

# The host reference of the MUL_MAT kind (tests only).
add_library(tsr_fw_ref STATIC ${TSR_CTRL_DIR}/ref/mulmat_ref.cpp)
set_target_properties(tsr_fw_ref PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_fw_ref PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_fw_ref PUBLIC ${TSR_CTRL_DIR}/ref ${TSR_CTRL_INCLUDE_DIR})
target_link_libraries(tsr_fw_ref PUBLIC tsr_ref Threads::Threads)

# --- RISC-V image on tsr_rv (the reference, SW §6.8) ---------------------------------------------
add_executable(test_fw_kernels_rv ${_k_dir}/test_fw_kernels_rv.cpp)
_tsr_k_cxx(test_fw_kernels_rv)
target_link_libraries(test_fw_kernels_rv PRIVATE tsr_fw_ref tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_kernels_rv tsr_fw)
add_test(NAME fw_kernels_rv COMMAND test_fw_kernels_rv ${TSR_FW_ELF} suite)
set_tests_properties(fw_kernels_rv PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_rv_v1conv COMMAND test_fw_kernels_rv ${TSR_FW_ELF} v1conv
         ${_k_traces}/model-v1conv/qwen05b-q8_0 ${_k_traces}/model-v1conv/qwen05b-q4_k_m-requant --sample 5)
set_tests_properties(fw_kernels_rv_v1conv PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_rv_v1conv_all COMMAND test_fw_kernels_rv ${TSR_FW_ELF} v1conv
         ${_k_traces}/model-v1conv/qwen05b-q8_0 ${_k_traces}/model-v1conv/qwen05b-q4_k_m-requant)
set_tests_properties(fw_kernels_rv_v1conv_all PROPERTIES LABELS "fw;slow" TIMEOUT 7200 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_rv_shapes COMMAND test_fw_kernels_rv ${TSR_FW_ELF} shapes)
set_tests_properties(fw_kernels_rv_shapes PROPERTIES LABELS "fw;slow" TIMEOUT 7200 SKIP_RETURN_CODE 77)

# Negative tests: mutated MUL_MAT kernels must be caught (test images with TSR_KMM_MUTANT; the kernel
# sources are compiled into the image itself, NO_KERNELS, since DEFINES reach only its own sources).
file(GLOB _k_kernels CONFIGURE_DEPENDS ${TSR_CTRL_DIR}/kernels/*.cpp)
tsr_fw_image(fw_kmm_mut_nodep NO_KERNELS SOURCES ${_k_kernels} DEFINES TSR_KMM_MUTANT=1)
tsr_fw_image(fw_kmm_mut_stride NO_KERNELS SOURCES ${_k_kernels} DEFINES TSR_KMM_MUTANT=2)
add_test(NAME fw_kernels_rv_mutant_race COMMAND test_fw_kernels_rv ${TSR_FW_TEST_ELF_fw_kmm_mut_nodep} suite
         --scheds 1 --only 0 --race report --expect-fail)
set_tests_properties(fw_kernels_rv_mutant_race PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_rv_mutant_v1 COMMAND test_fw_kernels_rv ${TSR_FW_TEST_ELF_fw_kmm_mut_stride} v1conv
         ${_k_traces}/model-v1conv/qwen05b-q8_0 --sample 1 --expect-fail)
set_tests_properties(fw_kernels_rv_mutant_v1 PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)

# --- replay kernel route of trace replay (C) -----------------------------------------------------
add_executable(test_fw_replay_sim ${_k_dir}/test_fw_replay_sim.c)
tsr_c_target(test_fw_replay_sim)
target_include_directories(test_fw_replay_sim PRIVATE ${_k_inc})
target_link_libraries(test_fw_replay_sim PRIVATE tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_replay_sim tsr_fw)
add_test(NAME fw_replay_sim COMMAND test_fw_replay_sim ${TSR_FW_ELF} ${_k_traces})
set_tests_properties(fw_replay_sim PROPERTIES LABELS fw TIMEOUT 900 SKIP_RETURN_CODE 77)
add_test(NAME fw_replay_sim_all COMMAND test_fw_replay_sim ${TSR_FW_ELF} ${_k_traces} --all)
set_tests_properties(fw_replay_sim_all PROPERTIES LABELS "fw;slow" TIMEOUT 14400 SKIP_RETURN_CODE 77)

# --- selftest on the simulator (C) ---------------------------------------------------------------
add_executable(test_fw_selftest_sim ${_k_dir}/test_fw_selftest_sim.c)
tsr_c_target(test_fw_selftest_sim)
target_include_directories(test_fw_selftest_sim PRIVATE ${_k_inc})
target_link_libraries(test_fw_selftest_sim PRIVATE tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_selftest_sim tsr_fw)
add_test(NAME fw_selftest_sim COMMAND test_fw_selftest_sim ${TSR_FW_ELF})
set_tests_properties(fw_selftest_sim PROPERTIES LABELS fw TIMEOUT 900 SKIP_RETURN_CODE 77)

# --- host build of the kernels under ASan + UBSan (SW §6.8) --------------------------------------
tsr_fw_host_lib(tsr_fw_host_kern_san SANITIZE)
add_executable(test_fw_kernels_host ${_k_dir}/test_fw_kernels_host.cpp ${TSR_CTRL_DIR}/ref/mulmat_ref.cpp)
_tsr_k_cxx(test_fw_kernels_host)
target_link_libraries(test_fw_kernels_host PRIVATE tsr_fw_host_kern_san)
add_dependencies(test_fw_kernels_host tsr_fw)
set(_k_err_trace ${_k_traces}/ctrl/errors)
file(GLOB _k_err_traces LIST_DIRECTORIES true ${_k_err_trace}/*)
list(SORT _k_err_traces)
set(_k_host_trace "")
foreach(_t IN LISTS _k_err_traces)
    if (EXISTS ${_t}/meta.json AND _k_host_trace STREQUAL "")
        set(_k_host_trace ${_t})
    endif()
endforeach()
add_test(NAME fw_kernels_host COMMAND test_fw_kernels_host ${TSR_FW_ELF} ${_k_host_trace})
set_tests_properties(fw_kernels_host PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77
                     ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")

# sw/ctrl/tests/e2e.cmake — the W2a integration tests (docs/v2/06-software-verification.md "SW" §11
# P2.1-P2.5, §10.1, §10.2; docs/v2/README.md §10 W2a). Included by sw/ctrl/host/CMakeLists.txt after
# the package snippets (it uses kernels.cmake's tsr_fw_ref and the tsr_elf library).
#
#   fw_m6_e2e    the production image tsr-fw.elf end to end on the simulator, fixed + 7 random
#                schedules with TSR_SIM_RACE fatal and launch capture: loader with VERIFY, HELLO (id =
#                tsr-elf's), PING, a two-layer 0.5B-shaped MUL_MAT op list through `graph` (Q8_0 and
#                Q4_K, bit-exact against sw/ctrl/ref/mulmat_ref.cpp), `selftest` ids 1-10, `replay` of
#                trace samples (traces/model-v1conv, vu, at, ctrl when on disk, and a synthetic stream);
#                identical observables and captured traces under every schedule; every captured launch
#                replays on fresh simulators (test_fw_m6_e2e.cpp)
#   fw_m6_perf   the M6 numbers of SW §7.8 / EX §4.11 on the production image: image sizes, launch costs,
#                RISC-V cycles per graph op (busy and spin-free work, U200 and deep queues) with a
#                per-function profile (tests/fw_prof.h), RISC-V vs tile time of a 0.5B decode layer
#                (printed; only regression ceilings asserted); with the fw_exec image an M7 projection
#                (the layer plus 25 short VU records per layer, exec_kinds.cpp kind 0x38)

if (NOT TARGET tsr_fw_ref)
    return()                       # kernels.cmake (P2.5) provides the MUL_MAT reference
endif()
set(_e_dir ${TSR_CTRL_DIR}/tests)
add_executable(test_fw_m6_e2e ${_e_dir}/test_fw_m6_e2e.cpp)
set_target_properties(test_fw_m6_e2e PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_fw_m6_e2e PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
target_include_directories(test_fw_m6_e2e PRIVATE ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim
                           ${TESSERACT_ROOT}/sw/sim/include ${TSR_FW_SIM_TESTS_DIR})
target_link_libraries(test_fw_m6_e2e PRIVATE tsr_fw_ref tsr_elf tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_m6_e2e tsr_fw)
set(_e_scratch ${CMAKE_CURRENT_BINARY_DIR}/scratch)
add_test(NAME fw_m6_e2e COMMAND test_fw_m6_e2e ${TSR_FW_ELF} ${_e_scratch} ${TESSERACT_ROOT}/traces)
set_tests_properties(fw_m6_e2e PROPERTIES LABELS fw TIMEOUT 600 SKIP_RETURN_CODE 77)
# The M7 projection of fw_m6_perf runs on the fw_exec test image (exec.cmake; its kind 0x38 VU stub).
add_test(NAME fw_m6_perf COMMAND test_fw_m6_e2e ${TSR_FW_ELF} ${_e_scratch} --perf ${TSR_FW_TEST_ELF_fw_exec})
set_tests_properties(fw_m6_perf PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)

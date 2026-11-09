# sw/ctrl/tests/runtime.cmake — fw_runtime (docs/design/06-software-verification.md "SW" §10.2):
# the SDK runtime (sw/ctrl/rt: crt0, R1-R10 of EX §4.7, traps and panic, log records) as the
# RISC-V image on the simulator, under the fixed and 7 random schedules with TSR_SIM_RACE=fatal.
#
# Included by sw/ctrl/host/CMakeLists.txt (main project). Provides:
#   image fw_rt         tsr_fw_image(fw_rt NO_KERNELS ...): sw/ctrl/rt plus the test kernels
#                       sw/ctrl/tests/fw/rt_*.cpp (no production kernels, so the test does not
#                       depend on the production kernels); TSR_FW_TEST_ELF_fw_rt
#   test_fw_runtime     the host driver (sw/ctrl/tests/test_fw_runtime.c, C11), with the helpers of
#                       sw/sim/tests/test_sim_rvhost.h / test_sim_obs.h and tsr_elf (image id,
#                       kernel table, log formats)
#   ctest fw_runtime    test_fw_runtime <fw_rt.elf>; prints the EX §4.11 launch costs
#   image fw_rt_stale   the same with RT_MUTANT_STALE (rt_blk1 reads beyond its staged line)
#   ctest fw_runtime_stale  the race detector reports exactly that read as SPM_STALE (SW-4)
#
# Run: ctest --test-dir <build> -R '^fw_runtime$' --output-on-failure

file(GLOB _rt_test_kernels CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/fw/rt_*.cpp)
list(SORT _rt_test_kernels)
tsr_fw_image(fw_rt NO_KERNELS SOURCES ${_rt_test_kernels})
tsr_fw_image(fw_rt_stale NO_KERNELS SOURCES ${_rt_test_kernels} DEFINES RT_MUTANT_STALE)

add_executable(test_fw_runtime ${CMAKE_CURRENT_LIST_DIR}/test_fw_runtime.c)
tsr_c_target(test_fw_runtime)
target_include_directories(test_fw_runtime PRIVATE ${TSR_FW_SIM_TESTS_DIR} ${TESSERACT_ROOT}/sw/sim)
target_link_libraries(test_fw_runtime PRIVATE tsr_sim tsr_ref tsr_elf Threads::Threads m)
add_dependencies(test_fw_runtime tsr_fw)
add_test(NAME fw_runtime COMMAND test_fw_runtime ${TSR_FW_TEST_ELF_fw_rt})
set_tests_properties(fw_runtime PROPERTIES TIMEOUT 300)
add_test(NAME fw_runtime_stale COMMAND test_fw_runtime --stale ${TSR_FW_TEST_ELF_fw_rt_stale})
set_tests_properties(fw_runtime_stale PROPERTIES TIMEOUT 60)

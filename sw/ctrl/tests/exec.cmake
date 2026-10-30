# sw/ctrl/tests/exec.cmake — P2.4 tests of plan.hpp, OPL v1 and the graph executor
# (docs/v2/06-software-verification.md "SW" §7.2, §7.3, §7.8, §10.2). Included by
# sw/ctrl/host/CMakeLists.txt (main project).
#
#   fw_exec      test image: the production sources plus tests/fw/exec_kinds.cpp (test kinds
#                0x30-0x3F of the executor tests; tsr_fw_image)
#   fw_exec_host the executor compiled for the host (SW §6.8; ASan + UBSan variant of tsr_fw_host)
#                with the test kinds, driven by tsr::host_run; plan.hpp against tsr_ref and v1
#   fw_exec_rv   the fw_exec image on tsr_rv through the LQ / LCQ: every scenario under the fixed
#                and 7 random schedules with TSR_SIM_RACE=fatal, observables and DONE records
#                compared (SW §4.7), and the race detector probe (≈ 8 s)
#   fw_exec_perf RISC-V cycles per graph op on 0.5B-like lists (SW §7.8 target <= 400: printed;
#                asserted: the executor's own overhead per record and a regression ceiling)

tsr_fw_image(fw_exec SOURCES ${TSR_CTRL_DIR}/tests/fw/exec_kinds.cpp)

set(_exec_src ${TSR_CTRL_DIR}/tests/test_fw_exec.cpp)
set(_exec_inc ${TSR_CTRL_INCLUDE_DIR} ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/tests)

function(_tsr_exec_test_target tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_exec_inc})
    if (TARGET tsr_lower)
        target_compile_definitions(${tgt} PRIVATE TSR_EXEC_HAVE_LOWER=1)
        target_link_libraries(${tgt} PRIVATE tsr_lower)
    endif()
endfunction()

# --- RISC-V: the reference (SW §6.8) -------------------------------------------------------------
add_executable(test_fw_exec_rv ${_exec_src})
_tsr_exec_test_target(test_fw_exec_rv)
target_link_libraries(test_fw_exec_rv PRIVATE tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_exec_rv tsr_fw)
add_test(NAME fw_exec_rv COMMAND test_fw_exec_rv ${TSR_FW_TEST_ELF_fw_exec} --full)
set_tests_properties(fw_exec_rv PROPERTIES TIMEOUT 300)
add_test(NAME fw_exec_perf COMMAND test_fw_exec_rv ${TSR_FW_TEST_ELF_fw_exec} --perf)
set_tests_properties(fw_exec_perf PROPERTIES TIMEOUT 300)

# --- host build (ASan + UBSan) -------------------------------------------------------------------
tsr_fw_host_lib(tsr_fw_host_exec_san SANITIZE)
add_executable(test_fw_exec_host ${_exec_src} ${TSR_CTRL_DIR}/tests/fw/exec_kinds.cpp)
_tsr_exec_test_target(test_fw_exec_host)
target_compile_definitions(test_fw_exec_host PRIVATE TSR_EXEC_TEST_HOST=1)
target_link_libraries(test_fw_exec_host PRIVATE tsr_fw_host_exec_san)
add_test(NAME fw_exec_host COMMAND test_fw_exec_host)
set_tests_properties(fw_exec_host PROPERTIES TIMEOUT 300)

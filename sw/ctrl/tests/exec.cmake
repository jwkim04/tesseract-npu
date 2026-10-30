# sw/ctrl/tests/exec.cmake — P2.4 tests of plan.hpp, OPL v1 and the graph executor
# (docs/v2/06-software-verification.md "SW" §7.2, §7.3, §7.8, §10.2). Included by
# sw/ctrl/host/CMakeLists.txt (main project).
#
#   fw_exec      test image: the production sources plus tests/fw/exec_kinds.cpp and exec_fast.cpp
#                (test kinds 0x30-0x3F of the executor tests; tsr_fw_image)
#   fw_exec_host the executor compiled for the host (SW §6.8; ASan + UBSan variant of tsr_fw_host)
#                with the test kinds, driven by tsr::host_run; plan.hpp against tsr_ref and v1
#   fw_exec_rv   the fw_exec image on tsr_rv through the LQ / LCQ: every scenario under the fixed
#                and 7 random schedules with TSR_SIM_RACE=fatal, observables and DONE records
#                compared (SW §4.7), and the race detector probe (≈ 8 s)
#   fw_exec_perf RISC-V cycles per graph op on 0.5B-like lists (SW §7.8 target <= 400: printed;
#                asserted: the executor's own overhead per record, a JOIN record, regression ceilings);
#                TSR_FW_PROF=1 adds a per-function profile per list (tests/fw_prof.h); W2b: the "short
#                VU record" scenarios (each M7 short kind of the image at a decode shape, printed against
#                the 350 RV-cycle requirement; the k_fast.hpp reference kind 0x39 of tests/fw/exec_fast.cpp
#                with a regression ceiling)
#   fw_exec_perf_m7  the short VU records only, every kind of the image (and SCALE with a bias) asserted
#                <= 350 RV cycles of work per record (W2b's requirement, no exemption since the W2b fix;
#                --kind <k> selects one when run by hand)
#   fw_exec_perf_ref the test image fw_exec_ref, whose kind_unary is the k_fast.hpp reference
#                (tests/fw/exec_ref_unary.cpp) on the production dispatch path: regression ceiling 380

tsr_fw_image(fw_exec SOURCES ${TSR_CTRL_DIR}/tests/fw/exec_kinds.cpp ${TSR_CTRL_DIR}/tests/fw/exec_fast.cpp)
# W2b: the k_fast.hpp reference short kind as a production table kind (kind_unary of
# tests/fw/exec_ref_unary.cpp; no W2b kernel package, the other M7 kinds are their stubs).
tsr_fw_image(fw_exec_ref NO_KERNELS SOURCES ${TSR_CTRL_DIR}/kernels/graph.cpp ${TSR_CTRL_DIR}/kernels/k_mulmat.cpp
             ${TSR_CTRL_DIR}/kernels/zz_kind_stubs.cpp ${TSR_CTRL_DIR}/kernels/selftest.cpp
             ${TSR_CTRL_DIR}/kernels/replay.cpp ${TSR_CTRL_DIR}/tests/fw/exec_ref_unary.cpp)

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
# W2b: every M7 short kind of the image at its decode shape must meet <= 350 RV cycles of work per
# record (fw_exec_perf prints them and asserts the k_fast.hpp reference kind only).
add_test(NAME fw_exec_perf_m7 COMMAND test_fw_exec_rv ${TSR_FW_TEST_ELF_fw_exec} --perf-m7 --strict)
set_tests_properties(fw_exec_perf_m7 PROPERTIES TIMEOUT 300)
# The reference short kind (fw_exec_ref's kind_unary, 3 pushes) through the production table path: the
# toolkit's floor (365 RV cycles at W2b infra), a regression ceiling (the 350 verdict is printed).
add_test(NAME fw_exec_perf_ref COMMAND test_fw_exec_rv ${TSR_FW_TEST_ELF_fw_exec_ref} --perf-m7 --kind 0x1b --ceiling 380)
set_tests_properties(fw_exec_perf_ref PROPERTIES TIMEOUT 300)

# --- host build (ASan + UBSan) -------------------------------------------------------------------
tsr_fw_host_lib(tsr_fw_host_exec_san SANITIZE)
add_executable(test_fw_exec_host ${_exec_src} ${TSR_CTRL_DIR}/tests/fw/exec_kinds.cpp ${TSR_CTRL_DIR}/tests/fw/exec_fast.cpp)
_tsr_exec_test_target(test_fw_exec_host)
target_compile_definitions(test_fw_exec_host PRIVATE TSR_EXEC_TEST_HOST=1)
target_link_libraries(test_fw_exec_host PRIVATE tsr_fw_host_exec_san)
add_test(NAME fw_exec_host COMMAND test_fw_exec_host)
set_tests_properties(fw_exec_host PROPERTIES TIMEOUT 300)

# --- W2b: OPL v1 M7 kinds and the M7 planning glue -------------------------------------------------
#   fw_opl_m7       writer / reader round trips of every M7 kind (named helpers, preload helpers, JOIN
#                   folding), BAD_RECORD for a wrong n_args (reader, check_args, plan check_op), slot
#                   counts, opl::check_args' structural rules per kind, plan_op = plan_kind, plan_rows and
#                   for_period_segments against brute force (test_fw_opl_m7.cpp; < 1 s)
#   fw_plan_cxx17   tsr/opl.hpp and tsr/plan.hpp with the three planner headers compile as C++17 (the
#                   backend's language, sw/ggml-tesseract: supports_op uses the same planners, V2-SW-20):
#                   the same source, -fsyntax-only (a compile-time check run as a test, so that a header
#                   that slips does not break every build)
add_executable(test_fw_opl_m7 ${TSR_CTRL_DIR}/tests/test_fw_opl_m7.cpp)
set_target_properties(test_fw_opl_m7 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_fw_opl_m7 PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
target_include_directories(test_fw_opl_m7 PRIVATE ${TSR_CTRL_INCLUDE_DIR})
add_test(NAME fw_opl_m7 COMMAND test_fw_opl_m7)
set_tests_properties(fw_opl_m7 PROPERTIES LABELS fw TIMEOUT 60)
add_test(NAME fw_plan_cxx17 COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only -Wall -Wextra -Wno-unused-function
         -I${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/tests/test_fw_opl_m7.cpp)
set_tests_properties(fw_plan_cxx17 PROPERTIES LABELS fw TIMEOUT 120)

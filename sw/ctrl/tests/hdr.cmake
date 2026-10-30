# sw/ctrl/tests/hdr.cmake — P2.3 tests of the tile/VU header library and the host backend
# (docs/v2/06-software-verification.md "SW" §6.7, §6.8, §10.2). Included by
# sw/ctrl/host/CMakeLists.txt (main project). The host backend itself (sw/ctrl/host/backend_host.cpp,
# shim_host.cpp) is part of tsr_fw_host there (every sw/ctrl/host/*.cpp).
#
#   fw_hdr            test image: the production sources plus tests/fw/hdr_kernels.cpp (tsr_fw_image)
#   fw_hdr_host       the library on the host build (ASan + UBSan; backend + shim + hdr_kernels): builders
#                     against the checker, tickets / deps / joins / errors, group folding, spm_ring
#                     against a line-level model, the hdr_* kernels through tsr::host_run
#   fw_hdr_rv         the fw_hdr image on tsr_rv under the fixed and 7 random schedules with
#                     TSR_SIM_RACE=fatal: results, DONE records, outputs and per-launch push streams
#                     identical across schedules and to the host build's; PERF PUSH_STALL = 0; queue
#                     occupancy; a soft stop
#   fw_hdr_rv_mutant  the image fw_hdr_mut (hdr_stream without spm_ring's must_wait deps) under the race
#                     detector: the reuse of ring blocks must be reported (must_wait is what orders it)
#   fw_hdr_constexpr  a builder whose field does not fit (or a rule the checker applies whatever the
#                     configuration) is a compile error in a constant expression: each case of
#                     tests/fw/hdr_constexpr.cpp must fail to compile, case 0 must compile

# ---------------------------------------------------------------------------
# Script mode (the fw_hdr_constexpr test): cmake -DTSR_HDR_CX_CHECK=1 -DCXX=<host c++> -DSRC=<file>
#     -DINC=<sw/ctrl/include> -DNCASES=<n> -P hdr.cmake
# Case 0 must compile (-fsyntax-only), cases 1..n must not.
# ---------------------------------------------------------------------------
if (TSR_HDR_CX_CHECK)
    set(_flags -std=c++20 -fsyntax-only -DTSR_FW_HOST=1 -I${INC})
    execute_process(COMMAND ${CXX} ${_flags} -DCASE=0 ${SRC} RESULT_VARIABLE _r ERROR_VARIABLE _e)
    if (NOT _r EQUAL 0)
        message(FATAL_ERROR "fw_hdr_constexpr: case 0 (legal builders) does not compile:\n${_e}")
    endif()
    foreach(_c RANGE 1 ${NCASES})
        execute_process(COMMAND ${CXX} ${_flags} -DCASE=${_c} ${SRC} RESULT_VARIABLE _r ERROR_VARIABLE _e)
        if (_r EQUAL 0)
            message(FATAL_ERROR "fw_hdr_constexpr: case ${_c} compiled, but its builder call must not be a constant expression")
        endif()
        if (NOT _e MATCHES "tsr::detail::op_")
            message(FATAL_ERROR "fw_hdr_constexpr: case ${_c} failed for another reason:\n${_e}")
        endif()
    endforeach()
    message(STATUS "fw_hdr_constexpr: case 0 compiles, cases 1..${NCASES} are compile errors")
    return()
endif()

tsr_fw_image(fw_hdr SOURCES ${TSR_CTRL_DIR}/tests/fw/hdr_kernels.cpp)
tsr_fw_image(fw_hdr_mut SOURCES ${TSR_CTRL_DIR}/tests/fw/hdr_kernels.cpp DEFINES HDR_MUTANT_NO_MUST_WAIT=1)

set(_hdr_src ${TSR_CTRL_DIR}/tests/test_fw_hdr.cpp ${TSR_CTRL_DIR}/tests/fw/hdr_kernels.cpp)
set(_hdr_inc ${TSR_CTRL_INCLUDE_DIR} ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/tests)

function(_tsr_hdr_test_target tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_hdr_inc})
endfunction()

# The host side of these tests needs only the backend and the shim (not the production kernels of
# tsr_fw_host): its own small libraries, so that the P2.3 tests do not depend on the other kernels.
set(_hdr_host_lib_src ${TSR_CTRL_DIR}/host/backend_host.cpp ${TSR_CTRL_DIR}/host/shim_host.cpp)
function(_tsr_hdr_host_lib tgt)
    cmake_parse_arguments(A "SANITIZE" "" "" ${ARGN})
    add_library(${tgt} STATIC ${_hdr_host_lib_src})
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                          POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${tgt} PUBLIC TSR_FW_HOST=1)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
    target_include_directories(${tgt} PUBLIC ${TSR_CTRL_INCLUDE_DIR})
    target_link_libraries(${tgt} PUBLIC tsr_sim tsr_ref Threads::Threads m)
    if (A_SANITIZE)
        target_compile_options(${tgt} PUBLIC -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
        target_link_options(${tgt} PUBLIC -fsanitize=address,undefined)
    endif()
endfunction()

# --- host build (ASan + UBSan) -------------------------------------------------------------------
_tsr_hdr_host_lib(tsr_fw_host_hdr_san SANITIZE)
add_executable(test_fw_hdr_host ${_hdr_src})
_tsr_hdr_test_target(test_fw_hdr_host)
target_compile_definitions(test_fw_hdr_host PRIVATE TSR_HDR_TEST_HOST=1)
target_link_libraries(test_fw_hdr_host PRIVATE tsr_fw_host_hdr_san)
add_test(NAME fw_hdr_host COMMAND test_fw_hdr_host)
set_tests_properties(fw_hdr_host PROPERTIES TIMEOUT 120)

# --- RISC-V: the reference (SW §6.8), against the host build -------------------------------------
_tsr_hdr_host_lib(tsr_fw_host_hdr)
add_executable(test_fw_hdr_rv ${_hdr_src})
_tsr_hdr_test_target(test_fw_hdr_rv)
target_compile_definitions(test_fw_hdr_rv PRIVATE TSR_HDR_TEST_RV=1)
target_link_libraries(test_fw_hdr_rv PRIVATE tsr_fw_host_hdr)
add_dependencies(test_fw_hdr_rv tsr_fw)
add_test(NAME fw_hdr_rv COMMAND test_fw_hdr_rv ${TSR_FW_TEST_ELF_fw_hdr})
set_tests_properties(fw_hdr_rv PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_test(NAME fw_hdr_rv_mutant COMMAND test_fw_hdr_rv ${TSR_FW_TEST_ELF_fw_hdr_mut} --mutant)
set_tests_properties(fw_hdr_rv_mutant PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

# --- constexpr field checks: compile failures --------------------------------------------------
set(_hdr_cx ${TSR_CTRL_DIR}/tests/fw/hdr_constexpr.cpp)
file(STRINGS ${_hdr_cx} _cx_cases REGEX "^#elif CASE == [0-9]+")
list(LENGTH _cx_cases _cx_n)
add_test(NAME fw_hdr_constexpr
         COMMAND ${CMAKE_COMMAND} -DTSR_HDR_CX_CHECK=1 -DCXX=${CMAKE_CXX_COMPILER} -DSRC=${_hdr_cx}
                 -DINC=${TSR_CTRL_INCLUDE_DIR} -DNCASES=${_cx_n} -P ${CMAKE_CURRENT_LIST_FILE})
set_tests_properties(fw_hdr_constexpr PROPERTIES TIMEOUT 120)

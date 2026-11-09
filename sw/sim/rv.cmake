# sw/sim/rv.cmake — the RISC-V side of the simulator (docs/design/06-software-verification.md
# §4.2, SW-1): tsr_rv.c (RV32IMC + Zicsr + Zifencei interpreter of the control core's Ibex,
# docs/design/04-control-core.md §3) and tsr_bus.c (firmware and test bus profiles, SBA, HTIF, ELF
# loader, and the RISC-V agent that runs the hart on the simulator's time base).
#
# Included by sw/sim/CMakeLists.txt after tsr_sim / tsr_sim_shared exist (needs tsr_c_target):
#     include(${CMAKE_CURRENT_SOURCE_DIR}/rv.cmake)
#
# Targets:
#   tsr_sim, tsr_sim_shared  gain tsr_rv.c and tsr_bus.c, compiled with TSR_RV_WITH_SIM (the agent:
#                            tsr_rv_agent_create / _destroy, tsr_bus.h) and TSR_SIM_HAVE_RV; the
#                            sources are also appended to TSR_SIM_SOURCES
#   tsr_rv                   static library: interpreter and bus profiles without the simulator
#                            (unit tests, tsr_rv_run, tools). Never link it together with tsr_sim
#                            (the same objects are inside tsr_sim)
#   tsr_rv_run               the runner of the RV32 conformance harness (sw/sim/rvtests.cmake:
#                            rv_isa, rv_random; contract in tests/rvtests/rvcommon.py)
#
# Tests (prefix rv_; run: ctest --test-dir build/dev -R '^rv_(unit|agent)' --output-on-failure):
#   rv_unit       hand-encoded instruction sequences (tests/test_rv_isa.c): every RV32I/M/C class,
#                 CSRs, traps, interrupts, WFI, debug mode, misaligned splits, the CC §4.1 map and
#                 its access rules, line buffer, SBA, HTIF, ELF loader; no toolchain needed
#   rv_unit_enc   the encoders of test_rv_isa.c against the GNU assembler: runs this file as a
#                 script (cmake -P, below); "Skipped" (77) when riscv-none-elf-as is not installed
#   rv_agent      the simulator agent (time slicing, yields, waiting pushes and their timeout,
#                 WFI, resets, state reporting) against a stub time base (tests/test_rv_agent.c)
# rv_isa, rv_arch, rv_lockstep and rv_random (riscv-tests, riscv-arch-test, lockstep against Spike)
# are defined by sw/sim/rvtests.cmake with tsr_rv_run as the runner.

# ---------------------------------------------------------------------------
# Script mode: cmake -DTSR_RV_ENC_CHECK=1 -DPREFIX=<riscv-none-elf-> -DEXE=<test_rv_isa>
#              -DOUT=<dir> -P rv.cmake
# Assembles the --dump-enc listing of test_rv_isa with the GNU assembler and compares the bytes
# (test_rv_isa --check-enc).
# ---------------------------------------------------------------------------
if (TSR_RV_ENC_CHECK)
    if (NOT EXISTS "${PREFIX}as" OR NOT EXISTS "${PREFIX}objcopy")
        message("rv_unit_enc: ${PREFIX}as not found; skipped (sw/ctrl/tools/install-toolchain.sh)")
        cmake_language(EXIT 77)
    endif()
    file(MAKE_DIRECTORY "${OUT}")
    execute_process(COMMAND "${EXE}" --dump-enc OUTPUT_VARIABLE _lst RESULT_VARIABLE _r)
    if (NOT _r EQUAL 0)
        message(FATAL_ERROR "rv_unit_enc: ${EXE} --dump-enc failed (${_r})")
    endif()
    string(REPLACE "\n" ";" _lines "${_lst}")
    set(_asm "")
    foreach(_l IN LISTS _lines)
        if (_l MATCHES "^([0-9a-f]+) (.*)$")
            string(LENGTH "${CMAKE_MATCH_1}" _n)
            if (_n EQUAL 8)
                string(APPEND _asm ".option norvc\n${CMAKE_MATCH_2}\n")
            else()
                string(APPEND _asm ".option rvc\n${CMAKE_MATCH_2}\n")
            endif()
        endif()
    endforeach()
    file(WRITE "${OUT}/enc.s" "${_asm}")
    execute_process(COMMAND "${PREFIX}as" -march=rv32imc_zicsr_zifencei -o "${OUT}/enc.o" "${OUT}/enc.s"
                    RESULT_VARIABLE _r)
    if (_r EQUAL 0)
        execute_process(COMMAND "${PREFIX}objcopy" -O binary -j .text "${OUT}/enc.o" "${OUT}/enc.bin"
                        RESULT_VARIABLE _r)
    endif()
    if (NOT _r EQUAL 0)
        message(FATAL_ERROR "rv_unit_enc: assembling ${OUT}/enc.s failed")
    endif()
    execute_process(COMMAND "${EXE}" --check-enc "${OUT}/enc.bin" RESULT_VARIABLE _r)
    if (NOT _r EQUAL 0)
        message(FATAL_ERROR "rv_unit_enc: encodings differ")
    endif()
    return()
endif()

# ---------------------------------------------------------------------------
# Sources in the simulator library
# ---------------------------------------------------------------------------
set(TSR_RV_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/tsr_rv.c
    ${CMAKE_CURRENT_LIST_DIR}/tsr_bus.c)

foreach(_tgt tsr_sim tsr_sim_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_RV_SOURCES})
        target_compile_definitions(${_tgt} PRIVATE TSR_RV_WITH_SIM=1 TSR_SIM_HAVE_RV=1)
    endif()
endforeach()
list(APPEND TSR_SIM_SOURCES ${TSR_RV_SOURCES})

# ---------------------------------------------------------------------------
# Standalone library and the conformance runner
# ---------------------------------------------------------------------------
add_library(tsr_rv STATIC ${TSR_RV_SOURCES})
tsr_c_target(tsr_rv)
target_include_directories(tsr_rv PUBLIC ${TSR_SIM_INCLUDE_DIR})

set(_rv_tests ${CMAKE_CURRENT_LIST_DIR}/tests)

add_executable(tsr_rv_run ${_rv_tests}/test_rv_run.c)
tsr_c_target(tsr_rv_run)
target_link_libraries(tsr_rv_run PRIVATE tsr_rv)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(test_rv_isa ${_rv_tests}/test_rv_isa.c)
tsr_c_target(test_rv_isa)
target_include_directories(test_rv_isa PRIVATE ${_rv_tests})
target_link_libraries(test_rv_isa PRIVATE tsr_rv m)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rv_scratch)
add_test(NAME rv_unit COMMAND test_rv_isa ${CMAKE_CURRENT_BINARY_DIR}/rv_scratch)

# The compiler prefix: sw/sim/CMakeLists.txt's TSR_RISCV_PREFIX_EFFECTIVE (cache TSR_RISCV_PREFIX,
# else the pinned xPack under TSR_RISCV_TOOLS).
if (TSR_RISCV_PREFIX_EFFECTIVE)
    set(_rv_prefix "${TSR_RISCV_PREFIX_EFFECTIVE}")
elseif (DEFINED TSR_RISCV_PREFIX AND TSR_RISCV_PREFIX)
    set(_rv_prefix "${TSR_RISCV_PREFIX}")
else()
    set(_rv_prefix "${TESSERACT_ROOT}/build/tools/riscv/xpack-riscv-none-elf-gcc-14.2.0-3/bin/riscv-none-elf-")
endif()
add_test(NAME rv_unit_enc
         COMMAND ${CMAKE_COMMAND} -DTSR_RV_ENC_CHECK=1 "-DPREFIX=${_rv_prefix}"
                 -DEXE=$<TARGET_FILE:test_rv_isa> -DOUT=${CMAKE_CURRENT_BINARY_DIR}/rv_scratch/enc
                 -P ${CMAKE_CURRENT_LIST_FILE})
set_tests_properties(rv_unit_enc PROPERTIES SKIP_RETURN_CODE 77)

# The agent glue (tsr_bus.c with TSR_RV_WITH_SIM) against a stub time base defined by the test
# itself: links neither tsr_sim nor tsr_rv.
add_executable(test_rv_agent ${_rv_tests}/test_rv_agent.c ${TSR_RV_SOURCES})
tsr_c_target(test_rv_agent)
target_compile_definitions(test_rv_agent PRIVATE TSR_RV_WITH_SIM=1)
target_include_directories(test_rv_agent PRIVATE ${TSR_SIM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR} ${_rv_tests})
target_link_libraries(test_rv_agent PRIVATE m)
add_test(NAME rv_agent COMMAND test_rv_agent)

# The agent on the real simulator: firmware booted through the host CSR port runs against the
# core model (issue port, DONE_SEQ, SPM window, RT_* mirrors, core window, timer, LCQ).
add_executable(test_rv_sim ${_rv_tests}/test_rv_sim.c)
tsr_c_target(test_rv_sim)
target_include_directories(test_rv_sim PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${_rv_tests})
target_link_libraries(test_rv_sim PRIVATE tsr_sim tsr_ref Threads::Threads m)
add_test(NAME rv_sim COMMAND test_rv_sim)

# ---------------------------------------------------------------------------
# sim_rv_issue (tests/test_sim_rv_issue.c). A test firmware (tests/rv_issue/:
# crt0.S, link.ld, fw_issue.c, a minimal EX §4.7 runtime with one kernel) built by the RISC-V
# compiler with the SW §6.1 flags (C11 instead of C++20), loaded through the host CSR port and run
# on the built-in agent: it pops launches, pushes LD / QACT / MVM / VOP / RED / ST through the
# issue port, waits for tickets, posts LCQ records and log records; the host checks everything
# against direct references, at two RV / core clock ratios. Without the compiler at configure time
# the ELF is not built and the test reports "Skipped" (77).
# ---------------------------------------------------------------------------
set(_rv_fw_dir ${_rv_tests}/rv_issue)
set(_rv_fw_elf ${CMAKE_CURRENT_BINARY_DIR}/rv_issue/fw_issue.elf)
if (EXISTS "${_rv_prefix}gcc")
    execute_process(COMMAND "${_rv_prefix}gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -print-libgcc-file-name
                    OUTPUT_VARIABLE _rv_libgcc OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rv_r)
    if (NOT _rv_r EQUAL 0 OR NOT EXISTS "${_rv_libgcc}")
        set(_rv_libgcc "")
    endif()
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rv_issue)
    add_custom_command(
        OUTPUT ${_rv_fw_elf}
        COMMAND "${_rv_prefix}gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -Os -ffreestanding
                -fno-tree-loop-distribute-patterns -ffp-contract=off -nostdlib -nostartfiles -g -std=c11
                -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Werror
                -I${TSR_SIM_INCLUDE_DIR} -T ${_rv_fw_dir}/link.ld
                -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/rv_issue/fw_issue.map
                ${_rv_fw_dir}/crt0.S ${_rv_fw_dir}/fw_issue.c ${_rv_libgcc} -o ${_rv_fw_elf}
        DEPENDS ${_rv_fw_dir}/crt0.S ${_rv_fw_dir}/fw_issue.c ${_rv_fw_dir}/link.ld
                ${TSR_SIM_INCLUDE_DIR}/tsr_isa.h ${TSR_SIM_INCLUDE_DIR}/tsr_tables.h
                ${TESSERACT_ROOT}/sw/ctrl/include/tsr/version.h
        COMMENT "RISC-V test firmware rv_issue/fw_issue.elf"
        VERBATIM)
    add_custom_target(rv_issue_fw ALL DEPENDS ${_rv_fw_elf})
else()
    message(STATUS "sim_rv_issue: no RISC-V compiler at ${_rv_prefix}gcc; the test reports Skipped "
                   "(sw/ctrl/tools/install-toolchain.sh, then reconfigure)")
endif()
add_executable(sim_rv_issue ${_rv_tests}/test_sim_rv_issue.c)
tsr_c_target(sim_rv_issue)
target_include_directories(sim_rv_issue PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${_rv_tests})
target_link_libraries(sim_rv_issue PRIVATE tsr_sim tsr_ref Threads::Threads m)
if (TARGET rv_issue_fw)
    add_dependencies(sim_rv_issue rv_issue_fw)
endif()
add_test(NAME sim_rv_issue COMMAND sim_rv_issue ${_rv_fw_elf})
set_tests_properties(sim_rv_issue PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

# sw/sim/race.cmake — the race detector (P1.9, docs/design/06-software-verification.md §4.6, SW-3 /
# SW-4): tsr_race.c (vector clocks over units / RISC-V / host, SPM_STALE; API tsr_race.h).
#
# Included by sw/sim/CMakeLists.txt's model snippet loop after tsr_sim / tsr_sim_shared and the
# units' sanitizer variants exist (-DTSR_SIM_SKIP_SNIPPETS=race leaves it out). It adds tsr_race.c
# to both libraries, to TSR_SIM_SOURCES (hence to sim_capture_ubsan / sim_queue_ubsan) and to
# sim_vu_ubsan / sim_at_ubsan; sw/sim/CMakeLists.txt then defines TSR_SIM_HAVE_RACE on each, and
# tsr_sim_create_opts attaches the detector when tsr_sim_opts.race or TSR_SIM_RACE asks for it.
#
# Tests (run: ctest --test-dir build/dev -R '^sim_(race|rv_issue_race)' --output-on-failure):
#   sim_race           tests/test_sim_race.c tile: directed tile programs through the test hooks
#                      (RAW / WAR / WAW across units, with and without a dep / FENCE / QCLEAR,
#                      same-unit order, s registers, host DRAM, ranges, a 1000-op random program,
#                      fatal mode); no toolchain needed
#   sim_race_rv        tests/test_sim_race.c rv: firmware on the built-in hart — the sim_rv_issue
#                      firmware under fatal, its mutant without the ST -> VOP dependency
#                      (fw_issue.c built with -DFW_ISSUE_MUTANT_ST_NODEP), and the kernels of
#                      tests/race/fw_race.c (RISC-V SPM accesses, DONE_SEQ / WAIT observation,
#                      SPM_STALE, EX §2.3 H1 / H2); "Skipped" (77) without the RISC-V compiler
#   sim_rv_issue_race  the W1a integration test (sim_rv_issue) with TSR_SIM_RACE=fatal
#   sim_vu_race, sim_at_race, sim_queue_race   sim_vu / sim_at / sim_queue with TSR_SIM_RACE=fatal
#                      (their queue-path programs, the AT §6.6 FA layers included, are race-free;
#                      W1b integration made sim_queue's credit test race-free)
# Overhead: `sim_race bench <ops> <0|1|2>` (not a test).

set(TSR_RACE_SOURCES ${CMAKE_CURRENT_LIST_DIR}/tsr_race.c)

foreach(_tgt tsr_sim tsr_sim_shared sim_vu_ubsan sim_at_ubsan)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_RACE_SOURCES})
    endif()
endforeach()
list(APPEND TSR_SIM_SOURCES ${TSR_RACE_SOURCES})

set(_race_tests ${CMAKE_CURRENT_LIST_DIR}/tests)
set(_race_issue_dir ${_race_tests}/rv_issue)
set(_race_bin ${CMAKE_CURRENT_BINARY_DIR}/race)
set(_race_issue_elf ${CMAKE_CURRENT_BINARY_DIR}/rv_issue/fw_issue.elf)   # built by rv.cmake
set(_race_nodep_elf ${_race_bin}/fw_issue_nodep.elf)
set(_race_fw_elf ${_race_bin}/fw_race.elf)

# The RV32 programs: the sim_rv_issue firmware's mutant and fw_race.c, with fw_issue's flags,
# crt0.S and link.ld (rv.cmake).
set(_race_pfx "${TSR_RISCV_PREFIX_EFFECTIVE}")
if (TSR_RISCV_PREFIX_EFFECTIVE AND EXISTS "${_race_pfx}gcc")
    execute_process(COMMAND "${_race_pfx}gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -print-libgcc-file-name
                    OUTPUT_VARIABLE _race_libgcc OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _race_r)
    if (NOT _race_r EQUAL 0 OR NOT EXISTS "${_race_libgcc}")
        set(_race_libgcc "")
    endif()
    file(MAKE_DIRECTORY ${_race_bin})
    set(_race_cflags -march=rv32imc_zicsr_zifencei -mabi=ilp32 -Os -ffreestanding
        -fno-tree-loop-distribute-patterns -ffp-contract=off -nostdlib -nostartfiles -g -std=c11
        -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Werror
        -I${TSR_SIM_INCLUDE_DIR} -T ${_race_issue_dir}/link.ld)
    set(_race_deps ${_race_issue_dir}/crt0.S ${_race_issue_dir}/link.ld
        ${TSR_SIM_INCLUDE_DIR}/tsr_isa.h ${TSR_SIM_INCLUDE_DIR}/tsr_tables.h)
    add_custom_command(
        OUTPUT ${_race_nodep_elf}
        COMMAND "${_race_pfx}gcc" ${_race_cflags} -DFW_ISSUE_MUTANT_ST_NODEP
                ${_race_issue_dir}/crt0.S ${_race_issue_dir}/fw_issue.c ${_race_libgcc} -o ${_race_nodep_elf}
        DEPENDS ${_race_deps} ${_race_issue_dir}/fw_issue.c
        COMMENT "RISC-V test firmware race/fw_issue_nodep.elf (mutant: ST without its VOP dependency)"
        VERBATIM)
    add_custom_command(
        OUTPUT ${_race_fw_elf}
        COMMAND "${_race_pfx}gcc" ${_race_cflags}
                ${_race_issue_dir}/crt0.S ${_race_tests}/race/fw_race.c ${_race_libgcc} -o ${_race_fw_elf}
        DEPENDS ${_race_deps} ${_race_tests}/race/fw_race.c
        COMMENT "RISC-V test firmware race/fw_race.elf"
        VERBATIM)
    add_custom_target(race_fw ALL DEPENDS ${_race_nodep_elf} ${_race_fw_elf})
else()
    message(STATUS "sim_race_rv: no RISC-V compiler at ${_race_pfx}gcc; the test reports Skipped")
endif()

find_package(Threads REQUIRED)
add_executable(sim_race ${_race_tests}/test_sim_race.c)
tsr_c_target(sim_race)
target_include_directories(sim_race PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${_race_tests})   # tsr_sim_int.h, test_util.h
target_link_libraries(sim_race PRIVATE tsr_sim tsr_ref Threads::Threads m)
foreach(_dep race_fw rv_issue_fw)
    if (TARGET ${_dep})
        add_dependencies(sim_race ${_dep})
    endif()
endforeach()

add_test(NAME sim_race COMMAND sim_race tile)
set_tests_properties(sim_race PROPERTIES TIMEOUT 60)
add_test(NAME sim_race_rv COMMAND sim_race rv ${_race_issue_elf} ${_race_nodep_elf} ${_race_fw_elf})
set_tests_properties(sim_race_rv PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
if (TARGET sim_rv_issue)
    add_test(NAME sim_rv_issue_race COMMAND sim_rv_issue ${_race_issue_elf})
    set_tests_properties(sim_rv_issue_race PROPERTIES ENVIRONMENT TSR_SIM_RACE=fatal SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
# The VU / AT unit tests' queue-path programs (incl. the AT §6.6 FA layers) are race-free.
# (sim_queue_race is defined with sim_queue in tests/CMakeLists.txt, after this snippet.)
foreach(_t vu at)
    if (TARGET sim_${_t})
        add_test(NAME sim_${_t}_race COMMAND sim_${_t})
        set_tests_properties(sim_${_t}_race PROPERTIES ENVIRONMENT TSR_SIM_RACE=fatal TIMEOUT 120)
    endif()
endforeach()

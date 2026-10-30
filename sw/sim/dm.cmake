# sw/sim/dm.cmake — the debug-module model of the simulator (P1.12, docs/v2/05-debug.md §3, §9.1;
# docs/v2/06-software-verification.md §4.2, §11): tsr_dm.c, a behavioural riscv-dbg DM implementing
# DB §9.1's DMI profile on the built-in hart (tsr_rv.c) and its bus (tsr_bus.c). Interface:
# include/tsr_dm.h.
#
# Included by sw/sim/CMakeLists.txt after rv.cmake (it needs the RISC-V model: without the tsr_rv
# target this snippet does nothing). It adds tsr_dm.c to tsr_sim and tsr_sim_shared and appends it
# to TSR_SIM_SOURCES; sw/sim/CMakeLists.txt then defines TSR_SIM_HAVE_DM on both, so
# tsr_sim_create_opts calls tsr_dm_attach (tsr_sim_int.h §5).
#
# Tests (prefix sim_; run: ctest --test-dir build/dev -R '^sim_dm' --output-on-failure):
#   sim_dm        tests/test_sim_dm.c with hand-encoded RV32 programs (no toolchain): every DB §3.5 /
#                 §3.6 DMI sequence verbatim (halt, resume, ndmreset, x_i / CSR / dpc access,
#                 cmderr clear, fence.i through the program buffer, SBA n-word reads with
#                 autoincrement and the last-word rule, n-word and 16-bit writes, every CC §4.3
#                 refusal with sberror), the profile's registers and reset values, cmderr paths
#                 (busy, exception, halt / resume, not supported), long / stuck programs, a halt
#                 from WFI, single step over 20 instructions against a free-running reference hart
#                 (fixed at two clock ratios and a random schedule), hardware breakpoints, ebreak
#                 with ebreakm and SBA-written software breakpoints, a halt while a push waits for
#                 a credit, halt-on-boot through the external request, a timer interrupt pending
#                 during a halt, state across SOFT_RESET, and (with the DBG model) the DMI window
#                 through host CSR 0xC00 + 4r gated by DBG_CFG.DMI_EN
#   sim_dm_fw     the same binary with tests/dm/fw_dm.elf (C, built by the pinned xPack compiler
#                 like tests/rv_issue): a GDB-style session (hardware breakpoint at a symbol,
#                 software breakpoint written through SBA + fence.i, step, SBA reads / writes of C
#                 globals while running); "Skipped" (77) without the compiler
#   sim_dm_ubsan  sim_dm built with -fsanitize=undefined (sources compiled in), when supported

if (NOT TARGET tsr_rv)
    message(STATUS "dm.cmake: no RISC-V model (rv.cmake skipped); the DM model is not built")
    return()
endif()

set(TSR_DM_SOURCES ${CMAKE_CURRENT_LIST_DIR}/tsr_dm.c)
foreach(_tgt tsr_sim tsr_sim_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_DM_SOURCES})
    endif()
endforeach()
list(APPEND TSR_SIM_SOURCES ${TSR_DM_SOURCES})

set(_dm_tests ${CMAKE_CURRENT_LIST_DIR}/tests)
find_package(Threads REQUIRED)

add_executable(sim_dm ${_dm_tests}/test_sim_dm.c)
tsr_c_target(sim_dm)
target_include_directories(sim_dm PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${_dm_tests})   # tsr_sim_int.h, test_util.h
target_link_libraries(sim_dm PRIVATE tsr_sim tsr_ref Threads::Threads m)
add_test(NAME sim_dm COMMAND sim_dm)
set_tests_properties(sim_dm PROPERTIES TIMEOUT 60)

# ---------------------------------------------------------------------------
# sim_dm_fw: tests/dm/fw_dm.c with tests/rv_issue's crt0.S and link.ld (the CC §4.1 map)
# ---------------------------------------------------------------------------
if (TSR_RISCV_PREFIX_EFFECTIVE)
    set(_dm_prefix "${TSR_RISCV_PREFIX_EFFECTIVE}")
else()
    set(_dm_prefix "${TESSERACT_ROOT}/build/tools/riscv/xpack-riscv-none-elf-gcc-14.2.0-3/bin/riscv-none-elf-")
endif()
set(_dm_fw_src ${_dm_tests}/dm)
set(_dm_rt_src ${_dm_tests}/rv_issue)
set(_dm_fw_elf ${CMAKE_CURRENT_BINARY_DIR}/dm/fw_dm.elf)
if (EXISTS "${_dm_prefix}gcc")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dm)
    add_custom_command(
        OUTPUT ${_dm_fw_elf}
        COMMAND "${_dm_prefix}gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -Os -ffreestanding
                -fno-tree-loop-distribute-patterns -ffp-contract=off -nostdlib -nostartfiles -g -std=c11
                -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Werror
                -I${TSR_SIM_INCLUDE_DIR} -T ${_dm_rt_src}/link.ld
                -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/dm/fw_dm.map
                ${_dm_rt_src}/crt0.S ${_dm_fw_src}/fw_dm.c -o ${_dm_fw_elf}
        DEPENDS ${_dm_rt_src}/crt0.S ${_dm_rt_src}/link.ld ${_dm_fw_src}/fw_dm.c
                ${TSR_SIM_INCLUDE_DIR}/tsr_isa.h ${TSR_SIM_INCLUDE_DIR}/tsr_tables.h
        COMMENT "RISC-V test firmware dm/fw_dm.elf"
        VERBATIM)
    add_custom_target(dm_fw ALL DEPENDS ${_dm_fw_elf})
    add_dependencies(sim_dm dm_fw)
else()
    message(STATUS "sim_dm_fw: no RISC-V compiler at ${_dm_prefix}gcc; the test reports Skipped")
endif()
add_test(NAME sim_dm_fw COMMAND sim_dm --fw ${_dm_fw_elf})
set_tests_properties(sim_dm_fw PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

# ---------------------------------------------------------------------------
# UBSan: the core, the RISC-V model, the DM and the references compiled into the test
# ---------------------------------------------------------------------------
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")
set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
check_c_source_compiles("int main(void) { return 0; }" TSR_DM_HAVE_UBSAN)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LINK_OPTIONS)
if (TSR_DM_HAVE_UBSAN)
    # The core sources (TSR_SIM_CORE_SOURCES: tsr_sim.c, tsr_q.c, tsr_sched.c) are listed here and,
    # through TSR_SIM_UBSAN_TARGETS, completed by sw/sim/CMakeLists.txt if the core gains files.
    if (DEFINED TSR_SIM_CORE_SOURCES)
        set(_dm_core ${TSR_SIM_CORE_SOURCES})
    else()
        set(_dm_core ${CMAKE_CURRENT_LIST_DIR}/tsr_sim.c ${CMAKE_CURRENT_LIST_DIR}/tsr_q.c)
    endif()
    set(_dm_ub_src ${_dm_tests}/test_sim_dm.c ${TSR_REF_SOURCES} ${_dm_core}
        ${CMAKE_CURRENT_LIST_DIR}/tsr_rv.c ${CMAKE_CURRENT_LIST_DIR}/tsr_bus.c ${TSR_DM_SOURCES})
    add_executable(sim_dm_ubsan ${_dm_ub_src})
    list(APPEND TSR_SIM_UBSAN_TARGETS sim_dm_ubsan)
    tsr_c_target(sim_dm_ubsan)
    target_include_directories(sim_dm_ubsan PRIVATE ${TSR_SIM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR} ${_dm_tests})
    target_compile_definitions(sim_dm_ubsan PRIVATE TSR_SIM_HAVE_RV=1 TSR_RV_WITH_SIM=1 TSR_SIM_HAVE_DM=1)
    target_compile_options(sim_dm_ubsan PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(sim_dm_ubsan PRIVATE -fsanitize=undefined)
    target_link_libraries(sim_dm_ubsan PRIVATE Threads::Threads m)
    add_test(NAME sim_dm_ubsan COMMAND sim_dm_ubsan)
    set_tests_properties(sim_dm_ubsan PROPERTIES TIMEOUT 120)
endif()

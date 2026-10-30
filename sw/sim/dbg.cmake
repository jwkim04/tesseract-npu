# sw/sim/dbg.cmake — the DB simulator models (P1.13, docs/design/05-debug.md §3.3, §3.7-§3.9, §5, §6,
# §7.1, §8.1-§8.4, §9.1, §10; docs/design/06-software-verification.md §11): tsr_dbg.c, the tile-level
# debug logic (stop points, QSTEP, serial mode, tile breakpoints, dispatch history, unit probes),
# the DBG register block and DB's control-window registers, the DMI gate in front of the DM model,
# DB's halt requests, RV watchpoints and halt-on-boot, completion records (REC) and the trace
# writer (TRW). Interface: include/tsr_dbg.h.
#
# Included by sw/sim/CMakeLists.txt after dm.cmake (snippet loop "rv units dm dbg race trace";
# -DTSR_SIM_SKIP_SNIPPETS=dbg leaves it out). It adds tsr_dbg.c to tsr_sim and tsr_sim_shared and
# appends it to TSR_SIM_SOURCES; sw/sim/CMakeLists.txt then defines TSR_SIM_HAVE_DBG on both, so
# tsr_sim_create_opts calls tsr_dbg_attach (tsr_sim_int.h §5). Without the RISC-V model (rv.cmake
# skipped) the model builds without its hart parts (RV watchpoints time out on RVWP_APPLY,
# RV_SAMPLE times out); without the DM model the DMI window reads 0 and halt requests go to the
# hart's debug_req_i directly.
#
# The existing sanitizer variants get the model too (DB §5.8 / §10.3: attached and unconfigured it
# changes nothing, so their tests double as the non-interference check): units.cmake's
# sim_vu_ubsan / sim_at_ubsan (TSR_SIM_HAVE_DBG from sw/sim/CMakeLists.txt's sanitizer loop) and
# dm.cmake's sim_dm_ubsan (defined here).
#
# Tests (prefix sim_; run: ctest --test-dir build/dev -R '^sim_dbg' --output-on-failure):
#   sim_dbg        tests/test_sim_dbg.c: every DBG register's access type and reset value, the DMI
#                  gate (DMI_EN, DMI_BLOCKED, core-window reads), stop points in push order
#                  (STOP_AT k holds exactly the entries from k on, DRAINED, the state equals the
#                  prefix run; RELEASE gives the unstopped result), STOP_NOW and its priority,
#                  QSTEP (one entry per step against the prefix runs, DBG_STEP_N), serial mode
#                  (push-order completions, the final state of normal mode), tile breakpoints
#                  (PIDX, OPCODE, SPM / DRAM extents with R / W, AFTER, COUNT_ONLY, TICKET,
#                  push-index wrap), the dispatch history (3 x HIST_DEPTH entries, PRESENT,
#                  HIST_CLEAR, HIST_FREEZE_ERR), REC records in completion order with their fields
#                  and irq_fast[4], TRW records (fields, monotonic times, ring wrap, TRW_STOP_FULL
#                  drops and DROP records), ERR_STOP, the hard hold against QABORT and SOFT_RESET,
#                  QSTEP through the debug-halt hold (dbg_step), RV watchpoints with HALT,
#                  halt-on-boot, STOP_HALT_RV and its one-shot rule, the halt's exact time
#                  (STATUS.RV_HALTED, DBG_STOP_CYCLES, the RVHALT record), dbg_stop freezing the
#                  ISSUE_WAIT_MAX count, the SBA refusal bit, and the §5.8 / §10.3 comparability
#                  checks (events, times, memories, records, Q_* and PERF identical with the model
#                  detached, for tile programs and for tests/rv_issue's firmware scenario, under the
#                  fixed and a random schedule); the stop / step / serial invariants also run under
#                  a random schedule. With the RISC-V model and without the DM model the hart parts
#                  run with the halt request on the hart's debug_req_i (the DM is detached for one
#                  pass when present)
#   sim_dbg_ubsan  the same test built with -fsanitize=undefined (sources compiled in), when the
#                  toolchain supports it

set(TSR_DBG_SOURCES ${CMAKE_CURRENT_LIST_DIR}/tsr_dbg.c)
foreach(_tgt tsr_sim tsr_sim_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_DBG_SOURCES})
    endif()
endforeach()
list(APPEND TSR_SIM_SOURCES ${TSR_DBG_SOURCES})

foreach(_tgt sim_vu_ubsan sim_at_ubsan)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_DBG_SOURCES})
    endif()
endforeach()
if (TARGET sim_dm_ubsan)
    target_sources(sim_dm_ubsan PRIVATE ${TSR_DBG_SOURCES})
    target_compile_definitions(sim_dm_ubsan PRIVATE TSR_SIM_HAVE_DBG=1)
endif()

set(_dbg_tests ${CMAKE_CURRENT_LIST_DIR}/tests)
find_package(Threads REQUIRED)

add_executable(sim_dbg ${_dbg_tests}/test_sim_dbg.c)
tsr_c_target(sim_dbg)
target_include_directories(sim_dbg PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${_dbg_tests})   # tsr_sim_int.h, test_util.h
target_link_libraries(sim_dbg PRIVATE tsr_sim tsr_ref Threads::Threads m)
# The test reaches the hart (tsr_rv.h) and detaches the DM only when the library has them.
if (TARGET tsr_rv)
    target_compile_definitions(sim_dbg PRIVATE TSR_SIM_HAVE_RV=1)
endif()
if (DEFINED TSR_DM_SOURCES)
    target_compile_definitions(sim_dbg PRIVATE TSR_SIM_HAVE_DM=1)
endif()
# The firmware comparability run uses rv.cmake's tests/rv_issue firmware (skipped without it).
set(_dbg_fw "")
if (DEFINED _rv_fw_elf)
    set(_dbg_fw ${_rv_fw_elf})
    if (TARGET rv_issue_fw)
        add_dependencies(sim_dbg rv_issue_fw)
    endif()
endif()
add_test(NAME sim_dbg COMMAND sim_dbg ${_dbg_fw})
set_tests_properties(sim_dbg PROPERTIES TIMEOUT 120)

# UBSan: the core, the RISC-V model, the DM (when built), this model and the references compiled
# into the test; sw/sim/CMakeLists.txt's sanitizer loop adds the core sources it lacks and the
# TSR_SIM_HAVE_* definitions of the model sources.
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")
set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
check_c_source_compiles("int main(void) { return 0; }" TSR_DBG_HAVE_UBSAN)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LINK_OPTIONS)
if (TSR_DBG_HAVE_UBSAN)
    set(_dbg_ub_src ${_dbg_tests}/test_sim_dbg.c ${TSR_REF_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/tsr_sim.c ${CMAKE_CURRENT_LIST_DIR}/tsr_q.c ${TSR_DBG_SOURCES})
    if (TARGET tsr_rv)
        list(APPEND _dbg_ub_src ${CMAKE_CURRENT_LIST_DIR}/tsr_rv.c ${CMAKE_CURRENT_LIST_DIR}/tsr_bus.c)
    endif()
    if (DEFINED TSR_DM_SOURCES)
        list(APPEND _dbg_ub_src ${TSR_DM_SOURCES})
    endif()
    set(_dbg_ub_defs "")
    if (DEFINED TSR_UNITS_SOURCES)                                   # the firmware scenario's VU ops
        list(APPEND _dbg_ub_src ${TSR_UNITS_SOURCES})
        set(_dbg_ub_defs TSR_SIM_HAVE_VU=1 TSR_SIM_HAVE_AT=1)
    endif()
    add_executable(sim_dbg_ubsan ${_dbg_ub_src})
    if (_dbg_ub_defs)
        target_compile_definitions(sim_dbg_ubsan PRIVATE ${_dbg_ub_defs})
    endif()
    if (TARGET rv_issue_fw)
        add_dependencies(sim_dbg_ubsan rv_issue_fw)
    endif()
    tsr_c_target(sim_dbg_ubsan)
    target_include_directories(sim_dbg_ubsan PRIVATE ${TSR_SIM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR} ${_dbg_tests})
    target_compile_options(sim_dbg_ubsan PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(sim_dbg_ubsan PRIVATE -fsanitize=undefined)
    target_link_libraries(sim_dbg_ubsan PRIVATE Threads::Threads m)
    add_test(NAME sim_dbg_ubsan COMMAND sim_dbg_ubsan ${_dbg_fw})
    set_tests_properties(sim_dbg_ubsan PROPERTIES TIMEOUT 300)
    list(APPEND TSR_SIM_UBSAN_TARGETS sim_dbg_ubsan)
endif()

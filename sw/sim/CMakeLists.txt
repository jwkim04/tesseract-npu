# sw/sim — Tesseract reference library (tsr_ref) and ISA simulator (tsr_sim).
#
# Spec rule 3: the reference numerics/layouts (repack, QA quantization, dot products,
# approximate math) exist exactly once, in this directory. Everything here is plain C11
# compiled with ${TSR_FP_FLAGS} (-ffp-contract=off -fno-fast-math): every f32 operation is a
# separately rounded IEEE binary32 operation.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

set(TSR_SIM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TSR_C_WARN -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wdouble-promotion)

# RISC-V tools (P1.11): sw/ctrl/tools/install-toolchain.sh installs the pins of
# docs/v2/toolchain.lock without sudo under TSR_RISCV_TOOLS. Every user in this directory (rv.cmake:
# rv_unit_enc and the sim_rv_issue firmware; rvtests.cmake: rv_isa, rv_arch, rv_lockstep,
# rv_random, ...) takes the compiler from TSR_RISCV_PREFIX_EFFECTIVE (below) and Spike from
# TSR_SPIKE (empty: TSR_RISCV_TOOLS/spike/bin/spike, resolved by tests/rvtests/rvcommon.py when a
# test runs). Tests that need a tool that is absent report "Skipped" (exit 77).
set(TSR_RISCV_TOOLS "${TESSERACT_ROOT}/build/tools/riscv" CACHE PATH
    "Install root of sw/ctrl/tools/install-toolchain.sh (RISC-V compiler, Spike, test sources)")
set(TSR_RISCV_PREFIX "" CACHE STRING
    "RISC-V compiler prefix (default: the pinned xPack under TSR_RISCV_TOOLS)")
set(TSR_SPIKE "" CACHE FILEPATH "Spike executable (default: TSR_RISCV_TOOLS/spike/bin/spike)")
if (TSR_RISCV_PREFIX)
    set(TSR_RISCV_PREFIX_EFFECTIVE "${TSR_RISCV_PREFIX}")
else()
    # xpack_gcc.dir / xpack_gcc.prefix of the lock file (the directory the install script unpacks).
    set(_tsr_xpack_dir "xpack-riscv-none-elf-gcc-14.2.0-3")
    set(_tsr_xpack_pfx "riscv-none-elf-")
    set(_tsr_lock "${TESSERACT_ROOT}/docs/v2/toolchain.lock")
    if (EXISTS "${_tsr_lock}")
        file(STRINGS "${_tsr_lock}" _tsr_lock_lines REGEX "^xpack_gcc\\.(dir|prefix) *=")
        foreach(_l ${_tsr_lock_lines})
            if (_l MATCHES "^xpack_gcc\\.dir *= *([^ #]+)")
                set(_tsr_xpack_dir "${CMAKE_MATCH_1}")
            elseif (_l MATCHES "^xpack_gcc\\.prefix *= *([^ #]+)")
                set(_tsr_xpack_pfx "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()
    set(TSR_RISCV_PREFIX_EFFECTIVE "${TSR_RISCV_TOOLS}/${_tsr_xpack_dir}/bin/${_tsr_xpack_pfx}")
endif()
# P2.1: published for sw/ctrl/host (the firmware build, SW §6.1), which reads it instead of
# deriving the prefix a second time.
set_property(GLOBAL PROPERTY TSR_RISCV_PREFIX_EFFECTIVE "${TSR_RISCV_PREFIX_EFFECTIVE}")

# Common C settings for every Tesseract C target in this directory.
function(tsr_c_target tgt)
    set_target_properties(${tgt} PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} ${TSR_C_WARN})
endfunction()

# ---------------------------------------------------------------------------
# tsr_ref — reference library (owner: M1 step 2)
#   tsr_ref         static, PIC: linked into tsr_sim, the ggml backend, tools
#   tsr_ref_shared  shared, OUTPUT_NAME tsr_ref -> libtsr_ref.so: cocotb / Python ctypes
# Approximate math (sw/sim/tsr_ref_math.c, docs/v2/02-vector-unit.md §6) is added to both
# variants and to TSR_REF_SOURCES by ref_math.cmake below, with its ref_math_* tests.
# ---------------------------------------------------------------------------
set(TSR_REF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tsr_ref.c)

add_library(tsr_ref STATIC ${TSR_REF_SOURCES})
tsr_c_target(tsr_ref)
target_include_directories(tsr_ref PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_ref PUBLIC m)

add_library(tsr_ref_shared SHARED ${TSR_REF_SOURCES})
tsr_c_target(tsr_ref_shared)
set_target_properties(tsr_ref_shared PROPERTIES OUTPUT_NAME tsr_ref)
target_include_directories(tsr_ref_shared PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_ref_shared PUBLIC m)

include(${CMAKE_CURRENT_SOURCE_DIR}/ref_math.cmake)   # tsr_ref_math.c + ref_math_* tests (P1.3)

# Optional snippets of the other W1 packages, included when present (each one owns its file):
#   refvu.cmake  VU references (P1.3)     refat.cmake  AT references (P1.3)
# They are included after tsr_ref / tsr_ref_shared exist: add sources with target_sources() to both
# and append them to TSR_REF_SOURCES (as ref_math.cmake does).
# TSR_SIM_SKIP_SNIPPETS (cache, a list of snippet names) leaves some out, e.g. while one is
# being written: -DTSR_SIM_SKIP_SNIPPETS="refat;units". "trace" also leaves out sw/tools' trace v2
# tools (they link the trace library); "trace_tools" / "trace_gen" leave out one sw/tools snippet.
# Every subset builds; the whole list builds the bare core (tsr_sim.c, tsr_q.c, tsr_sched.c).
set(TSR_SIM_SKIP_SNIPPETS "" CACHE STRING
    "snippets (sw/sim: refvu refat rv units dm dbg race trace rvtests; sw/tools: trace_tools trace_gen) not to include")
foreach(_snip refvu refat)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_snip}.cmake AND NOT _snip IN_LIST TSR_SIM_SKIP_SNIPPETS)
        include(${CMAKE_CURRENT_SOURCE_DIR}/${_snip}.cmake)
    endif()
endforeach()

# ---------------------------------------------------------------------------
# tsr_sim — simulator (owner: M1 step 3; v2: docs/v2/06-software-verification.md §4)
#   tsr_sim         static, PIC: the oracle (core, host CSR port, queue model, units, capture)
#   tsr_sim_shared  shared, OUTPUT_NAME tsr_sim -> libtsr_sim.so (tsr_ref linked in): Python
#                   ctypes, i.e. tb/common/test_common.py's whole-MVM check of mvm_model against
#                   tsr_sim_op (the oracle of the tb/mvm benches)
# Sources: tsr_sim.c (memories, host CSR port, loader, rings, SOFT_RESET, PERF, handler table, the
# tile units), tsr_q.c (issue port and control window, dispatcher / checker order, unit queues,
# SYNC, errors, time base, listeners), tsr_sched.c (P1.8: TSR_SIM_SCHED, the random schedule's
# streams and draws). sw/sim/tsr_sim_int.h is the internal interface (not installed).
# The v1 trace library (cmd.bin trace directories and their replay) is deleted (X0); trace v2
# (sw/sim/tsr_trace.c, P1.10) replaces it. Unit tests live under sw/sim/tests/ (prefix sim_).
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

set(TSR_SIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tsr_sim.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tsr_q.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tsr_sched.c)
# The core sources every simulator build compiles (also the sanitizer variants that list sources
# themselves; see the end of the snippet section).
set(TSR_SIM_CORE_SOURCES ${TSR_SIM_SOURCES})
set(TSR_SIM_INT_DIR ${CMAKE_CURRENT_SOURCE_DIR})      # tsr_sim_int.h, for sim-internal tests

add_library(tsr_sim STATIC ${TSR_SIM_SOURCES})
tsr_c_target(tsr_sim)
target_include_directories(tsr_sim PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_sim PUBLIC tsr_ref Threads::Threads)

add_library(tsr_sim_shared SHARED ${TSR_SIM_SOURCES})
tsr_c_target(tsr_sim_shared)
set_target_properties(tsr_sim_shared PROPERTIES OUTPUT_NAME tsr_sim)
target_include_directories(tsr_sim_shared PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_sim_shared PRIVATE tsr_ref Threads::Threads)

# Optional snippets of the other W1 packages, included when present (each one owns its file):
#   rv.cmake     tsr_rv.c / tsr_bus.c, the RISC-V model and its bus (P1.7)
#   units.cmake  tsr_vu.c / tsr_at.c, VU and AT unit semantics (P1.4 / P1.5)
# They are included after tsr_sim / tsr_sim_shared exist. A snippet adds simulator sources with
# target_sources() to both targets, or appends them to TSR_SIM_SOURCES (added to both below);
# either way TSR_SIM_SOURCES ends up complete for later targets (sanitizer builds). When the
# sources contain tsr_vu.c / tsr_at.c the targets get TSR_SIM_HAVE_VU / TSR_SIM_HAVE_AT, and
# tsr_sim_create registers their handlers (tsr_vu_register / tsr_at_register, tsr_sim_int.h).
# W1b model snippets (P1.9 / P1.10 / P1.12 / P1.13), each owned by its package, the same way:
#   dm.cmake     tsr_dm.c, the debug-module model (P1.12)
#   dbg.cmake    tsr_dbg.c, the DB simulator models (P1.13)
#   race.cmake   tsr_race.c, the race detector (P1.9)
#   trace.cmake  tsr_trace.c, trace v2 library, capture and sim replay (P1.10)
# When the simulator's sources contain tsr_dm.c / tsr_dbg.c / tsr_race.c / tsr_trace.c the targets
# get TSR_SIM_HAVE_DM / _DBG / _RACE / _TRACE, and tsr_sim_create_opts attaches the models
# (tsr_<m>_attach, tsr_sim_int.h §5). A snippet that adds its source to a sanitizer variant (the
# targets listed in TSR_SIM_UBSAN_TARGETS) gets the define there too (below).
foreach(_snip rv units dm dbg race trace)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_snip}.cmake AND NOT _snip IN_LIST TSR_SIM_SKIP_SNIPPETS)
        include(${CMAKE_CURRENT_SOURCE_DIR}/${_snip}.cmake)
    endif()
endforeach()
# Merge both ways on absolute paths (a snippet may use relative ones).
function(_tsr_abs_list out)
    set(_r "")
    foreach(_f ${ARGN})
        if (_f MATCHES "^\\$<")                       # generator expression: kept as is
            list(APPEND _r "${_f}")
        else()
            get_filename_component(_a "${_f}" ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
            list(APPEND _r "${_a}")
        endif()
    endforeach()
    if (_r)
        list(REMOVE_DUPLICATES _r)
    endif()
    set(${out} "${_r}" PARENT_SCOPE)
endfunction()
_tsr_abs_list(TSR_SIM_SOURCES ${TSR_SIM_SOURCES})
foreach(_tgt tsr_sim tsr_sim_shared)
    get_target_property(_have ${_tgt} SOURCES)
    _tsr_abs_list(_have ${_have})
    foreach(_src ${TSR_SIM_SOURCES})
        if (NOT _src IN_LIST _have)
            target_sources(${_tgt} PRIVATE ${_src})
        endif()
    endforeach()
    list(APPEND TSR_SIM_SOURCES ${_have})
endforeach()
list(REMOVE_DUPLICATES TSR_SIM_SOURCES)
# The TSR_SIM_HAVE_* definitions a list of simulator sources implies.
function(tsr_sim_defs_of out)
    set(_d "")
    foreach(_src ${ARGN})
        if (_src MATCHES "(^|/)tsr_vu\\.c$")
            list(APPEND _d TSR_SIM_HAVE_VU=1)
        elseif (_src MATCHES "(^|/)tsr_at\\.c$")
            list(APPEND _d TSR_SIM_HAVE_AT=1)
        elseif (_src MATCHES "(^|/)tsr_bus\\.c$")
            # tsr_bus.c's simulator agent, created by tsr_sim_create (tsr_sim_int.h §4)
            list(APPEND _d TSR_SIM_HAVE_RV=1 TSR_RV_WITH_SIM=1)
        elseif (_src MATCHES "(^|/)tsr_dm\\.c$")
            list(APPEND _d TSR_SIM_HAVE_DM=1)
        elseif (_src MATCHES "(^|/)tsr_dbg\\.c$")
            list(APPEND _d TSR_SIM_HAVE_DBG=1)
        elseif (_src MATCHES "(^|/)tsr_race\\.c$")
            list(APPEND _d TSR_SIM_HAVE_RACE=1)
        elseif (_src MATCHES "(^|/)tsr_trace\\.c$")
            list(APPEND _d TSR_SIM_HAVE_TRACE=1)
        endif()
    endforeach()
    if (_d)
        list(REMOVE_DUPLICATES _d)
    endif()
    set(${out} "${_d}" PARENT_SCOPE)
endfunction()
tsr_sim_defs_of(TSR_SIM_DEFS ${TSR_SIM_SOURCES})
if (TSR_SIM_DEFS)
    target_compile_definitions(tsr_sim PRIVATE ${TSR_SIM_DEFS})
    target_compile_definitions(tsr_sim_shared PRIVATE ${TSR_SIM_DEFS})
endif()

# Sanitizer variants that compile the simulator sources into the test themselves (units.cmake:
# sim_vu_ubsan, sim_at_ubsan list tsr_sim.c / tsr_q.c by name): they get every core source they
# lack (tsr_sched.c), and the TSR_SIM_HAVE_* definitions of the model sources a snippet added to
# them. A snippet may append its own such targets to TSR_SIM_UBSAN_TARGETS.
list(APPEND TSR_SIM_UBSAN_TARGETS sim_vu_ubsan sim_at_ubsan)
foreach(_tgt ${TSR_SIM_UBSAN_TARGETS})
    if (NOT TARGET ${_tgt})
        continue()
    endif()
    get_target_property(_have ${_tgt} SOURCES)
    _tsr_abs_list(_have ${_have})
    foreach(_src ${TSR_SIM_CORE_SOURCES})
        if (NOT _src IN_LIST _have)
            target_sources(${_tgt} PRIVATE ${_src})
        endif()
    endforeach()
    tsr_sim_defs_of(_defs ${_have})
    list(REMOVE_ITEM _defs TSR_SIM_HAVE_VU=1 TSR_SIM_HAVE_AT=1)   # units.cmake defines these itself
    if (_defs)
        target_compile_definitions(${_tgt} PRIVATE ${_defs})
    endif()
endforeach()

# Tests whose assertions include the fixed schedule's timing (op durations = cycles(), PERF busy
# counters, crossing times, RISC-V-side reads without the random schedule's lag) and that leave
# tsr_sim_opts.sched unset: an exported TSR_SIM_SCHED (SW §4.9) must not change them. sim_dbg
# reruns its own parts under random:7. (sim_csr / sim_queue: tests/CMakeLists.txt.)
foreach(_t sim_vu sim_at sim_vu_ubsan sim_at_ubsan sim_vu_race sim_at_race sim_dbg sim_dbg_ubsan)
    if (TEST ${_t})
        set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT "TSR_SIM_SCHED=")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_subdirectory(tests)
# RV32 conformance harness (P1.11): riscv-tests, riscv-arch-test, lockstep and random programs on
# Spike and tsr_rv (tests only; the runner target tsr_rv_run comes from rv.cmake).
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rvtests.cmake AND NOT "rvtests" IN_LIST TSR_SIM_SKIP_SNIPPETS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/rvtests.cmake)
endif()

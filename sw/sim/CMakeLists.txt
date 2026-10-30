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
# being written: -DTSR_SIM_SKIP_SNIPPETS="refat;units".
set(TSR_SIM_SKIP_SNIPPETS "" CACHE STRING "sw/sim snippets (refvu refat rv units rvtests) not to include")
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
# SYNC, errors, time base). sw/sim/tsr_sim_int.h is the internal interface (not installed).
# The v1 trace library (cmd.bin trace directories and their replay) is deleted (X0); trace v2
# (sw/sim/tsr_trace.c, P1.10) replaces it. Unit tests live under sw/sim/tests/ (prefix sim_).
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

set(TSR_SIM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tsr_sim.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tsr_q.c)
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
foreach(_snip rv units)
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
set(TSR_SIM_DEFS "")
foreach(_src ${TSR_SIM_SOURCES})
    if (_src MATCHES "(^|/)tsr_vu\\.c$")
        list(APPEND TSR_SIM_DEFS TSR_SIM_HAVE_VU=1)
    elseif (_src MATCHES "(^|/)tsr_at\\.c$")
        list(APPEND TSR_SIM_DEFS TSR_SIM_HAVE_AT=1)
    elseif (_src MATCHES "(^|/)tsr_bus\\.c$")
        # tsr_bus.c's simulator agent, created by tsr_sim_create (tsr_sim_int.h §4)
        list(APPEND TSR_SIM_DEFS TSR_SIM_HAVE_RV=1 TSR_RV_WITH_SIM=1)
    endif()
endforeach()
if (TSR_SIM_DEFS)
    list(REMOVE_DUPLICATES TSR_SIM_DEFS)
    target_compile_definitions(tsr_sim PRIVATE ${TSR_SIM_DEFS})
    target_compile_definitions(tsr_sim_shared PRIVATE ${TSR_SIM_DEFS})
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_subdirectory(tests)
# RV32 conformance harness (P1.11): riscv-tests, riscv-arch-test, lockstep and random programs on
# Spike and tsr_rv (tests only; the runner target tsr_rv_run comes from rv.cmake).
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rvtests.cmake AND NOT "rvtests" IN_LIST TSR_SIM_SKIP_SNIPPETS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/rvtests.cmake)
endif()

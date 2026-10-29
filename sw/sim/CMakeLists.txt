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

# ---------------------------------------------------------------------------
# tsr_sim — ISA simulator, op-level core (owner: M1 step 3; v2: docs/v2/06-software-verification.md §4)
#   tsr_sim         static, PIC: the oracle (op checks and units, SPM, DRAM, CSRs, capture)
#   tsr_sim_shared  shared, OUTPUT_NAME tsr_sim -> libtsr_sim.so (tsr_ref linked in): Python
#                   ctypes, i.e. tb/common/test_common.py's whole-MVM check of mvm_model against
#                   tsr_sim_op (the oracle of the tb/mvm benches)
# The v1 trace library (cmd.bin trace directories and their replay) is deleted (X0); trace v2
# (sw/sim/tsr_trace.c, P1.10) replaces it. Unit tests live under sw/sim/tests/ (prefix sim_).
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

add_library(tsr_sim STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tsr_sim.c)
tsr_c_target(tsr_sim)
target_include_directories(tsr_sim PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_sim PUBLIC tsr_ref Threads::Threads)

add_library(tsr_sim_shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/tsr_sim.c)
tsr_c_target(tsr_sim_shared)
set_target_properties(tsr_sim_shared PROPERTIES OUTPUT_NAME tsr_sim)
target_include_directories(tsr_sim_shared PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_sim_shared PRIVATE tsr_ref Threads::Threads)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_subdirectory(tests)

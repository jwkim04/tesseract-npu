# sw/sim — Tesseract reference library (tsr_ref), ISA simulator (tsr_sim), trace I/O (tsr_trace).
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
# Approximate math (sw/sim/tsr_ref_math.c, spec 근사 함수, M6) is to be appended to
# TSR_REF_SOURCES so both variants carry it.
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

# ---------------------------------------------------------------------------
# tsr_sim / tsr_trace — ISA simulator and trace directories (owner: M1 step 3/4)
#   tsr_sim    static, PIC: the oracle (decoder, registers, SPM, DRAM, CSRs, capture)
#   tsr_trace  static, PIC: trace directories (meta.json, cmd.bin, init/, expect/), replay
# Their unit tests live under sw/sim/tests/ (prefix sim_ / trace_).
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

add_library(tsr_sim STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tsr_sim.c)
tsr_c_target(tsr_sim)
target_include_directories(tsr_sim PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_sim PUBLIC tsr_ref Threads::Threads)

add_library(tsr_trace STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tsr_trace.c)
tsr_c_target(tsr_trace)
target_include_directories(tsr_trace PUBLIC ${TSR_SIM_INCLUDE_DIR})
target_link_libraries(tsr_trace PUBLIC tsr_sim)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_subdirectory(tests)

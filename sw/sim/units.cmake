# sw/sim/units.cmake — the VU and AT unit semantics of the simulator (P1.4 / P1.5,
# docs/v2/06-software-verification.md §4.3, §4.5, §11): tsr_vu.c (VOP, RED, ROPE, TOPK, SOP, DQROW;
# docs/v2/02-vector-unit.md §5) and tsr_at.c (KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM;
# docs/v2/03-attention-kv.md §5), each with its dispatch checks, operand extents, execution through
# the references (tsr_ref_vu.c, tsr_ref_at.c) and cycle model.
#
# Included by sw/sim/CMakeLists.txt after tsr_sim / tsr_sim_shared exist (snippet loop "rv units";
# -DTSR_SIM_SKIP_SNIPPETS=units leaves it out). It adds both sources to tsr_sim and tsr_sim_shared
# and appends them to TSR_SIM_SOURCES; sw/sim/CMakeLists.txt then defines TSR_SIM_HAVE_VU /
# TSR_SIM_HAVE_AT on both targets, so tsr_sim_create registers the handlers (tsr_vu_register,
# tsr_at_register; tsr_sim_int.h §2) and CAPS can report VU, TOPK, DQROW and ATT.
#
# Tests (prefix sim_; run: ctest --test-dir build/dev -R '^sim_(vu|at)' --output-on-failure):
#   sim_vu        tests/test_sim_vu.c: every VOP function and cmode, RED kinds / posts / TO_S, ROPE
#                 (NEOX / NORM, FF, FS, ATAB, 8 tokens, strides, in place), TOPK, SOP, DQROW against
#                 direct tsr_ref_vu calls on a copy of the SPM (whole images, bit for bit); every
#                 check code with ERR_INFO, the check order, CAPS gating, reserved VU opcodes;
#                 extents; the cycle model; PERF (VU_CYCLES, VU_LINES) on the op hook and on the
#                 queue path under ACTIVE; RMSNorm (RED RMS + VOP MULCB, and the SOP recipe), per-head
#                 norms and a softmax (RED MAX, EXPSUBC, RED SUM RCP, VOP MULC) as op sequences; the
#                 queue model's records (OPERAND, BAD_DEP, DEP_FAILED, ABORTED) for VU ops
#   sim_at        tests/test_sim_at.c: KVST (K / V, RMW, PERF, capture), ATTQK, ATTPV and the four
#                 vector ops against direct tsr_ref_at calls; every check code with ERR_INFO; CAPS;
#                 extents; the cycle model; PERF (ATT_QK / ATT_PV / MVM / VU cycles, CH_RD_BEATS,
#                 KVST_RMW, ATT_PATH_STALL, VU_LINES); complete AT §6.6 FA decode layers (KVST of the
#                 new token, QACT, then per chunk ATTQK, ASMAX, ATTPV, AMERGE, and ANORM) pushed
#                 through the queue model with dependencies, against tsr_ref_fa bit for bit
#   sim_vu_ubsan, sim_at_ubsan   the same tests built with -fsanitize=undefined (sources compiled
#                 in, -fno-sanitize-recover), when the toolchain supports it

set(TSR_UNITS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/tsr_vu.c
    ${CMAKE_CURRENT_LIST_DIR}/tsr_at.c)

foreach(_tgt tsr_sim tsr_sim_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_UNITS_SOURCES})
    endif()
endforeach()
list(APPEND TSR_SIM_SOURCES ${TSR_UNITS_SOURCES})

set(_un_tests ${CMAKE_CURRENT_LIST_DIR}/tests)
find_package(Threads REQUIRED)

foreach(_t vu at)
    add_executable(sim_${_t} ${_un_tests}/test_sim_${_t}.c)
    tsr_c_target(sim_${_t})
    target_include_directories(sim_${_t} PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${_un_tests})   # tsr_sim_int.h, test_util.h
    target_link_libraries(sim_${_t} PRIVATE tsr_sim tsr_ref Threads::Threads m)
    add_test(NAME sim_${_t} COMMAND sim_${_t})
    set_tests_properties(sim_${_t} PROPERTIES TIMEOUT 120)
endforeach()

# UBSan builds: the simulator core, the units and the references compiled into the test (no RISC-V
# model), TSR_SIM_HAVE_VU / TSR_SIM_HAVE_AT defined so the handlers register.
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")
set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
check_c_source_compiles("int main(void) { return 0; }" TSR_UNITS_HAVE_UBSAN)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LINK_OPTIONS)
if (TSR_UNITS_HAVE_UBSAN)
    foreach(_t vu at)
        add_executable(sim_${_t}_ubsan ${_un_tests}/test_sim_${_t}.c ${TSR_REF_SOURCES}
                       ${CMAKE_CURRENT_LIST_DIR}/tsr_sim.c ${CMAKE_CURRENT_LIST_DIR}/tsr_q.c ${TSR_UNITS_SOURCES})
        tsr_c_target(sim_${_t}_ubsan)
        target_include_directories(sim_${_t}_ubsan PRIVATE ${TSR_SIM_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR} ${_un_tests})
        target_compile_definitions(sim_${_t}_ubsan PRIVATE TSR_SIM_HAVE_VU=1 TSR_SIM_HAVE_AT=1)
        target_compile_options(sim_${_t}_ubsan PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
        target_link_options(sim_${_t}_ubsan PRIVATE -fsanitize=undefined)
        target_link_libraries(sim_${_t}_ubsan PRIVATE Threads::Threads m)
        add_test(NAME sim_${_t}_ubsan COMMAND sim_${_t}_ubsan)
        set_tests_properties(sim_${_t}_ubsan PROPERTIES TIMEOUT 300)
    endforeach()
endif()

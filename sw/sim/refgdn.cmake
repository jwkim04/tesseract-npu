# sw/sim/refgdn.cmake — the fused Gated DeltaNet state update reference (sw/sim/tsr_ref_gdn.c: GDN,
# VU opcode 0x39; docs/design/02-vector-unit.md §7.12 and VU-28 … VU-33) in the reference library,
# and its tests.
#
# Include from sw/sim/CMakeLists.txt after ref_math.cmake:
#     include(${CMAKE_CURRENT_SOURCE_DIR}/refgdn.cmake)
# Like refvu.cmake: after the tsr_ref / tsr_ref_shared add_library() calls it adds the source to
# both targets and to TSR_REF_SOURCES. The header is sw/sim/include/tsr_ref_gdn.h.
#
# Tests (run: ctest --test-dir build/dev -R '^ref_gdn' --output-on-failure):
#   ref_gdn     tests/test_ref_gdn.c: the fixed-point conversion against an exact rational model
#               (RNE, overflow, the FTZ edge at 2^-126 - 2^-150), integer dot products that are
#               exact, the tables, the head against the float64 recurrence, poison by Inf / NaN,
#               alpha = 0, tails, in place.
#   ref_gdn_np  tests/ref_gdn_np.py --self-test --compare libtsr_ref.so: the independent numpy model
#               (written from the text of VU §7.12.3) against the C through ctypes, bit for bit on
#               random and edge-case heads.

set(TSR_REF_GDN_SOURCE ${CMAKE_CURRENT_LIST_DIR}/tsr_ref_gdn.c)
list(APPEND TSR_REF_SOURCES ${TSR_REF_GDN_SOURCE})
foreach(_tgt tsr_ref tsr_ref_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_REF_GDN_SOURCE})
    endif()
endforeach()

set(_rg_tests ${CMAKE_CURRENT_LIST_DIR}/tests)

add_executable(ref_gdn ${_rg_tests}/test_ref_gdn.c)
tsr_c_target(ref_gdn)
target_include_directories(ref_gdn PRIVATE ${_rg_tests})
target_link_libraries(ref_gdn PRIVATE tsr_ref m)
add_test(NAME ref_gdn COMMAND ref_gdn)
set_tests_properties(ref_gdn PROPERTIES TIMEOUT 300)

# The numpy model needs numpy; ref_vu.cmake has already chosen the interpreter (_rv_py).
if (_rv_py)
    add_test(NAME ref_gdn_np COMMAND "${_rv_py}" ${_rg_tests}/ref_gdn_np.py --self-test
                                     --compare $<TARGET_FILE:tsr_ref_shared>)
    set_tests_properties(ref_gdn_np PROPERTIES TIMEOUT 600)
else()
    set(_rg_msg "ref_gdn_np: no Python with numpy was found by refvu.cmake")
    message(WARNING "${_rg_msg}")
    add_test(NAME ref_gdn_np COMMAND sh -c "echo \"$1\" >&2; exit 1" sh "${_rg_msg}")
endif()

# sw/ctrl/tests/gdn.cmake — tests of the Gated DeltaNet kinds SOFTPLUS (0x2D), SSM_CONV (0x2E) and
# GATED_DELTA_NET (0x2F) (docs/design/06-software-verification.md "SW" §7.10). Included by
# sw/ctrl/host/CMakeLists.txt (main project, after elem.cmake). Every test has label fw.
#
#   fw_gdn_rv         the kinds in tsr-fw.elf on tsr_rv through the graph executor: the 2B / 27B shapes,
#                     tails, strides, sequences, in-place state, small-SPM tiling; fixed + 7 random
#                     schedules with TSR_SIM_RACE fatal, every output bit-exact against
#                     sw/ctrl/ref/gdn_ref.cpp, SW §4.7 observables equal (test_fw_gdn_rv.cpp)
#   fw_gdn_rv_composed the same with CAPS bit 8 clear (a device without the fused GDN op): GATED_DELTA_NET takes the
#                     composition (numerics revision 1's bits) for every S_v, SW-78
#   fw_gdn_rv_errors  refused records answer BAD_RECORD
#   fw_gdn_ggml       the reference (composed and fused path) against float64 math and the pinned ggml-cpu ops

if (NOT TARGET tsr_fw_ref)
    return()                       # kernels.cmake (P2.5) provides tsr_fw_ref (mulmat_ref, dram_view)
endif()
set(_gd_dir ${TSR_CTRL_DIR}/tests)
set(_gd_inc ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/include
            ${TSR_FW_SIM_TESTS_DIR})

function(_tsr_gd_cxx tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_gd_inc})
endfunction()

# The host reference of the Gated DeltaNet kinds.
add_library(tsr_fw_gdn_ref STATIC ${TSR_CTRL_DIR}/ref/gdn_ref.cpp)
set_target_properties(tsr_fw_gdn_ref PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_fw_gdn_ref PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_fw_gdn_ref PUBLIC ${TSR_CTRL_DIR}/ref ${TSR_CTRL_INCLUDE_DIR})
target_link_libraries(tsr_fw_gdn_ref PUBLIC tsr_fw_ref tsr_ref)

# --- RISC-V image on tsr_rv (the reference, SW §6.8) ---------------------------------------------
add_executable(test_fw_gdn_rv ${_gd_dir}/test_fw_gdn_rv.cpp)
_tsr_gd_cxx(test_fw_gdn_rv)
target_link_libraries(test_fw_gdn_rv PRIVATE tsr_fw_gdn_ref tsr_fw_ref tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_gdn_rv tsr_fw)
add_test(NAME fw_gdn_rv COMMAND test_fw_gdn_rv ${TSR_FW_ELF} suite)
set_tests_properties(fw_gdn_rv PROPERTIES LABELS fw TIMEOUT 1800 SKIP_RETURN_CODE 77)
add_test(NAME fw_gdn_rv_composed COMMAND test_fw_gdn_rv ${TSR_FW_ELF} suite --gdn 0)
set_tests_properties(fw_gdn_rv_composed PROPERTIES LABELS fw TIMEOUT 1800 SKIP_RETURN_CODE 77)
add_test(NAME fw_gdn_rv_errors COMMAND test_fw_gdn_rv ${TSR_FW_ELF} errors)
set_tests_properties(fw_gdn_rv_errors PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)

# --- the reference against the pinned ggml-cpu (KL / ulp level) and float64 -----------------------
if (TSR_WITH_LLAMA)
    add_executable(test_fw_gdn_ggml ${_gd_dir}/test_fw_gdn_ggml.cpp)
    _tsr_gd_cxx(test_fw_gdn_ggml)
    target_include_directories(test_fw_gdn_ggml SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    target_link_libraries(test_fw_gdn_ggml PRIVATE tsr_fw_gdn_ref tsr_ref ggml-cpu ggml-base Threads::Threads m)
    add_test(NAME fw_gdn_ggml COMMAND test_fw_gdn_ggml)
    set_tests_properties(fw_gdn_ggml PROPERTIES LABELS fw TIMEOUT 600)
endif()

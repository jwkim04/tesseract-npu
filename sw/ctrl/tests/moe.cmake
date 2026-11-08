# sw/ctrl/tests/moe.cmake — the MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F) (docs/design/06-software-
# verification.md "SW" §7.2, §7.5, §10.2; SW-60 to SW-65). Included by sw/ctrl/host/CMakeLists.txt (main project,
# after kernels.cmake). Every test has label fw.
#
#   fw_kernels_moe_rv        the kinds in tsr-fw.elf on tsr_rv through the graph executor: routing of 128 experts
#                            (tiles of 16 tokens, strided rows, ties, NaN / inf logits), the gate-up and down
#                            forms of MUL_MAT_ID with and without the scale table, ids from the host and from
#                            a MOE_ROUTE of the same list, several ids chunks, a small SPM; fixed + 7 random
#                            schedules with TSR_SIM_RACE fatal, every output bit-exact against
#                            sw/ctrl/ref/moe_ref.cpp, SW §4.7 observables equal, written DRAM = the outputs
#   fw_kernels_moe_rv_errors an expert number outside the tensor ends the launch with TILE_ERR, refused
#                            records with BAD_RECORD
#   fw_kernels_moe_host      the kinds compiled for the host (SW §6.8) under ASan + UBSan: the planners, the
#                            selection against tsr_ref_topk, records against moe_ref (test_fw_moe_host.cpp)

if (NOT TARGET tsr_fw_ref)
    return()
endif()
set(_mo_dir ${TSR_CTRL_DIR}/tests)
set(_mo_inc ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/include
            ${TSR_FW_SIM_TESTS_DIR})

function(_tsr_mo_cxx tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_mo_inc})
endfunction()

# The host reference of the MoE kinds (tests only).
add_library(tsr_fw_moe_ref STATIC ${TSR_CTRL_DIR}/ref/moe_ref.cpp)
set_target_properties(tsr_fw_moe_ref PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_fw_moe_ref PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_fw_moe_ref PUBLIC ${TSR_CTRL_DIR}/ref ${TSR_CTRL_INCLUDE_DIR})
target_link_libraries(tsr_fw_moe_ref PUBLIC tsr_fw_ref tsr_ref)

# --- RISC-V image on tsr_rv (the reference, SW §6.8) ---------------------------------------------
add_executable(test_fw_moe_rv ${_mo_dir}/test_fw_moe_rv.cpp)
_tsr_mo_cxx(test_fw_moe_rv)
target_link_libraries(test_fw_moe_rv PRIVATE tsr_fw_moe_ref tsr_fw_ref tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_moe_rv tsr_fw)
add_test(NAME fw_kernels_moe_rv COMMAND test_fw_moe_rv ${TSR_FW_ELF} suite)
set_tests_properties(fw_kernels_moe_rv PROPERTIES LABELS fw TIMEOUT 900 SKIP_RETURN_CODE 77)
add_test(NAME fw_kernels_moe_rv_errors COMMAND test_fw_moe_rv ${TSR_FW_ELF} errors)
set_tests_properties(fw_kernels_moe_rv_errors PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)

# --- host build of the kernels under ASan + UBSan (SW §6.8) --------------------------------------
if (NOT TARGET tsr_fw_host_kern_san)
    tsr_fw_host_lib(tsr_fw_host_kern_san SANITIZE)
endif()
add_executable(test_fw_moe_host ${_mo_dir}/test_fw_moe_host.cpp)
_tsr_mo_cxx(test_fw_moe_host)
target_link_libraries(test_fw_moe_host PRIVATE tsr_fw_host_kern_san tsr_ref)
add_test(NAME fw_kernels_moe_host COMMAND test_fw_moe_host)
set_tests_properties(fw_kernels_moe_host PROPERTIES LABELS fw TIMEOUT 300
                     ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")

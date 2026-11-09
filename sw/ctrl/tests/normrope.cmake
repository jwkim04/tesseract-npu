# sw/ctrl/tests/normrope.cmake — tests of the RMS_NORM (0x14) and ROPE (0x16) kinds
# (sw/ctrl/kernels/k_norm.cpp, k_rope.cpp; docs/design/06-software-verification.md "SW" §7.5, §7.8,
# §10.2). Included by sw/ctrl/host/CMakeLists.txt (main project) after kernels.cmake. Label fw.
#
#   fw_kernels_normrope_rv          the production image on tsr_rv through `graph`: suite "decode"
#                                   (Qwen2.5-0.5B / 32B, Gemma 4 12B decode records and the other forms),
#                                   fixed + 7 random schedules, TSR_SIM_RACE fatal, rows bit-exact against
#                                   sw/ctrl/ref/rmsnorm_ref.cpp / rope_ref.cpp, written DRAM = the rows,
#                                   SW §4.7 observables equal; and the refused records (RET codes, no push)
#   fw_kernels_normrope_rv_prefill  the same for suite "prefill" (multi-tile ops on a 512 KiB SPM)
#   fw_kernels_normrope_rv_refuse   refused records (BAD_RECORD / SCRATCH as KERNEL_ERR, nothing pushed)
#   fw_kernels_normrope_rv_mutant_race  negative test: an image whose RED lacks its LD dep must be caught
#                                   by the race detector
#   fw_kernels_normrope_perf        RV cycles of work per decode record (SW §7.8; target <= 350 on
#                                   Ibex maxperf; printed per shape, regression ceiling asserted;
#                                   TSR_FW_PROF=1 adds per-function profiles), PUSH_STALL = 0
#   fw_kernels_normrope_host        the kinds compiled for the host (SW §6.8, ASan + UBSan): records
#                                   against the references, fast vs general path push for push on twin
#                                   simulators, refused records push nothing, the 2-op RMS form = the
#                                   spec's 7-op lowering
#   fw_kernels_normrope_ggml        (TSR_WITH_LLAMA builds) kernel outputs on tsr_rv against ggml-cpu
#                                   rms_norm / rope_ext: ulp and absolute differences (reported, gated)

set(_nr_dir ${TSR_CTRL_DIR}/tests)
set(_nr_inc ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/include
            ${TSR_FW_SIM_TESTS_DIR})
set(_nr_refs ${TSR_CTRL_DIR}/ref/rmsnorm_ref.cpp ${TSR_CTRL_DIR}/ref/rope_ref.cpp)

function(_tsr_nr_cxx tgt)
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
    target_include_directories(${tgt} PRIVATE ${_nr_inc})
endfunction()

# --- RISC-V image on tsr_rv (the reference, SW §6.8) ---------------------------------------------
add_executable(test_fw_normrope_rv ${_nr_dir}/test_fw_normrope.cpp ${_nr_refs})
_tsr_nr_cxx(test_fw_normrope_rv)
target_link_libraries(test_fw_normrope_rv PRIVATE tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_normrope_rv tsr_fw)
add_test(NAME fw_kernels_normrope_rv COMMAND test_fw_normrope_rv ${TSR_FW_ELF} suite --only 0)
add_test(NAME fw_kernels_normrope_rv_refuse COMMAND test_fw_normrope_rv ${TSR_FW_ELF} refuse)
add_test(NAME fw_kernels_normrope_rv_prefill COMMAND test_fw_normrope_rv ${TSR_FW_ELF} suite --only 1)
add_test(NAME fw_kernels_normrope_perf COMMAND test_fw_normrope_rv ${TSR_FW_ELF} perf)
set_tests_properties(fw_kernels_normrope_rv fw_kernels_normrope_rv_refuse fw_kernels_normrope_rv_prefill
                     fw_kernels_normrope_perf PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)

# Negative test: the image whose RMS_NORM RED lacks its dep on the tile's LD must be caught by the
# race detector (the kernel sources compiled into the image itself, NO_KERNELS, as kernels.cmake's
# mutants: DEFINES reach only an image's own sources).
file(GLOB _nr_kernels CONFIGURE_DEPENDS ${TSR_CTRL_DIR}/kernels/*.cpp)
tsr_fw_image(fw_knorm_mut_nodep NO_KERNELS SOURCES ${_nr_kernels} DEFINES TSR_KNORM_MUTANT=1)
add_test(NAME fw_kernels_normrope_rv_mutant_race COMMAND test_fw_normrope_rv ${TSR_FW_TEST_ELF_fw_knorm_mut_nodep} suite
         --only 0 --scheds 1 --race report --expect-fail)
set_tests_properties(fw_kernels_normrope_rv_mutant_race PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)

# --- host build of the kinds under ASan + UBSan (SW §6.8) ----------------------------------------
if (NOT TARGET tsr_fw_host_kern_san)
    tsr_fw_host_lib(tsr_fw_host_kern_san SANITIZE)
endif()
add_executable(test_fw_normrope_host ${_nr_dir}/test_fw_normrope.cpp ${_nr_refs})
_tsr_nr_cxx(test_fw_normrope_host)
target_link_libraries(test_fw_normrope_host PRIVATE tsr_fw_host_kern_san)
add_test(NAME fw_kernels_normrope_host COMMAND test_fw_normrope_host)
set_tests_properties(fw_kernels_normrope_host PROPERTIES LABELS fw TIMEOUT 300
                     ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")

# --- kernel outputs against ggml-cpu (needs the pinned llama.cpp) --------------------------------
if (TSR_WITH_LLAMA AND EXISTS ${_nr_dir}/test_fw_normrope_ggml.cpp)
    add_executable(test_fw_normrope_ggml ${_nr_dir}/test_fw_normrope_ggml.cpp ${_nr_refs})
    _tsr_nr_cxx(test_fw_normrope_ggml)
    target_include_directories(test_fw_normrope_ggml SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    target_link_libraries(test_fw_normrope_ggml PRIVATE tsr_sim tsr_ref ggml-cpu ggml-base Threads::Threads m)
    add_dependencies(test_fw_normrope_ggml tsr_fw)
    add_test(NAME fw_kernels_normrope_ggml COMMAND test_fw_normrope_ggml ${TSR_FW_ELF})
    set_tests_properties(fw_kernels_normrope_ggml PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)
endif()

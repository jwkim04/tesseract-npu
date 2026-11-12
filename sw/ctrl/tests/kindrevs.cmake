# sw/ctrl/tests/kindrevs.cmake — the numerics revisions of the op list kinds (docs/design/06-software-
# verification.md "SW" §6.9, SW-54). Included by sw/ctrl/host/CMakeLists.txt (main project). Label fw.
#
#   fw_kind_revs   tsr-fw.elf on tsr_rv runs a fixed corpus per kind (test_fw_kind_revs.cpp); each kind's
#                  output digest must equal its row of sw/ctrl/tests/kind_revs.txt at the revision
#                  tsr/opl.hpp kind_rev gives (outputs changed: bump the revision; a bump without a
#                  change: revert it), every corpus launch answers OK, the image's .tsr_kinds declares
#                  exactly the kinds with a revision, and each launch takes the path it is meant to (the
#                  decode fast entries of production decode, or the general path: a retire hook counts
#                  the entries into the image's functions). After a deliberate bump:
#                    <build>/sw/ctrl/host/test_fw_kind_revs <build>/fw/tsr-fw.elf sw/ctrl/tests/kind_revs.txt --update
#                  (--update applies the table rules first: it records only kinds whose revision went up)
#   fw_kind_revs_hold, fw_kind_revs_hold_r1 ... r7   the same corpus with every launch's list wrapped in a MUL_MAT_BEGIN (first record)
#                  and its MUL_MAT_END (last): every kind runs with a ring block held across its records, under
#                  TSR_SIM_RACE fatal, on the fixed schedule and on seven random ones; the digests must be the table's
#                  and the held MUL_MAT's y rows a plain MUL_MAT's (--hold --digests-only)
#   fw_basis_old   test image: the production sources linked as Tesseract Basis 0.0.5 (below)
#   fw_no_mulmat   test image: the production sources without the MUL_MAT kind declaration (below)

add_executable(test_fw_kind_revs ${TSR_CTRL_DIR}/tests/test_fw_kind_revs.cpp)
set_target_properties(test_fw_kind_revs PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_fw_kind_revs PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
target_include_directories(test_fw_kind_revs PRIVATE ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref ${TESSERACT_ROOT}/sw/sim
                           ${TESSERACT_ROOT}/sw/sim/include ${TSR_FW_SIM_TESTS_DIR})
target_link_libraries(test_fw_kind_revs PRIVATE tsr_elf tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_kind_revs tsr_fw)
add_test(NAME fw_kind_revs COMMAND test_fw_kind_revs ${TSR_FW_ELF} ${TSR_CTRL_DIR}/tests/kind_revs.txt)
set_tests_properties(fw_kind_revs PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)
add_test(NAME fw_kind_revs_hold COMMAND test_fw_kind_revs ${TSR_FW_ELF} ${TSR_CTRL_DIR}/tests/kind_revs.txt --hold --digests-only)
set_tests_properties(fw_kind_revs_hold PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)
foreach(_s 1 2 3 4 5 6 7)   # the fixed schedule and seven random ones
    add_test(NAME fw_kind_revs_hold_r${_s} COMMAND test_fw_kind_revs ${TSR_FW_ELF} ${TSR_CTRL_DIR}/tests/kind_revs.txt --hold
             --digests-only --sched random:${_s})
    set_tests_properties(fw_kind_revs_hold_r${_s} PROPERTIES LABELS fw TIMEOUT 300 SKIP_RETURN_CODE 77)
endforeach()

# A production image linked as Tesseract Basis 0.0.5 (.note.tsr word 6 and HELLO r6; SW §6.9): the
# transport boots it (the Basis rule is the backends', not the transport's: transport_protocol), the
# ggml backend refuses it as older than itself (ggml_tesseract_versions_old).
tsr_fw_image(fw_basis_old BASIS_VERSION 5)

# A production image without the MUL_MAT kind (TSR_TEST_NO_KIND_MUL_MAT leaves out its TSR_KIND; the kernel
# sources are compiled into the image itself, NO_KERNELS, since DEFINES reach only its own sources): the
# transport boots it, sw/rt refuses it (rt_sim_no_mulmat).
file(GLOB _kr_kernels CONFIGURE_DEPENDS ${TSR_CTRL_DIR}/kernels/*.cpp)
tsr_fw_image(fw_no_mulmat NO_KERNELS SOURCES ${_kr_kernels} DEFINES TSR_TEST_NO_KIND_MUL_MAT=1)

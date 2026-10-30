# sw/ctrl/tests/build.cmake — tests of the firmware build and tsr-elf (P2.1; SW §6.1-§6.2, §9.5).
# Included by sw/ctrl/host/CMakeLists.txt (main project); also run as a script (below).
#
# Test images: sw/ctrl/tests/elf/elf_img.S linked with the real sw/ctrl/rt/link.ld and
# sw/ctrl/cmake/tsr_note.S by the RISC-V compiler, without TCM --defsym (so link.ld's defaults
# are what the tests see) and with --fatal-warnings (link.ld must place every section):
#   fw_elf/elf_img.elf            five kernels, every SW §6.2 section, not stamped
#   fw_elf/elf_img_no_base.elf    no graph / selftest / replay
#   fw_elf/elf_img_big_text.elf   over SW §7.8's ITCM budget (80 KiB)
#
# ctests (label fw):
#   fw_elf        tsr-elf as a library and a tool (sw/ctrl/tests/test_fw_elf.cpp "unit"): SHA-256 /
#                 FNV-1a known answers, id golden values, the id against sw/sim/tsr_trace.c's
#                 implementation (also on sw/sim's rv_issue firmware), layout of the test images,
#                 stamping, tamper detection, every validation rule, 3,000 corrupted / truncated
#                 images, the command line
#   fw_elf_asan   the same with tsr-elf's reader under ASan / UBSan
#   fw_elf_link   link.ld's link-time checks: _start not at +0x80, a partial kernel table entry,
#                 a wrong-size .tsr_dbg, .bss running into the stacks (this file in script mode)
#   fw_elf_image  the production image tsr-fw.elf: every check with the stored id, the base kernels
#                 at 0-2, SW §7.8's budget as a failure, ids and segments against the simulator's code
#   fw_elf_hello  tsr-fw.elf booted on the simulator through the CSR loader: HELLO carries the id
#                 tsr-elf computed, the note's SDK ABI and the kernel count
#   fw_elf_test_image, _nok, _hello
#                 the tsr_fw_image() path: fw_elf_probe (production + tests/elf/k_probe.cpp: the
#                 probe kernel after the base kernels) and fw_elf_probe_nok (NO_KERNELS, DEFINES:
#                 the probe kernel alone; booted, HELLO counts one kernel)

# ---------------------------------------------------------------------------
# Script mode: cmake -DTSR_FW_ELF_LINK_CHECK=1 -DPREFIX=<riscv-none-elf-> -DSRC=<elf_img.S>
#              -DLD=<link.ld> -DNOTE=<tsr_note.S> -DOUT=<dir> -P build.cmake
# ---------------------------------------------------------------------------
if (TSR_FW_ELF_LINK_CHECK)
    file(MAKE_DIRECTORY "${OUT}")
    set(_fail 0)
    # variant;expected linker message (empty: must link)
    set(_cases
        "NONE|"
        "BAD_START|the reset entry _start must be at RV_BOOT \\+ 0x80"
        "BAD_KT|is not a whole number of kernel table entries"
        "BAD_DBG|.tsr_dbg must be 64 bytes"
        "BIG_BSS|DTCM overflow: .bss / .tsr_crash run into the stacks")
    foreach(_c IN LISTS _cases)
        string(REPLACE "|" ";" _c "${_c}")
        list(GET _c 0 _v)
        list(LENGTH _c _n)
        set(_msg "")
        if (_n GREATER 1)
            list(GET _c 1 _msg)
        endif()
        execute_process(
            COMMAND "${PREFIX}gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -g -nostdlib -nostartfiles -D${_v}
                    -T "${LD}" -Wl,--orphan-handling=warn "${SRC}" "${NOTE}" -o "${OUT}/link_${_v}.elf"
            RESULT_VARIABLE _r OUTPUT_VARIABLE _o ERROR_VARIABLE _o)
        if (_msg STREQUAL "")
            if (NOT _r EQUAL 0)
                message(SEND_ERROR "fw_elf_link: ${_v}: the good image does not link:\n${_o}")
                set(_fail 1)
            endif()
        elseif (_r EQUAL 0)
            message(SEND_ERROR "fw_elf_link: ${_v}: linked, expected the error '${_msg}'")
            set(_fail 1)
        elseif (NOT _o MATCHES "${_msg}")
            message(SEND_ERROR "fw_elf_link: ${_v}: failed without the expected message '${_msg}':\n${_o}")
            set(_fail 1)
        else()
            message(STATUS "fw_elf_link: ${_v}: refused by the linker as expected")
        endif()
    endforeach()
    if (_fail)
        message(FATAL_ERROR "fw_elf_link failed")
    endif()
    return()
endif()

# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------
set(_fwe_dir ${CMAKE_CURRENT_BINARY_DIR}/fw_elf)
set(_fwe_src ${TSR_CTRL_DIR}/tests/elf/elf_img.S)
set(_fwe_ld ${TSR_CTRL_DIR}/rt/link.ld)
set(_fwe_note ${TSR_CTRL_DIR}/cmake/tsr_note.S)
file(MAKE_DIRECTORY ${_fwe_dir})
set(_fwe_imgs "")
foreach(_v "elf_img|" "elf_img_no_base|-DNO_BASE" "elf_img_big_text|-DBIG_TEXT")
    string(REPLACE "|" ";" _v "${_v}")
    list(GET _v 0 _name)
    list(LENGTH _v _n)
    set(_def "")
    if (_n GREATER 1)
        list(GET _v 1 _def)
    endif()
    add_custom_command(
        OUTPUT ${_fwe_dir}/${_name}.elf
        COMMAND "${TSR_RISCV_PREFIX_EFFECTIVE}gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -g -nostdlib -nostartfiles
                ${_def} -T ${_fwe_ld} -Wl,--orphan-handling=warn -Wl,--fatal-warnings
                -Wl,-Map=${_fwe_dir}/${_name}.map ${_fwe_src} ${_fwe_note} -o ${_fwe_dir}/${_name}.elf
        DEPENDS ${_fwe_src} ${_fwe_ld} ${_fwe_note}
        COMMENT "fw_elf test image ${_name}.elf"
        VERBATIM)
    list(APPEND _fwe_imgs ${_fwe_dir}/${_name}.elf)
endforeach()
add_custom_target(fw_elf_images ALL DEPENDS ${_fwe_imgs})

# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
add_executable(test_fw_elf ${TSR_CTRL_DIR}/tests/test_fw_elf.cpp)
set_target_properties(test_fw_elf PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_fw_elf PRIVATE -Wall -Wextra -Wshadow)
target_link_libraries(test_fw_elf PRIVATE tsr_elf tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_elf fw_elf_images tsr-elf)

# The same driver with tsr-elf's library compiled under ASan / UBSan (the reader parses files).
add_executable(test_fw_elf_asan ${TSR_CTRL_DIR}/tests/test_fw_elf.cpp ${TSR_CTRL_DIR}/tools/tsr-elf/tsr_elf.cpp)
set_target_properties(test_fw_elf_asan PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(test_fw_elf_asan PRIVATE ${TSR_CTRL_DIR}/tools/tsr-elf ${TESSERACT_ROOT}/sw/sim/include)
target_compile_options(test_fw_elf_asan PRIVATE -Wall -Wextra -Wshadow -fsanitize=address,undefined
                       -fno-sanitize-recover=all -fno-omit-frame-pointer)
target_link_options(test_fw_elf_asan PRIVATE -fsanitize=address,undefined)
target_link_libraries(test_fw_elf_asan PRIVATE tsr_sim tsr_ref Threads::Threads m)
add_dependencies(test_fw_elf_asan fw_elf_images tsr-elf)

set(_rv_issue_elf ${CMAKE_BINARY_DIR}/sw/sim/rv_issue/fw_issue.elf)   # sw/sim/rv.cmake's test firmware
add_test(NAME fw_elf COMMAND test_fw_elf unit ${_fwe_dir} $<TARGET_FILE:tsr-elf> ${_rv_issue_elf})
add_test(NAME fw_elf_asan COMMAND test_fw_elf_asan unit ${_fwe_dir} $<TARGET_FILE:tsr-elf> ${_rv_issue_elf})
add_test(NAME fw_elf_link
         COMMAND ${CMAKE_COMMAND} -DTSR_FW_ELF_LINK_CHECK=1 "-DPREFIX=${TSR_RISCV_PREFIX_EFFECTIVE}" -DSRC=${_fwe_src}
                 -DLD=${_fwe_ld} -DNOTE=${_fwe_note} -DOUT=${_fwe_dir}/link -P ${CMAKE_CURRENT_LIST_FILE})
add_test(NAME fw_elf_image COMMAND test_fw_elf image ${TSR_FW_ELF} --base)
add_test(NAME fw_elf_hello COMMAND test_fw_elf hello ${TSR_FW_ELF})

# tsr_fw_image(): a test image with one extra kernel (production sources + k_probe.cpp) and one
# with the runtime only (NO_KERNELS, DEFINES).
tsr_fw_image(fw_elf_probe SOURCES ${TSR_CTRL_DIR}/tests/elf/k_probe.cpp)
tsr_fw_image(fw_elf_probe_nok NO_KERNELS SOURCES ${TSR_CTRL_DIR}/tests/elf/k_probe.cpp DEFINES TSR_PROBE_RET=7)
add_test(NAME fw_elf_test_image
         COMMAND test_fw_elf image ${TSR_FW_TEST_ELF_fw_elf_probe} --base --has k_fw_elf_probe)
add_test(NAME fw_elf_test_image_nok
         COMMAND test_fw_elf image ${TSR_FW_TEST_ELF_fw_elf_probe_nok} --only k_fw_elf_probe)
add_test(NAME fw_elf_test_image_hello COMMAND test_fw_elf hello ${TSR_FW_TEST_ELF_fw_elf_probe_nok})
set_tests_properties(fw_elf fw_elf_asan fw_elf_link fw_elf_image fw_elf_hello fw_elf_test_image fw_elf_test_image_nok
                     fw_elf_test_image_hello PROPERTIES LABELS fw TIMEOUT 60)

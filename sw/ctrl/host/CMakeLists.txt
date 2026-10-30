# sw/ctrl/host/CMakeLists.txt — the firmware SDK's main-project side (SW §6.1, §6.8, §9.5; P2.1).
#
# Added by the root CMakeLists.txt after sw/sim. Provides:
#   tsr_elf, tsr-elf   host library and tool for firmware images (sw/ctrl/tools/tsr-elf): image id
#                      (EX §3.2) computed and written into .tsr_id, kernel table and .note.tsr
#                      checks, the log-format table, the size report and SW §7.8's budget
#   tsr_fw             ExternalProject_Add of the firmware project sw/ctrl (its own CMake project with
#                      the RISC-V compiler, sw/ctrl/cmake/riscv32.cmake), built after tsr-elf on every
#                      build (BUILD_ALWAYS: the inner build decides what is out of date)
#   tsr_fw_paths       INTERFACE library: include directory of the generated tsr_fw_paths.h
#                      (TSR_FW_DEFAULT_IMAGE, the default image of the host libraries; TSR_IMAGE
#                      overrides it at run time, EX §10.2) and the test images' TSR_FW_TEST_IMAGE_<name>
#   tsr_fw_host        host build of the kernels (SW §6.8): every sw/ctrl/kernels/*.cpp and the host
#                      shim sw/ctrl/host/*.cpp, C++20 with TSR_FW_HOST defined (tsr/backend_host.hpp),
#                      linking tsr_sim. tsr_fw_host_lib(<target> [SANITIZE]) makes another variant,
#                      e.g. the ASan / UBSan one of fw_kernels_host.
#   tsr_fw_image(<name> [SOURCES <file> ...] [NO_KERNELS] [DEFINES <def> ...] [BASIS_VERSION <code>])
#                      requests a test image built by the firmware project from the production sources
#                      plus the extra ones (sw/ctrl/CMakeLists.txt has the same signature); sets
#                      TSR_FW_TEST_ELF_<name> (also as an internal cache variable)
# Variables for the test snippets: TSR_FW_ELF (production image), TSR_FW_BINARY_DIR, TSR_CTRL_DIR,
# TSR_CTRL_INCLUDE_DIR, TSR_FW_SIM_TESTS_DIR (sw/sim/tests: test_sim_rvhost.h, test_sim_obs.h),
# TSR_RISCV_PREFIX_EFFECTIVE.
#
# Test snippets, each owned by its package and included when present (in this order, after the
# targets above and before the ExternalProject, so that their tsr_fw_image() calls reach it):
#   sw/ctrl/tests/build.cmake    fw_elf (P2.1)          sw/ctrl/tests/hdr.cmake     (P2.3)
#   sw/ctrl/tests/runtime.cmake  fw_runtime (P2.2)      sw/ctrl/tests/exec.cmake    (P2.4)
#   sw/ctrl/tests/kernels.cmake  fw_kernels_*, fw_selftest_sim, fw_replay_sim (P2.5)
#   sw/ctrl/tests/e2e.cmake      fw_m6_e2e, fw_m6_perf (W2a integration)
# and, required (W2b): sw/ctrl/tests/elem.cmake (fw_kernels_elem_*, P2.6), kvfa.cmake (fw_kvfa_*,
# P2.7), normrope.cmake (fw_kernels_normrope_*, P2.6), m7layer.cmake (fw_m7_layer_*, which links the
# other three's references), kindrevs.cmake (fw_kind_revs: the kinds' numerics revisions, SW §6.9).
#
# The RISC-V toolchain is required (SW §6.1): configure fails with the install hint of SW §9.6
# when the compiler is absent. The prefix is sw/sim/CMakeLists.txt's (cache TSR_RISCV_PREFIX, else
# the pinned xPack under TSR_RISCV_TOOLS, docs/design/toolchain.lock).

include(ExternalProject)
find_package(Threads REQUIRED)

set(TSR_CTRL_DIR ${TESSERACT_ROOT}/sw/ctrl)
set(TSR_CTRL_INCLUDE_DIR ${TSR_CTRL_DIR}/include)
set(TSR_FW_SIM_TESTS_DIR ${TESSERACT_ROOT}/sw/sim/tests)
set(TSR_FW_BINARY_DIR ${CMAKE_BINARY_DIR}/fw)
set(TSR_FW_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

# ---------------------------------------------------------------------------
# RISC-V compiler (required)
# ---------------------------------------------------------------------------
get_property(TSR_RISCV_PREFIX_EFFECTIVE GLOBAL PROPERTY TSR_RISCV_PREFIX_EFFECTIVE)
if (NOT TSR_RISCV_PREFIX_EFFECTIVE)
    message(FATAL_ERROR "sw/ctrl/host: sw/sim did not publish TSR_RISCV_PREFIX_EFFECTIVE (add sw/sim first)")
endif()
set(_tsr_cxx "${TSR_RISCV_PREFIX_EFFECTIVE}g++")
if (NOT IS_ABSOLUTE "${_tsr_cxx}")
    find_program(_tsr_cxx_found "${_tsr_cxx}")
    if (_tsr_cxx_found)
        set(_tsr_cxx "${_tsr_cxx_found}")
    endif()
endif()
if (NOT EXISTS "${_tsr_cxx}")
    message(FATAL_ERROR
        "The RISC-V toolchain is a required dependency of the build (SW §6.1): no ${_tsr_cxx}.\n"
        "Install the pinned tools in user space, no sudo (SW §9.6, versions in docs/design/toolchain.lock):\n"
        "    sw/ctrl/tools/install-toolchain.sh            # installs under build/tools/riscv\n"
        "then reconfigure. Another install root: -DTSR_RISCV_TOOLS=<prefix of install-toolchain.sh>; "
        "another bare-metal RISC-V GCC: -DTSR_RISCV_PREFIX=<dir>/bin/riscv-none-elf-")
endif()

# ---------------------------------------------------------------------------
# tsr-elf (SW §9.5): library (also for the host decoders and loaders) and tool
# ---------------------------------------------------------------------------
set(_elf_dir ${TSR_CTRL_DIR}/tools/tsr-elf)
add_library(tsr_elf STATIC ${_elf_dir}/tsr_elf.cpp)
set_target_properties(tsr_elf PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_elf PRIVATE -Wall -Wextra -Wshadow)
# PUBLIC sw/ctrl/include: tsr_elf.h includes tsr/version.h (the Basis ABI and version, SW §6.9).
target_include_directories(tsr_elf PUBLIC ${_elf_dir} ${TSR_CTRL_INCLUDE_DIR} PRIVATE ${TESSERACT_ROOT}/sw/sim/include)

add_executable(tsr-elf ${_elf_dir}/main.cpp)
set_target_properties(tsr-elf PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(tsr-elf PRIVATE -Wall -Wextra -Wshadow)
target_link_libraries(tsr-elf PRIVATE tsr_elf)

# The versions every firmware link writes into .note.tsr (SW §6.9): TSR_BASIS_VERSION_STRING,
# TSR_FW_VERSION_DEFSYMS ... (sw/ctrl/cmake/tsr_version.cmake; the fw_elf test images link with them).
include(${TSR_CTRL_DIR}/cmake/tsr_version.cmake)
tsr_read_versions(${TESSERACT_ROOT})

# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------
set(TSR_FW_ELF ${TSR_FW_BINARY_DIR}/tsr-fw.elf)
set(TSR_FW_ELF ${TSR_FW_ELF} CACHE INTERNAL "production firmware image of this build tree (SW §6.1)")
add_library(tsr_fw_paths INTERFACE)
target_include_directories(tsr_fw_paths INTERFACE ${TSR_FW_GEN_DIR})

# ---------------------------------------------------------------------------
# Host build of the kernels (SW §6.8)
# ---------------------------------------------------------------------------
function(tsr_fw_host_lib tgt)
    cmake_parse_arguments(A "SANITIZE" "" "" ${ARGN})
    file(GLOB _src CONFIGURE_DEPENDS ${TSR_CTRL_DIR}/kernels/*.cpp ${TSR_CTRL_DIR}/host/*.cpp)
    list(SORT _src)
    if (NOT _src)   # nothing to build yet: an empty library keeps the target usable
        set(_src ${CMAKE_CURRENT_BINARY_DIR}/tsr_fw_host_empty.cpp)
        if (NOT EXISTS ${_src})
            file(WRITE ${_src} "// no sw/ctrl/kernels or sw/ctrl/host sources yet\n")
        endif()
    endif()
    add_library(${tgt} STATIC ${_src})
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                          POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${tgt} PUBLIC TSR_FW_HOST=1)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
    target_include_directories(${tgt} PUBLIC ${TSR_CTRL_INCLUDE_DIR} ${TSR_FW_GEN_DIR})
    target_link_libraries(${tgt} PUBLIC tsr_sim tsr_ref Threads::Threads m)
    if (A_SANITIZE)
        target_compile_options(${tgt} PUBLIC -fsanitize=address,undefined -fno-sanitize-recover=all
                               -fno-omit-frame-pointer)
        target_link_options(${tgt} PUBLIC -fsanitize=address,undefined)
    endif()
endfunction()
tsr_fw_host_lib(tsr_fw_host)

# ---------------------------------------------------------------------------
# Test images: tsr_fw_image() records the request for the firmware project
# ---------------------------------------------------------------------------
set_property(GLOBAL PROPERTY TSR_FW_IMAGE_CALLS "")
function(tsr_fw_image name)
    cmake_parse_arguments(A "NO_KERNELS" "BASIS_VERSION" "SOURCES;DEFINES" ${ARGN})
    # The name is a target of the firmware project: not the production image's or its libraries'.
    if (name MATCHES "^(tsr-fw|tsr_fw_rt|tsr_fw_kernels|tsr_fw_flags)$" OR NOT name MATCHES "^[A-Za-z0-9_][A-Za-z0-9_+-]*$")
        message(FATAL_ERROR "tsr_fw_image: bad image name '${name}'")
    endif()
    get_property(_names GLOBAL PROPERTY TSR_FW_IMAGE_NAMES)
    if (name IN_LIST _names)
        message(FATAL_ERROR "tsr_fw_image: image '${name}' requested twice")
    endif()
    set(_c "tsr_fw_image(${name}")
    if (A_NO_KERNELS)
        string(APPEND _c " NO_KERNELS")
    endif()
    if (A_SOURCES)
        string(APPEND _c " SOURCES")
        foreach(_s ${A_SOURCES})
            get_filename_component(_s "${_s}" ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
            string(APPEND _c " \"${_s}\"")
        endforeach()
    endif()
    if (A_DEFINES)
        string(APPEND _c " DEFINES")
        foreach(_d ${A_DEFINES})
            string(APPEND _c " \"${_d}\"")
        endforeach()
    endif()
    if (DEFINED A_BASIS_VERSION)
        string(APPEND _c " BASIS_VERSION ${A_BASIS_VERSION}")
    endif()
    string(APPEND _c ")")
    set_property(GLOBAL APPEND PROPERTY TSR_FW_IMAGE_CALLS "${_c}")
    set_property(GLOBAL APPEND PROPERTY TSR_FW_IMAGE_NAMES "${name}")
    set(TSR_FW_TEST_ELF_${name} ${TSR_FW_BINARY_DIR}/${name}.elf PARENT_SCOPE)
    set(TSR_FW_TEST_ELF_${name} ${TSR_FW_BINARY_DIR}/${name}.elf CACHE INTERNAL "test firmware image ${name}")
endfunction()

# ---------------------------------------------------------------------------
# Test snippets of the firmware packages
# ---------------------------------------------------------------------------
foreach(_snip build hdr runtime exec kernels e2e)   # e2e: W2a integration (fw_m6_e2e, fw_m6_perf)
    if (EXISTS ${TSR_CTRL_DIR}/tests/${_snip}.cmake)
        include(${TSR_CTRL_DIR}/tests/${_snip}.cmake)
    endif()
endforeach()
# W2b: the M7 kinds' tests, required (m7layer links the other three snippets' reference libraries);
# kindrevs: the kinds' numerics revisions (fw_kind_revs, SW §6.9).
foreach(_snip elem kvfa normrope m7layer kindrevs)
    include(${TSR_CTRL_DIR}/tests/${_snip}.cmake)
endforeach()
# [additive] SDK example kernel (sw/ctrl/examples/vadd.cpp, ctest fw_example_vadd)
if (EXISTS ${TSR_CTRL_DIR}/tests/examples.cmake)
    include(${TSR_CTRL_DIR}/tests/examples.cmake)
endif()
# W2a: every firmware test carries the label fw (ctest -L fw), whichever snippet added it.
get_property(_fw_tests DIRECTORY PROPERTY TESTS)
foreach(_t IN LISTS _fw_tests)
    get_test_property(${_t} LABELS _l)
    if (NOT "fw" IN_LIST _l)
        set_property(TEST ${_t} APPEND PROPERTY LABELS fw)
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Generated files: the firmware project's test-image list and tsr_fw_paths.h
# ---------------------------------------------------------------------------
get_property(_calls GLOBAL PROPERTY TSR_FW_IMAGE_CALLS)
get_property(_names GLOBAL PROPERTY TSR_FW_IMAGE_NAMES)
set(_img_file ${CMAKE_CURRENT_BINARY_DIR}/tsr_fw_images.cmake)
set(_txt "# GENERATED by sw/ctrl/host/CMakeLists.txt: test images requested with tsr_fw_image().\n")
foreach(_c IN LISTS _calls)
    string(APPEND _txt "${_c}\n")
endforeach()
file(CONFIGURE OUTPUT ${_img_file} CONTENT "${_txt}" @ONLY)

set(_hdr "// GENERATED by sw/ctrl/host/CMakeLists.txt (P2.1): firmware images of this build tree.\n")
string(APPEND _hdr "#ifndef TSR_FW_PATHS_H\n#define TSR_FW_PATHS_H\n\n")
string(APPEND _hdr "// SW §6.1: the default image of the host libraries (TSR_IMAGE overrides it, EX §10.2).\n")
string(APPEND _hdr "#define TSR_FW_DEFAULT_IMAGE \"${TSR_FW_ELF}\"\n")
string(APPEND _hdr "#define TSR_FW_IMAGE_DIR \"${TSR_FW_BINARY_DIR}\"\n")
# Only the stamped images: a .map in BUILD_BYPRODUCTS would be deleted by an outer `clean` and never
# come back (the inner link target stays up to date), breaking SW §6.1's tsr-fw.map (fix W2a).
set(_byproducts ${TSR_FW_ELF})
foreach(_n IN LISTS _names)
    string(MAKE_C_IDENTIFIER "${_n}" _id)
    string(APPEND _hdr "#define TSR_FW_TEST_IMAGE_${_id} \"${TSR_FW_BINARY_DIR}/${_n}.elf\"\n")
    list(APPEND _byproducts ${TSR_FW_BINARY_DIR}/${_n}.elf)
endforeach()
string(APPEND _hdr "\n#endif\n")
file(CONFIGURE OUTPUT ${TSR_FW_GEN_DIR}/tsr_fw_paths.h CONTENT "${_hdr}" @ONLY)

# ---------------------------------------------------------------------------
# The firmware project
# ---------------------------------------------------------------------------
set(TSR_FW_BUILD_TYPE Release CACHE STRING "Firmware build type: Release (NDEBUG) or Debug (runtime debug checks)")
ExternalProject_Add(tsr_fw
    SOURCE_DIR ${TSR_CTRL_DIR}
    BINARY_DIR ${TSR_FW_BINARY_DIR}
    CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${TSR_CTRL_DIR}/cmake/riscv32.cmake
        -DTSR_RISCV_PREFIX=${TSR_RISCV_PREFIX_EFFECTIVE}
        -DCMAKE_BUILD_TYPE=${TSR_FW_BUILD_TYPE}
        -DTSR_ELF=$<TARGET_FILE:tsr-elf>
        -DTSR_FW_IMAGES_FILE=${_img_file}
    CONFIGURE_HANDLED_BY_BUILD ON
    BUILD_ALWAYS ON
    INSTALL_COMMAND ""
    DEPENDS tsr-elf
    BUILD_BYPRODUCTS ${_byproducts})

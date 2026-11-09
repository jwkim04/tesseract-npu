# sw/ctrl/CMakeLists.txt — the firmware project (SW §6.1-§6.2): RISC-V images of the
# control core, built with the pinned bare-metal GCC. Its own CMake project, because one project
# cannot have two C++ compilers: the main build configures and builds it as
# ExternalProject_Add(tsr_fw) (sw/ctrl/host/CMakeLists.txt), passing
#   CMAKE_TOOLCHAIN_FILE  sw/ctrl/cmake/riscv32.cmake
#   TSR_RISCV_PREFIX      the compiler prefix (the main build's effective TSR_RISCV_PREFIX)
#   TSR_ELF               the host tool tsr-elf (built by the main project first)
#   TSR_FW_IMAGES_FILE    generated list of test images (tsr_fw_image() calls of the main project)
#   CMAKE_BUILD_TYPE      Release (default: NDEBUG) or Debug (runtime debug checks, SW §6.3 R10)
#
# Images (in this project's binary directory, the main build's <build>/fw/):
#   tsr-fw.elf           the production image: every sw/ctrl/rt/*.S, sw/ctrl/rt/*.cpp and
#                        sw/ctrl/kernels/*.cpp, linked with sw/ctrl/rt/link.ld, with DWARF; the image
#                        id (EX §3.2) written into .tsr_id by `tsr-elf stamp`
#   tsr-fw.map           linker map
#   tsr-fw.size.txt      tsr-elf's size report (SW §7.8 budget)
#   tsr-fw.logfmt.txt    log-format table (address, string) of .tsr_logfmt for the host decoders
#   tsr-fw.unstamped.elf the linker output before stamping (id 0)
#   <name>.elf ...       test images: tsr_fw_image(<name> SOURCES <extra.cpp> ...) = production
#                        sources plus the extra ones (NO_KERNELS: sw/ctrl/rt only plus the extra ones)
#
# A missing source set builds nothing: without sw/ctrl/rt sources no image is linked (there is no
# crt0); an empty sw/ctrl/kernels gives an image with only the extra kernels. Sources are globbed
# with CONFIGURE_DEPENDS, so adding a file is picked up by the next build.

cmake_minimum_required(VERSION 3.20)

if (NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/riscv32.cmake")
endif()
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Release (NDEBUG) or Debug" FORCE)
endif()

project(tsr_fw LANGUAGES CXX ASM)

get_filename_component(TESSERACT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TSR_CTRL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TSR_FW_LINK_SCRIPT ${TSR_CTRL_DIR}/rt/link.ld)
set(TSR_FW_NOTE_SRC ${TSR_CTRL_DIR}/cmake/tsr_note.S)

set(TSR_ELF "" CACHE FILEPATH "tsr-elf host tool (stamps the image id; built by the main project)")
if (NOT TSR_ELF)
    message(FATAL_ERROR "sw/ctrl: TSR_ELF is not set. Configure the firmware through the main project "
                        "(cmake -S . -B build/...; target tsr_fw), which builds tsr-elf first.")
endif()
set(TSR_FW_IMAGES_FILE "" CACHE FILEPATH "generated tsr_fw_image() calls of the main project (test images)")

# ---------------------------------------------------------------------------
# Memory sizes from the generated table header (CC §4.1); versions (SW §6.9)
# ---------------------------------------------------------------------------
set(_tables_h ${TESSERACT_ROOT}/sw/sim/include/tsr_tables.h)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_tables_h})
function(_tsr_table_value name out)
    file(STRINGS ${_tables_h} _l REGEX "^#define TSR_${name} +")
    if (NOT _l MATCHES "^#define TSR_${name} +(0x[0-9A-Fa-f]+|[0-9]+)u?")
        message(FATAL_ERROR "sw/ctrl: TSR_${name} not found in ${_tables_h}")
    endif()
    math(EXPR _v "${CMAKE_MATCH_1}" OUTPUT_FORMAT DECIMAL)
    set(${out} ${_v} PARENT_SCOPE)
endfunction()
_tsr_table_value(ITCM_BYTES _itcm)
_tsr_table_value(DTCM_BYTES _dtcm)
# .note.tsr words 0, 1 and 6: the Basis ABI and the Tesseract Basis version (tsr/version.h), the
# Tesseract Core capability the images need (VERSION_VALUE); passed to every link as --defsym.
include(${TSR_CTRL_DIR}/cmake/tsr_version.cmake)
tsr_read_versions(${TESSERACT_ROOT})
set(TSR_FW_ITCM_BYTES ${_itcm} CACHE STRING "ITCM size the images are linked for (RV_CFG; default from tsr_tables.h)")
set(TSR_FW_DTCM_BYTES ${_dtcm} CACHE STRING "DTCM size the images are linked for (RV_CFG; default from tsr_tables.h)")

# ---------------------------------------------------------------------------
# Flags (SW §6.1, exactly; docs/design/toolchain.lock firmware.cxxflags records the same list)
# ---------------------------------------------------------------------------
set(TSR_FW_ARCH -march=rv32imc_zicsr_zifencei -mabi=ilp32)
set(TSR_FW_CXX_FLAGS ${TSR_FW_ARCH} -Os -ffreestanding -fno-exceptions -fno-rtti -fno-threadsafe-statics
                     -fno-use-cxa-atexit -ffp-contract=off -nostdlib -nostartfiles -g -std=c++20)
set(TSR_FW_ASM_FLAGS ${TSR_FW_ARCH} -g)
# Hot path (SW §6.1): the graph executor and the kind implementations run once per op record and
# are compiled for speed; these flags come after TSR_FW_CXX_FLAGS on their command lines (the last
# -O wins). Everything else (runtime, selftest, replay, libc) stays -Os for the image budget
# (SW §7.8). docs/design/toolchain.lock firmware.cxxflags_hot records the same list.
set(TSR_FW_HOT_FLAGS -O2)
set(TSR_FW_HOT_GLOBS kernels/graph.cpp kernels/k_*.cpp)
option(TSR_FW_WERROR "Treat firmware compiler warnings as errors" ON)
set(TSR_FW_WARN -Wall -Wextra)
if (TSR_FW_WERROR)
    list(APPEND TSR_FW_WARN -Werror)
endif()
# The build-type flags are ours alone (CMake's default -O3 / -O0 would override -Os).
set(CMAKE_CXX_FLAGS "")
set(CMAKE_ASM_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
set(CMAKE_ASM_FLAGS_RELEASE "-DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_ASM_FLAGS_DEBUG "")
foreach(_c RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_CXX_FLAGS_${_c} "-DNDEBUG")
    set(CMAKE_ASM_FLAGS_${_c} "-DNDEBUG")
endforeach()

# The lock file's record of the flags must not drift from what is built.
set(_lock ${TESSERACT_ROOT}/docs/design/toolchain.lock)
if (EXISTS ${_lock})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_lock})
    foreach(_k cxxflags cxxflags_hot)
        set(_v TSR_FW_CXX_FLAGS)
        if (_k STREQUAL "cxxflags_hot")
            set(_v TSR_FW_HOT_FLAGS)
        endif()
        file(STRINGS ${_lock} _l REGEX "^firmware\\.${_k} *=")
        if (_l MATCHES "^firmware\\.${_k} *= *([^#]*)")
            string(STRIP "${CMAKE_MATCH_1}" _lock_flags)
            string(REPLACE ";" " " _ours "${${_v}}")
            if (NOT _lock_flags STREQUAL _ours)
                message(WARNING "sw/ctrl: firmware flags differ from docs/design/toolchain.lock firmware.${_k}:\n"
                                "  built: ${_ours}\n  lock:  ${_lock_flags}")
            endif()
        endif()
    endforeach()
endif()

# libgcc (integer helpers only) of the same multilib; rv32im_zicsr / ilp32 if that is absent (SW §6.1).
function(_tsr_libgcc out)
    foreach(_m "${TSR_FW_ARCH}" "-march=rv32im_zicsr;-mabi=ilp32")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${_m} -print-libgcc-file-name
                        OUTPUT_VARIABLE _p OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _r)
        if (_r EQUAL 0 AND EXISTS "${_p}")
            get_filename_component(_p "${_p}" ABSOLUTE)
            set(${out} "${_p}" PARENT_SCOPE)
            if (NOT _m STREQUAL "${TSR_FW_ARCH}")
                message(STATUS "sw/ctrl: no rv32imc/ilp32 libgcc; using the ${_m} one")
            endif()
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "sw/ctrl: no rv32imc or rv32im libgcc.a from ${CMAKE_CXX_COMPILER} -print-libgcc-file-name")
endfunction()
_tsr_libgcc(TSR_FW_LIBGCC)
message(STATUS "sw/ctrl: ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION}), libgcc ${TSR_FW_LIBGCC}, "
               "ITCM ${TSR_FW_ITCM_BYTES} B, DTCM ${TSR_FW_DTCM_BYTES} B, ${CMAKE_BUILD_TYPE}")
message(STATUS "sw/ctrl: Tesseract Basis ${TSR_BASIS_VERSION_STRING} (Basis ABI ${TSR_SDK_ABI_VALUE}), images need "
               "Tesseract Core capability ${TSR_CAP_STRING}")

# Settings shared by every firmware object.
add_library(tsr_fw_flags INTERFACE)
target_compile_options(tsr_fw_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${TSR_FW_CXX_FLAGS} ${TSR_FW_WARN}>
    $<$<COMPILE_LANGUAGE:ASM>:${TSR_FW_ASM_FLAGS}>)
target_include_directories(tsr_fw_flags INTERFACE ${TSR_CTRL_DIR}/include)

# ---------------------------------------------------------------------------
# Production sources
# ---------------------------------------------------------------------------
file(GLOB TSR_FW_RT_SOURCES CONFIGURE_DEPENDS ${TSR_CTRL_DIR}/rt/*.S ${TSR_CTRL_DIR}/rt/*.cpp)
file(GLOB TSR_FW_KERNEL_SOURCES CONFIGURE_DEPENDS ${TSR_CTRL_DIR}/kernels/*.cpp)
list(SORT TSR_FW_RT_SOURCES)
list(SORT TSR_FW_KERNEL_SOURCES)
list(TRANSFORM TSR_FW_HOT_GLOBS PREPEND ${TSR_CTRL_DIR}/ OUTPUT_VARIABLE _hot_globs)
file(GLOB TSR_FW_HOT_SOURCES CONFIGURE_DEPENDS ${_hot_globs})
if (TSR_FW_HOT_SOURCES AND TSR_FW_HOT_FLAGS)
    set_source_files_properties(${TSR_FW_HOT_SOURCES} PROPERTIES COMPILE_OPTIONS "${TSR_FW_HOT_FLAGS}")
    list(TRANSFORM TSR_FW_HOT_SOURCES REPLACE "^${TSR_CTRL_DIR}/" "" OUTPUT_VARIABLE _hot_names)
    message(STATUS "sw/ctrl: hot-path sources (${TSR_FW_HOT_FLAGS}): ${_hot_names}")
endif()

if (TSR_FW_RT_SOURCES)
    add_library(tsr_fw_rt OBJECT ${TSR_FW_RT_SOURCES} ${TSR_FW_NOTE_SRC})
    target_link_libraries(tsr_fw_rt PUBLIC tsr_fw_flags)
endif()
if (TSR_FW_KERNEL_SOURCES)
    add_library(tsr_fw_kernels OBJECT ${TSR_FW_KERNEL_SOURCES})
    target_link_libraries(tsr_fw_kernels PUBLIC tsr_fw_flags)
endif()

# ---------------------------------------------------------------------------
# tsr_fw_image(<name> [SOURCES <file> ...] [NO_KERNELS] [DEFINES <def> ...] [BASIS_VERSION <code>])
#   Links <name>.unstamped.elf from the sw/ctrl/rt objects, the sw/ctrl/kernels objects (unless
#   NO_KERNELS) and the given extra sources (compiled with DEFINES), with the Tesseract Basis version
#   <code> (major << 16 | minor << 8 | patch) instead of tsr/version.h's in .note.tsr and HELLO r6
#   when BASIS_VERSION is given (a test image of an older Basis, SW §6.9), then `tsr-elf stamp` writes
#   <name>.elf (image id in .tsr_id), <name>.size.txt and <name>.logfmt.txt. Target <name> (ALL).
#   The main project has a function of the same name and signature that records the call for this
#   project (TSR_FW_IMAGES_FILE) and sets TSR_FW_TEST_ELF_<name>.
# ---------------------------------------------------------------------------
function(tsr_fw_image name)
    cmake_parse_arguments(A "NO_KERNELS" "BASIS_VERSION" "SOURCES;DEFINES" ${ARGN})
    if (NOT TARGET tsr_fw_rt)
        message(STATUS "sw/ctrl: image ${name} not built: no sw/ctrl/rt sources yet")
        return()
    endif()
    set(_link ${name}.link)
    add_executable(${_link} $<TARGET_OBJECTS:tsr_fw_rt>)
    if (TARGET tsr_fw_kernels AND NOT A_NO_KERNELS)
        target_sources(${_link} PRIVATE $<TARGET_OBJECTS:tsr_fw_kernels>)
    endif()
    if (A_SOURCES)
        target_sources(${_link} PRIVATE ${A_SOURCES})
    endif()
    if (A_DEFINES)   # only the extra sources compile in this target (the others are object libraries)
        target_compile_definitions(${_link} PRIVATE ${A_DEFINES})
    endif()
    target_link_libraries(${_link} PRIVATE tsr_fw_flags ${TSR_FW_LIBGCC})
    set_target_properties(${_link} PROPERTIES OUTPUT_NAME ${name}.unstamped SUFFIX .elf
                          LINK_DEPENDS "${TSR_FW_LINK_SCRIPT}")
    set(_out ${CMAKE_CURRENT_BINARY_DIR})
    set(_versions ${TSR_FW_VERSION_DEFSYMS})
    if (DEFINED A_BASIS_VERSION)
        list(TRANSFORM _versions REPLACE "^-Wl,--defsym=__tsr_basis_version=.*$"
             "-Wl,--defsym=__tsr_basis_version=${A_BASIS_VERSION}")
    endif()
    target_link_options(${_link} PRIVATE ${TSR_FW_ARCH} -nostdlib -nostartfiles
        -T ${TSR_FW_LINK_SCRIPT}
        -Wl,--defsym=__tsr_itcm_bytes=${TSR_FW_ITCM_BYTES}
        -Wl,--defsym=__tsr_dtcm_bytes=${TSR_FW_DTCM_BYTES}
        ${_versions}
        -Wl,--orphan-handling=warn
        -Wl,-Map=${_out}/${name}.map)
    # SW §7.8's budget (ITCM 96 KiB, SW-75; tsr-elf's default, TSR_ELF_BUDGET_ITCM) is the production
    # image's (tsr-fw; fw_elf_image / fw_elf_test_image check it on
    # tsr-fw and on fw_elf_probe = production + one probe kernel); a test image (production code plus
    # test kernels, never shipped) is bound by the ITCM only (with the M7 kinds the
    # images with the executor's / header library's test kernels exceed the production budget).
    set(_budget "")
    if (NOT name STREQUAL "tsr-fw")
        set(_budget --budget-itcm ${TSR_FW_ITCM_BYTES})
    endif()
    add_custom_command(
        OUTPUT ${_out}/${name}.elf ${_out}/${name}.size.txt ${_out}/${name}.logfmt.txt
        COMMAND ${TSR_ELF} stamp $<TARGET_FILE:${_link}> -o ${_out}/${name}.elf
                --itcm ${TSR_FW_ITCM_BYTES} --dtcm ${TSR_FW_DTCM_BYTES} ${_budget}
                --report ${_out}/${name}.size.txt --logfmt ${_out}/${name}.logfmt.txt
        DEPENDS ${_link} ${TSR_ELF}
        COMMENT "tsr-elf stamp ${name}.elf (image id, kernel table, note, size budget)"
        VERBATIM)
    add_custom_target(${name} ALL DEPENDS ${_out}/${name}.elf)
endfunction()

tsr_fw_image(tsr-fw)

# Test images requested by the main project (sw/ctrl/tests/*.cmake call tsr_fw_image there).
if (TSR_FW_IMAGES_FILE AND EXISTS "${TSR_FW_IMAGES_FILE}")
    include(${TSR_FW_IMAGES_FILE})
endif()

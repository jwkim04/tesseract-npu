# sw/ctrl/cmake/riscv32.cmake — CMake toolchain file of the firmware project (SW §6.1, P2.1).
#
# Selects the bare-metal RISC-V GCC for sw/ctrl/CMakeLists.txt. Only the compilers and the
# try-compile mode live here; the firmware flags (SW §6.1) are set by the project itself, so that
# exactly one place spells them out.
#
# The compiler prefix is TSR_RISCV_PREFIX (e.g. /x/bin/riscv-none-elf-). The main build passes its
# effective prefix (sw/sim/CMakeLists.txt: the cache variables TSR_RISCV_PREFIX / TSR_RISCV_TOOLS,
# else the pinned xPack of docs/v2/toolchain.lock). A standalone configure of sw/ctrl without it
# falls back to the same default: the xPack directory of the lock file under build/tools/riscv
# (sw/ctrl/tools/install-toolchain.sh's default prefix).

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv32)

if (NOT TSR_RISCV_PREFIX)
    get_filename_component(_tsr_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    set(_tsr_dir "xpack-riscv-none-elf-gcc-14.2.0-3")
    set(_tsr_pfx "riscv-none-elf-")
    if (EXISTS "${_tsr_root}/docs/v2/toolchain.lock")
        file(STRINGS "${_tsr_root}/docs/v2/toolchain.lock" _tsr_lines REGEX "^xpack_gcc\\.(dir|prefix) *=")
        foreach(_l ${_tsr_lines})
            if (_l MATCHES "^xpack_gcc\\.dir *= *([^ #]+)")
                set(_tsr_dir "${CMAKE_MATCH_1}")
            elseif (_l MATCHES "^xpack_gcc\\.prefix *= *([^ #]+)")
                set(_tsr_pfx "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()
    set(TSR_RISCV_PREFIX "${_tsr_root}/build/tools/riscv/${_tsr_dir}/bin/${_tsr_pfx}")
endif()
# try_compile projects re-read this file: hand the prefix down to them.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES TSR_RISCV_PREFIX)

if (IS_ABSOLUTE "${TSR_RISCV_PREFIX}" AND NOT EXISTS "${TSR_RISCV_PREFIX}g++")
    message(FATAL_ERROR
        "RISC-V compiler not found: ${TSR_RISCV_PREFIX}g++ (SW §6.1, §9.6). Install the pinned tools "
        "without sudo with sw/ctrl/tools/install-toolchain.sh (default prefix build/tools/riscv, pins in "
        "docs/v2/toolchain.lock), or set TSR_RISCV_PREFIX to another bare-metal RISC-V GCC prefix.")
endif()

set(CMAKE_C_COMPILER   "${TSR_RISCV_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${TSR_RISCV_PREFIX}g++")
set(CMAKE_ASM_COMPILER "${TSR_RISCV_PREFIX}gcc")
set(CMAKE_AR           "${TSR_RISCV_PREFIX}ar"      CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB       "${TSR_RISCV_PREFIX}ranlib"  CACHE FILEPATH "" FORCE)
set(CMAKE_OBJCOPY      "${TSR_RISCV_PREFIX}objcopy" CACHE FILEPATH "" FORCE)
set(CMAKE_OBJDUMP      "${TSR_RISCV_PREFIX}objdump" CACHE FILEPATH "" FORCE)
set(CMAKE_SIZE         "${TSR_RISCV_PREFIX}size"    CACHE FILEPATH "")

# No C library and no start files (-nostdlib): the compiler checks build static libraries only.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# sw/ctrl/cmake/tsr_version.cmake — the versions a firmware link writes into .note.tsr (SW §6.2,
# §6.9). Included by sw/ctrl/CMakeLists.txt (the firmware project) and sw/ctrl/host/CMakeLists.txt
# (the main project: the fw_elf test images link with the same --defsym).
#
# tsr_read_versions(<root>) reads
#   <root>/sw/ctrl/include/tsr/version.h   TSR_BASIS_VERSION_MAJOR / _MINOR / _PATCH, TSR_SDK_ABI
#   <root>/sw/sim/include/tsr_tables.h     TSR_VERSION_VALUE (the Tesseract Core capability)
# with a regex (both files are reconfigure dependencies) and sets in the caller's scope
#   TSR_BASIS_VERSION_STRING   "MAJOR.MINOR.PATCH"
#   TSR_BASIS_VERSION_CODE     major << 16 | minor << 8 | patch (decimal; .note.tsr word 6, HELLO r6)
#   TSR_SDK_ABI_VALUE          the Basis ABI (.note.tsr word 0, HELLO r4[15:0])
#   TSR_CAP_MIN                VERSION_VALUE[15:0]: the capability the images need (.note.tsr word 1)
#   TSR_CAP_STRING             "MAJOR.MINOR" of TSR_CAP_MIN
#   TSR_FW_VERSION_DEFSYMS     -Wl,--defsym=__tsr_sdk_abi=..., __tsr_basis_version, __tsr_cap_min
function(tsr_read_versions root)
    set(_vh ${root}/sw/ctrl/include/tsr/version.h)
    set(_th ${root}/sw/sim/include/tsr_tables.h)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_vh} ${_th})
    foreach(_n BASIS_VERSION_MAJOR BASIS_VERSION_MINOR BASIS_VERSION_PATCH SDK_ABI)
        file(STRINGS ${_vh} _l REGEX "^#define TSR_${_n} +")
        if (NOT _l MATCHES "^#define TSR_${_n} +([0-9]+)u?")
            message(FATAL_ERROR "tsr_version.cmake: TSR_${_n} not found in ${_vh}")
        endif()
        set(_${_n} ${CMAKE_MATCH_1})
    endforeach()
    file(STRINGS ${_th} _l REGEX "^#define TSR_VERSION_VALUE +")
    if (NOT _l MATCHES "^#define TSR_VERSION_VALUE +(0x[0-9A-Fa-f]+|[0-9]+)u?")
        message(FATAL_ERROR "tsr_version.cmake: TSR_VERSION_VALUE not found in ${_th}")
    endif()
    math(EXPR _cap "(${CMAKE_MATCH_1}) & 0xFFFF" OUTPUT_FORMAT DECIMAL)
    math(EXPR _cap_major "${_cap} >> 8")
    math(EXPR _cap_minor "${_cap} & 0xFF")
    math(EXPR _code "(${_BASIS_VERSION_MAJOR} << 16) | (${_BASIS_VERSION_MINOR} << 8) | ${_BASIS_VERSION_PATCH}"
         OUTPUT_FORMAT DECIMAL)
    set(TSR_BASIS_VERSION_STRING "${_BASIS_VERSION_MAJOR}.${_BASIS_VERSION_MINOR}.${_BASIS_VERSION_PATCH}" PARENT_SCOPE)
    set(TSR_BASIS_VERSION_CODE ${_code} PARENT_SCOPE)
    set(TSR_SDK_ABI_VALUE ${_SDK_ABI} PARENT_SCOPE)
    set(TSR_CAP_MIN ${_cap} PARENT_SCOPE)
    set(TSR_CAP_STRING "${_cap_major}.${_cap_minor}" PARENT_SCOPE)
    set(TSR_FW_VERSION_DEFSYMS
        -Wl,--defsym=__tsr_sdk_abi=${_SDK_ABI}
        -Wl,--defsym=__tsr_basis_version=${_code}
        -Wl,--defsym=__tsr_cap_min=${_cap}
        PARENT_SCOPE)
endfunction()

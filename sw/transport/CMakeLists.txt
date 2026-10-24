# sw/transport — tsr_transport: host <-> device transport (spec "ggml 백엔드 통합", 전송 계층;
# docs/xrt-host.md).
#
#   tsr_transport   static, PIC (linked into libggml-tesseract.so): dispatcher, the "sim" transport
#                   (the ISA simulator in-process), the kernel-protocol layer over a kdev
#                   (tsr_kdev.cpp) with the "mock" kdev (adapter register model around the
#                   simulator), the "vkernel" kdev (client of the Verilated kernel server,
#                   tb/platform/vkernel) and, with -DTSR_WITH_XRT=ON, the "xrt" kdev (Alveo through XRT).
#
# XRT: -DTSR_WITH_XRT=ON -DTSR_XRT_ROOT=<dir with include/ and lib/> (default /opt/xilinx/xrt).
# Without an installed XRT, the 2.18.179 deb can be unpacked into build/xrt-local (docs/xrt-host.md
# §7) and used with -DTSR_XRT_ROOT=build/xrt-local/opt/xilinx/xrt; binaries find libxrt_coreutil
# through the build RPATH, and XRT finds its driver shim through XILINX_XRT or /opt/xilinx/xrt.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

add_library(tsr_transport STATIC
    tsr_transport.c
    tsr_transport_sim.c
    tsr_kdev.cpp
    tsr_kdev_mock.cpp
    tsr_kdev_vkernel.cpp)
set_target_properties(tsr_transport PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_transport PRIVATE ${TSR_FP_FLAGS}
    $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wshadow>)
target_include_directories(tsr_transport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tsr_transport PUBLIC tsr_trace tsr_sim tsr_ref)

if (TSR_WITH_XRT)
    set(TSR_XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT installation root (include/, lib/)")
    get_filename_component(_tsr_xrt_root "${TSR_XRT_ROOT}" ABSOLUTE BASE_DIR "${TESSERACT_ROOT}")
    set(_tsr_xrt_inc ${_tsr_xrt_root}/include)
    set(_tsr_xrt_lib ${_tsr_xrt_root}/lib/libxrt_coreutil.so)
    if (NOT EXISTS ${_tsr_xrt_inc}/xrt/xrt_kernel.h OR NOT EXISTS ${_tsr_xrt_inc}/experimental/xrt_xclbin.h
        OR NOT EXISTS ${_tsr_xrt_lib})
        message(FATAL_ERROR "TSR_WITH_XRT: no XRT under TSR_XRT_ROOT=${_tsr_xrt_root} "
                            "(needs include/xrt/xrt_kernel.h and lib/libxrt_coreutil.so; see docs/xrt-host.md)")
    endif()
    # The XRT headers include <uuid/uuid.h> and call libuuid inline (uuid-dev is a dependency of
    # the XRT package). Without uuid-dev, xrt-compat/uuid/uuid.h declares the few functions they
    # use, and libuuid is linked by soname when the development symlink is missing.
    set(_tsr_xrt_incs ${_tsr_xrt_inc})
    find_file(TSR_UUID_HEADER uuid/uuid.h PATHS ${_tsr_xrt_inc} NO_CACHE)
    if (NOT TSR_UUID_HEADER)
        set(TSR_UUID_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/xrt-compat/uuid/uuid.h)
        list(APPEND _tsr_xrt_incs ${CMAKE_CURRENT_SOURCE_DIR}/xrt-compat)
    endif()
    find_library(TSR_UUID_LIB NAMES uuid libuuid.so.1)
    if (NOT TSR_UUID_LIB)
        message(FATAL_ERROR "TSR_WITH_XRT: libuuid not found (package libuuid1 / uuid-dev, an XRT dependency)")
    endif()
    message(STATUS "tsr_transport: XRT from ${_tsr_xrt_root} (uuid: ${TSR_UUID_HEADER}, ${TSR_UUID_LIB})")
    # tsr_xrt: XRT native API (headers + libxrt_coreutil) for any target that calls XRT directly
    # (the transport, sw/tools/tsr_bw).
    add_library(tsr_xrt INTERFACE)
    target_include_directories(tsr_xrt SYSTEM INTERFACE ${_tsr_xrt_incs})
    target_link_libraries(tsr_xrt INTERFACE ${_tsr_xrt_lib} ${TSR_UUID_LIB})
    target_sources(tsr_transport PRIVATE tsr_transport_xrt.cpp)
    target_compile_definitions(tsr_transport PRIVATE TSR_HAVE_XRT=1)
    target_link_libraries(tsr_transport PRIVATE tsr_xrt)
endif()

# ---------------------------------------------------------------------------
# Tests (prefix transport_). No hardware: the protocol layer runs on the mock kdev.
# ---------------------------------------------------------------------------
add_executable(test-tsr-kdev tests/test_kdev.cpp)
set_target_properties(test-tsr-kdev PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(test-tsr-kdev PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_link_libraries(test-tsr-kdev PRIVATE tsr_transport)
# Arg packing, window offsets, start / done, timeout -> ABORT, ADAPTER_STATUS errors, chunked
# transfers at odd offsets / sizes under several DMA rules, offsets beyond 4 GiB.
add_test(NAME transport_kdev COMMAND test-tsr-kdev)

# The xrt kdev against XRT's userspace with its no-op shim (no card, no driver): xclbin load,
# kernel / argument / memory-group discovery, device-only buffers, xrt::run start / wait, and
# refusal of xclbins that break the kernel contract.
if (TSR_WITH_XRT AND EXISTS ${_tsr_xrt_root}/lib/libxrt_noop.so)
    add_executable(test-tsr-xrt-noop tests/test_xrt_noop.cpp)
    set_target_properties(test-tsr-xrt-noop PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(test-tsr-xrt-noop PRIVATE -Wall -Wextra)
    target_link_libraries(test-tsr-xrt-noop PRIVATE tsr_transport tsr_xrt)
    add_test(NAME transport_xrt_noop
             COMMAND test-tsr-xrt-noop ${_tsr_xrt_root} ${CMAKE_CURRENT_BINARY_DIR}/xrt_noop)
endif()

# The ggml backend end to end on the kernel protocol (mock kdev; DMA rules that force the
# bounce / read-modify-write path). test-ggml-tesseract is created after this directory
# (llama.cpp), so the test is added by a deferred call.
function(tsr_add_transport_mock_e2e)
    if (TARGET test-ggml-tesseract)
        add_test(NAME transport_ggml_e2e_mock COMMAND test-ggml-tesseract)
        set_tests_properties(transport_ggml_e2e_mock PROPERTIES
            ENVIRONMENT "TSR_DEVICE=mock;TSR_MOCK_DMA_ALIGN=64;TSR_MOCK_DMA_CHUNK_KIB=64")
        add_test(NAME transport_ggml_e2e_mock_tmax8 COMMAND test-ggml-tesseract)
        set_tests_properties(transport_ggml_e2e_mock_tmax8 PROPERTIES
            ENVIRONMENT "TSR_DEVICE=mock;TSR_MOCK_TMAX=8;TSR_MOCK_NUM_CH=3;TSR_MOCK_DMA_ALIGN=4096;TSR_MOCK_DMA_CHUNK_KIB=12")
    endif()
endfunction()
cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_transport_mock_e2e)

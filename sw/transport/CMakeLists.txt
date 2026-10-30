# sw/transport — tsr_transport: host <-> device transport (docs/design/01-execution-model.md "EX" §10;
# docs/design/06-software-verification.md "SW" §9.1-§9.2; docs/design/05-debug.md "DB" §4.1, §7.3, §9.2).
#
#   tsr_transport   static, PIC (linked into libggml-tesseract.so, tsr_rt, the tools): the C API
#                   (tsr_transport.h), one protocol layer for every kind (tsr_kdev.cpp: open, bind,
#                   boot with VERIFY and warm restart, launch, wait with the debug-aware timeout and
#                   DB's recovery ladder, TRAP handling, cancel, stop, soft reset, log), firmware
#                   images (tsr_image.cpp: segments, CRC, .note.tsr, symbols and DWARF lines), the log
#                   decoder (tsr_logdec.cpp), and the kdevs: "sim" (the simulator in-process behind the
#                   core CSR port, tsr_kdev_sim.cpp), "mock" (the sim kdev in the ap_ctrl_none adapter
#                   model with fault injection, tsr_kdev_mock.cpp), "vkernel" (client of the Verilated
#                   kernel server; the two-clock server is P5.3, until then the open says so) and,
#                   with -DTSR_WITH_XRT=ON, "xrt" (xrt::ip with exclusive access + one device-only buffer
#                   per channel, tsr_transport_xrt.cpp; on the card from P6).
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
    tsr_kdev.cpp
    tsr_kdev_sim.cpp
    tsr_kdev_mock.cpp
    tsr_kdev_vkernel.cpp
    tsr_image.cpp
    tsr_logdec.cpp)
set_target_properties(tsr_transport PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_transport PRIVATE ${TSR_FP_FLAGS}
    $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wshadow>)
target_include_directories(tsr_transport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${TESSERACT_ROOT}/sw/sim)   # tsr_sim_int.h: the sim kdev's "nothing pending" test, the TCMs
find_package(Threads REQUIRED)
# tsr_elf (sw/ctrl/host): image parsing, id, kernel table, .tsr_logfmt; tsr_fw_paths: the build's
# default image (TSR_FW_DEFAULT_IMAGE, tsr_image_open_default).
target_link_libraries(tsr_transport PUBLIC tsr_sim tsr_ref PRIVATE tsr_elf tsr_fw_paths Threads::Threads)

if (TSR_WITH_XRT)
    set(TSR_XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT installation root (include/, lib/)")
    get_filename_component(_tsr_xrt_root "${TSR_XRT_ROOT}" ABSOLUTE BASE_DIR "${TESSERACT_ROOT}")
    set(_tsr_xrt_inc ${_tsr_xrt_root}/include)
    set(_tsr_xrt_lib ${_tsr_xrt_root}/lib/libxrt_coreutil.so)
    if (NOT EXISTS ${_tsr_xrt_inc}/xrt/xrt_kernel.h OR NOT EXISTS ${_tsr_xrt_inc}/experimental/xrt_xclbin.h
        OR NOT EXISTS ${_tsr_xrt_inc}/experimental/xrt_ip.h OR NOT EXISTS ${_tsr_xrt_lib})
        message(FATAL_ERROR "TSR_WITH_XRT: no XRT under TSR_XRT_ROOT=${_tsr_xrt_root} "
                            "(needs include/xrt/xrt_kernel.h, include/experimental/xrt_ip.h and lib/libxrt_coreutil.so; "
                            "see docs/xrt-host.md)")
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
# Tests (prefix transport_; SW §10.2). No hardware. Firmware: the production image tsr-fw.elf and
# the fw_rt test image (sw/ctrl/tests/runtime.cmake: the runtime's test kernels rt_args0-7, rt_nop,
# rt_ret, rt_copy, rt_tile_err, rt_stop, rt_cancel, rt_log, rt_trap), both from tsr_fw_paths.h.
# ---------------------------------------------------------------------------
function(tsr_transport_test tgt src)
    add_executable(${tgt} ${src})
    set_target_properties(${tgt} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(${tgt} PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
    target_include_directories(${tgt} PRIVATE ${TESSERACT_ROOT}/sw/sim)
    target_link_libraries(${tgt} PRIVATE tsr_transport tsr_elf tsr_fw_paths Threads::Threads)
    if (TARGET tsr_fw)
        add_dependencies(${tgt} tsr_fw)
    endif()
endfunction()

# transport_kdev: the protocol layer on the mock kdev (SW §10.2): identity checks and adapter
# faults at open (ID, ISA major, ADAPTER_VERSION, NUM_CH, RV_FREQ = 0, ADAPTER_STATUS left by an
# earlier process), bind (MEM_EN, misaligned base, BASE_LOCKED, held requests), timeouts (SOFT_RESET
# stuck, no HELLO), the stale core at open (a booted device with a launch running left by a "dead
# process", with and without TSR_ATTACH), adapter errors during a launch (PLAT_ERR), a hung unit
# ("card reset needed"), the registry and TSR_MOCK_* parsing, and the chunked / aligned transfer
# engine (byte-granular transfers under DMA rules from align 1 to 4096, across 4 GiB).
tsr_transport_test(test-tsr-kdev tests/test_kdev.cpp)
add_test(NAME transport_kdev COMMAND test-tsr-kdev)
set_tests_properties(transport_kdev PROPERTIES TIMEOUT 120)

# transport_protocol: EX §15 item 2, one binary for every kind it can run (sim, mock; vkernel when
# TSR_VKERNEL_SERVER / _SOCKET name a two-clock server, P5.3): cold / warm boot, image-id mismatch,
# .note.tsr refusal, VERIFY with an injected TCM fault (mock), PING, 0-7 args, BAD_DESC, 8
# outstanding and the credit rule, forced LQ_OVERFLOW and RING_RD_BAD, events while waiting, the
# log (FMT / TEXT / EVENT / BLOB decode, drop), cancel and stop, SOFT_RESET mid-launch, the
# debug-aware timeout with a hart halted through the DM model, the recovery ladder (levels 2 and
# 3), TRAP handling (decode with mepc -> function / line against addr2line, warm restart), the
# minimal record reads against full reads; records r0-r6 and DRAM identical across the kinds;
# the EX §4.11 numbers on sim (boot, empty-launch round trip, pipelined rate).
tsr_transport_test(test-tsr-protocol tests/test_protocol.cpp)
get_property(_tsr_pfx GLOBAL PROPERTY TSR_RISCV_PREFIX_EFFECTIVE)   # sw/sim: the pinned RISC-V tools
set(_tsr_addr2line "${_tsr_pfx}addr2line")
add_test(NAME transport_protocol COMMAND test-tsr-protocol --addr2line ${_tsr_addr2line}
         --scratch ${CMAKE_CURRENT_BINARY_DIR}/protocol)
set_tests_properties(transport_protocol PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

# transport_xrt_noop: the xrt kdev against XRT's userspace with its no-op shim (no card, no driver):
# xclbin load, kernel / argument / memory discovery from the xclbin, xrt::ip with exclusive access
# for the ap_ctrl_none kernel of the contract (and the EX §9.1 fallback ap_ctrl_hs without an
# interrupt), device-only buffers, and refusal of xclbins that break the kernel contract (run
# arguments such as cmd_addr / cmd_count included).
if (TSR_WITH_XRT AND EXISTS ${_tsr_xrt_root}/lib/libxrt_noop.so)
    add_executable(test-tsr-xrt-noop tests/test_xrt_noop.cpp)
    set_target_properties(test-tsr-xrt-noop PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(test-tsr-xrt-noop PRIVATE -Wall -Wextra)
    target_link_libraries(test-tsr-xrt-noop PRIVATE tsr_transport tsr_xrt)
    add_test(NAME transport_xrt_noop
             COMMAND test-tsr-xrt-noop ${_tsr_xrt_root} ${CMAKE_CURRENT_BINARY_DIR}/xrt_noop)
endif()

# transport_ggml_e2e_mock and transport_ggml_e2e_mock_tmax8 (the ggml backend end to end on the
# mock kdev) return with the backend's OPL path (P3.2, SW §10.2; sw/ggml-tesseract).

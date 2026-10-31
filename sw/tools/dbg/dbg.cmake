# sw/tools/dbg/dbg.cmake — libtsr_dbg, tsr-dbg and tsr-prof (P3.7, P3.8; docs/design/05-debug.md "DB"
# §3-§9, docs/design/06-software-verification.md "SW" §11). Included by sw/tools/CMakeLists.txt.
#
#   tsr_dbg        static, PIC: the DMI driver, the RISC-V debug target, tile-level debug and probes,
#                  the monitor commands, the GDB RSP server, the trace writer and its timeline, the
#                  embedded service (tsr_dbg_lib.hpp). Only the transport API (sw/transport), so it
#                  runs on sim, mock, vkernel and xrt.
#   tsr_dbg_embed  OBJECT: linking it registers the embedded service (TSR_GDB / TSR_DEBUGGER) before
#                  main; for the owning processes (the ggml backend, tsr_rt, the torch extension).
#   tsr-dbg        the CLI and the standalone GDB server (tsr_dbg.cpp)
#   tsr-prof       the trw source (Perfetto / Chrome JSON timeline, DB §8.7 checks) and the DB §8.6
#                  PERF summary (tsr_prof.cpp)
#
# Tests (prefix tools_dbg_ / tools_prof_; ctest -R 'tools_(dbg|prof)'), all on the sim kind with the
# fw_rt test image (rt_nop, rt_copy, ...):
#   tools_dbg_lib    tests/test_dbg.cpp: the library end to end: halt-on-boot, registers, CSRs,
#                    memory against the ELF, the write policy (windows refused, SPM only stopped,
#                    TCM writes through LOAD_* while RV_RUN = 0), software / hardware breakpoints,
#                    step, the launch break, an RV watchpoint, a tile breakpoint that halts the hart
#                    with the stop classified and QSTEP, the history with push PCs in rt_copy, the
#                    trace writer timeline and its DB §8.7 checks, detach and a PING after it
#   tools_dbg_rsp    tests/rsp_client.py: tsr-dbg gdb on sim and a scripted RSP client (python
#                    stdlib): halt at boot, pc and x1-x31, CSRs, target.xml / memory map, a software
#                    breakpoint on a firmware symbol, continue, hit, memory read / write, step, a
#                    hardware breakpoint, monitor launch / tile break / hist / units / trw, Ctrl-C,
#                    detach
#   tools_dbg_rsp_drop, tools_dbg_embed_drop
#                    tests/rsp_client.py --drop [--embedded]: the client vanishes (RST) with a request
#                    outstanding; the server (or the owning process) must end the session with the
#                    detach and exit 0, never die of SIGPIPE (DB §4.1, §4.2)
#   tools_dbg_embed  tests/rsp_client.py --embedded: TSR_GDB=tcp:0 with TSR_GDB_WAIT=launch in an
#                    owning process (test-tsr-dbg embed-host): the first launch halts for GDB
#   tools_dbg_gdb    tests/gdb_smoke.sh: the same scenario through the toolchain's riscv-none-elf-gdb
#                    (skipped, 77, without it)
#   tools_prof_trw   tests/prof_trw.py: tsr-prof trw on sim (launches of rt_copy and rt_nop, --check,
#                    --stats), the JSON validated, --from-raw reproduces it

find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter)

add_library(tsr_dbg STATIC
    dbg/dmi.cpp dbg/target.cpp dbg/tile.cpp dbg/monitor.cpp dbg/rsp.cpp dbg/trw.cpp dbg/embed.cpp)
set_target_properties(tsr_dbg PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_dbg PRIVATE -Wall -Wextra -Wshadow)
target_include_directories(tsr_dbg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/dbg)
target_link_libraries(tsr_dbg PUBLIC tsr_transport PRIVATE tsr_elf Threads::Threads)

add_library(tsr_dbg_embed OBJECT dbg/embed_auto.cpp)
set_target_properties(tsr_dbg_embed PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tsr_dbg_embed PUBLIC tsr_dbg)

foreach(_t tsr-dbg tsr-prof)
    string(REPLACE "-" "_" _src ${_t})
    add_executable(${_t} ${_src}.cpp)
    set_target_properties(${_t} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_compile_options(${_t} PRIVATE -Wall -Wextra -Wshadow)
    target_link_libraries(${_t} PRIVATE tsr_dbg tsr_transport)
endforeach()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
get_filename_component(_tsr_fw_dir "${TSR_FW_ELF}" DIRECTORY)
set(_tsr_fw_rt ${_tsr_fw_dir}/fw_rt.elf)

add_executable(test-tsr-dbg dbg/tests/test_dbg.cpp)
set_target_properties(test-tsr-dbg PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(test-tsr-dbg PRIVATE -Wall -Wextra -Wshadow)
target_link_libraries(test-tsr-dbg PRIVATE tsr_dbg tsr_dbg_embed tsr_transport tsr_elf tsr_fw_paths Threads::Threads)
if (TARGET tsr_fw)
    add_dependencies(test-tsr-dbg tsr_fw)
    add_dependencies(tsr-dbg tsr_fw)
endif()

add_test(NAME tools_dbg_lib COMMAND test-tsr-dbg lib ${_tsr_fw_rt} ${CMAKE_CURRENT_BINARY_DIR}/dbg_lib)
set_tests_properties(tools_dbg_lib PROPERTIES TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")

if (Python3_EXECUTABLE)
    add_test(NAME tools_dbg_rsp
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dbg/tests/rsp_client.py
                     --server $<TARGET_FILE:tsr-dbg> --elf ${_tsr_fw_rt} --scratch ${CMAKE_CURRENT_BINARY_DIR}/dbg_rsp)
    set_tests_properties(tools_dbg_rsp PROPERTIES TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")
    add_test(NAME tools_dbg_embed
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dbg/tests/rsp_client.py --embedded
                     --server $<TARGET_FILE:test-tsr-dbg> --elf ${_tsr_fw_rt} --scratch ${CMAKE_CURRENT_BINARY_DIR}/dbg_embed)
    set_tests_properties(tools_dbg_embed PROPERTIES TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")
    add_test(NAME tools_dbg_rsp_drop
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dbg/tests/rsp_client.py --drop
                     --server $<TARGET_FILE:tsr-dbg> --elf ${_tsr_fw_rt} --scratch ${CMAKE_CURRENT_BINARY_DIR}/dbg_rsp_drop)
    set_tests_properties(tools_dbg_rsp_drop PROPERTIES TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")
    add_test(NAME tools_dbg_embed_drop
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dbg/tests/rsp_client.py --embedded --drop
                     --server $<TARGET_FILE:test-tsr-dbg> --elf ${_tsr_fw_rt} --scratch ${CMAKE_CURRENT_BINARY_DIR}/dbg_embed_drop)
    set_tests_properties(tools_dbg_embed_drop PROPERTIES TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")
    add_test(NAME tools_prof_trw
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dbg/tests/prof_trw.py
                     --prof $<TARGET_FILE:tsr-prof> --elf ${_tsr_fw_rt} --scratch ${CMAKE_CURRENT_BINARY_DIR}/prof_trw)
    set_tests_properties(tools_prof_trw PROPERTIES TIMEOUT 300 ENVIRONMENT "TSR_SIM_THREADS=2")
endif()

# The pinned toolchain's GDB (SW-7: xPack riscv-none-elf-gdb), else gdb-multiarch.
get_property(_tsr_pfx GLOBAL PROPERTY TSR_RISCV_PREFIX_EFFECTIVE)
if (_tsr_pfx AND EXISTS "${_tsr_pfx}gdb")
    set(TSR_RISCV_GDB "${_tsr_pfx}gdb")
else()
    find_program(TSR_RISCV_GDB NAMES riscv-none-elf-gdb gdb-multiarch
                 PATHS ${TESSERACT_ROOT}/build/tools/riscv/xpack-riscv-none-elf-gcc-14.2.0-3/bin)
endif()
if (TSR_BASH_TOOLS)
    add_test(NAME tools_dbg_gdb
             COMMAND ${TSR_BASH_TOOLS} ${CMAKE_CURRENT_SOURCE_DIR}/dbg/tests/gdb_smoke.sh "${TSR_RISCV_GDB}" $<TARGET_FILE:tsr-dbg>
                     ${_tsr_fw_rt} ${CMAKE_CURRENT_BINARY_DIR}/dbg_gdb)
    set_tests_properties(tools_dbg_gdb PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT "TSR_SIM_THREADS=2")
endif()

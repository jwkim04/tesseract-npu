# sw/tables — the single table source (tsr_tables.py) and its generator (EX-30).
#
# The generated files are checked in (rtl/core/tsr_tables_pkg.sv, sw/sim/include/tsr_tables.h,
# sw/ctrl/include/tsr/{mmio,vu,at}.hpp, tb/common/tables.py); nothing is generated at build
# time. After changing tsr_tables.py, run sw/tables/gen_tables.py and commit its outputs.
#   tables_check     gen_tables.py --check: every generated file is current (also in rtl/lint.sh)
#   tables_selftest  test_tables.py: the table's self-consistency checks and its expected values
# Both need only the Python standard library.
#   tsr_tables_hdrcheck  object library (built with every build, nothing linked): the firmware
#                        headers and tsr_tables.h compile together as C++20 and agree (hdrcheck.cpp)
find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_test(NAME tables_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_tables.py --check)
    add_test(NAME tables_selftest COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tables.py)
endif()

# Firmware SDK headers (sw/ctrl/include, generated): an interface target for their users, and the
# compile check. The RV32 SDK build compiles them with its own toolchain.
add_library(tsr_ctrl_headers INTERFACE)
target_include_directories(tsr_ctrl_headers INTERFACE ${TESSERACT_ROOT}/sw/ctrl/include)   # "tsr/mmio.hpp"
add_library(tsr_tables_hdrcheck OBJECT hdrcheck.cpp)
set_target_properties(tsr_tables_hdrcheck PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(tsr_tables_hdrcheck PRIVATE ${TESSERACT_ROOT}/sw/ctrl/include ${TESSERACT_ROOT}/sw/sim/include)
target_compile_options(tsr_tables_hdrcheck PRIVATE -Wall -Wextra -Wpedantic -ffreestanding)

# sw/ctrl/tests/examples.cmake — the SDK example kernel sw/ctrl/examples/vadd.cpp (shown in README.md).
# Included by sw/ctrl/host/CMakeLists.txt.
#
#   fw_example_vadd   boots the image fw_example_vadd on the simulator through the transport, launches
#                     vadd (n = 1024) and compares every word with tsr_fadd; under the fixed schedule and
#                     under TSR_SIM_SCHED=random:1 with TSR_SIM_RACE=fatal

tsr_fw_image(fw_example_vadd SOURCES ${TSR_CTRL_DIR}/examples/vadd.cpp)

add_executable(test_fw_example_vadd ${TSR_CTRL_DIR}/tests/test_fw_example_vadd.cpp)
set_target_properties(test_fw_example_vadd PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_fw_example_vadd PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
target_include_directories(test_fw_example_vadd PRIVATE ${TESSERACT_ROOT}/sw/sim)
target_link_libraries(test_fw_example_vadd PRIVATE tsr_transport tsr_fw_paths Threads::Threads)
add_dependencies(test_fw_example_vadd tsr_fw)
add_test(NAME fw_example_vadd COMMAND test_fw_example_vadd)
set_tests_properties(fw_example_vadd PROPERTIES TIMEOUT 60)

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

# ---------------------------------------------------------------------------------------------
# The Ising machine example (sw/ctrl/examples/ising/): the kernel `ising`, its host side and the reference. It solves
# the Sherrington-Kirkpatrick spin glass by discrete simulated bifurcation.
#
#   fw_example_ising       boots the image fw_example_ising on the simulator through the transport and checks the
#                          kernel against the bit-exact reference (snapshots, energies, final state; the fixed
#                          schedule and random ones with the race detector fatal), the exactness of the energy words,
#                          several column tensors, t = 8, a run cut into launches, the status codes of bad parameter
#                          blocks, a soft stop, the quality of the answers, and the schedule constants, the couplings
#                          and the sign step against the design
#   fw_example_ising_tool  the host tool tsr-ising (ising_host.cpp) on the simulator with --check and --dump, the exit
#                          status, a Ctrl-C stop, bad command lines, and the video script sk_video.py on the dump
#                          (ising_tool.cmake)
# ---------------------------------------------------------------------------------------------
set(TSR_ISING_DIR ${TSR_CTRL_DIR}/examples/ising)
tsr_fw_image(fw_example_ising SOURCES ${TSR_ISING_DIR}/ising.cpp)

add_library(tsr_ising_host STATIC ${TSR_ISING_DIR}/ising_problem.cpp ${TSR_ISING_DIR}/ising_ref.cpp
            ${TSR_ISING_DIR}/ising_run.cpp ${TSR_ISING_DIR}/ising_dump.cpp)
set_target_properties(tsr_ising_host PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(tsr_ising_host PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_ising_host PUBLIC ${TSR_ISING_DIR} ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/transport)
target_link_libraries(tsr_ising_host PUBLIC tsr_transport)

add_executable(test_fw_example_ising ${TSR_CTRL_DIR}/tests/test_fw_example_ising.cpp)
set_target_properties(test_fw_example_ising PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_fw_example_ising PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
target_link_libraries(test_fw_example_ising PRIVATE tsr_ising_host tsr_fw_paths Threads::Threads)
add_dependencies(test_fw_example_ising tsr_fw)
add_test(NAME fw_example_ising COMMAND test_fw_example_ising)
set_tests_properties(fw_example_ising PROPERTIES TIMEOUT 120)

add_executable(tsr-ising ${TSR_ISING_DIR}/ising_host.cpp)
set_target_properties(tsr-ising PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(tsr-ising PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_link_libraries(tsr-ising PRIVATE tsr_ising_host tsr_fw_paths Threads::Threads)
add_dependencies(tsr-ising tsr_fw)
add_test(NAME fw_example_ising_tool
         COMMAND ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:tsr-ising> -DVIDEO=${TSR_ISING_DIR}/sk_video.py
                 -DOUT=${CMAKE_CURRENT_BINARY_DIR}/scratch/ising_tool
                 -P ${TSR_CTRL_DIR}/tests/ising_tool.cmake)
set_tests_properties(fw_example_ising_tool PROPERTIES TIMEOUT 300)

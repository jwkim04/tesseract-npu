# examples/ising: the Ising machine, a custom kernel example that uses the Basis SDK (sw/ctrl). It is a project of
# its own and not part of the SDK. Added by the root CMakeLists.txt (option TSR_WITH_EXAMPLES) after sw/sim,
# sw/ctrl/host and sw/transport. Provides:
#   tsr_ising_fw       ExternalProject_Add that runs the firmware project sw/ctrl a second time, in
#                      <build>/examples/ising/fw, with its own images file: one image, example_ising,
#                      the production sources plus ising.cpp. This is how a kernel project outside the SDK
#                      builds its image.
#   tsr_ising_host     static library: the problem, the bit-exact model, the launches and the dump
#   tsr-ising          the tool (ising_host.cpp)
#   test_ising         the test program
# ctest: example_ising, example_ising_tool (label example).
# The host side takes the path of the image as the definition TSR_ISING_IMAGE; --image and $TSR_IMAGE override it.

set(TSR_ISING_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TSR_ISING_FW_DIR ${CMAKE_CURRENT_BINARY_DIR}/fw)
set(TSR_ISING_IMAGE_PATH ${TSR_ISING_FW_DIR}/example_ising.elf)

# The images file of the firmware project: one line, the image of the example.
set(_ising_images ${CMAKE_CURRENT_BINARY_DIR}/tsr_fw_images.cmake)
file(CONFIGURE OUTPUT ${_ising_images}
     CONTENT "tsr_fw_image(example_ising SOURCES ${TSR_ISING_DIR}/ising.cpp)\n" @ONLY)

get_property(_ising_riscv_prefix GLOBAL PROPERTY TSR_RISCV_PREFIX_EFFECTIVE)
if (NOT TSR_FW_BUILD_TYPE)
    set(TSR_FW_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
include(ExternalProject)
ExternalProject_Add(tsr_ising_fw
    SOURCE_DIR ${TESSERACT_ROOT}/sw/ctrl
    BINARY_DIR ${TSR_ISING_FW_DIR}
    CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${TESSERACT_ROOT}/sw/ctrl/cmake/riscv32.cmake
        -DTSR_RISCV_PREFIX=${_ising_riscv_prefix}
        -DCMAKE_BUILD_TYPE=${TSR_FW_BUILD_TYPE}
        -DTSR_ELF=$<TARGET_FILE:tsr-elf>
        -DTSR_FW_IMAGES_FILE=${_ising_images}
    CONFIGURE_HANDLED_BY_BUILD ON
    BUILD_ALWAYS ON
    INSTALL_COMMAND ""
    DEPENDS tsr-elf
    BUILD_BYPRODUCTS ${TSR_ISING_IMAGE_PATH})

add_library(tsr_ising_host STATIC ising_problem.cpp ising_ref.cpp ising_run.cpp ising_dump.cpp)
set_target_properties(tsr_ising_host PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(tsr_ising_host PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_ising_host PUBLIC ${TSR_ISING_DIR} ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/transport)
target_compile_definitions(tsr_ising_host PUBLIC TSR_ISING_IMAGE="${TSR_ISING_IMAGE_PATH}")
target_link_libraries(tsr_ising_host PUBLIC tsr_transport)

add_executable(tsr-ising ising_host.cpp)
set_target_properties(tsr-ising PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(tsr-ising PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_link_libraries(tsr-ising PRIVATE tsr_ising_host Threads::Threads)
add_dependencies(tsr-ising tsr_ising_fw)

add_executable(test_ising tests/test_ising.cpp)
set_target_properties(test_ising PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_ising PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
target_link_libraries(test_ising PRIVATE tsr_ising_host Threads::Threads)
add_dependencies(test_ising tsr_ising_fw)

# example_ising       boots the image on the simulator through the transport and checks the kernel against the
#                     bit-exact model (snapshots, energies, final state; the fixed schedule and random ones with the
#                     race detector fatal), the exactness of the energy words, several column tensors, t = 8, a run
#                     cut into launches, the status codes of bad parameter blocks, a soft stop, the quality of the
#                     answers, and the schedule constants, the couplings and the sign step against the design
# example_ising_tool  the tool tsr-ising on the simulator with --check and --dump, the exit status, a Ctrl-C stop,
#                     bad command lines, and the video script sk_video.py on the dump (tests/ising_tool.cmake)
add_test(NAME example_ising COMMAND test_ising)
set_tests_properties(example_ising PROPERTIES TIMEOUT 120 LABELS example)
add_test(NAME example_ising_tool
         COMMAND ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:tsr-ising> -DVIDEO=${TSR_ISING_DIR}/sk_video.py
                 -DOUT=${CMAKE_CURRENT_BINARY_DIR}/scratch/ising_tool
                 -P ${TSR_ISING_DIR}/tests/ising_tool.cmake)
set_tests_properties(example_ising_tool PROPERTIES TIMEOUT 300 LABELS example)

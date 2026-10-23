# sw/transport — tsr_transport: host <-> device transport (spec "ggml 백엔드 통합", 전송 계층).
#
#   tsr_transport   static, PIC (linked into libggml-tesseract.so): dispatcher + "sim" transport
#                   (the ISA simulator in-process); "xrt" only with -DTSR_WITH_XRT=ON (M2, stub now).

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

add_library(tsr_transport STATIC
    tsr_transport.c
    tsr_transport_sim.c)
set_target_properties(tsr_transport PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_transport PRIVATE ${TSR_FP_FLAGS}
    $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes>)
target_include_directories(tsr_transport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tsr_transport PUBLIC tsr_trace tsr_sim tsr_ref)

if (TSR_WITH_XRT)
    target_sources(tsr_transport PRIVATE tsr_transport_xrt.cpp)
    target_compile_definitions(tsr_transport PRIVATE TSR_HAVE_XRT=1)
    set_target_properties(tsr_transport PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endif()

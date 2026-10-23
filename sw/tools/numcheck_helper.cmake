# numcheck_helper — shared library for sw/tools/numcheck.py (docs/numcheck.md), loaded with ctypes:
# ggml to_float / quantize_row_q8_0_ref (bit-exact device-format weights) and the libtsr_ref
# B1 model loop. Links ggml-base, which the top-level CMakeLists adds after sw/tools, so the
# target is created by a call deferred to the end of the top-level directory (like klcheck).
#   cmake --build <build> --target numcheck_helper tsr_ref_shared -j4
function(tsr_add_numcheck_helper)
    if (NOT TARGET ggml-base OR NOT TARGET tsr_ref_shared)
        message(STATUS "numcheck_helper: ggml-base not available, not built")
        return()
    endif()
    add_library(numcheck_helper SHARED ${TESSERACT_ROOT}/sw/tools/numcheck_helper.c)
    set_target_properties(numcheck_helper PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        C_VISIBILITY_PRESET hidden
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/sw/tools)
    target_compile_options(numcheck_helper PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes)
    find_package(Threads REQUIRED)
    target_link_libraries(numcheck_helper PRIVATE tsr_ref_shared ggml-base Threads::Threads)
    # Model-free self test: QACT model vs libtsr_ref, the B2 bound on random/adversarial rows,
    # the comparer on a synthetic run set (python3 + numpy + gguf-py needed).
    find_program(TSR_PYTHON3 python3)
    if (TSR_PYTHON3)
        add_test(NAME tools_numcheck_selftest
                 COMMAND ${TSR_PYTHON3} ${TESSERACT_ROOT}/sw/tools/numcheck.py --build ${CMAKE_BINARY_DIR}
                         selftest -o ${CMAKE_BINARY_DIR}/sw/tools/numcheck_selftest)
        set_tests_properties(tools_numcheck_selftest PROPERTIES TIMEOUT 1800)
    endif()
endfunction()
cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_numcheck_helper)

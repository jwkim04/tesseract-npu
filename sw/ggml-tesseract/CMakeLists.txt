# sw/ggml-tesseract — tests of the ggml-tesseract backend.
#
# The backend library itself (target ggml-tesseract) is built inside llama.cpp's ggml tree:
# third_party/patches/0001-ggml-tesseract.patch adds ggml/src/ggml-tesseract/CMakeLists.txt,
# which includes ./sources.cmake. This directory is added by the top-level CMakeLists after
# third_party/llama.cpp, so the ggml targets exist here.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

# Allocator + SPM tiling (the tsr_lower library, lower.cmake), no ggml needed.
add_executable(test-tsr-backend-units tests/test-tsr-backend-units.cpp)
target_link_libraries(test-tsr-backend-units PRIVATE tsr_lower tsr_ref)
target_compile_features(test-tsr-backend-units PRIVATE cxx_std_17)
target_compile_options(test-tsr-backend-units PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
set_target_properties(test-tsr-backend-units PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME ggml_tesseract_units COMMAND test-tsr-backend-units)

if (TARGET ggml AND TARGET ggml-tesseract)
    # X0 (docs/v2/06-software-verification.md §10.2): the backend executes no op on the device until
    # the op-list path (P3.2), so the ctests that run MUL_MATs on it — ggml_tesseract_e2e,
    # _e2e_tmax8, _e2e_hosted, _e2e_hosted_tmax8, ggml_tesseract_scalar_nmse and
    # ggml_tesseract_backend_ops — are removed and return with P3.2, with the same assertions. The
    # executables are still built so that they keep compiling against the backend's public API.

    # End-to-end through the ggml API on the transport selected by TSR_DEVICE (default sim).
    add_executable(test-ggml-tesseract tests/test-ggml-tesseract.cpp)
    target_include_directories(test-ggml-tesseract PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)          # ggml-quants.h (quantize_row_q8_0_ref)
    target_link_libraries(test-ggml-tesseract PRIVATE ggml ggml-base tsr_ref)
    target_compile_features(test-ggml-tesseract PRIVATE cxx_std_17)
    target_compile_options(test-ggml-tesseract PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-ggml-tesseract PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # The single-output MUL_MAT case of test-backend-ops (f32 x f32, m=1 n=1 k=2048), checked
    # bit-exactly against the spec model; see the header of the source for why.
    add_executable(test-tsr-scalar-nmse tests/test-tsr-scalar-nmse.cpp)
    target_include_directories(test-tsr-scalar-nmse PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)          # ggml-quants.h
    target_link_libraries(test-tsr-scalar-nmse PRIVATE ggml ggml-base tsr_ref)
    target_compile_features(test-tsr-scalar-nmse PRIVATE cxx_std_17)
    target_compile_options(test-tsr-scalar-nmse PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-tsr-scalar-nmse PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# llama.cpp's own tests that load models with default offload settings (no --no-kv-offload):
# with the Tesseract device registered (GPU type) llama.cpp puts the KV cache in a Tesseract
# buffer, and v1 runs no KV ops (SET_ROWS, attention), so the scheduler aborts — the documented
# v1 constraint (spec "v1 lowering 제약", docs/llama-pin.md: run llama.cpp with -nkvo). These
# tests check llama.cpp itself, so they run with TSR_DEVICE=none (no Tesseract device).
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    foreach (t test-llama-context test-llama-context-model test-thread-safety)
        get_property(_tsr_has_test DIRECTORY ${TESSERACT_ROOT}/third_party/llama.cpp/tests PROPERTY TESTS)
        if (t IN_LIST _tsr_has_test)
            set_property(TEST ${t} DIRECTORY ${TESSERACT_ROOT}/third_party/llama.cpp/tests
                         APPEND PROPERTY ENVIRONMENT "TSR_DEVICE=none")
        endif()
    endforeach()
endif()

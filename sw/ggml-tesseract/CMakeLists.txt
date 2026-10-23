# sw/ggml-tesseract — tests of the ggml-tesseract backend.
#
# The backend library itself (target ggml-tesseract) is built inside llama.cpp's ggml tree:
# third_party/patches/0001-ggml-tesseract.patch adds ggml/src/ggml-tesseract/CMakeLists.txt,
# which includes ./sources.cmake. This directory is added by the top-level CMakeLists after
# third_party/llama.cpp, so the ggml targets exist here.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

# Allocator + lowering, no ggml needed.
add_executable(test-tsr-backend-units
    tests/test-tsr-backend-units.cpp
    tsr-alloc.cpp
    tsr-lower.cpp)
target_include_directories(test-tsr-backend-units PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test-tsr-backend-units PRIVATE tsr_ref)
target_compile_features(test-tsr-backend-units PRIVATE cxx_std_17)
target_compile_options(test-tsr-backend-units PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
set_target_properties(test-tsr-backend-units PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME ggml_tesseract_units COMMAND test-tsr-backend-units)

if (TARGET ggml AND TARGET ggml-tesseract)
    # End-to-end through the ggml API on the transport selected by TSR_DEVICE (default sim).
    add_executable(test-ggml-tesseract tests/test-ggml-tesseract.cpp)
    target_include_directories(test-ggml-tesseract PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)          # ggml-quants.h (quantize_row_q8_0_ref)
    target_link_libraries(test-ggml-tesseract PRIVATE ggml ggml-base tsr_ref)
    target_compile_features(test-ggml-tesseract PRIVATE cxx_std_17)
    target_compile_options(test-ggml-tesseract PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-ggml-tesseract PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    add_test(NAME ggml_tesseract_e2e       COMMAND test-ggml-tesseract)
    add_test(NAME ggml_tesseract_e2e_tmax8 COMMAND test-ggml-tesseract)
    set_tests_properties(ggml_tesseract_e2e_tmax8 PROPERTIES ENVIRONMENT "TSR_SIM_TMAX=8")
    # Every activation buffer in host memory and a 64 KiB IO stage: every MUL_MAT takes the
    # staged path (src1 host -> stage, dst stage -> host) and the larger ones run in several
    # T tiles; the same bit-exact checks must hold.
    add_test(NAME ggml_tesseract_e2e_hosted       COMMAND test-ggml-tesseract)
    set_tests_properties(ggml_tesseract_e2e_hosted PROPERTIES ENVIRONMENT "TSR_FORCE_HOST=1;TSR_STAGE_KIB=64")
    add_test(NAME ggml_tesseract_e2e_hosted_tmax8 COMMAND test-ggml-tesseract)
    set_tests_properties(ggml_tesseract_e2e_hosted_tmax8 PROPERTIES ENVIRONMENT "TSR_FORCE_HOST=1;TSR_STAGE_KIB=64;TSR_SIM_TMAX=8")

    # The single-output MUL_MAT case of test-backend-ops (f32 x f32, m=1 n=1 k=2048), checked
    # bit-exactly against the spec model; see the header of the source for why.
    add_executable(test-tsr-scalar-nmse tests/test-tsr-scalar-nmse.cpp)
    target_include_directories(test-tsr-scalar-nmse PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)          # ggml-quants.h
    target_link_libraries(test-tsr-scalar-nmse PRIVATE ggml ggml-base tsr_ref)
    target_compile_features(test-tsr-scalar-nmse PRIVATE cxx_std_17)
    target_compile_options(test-tsr-scalar-nmse PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-tsr-scalar-nmse PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ggml_tesseract_scalar_nmse COMMAND test-tsr-scalar-nmse)

    # test-backend-ops: Tesseract0 vs the CPU backend (spec 검증 전략, 연산 단위), every MUL_MAT
    # case, tolerances untouched. One case is left out here and covered by
    # ggml_tesseract_scalar_nmse instead: MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048) has a
    # single output element, so its NMSE is the squared relative error of one dot product of
    # fresh random data; with the spec's 8-bit weights and activations it exceeds 5e-4 in ~15%
    # of runs on a correct device (measured: 299/2000 trials, the same count as an exact dot
    # product of the 8-bit operands). The full, unfiltered command is in README.md.
    if (TARGET test-backend-ops)
        add_test(NAME ggml_tesseract_backend_ops
                 COMMAND test-backend-ops test -o MUL_MAT -b Tesseract0
                         -p "^(?!type_a=f32,type_b=f32,m=1,n=1,k=2048,)")
    endif()
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

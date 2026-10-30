# sw/ggml-tesseract — tests of the ggml-tesseract backend.
#
# The backend library itself (target ggml-tesseract) is built inside llama.cpp's ggml tree:
# third_party/patches/0001-ggml-tesseract.patch adds ggml/src/ggml-tesseract/CMakeLists.txt,
# which includes ./sources.cmake. This directory is added by the top-level CMakeLists after
# third_party/llama.cpp, so the ggml targets exist here.

if (NOT DEFINED TSR_FP_FLAGS)
    set(TSR_FP_FLAGS -ffp-contract=off -fno-fast-math)
endif()

# Allocator, SPM tiling, OPL builder and cache (the tsr_lower library, lower.cmake), no ggml and no
# device needed (docs/v2/06-software-verification.md §10.2: ggml_tesseract_units rewritten for the
# OPL builder).
add_executable(test-tsr-backend-units tests/test-tsr-backend-units.cpp)
target_link_libraries(test-tsr-backend-units PRIVATE tsr_lower tsr_ref)
target_compile_features(test-tsr-backend-units PRIVATE cxx_std_17)
target_compile_options(test-tsr-backend-units PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
set_target_properties(test-tsr-backend-units PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME ggml_tesseract_units COMMAND test-tsr-backend-units)

# The launcher on a device (TSR_DEVICE, default sim; the build's firmware image): a list with a JOIN
# and an ACC_DST record against the spec model, an OPL cache hit, a TILE_ERR (DRAM_RANGE in op 1)
# and a KERNEL_ERR (BAD_RECORD) answer mapped to their op and named, then a valid list again.
add_executable(test-tsr-opl-launch tests/test-tsr-opl-launch.cpp)
target_link_libraries(test-tsr-opl-launch PRIVATE tsr_lower tsr_ref)
target_compile_features(test-tsr-opl-launch PRIVATE cxx_std_17)
target_compile_options(test-tsr-opl-launch PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
set_target_properties(test-tsr-opl-launch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME ggml_tesseract_opl_launch COMMAND test-tsr-opl-launch)
add_test(NAME ggml_tesseract_opl_launch_mock COMMAND test-tsr-opl-launch)
set_tests_properties(ggml_tesseract_opl_launch PROPERTIES ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal")
set_tests_properties(ggml_tesseract_opl_launch_mock PROPERTIES ENVIRONMENT
    "TSR_DEVICE=mock;TSR_SIM_RACE=fatal;TSR_MOCK_DMA_ALIGN=64;TSR_MOCK_DMA_CHUNK_KIB=64")

if (TARGET ggml AND TARGET ggml-tesseract)
    # The ctests below are v1's with the same assertions, run on v2 (docs/v2/06-software-verification.md
    # §10.2 "Kept, run on v2"): every MUL_MAT goes through an op list and the firmware's `graph` kernel
    # (TSR_IMAGE: the build's image), on the sim transport unless TSR_DEVICE says otherwise, with the
    # simulator's race detector fatal (SW §4.6: ctest uses fatal).

    # End-to-end through the ggml API on the transport selected by TSR_DEVICE (default sim).
    add_executable(test-ggml-tesseract tests/test-ggml-tesseract.cpp)
    target_include_directories(test-ggml-tesseract PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)          # ggml-quants.h (quantize_row_q8_0_ref)
    target_link_libraries(test-ggml-tesseract PRIVATE ggml ggml-base tsr_ref)
    target_compile_features(test-ggml-tesseract PRIVATE cxx_std_17)
    target_compile_options(test-ggml-tesseract PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-ggml-tesseract PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    add_test(NAME ggml_tesseract_e2e       COMMAND test-ggml-tesseract)
    set_tests_properties(ggml_tesseract_e2e PROPERTIES ENVIRONMENT "TSR_SIM_RACE=fatal")
    add_test(NAME ggml_tesseract_e2e_tmax8 COMMAND test-ggml-tesseract)
    set_tests_properties(ggml_tesseract_e2e_tmax8 PROPERTIES ENVIRONMENT "TSR_SIM_TMAX=8;TSR_SIM_RACE=fatal")
    # Every activation buffer in host memory and a 64 KiB IO stage: every MUL_MAT takes the
    # staged path (src1 host -> stage, dst stage -> host) and the larger ones run in several
    # T tiles (one launch each); the same bit-exact checks must hold.
    add_test(NAME ggml_tesseract_e2e_hosted       COMMAND test-ggml-tesseract)
    set_tests_properties(ggml_tesseract_e2e_hosted PROPERTIES ENVIRONMENT "TSR_FORCE_HOST=1;TSR_STAGE_KIB=64;TSR_SIM_RACE=fatal")
    add_test(NAME ggml_tesseract_e2e_hosted_tmax8 COMMAND test-ggml-tesseract)
    set_tests_properties(ggml_tesseract_e2e_hosted_tmax8 PROPERTIES ENVIRONMENT
        "TSR_FORCE_HOST=1;TSR_STAGE_KIB=64;TSR_SIM_TMAX=8;TSR_SIM_RACE=fatal")
    # The same on the mock kdev (the protocol layer with the ap_ctrl_none adapter model and DMA rules
    # that force bounce / read-modify-write transfers). v1 defined these two in sw/transport; they
    # moved here with the backend's op-list path (P3.2), as they test the backend end to end.
    add_test(NAME transport_ggml_e2e_mock COMMAND test-ggml-tesseract)
    set_tests_properties(transport_ggml_e2e_mock PROPERTIES
        ENVIRONMENT "TSR_DEVICE=mock;TSR_MOCK_DMA_ALIGN=64;TSR_MOCK_DMA_CHUNK_KIB=64;TSR_SIM_RACE=fatal")
    add_test(NAME transport_ggml_e2e_mock_tmax8 COMMAND test-ggml-tesseract)
    set_tests_properties(transport_ggml_e2e_mock_tmax8 PROPERTIES ENVIRONMENT
        "TSR_DEVICE=mock;TSR_MOCK_TMAX=8;TSR_MOCK_NUM_CH=3;TSR_MOCK_DMA_ALIGN=4096;TSR_MOCK_DMA_CHUNK_KIB=12;TSR_SIM_RACE=fatal")

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
    set_tests_properties(ggml_tesseract_scalar_nmse PROPERTIES ENVIRONMENT "TSR_SIM_RACE=fatal")

    # test-backend-ops: Tesseract0 vs the CPU backend (spec 검증 전략, 연산 단위), every MUL_MAT
    # case, tolerances untouched. One case is left out here and covered by
    # ggml_tesseract_scalar_nmse instead: MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048) has a
    # single output element, so its NMSE is the squared relative error of one dot product of
    # fresh random data; with the spec's 8-bit weights and activations it exceeds 5e-4 in ~15%
    # of runs on a correct device (measured: 299/2000 trials, the same count as an exact dot
    # product of the 8-bit operands). The full, unfiltered command is in README.md.
    # v2 (P3.2): the case MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32) (2.1 M tokens, staged in
    # 16 launches of 131,072 single-token tiles at the sim's T_MAX 1: 8.4 M tile ops through the
    # interpreted firmware, ~50 s, and over 15 min with the race detector) runs in its own ctest
    # labelled slow, without the race detector; the two ctests together run v1's case set.
    if (TARGET test-backend-ops)
        add_test(NAME ggml_tesseract_backend_ops
                 COMMAND test-backend-ops test -o MUL_MAT -b Tesseract0
                         -p "^(?!type_a=f32,type_b=f32,m=1,n=1,k=2048,|type_a=f16,type_b=f32,m=16,n=2105344,k=32,)")
        set_tests_properties(ggml_tesseract_backend_ops PROPERTIES ENVIRONMENT "TSR_SIM_RACE=fatal")
        add_test(NAME ggml_tesseract_backend_ops_bigt
                 COMMAND test-backend-ops test -o MUL_MAT -b Tesseract0 -p "^type_a=f16,type_b=f32,m=16,n=2105344,k=32,")
        set_tests_properties(ggml_tesseract_backend_ops_bigt PROPERTIES LABELS slow ENVIRONMENT "TSR_SIM_RACE=off")
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
    # test-tokenizers-ggml-vocabs (tests/test-tokenizers-repo.sh) runs `git clone` / `git pull` of
    # https://huggingface.co/ggml-org/vocabs into third_party/llama.cpp/models/ggml-vocabs: network
    # access and writes inside the pinned llama.cpp tree from `ctest`, and a failure whenever the
    # remote changes or is unreachable. It tests llama.cpp's tokenizers only: disabled here.
    get_property(_tsr_has_test DIRECTORY ${TESSERACT_ROOT}/third_party/llama.cpp/tests PROPERTY TESTS)
    if (test-tokenizers-ggml-vocabs IN_LIST _tsr_has_test)
        set_property(TEST test-tokenizers-ggml-vocabs DIRECTORY ${TESSERACT_ROOT}/third_party/llama.cpp/tests
                     PROPERTY DISABLED ON)
    endif()
endif()

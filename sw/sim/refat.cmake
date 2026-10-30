# sw/sim/refat.cmake — the AT references (sw/sim/tsr_ref_at.c, docs/design/03-attention-kv.md §13.1:
# KV plane layout and conversion, KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM, the MSUM and the
# composite FA of AT §6.6) in the reference library, and their tests (AT §13.2,
# docs/design/06-software-verification.md §10.1-§10.2).
#
# sw/sim/CMakeLists.txt includes it (snippet loop "refvu refat", after ref_math.cmake: tsr_ref_at.c
# calls tsr_m_exp; -DTSR_SIM_SKIP_SNIPPETS=refat leaves it out). Before the tsr_ref /
# tsr_ref_shared add_library() calls it extends TSR_REF_SOURCES; after them it adds the source to
# both targets. Either way TSR_REF_SOURCES carries tsr_ref_at.c for any later target built from the
# list (e.g. sanitizer builds).
#
# Tests (run: ctest --test-dir build/dev -R '^ref_at' --output-on-failure):
#   ref_at       self tests: layout round trips and placement (model shapes, random E / kv_size /
#                C), KVST bytes and RMW counts (and, with ggml, against quantize_row_q8_0_ref),
#                ATTQK SEG vs tsr_dot_row and the short tree, bounds, GQA rows and MAXOUT,
#                ATTPV against independent 128-bit / exact-rational models and directed cases,
#                the vector ops, MSUM, the chunk planner, and tsr_ref_fa against an independent
#                double-precision model
#   ref_at_ggml  tsr_ref_fa against ggml-cpu's FLASH_ATTN_EXT (q8_0 K/V views of a [E, kv_size]
#                cache, F16 mask, GQA) for decode and prefill, causal, sliding-window and wrapped
#                masks, D = 64..512, n_kv up to 16K, C 1..16: NMSE, max abs / rel error,
#                softmax-KL of the output rows, the x86 Q-quantization ties, and the error of both
#                against the double model (TSR_WITH_LLAMA only), including finite masks with
#                scale 1 and scale != 1 (the f32 mask copy divided by scale, AT §5.5).
# Both run in a few seconds; `ref_at self -v` / `ref_at ggml -v` print more detail.

set(TSR_REF_AT_SOURCE ${CMAKE_CURRENT_LIST_DIR}/tsr_ref_at.c)
list(APPEND TSR_REF_SOURCES ${TSR_REF_AT_SOURCE})
foreach(_tgt tsr_ref tsr_ref_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_REF_AT_SOURCE})
    endif()
endforeach()

set(_ra_tests ${CMAKE_CURRENT_LIST_DIR}/tests)

add_executable(ref_at ${_ra_tests}/test_ref_at.c)
tsr_c_target(ref_at)
target_include_directories(ref_at PRIVATE ${_ra_tests})
target_link_libraries(ref_at PRIVATE tsr_ref m)
if (TSR_WITH_LLAMA)
    # ggml-base (quantize_row_q8_0_ref) and ggml-cpu (FLASH_ATTN_EXT) are defined later in the
    # top-level project (third_party/llama.cpp); target names resolve at generate time.
    target_compile_definitions(ref_at PRIVATE TSR_REF_AT_GGML=1)
    target_include_directories(ref_at SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include
                                                     ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)
    target_link_libraries(ref_at PRIVATE ggml-cpu ggml-base)
endif()

add_test(NAME ref_at COMMAND ref_at self)
set_tests_properties(ref_at PROPERTIES TIMEOUT 120)
if (TSR_WITH_LLAMA)
    add_test(NAME ref_at_ggml COMMAND ref_at ggml)
    set_tests_properties(ref_at_ggml PROPERTIES TIMEOUT 120 PROCESSORS 4)
endif()

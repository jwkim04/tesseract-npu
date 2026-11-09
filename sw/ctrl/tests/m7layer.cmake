# sw/ctrl/tests/m7layer.cmake — one whole decoder layer per OPL list on the RISC-V
# image (docs/design/06-software-verification.md "SW" §7.2, §7.5, §7.8, §10.2). Included by sw/ctrl/host/CMakeLists.txt after elem / kvfa / normrope (it links their
# references). Every test has label fw; long ones add slow.
#
#   fw_m7_layer_qwen05b    Qwen2.5-0.5B layer 0 (real Q8_0 weights from models/qwen2.5-0.5b-instruct-
#                          q8_0.gguf when the file is there, else random weights in the same shapes),
#                          decode (T = 1) and a prefill tile (T = 8): RMS_NORM+weight, Q / K / V MUL_MATs
#                          with bias addends, ROPE on Q and K, SET_ROWS_KV K and V, FLASH_ATTN (causal
#                          MSUM), MUL_MAT wo + residual, RMS_NORM, gate / up, GLU SWIGLU, down + residual;
#                          fixed + 7 random schedules, race fatal: every record's output bit-exact against
#                          the host composition of the per-kind references in op order, the SW §4.7
#                          observables equal, no write outside the outputs and the KV planes
#   fw_m7_layer_gemma12b   the same for one Gemma 4 12B SWA layer shape with random weights (about 15 s)
#                          (per-head q / k norms with weight and v norm without, GEGLU16, post-attention
#                          and post-FFN norms, ADD residuals, MUL by the layer's out_scale)
#   fw_m7_layer_k4min      the device of the reduced card image tsr_u200_k4 (4 channels, T_MAX 8, CAPS 0): the
#                          layer's seven MUL_MAT records only (M6), decode and a T = 8 prefill tile, fixed + 3 random
#                          schedules against mulmat_ref (docs/platform-u200.md §8.7)
#   fw_m7_layer_ggml       (TSR_WITH_LLAMA) the device layers against ggml-cpu computing the same layer
#                          (llama.cpp's qwen2 / gemma4 graph on the same weights, inputs and KV history):
#                          NMSE and row-softmax KL of the layer output
#   fw_m7_layer_perf       measurements (SW §7.8): RISC-V cycles per record per kind inside a 0.5B decode
#                          layer, RISC-V time vs unit time, MVM / VU issue-idle, push -> DONE, at 166.7
#                          and 250 MHz, against a near-ideal issuer

set(_ml_dir ${TSR_CTRL_DIR}/tests)
add_executable(test_fw_m7_layer ${_ml_dir}/test_fw_m7_layer.cpp ${TSR_CTRL_DIR}/ref/rmsnorm_ref.cpp
               ${TSR_CTRL_DIR}/ref/rope_ref.cpp)
set_target_properties(test_fw_m7_layer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_fw_m7_layer PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wno-unused-function)
target_include_directories(test_fw_m7_layer PRIVATE ${TSR_CTRL_INCLUDE_DIR} ${TSR_CTRL_DIR}/ref
                           ${TESSERACT_ROOT}/sw/sim ${TESSERACT_ROOT}/sw/sim/include ${TSR_FW_SIM_TESTS_DIR})
target_link_libraries(test_fw_m7_layer PRIVATE tsr_fw_ref tsr_fw_elem_ref tsr_fw_kvfa_ref tsr_sim tsr_ref
                      Threads::Threads m)
target_compile_definitions(test_fw_m7_layer PRIVATE TSR_M7L_MODELS="${TESSERACT_ROOT}/models")
if (TSR_WITH_LLAMA)
    target_compile_definitions(test_fw_m7_layer PRIVATE TSR_M7L_GGML=1)
    target_include_directories(test_fw_m7_layer SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    target_link_libraries(test_fw_m7_layer PRIVATE ggml-cpu ggml-base)
endif()
add_dependencies(test_fw_m7_layer tsr_fw)

add_test(NAME fw_m7_layer_qwen05b COMMAND test_fw_m7_layer ${TSR_FW_ELF} suite qwen05b)
set_tests_properties(fw_m7_layer_qwen05b PROPERTIES LABELS fw TIMEOUT 900 SKIP_RETURN_CODE 77 PROCESSORS 2)
add_test(NAME fw_m7_layer_gemma12b COMMAND test_fw_m7_layer ${TSR_FW_ELF} suite gemma12b)
set_tests_properties(fw_m7_layer_gemma12b PROPERTIES LABELS fw TIMEOUT 900 SKIP_RETURN_CODE 77 PROCESSORS 2)
add_test(NAME fw_m7_layer_k4min COMMAND test_fw_m7_layer ${TSR_FW_ELF} suite k4min)
set_tests_properties(fw_m7_layer_k4min PROPERTIES LABELS fw TIMEOUT 900 SKIP_RETURN_CODE 77 PROCESSORS 2)
if (TSR_WITH_LLAMA)
    add_test(NAME fw_m7_layer_ggml COMMAND test_fw_m7_layer ${TSR_FW_ELF} ggml)
    set_tests_properties(fw_m7_layer_ggml PROPERTIES LABELS fw TIMEOUT 1800 SKIP_RETURN_CODE 77 PROCESSORS 4)
endif()
add_test(NAME fw_m7_layer_perf COMMAND test_fw_m7_layer ${TSR_FW_ELF} perf)
set_tests_properties(fw_m7_layer_perf PROPERTIES LABELS fw TIMEOUT 900 SKIP_RETURN_CODE 77)

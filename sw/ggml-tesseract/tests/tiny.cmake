# sw/ggml-tesseract/tests/tiny.cmake — M7 on the simulator (docs/design/06-software-verification.md
# §10.2, §10.4): the tiny-model graph tests and the test-backend-ops coverage of the M7 kinds.
# Included from ../CMakeLists.txt (after the llama.cpp targets exist).
#
# Tiny models (SW-21): gen_tiny_models.py (gguf-py from the pinned llama.cpp + numpy) writes, at
# build time, into ${TSR_TINY_MODEL_DIR}:
#   qwen2-tiny-q8_0.gguf, qwen2-tiny-q4_k.gguf (2 layers), gemma4-tiny-q8_0.gguf (4 layers, 3 SWA +
#   1 full), gemma4-tiny-moe-q8_0.gguf (the same layers with a dense FFN beside 16 experts), gemma4-tiny-shared-q8_0.gguf (4 layers whose last two share the KV of the first two, as Gemma 4
#   E4B does), qwen35-tiny-q8_0.gguf and qwen35-tiny-s128-q8_0.gguf (Qwen3.5 / 3.8 layout: 3 Gated DeltaNet
#   layers and 1 full attention layer; S_v 32 and 128), qwen35-tiny-mtp-q8_0.gguf and qwen35-tiny-s128-mtp-q8_0.gguf (the
#   same with the multi-token prediction block, SW-83), qwen3moe-tiny-q8_0.gguf and qwen3moe-tiny-q4_k.gguf
#   (Qwen3-30B-A3B's graph: 2 layers, 64 experts, 8 used), tiny-prompts.txt — random weights, 64-row
#   vocabularies, no binary in git.
#
# Graph tests (m7_tests.py graph): klcheck generates with the M7 settings (--phase m7) on the device;
# the log must show exactly one Tesseract split in every single-token decode graph the scheduler
# splits (GGML_SCHED_DEBUG=2), exactly one launch per single-token decode (64 of them: 4 prompts x 16)
# and one launch per ubatch in prefill (-ub 16: prompts of 48-60 and 251-343 tokens, 3-22 ubatches;
# the long ones cross the 256-cell KV bucket in decode and in prefill and wrap Gemma 4's 256-cell SWA
# ring, gen_tiny_models.py PROMPTS); then the KL against the CPU backend with klcheck's criterion,
# mean KL <= max(--min-kl, 3 x floor) and top-1 >= --min-top1 (docs/klcheck.md "Pass"). The floor is
# the mean over the 1-ulp floor runs of the seeds 1 to 3 (m7_tests.py --floor-seeds, as klcheck all),
# and the device run takes no TSR_KINDS from the environment (the tests check the full kind set).
#
# The bounds (measurements, these prompts, 96 positions per model: 8 prompt-tail rows + 16
# generated rows of 4 prompts; deterministic, the same every run). The M6 ratios and the first M7
# ratio are against the floor of seed 1 alone; the second
# M7 ratio is against the mean floor of the seeds 1 to 3, which the tests use:
#   model        M6 mean KL (ratio, seed 1)  M7 mean KL (ratio seed 1 / mean)  floor seed 1 / mean  top-1
#   qwen2 q8_0   1.42e-4 (0.65)              1.03e-4 (0.47 / 0.60)             2.20e-4 / 1.71e-4    100 %
#   qwen2 q4_k   1.17e-4 (0.79)              1.47e-4 (0.99 / 1.47)             1.48e-4 / 1.00e-4    100 %
#   gemma4 q8_0  6.05e-4 (4.2)               6.89e-4 (4.8 / 1.69)              1.44e-4 / 4.07e-4    100 %
# The Qwen2 tests use klcheck's criterion as it is (3 x floor). Gemma 4 adds --min-kl 3e-3
# (openTPU's absolute MAX_KL form, docs/klcheck.md): the 1-ulp floor of a 4-layer random model
# underestimates its sensitivity, which M6 (the MUL_MATs on the device, bit-exact with the spec model
# by numcheck B1; attention on the CPU) already shows at ratio 4.2: the floor's nudge is absorbed by
# the first q8_0 activation quantization in most positions (24 of 96 floor rows bit-identical), the
# device's f32-level differences after every MUL_MAT flip q8_0 roundings of the KV cache and later
# activations. Injected errors, the CPU on the device's tokens with an edited GGUF: Gemma 4 sliding
# window 16 -> 15: mean KL 0.41, top-1 90.6 %; freq_base_swa x 1.001: 6.4e-3; RMS eps 1e-6 -> 1e-3:
# 8.1e-3; Qwen2 freq_base x 1.001: 4.2e-4; eps 1e-6 -> 1e-3: 4.4e-4. --min-top1 0.95 (the spec's
# 0.8 would pass the sliding-window error). Against the mean of three floors the Gemma 4 M7 ratio is
# 1.7 (0.79 to 4.8 per seed); --min-kl stays.
# The Qwen3 MoE models (SW-77; 96 positions, mean of three floors) have the same shape of result: the 1-ulp floor of a
# 2-layer random model with a router is small, the device's mean KL is 3.98e-4 (Q8_0, ratio 11.5, floor 3.5e-5) and
# 5.84e-4 (Q4_K, ratio 4.7, floor 1.2e-4), top-1 100 %. M6 on the Q8_0 model (the CPU selects the experts, the device
# runs the matrix products) gives 4.85e-4, so the routing on the device is not the larger part. --min-kl 1e-3
# (openTPU's MAX_KL form, docs/klcheck.md) is 2.5 times the Q8_0 value; a router or expert error of the size of a wrong
# expert gives mean KL above 1e-1.
#
# test-backend-ops (m7_tests.py ops): one ctest per op filter, Tesseract0 vs CPU on the sim with the
# race detector fatal: every supported case must pass, at least MIN_RUN cases must be supported, the
# unsupported cases are counted and listed (all of them in the build tree's
# sw/ggml-tesseract/backend_ops/<op>.unsupported.txt). FLASH_ATTN_EXT and SET_ROWS are supported only
# on KV-cache tensors (SW §9.3 "KV buffers"; test-backend-ops' tensors are not), so MIN_RUN is 0 there
# and ggml_tesseract_attn covers them; the M7 gate's "test-backend-ops passes every supported op"
# (SW §10.4 item 4) is these ctests plus ggml_tesseract_backend_ops (MUL_MAT).

if (NOT (TARGET ggml AND TARGET ggml-tesseract))
    return()
endif()

# where the build writes the models (also read by other tests: ggml_tesseract_kvstate)
set(TSR_TINY_MODEL_DIR ${CMAKE_BINARY_DIR}/models/tiny)

# Python with numpy that imports gguf-py (the pinned llama.cpp's). Candidates: TSR_TINY_PYTHON, the
# project venv, TSR_REF_MATH_PYTHON, the found python3 (the dev host's first python3 is an unrelated
# venv, which works too as long as it has numpy).
set(TSR_TINY_PYTHON "" CACHE FILEPATH "Python with numpy for gen_tiny_models.py and m7_tests.py")
set(_tiny_gguf_py ${TESSERACT_ROOT}/third_party/llama.cpp/gguf-py)
set(_tiny_candidates "${TSR_TINY_PYTHON}")
if (NOT TSR_TINY_PYTHON)
    set(_tiny_candidates "")
    if (EXISTS ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
        list(APPEND _tiny_candidates ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
    endif()
    if (TSR_REF_MATH_PYTHON)
        list(APPEND _tiny_candidates ${TSR_REF_MATH_PYTHON})
    endif()
    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_EXECUTABLE)
        list(APPEND _tiny_candidates ${Python3_EXECUTABLE})
    endif()
endif()
set(_tiny_py "")
foreach (_cand ${_tiny_candidates})
    if (NOT _tiny_py)
        execute_process(COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${_tiny_gguf_py} "${_cand}" -c "import numpy, gguf"
                        RESULT_VARIABLE _tiny_rc OUTPUT_QUIET ERROR_QUIET)
        if (_tiny_rc EQUAL 0)
            set(_tiny_py "${_cand}")
        endif()
    endif()
endforeach()

set(_tiny_tests ggml_tesseract_graph_qwen_tiny ggml_tesseract_graph_qwen_tiny_q4k ggml_tesseract_graph_gemma_tiny
                ggml_tesseract_graph_gemma_tiny_shared ggml_tesseract_graph_gemma_tiny_shared_ub1024
                ggml_tesseract_graph_qwen35_tiny ggml_tesseract_graph_qwen35_tiny_s128
                ggml_tesseract_graph_qwen3moe_tiny ggml_tesseract_graph_qwen3moe_tiny_q4k)
if (NOT _tiny_py)
    set(_tiny_msg "tiny models: no Python with numpy and gguf-py among '${_tiny_candidates}'; set -DTSR_TINY_PYTHON=<python with numpy> (sw/ggml-tesseract/tests/tiny.cmake)")
    message(WARNING "${_tiny_msg}")
    foreach (_t ${_tiny_tests})
        add_test(NAME ${_t} COMMAND sh -c "echo \"$1\" >&2; exit 1" sh "${_tiny_msg}")
    endforeach()
    return()
endif()
message(STATUS "tiny models: gen_tiny_models.py run by ${_tiny_py}")

set(_tiny_outputs ${TSR_TINY_MODEL_DIR}/qwen2-tiny-q8_0.gguf ${TSR_TINY_MODEL_DIR}/qwen2-tiny-q4_k.gguf
                  ${TSR_TINY_MODEL_DIR}/gemma4-tiny-q8_0.gguf ${TSR_TINY_MODEL_DIR}/gemma4-tiny-moe-q8_0.gguf ${TSR_TINY_MODEL_DIR}/gemma4-tiny-shared-q8_0.gguf
                  ${TSR_TINY_MODEL_DIR}/qwen35-tiny-q8_0.gguf
                  ${TSR_TINY_MODEL_DIR}/qwen35-tiny-s128-q8_0.gguf ${TSR_TINY_MODEL_DIR}/qwen3moe-tiny-q8_0.gguf
                  ${TSR_TINY_MODEL_DIR}/qwen3moe-tiny-q4_k.gguf ${TSR_TINY_MODEL_DIR}/qwen35-tiny-mtp-q8_0.gguf
                  ${TSR_TINY_MODEL_DIR}/qwen35-tiny-s128-mtp-q8_0.gguf ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt)
add_custom_command(
    OUTPUT ${_tiny_outputs}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${_tiny_gguf_py}
            ${_tiny_py} ${CMAKE_CURRENT_LIST_DIR}/gen_tiny_models.py --out-dir ${TSR_TINY_MODEL_DIR} --check -q
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/gen_tiny_models.py ${_tiny_gguf_py}/gguf/quants.py ${_tiny_gguf_py}/gguf/gguf_writer.py
    COMMENT "Generating the tiny Qwen2 / Gemma 4 GGUFs (SW-21)"
    VERBATIM)
add_custom_target(tsr_tiny_models ALL DEPENDS ${_tiny_outputs})

# ---- graph tests (klcheck is created by a deferred call in sw/tools under the same condition) ------
if (TARGET llama)
    set(_tiny_m7 ${CMAKE_CURRENT_LIST_DIR}/m7_tests.py)
    # name | model | --min-kl | --min-top1 (the bounds: header)
    set(_tiny_graph_cases
        "ggml_tesseract_graph_qwen_tiny|qwen2-tiny-q8_0.gguf|0|0.95"
        "ggml_tesseract_graph_qwen_tiny_q4k|qwen2-tiny-q4_k.gguf|0|0.95"
        "ggml_tesseract_graph_gemma_tiny|gemma4-tiny-q8_0.gguf|3e-3|0.95"
        "ggml_tesseract_graph_gemma_tiny_shared|gemma4-tiny-shared-q8_0.gguf|3e-3|0.95"
        "ggml_tesseract_graph_qwen35_tiny|qwen35-tiny-q8_0.gguf|0|0.95"
        "ggml_tesseract_graph_qwen35_tiny_s128|qwen35-tiny-s128-q8_0.gguf|0|0.95"
        "ggml_tesseract_graph_qwen3moe_tiny|qwen3moe-tiny-q8_0.gguf|1e-3|0.95"
        "ggml_tesseract_graph_qwen3moe_tiny_q4k|qwen3moe-tiny-q4_k.gguf|1e-3|0.95")
    foreach (_case ${_tiny_graph_cases})
        string(REPLACE "|" ";" _f "${_case}")
        list(GET _f 0 _name)
        list(GET _f 1 _model)
        list(GET _f 2 _min_kl)
        list(GET _f 3 _min_top1)
        set(_cells_arg "")
        if (_model MATCHES "^qwen35")
            set(_cells_arg --gdn-cells all)   # SW-82: every GATED_DELTA_NET record runs on its state cache cell
        endif()
        add_test(NAME ${_name}
                 COMMAND ${_tiny_py} ${_tiny_m7} graph --klcheck $<TARGET_FILE:klcheck>
                         --model ${TSR_TINY_MODEL_DIR}/${_model} --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                         --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/${_name}
                         --n-prompts 4 -n 17 --prompt-tail 8 -ub 16 -t 4 --min-decodes 64
                         --min-kl ${_min_kl} --min-top1 ${_min_top1} ${_cells_arg})
        set(_timeout 300)
        if (_model MATCHES "^qwen35")
            set(_timeout 900)      # S_v 128: 40 s alone, several times that on a loaded host
            # SW-82: the same run with TSR_GDN_CELL=0 has the gather and the copy back (the three passes of the pinned graph)
            add_test(NAME ${_name}_nocell
                     COMMAND ${_tiny_py} ${_tiny_m7} graph --klcheck $<TARGET_FILE:klcheck>
                             --model ${TSR_TINY_MODEL_DIR}/${_model} --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                             --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/${_name}_nocell
                             --n-prompts 4 -n 17 --prompt-tail 8 -ub 16 -t 4 --min-decodes 64
                             --min-kl ${_min_kl} --min-top1 ${_min_top1} --gdn-cells none)
            set_tests_properties(${_name}_nocell PROPERTIES TIMEOUT ${_timeout}
                                 ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_GDN_CELL=0")
        endif()
        set_tests_properties(${_name} PROPERTIES TIMEOUT ${_timeout} ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal")
    endforeach()
    # SW-92, SW-94: the device output with and without the MUL_MAT deferral (TSR_MM_DEFER=0 lowers every MUL_MAT as one record) is bit-identical:
    # klcheck generates two prompts (prefill in ubatches of 16 tokens, then single-token decodes) with the default and with the knob off on the
    # dense (Qwen2), Gemma 4, Qwen3.5 (Gated DeltaNet) and MoE tiny models, and `klcheck compare --expect-identical` compares the logits. The default run
    # must have launched MUL_MAT_BEGIN / MUL_MAT_END records, the other none, with the same launches and the same other records.
    foreach (_m qwen2-tiny-q8_0 gemma4-tiny-q8_0 qwen35-tiny-q8_0 qwen3moe-tiny-q8_0)
        string(REPLACE "-tiny-q8_0" "" _short ${_m})
        # T_MAX 4 with ubatches of 16 tokens (a prefill ubatch is several T-tiles: those MUL_MATs stay one record), and T_MAX 8 with ubatches of
        # 8 (a prefill ubatch is one tile, so its MUL_MATs are split too)
        foreach (_v "|16|" "_tmax8|8|TSR_SIM_TMAX=8")
            string(REPLACE "|" ";" _vf "${_v}")
            list(GET _vf 0 _sfx)
            list(GET _vf 1 _ub)
            list(GET _vf 2 _tm)
            add_test(NAME ggml_tesseract_defer_${_short}${_sfx}
                     COMMAND ${_tiny_py} ${_tiny_m7} defer --klcheck $<TARGET_FILE:klcheck>
                             --model ${TSR_TINY_MODEL_DIR}/${_m}.gguf --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                             --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/ggml_tesseract_defer_${_short}${_sfx}
                             --n-prompts 2 -n 9 --prompt-tail 8 -ub ${_ub} -t 4 --min-pairs 1)
            set_tests_properties(ggml_tesseract_defer_${_short}${_sfx} PROPERTIES TIMEOUT 900
                                 ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;${_tm}")
        endforeach()
    endforeach()
    # The same with the lists cut after 7 records (TSR_OPL_SPLIT, T_MAX 8, ubatches of 8): the builder adds the END of every open BEGIN before each
    # cut, so a BEGIN and its END are always in one list, and the output does not depend on where the lists are cut.
    foreach (_m gemma4-tiny-q8_0 qwen35-tiny-q8_0 qwen3moe-tiny-q8_0)
        string(REPLACE "-tiny-q8_0" "" _short ${_m})
        add_test(NAME ggml_tesseract_defer_${_short}_split
                 COMMAND ${_tiny_py} ${_tiny_m7} defer --klcheck $<TARGET_FILE:klcheck>
                         --model ${TSR_TINY_MODEL_DIR}/${_m}.gguf --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                         --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/ggml_tesseract_defer_${_short}_split
                         --n-prompts 2 -n 9 --prompt-tail 8 -ub 8 -t 4 --min-pairs 1 --split)
        set_tests_properties(ggml_tesseract_defer_${_short}_split PROPERTIES TIMEOUT 900
                             ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_TMAX=8;TSR_OPL_SPLIT=7")
    endforeach()
    # Tesseract Pivot, the graph scheduler (docs/design/07-graph-scheduler.md): the device output with the pass (the default) and with TSR_PIVOT=0
    # is bit-identical, launch for launch, for the prompt (ubatches of 16 tokens), the single-token decodes and a replay in verification
    # windows of 4 tokens (the batches of speculative decoding, T_MAX 4); the pass must have moved nodes. The MoE model with the dense FFN
    # beside its experts and the Qwen3.5 model also have the pass's mark checked in the decode launch's records: the dense gate product is a
    # MUL_MAT_BEGIN in front of MOE_ROUTE and its MUL_MAT_END comes after the MUL_MAT_ID records; the z product is a MUL_MAT_BEGIN in front of the
    # GATED_DELTA_NET record with its MUL_MAT_END behind it.
    # name | model | --min-moves | extra arguments
    foreach (_pv "qwen2|qwen2-tiny-q8_0|0|" "gemma4|gemma4-tiny-q8_0|1|" "gemma4moe|gemma4-tiny-moe-q8_0|1|--expect=moe"
                 "qwen35|qwen35-tiny-q8_0|1|--expect=gdn" "qwen3moe|qwen3moe-tiny-q8_0|0|")
        string(REPLACE "|" ";" _pvf "${_pv}")
        list(GET _pvf 0 _pvn)
        list(GET _pvf 1 _pvm)
        list(GET _pvf 2 _pvmin)
        list(LENGTH _pvf _pvlen)
        set(_pvx "")
        if (_pvlen GREATER 3)
            list(GET _pvf 3 _pvx)
        endif()
        add_test(NAME ggml_tesseract_pivot_${_pvn}
                 COMMAND ${_tiny_py} ${_tiny_m7} pivot --klcheck $<TARGET_FILE:klcheck>
                         --model ${TSR_TINY_MODEL_DIR}/${_pvm}.gguf --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                         --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/ggml_tesseract_pivot_${_pvn}
                         --n-prompts 2 -n 9 --prompt-tail 8 -ub 16 -t 4 --min-moves ${_pvmin} --verify-k 3 ${_pvx})
        set_tests_properties(ggml_tesseract_pivot_${_pvn} PROPERTIES TIMEOUT 900
                             ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_TMAX=4")
    endforeach()
    # SW-83: klcheck's verification windows (--verify-k 3): the recorded tokens of the plain device run decoded in windows of 4
    # tokens, every row an output, the accepted length cycling through 0 .. 3 (wrong tokens behind it, a rollback to the
    # recurrent snapshot). The rows must meet klcheck's criterion against the CPU (the openTPU absolute bound: the tiny
    # models' floors are small) and must be bit-identical to the plain device run (T_MAX 4: a window is one pass).
    foreach (_m qwen35-tiny-mtp-q8_0 qwen35-tiny-s128-mtp-q8_0)
        add_test(NAME ggml_tesseract_verify_${_m}
                 COMMAND klcheck all -m ${TSR_TINY_MODEL_DIR}/${_m}.gguf --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                         --n-prompts 4 -n 17 --prompt-tail 8 -ub 16 -t 4 --phase m7 --verify-k 3 --verify-identical
                         --min-kl 1e-3 --min-top1 0.95)
        set_tests_properties(ggml_tesseract_verify_${_m} PROPERTIES TIMEOUT 900
                             ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_TMAX=4")
    endforeach()
    # SW-43: a prefill ubatch of 692 tokens (the first four prompts joined, -ub 1024): its
    # FLASH_ATTN needs 174 MSUM + 87 FA_NEW preload lines (> 256, one list), so supports_op
    # keeps it on the CPU (extra launches in that prefill, reported, not checked); the decodes stay one
    # launch each and the KL criterion holds.
    add_test(NAME ggml_tesseract_graph_qwen_tiny_ub1024
             COMMAND ${_tiny_py} ${_tiny_m7} graph --klcheck $<TARGET_FILE:klcheck>
                     --model ${TSR_TINY_MODEL_DIR}/qwen2-tiny-q8_0.gguf --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                     --join-prompts 4 --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/ggml_tesseract_graph_qwen_tiny_ub1024
                     --n-prompts 1 -n 17 --prompt-tail 8 -ub 1024 -t 4 --min-decodes 16 --prefill any
                     --min-prefill-tokens 681 --min-kl 0 --min-top1 0.95)
    set_tests_properties(ggml_tesseract_graph_qwen_tiny_ub1024 PROPERTIES TIMEOUT 300
                         ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal")
    # KV sharing across list cuts (SW-43, SW §9.3 FA_NEW): the tiny Gemma 4 whose last two layers attend to the K and V planes of the
    # first two, a prefill ubatch of 365 tokens (the first three prompts joined, -ub 1024) whose MSUM and new-cell indices take 138
    # preload lines per FLASH_ATTN, so the op lists are cut (4 launches), then the decodes (one launch each). The FLASH_ATTNs of the sharing layers name the SET_ROWS indices of the owner layers as their new cells.
    add_test(NAME ggml_tesseract_graph_gemma_tiny_shared_ub1024
             COMMAND ${_tiny_py} ${_tiny_m7} graph --klcheck $<TARGET_FILE:klcheck>
                     --model ${TSR_TINY_MODEL_DIR}/gemma4-tiny-shared-q8_0.gguf --prompts ${TSR_TINY_MODEL_DIR}/tiny-prompts.txt
                     --join-prompts 3 --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/ggml_tesseract_graph_gemma_tiny_shared_ub1024
                     --n-prompts 1 -n 17 --prompt-tail 8 -ub 1024 -t 4 --min-decodes 16 --prefill any
                     --min-prefill-tokens 300 --min-kl 3e-3 --min-top1 0.95)
    set_tests_properties(ggml_tesseract_graph_gemma_tiny_shared_ub1024 PROPERTIES TIMEOUT 300
                         ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal")
    # The same checks on the real Qwen2.5-0.5B Q8_0 (SW §10.4 M7 gate item 1 on the simulator; label
    # model, registered when the GGUF is present; models/ is not in git): klcheck's 4 built-in
    # prompts x 33 tokens (128 single-token decodes), -ub 512, klcheck's criterion unchanged:
    # one launch per token (365 op records), KL ratio 0.77 at 2 x 17 tokens, 86 ms per token on the sim.
    set(_tiny_05b ${TESSERACT_ROOT}/models/qwen2.5-0.5b-instruct-q8_0.gguf)
    if (EXISTS ${_tiny_05b})
        add_test(NAME ggml_tesseract_graph_qwen05b
                 COMMAND ${_tiny_py} ${_tiny_m7} graph --klcheck $<TARGET_FILE:klcheck> --model ${_tiny_05b}
                         --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/ggml_tesseract_graph_qwen05b
                         --n-prompts 4 -n 33 --prompt-tail 1 -ub 512 -t 4 --min-decodes 128 --min-top1 0.8)
        set_tests_properties(ggml_tesseract_graph_qwen05b PROPERTIES TIMEOUT 900 LABELS model
                             ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_THREADS=4")
    endif()
endif()

# ---- the profile report (sw/tools/prof, docs/prof-dbg.md "The profile report") ----------------------
# TSR_DUMP_PLAN on a tiny MoE model and a tiny dense Gemma 4 on the simulator: the dump's files, the same generated text as
# without it, prof_report.py's report.html / trace.perfetto.json / summary.json and the rules of the page. The ideal MVM
# cycles are checked on the plain simulator and on the card image's (TSR_SIM_CARD=k4dual) with the dual weight stream on and
# off (TSR_MVM_DUAL=0): the rule, the simulated run time of every MVM op, meta.json's mvm_dual and the E and F of the utilization split.
if (TARGET llama-completion AND TSR_FW_ELF)
    foreach (_prof_case "qwen3moe_tiny|qwen3moe-tiny-q8_0.gguf" "gemma_tiny|gemma4-tiny-q8_0.gguf")
        string(REPLACE "|" ";" _f "${_prof_case}")
        list(GET _f 0 _pname)
        list(GET _f 1 _pmodel)
        add_test(NAME tools_prof_report_${_pname}
                 COMMAND ${_tiny_py} ${TESSERACT_ROOT}/sw/tools/prof/test_prof_report.py --build-dir ${CMAKE_BINARY_DIR}
                         --model ${TSR_TINY_MODEL_DIR}/${_pmodel} --out-dir ${CMAKE_BINARY_DIR}/sw/tools/prof/${_pname} --fw ${TSR_FW_ELF})
        set_tests_properties(tools_prof_report_${_pname} PROPERTIES TIMEOUT 300)
    endforeach()
endif()

# ---- llama-server on the device (tests/server_smoke.py) ---------------------------------------------
# Two llama-server processes on the tiny Qwen2 model, the device (sim) and the CPU, with the M7 settings
# and 2 slots (--kv-unified, one KV stream: SW §8 R10): /health, /v1/models, a chat completion, the same
# chat streamed (SSE), a completion, 2 concurrent chats at temperature 0 and the first chat again, which
# the server restores from its RAM prompt cache (the KV state read back from and written to the device).
# Every text must equal the CPU's, both servers must exit 0 on SIGINT, and the device's TSR_STATS must
# show one launch per graph_compute. About 2 s; Python standard library only. The script's own time
# limits add up to 105 s, below the ctest TIMEOUT, so a hang ends in a FAIL line, not in a ctest kill.
if (TARGET llama-server)
    add_test(NAME ggml_tesseract_server_tiny
             COMMAND ${_tiny_py} ${CMAKE_CURRENT_LIST_DIR}/server_smoke.py --server $<TARGET_FILE:llama-server>
                     --model ${TSR_TINY_MODEL_DIR}/qwen2-tiny-q8_0.gguf
                     --out-dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/ggml_tesseract_server_tiny)
    set_tests_properties(ggml_tesseract_server_tiny PROPERTIES TIMEOUT 120
                         ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal")
endif()

# ---- test-backend-ops, one ctest per M7 op filter -------------------------------------------------
if (TARGET test-backend-ops)
    set(_tbo_dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/backend_ops)
    file(MAKE_DIRECTORY ${_tbo_dir})
    # test-backend-ops -o value | MIN_RUN | -p value (none: "-")
    set(_tbo_cases
        "ADD|1|-" "MUL|1|-" "RMS_NORM|1|-" "SWIGLU|1|-" "GEGLU|1|-" "ROPE|1|-" "GET_ROWS|1|-"
        "CPY|1|-" "CONT|1|-" "TANH|1|-" "GELU|1|-"
        # every SCALE / SOFTCAP case of the pinned test-backend-ops has rows of 10 or 100 f32: not
        # 64-byte rows (SW §9.3), so none is supported; the tiny Gemma graph runs both on the device
        "SCALE|0|-" "SOFTCAP|0|-"
        # fused / multi-node cases of the builder (SW §9.3 pass 3)
        "RMS_NORM_MUL_ADD|1|-" "ADD_RMS_NORM|1|-" "ADD_ADD|1|-" "MUL_MAT_ADD|1|-" "RMS_NORM_MUL_ROPE|1|-"
        # KV-cache ops: test-backend-ops' tensors are not KV-cache tensors (unsupported by design);
        # FLASH_ATTN_EXT limited to q8_0 K / V (5,343 cases otherwise, all refused for their types)
        "SET_ROWS|0|-" "ROPE_SET_ROWS|0|-" "FLASH_ATTN_EXT|0|type_K=q8_0,type_V=q8_0")
    foreach (_case ${_tbo_cases})
        string(REPLACE "|" ";" _f "${_case}")
        list(GET _f 0 _op)
        list(GET _f 1 _min)
        list(GET _f 2 _p)
        string(TOLOWER "${_op}" _lop)
        set(_args --test-backend-ops $<TARGET_FILE:test-backend-ops> --op ${_op} --min-run ${_min}
                  --list ${_tbo_dir}/${_lop}.unsupported.txt)
        if (NOT _p STREQUAL "-")
            list(APPEND _args --params ${_p})
        endif()
        add_test(NAME ggml_tesseract_backend_ops_${_lop} COMMAND ${_tiny_py} ${CMAKE_CURRENT_LIST_DIR}/m7_tests.py ops ${_args})
        set_tests_properties(ggml_tesseract_backend_ops_${_lop} PROPERTIES TIMEOUT 600 ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal")
    endforeach()
endif()

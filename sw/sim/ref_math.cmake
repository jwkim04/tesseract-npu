# sw/sim/ref_math.cmake — the VU's approximate and composite math (sw/sim/tsr_ref_math.c,
# docs/design/02-vector-unit.md §6) in the reference library, and its tests (VU §10 items 1-2,
# docs/design/06-software-verification.md §10.1-§10.2, SW-23).
#
# Include from sw/sim/CMakeLists.txt after tsr_c_target() is defined:
#     include(${CMAKE_CURRENT_SOURCE_DIR}/ref_math.cmake)
# Before the tsr_ref / tsr_ref_shared add_library() calls it extends TSR_REF_SOURCES; after them
# it adds the source to both targets. Either way TSR_REF_SOURCES carries tsr_ref_math.c for any
# later target built from the list (e.g. sanitizer builds).
#
# Tests (prefix ref_math_; run: ctest --test-dir build/dev -R '^ref_math_' --output-on-failure):
#   ref_math_specials       directed cases (§6.5 specials table, EXPSUB / GEGLU16 rules, binary16
#                           helpers, symmetries) and an int64 / __int128 transcription of the
#                           EXPK and SIN/COS integer steps against the portable code
#   ref_math_acc_<fn>       accuracy against double libm on 2^24 strided inputs + specials;
#                           gate: measured <= 2x the §6.9 model figure, plus the property checks
#   ref_math_acc_<fn>_full  the same over all 2^32 inputs. LABELS slow; runs only when the
#                           environment has TSR_REF_MATH_FULL=1, otherwise reports "skipped":
#                               TSR_REF_MATH_FULL=1 ctest --test-dir build/dev -L slow -R '^ref_math_'
#   ref_math_gelu16         all 65,536 fp16 inputs of GELU16 against the pinned ggml CPU table
#                           ggml_table_gelu_f16 (links ggml-cpu; TSR_WITH_LLAMA only)
#   ref_math_dump_smoke     the dump CLI against its own --check mode
#   ref_math_np_<fn>        the numpy model sw/sim/tests/ref_math_np.py and ref_math_dump bit for
#                           bit (2^24 inputs at stride 256 and 257 + special windows;
#                           tests/ref_math_np_cmp.sh); ref_math_np_<fn>_full (LABELS slow,
#                           TSR_REF_MATH_FULL=1) over 2^32; ref_math_np_selftest: the model's
#                           --self-test
#
# Python for the numpy model: -DTSR_REF_MATH_PYTHON=<interpreter with numpy> (mpmath optional:
# the self-test then also checks the constants). Default: the first of
# build/tools/torchvenv/bin/python (the project venv, as sw/torch uses) and the python3 CMake finds
# that imports numpy. On the dev host use the torchvenv or ~/miniconda3/bin/python (numpy 2.4 and
# mpmath): the python3 first on PATH there is an unrelated venv. If no candidate has numpy,
# configure warns and registers ref_math_np_selftest as a test that fails with that message (the
# comparison is a required check of VU §10 item 1, not an optional one).

set(TSR_REF_MATH_SOURCE ${CMAKE_CURRENT_LIST_DIR}/tsr_ref_math.c)
list(APPEND TSR_REF_SOURCES ${TSR_REF_MATH_SOURCE})
foreach(_tgt tsr_ref tsr_ref_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_REF_MATH_SOURCE})
    endif()
endforeach()

set(_rm_tests ${CMAKE_CURRENT_LIST_DIR}/tests)
find_package(Threads REQUIRED)

function(tsr_ref_math_exe name src)
    add_executable(${name} ${_rm_tests}/${src})
    tsr_c_target(${name})
    target_include_directories(${name} PRIVATE ${_rm_tests})
    target_link_libraries(${name} PRIVATE tsr_ref ${ARGN} m)
endfunction()

tsr_ref_math_exe(ref_math_dump     ref_math_dump.c)
tsr_ref_math_exe(ref_math_acc      ref_math_acc.c Threads::Threads)
tsr_ref_math_exe(ref_math_specials ref_math_specials.c)

add_test(NAME ref_math_specials COMMAND ref_math_specials)

# Functions with a §6.9 figure (names as ref_math_acc and ref_math_dump spell them).
set(TSR_REF_MATH_FUNCS expk_p1 expk_n1 expk_n2 expk_ng sigmoid silu gelu_tanh tanh sin cos)
foreach(_fn ${TSR_REF_MATH_FUNCS})
    add_test(NAME ref_math_acc_${_fn} COMMAND ref_math_acc --threads 4 ${_fn})
    set_tests_properties(ref_math_acc_${_fn} PROPERTIES PROCESSORS 4 TIMEOUT 300)
    add_test(NAME ref_math_acc_${_fn}_full COMMAND ref_math_acc --slow ${_fn})
    set_tests_properties(ref_math_acc_${_fn}_full PROPERTIES
                         LABELS slow SKIP_RETURN_CODE 77 PROCESSORS 8 TIMEOUT 3600)
endforeach()

add_test(NAME ref_math_dump_smoke
         COMMAND sh -c "\"$<TARGET_FILE:ref_math_dump>\" sincos 0 70000 61681 | \"$<TARGET_FILE:ref_math_dump>\" --check sincos 0 70000 61681 && \"$<TARGET_FILE:ref_math_dump>\" geglu16 0xC1300000 4096 4099 0x40000000 | \"$<TARGET_FILE:ref_math_dump>\" --check geglu16 0xC1300000 4096 4099 0x40000000")

# GELU16 against the pinned CPU table. ggml-cpu is defined later in the top-level project
# (third_party/llama.cpp); target names resolve at generate time, as for ref_ggml.
if (TSR_WITH_LLAMA)
    tsr_ref_math_exe(ref_math_gelu16 ref_math_gelu16.c ggml-cpu)
    target_include_directories(ref_math_gelu16 SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    add_test(NAME ref_math_gelu16 COMMAND ref_math_gelu16)
endif()

# Bit-exact comparison with the independent numpy model (same CLI as ref_math_dump), when the
# model is present (VU §10 item 1, SW §10.1). sw/sim/tests/ref_math_np_cmp.sh runs the windows:
# 2^24 inputs at stride 256 and at stride 257 from 0, then special windows (zeros, subnormals,
# smallest normals, largest finites, infinities, NaNs and each function's thresholds). Every
# function both CLIs know is compared: the VU math, the binary16 helpers and the f32 primitives
# (the model's primitive emulation against tsr_ref.c's). Binary functions run once per fixed
# second operand below. Interpreter: see the header (TSR_REF_MATH_PYTHON).
if (EXISTS ${_rm_tests}/ref_math_np.py)
    set(TSR_REF_MATH_PYTHON "" CACHE FILEPATH
        "Python with numpy for ref_math_np.py (default: build/tools/torchvenv/bin/python, else the found python3)")
    set(_rm_py_candidates "${TSR_REF_MATH_PYTHON}")
    if (NOT TSR_REF_MATH_PYTHON)
        set(_rm_py_candidates "")
        if (EXISTS ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
            list(APPEND _rm_py_candidates ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
        endif()
        find_package(Python3 COMPONENTS Interpreter)
        if (Python3_EXECUTABLE)
            list(APPEND _rm_py_candidates ${Python3_EXECUTABLE})
        endif()
    endif()
    set(_rm_have_numpy FALSE)
    set(_rm_py "")
    foreach(_cand ${_rm_py_candidates})
        if (NOT _rm_have_numpy)
            execute_process(COMMAND "${_cand}" -c "import numpy"
                            RESULT_VARIABLE _rm_np_rc OUTPUT_QUIET ERROR_QUIET)
            if (_rm_np_rc EQUAL 0)
                set(_rm_have_numpy TRUE)
                set(_rm_py "${_cand}")
            endif()
        endif()
    endforeach()
    if (NOT _rm_have_numpy)
        set(_rm_msg "ref_math_np_*: no Python with numpy among '${_rm_py_candidates}'; set -DTSR_REF_MATH_PYTHON=<python with numpy> (see sw/sim/ref_math.cmake)")
        message(WARNING "${_rm_msg}")
        add_test(NAME ref_math_np_selftest COMMAND sh -c "echo \"$1\" >&2; exit 1" sh "${_rm_msg}")
    else()
        message(STATUS "ref_math_np_*: numpy model run by ${_rm_py}")
        set(_rm_cmp sh ${_rm_tests}/ref_math_np_cmp.sh "${_rm_py}" ${_rm_tests}/ref_math_np.py
                    $<TARGET_FILE:ref_math_dump>)
        # Unary functions (sincos: two words per input), the binary16 helpers and SQ / RC.
        set(_rm_np_unary exp ${TSR_REF_MATH_FUNCS} gelu16 sincos f32_to_f16_rne f16_to_f32 fsqrt frecip)
        # Binary functions: the swept input is the first operand; B = 1, -0, +-inf, NaN, a
        # subnormal (DAZ), -10 and 2^25 (for expsub B is the subtrahend x, the running maximum of
        # an online softmax). The special windows run with B = 1.
        set(_rm_np_binary expsub geglu16 geglu swiglu fadd fsub fmul fdiv)
        set(_rm_np_b 3F800000 80000000 7F800000 FF800000 7FC00000 00400000 C1200000 4C000000)
        foreach(_fn ${_rm_np_unary} ${_rm_np_binary})
            set(_b "")
            if (_fn IN_LIST _rm_np_binary)
                set(_b ${_rm_np_b})
            endif()
            add_test(NAME ref_math_np_${_fn} COMMAND ${_rm_cmp} strided ${_fn} ${_b})
            set_tests_properties(ref_math_np_${_fn} PROPERTIES TIMEOUT 1800)
            # All 2^32 inputs in 256 chunks of 2^24 (bounded memory on the numpy side); binary
            # functions with B = 1. LABELS slow; runs only with TSR_REF_MATH_FULL=1:
            #     TSR_REF_MATH_FULL=1 ctest --test-dir build/dev -L slow -R '^ref_math_np_'
            add_test(NAME ref_math_np_${_fn}_full COMMAND ${_rm_cmp} full ${_fn} ${_b})
            set_tests_properties(ref_math_np_${_fn}_full PROPERTIES
                                 LABELS slow SKIP_RETURN_CODE 77 TIMEOUT 7200)
        endforeach()
        # expk with κ given as the fifth argument (the CLI form both share): every κ index.
        add_test(NAME ref_math_np_expk COMMAND ${_rm_cmp} strided expk 0 1 2 3)
        set_tests_properties(ref_math_np_expk PROPERTIES TIMEOUT 1800)
        # The model's own checks (constants against mpmath when present, scalar transcriptions,
        # properties, accuracy against double), so a model regression is not hidden by a matching
        # C regression.
        add_test(NAME ref_math_np_selftest COMMAND "${_rm_py}" ${_rm_tests}/ref_math_np.py --self-test)
        set_tests_properties(ref_math_np_selftest PROPERTIES TIMEOUT 600)
    endif()
endif()

# sw/sim/refvu.cmake — the VU op references (sw/sim/tsr_ref_vu.c: VOP, RED, ROPE, TOPK, SOP, DQROW;
# docs/design/02-vector-unit.md §4–§6) in the reference library, and their tests (VU §10 items 1 and 3,
# docs/design/06-software-verification.md §10.1 "VU/AT references", SW-23).
#
# Include from sw/sim/CMakeLists.txt after ref_math.cmake (tsr_ref_vu.c calls tsr_ref_math.c):
#     include(${CMAKE_CURRENT_SOURCE_DIR}/refvu.cmake)
# Like ref_math.cmake: before the tsr_ref / tsr_ref_shared add_library() calls it extends
# TSR_REF_SOURCES; after them it adds the source to both targets. Either way TSR_REF_SOURCES carries
# tsr_ref_vu.c for any later target built from the list (e.g. sanitizer builds). The header is
# sw/sim/include/tsr_ref_vu.h (in TSR_SIM_INCLUDE_DIR).
#
# Tests (run: ctest --test-dir build/dev -R '^ref_vu' --output-on-failure):
#   ref_vu      tests/test_ref_vu.c: directed cases (order / MAXT / MINT, every VOP function's
#               specials, shapes with strides / broadcast / cmode / tails / in place, RED's -0
#               subtrees at m = 8k + r against both skip models, post-ops = the SOP recipe, ROPE
#               angles against a transcription of ggml_rope_cache_init at positions 0 .. 2^17 - 1,
#               rotation accuracy against double, TOPK, SOP, DQROW, EXPSUB softmax sequences).
#               With TSR_WITH_LLAMA it links the pinned ggml-cpu and adds ggml_rope_ext (angles
#               bit-identical for Qwen2.5 / Gemma 4 parameters, positions 0 .. 32767) and
#               ggml_rms_norm (RED RMS + VOP.MULC; ulp difference reported).
#   ref_vu_np   tests/ref_vu_np.py --self-test --compare libtsr_ref.so: the independent numpy model
#               (written from the VU text; element math from tests/ref_math_np.py) against the C
#               through ctypes, bit for bit on generated SPM images (VOP every function / cmode /
#               shape, RED every kind / post / TO_S / skip model, ROPE both forms, TOPK, SOP).
#
# Python for the numpy model: -DTSR_REF_VU_PYTHON=<interpreter with numpy>. Default: the first of
# TSR_REF_MATH_PYTHON, build/tools/torchvenv/bin/python and the python3 CMake finds that imports
# numpy (on the dev host the python3 first on PATH is an unrelated venv; see ref_math.cmake). If
# none has numpy, configure warns and ref_vu_np fails with that message (a required check, SW §10.1).

set(TSR_REF_VU_SOURCE ${CMAKE_CURRENT_LIST_DIR}/tsr_ref_vu.c)
list(APPEND TSR_REF_SOURCES ${TSR_REF_VU_SOURCE})
foreach(_tgt tsr_ref tsr_ref_shared)
    if (TARGET ${_tgt})
        target_sources(${_tgt} PRIVATE ${TSR_REF_VU_SOURCE})
    endif()
endforeach()

set(_rv_tests ${CMAKE_CURRENT_LIST_DIR}/tests)

add_executable(ref_vu ${_rv_tests}/test_ref_vu.c)
tsr_c_target(ref_vu)
target_include_directories(ref_vu PRIVATE ${_rv_tests})
target_link_libraries(ref_vu PRIVATE tsr_ref m)
if (TSR_WITH_LLAMA)
    # ggml-cpu / ggml-base are defined later in the top-level project (third_party/llama.cpp);
    # target names resolve at generate time, as for ref_ggml and ref_math_gelu16.
    target_compile_definitions(ref_vu PRIVATE TSR_REF_VU_GGML=1)
    target_include_directories(ref_vu SYSTEM PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    target_link_libraries(ref_vu PRIVATE ggml-cpu ggml-base)
endif()
add_test(NAME ref_vu COMMAND ref_vu)
set_tests_properties(ref_vu PROPERTIES TIMEOUT 300 PROCESSORS 4)

set(TSR_REF_VU_PYTHON "" CACHE FILEPATH
    "Python with numpy for ref_vu_np.py (default: TSR_REF_MATH_PYTHON, build/tools/torchvenv/bin/python, else the found python3)")
set(_rv_candidates ${TSR_REF_VU_PYTHON} ${TSR_REF_MATH_PYTHON})
if (EXISTS ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
    list(APPEND _rv_candidates ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
endif()
find_package(Python3 COMPONENTS Interpreter)
if (Python3_EXECUTABLE)
    list(APPEND _rv_candidates ${Python3_EXECUTABLE})
endif()
set(_rv_py "")
foreach(_cand ${_rv_candidates})
    if (NOT _rv_py)
        execute_process(COMMAND "${_cand}" -c "import numpy"
                        RESULT_VARIABLE _rv_rc OUTPUT_QUIET ERROR_QUIET)
        if (_rv_rc EQUAL 0)
            set(_rv_py "${_cand}")
        endif()
    endif()
endforeach()
if (NOT _rv_py)
    set(_rv_msg "ref_vu_np: no Python with numpy among '${_rv_candidates}'; set -DTSR_REF_VU_PYTHON=<python with numpy> (see sw/sim/refvu.cmake)")
    message(WARNING "${_rv_msg}")
    add_test(NAME ref_vu_np COMMAND sh -c "echo \"$1\" >&2; exit 1" sh "${_rv_msg}")
else()
    message(STATUS "ref_vu_np: numpy model run by ${_rv_py}")
    add_test(NAME ref_vu_np COMMAND "${_rv_py}" ${_rv_tests}/ref_vu_np.py --self-test
                                    --compare $<TARGET_FILE:tsr_ref_shared>)
    set_tests_properties(ref_vu_np PROPERTIES TIMEOUT 600)
endif()

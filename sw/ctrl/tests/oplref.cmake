# sw/ctrl/tests/oplref.cmake — tsr_opl_ref: the host reference executor of an op-list blob (sw/ctrl/ref/opl_ref.cpp) with the
# per-kind references it composes (mulmat_ref, elem_ref, rmsnorm_ref, rope_ref, kv_ref, fattn_ref), as one static PIC library.
# Linked by the runtime sw/rt (tsr_rt: graph_run_ref) and by the tests of the framework-free encoders (ggml_tesseract_kinds).
# Included by sw/ctrl/host/CMakeLists.txt (main project, after the snippets that define the kinds' tests).
#
#   tsr_opl_ref   opl_exec: the preload copy, then every record in list order through its reference; no simulator, no firmware
#                 (docs/design/06-software-verification.md §7.2, §7.3)

add_library(tsr_opl_ref STATIC
    ${TSR_CTRL_DIR}/ref/opl_ref.cpp
    ${TSR_CTRL_DIR}/ref/mulmat_ref.cpp
    ${TSR_CTRL_DIR}/ref/elem_ref.cpp
    ${TSR_CTRL_DIR}/ref/rmsnorm_ref.cpp
    ${TSR_CTRL_DIR}/ref/rope_ref.cpp
    ${TSR_CTRL_DIR}/ref/kv_ref.cpp
    ${TSR_CTRL_DIR}/ref/fattn_ref.cpp)
set_target_properties(tsr_opl_ref PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(tsr_opl_ref PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_include_directories(tsr_opl_ref PUBLIC ${TSR_CTRL_DIR}/ref ${TSR_CTRL_INCLUDE_DIR} ${TESSERACT_ROOT}/sw/sim/include)
target_link_libraries(tsr_opl_ref PUBLIC tsr_ref Threads::Threads)

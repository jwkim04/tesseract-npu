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
# device needed (docs/design/06-software-verification.md §10.2: ggml_tesseract_units).
add_executable(test-tsr-backend-units tests/test-tsr-backend-units.cpp)
target_link_libraries(test-tsr-backend-units PRIVATE tsr_lower tsr_ref)
target_compile_features(test-tsr-backend-units PRIVATE cxx_std_17)
target_compile_options(test-tsr-backend-units PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
set_target_properties(test-tsr-backend-units PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME ggml_tesseract_units COMMAND test-tsr-backend-units)
# the M7 builder (tsr-build.cpp: the records of ggml nodes, supports_op M7, fusion,
# the graph key; SW §9.3) is ggml-aware but device-free: the unit tests compile it with ggml-base
# when it exists (fusion, dependencies of views, the preload table, the plan key, supports_op against
# the planners), else they test the ggml-free parts only.
if (TARGET ggml-base)
    target_sources(test-tsr-backend-units PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tsr-build.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tsr-weights.cpp)
    target_include_directories(test-tsr-backend-units PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)
    target_link_libraries(test-tsr-backend-units PRIVATE ggml-base)
    target_compile_definitions(test-tsr-backend-units PRIVATE TSR_UNITS_BUILD=1)
endif()

# KV-cache buffers (tsr-kvbuf: planes, conversion, state save / load) and the mask-summary
# producer (tsr-msum: MSUM, f32 copies, readers, promotion) on a host image of the channels (SW §9.3 M7).
add_executable(test-tsr-kvbuf tests/test-tsr-kvbuf.cpp)
target_link_libraries(test-tsr-kvbuf PRIVATE tsr_lower tsr_ref)
target_compile_features(test-tsr-kvbuf PRIVATE cxx_std_17)
target_compile_options(test-tsr-kvbuf PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
set_target_properties(test-tsr-kvbuf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME ggml_tesseract_kvbuf COMMAND test-tsr-kvbuf)

# Tesseract Pivot's core (tsr-pivot: the machine model, the hoist search, the validator; no ggml, no device):
# hand-built graphs with their expected orders, the model's arithmetic, and random DAGs (valid order, never slower,
# a fixed point, deterministic, the recorded gains replayed), and the time of a graph of 3000 units.
add_executable(test-tsr-pivot tests/test-tsr-pivot.cpp)
target_link_libraries(test-tsr-pivot PRIVATE tsr_lower)
target_compile_features(test-tsr-pivot PRIVATE cxx_std_17)
target_compile_options(test-tsr-pivot PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
set_target_properties(test-tsr-pivot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME ggml_tesseract_pivot COMMAND test-tsr-pivot)

# Tesseract Pivot's ggml adapter (tsr-pivot-graph: units from nodes, the order constraints, the costs, the node validator) on
# shape-only graphs in llama.cpp's node order (Gemma 4 26B-A4B and Qwen3.8 27B layers): needs ggml-base, like the builder tests.
if (TARGET ggml-base)
    add_executable(test-tsr-pivot-graph tests/test-tsr-pivot-graph.cpp tsr-pivot-graph.cpp tsr-build.cpp tsr-weights.cpp)
    target_link_libraries(test-tsr-pivot-graph PRIVATE tsr_lower ggml-base)
    target_include_directories(test-tsr-pivot-graph PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)
    target_compile_features(test-tsr-pivot-graph PRIVATE cxx_std_17)
    target_compile_options(test-tsr-pivot-graph PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-tsr-pivot-graph PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ggml_tesseract_pivot_graph COMMAND test-tsr-pivot-graph)
endif()

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
    # The ctests below are the MUL_MAT end-to-end tests (docs/design/06-software-verification.md
    # §10.2): every MUL_MAT goes through an op list and the firmware's `graph` kernel
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
    # SW-88: a matrix whose T_MAX tokens do not fit one tile of the planner is stored in row blocks (one MUL_MAT record each);
    # the debug log says so, and the cases above (head N = 131072 at T = 4, 1 and a split upload) are bit-exact
    add_test(NAME ggml_tesseract_head_blocks COMMAND sh -c "TSR_DEBUG=1 $<TARGET_FILE:test-ggml-tesseract> 2>&1 | grep -q 'in 2 row blocks of 65536 rows'")
    set_tests_properties(ggml_tesseract_head_blocks PROPERTIES ENVIRONMENT "TSR_SIM_TMAX=4;TSR_SIM_RACE=fatal" TIMEOUT 600)
    # SW-97: the logit soft cap of a head stored in row blocks (tests/test-tsr-headcap.cpp), in the default lowering (deferred blocks, the cap per
    # block slice), with TSR_MM_DEFER=0 and with TSR_HEAD_SPLIT=0: the logits are byte for byte the same, and the records of the launch are
    # the expected ones (tests/check_headcap.py on TSR_DEBUG=2). The lists cut after 5 records (TSR_OPL_SPLIT) cut inside the group. The _fault
    # tests (TSR_TEST_CAP_FAULT: a slice one line off, a slice 16 elements short) must fail the byte comparison.
    add_executable(test-tsr-headcap tests/test-tsr-headcap.cpp)
    target_link_libraries(test-tsr-headcap PRIVATE ggml ggml-base)
    target_compile_features(test-tsr-headcap PRIVATE cxx_std_17)
    target_compile_options(test-tsr-headcap PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-tsr-headcap PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_Interpreter_FOUND)
        set(_hc_dir ${CMAKE_BINARY_DIR}/sw/ggml-tesseract/headcap)
        set(_hc_bin $<TARGET_FILE:test-tsr-headcap>)
        set(_hc_env "TSR_SIM_TMAX=4 TSR_SIM_RACE=fatal TSR_DEVICE=sim")
        # name | extra environment | what the records must be
        add_test(NAME ggml_tesseract_headcap
                 COMMAND sh -c "set -e; mkdir -p ${_hc_dir}; export ${_hc_env}; \
                     TSR_DEBUG=2 ${_hc_bin} ${_hc_dir}/new.bin > ${_hc_dir}/new.log 2>&1; \
                     TSR_MM_DEFER=0 TSR_DEBUG=2 ${_hc_bin} ${_hc_dir}/nodefer.bin > ${_hc_dir}/nodefer.log 2>&1; \
                     TSR_HEAD_SPLIT=0 TSR_DEBUG=2 ${_hc_bin} ${_hc_dir}/whole.bin > ${_hc_dir}/whole.log 2>&1; \
                     TSR_OPL_SPLIT=5 TSR_DEBUG=2 ${_hc_bin} ${_hc_dir}/cut.bin > ${_hc_dir}/cut.log 2>&1; \
                     cmp ${_hc_dir}/new.bin ${_hc_dir}/nodefer.bin; cmp ${_hc_dir}/new.bin ${_hc_dir}/whole.bin; \
                     cmp ${_hc_dir}/new.bin ${_hc_dir}/cut.bin; \
                     rm -rf ${_hc_dir}/plan; mkdir -p ${_hc_dir}/plan; TSR_DUMP_PLAN=${_hc_dir}/plan ${_hc_bin} > /dev/null 2>&1; \
                     ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_headcap.py plan ${_hc_dir}/plan; \
                     rm -rf ${_hc_dir}/dump; mkdir -p ${_hc_dir}/dump; \
                     TSR_DUMP_RE='result_output_pre|cap_scale_in|cap_tanh|result_output' TSR_DUMP_DIR=${_hc_dir}/dump ${_hc_bin} > /dev/null 2> ${_hc_dir}/dump.log; \
                     ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_headcap.py dump ${_hc_dir}/dump.log; \
                     ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_headcap.py new ${_hc_dir}/new.log; \
                     ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_headcap.py nodefer ${_hc_dir}/nodefer.log; \
                     ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_headcap.py whole ${_hc_dir}/whole.log; \
                     ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_headcap.py cut ${_hc_dir}/cut.log")
        set_tests_properties(ggml_tesseract_headcap PROPERTIES TIMEOUT 1800)
        foreach (_f 1 2)
            add_test(NAME ggml_tesseract_headcap_fault${_f}
                     COMMAND sh -c "set -e; export ${_hc_env}; mkdir -p ${_hc_dir}; \
                         if TSR_TEST_CAP_FAULT=${_f} ${_hc_bin} ${_hc_dir}/fault${_f}.bin > ${_hc_dir}/fault${_f}.log 2>&1; then exit 1; fi; \
                         grep -q 'FAIL .*NMSE vs the CPU' ${_hc_dir}/fault${_f}.log; \
                         TSR_MM_DEFER=0 ${_hc_bin} ${_hc_dir}/ref${_f}.bin > /dev/null 2>&1; \
                         ! cmp -s ${_hc_dir}/fault${_f}.bin ${_hc_dir}/ref${_f}.bin")
            set_tests_properties(ggml_tesseract_headcap_fault${_f} PROPERTIES TIMEOUT 1800)
        endforeach()
    endif()
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
    # that force bounce / read-modify-write transfers). These two live here, not in sw/transport,
    # as they test the backend end to end.
    add_test(NAME transport_ggml_e2e_mock COMMAND test-ggml-tesseract)
    set_tests_properties(transport_ggml_e2e_mock PROPERTIES
        ENVIRONMENT "TSR_DEVICE=mock;TSR_MOCK_DMA_ALIGN=64;TSR_MOCK_DMA_CHUNK_KIB=64;TSR_SIM_RACE=fatal")
    add_test(NAME transport_ggml_e2e_mock_tmax8 COMMAND test-ggml-tesseract)
    set_tests_properties(transport_ggml_e2e_mock_tmax8 PROPERTIES ENVIRONMENT
        "TSR_DEVICE=mock;TSR_MOCK_TMAX=8;TSR_MOCK_NUM_CH=3;TSR_MOCK_DMA_ALIGN=4096;TSR_MOCK_DMA_CHUNK_KIB=12;TSR_SIM_RACE=fatal")
    # MoE (SW-60 to SW-65): build_moe_ffn's graph with Gemma 4's router, the expert tensors placed per channel with the
    # expert stride, T = 1 .. 70, against the CPU. Three device configurations: every kind (the routing group is one
    # MOE_ROUTE record, the scale group fused into MUL_MAT_ID), M6's kinds plus MUL_MAT_ID (the CPU selects the experts,
    # the device runs the matrix products) and every kind but MOE_ROUTE.
    add_executable(test-tsr-moe tests/test-tsr-moe.cpp)
    target_link_libraries(test-tsr-moe PRIVATE ggml ggml-base)
    target_compile_features(test-tsr-moe PRIVATE cxx_std_17)
    target_compile_options(test-tsr-moe PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-tsr-moe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ggml_tesseract_moe COMMAND test-tsr-moe)
    set_tests_properties(ggml_tesseract_moe PROPERTIES ENVIRONMENT "TSR_SIM_RACE=fatal;TSR_TEST_ROUTE=1;TSR_TEST_MMID=1")
    add_test(NAME ggml_tesseract_moe_tmax8 COMMAND test-tsr-moe)
    set_tests_properties(ggml_tesseract_moe_tmax8 PROPERTIES ENVIRONMENT
        "TSR_SIM_TMAX=8;TSR_SIM_RACE=fatal;TSR_TEST_ROUTE=1;TSR_TEST_MMID=1")
    add_test(NAME ggml_tesseract_moe_m6 COMMAND test-tsr-moe)
    set_tests_properties(ggml_tesseract_moe_m6 PROPERTIES ENVIRONMENT
        "TSR_KINDS=m6,+MUL_MAT_ID;TSR_SIM_RACE=fatal;TSR_TEST_ROUTE=0;TSR_TEST_MMID=1")
    add_test(NAME ggml_tesseract_moe_host_route COMMAND test-tsr-moe)
    set_tests_properties(ggml_tesseract_moe_host_route PROPERTIES ENVIRONMENT
        "TSR_KINDS=all,-MOE_ROUTE;TSR_SIM_RACE=fatal;TSR_TEST_ROUTE=0;TSR_TEST_MMID=1")

    # these e2e cases are M6's (MUL_MAT only: "ADD must be rejected"; the IO region
    # filled to force host memory and the staged path): they run with the device restricted to M6's
    # kinds (TSR_KINDS=m6, SW-14) and IO-only compute windows (TSR_COMPUTE_WINDOWS=io). M7 is
    # covered by ggml_tesseract_units, _opl_launch, _attn and the tiny-model graph tests.
    foreach (_tsr_t ggml_tesseract_e2e ggml_tesseract_e2e_tmax8 ggml_tesseract_e2e_hosted ggml_tesseract_e2e_hosted_tmax8
                    transport_ggml_e2e_mock transport_ggml_e2e_mock_tmax8)
        set_property(TEST ${_tsr_t} APPEND PROPERTY ENVIRONMENT "TSR_KINDS=m6" "TSR_COMPUTE_WINDOWS=io")
    endforeach()

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

    # SW §6.9 versions: the description and the versions hook on tsr-fw.elf (full); a subset image
    # (fw_exec_ref: NOP, JOIN, MUL_MAT) and TSR_KINDS=m6 route MUL_MAT only, the ADD of two MUL_MAT
    # results runs on the CPU (subset, m6; on the device with the full image); an image older than this
    # backend's Tesseract Basis (fw_basis_old, 0.0.5) is not opened (old). The expected numbers are
    # derived from tsr/version.h, tsr_tables.h and tsr/opl.hpp.
    add_executable(test-tsr-versions tests/test-tsr-versions.cpp)
    target_link_libraries(test-tsr-versions PRIVATE ggml ggml-base ggml-cpu)
    target_include_directories(test-tsr-versions PRIVATE ${TESSERACT_ROOT}/sw/transport ${TESSERACT_ROOT}/sw/sim/include
                               ${TESSERACT_ROOT}/sw/ctrl/include)
    target_compile_features(test-tsr-versions PRIVATE cxx_std_17)
    target_compile_options(test-tsr-versions PRIVATE -Wall -Wextra)
    set_target_properties(test-tsr-versions PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ggml_tesseract_versions_full COMMAND test-tsr-versions full)
    add_test(NAME ggml_tesseract_versions_no_local_scale COMMAND test-tsr-versions full)
    set_tests_properties(ggml_tesseract_versions_no_local_scale PROPERTIES ENVIRONMENT
                         "TSR_DEVICE=sim;TSR_SIM_MVM_LOCAL_SCALE=0;TSR_SIM_RACE=fatal")
    add_test(NAME ggml_tesseract_versions_no_local_softcap COMMAND test-tsr-versions full)
    set_tests_properties(ggml_tesseract_versions_no_local_softcap PROPERTIES ENVIRONMENT
                         "TSR_DEVICE=sim;TSR_SIM_MVM_LOCAL_SOFTCAP=0;TSR_SIM_RACE=fatal")
    add_test(NAME ggml_tesseract_versions_subset COMMAND test-tsr-versions subset ${TSR_FW_TEST_ELF_fw_exec_ref})
    add_test(NAME ggml_tesseract_versions_m6 COMMAND test-tsr-versions m6)
    add_test(NAME ggml_tesseract_versions_old COMMAND test-tsr-versions old ${TSR_FW_TEST_ELF_fw_basis_old})
    set_tests_properties(ggml_tesseract_versions_full ggml_tesseract_versions_subset ggml_tesseract_versions_m6
                         ggml_tesseract_versions_old PROPERTIES TIMEOUT 120 ENVIRONMENT "TSR_SIM_RACE=fatal")

    # SW §10.2 ggml_tesseract_attn: KV buffers (planes, conversion, state save / load), mask
    # summaries and f32 copies, and SET_ROWS + FLASH_ATTN_EXT on the device against the CPU backend
    # (NMSE) and tsr_ref_fa (bit for bit), D 64 / 128 / 256, GQA, decode and prefill, causal / SWA /
    # wrapped ring / GENERAL masks; _tmax8: T_MAX 8 (query tiles of several tokens, head subsets).
    add_executable(test-tsr-attn tests/test-tsr-attn.cpp)
    target_include_directories(test-tsr-attn PRIVATE
        ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/src)          # ggml-quants.h (quantize_row_q8_0_ref)
    target_link_libraries(test-tsr-attn PRIVATE ggml ggml-base ggml-cpu tsr_ref)
    target_compile_features(test-tsr-attn PRIVATE cxx_std_17)
    target_compile_options(test-tsr-attn PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra)
    set_target_properties(test-tsr-attn PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ggml_tesseract_attn COMMAND test-tsr-attn)
    set_tests_properties(ggml_tesseract_attn PROPERTIES ENVIRONMENT "TSR_SIM_RACE=fatal")
    add_test(NAME ggml_tesseract_attn_tmax8 COMMAND test-tsr-attn)
    set_tests_properties(ggml_tesseract_attn_tmax8 PROPERTIES ENVIRONMENT "TSR_SIM_TMAX=8;TSR_SIM_RACE=fatal")
    # every record in its own op list, so FA_NEW crosses a cut (SW-43)
    add_test(NAME ggml_tesseract_attn_tmax4 COMMAND test-tsr-attn)
    set_tests_properties(ggml_tesseract_attn_tmax4 PROPERTIES ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_TMAX=4;TSR_SIM_RACE=fatal")
    add_test(NAME ggml_tesseract_attn_split1 COMMAND test-tsr-attn)
    set_tests_properties(ggml_tesseract_attn_split1 PROPERTIES ENVIRONMENT "TSR_OPL_SPLIT=1;TSR_SIM_RACE=fatal")

    # test-backend-ops: Tesseract0 vs the CPU backend (spec "Verification strategy", op level), every MUL_MAT
    # case, tolerances untouched. One case is left out here and covered by
    # ggml_tesseract_scalar_nmse instead: MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048) has a
    # single output element, so its NMSE is the squared relative error of one dot product of
    # fresh random data; with the spec's 8-bit weights and activations it exceeds 5e-4 in ~15%
    # of runs on a correct device (measured: 299/2000 trials, the same count as an exact dot
    # product of the 8-bit operands). The full, unfiltered command is in README.md.
    # the case MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32) (2.1 M tokens, staged in
    # 16 launches of 131,072 single-token tiles at the sim's T_MAX 1: 8.4 M tile ops through the
    # interpreted firmware, ~50 s, and over 15 min with the race detector) runs in its own ctest
    # labeled slow, without the race detector; the two ctests together run the whole case set.
    if (TARGET test-backend-ops)
        add_test(NAME ggml_tesseract_backend_ops
                 COMMAND test-backend-ops test -o MUL_MAT -b Tesseract0
                         -p "^(?!type_a=f32,type_b=f32,m=1,n=1,k=2048,|type_a=f16,type_b=f32,m=16,n=2105344,k=32,)")
        set_tests_properties(ggml_tesseract_backend_ops PROPERTIES ENVIRONMENT "TSR_SIM_RACE=fatal")
        add_test(NAME ggml_tesseract_backend_ops_bigt
                 COMMAND test-backend-ops test -o MUL_MAT -b Tesseract0 -p "^type_a=f16,type_b=f32,m=16,n=2105344,k=32,")
        set_tests_properties(ggml_tesseract_backend_ops_bigt PROPERTIES LABELS slow ENVIRONMENT "TSR_SIM_RACE=off")
        # MoE (SW-62): the whole MUL_MAT_ID case set of the pin (945 cases: 928 supported and compared with the CPU, 17
        # refused: m not a multiple of 16, k not a multiple of the block, iq2_xxs).
        add_test(NAME ggml_tesseract_backend_ops_mul_mat_id
                 COMMAND test-backend-ops test -o MUL_MAT_ID -b Tesseract0)
        set_tests_properties(ggml_tesseract_backend_ops_mul_mat_id PROPERTIES TIMEOUT 1200 ENVIRONMENT "TSR_SIM_RACE=fatal")
    endif()
endif()

# llama.cpp's own tests that load models with default offload settings (no --no-kv-offload):
# with the Tesseract device registered (GPU type) llama.cpp puts the KV cache in a Tesseract
# buffer, and M6 runs no KV ops (SET_ROWS, attention), so the scheduler aborts — the documented
# M6 constraint (spec "lowering constraints", docs/llama-pin.md: run llama.cpp with -nkvo). These
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

# Tiny-model graph tests (ggml_tesseract_graph_*_tiny) and the M7 test-backend-ops
# coverage (ggml_tesseract_backend_ops_<op>); the generator, the drivers and the rationale are in tests/.
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny.cmake)

# llama.cpp's state save / load with the KV cache in device KV planes (SW §9.3 "KV buffers",
# AT §4.6) on the tiny Qwen2 model: device -> device bit-identical logits, device -> CPU state bytes,
# sequence state into a cleared context (tests/test-tsr-kvstate.cpp).
if (TARGET llama AND TARGET tsr_tiny_models AND TARGET ggml-tesseract)
    add_executable(test-tsr-kvstate tests/test-tsr-kvstate.cpp)
    target_link_libraries(test-tsr-kvstate PRIVATE llama ggml)
    target_compile_features(test-tsr-kvstate PRIVATE cxx_std_17)
    target_compile_options(test-tsr-kvstate PRIVATE -Wall -Wextra)
    set_target_properties(test-tsr-kvstate PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME ggml_tesseract_kvstate COMMAND test-tsr-kvstate ${TSR_TINY_MODEL_DIR}/qwen2-tiny-q8_0.gguf)
    set_tests_properties(ggml_tesseract_kvstate PROPERTIES TIMEOUT 300
        ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;LLAMA_ATTN_ROT_DISABLE=1")
endif()

# Gated DeltaNet logits, device against CPU, short prompts and single-token steps, the tiny Qwen3.5-style models
# (tests/test-tsr-gdn-logits.cpp, SW §7.10). The second model has the real models' S_v = 128.
if (TARGET llama AND TARGET tsr_tiny_models AND TARGET ggml-tesseract)
    add_executable(test-tsr-gdn-logits tests/test-tsr-gdn-logits.cpp)
    target_link_libraries(test-tsr-gdn-logits PRIVATE llama ggml)
    target_compile_features(test-tsr-gdn-logits PRIVATE cxx_std_17)
    target_compile_options(test-tsr-gdn-logits PRIVATE -Wall -Wextra)
    target_include_directories(test-tsr-gdn-logits PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    set_target_properties(test-tsr-gdn-logits PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    foreach (_m qwen35-tiny-q8_0 qwen35-tiny-s128-q8_0)
        add_test(NAME ggml_tesseract_gdn_logits_${_m} COMMAND test-tsr-gdn-logits ${TSR_TINY_MODEL_DIR}/${_m}.gguf 3)
        set_tests_properties(ggml_tesseract_gdn_logits_${_m} PROPERTIES TIMEOUT 300
            ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;LLAMA_ATTN_ROT_DISABLE=1")
    endforeach()
endif()

# The verification batch of multi-token prediction (tests/test-tsr-mtp.cpp, SW §7.11, SW-83): a context with rollback snapshots
# decodes batches of up to K + 1 tokens with every row's logits and rolls the rejected tail back; the rows are compared with plain
# decoding on the same device and with the CPU. The models are the tiny Qwen3.5-style ones (S_v 32: the composed Gated DeltaNet
# path, S_v 128: the fused path on the simulator's CAPS); the second run is the card configuration (T_MAX 4, no GDN op).
if (TARGET llama AND TARGET tsr_tiny_models AND TARGET ggml-tesseract)
    add_executable(test-tsr-mtp tests/test-tsr-mtp.cpp)
    target_link_libraries(test-tsr-mtp PRIVATE llama ggml)
    target_compile_features(test-tsr-mtp PRIVATE cxx_std_17)
    target_compile_options(test-tsr-mtp PRIVATE -Wall -Wextra)
    target_include_directories(test-tsr-mtp PRIVATE ${TESSERACT_ROOT}/third_party/llama.cpp/ggml/include)
    set_target_properties(test-tsr-mtp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    foreach (_m qwen35-tiny-mtp-q8_0 qwen35-tiny-s128-mtp-q8_0)
        add_test(NAME ggml_tesseract_mtp_${_m} COMMAND test-tsr-mtp ${TSR_TINY_MODEL_DIR}/${_m}.gguf 3 12)
        set_tests_properties(ggml_tesseract_mtp_${_m} PROPERTIES TIMEOUT 600
            ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_TMAX=4;LLAMA_ATTN_ROT_DISABLE=1")
        # without the state group (SW-82, TSR_GDN_CELL=0): the chain writes the result tensor's slots and generic CPY records move them
        add_test(NAME ggml_tesseract_mtp_nocell_${_m} COMMAND test-tsr-mtp ${TSR_TINY_MODEL_DIR}/${_m}.gguf 3 8)
        set_tests_properties(ggml_tesseract_mtp_nocell_${_m} PROPERTIES TIMEOUT 600
            ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_TMAX=4;TSR_GDN_CELL=0;LLAMA_ATTN_ROT_DISABLE=1")
        add_test(NAME ggml_tesseract_mtp_card_${_m} COMMAND test-tsr-mtp ${TSR_TINY_MODEL_DIR}/${_m}.gguf 2 9)
        set_tests_properties(ggml_tesseract_mtp_card_${_m} PROPERTIES TIMEOUT 600
            ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_CARD=k4full;LLAMA_ATTN_ROT_DISABLE=1")
    endforeach()
    # a target without recurrent layers (tiny Gemma 4): no snapshots, the rejected tail is a KV removal; tsr-mtp accepts n_rs_seq 0
    add_test(NAME ggml_tesseract_mtp_gemma4-tiny-q8_0 COMMAND test-tsr-mtp ${TSR_TINY_MODEL_DIR}/gemma4-tiny-q8_0.gguf 3 12)
    set_tests_properties(ggml_tesseract_mtp_gemma4-tiny-q8_0 PROPERTIES TIMEOUT 600
        ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_TMAX=4;LLAMA_ATTN_ROT_DISABLE=1")
    add_test(NAME ggml_tesseract_mtp_card_gemma4-tiny-q8_0 COMMAND test-tsr-mtp ${TSR_TINY_MODEL_DIR}/gemma4-tiny-q8_0.gguf 2 9)
    set_tests_properties(ggml_tesseract_mtp_card_gemma4-tiny-q8_0 PROPERTIES TIMEOUT 600
        ENVIRONMENT "TSR_DEVICE=sim;TSR_SIM_RACE=fatal;TSR_SIM_CARD=k4full;LLAMA_ATTN_ROT_DISABLE=1")
endif()

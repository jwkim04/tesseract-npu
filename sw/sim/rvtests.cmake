# sw/sim/rvtests.cmake — RV32 conformance harness (P1.11): riscv-tests, riscv-arch-test and random
# lockstep programs on Spike and on tsr_rv's test bus profile (docs/v2/06-software-verification.md
# §4.2, §9.6, §10.1-§10.2; README risk 14). Scripts: sw/sim/tests/rvtests/ (stdlib Python; the
# Spike profiles and the tsr_rv runner contract are documented in rvcommon.py).
#
# Include from sw/sim/CMakeLists.txt (any position; it defines tests only, no targets):
#     include(${CMAKE_CURRENT_SOURCE_DIR}/rvtests.cmake)
#
# Tools: sw/ctrl/tools/install-toolchain.sh installs them without sudo under build/tools/riscv/
# with the pins of docs/v2/toolchain.lock (xPack riscv-none-elf-gcc 14.2.0-3, dtc, Spike, the
# riscv-tests and riscv-arch-test sources). Cache variables (defaults follow that layout):
#   TSR_RISCV_TOOLS     install root (default ${TESSERACT_ROOT}/build/tools/riscv)
#   TSR_RISCV_PREFIX    compiler prefix (default <root>/xpack-riscv-none-elf-gcc-14.2.0-3/bin/riscv-none-elf-)
#   TSR_SPIKE           spike (default <root>/spike/bin/spike)
#   TSR_RV_RUNNER       the tsr_rv runner: a CMake target name (default tsr_rv_run, P1.7; the target
#                       may be defined before or after this file) or an executable path
#   TSR_RV_RUNNER_ARGS  arguments placed before the harness's own (e.g. a profile switch)
#   TSR_RVTESTS_PYTHON  Python >= 3.8, stdlib only (default: the python3 CMake finds)
# The tools are looked up when a test runs, not at configure time: a test whose tools, sources or
# runner are missing reports "Skipped" (exit 77) with the install hint, so installing the tools
# later needs no reconfigure.
#
# Tests (run: ctest --test-dir build/dev -R '^rv_' --output-on-failure):
#   rv_harness_selftest  commit-log parser and comparator, arch-test case selection, generator
#                        determinism and register discipline; assembles and links 4 random
#                        programs when the compiler is installed (Spike not needed)
#   rv_spike_upstream    riscv-tests rv32ui/um/uc/mi -p exactly as published (env/p) on Spike
#                        profile "upstream": the sanity check of the installed compiler and Spike
#   rv_spike_profile     riscv-tests in the profile env + riscv-arch-test 3.10.0 RV32IMC/Zicsr/
#                        Zifencei/privilege on Spike profile "test" (the oracle configuration)
#   rv_spike_random      20 random programs on Spike profile "test" (each must end through tohost)
#   rv_isa               the self-checking riscv-tests pass on tsr_rv, and on Spike when it is
#                        installed (--spike-optional: without Spike the Spike runs are reported as
#                        not run; tsr_rv's result is complete either way, the tests check
#                        themselves); skipped only without the compiler, sources or runner
#   rv_arch              riscv-arch-test signatures of tsr_rv equal to Spike's (SW §10.2); the
#                        suite has no reference signatures, so it is skipped without Spike
#   rv_lockstep          the riscv-tests on Spike and tsr_rv, commit logs compared retire by retire
#                        up to the store to tohost (rvtests.py --lockstep; the two tests that read
#                        implementation-defined CSRs are left out); skipped without Spike
#   rv_random            50 random programs (1500 units each), tsr_rv vs Spike retire by retire up
#                        to the store to tohost, then the signatures (SW §10.2)
#   rv_random_long       LABELS slow: 2000 programs; runs only with TSR_RV_RANDOM_LONG=1, otherwise
#                        "Skipped": TSR_RV_RANDOM_LONG=1 ctest --test-dir build/dev -R rv_random_long
# Each test builds into its own directory under ${CMAKE_CURRENT_BINARY_DIR}/rvtests/.

set(TSR_RISCV_TOOLS "${TESSERACT_ROOT}/build/tools/riscv" CACHE PATH
    "Install root of sw/ctrl/tools/install-toolchain.sh")
set(TSR_RISCV_PREFIX "" CACHE STRING
    "RISC-V compiler prefix (default: the pinned xPack under TSR_RISCV_TOOLS)")
set(TSR_SPIKE "" CACHE FILEPATH "Spike executable (default: TSR_RISCV_TOOLS/spike/bin/spike)")
set(TSR_RV_RUNNER "tsr_rv_run" CACHE STRING
    "tsr_rv runner for rv_isa / rv_random: target name or executable path")
set(TSR_RV_RUNNER_ARGS "" CACHE STRING "Extra tsr_rv runner arguments (before the harness's)")
set(TSR_RVTESTS_PYTHON "" CACHE FILEPATH "Python 3 for sw/sim/tests/rvtests (stdlib only)")

set(_rv_py "${TSR_RVTESTS_PYTHON}")
if (NOT _rv_py)
    find_package(Python3 COMPONENTS Interpreter)
    set(_rv_py "${Python3_EXECUTABLE}")
endif()

if (NOT _rv_py)
    message(WARNING "rv_* tests: no Python 3 found; set -DTSR_RVTESTS_PYTHON=<python3>")
else()
    set(_rv_dir ${CMAKE_CURRENT_LIST_DIR}/tests/rvtests)
    set(_rv_out ${CMAKE_CURRENT_BINARY_DIR}/rvtests)
    set(_rv_tools --tools-prefix ${TSR_RISCV_TOOLS})
    if (TSR_RISCV_PREFIX)
        list(APPEND _rv_tools --gcc-prefix ${TSR_RISCV_PREFIX})
    endif()
    if (TSR_SPIKE)
        list(APPEND _rv_tools --spike ${TSR_SPIKE})
    endif()
    # The runner as one argument: the target's file when the target exists (evaluated at generate
    # time, so the definition order does not matter), else the path; empty -> rv_isa / rv_random
    # report "Skipped".
    if (TSR_RV_RUNNER MATCHES "/")
        set(_rv_runner "${TSR_RV_RUNNER}")
    else()
        set(_rv_runner "$<$<TARGET_EXISTS:${TSR_RV_RUNNER}>:$<TARGET_FILE:${TSR_RV_RUNNER}>>")
    endif()
    if (TSR_RV_RUNNER_ARGS)
        string(REPLACE ";" " " _rv_rargs "${TSR_RV_RUNNER_ARGS}")
        set(_rv_runner "${_rv_runner}$<$<BOOL:${_rv_runner}>: ${_rv_rargs}>")
    endif()

    set(_rv_tests ${_rv_py} ${_rv_dir}/rvtests.py)
    set(_rv_random ${_rv_py} ${_rv_dir}/rv_random.py)

    add_test(NAME rv_harness_selftest
             COMMAND ${_rv_random} selftest ${_rv_tools} --out ${_rv_out}/selftest)
    add_test(NAME rv_spike_upstream
             COMMAND ${_rv_tests} riscv-tests --env upstream ${_rv_tools} --out ${_rv_out}/upstream)
    add_test(NAME rv_spike_profile
             COMMAND ${_rv_tests} all --env tsr ${_rv_tools} --out ${_rv_out}/spike_profile)
    add_test(NAME rv_spike_random
             COMMAND ${_rv_random} run ${_rv_tools} --out ${_rv_out}/spike_random --count 20)
    add_test(NAME rv_isa
             COMMAND ${_rv_tests} riscv-tests --env tsr ${_rv_tools} --out ${_rv_out}/isa --spike-optional
                     "--runner=${_rv_runner}")
    add_test(NAME rv_arch
             COMMAND ${_rv_tests} arch-test --env tsr ${_rv_tools} --out ${_rv_out}/arch
                     "--runner=${_rv_runner}")
    add_test(NAME rv_lockstep
             COMMAND ${_rv_tests} riscv-tests --env tsr ${_rv_tools} --out ${_rv_out}/lockstep
                     --lockstep "--runner=${_rv_runner}")
    add_test(NAME rv_random
             COMMAND ${_rv_random} run ${_rv_tools} --out ${_rv_out}/random --count 50
                     "--runner=${_rv_runner}")
    add_test(NAME rv_random_long
             COMMAND ${_rv_random} run ${_rv_tools} --out ${_rv_out}/random_long --seed 1000
                     --count 2000 --require-env TSR_RV_RANDOM_LONG "--runner=${_rv_runner}")
    set_tests_properties(rv_harness_selftest rv_spike_upstream rv_spike_profile rv_spike_random
                         rv_isa rv_arch rv_lockstep rv_random PROPERTIES
                         SKIP_RETURN_CODE 77 PROCESSORS 4 TIMEOUT 300 LABELS rv)
    set_tests_properties(rv_random_long PROPERTIES
                         SKIP_RETURN_CODE 77 PROCESSORS 4 TIMEOUT 7200 LABELS "rv;slow")
endif()

#!/usr/bin/env bash
# tb/run_all.sh [--no-lint] [pytest args...] — RTL lint (rtl/lint.sh, which also runs the control
# subsystem's, the Xilinx technology, XRT platform and M0 bandwidth-kernel lint scripts), then every
# testbench under tb/:
#   block benches   common, dma, fp, qact, spm (2R2W banks, client adapter, RISC-V bridge, the
#                   six-client system bench); M3: mvmfront, mvmback, mvm
#   issue path      chk (tsr_chk against the simulator's checker), issue (tsr_issue: random programs
#                   and the trace sets, against the simulator's queue model), csr (tsr_csr + tsr_issue
#                   against the simulator's host port and control window)
#   control core    ctrl (tsr_ctrl: Ibex + debug module + TCMs + control window + CDC; lockstep
#                   against the simulator's tsr_rv, DMI against tsr_dm.c, directed CDC tests)
#   integration     core (tsr_core: the firmware tb/core/fw on the RTL Ibex pushes a MUL_MAT-shaped
#                   op stream through the issue path into the units; DRAM / SPM / LCQ / log / CSR
#                   state against the simulator running the same firmware)
#   W4-B units      vu (the vector unit: the arithmetic cores and lanes against libtsr_ref, the unit against
#                   the simulator's VU / AT execution, the checker class), at (KVST, the ATTQK / ATTPV cluster
#                   modes and the gather path, the AT checker class, the AT §6.6 flash-attention sequence against
#                   tsr_ref_fa), dbg (the DB block: DBG registers, stops, history, trace writer, records,
#                   against the simulator's DBG and DM models)
#   other           tech (Xilinx technology wrappers against the generic ones, the MAC lane's PV mode),
#                   fastreplay (the C++ memory model self-test), platform (the adapter lint and bench against
#                   the mock adapter model, the ap_clk_2 / MEM_EN kernel smoke, hw/u200/build_xclbin.sh: k1 /
#                   k4 disabled until Y5, platform lookup), platform/bw (the M0 bandwidth kernel)
# core also runs, besides the W4-A smoke, the W4-B integration smokes: core_m7 (the production firmware image on
# the RTL Ibex executes an op list of RMS_NORM, MUL_MAT, ROPE, SET_ROWS_KV, FLASH_ATTN and GLU records through
# the issue path into the VU, the attention modes of the MVM unit and KVST on the DMA: a decode token and a
# prefill tile of a 128-wide decoder layer; the reference is the simulator's run, dumped by
# build/dev/sw/ctrl/host/test_fw_m7_layer: `cmake --build build/dev --target test_fw_m7_layer tsr_fw`) and
# core_db (the DB block in the core: DBG registers, the DMI gate, history, trace writer, tile breakpoints).
# The trace-replay and full-system benches of SW §10.3 beyond tb/core (tb/opreplay, tb/platform
# kernel bench, tb/platform/vkernel) come with W5.
# Most benches compare with the simulator through build/dev/sw/sim/libtsr_sim.so (CMake target
# tsr_sim_shared; TSR_SIM_LIB overrides it); tb/core and tb/ctrl also need the xPack RISC-V GCC under
# build/tools/riscv (docs/design/toolchain.lock).
# Slow tests (marker `slow`) are opt-in. tb/pytest.ini does not deselect the marker; each slow test
# skips itself unless it was asked for, by tb/fp/test_fp.py's rule (slow_requested): an -m
# expression naming `slow` (all of them), -k with the test's name, its node id, or its directory's
# switch: TSR_FP_SLOW=1 (tb/fp: the full sweeps; the Vivado OOC runs also need TSR_VIVADO),
# TSR_PLATFORM_SLOW=1 (tb/platform/bw test_bw_mutants: the 57 mutants of mutate_bw.py, 235 s;
# 2026-10-24, cold builds, pass when opted in), TSR_ISSUE_MUTANTS=1 (tb/issue test_issue_mutants:
# 31 mutants of tsr_chk / tsr_issue / tsr_csr), TSR_CHK_VIVADO=1 (tb/chk test_chk_netlist: the Vivado
# netlist of tsr_chk against the simulator, 1 min), TSR_CORE_MUTANTS=1 (tb/core test_core_mutants: 8
# wiring mutants of tsr_core / tsr_core_dp; test_core_w4b_mutants: 11 wiring mutants of the VU / AT / DB
# integration, about 5 min each), TSR_VU_MUTANTS=1 (tb/vu: 36 mutants), TSR_DBG_MUTANTS=1 (tb/dbg: 12),
# TSR_ADAPTER_MUTANTS=1 (tb/platform/test_adapter.py: 6).
#   TSR_PLATFORM_SLOW=1 tb/run_all.sh --no-lint platform/bw -k mutants     the bw mutant test
# Opt-in campaigns (not run here, docs/rtl-m3.md §13, docs/platform-u200.md): tb/mvmback/mutate_back.py,
# tb/spm/mutate_spm.py (22 mutants of the SPM / bridge / DMA reservation), tb/ctrl/mutate_ctrl.py (12
# mutants of rtl/ctrl), Vivado netlist simulation of the technology wrappers (TSR_TECH_VIVADO=1).
#   tb/run_all.sh                       lint + all tests: lint all clean (397 runs: 376 Verilator lints, 9 XSim
#                                       static elaborations, 8 Vivado syntheses, the generated-file checks;
#                                       20 min with TSR_VIVADO_WRAP); 232 passed, 111 opt-in skipped, 0 failed,
#                                       47 min wall with TSR_BUILD_JOBS=8 and the Verilator builds warm (the
#                                       cold run took 64 min) (2026-11-01, W4-B integration)
#   tb/run_all.sh -k dma                lint + tests matching "dma"
#   tb/run_all.sh --no-lint common      only tb/common (self-test)
#   tb/run_all.sh mvm --tsr-seed 7      only the MVM unit tests, another seed
# Builds use make -j${TSR_BUILD_JOBS:-4}; build dirs under ${TSR_SIM_BUILD_ROOT:-build/cocotb}.
# The Python with cocotb and numpy: ${TSR_PYTHON:-python3}.
# Opt-in extras (not run here): fp slow sweeps (-m slow or TSR_FP_SLOW=1), the bw mutant test
# (-m slow or TSR_PLATFORM_SLOW=1), Vivado out-of-context runs (tb/qact TSR_QACT_VIVADO=1,
# tb/core/viv/core_ooc.sh (also the clock-domain-crossing gate: report_cdc, report_clock_interaction),
# tb/platform/viv/kernel_ooc.tcl, tb/platform/viv/xo_cdc.tcl (the packaged kernel IP carries the CDC
# constraints: create_ip + synth_ip of an .xo, then the same gate), tb/spm/viv/spm_ooc.tcl,
# rtl/ctrl/viv/ctrl_ooc.sh for gate T1).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" = "--no-lint" ]; then
    shift
else
    "$ROOT/rtl/lint.sh" || { echo "run_all.sh: lint failed"; exit 1; }
fi
cd "$ROOT/tb" || exit 1
exec "${TSR_PYTHON:-python3}" -m pytest "$@"

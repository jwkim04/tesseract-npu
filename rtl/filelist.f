// rtl/filelist.f — generic (vendor-neutral) target: simulation (Verilator/cocotb) and lint.
// Paths are relative to this file's directory: use `verilator -F rtl/filelist.f`
// (tb/common/runner.py reads the same list). Order: packages first, then tech, then core.
// A target filelist (e.g. rtl/filelist_u200.f, M4) swaps tech/generic for tech/xilinx.
ctrl/waivers/vendor.vlt
core/tsr_tables_pkg.sv
core/tsr_pkg.sv
core/tsr_fp_pkg.sv
tech/generic/tsr_sram_1r1w.sv
tech/generic/tsr_mac_lane.sv
// the RISC-V control subsystem and its vendored IP (rtl/ctrl, CC §2-§5, §11; W4-A): after the
// tables package and the SRAM wrapper, which it uses
-F ctrl/filelist.f
core/tsr_fp_mul.sv
core/tsr_fp_div.sv
core/tsr_fp_add.sv
core/tsr_spm.sv
// the single-port SPM client adapter and the RISC-V SPM bridge (CC §10.2, §10.5)
core/tsr_spm_cport.sv
core/tsr_spm_bridge.sv
core/tsr_mem_arb.sv
core/tsr_vpipe.sv
core/tsr_rpipe.sv
core/tsr_dma.sv
core/tsr_qact.sv
core/tsr_mvm_ws.sv
core/tsr_mvm_front.sv
core/tsr_mvm_actbuf.sv
core/tsr_mvm_back.sv
core/tsr_mvm_cluster.sv
core/tsr_mvm_gather.sv
core/tsr_mvm.sv
core/tsr_csr.sv
// the dispatch checker and the issue path (CC §8-§9; tsr_issue instantiates tsr_chk)
core/tsr_chk.sv
core/tsr_issue.sv
// the core-clock datapath and the core (CC §13.2)
core/tsr_core_dp.sv
core/tsr_core.sv

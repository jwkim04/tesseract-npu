// rtl/filelist_xilinx.f — Xilinx UltraScale+ target (Alveo U200): rtl/filelist.f with the
// technology wrappers of rtl/tech/generic replaced by rtl/tech/xilinx (inferred URAM / BRAM /
// LUTRAM by size, DSP48E2-packed MAC lanes; no unisim primitives, so Verilator simulates it too).
// Paths are relative to this file's directory, order as rtl/filelist.f: packages, tech, core.
// Keep the core list identical to rtl/filelist.f (rtl/tech/xilinx/lint.sh checks it).
ctrl/waivers/vendor.vlt
core/tsr_tables_pkg.sv
core/tsr_pkg.sv
core/tsr_fp_pkg.sv
tech/xilinx/tsr_sram_1r1w.sv
tech/xilinx/tsr_mac_lane.sv
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

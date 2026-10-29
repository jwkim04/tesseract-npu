// rtl/filelist_xilinx.f — Xilinx UltraScale+ target (Alveo U200): rtl/filelist.f with the
// technology wrappers of rtl/tech/generic replaced by rtl/tech/xilinx (inferred URAM / BRAM /
// LUTRAM by size, DSP48E2-packed MAC lanes; no unisim primitives, so Verilator simulates it too).
// Paths are relative to this file's directory, order as rtl/filelist.f: packages, tech, core.
// Keep the core list identical to rtl/filelist.f (rtl/tech/xilinx/lint.sh checks it).
core/tsr_pkg.sv
core/tsr_fp_pkg.sv
tech/xilinx/tsr_sram_1r1w.sv
tech/xilinx/tsr_mac_lane.sv
core/tsr_fp_mul.sv
core/tsr_fp_div.sv
core/tsr_fp_add.sv
core/tsr_spm.sv
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
core/tsr_core.sv

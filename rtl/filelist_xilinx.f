// rtl/filelist_xilinx.f — Xilinx UltraScale+ target (Alveo U200): rtl/filelist.f with the
// technology wrappers of rtl/tech/generic replaced by rtl/tech/xilinx (inferred URAM / BRAM /
// LUTRAM by size, DSP48E2-packed MAC lanes; no unisim primitives, so Verilator simulates it too).
// Paths are relative to this file's directory, order as rtl/filelist.f: packages, tech, core.
// Keep the core list identical to rtl/filelist.f (rtl/tech/xilinx/lint.sh checks it).
ctrl/waivers/vendor.vlt
core/tsr_tables_pkg.sv
core/tsr_pkg.sv
core/tsr_fp_pkg.sv
// W4-B at: area AT's payloads and helpers (after tsr_fp_pkg, which it imports)
core/tsr_att_pkg.sv
// W4-B vu: the external checker-class contract and the vector unit's package (VU §7, CC §8.2)
core/tsr_xc_pkg.sv
core/tsr_vu_pkg.sv
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
// W4-B at: the DMA's KVST engine (AT §8.3), instantiated by tsr_dma
core/tsr_dma_kvst.sv
core/tsr_dma.sv
core/tsr_qact.sv
core/tsr_mvm_ws.sv
core/tsr_mvm_front.sv
core/tsr_mvm_actbuf.sv
core/tsr_mvm_back.sv
// W4-B at: the attention units of a cluster (AT §8.1 score assembler and row-vector emitter, §8.2 PV)
core/tsr_mvm_att.sv
core/tsr_mvm_pv.sv
core/tsr_mvm_cluster.sv
core/tsr_mvm_gather.sv
core/tsr_mvm.sv
// W4-B vu: the vector unit (unit 3, VU §7) and its blocks
core/tsr_vu_dly.sv
core/tsr_vu_exp.sv
core/tsr_vu_rcp.sv
core/tsr_vu_sqrt.sv
core/tsr_vu_sincos.sv
core/tsr_vu_lane.sv
core/tsr_vu_gen.sv
core/tsr_vu_rd.sv
core/tsr_vu_dq.sv
core/tsr_vu_inj.sv
core/tsr_vu_wr.sv
core/tsr_vu_red.sv
core/tsr_vu_angle.sv
core/tsr_vu_topk.sv
core/tsr_vu.sv
core/tsr_csr.sv
// the dispatch checker and the issue path (CC §8-§9; tsr_issue instantiates tsr_chk)
core/tsr_chk.sv
// W4-B at: the AT checker classes of tsr_chk's external-class port (CC §8.2, AT §5.8)
core/tsr_chk_at.sv
// W4-B vu: the VU checker class and the xc mux of the VU / AT classes (CC §8.2, VU §5.9)
core/tsr_vu_chk.sv
core/tsr_vu_xc.sv
core/tsr_issue.sv
// W4-B db: the DB block (DB §10.1): stop logic, shadow queues, history, trace writer, the DBG block
// (tsr_dbg, core clock), the clk_rv part with its crossings (tsr_dbg_rv, uses rtl/ctrl/cdc) and the
// trace writer's write mux on channel TRW_CH
core/tsr_dbg_stop.sv
core/tsr_dbg_shq.sv
core/tsr_dbg_hist.sv
core/tsr_dbg_trw.sv
core/tsr_dbg.sv
core/tsr_dbg_rv.sv
core/tsr_dbg_wrmux.sv
// the core-clock datapath and the core (CC §13.2)
core/tsr_core_dp.sv
core/tsr_core.sv

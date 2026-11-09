// rtl/ctrl/filelist.f — the RISC-V control subsystem (CC §2-§5, §11-§12) and the
// vendored IP it instantiates, unmodified (CC §3.1: third_party/ibex, third_party/riscv-dbg,
// third_party/common_cells). Paths are relative to this file's directory (verilator -F, Vivado
// via rtl/ctrl/viv/ctrl_ooc.tcl and the flows that read rtl/filelist*.f).
//
// Expects, before this list: rtl/core/tsr_tables_pkg.sv and rtl/tech/<target>/tsr_sram_1r1w.sv
// (both in rtl/filelist.f / rtl/filelist_xilinx.f, which include this list with -F). Verilator also
// needs rtl/ctrl/waivers/vendor.vlt before any source (file-scoped to third_party/; rtl/filelist.f
// lists it first; Ibex's own waiver file is not needed, and its global PINCONNECTEMPTY waiver would
// cover the whole core). Simulation builds may define RVFI for Ibex's RISC-V Formal Interface
// (tb/ctrl's lockstep, CC §15.4).
//
// common_cells/src/fifo_v3.sv (riscv-dbg's response FIFO) is read without common_cells' assertion
// header (+define+ lines below), so the two assertion macro sets never meet.
+incdir+../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl
+incdir+../../third_party/ibex/vendor/lowrisc_ip/dv/sv/dv_utils
+incdir+../../third_party/common_cells/include
// fifo_v3 without its common_cells assertion header (whose ASSERT macros would replace
// lowRISC's) and without its three assertions
+define+COMMON_CELLS_ASSERTIONS_SVH
+define+COMMON_CELLS_ASSERTS_OFF
// lowRISC prim packages and primitives (Ibex's vendor/lowrisc_ip; generic implementations)
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_util_pkg.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_mubi_pkg.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_secded_pkg.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_count_pkg.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_cipher_pkg.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim_generic/rtl/prim_ram_1p_pkg.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim_generic/rtl/prim_buf.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim_generic/rtl/prim_flop.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim_generic/rtl/prim_and2.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim_generic/rtl/prim_clock_mux2.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_count.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_lfsr.sv
../../third_party/ibex/vendor/lowrisc_ip/ip/prim/rtl/prim_fifo_sync.sv
prim/prim_clock_gating.sv
// Ibex (lowRISC/ibex; ibex_top.core / ibex_core.core / ibex_pkg.core file sets)
../../third_party/ibex/rtl/ibex_pkg.sv
../../third_party/ibex/rtl/ibex_cheriot_pkg.sv
../../third_party/ibex/vendor/pulp_common_cells/rtl/stream_fork.sv
../../third_party/ibex/vendor/pulp_common_cells/rtl/stream_join_dynamic.sv
../../third_party/ibex/rtl/ibex_cheriot_ex.sv
../../third_party/ibex/rtl/ibex_alu.sv
../../third_party/ibex/rtl/ibex_branch_predict.sv
../../third_party/ibex/rtl/ibex_compressed_decoder.sv
../../third_party/ibex/rtl/ibex_controller.sv
../../third_party/ibex/rtl/ibex_cs_registers.sv
../../third_party/ibex/rtl/ibex_csr.sv
../../third_party/ibex/rtl/ibex_counter.sv
../../third_party/ibex/rtl/ibex_decoder.sv
../../third_party/ibex/rtl/ibex_ex_block.sv
../../third_party/ibex/rtl/ibex_fetch_fifo.sv
../../third_party/ibex/rtl/ibex_id_stage.sv
../../third_party/ibex/rtl/ibex_if_stage.sv
../../third_party/ibex/rtl/ibex_load_store_unit.sv
../../third_party/ibex/rtl/ibex_multdiv_fast.sv
../../third_party/ibex/rtl/ibex_multdiv_slow.sv
../../third_party/ibex/rtl/ibex_prefetch_buffer.sv
../../third_party/ibex/rtl/ibex_pmp.sv
../../third_party/ibex/rtl/ibex_wb_stage.sv
../../third_party/ibex/rtl/ibex_dummy_instr.sv
../../third_party/ibex/rtl/ibex_core.sv
../../third_party/ibex/rtl/ibex_register_file_ff.sv
../../third_party/ibex/rtl/ibex_register_file_fpga.sv
../../third_party/ibex/rtl/ibex_register_file_latch.sv
../../third_party/ibex/rtl/ibex_trvk.sv
../../third_party/ibex/rtl/ibex_lockstep.sv
../../third_party/ibex/rtl/ibex_top.sv
// riscv-dbg (pulp-platform/riscv-dbg; Bender.yml sources without the JTAG DTM)
../../third_party/riscv-dbg/src/dm_pkg.sv
../../third_party/riscv-dbg/debug_rom/debug_rom.sv
../../third_party/riscv-dbg/debug_rom/debug_rom_one_scratch.sv
../../third_party/riscv-dbg/src/dm_csrs.sv
../../third_party/riscv-dbg/src/dm_mem.sv
../../third_party/riscv-dbg/src/dm_sba.sv
../../third_party/riscv-dbg/src/dm_top.sv
// Tesseract control subsystem
tsr_ctrl_pkg.sv
cdc/tsr_sync.sv
cdc/tsr_rst_sync.sv
cdc/tsr_cdc_gray.sv
cdc/tsr_cdc_fifo.sv
cdc/tsr_cdc_hs.sv
tsr_rv_bus.sv
tsr_ctl_win.sv
tsr_ctrl_rv.sv
tsr_ctrl_cdc.sv
tsr_ctrl.sv
// pulp common_cells (riscv-dbg's dependency)
../../third_party/common_cells/src/fifo_v3.sv

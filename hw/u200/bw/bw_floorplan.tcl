# hw/u200/bw/bw_floorplan.tcl — DRAFT floorplan of the M0 bandwidth kernel tsr_bw_k4 (not used by
# default; TSR_BW_FLOORPLAN=1 in hw/u200/bw/build_bw_xclbin.sh passes it as the OPT_DESIGN pre-hook,
# which v++ sources after the platform's own preopt.tcl). Not run in a link yet. The bank SLRs below
# and the pblock names are those of xilinx_u200_gen3x16_xdma_2_202110_1 (platforminfo: bank0 SLR0,
# bank1 / bank2 SLR1, bank3 SLR2; hw.xsa impl.xdc: pblock_dynamic_SLR0..2, docs/platform-u200.md §8).
# Per channel c the engine (u_kernel/g_ch[c].u_eng) and the AXI bridge (u_kernel/g_ch[c].u_br) go
# to the platform's dynamic-region pblock of the SLR of DDR[c] (created over the whole SLR when the
# platform has none); the control block stays where the placer puts it.
set TSR_DDR_SLR {0 SLR0 1 SLR1 2 SLR1 3 SLR2}
set kinst [get_cells -hierarchical -filter {REF_NAME == tsr_bw_k4 || ORIG_REF_NAME == tsr_bw_k4}]
if {[llength $kinst] != 1} {
  puts "bw_floorplan: tsr_bw_k4 instance not found (found [llength $kinst]); no pblocks"
  return
}
array set pb {}
foreach slr {SLR0 SLR1 SLR2} {
  set p [get_pblocks -quiet pblock_dynamic_$slr]
  if {[llength $p] == 0} {
    set p [create_pblock tsr_bw_pb_$slr]
    resize_pblock $p -add $slr
    puts "bw_floorplan: no pblock_dynamic_$slr in the platform; created [get_property NAME $p]"
  }
  set pb($slr) $p
}
foreach {c slr} $TSR_DDR_SLR {
  set cells [list]
  foreach pat [list "$kinst/u_kernel/g_ch\[$c\].u_eng" "$kinst/u_kernel/g_ch\[$c\].u_br"] {
    set cl [get_cells -quiet $pat]
    if {[llength $cl] == 0} { puts "bw_floorplan: no cell $pat" } else { lappend cells {*}$cl }
  }
  if {[llength $cells]} { add_cells_to_pblock $pb($slr) $cells }
}

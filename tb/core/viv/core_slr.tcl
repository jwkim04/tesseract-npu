# tb/core/viv/core_slr.tcl — SLR floorplan C'' of the whole-core out-of-context runs
# (tb/core/viv/core_ooc.tcl floorplan=tb/core/viv/core_slr.tcl, sourced after opt_design).
#
# The floorplan itself is hw/u200/k4_floorplan.tcl, the one the v++ link applies (docs/platform-u200.md §8.1,
# docs/design/04-control-core.md §14.2): clusters 0 / 2 in SLR0, clusters 1 / 3 in SLR2.
# VU banks put four lanes in each outer SLR, with two back-equipped lanes per bank.
# DDR0 / 3 arbiters and weight-pipe ends stay local; five DMA boundary slices per outer
# channel stay in SLR1. Paired gather keeps source slots and the first shared stage local,
# with the final shared stage central. The control core has its own SLR1 sub-pblock
# (>= 64 RAMB36, no URAM column); issue, SPM and the other central units stay in SLR1.
# Every crossing pipe retains registered endpoints on the corresponding sides.
# This file only makes the stand-ins of the platform's pblocks the script expects and sources it for a bare
# tsr_core (TSR_K4_FP_CORE = ""): the ports are unplaced, the AXI bridges / DDR4 controllers are not modelled.
#   pblock_dynamic_SLR0  the whole SLR0
#   pblock_dynamic_SLR1  CLOCKREGION_X0Y5:CLOCKREGION_X2Y9, the columns that match the U200 platform's dynamic
#                        region (UG1120: 205 K LUT / 1,320 DSP / 160 URAM), the rest of SLR1 being the shell
#   pblock_dynamic_SLR2  the whole SLR2
# Environment: TSR_SLR_CR  per-SLR clock-region ranges replacing the whole SLR, "SLR:range[,range] ...";
#   default "1:CLOCKREGION_X0Y5:CLOCKREGION_X2Y9"; "none" = whole SLRs everywhere. TSR_K4_FP_CTRL: the
#   control core's clock-region range (hw/u200/k4_floorplan.tcl).
# Example (k4, the T2 run): tb/core/viv/core_ooc.sh k4 OUT from=SYNTH/post_synth.dcp dcp \
#   floorplan=tb/core/viv/core_slr.tcl place=AltSpreadLogic_high prphys
# Prints K4FP lines (cells per group and pblock, pinned stages) and COOC pblock_cap <name> LUT .. FF .. DSP
# .. RAMB36 .. URAM .. (the sites inside each pblock).
set tsr_slr_crs [expr {[info exists ::env(TSR_SLR_CR)] ? $::env(TSR_SLR_CR) : "1:CLOCKREGION_X0Y5:CLOCKREGION_X2Y9"}]
array set tsr_slr_cr {}
if {$tsr_slr_crs ne "none"} {
  foreach e $tsr_slr_crs {
    set i [string first : $e]
    set tsr_slr_cr([string range $e 0 [expr {$i - 1}]]) [split [string range $e [expr {$i + 1}] end] ,]
  }
}
foreach s {0 1 2} {
  set pb [create_pblock pblock_dynamic_SLR$s]
  if {[info exists tsr_slr_cr($s)]} {
    foreach r $tsr_slr_cr($s) { resize_pblock $pb -add $r }
  } else {
    resize_pblock $pb -add [get_slrs SLR$s]
  }
}
set ::env(TSR_K4_FP_CORE) ""
source [file join $::repo hw u200 k4_floorplan.tcl]
foreach pb [get_pblocks] {
  # Capacity of the pblock (sites): LUT = 8 per SLICE, FF = 16 per SLICE
  set sl [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE =~ SLICE*}]]
  set ds [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE == DSP48E2}]]
  set br [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE == RAMBFIFO36}]]
  set ur [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE == URAM288}]]
  puts "COOC pblock_cap $pb LUT [expr {8 * $sl}] FF [expr {16 * $sl}] DSP $ds RAMB36 $br URAM $ur"
}

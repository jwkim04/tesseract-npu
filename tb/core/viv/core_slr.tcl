# tb/core/viv/core_slr.tcl — SLR floorplan of the whole-core out-of-context runs
# (tb/core/viv/core_ooc.tcl floorplan=tb/core/viv/core_slr.tcl, sourced after opt_design).
#
# Places MVM cluster c (with its result assembler u_gather/g_asm[c].u_asm) and the memory
# arbiter of channel c in given SLRs and the rest of the core (SPM, central gather, cmdproc, DMA,
# QACT, fetch, CSR, MVM dispatch) in one SLR. Of the broadcast / result / weight-stream / job
# pipes (g_cl[c].u_bcp / u_rp / u_wqp / u_wrp, u_gather/g_asm[c].u_sp / u_jp) the first stage
# goes to the source side and the last stage to the destination side, so every crossing is one
# register-to-register hop; the reset copies stay free. Ports are unplaced (out of context): the
# AXI bridges / DDR4 controllers are not modelled.
# Environment:
#   TSR_SLR_MAP   "cluster:SLR ..." (default "0:0 1:2 2:0 3:2": clusters 0 / 2 in SLR0, 1 / 3 in
#                 SLR2, floorplan C' of docs/rtl-m3.md §14.5, which met 300 MHz; "0:0 1:1 2:1 3:2"
#                 = every cluster in its DDR bank's SLR, which does not fit SLR1's dynamic region
#                 at T_MAX 8); clusters >= NUM_CH ignored
#   TSR_SLR_ARB   "channel:SLR ..." for the memory arbiters (default "0:1 1:1 2:1 3:1": all with
#                 the DMA; the weight-stream pipes cross to the clusters)
#   TSR_SLR_REST  SLR of the rest of the core (default 1)
#   TSR_SLR_CR    per-SLR clock-region ranges replacing the whole SLR, "SLR:range[,range] ...";
#                 default "1:CLOCKREGION_X0Y5:CLOCKREGION_X2Y9": SLR1 limited to the columns that
#                 match the U200 platform's dynamic region (UG1120: 205 K LUT / 1,320 DSP / 160
#                 URAM), the rest of SLR1 being the shell; "none" = whole SLRs everywhere
# Example (k4, the §14 run): tb/core/viv/core_ooc.sh k4 OUT from=SYNTH/post_synth.dcp dcp \
#   floorplan=tb/core/viv/core_slr.tcl place=AltSpreadLogic_high prphys
# Prints COOC pblock <name> <SLR> <range> <cells ...> and COOC pblock_cap <name> LUT .. FF .. DSP
# .. RAMB36 .. URAM .. (the sites inside the pblock).
set map  [expr {[info exists ::env(TSR_SLR_MAP)]  ? $::env(TSR_SLR_MAP)  : "0:0 1:2 2:0 3:2"}]
set rest [expr {[info exists ::env(TSR_SLR_REST)] ? $::env(TSR_SLR_REST) : 1}]
set amap [expr {[info exists ::env(TSR_SLR_ARB)]  ? $::env(TSR_SLR_ARB)  : "0:1 1:1 2:1 3:1"}]
set crs  [expr {[info exists ::env(TSR_SLR_CR)]   ? $::env(TSR_SLR_CR)   : "1:CLOCKREGION_X0Y5:CLOCKREGION_X2Y9"}]
array set cr {}
if {$crs ne "none"} {
  foreach e $crs {
    set i [string first : $e]
    set cr([string range $e 0 [expr {$i - 1}]]) [split [string range $e [expr {$i + 1}] end] ,]
  }
}
# One pblock per SLR used, created on first use; cells are added as objects straight from
# get_cells (a Tcl list of their names would lose the object type).
array set pbs {}
array set names {}
proc tsr_pb {s} {
  global pbs cr
  if {![info exists pbs($s)]} {
    set pb [create_pblock tsr_pb_slr$s]
    if {[info exists cr($s)]} {
      foreach r $cr($s) { resize_pblock $pb -add $r }
    } else {
      resize_pblock $pb -add [get_slrs SLR$s]
    }
    set pbs($s) $pb
  }
  return $pbs($s)
}
proc tsr_add {s cl} {
  global names
  if {[llength $cl] == 0} { return }
  add_cells_to_pblock [tsr_pb $s] $cl
  if {[llength $cl] <= 4} { lappend names($s) {*}$cl } else { lappend names($s) "([llength $cl] cells)" }
}
foreach e $map {
  lassign [split $e :] c s
  tsr_add $s [get_cells -quiet "g_mvm.u_mvm/g_cl\[$c\].u_cl"]
  # the cluster's result assembler (tsr_mvm_gasm) goes with it; its start / job pipes stay free
  tsr_add $s [get_cells -quiet "g_mvm.u_mvm/u_gather/g_asm\[$c\].u_asm"]
}
foreach e $amap {
  lassign [split $e :] c s
  tsr_add $s [get_cells -quiet "g_ch\[$c\].u_arb"]
}
foreach pat {u_spm u_cmdproc u_dma u_qact u_fetch u_csr} {
  tsr_add $rest [get_cells -quiet $pat]
}
# central part of the gather: every leaf cell outside the per-cluster assemblers and pipes
tsr_add $rest [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE && NAME =~ g_mvm.u_mvm/u_gather/* && \
  NAME !~ g_mvm.u_mvm/u_gather/g_asm*}]
# MVM top-level logic (dispatch, broadcast source, SPM client mux): every leaf cell directly in
# g_mvm.u_mvm outside the clusters, pipes, gather and reset copies.
tsr_add $rest [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE && NAME =~ g_mvm.u_mvm/* && \
  NAME !~ g_mvm.u_mvm/g_cl* && NAME !~ g_mvm.u_mvm/u_gather/* && NAME !~ g_mvm.u_mvm/g_rst*}]
# Pipes between a cluster (with its assembler) and the centre: the first stage on the source
# side, the last on the destination side (DEPTH >= 2), so each SLR crossing is one
# register-to-register hop; the result pipe (cluster -> assembler) stays with the cluster.
proc tsr_pipe_ends {inst first_s last_s} {
  array set byst {}
  set maxs -1
  set kids [get_cells -quiet "$inst/*"]
  foreach n [get_property -quiet NAME $kids] {
    if {[regexp {/(?:g_keep|g_regs)\.[vd]_q_reg\[([0-9]+)\]} $n -> st] ||
        [regexp {/g_slices\.g_s\[([0-9]+)\]\.u_slice$} $n -> st]} {
      lappend byst($st) $n
      if {$st > $maxs} { set maxs $st }
    }
  }
  if {$maxs < 1} { return 0 }
  add_cells_to_pblock [tsr_pb $first_s] [get_cells -quiet $byst(0)]
  add_cells_to_pblock [tsr_pb $last_s]  [get_cells -quiet $byst($maxs)]
  return [expr {$maxs + 1}]
}
set npin 0
foreach e $map {
  lassign [split $e :] c s
  set p "g_mvm.u_mvm/g_cl\[$c\]"
  set g "g_mvm.u_mvm/u_gather/g_asm\[$c\]"
  incr npin [tsr_pipe_ends "$p.u_bcp" $rest $s]
  incr npin [tsr_pipe_ends "$p.u_wrp" $rest $s]
  incr npin [tsr_pipe_ends "$p.u_wqp" $s $rest]
  incr npin [tsr_pipe_ends "$p.u_rp"  $s $s]
  incr npin [tsr_pipe_ends "$g.u_sp"  $rest $s]
  incr npin [tsr_pipe_ends "$g.u_jp"  $s $rest]
}
puts "COOC pipes pinned: $npin stages total (first / last stage of every pipe with DEPTH >= 2)"
foreach s [lsort [array names pbs]] {
  set pb $pbs($s)
  puts "COOC pblock tsr_pb_slr$s SLR$s [expr {[info exists cr($s)] ? $cr($s) : "whole"}] $names($s)"
  # Capacity of the pblock (sites): LUT = 8 per SLICE, FF = 16 per SLICE
  set sl [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE =~ SLICE*}]]
  set ds [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE == DSP48E2}]]
  set br [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE == RAMBFIFO36}]]
  set ur [llength [get_sites -quiet -of_objects $pb -filter {SITE_TYPE == URAM288}]]
  puts "COOC pblock_cap tsr_pb_slr$s LUT [expr {8 * $sl}] FF [expr {16 * $sl}] DSP $ds RAMB36 $br URAM $ur"
}

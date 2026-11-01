# hw/u200/k4_floorplan.tcl — SLR floorplan C'' of tsr_u200_k4 in the v++ link (docs/platform-u200.md
# §8, docs/design/04-control-core.md §14.2): floorplan C' (docs/rtl-m3.md §14.5, the out-of-context floorplan
# that met 300 MHz: k4 +0.015 ns, k4t1 +0.108 ns) translated to the kernel hierarchy and to the platform's
# dynamic-region pblocks, extended by the control core, the issue path and the vector unit.
#
# Applied by hw/u200/k4.cfg as the place_design pre-hook (run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE):
# after opt_design, where tb/core/viv/core_ooc.tcl sources tb/core/viv/core_slr.tcl (which makes stand-in
# pblocks and sources this file). v++ sources it after its own and the platform's hooks of that step
# (scripts/impl_1/_full_place_pre.tcl).
#
# Pblocks: the platform's, never created here (xilinx_u200_gen3x16_xdma_2_202110_1, hw.xsa
# tcl_hooks/impl.xdc; children of pblock_dynamic_region; platforminfo: DDR bank0 SLR0, bank1 and
# bank2 SLR1, bank3 SLR2), plus one child of pblock_dynamic_SLR1 for the control core (below). The script
# stops with an error when a platform pblock is missing.
#   pblock_dynamic_SLR0  CLOCKREGION_X0Y0:CLOCKREGION_X5Y4 minus the isolation pipeline slices
#   pblock_dynamic_SLR1  CLOCKREGION_X0Y5:CLOCKREGION_X2Y9 (the rest of SLR1 is the static shell;
#                        the out-of-context runs used exactly this range as the SLR1 stand-in)
#   pblock_dynamic_SLR2  CLOCKREGION_X0Y10:CLOCKREGION_X5Y14 minus the isolation pipeline slices
#   pblock_tsr_ctrl_rv   CLOCKREGION_X0Y6:CLOCKREGION_X0Y8, a child of pblock_dynamic_SLR1: 108 RAMB36
#                        (the TCMs take 48; the gate asks for >= 64), 0 URAM (the SPM's 128 URAM sit in
#                        the X1 / X2 columns), ~5,100 slices, 360 DSP; holds tsr_ctrl_rv (Ibex, debug
#                        module, TCMs, bus, control window)
#
# Placement (C''):
#   SLR0  MVM clusters 0 and 2 with their result assemblers (u_gather/g_asm[c].u_asm); the 16 lanes of the
#         vector unit (g_vu.u_vu/u_lanes/g_lane[l].u_lane: CC §14.1 remedy 3, VU §7.10), behind
#         VU_LANE_PIPE register stages on each side (rtl/core/tsr_vu_lanes.sv): of the input pipes the first
#         stage (injector side) is pinned to SLR1 and the last to SLR0, of the output pipes the first to SLR0
#         and the last to SLR1, so each crossing is one register-to-register hop
#   SLR2  MVM clusters 1 and 3 with their assemblers
#   SLR1  everything else of the core: the control subsystem (u_ctrl; tsr_ctrl_rv in pblock_tsr_ctrl_rv,
#         the CDC and reset synchronisers beside it), the issue path and CSR block (u_issue, u_csr: between
#         the CSR bridge and the units), the DB block, the four memory arbiters, SPM, DMA, QACT, the
#         SPM-facing part of the vector unit (reader, injector, writer, reduction, angle unit),
#         central gather, MVM dispatch; the control bridge (u_ctrl of the adapter), the reset block
#         (u_rst), the bridges of channels 1 and 2 and the core side of the bridges of channels 0 and 3
#   Pipes between a cluster (with its assembler) and the centre (g_cl[c].u_bcp / u_wrp / u_wqp / g_ajp.u_ajp,
#   u_gather/g_asm[c].u_sp / g_jp.u_jp + u_jl): the first stage on the source side, the last on the
#   destination side, so every crossing is one register-to-register hop; the result pipe
#   g_cl[c].u_rp stays with its cluster (both ends). The core's reset copies stay free.
#   Bridges 0 and 3 cross in their AXI register slices (AXI_SLICES >= 2): the AXI-side stage of
#   every slice chain (the last stage of AR / AW / W, the first of R / B) and the registered AR / AW
#   handshake (g_hs_q) go to the SLR of DDR[0] (SLR0) / DDR[3] (SLR2), the rest of the bridge
#   (ID filter, counters, output slices, base adders) to SLR1.
#
# Environment (validation outside the link only; docs/platform-u200.md §8):
#   TSR_K4_FP_BODY  hierarchical name of the tsr_xrt_kernel instance ("" = the top cell). Default:
#                   <tsr_u200_k4 instance>/u_kernel, the instance found by REF_NAME / ORIG_REF_NAME
#                   tsr_u200_k4, else by the compute-unit name (*/tsr_k4/inst, hw/u200/k4.cfg)
#   TSR_K4_FP_CORE  hierarchical name of a tsr_core with no adapter around it ("" = the top cell,
#                   e.g. a tb/core/viv/core_ooc.tcl checkpoint): only the core part is applied
#   TSR_K4_FP_CTRL  clock-region range of pblock_tsr_ctrl_rv (default CLOCKREGION_X0Y6:CLOCKREGION_X0Y8)
# Prints (grep K4FP): the cells of every group and pblock, the pinned pipe stages (56 at NUM_CH 4, plus the
# lane pipes' ends) and the bridge cells per side.

set k4fp_map  {0 0 1 2 2 0 3 2}       ;# cluster -> SLR (C')
set k4fp_rest 1                       ;# SLR of the rest of the core and of the adapter
set k4fp_lane 0                       ;# SLR of the VU lanes (CC §14.1 remedy 3)
set k4fp_ddr  {0 0 3 2}               ;# bridges that cross: channel -> SLR of DDR[channel]

proc k4fp_path {base rel} {
  if {$base eq ""} { return $rel }
  return "$base/$rel"
}

current_instance -quiet
if {[info exists ::env(TSR_K4_FP_CORE)]} {
  set k4fp_mode core
  set k4fp_body ""
  set k4fp_core $::env(TSR_K4_FP_CORE)
} else {
  set k4fp_mode kernel
  if {[info exists ::env(TSR_K4_FP_BODY)]} {
    set k4fp_body $::env(TSR_K4_FP_BODY)
  } else {
    set kinst [get_cells -quiet -hierarchical -filter {REF_NAME == tsr_u200_k4 || ORIG_REF_NAME == tsr_u200_k4}]
    if {[llength $kinst] == 0} {
      set kinst [get_cells -quiet -hierarchical -filter {NAME =~ */tsr_k4/inst && !IS_PRIMITIVE}]
    }
    if {[llength $kinst] != 1} {
      error "k4_floorplan: expected one tsr_u200_k4 instance, found [llength $kinst] ($kinst)"
    }
    set k4fp_body "[get_property NAME $kinst]/u_kernel"
  }
  set k4fp_core [k4fp_path $k4fp_body u_core]
}
puts "K4FP mode $k4fp_mode body '$k4fp_body' core '$k4fp_core'"

# The platform's pblocks
array set k4fp_pb {}
foreach s {0 1 2} {
  set p [get_pblocks -quiet pblock_dynamic_SLR$s]
  if {[llength $p] != 1} {
    error "k4_floorplan: platform pblock pblock_dynamic_SLR$s not found (xilinx_u200_gen3x16_xdma_2_202110_1 defines pblock_dynamic_SLR0..2)"
  }
  set k4fp_pb($s) $p
}

array set k4fp_names {}
array set k4fp_count {}
foreach s {0 1 2} { set k4fp_names($s) {}; set k4fp_count($s) 0 }

# Add the cells (objects from get_cells) of one group to the pblock of SLR s; empty = error.
proc k4fp_add {s cl what} {
  global k4fp_pb k4fp_names k4fp_count
  if {[llength $cl] == 0} { error "k4_floorplan: no cells for $what" }
  add_cells_to_pblock $k4fp_pb($s) $cl
  incr k4fp_count($s) [llength $cl]
  if {[llength $cl] <= 2} {
    lappend k4fp_names($s) $what
  } else {
    lappend k4fp_names($s) "$what ([llength $cl] cells)"
  }
}
proc k4fp_cell {name} {
  set c [get_cells -quiet $name]
  if {[llength $c] != 1} { error "k4_floorplan: cell $name not found" }
  return $c
}

# ---- core: clusters, assemblers and VU lanes away from SLR1, the control core in its sub-pblock ----
set dp  [k4fp_path $k4fp_core u_dp]
set mvm [k4fp_path $dp g_mvm.u_mvm]
set vu  [k4fp_path $dp g_vu.u_vu]
set nch 0
foreach {c s} $k4fp_map {
  set cl [get_cells -quiet "$mvm/g_cl\[$c\].u_cl"]
  if {[llength $cl] == 0} { continue }        ;# NUM_CH < 4: no such cluster
  incr nch
  k4fp_add $s $cl "g_cl\[$c\].u_cl"
  k4fp_add $s [k4fp_cell "$mvm/u_gather/g_asm\[$c\].u_asm"] "g_asm\[$c\].u_asm"
}
if {$nch == 0} { error "k4_floorplan: no MVM cluster under $mvm" }

# the VU lanes (tsr_vu_lanes; the pipe stages are pinned below)
set lanes {}
for {set l 0} {$l < 16} {incr l} { lappend lanes [k4fp_cell "$vu/u_lanes/g_lane\[$l\].u_lane"] }
k4fp_add $k4fp_lane $lanes "u_vu/u_lanes (16 lanes)"

# the control core's sub-pblock of SLR1: tsr_ctrl_rv (Ibex, debug module, TCMs, bus, control window)
set k4fp_ctrl_cr "CLOCKREGION_X0Y6:CLOCKREGION_X0Y8"
if {[info exists ::env(TSR_K4_FP_CTRL)]} { set k4fp_ctrl_cr $::env(TSR_K4_FP_CTRL) }
set k4fp_ctrl_pb [get_pblocks -quiet pblock_tsr_ctrl_rv]
if {[llength $k4fp_ctrl_pb] == 0} {
  set k4fp_ctrl_pb [create_pblock pblock_tsr_ctrl_rv]
  resize_pblock $k4fp_ctrl_pb -add $k4fp_ctrl_cr
  if {[catch {set_property PARENT $k4fp_pb(1) $k4fp_ctrl_pb} err]} {
    puts "K4FP warning: pblock_tsr_ctrl_rv stays a top-level pblock ($err)"
  }
}
set k4fp_ctrl_cells [k4fp_cell [k4fp_path $k4fp_core u_ctrl/u_rv]]
add_cells_to_pblock $k4fp_ctrl_pb $k4fp_ctrl_cells
puts "K4FP pblock_tsr_ctrl_rv $k4fp_ctrl_cr: u_ctrl/u_rv, [llength [get_sites -quiet -of_objects $k4fp_ctrl_pb -filter {SITE_TYPE == RAMBFIFO36}]] RAMB36 sites, [llength [get_sites -quiet -of_objects $k4fp_ctrl_pb -filter {SITE_TYPE == URAM288}]] URAM sites"

# the rest of the control subsystem, the DB block's clk_rv part and every other block of the core: SLR1, whole
# hierarchical cells (the reset stage registers of u_dp are leaf cells there and stay free)
set k4fp_rest_cells {}
foreach c [get_cells -quiet "[k4fp_path $k4fp_core *]" -filter {!IS_PRIMITIVE}] {
  set n [get_property NAME $c]
  if {$n eq [k4fp_path $k4fp_core u_dp]} continue
  if {$n eq [k4fp_path $k4fp_core u_ctrl]} {
    foreach cc [get_cells -quiet "$n/*"] {
      if {[get_property NAME $cc] ne [k4fp_path $k4fp_core u_ctrl/u_rv]} { lappend k4fp_rest_cells $cc }
    }
  } else {
    lappend k4fp_rest_cells $c
  }
}
foreach c [get_cells -quiet "$dp/*" -filter {!IS_PRIMITIVE}] {
  set n [get_property NAME $c]
  if {$n eq $mvm || $n eq $vu} continue          ;# partitioned below
  lappend k4fp_rest_cells $c
}
k4fp_add $k4fp_rest $k4fp_rest_cells "core blocks (u_ctrl, u_dp: issue, CSR, SPM, DMA, QACT, arbiters, DB, ...; [llength $k4fp_rest_cells] cells)"
# MVM top-level logic (dispatch, broadcast source, SPM client mux): the leaf cells directly in
# g_mvm.u_mvm outside the clusters, pipes, gather and reset copies; central part of the gather: every leaf
# cell outside the per-cluster assemblers and their pipes
k4fp_add $k4fp_rest [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $mvm/u_gather/* && \
  NAME !~ $mvm/u_gather/g_asm*"] "u_gather (centre)"
k4fp_add $k4fp_rest [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $mvm/* && \
  NAME !~ $mvm/g_cl* && NAME !~ $mvm/u_gather/* && NAME !~ $mvm/g_rst*"] "u_mvm (top)"
# the vector unit outside its lanes: injector, reader, writer, reduction, angle unit, sequencer, local memory
k4fp_add $k4fp_rest [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $vu/* && \
  NAME !~ $vu/u_lanes/*"] "u_vu (SPM-facing part)"

# ---- pipe ends (as core_slr.tcl): first stage at the source, last at the destination ----------
proc k4fp_pipe_ends {inst first_s last_s} {
  array set byst {}
  set maxs -1
  foreach n [get_property -quiet NAME [get_cells -quiet "$inst/*"]] {
    if {[regexp {/(?:g_keep|g_regs)\.[vd]_q_reg\[([0-9]+)\]} $n -> st] ||
        [regexp {/g_slices\.g_s\[([0-9]+)\]\.u_slice$} $n -> st]} {
      lappend byst($st) $n
      if {$st > $maxs} { set maxs $st }
    }
  }
  if {$maxs < 1} { error "k4_floorplan: pipe $inst has fewer than 2 stages (MVM_BC_PIPE / MVM_RES_PIPE >= 2)" }
  global k4fp_pb
  add_cells_to_pblock $k4fp_pb($first_s) [get_cells -quiet $byst(0)]
  add_cells_to_pblock $k4fp_pb($last_s)  [get_cells -quiet $byst($maxs)]
  return [expr {$maxs + 1}]
}
# The job pipe of an assembler (tsr_mvm_gather g_jp): u_jp (JOB_PIPE - 1 slices, the first at the assembler)
# and u_jl (the last slice, in front of the central take)
proc k4fp_jp_ends {inst first_s last_s} {
  global k4fp_pb
  set slices {}
  foreach n [get_property -quiet NAME [get_cells -quiet "$inst.u_jp/*"]] {
    if {[regexp {/g_slices\.g_s\[0\]\.u_slice$} $n]} { lappend slices $n }
  }
  if {[llength $slices] != 1} { error "k4_floorplan: job pipe $inst.u_jp has no stage 0 (JOB_PIPE >= 2)" }
  add_cells_to_pblock $k4fp_pb($first_s) [get_cells -quiet $slices]
  add_cells_to_pblock $k4fp_pb($last_s) [k4fp_cell "$inst.u_jl"]
  return 2
}
set npin 0
foreach {c s} $k4fp_map {
  if {$c >= $nch} { continue }
  set p "$mvm/g_cl\[$c\]"
  set g "$mvm/u_gather/g_asm\[$c\]"
  incr npin [k4fp_pipe_ends "$p.u_bcp" $k4fp_rest $s]
  incr npin [k4fp_pipe_ends "$p.u_wrp" $k4fp_rest $s]
  incr npin [k4fp_pipe_ends "$p.u_wqp" $s $k4fp_rest]
  incr npin [k4fp_pipe_ends "$p.u_rp"  $s $s]
  incr npin [k4fp_pipe_ends "$p.g_ajp.u_ajp" $s $k4fp_rest]
  incr npin [k4fp_pipe_ends "$g.u_sp"  $k4fp_rest $s]
  incr npin [k4fp_jp_ends   "$g.g_jp"  $s $k4fp_rest]
}
puts "K4FP pipes pinned: $npin stages (first / last stage of the 7 pipes of each of $nch clusters)"

# the lane pipes of tsr_vu_lanes (g_pipe.ip_q: injector -> lanes, rp_q: the lanes' reset, op_q: lanes -> writer):
# stage 0 / last stage by the direction (rest = the SPM-facing side)
set nlp 0
array set lps {}
set lmax 0
foreach n [get_property -quiet NAME [get_cells -quiet "$vu/u_lanes/*"]] {
  if {[regexp {/g_pipe\.(ip|rp|op)_q_reg\[([0-9]+)\]} $n -> lp_k lp_st]} {
    lappend lps($lp_k,$lp_st) $n
    if {$lp_st > $lmax} { set lmax $lp_st }
  }
}
if {$lmax < 1} { error "k4_floorplan: the lane pipes of $vu/u_lanes have fewer than 2 stages (VU_LANE_PIPE >= 2)" }
foreach {lp_k lp_first lp_last} [list ip $k4fp_rest $k4fp_lane  rp $k4fp_rest $k4fp_lane  op $k4fp_lane $k4fp_rest] {
  if {![info exists lps($lp_k,0)] || ![info exists lps($lp_k,$lmax)]} { error "k4_floorplan: lane pipe ${lp_k}_q has no stage $lmax / 0" }
  add_cells_to_pblock $k4fp_pb($lp_first) [get_cells -quiet $lps($lp_k,0)]
  add_cells_to_pblock $k4fp_pb($lp_last)  [get_cells -quiet $lps($lp_k,$lmax)]
  incr nlp [expr {[llength $lps($lp_k,0)] + [llength $lps($lp_k,$lmax)]}]
}
puts "K4FP lane pipes pinned: $nlp flip-flops (stages 0 and $lmax of the input, reset and output pipes)"

# ---- adapter: control bridge, reset block, bridges ---------------------------------------------
if {$k4fp_mode eq "kernel"} {
  foreach b {u_ctrl u_rst} {
    k4fp_add $k4fp_rest [k4fp_cell [k4fp_path $k4fp_body $b]] $b
  }
  array set ddr $k4fp_ddr
  for {set c 0} {$c < $nch} {incr c} {
    set br [k4fp_cell [k4fp_path $k4fp_body "g_ch\[$c\].u_br"]]
    if {![info exists ddr($c)]} {
      k4fp_add $k4fp_rest $br "g_ch\[$c\].u_br"
      continue
    }
    # Crossing bridge: partition its leaf cells by name (no nested pblock assignment). Stage k of
    # the slice chain u_<ch> is g_st[k].u_s; the AXI side is the last stage of AR / AW / W and the
    # first of R / B.
    set brn [get_property NAME $br]
    set leaves [get_property -quiet NAME [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $brn/*"]]
    set re {^u_(ar|aw|w|r|b)/g_st\[([0-9]+)\]\.u_s/}
    set plen [string length "$brn/"]
    array unset last
    foreach n $leaves {
      if {[regexp $re [string range $n $plen end] -> ch st] && (![info exists last($ch)] || $st > $last($ch))} {
        set last($ch) $st
      }
    }
    foreach ch {ar aw w r b} {
      if {![info exists last($ch)] || $last($ch) < 1} {
        error "k4_floorplan: slice chain u_$ch of $brn has fewer than 2 stages (AXI_SLICES >= 2)"
      }
    }
    set far {}
    set near {}
    foreach n $leaves {
      set rel [string range $n $plen end]
      if {[regexp $re $rel -> ch st]} {
        set axi [expr {$ch in {ar aw w} ? $st == $last($ch) : $st == 0}]
      } else {
        set axi [string match {g_hs_q.*} $rel]
      }
      if {$axi} { lappend far $n } else { lappend near $n }
    }
    k4fp_add $ddr($c) [get_cells -quiet $far] "g_ch\[$c\].u_br (AXI side)"
    k4fp_add $k4fp_rest [get_cells -quiet $near] "g_ch\[$c\].u_br (core side)"
    puts "K4FP bridge $c: [llength $far] cells in SLR$ddr($c) (AXI-side slice stages, AR / AW handshake), [llength $near] in SLR$k4fp_rest"
  }
}

foreach s {0 1 2} {
  puts "K4FP pblock_dynamic_SLR$s: $k4fp_count($s) cells assigned (excl. pinned pipe stages): [join $k4fp_names($s) {, }]"
}
current_instance -quiet

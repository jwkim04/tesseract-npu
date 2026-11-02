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
#   Broadcast: u_bc_head is the shared first stage in SLR1; every stage of g_cl[c].u_bcp
#   and its rebuilt decode stays with its cluster. Older checkpoints without u_bc_head keep
#   the first per-cluster broadcast stage in SLR1 and the remaining stages local.
#   Other pipes between a cluster (with its assembler) and the centre (g_cl[c].u_wrp / u_wqp,
#   u_gather/g_asm[c].u_sp / g_jp.u_jp + u_jl): the first stage on the source side, the last on the
#   destination side, so every crossing is one register-to-register hop; the result pipe
#   g_cl[c].u_rp and the attention job pipe g_ajp.u_ajp stay with their cluster (both ends).
#   Attention and MVM jobs are selected there before the shared job pipe crosses to the centre.
#   Direct broadcast decode leaves stay with the destination cluster; the lane wrapper's direct
#   writer/reduction helpers stay in SLR1. Synthesis can rebuild these cones under the pipe wrappers,
#   outside the original consumer hierarchy. The split pipe parents are never assigned wholesale.
#   The optional u_lanes/u_share result mux and its selection delay stay with the lanes in SLR0.
#   The ATT_PATH_STALL reduction LUTs stay beside att_stall_q in SLR1 even when synthesis puts
#   them under a cluster. Their source registers and shared functional decode stay with the cluster.
#   The core's reset copies stay free.
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
# k4fp_check_locality ?placed? is a read-only check of these scopes; with placed=1 it also checks
# their sites' SLRs. It can be repeated after placement / phys_opt to catch new, unowned replicas.

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

# Select by the actual PERF endpoint cone, not by the cluster that inherited a LUT during
# synthesis. The name/type guard excludes unrelated functional decode in its transitive fan-in;
# exclusive fanout proves that moving these LUTs cannot drag a functional consumer with them.
proc k4fp_att_stall_cone {mvm} {
  set endpoint [k4fp_cell "$mvm/att_stall_q_reg"]
  if {[get_property REF_NAME $endpoint] ne "FDRE"} {
    error "k4_floorplan: unknown stall aggregation endpoint $endpoint"
  }
  set pin [get_pins -quiet "$mvm/att_stall_q_reg/D"]
  if {[llength $pin] != 1} { error "k4_floorplan: missing stall aggregation D pin" }
  set cone [all_fanin -flat -only_cells -trace_arcs all -to $pin]
  set result {}
  foreach cell $cone {
    if {$cell eq $endpoint} continue
    set name [get_property NAME $cell]
    set ref [get_property REF_NAME $cell]
    if {![get_property IS_PRIMITIVE $cell]} { error "k4_floorplan: unknown stall aggregation object $cell" }
    if {[get_property IS_SEQUENTIAL $cell] || $ref in {GND VCC}} continue
    set outputs [get_pins -quiet -of_objects $cell -filter {DIRECTION == OUT}]
    if {[llength $outputs] == 0} { error "k4_floorplan: stall aggregation leaf $cell has no output" }
    set sinks [lsort -unique [get_property NAME [all_fanout -flat -endpoints_only -trace_arcs all -from $outputs]]]
    if {[lsearch -exact $sinks [get_property NAME $pin]] < 0} {
      error "k4_floorplan: stall aggregation leaf $cell does not reach $pin"
    }
    # Shared functional decode is a boundary, not part of this PERF-only cone.
    if {$sinks ne [list [get_property NAME $pin]]} continue
    if {[llength $outputs] != 1 || [string first "$mvm/" $name] != 0 ||
        ![regexp {^att_stall_q_i_[0-9]+(_comp)?(_replica.*|_rep__.*)?$} [file tail $name]] ||
        ![regexp {^LUT[1-6]$} $ref]} {
      error "k4_floorplan: unknown stall aggregation leaf $cell ($ref)"
    }
    lappend result $cell
  }
  if {[llength $result] == 0} { error "k4_floorplan: empty stall aggregation cone at $pin" }
  set selected [get_property NAME $result]
  foreach cell [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $mvm/*att_stall_q*"] {
    if {$cell ne $endpoint && [lsearch -exact $selected [get_property NAME $cell]] < 0} {
      error "k4_floorplan: disconnected or shared stall aggregation leaf $cell"
    }
  }
  return [linsert $result 0 $endpoint]
}

# Keep maximal cluster subtrees intact, descending only along a branch that contains an excluded
# PERF LUT. Assigning its u_cl ancestor would otherwise conflict with the LUT's central pblock.
proc k4fp_cluster_without {inst excluded} {
  set result {}
  foreach cell [get_cells -quiet "$inst/*"] {
    set name [get_property NAME $cell]
    if {[lsearch -exact $excluded $name] >= 0} continue
    set split 0
    foreach target $excluded {
      if {[string first "$name/" $target] == 0} { set split 1; break }
    }
    if {$split} {
      if {[get_property IS_PRIMITIVE $cell]} { error "k4_floorplan: cannot partition cluster leaf $cell" }
      lappend result {*}[k4fp_cluster_without $name $excluded]
    } else { lappend result $cell }
  }
  return $result
}

# ---- core: clusters, assemblers and VU lanes away from SLR1, the control core in its sub-pblock ----
set dp  [k4fp_path $k4fp_core u_dp]
set mvm [k4fp_path $dp g_mvm.u_mvm]
set vu  [k4fp_path $dp g_vu.u_vu]
set k4fp_stall [k4fp_att_stall_cone $mvm]
set k4fp_stall_names [get_property NAME $k4fp_stall]
set nch 0
foreach {c s} $k4fp_map {
  set cl [get_cells -quiet "$mvm/g_cl\[$c\].u_cl"]
  if {[llength $cl] == 0} { continue }        ;# NUM_CH < 4: no such cluster
  incr nch
  set k4fp_cl_name [get_property NAME $cl]
  set k4fp_excluded {}
  foreach k4fp_target $k4fp_stall_names {
    if {[string first "$k4fp_cl_name/" $k4fp_target] == 0} { lappend k4fp_excluded $k4fp_target }
  }
  if {[llength $k4fp_excluded]} {
    # Also supports reapplying this floorplan to a checkpoint with the old parent assignment.
    set k4fp_old_blocks [get_pblocks -quiet -of_objects $cl]
    if {[llength $k4fp_old_blocks] > 0 &&
        [lsearch -exact [get_property NAME $k4fp_old_blocks] $k4fp_pb($s)] >= 0} {
      remove_cells_from_pblock $k4fp_pb($s) $cl
    }
    k4fp_add $s [k4fp_cluster_without $k4fp_cl_name $k4fp_excluded] "g_cl\[$c\].u_cl (outside stall aggregation)"
  } else { k4fp_add $s $cl "g_cl\[$c\].u_cl" }
  k4fp_add $s [k4fp_cell "$mvm/u_gather/g_asm\[$c\].u_asm"] "g_asm\[$c\].u_asm"
}
if {$nch == 0} { error "k4_floorplan: no MVM cluster under $mvm" }
k4fp_add $k4fp_rest $k4fp_stall "ATT_PATH_STALL aggregation and endpoint"

# the VU lanes (tsr_vu_lanes; the pipe stages are pinned below)
set lanes {}
for {set l 0} {$l < 16} {incr l} { lappend lanes [k4fp_cell "$vu/u_lanes/g_lane\[$l\].u_lane"] }
k4fp_add $k4fp_lane $lanes "u_vu/u_lanes (16 lanes)"
set k4fp_shared_vu [expr {[llength [get_cells -quiet "$vu/u_lanes/u_share"]] == 1}]
if {$k4fp_shared_vu} {
  k4fp_add $k4fp_lane [k4fp_cell "$vu/u_lanes/u_share"] "u_vu shared lane result mux"
}

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

# Direct primitive leaves exclude submodules (in particular the lanes and FIFO slices). Constants
# have no placement. Use literal prefixes below: generated indices contain square brackets.
proc k4fp_direct_leaves {inst} {
  set result {}
  foreach cell [get_cells -quiet "$inst/*" -filter {IS_PRIMITIVE}] {
    if {[get_property REF_NAME $cell] ni {GND VCC}} { lappend result $cell }
  }
  return $result
}
proc k4fp_broadcast_parts {inst first_s last_s {min_stages 2}} {
  set parts [dict create]
  set stages [dict create]
  set plen [string length "$inst/"]
  foreach cell [k4fp_direct_leaves $inst] {
    set rel [string range [get_property NAME $cell] $plen end]
    if {[regexp {^(?:g_keep|g_regs)\.[vd]_q_reg\[([0-9]+)\]} $rel -> st]} {
      dict set stages $st 1
      set s [expr {$st == 0 ? $first_s : $last_s}]
    } else {
      # tsr_vpipe itself contains only stage registers. Rebuilt combinational decode belongs
      # to its consumer, even when Vivado puts e.g. g_att.lo_q or fmt_q_rep under u_bcp.
      set ref [get_property REF_NAME $cell]
      if {![string match LUT* $ref] && ![string match CARRY* $ref]} {
        error "k4_floorplan: unknown broadcast leaf $cell ($ref)"
      }
      set s $last_s
    }
    dict lappend parts $s $cell
  }
  set indices [lsort -integer [dict keys $stages]]
  if {[llength $indices] < $min_stages} { error "k4_floorplan: broadcast $inst has fewer than $min_stages register stages" }
  for {set i 0} {$i <= [lindex $indices end]} {incr i} {
    if {![dict exists $stages $i]} { error "k4_floorplan: broadcast $inst is missing stage $i" }
  }
  return $parts
}
# The shared head may be replicated by physical optimisation, but all copies
# must stay central. A reconstructed consumer cone here needs an explicit
# ownership decision; do not silently assign arbitrary decode to SLR1.
proc k4fp_broadcast_head {mvm} {
  set inst "$mvm/u_bc_head"
  set cells [k4fp_direct_leaves $inst]
  if {[llength $cells] == 0} { error "k4_floorplan: shared broadcast head is empty" }
  set plen [string length "$inst/"]
  foreach cell $cells {
    set rel [string range [get_property NAME $cell] $plen end]
    if {![regexp {^g_keep\.[vd]_q_reg\[0\]} $rel] || [get_property REF_NAME $cell] ne "FDRE"} {
      error "k4_floorplan: unknown shared broadcast head leaf $cell"
    }
  }
  return $cells
}
proc k4fp_gather_helpers {inst} {
  set result {}
  set nselector 0
  set parent [file dirname $inst]
  set plen [string length "$inst."]
  foreach cell [k4fp_direct_leaves $parent] {
    set name [get_property NAME $cell]
    if {[string first "$inst." $name] != 0} continue
    lappend result $cell
    if {[string first "g_aj.a_att_q_reg" [string range $name $plen end]] == 0} { incr nselector }
  }
  if {$nselector == 0} { error "k4_floorplan: attention selector missing under $inst" }
  return $result
}
proc k4fp_lane_parts {inst} {
  global k4fp_rest k4fp_lane
  set parts [dict create]
  set stages [dict create]
  set pend {}
  set plen [string length "$inst/"]
  foreach cell [k4fp_direct_leaves $inst] {
    set rel [string range [get_property NAME $cell] $plen end]
    if {[regexp {^g_pipe\.(ip|rp|op)_q_reg\[([0-9]+)\]} $rel -> kind st]} {
      dict set stages $kind $st 1
      lappend pend [list $cell $kind $st]
    } elseif {[regexp {^(g_q|lb_q)\[} $rel]} {
      # The reduction drain mux and line assembler are consumers of the final output stage.
      dict lappend parts $k4fp_rest $cell
    } else {
      error "k4_floorplan: unknown lane-wrapper leaf $cell"
    }
  }
  set last -1
  foreach kind {ip rp op} {
    if {![dict exists $stages $kind]} { error "k4_floorplan: lane pipe $kind missing under $inst" }
    set indices [lsort -integer [dict keys [dict get $stages $kind]]]
    if {[llength $indices] < 2} { error "k4_floorplan: lane pipe $kind has fewer than 2 stages" }
    if {$last < 0} { set last [lindex $indices end] }
    if {[lindex $indices end] != $last} { error "k4_floorplan: lane pipe depths differ under $inst" }
    for {set st 0} {$st <= $last} {incr st} {
      if {![dict exists $stages $kind $st]} { error "k4_floorplan: lane pipe $kind missing stage $st" }
    }
  }
  foreach row $pend {
    lassign $row cell kind st
    set near [expr {$kind eq "op" ? $st == $last : $st == 0}]
    dict lappend parts [expr {$near ? $k4fp_rest : $k4fp_lane}] $cell
  }
  return $parts
}
proc k4fp_assign_parts {parts what} {
  dict for {s cells} $parts { k4fp_add $s $cells $what }
}
proc k4fp_check_cells {s cells placed} {
  global k4fp_pb
  if {[llength $cells] == 0} { error "k4_floorplan: empty locality check for SLR$s" }
  foreach cell $cells {
    # Membership includes a leaf covered by an assigned hierarchy. This is the
    # same query used by the failed-checkpoint census; do not require a leaf
    # PBLOCK property or assume it lists only one enclosing pblock.
    set blocks [get_pblocks -quiet -of_objects $cell]
    if {[llength $blocks] == 0 ||
        [lsearch -exact [get_property NAME $blocks] $k4fp_pb($s)] < 0} {
      error "k4_floorplan: $cell is not assigned to $k4fp_pb($s)"
    }
    if {$placed} {
      set sites [get_sites -quiet -of_objects $cell]
      set slrs [get_slrs -quiet -of_objects $sites]
      if {[llength $sites] == 0 || [get_property NAME $slrs] ne "SLR$s"} {
        error "k4_floorplan: $cell is not placed in SLR$s"
      }
    }
  }
}
proc k4fp_check_locality {{placed 0}} {
  global k4fp_scope
  if {![info exists k4fp_scope]} { error "k4_floorplan: locality scope was not captured" }
  set mvm [dict get $k4fp_scope mvm]
  set vu [dict get $k4fp_scope vu]
  set shared_bc [dict get $k4fp_scope shared_bc]
  if {$shared_bc} { k4fp_check_cells 1 [k4fp_broadcast_head $mvm] $placed }
  if {[dict get $k4fp_scope shared_vu]} {
    set share "$vu/u_lanes/u_share"
    set leaves [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $share/* && REF_NAME != GND && REF_NAME != VCC"]
    k4fp_check_cells 0 $leaves $placed
  }
  set stall [k4fp_att_stall_cone $mvm]
  k4fp_check_cells 1 $stall $placed
  # The reduction's source FFs remain in their original clusters, including
  # replicas added by phys_opt after the hierarchy was partitioned.
  set endpoint [lindex $stall 0]
  set pin [get_pins -quiet "[get_property NAME $endpoint]/D"]
  foreach cell [all_fanin -flat -only_cells -trace_arcs all -to $pin] {
    if {$cell eq $endpoint || ![get_property IS_SEQUENTIAL $cell]} continue
    set found 0
    foreach {c s} [dict get $k4fp_scope map] {
      if {[string first "$mvm/g_cl\[$c\].u_cl/" [get_property NAME $cell]] != 0} continue
      k4fp_check_cells $s [list $cell] $placed
      set found 1
      break
    }
    if {!$found} { error "k4_floorplan: stall source $cell is outside the cluster scope" }
  }
  foreach {c s} [dict get $k4fp_scope map] {
    if {$c >= [dict get $k4fp_scope nch]} continue
    set p "$mvm/g_cl\[$c\]"
    set g "$mvm/u_gather/g_asm\[$c\]"
    set inst "$p.g_ajp.u_ajp"
    set leaves [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $inst/* && REF_NAME != GND && REF_NAME != VCC"]
    k4fp_check_cells $s $leaves $placed
    k4fp_check_cells $s [k4fp_gather_helpers $g] $placed
    if {$shared_bc} {
      set bc_parts [k4fp_broadcast_parts "$p.u_bcp" $s $s 1]
    } else { set bc_parts [k4fp_broadcast_parts "$p.u_bcp" 1 $s] }
    dict for {side cells} $bc_parts { k4fp_check_cells $side $cells $placed }
  }
  dict for {s cells} [k4fp_lane_parts "$vu/u_lanes"] { k4fp_check_cells $s $cells $placed }
  puts "K4FP locality PASS (placed=$placed)"
}

set npin 0
set k4fp_shared_bc [expr {[llength [get_cells -quiet "$mvm/u_bc_head"]] == 1}]
if {$k4fp_shared_bc} {
  k4fp_add $k4fp_rest [k4fp_broadcast_head $mvm] "shared broadcast head"
  puts "K4FP broadcast: shared source stage in SLR1, separate local cluster tails"
}
foreach {c s} $k4fp_map {
  if {$c >= $nch} { continue }
  set p "$mvm/g_cl\[$c\]"
  set g "$mvm/u_gather/g_asm\[$c\]"
  if {!$k4fp_shared_bc} { incr npin [k4fp_pipe_ends "$p.u_bcp" $k4fp_rest $s] }
  incr npin [k4fp_pipe_ends "$p.u_wrp" $k4fp_rest $s]
  incr npin [k4fp_pipe_ends "$p.u_wqp" $s $k4fp_rest]
  incr npin [k4fp_pipe_ends "$p.u_rp"  $s $s]
  incr npin [k4fp_pipe_ends "$p.g_ajp.u_ajp" $s $s]
  k4fp_add $s [k4fp_cell "$p.g_ajp.u_ajp"] "g_cl\[$c\] attention job pipe (local)"
  incr npin [k4fp_pipe_ends "$g.u_sp"  $k4fp_rest $s]
  incr npin [k4fp_jp_ends   "$g.g_jp"  $s $k4fp_rest]
  if {$k4fp_shared_bc} {
    set k4fp_bc_parts [k4fp_broadcast_parts "$p.u_bcp" $s $s 1]
  } else { set k4fp_bc_parts [k4fp_broadcast_parts "$p.u_bcp" $k4fp_rest $s] }
  k4fp_assign_parts $k4fp_bc_parts "g_cl\[$c\] broadcast stages and decode"
  k4fp_add $s [k4fp_gather_helpers $g] "g_asm\[$c\] attention selector and mux"
}
puts "K4FP pipes pinned: $npin stages (shared broadcast head/tails assigned separately when present)"

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
k4fp_assign_parts [k4fp_lane_parts "$vu/u_lanes"] "u_vu lane stages and SPM-facing helpers"

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
# Later implementation hooks may overwrite generic Tcl globals such as nch,
# mvm and vu. Retain this floorplan's scope for read-only post-step checks.
set k4fp_scope [dict create map $k4fp_map nch $nch mvm $mvm vu $vu \
  shared_bc $k4fp_shared_bc shared_vu $k4fp_shared_vu]
k4fp_check_locality
current_instance -quiet

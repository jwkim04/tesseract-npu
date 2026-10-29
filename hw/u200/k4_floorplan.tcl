# hw/u200/k4_floorplan.tcl — SLR floorplan C' of tsr_u200_k4 in the v++ link (docs/platform-u200.md
# §8, docs/rtl-m3.md §14.5): tb/core/viv/core_slr.tcl, the floorplan of the out-of-context runs that
# met 300 MHz (k4 +0.015 ns, k4t1 +0.108 ns), translated to the kernel hierarchy and to the
# platform's dynamic-region pblocks.
#
# Applied by hw/u200/k4.cfg as the place_design pre-hook (run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE):
# after opt_design, where tb/core/viv/core_ooc.tcl sources core_slr.tcl. v++ sources it after its
# own and the platform's hooks of that step (scripts/impl_1/_full_place_pre.tcl).
#
# Pblocks: the platform's, never created here (xilinx_u200_gen3x16_xdma_2_202110_1, hw.xsa
# tcl_hooks/impl.xdc; children of pblock_dynamic_region; platforminfo: DDR bank0 SLR0, bank1 and
# bank2 SLR1, bank3 SLR2). The script stops with an error when one of them is missing.
#   pblock_dynamic_SLR0  CLOCKREGION_X0Y0:CLOCKREGION_X5Y4 minus the isolation pipeline slices
#   pblock_dynamic_SLR1  CLOCKREGION_X0Y5:CLOCKREGION_X2Y9 (the rest of SLR1 is the static shell;
#                        the out-of-context runs used exactly this range as the SLR1 stand-in)
#   pblock_dynamic_SLR2  CLOCKREGION_X0Y10:CLOCKREGION_X5Y14 minus the isolation pipeline slices
#
# Placement (C'):
#   SLR0  MVM clusters 0 and 2 with their result assemblers (u_gather/g_asm[c].u_asm)
#   SLR2  MVM clusters 1 and 3 with their assemblers
#   SLR1  the rest of the core (the four memory arbiters, SPM, central gather, MVM dispatch,
#         DMA, QACT, CSR), the control bridge (u_ctrl), the reset block (u_rst),
#         the bridges of channels 1 and 2 and the core side of the bridges of channels 0 and 3
#   Pipes between a cluster (with its assembler) and the centre (g_cl[c].u_bcp / u_wrp / u_wqp,
#   u_gather/g_asm[c].u_sp / u_jp): the first stage on the source side, the last on the
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
# Prints (grep K4FP): the cells of every group and pblock, the pinned pipe stages (48 at NUM_CH 4)
# and the bridge cells per side.

set k4fp_map  {0 0 1 2 2 0 3 2}       ;# cluster -> SLR (C')
set k4fp_rest 1                       ;# SLR of the rest of the core and of the adapter
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

# ---- core: clusters and assemblers by the map, arbiters and the rest in SLR1 -------------------
set mvm [k4fp_path $k4fp_core g_mvm.u_mvm]
set nch 0
foreach {c s} $k4fp_map {
  set cl [get_cells -quiet "$mvm/g_cl\[$c\].u_cl"]
  if {[llength $cl] == 0} { continue }        ;# NUM_CH < 4: no such cluster
  incr nch
  k4fp_add $s $cl "g_cl\[$c\].u_cl"
  k4fp_add $s [k4fp_cell "$mvm/u_gather/g_asm\[$c\].u_asm"] "g_asm\[$c\].u_asm"
}
if {$nch == 0} { error "k4_floorplan: no MVM cluster under $mvm" }
for {set c 0} {$c < $nch} {incr c} {
  k4fp_add $k4fp_rest [k4fp_cell [k4fp_path $k4fp_core "g_ch\[$c\].u_arb"]] "g_ch\[$c\].u_arb"
}
# X0 (docs/v2 EX §12): u_cmdproc and u_fetch are deleted; Y5 adds the v2 blocks (CC §13) here.
foreach b {u_spm u_dma u_qact u_csr} {
  k4fp_add $k4fp_rest [k4fp_cell [k4fp_path $k4fp_core $b]] $b
}
# central part of the gather: every leaf cell outside the per-cluster assemblers and their pipes
k4fp_add $k4fp_rest [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $mvm/u_gather/* && \
  NAME !~ $mvm/u_gather/g_asm*"] "u_gather (centre)"
# MVM top-level logic (dispatch, broadcast source, SPM client mux): the leaf cells directly in
# g_mvm.u_mvm outside the clusters, pipes, gather and reset copies
k4fp_add $k4fp_rest [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ $mvm/* && \
  NAME !~ $mvm/g_cl* && NAME !~ $mvm/u_gather/* && NAME !~ $mvm/g_rst*"] "u_mvm (top)"

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
set npin 0
foreach {c s} $k4fp_map {
  if {$c >= $nch} { continue }
  set p "$mvm/g_cl\[$c\]"
  set g "$mvm/u_gather/g_asm\[$c\]"
  incr npin [k4fp_pipe_ends "$p.u_bcp" $k4fp_rest $s]
  incr npin [k4fp_pipe_ends "$p.u_wrp" $k4fp_rest $s]
  incr npin [k4fp_pipe_ends "$p.u_wqp" $s $k4fp_rest]
  incr npin [k4fp_pipe_ends "$p.u_rp"  $s $s]
  incr npin [k4fp_pipe_ends "$g.u_sp"  $k4fp_rest $s]
  incr npin [k4fp_pipe_ends "$g.u_jp"  $s $k4fp_rest]
}
puts "K4FP pipes pinned: $npin stages (first / last stage of the 6 pipes of each of $nch clusters)"

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

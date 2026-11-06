# hw/u200/k4_major_blocks.tcl — Pblocks of whole major functional blocks for tsr_u200_k4, the reduced first card
# image (docs/platform-u200.md §8.7). Applied by hw/u200/k4.cfg as the place_design pre-hook; for the
# out-of-context core run by tb/core/viv/core_slr.tcl (TSR_K4_FP_CORE set: a bare tsr_core, no adapter).
#
# The elaboration has four MVM clusters (T_MAX 8), the RISC-V control core, the issue path, the CSR block, the
# 2R2W SPM, DMA without KVST, QACT and the DB block without trace writer and history. It has no vector unit, no
# attention (ATTQK / ATTPV / KVST) and no local-intermediate path, so there is no VU bank, VU front end or
# local attention block to place. A block that is absent here is not in the netlist. The map of the full
# elaboration is hw/u200/k4_major_blocks_full.tcl.
#
# Each block goes whole into one Pblock: the platform's pblock of an SLR, or the control Pblock below. Nothing here
# names a LUT, a register, a pipeline stage, a site or a BEL. Vivado places and routes inside the regions.
# hw/viz/k4_pblock_preview.py draws the map from this file.
#
#   SLR0  clusters 0 and 2 with their result assemblers; the DDR0 bridge
#   SLR0  control Pblock (pblock_tsr_ctrl_plane, below): the control core (Ibex, debug module, TCMs, bus, control
#         window, every clock-domain crossing, the RV side of the DB block), the issue path, the CSR block and
#         the DB block's core side, with the issue and LCQ ends of the control links (TSR_K4_CTRL, below)
#   SLR2  clusters 1 and 3 with their result assemblers; the DDR3 bridge
#   SLR1  the data path: SPM, the RISC-V SPM bridge (and the SPM end of its link), DMA, QACT and the four memory
#         arbiters; the central gather, the DDR1 / DDR2 bridges and the adapter's control logic are left to Vivado
#
# Why the control plane is in SLR0: the placed out-of-context core (docs/platform-u200.md §8.7) puts 9,292 CLB sites
# in SLR1 with issue / CSR / DB there (estimated SLR1 occupancy with the shell 94.7 % or more; the routed
# 2026-10-25 image had 92.1 %) and 6,122 without them (88 to 92 %).
#
# Why it has a Pblock of its own: in the first link of the image (2026-11-06) the control plane, 7,600 CLBs, sat
# at X0 to X3 of SLR0, partly in the DDR0 controller's clock regions X2Y1 to X2Y3, which were 99.5 % full and
# failed their clock by 0.33 ns (docs/platform-u200.md §8.7). The control Pblock is a child of
# pblock_dynamic_SLR0: the two rows at the SLR1 boundary, Y3 and Y4, of all six clock-region columns, without
# X2Y3. It
#   - leaves out X2Y1 to X2Y3, the DDR0 controller, and the rows below it, where cluster 0 has its home;
#   - holds all of row Y4 at X0 to X2, under SLR1's dynamic Pblock (X0 to X2), where SPM, DMA and QACT are;
#   - has 16,300 CLBs, 252 RAMB36 and 840 DSP in the checkpoint's platform: the plane (7,600 CLBs, 48 RAMB36,
#     8 DSP) and the platform's own 2,600 CLBs fill 63 %, and 85 % once about 4,500 slices of other blocks
#     (cluster 0 and 2 and the result assemblers) have moved out of it.
# The Pblock does not exclude other logic: clusters and DSP / BRAM columns inside it stay usable. Cluster 0 stays
# in SLR0 around DDR0, and the slices the plane leaves in X0 to X2 of Y1 to Y2 are free for it.
#
# Environment (validation outside the link only): TSR_K4_FP_CORE (hierarchical name of the tsr_core, "" = the top
# cell: the out-of-context run), TSR_K4_FP_BODY (name of the tsr_xrt_kernel instance; default: the tsr_u200_k4
# instance found by REF_NAME, else by the compute-unit name */tsr_k4/inst), TSR_K4_CTRL (where the issue path,
# the CSR block and the DB block go: slr0 = default, in the control Pblock beside the control core; slr1 =
# beside the SPM; free = unassigned).
current_instance -quiet

proc tsr_k4_path {base rel} {
  if {$base eq ""} { return $rel }
  return "$base/$rel"
}

if {[info exists ::env(TSR_K4_FP_CORE)]} {
  set tsr_k4_kernel 0
  set tsr_k4_body ""
  set tsr_k4_core $::env(TSR_K4_FP_CORE)
} else {
  set tsr_k4_kernel 1
  if {[info exists ::env(TSR_K4_FP_BODY)]} {
    set tsr_k4_body $::env(TSR_K4_FP_BODY)
  } else {
    set tsr_k4_instances [get_cells -quiet -hierarchical -filter {
      REF_NAME == tsr_u200_k4 || ORIG_REF_NAME == tsr_u200_k4
    }]
    if {[llength $tsr_k4_instances] == 0} {
      set tsr_k4_instances [get_cells -quiet -hierarchical -filter {
        NAME =~ */tsr_k4/inst && !IS_PRIMITIVE
      }]
    }
    if {[llength $tsr_k4_instances] != 1} {
      error "K4 major blocks: expected one tsr_u200_k4 instance"
    }
    set tsr_k4_body "[get_property NAME $tsr_k4_instances]/u_kernel"
  }
  set tsr_k4_core [tsr_k4_path $tsr_k4_body u_core]
}
set tsr_k4_dp  [tsr_k4_path $tsr_k4_core u_dp]
set tsr_k4_mvm "$tsr_k4_dp/g_mvm.u_mvm"
set tsr_k4_ctrl_plane slr0
if {[info exists ::env(TSR_K4_CTRL)]} { set tsr_k4_ctrl_plane $::env(TSR_K4_CTRL) }
if {$tsr_k4_ctrl_plane ni {slr0 slr1 free}} {
  error "K4 major blocks: TSR_K4_CTRL must be slr0, slr1 or free"
}

# The control Pblock: a child of the SLR0 platform pblock. Clock-region ranges only. The platform pblock cuts rows
# of X0 to X5 in Y4 (its pipeline Pblocks); with SNAPPING_MODE NESTED Vivado resizes the child to the parent when
# the first cell is assigned (INFO Constraints 18-12550), so the capacity is read after the assignments.
proc tsr_k4_ctrl_pblock_name {} { return pblock_tsr_ctrl_plane }
set tsr_k4_ctrl_ranges {
  CLOCKREGION_X0Y3:CLOCKREGION_X1Y4
  CLOCKREGION_X2Y4:CLOCKREGION_X2Y4
  CLOCKREGION_X3Y3:CLOCKREGION_X5Y4
}
# The TCMs are 48 RAMB36 (CC §14.2); the DB block and the rest of the plane need a few more.
set tsr_k4_ctrl_min_ramb36 64

proc tsr_k4_ctrl_pblock_make {name parent_name ranges} {
  set parent [get_pblocks -quiet $parent_name]
  if {[llength $parent] != 1} { error "K4 major blocks: missing platform pblock $parent_name" }
  set pb [get_pblocks -quiet $name]
  if {[llength $pb] == 0} {
    set pb [create_pblock $name]
    foreach r $ranges { resize_pblock $pb -add $r }
    set_property PARENT $parent $pb
  }
  if {[llength $pb] != 1 || [get_property PARENT $pb] ne $parent_name} {
    error "K4 major blocks: pblock $name is not a child of $parent_name"
  }
  return $pb
}
tsr_k4_ctrl_pblock_make [tsr_k4_ctrl_pblock_name] pblock_dynamic_SLR0 $tsr_k4_ctrl_ranges

# The sites the control Pblock holds once Vivado has fitted it to its parent. Sites are counted by name: an occupied
# RAMB36 site of a placed checkpoint no longer reports the type RAMBFIFO36.
proc tsr_k4_ctrl_pblock_check {name min_ramb36} {
  set pb [get_pblocks -quiet $name]
  set slices [llength [get_sites -quiet -of_objects $pb -filter {NAME =~ SLICE_*}]]
  set ramb36 [llength [get_sites -quiet -of_objects $pb -filter {NAME =~ RAMB36_*}]]
  if {$slices == 0} { error "K4 major blocks: pblock $name has no SLICE sites" }
  if {$ramb36 < $min_ramb36} {
    error "K4 major blocks: pblock $name has $ramb36 RAMB36 sites, the control plane needs $min_ramb36"
  }
  puts "K4 major blocks: $name: $slices SLICE, $ramb36 RAMB36 sites"
}

# where: 0, 1 or 2 = the platform pblock of that SLR; ctrl = the control Pblock.
proc tsr_k4_major_block {where path} {
  if {$where eq "ctrl"} {
    set name [tsr_k4_ctrl_pblock_name]
  } else {
    set name pblock_dynamic_SLR$where
  }
  set pblock [get_pblocks -quiet $name]
  set cell [get_cells -quiet $path]
  if {[llength $pblock] != 1} {
    error "K4 major blocks: missing pblock $name"
  }
  if {[llength $cell] != 1 || [get_property IS_PRIMITIVE $cell]} {
    error "K4 major blocks: missing hierarchy $path"
  }
  add_cells_to_pblock $pblock $cell
}

# Clusters and their result assemblers: 0 and 2 beside DDR0, 1 and 3 beside DDR3.
foreach {cluster slr} {0 0 1 2 2 0 3 2} {
  tsr_k4_major_block $slr "$tsr_k4_mvm/g_cl\[$cluster\].u_cl"
  tsr_k4_major_block $slr "$tsr_k4_mvm/u_gather/g_asm\[$cluster\].u_asm"
}

# The AXI bridges of the outer DDR banks (the kernel body only; the bare core ends at the memory ports).
if {$tsr_k4_kernel} {
  foreach {channel slr} {0 0 3 2} {
    tsr_k4_major_block $slr "$tsr_k4_body/g_ch\[$channel\].u_br"
  }
}

# Control core, TCMs, local bus, clock-domain crossings and the RV side of the DB block stay together in the
# control Pblock. Only the three narrow links below face the central SPM, issue path and CSR block.
tsr_k4_major_block ctrl [tsr_k4_path $tsr_k4_core u_ctrl]
tsr_k4_major_block ctrl [tsr_k4_path $tsr_k4_core g_db.u_dbg_rv]
foreach endpoint {u_spm_rx u_issue_tx u_lq_tx} {
  tsr_k4_major_block ctrl [tsr_k4_path $tsr_k4_core g_rv_local.$endpoint]
}
# The SPM link ends at the SPM bridge in SLR1. The issue and LCQ links end at the issue path and the CSR block:
# in the control Pblock with them when they are beside the control core, else in SLR1.
set tsr_k4_ctrl_slr [expr {$tsr_k4_ctrl_plane eq "slr0" ? "ctrl" : 1}]
tsr_k4_major_block 1 [tsr_k4_path $tsr_k4_core g_rv_local.u_spm_tx]
foreach endpoint {u_issue_rx u_lq_rx} {
  tsr_k4_major_block $tsr_k4_ctrl_slr [tsr_k4_path $tsr_k4_core g_rv_local.$endpoint]
}

# Central data path: SPM and the RISC-V bridge into it, DMA, QACT and the memory arbiters.
foreach block {u_spm u_bridge u_dma u_qact} {
  tsr_k4_major_block 1 "$tsr_k4_dp/$block"
}
for {set channel 0} {$channel < 4} {incr channel} {
  tsr_k4_major_block 1 "$tsr_k4_dp/g_ch\[$channel\].u_arb"
}

# Issue path, CSR block and the DB block's core side.
if {$tsr_k4_ctrl_plane ne "free"} {
  foreach block {u_issue u_csr g_db.u_dbg} {
    tsr_k4_major_block $tsr_k4_ctrl_slr "$tsr_k4_dp/$block"
  }
}
tsr_k4_ctrl_pblock_check [tsr_k4_ctrl_pblock_name] $tsr_k4_ctrl_min_ramb36
puts "K4 major blocks: clusters and assemblers 0/2 in SLR0 and 1/3 in SLR2; control core in the SLR0 control Pblock; SPM, DMA, QACT, arbiters in SLR1; issue/CSR/DB $tsr_k4_ctrl_plane; central gather and individual stages free"

# hw/u200/k4_major_blocks.tcl — Pblocks of whole major functional blocks for tsr_u200_k4, the full-feature card
# image (docs/platform-u200.md §8.8). Applied by hw/u200/k4.cfg as the place_design pre-hook; for the
# out-of-context core run by tb/core/viv/core_slr.tcl (TSR_K4_FP_CORE set: a bare tsr_core, no adapter).
#
# The elaboration has four MVM clusters (T_MAX 4) with attention, the vector unit (8 / 8 / 4 lanes, two lane banks),
# the RISC-V control core behind the narrow RV transport, the issue path with both checker classes, the CSR block,
# the 2R2W SPM, DMA with KVST, QACT, the DB block with trace writer and history, the paired two-flit gather and the
# DDR0 / DDR3 arbitration boundaries. It has no local-intermediate bank (LOCAL_INTERMEDIATES, LOCAL_SCALE,
# LOCAL_SOFTCAP 0). A block that is absent from the netlist is not named here; a block that is named and absent
# stops the hook.
#
# Each block goes whole into one Pblock: the platform's pblock of an SLR, or one of the two Pblocks below. Nothing
# here names a LUT, a register, a pipeline stage, a site or a BEL. Vivado places and routes inside the regions.
# hw/viz/k4_pblock_preview.py draws the map from this file.
#
#   SLR0  clusters 0 and 2 with their result assemblers (each cluster holds both of its weight streams and stream B has an
#         own row assembler); the pair source of clusters 0 and 2; the DDR0 bridge, the read-only bridge of the second weight
#         stream of channel 0 and the DDR0 arbiter
#   SLR0  control Pblock (pblock_tsr_ctrl_plane, below): the control core (Ibex, debug module, TCMs, bus, control
#         window, every clock-domain crossing, the RV side of the DB block), the issue path with its two checker
#         classes, the CSR block and the DB block's core side, with the issue and LCQ ends of the control links
#         (TSR_K4_CTRL, below)
#   SLR2  clusters 1 and 3 with their result assemblers (both streams); the pair source of clusters 1 and 3; the DDR3 bridge,
#         the read-only bridge of the second weight stream of channel 3 and the DDR3 arbiter
#   SLR2  vector unit Pblock (pblock_tsr_vu, below): the whole vector unit, front end and both lane banks
#         (TSR_K4_VU, below)
#   SLR1  the data hub: SPM, the RISC-V SPM bridge (and the SPM end of its link), DMA with KVST, QACT, the
#         DDR1 / DDR2 arbiters and bridges (with the read-only bridges of the second weight stream of channels 1
#         and 2) and the central gather receiver (the receivers of the two pair links, the row-maximum emitter and
#         the accumulate adders); the stages around the receiver, the DMA-side boundary slices of channels 0 and
#         3, the request and response stages of the second weight stream in the core's data path and the
#         adapter's control logic are left to Vivado
#
# Why the DDR1 / DDR2 bridges and the central gather receiver are named: a link that left them free placed both
# bridges and the receiver in SLR2. Each bridge then crossed the SLR1 / SLR2 boundary about 2,300 times (toward
# its arbiter and toward the memory subsystem, both in SLR1) and the receiver about 1,300 times (the SPM read
# line into the adders and the SPM write line out of them). The boundary needed 15,863 of its 17,280 SLLs and
# the router could not resolve the demand. In SLR1 these blocks reach the SPM, the arbiters and the DDR1 /
# DDR2 banks without a crossing; the pair links (two flits) and the weight streams of clusters 1 and 2 are
# what crosses.
#
# Why the control plane is in SLR0: the placed out-of-context core puts 9,292 CLB sites
# in SLR1 with issue / CSR / DB there (estimated SLR1 occupancy with the shell 94.7 % or more; an earlier
# routed image had 92.1 %) and 6,122 without them (88 to 92 %). The census of the full runs and of the reduced
# image gives the same split: the control plane, 5.5 to 6.4 K CLBs, is the difference
# between the full runs, with SLR1 at 98.5 to 99.9 %, and the reduced image that routed, with 90.7 %. The two
# checker classes of the issue path (g_xc.u_xc, the VU class, and g_xc.g_at.u_chk_at, the AT class; 0.7 to 0.8 K
# CLBs) belong with the issue path and go with it.
#
# Why the whole vector unit is in SLR2: with the front end in SLR1 it takes 4.4 K CLBs of the SLR1 strip (31 K LUT
# cells); with the front end in SLR0 and a bank in SLR2 the compact packet crosses SLR1 twice (the worst full run,
# 35,335 route overlaps). One SLR for all of it leaves only its SPM ports (two reads, one write, 512 bits each) and
# its command, done and register links to cross one boundary toward SLR1 and two toward the control plane. It is
# SLR2 because the control plane takes SLR0.
#
# Why the two Pblocks of their own: in the first link of the reduced image the control plane,
# 7,600 CLBs, sat at X0 to X3 of SLR0, partly in the DDR0 controller's clock regions X2Y1 to X2Y3, which were
# 99.5 % full and failed their clock by 0.33 ns (the controller footprints are in docs/platform-u200.md §8.7). The SLR2 controller (MIG 2, DDR[3]) is
# at X3Y10 to X4Y13 (76 to 96 % full, -0.065 ns in the same link). Both Pblocks are children of the platform's SLR
# pblock and keep a large block off the controller's clock regions.
#   control Pblock  the two rows at the SLR1 boundary, Y3 and Y4, of all six clock-region columns, without X2Y3:
#                   - leaves out X2Y1 to X2Y3, the DDR0 controller, and the rows below it, where cluster 0 has its home;
#                   - holds all of row Y4 at X0 to X2, under SLR1's dynamic Pblock (X0 to X2), where SPM, DMA and
#                     QACT are;
#                   - has 16,300 CLBs, 252 RAMB36 and 840 DSP in the checkpoint's platform: the plane (7,600 CLBs,
#                     48 RAMB36, 8 DSP) and the platform's own 2,600 CLBs fill 63 %, and about 72 % with the two
#                     checkers and the full DB block (about 1.4 K CLBs, the SLR1 study's estimate).
#   VU Pblock       X0Y10 to X2Y14 of SLR2: the three left columns of all five rows, next to the SPM across the
#                   SLR1 boundary. It leaves out the DDR3 controller (X3Y10 to X3Y11, X4Y10 to X4Y13) and the
#                   columns beyond it, and has 25,320 CLBs and 420 RAMB36. The vector unit synthesizes to 60,104 LUT
#                   cells, 68,799 registers, 16 RAMB36, 13 RAMB18 and 249 DSP (out-of-context core, docs/platform-u200.md
#                   §8.8): 10.6 to 13.8 K CLBs at 0.176 to 0.229 CLBs per LUT cell, 42 to 55 % of the region.
# A Pblock does not exclude other logic: clusters and DSP / BRAM columns inside it stay usable, and the slices a
# block leaves free are free for the others.
#
# Environment (validation outside the link only): TSR_K4_FP_CORE (hierarchical name of the tsr_core, "" = the top
# cell: the out-of-context run), TSR_K4_FP_BODY (name of the tsr_xrt_kernel instance; default: the tsr_u200_k4
# instance found by REF_NAME, else by the compute-unit name */tsr_k4/inst), TSR_K4_CTRL (where the issue path with
# its checkers, the CSR block and the DB block go: slr0 = default, in the control Pblock beside the control core;
# slr1 = beside the SPM; free = unassigned), TSR_K4_VU (where the vector unit goes: pblock = default, the VU
# Pblock; slr2 = the platform pblock of SLR2), TSR_K4_SKIP_MISSING (set: a named block that is not in the netlist is
# reported and skipped instead of stopping the hook; only tb/platform/viv/k4_pblock_check.tcl sets it, to check the
# Pblocks on the checkpoint of an image that lacks some blocks).
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
set tsr_k4_vu_where pblock
if {[info exists ::env(TSR_K4_VU)]} { set tsr_k4_vu_where $::env(TSR_K4_VU) }
if {$tsr_k4_vu_where ni {pblock slr2}} {
  error "K4 major blocks: TSR_K4_VU must be pblock or slr2"
}

# The two child Pblocks: children of the platform's SLR pblocks, clock-region ranges only. The platform pblock cuts
# rows of X0 to X5 in Y4 (its pipeline Pblocks); with SNAPPING_MODE NESTED Vivado resizes a child to the parent when
# the first cell is assigned (INFO Constraints 18-12550), so the capacity is read after the assignments.
proc tsr_k4_ctrl_pblock_name {} { return pblock_tsr_ctrl_plane }
proc tsr_k4_vu_pblock_name {} { return pblock_tsr_vu }
set tsr_k4_ctrl_ranges {
  CLOCKREGION_X0Y3:CLOCKREGION_X1Y4
  CLOCKREGION_X2Y4:CLOCKREGION_X2Y4
  CLOCKREGION_X3Y3:CLOCKREGION_X5Y4
}
set tsr_k4_vu_ranges {
  CLOCKREGION_X0Y10:CLOCKREGION_X2Y14
}
# The TCMs are 48 RAMB36 (CC §14.2); the DB block and the rest of the plane need a few more.
set tsr_k4_ctrl_min_ramb36 64
# The vector unit needs 16 RAMB36 and 13 RAMB18 (docs/platform-u200.md §8.8); the region has 420.
set tsr_k4_vu_min_ramb36 32

proc tsr_k4_pblock_make {name parent_name ranges} {
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
tsr_k4_pblock_make [tsr_k4_ctrl_pblock_name] pblock_dynamic_SLR0 $tsr_k4_ctrl_ranges
if {$tsr_k4_vu_where eq "pblock"} {
  tsr_k4_pblock_make [tsr_k4_vu_pblock_name] pblock_dynamic_SLR2 $tsr_k4_vu_ranges
}

# The sites a child Pblock holds once Vivado has fitted it to its parent. Sites are counted by name: an occupied
# RAMB36 site of a placed checkpoint no longer reports the type RAMBFIFO36.
proc tsr_k4_pblock_check {name min_ramb36 what} {
  set pb [get_pblocks -quiet $name]
  set slices [llength [get_sites -quiet -of_objects $pb -filter {NAME =~ SLICE_*}]]
  set ramb36 [llength [get_sites -quiet -of_objects $pb -filter {NAME =~ RAMB36_*}]]
  if {$slices == 0} { error "K4 major blocks: pblock $name has no SLICE sites" }
  if {$ramb36 < $min_ramb36} {
    error "K4 major blocks: pblock $name has $ramb36 RAMB36 sites, the $what needs $min_ramb36"
  }
  puts "K4 major blocks: $name: $slices SLICE, $ramb36 RAMB36 sites"
}

# where: 0, 1 or 2 = the platform pblock of that SLR; ctrl = the control Pblock; vu = the VU Pblock.
proc tsr_k4_major_block {where path} {
  if {$where eq "ctrl"} {
    set name [tsr_k4_ctrl_pblock_name]
  } elseif {$where eq "vu"} {
    set name [tsr_k4_vu_pblock_name]
  } else {
    set name pblock_dynamic_SLR$where
  }
  set pblock [get_pblocks -quiet $name]
  set cell [get_cells -quiet $path]
  if {[llength $pblock] != 1} {
    error "K4 major blocks: missing pblock $name"
  }
  if {[llength $cell] != 1 || [get_property IS_PRIMITIVE $cell]} {
    if {[info exists ::env(TSR_K4_SKIP_MISSING)]} {
      puts "K4 major blocks: skipped, not in this netlist: $path"
      return
    }
    error "K4 major blocks: missing hierarchy $path"
  }
  add_cells_to_pblock $pblock $cell
}

# The blocks of the dual weight stream (MVM_DUAL, docs/rtl-mvm.md §15): the row assembler of stream B of every cluster and,
# in the kernel, the read-only bridge of every DDR channel. A netlist without the stream has none of them and the
# hook says so. A netlist with some of them and not all of them stops it.
set tsr_k4_dual_paths [list]
foreach cluster {0 1 2 3} {
  lappend tsr_k4_dual_paths "$tsr_k4_mvm/u_gather/g_asm\[$cluster\].g_b.u_asm_b"
}
if {$tsr_k4_kernel} {
  foreach channel {0 1 2 3} { lappend tsr_k4_dual_paths "$tsr_k4_body/g_dual.g_chb\[$channel\].u_br" }
}
set tsr_k4_dual_have 0
foreach path $tsr_k4_dual_paths {
  if {[llength [get_cells -quiet $path]] == 1} { incr tsr_k4_dual_have }
}
if {$tsr_k4_dual_have != 0 && $tsr_k4_dual_have != [llength $tsr_k4_dual_paths] && ![info exists ::env(TSR_K4_SKIP_MISSING)]} {
  error "K4 major blocks: $tsr_k4_dual_have of [llength $tsr_k4_dual_paths] blocks of the dual weight stream are in the netlist"
}
set tsr_k4_has_dual [expr {$tsr_k4_dual_have != 0}]

# Clusters and their result assemblers: 0 and 2 beside DDR0, 1 and 3 beside DDR3. A pair source collects the results
# of one such pair (pair 0: clusters 0 and 2, pair 1: clusters 1 and 3) and ends in the two-flit link to the
# central receiver. The row assembler of stream B goes with the assembler of stream A.
foreach {cluster slr} {0 0 1 2 2 0 3 2} {
  tsr_k4_major_block $slr "$tsr_k4_mvm/g_cl\[$cluster\].u_cl"
  tsr_k4_major_block $slr "$tsr_k4_mvm/u_gather/g_asm\[$cluster\].u_asm"
  if {$tsr_k4_has_dual} { tsr_k4_major_block $slr "$tsr_k4_mvm/u_gather/g_asm\[$cluster\].g_b.u_asm_b" }
}
foreach {pair slr} {0 0 1 2} {
  tsr_k4_major_block $slr "$tsr_k4_mvm/u_gather/g_pairs.g_pair\[$pair\].u_source"
}

# The AXI bridges of the DDR banks (the kernel body only; the bare core ends at the memory ports): the bridge of
# the read-write port and the read-only bridge of the second weight stream (MVM_DUAL) of the channel, in the SLR
# of the channel's bank and arbiter.
if {$tsr_k4_kernel} {
  foreach {channel slr} {0 0 1 1 2 1 3 2} {
    tsr_k4_major_block $slr "$tsr_k4_body/g_ch\[$channel\].u_br"
    if {$tsr_k4_has_dual} { tsr_k4_major_block $slr "$tsr_k4_body/g_dual.g_chb\[$channel\].u_br" }
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

# Central data hub: SPM and the RISC-V bridge into it, DMA with KVST, QACT. The arbiters of the outer channels 0
# and 3 sit with their DDR bridges; those of channels 1 and 2 stay in SLR1.
foreach block {u_spm u_bridge u_dma u_qact} {
  tsr_k4_major_block 1 "$tsr_k4_dp/$block"
}
foreach {channel slr} {0 0 1 1 2 1 3 2} {
  tsr_k4_major_block $slr "$tsr_k4_dp/g_ch\[$channel\].u_arb"
}

# The central gather receiver beside the SPM it reads and writes: the receivers of the two pair links, the
# row-maximum emitter and the 16 accumulate adders. Its issue slice, pipes and line registers are not named.
foreach pair {0 1} {
  tsr_k4_major_block 1 "$tsr_k4_mvm/u_gather/g_pairs.g_pair\[$pair\].g_narrow.u_remote"
}
tsr_k4_major_block 1 "$tsr_k4_mvm/u_gather/g_mx.u_rv"
for {set tsr_k4_lane 0} {$tsr_k4_lane < 16} {incr tsr_k4_lane} {
  tsr_k4_major_block 1 "$tsr_k4_mvm/u_gather/g_add\[$tsr_k4_lane\].u_add"
}

# Issue path with its checker classes, CSR block and the DB block's core side.
if {$tsr_k4_ctrl_plane ne "free"} {
  foreach block {u_issue u_csr g_db.u_dbg g_xc.u_xc g_xc.g_at.u_chk_at} {
    tsr_k4_major_block $tsr_k4_ctrl_slr "$tsr_k4_dp/$block"
  }
}

# The whole vector unit, front end and both lane banks, in SLR2.
tsr_k4_major_block [expr {$tsr_k4_vu_where eq "pblock" ? "vu" : 2}] "$tsr_k4_dp/g_vu.u_vu"

tsr_k4_pblock_check [tsr_k4_ctrl_pblock_name] $tsr_k4_ctrl_min_ramb36 "control plane"
if {$tsr_k4_vu_where eq "pblock"} {
  tsr_k4_pblock_check [tsr_k4_vu_pblock_name] $tsr_k4_vu_min_ramb36 "vector unit"
}
puts "K4 major blocks: dual weight stream blocks in the netlist: $tsr_k4_has_dual"
puts "K4 major blocks: clusters, assemblers and pair sources 0/2 in SLR0 and 1/3 in SLR2; control core, issue path with checkers, CSR and DB in the control Pblock ($tsr_k4_ctrl_plane); whole vector unit in SLR2 ($tsr_k4_vu_where); SPM, DMA, QACT, the arbiters and bridges of channels 1 and 2 and the central gather receiver in SLR1; individual stages free"

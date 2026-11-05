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
# Each block goes whole into the platform's pblock of one SLR; nothing here names a LUT, a register, a pipeline
# stage, a site or a BEL. Vivado places and routes inside the regions. hw/viz/k4_pblock_preview.py draws the map
# from this file.
#
#   SLR0  clusters 0 and 2 with their result assemblers; the DDR0 bridge; the control plane: the control core
#         (Ibex, debug module, TCMs, bus, control window, every clock-domain crossing, the RV side of the DB
#         block), the issue path, the CSR block and the DB block's core side, with the issue and LCQ ends of the
#         control links (TSR_K4_CTRL, below)
#   SLR2  clusters 1 and 3 with their result assemblers; the DDR3 bridge
#   SLR1  the data path: SPM, the RISC-V SPM bridge (and the SPM end of its link), DMA, QACT and the four memory
#         arbiters; the central gather, the DDR1 / DDR2 bridges and the adapter's control logic are left to Vivado
#
# Why the control plane is in SLR0: the placed out-of-context core (docs/platform-u200.md §8.7) puts 9,292 CLB sites
# in SLR1 with issue / CSR / DB there (estimated SLR1 occupancy with the shell 94.7 % or more; the routed
# 2026-10-25 image had 92.1 %) and 6,122 without them (88 to 92 %).
#
# Environment (validation outside the link only): TSR_K4_FP_CORE (hierarchical name of the tsr_core, "" = the top
# cell: the out-of-context run), TSR_K4_FP_BODY (name of the tsr_xrt_kernel instance; default: the tsr_u200_k4
# instance found by REF_NAME, else by the compute-unit name */tsr_k4/inst), TSR_K4_CTRL (where the issue path,
# the CSR block and the DB block go: slr0 = default, beside the control core; slr1 = beside the SPM; free =
# unassigned).
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

proc tsr_k4_major_block {slr path} {
  set pblock [get_pblocks -quiet pblock_dynamic_SLR$slr]
  set cell [get_cells -quiet $path]
  if {[llength $pblock] != 1} {
    error "K4 major blocks: missing platform SLR$slr pblock"
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

# Control core, TCMs, local bus, clock-domain crossings and the RV side of the DB block stay together in SLR0.
# Only the three narrow links below face the central SPM, issue path and CSR block.
tsr_k4_major_block 0 [tsr_k4_path $tsr_k4_core u_ctrl]
tsr_k4_major_block 0 [tsr_k4_path $tsr_k4_core g_db.u_dbg_rv]
foreach endpoint {u_spm_rx u_issue_tx u_lq_tx} {
  tsr_k4_major_block 0 [tsr_k4_path $tsr_k4_core g_rv_local.$endpoint]
}
# The SPM link ends at the SPM bridge in SLR1. The issue and LCQ links end at the issue path and the CSR block:
# in SLR0 with them when they are beside the control core, else in SLR1.
set tsr_k4_ctrl_slr [expr {$tsr_k4_ctrl_plane eq "slr0" ? 0 : 1}]
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
puts "K4 major blocks: clusters and assemblers 0/2 in SLR0 and 1/3 in SLR2; control core in SLR0; SPM, DMA, QACT, arbiters in SLR1; issue/CSR/DB $tsr_k4_ctrl_plane; central gather and individual stages free"

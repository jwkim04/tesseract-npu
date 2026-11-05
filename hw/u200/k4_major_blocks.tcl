# K4 placement uses whole major blocks and the platform's existing SLR pblocks.
# Local intermediate consumers and complete pair-source endpoints follow their
# owner groups. Central gather and individual pipeline stages remain free to place.
# This candidate map requires LOCAL_INTERMEDIATES=1 in the eventual K4 image.
current_instance -quiet
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
set tsr_k4_core "$tsr_k4_body/u_core"
set tsr_k4_dp "$tsr_k4_core/u_dp"
set tsr_k4_mvm "$tsr_k4_dp/g_mvm.u_mvm"
set tsr_k4_vu "$tsr_k4_dp/g_vu.u_vu"

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

foreach {cluster slr} {0 0 1 2 2 0 3 2} {
  tsr_k4_major_block $slr "$tsr_k4_mvm/g_cl\[$cluster\].u_cl"
  tsr_k4_major_block $slr "$tsr_k4_mvm/u_gather/g_asm\[$cluster\].u_asm"
}
foreach {bank slr} {0 0 1 2} {
  tsr_k4_major_block $slr "$tsr_k4_vu/u_lanes/g_bank\[$bank\].u_bank"
  tsr_k4_major_block $slr "$tsr_k4_vu/g_local_att.g_pair\[$bank\].u_local_asmax"
  tsr_k4_major_block $slr "$tsr_k4_mvm/u_gather/g_pairs.g_pair\[$bank\].u_source"
}
foreach {channel slr} {0 0 3 2} {
  tsr_k4_major_block $slr "$tsr_k4_body/g_ch\[$channel\].u_br"
  tsr_k4_major_block $slr "$tsr_k4_dp/g_ch\[$channel\].u_arb"
}
tsr_k4_major_block 1 "$tsr_k4_dp/u_spm"
tsr_k4_major_block 1 "$tsr_k4_dp/u_bridge"
# Keep the RV, TCMs, local bus and their CDC together. Only the transport
# endpoints below face the central SPM and issue/LCQ consumers.
tsr_k4_major_block 0 "$tsr_k4_core/u_ctrl"
tsr_k4_major_block 0 "$tsr_k4_core/g_db.u_dbg_rv"
foreach endpoint {u_spm_rx u_issue_tx u_lq_tx} {
  tsr_k4_major_block 0 "$tsr_k4_core/g_rv_local.$endpoint"
}
foreach endpoint {u_spm_tx u_issue_rx u_lq_rx} {
  tsr_k4_major_block 1 "$tsr_k4_core/g_rv_local.$endpoint"
}
# Keep SPM-facing VU selection beside SPM. Each outer lane bank owns its
# compressed-packet expansion and coefficient stores. Writer/reduction and
# individual stages remain under native placement.
foreach block {u_gen u_rd u_ang u_inj u_lma u_lmb} {
  tsr_k4_major_block 1 "$tsr_k4_vu/$block"
}
tsr_k4_major_block 1 "$tsr_k4_dp/u_dma"
puts "K4 major blocks: RV/TCM/debug in SLR0; VU banks, local attention and complete pair-source collectors in SLR0/2; SPM, VU frontend and DMA in SLR1; central gather/RX and individual stages free"

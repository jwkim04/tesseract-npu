# rtl/ctrl/viv/tsr_cdc_waivers.tcl — the report_cdc waivers of the control core and the DB block (CC §11.3:
# only the intended crossings, all safe). Vivado does not see these protocols and reports them as "unknown"
# structures. Tcl, not XDC (it needs loops and procedures): sourced after rtl/ctrl/viv/tsr_ctrl_cdc.xdc by
# the out-of-context scripts (tb/core/viv/core_ooc.tcl, rtl/ctrl/viv/ctrl_ooc.tcl) and by the post-route gate
# of the link (hw/u200/cdc_check.tcl). A waiver only changes what report_cdc lists, never the timing.
#
#  - tsr_cdc_hs request / response registers (req_q, rsp_q): held stable from the toggle that announces
#    them until the toggle that frees them; the other domain loads them into an enabled register only
#    after its synchronised copy of the toggle changed (CC §11.2).
#  - tsr_cdc_fifo LUTRAM (mem): an entry is read only behind the synchronised gray write pointer that
#    covers it, and rewritten only after the gray read pointer freed it. Every FIFO instance is found by
#    its write pointer register (wgray_q), whatever the instance is called.
#  - gray-coded buses into their multi-bit synchronisers (counters, FIFO pointers).
#  - the level buses (u_sync_lv, u_sync_rl, u_sync_rv) and the event toggles (u_sync_ev), whose bits are
#    independent (each synchronised on its own), and RV_BOOT.
# create_waiver takes pins (the clock pins of the source cells) and one rule id per call.
proc tsr_cdc_waive {cells ids desc} {
  set pins [get_pins -quiet -of_objects $cells -filter {REF_PIN_NAME == C || REF_PIN_NAME == CLK || REF_PIN_NAME == WCLK}]
  if {[llength $pins] == 0} { return }
  foreach id $ids { create_waiver -type CDC -id $id -from $pins -description $desc }
}
proc tsr_cdc_waive_to {cells ids desc} {
  set pins [get_pins -quiet -of_objects $cells -filter {REF_PIN_NAME == D}]
  if {[llength $pins] == 0} { return }
  foreach id $ids { create_waiver -type CDC -id $id -to $pins -description $desc }
}

tsr_cdc_waive [get_cells -quiet -hier -filter {(NAME =~ *u_hs*/req_q_reg* || NAME =~ *u_hs*/rsp_q_reg*) && IS_SEQUENTIAL}] \
  {CDC-1 CDC-4 CDC-15} "tsr_cdc_hs request / response data, loaded behind the synchronised toggle (CC 11.2)"

# tsr_cdc_fifo memories: the memory cells of an instance (and their LUTRAM sub-cells) are the ones whose name
# starts with <instance>/mem_reg
set tsr_fifo_dirs [dict create]
foreach c [get_cells -quiet -hier -filter {NAME =~ *wgray_q_reg* && IS_SEQUENTIAL}] {
  dict set tsr_fifo_dirs "[file dirname [get_property NAME $c]]/mem_reg" 1
}
set tsr_fifo_mem [list]
foreach c [get_cells -quiet -hier -filter {NAME =~ *mem_reg*}] {
  set n [get_property NAME $c]
  set i [string first "/mem_reg" $n]
  if {$i >= 0 && [dict exists $tsr_fifo_dirs [string range $n 0 [expr {$i + 7}]]]} { lappend tsr_fifo_mem $c }
}
tsr_cdc_waive $tsr_fifo_mem {CDC-1 CDC-4 CDC-13 CDC-15 CDC-26} \
  "tsr_cdc_fifo LUTRAM, read behind the synchronised gray write pointer (CC 11.2)"

tsr_cdc_waive [get_cells -quiet -hier -filter {NAME =~ *gray_q_reg* && IS_SEQUENTIAL}] {CDC-6} \
  "gray-coded counter / pointer bus, bounded by set_bus_skew (CC 11.3)"

# RV_BOOT: a 32-bit bus through plain synchronisers. It is quasi-static: tsr_csr ignores RV_BOOT writes while
# CTRL.RV_RUN = 1, and the clk_rv sequencer samples the synchronised value two cycles behind the synchronised
# RV_RUN, at the reset release (CC 11.2).
tsr_cdc_waive_to [get_cells -quiet -hier -filter {NAME =~ *u_sync_boot/s_q_reg* && IS_SEQUENTIAL}] {CDC-6} \
  "RV_BOOT, written only while RV_RUN = 0, sampled at the reset release (CC 11.2)"

# The level / event bundles whose bits are independent (one synchroniser per bit, PER_BIT): the core -> RV levels
# u_sync_lv, the RV -> core levels u_sync_rl and the sticky-event toggles u_sync_ev of tsr_ctrl_cdc, and
# tsr_dbg_rv's u_sync_rv.
tsr_cdc_waive_to [get_cells -quiet -hier -filter {(NAME =~ *u_sync_lv/s_q_reg* || NAME =~ *u_sync_rl/s_q_reg* || NAME =~ *u_sync_ev/s_q_reg* || NAME =~ *u_sync_rv/s_q_reg*) && IS_SEQUENTIAL}] {CDC-6} \
  "independent levels / event toggles, one synchroniser per bit (CC 11.2)"

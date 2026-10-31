# rtl/ctrl/viv/tsr_ctrl_cdc.xdc — clock-domain-crossing constraints of the control subsystem
# (CC §11.3; docs/ctrl-riscv.md §7.2). Scoped to tsr_ctrl (read with -ref tsr_ctrl, or sourced in
# an out-of-context run of tsr_ctrl). The two clocks are the core clock (port clk, ap_clk) and
# clk_rv (port clk_rv, ap_clk_2); their objects are taken from the ports so the file works in an
# out-of-context run and when scoped to an instance.
#
# - ASYNC_REG on every synchroniser stage (tsr_sync's s_q registers, tsr_rst_sync's s_q).
# - set_max_delay -datapath_only = min(period) on every path between the two clocks (the gray
#   pointers and counters, the handshake request / response registers, the FIFO LUTRAM -> output
#   register paths and the level synchronisers' first stage). No set_clock_groups: it would
#   override the max delays (CC §11.3).
# - set_bus_skew on the gray-coded buses (pointer and counter registers gray_q / wgray_q / rgray_q),
#   to the smaller period, so a sample sees at most one step.
set clk_core [get_clocks -of_objects [get_ports clk]]
set clk_rv   [get_clocks -of_objects [get_ports clk_rv]]
set p_core   [get_property PERIOD $clk_core]
set p_rv     [get_property PERIOD $clk_rv]
set p_min    [expr {min($p_core, $p_rv)}]

set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_sync*/s_q_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_rst_*/g_s*.s_q_reg*}]

set_max_delay -datapath_only -from $clk_core -to $clk_rv $p_min
set_max_delay -datapath_only -from $clk_rv -to $clk_core $p_min

# Each gray register bus (tsr_cdc_gray gray_q, tsr_cdc_fifo wgray_q / rgray_q) with the
# synchroniser stage it drives, grouped per instance (no name patterns over generate brackets).
proc tsr_ctrl_skew {p} {
  set groups [dict create]
  foreach c [get_cells -hier -filter {NAME =~ *gray_q_reg* && IS_SEQUENTIAL}] {
    dict lappend groups [file dirname [get_property NAME $c]] $c
  }
  dict for {inst cells} $groups {
    set q  [get_pins -filter {REF_PIN_NAME == Q} -of_objects $cells]
    set to [get_cells -filter {IS_SEQUENTIAL} -of_objects \
              [get_pins -leaf -filter {DIRECTION == IN} -of_objects [get_nets -of_objects $q]]]
    if {[llength $to] > 0} { set_bus_skew -from $cells -to $to $p }
  }
}
tsr_ctrl_skew $p_min

# Waivers for report_cdc (CC §11.3: only the intended crossings, all safe). Vivado does not see
# these protocols and reports them as "unknown" structures:
#  - tsr_cdc_hs request / response registers (req_q, rsp_q): held stable from the toggle that
#    announces them until the toggle that frees them; the other domain loads them into an enabled
#    register only after its synchronised copy of the toggle changed (CC §11.2 "tagged handshakes").
#  - tsr_cdc_fifo LUTRAM (mem): an entry is read only behind the synchronised gray write pointer
#    that covers it, and rewritten only after the gray read pointer freed it.
#  - gray-coded buses into their multi-bit synchronisers (counters, FIFO pointers), and the level
#    bus u_sync_lv, whose bits are independent levels (each synchronised on its own).
# create_waiver takes pins (the clock pins of the source cells) and one rule id per call.
proc tsr_ctrl_waive {cells ids desc} {
  set pins [get_pins -quiet -of_objects $cells -filter {REF_PIN_NAME == C || REF_PIN_NAME == CLK || REF_PIN_NAME == WCLK}]
  if {[llength $pins] == 0} { return }
  foreach id $ids { create_waiver -type CDC -id $id -from $pins -description $desc }
}
tsr_ctrl_waive [get_cells -hier -filter {(NAME =~ *u_hs*/req_q_reg* || NAME =~ *u_hs*/rsp_q_reg*) && IS_SEQUENTIAL}] \
  {CDC-1 CDC-4 CDC-15} "tsr_cdc_hs request / response data, loaded behind the synchronised toggle (CC 11.2)"
tsr_ctrl_waive [get_cells -hier -filter {NAME =~ *u_ififo/mem_reg* || NAME =~ *u_lfifo/mem_reg*}] \
  {CDC-1 CDC-4 CDC-13 CDC-15 CDC-26} "tsr_cdc_fifo LUTRAM, read behind the synchronised gray write pointer (CC 11.2)"
tsr_ctrl_waive [get_cells -hier -filter {NAME =~ *gray_q_reg* && IS_SEQUENTIAL}] \
  {CDC-6} "gray-coded counter / pointer bus, bounded by set_bus_skew (CC 11.3)"
# RV_BOOT: a 32-bit bus through plain synchronisers. It is quasi-static: tsr_csr ignores RV_BOOT writes
# while CTRL.RV_RUN = 1, and the clk_rv sequencer samples the synchronised value two cycles behind
# the synchronised RV_RUN, at the reset release (CC 11.2).
set boot_sync [get_cells -hier -filter {NAME =~ *u_sync_boot/s_q_reg* && IS_SEQUENTIAL}]
if {[llength $boot_sync] > 0} {
  create_waiver -type CDC -id CDC-6 -to [get_pins -of_objects $boot_sync -filter {REF_PIN_NAME == D}] \
    -description "RV_BOOT, written only while RV_RUN = 0, sampled at the reset release (CC 11.2)"
}
set lv_sync [get_cells -hier -filter {NAME =~ *u_sync_lv/s_q_reg* && IS_SEQUENTIAL}]
if {[llength $lv_sync] > 0} {
  foreach id {CDC-6} {
    create_waiver -type CDC -id $id -to [get_pins -of_objects $lv_sync -filter {REF_PIN_NAME == D}] \
      -description "independent core-domain levels, one synchroniser per bit (CC 11.2)"
  }
}

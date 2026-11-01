# hw/u200/cdc_check.tcl — the clock-domain-crossing gate on the routed kernel (CC §11.3, §14.3 gate T3:
# zero unconstrained ap_clk <-> ap_clk_2 paths, report_cdc clean). A v++ hook: hw/u200/k1.cfg and k4.cfg set
# it as run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST, so it runs in the implementation run after route_design.
# The kernel's constraints are the IP's own (rtl/ctrl/viv/tsr_ctrl_cdc.xdc, packaged by
# rtl/platform/alveo_xrt/package_xo.tcl); this checks that they took effect on the linked design: it finds the
# kernel's two clocks from the clock pins of the synchroniser registers (whatever the platform calls them),
# applies the report_cdc waivers (rtl/ctrl/viv/tsr_cdc_waivers.tcl: not constraints, so not in the IP) and
# runs rtl/ctrl/viv/cdc_gate.tcl: report_cdc without Critical / Warning rows between the clocks,
# report_clock_interaction showing "Max Delay Datapath Only" in both directions. The reports go to the
# implementation run directory (cdc_link.rpt, clkint_link.rpt); lines "CDCCHK ..." are in its log.
# Clock/reset wiring is rechecked first and is always fatal. A failing CDC gate stops the implementation run (error), so a link that leaves the crossings unconstrained
# does not produce an xclbin. TSR_CDC_GATE=warn in the environment reports the failure without stopping it.
set tsr_cdc_dir [file dirname [file normalize [info script]]]
source $tsr_cdc_dir/clock_net_check.tcl
source $tsr_cdc_dir/../../rtl/ctrl/viv/cdc_gate.tcl
# the report_cdc waivers are not part of the IP's constraints
source $tsr_cdc_dir/../../rtl/ctrl/viv/tsr_cdc_waivers.tcl

set core_cells [get_cells -quiet -hier -filter {NAME =~ *u_cdc/lv_q_reg* && IS_SEQUENTIAL}]
set rv_cells   [get_cells -quiet -hier -filter {NAME =~ *u_cdc/u_sync_lv/s_q_reg* && IS_SEQUENTIAL}]
set c_core [lsort -unique [get_clocks -quiet -of_objects [get_pins -quiet -of_objects $core_cells -filter {REF_PIN_NAME == C}]]]
set c_rv   [lsort -unique [get_clocks -quiet -of_objects [get_pins -quiet -of_objects $rv_cells -filter {REF_PIN_NAME == C}]]]
if {[llength $c_core] != 1 || [llength $c_rv] != 1 || $c_core eq $c_rv} {
  set msg "CDCCHK cannot find the kernel's two clocks (core: '$c_core', clk_rv: '$c_rv')"
  puts $msg
  if {[info exists ::env(TSR_CDC_GATE)] && $::env(TSR_CDC_GATE) eq "warn"} {
    puts "CRITICAL WARNING: $msg"
  } else {
    error $msg
  }
} else {
  puts "CDCCHK kernel clocks: core $c_core, clk_rv $c_rv"
  set pass [tsr_cdc_gate CDCCHK link [pwd] $c_core $c_rv]
  if {!$pass} {
    set msg "CDCCHK the clock-domain crossings of the kernel are not clean (cdc_link.rpt, clkint_link.rpt)"
    if {[info exists ::env(TSR_CDC_GATE)] && $::env(TSR_CDC_GATE) eq "warn"} {
      puts "CRITICAL WARNING: $msg"
    } else {
      error $msg
    }
  }
}

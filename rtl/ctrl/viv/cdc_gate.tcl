# rtl/ctrl/viv/cdc_gate.tcl — the clock-domain-crossing gate of the control core and the DB block (CC §11.3,
# §14.3 T3), a Tcl procedure on the open design:
#
#   source rtl/ctrl/viv/cdc_gate.tcl
#   tsr_cdc_gate <prefix> <tag> <outdir> <core clock> <rv clock>      returns 1 (PASS) or 0 (FAIL)
#
#   - report_cdc -details to <outdir>/cdc_<tag>.rpt; every Critical or Warning row between the two clocks
#     fails the gate (what is left after the waivers of rtl/ctrl/viv/tsr_ctrl_cdc.xdc are only Info rows);
#   - report_clock_interaction to <outdir>/clkint_<tag>.rpt: both directions between the two clocks must be
#     listed and constrained by "Max Delay Datapath Only" (an unconstrained or merely timed asynchronous
#     pair fails the gate: the synchroniser structures are then not covered by the max delays).
# Prints <prefix> cdc(<tag>) <summary rows>, <prefix> clkint(<tag>) <the two inter-clock rows> and
# <prefix> cdc_gate(<tag>) PASS|FAIL unsafe <n> unconstrained <n>. The clocks are objects or names; they are the ones
# on tsr_core's ports clk / clk_rv, on the kernel's ap_clk / ap_clk_2, or whatever the linked design calls them.
proc tsr_cdc_gate {prefix tag out core rv} {
  set c0 [get_property NAME [get_clocks $core]]
  set c1 [get_property NAME [get_clocks $rv]]
  report_cdc -details -file $out/cdc_$tag.rpt
  set unsafe 0
  foreach line [split [report_cdc -return_string] "\n"] {
    if {[regexp "^(Critical|Warning|Info|Safe|Unknown) +($c0|$c1) +($c0|$c1) " $line]} {
      puts "$prefix cdc($tag) $line"
      if {[regexp {^(Critical|Warning)} $line]} { incr unsafe }
    }
  }
  report_clock_interaction -delay_type min_max -file $out/clkint_$tag.rpt
  set un 0
  set seen 0
  foreach line [split [report_clock_interaction -delay_type min_max -return_string] "\n"] {
    if {[regexp "^ *($c0|$c1) +($c0|$c1) " $line -> from to] && $from ne $to} {
      puts "$prefix clkint($tag) $line"
      incr seen
      if {![string match "*Max Delay Datapath Only*" $line]} { incr un }
    }
  }
  # both directions must be listed and constrained
  if {$seen < 2} { incr un }
  set pass [expr {$unsafe == 0 && $un == 0}]
  puts "$prefix cdc_gate($tag) [expr {$pass ? "PASS" : "FAIL"}] unsafe $unsafe unconstrained $un"
  return $pass
}

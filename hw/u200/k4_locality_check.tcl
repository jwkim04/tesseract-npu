# Read-only PLACE_DESIGN.POST / PHYS_OPT_DESIGN.POST check of the scopes
# captured by k4_floorplan.tcl at PLACE_DESIGN.PRE.
if {[llength [info procs k4fp_check_locality]] != 1} {
  error "K4FP locality check unavailable: k4_floorplan.tcl did not define k4fp_check_locality"
}
k4fp_check_locality 1

# These reports only query the current design. Keep stage names distinct so
# later physical optimisation does not replace earlier placement evidence.
proc k4fp_report_diagnostics {} {
  set step locality
  if {[info exists ::ACTIVE_STEP] && $::ACTIVE_STEP in {
      place_design phys_opt_design post_route_phys_opt_design}} {
    set step $::ACTIVE_STEP
  }
  set timing [file join [pwd] k4_${step}_timing_summary.rpt]
  set congestion [file join [pwd] k4_${step}_congestion.rpt]
  puts "K4FP diagnostics step=$step timing=$timing congestion=$congestion"
  report_timing_summary -delay_type min_max -report_unconstrained \
      -check_timing_verbose -max_paths 5 -file $timing
  report_design_analysis -congestion -file $congestion
}
k4fp_report_diagnostics

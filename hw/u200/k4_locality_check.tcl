# Read-only PLACE_DESIGN.POST / PHYS_OPT_DESIGN.POST check of the scopes
# captured by k4_floorplan.tcl at PLACE_DESIGN.PRE.
if {[llength [info procs k4fp_check_locality]] != 1} {
  error "K4FP locality check unavailable: k4_floorplan.tcl did not define k4fp_check_locality"
}
k4fp_check_locality 1

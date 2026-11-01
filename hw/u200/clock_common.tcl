# Shared fail-closed object checks for the U200 clock hooks.
proc tsr_clock_one {objects what} {
  if {[llength $objects] != 1} {
    error "CLKCHK expected one $what, found [llength $objects]: $objects"
  }
  return [lindex $objects 0]
}

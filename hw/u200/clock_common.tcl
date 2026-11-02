# Shared fail-closed object checks for the U200 clock hooks.
proc tsr_clock_one {objects what} {
  if {[llength $objects] != 1} {
    error "CLKCHK expected one $what, found [llength $objects]: $objects"
  }
  return [lindex $objects 0]
}

# Read the request from the same checked-in config used by build_xclbin.sh.
# Vitis 2024.2 truncates fractional scalable-clock requests to integer MHz.
namespace eval tsr_clock {
  variable config_dir [file dirname [file normalize [info script]]]
}
proc tsr_clock_request {kernel} {
  set kind ""
  foreach property {REF_NAME ORIG_REF_NAME} {
    if {[regexp {^tsr_u200_(k[14])$} [get_property $property $kernel] -> kind]} break
  }
  if {$kind eq ""} { error "CLKNET unknown kernel clock request for $kernel" }
  set f [open [file join $::tsr_clock::config_dir $kind.cfg] r]
  try {set config [read $f]} finally {close $f}
  set requests {}
  foreach line [split $config \n] {
    if {[string match kernel_frequency=* [string trim $line]]} { lappend requests [string trim $line] }
  }
  if {[llength $requests] != 1 ||
      ![regexp {^kernel_frequency=0:([0-9]+(?:\.[0-9]+)?)\|1:([0-9]+(?:\.[0-9]+)?)$} \
          [lindex $requests 0] -> core rv] || $core < 1 || $rv < 1} {
    error "CLKNET invalid scalable-clock request in $kind.cfg"
  }
  puts "CLKNET request $kind core=$core MHz RV=$rv MHz (integer MHz in Vitis)"
  return [list [expr {1000.0 / floor($core)}] [expr {1000.0 / floor($rv)}]]
}

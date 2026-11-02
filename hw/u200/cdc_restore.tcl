# CC §11.3: bind the packaged CDC contract to the effective linked clocks.
# A linked checkpoint need not retain the IP's OOC exceptions. Keep the original
# XDC as the only definition of the crossing bounds; do not infer why it was lost.
source [file join [file dirname [file normalize [info script]]] clock_common.tcl]
namespace eval tsr_cdc_link {
  variable repo [file normalize [file join [file dirname [info script]] ../..]]

  proc context {} {
    set kernel [tsr_clock_one [get_cells -quiet -hierarchical -filter {
      REF_NAME =~ tsr_u200_k* || ORIG_REF_NAME =~ tsr_u200_k*
    }] {CDC kernel instance}]
    set kernel [get_property NAME $kernel]
    set clocks {}
    foreach port {ap_clk ap_clk_2} {
      set pin [tsr_clock_one [get_pins -quiet $kernel/$port] "CDC $port pin"]
      lappend clocks [tsr_clock_one [get_clocks -quiet -of_objects $pin] "CDC $port clock"]
    }
    lassign $clocks core rv
    if {$core eq $rv} { error "CDCBIND kernel clocks alias: $core" }
    set periods {}
    foreach clock $clocks {
      set p [get_property PERIOD $clock]
      if {![string is double -strict $p] ||
          [catch {expr {$p > 0 && $p < Inf}} valid] || !$valid} {
        error "CDCBIND invalid period '$p' on $clock"
      }
      lappend periods $p
    }
    # Match the packaged XDC's conservative cap when the core runs below 300 MHz.
    return [list $kernel $core $rv [expr {min(3.333, [lindex $periods 0], [lindex $periods 1])}]]
  }

  proc check {tag {out .}} {
    variable repo
    lassign [context] kernel core rv bound
    foreach pattern {*gray_q_reg* *u_sync*/s_q_reg*} {
      set cells [get_cells -quiet -hierarchical -filter \
          [format {NAME =~ %s/%s && IS_SEQUENTIAL} $kernel $pattern]]
      if {[llength $cells] == 0} {
        error "CDCBIND missing kernel CDC cells for $pattern"
      }
    }
    # A clock-interaction label alone must not hide an empty clock pair or a
    # related-clock edge-distance check after an exception was lost.
    foreach {from to} [list $core $rv $rv $core] {
      set paths [get_timing_paths -from $from -to $to -delay_type max -max_paths 1]
      if {[llength $paths] == 0} { error "CDCBIND no timed paths from $from to $to" }
      foreach path $paths {
        set requirement [get_property REQUIREMENT $path]
        if {![string is double -strict $requirement] ||
            [catch {expr {abs($requirement - $bound) <= 0.001}} valid] || !$valid} {
          error "CDCBIND $from -> $to requirement '$requirement', expected datapath bound $bound ns"
        }
      }
    }
    source $repo/rtl/ctrl/viv/tsr_cdc_waivers.tcl
    source $repo/rtl/ctrl/viv/cdc_gate.tcl
    if {![tsr_cdc_gate CDCBIND $tag $out $core $rv]} {
      error "CDCBIND CDC structure or clock-pair constraints failed ($tag)"
    }
    set skew_report [file join $out bus_skew_$tag.rpt]
    report_bus_skew -delay_type min_max -file $skew_report
    set f [open $skew_report r]
    try {set skew [read $f]} finally {close $f}
    # Parse the pinned 2024.2 summary, whose endpoint lists can span many lines.
    # A platform-only report also exists in the broken checkpoint.
    set rows 0
    set kernel_bus 0
    set checked 0
    foreach line [split $skew \n] {
      if {$rows && [string match "2. Bus Skew Report Per Constraint*" $line]} break
      if {[regexp {^[0-9]+ +[0-9]+ +} $line]} {
        incr rows
        set kernel_bus 0
      }
      if {[string first "$kernel/" $line] >= 0 &&
          [string first gray_q_reg $line] >= 0} {set kernel_bus 1}
      if {$kernel_bus && [regexp {^ *(Slow|Fast) +([^ ]+) +([^ ]+) +([^ ]+) *$} \
          $line -> corner requirement actual slack]} {
        if {![string is double -strict $requirement] ||
            [catch {expr {$requirement > 0 && $requirement <= $bound + 0.001}} valid] || !$valid} {
          error "CDCBIND gray-bus skew bound '$requirement' exceeds $bound ns ($tag)"
        }
        incr checked
      }
    }
    if {!$checked} {
      error "CDCBIND kernel gray-bus skew constraint is missing ($tag)"
    }
    puts "CDCBIND PASS ($tag) kernel=$kernel core=$core rv=$rv bound=$bound ns"
  }

  proc restore {{tag pre_impl} {out .}} {
    variable repo
    lassign [context] kernel core rv bound
    set original [file join $repo rtl/ctrl/viv/tsr_ctrl_cdc.xdc]
    set f [open $original r]
    try {set body [read $f]} finally {close $f}
    # Match package_xo.tcl's prelude, then apply to the actual instance. The
    # generated file belongs to the run, never to the source/IP installation.
    set generated [file normalize [file join $out tsr_ctrl_cdc_link.xdc]]
    set f [open $generated w]
    try {
      puts $f "# Generated from $original by hw/u200/cdc_restore.tcl"
      puts $f "set tsr_cdc_core_port ap_clk"
      puts $f "set tsr_cdc_rv_port ap_clk_2"
      puts -nonewline $f $body
    } finally {close $f}
    puts "CDCBIND restoring $original on $kernel for effective linked clocks (bound=$bound ns)"
    read_xdc -cells [list $kernel] $generated
    check $tag $out
  }
}

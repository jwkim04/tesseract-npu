# tb/core/viv/blk_rpt.tcl — the reports of blk_ooc.tcl (blk_constrain, blk_grp, blk_report), also used by retime.tcl.
# blk_report <tag> <outdir> <thr>: BOOC util / dev / <tag> wns whs tns fail_ep (r2r, in, out, through), the failing endpoints
# grouped by (start group, end group) and the worst path of each group; <outdir>/ends_<tag>.tsv lists every endpoint below
# <thr> with its worst path. Needs the procs of core_rpt.tcl (cooc_util).
proc blk_constrain {per io} {
  create_clock -name clk -period $per [get_ports clk]
  set_input_delay  -clock clk $io [get_ports -filter {DIRECTION == IN && NAME != clk}]
  set_output_delay -clock clk $io [get_ports -filter {DIRECTION == OUT}]
}

# group of a pin: the hierarchy of its cell with the indices stripped, at most <depth> components
proc blk_grp {pin depth} {
  set d [file dirname $pin]
  regsub -all {\[[0-9]+\]} $d {[*]} d
  regsub -all {_rep(__[0-9]+)?|_replica(_[0-9]+)?} $d {} d
  regsub {/RAM[A-H](_D1)?$} $d {} d
  regsub {(_[0-9]+)+$} $d {} d
  set parts [split $d /]
  if {[llength $parts] > $depth} { set parts [lrange $parts 0 [expr {$depth - 1}]] }
  return [join $parts /]
}

proc blk_report {tag out thr} {
  cooc_util $tag
  set r [report_utilization -return_string]
  set d "BOOC dev($tag)"
  foreach {lbl re} {CLBs {\|\s*CLB\s*\|\s*([0-9]+)} CLB_LUTs {\|\s*CLB LUTs\*?\s*\|\s*([0-9]+)} CLB_regs {\|\s*CLB Registers\s*\|\s*([0-9]+)}} {
    if {[regexp $re $r -> v]} { append d " $lbl $v" } else { append d " $lbl ?" }
  }
  puts $d
  report_timing_summary -max_paths 20 -file $out/timing_$tag.rpt
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  set tns "?"; set fep "?"
  set ts [report_timing_summary -no_detailed_paths -no_check_timing -return_string]
  if {[regexp {WNS\(ns\)\s+TNS\(ns\)\s+TNS Failing Endpoints.*?\n\s*-+.*?\n\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s+([0-9]+)} $ts -> w t f]} { set tns $t; set fep $f }
  set s "BOOC $tag wns $wns whs $whs tns $tns fail_ep $fep"
  foreach {lbl a} [list r2r {-from [all_registers] -to [all_registers]} io_in {-from [all_inputs] -to [all_registers]} io_out {-from [all_registers] -to [all_outputs]} io_thru {-from [all_inputs] -to [all_outputs]}] {
    set p [eval get_timing_paths -quiet -setup -max_paths 1 $a]
    if {[llength $p]} { append s " | ${lbl}_wns [get_property SLACK $p]" }
  }
  puts $s
  # every endpoint below thr, with its worst path: ends.tsv; the failing ones grouped by (start, end) group
  set f [open $out/ends_$tag.tsv w]
  array unset gn; array unset gw; array unset gl; array unset gp
  foreach p [get_timing_paths -quiet -setup -max_paths 200000 -nworst 1 -slack_lesser_than $thr] {
    set sp [get_property STARTPOINT_PIN $p]; set ep [get_property ENDPOINT_PIN $p]
    set sl [get_property SLACK $p]; set lv [get_property LOGIC_LEVELS $p]
    puts $f "$sl\t$lv\t[get_property DATAPATH_DELAY $p]\t$sp\t$ep"
    if {$sl >= 0} { continue }
    set k "[blk_grp $sp 3] -> [blk_grp $ep 3]"
    if {[info exists gn($k)]} { incr gn($k) } else { set gn($k) 1; set gw($k) $sl; set gl($k) $lv; set gp($k) "$sp -> $ep" }
    if {$sl < $gw($k)} { set gw($k) $sl; set gl($k) $lv; set gp($k) "$sp -> $ep" }
  }
  close $f
  set l [list]
  foreach k [array names gn] { lappend l [list $gn($k) $gw($k) $gl($k) $k $gp($k)] }
  set i 0
  foreach e [lsort -integer -decreasing -index 0 $l] {
    puts "BOOC $tag group [lindex $e 0] [lindex $e 1] [lindex $e 2] [lindex $e 3]"
    if {[incr i] >= 25} { break }
  }
  set i 0
  foreach e [lsort -real -index 1 $l] {
    puts "BOOC $tag path [lindex $e 1] levels [lindex $e 2] [lindex $e 4]"
    if {[incr i] >= 12} { break }
  }
}


# tb/core/viv/retime.tcl — time a placed or routed out-of-context checkpoint of the core at other clock periods, with its
# own placement: the same cells and routes against a tighter clock (docs/platform-u200.md §8.9).
#
# usage: vivado -mode batch -nojournal -nolog -source tb/core/viv/retime.tcl -tclargs <dcp> <outdir> <period ns> [<rvperiod ns>]
#          [io=<ns>] [thr=<ns>]
# The clock `clk` (and `clk_rv`) is created again with the new period; the port delays and the CDC constraints stay. Prints
# BOOC retime wns .. (as blk_ooc.tcl) and the failing groups; <outdir>/ends_retime.tsv.
set dcp    [lindex $argv 0]
set out    [lindex $argv 1]
set per    [lindex $argv 2]
set rvper  [expr {[llength $argv] > 3 && ![string match *=* [lindex $argv 3]] ? [lindex $argv 3] : ""}]
set thr 0.3
foreach a [lrange $argv 3 end] {
  if {[regexp {^thr=(.*)$} $a -> v]} { set thr $v }
}
set_param general.maxThreads 4
file mkdir $out
set cooc_flow 1
source [file join [file dirname [file normalize [info script]]] core_rpt.tcl]
source [file join [file dirname [file normalize [info script]]] blk_rpt.tcl]
open_checkpoint $dcp
create_clock -name clk -period $per [get_ports clk]
if {$rvper ne "" && [llength [get_ports -quiet clk_rv]]} { create_clock -name clk_rv -period $rvper [get_ports clk_rv] }
puts "BOOC retime period $per rv $rvper dcp $dcp"
blk_report retime $out $thr
puts "BOOC DONE"

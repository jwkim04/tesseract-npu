# tb/gdn/viv/gdn_ooc.tcl — out-of-context implementation of a GDN module on the U200 part, 5.0 ns (the core clock),
# input and output delay 0.48 * clk, default synth_design / opt_design / place_design / phys_opt_design / route_design:
# the method of tb/fp/viv/fp_area.tcl (docs/design/02-vector-unit.md §7.10, §7.12.7).
#
# usage: vivado -mode batch -nojournal -nolog -source gdn_ooc.tcl -tclargs <repo> <outdir> top=<module>
#          [srcs=<preset|file,...>] [clk=<ns>] [iodly=<ns>] [flow=synth|place|route] [define=<NAME>] [GEN=value ...]
# Prints (grep GOOC):
#   GOOC <stage> CLB <n> LUT <n> FF <n> CARRY8 <n> DSP <n> URAM <n> RAMB36 <n> RAMB18 <n> LUTMEM <n> WNS <ns> WHS <ns>
#        (stage = synth | place | route; from report_utilization of the current design, CLB = CLB tiles)
#   GOOC worst <start> -> <end> slack <ns> levels <n>      (10 register-to-register paths)
#   GOOC failing_endpoints <n>
# Reports in <outdir>: util_<stage>.rpt, util_hier_<stage>.rpt, timing_<stage>.rpt.
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set top tsr_vu_gdn
set srcs unit
set clk 5.0
set iodly -
set flow route
set vdef [list]
set gens [list]
foreach a [lrange $argv 2 end] {
  if {[string match top=* $a]} { set top [string range $a 4 end] } \
  elseif {[string match srcs=* $a]} { set srcs [string range $a 5 end] } \
  elseif {[string match clk=* $a]} { set clk [string range $a 4 end] } \
  elseif {[string match iodly=* $a]} { set iodly [string range $a 6 end] } \
  elseif {[string match flow=* $a]} { set flow [string range $a 5 end] } \
  elseif {[string match define=* $a]} { lappend vdef -verilog_define [string range $a 7 end] } \
  else { lappend gens $a }
}
if {$iodly eq "-"} { set iodly [expr {0.48 * $clk}] }
set_param general.maxThreads 4
set c $repo/rtl/core
set presets [dict create \
  fx2f [list $c/tsr_vu_gdn_fx2f.sv] \
  dot  [list $c/tsr_vu_gdn_ram.sv $c/tsr_vu_gdn_dlane.sv $c/tsr_vu_gdn_mac.sv $c/tsr_vu_gdn_dot.sv] \
  unit [list $c/tsr_fp_pkg.sv $c/tsr_fp_mul.sv $c/tsr_fp_add.sv $c/tsr_fp_add_dsp.sv $c/tsr_vu_gdn_ram.sv \
             $c/tsr_vu_gdn_fx2f.sv $c/tsr_vu_gdn_dlane.sv $c/tsr_vu_gdn_mac.sv $c/tsr_vu_gdn_dot.sv $c/tsr_vu_gdn_ulane.sv $c/tsr_vu_gdn_upd.sv \
             $c/tsr_vu_gdn_coef.sv $c/tsr_vu_gdn.sv]]
if {[dict exists $presets $srcs]} { set files [dict get $presets $srcs] } else {
  set files [list]
  foreach f [split $srcs ,] { lappend files [expr {[file pathtype $f] eq "absolute" ? $f : "$repo/$f"}] }
}
set gopt [list]
foreach g $gens { lappend gopt -generic $g }
file mkdir $out
read_verilog -sv $files
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top {*}$gopt {*}$vdef
create_clock -name clk -period $clk [get_ports clk]
set_input_delay  -clock clk $iodly [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk $iodly [get_ports -filter {DIRECTION == OUT}]
proc grab {rep name} {
  if {[regexp "\\|\\s*[string map {( \\( ) \\) + \\+} $name]\\s*\\|\\s*(\\d+)\\s*\\|" $rep -> v]} { return $v }
  return 0
}
proc stage_report {tag out} {
  set rep [report_utilization -return_string]
  set f [open $out/util_$tag.rpt w]; puts $f $rep; close $f
  report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier_$tag.rpt
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  report_timing_summary -max_paths 30 -file $out/timing_$tag.rpt
  puts "GOOC $tag CLB [grab $rep CLB] LUT [grab $rep {CLB LUTs}] FF [grab $rep {CLB Registers}] CARRY8 [grab $rep CARRY8] DSP [grab $rep DSPs] URAM [grab $rep URAM] RAMB36 [grab $rep {RAMB36/FIFO*}] RAMB18 [grab $rep RAMB18] LUTMEM [grab $rep {LUT as Memory}] WNS $wns WHS $whs"
}
stage_report synth $out
if {$flow ne "synth"} {
  opt_design
  place_design
  stage_report place $out
  if {$flow eq "route"} {
    phys_opt_design
    route_design
    stage_report route $out
  }
}
foreach p [get_timing_paths -setup -max_paths 10 -nworst 1 -from [all_registers] -to [all_registers]] {
  puts "GOOC worst [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p]"
}
set nfail [llength [get_timing_paths -quiet -setup -max_paths 100000 -nworst 1 -slack_lesser_than 0]]
puts "GOOC failing_endpoints $nfail"

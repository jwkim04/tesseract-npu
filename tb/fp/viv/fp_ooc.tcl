# tb/fp/viv/fp_ooc.tcl — out-of-context implementation of one fp module at 300 MHz on the U200
# part, with a logic-depth summary for tb/fp/test_fp.py::test_fp_ooc_timing (opt-in: needs
# Vivado; not part of the default regression).
#
# usage: vivado -mode batch -nojournal -nolog -source fp_ooc.tcl \
#          -tclargs <repo> <top> <generic or -> <outdir>
# Prints (parsed by the test):
#   FPOOC wns <ns> whs <ns>
#   FPOOC depth max_lut <n> max_fabric <n> paths <n>
#   FPOOC fabric_wns <ns>                      worst slack of the paths with fabric logic cells
#   FPOOC lut_path <start> -> <end> slack <ns>     (a path with max_lut LUTs)
#   FPOOC fabric_path <start> -> <end> slack <ns>  (a path with max_fabric cells)
#   FPOOC dsp <cell> AREG <n> BREG <n> MREG <n> PREG <n>
#   FPOOC stage <group> slack <ns> max_lut <n> max_fabric <n>   per destination register group
#                                              (register-to-register paths only)
# Over the 10 worst setup paths of every endpoint (register-to-register and port paths alike):
# max_lut is the largest number of LUT cells (LUT1..LUT6) on one path, max_fabric the largest
# number of fabric logic cells (LUT*, CARRY*, MUXF*). DSP48E2-internal cells are not counted
# (Vivado's LOGIC_LEVELS counts them).
set repo [lindex $argv 0]
set top  [lindex $argv 1]
set gen  [lindex $argv 2]
set out  [lindex $argv 3]
set_param general.maxThreads 2
set root $repo/rtl/core
read_verilog -sv [list $root/tsr_fp_pkg.sv $root/tsr_fp_mul.sv $root/tsr_fp_div.sv $root/tsr_fp_add.sv]
if {$gen ne "-"} {
  synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top -generic $gen
} else {
  synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top
}
create_clock -name clk -period 3.333 [get_ports clk]
# Registered I/O assumed on the other side of the module boundary: budget half a cycle.
set_input_delay  -clock clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk 1.6 [get_ports -filter {DIRECTION == OUT}]
opt_design
place_design
phys_opt_design
route_design
file mkdir $out
report_timing_summary -max_paths 10 -file $out/timing.rpt
report_utilization -file $out/util.rpt

set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
puts "FPOOC wns $wns whs $whs"

set max_lut 0
set max_fab 0
set lut_path "-"
set fab_path "-"
set fab_wns 1e9
set paths [get_timing_paths -setup -max_paths 100000 -nworst 10]
foreach p $paths {
  set nl 0
  set nf 0
  foreach c [get_cells -quiet -of_objects $p] {
    set r [get_property REF_NAME $c]
    if {[string match "LUT*" $r]} { incr nl; incr nf }
    if {[string match "CARRY*" $r] || [string match "MUXF*" $r]} { incr nf }
  }
  set sl [get_property SLACK $p]
  if {$nf > 0 && $sl < $fab_wns} { set fab_wns $sl }
  if {$nl > $max_lut} {
    set max_lut $nl
    set lut_path "[get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] slack $sl"
  }
  if {$nf > $max_fab} {
    set max_fab $nf
    set fab_path "[get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] slack $sl"
  }
}
puts "FPOOC depth max_lut $max_lut max_fabric $max_fab paths [llength $paths]"

# Register-to-register paths per destination register group (the cut / stage registers: the
# hierarchical prefix of the endpoint, e.g. g_c2 or y4q): worst slack and logic depth of the
# logic segment that ends there (start: any flip-flop or DSP register). Paths from input ports
# and to output ports (I/O budget) are left out.
array set st_slack {}
array set st_lut {}
array set st_fab {}
foreach p $paths {
  set sp [get_property STARTPOINT_PIN $p]
  if {![string match "*/*" $sp]} { continue }     ;# input port: I/O budget, not a stage
  set ep [get_property ENDPOINT_PIN $p]
  if {![string match "*/*" $ep]} { continue }     ;# output port: I/O budget, not a stage
  set grp [lindex [split $ep "./"] 0]
  regsub {_reg.*$} $grp "" grp
  set nl 0
  set nf 0
  foreach c [get_cells -quiet -of_objects $p] {
    set r [get_property REF_NAME $c]
    if {[string match "LUT*" $r]} { incr nl; incr nf }
    if {[string match "CARRY*" $r] || [string match "MUXF*" $r]} { incr nf }
  }
  set sl [get_property SLACK $p]
  if {![info exists st_slack($grp)] || $sl < $st_slack($grp)} { set st_slack($grp) $sl }
  if {![info exists st_lut($grp)] || $nl > $st_lut($grp)} { set st_lut($grp) $nl }
  if {![info exists st_fab($grp)] || $nf > $st_fab($grp)} { set st_fab($grp) $nf }
}
foreach grp [lsort [array names st_slack]] {
  puts "FPOOC stage $grp slack $st_slack($grp) max_lut $st_lut($grp) max_fabric $st_fab($grp)"
}
puts "FPOOC fabric_wns $fab_wns"
puts "FPOOC lut_path $lut_path"
puts "FPOOC fabric_path $fab_path"

foreach d [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}] {
  puts "FPOOC dsp $d AREG [get_property AREG $d] BREG [get_property BREG $d] MREG [get_property MREG $d] PREG [get_property PREG $d]"
}

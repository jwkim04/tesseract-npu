# tb/qact/viv/qact_ooc.tcl — out-of-context implementation of tsr_qact at 300 MHz on the U200
# part (opt-in timing check: needs Vivado; tb/qact/test_qact.py::test_qact_ooc_timing).
#
# usage: vivado -mode batch -nojournal -nolog -source qact_ooc.tcl \
#          -tclargs <repo> <generic or -> <outdir> [keep]
#   keep: synth_design -keep_equivalent_registers (Vivado otherwise merges the 16 identical
#         scale-operand input registers of the multipliers into one register driving 32 DSPs)
# Same constraints as tb/fp/viv/fp_ooc.tcl: 3.333 ns clock, every I/O registered on the other side
# of the boundary with half a cycle (1.6 ns) budgeted outside. Prints (parsed by the test):
#   QOOC wns <ns> whs <ns>
#   QOOC worst <start> -> <end> slack <ns> levels <n>
#   QOOC util LUT <n> FF <n> DSP <n> RAMB36 <n> RAMB18 <n> LUTRAM <n>
set repo [lindex $argv 0]
set gen  [lindex $argv 1]
set out  [lindex $argv 2]
set keep [expr {[llength $argv] > 3 && [lindex $argv 3] eq "keep"}]
set sopt [expr {$keep ? "-keep_equivalent_registers" : ""}]
set_param general.maxThreads 4
set root $repo/rtl/core
read_verilog -sv [list $root/tsr_tables_pkg.sv $root/tsr_pkg.sv $root/tsr_fp_pkg.sv $repo/rtl/tech/generic/tsr_sram_1r1w.sv \
                       $root/tsr_fp_mul.sv $root/tsr_fp_div.sv $root/tsr_qact.sv]
set gopt [expr {$gen ne "-" ? [list -generic $gen] : [list]}]
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top tsr_qact {*}$gopt {*}$sopt
create_clock -name clk -period 3.333 [get_ports clk]
set_input_delay  -clock clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk 1.6 [get_ports -filter {DIRECTION == OUT}]
opt_design
place_design
phys_opt_design
route_design
file mkdir $out
report_timing_summary -max_paths 20 -file $out/timing.rpt
report_utilization -file $out/util.rpt
report_utilization -hierarchical -file $out/util_hier.rpt

set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
puts "QOOC wns $wns whs $whs"
foreach p [get_timing_paths -setup -max_paths 10 -nworst 1] {
  puts "QOOC worst [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p]"
}
# Register-to-register paths only (the I/O budget above is an assumption about the neighbours).
set r2r [get_timing_paths -setup -max_paths 1 -from [all_registers] -to [all_registers]]
puts "QOOC r2r_wns [get_property SLACK $r2r] [get_property STARTPOINT_PIN $r2r] -> [get_property ENDPOINT_PIN $r2r] levels [get_property LOGIC_LEVELS $r2r]"
# Worst path per start-point group (the merged scale register would otherwise hide the rest).
set seen [dict create]
foreach p [get_timing_paths -setup -max_paths 2000 -nworst 1] {
  set sp [get_property STARTPOINT_PIN $p]
  regsub {\[[0-9]+\]} [regsub {/[A-Z]+$} $sp ""] "" grp
  regsub -all {\[[0-9]+\]} $grp "\[*\]" grp
  if {![dict exists $seen $grp]} {
    dict set seen $grp 1
    if {[dict size $seen] <= 12} {
      puts "QOOC group $grp slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] -> [get_property ENDPOINT_PIN $p]"
    }
  }
}
set luts [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
set ffs  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
set dsp  [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}]]
set b36  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36*}]]
set b18  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18*}]]
set lram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D*}]]
puts "QOOC util LUT $luts FF $ffs DSP $dsp RAMB36 $b36 RAMB18 $b18 LUTRAM $lram"

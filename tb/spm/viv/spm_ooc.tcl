# tb/spm/viv/spm_ooc.tcl — out-of-context implementation of tsr_spm (the even / odd 2R2W SPM,
# CC §10.1) or tsr_dma (with its 512-beat LD queue, CC §10.4) on the U200 part with the Xilinx
# technology wrapper (rtl/tech/xilinx/tsr_sram_1r1w.sv). Opt-in, not part of tb/run_all.sh.
#
# usage: vivado -mode batch -nojournal -nolog -source tb/spm/viv/spm_ooc.tcl \
#          -tclargs <repo> <top: tsr_spm | tsr_dma> <outdir> [generic ...]
#   e.g. ... -tclargs $PWD tsr_spm /tmp/spm_ooc RD_LAT=4
#        ... -tclargs $PWD tsr_dma /tmp/dma_ooc SPM_RD_LAT=4
# Constraints as tb/qact/viv/qact_ooc.tcl: 3.333 ns clock, every I/O registered on the other side
# of the boundary with 1.6 ns budgeted outside. Prints:
#   SOOC wns <ns> whs <ns>
#   SOOC r2r_wns <ns> <start> -> <end> levels <n>
#   SOOC worst <start> -> <end> slack <ns> levels <n>        (10 paths)
#   SOOC util LUT <n> FF <n> URAM <n> RAMB36 <n> RAMB18 <n> LUTRAM <n>
set repo [lindex $argv 0]
set top  [lindex $argv 1]
set out  [lindex $argv 2]
set gens [lrange $argv 3 end]
set_param general.maxThreads 4
set root $repo/rtl/core
read_verilog -sv [list $root/tsr_tables_pkg.sv $root/tsr_pkg.sv $repo/rtl/tech/xilinx/tsr_sram_1r1w.sv \
                       $root/tsr_spm.sv $root/tsr_dma.sv]
set gopt [list]
foreach g $gens { lappend gopt -generic $g }
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top {*}$gopt
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
puts "SOOC wns $wns whs $whs"
set r2r [get_timing_paths -setup -max_paths 1 -from [all_registers] -to [all_registers]]
puts "SOOC r2r_wns [get_property SLACK $r2r] [get_property STARTPOINT_PIN $r2r] -> [get_property ENDPOINT_PIN $r2r] levels [get_property LOGIC_LEVELS $r2r]"
foreach p [get_timing_paths -setup -max_paths 10 -nworst 1] {
  puts "SOOC worst [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p]"
}
set luts [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
set ffs  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
set uram [llength [get_cells -quiet -hierarchical -filter {REF_NAME == URAM288}]]
set b36  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36*}]]
set b18  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18*}]]
set lram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D*}]]
puts "SOOC util LUT $luts FF $ffs URAM $uram RAMB36 $b36 RAMB18 $b18 LUTRAM $lram"

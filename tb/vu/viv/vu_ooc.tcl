# tb/vu/viv/vu_ooc.tcl — out-of-context implementation of the vector unit tsr_vu (rtl/core/tsr_vu.sv) on the
# U200 part, for the VU's utilisation against CC §14.1's SLR1 budget and timing at 3.333 ns (VU §13 item 4,
# CC §14.3 the run before gate T2). Opt-in (tb/vu/viv/vu_ooc.sh), not part of tb/run_all.sh.
#
# usage: vivado -mode batch -nojournal -nolog -source tb/vu/viv/vu_ooc.tcl -tclargs <repo> <outdir> [synth_only]
#          [GEN=value ...]      e.g. EXP_LANES=8 BACK_LANES=4 SPM_RD_LAT=4
# Constraints as tb/spm/viv/spm_ooc.tcl: 3.333 ns clock, I/O delay 1.6 ns on both sides of the boundary.
# Prints (grep VOOC):
#   VOOC util LUT <n> FF <n> DSP <n> RAMB36 <n> RAMB18 <n> URAM <n> LUTRAM <n> SRL <n>
#   VOOC wns <ns> whs <ns>           (after route; synth_only: the post-synthesis estimate)
#   VOOC r2r_wns <ns> <start> -> <end> levels <n>
#   VOOC worst <start> -> <end> slack <ns> levels <n>     (20 paths)
#   VOOC failing_endpoints <n>
#   VOOC blk <instance> slack <ns> <start> -> <end> levels <n>   (worst path into each block)
#   VOOC hier <instance> LUT <n> FF <n> DSP <n> RAMB36 <n>   (per block)
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set rest [lrange $argv 2 end]
set synth_only 0
set gens [list]
foreach a $rest {
  if {$a eq "synth_only"} { set synth_only 1 } else { lappend gens $a }
}
set_param general.maxThreads 4
set c $repo/rtl/core
read_verilog -sv [list $c/tsr_tables_pkg.sv $c/tsr_pkg.sv $c/tsr_fp_pkg.sv $c/tsr_xc_pkg.sv $c/tsr_vu_pkg.sv \
  $repo/rtl/tech/xilinx/tsr_sram_1r1w.sv $c/tsr_fp_mul.sv $c/tsr_fp_add.sv $c/tsr_fp_div.sv $c/tsr_vu_dly.sv \
  $c/tsr_vu_exp.sv $c/tsr_vu_rcp.sv $c/tsr_vu_sqrt.sv $c/tsr_vu_sincos.sv $c/tsr_vu_lane.sv $c/tsr_vu_lanes.sv $c/tsr_vu_gen.sv \
  $c/tsr_vu_rd.sv $c/tsr_vu_dq.sv $c/tsr_vu_inj.sv $c/tsr_vu_wr.sv $c/tsr_vu_red.sv $c/tsr_vu_angle.sv $c/tsr_vu_topk.sv $c/tsr_vu.sv]
set gopt [list]
foreach g $gens { lappend gopt -generic $g }
file mkdir $out
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top tsr_vu -flatten_hierarchy rebuilt {*}$gopt
create_clock -name clk -period 3.333 [get_ports clk]
set_input_delay  -clock clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk 1.6 [get_ports -filter {DIRECTION == OUT}]
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier_synth.rpt
if {!$synth_only} {
  opt_design
  place_design
  phys_opt_design
  route_design
  report_utilization -file $out/util.rpt
  report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier.rpt
}
report_timing_summary -max_paths 30 -file $out/timing.rpt
set luts [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
set ffs  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
set dsp  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP48E2}]]
set uram [llength [get_cells -quiet -hierarchical -filter {REF_NAME == URAM288}]]
set b36  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36*}]]
set b18  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18*}]]
set lram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM32* || REF_NAME =~ RAM64* || REF_NAME =~ RAM16* || REF_NAME =~ RAM128* || REF_NAME =~ RAM256*}]]
set srl  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ SRL*}]]
puts "VOOC util LUT $luts FF $ffs DSP $dsp RAMB36 $b36 RAMB18 $b18 URAM $uram LUTRAM $lram SRL $srl"
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
puts "VOOC wns $wns whs $whs"
set r2r [get_timing_paths -setup -max_paths 1 -from [all_registers] -to [all_registers]]
puts "VOOC r2r_wns [get_property SLACK $r2r] [get_property STARTPOINT_PIN $r2r] -> [get_property ENDPOINT_PIN $r2r] levels [get_property LOGIC_LEVELS $r2r]"
foreach p [get_timing_paths -setup -max_paths 20 -nworst 1 -from [all_registers] -to [all_registers]] {
  puts "VOOC worst [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p]"
}
# worst register-to-register path ending in each block, and the failing endpoints
set nfail [llength [get_timing_paths -quiet -setup -max_paths 100000 -nworst 1 -slack_lesser_than 0]]
puts "VOOC failing_endpoints $nfail"
foreach inst [list u_gen u_rd u_inj u_wr u_red u_ang g_topk.u_topk g_lane\[0\].u_lane g_lane\[1\].u_lane g_lane\[15\].u_lane] {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ ${inst}/* && IS_SEQUENTIAL"]
  if {[llength $cs] == 0} { continue }
  set p [get_timing_paths -quiet -setup -max_paths 1 -to $cs]
  if {[llength $p] == 0} { continue }
  puts "VOOC blk $inst slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
}
set cs [get_cells -quiet -filter {IS_SEQUENTIAL}]
if {[llength $cs] > 0} {
  set p [get_timing_paths -quiet -setup -max_paths 1 -to $cs]
  puts "VOOC blk top slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
}
foreach inst [list u_gen u_rd u_inj u_wr u_red u_ang u_lma u_lmb g_topk.u_topk u_inj/g_dq.u_dq] {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ ${inst}/*"]
  puts "VOOC hier $inst LUT [llength [filter $cs {REF_NAME =~ LUT*}]] FF [llength [filter $cs {REF_NAME =~ FD*}]] DSP [llength [filter $cs {REF_NAME =~ DSP48E2}]] RAMB36 [llength [filter $cs {REF_NAME =~ RAMB36*}]]"
}
for {set l 0} {$l < 16} {incr l} {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ g_lane\[$l\].u_lane/*"]
  puts "VOOC hier lane$l LUT [llength [filter $cs {REF_NAME =~ LUT*}]] FF [llength [filter $cs {REF_NAME =~ FD*}]] DSP [llength [filter $cs {REF_NAME =~ DSP48E2}]] RAMB36 [llength [filter $cs {REF_NAME =~ RAMB36*}]]"
}

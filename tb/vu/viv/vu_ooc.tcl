# tb/vu/viv/vu_ooc.tcl — out-of-context implementation of the vector unit tsr_vu (rtl/core/tsr_vu.sv) on the
# U200 part, for the VU's utilisation against CC §14.1's resource budget and timing at 3.333 ns (VU §7.10,
# VU §7.12.9). Opt-in (tb/vu/viv/vu_ooc.sh), not part of tb/run_all.sh.
#
# usage: vivado -mode batch -nojournal -nolog -source tb/vu/viv/vu_ooc.tcl -tclargs <repo> <outdir> [synth_only]
#          [place_only] [clk=<ns>] [iodly=<ns>] [define=<NAME>] [GEN=value ...]
#          e.g. EXP_LANES=8 BACK_LANES=4 SPM_RD_LAT=4
#   place_only   stop after place_design (no phys_opt_design, no route_design); the VOOC lines carry the
#                placed utilization and the WNS of the placed (estimated-route) design
#   clk=<ns>     clock period (default 3.333)
#   iodly=<ns>   input and output delay on the boundary (default 1.6)
#   define=<N>   verilog define (e.g. TSR_FP_ADD_DSP=1: every tsr_fp_add is tsr_fp_add_dsp, rtl/core/tsr_fp_add.sv)
#   usedsp=<module>   put (* use_dsp = "yes" *) on that module's declaration (text-patched copy in <outdir>/src),
#                the synthesis attribute that lets Vivado map the module's adders into DSP48E2
# Constraints as tb/spm/viv/spm_ooc.tcl: 3.333 ns clock, I/O delay 1.6 ns on both sides of the boundary.
# The K4 configuration (docs/platform-u200.md §8.8, rtl/platform/alveo_xrt/gen_tops.py):
#   FRONT_LANES=8 EXP_LANES=8 BACK_LANES=4 LANE_PIPE=2 SPM_RD_LAT=4 NUM_CH=4
# The GDN engine (HAS_GDN=1) is read always; its blocks have VOOC hier lines (g_gdn.u_gdn, u_dot, u_upd, u_coef, u_cvt).
# Prints (grep VOOC):
#   VOOC <stage> CLB <n> LUT <n> FF <n> CARRY8 <n> DSP <n> LUTMEM <n> WNS <ns> WHS <ns>  (stage = place | route;
#                from report_utilization of the placed / routed design, CLB = CLB tiles)
#   VOOC util LUT <n> FF <n> DSP <n> RAMB36 <n> RAMB18 <n> URAM <n> LUTRAM <n> SRL <n>
#   VOOC wns <ns> whs <ns>           (after route; synth_only: the post-synthesis estimate)
#   VOOC r2r_wns <ns> <start> -> <end> levels <n>
#   VOOC worst <start> -> <end> slack <ns> levels <n>     (20 paths)
#   VOOC failing_endpoints <n>
#   VOOC blk <instance> slack <ns> <start> -> <end> levels <n>   (worst path into each block)
#   VOOC gdn_from / gdn_to / gdn_lanes / gdn_wr slack <ns> ...   (HAS_GDN: worst paths from and to the engine, into the lanes, into the writer)
#   VOOC hier <instance> LUT <n> FF <n> DSP <n> RAMB36 <n>   (per block)
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set rest [lrange $argv 2 end]
set synth_only 0
set place_only 0
set clk 3.333
set iodly 1.6
set vdef [list]
set usedsp [list]
set gens [list]
foreach a $rest {
  if {$a eq "synth_only"} { set synth_only 1 } elseif {$a eq "place_only"} { set place_only 1 } \
  elseif {[string match clk=* $a]} { set clk [string range $a 4 end] } \
  elseif {[string match iodly=* $a]} { set iodly [string range $a 6 end] } \
  elseif {[string match define=* $a]} { lappend vdef -verilog_define [string range $a 7 end] } \
  elseif {[string match usedsp=* $a]} { lappend usedsp [string range $a 7 end] } \
  else { lappend gens $a }
}
set_param general.maxThreads 4
set c $repo/rtl/core
proc src {c out usedsp name} {
  if {[lsearch -exact $usedsp [file rootname $name]] < 0} { return $c/$name }
  file mkdir $out/src
  set f [open $c/$name]; set t [read $f]; close $f
  regsub -all "(\n|^)module [file rootname $name]" $t "\\1(* use_dsp = \"yes\" *) module [file rootname $name]" t
  set f [open $out/src/$name w]; puts -nonewline $f $t; close $f
  return $out/src/$name
}
read_verilog -sv [list $c/tsr_tables_pkg.sv $c/tsr_pkg.sv $c/tsr_fp_pkg.sv $c/tsr_att_pkg.sv $c/tsr_xc_pkg.sv $c/tsr_vu_pkg.sv \
  $repo/rtl/tech/xilinx/tsr_sram_1r1w.sv $c/tsr_fp_mul.sv [src $c $out $usedsp tsr_fp_add.sv] $c/tsr_fp_add_dsp.sv $c/tsr_fp_div.sv $c/tsr_vu_dly.sv \
  $c/tsr_vu_exp.sv $c/tsr_vu_rcp.sv $c/tsr_vu_sqrt.sv $c/tsr_vu_sincos.sv $c/tsr_vu_lane.sv $c/tsr_vu_lane_share.sv $c/tsr_vu_lanes.sv \
  $c/tsr_local_asmax.sv $c/tsr_vu_gen.sv \
  $c/tsr_vu_rd.sv $c/tsr_vu_dq.sv $c/tsr_vu_inj.sv $c/tsr_vu_wr.sv $c/tsr_vu_red.sv $c/tsr_vu_angle.sv $c/tsr_vu_topk.sv \
  $c/tsr_vu_gdn_ram.sv $c/tsr_vu_gdn_fx2f.sv $c/tsr_vu_gdn_dlane.sv $c/tsr_vu_gdn_mac.sv $c/tsr_vu_gdn_dot.sv \
  $c/tsr_vu_gdn_upd.sv $c/tsr_vu_gdn_coef.sv $c/tsr_vu_gdn.sv $c/tsr_vu.sv]
set gopt [list]
foreach g $gens { lappend gopt -generic $g }
file mkdir $out
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top tsr_vu -flatten_hierarchy rebuilt {*}$gopt {*}$vdef
create_clock -name clk -period $clk [get_ports clk]
set_input_delay  -clock clk $iodly [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk $iodly [get_ports -filter {DIRECTION == OUT}]
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier_synth.rpt
proc grab {name} {
  upvar rep rep
  if {[regexp "\\|\\s*[string map {( \\( ) \\) + \\+} $name]\\s*\\|\\s*(\\d+)\\s*\\|" $rep -> v]} { return $v }
  return 0
}
proc stage_report {tag out} {
  set rep [report_utilization -return_string]
  set f [open $out/util_$tag.rpt w]; puts $f $rep; close $f
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  puts "VOOC $tag CLB [grab CLB] LUT [grab {CLB LUTs}] FF [grab {CLB Registers}] CARRY8 [grab CARRY8] DSP [grab DSPs] LUTMEM [grab {LUT as Memory}] WNS $wns WHS $whs"
}
if {!$synth_only} {
  opt_design
  place_design
  stage_report place $out
  report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier_place.rpt
  report_timing_summary -max_paths 30 -file $out/timing_place.rpt
  if {!$place_only} {
    phys_opt_design
    route_design
    stage_report route $out
    report_utilization -file $out/util.rpt
    report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier.rpt
  }
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
foreach inst [list u_gen u_rd u_inj u_wr u_red u_ang g_topk.u_topk g_gdn.u_gdn g_gdn.u_gdn/u_dot g_gdn.u_gdn/u_upd g_gdn.u_gdn/u_coef g_gdn.u_gdn/u_cvt g_lane\[0\].u_lane g_lane\[1\].u_lane g_lane\[15\].u_lane] {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ ${inst}/* && IS_SEQUENTIAL"]
  if {[llength $cs] == 0} { continue }
  set p [get_timing_paths -quiet -setup -max_paths 1 -to $cs]
  if {[llength $p] == 0} { continue }
  puts "VOOC blk $inst slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
}
# the paths that start in the GDN engine (operand beats and the lane select into the lanes, the writer and the checker) and
# end in it (the injector's slot and line, the writer's credits)
set gs [get_cells -quiet -hierarchical -filter {NAME =~ g_gdn.u_gdn/* && IS_SEQUENTIAL}]
if {[llength $gs] > 0} {
  foreach p [get_timing_paths -quiet -setup -max_paths 10 -nworst 1 -from $gs] {
    puts "VOOC gdn_from slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
  }
  foreach p [get_timing_paths -quiet -setup -max_paths 10 -nworst 1 -to $gs] {
    puts "VOOC gdn_to slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
  }
  set p [get_timing_paths -quiet -setup -max_paths 1 -from $gs -to [get_cells -quiet -hierarchical -filter {NAME =~ u_lanes/* && IS_SEQUENTIAL}]]
  if {[llength $p] > 0} {
    puts "VOOC gdn_lanes slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
  }
  set p [get_timing_paths -quiet -setup -max_paths 1 -from $gs -to [get_cells -quiet -hierarchical -filter {NAME =~ u_wr/* && IS_SEQUENTIAL}]]
  if {[llength $p] > 0} {
    puts "VOOC gdn_wr slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
  }
}
set cs [get_cells -quiet -filter {IS_SEQUENTIAL}]
if {[llength $cs] > 0} {
  set p [get_timing_paths -quiet -setup -max_paths 1 -to $cs]
  puts "VOOC blk top slack [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p] levels [get_property LOGIC_LEVELS $p]"
}
foreach inst [list u_gen u_rd u_inj u_wr u_red u_ang u_lma u_lmb g_topk.u_topk u_inj/g_dq.u_dq g_gdn.u_gdn g_gdn.u_gdn/u_dot g_gdn.u_gdn/u_upd \
                  g_gdn.u_gdn/u_coef g_gdn.u_gdn/u_cvt g_gdn.u_gdn/u_sfm g_gdn.u_gdn/u_sfa g_gdn.u_gdn/u_bv] {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ ${inst}/*"]
  puts "VOOC hier $inst LUT [llength [filter $cs {REF_NAME =~ LUT*}]] FF [llength [filter $cs {REF_NAME =~ FD*}]] DSP [llength [filter $cs {REF_NAME =~ DSP48E2}]] RAMB36 [llength [filter $cs {REF_NAME =~ RAMB36*}]] RAMB18 [llength [filter $cs {REF_NAME =~ RAMB18*}]] URAM [llength [filter $cs {REF_NAME == URAM288}]]"
}
for {set l 0} {$l < 16} {incr l} {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ g_lane\[$l\].u_lane/*"]
  puts "VOOC hier lane$l LUT [llength [filter $cs {REF_NAME =~ LUT*}]] FF [llength [filter $cs {REF_NAME =~ FD*}]] DSP [llength [filter $cs {REF_NAME =~ DSP48E2}]] RAMB36 [llength [filter $cs {REF_NAME =~ RAMB36*}]]"
}

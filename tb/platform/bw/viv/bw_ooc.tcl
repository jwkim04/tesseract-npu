# tb/platform/bw/viv/bw_ooc.tcl — out-of-context synthesis (and optionally placement / routing) of
# the bandwidth kernel tsr_bw_k4 at 300 MHz on the U200 part: resources per block and the worst
# paths (opt-in; needs Vivado; docs/bandwidth-kernel.md §4).
#
# usage: vivado -mode batch -nojournal -nolog -source tb/platform/bw/viv/bw_ooc.tcl \
#          -tclargs <repo> <outdir> [impl]
# Constraints as tb/platform/viv/kernel_ooc.tcl: 3.333 ns ap_clk, every I/O budgeted 1.6 ns outside
# (the platform's AXI interconnect registers both sides). No pblocks / SLR crossings modelled.
# Prints (grep "BWOOC"):
#   BWOOC util(<stage>) LUT .. FF .. DSP .. RAMB36 .. RAMB18 .. URAM .. LUTRAM ..
#   BWOOC hier <cell> LUT .. FF .. LUTRAM ..
#   BWOOC <stage> wns .. tns .. whs .. | r2r_wns .. levels ..
#   BWOOC <stage> path <slack> <levels> <start> -> <end>   worst paths per (start, end) cell pair
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set impl [expr {[lsearch -exact [lrange $argv 2 end] impl] >= 0}]
set_param general.maxThreads 4
file mkdir $out

proc read_fl {fl} {
  set dir [file dirname $fl]
  set srcs [list]
  set f [open $fl r]
  foreach line [split [read $f] "\n"] {
    regsub {//.*$} $line "" line
    set line [string trim $line]
    if {$line ne ""} { lappend srcs [file normalize $dir/$line] }
  }
  close $f
  return $srcs
}
foreach s [read_fl $repo/rtl/platform/alveo_xrt/bw/filelist.f] {
  if {[string match *.sv $s]} { read_verilog -sv $s } else { read_verilog $s }
}
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top tsr_bw_k4
create_clock -name ap_clk -period 3.333 [get_ports ap_clk]
set_input_delay  -clock ap_clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != ap_clk}]
set_output_delay -clock ap_clk 1.6 [get_ports -filter {DIRECTION == OUT}]

proc util {tag} {
  set luts [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
  set ffs  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
  set dsp  [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}]]
  set b36  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36*}]]
  set b18  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18*}]]
  set uram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ URAM288*}]]
  set lram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D* || REF_NAME =~ SRL*}]]
  puts "BWOOC util($tag) LUT $luts FF $ffs DSP $dsp RAMB36 $b36 RAMB18 $b18 URAM $uram LUTRAM/SRL $lram"
}

proc hier_util {pat} {
  foreach c [get_cells -quiet -hierarchical -filter "NAME =~ $pat"] {
    set luts [llength [get_cells -quiet -hierarchical -filter "NAME =~ $c/* && REF_NAME =~ LUT*"]]
    set ffs  [llength [get_cells -quiet -hierarchical -filter "NAME =~ $c/* && REF_NAME =~ FD*"]]
    set ram  [llength [get_cells -quiet -hierarchical -filter "NAME =~ $c/* && (REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D* || REF_NAME =~ SRL*)"]]
    puts "BWOOC hier $c LUT $luts FF $ffs LUTRAM/SRL $ram"
  }
}

# cell of a pin, cut to the hierarchy below u_kernel (g_ch[c].u_eng / u_br / u_ctrl / u_rst)
proc where {p} {
  if {$p eq ""} { return "?" }
  set n [get_property NAME $p]
  if {[get_property CLASS $p] eq "port"} { return "port:$n" }
  set parts [split $n "/"]
  return [join [lrange $parts 0 [expr {min(2, [llength $parts] - 2)}]] "/"]
}

proc report_stage {tag} {
  report_timing_summary -file $::out/timing_$tag.rpt -max_paths 20
  set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
  set whs [get_property SLACK [get_timing_paths -max_paths 1 -hold]]
  set tns 0.0
  foreach p [get_timing_paths -max_paths 1000 -setup -slack_lesser_than 0] { set tns [expr {$tns + [get_property SLACK $p]}] }
  set r2r [get_timing_paths -max_paths 1 -setup -from [all_registers] -to [all_registers]]
  puts "BWOOC $tag wns $wns tns $tns whs $whs | r2r_wns [get_property SLACK $r2r] levels [get_property LOGIC_LEVELS $r2r]"
  set seen [dict create]
  foreach p [get_timing_paths -max_paths 400 -nworst 1 -setup] {
    set k "[where [get_property STARTPOINT_PIN $p]] -> [where [get_property ENDPOINT_PIN $p]]"
    if {[dict exists $seen $k]} { continue }
    dict set seen $k 1
    puts [format "BWOOC %s path %.3f %d %s" $tag [get_property SLACK $p] [get_property LOGIC_LEVELS $p] $k]
    if {[dict size $seen] >= 14} { break }
  }
}

util synth
hier_util "u_kernel/g_ch*.u_eng"
hier_util "u_kernel/g_ch*.u_br"
hier_util "u_kernel/u_ctrl"
hier_util "u_kernel/u_rst"
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier_synth.rpt
report_stage synth

if {$impl} {
  opt_design
  place_design
  phys_opt_design
  route_design
  util route
  report_utilization -file $out/util_route.rpt
  report_stage route
}
puts "BWOOC done"

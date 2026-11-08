# tb/fp/viv/fp_area.tcl — out-of-context area run of an array of f32 operators on the U200 part
# (experiment E1 of docs/design/02-vector-unit.md §7.10). Reports CLB tiles, CLB LUTs, CLB registers,
# CARRY8, DSPs and WNS after place_design and (flow=route) after route_design, plus per-instance numbers.
#
# usage: vivado -mode batch -nojournal -nolog -source fp_area.tcl -tclargs <repo> <outdir> [key=value ...]
#   kind=add|mul|add_dsp|add_dsp2|ip_add|ip_mul   operator (add: tsr_fp_add, mul: tsr_fp_mul, add_dsp: tsr_fp_add_dsp,
#                                         add_dsp2: tsr_fp_add_dsp with NORM_DSP = 1,
#                                         ip_*: Xilinx Floating-Point Operator, area reference only)
#   lat=<n>        LAT of the RTL operators (default: 5 for add and add_dsp, 4 for mul)
#   n=<n>          instances in the array (default 32)
#   clk=<ns>       clock period (default 5.0)
#   iodly=<ns>     input and output delay (default 0.48 * clk)
#   usedsp=yes|no RTL variant: (* use_dsp = "yes" *) (or "no") on the operator module (text-patched copy in
#                  <outdir>/src); 1 means yes
#   ipdsp=<usage>  IP DSP usage: Full_Usage, Medium_Usage, Max_Usage, No_Usage (default Full_Usage)
#   iplat=<n>      IP latency (default: the IP's automatic latency)
#   flow=place|route  (default route)
#   define=<NAME>  verilog define (e.g. TSR_FP_ADD_DSP)
# Prints (grep FPAREA): FPAREA <stage> CLB .. LUT .. FF .. CARRY8 .. DSP .. WNS .. TNS ..
# (stage = place | route), and the same divided by n (FPAREA_PER ...). Reports in <outdir>.
set repo [lindex $argv 0]
set out  [lindex $argv 1]
array set o {kind add lat - n 32 clk 5.0 iodly - usedsp 0 ipdsp Full_Usage iplat - flow route define -}
foreach a [lrange $argv 2 end] {
  set i [string first = $a]
  set o([string range $a 0 [expr {$i - 1}]]) [string range $a [expr {$i + 1}] end]
}
if {$o(lat) eq "-"} { set o(lat) [expr {$o(kind) eq "mul" ? 4 : 5}] }
if {$o(iodly) eq "-"} { set o(iodly) [expr {0.48 * $o(clk)}] }
set_param general.maxThreads 4
set part xcu200-fsgd2104-2-e
file mkdir $out
set viv $repo/tb/fp/viv
set c $repo/rtl/core
set kindnum [dict get {add 0 mul 1 add_dsp 2 add_dsp2 3} [expr {[string match ip_* $o(kind)] ? "add" : $o(kind)}]]

# sources (the operator module text-patched for usedsp=1)
proc patched {src dst mod val} {
  set f [open $src]; set t [read $f]; close $f
  regsub -all "(\n|^)module $mod" $t "\\1(* use_dsp = \"$val\" *) module $mod" t
  set f [open $dst w]; puts -nonewline $f $t; close $f
}
file mkdir $out/src
set srcs [list $c/tsr_fp_pkg.sv]
foreach m {tsr_fp_mul tsr_fp_add tsr_fp_add_dsp} {
  if {![file exists $c/$m.sv]} { continue }
  if {$o(usedsp) ne "0" && $m eq "tsr_fp_$o(kind)"} {
    patched $c/$m.sv $out/src/$m.sv $m [expr {$o(usedsp) eq "no" ? "no" : "yes"}]
    lappend srcs $out/src/$m.sv
  } else {
    lappend srcs $c/$m.sv
  }
}
set vdef [list]
if {$o(define) ne "-"} { lappend vdef -verilog_define $o(define) }

if {[string match ip_* $o(kind)]} {
  file mkdir $out/ip
  create_project -in_memory -part $part
  lappend srcs $viv/tsr_fpip_arr.sv
  set props [list CONFIG.A_Precision_Type Single CONFIG.C_Mult_Usage $o(ipdsp) CONFIG.Flow_Control NonBlocking \
    CONFIG.Has_ACLKEN false CONFIG.Has_ARESETn false CONFIG.RESULT_TLAST_Behv Null]
  if {$o(kind) eq "ip_add"} {
    lappend props CONFIG.Operation_Type Add_Subtract CONFIG.Add_Sub_Value Add
  } else {
    lappend props CONFIG.Operation_Type Multiply
  }
  if {$o(iplat) ne "-"} { lappend props CONFIG.Maximum_Latency false CONFIG.C_Latency $o(iplat) }
  create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name fpip_op -dir $out/ip
  set_property -dict $props [get_ips fpip_op]
  set_property generate_synth_checkpoint false [get_files fpip_op.xci]
  generate_target all [get_ips fpip_op]
  puts "FPAREA ip_latency [get_property CONFIG.C_Latency [get_ips fpip_op]] usage [get_property CONFIG.C_Mult_Usage [get_ips fpip_op]]"
  read_verilog -sv $srcs
  set top tsr_fpip_arr
  set gen [list -generic N=$o(n)]
} else {
  read_verilog -sv [concat $srcs [list $viv/tsr_fp_arr.sv]]
  set top tsr_fp_arr
  set gen [list -generic N=$o(n) -generic KIND=$kindnum -generic LAT=$o(lat)]
}
synth_design -mode out_of_context -part $part -top $top {*}$gen {*}$vdef
create_clock -name clk -period $o(clk) [get_ports clk]
set_input_delay  -clock clk $o(iodly) [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk $o(iodly) [get_ports -filter {DIRECTION == OUT}]
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -file $out/util_hier_synth.rpt

proc grab {name} {
  # one count out of the utilization report text of the current design
  upvar rep rep
  if {[regexp "\\|\\s*[string map {( \\( ) \\) + \\+} $name]\\s*\\|\\s*(\\d+)\\s*\\|" $rep -> v]} { return $v }
  return 0
}
proc stage_report {tag out olist} {
  array set o $olist
  set rep [report_utilization -return_string]
  set f [open $out/util_$tag.rpt w]; puts $f $rep; close $f
  set clb [grab "CLB"]
  set lut [grab "CLB LUTs"]
  set ff  [grab "CLB Registers"]
  set cy  [grab "CARRY8"]
  set dsp [grab "DSPs"]
  set lram [grab "LUT as Memory"]
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  set nfail [llength [get_timing_paths -quiet -setup -max_paths 100000 -nworst 1 -slack_lesser_than 0]]
  report_timing_summary -max_paths 5 -file $out/timing_$tag.rpt
  set n $o(n)
  puts "FPAREA $tag CLB $clb LUT $lut FF $ff CARRY8 $cy DSP $dsp LUTMEM $lram WNS $wns WHS $whs fail_ep $nfail"
  foreach d [lrange [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}] 0 3] {
    puts "FPAREA dsp $tag $d AREG [get_property AREG $d] BREG [get_property BREG $d] MREG [get_property MREG $d] PREG [get_property PREG $d] CREG [get_property CREG $d]"
  }
  puts [format "FPAREA_PER $tag n %d CLB %.2f LUT %.1f FF %.1f CARRY8 %.2f DSP %.2f" $n [expr {double($clb)/$n}] [expr {double($lut)/$n}] [expr {double($ff)/$n}] [expr {double($cy)/$n}] [expr {double($dsp)/$n}]]
}
opt_design
place_design
stage_report place $out [array get o]
if {$o(flow) eq "route"} {
  phys_opt_design
  route_design
  stage_report route $out [array get o]
}

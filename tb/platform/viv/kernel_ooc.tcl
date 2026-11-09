# tb/platform/viv/kernel_ooc.tcl — out-of-context synthesis (optionally implementation) of a
# packaged U200 kernel top at 300 MHz on the U200 part: the adapter's share of the resources and
# its worst paths (opt-in; needs Vivado, not part of tb/run_all.sh). docs/platform-u200.md §9.
#
# usage: vivado -mode batch -nojournal -nolog -source tb/platform/viv/kernel_ooc.tcl \
#          -tclargs <repo> <outdir> [top] [synth_only] [filelist=<core filelist>]
#   top        tsr_u200_k1 (default) | tsr_u200_k4 | tsr_xrt_kernel
#   filelist   core sources (default rtl/filelist_xilinx.f); rtl/platform/alveo_xrt/filelist.f added
# Constraints as tb/core/viv/core_ooc.tcl: 3.333 ns ap_clk, every I/O budgeted 1.6 ns outside (the
# platform's AXI interconnect registers both sides). No pblocks / SLR crossings modelled.
# Prints (grep "KOOC"):
#   KOOC util(<stage>) LUT .. FF .. DSP .. RAMB36 .. RAMB18 .. URAM .. LUTRAM .. LATCH ..
#   KOOC hier <cell> LUT .. FF ..                     (adapter blocks and the core)
#   KOOC <stage> wns .. whs .. | r2r_wns .. levels ..
#   KOOC <stage> adapter_path(<kind>) <slack> <levels> <start> -> <end>   worst adapter paths: from
#        the kernel inputs, to the outputs, adapter to adapter, adapter to any register, any to adapter
#   KOOC <stage> path <slack> <levels> <start> -> <end>          worst path per (start, end) module
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set top  tsr_u200_k1
set synth_only 0
set corefl $repo/rtl/filelist_xilinx.f
foreach a [lrange $argv 2 end] {
  if {$a eq "synth_only"} {
    set synth_only 1
  } elseif {[string match "filelist=*" $a]} {
    set corefl [string range $a 9 end]
  } else {
    set top $a
  }
}
set_param general.maxThreads 4
file mkdir $out

# core filelist (nested lists, include directories, defines: rtl/filelist.tcl), then the
# platform filelist
source $repo/rtl/filelist.tcl
set fld [tsr_read_filelist $corefl]
set fld [tsr_read_filelist $repo/rtl/platform/alveo_xrt/filelist.f $fld]
foreach s [dict get $fld srcs] {
  if {[string match *.sv $s]} { read_verilog -sv $s } else { read_verilog $s }
}
read_xdc $repo/rtl/ctrl/viv/tsr_ctrl_tcm.xdc     ;# the control core's TCMs in RAMB36, not URAM (CC §4.2)
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top {*}[tsr_synth_opts $fld]
create_clock -name ap_clk -period 3.333 [get_ports ap_clk]
set_input_delay  -clock ap_clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != ap_clk}]
set_output_delay -clock ap_clk 1.6 [get_ports -filter {DIRECTION == OUT}]
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.rpt

proc util {tag} {
  set luts [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
  set ffs  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
  set dsp  [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}]]
  set b36  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36*}]]
  set b18  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18*}]]
  set uram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ URAM288*}]]
  set lram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D*}]]
  set ltch [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LD*}]]
  puts "KOOC util($tag) LUT $luts FF $ffs DSP $dsp RAMB36 $b36 RAMB18 $b18 URAM $uram LUTRAM $lram LATCH $ltch"
}

# LUT / FF count below a hierarchical cell
proc hier_util {pat} {
  foreach c [get_cells -quiet -hierarchical -filter "NAME =~ $pat"] {
    set luts [llength [get_cells -quiet -hierarchical -filter "NAME =~ $c/* && REF_NAME =~ LUT*"]]
    set ffs  [llength [get_cells -quiet -hierarchical -filter "NAME =~ $c/* && REF_NAME =~ FD*"]]
    set ram  [llength [get_cells -quiet -hierarchical -filter "NAME =~ $c/* && (REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D*)"]]
    puts "KOOC hier $c LUT $luts FF $ffs LUTRAM $ram"
  }
}

proc paths {tag n} {
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  set r2r [get_timing_paths -setup -max_paths 1 -from [all_registers] -to [all_registers]]
  puts "KOOC $tag wns $wns whs $whs | r2r_wns [get_property SLACK $r2r] levels [get_property LOGIC_LEVELS $r2r] [get_property STARTPOINT_PIN $r2r] -> [get_property ENDPOINT_PIN $r2r]"
  # worst paths that start or end in the adapter (control bridge, channel bridges, reset)
  set apat {*u_ctrl/* || NAME =~ *u_br/* || NAME =~ *u_rst/*}
  set acells [get_cells -quiet -hierarchical -filter "IS_SEQUENTIAL && (NAME =~ $apat)"]
  if {[llength $acells]} {
    foreach {kind from to} [list in [all_inputs] $acells out $acells [all_outputs] internal $acells $acells \
                              to_core $acells [all_registers] from_core [all_registers] $acells] {
      foreach p [get_timing_paths -setup -max_paths 4 -nworst 1 -from $from -to $to] {
        puts "KOOC $tag adapter_path($kind) [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p]"
      }
    }
  }
  # worst path per (start module, end module) pair, register to register
  set seen [dict create]
  foreach p [get_timing_paths -setup -max_paths 3000 -nworst 1 -from [all_registers] -to [all_registers]] {
    set sp [get_property STARTPOINT_PIN $p]
    set ep [get_property ENDPOINT_PIN $p]
    set sm [join [lrange [split $sp /] 0 2] /]
    set em [join [lrange [split $ep /] 0 2] /]
    set key "$sm>$em"
    if {![dict exists $seen $key]} {
      dict set seen $key 1
      if {[dict size $seen] <= $n} {
        puts "KOOC $tag path [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] $sp -> $ep"
      }
    }
  }
}

write_checkpoint -force $out/synth.dcp
util synth
foreach pat {*u_ctrl *u_rst *g_ch*.u_br *u_core} { hier_util $pat }
paths synth 30
report_timing_summary -max_paths 20 -file $out/timing_synth.rpt
if {$synth_only} { exit }

opt_design
place_design
phys_opt_design
route_design
report_timing_summary -max_paths 30 -file $out/timing.rpt
report_utilization -file $out/util.rpt
util routed
paths routed 30

# tb/core/viv/core_ooc.tcl — out-of-context implementation of the whole tsr_core at 300 MHz on the
# U200 part (opt-in synthesis sanity check; needs Vivado, not part of tb/run_all.sh).
#
# usage: vivado -mode batch -nojournal -nolog -source tb/core/viv/core_ooc.tcl \
#          -tclargs <repo> <outdir> [synth_only] [sram=<file>] [NAME=VALUE ...]
#   NAME=VALUE  tsr_core parameters passed as synth_design -generic (e.g. SPM_BYTES=262144
#               SPM_RD_LAT=4); omitted ones keep the U200 v1 defaults of tsr_pkg.
#   synth_only  stop after synthesis (utilisation, latches, logic depth estimate).
#   sram=<file> use this tsr_sram_1r1w instead of rtl/tech/generic's (e.g. a draft of the
#               rtl/tech/xilinx URAM variant).
# Sources: rtl/filelist.f (generic target: behavioural tsr_sram_1r1w; Vivado 2024.2 maps the
# 4 MiB U200 SPM to 1032 RAMB36, not URAM — the M4 xilinx variant has to force URAM).
# Constraints as tb/qact/viv/qact_ooc.tcl: 3.333 ns clock, every I/O budgeted 1.6 ns outside (the
# memory ports and the CSR port are registered on the far side). No pblocks: SLR crossings of the
# real U200 floorplan (DMA <-> channel arbiters) are NOT modelled here.
# Prints (grep "COOC"):
#   COOC wns <ns> whs <ns> | COOC r2r_wns ... | COOC path <slack> <levels> <start> -> <end>
#   COOC util LUT <n> FF <n> DSP <n> RAMB36 <n> RAMB18 <n> URAM <n> LUTRAM <n> LATCH <n>
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set synth_only 0
set sram ""
set gens [list]
foreach a [lrange $argv 2 end] {
  if {$a eq "synth_only"} {
    set synth_only 1
  } elseif {[string match "sram=*" $a]} {
    set sram [string range $a 5 end]
  } else {
    lappend gens -generic $a
  }
}
set_param general.maxThreads 4
file mkdir $out

set fl [open $repo/rtl/filelist.f r]
set srcs [list]
foreach line [split [read $fl] "\n"] {
  regsub {//.*$} $line "" line
  set line [string trim $line]
  if {$line eq ""} { continue }
  if {$sram ne "" && [string match "*/tsr_sram_1r1w.sv" $line]} {
    lappend srcs $sram
  } else {
    lappend srcs $repo/rtl/$line
  }
}
close $fl
read_verilog -sv $srcs
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top tsr_core {*}$gens
create_clock -name clk -period 3.333 [get_ports clk]
set_input_delay  -clock clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk 1.6 [get_ports -filter {DIRECTION == OUT}]
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -file $out/util_hier.rpt

proc util {tag} {
  set luts [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
  set ffs  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
  set dsp  [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}]]
  set b36  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36*}]]
  set b18  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18*}]]
  set uram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ URAM288*}]]
  set lram [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D*}]]
  set ltch [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LD*}]]
  puts "COOC util($tag) LUT $luts FF $ffs DSP $dsp RAMB36 $b36 RAMB18 $b18 URAM $uram LUTRAM $lram LATCH $ltch"
}

proc paths {tag n} {
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  puts "COOC $tag wns $wns whs $whs"
  set r2r [get_timing_paths -setup -max_paths 1 -from [all_registers] -to [all_registers]]
  puts "COOC $tag r2r_wns [get_property SLACK $r2r] levels [get_property LOGIC_LEVELS $r2r] [get_property STARTPOINT_PIN $r2r] -> [get_property ENDPOINT_PIN $r2r]"
  # Worst path per (start module, end module) pair, register to register.
  set seen [dict create]
  foreach p [get_timing_paths -setup -max_paths 3000 -nworst 1 -from [all_registers] -to [all_registers]] {
    set sp [get_property STARTPOINT_PIN $p]
    set ep [get_property ENDPOINT_PIN $p]
    set sm [lindex [split $sp /] 0]
    set em [lindex [split $ep /] 0]
    set key "$sm>$em"
    if {![dict exists $seen $key]} {
      dict set seen $key 1
      if {[dict size $seen] <= $n} {
        puts "COOC $tag path [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] $sp -> $ep"
      }
    }
  }
}

util synth
paths synth 25
if {$synth_only} { exit }

opt_design
place_design
phys_opt_design
route_design
report_timing_summary -max_paths 30 -file $out/timing.rpt
report_utilization -file $out/util.rpt
report_drc -file $out/drc.rpt
util routed
paths routed 25

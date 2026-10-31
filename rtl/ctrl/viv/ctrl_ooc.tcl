# rtl/ctrl/viv/ctrl_ooc.tcl — gate T1 (CC §14.3): out-of-context synthesis and implementation of the
# RISC-V control subsystem on the U200 part, xcu200-fsgd2104-2-e. Opt-in (needs Vivado 2024.2);
# runner: rtl/ctrl/viv/ctrl_ooc.sh.
#
# usage: vivado -mode batch -nojournal -nolog -source rtl/ctrl/viv/ctrl_ooc.tcl \
#          -tclargs <repo> <outdir> [top=tsr_ctrl_rv|tsr_ctrl] [rv=<ns>] [core=<ns>] [synth_only]
#                   [NAME=VALUE ...]
#   top       tsr_ctrl_rv (default; T1 proper: the clk_rv side alone, every port budgeted 40 % of the
#             period outside, as the CDC registers it talks to sit next to it) or tsr_ctrl (the
#             whole subsystem with the CDC: two clocks, tsr_ctrl_cdc.xdc, report_cdc).
#   rv=<ns>   clk_rv period (default 6.0 = 166.7 MHz; 4.0 for the 250 MHz question of D12).
#   core=<ns> core clock period for top=tsr_ctrl (default 3.62 = 276.3 MHz).
#   sram=     generic (default: the attribute-free SRAM model with tsr_ctrl_tcm.xdc mapping the TCMs
#             to block RAM, CC §4.2) or xilinx (rtl/tech/xilinx's wrapper, which maps them to URAM)
#   place= / phys= / route=  -directive of place_design / phys_opt_design / route_design;
#   prphys    phys_opt_design -directive AggressiveExplore after routing.
#   NAME=VALUE  generics of the top (e.g. RV32M=2 WB_STAGE=0 BT_ALU=0 for Ibex "small" with
#             RV32MFast; ibex_pkg enum values: RV32MFast 2, RV32MSingleCycle 3).
# Sources: rtl/core/tsr_tables_pkg.sv, rtl/tech/<sram>/tsr_sram_1r1w.sv,
# rtl/ctrl/filelist.f (+incdir / +define lines become synth_design -include_dirs /
# -verilog_define); FPGA_XILINX is defined for Ibex (DSP pragmas of its counters).
# Prints (grep CTRLOOC): wns / whs per clock, util (LUT, FF, RAMB36, RAMB18, DSP, LUTRAM), the
# worst paths.
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set top tsr_ctrl_rv
set rv 6.0
set core 3.62
set synth_only 0
set sram generic
set pdir ""; set odir ""; set rdir ""; set prphys 0
set gens [dict create]
foreach a [lrange $argv 2 end] {
  if {$a eq "synth_only"} {
    set synth_only 1
  } elseif {[regexp {^top=(.*)$} $a -> v]} {
    set top $v
  } elseif {[regexp {^rv=(.*)$} $a -> v]} {
    set rv $v
  } elseif {[regexp {^core=(.*)$} $a -> v]} {
    set core $v
  } elseif {[regexp {^sram=(generic|xilinx)$} $a -> v]} {
    set sram $v
  } elseif {[regexp {^place=(.*)$} $a -> v]} {
    set pdir $v
  } elseif {[regexp {^phys=(.*)$} $a -> v]} {
    set odir $v
  } elseif {[regexp {^route=(.*)$} $a -> v]} {
    set rdir $v
  } elseif {$a eq "prphys"} {
    set prphys 1
  } elseif {[regexp {^([A-Za-z_][A-Za-z0-9_]*)=(.*)$} $a -> k v]} {
    dict set gens $k $v
  } else {
    error "ctrl_ooc.tcl: unknown argument '$a'"
  }
}
set genargs [list]
dict for {k v} $gens { lappend genargs -generic $k=$v }
puts "CTRLOOC top $top rv $rv core $core generics $gens place '$pdir' phys '$odir' route '$rdir' prphys $prphys"
set_param general.maxThreads 4
file mkdir $out

set srcs [list $repo/rtl/core/tsr_tables_pkg.sv $repo/rtl/tech/$sram/tsr_sram_1r1w.sv]
set incs [list]
set defs [list FPGA_XILINX]
set fdir $repo/rtl/ctrl
set fl [open $fdir/filelist.f r]
foreach line [split [read $fl] "\n"] {
  regsub {//.*$} $line "" line
  set line [string trim $line]
  if {$line eq ""} { continue }
  if {[regexp {^\+incdir\+(.*)$} $line -> d]} {
    lappend incs [file normalize $fdir/$d]
  } elseif {[regexp {^\+define\+(.*)$} $line -> d]} {
    lappend defs $d
  } else {
    lappend srcs [file normalize $fdir/$line]
  }
}
close $fl
read_verilog -sv $srcs
if {$sram eq "generic"} { read_xdc $repo/rtl/ctrl/viv/tsr_ctrl_tcm.xdc }
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top -include_dirs $incs \
  -verilog_define $defs {*}$genargs

if {$top eq "tsr_ctrl_rv"} {
  create_clock -name clk_rv -period $rv [get_ports clk]
  set io [expr {0.4 * $rv}]
  set_input_delay  -clock clk_rv $io [get_ports -filter {DIRECTION == IN && NAME != clk}]
  set_output_delay -clock clk_rv $io [get_ports -filter {DIRECTION == OUT}]
} else {
  create_clock -name clk    -period $core [get_ports clk]
  create_clock -name clk_rv -period $rv   [get_ports clk_rv]
  set ins_core [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != clk_rv && NAME !~ db_*}]
  set_input_delay  -clock clk 1.6 $ins_core
  set_output_delay -clock clk 1.6 [get_ports -filter {DIRECTION == OUT && NAME !~ db_*}]
  set_input_delay  -clock clk_rv [expr {0.4 * $rv}] [get_ports db_*_i]
  set_output_delay -clock clk_rv [expr {0.4 * $rv}] [get_ports -filter {DIRECTION == OUT && NAME =~ db_*}]
  source $repo/rtl/ctrl/viv/tsr_ctrl_cdc.xdc
}
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier_synth.rpt
if {$synth_only} {
  write_checkpoint -force $out/post_synth.dcp
  if {$top eq "tsr_ctrl"} {
    report_cdc -details -file $out/cdc_synth.rpt
    foreach line [split [report_cdc -return_string] "\n"] {
      if {[regexp {^(Critical|Warning|Info) +clk} $line]} { puts "CTRLOOC cdc $line" }
    }
  }
  exit
}

# BRAM power optimisation off for the TCMs: its observability-don't-care enable gating puts the
# fetch / data address back on the BRAM enables (the path the bus keeps free, gate T1).
set_power_opt -exclude_cells [get_cells -hier -filter {REF_NAME =~ RAMB36* && (NAME =~ *u_itcm/* || NAME =~ *u_dtcm/*)}]
opt_design
if {$pdir ne ""} { place_design -directive $pdir } else { place_design }
if {$odir ne ""} { phys_opt_design -directive $odir } else { phys_opt_design }
if {$rdir ne ""} { route_design -directive $rdir } else { route_design }
if {$prphys} { phys_opt_design -directive AggressiveExplore }
report_timing_summary -max_paths 30 -file $out/timing.rpt
report_utilization -file $out/util.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.rpt
report_design_analysis -logic_level_distribution -file $out/levels.rpt
if {$top eq "tsr_ctrl"} {
  report_cdc -details -file $out/cdc.rpt
  foreach line [split [report_cdc -return_string] "\n"] {
    if {[regexp {^(Critical|Warning|Info) +clk} $line]} { puts "CTRLOOC cdc $line" }
  }
}

foreach c [get_clocks] {
  set p [get_timing_paths -max_paths 1 -setup -group $c]
  set h [get_timing_paths -max_paths 1 -hold -group $c]
  set wns [expr {[llength $p] ? [get_property SLACK $p] : "none"}]
  set whs [expr {[llength $h] ? [get_property SLACK $h] : "none"}]
  puts "CTRLOOC clock [get_property NAME $c] period [get_property PERIOD $c] wns $wns whs $whs"
}
foreach c [get_clocks] {
  set p [get_timing_paths -max_paths 1 -setup -group $c -from [all_registers] -to [all_registers]]
  if {[llength $p]} {
    puts "CTRLOOC r2r [get_property NAME $c] slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] from [get_property STARTPOINT_PIN $p] to [get_property ENDPOINT_PIN $p]"
  }
}
foreach p [get_timing_paths -max_paths 8 -setup] {
  puts "CTRLOOC path slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] from [get_property STARTPOINT_PIN $p] to [get_property ENDPOINT_PIN $p]"
}
set u [report_utilization -return_string]
foreach k {"CLB LUTs" "CLB Registers" "LUT as Memory" "Block RAM Tile" "RAMB36/FIFO" "RAMB18" "DSPs"} {
  foreach line [split $u "\n"] {
    if {[string first "| $k" $line] == 0} { puts "CTRLOOC util $line"; break }
  }
}

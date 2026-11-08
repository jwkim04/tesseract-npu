# tb/core/viv/blk_ooc.tcl — out-of-context implementation of one block of the core (a cluster, the gather, the
# vector unit, the DMA, the SPM, the issue / CSR / DB part) on the U200 part, at any core clock period. Opt-in
# (Vivado 2024.2); the runner with the K4 parameter sets is blk_ooc.sh. Results: docs/platform-u200.md §8.9.
#
# usage: vivado -mode batch -nojournal -nolog -source tb/core/viv/blk_ooc.tcl \
#          -tclargs <repo> <outdir> <top> <period ns> [synth_only] [place_only] [dcp] [from=<dcp>] [io=<ns>]
#                   [thr=<ns>] [NAME=VALUE ...]
#   top           the module (tsr_mvm_cluster, tsr_mvm_gather, tsr_vu, tsr_dma, tsr_spm, tsr_qact, tsr_core_dp)
#   period        the clock period of port `clk` (ns)
#   synth_only    stop after synthesis; place_only: stop after place_design (estimate, no routing)
#   dcp           write <outdir>/post_synth.dcp (and post_route.dcp)
#   from=<dcp>    skip synthesis: implement a post_synth.dcp of an earlier run; the clock is re-created
#                 with <period>, so one synthesis serves the 4.0 ns and the 5.0 ns runs
#   io=<ns>       input and output delay of every port (default 1.6: the other side of the boundary is
#                 a register, as in tb/spm/viv and core_ooc.tcl)
#   thr=<ns>      endpoints with slack below this are written to <outdir>/ends.tsv (default 0.3)
#   NAME=VALUE    top-module parameters (synth_design -generic)
# Flow after synthesis: opt_design, place_design, phys_opt_design -directive AggressiveExplore, route_design
# -directive AggressiveExplore, phys_opt_design -directive AggressiveExplore (the directives of hw/u200/k4.cfg).
# Prints (grep BOOC):
#   BOOC util(<stage>) LUT .. FF .. DSP .. RAMB36 .. RAMB18 .. URAM .. LUTRAM .. SRL .. CARRY8 ..
#   BOOC dev(<stage>) CLBs .. CLB_LUTs .. CLB_regs .. (report_utilization, the device view)
#   BOOC <stage> wns .. whs .. tns .. fail_ep .. (setup endpoints below 0) | r2r_wns .. | io_in_wns .. | io_out_wns ..
#   BOOC <stage> group <n failing endpoints> <worst slack> <levels> <start group> -> <end group>   (the failing pairs)
#   BOOC <stage> path <slack> levels <n> <start pin> -> <end pin>                                   (worst per pair)
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set top  [lindex $argv 2]
set per  [lindex $argv 3]
set synth_only 0; set place_only 0; set dcp 0; set from ""; set io 1.6; set thr 0.3
set gens [dict create]
foreach a [lrange $argv 4 end] {
  if {$a eq "synth_only"} { set synth_only 1
  } elseif {$a eq "place_only"} { set place_only 1
  } elseif {$a eq "dcp"} { set dcp 1
  } elseif {[regexp {^from=(.*)$} $a -> v]} { set from $v
  } elseif {[regexp {^io=(.*)$} $a -> v]} { set io $v
  } elseif {[regexp {^thr=(.*)$} $a -> v]} { set thr $v
  } elseif {[regexp {^([A-Za-z_][A-Za-z0-9_]*)=(.*)$} $a -> k v]} { dict set gens $k $v
  } else { error "blk_ooc.tcl: unknown argument '$a'" }
}
set genargs [list]
dict for {k v} $gens { lappend genargs -generic $k=$v }
puts "BOOC top $top period $per io $io generics $gens"
set_param general.maxThreads 4
file mkdir $out
set cooc_flow 1
source [file join [file dirname [file normalize [info script]]] core_rpt.tcl]

proc blk_constrain {per io} {
  create_clock -name clk -period $per [get_ports clk]
  set_input_delay  -clock clk $io [get_ports -filter {DIRECTION == IN && NAME != clk}]
  set_output_delay -clock clk $io [get_ports -filter {DIRECTION == OUT}]
}

# group of a pin: the hierarchy of its cell with the indices stripped, at most <depth> components
proc blk_grp {pin depth} {
  set d [file dirname $pin]
  regsub -all {\[[0-9]+\]} $d {[*]} d
  regsub -all {_rep(__[0-9]+)?|_replica(_[0-9]+)?} $d {} d
  regsub {/RAM[A-H](_D1)?$} $d {} d
  regsub {(_[0-9]+)+$} $d {} d
  set parts [split $d /]
  if {[llength $parts] > $depth} { set parts [lrange $parts 0 [expr {$depth - 1}]] }
  return [join $parts /]
}

proc blk_report {tag out thr} {
  cooc_util $tag
  set r [report_utilization -return_string]
  set d "BOOC dev($tag)"
  foreach {lbl re} {CLBs {\|\s*CLB\s*\|\s*([0-9]+)} CLB_LUTs {\|\s*CLB LUTs\*?\s*\|\s*([0-9]+)} CLB_regs {\|\s*CLB Registers\s*\|\s*([0-9]+)}} {
    if {[regexp $re $r -> v]} { append d " $lbl $v" } else { append d " $lbl ?" }
  }
  puts $d
  report_timing_summary -max_paths 20 -file $out/timing_$tag.rpt
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  set tns "?"; set fep "?"
  set ts [report_timing_summary -no_detailed_paths -no_check_timing -return_string]
  if {[regexp {WNS\(ns\)\s+TNS\(ns\)\s+TNS Failing Endpoints.*?\n\s*-+.*?\n\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s+([0-9]+)} $ts -> w t f]} { set tns $t; set fep $f }
  set s "BOOC $tag wns $wns whs $whs tns $tns fail_ep $fep"
  foreach {lbl a} [list r2r {-from [all_registers] -to [all_registers]} io_in {-from [all_inputs] -to [all_registers]} io_out {-from [all_registers] -to [all_outputs]} io_thru {-from [all_inputs] -to [all_outputs]}] {
    set p [eval get_timing_paths -quiet -setup -max_paths 1 $a]
    if {[llength $p]} { append s " | ${lbl}_wns [get_property SLACK $p]" }
  }
  puts $s
  # every endpoint below thr, with its worst path: ends.tsv; the failing ones grouped by (start, end) group
  set f [open $out/ends_$tag.tsv w]
  array unset gn; array unset gw; array unset gl; array unset gp
  foreach p [get_timing_paths -quiet -setup -max_paths 200000 -nworst 1 -slack_lesser_than $thr] {
    set sp [get_property STARTPOINT_PIN $p]; set ep [get_property ENDPOINT_PIN $p]
    set sl [get_property SLACK $p]; set lv [get_property LOGIC_LEVELS $p]
    puts $f "$sl\t$lv\t[get_property DATAPATH_DELAY $p]\t$sp\t$ep"
    if {$sl >= 0} { continue }
    set k "[blk_grp $sp 3] -> [blk_grp $ep 3]"
    if {[info exists gn($k)]} { incr gn($k) } else { set gn($k) 1; set gw($k) $sl; set gl($k) $lv; set gp($k) "$sp -> $ep" }
    if {$sl < $gw($k)} { set gw($k) $sl; set gl($k) $lv; set gp($k) "$sp -> $ep" }
  }
  close $f
  set l [list]
  foreach k [array names gn] { lappend l [list $gn($k) $gw($k) $gl($k) $k $gp($k)] }
  set i 0
  foreach e [lsort -integer -decreasing -index 0 $l] {
    puts "BOOC $tag group [lindex $e 0] [lindex $e 1] [lindex $e 2] [lindex $e 3]"
    if {[incr i] >= 25} { break }
  }
  set i 0
  foreach e [lsort -real -index 1 $l] {
    puts "BOOC $tag path [lindex $e 1] levels [lindex $e 2] [lindex $e 4]"
    if {[incr i] >= 12} { break }
  }
}

if {$from eq ""} {
  source $repo/rtl/filelist.tcl
  set fld [tsr_read_filelist $repo/rtl/filelist_xilinx.f]
  read_verilog -sv [dict get $fld srcs]
  read_xdc $repo/rtl/ctrl/viv/tsr_ctrl_tcm.xdc
  synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top {*}[tsr_synth_opts $fld] {*}$genargs
  blk_constrain $per $io
  if {$dcp} { write_checkpoint -force $out/post_synth.dcp }
  report_utilization -file $out/util_synth.rpt
  report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier_synth.rpt
  cooc_util synth
  if {$synth_only} { blk_report synth $out $thr; exit }
} else {
  open_checkpoint $from
  blk_constrain $per $io
}
opt_design
place_design
phys_opt_design -directive AggressiveExplore
blk_report placed $out $thr
if {$place_only} { exit }
route_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore
if {$dcp} { write_checkpoint -force $out/post_route.dcp }
report_utilization -file $out/util.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.rpt
report_design_analysis -logic_level_distribution -file $out/levels.rpt
blk_report routed $out $thr
puts "BOOC DONE"

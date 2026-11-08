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

source [file join [file dirname [file normalize [info script]]] blk_rpt.tcl]

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

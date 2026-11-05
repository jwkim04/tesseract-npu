# tb/core/viv/core_ooc.tcl — out-of-context synthesis / implementation of the whole tsr_core at
# 300 MHz on the U200 part (opt-in: needs Vivado, not part of tb/run_all.sh). Results of the
# M3 part C runs: docs/rtl-m3.md §14; runner for the three U200 configurations: core_ooc.sh.
#
# usage: vivado -mode batch -nojournal -nolog -source tb/core/viv/core_ooc.tcl \
#          -tclargs <repo> <outdir> [synth_only] [dcp] [from=<dcp>] [filelist=<f>] [floorplan=<tcl>]
#                   [place=<directive>] [phys=<directive>] [route=<directive>] [prphys] [unc=<ns>] [place_only]
#                   [NAME=VALUE ...]
#   NAME=VALUE    tsr_core parameters passed as synth_design -generic (e.g. NUM_CH=4 T_MAX=8
#                 SPM_RD_LAT=4 MVM_MAC_LAT=9); omitted ones keep the tsr_pkg defaults; a later
#                 NAME=VALUE overrides an earlier one (core_ooc.sh puts its defaults first).
#   synth_only    stop after synthesis (utilisation per hierarchy, latches, logic depth estimate).
#   place_only    stop after place_design (with the floorplan sourced before it): the placed timing, the
#                 pblock capacities and, with dcp, <outdir>/post_place.dcp.
#   dcp           write <outdir>/post_synth.dcp (with the constraints) and post_route.dcp.
#   from=<dcp>    skip synthesis: implement a post_synth.dcp of an earlier run (same constraints).
#   filelist=<f>  sources (default rtl/filelist_xilinx.f: the U200 target with the rtl/tech/xilinx
#                 URAM / BRAM / DSP-packed wrappers; rtl/filelist.f = generic target, where Vivado
#                 maps the 4 MiB SPM to 1032 RAMB36 and the MAC lanes to one DSP per product).
#   floorplan=<f> Tcl sourced after opt_design, before place_design (pblocks; e.g. core_slr.tcl,
#                 configured by TSR_SLR_* environment variables, see its header).
#   place= / phys= / route=  -directive of place_design / phys_opt_design / route_design.
#   prphys        phys_opt_design once more after routing.
#   unc=<ns>      extra setup uncertainty on clk (the platform's MMCM jitter is not modelled by
#                 default, as in the block runs of tb/tech/viv and tb/qact/viv).
# Constraints as tb/qact/viv/qact_ooc.tcl: 3.333 ns clock, every I/O budgeted 1.6 ns outside (the
# memory ports, the CSR port and rst_n are registered on the far side: tsr_xrt_kernel's AXI
# slices and reset copies). Without floorplan= no pblocks: the placer spreads the core over the
# SLRs it needs, the SLR crossings of the real U200 floorplan (clusters next to their DDR4
# controller) are not imposed.
# Prints (grep "COOC"; procs in core_rpt.tcl): util / dev (totals), hier (per hierarchy key),
# wns / whs / tns, r2r, the worst path per (start group, end group), io_in / io_out, the clock-domain-crossing
# gate (cdc(<stage>) rows of report_cdc, clkint(<stage>) rows of report_clock_interaction and
# cdc_gate(<stage>) PASS|FAIL: gate T2 / T3 of CC §14.3, constraints in rtl/ctrl/viv/tsr_ctrl_cdc.xdc and
# tsr_cdc_waivers.tcl), and after placement slr / sll (cells per SLR, SLR-crossing nets by driver).
set repo [lindex $argv 0]
set out  [lindex $argv 1]
set synth_only 0
set dcp 0
set from ""
set flist $repo/rtl/filelist_xilinx.f
set fplan ""
set pdir ""; set odir ""; set rdir ""
set prphys 0
set place_only 0
set unc ""
set gens [dict create]
foreach a [lrange $argv 2 end] {
  if {$a eq "synth_only"} {
    set synth_only 1
  } elseif {$a eq "dcp"} {
    set dcp 1
  } elseif {$a eq "prphys"} {
    set prphys 1
  } elseif {$a eq "place_only"} {
    set place_only 1
  } elseif {[regexp {^from=(.*)$} $a -> v]} {
    set from $v
  } elseif {[regexp {^filelist=(.*)$} $a -> v]} {
    set flist $v
  } elseif {[regexp {^floorplan=(.*)$} $a -> v]} {
    set fplan $v
  } elseif {[regexp {^place=(.*)$} $a -> v]} {
    set pdir $v
  } elseif {[regexp {^phys=(.*)$} $a -> v]} {
    set odir $v
  } elseif {[regexp {^route=(.*)$} $a -> v]} {
    set rdir $v
  } elseif {[regexp {^unc=(.*)$} $a -> v]} {
    set unc $v
  } elseif {[regexp {^([A-Za-z_][A-Za-z0-9_]*)=(.*)$} $a -> k v]} {
    dict set gens $k $v      ;# a later NAME=VALUE overrides an earlier one
  } else {
    error "core_ooc.tcl: unknown argument '$a'"
  }
}
set genargs [list]
dict for {k v} $gens { lappend genargs -generic $k=$v }
puts "COOC generics $gens"
set_param general.maxThreads 4
file mkdir $out
set cooc_flow 1
source [file join [file dirname [file normalize [info script]]] core_rpt.tcl]

if {$from eq ""} {
  # the filelist with its nested lists, include directories and defines (rtl/filelist.tcl, W4-A)
  source $repo/rtl/filelist.tcl
  set fld [tsr_read_filelist $flist]
  read_verilog -sv [dict get $fld srcs]
  # the control core's TCMs in RAMB36 (32K x 1 columns), never URAM (CC §4.2, CC-3): the XDC overrides the
  # size rule of rtl/tech/xilinx/tsr_sram_1r1w.sv (synth WNS -1.257 ns with the ITCM in a URAM cascade)
  read_xdc $repo/rtl/ctrl/viv/tsr_ctrl_tcm.xdc
  synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top tsr_core {*}[tsr_synth_opts $fld] {*}$genargs
  create_clock -name clk -period 3.333 [get_ports clk]
  # the control core's clock (CC §11.1, 166.7 MHz) and its crossings (rtl/ctrl/viv/tsr_ctrl_cdc.xdc takes
  # both clocks from the ports named below; the report_cdc waivers are tsr_cdc_waivers.tcl)
  create_clock -name clk_rv -period 6.000 [get_ports clk_rv]
  set_input_delay  -clock clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != clk_rv}]
  set_output_delay -clock clk 1.6 [get_ports -filter {DIRECTION == OUT}]
  set tsr_cdc_core_port clk
  set tsr_cdc_rv_port   clk_rv
  source $repo/rtl/ctrl/viv/tsr_ctrl_cdc.xdc
  source $repo/rtl/ctrl/viv/tsr_cdc_waivers.tcl
  if {$unc ne ""} { set_clock_uncertainty -setup $unc [get_clocks clk] }
  if {$dcp} { write_checkpoint -force $out/post_synth.dcp }
  if {![cooc_cdc synth $out]} { error "COOC synthesis CDC gate failed" }
  report_utilization -file $out/util_synth.rpt
  report_utilization -hierarchical -hierarchical_depth 6 -file $out/util_hier.rpt
  cooc_util synth
  cooc_hier synth
  cooc_paths synth 30
  if {$synth_only} { exit }
} else {
  open_checkpoint $from
  if {$unc ne ""} { set_clock_uncertainty -setup $unc [get_clocks clk] }
}

# BRAM power optimisation off for the TCMs (as rtl/ctrl/viv/ctrl_ooc.tcl): its enable gating puts the fetch
# address back on the BRAM enables
set_power_opt -exclude_cells [get_cells -hier -filter {REF_NAME =~ RAMB36* && (NAME =~ *u_itcm/* || NAME =~ *u_dtcm/*)}]
opt_design
if {$fplan ne ""} { source $fplan }
if {$pdir ne ""} { place_design -directive $pdir } else { place_design }
cooc_paths placed 10
if {$place_only} {
  if {$dcp} { write_checkpoint -force $out/post_place.dcp }
  report_utilization -hierarchical -hierarchical_depth 6 -file $out/util_hier_placed.rpt
  report_timing_summary -max_paths 50 -file $out/timing_placed.rpt
  report_design_analysis -congestion -file $out/congestion_placed.rpt
  report_clock_utilization -file $out/clock_util_placed.rpt
  cooc_slr placed $out
  exit
}
if {$odir ne ""} { phys_opt_design -directive $odir } else { phys_opt_design }
if {$rdir ne ""} { route_design -directive $rdir } else { route_design }
if {$prphys} { phys_opt_design }
if {$dcp} { write_checkpoint -force $out/post_route.dcp }
report_timing_summary -max_paths 50 -file $out/timing.rpt
report_utilization -file $out/util.rpt
report_utilization -hierarchical -hierarchical_depth 6 -file $out/util_hier_routed.rpt
report_design_analysis -logic_level_distribution -file $out/levels.rpt
report_drc -file $out/drc.rpt
if {![cooc_cdc routed $out]} { error "COOC routed CDC gate failed" }
cooc_util routed
cooc_hier routed
cooc_paths routed 30
cooc_slr routed $out

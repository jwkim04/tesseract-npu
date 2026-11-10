# tb/platform/viv/xo_cdc.tcl — does the packaged kernel carry its clock-domain-crossing constraints? (opt-in;
# needs Vivado; CC §11.3 "kernel XDC packaged in the .xo", gate T3 "zero unconstrained ap_clk <-> ap_clk_2
# paths; report_cdc clean", docs/platform-u200.md.) What a v++ link does with the kernel IP, in miniature:
# the .xo's IP is instantiated (create_ip) and synthesized out of context (synth_ip, non-project mode), which
# applies the constraint files of the IP's synthesis view with the clocks the IP's interface parameters
# (FREQ_HZ of ap_clk, ap_clk_2) define. This also proves that the packaged sources synthesize on their own
# (include files, defines). The synthesized IP checkpoint is then checked:
#   - XCDC ip_xdc: the IP's constraint file list contains tsr_ctrl_cdc.xdc;
#   - XCDC async_reg <n>: cells with ASYNC_REG (the synchronizer stages, > 0);
#   - report_cdc and report_clock_interaction of the two kernel clocks (cooc_cdc of tb/core/viv/core_rpt.tcl):
#     only Info rows once the waivers are applied, both directions "Max Delay Datapath Only".
# Prints XCDC lines, the COOC lines of cooc_cdc and, last, "XCDC gate PASS|FAIL".
#
# usage: vivado -mode batch -nojournal -nolog -source tb/platform/viv/xo_cdc.tcl -tclargs <repo> <xo> <outdir>
#   <xo>  build/u200/xo/tsr_u200_k1.xo (hw/u200/build_xclbin.sh; rtl/platform/alveo_xrt/package_xo.tcl)
set repo [lindex $argv 0]
set xo   [file normalize [lindex $argv 1]]
set out  [file normalize [lindex $argv 2]]
set_param general.maxThreads 4
file mkdir $out
source $repo/tb/core/viv/core_rpt.tcl

# the IP repository inside the .xo (a zip: ip_repo/<vendor>_<library>_<name>_<version>/component.xml)
exec unzip -q -o $xo -d $out/xo
set_part xcu200-fsgd2104-2-e
set_property ip_repo_paths [glob -directory $out/xo ip_repo] [current_project]
update_ip_catalog
set defs [get_ipdefs -filter {NAME =~ tsr_u200_k*}]
if {[llength $defs] != 1} { puts "XCDC error: IP definitions found: $defs"; puts "XCDC gate FAIL"; exit 1 }
file mkdir $out/ip
create_ip -vlnv $defs -module_name kip -dir $out/ip
set ip [get_ips kip]
generate_target all $ip
synth_ip $ip
set xdcs [get_files -all -quiet -of_objects $ip *tsr_ctrl_cdc.xdc]
puts "XCDC ip_xdc [llength $xdcs] [join $xdcs { }]"
set ok [expr {[llength $xdcs] > 0}]

set dcp [lindex [glob -nocomplain $out/ip/kip/kip.dcp] 0]
puts "XCDC dcp $dcp"
if {$dcp eq ""} { puts "XCDC no checkpoint: the IP did not synthesize"; puts "XCDC gate FAIL"; exit 1 }
open_checkpoint $dcp
puts "XCDC clocks [get_clocks]"
set ar [llength [get_cells -hier -quiet -filter {ASYNC_REG == TRUE}]]
puts "XCDC async_reg $ar"
# the report_cdc waivers are not part of the IP's constraints
source $repo/rtl/ctrl/viv/tsr_cdc_waivers.tcl
if {$ar == 0} { set ok 0 }
if {[llength [get_clocks -quiet ap_clk_2]] == 0} { set ok 0 }
set gate [cooc_cdc ip $out ap_clk ap_clk_2]
puts "XCDC gate [expr {$ok && $gate ? {PASS} : {FAIL}}]"
if {!$ok || !$gate} { error "XCDC packaged kernel CDC gate failed" }

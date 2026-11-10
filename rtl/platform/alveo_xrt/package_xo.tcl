# rtl/platform/alveo_xrt/package_xo.tcl — package a Tesseract U200 RTL kernel as a Vitis .xo
# (Vivado batch; docs/platform-u200.md §6).
#
# usage: vivado -mode batch -nojournal -nolog -source rtl/platform/alveo_xrt/package_xo.tcl \
#          -tclargs <repo> <kernel> <outdir> [core_filelist]
#   kernel         tsr_u200_k1 | tsr_u200_k4 (module name = kernel name; kernel_k1.xml / kernel_k4.xml)
#   outdir         the .xo is written to <outdir>/<kernel>.xo (hw/u200/build_xclbin.sh: build/u200/xo);
#                  the IP and the temporary project go to <outdir>/<kernel>_pkg/
#   core_filelist  core sources (default rtl/filelist_xilinx.f: the Xilinx technology wrappers,
#                  behavior identical to rtl/filelist.f); rtl/platform/alveo_xrt/filelist.f is added
# Steps: a project with every source (SystemVerilog) and the kernel as top -> ipx::package_project
# (sources imported into the IP) -> Vitis kernel properties (sdx_kernel rtl, no user parameters,
# ap_clk associated with s_axi_control / m_axi_ch<c> / ap_rst_n, FREQ_HZ 300 MHz; ap_clk_2 (the
# control core's clock, tsr_core clk_rv) with ap_rst_n_2, FREQ_HZ 166.7 MHz; m_axi burst and
# outstanding hints, the s_axi_control register map with the args of kernel_<k>.xml) ->
# the clock-domain-crossing constraints (rtl/ctrl/viv/tsr_ctrl_cdc.xdc; CC §11.3) as the IP's
# synthesis + implementation constraints (processed last, scoped to the kernel), so that a v++ link applies
# the ASYNC_REG properties, the max delays between ap_clk and ap_clk_2, the bus skews and the report_cdc waivers ->
# the TCM block RAM constraints (rtl/ctrl/viv/tsr_ctrl_tcm.xdc, synthesis only) ->
# ipx::check_integrity -kernel / -xrt -> package_xo with kernel_<k>.xml -> the source manifest
# <outdir>/<kernel>.sources.sha256 (sha256sum format, paths relative to <repo>: both filelists, every
# source, kernel_<k>.xml, the CDC and TCM constraints and this script), so `sha256sum -c` in <repo> tells whether the .xo still
# matches the working tree (hw/u200/build_xclbin.sh repackages when it does not).
# Prints "PKG <kernel> xo <path>" on success.
set script [file normalize [info script]]
set repo   [file normalize [lindex $argv 0]]
set kernel [lindex $argv 1]
set outdir [file normalize [lindex $argv 2]]
set corefl [expr {[llength $argv] > 3 ? [file normalize [lindex $argv 3]] : "$repo/rtl/filelist_xilinx.f"}]
set_param general.maxThreads 4

array set short {tsr_u200_k1 k1 tsr_u200_k4 k4}
array set nch   {tsr_u200_k1 1  tsr_u200_k4 4}
if {![info exists short($kernel)]} {
  puts "PKG error: unknown kernel '$kernel' (tsr_u200_k1 | tsr_u200_k4)"
  exit 2
}
set pdir  $repo/rtl/platform/alveo_xrt
set kxml  $pdir/kernel_$short($kernel).xml
set work  $outdir/${kernel}_pkg
set ipdir $work/ip
file delete -force $work
file mkdir $work
file delete -force $outdir/$kernel.xo $outdir/$kernel.sources.sha256

# ---- sources: core filelist (paths relative to its directory), then the platform filelist ----
# (nested lists, include directories, defines: rtl/filelist.tcl)
source $repo/rtl/filelist.tcl
if {![file exists $corefl]} {
  puts "PKG error: core filelist $corefl not found"
  exit 2
}
set fld [tsr_read_filelist $corefl]
set fld [tsr_read_filelist $pdir/filelist.f $fld]
set srcs [dict get $fld srcs]

create_project -force kpack $work/proj -part xcu200-fsgd2104-2-e
add_files -norecurse $srcs
if {[llength [dict get $fld incdirs]]} { set_property include_dirs [dict get $fld incdirs] [current_fileset] }
set_property verilog_define [dict get $fld defines] [current_fileset]
set_property file_type SystemVerilog [get_files *.sv]
set_property top $kernel [current_fileset]
update_compile_order -fileset sources_1

# ---- IP ----
ipx::package_project -root_dir $ipdir -vendor tesseract -library kernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $ipdir/component.xml
ipx::edit_ip_in_project -upgrade true -name kedit -directory $work/edit $ipdir/component.xml
set core [ipx::current_core]
set_property core_revision 1 $core
foreach up [ipx::get_user_parameters -of_objects $core] {
  ipx::remove_user_parameter [get_property NAME $up] $core
}
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property supported_families { } $core
set_property auto_family_support_level level_2 $core

ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core
for {set c 0} {$c < $nch($kernel)} {incr c} {
  set m m_axi_ch$c
  ipx::associate_bus_interfaces -busif $m -clock ap_clk $core
  set bif [ipx::get_bus_interfaces $m -of_objects $core]
  foreach {p v} {MAX_BURST_LENGTH 64 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 32} {
    set bp [ipx::add_bus_parameter -quiet $p $bif]
    set_property value $v $bp
    set_property value_source constant $bp
  }
}
ipx::associate_bus_interfaces -clock ap_clk -reset ap_rst_n $core
set clkbif [ipx::get_bus_interfaces ap_clk -of_objects $core]
set fp [ipx::add_bus_parameter -quiet FREQ_HZ $clkbif]
set_property value 300000000 $fp
set_property value_resolve_type user $fp
# the control core's clock (CC §11.1, §13.3): ap_clk_2 with its reset ap_rst_n_2; no bus interface
# runs on it (every crossing is inside tsr_core). The link requests 166.7 MHz for k1 and 150 MHz for k4
# (hw/u200/k*.cfg); FREQ_HZ here is only the IP's declared value.
# ipx::package_project infers only ap_clk / ap_rst_n by name: the second pair is inferred here.
if {[llength [ipx::get_bus_interfaces ap_clk_2 -of_objects $core]] == 0} {
  ipx::infer_bus_interface ap_clk_2 xilinx.com:signal:clock_rtl:1.0 $core
}
if {[llength [ipx::get_bus_interfaces ap_rst_n_2 -of_objects $core]] == 0} {
  ipx::infer_bus_interface ap_rst_n_2 xilinx.com:signal:reset_rtl:1.0 $core
}
set rst2bif [ipx::get_bus_interfaces ap_rst_n_2 -of_objects $core]
set pp2 [ipx::add_bus_parameter -quiet POLARITY $rst2bif]
set_property value ACTIVE_LOW $pp2
ipx::associate_bus_interfaces -clock ap_clk_2 -reset ap_rst_n_2 $core
set clk2bif [ipx::get_bus_interfaces ap_clk_2 -of_objects $core]
set fp2 [ipx::add_bus_parameter -quiet FREQ_HZ $clk2bif]
set_property value 166666667 $fp2
set_property value_resolve_type user $fp2

# s_axi_control register map: the args of kernel_<k>.xml (ap_ctrl_none, docs/design EX §9.1: no
# AP_CTRL / GIE / IER / ISR block)
set mm [ipx::add_memory_map -quiet s_axi_control $core]
set ab [ipx::add_address_block -quiet reg0 $mm]
set_property range 0x2000 $ab
set xf [open $kxml r]
set xml [read $xf]
close $xf
foreach {all name aq port size off} [regexp -all -inline {<arg name="([^"]+)" addressQualifier="([0-9])" id="[0-9]+" port="([^"]+)" size="(0x[0-9a-fA-F]+)" offset="(0x[0-9a-fA-F]+)"} $xml] {
  set reg [ipx::add_register -quiet $name $ab]
  set_property address_offset $off $reg
  set_property size [expr {$size * 8}] $reg
  if {$aq == 1} {
    set rp [ipx::add_register_parameter -quiet ASSOCIATED_BUSIF $reg]
    set_property value $port $rp
  }
}
set_property slave_memory_map_ref s_axi_control [ipx::get_bus_interfaces s_axi_control -of_objects $core]

# ---- include files in subdirectories of the include directories ----
# The IP imports include files flat into src/, but fifo_v3.sv (riscv-dbg's response FIFO) says
# `include "common_cells/assertions.svh"` (the guard of the file is defined by the filelist, but the file
# must be found): the files that sit one directory below an include directory are added under that
# relative path as well, so that src/ resolves them in the IP's synthesis.
foreach d [dict get $fld incdirs] {
  foreach f [glob -nocomplain -types f -directory $d */*.svh */*.vh] {
    set rel [string range $f [expr {[string length $d] + 1}] end]
    file mkdir [file dirname $ipdir/src/$rel]
    file copy -force $f $ipdir/src/$rel
    foreach fgn {xilinx_anylanguagesynthesis xilinx_anylanguagebehavioralsimulation} {
      set fg [ipx::get_file_groups $fgn -of_objects $core]
      if {[llength $fg] == 0} { continue }
      set inc [ipx::add_file src/$rel $fg]
      set_property type systemVerilogSource $inc
      set_property is_include true $inc
    }
  }
}

# ---- clock-domain-crossing constraints (CC §11.3, gate T3): ap_clk <-> ap_clk_2 ----
# The file finds its clocks from the ports ap_clk / ap_clk_2 (a v++ link has no other constraint for the
# crossings: without it the synchronizers have no ASYNC_REG and the paths between the clocks are timed
# as unrelated or related clocks at their real edge distance). USED_IN synthesis and implementation
# (also out of context), in the synthesis file group (the one the out-of-context IP run reads), PROCESSING_ORDER
# LATE so that the platform's clock definitions come first.
# The file is plain XDC (no control flow: the constraint reader has none) and takes its two clock ports from
# variables; the IP's copy sets them to the kernel's ap_clk / ap_clk_2 in front. The report_cdc waivers
# (rtl/ctrl/viv/tsr_cdc_waivers.tcl) are not constraints and stay out of the IP; hw/u200/cdc_check.tcl applies
# them to the linked design before its gate.
set xdc $repo/rtl/ctrl/viv/tsr_ctrl_cdc.xdc
file mkdir $ipdir/xdc
set xh [open $xdc r]
set xbody [read $xh]
close $xh
set xh [open $ipdir/xdc/tsr_ctrl_cdc.xdc w]
puts $xh "# generated by rtl/platform/alveo_xrt/package_xo.tcl from rtl/ctrl/viv/tsr_ctrl_cdc.xdc"
puts $xh "set tsr_cdc_core_port ap_clk"
puts $xh "set tsr_cdc_rv_port   ap_clk_2"
puts -nonewline $xh $xbody
close $xh
set xg [ipx::get_file_groups xilinx_anylanguagesynthesis -of_objects $core]
set xf_ [ipx::add_file xdc/tsr_ctrl_cdc.xdc $xg]
set_property type xdc $xf_
set_property used_in {synthesis implementation out_of_context} $xf_
set_property processing_order LATE $xf_

# ---- control core TCMs in block RAM (CC §4.2, CC-3) ----
# rtl/ctrl/viv/tsr_ctrl_tcm.xdc: RAM_STYLE BLOCK and CASCADE_HEIGHT 1 on the ITCM / DTCM arrays, which
# overrides the size rule of rtl/tech/xilinx/tsr_sram_1r1w.sv (32K x 32 would be a URAM cascade: the worst
# path of the first whole-core synthesis, Ibex fetch -> ITCM, -1.257 ns). Synthesis only: the cells are
# the inferred arrays (the implementation sees RAMB36 primitives); scoped by cell names, so it needs no ports.
set tcmx $repo/rtl/ctrl/viv/tsr_ctrl_tcm.xdc
file copy -force $tcmx $ipdir/xdc/tsr_ctrl_tcm.xdc
set xf_ [ipx::add_file xdc/tsr_ctrl_tcm.xdc $xg]
set_property type xdc $xf_
set_property used_in {synthesis out_of_context} $xf_

ipx::create_xgui_files $core
ipx::update_checksums $core
set ok_kernel [ipx::check_integrity -kernel $core]
set ok_xrt    [ipx::check_integrity -xrt $core]
ipx::save_core $core
close_project -delete

# ---- .xo ----
package_xo -force -xo_path $outdir/$kernel.xo -kernel_name $kernel -ip_directory $ipdir -kernel_xml $kxml
if {![file exists $outdir/$kernel.xo]} {
  puts "PKG error: package_xo did not write $outdir/$kernel.xo"
  exit 1
}
# ---- source manifest ----
set mfiles [list]
foreach f [concat [list $corefl $pdir/filelist.f] $srcs [list $kxml $xdc $tcmx $script]] {
  lappend mfiles [expr {[string first "$repo/" $f] == 0 ? [string range $f [string length "$repo/"] end] : $f}]
}
set mf [open $outdir/$kernel.sources.sha256 w]
puts -nonewline $mf [exec sh -c {cd "$1" && shift && sha256sum -- "$@"} sh $repo {*}$mfiles]
puts $mf ""
close $mf
puts "PKG $kernel integrity kernel=$ok_kernel xrt=$ok_xrt"
puts "PKG $kernel cdc constraints xdc/tsr_ctrl_cdc.xdc, TCM block RAM constraints xdc/tsr_ctrl_tcm.xdc in the IP"
puts "PKG $kernel sources [llength $mfiles] files, manifest $outdir/$kernel.sources.sha256"
puts "PKG $kernel xo $outdir/$kernel.xo"

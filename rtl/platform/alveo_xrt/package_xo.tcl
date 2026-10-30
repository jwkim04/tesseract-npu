# rtl/platform/alveo_xrt/package_xo.tcl — package a Tesseract U200 RTL kernel as a Vitis .xo
# (Vivado batch; docs/platform-u200.md §6).
#
# usage: vivado -mode batch -nojournal -nolog -source rtl/platform/alveo_xrt/package_xo.tcl \
#          -tclargs <repo> <kernel> <outdir> [core_filelist]
#   kernel         tsr_u200_k1 | tsr_u200_k4 (module name = kernel name; kernel_k1.xml / kernel_k4.xml)
#   outdir         the .xo is written to <outdir>/<kernel>.xo (hw/u200/build_xclbin.sh: build/u200/xo);
#                  the IP and the temporary project go to <outdir>/<kernel>_pkg/
#   core_filelist  core sources (default rtl/filelist_xilinx.f: the Xilinx technology wrappers,
#                  behaviour identical to rtl/filelist.f); rtl/platform/alveo_xrt/filelist.f is added
# Steps: a project with every source (SystemVerilog) and the kernel as top -> ipx::package_project
# (sources imported into the IP) -> Vitis kernel properties (sdx_kernel rtl, no user parameters,
# ap_clk associated with s_axi_control / m_axi_ch<c> / ap_rst_n, FREQ_HZ 300 MHz, m_axi burst and
# outstanding hints, the s_axi_control register map with the args of kernel_<k>.xml) ->
# ipx::check_integrity -kernel / -xrt -> package_xo with kernel_<k>.xml -> the source manifest
# <outdir>/<kernel>.sources.sha256 (sha256sum format, paths relative to <repo>: both filelists, every
# source, kernel_<k>.xml and this script), so `sha256sum -c` in <repo> tells whether the .xo still
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
proc read_fl {fl} {
  set dir [file dirname $fl]
  set out [list]
  set f [open $fl r]
  foreach line [split [read $f] "\n"] {
    regsub {//.*$} $line "" line
    set line [string trim $line]
    if {$line ne ""} { lappend out [file normalize $dir/$line] }
  }
  close $f
  return $out
}
if {![file exists $corefl]} {
  puts "PKG error: core filelist $corefl not found"
  exit 2
}
set srcs [concat [read_fl $corefl] [read_fl $pdir/filelist.f]]

create_project -force kpack $work/proj -part xcu200-fsgd2104-2-e
add_files -norecurse $srcs
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

# s_axi_control register map: the args of kernel_<k>.xml (ap_ctrl_none, docs/design EX §9.1: no
# AP_CTRL / GIE / IER / ISR block; hw/u200/build_xclbin.sh does not package the k*
# kernels until Y5, whose first smoke link confirms this map, fallback in EX §9.1)
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
foreach f [concat [list $corefl $pdir/filelist.f] $srcs [list $kxml $script]] {
  lappend mfiles [expr {[string first "$repo/" $f] == 0 ? [string range $f [string length "$repo/"] end] : $f}]
}
set mf [open $outdir/$kernel.sources.sha256 w]
puts -nonewline $mf [exec sh -c {cd "$1" && shift && sha256sum -- "$@"} sh $repo {*}$mfiles]
puts $mf ""
close $mf
puts "PKG $kernel integrity kernel=$ok_kernel xrt=$ok_xrt"
puts "PKG $kernel sources [llength $mfiles] files, manifest $outdir/$kernel.sources.sha256"
puts "PKG $kernel xo $outdir/$kernel.xo"

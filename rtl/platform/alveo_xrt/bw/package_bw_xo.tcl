# rtl/platform/alveo_xrt/bw/package_bw_xo.tcl — package the M0 bandwidth kernel tsr_bw_k4 as a Vitis
# .xo (Vivado batch; docs/m0-bw.md §4). The flow of ../package_xo.tcl (the adapter's) for this kernel.
#
# usage: vivado -mode batch -nojournal -nolog -source rtl/platform/alveo_xrt/bw/package_bw_xo.tcl \
#          -tclargs <repo> <outdir> [core|wide]
#   outdir   the .xo is written to <outdir>/tsr_bw_k4.xo (hw/u200/bw/build_bw_xclbin.sh: build/u200/xo,
#            build/u200/xo/wide for `wide`); the IP and the temporary project go to <outdir>/tsr_bw_k4_pkg/
#   hints    m_axi interconnect hints of m_axi_ch0..3 (component.xml bus parameters, which the v++
#            linker uses to size the kernel-side interconnect):
#              core (default)  MAX_BURST_LENGTH 64, NUM_READ_OUTSTANDING 16, NUM_WRITE_OUTSTANDING 32 =
#                              the Tesseract kernels' (../package_xo.tcl), so M0 measures the memory
#                              path the core gets (docs/m0-bw.md §1; lint.sh checks the two agree)
#              wide            64 / 64 / 64, the engine's maximum: shows whether the interconnect
#                              sized for the core limits the outstanding sweep above 16 / 32
# Sources: rtl/platform/alveo_xrt/bw/filelist.f (complete; no technology wrappers are involved).
# Steps: a project with every source (SystemVerilog) and tsr_bw_k4 as top -> ipx::package_project
# (sources imported into the IP) -> Vitis kernel properties (sdx_kernel rtl, no user parameters,
# ap_clk associated with s_axi_control / m_axi_ch0..3 / ap_rst_n, FREQ_HZ 300 MHz, the m_axi hints)
# -> the s_axi_control register map with the args of kernel_bw_k4.xml -> ipx::check_integrity
# -kernel / -xrt -> package_xo with kernel_bw_k4.xml -> the source manifest
# <outdir>/tsr_bw_k4.sources.sha256 (sha256sum format, paths relative to <repo>: filelist.f, every
# source, kernel_bw_k4.xml and this script; hw/u200/bw/build_bw_xclbin.sh repackages when
# `sha256sum -c` of it fails in <repo>).
# Prints "PKG tsr_bw_k4 hints <core|wide> ..." and "PKG tsr_bw_k4 xo <path>" on success.
set script [file normalize [info script]]
set repo   [file normalize [lindex $argv 0]]
set outdir [file normalize [lindex $argv 1]]
set hints  [expr {[llength $argv] > 2 ? [lindex $argv 2] : "core"}]
set kernel tsr_bw_k4
# m_axi hints per variant (core: keep equal to ../package_xo.tcl)
array set hint_vals {
  core {MAX_BURST_LENGTH 64 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 32}
  wide {MAX_BURST_LENGTH 64 NUM_READ_OUTSTANDING 64 NUM_WRITE_OUTSTANDING 64}
}
if {![info exists hint_vals($hints)]} {
  puts "PKG error: unknown hints '$hints' (core | wide)"
  exit 2
}
set nch    4
set_param general.maxThreads 4

set bdir  $repo/rtl/platform/alveo_xrt/bw
set kxml  $bdir/kernel_bw_k4.xml
set work  $outdir/${kernel}_pkg
set ipdir $work/ip
file delete -force $work
file mkdir $work
file delete -force $outdir/$kernel.xo $outdir/$kernel.sources.sha256

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
set srcs [read_fl $bdir/filelist.f]

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
for {set c 0} {$c < $nch} {incr c} {
  set m m_axi_ch$c
  ipx::associate_bus_interfaces -busif $m -clock ap_clk $core
  set bif [ipx::get_bus_interfaces $m -of_objects $core]
  foreach {p v} $hint_vals($hints) {
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

# s_axi_control register map: ap_ctrl_hs block and the args of kernel_bw_k4.xml
set mm [ipx::add_memory_map -quiet s_axi_control $core]
set ab [ipx::add_address_block -quiet reg0 $mm]
set_property range 0x1000 $ab
foreach {name off} {CTRL 0x000 GIER 0x004 IP_IER 0x008 IP_ISR 0x00C} {
  set reg [ipx::add_register $name $ab]
  set_property address_offset $off $reg
  set_property size 32 $reg
}
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
foreach f [concat [list $bdir/filelist.f] $srcs [list $kxml $script]] {
  lappend mfiles [expr {[string first "$repo/" $f] == 0 ? [string range $f [string length "$repo/"] end] : $f}]
}
set mf [open $outdir/$kernel.sources.sha256 w]
puts -nonewline $mf [exec sh -c {cd "$1" && shift && sha256sum -- "$@"} sh $repo {*}$mfiles]
puts $mf ""
close $mf
puts "PKG $kernel sources [llength $mfiles] files, manifest $outdir/$kernel.sources.sha256"
puts "PKG $kernel hints $hints ($hint_vals($hints))"
puts "PKG $kernel integrity kernel=$ok_kernel xrt=$ok_xrt"
puts "PKG $kernel xo $outdir/$kernel.xo"

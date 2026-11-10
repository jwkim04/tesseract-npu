# rtl/filelist.tcl — Tcl reader of the Verilator filelists (rtl/filelist.f, rtl/filelist_xilinx.f
# and the lists they nest with -F, e.g. rtl/ctrl/filelist.f) for the Vivado flows (tb/core/viv,
# tb/tech/viv, tb/platform/viv, rtl/platform/alveo_xrt/package_xo.tcl).
#
#   source $repo/rtl/filelist.tcl
#   set fl [tsr_read_filelist $repo/rtl/filelist_xilinx.f]
#   dict get $fl srcs      HDL files in list order (normalized paths)
#   dict get $fl incdirs   +incdir+ directories
#   dict get $fl defines   +define+ NAME[=VALUE], plus FPGA_XILINX (Ibex's DSP pragmas, as
#                          rtl/ctrl/viv/ctrl_ooc.tcl), for synth_design -verilog_define
#   tsr_synth_opts $fl     the synth_design options -include_dirs / -verilog_define (empty parts
#                          left out)
# Rules (as Verilator's -F): // comments; paths relative to the directory of the list that names
# them; -F / -f <list> nests a list in place; Verilator configuration files (*.vlt) are skipped.

proc tsr_read_filelist {fl {acc ""}} {
  if {$acc eq ""} { set acc [dict create srcs {} incdirs {} defines {FPGA_XILINX}] }
  set dir [file dirname [file normalize $fl]]
  set f [open $fl r]
  set lines [split [read $f] "\n"]
  close $f
  foreach line $lines {
    regsub {//.*$} $line "" line
    set line [string trim $line]
    if {$line eq ""} { continue }
    if {[regexp {^-[Ff]\s+(.*)$} $line -> sub]} {
      set acc [tsr_read_filelist [file normalize $dir/[string trim $sub]] $acc]
    } elseif {[regexp {^\+incdir\+(.*)$} $line -> d]} {
      set d [file normalize $dir/$d]
      if {[lsearch -exact [dict get $acc incdirs] $d] < 0} { dict lappend acc incdirs $d }
    } elseif {[regexp {^\+define\+(.*)$} $line -> d]} {
      if {[lsearch -exact [dict get $acc defines] $d] < 0} { dict lappend acc defines $d }
    } elseif {[string match *.vlt $line]} {
      continue
    } else {
      dict lappend acc srcs [file normalize $dir/$line]
    }
  }
  return $acc
}

proc tsr_synth_opts {fl} {
  set o [list]
  if {[llength [dict get $fl incdirs]]} { lappend o -include_dirs [dict get $fl incdirs] }
  if {[llength [dict get $fl defines]]} { lappend o -verilog_define [dict get $fl defines] }
  return $o
}

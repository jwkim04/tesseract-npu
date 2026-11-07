# tb/at/viv/lsel_ooc.tcl — out-of-context synthesis of a lane-select wrapper (tb/at/viv/tsr_lsel_flat.sv) on the U200
# part from rtl/filelist_xilinx.f, and its funcsim netlist for tb/at/viv/lsel_net.sh.
# usage: vivado -mode batch -nojournal -nolog -source tb/at/viv/lsel_ooc.tcl -tclargs <repo> <top> <outdir>
set repo [lindex $argv 0]
set top  [lindex $argv 1]
set out  [lindex $argv 2]
file mkdir $out
set_param general.maxThreads 4
source $repo/rtl/filelist.tcl
set fld [tsr_read_filelist $repo/rtl/filelist_xilinx.f]
read_verilog -sv [concat [dict get $fld srcs] $repo/tb/at/viv/tsr_lsel_flat.sv]
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top {*}[tsr_synth_opts $fld]
write_verilog -mode funcsim -force $out/net.v
# how many loads the select inputs have (a dropped bit of l shows here too)
foreach p [get_ports -quiet {l[*]}] {
  puts "LSEL [get_property NAME $p] loads [llength [get_pins -of [get_nets -of $p] -filter {DIRECTION == IN}]]"
}
exit

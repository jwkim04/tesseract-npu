# tb/tech/viv/tech_ooc.tcl — out-of-context synthesis / implementation of one module of the Xilinx
# target (rtl/filelist_xilinx.f: rtl/tech/xilinx wrappers) at 300 MHz on the U200 part, with a
# summary of how the RAMs and DSPs were mapped (opt-in: needs Vivado; also the synthesis step of
# the netlist simulations, tb/tech/viv/net_sim.sh).
#
# usage: vivado -mode batch -nojournal -nolog -source tb/tech/viv/tech_ooc.tcl \
#          -tclargs <repo> <top> <outdir> [synth_only] [netlist] [filelist=<f>] [extra=<f>] [NAME=VALUE ...]
#   NAME=VALUE  synth_design -generic of <top> (e.g. T_MAX=8 MAC_LAT=5)
#   synth_only  stop after synthesis (utilisation and mapping summary)
#   netlist     write the synthesised netlist for simulation: <outdir>/net.v (write_verilog
#               -mode funcsim), used by tb/tech/viv/net_sim.sh
#   filelist=   another filelist (default rtl/filelist_xilinx.f)
#   extra=<f>   one more source file (e.g. the flat-port wrapper of the netlist simulation)
# Constraints as tb/core/viv/core_ooc.tcl: 3.333 ns clock, every I/O budgeted 1.6 ns outside (the
# neighbours are registered); r2r = register-to-register only. Prints (grep "TOOC"):
#   TOOC util(<stage>) LUT .. FF .. DSP .. RAMB36 .. RAMB18 .. URAM .. LUTRAM .. SRL .. CARRY8 ..
#   TOOC dsp <count> <AREG/BREG/DREG/ADREG/MREG/PREG/AMULTSEL/PREADDINSEL ...>   (per setting)
#   TOOC dsp_in <hier> <count>                 DSPs per tsr_mac_lane instance (u_mac)
#   TOOC uram <count> <OREG/REG_CAS/CASCADE_ORDER ...> rdb_wr_a=<GROUND|POWER|signal> rdb_wr_b=..
#   TOOC bram <count> <DOA_REG/DOB_REG/WRITE_MODE/READ_WIDTH ...>
#   TOOC <stage> wns <ns> whs <ns> | TOOC <stage> r2r_wns ... | TOOC <stage> path ...
set repo [lindex $argv 0]
set top  [lindex $argv 1]
set out  [lindex $argv 2]
set synth_only 0
set netlist 0
set flist $repo/rtl/filelist_xilinx.f
set extra [list]
set gens [list]
foreach a [lrange $argv 3 end] {
  if {$a eq "synth_only"} {
    set synth_only 1
  } elseif {$a eq "netlist"} {
    set netlist 1
  } elseif {[string match "filelist=*" $a]} {
    set flist [string range $a 9 end]
  } elseif {[string match "extra=*" $a]} {
    lappend extra [string range $a 6 end]
  } else {
    lappend gens -generic $a
  }
}
set_param general.maxThreads 4
file mkdir $out

# the filelist with its nested lists, include directories and defines (rtl/filelist.tcl)
source $repo/rtl/filelist.tcl
set fld [tsr_read_filelist $flist]
read_verilog -sv [concat [dict get $fld srcs] $extra]
synth_design -mode out_of_context -part xcu200-fsgd2104-2-e -top $top {*}[tsr_synth_opts $fld] {*}$gens
create_clock -name clk -period 3.333 [get_ports clk]
set_input_delay  -clock clk 1.6 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock clk 1.6 [get_ports -filter {DIRECTION == OUT}]
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -file $out/util_hier.rpt
if {$netlist} {
  write_verilog -mode funcsim -force $out/net.v
}

proc util {tag} {
  set c(LUT)    [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LUT*}]]
  set c(FF)     [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ FD*}]]
  set c(DSP)    [llength [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}]]
  set c(RAMB36) [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36*}]]
  set c(RAMB18) [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18*}]]
  set c(URAM)   [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ URAM288*}]]
  set c(LUTRAM) [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D*}]]
  set c(SRL)    [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ SRL*}]]
  set c(CARRY8) [llength [get_cells -quiet -hierarchical -filter {REF_NAME == CARRY8}]]
  set c(LATCH)  [llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LD*}]]
  set s "TOOC util($tag)"
  foreach k {LUT FF DSP RAMB36 RAMB18 URAM LUTRAM SRL CARRY8 LATCH} { append s " $k $c($k)" }
  puts $s
}

# Histogram of a property tuple over cells.
proc histo {tag cells props {extra ""}} {
  set h [dict create]
  foreach c $cells {
    set k ""
    foreach p $props { append k " $p=[get_property -quiet $p $c]" }
    if {$extra ne ""} { append k [$extra $c] }
    dict incr h $k
  }
  dict for {k n} $h { puts "TOOC $tag $n$k" }
}

proc pin_src {c pin} {
  set n [get_nets -quiet -of_objects [get_pins -quiet $c/$pin]]
  if {$n eq ""} { return "open" }
  set t [get_property -quiet TYPE $n]
  if {$t eq "GROUND" || $t eq "POWER"} { return $t }
  return "signal"
}
# RDB_WR_x = 0 read / 1 write (the FIRST URAM of a cascade carries it down the chain): read-first
# needs the read on port A (executed before port B).
proc uram_ports {c} { return " rdb_wr_a=[pin_src $c RDB_WR_A] rdb_wr_b=[pin_src $c RDB_WR_B]" }

proc mapping {} {
  set dsps [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2}]
  histo dsp $dsps {AREG ACASCREG BREG BCASCREG DREG ADREG MREG PREG CREG AMULTSEL BMULTSEL PREADDINSEL USE_MULT}
  # DSPs per MAC lane instance (path up to and including u_mac)
  set per [dict create]
  foreach c $dsps {
    set n [get_property NAME $c]
    if {[regexp {^(.*u_mac)/} $n -> h]} { dict incr per $h } else { dict incr per "(not in u_mac)" }
  }
  dict for {k n} $per { puts "TOOC dsp_in $k $n" }
  histo uram [get_cells -quiet -hierarchical -filter {REF_NAME =~ URAM288*}] \
    {OREG_A OREG_B REG_CAS_A REG_CAS_B CASCADE_ORDER_A CASCADE_ORDER_B} uram_ports
  histo bram [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB*}] \
    {REF_NAME DOA_REG DOB_REG WRITE_MODE_A WRITE_MODE_B READ_WIDTH_A READ_WIDTH_B WRITE_WIDTH_B}
  histo lutram [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAM*M* || REF_NAME =~ RAM*S* || REF_NAME =~ RAM*D*}] {REF_NAME}
}

proc paths {tag n} {
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  puts "TOOC $tag wns $wns whs $whs"
  set r2r [get_timing_paths -setup -max_paths 1 -from [all_registers] -to [all_registers]]
  if {$r2r ne ""} {
    puts "TOOC $tag r2r_wns [get_property SLACK $r2r] levels [get_property LOGIC_LEVELS $r2r] [get_property STARTPOINT_PIN $r2r] -> [get_property ENDPOINT_PIN $r2r]"
  }
  # Worst register-to-register path per (start group, end group): indices stripped.
  set seen [dict create]
  foreach p [get_timing_paths -setup -max_paths 4000 -nworst 1 -from [all_registers] -to [all_registers]] {
    set sp [get_property STARTPOINT_PIN $p]
    set ep [get_property ENDPOINT_PIN $p]
    regsub -all {\[[0-9]+\]} [file dirname $sp] {[*]} sg
    regsub -all {\[[0-9]+\]} [file dirname $ep] {[*]} eg
    set key "$sg>$eg"
    if {![dict exists $seen $key]} {
      dict set seen $key 1
      if {[dict size $seen] <= $n} {
        puts "TOOC $tag path [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] $sp -> $ep"
      }
    }
  }
}

util synth
mapping
paths synth 12
if {$synth_only} { exit }

opt_design
place_design
phys_opt_design
route_design
report_timing_summary -max_paths 30 -file $out/timing.rpt
report_utilization -file $out/util.rpt
report_drc -file $out/drc.rpt
util routed
paths routed 20

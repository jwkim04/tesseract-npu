# Post-system-link BD hook, before synthesis (CC §11.1, §14.3).
# The U200 platform exposes clock IDs 0/1 as ss_ucs/aclk_kernel_00/01.
# Check actual nets, including each proc_sys_reset's clock, rather than the
# requested frequency attributes. Scalable clock periods are checked after synth.
source [file join [file dirname [file normalize [info script]]] clock_common.tcl]
proc tsr_clock_check_bd {} {
  puts "CLKBD checking kernel clock/reset connections"
  set tsr_bd_kernel [tsr_clock_one [get_bd_cells -quiet -hierarchical -filter {
    VLNV =~ tesseract:kernel:tsr_u200_k*:*
  }] {Tesseract compute unit}]
  set tsr_bd_kernel [get_property NAME $tsr_bd_kernel]
  foreach {port index reset} {ap_clk 00 ap_rst_n ap_clk_2 01 ap_rst_n_2} {
    set sink [tsr_clock_one [get_bd_pins -quiet $tsr_bd_kernel/$port] $port]
    set source_pin [tsr_clock_one [get_bd_pins -quiet ss_ucs/aclk_kernel_$index] "platform clock $index"]
    set clock_net [tsr_clock_one [get_bd_nets -quiet -of_objects $sink] "$port net"]
    set source_net [tsr_clock_one [get_bd_nets -quiet -of_objects $source_pin] "clock $index net"]
    if {$clock_net ne $source_net} {
      error "CLKBD $port must use $source_pin; actual net $clock_net, expected $source_net"
    }
    set reset_pin [tsr_clock_one [get_bd_pins -quiet $tsr_bd_kernel/$reset] $reset]
    set reset_net [tsr_clock_one [get_bd_nets -quiet -of_objects $reset_pin] "$reset net"]
    set driver [tsr_clock_one [get_bd_pins -quiet -of_objects $reset_net -filter {DIR == O}] "$reset driver"]
    set reset_cell [tsr_clock_one [get_bd_cells -quiet -of_objects $driver] "$reset source cell"]
    if {![string match xilinx.com:ip:proc_sys_reset:* [get_property VLNV $reset_cell]] ||
        [file tail [get_property NAME $driver]] ne "peripheral_aresetn"} {
      error "CLKBD $reset must use proc_sys_reset/peripheral_aresetn; actual $driver"
    }
    set reset_clock [tsr_clock_one [get_bd_pins -quiet $reset_cell/slowest_sync_clk] "$reset source clock"]
    set reset_clock_net [tsr_clock_one [get_bd_nets -quiet -of_objects $reset_clock] "$reset clock net"]
    if {$reset_clock_net ne $clock_net} {
      error "CLKBD $reset source uses $reset_clock_net, but $port uses $clock_net"
    }
    puts "CLKBD $port <- $source_pin; $reset <- $driver"
  }
  puts "CLKBD PASS"
}
tsr_clock_check_bd

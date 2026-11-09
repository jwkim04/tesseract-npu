# opt_design pre-hook: inspect the synthesized linked design before placement.
# CC §11.1 and §14.3. The source identities are checked earlier in the BD hook.
# Check the active kernel config's request before final frequency scaling.
# These PERIOD properties retain the request after routing; final acceptance
# (k4: core from 200 MHz up to the request and RV at the request, k1: core >= 250 MHz) uses Vitis's _new_clk_freq
# in clock_final_check.tcl instead. The period window is 0.5 % around the request in
# the kernel's own config, so it follows the config and needs no edit for a new clock.
# A 125 MHz fallback requires an explicit clock-request/check-window change.
source [file join [file dirname [file normalize [info script]]] clock_common.tcl]
proc tsr_clock_check_net {} {
  puts "CLKNET checking synthesized kernel clocks and reset sources"
  set tsr_clock_kernel [tsr_clock_one [get_cells -quiet -hierarchical -filter {
    REF_NAME =~ tsr_u200_k* || ORIG_REF_NAME =~ tsr_u200_k*
  }] {Tesseract kernel instance}]
  set requested [tsr_clock_request $tsr_clock_kernel]
  set tsr_clock_core_regs [get_cells -quiet -hierarchical -filter {NAME =~ *u_cdc/lv_q_reg* && IS_SEQUENTIAL}]
  set tsr_clock_rv_regs [get_cells -quiet -hierarchical -filter {NAME =~ *u_cdc/u_sync_lv/s_q_reg* && IS_SEQUENTIAL}]
  if {[llength $tsr_clock_core_regs] == 0 || [llength $tsr_clock_rv_regs] == 0} {
    error "CLKNET missing core/RV CDC reference registers"
  }
  set tsr_clock_core_pins [get_pins -quiet -of_objects $tsr_clock_core_regs -filter {REF_PIN_NAME == C}]
  set tsr_clock_rv_pins [get_pins -quiet -of_objects $tsr_clock_rv_regs -filter {REF_PIN_NAME == C}]
  set tsr_clock_core [tsr_clock_one [lsort -unique [get_clocks -quiet -of_objects $tsr_clock_core_pins]] {core clock}]
  set tsr_clock_rv [tsr_clock_one [lsort -unique [get_clocks -quiet -of_objects $tsr_clock_rv_pins]] {RV clock}]
  if {$tsr_clock_core eq $tsr_clock_rv} { error "CLKNET core and RV clocks alias: $tsr_clock_core" }
  foreach clock [list $tsr_clock_core $tsr_clock_rv] expected $requested {
    set low [expr {$expected * 0.995}]
    set high [expr {$expected * 1.005}]
    set period [get_property PERIOD [get_clocks $clock]]
    if {![string is double -strict $period] || !($period >= $low && $period <= $high)} {
      error "CLKNET $clock period '$period' ns is outside $low..$high ns"
    }
    puts "CLKNET clock $clock period $period ns"
  }
  set tsr_clock_hart_regs [get_cells -quiet -hierarchical -filter {NAME =~ *u_ctrl/u_rv/* && REF_NAME =~ FD*}]
  if {[llength $tsr_clock_hart_regs] == 0} { error "CLKNET no RV hart registers found" }
  set tsr_clock_hart [lsort -unique [get_clocks -quiet -of_objects [get_pins -quiet -of_objects $tsr_clock_hart_regs -filter {REF_PIN_NAME == C}]]]
  if {[llength $tsr_clock_hart] != 1 || $tsr_clock_hart ne $tsr_clock_rv} {
    error "CLKNET RV hart registers use '$tsr_clock_hart', expected $tsr_clock_rv"
  }
  # A union alone misses an unclocked register when all other FFs use clk_rv.
  foreach reg $tsr_clock_hart_regs {
    set pin [tsr_clock_one [get_pins -quiet -of_objects $reg -filter {REF_PIN_NAME == C}] "$reg clock pin"]
    set clocks [get_clocks -quiet -of_objects $pin]
    if {[llength $clocks] != 1 || $clocks ne $tsr_clock_rv} {
      error "CLKNET $reg clock '$clocks', expected $tsr_clock_rv"
    }
  }
  puts "CLKNET RV hart clock checked on [llength $tsr_clock_hart_regs] registers"
  set tsr_clock_kernel [get_property NAME $tsr_clock_kernel]
  set tsr_reset_seen {}
  foreach {reset clock} [list ap_rst_n $tsr_clock_core ap_rst_n_2 $tsr_clock_rv] {
    set pin [tsr_clock_one [get_pins -quiet $tsr_clock_kernel/$reset] $reset]
    set nets [get_nets -quiet -segments -of_objects $pin]
    if {[llength $nets] == 0} { error "CLKNET $reset is unconnected" }
    foreach net $nets {
      if {$net in $tsr_reset_seen} { error "CLKNET kernel resets alias on $net" }
    }
    set tsr_reset_seen [concat $tsr_reset_seen $nets]
    # Stop at sequential startpoints, allowing an inverted reset output as well
    # as a direct FF output. Do not trace through the reset generator's flops.
    set roots [all_fanin -flat -startpoints_only -to $pin]
    if {[llength $roots] == 0} { error "CLKNET $reset has no reset-generator startpoints" }
    foreach root $roots {
      if {[llength [get_cells -quiet -of_objects $root -filter {IS_SEQUENTIAL}]] != 1} {
        error "CLKNET $reset has a non-sequential reset source: $root"
      }
    }
    set regs [get_cells -quiet -of_objects $roots -filter {IS_SEQUENTIAL}]
    set clocks [lsort -unique [get_clocks -quiet -of_objects [get_pins -quiet -of_objects $regs -filter {REF_PIN_NAME == C}]]]
    if {[llength $clocks] != 1 || $clocks ne $clock} {
      error "CLKNET $reset source clocks '$clocks', expected $clock"
    }
    puts "CLKNET $reset source clock $clock"
  }
  puts "CLKNET PASS"
}
tsr_clock_check_net

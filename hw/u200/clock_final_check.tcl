# WRITE_BITSTREAM.PRE user hook, after Vitis's timing/frequency scaling hook.
# Scaling writes _new_clk_freq in the implementation run directory without
# changing the clock PERIOD properties. Preserve its MHz decimal precision.
source [file join [file dirname [file normalize [info script]]] clock_common.tcl]

# Accepted scaled frequencies in MHz by kernel: {core_low core_high rv_low rv_high}.
#   k1  core request 300, floor 250. RV request 166 (Vitis truncates 166.67), 5 % tolerance.
#   k4  the card images of tsr_u200_k4 accept exactly the request of hw/u200/k4.cfg (kernel_frequency=0:<core>|1:<rv>; 200 / 100
#       in the images of the reduced and the full-feature cores, user decision 2026-11-06; 250 / 100 for the 250 MHz work, docs/platform-u200.md
#       §8.10): a core or RV clock that Vitis scaled down fails the run. The 0.1 MHz above the request only absorbs the decimal
#       precision of _new_clk_freq.
# Faster clocks or the 125 MHz fallback require an explicit request and window change after the timing
# choice has been validated.
proc tsr_clock_final_window {kind} {
  switch -- $kind {
    k1 { return {250.0 300.1 158.33 166.8} }
    k4 {
      lassign [tsr_clock_request_mhz k4] core rv
      return [list $core [expr {$core + 0.1}] $rv [expr {$rv + 0.1}]]
    }
  }
  error "CLKFINAL no acceptance window for kernel $kind"
}
proc tsr_clock_check_final {} {
  # The design is open at this step (Vitis's scaling hook has just timed it): the kernel cell names the window.
  set kernel [tsr_clock_one [get_cells -quiet -hierarchical -filter {
    REF_NAME =~ tsr_u200_k* || ORIG_REF_NAME =~ tsr_u200_k*
  }] {Tesseract kernel instance}]
  set kind [tsr_clock_kernel_kind $kernel CLKFINAL]
  lassign [tsr_clock_final_window $kind] core_low core_high rv_low rv_high
  set path [file join [pwd] _new_clk_freq]
  puts "CLKFINAL checking $path for $kind"
  set f [open $path r]
  set data [read $f]
  close $f
  set seen [dict create]
  foreach line [split $data \n] {
    set line [string trim $line]
    if {![string match kernel:* $line]} continue
    set fields [split $line :]
    if {[llength $fields] != 4} { error "CLKFINAL malformed clock record: $line" }
    lassign $fields rec id pin mhz
    if {$id ni {0 1} || [dict exists $seen $id]} {
      error "CLKFINAL unexpected or duplicate clock ID: $line"
    }
    set expected [format ss_ucs/aclk_kernel_%02d $id]
    if {$pin ne $expected || ![string is double -strict $mhz]} {
      error "CLKFINAL invalid clock source/frequency: $line"
    }
    lassign [expr {$id == 0 ? [list $core_low $core_high] : [list $rv_low $rv_high]}] low high
    if {!($mhz >= $low && $mhz <= $high)} {
      error "CLKFINAL clock $id frequency $mhz MHz is outside $low..$high MHz ($kind)"
    }
    dict set seen $id $mhz
    puts "CLKFINAL clock $id $pin $mhz MHz"
  }
  if {[dict size $seen] != 2} { error "CLKFINAL missing core or RV clock in $path" }
  puts "CLKFINAL PASS $kind"
}
tsr_clock_check_final

# WRITE_BITSTREAM.PRE user hook, after Vitis's timing/frequency scaling hook.
# Scaling writes _new_clk_freq in the implementation run directory without
# changing the clock PERIOD properties. Preserve its MHz decimal precision.
proc tsr_clock_check_final {} {
  set path [file join [pwd] _new_clk_freq]
  puts "CLKFINAL checking $path"
  set f [open $path r]
  set data [read $f]
  close $f
  set seen [dict create]
  foreach line [split $data \n] {
    set line [string trim $line]
    if {![string match kernel:* $line]} continue
    set fields [split $line :]
    if {[llength $fields] != 4} { error "CLKFINAL malformed clock record: $line" }
    lassign $fields kind id pin mhz
    if {$id ni {0 1} || [dict exists $seen $id]} {
      error "CLKFINAL unexpected or duplicate clock ID: $line"
    }
    set expected [format ss_ucs/aclk_kernel_%02d $id]
    if {$pin ne $expected || ![string is double -strict $mhz]} {
      error "CLKFINAL invalid clock source/frequency: $line"
    }
    # Final core acceptance floor is 250 MHz (K4 requests 250; K1 requests
    # 300 MHz). RV stays within the current request's 5% tolerance.
    # Faster RV clocks or the 125 MHz fallback require an explicit request
    # and gate change after the corresponding timing choice has been validated.
    lassign [expr {$id == 0 ? {250.0 300.1} : {158.33 166.8}}] low high
    if {!($mhz >= $low && $mhz <= $high)} {
      error "CLKFINAL clock $id frequency $mhz MHz is outside $low..$high MHz"
    }
    dict set seen $id $mhz
    puts "CLKFINAL clock $id $pin $mhz MHz"
  }
  if {[dict size $seen] != 2} { error "CLKFINAL missing core or RV clock in $path" }
  puts "CLKFINAL PASS"
}
tsr_clock_check_final

# INIT_DESIGN.PRE, before Vitis's INIT_DESIGN.POST clock override. Vitis 2024.2
# ocl_util passes Project to the part-keyed clk_wiz API and omits timePeriod.
# Keep its native generated-clock XDC flow; adapt only the calculator helpers.
# Signature: clk_wiz_v6_0/xgui/clk_wiz_v6_0.tcl GetClosestSolution call.
if {[version -short] ne "2024.2"} {
  error "CLKAPI compatibility hook is pinned to Vivado 2024.2"
}
if {![namespace exists ::ocl_util]} { error "CLKAPI missing Vitis ocl_util namespace" }
foreach tsr_api_helper {initialize_clkwiz_debug uninitialize_clkwiz_debug get_clkwiz_prop set_clkwiz_prop write_user_impl_clock_constraint} {
  if {[llength [info procs ::ocl_util::$tsr_api_helper]] != 1} {
    error "CLKAPI missing Vitis helper $tsr_api_helper"
  }
}
unset tsr_api_helper
if {[info exists ::tsr_clock_api::installed]} { return }

namespace eval ::tsr_clock_api {
  variable active 0
  variable part ""
  variable inst tsr_vitis_clock_compat
  variable chosen [dict create]

  proc positive {value what} {
    if {![string is double -strict $value] ||
        [catch {expr {$value > 0 && $value < Inf}} finite] || !$finite} {
      error "CLKAPI invalid $what: '$value' (need a finite positive number)"
    }
    return $value
  }
  proc cleanup {} {
    variable active
    variable part
    variable inst
    variable chosen
    if {$active} {
      try {
        UnInit_Clkwiz $part $inst
      } finally {
        set active 0
        set chosen [dict create]
      }
    }
  }
  proc require_active {} {
    variable active
    if {!$active} { error "CLKAPI calculator is not initialized" }
  }
}

proc ::ocl_util::initialize_clkwiz_debug {is_versal partinfo} {
  # INIT.PRE runs before create_project. Resolve PART only when the native
  # INIT.POST writer calls us, when the linked design/project is available.
  if {$is_versal || $partinfo ne "xcu200-fsgd2104-2-e" ||
      [get_property PART [current_project]] ne $partinfo} {
    error "CLKAPI compatibility hook requires the pinned non-Versal U200 part"
  }
  if {$::tsr_clock_api::active} { error "CLKAPI calculator is already initialized" }
  if {[llength [info commands ::Init_Clkwiz]] == 0} {
    load librdi_iptasks[info sharedlibextension]
  }
  set ::tsr_clock_api::part $partinfo
  set ::tsr_clock_api::active 1
  try {
    Init_Clkwiz $partinfo $::tsr_clock_api::inst $partinfo
  } on error {reason options} {
    catch {::tsr_clock_api::cleanup}
    return -options $options $reason
  }
}

proc ::ocl_util::uninitialize_clkwiz_debug {is_versal partinfo} {
  if {$is_versal || $partinfo ne $::tsr_clock_api::part} {
    error "CLKAPI calculator cleanup part mismatch"
  }
  ::tsr_clock_api::cleanup
}

proc ::ocl_util::set_clkwiz_prop {clock_freq_orig clock_freq} {
  ::tsr_clock_api::require_active
  try {
    ::tsr_clock_api::positive $clock_freq_orig "input frequency"
    ::tsr_clock_api::positive $clock_freq "requested frequency"
    set period [expr {1000.0 / $clock_freq_orig}]
    ::tsr_clock_api::positive $period "input period"
    set part $::tsr_clock_api::part
    set inst $::tsr_clock_api::inst
    SetClkwizProperty $part $inst UseFinePS true
    set ok [GetClosestSolution $part $inst $clock_freq 0 50 $clock_freq_orig 0 1 false false false false $period false 0]
    if {$ok ne "1"} { error "CLKAPI clock calculation failed: $ok" }
    set choices [dict create]
    foreach prop {ChosenM ChosenD ChosenDiv0} {
      dict set choices $prop [::tsr_clock_api::positive [GetClkwizProperty $part $inst $prop] $prop]
    }
    set m [dict get $choices ChosenM]
    set d [dict get $choices ChosenD]
    set o [dict get $choices ChosenDiv0]
    if {$m == 1 && $d == 1 && $o == 1} { error "CLKAPI rejected default 1:1 clock solution" }
    # The native writer floors all three fields before emitting multiply_by /
    # divide_by. Check the ratio it will actually write, including fractions.
    set fm [expr {floor($m)}]
    set fd [expr {floor($d)}]
    set fo [expr {floor($o)}]
    if {$fm < 1 || $fd < 1 || $fo < 1} { error "CLKAPI invalid effective clock divisors" }
    set effective [expr {$clock_freq_orig * $fm / ($fd * $fo)}]
    ::tsr_clock_api::positive $effective "effective frequency"
    if {abs($effective / $clock_freq - 1.0) > 0.01} {
      error "CLKAPI effective $effective MHz differs from requested $clock_freq MHz by more than 1% (M=$m D=$d O=$o)"
    }
    set ::tsr_clock_api::chosen $choices
    puts "CLKAPI part=$part input=$clock_freq_orig requested=$clock_freq M=$m D=$d O=$o effective=$effective MHz"
  } on error {reason options} {
    catch {::tsr_clock_api::cleanup}
    return -options $options $reason
  }
}

proc ::ocl_util::get_clkwiz_prop {prop} {
  ::tsr_clock_api::require_active
  if {![dict exists $::tsr_clock_api::chosen $prop]} {
    error "CLKAPI no validated clock property $prop"
  }
  return [dict get $::tsr_clock_api::chosen $prop]
}

# Preserve the native writer's namespace and XDC creation/read_xdc behavior.
# Always clean up if a later native operation fails after a successful solve.
rename ::ocl_util::write_user_impl_clock_constraint ::ocl_util::tsr_clock_native_write_user_impl_clock_constraint
proc ::ocl_util::write_user_impl_clock_constraint {inst freqs steps_log out is_versal} {
  try {
    ::ocl_util::tsr_clock_native_write_user_impl_clock_constraint $inst $freqs $steps_log $out $is_versal
  } finally {
    ::tsr_clock_api::cleanup
  }
}
# _vpl_pre_init already imports this command before the user INIT.PRE hook.
# Replace that imported alias too, so INIT.POST reaches the cleanup wrapper.
namespace eval :: { namespace import -force ::ocl_util::write_user_impl_clock_constraint }
set ::tsr_clock_api::installed 1
puts "CLKAPI installed Vivado 2024.2 U200 clock calculator compatibility"

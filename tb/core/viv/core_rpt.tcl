# tb/core/viv/core_rpt.tcl — reports of the whole-core out-of-context runs (sourced by
# tb/core/viv/core_ooc.tcl; also standalone on a checkpoint the flow wrote):
#
#   vivado -mode batch -nojournal -nolog -source tb/core/viv/core_rpt.tcl -tclargs <dcp> <tag> [<outdir>]
#
# Procs (all print lines starting with "COOC", grep them):
#   cooc_util <tag>         COOC util(<tag>) LUT .. FF .. DSP .. RAMB36 .. RAMB18 .. URAM .. LUTRAM .. SRL .. CARRY8 .. LATCH ..
#                           (primitive cell counts; LUT = LUT1..LUT6 cells, so two LUTs sharing a
#                           LUT6_2 site count 2) and COOC dev(<tag>) <report_utilization totals:
#                           CLB LUTs / registers / RAMB36 tiles / URAM / DSP, the device-capacity view>
#   cooc_hier <tag>         COOC hier(<tag>) <key> LUT .. FF .. DSP .. RAMB36 .. RAMB18 .. URAM .. LUTRAM .. SRL ..
#                           per hierarchy key (cooc_hkey below; every key also accumulates into
#                           its prefixes, and keys under mvm/cl<c> also into mvm/cl*: the sum
#                           over the clusters)
#   cooc_paths <tag> <n>    COOC <tag> wns .. whs .. tns .. fail_ep .. | r2r_wns .. | then the worst
#                           register-to-register path of the <n> worst (start group, end group)
#                           pairs: COOC <tag> path <slack> levels <n> <start pin> -> <end pin>
#   cooc_slr <tag> <outdir> placed designs: report_utilization -slr, cells per SLR per top-level
#                           hierarchy key (COOC slr(<tag>) <key> SLR0 .. SLR1 .. SLR2 ..), and the
#                           nets that cross SLRs grouped by driver (COOC sll(<tag>) <n> <group>)
# Hierarchy keys (cooc_hkey): csr cmdproc fetch dma qact spm arb<c> core_top, mvm/top
# (dispatch, broadcast source, SPM client mux, reset copies), mvm/pipes (per cluster: broadcast
# vpipe, result rpipe, weight-stream request rpipe / response vpipe), mvm/gather/{asm (the
# per-cluster assemblers), pipes (their start / job pipes), central},
# mvm/cl<c>/{top,front,act,back/mac<i>,back/tok<t>,back/shared}: back/mac<i> = MAC
# lane i (one per token pair with the xilinx DSP lane at T_MAX >= 2: tokens 2i, 2i+1), back/tok<t>
# = the per-token back end of token t (integer scale, i2f, fp multiply / subtract / accumulate /
# tree), back/shared = beat decode, delay lines, control.

proc cooc_hkey {n} {
  if {[regexp {^g_mvm\.u_mvm/(.*)$} $n -> r]} {
    if {[regexp {^g_cl\[([0-9]+)\]\.u_cl/(.*)$} $r -> c r2]} {
      if {[regexp {^u_back/(.*)$} $r2 -> r3]} {
        if {[regexp {^g_mac\[([0-9]+)\]\.} $r3 -> i]} { return "mvm/cl$c/back/mac$i" }
        if {[regexp {^g_tok\[([0-9]+)\]\.} $r3 -> t]} { return "mvm/cl$c/back/tok$t" }
        return "mvm/cl$c/back/shared"
      }
      if {[regexp {^u_front/} $r2]} { return "mvm/cl$c/front" }
      if {[regexp {^u_act/} $r2]}   { return "mvm/cl$c/act" }
      return "mvm/cl$c/top"
    }
    if {[regexp {^g_cl\[[0-9]+\]\.(u_bcp|u_rp|u_wqp|u_wrp)/} $r]} { return "mvm/pipes" }
    if {[regexp {^u_gather/(.*)$} $r -> r2]} {
      if {[regexp {^g_asm\[[0-9]+\]\.u_asm/} $r2]} { return "mvm/gather/asm" }
      if {[regexp {^g_asm\[[0-9]+\]\.(u_sp|u_jp)/} $r2]} { return "mvm/gather/pipes" }
      return "mvm/gather/central"
    }
    return "mvm/top"
  }
  if {[regexp {^g_ch\[([0-9]+)\]\.u_arb/} $n -> c]} { return "arb$c" }
  if {[regexp {^u_([a-z]+)/} $n -> b]} { return $b }
  return "core_top"
}

# Every prefix of a key, plus the mvm/cl* (sum over clusters) variants.
proc cooc_hkeys {k} {
  set out [list]
  set parts [split $k /]
  for {set i 1} {$i <= [llength $parts]} {incr i} {
    lappend out [join [lrange $parts 0 [expr {$i - 1}]] /]
  }
  if {[regexp {^mvm/cl[0-9]+(.*)$} $k -> rest]} {
    set parts [split "mvm/cl*$rest" /]
    for {set i 2} {$i <= [llength $parts]} {incr i} {
      lappend out [join [lrange $parts 0 [expr {$i - 1}]] /]
    }
  }
  return $out
}

proc cooc_filters {} {
  return [list \
    LUT    {REF_NAME =~ LUT*} \
    FF     {REF_NAME =~ FD*} \
    DSP    {REF_NAME == DSP48E2} \
    RAMB36 {REF_NAME =~ RAMB36*} \
    RAMB18 {REF_NAME =~ RAMB18*} \
    URAM   {REF_NAME =~ URAM288*} \
    LUTRAM {REF_NAME =~ RAM*M* || REF_NAME =~ RAMS* || REF_NAME =~ RAMD*} \
    SRL    {REF_NAME =~ SRL*} \
    CARRY8 {REF_NAME == CARRY8} \
    LATCH  {REF_NAME =~ LD*}]
}

proc cooc_util {tag} {
  set s "COOC util($tag)"
  foreach {k f} [cooc_filters] {
    append s " $k [llength [get_cells -quiet -hierarchical -filter $f]]"
  }
  puts $s
  # Device-capacity view (site-based totals of report_utilization)
  set r [report_utilization -return_string]
  set d "COOC dev($tag)"
  foreach {lbl re} {CLB_LUTs {\|\s*CLB LUTs\*?\s*\|\s*([0-9]+)} CLB_regs {\|\s*CLB Registers\s*\|\s*([0-9]+)} \
                    BRAM_tiles {\|\s*Block RAM Tile\s*\|\s*([0-9.]+)} URAM {\|\s*URAM\s*\|\s*([0-9]+)} \
                    DSP {\|\s*DSPs\s*\|\s*([0-9]+)}} {
    if {[regexp $re $r -> v]} { append d " $lbl $v" } else { append d " $lbl ?" }
  }
  puts $d
}

proc cooc_hier {tag} {
  set types [list]
  array set cnt {}
  set keys [dict create]
  foreach {k f} [cooc_filters] {
    if {$k eq "CARRY8" || $k eq "LATCH"} { continue }
    lappend types $k
    foreach n [get_property -quiet NAME [get_cells -quiet -hierarchical -filter $f]] {
      foreach h [cooc_hkeys [cooc_hkey $n]] {
        dict set keys $h 1
        if {[info exists cnt($h,$k)]} { incr cnt($h,$k) } else { set cnt($h,$k) 1 }
      }
    }
  }
  foreach h [lsort [dict keys $keys]] {
    set s "COOC hier($tag) $h"
    foreach k $types {
      if {[info exists cnt($h,$k)]} { append s " $k $cnt($h,$k)" } else { append s " $k 0" }
    }
    puts $s
  }
}

proc cooc_paths {tag n} {
  set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
  set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
  set tns "?"; set fep "?"
  set ts [report_timing_summary -no_detailed_paths -no_check_timing -return_string]
  if {[regexp {WNS\(ns\)\s+TNS\(ns\)\s+TNS Failing Endpoints.*?\n\s*-+.*?\n\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s+([0-9]+)} $ts -> w t f]} {
    set tns $t; set fep $f
  }
  puts "COOC $tag wns $wns whs $whs tns $tns fail_ep $fep"
  set r2r [get_timing_paths -setup -max_paths 1 -from [all_registers] -to [all_registers]]
  if {$r2r ne ""} {
    puts "COOC $tag r2r_wns [get_property SLACK $r2r] levels [get_property LOGIC_LEVELS $r2r] [get_property STARTPOINT_PIN $r2r] -> [get_property ENDPOINT_PIN $r2r]"
  }
  # Worst register-to-register path per (start group, end group); group = the cell's
  # hierarchy with every index stripped (all clusters / tokens / lanes fold together).
  set seen [dict create]
  foreach p [get_timing_paths -setup -max_paths 6000 -nworst 1 -from [all_registers] -to [all_registers]] {
    set sp [get_property STARTPOINT_PIN $p]
    set ep [get_property ENDPOINT_PIN $p]
    regsub -all {\[[0-9]+\]} [file dirname $sp] {[*]} sg
    regsub -all {\[[0-9]+\]} [file dirname $ep] {[*]} eg
    foreach v {sg eg} {
      regsub -all {_rep(__[0-9]+)?|_replica(_[0-9]+)?} [set $v] {} $v
      regsub {/RAM[A-H](_D1)?$} [set $v] {} $v
      regsub {(_[0-9]+)+$} [set $v] {} $v
    }
    set key "$sg>$eg"
    if {![dict exists $seen $key]} {
      dict set seen $key 1
      if {[dict size $seen] <= $n} {
        puts "COOC $tag path [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] $sp -> $ep"
      }
    }
  }
  # I/O paths (budgeted 1.6 ns outside), worst in and worst out
  foreach {lbl args} [list in [list -from [all_inputs] -to [all_registers]] out [list -from [all_registers] -to [all_outputs]]] {
    set p [get_timing_paths -setup -max_paths 1 {*}$args]
    if {$p ne ""} {
      puts "COOC $tag io_$lbl [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p]"
    }
  }
}

proc cooc_slr {tag out} {
  set slrs [lsort [get_slrs]]
  report_utilization -slr -file $out/util_slr_$tag.rpt
  # Cells per SLR per top-level key (leaf cells: LUT / FF / DSP / RAM primitives).
  array set cnt {}
  set keys [dict create]
  foreach slr $slrs {
    set cells [get_cells -quiet -of_objects [get_clock_regions -quiet -of_objects $slr] -filter {IS_PRIMITIVE}]
    foreach n [get_property -quiet NAME $cells] {
      set k [cooc_hkey $n]
      regsub {^(mvm/cl[0-9]+)/.*$} $k {\1} k
      dict set keys $k 1
      if {[info exists cnt($k,$slr)]} { incr cnt($k,$slr) } else { set cnt($k,$slr) 1 }
    }
  }
  foreach k [lsort [dict keys $keys]] {
    set s "COOC slr($tag) $k"
    foreach slr $slrs {
      if {[info exists cnt($k,$slr)]} { append s " $slr $cnt($k,$slr)" } else { append s " $slr 0" }
    }
    puts $s
  }
  # Nets crossing an SLR boundary, grouped by the driver's hierarchy (indices stripped).
  set nets [get_nets -quiet -hierarchical -top_net_of_hierarchical_group -filter {CROSSING_SLRS != ""}]
  puts "COOC sll($tag) total [llength $nets]"
  set g [dict create]
  foreach net $nets {
    set drv [get_pins -quiet -of_objects $net -filter {DIRECTION == OUT && IS_LEAF}]
    if {$drv eq ""} {
      set grp "(port) [get_property NAME $net]"
      regsub -all {\[[0-9]+\]} $grp {[*]} grp
    } else {
      set grp "[cooc_hkey [lindex $drv 0]] [file tail [file dirname [lindex $drv 0]]]"
      regsub -all {\[[0-9]+\]} $grp {[*]} grp
      regsub -all {_rep(__[0-9]+)?|_replica(_[0-9]+)?} $grp {} grp
    }
    dict incr g $grp
  }
  set l [list]
  dict for {k v} $g { lappend l [list $v $k] }
  foreach e [lsort -integer -decreasing -index 0 $l] {
    puts "COOC sll($tag) [lindex $e 0] [lindex $e 1]"
  }
}

# Standalone use on a checkpoint
if {![info exists ::cooc_flow] && [llength $argv] >= 2 && [string match "*.dcp" [lindex $argv 0]]} {
  set_param general.maxThreads 4
  open_checkpoint [lindex $argv 0]
  set tag [lindex $argv 1]
  set out [expr {[llength $argv] >= 3 ? [lindex $argv 2] : [file dirname [lindex $argv 0]]}]
  cooc_util $tag
  cooc_hier $tag
  cooc_paths $tag 30
  if {[llength [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE && STATUS == PLACED}]] > 0 ||
      [llength [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE && IS_FIXED}]] > 0} {
    cooc_slr $tag $out
  }
}

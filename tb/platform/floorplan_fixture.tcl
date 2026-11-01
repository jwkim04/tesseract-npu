# A small object model of the Vivado queries used by k4_floorplan.tcl. The
# primitive names include the synthesis reparenting seen in the failed k4 DCP.
# This tests ownership and split-pipe coverage without opening a checkpoint.
set fp_script [lindex $argv 0]
set fixture_mode [lindex $argv 1]
set fixture_case [lindex $argv 2]
set cells [dict create]
set pblocks [dict create]
set owners [dict create]
set placements [dict create]
set fanin [dict create]
set assignments [dict create]
set added {}
foreach s {0 1 2} { dict set pblocks pblock_dynamic_SLR$s 1 }

proc fixture_cell {name {type HIER}} {
  global cells
  set p [file dirname $name]
  if {$p ne "." && ![dict exists $cells $p]} { fixture_cell $p }
  dict set cells $name $type
}
proc fixture_match {pattern value} {
  return [string match [string map [list {[} {\[} {]} {\]}] $pattern] $value]
}
proc fixture_filter {name filter} {
  global cells
  if {$filter eq ""} { return 1 }
  foreach alternative [split [string map {|| |} $filter] |] {
    set yes 1
    foreach term [split [string map {&& &} $alternative] &] {
      set term [string trim $term]
      if {$term eq "IS_PRIMITIVE"} {
        set v [expr {[dict get $cells $name] ne "HIER"}]
      } elseif {$term eq "!IS_PRIMITIVE"} {
        set v [expr {[dict get $cells $name] eq "HIER"}]
      } elseif {[regexp {^(NAME|REF_NAME|ORIG_REF_NAME)\s+(=~|!~|==|!=)\s+(.*)$} $term -> prop op rhs]} {
        set val [expr {$prop eq "NAME" ? $name : [dict get $cells $name]}]
        if {$op in {=~ !~}} { set v [fixture_match $rhs $val] } else { set v [expr {$val eq $rhs}] }
        if {$op in {!~ !=}} { set v [expr {!$v}] }
      } else { error "fixture: unsupported filter '$term'" }
      set yes [expr {$yes && $v}]
    }
    if {$yes} { return 1 }
  }
  return 0
}
proc get_cells {args} {
  global cells
  set hierarchical 0; set filter ""; set patterns {}
  for {set i 0} {$i < [llength $args]} {incr i} {
    set a [lindex $args $i]
    switch -- $a {
      -quiet {}
      -hierarchical { set hierarchical 1 }
      -filter { incr i; set filter [lindex $args $i] }
      default { set patterns $a }
    }
  }
  set out {}
  foreach n [dict keys $cells] {
    set hit [expr {$patterns eq ""}]
    foreach pat $patterns {
      if {$n eq $pat} { set hit 1; break }
      if {![fixture_match $pat $n]} continue
      if {!$hierarchical && [string match */* $pat] && [string range $pat end-1 end] eq "/*"} {
        set tail [string range $n [expr {[string length $pat]-1}] end]
        if {[string first / $tail] >= 0} continue
      }
      if {!$hierarchical && $pat eq "*" && [string first / $n] >= 0} continue
      set hit 1
    }
    if {$hit && [fixture_filter $n $filter]} { lappend out $n }
  }
  return $out
}
proc get_property {args} {
  global cells owners
  if {[lindex $args 0] eq "-quiet"} { set args [lrange $args 1 end] }
  lassign $args prop objects
  if {[llength $objects] == 0} { error "fixture: get_property $prop on an empty collection (Common 17-55)" }
  set out {}
  foreach obj $objects {
    switch -- $prop {
      NAME { set v $obj }
      REF_NAME - ORIG_REF_NAME { set v [dict get $cells $obj] }
      IS_PRIMITIVE { set v [expr {[dict get $cells $obj] ne "HIER"}] }
      IS_SEQUENTIAL { set v [string match FD* [dict get $cells $obj]] }
      PBLOCK {
        # Locality checks should use get_pblocks membership, including ownership
        # inherited from a hierarchy. Do not depend on the leaf's property view.
        if {$::fixture_case eq "membership"} { error "fixture: query pblock membership, not leaf PBLOCK" }
        set v [expr {[dict exists $owners $obj] ? [dict get $owners $obj] : "pblock_dynamic_region"}]
      }
      default { error "fixture: unsupported property '$prop'" }
    }
    lappend out $v
  }
  if {[llength $out] == 1} { return [lindex $out 0] }
  return $out
}
proc get_pins {args} {
  set i [lsearch -exact $args -of_objects]
  if {$i >= 0} {
    set cell [lindex $args [expr {$i+1}]]
    if {![dict exists $::cells $cell]} { error "fixture: unknown pin owner $cell" }
    return "$cell/O"
  }
  set pin [lindex $args end]
  if {[dict exists $::cells [file dirname $pin]]} { return $pin }
  return {}
}
proc all_fanin {args} {
  set todo [list [file dirname [lindex $args end]]]
  set seen [dict create]
  while {[llength $todo]} {
    set cell [lindex $todo 0]; set todo [lrange $todo 1 end]
    if {![dict exists $::fanin $cell]} continue
    foreach driver [dict get $::fanin $cell] {
      if {[dict exists $seen $driver]} continue
      dict set seen $driver 1
      if {![string match FD* [dict get $::cells $driver]]} { lappend todo $driver }
    }
  }
  return [dict keys $seen]
}
proc all_fanout {args} {
  set todo [list [file dirname [lindex $args end]]]
  set seen [dict create]; set result {}
  while {[llength $todo]} {
    set driver [lindex $todo 0]; set todo [lrange $todo 1 end]
    dict for {sink inputs} $::fanin {
      if {[lsearch -exact $inputs $driver] < 0 || [dict exists $seen $sink]} continue
      dict set seen $sink 1
      if {[string match FD* [dict get $::cells $sink]]} {
        lappend result "$sink/D"
      } else { lappend todo $sink }
    }
  }
  return $result
}
proc get_pblocks {args} {
  global pblocks owners
  set i [lsearch -exact $args -of_objects]
  if {$i >= 0} {
    set out {}
    foreach obj [lindex $args [expr {$i+1}]] {
      if {[dict exists $owners $obj]} { lappend out [dict get $owners $obj] }
    }
    return [lsort -unique $out]
  }
  set name [lindex $args end]
  return [expr {[dict exists $pblocks $name] ? $name : ""}]
}
proc create_pblock {name} { global pblocks; dict set pblocks $name 1; return $name }
proc resize_pblock {args} {}
proc set_property {args} {}
proc current_instance {args} {}
proc add_cells_to_pblock {pb objects} {
  global cells owners added assignments
  foreach obj $objects {
    # Vivado accepts a list of single-cell collections as well as a flat list.
    if {![dict exists $cells $obj] && [llength $obj] == 1} { set obj [lindex $obj 0] }
    if {![dict exists $cells $obj]} { error "fixture: unknown cell '$obj'" }
    dict for {prior owner} $assignments {
      if {$owner ne $pb && ([string first "$prior/" $obj] == 0 ||
                           [string first "$obj/" $prior] == 0)} {
        error "fixture: conflicting hierarchical pblocks for $prior and $obj"
      }
    }
    dict set assignments $obj $pb
    lappend added [list $pb $obj]
    foreach n [dict keys $cells] {
      if {$n eq $obj || [string first "$obj/" $n] == 0} { dict set owners $n $pb }
    }
  }
}
proc remove_cells_from_pblock {pb objects} {
  foreach obj $objects {
    foreach n [dict keys $::assignments] {
      if {($n eq $obj || [string first "$obj/" $n] == 0) &&
          [dict get $::assignments $n] eq $pb} { dict unset ::assignments $n }
    }
    foreach n [dict keys $::owners] {
      if {($n eq $obj || [string first "$obj/" $n] == 0) &&
          [dict get $::owners $n] eq $pb} { dict unset ::owners $n }
    }
  }
}
proc get_sites {args} {
  global placements
  set i [lsearch -exact $args -of_objects]
  set obj [lindex $args [expr {$i+1}]]
  if {[string match pblock_* $obj]} {
    if {[string match *URAM* [lindex $args end]]} { return {} }
    return {RAMB36_site}
  }
  return [expr {[dict exists $placements $obj] ? [dict get $placements $obj] : ""}]
}
proc get_slrs {args} { return [lindex $args end] }
proc fixture_assert_owner {obj slr} {
  set got [get_pblocks -quiet -of_objects $obj]
  if {[lsearch -exact $got "pblock_dynamic_SLR$slr"] < 0} { error "ownership: $obj expected SLR$slr, got $got" }
}
proc fixture_rpipe {root n} {
  for {set s 0} {$s < $n} {incr s} {
    set slice "$root/g_slices.g_s\[$s\].u_slice"
    fixture_cell "$slice/g_f0.main_q_reg\[0\]" FDRE
    fixture_cell "$slice/g_f0.main_v_q_reg" FDRE
  }
}
proc fixture_vpipe {root} {
  foreach s {0 1} {
    fixture_cell "$root/g_keep.d_q_reg\[$s\]\[0\]" FDRE
    fixture_cell "$root/g_keep.v_q_reg\[$s\]\[0\]" FDRE
  }
}

unset -nocomplain ::env(TSR_K4_FP_CORE) ::env(TSR_K4_FP_BODY) ::env(TSR_K4_FP_CTRL)
if {$fixture_mode eq "core"} {
  set ::env(TSR_K4_FP_CORE) ""
  set core ""
} else {
  set ::env(TSR_K4_FP_BODY) top/u_kernel
  set core top/u_kernel/u_core
}
set dp [expr {$core eq "" ? "u_dp" : "$core/u_dp"}]
set mvm "$dp/g_mvm.u_mvm"
set vu "$dp/g_vu.u_vu"
set ctrl [expr {$core eq "" ? "u_ctrl" : "$core/u_ctrl"}]
fixture_cell "$ctrl/u_rv/rv_q" FDRE
fixture_cell "$ctrl/u_cdc/cdc_q" FDRE
fixture_cell "$dp/u_spm/data_q" FDRE
fixture_cell "$mvm/dispatch_q" FDRE
set stall_q "$mvm/att_stall_q_reg"
set stall_a "$mvm/g_cl\[2\].u_cl/g_att.u_pv/att_stall_q_i_5_comp"
set stall_b "$mvm/g_cl\[1\].u_cl/g_att.u_pv/att_stall_q_i_1_comp"
if {$fixture_case eq "stall_reparented"} {
  set stall_a "$mvm/g_cl\[3\].u_cl/other_helper/att_stall_q_i_17_comp"
}
if {$fixture_case eq "stall_unknown_name"} {
  set stall_a "$mvm/g_cl\[2\].u_cl/unknown_aggregation_i_1"
}
if {$fixture_case eq "stall_outside_scope"} { set stall_a "unrelated/att_stall_q_i_5_comp" }
fixture_cell $stall_q FDRE
fixture_cell $stall_a LUT4
fixture_cell $stall_b LUT6
set stall_source "$mvm/g_cl\[2\].u_cl/g_att.at_qk_q_reg"
set shared_decode "$mvm/g_cl\[2\].u_cl/functional_decode_i_1"
set functional_sink "$mvm/g_cl\[2\].u_cl/g_att.qk_gate_q_reg"
fixture_cell $stall_source FDRE
fixture_cell $shared_decode LUT2
fixture_cell $functional_sink FDRE
dict set fanin $shared_decode [list $stall_source]
dict set fanin $functional_sink [list $shared_decode]
dict set fanin $stall_a [list $stall_source $shared_decode]
dict set fanin $stall_b [list $stall_a]
dict set fanin $stall_q [list $stall_b]
set stall_other {}
foreach {c i} {0 3 1 2 3 4} {
  set term "$mvm/g_cl\[$c\].u_cl/g_att.u_pv/att_stall_q_i_$i"
  fixture_cell $term LUT6
  dict set fanin $term [list "$mvm/g_cl\[$c\].u_cl/acc_q"]
  dict lappend fanin $stall_b $term
  lappend stall_other $term
}
fixture_cell "$mvm/u_gather/sel_q" FDRE
fixture_cell "$vu/u_wr/lb_q_reg\[0\]" FDRE
foreach {c s} {0 0 1 2 2 0 3 2} {
  set p "$mvm/g_cl\[$c\]"
  set g "$mvm/u_gather/g_asm\[$c\]"
  fixture_cell "$p.u_cl/acc_q" FDRE
  fixture_cell "$g.u_asm/mux_i_1" LUT3
  fixture_vpipe "$p.u_bcp"
  fixture_cell "$p.u_bcp/g_att.lo_q\[0\]\[17\]_i_1" LUT6
  fixture_cell "$p.u_bcp/g_att.lo_q_carry" CARRY8
  fixture_cell "$p.u_bcp/fmt_q_rep_i_1" LUT2
  fixture_vpipe "$p.u_wrp"
  fixture_rpipe "$p.u_wqp" 2
  fixture_rpipe "$p.u_rp" 2
  fixture_rpipe "$p.g_ajp.u_ajp" 2
  fixture_vpipe "$g.u_sp"
  fixture_rpipe "$g.g_jp.u_jp" 1
  fixture_cell "$g.g_jp.u_jl/g_f1.slot_q_reg\[0\]" FDRE
  fixture_cell "$g.g_aj.a_att_q_reg" FDRE
  fixture_cell "$g.g_aj.a_att_q_reg_rep__3_replica" FDRE
  fixture_cell "$g.p_d_i_1" LUT3
}
for {set l 0} {$l < 16} {incr l} { fixture_cell "$vu/u_lanes/g_lane\[$l\].u_lane/data_q" FDRE }
foreach kind {ip rp op} {
  foreach st {0 1} { fixture_cell "$vu/u_lanes/g_pipe.${kind}_q_reg\[$st\]\[0\]" FDRE }
}
fixture_cell "$vu/u_lanes/g_q\[0\]\[0\]_i_1" LUT4
fixture_cell "$vu/u_lanes/lb_q\[0\]\[0\]_i_2" LUT6
if {$fixture_mode eq "kernel"} {
  fixture_cell top/u_kernel/u_ctrl/control_q FDRE
  fixture_cell top/u_kernel/u_rst/reset_q FDRE
  for {set c 0} {$c < 4} {incr c} {
    set br "top/u_kernel/g_ch\[$c\].u_br"
    fixture_cell "$br/core_q" FDRE
    fixture_cell "$br/g_hs_q.hs_q" FDRE
    foreach ch {ar aw w r b} {
      foreach st {0 1} { fixture_cell "$br/u_$ch/g_st\[$st\].u_s/data_q" FDRE }
    }
  }
}

if {$fixture_case eq "missing_selector"} {
  foreach n [dict keys $cells] {
    if {[string first "$mvm/u_gather/g_asm\[1\].g_aj." $n] == 0} { dict unset cells $n }
  }
}
if {$fixture_case eq "unknown_lane_leaf"} { fixture_cell "$vu/u_lanes/unknown_helper_i_1" LUT2 }
if {$fixture_case eq "missing_att_pipe"} {
  foreach n [dict keys $cells] {
    if {[string first "$mvm/g_cl\[1\].g_ajp.u_ajp" $n] == 0} { dict unset cells $n }
  }
}
if {$fixture_case eq "stall_shared"} { dict lappend fanin $functional_sink $stall_a }
if {$fixture_case eq "stall_orphan"} {
  fixture_cell "$mvm/g_cl\[0\].u_cl/att_stall_q_i_99" LUT2
}
if {$fixture_case eq "stall_unknown"} { dict set cells $stall_a CARRY8 }
if {$fixture_case eq "stall_missing"} { dict unset cells $stall_q }
if {$fixture_case in {stall_existing_parent repeat}} {
  foreach {c s} {0 0 1 2 2 0 3 2} {
    add_cells_to_pblock pblock_dynamic_SLR$s [list "$mvm/g_cl\[$c\].u_cl"]
  }
}

source $fp_script
if {$fixture_case eq "repeat"} {
  set previous_owners $owners
  source $fp_script
  if {[dict size $owners] != [dict size $previous_owners]} { error "reapply changed ownership coverage" }
  dict for {cell owner} $previous_owners {
    if {![dict exists $owners $cell] || [dict get $owners $cell] ne $owner} {
      error "reapply changed ownership of $cell"
    }
  }
}

fixture_assert_owner $stall_a 1
fixture_assert_owner $stall_b 1
fixture_assert_owner $stall_q 1
foreach term $stall_other { fixture_assert_owner $term 1 }
fixture_assert_owner $stall_source 0
fixture_assert_owner $shared_decode 0
fixture_assert_owner $functional_sink 0

foreach {c s} {0 0 1 2 2 0 3 2} {
  set p "$mvm/g_cl\[$c\]"
  set g "$mvm/u_gather/g_asm\[$c\]"
  foreach st {0 1} { fixture_assert_owner "$p.g_ajp.u_ajp/g_slices.g_s\[$st\].u_slice/g_f0.main_q_reg\[0\]" $s }
  foreach n [list "$g.g_aj.a_att_q_reg" "$g.g_aj.a_att_q_reg_rep__3_replica" "$g.p_d_i_1" \
                 "$p.u_bcp/g_att.lo_q\[0\]\[17\]_i_1" "$p.u_bcp/g_att.lo_q_carry" "$p.u_bcp/fmt_q_rep_i_1"] {
    fixture_assert_owner $n $s
  }
  fixture_assert_owner "$p.u_bcp/g_keep.d_q_reg\[0\]\[0\]" 1
  fixture_assert_owner "$p.u_bcp/g_keep.d_q_reg\[1\]\[0\]" $s
  fixture_assert_owner "$g.u_sp/g_keep.d_q_reg\[0\]\[0\]" 1
  fixture_assert_owner "$g.u_sp/g_keep.d_q_reg\[1\]\[0\]" $s
  fixture_assert_owner "$g.g_jp.u_jp/g_slices.g_s\[0\].u_slice/g_f0.main_q_reg\[0\]" $s
  fixture_assert_owner "$g.g_jp.u_jl/g_f1.slot_q_reg\[0\]" 1
  fixture_assert_owner "$p.u_cl/acc_q" $s
  foreach row $added {
    if {[lindex $row 1] eq "$p.u_bcp"} { error "split broadcast parent was pinned" }
  }
}
fixture_assert_owner "$vu/u_lanes/g_q\[0\]\[0\]_i_1" 1
fixture_assert_owner "$vu/u_lanes/lb_q\[0\]\[0\]_i_2" 1
fixture_assert_owner "$vu/u_lanes/g_lane\[0\].u_lane/data_q" 0
foreach row $added { if {[lindex $row 1] eq "$vu/u_lanes"} { error "split lane parent was pinned" } }
foreach {kind first fp_last_slr} {ip 1 0 rp 1 0 op 0 1} {
  fixture_assert_owner "$vu/u_lanes/g_pipe.${kind}_q_reg\[0\]\[0\]" $first
  fixture_assert_owner "$vu/u_lanes/g_pipe.${kind}_q_reg\[1\]\[0\]" $fp_last_slr
}
if {$fixture_case eq "late_selector"} {
  fixture_cell "$mvm/u_gather/g_asm\[1\].g_aj.a_att_q_reg_replica_99" FDRE
  k4fp_check_locality
}
set checked_adds [llength $added]
if {[string match post_* $fixture_case]} {
  dict for {n pb} $owners {
    if {[regexp {^pblock_dynamic_SLR([012])$} $pb -> s]} { dict set placements $n SLR$s }
  }
  if {$fixture_case eq "post_late_selector"} {
    fixture_cell "$mvm/u_gather/g_asm\[1\].g_aj.a_att_q_reg_replica_99" FDRE
  }
  if {$fixture_case eq "post_misplaced"} {
    dict set placements "$vu/u_lanes/g_pipe.op_q_reg\[1\]\[0\]" SLR0
  }
  if {$fixture_case eq "post_stall_misplaced"} { dict set placements $stall_a SLR0 }
  if {$fixture_case eq "post_stall_source_misplaced"} { dict set placements $stall_source SLR1 }
  if {$fixture_case eq "post_stall_source_replica"} {
    set replica "${stall_source}_replica_99"
    fixture_cell $replica FDRE
    dict set fanin $stall_a [list $replica $shared_decode]
  }
  if {$fixture_case eq "post_stall_replica"} {
    set replica "${stall_b}_replica_99"
    fixture_cell $replica LUT1
    dict set fanin $replica [list $stall_b]
    dict set fanin $stall_q [list $replica]
  }
  if {$fixture_case eq "post_missing_proc"} { rename k4fp_check_locality {} }
  # Other tool hooks use generic globals. The post-hook must retain the scope
  # captured when the floorplan was applied rather than trust these values.
  set nch 0
  set mvm unrelated_mvm
  set vu unrelated_vu
  proc report_timing_summary {args} {}
  proc report_design_analysis {args} {}
  source [file join [file dirname $fp_script] k4_locality_check.tcl]
} elseif {$fixture_case in {placed misplaced}} {
  dict for {n pb} $owners {
    if {[regexp {^pblock_dynamic_SLR([012])$} $pb -> s]} { dict set placements $n SLR$s }
  }
  if {$fixture_case eq "misplaced"} {
    dict set placements "$vu/u_lanes/g_pipe.op_q_reg\[1\]\[0\]" SLR0
  }
  k4fp_check_locality 1
} else {
  k4fp_check_locality
}
if {[llength $added] != $checked_adds} { error "locality check changed the pblock assignments" }
puts "FIXTURE PASS $fixture_mode $fixture_case"

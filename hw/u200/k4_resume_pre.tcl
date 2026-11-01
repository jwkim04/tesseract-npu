# PHYS_OPT_DESIGN.PRE for a preserved placed K4 checkpoint. Rebind the CDC
# constraints, rebuild floorplan ownership, and move only the PERF reduction
# LUTs. No cells are recreated, and no fixed placement is relaxed.
proc k4fp_resume_clock {endpoint} {
  set pins [get_pins -quiet "[get_property NAME $endpoint]/C"]
  set clocks [get_clocks -quiet -of_objects $pins]
  if {[llength $pins] != 1 || [llength $clocks] != 1} {
    error "K4ECO missing ATT stall endpoint clock"
  }
  return [list [get_property NAME $clocks] [get_property PERIOD $clocks]]
}

proc k4fp_resume_bel_free {bel} {
  if {[llength $bel] != 1 || [get_property CLASS $bel] ne "bel"} {
    error "K4ECO expected one BEL for occupancy check: $bel"
  }
  set properties [list_property $bel]
  set blocked 0
  foreach property {IS_USED IS_RESERVED PROHIBIT} {
    if {[lsearch -exact $properties $property] < 0} {
      error "K4ECO unsupported BEL occupancy property $property on $bel"
    }
    set value [get_property $property $bel]
    if {![string is boolean -strict $value]} {
      error "K4ECO invalid $property '$value' on $bel"
    }
    if {$value} { set blocked 1 }
  }
  # IS_USED includes physical route-throughs that have no logical cell.
  return [expr {!$blocked && [llength [get_cells -quiet -of_objects $bel]] == 0}]
}

proc k4fp_resume_bel {endpoint} {
  set sink [get_sites -quiet -of_objects $endpoint]
  if {[llength $sink] != 1 ||
      ![regexp {^SLICE_X([0-9]+)Y([0-9]+)$} [get_property NAME $sink] -> x y]} {
    error "K4ECO unexpected ATT stall endpoint site $sink"
  }
  set choices {}
  foreach site [get_sites -quiet -of_objects [get_pblocks pblock_dynamic_SLR1] \
      -filter {SITE_TYPE =~ SLICE*}] {
    set name [get_property NAME $site]
    if {![regexp {^SLICE_X([0-9]+)Y([0-9]+)$} $name -> sx sy]} {
      error "K4ECO unknown SLR1 slice $name"
    }
    set distance [expr {abs($sx - $x) + abs($sy - $y)}]
    if {$distance <= 16} { lappend choices [list $distance $name $site] }
  }
  # Search only nearby legal-region slices; never consume another cell's BEL.
  foreach row [lsort -integer -index 0 [lsort -dictionary -index 1 $choices]] {
    set site [lindex $row 2]
    if {[get_property NAME [get_slrs -quiet -of_objects $site]] ne "SLR1"} {
      error "K4ECO central pblock contains a slice outside SLR1: $site"
    }
    foreach bel [get_bels -quiet -of_objects $site -filter {NAME =~ */*6LUT}] {
      set pair [get_bels -quiet [string map {6LUT 5LUT} [get_property NAME $bel]]]
      if {[llength $pair] != 1} { error "K4ECO missing paired 5LUT for $bel" }
      if {[k4fp_resume_bel_free $bel] && [k4fp_resume_bel_free $pair]} {
        return $bel
      }
    }
  }
  error "K4ECO no unoccupied LUT pair within 16 slice coordinates of $sink"
}

proc k4fp_resume_pre {dir} {
  current_instance -quiet
  set endpoint [get_cells -quiet -hierarchical -filter {
    IS_PRIMITIVE && NAME =~ *g_mvm.u_mvm/att_stall_q_reg
  }]
  if {[llength $endpoint] != 1} { error "K4ECO expected one ATT stall endpoint" }
  set clock_before [k4fp_resume_clock $endpoint]
  # Batched snapshots verify every other primitive, including all source FFs
  # and functional consumers. Keep both site and BEL: unchanged LOC alone can
  # hide movement within the same slice.
  set cells [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE}]
  set names [get_property NAME $cells]
  set loc_before [get_property LOC $cells]
  set bel_before [get_property BEL $cells]
  if {[llength $loc_before] != [llength $cells] || [llength $bel_before] != [llength $cells]} {
    error "K4ECO incomplete placement snapshot"
  }
  uplevel #0 [list source [file join $dir clock_net_check.tcl]]
  uplevel #0 [list source [file join $dir cdc_restore.tcl]]
  ::tsr_cdc_link::restore pre_physopt [pwd]
  uplevel #0 [list source [file join $dir k4_floorplan.tcl]]
  set cone [k4fp_att_stall_cone [dict get $::k4fp_scope mvm]]
  set moved [dict create]
  set targets [dict create]
  set report_path [file join [pwd] k4_resume_eco.tsv]
  set new_report [expr {![file exists $report_path] || [file size $report_path] == 0}]
  set report [open $report_path a]
  set stamp [clock milliseconds]
  if {$new_report} {
    puts $report "run_ms\tcell\told_ref\told_init\told_loc\told_bel\tnew_ref\tnew_init\tnew_loc\tnew_bel"
  }
  foreach cell $cone {
    if {$cell eq $endpoint} continue
    set name [get_property NAME $cell]
    dict set targets $name 1
    set ref [get_property REF_NAME $cell]
    set init [get_property INIT $cell]
    set old_loc [get_property LOC $cell]
    set old_bel [get_property BEL $cell]
    set sites [get_sites -quiet -of_objects $cell]
    if {[llength $sites] != 1} { error "K4ECO expected a placed LUT: $name" }
    set slr [get_slrs -quiet -of_objects $sites]
    if {[get_property NAME $slr] ne "SLR1"} {
      if {[get_property IS_LOC_FIXED $cell] || [get_property IS_BEL_FIXED $cell]} {
        error "K4ECO refuses to move fixed cell $name LOC=$old_loc BEL=$old_bel"
      }
      set destination [k4fp_resume_bel $endpoint]
      place_cell [list $cell [get_property NAME $destination]]
      dict set moved $name 1
    }
    if {[get_property REF_NAME $cell] ne $ref || [get_property INIT $cell] ne $init} {
      error "K4ECO changed REF/INIT of $name"
    }
    puts $report [join [list $stamp $name $ref $init $old_loc $old_bel \
        [get_property REF_NAME $cell] [get_property INIT $cell] \
        [get_property LOC $cell] [get_property BEL $cell]] "\t"]
  }
  close $report
  set loc_after [get_property LOC $cells]
  set bel_after [get_property BEL $cells]
  if {[llength [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE}]] != [llength $cells]} {
    error "K4ECO changed the primitive cell count"
  }
  if {[llength $loc_after] != [llength $cells] || [llength $bel_after] != [llength $cells]} {
    error "K4ECO incomplete placement verification"
  }
  foreach name $names before_l $loc_before before_b $bel_before after_l $loc_after after_b $bel_after {
    if {![dict exists $targets $name] &&
        ($before_l ne $after_l || $before_b ne $after_b)} {
      error "K4ECO moved a non-target cell $name: $before_l/$before_b -> $after_l/$after_b"
    }
  }
  if {[k4fp_resume_clock $endpoint] ne $clock_before} { error "K4ECO changed the endpoint clock" }
  k4fp_check_locality 1
  puts "K4ECO PASS run=$stamp moved=[dict size $moved] targets=[dict size $targets] preserved=[expr {[llength $cells] - [dict size $targets]}] endpoint_clock=$clock_before"
}
k4fp_resume_pre [file dirname [file normalize [info script]]]

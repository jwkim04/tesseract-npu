# tb/platform/viv/k4_pblock_check.tcl — legality check of the K4 control Pblock against the U200 platform.
#
#   vivado -mode batch -source tb/platform/viv/k4_pblock_check.tcl -tclargs <routed.dcp> <outdir> [<repo>]
#
# Opens a routed (or placed) checkpoint of a K4 link, which carries the platform's own pblocks and the kernel
# hierarchy, and sources hw/u200/k4_major_blocks.tcl on it as the link does before place_design. Nothing is
# written back to the checkpoint. The hook creates the control Pblock as a child of pblock_dynamic_SLR0 and
# assigns every major block; any failure of the hook stops this script with an error.
#
# Checks and prints (lines start with K4CHK):
#   - the control Pblock is a child of pblock_dynamic_SLR0 and every site of it is also a site of the parent;
#   - every clock-region range of the Pblock names clock regions that exist and holds SLICE, RAMB36 and DSP sites;
#   - per clock region of SLR0 and SLR2: the sites the platform's pblocks allow (SLICE, RAMB36, DSP48E2, URAM288,
#     LAGUNA) and the BRAM / DSP / URAM sites the checkpoint uses;
#   - per pblock: CLB (SLICE), RAMB36, RAMB18, DSP48E2, URAM288 capacity;
#   - the RAMB / DSP / URAM primitives of each control-plane block;
#   - DRC checks of the PBLK category, when the install has them.
# Writes <outdir>/pblock_properties.txt (the format hw/viz/k4_pblock_preview.py reads, with a CAPACITY line per
# pblock and the Pblock's DERIVED_RANGES as Vivado resolved them).
set dcp    [lindex $argv 0]
set outdir [lindex $argv 1]
set repo   [expr {[llength $argv] > 2 ? [lindex $argv 2] : [file normalize [file join [file dirname [info script]] ../../..]]}]
file mkdir $outdir
set_param general.maxThreads 8
set t0 [clock seconds]
proc k4chk_lap {what} { global t0; puts "K4CHK time $what [expr {[clock seconds] - $t0}] s" }

open_checkpoint $dcp
k4chk_lap "open_checkpoint"

# ---- the site types of the capacity tables ----
set types {SLICE RAMB36 RAMB18 DSP48E2 URAM288 LAGUNA}
array set filter {
  SLICE   {NAME =~ SLICE_*}
  RAMB36  {NAME =~ RAMB36_*}
  RAMB18  {NAME =~ RAMB18_*}
  DSP48E2 {NAME =~ DSP48E2_*}
  URAM288 {NAME =~ URAM288_*}
  LAGUNA  {NAME =~ LAGUNA_*}
}
# Sites are selected by name: an occupied RAMB36 site of a placed checkpoint no longer reports SITE_TYPE RAMBFIFO36.
proc k4chk_sites {pb type} {
  global filter
  return [get_sites -quiet -of_objects $pb -filter $filter($type)]
}
proc k4chk_by_cr {sites} {
  set d [dict create]
  if {[llength $sites] == 0} { return $d }
  foreach c [get_property CLOCK_REGION $sites] { dict incr d $c }
  return $d
}

# ---- before the hook: the platform's pblocks ----
foreach n {pblock_dynamic_SLR0 pblock_dynamic_SLR1 pblock_dynamic_SLR2} {
  if {[llength [get_pblocks -quiet $n]] != 1} { error "K4CHK platform pblock $n is missing in $dcp" }
  puts "K4CHK platform $n PARENT [get_property PARENT [get_pblocks $n]] SNAPPING_MODE [get_property SNAPPING_MODE [get_pblocks $n]]"
}
set platform_cr [dict create]
foreach n {pblock_dynamic_SLR0 pblock_dynamic_SLR2} {
  foreach type $types {
    dict set platform_cr $n $type [k4chk_by_cr [k4chk_sites [get_pblocks $n] $type]]
  }
}
k4chk_lap "platform sites"

# the parent's site sets, for the containment check
set parent_sites [dict create]
foreach type {SLICE RAMB36 DSP48E2} {
  set d [dict create]
  foreach s [k4chk_sites [get_pblocks pblock_dynamic_SLR0] $type] { dict set d [get_property NAME $s] 1 }
  dict set parent_sites $type $d
}

# ---- sites the checkpoint uses, by clock region ----
foreach type {RAMB36 DSP48E2 URAM288} {
  set used($type) [k4chk_by_cr [get_sites -quiet -filter "$filter($type) && IS_USED"]]
}
k4chk_lap "used sites"

# ---- primitives of the control-plane blocks (before they are moved) ----
set kernel [get_cells -quiet -hierarchical -filter {REF_NAME == tsr_u200_k4 || ORIG_REF_NAME == tsr_u200_k4}]
if {[llength $kernel] != 1} { error "K4CHK expected one tsr_u200_k4 instance" }
set core "[get_property NAME $kernel]/u_kernel/u_core"
set rams [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE && (REF_NAME =~ RAMB* || REF_NAME =~ URAM* || REF_NAME =~ DSP48*)}]
set names [get_property NAME $rams]
set refs  [get_property REF_NAME $rams]
foreach {label rel} {
  u_ctrl u_ctrl  dbg_rv g_db.u_dbg_rv  issue u_dp/u_issue  csr u_dp/u_csr  dbg u_dp/g_db.u_dbg
  rv_links g_rv_local.*
} {
  set count [dict create]
  foreach n $names r $refs {
    if {[string match "$core/$rel*" $n]} { dict incr count $r }
  }
  if {[dict size $count] == 0} { set count none }
  puts "K4CHK primitives $label: $count"
}
k4chk_lap "primitives"

# ---- the hook, as the link runs it ----
set ::env(TSR_K4_CTRL) slr0
source [file join $repo hw/u200/k4_major_blocks.tcl]
k4chk_lap "hook"

set ctrl [get_pblocks -quiet pblock_tsr_ctrl_plane]
if {[llength $ctrl] != 1} { error "K4CHK the hook did not create pblock_tsr_ctrl_plane" }
puts "K4CHK pblock_tsr_ctrl_plane PARENT [get_property PARENT $ctrl] SNAPPING_MODE [get_property SNAPPING_MODE $ctrl]"
puts "K4CHK pblock_tsr_ctrl_plane GRID_RANGES [get_property GRID_RANGES $ctrl]"
if {[get_property PARENT $ctrl] ne "pblock_dynamic_SLR0"} { error "K4CHK control Pblock parent is not pblock_dynamic_SLR0" }

# ---- capacity of every pblock ----
puts "K4CHK capacity pblock SLICE(CLB) RAMB36 RAMB18 DSP48E2 URAM288 LAGUNA"
foreach pb [get_pblocks] {
  set row {}
  foreach type $types { lappend row [llength [k4chk_sites $pb $type]] }
  set cap([get_property NAME $pb]) $row
  puts "K4CHK capacity [get_property NAME $pb] [join $row { }]"
}

# ---- every site of the child is a site of the parent ----
foreach type {SLICE RAMB36 DSP48E2} {
  set outside 0
  foreach s [k4chk_sites $ctrl $type] {
    if {![dict exists [dict get $parent_sites $type] [get_property NAME $s]]} { incr outside }
  }
  puts "K4CHK containment $type sites of the control Pblock outside pblock_dynamic_SLR0: $outside"
  if {$outside} { error "K4CHK the control Pblock holds $outside $type sites outside its parent" }
}

# ---- every range resolves, clock region by clock region ----
set ctrl_cr [dict create]
foreach type $types { dict set ctrl_cr $type [k4chk_by_cr [k4chk_sites $ctrl $type]] }
foreach range [get_property GRID_RANGES $ctrl] {
  if {![regexp {^CLOCKREGION_(X[0-9]+Y[0-9]+):CLOCKREGION_(X[0-9]+Y[0-9]+)$} $range -> a b]} {
    error "K4CHK range $range is not a clock-region range"
  }
  regexp {X([0-9]+)Y([0-9]+)} $a -> x0 y0
  regexp {X([0-9]+)Y([0-9]+)} $b -> x1 y1
  set slices 0
  for {set x $x0} {$x <= $x1} {incr x} {
    for {set y $y0} {$y <= $y1} {incr y} {
      set cr X${x}Y$y
      if {[llength [get_clock_regions -quiet $cr]] != 1} { error "K4CHK clock region $cr of $range does not exist" }
      set row {}
      foreach type $types {
        set d [dict get $ctrl_cr $type]
        lappend row [expr {[dict exists $d $cr] ? [dict get $d $cr] : 0}]
      }
      if {[lindex $row 0] == 0 || [lindex $row 1] == 0} { error "K4CHK clock region $cr of $range has no SLICE or no RAMB36 site in the Pblock" }
      puts "K4CHK range $range $cr SLICE [lindex $row 0] RAMB36 [lindex $row 1] RAMB18 [lindex $row 2] DSP48E2 [lindex $row 3] URAM288 [lindex $row 4] LAGUNA [lindex $row 5]"
      incr slices [lindex $row 0]
    }
  }
  puts "K4CHK range $range resolves to $slices SLICE sites"
}

# ---- per clock region of SLR0 and SLR2: platform capacity and sites used by the routed first link ----
puts "K4CHK crtable cr  slice_allowed ramb36_allowed ramb36_used dsp_allowed dsp_used uram_allowed uram_used laguna_allowed"
foreach n {pblock_dynamic_SLR0 pblock_dynamic_SLR2} {
  set rows {}
  foreach y [expr {$n eq "pblock_dynamic_SLR0" ? {0 1 2 3 4} : {10 11 12 13 14}}] {
    for {set x 0} {$x < 6} {incr x} {
      set cr X${x}Y$y
      set line [list $cr]
      foreach {type used_key} {SLICE {} RAMB36 RAMB36 DSP48E2 DSP48E2 URAM288 URAM288 LAGUNA {}} {
        set d [dict get $platform_cr $n $type]
        lappend line [expr {[dict exists $d $cr] ? [dict get $d $cr] : 0}]
        if {$used_key ne "" && $type ne "LAGUNA"} {
          lappend line [expr {[dict exists $used($used_key) $cr] ? [dict get $used($used_key) $cr] : 0}]
        }
      }
      puts "K4CHK crtable [join $line { }]"
    }
  }
}

# ---- DRC, when the install has Pblock checks ----
set checks [get_drc_checks -quiet PBLK-*]
if {[llength $checks]} {
  catch {report_drc -checks $checks -file [file join $outdir drc_pblk.rpt]} err
  puts "K4CHK drc PBLK checks: [llength $checks] ($err)"
} else {
  puts "K4CHK drc no PBLK checks in this install"
}

# ---- the properties file for the preview ----
set f [open [file join $outdir pblock_properties.txt] w]
foreach pb [get_pblocks] {
  set name [get_property NAME $pb]
  puts $f "PBLOCK $name"
  foreach p {IS_SOFT EXCLUDE_PLACEMENT CONTAIN_ROUTING PARENT GRID_RANGES DERIVED_RANGES SNAPPING_MODE} {
    puts $f "$p [get_property $p $pb]"
  }
  puts $f "CAPACITY $cap($name)"
}
close $f
puts "K4CHK wrote [file join $outdir pblock_properties.txt]"
k4chk_lap "done"
puts "K4CHK PASS"
exit 0

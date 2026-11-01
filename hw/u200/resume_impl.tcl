# Resume the saved U200 project from its placed checkpoint through routing.
# Usage: vivado -mode batch -source resume_impl.tcl -tclargs \
#   <saved-project.xpr> <phys-pre-hook.tcl> <route-pre-hook.tcl>
# Hooks run in the implementation child, not in this project-control session.
# The phys-pre hook must restore CDC constraints and the K4FP Tcl scope before
# physical optimisation. No synthesis, bitstream or packaging step is launched.

proc tsr_resume_file {path what} {
  set path [file normalize $path]
  if {![file isfile $path] || ![file readable $path]} {
    error "RESUME $what is not a readable file: $path"
  }
  return $path
}

# A Vitis shell-linked GateLvl fileset can have no TOP. The RM synthesis
# fileset names the kernel, not the linked shell. Use this implementation
# run's typed output records instead of guessing either module name.
proc tsr_resume_checkpoints {run_dir run_name part} {
  set metadata [tsr_resume_file [file join $run_dir gen_run.xml] run-metadata]
  set f [open $metadata r]
  set xml [read $f]
  close $f
  if {![regexp {<GenRun[[:space:]][^>]*>} $xml header] ||
      ![regexp {[[:space:]]Id="([^"]+)"} $header -> id] ||
      ![regexp {[[:space:]]LaunchPart="([^"]+)"} $header -> launch_part] ||
      $id ne $run_name || $launch_part ne $part} {
    error "RESUME run metadata does not match $run_name / $part: $metadata"
  }
  set outputs [dict create]
  foreach tag [regexp -all -inline {<File[[:space:]][^>]*>} $xml] {
    if {![regexp {[[:space:]]Type="([^"]+)"} $tag -> type] ||
        $type ni {PLACE-DCP ROUTE-DCP}} { continue }
    if {[dict exists $outputs $type] ||
        ![regexp {[[:space:]]Name="([^"]+)"} $tag -> name] ||
        [file pathtype $name] ne "relative" || [file tail $name] ne $name} {
      error "RESUME invalid or duplicate $type record in $metadata: $tag"
    }
    dict set outputs $type $name
  }
  if {![dict exists $outputs PLACE-DCP] || ![dict exists $outputs ROUTE-DCP]} {
    error "RESUME missing placed/routed output records in $metadata"
  }
  set placed [dict get $outputs PLACE-DCP]
  set routed [dict get $outputs ROUTE-DCP]
  if {![regexp {^([[:alnum:]_.-]+)_placed[.]dcp$} $placed -> stem] ||
      $routed ne "${stem}_routed.dcp"} {
    error "RESUME inconsistent checkpoint names in $metadata: $placed / $routed"
  }
  puts "RESUME metadata=$metadata run=$id part=$launch_part placed=$placed routed=$routed"
  return [list [file join $run_dir $placed] [file join $run_dir $routed]]
}

proc tsr_resume_impl {} {
  global argv
  if {[llength $argv] != 3} {
    error "usage: resume_impl.tcl <saved-project.xpr> <phys-pre-hook.tcl> <route-pre-hook.tcl>"
  }
  set project [tsr_resume_file [lindex $argv 0] project]
  set phys_pre [tsr_resume_file [lindex $argv 1] PHYS_OPT.PRE]
  set route_pre [tsr_resume_file [lindex $argv 2] ROUTE.PRE]
  set phys_post [tsr_resume_file \
      [file join [file dirname [file normalize [info script]]] k4_locality_check.tcl] \
      PHYS_OPT.POST]
  if {[file extension $project] ne ".xpr"} {
    error "RESUME expected a saved .xpr project: $project"
  }
  if {[version -short] ne "2024.2"} {
    error "RESUME requires Vivado 2024.2, got [version -short]"
  }
  set_param general.maxThreads 8
  puts "RESUME open_project $project"
  open_project $project
  set part [get_property PART [current_project]]
  if {$part ne "xcu200-fsgd2104-2-e"} {
    error "RESUME expected xcu200-fsgd2104-2-e, got $part"
  }
  set run [get_runs -quiet impl_1]
  if {[llength $run] != 1 || ![get_property IS_IMPLEMENTATION $run]} {
    error "RESUME expected one implementation run named impl_1"
  }
  if {![get_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED $run]} {
    error "RESUME impl_1 has PHYS_OPT_DESIGN disabled"
  }
  set run_dir [file normalize [get_property DIRECTORY $run]]
  lassign [tsr_resume_checkpoints $run_dir [get_property NAME $run] $part] placed routed
  set placed [tsr_resume_file $placed placed-checkpoint]
  if {[file size $placed] == 0} { error "RESUME empty placed checkpoint: $placed" }
  if {![file isfile [file join $run_dir .place_design.end.rst]]} {
    error "RESUME missing completed place_design marker in $run_dir"
  }
  set route_post [get_property STEPS.ROUTE_DESIGN.TCL.POST $run]
  if {$route_post eq ""} {
    error "RESUME missing existing native ROUTE.POST hook"
  }
  puts "RESUME part=$part run=$run status=[get_property STATUS $run] progress=[get_property PROGRESS $run]"
  puts "RESUME placed=$placed bytes=[file size $placed]"
  puts "RESUME preserve STEPS.ROUTE_DESIGN.TCL.POST {$route_post}"

  # Configure only the three hooks needed by this restart. Leave the generated
  # native ROUTE.POST wrapper, its Vitis metadata work and CDC gate untouched.
  set hooks [list \
      STEPS.PHYS_OPT_DESIGN.TCL.PRE $phys_pre \
      STEPS.PHYS_OPT_DESIGN.TCL.POST $phys_post \
      STEPS.ROUTE_DESIGN.TCL.PRE $route_pre]
  foreach {property path} $hooks {
    puts "RESUME hook $property old={[get_property $property $run]} new={$path}"
  }
  foreach {property path} $hooks {
    set_property $property $path $run
    if {[get_property $property $run] ne $path} {
      error "RESUME failed to set $property to $path"
    }
  }
  if {[get_property STEPS.ROUTE_DESIGN.TCL.POST $run] ne $route_post} {
    error "RESUME native ROUTE.POST changed while setting resume hooks"
  }
  puts "RESUME reset_runs impl_1 -from_step phys_opt_design"
  reset_runs impl_1 -from_step phys_opt_design
  if {![file isfile $placed]} {
    error "RESUME placed checkpoint was removed by the step reset: $placed"
  }
  if {[file exists $routed]} {
    error "RESUME step reset left a stale routed checkpoint: $routed"
  }
  if {[get_property STEPS.ROUTE_DESIGN.TCL.POST $run] ne $route_post} {
    error "RESUME native ROUTE.POST changed during the step reset"
  }
  set started [clock seconds]
  puts "RESUME launch_runs impl_1 -to_step route_design -jobs 8"
  launch_runs impl_1 -to_step route_design -jobs 8
  wait_on_run impl_1
  set status [get_property STATUS $run]
  set progress [get_property PROGRESS $run]
  puts "RESUME finished status={$status} progress={$progress}"
  if {$status ne "route_design Complete!" || $progress ne "100%"} {
    error "RESUME implementation did not complete route_design"
  }
  if {![file isfile $routed] || [file size $routed] == 0 ||
      [file mtime $routed] < $started} {
    error "RESUME missing new nonempty routed checkpoint: $routed"
  }
  if {![file isfile [file join $run_dir .route_design.end.rst]] ||
      [file exists [file join $run_dir .route_design.error.rst]]} {
    error "RESUME route_design completion markers do not indicate success"
  }
  puts "RESUME PASS routed=$routed bytes=[file size $routed]; stopped before bitstream"
  close_project
}

if {[catch {tsr_resume_impl} tsr_resume_error tsr_resume_options]} {
  puts stderr "RESUME FAIL: $tsr_resume_error"
  if {[dict exists $tsr_resume_options -errorinfo]} {
    puts stderr [dict get $tsr_resume_options -errorinfo]
  }
  if {[llength [get_projects -quiet]] != 0} { catch {close_project} }
  exit 1
}
exit 0

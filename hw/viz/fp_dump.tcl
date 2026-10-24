# hw/viz/fp_dump.tcl — dump the placement of a routed checkpoint for hw/viz/fp_render.py (no GUI).
#
#   vivado -mode batch -nojournal -log OUT/vivado.log -source hw/viz/fp_dump.tcl \
#     -tclargs <checkpoint.dcp> <outdir>
#
# Writes into <outdir>:
#   cells.tsv    one line per placed primitive: hierarchical name, site (LOC), REF_NAME
#   sites.tsv    one line per site of the part: name, SITE_TYPE, RPM_X, RPM_Y, CLOCK_REGION
#   pblocks.tsv  pblock name and GRID_RANGES
# Every property is read with one get_property call on the whole collection, so a full VU9P
# design (about 600 K primitives, 240 K sites) takes 2-4 min and 6-8 GB, most of it opening the
# checkpoint.
set dcp [lindex $argv 0]; set out [lindex $argv 1]
set t0 [clock seconds]
proc fpd {msg} { global t0; puts "FPD [expr {[clock seconds]-$t0}]s $msg"; flush stdout }
file mkdir $out
open_checkpoint $dcp
fpd "opened $dcp"
set cells [get_cells -hier -quiet -filter {IS_PRIMITIVE && LOC != ""}]
fpd "[llength $cells] placed primitives"
set names [get_property NAME $cells]
set locs  [get_property LOC $cells]
set refs  [get_property REF_NAME $cells]
set fh [open $out/cells.tsv w]
foreach n $names l $locs r $refs { puts $fh "$n\t$l\t$r" }
close $fh
fpd "cells.tsv written"
set sites [get_sites]
set sn [get_property NAME $sites]
set st [get_property SITE_TYPE $sites]
set rx [get_property RPM_X $sites]
set ry [get_property RPM_Y $sites]
set cr [get_property CLOCK_REGION $sites]
set fh [open $out/sites.tsv w]
foreach a $sn b $st c $rx d $ry e $cr { puts $fh "$a\t$b\t$c\t$d\t$e" }
close $fh
fpd "[llength $sites] sites written"
set fh [open $out/pblocks.tsv w]
foreach pb [get_pblocks -quiet] { puts $fh "$pb\t[get_property -quiet GRID_RANGES $pb]" }
close $fh
fpd "done"

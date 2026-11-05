# tb/core/viv/slr_clb.tcl — dump the placed primitives of a checkpoint for tb/core/viv/slr_clb.py (read-only).
#
# usage: vivado -mode batch -nojournal -nolog -source tb/core/viv/slr_clb.tcl -tclargs <dcp> <outdir>
# Writes into <outdir>: cells.tsv (hierarchical name, LOC, REF_NAME of every placed primitive on a SLICE site) and
# util_slr_dcp.rpt (report_utilization -slr). slr_clb.py counts CLBs and LUTs per SLR and hierarchy from them: the
# counting is done outside Vivado because a per-cell Tcl loop over a full device (1.5 M primitives) takes hours.
set dcp [lindex $argv 0]
set out [lindex $argv 1]
file mkdir $out
open_checkpoint $dcp
set cells [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE && LOC != "" && LOC =~ SLICE_*}]
puts "SLRCLB [llength $cells] placed primitives on SLICE sites"
set names [get_property NAME $cells]
set locs  [get_property LOC $cells]
set refs  [get_property REF_NAME $cells]
set fh [open $out/cells.tsv w]
foreach n $names l $locs r $refs { puts $fh "$n\t$l\t$r" }
close $fh
puts "SLRCLB cells.tsv written"
report_utilization -slr -file $out/util_slr_dcp.rpt
puts "SLRCLB done"
exit

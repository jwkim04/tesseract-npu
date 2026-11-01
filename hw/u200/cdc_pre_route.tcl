# Read-only gate immediately before route_design. Restoration belongs to
# pre-implementation hooks; a later lost exception must fail here.
set tsr_cdc_pre_route_dir [file dirname [file normalize [info script]]]
source $tsr_cdc_pre_route_dir/clock_net_check.tcl
source $tsr_cdc_pre_route_dir/cdc_restore.tcl
tsr_cdc_link::check pre_route [pwd]

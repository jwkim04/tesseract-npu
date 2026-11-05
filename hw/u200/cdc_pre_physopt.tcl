# Rebind the linked CDC constraints before physical optimisation. Independent
# of placement policy and also active when TSR_U200_FLOORPLAN=0.
set tsr_cdc_physopt_dir [file dirname [file normalize [info script]]]
source $tsr_cdc_physopt_dir/clock_net_check.tcl
source $tsr_cdc_physopt_dir/cdc_restore.tcl
tsr_cdc_link::restore pre_physopt [pwd]

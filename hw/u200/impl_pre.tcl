# Inspect clock/reset connectivity before implementation, then preserve the T1
# exclusion of BRAM power optimisation on the control core's TCMs.
set tsr_impl_hook_dir [file dirname [file normalize [info script]]]
source $tsr_impl_hook_dir/clock_net_check.tcl
source $tsr_impl_hook_dir/cdc_restore.tcl
tsr_cdc_link::restore pre_impl [pwd]
source $tsr_impl_hook_dir/tcm_opt.tcl

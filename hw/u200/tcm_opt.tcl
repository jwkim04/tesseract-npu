# hw/u200/tcm_opt.tcl — opt_design pre-hook of the k1 and k4 links (hw/u200/k1.cfg, k4.cfg): no BRAM power
# optimization on the control core's TCMs. opt_design's power task gates the BRAM enables with the observability
# don't-care of the data (the fetch / data address back on the enables), the path the TCM bus keeps free: gate T1
# (CC §14.3) was measured with this exclusion, as rtl/ctrl/viv/ctrl_ooc.tcl and tb/core/viv/core_ooc.tcl do it.
# The TCMs are the RAMB36 of u_itcm / u_dtcm (rtl/ctrl/viv/tsr_ctrl_tcm.xdc maps them to block RAM).
set tsr_tcm_ram [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36* && (NAME =~ *u_itcm/* || NAME =~ *u_dtcm/*)}]
puts "TCMOPT [llength $tsr_tcm_ram] TCM block RAMs excluded from power optimization"
if {[llength $tsr_tcm_ram] > 0} { set_power_opt -exclude_cells $tsr_tcm_ram }

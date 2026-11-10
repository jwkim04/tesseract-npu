# rtl/ctrl/viv/tsr_ctrl_cdc.xdc — clock-domain-crossing constraints of the control core and the DB block
# (CC §11.3; docs/ctrl-riscv.md §7.2): every crossing between the core clock and clk_rv, whichever block
# it sits in (tsr_ctrl_cdc, tsr_dbg_rv, any later one built from the rtl/ctrl/cdc primitives).
#
# Plain XDC: Vivado's constraint reader accepts no control flow here (no if, foreach, proc, dict), so the file
# is straight-line commands, which is also what a v++ link can apply. It is used in three places, each
# setting the two clock ports first (set tsr_cdc_core_port ..; set tsr_cdc_rv_port ..):
#   - the packaged kernel: rtl/platform/alveo_xrt/package_xo.tcl copies it into the IP with the ports
#     ap_clk / ap_clk_2 in front (USED_IN synthesis, implementation and out of context, processed late), so a
#     v++ link applies it, scoped to the kernel;
#   - out-of-context runs of tsr_core (tb/core/viv/core_ooc.tcl) and tsr_ctrl (rtl/ctrl/viv/ctrl_ooc.tcl): the
#     ports clk / clk_rv, sourced after the clocks are created on them.
# The report_cdc waivers need loops and procs and are not constraints: rtl/ctrl/viv/tsr_cdc_waivers.tcl, sourced
# by the out-of-context scripts and by the post-route gate of the link (hw/u200/cdc_check.tcl).
#
# Conventions the patterns rely on (checked by rtl/ctrl/viv/check_cdc_names.py in rtl/ctrl/lint.sh):
# tsr_sync instances are named u_sync*, tsr_rst_sync u_rst*, tsr_cdc_hs u_hs*; the gray registers of
# tsr_cdc_gray / tsr_cdc_fifo are called gray_q (wgray_q / rgray_q). A signal enters a synchronizer straight
# from a flop that drives nothing else (report_cdc CDC-10, CDC-11: tsr_ctrl_cdc's level bundle, tsr_dbg_rv's
# levels and the reset sources of the clk_rv synchronizers have a register of their own for it).
#
# - ASYNC_REG on every synchronizer stage (tsr_sync's s_q registers, tsr_rst_sync's s_q).
# - set_max_delay -datapath_only <= min(period) on every path between the two clocks (the gray pointers and
#   counters, the handshake request / response registers, the FIFO LUTRAM -> output register paths and the
#   level synchronizers' first stage). No set_clock_groups: it would override the max delays (CC §11.3).
# - set_bus_skew on the gray-coded buses (the gray registers to the synchronizer stages they drive), to the
#   smaller period, so that a sample sees at most one step. Cap both bounds at 3.333 ns to retain
#   the original 300 MHz crossing budget when a kernel runs slower (K4: core 200 MHz, clk_rv 150 MHz, so the
#   cap is below both periods, 5.0 and 6.667 ns, and stays the bound).
# Gate (CC §14.3 T2 / T3): rtl/ctrl/viv/cdc_gate.tcl.
set clk_core [get_clocks -of_objects [get_ports $tsr_cdc_core_port]]
set clk_rv   [get_clocks -of_objects [get_ports $tsr_cdc_rv_port]]
set p_min    [expr {min(3.333, [get_property PERIOD $clk_core], [get_property PERIOD $clk_rv])}]

set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_sync*/s_q_reg*}]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *u_rst*/g_s*.s_q_reg*}]

set_max_delay -datapath_only -from $clk_core -to $clk_rv $p_min
set_max_delay -datapath_only -from $clk_rv -to $clk_core $p_min

set_bus_skew -from [get_cells -hier -filter {NAME =~ *gray_q_reg* && IS_SEQUENTIAL}] -to [get_cells -hier -filter {NAME =~ *u_sync*/s_q_reg* && IS_SEQUENTIAL}] $p_min

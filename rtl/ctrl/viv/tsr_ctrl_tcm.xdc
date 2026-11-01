# rtl/ctrl/viv/tsr_ctrl_tcm.xdc — synthesis constraints of the control subsystem's TCMs (CC §4.2,
# CC-3): ITCM and DTCM map to block RAM (RAMB36 in 32K x 1 columns, read latency 1), never to
# URAM, which the SPM keeps. Read before synth_design with either SRAM model: the attribute-free
# rtl/tech/generic/tsr_sram_1r1w.sv, or the U200 target's rtl/tech/xilinx/tsr_sram_1r1w.sv, which picks
# "ultra" for every memory of DEPTH >= 4096 and >= 256 Kib, the TCMs included, and whose attribute this
# file overrides (measured 2026-11-01, Vivado 2024.2: a 32K x 32 memory of the Xilinx wrapper is 8 URAM
# without this file and 32 RAMB36, no URAM, with it). Applied in every core / kernel flow of the U200
# target: tb/core/viv/core_ooc.tcl, tb/platform/viv/kernel_ooc.tcl and, as the IP's synthesis constraint,
# rtl/platform/alveo_xrt/package_xo.tcl (so the v++ links get it); rtl/ctrl/viv/ctrl_ooc.tcl with sram=generic.
# With the ITCM in a URAM cascade the first whole-core synthesis had its worst path (-1.257 ns at 3.333 ns
# core / 6 ns clk_rv: Ibex instruction fetch -> ITCM).
# CASCADE_HEIGHT 1: no BRAM cascades, so a 32 K-deep TCM is 32K x 1 columns (ITCM 32, DTCM 16
# RAMB36) with neither an address decode into the BRAM enables nor an output mux (CC §4.2; the
# default 8-high / 4-high cascades put the fetch address on the enables, gate T1).
set tcms [get_cells -hier -filter {NAME =~ *u_itcm/mem_reg* || NAME =~ *u_dtcm/mem_reg*}]
set_property RAM_STYLE BLOCK $tcms
set_property CASCADE_HEIGHT 1 $tcms

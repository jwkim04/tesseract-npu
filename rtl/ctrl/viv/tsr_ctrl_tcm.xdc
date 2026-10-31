# rtl/ctrl/viv/tsr_ctrl_tcm.xdc — synthesis constraints of the control subsystem's TCMs (CC §4.2,
# CC-3): ITCM and DTCM map to block RAM (RAMB36 in 32K x 1 columns, read latency 1), never to
# URAM, which the SPM keeps. Read before synth_design together with
# rtl/tech/generic/tsr_sram_1r1w.sv (the attribute-free model). The U200 target's
# rtl/tech/xilinx/tsr_sram_1r1w.sv picks "ultra" for every memory of DEPTH >= 4096 and >= 256 Kib,
# the TCMs included, which is what this file avoids until that wrapper takes a style override
# (open item of W4 item 3).
# CASCADE_HEIGHT 1: no BRAM cascades, so a 32 K-deep TCM is 32K x 1 columns (ITCM 32, DTCM 16
# RAMB36) with neither an address decode into the BRAM enables nor an output mux (CC §4.2; the
# default 8-high / 4-high cascades put the fetch address on the enables, gate T1).
set tcms [get_cells -hier -filter {NAME =~ *u_itcm/mem_reg* || NAME =~ *u_dtcm/mem_reg*}]
set_property RAM_STYLE BLOCK $tcms
set_property CASCADE_HEIGHT 1 $tcms

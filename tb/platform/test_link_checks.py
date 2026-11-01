"""Link gate failures and U200 clock/reset binding, without launching Vivado."""
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]


def run_tcl(tmp_path, body):
    script = tmp_path / "check.tcl"
    script.write_text(body)
    return subprocess.run(["tclsh", str(script)], cwd=REPO, text=True,
                          capture_output=True, timeout=30)


@pytest.mark.parametrize("flow", ["core", "xo"])
@pytest.mark.parametrize("unsafe", [True, False])
def test_cdc_failure_stops_flow(tmp_path, flow, unsafe):
    # Exercise the real callers. Only tool APIs/reports and unrelated reporting
    # are stubbed; a FAIL result must become the Tcl process's exit status.
    (tmp_path / "ip" / "kip").mkdir(parents=True)
    (tmp_path / "ip" / "kip" / "kip.dcp").touch()
    (tmp_path / "xo" / "ip_repo").mkdir(parents=True)
    entry = "tb/core/viv/core_ooc.tcl" if flow == "core" else "tb/platform/viv/xo_cdc.tcl"
    args = [str(REPO), str(tmp_path), "synth_only"] if flow == "core" else [str(REPO), "unused.xo", str(tmp_path)]
    body = r'''
proc unknown {args} { return {} }
proc get_property {prop objects} { return $objects }
proc get_clocks {args} { return [lindex $args end] }
proc get_cells {args} { return sync_reg }
proc get_ipdefs {args} { return tesseract:kernel:tsr_u200_k4:1.0 }
proc get_files {args} { return tsr_ctrl_cdc.xdc }
proc exec {args} { return {} }
proc report_cdc {args} { return "Critical ap_clk ap_clk_2 unsafe\nCritical clk clk_rv unsafe" }
proc report_clock_interaction {args} { return {} }
rename source original_source
proc source {path} {
    if {[file extension $path] eq ".xdc"} { return {} }
    set result [uplevel 1 [list original_source $path]]
    if {[file tail $path] eq "core_rpt.tcl"} {
        foreach p {cooc_util cooc_hier cooc_paths} { proc $p {args} {} }
    }
    return $result
}
'''
    body += "\nset argv [list " + " ".join("{" + a + "}" for a in args) + "]\n"
    if not unsafe:
        body += r'''
proc report_cdc {args} { return {} }
proc report_clock_interaction {args} {
    return "clk clk_rv Max Delay Datapath Only\nclk_rv clk Max Delay Datapath Only\nap_clk ap_clk_2 Max Delay Datapath Only\nap_clk_2 ap_clk Max Delay Datapath Only"
}
'''
    body += f"source {{{REPO / entry}}}\n"
    r = run_tcl(tmp_path, body)
    assert ("FAIL" if unsafe else "PASS") in r.stdout, (r.stdout, r.stderr)
    assert (r.returncode != 0) == unsafe, (r.stdout, r.stderr)


BD_MOCK = r'''
set cells {tsr_k4 ss_ucs psr0 psr1}
array set nets {
    tsr_k4/ap_clk c0 tsr_k4/ap_clk_2 c1
    tsr_k4/ap_rst_n r0 tsr_k4/ap_rst_n_2 r1
    ss_ucs/aclk_kernel_00 c0 ss_ucs/aclk_kernel_01 c1
    psr0/slowest_sync_clk c0 psr1/slowest_sync_clk c1
    psr0/peripheral_aresetn r0 psr1/peripheral_aresetn r1
}
array set drivers {c0 ss_ucs/aclk_kernel_00 c1 ss_ucs/aclk_kernel_01
                  r0 psr0/peripheral_aresetn r1 psr1/peripheral_aresetn}
proc get_bd_cells {args} {
    if {[lsearch $args -filter] >= 0} {return tsr_k4}
    if {[lsearch $args -of_objects] >= 0} {
        return [file dirname [lindex $args end]]
    }
    set n [string trimleft [lindex $args end] /]
    if {$n in $::cells} {return $n}
    return {}
}
proc get_bd_pins {args} {
    set i [lsearch $args -of_objects]
    if {$i >= 0} {
        set net [lindex $args [expr {$i+1}]]
        if {[info exists ::drivers($net)]} {return $::drivers($net)}
        return {}
    }
    set n [string trimleft [lindex $args end] /]
    if {[info exists ::nets($n)]} {return $n}
    return {}
}
proc get_bd_nets {args} {
    set pin [lindex $args end]
    if {[info exists ::nets($pin)]} {return $::nets($pin)}
    return {}
}
proc get_property {prop obj} {
    switch -- $prop {
        NAME { return $obj }
        VLNV { return xilinx.com:ip:proc_sys_reset:5.0 }
        default {error "unexpected property $prop on $obj"}
    }
}
'''


@pytest.mark.parametrize("mutation", ["", "alias", "swapped", "reset", "reset_clock", "missing"])
def test_bd_clock_reset_binding(tmp_path, mutation):
    changes = {
        "": "",
        "alias": "set nets(tsr_k4/ap_clk_2) c0",
        "swapped": "set nets(tsr_k4/ap_clk) c1; set nets(tsr_k4/ap_clk_2) c0",
        "reset": "set nets(tsr_k4/ap_rst_n_2) r0",
        "reset_clock": "set nets(psr1/slowest_sync_clk) c0",
        "missing": "unset nets(tsr_k4/ap_clk_2)",
    }
    r = run_tcl(tmp_path, BD_MOCK + "\n" + changes[mutation] +
                f"\nsource {{{REPO / 'hw/u200/clock_bd_check.tcl'}}}\n")
    assert (r.returncode == 0) == (mutation == ""), (r.stdout, r.stderr)
    assert "CLKBD" in r.stdout + r.stderr, (r.stdout, r.stderr)


NET_MOCK = r'''
array set periods {core 3.333 rv 6.000}
set hart_regs {rv_hart}
array set clock_for {core_c core rv_c rv rv_hart rv core_reset_q core rv_reset_q rv}
array set reset_net {kernel/ap_rst_n reset0 kernel/ap_rst_n_2 reset1}
array set roots {kernel/ap_rst_n core_reset_q kernel/ap_rst_n_2 rv_reset_q}
proc get_cells {args} {
    set i [lsearch $args -of_objects]
    if {$i >= 0} {
        set out {}
        foreach n [lindex $args [expr {$i+1}]] {
            if {$n ne "raw_async"} {lappend out $n}
        }
        return $out
    }
    set f [lindex $args end]
    if {[string match *u_sync_lv* $f]} {return rv_c}
    if {[string match *lv_q_reg* $f]} {return core_c}
    if {[string match *u_ctrl/u_rv/* $f]} {return $::hart_regs}
    if {[string match *tsr_u200_k* $f]} {return kernel}
    return {}
}
proc get_pins {args} {
    set i [lsearch $args -of_objects]
    if {$i >= 0} {return [lindex $args [expr {$i+1}]]}
    set n [lindex $args end]
    if {[info exists ::reset_net($n)]} {return $n}
    return {}
}
proc get_clocks {args} {
    set i [lsearch $args -of_objects]
    if {$i >= 0} {
        set out {}
        foreach n [lindex $args [expr {$i+1}]] {
            if {[info exists ::clock_for($n)]} {lappend out $::clock_for($n)}
        }
        return [lsort -unique $out]
    }
    return [lindex $args end]
}
proc get_nets {args} {
    set pin [lindex $args end]
    if {[info exists ::reset_net($pin)]} {return $::reset_net($pin)}
    return {}
}
proc all_fanin {args} {
    set pin [lindex $args end]
    if {[info exists ::roots($pin)]} {return $::roots($pin)}
    return {}
}
proc get_property {prop obj} {
    switch -- $prop {
        NAME {return $obj}
        PERIOD {return $::periods($obj)}
        default {error "unexpected property $prop on $obj"}
    }
}
'''


@pytest.mark.parametrize("mutation", ["", "alias", "rv500", "swapped", "reset", "reset_clock", "missing", "unclocked_hart", "async_reset", "nan"])
def test_synth_clock_reset_binding(tmp_path, mutation):
    changes = {
        "": "",
        "alias": "set clock_for(rv_c) core",
        "rv500": "set periods(rv) 2.0",
        "swapped": "set clock_for(core_c) rv; set clock_for(rv_c) core",
        "reset": "set reset_net(kernel/ap_rst_n_2) reset0",
        "reset_clock": "set clock_for(rv_reset_q) core",
        "missing": "unset clock_for(rv_c)",
        "unclocked_hart": "lappend hart_regs unclocked_ff",
        "async_reset": "lappend roots(kernel/ap_rst_n_2) raw_async",
        "nan": "set periods(rv) NaN",
    }
    r = run_tcl(tmp_path, NET_MOCK + "\n" + changes[mutation] +
                f"\nsource {{{REPO / 'hw/u200/clock_net_check.tcl'}}}\n")
    assert (r.returncode == 0) == (mutation == ""), (r.stdout, r.stderr)
    assert "CLKNET" in r.stdout + r.stderr, (r.stdout, r.stderr)


@pytest.mark.parametrize("records,valid", [
    (["kernel:0:ss_ucs/aclk_kernel_00:300.0", "kernel:1:ss_ucs/aclk_kernel_01:166.666667"], True),
    (["kernel:0:ss_ucs/aclk_kernel_00:276.3", "kernel:1:ss_ucs/aclk_kernel_01:166.6"], True),
    (["kernel:0:ss_ucs/aclk_kernel_00:276.29", "kernel:1:ss_ucs/aclk_kernel_01:166.6"], False),
    (["kernel:0:ss_ucs/aclk_kernel_00:276.3", "kernel:1:ss_ucs/aclk_kernel_01:500.0"], False),
    (["kernel:0:ss_ucs/aclk_kernel_00:276.3"], False),
    (["kernel:0:ss_ucs/aclk_kernel_00:300", "kernel:0:ss_ucs/aclk_kernel_00:300"], False),
    (["kernel:0:ss_ucs/aclk_kernel_01:300", "kernel:1:ss_ucs/aclk_kernel_00:166.6"], False),
    (["kernel:0:ss_ucs/aclk_kernel_00:NaN", "kernel:1:ss_ucs/aclk_kernel_01:166.6"], False),
    (["kernel:0:ss_ucs/aclk_kernel_00:Inf", "kernel:1:ss_ucs/aclk_kernel_01:166.6"], False),
    (["kernel:0:ss_ucs/aclk_kernel_00:300:extra", "kernel:1:ss_ucs/aclk_kernel_01:166.6"], False),
])
def test_final_scaled_frequencies(tmp_path, records, valid):
    (tmp_path / "_new_clk_freq").write_text("\n".join(records) + "\n")
    r = run_tcl(tmp_path, f"cd {{{tmp_path}}}\nsource {{{REPO / 'hw/u200/clock_final_check.tcl'}}}\n")
    assert (r.returncode == 0) == valid, (r.stdout, r.stderr)
    assert "CLKFINAL" in r.stdout + r.stderr


@pytest.mark.parametrize("kernel", ["k1", "k4"])
def test_link_config_clock_requests_and_hooks(kernel):
    # Protect the explicit platform clock selection and all three check stages.
    text = (REPO / f"hw/u200/{kernel}.cfg").read_text()
    assert "freqHz=" not in text
    # U200's scalable clocks use kernel_frequency. clock.id is restricted to
    # fixed references: a real CFGEN run rejects IDs 0/1 with CFGEN83-2244.
    assert "kernel_frequency=0:300|1:166.666667\n" in text
    assert not any(line.startswith("id=") for line in text.splitlines())
    for hook in ("clock_bd_check.tcl", "impl_pre.tcl", "cdc_check.tcl", "clock_final_check.tcl"):
        assert f"=../../../hw/u200/{hook}\n" in text
    if kernel == "k4":
        for step in ("PLACE_DESIGN", "PHYS_OPT_DESIGN", "POST_ROUTE_PHYS_OPT_DESIGN"):
            assert (f"prop=run.impl_1.STEPS.{step}.TCL.POST="
                    "../../../hw/u200/k4_locality_check.tcl\n") in text

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
    for hook in ("clock_bd_check.tcl", "clock_vitis_compat.tcl", "impl_pre.tcl", "cdc_check.tcl", "clock_final_check.tcl"):
        assert f"=../../../hw/u200/{hook}\n" in text
    assert ("prop=run.impl_1.STEPS.INIT_DESIGN.TCL.PRE="
            "../../../hw/u200/clock_vitis_compat.tcl\n") in text
    if kernel == "k4":
        for step in ("PLACE_DESIGN", "PHYS_OPT_DESIGN", "POST_ROUTE_PHYS_OPT_DESIGN"):
            assert (f"prop=run.impl_1.STEPS.{step}.TCL.POST="
                    "../../../hw/u200/k4_locality_check.tcl\n") in text


# Model the 2024.2 part-keyed APIs, including the new period argument. The
# legacy helpers below reproduce the installed Vitis caller's Project key and
# missing argument; the fixture rejects those before a default 1:1 can escape.
CLKAPI_MOCK = r'''
set mode normal
set project_ready 0
set api_live 0
set api_cleanups 0
set tool_version 2024.2
set part xcu200-fsgd2104-2-e
array set chosen {ChosenM 83.000 ChosenD 10 ChosenDiv0 5.000}
proc version {args} {return $::tool_version}
proc current_project {} {
    if {!$::project_ready} {error "PART lookup before project exists"}
    return Project
}
proc get_property {prop obj} {
    if {$prop ne "PART" || $obj ne "Project"} {error "unexpected property lookup"}
    return $::part
}
proc api_key {key inst} {
    if {$key ne $::part || $key eq "Project"} {error "API expected Part key, got $key"}
    if {$inst eq ""} {error "API instance missing"}
}
proc Init_Clkwiz {key inst part} {
    api_key $key $inst
    if {$part ne $key || $::api_live} {error "bad init or leaked instance"}
    set ::api_live 1
}
proc SetClkwizProperty {key inst prop val} {
    api_key $key $inst
    if {!$::api_live || $prop ne "UseFinePS" || $val ne "true"} {error "bad property setter"}
}
proc GetClosestSolution {args} {
    if {[llength $args] != 15} {error "API requires 15 args including timePeriod"}
    lassign $args key inst req phase duty input second count minj nond pll dbg period phy phyfreq
    api_key $key $inst
    if {!$::api_live || abs($period - 1000.0/$input) > 1e-9 ||
        $phase != 0 || $duty != 50 || $second != 0 || $count != 1 ||
        $minj ne "false" || $nond ne "false" || $pll ne "false" ||
        $dbg ne "false" || $phy ne "false" || $phyfreq != 0} {error "bad calculation arguments"}
    if {$::mode eq "return_zero"} {return 0}
    if {$::mode eq "api_error"} {error "calculation raised an error"}
    return 1
}
proc GetClkwizProperty {key inst prop} {
    api_key $key $inst
    if {!$::api_live} {error "property without instance"}
    return $::chosen($prop)
}
proc UnInit_Clkwiz {key inst} {
    api_key $key $inst
    if {!$::api_live} {error "duplicate cleanup"}
    set ::api_live 0
    incr ::api_cleanups
}
namespace eval ocl_util {
    namespace export write_user_impl_clock_constraint
    proc initialize_clkwiz_debug {is_versal part} {Init_Clkwiz [current_project] test1 $part}
    proc uninitialize_clkwiz_debug {is_versal part} {UnInit_Clkwiz [current_project] test1}
    proc set_clkwiz_prop {input req} {
        GetClosestSolution [current_project] test1 $req 0 50 $input 0 1 false false false false false false
    }
    proc get_clkwiz_prop {prop} {GetClkwizProperty [current_project] test1 $prop}
    proc write_user_impl_clock_constraint {inst freqs log out versal} {
        set part [get_property PART [current_project]]
        initialize_clkwiz_debug $versal $part
        set_clkwiz_prop 100 $freqs
        foreach p {ChosenM ChosenD ChosenDiv0} {get_clkwiz_prop $p}
        if {$::mode eq "writer_error"} {error "native XDC application failed"}
        uninitialize_clkwiz_debug $versal $part
        return native_writer_result
    }
}
namespace import ocl_util::*
'''


def test_clock_api_legacy_caller_is_rejected(tmp_path):
    r = run_tcl(tmp_path, CLKAPI_MOCK + r'''
set project_ready 1
ocl_util::initialize_clkwiz_debug false $part
''')
    assert r.returncode != 0
    assert "API expected Part key, got Project" in r.stderr


@pytest.mark.parametrize("mutation,diagnostic", [
    ("", "CLKAPI"),
    ("return_zero", "calculation failed"),
    ("api_error", "calculation raised"),
    ("writer_error", "native XDC application failed"),
    ("default", "default 1:1"),
    ("empty", "ChosenM"),
    ("nan_result", "ChosenM"),
    ("floor_error", "effective"),
    ("zero", "requested"),
    ("nan", "requested"),
    ("inf", "requested"),
    ("zero_input", "input"),
    ("bad_part", "U200"),
    ("versal", "U200"),
])
def test_vitis_clock_api_compatibility(tmp_path, mutation, diagnostic):
    changes = {
        "": "", "return_zero": "set mode return_zero", "api_error": "set mode api_error",
        "writer_error": "set mode writer_error",
        "default": "array set chosen {ChosenM 1 ChosenD 1 ChosenDiv0 1}",
        "empty": 'set chosen(ChosenM) ""', "nan_result": "set chosen(ChosenM) NaN",
        "floor_error": "set chosen(ChosenM) 8.375; set chosen(ChosenD) 1",
        "zero": "set req 0", "nan": "set req NaN", "inf": "set req Inf",
        "zero_input": "set input 0", "bad_part": "set part xcvu9p-flga2104-2L-e",
        "versal": "set versal true",
    }
    body = CLKAPI_MOCK + f"\nsource {{{REPO / 'hw/u200/clock_vitis_compat.tcl'}}}\n"
    body += "set project_ready 1\nset req 166\nset input 100\nset versal false\n" + changes[mutation]
    body += r'''
set rc [catch {
    if {$input != 100} {
        ocl_util::initialize_clkwiz_debug $versal $part
        ocl_util::set_clkwiz_prop $input $req
    } else {
        set result [write_user_impl_clock_constraint unused $req {} unused $versal]
        if {$result ne "native_writer_result"} {error "native writer result changed"}
    }
} reason options]
if {$api_live} {error "compatibility helper leaked clock instance"}
if {$rc} {puts stderr $reason; exit 1}
if {$api_cleanups != 1} {error "expected exactly one instance cleanup"}
puts "CLKAPI fixture PASS"
'''
    r = run_tcl(tmp_path, body)
    assert (r.returncode == 0) == (mutation == ""), (r.stdout, r.stderr)
    assert diagnostic in r.stdout + r.stderr, (r.stdout, r.stderr)
    assert "leaked clock instance" not in r.stderr


@pytest.mark.parametrize("guard", ["version", "namespace", "helper"])
def test_vitis_clock_api_guard(tmp_path, guard):
    changes = {"version": "set tool_version 2025.1",
               "namespace": "namespace delete ocl_util",
               "helper": "rename ocl_util::get_clkwiz_prop {}"}
    r = run_tcl(tmp_path, CLKAPI_MOCK + "\n" + changes[guard] +
                f"\nsource {{{REPO / 'hw/u200/clock_vitis_compat.tcl'}}}\n")
    assert r.returncode != 0
    assert "CLKAPI" in r.stdout + r.stderr

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
array set periods {core 4.000 rv 6.000}
set kernel_ref tsr_u200_k4
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
        REF_NAME - ORIG_REF_NAME {return $::kernel_ref}
        default {error "unexpected property $prop on $obj"}
    }
}
'''


@pytest.mark.parametrize("mutation", ["", "alias", "core300", "core240", "rv500", "swapped", "reset", "reset_clock", "missing", "unclocked_hart", "async_reset", "nan", "unknown_kernel"])
def test_synth_clock_reset_binding(tmp_path, mutation):
    changes = {
        "": "",
        "alias": "set clock_for(rv_c) core",
        # A stale 300 MHz checkpoint is not a valid native 250 MHz request.
        "core300": "set periods(core) 3.333",
        "core240": "set periods(core) 4.166667",
        "rv500": "set periods(rv) 2.0",
        "swapped": "set clock_for(core_c) rv; set clock_for(rv_c) core",
        "reset": "set reset_net(kernel/ap_rst_n_2) reset0",
        "reset_clock": "set clock_for(rv_reset_q) core",
        "missing": "unset clock_for(rv_c)",
        "unclocked_hart": "lappend hart_regs unclocked_ff",
        "async_reset": "lappend roots(kernel/ap_rst_n_2) raw_async",
        "nan": "set periods(rv) NaN",
        "unknown_kernel": "set kernel_ref tsr_unknown",
    }
    r = run_tcl(tmp_path, NET_MOCK + "\n" + changes[mutation] +
                f"\nsource {{{REPO / 'hw/u200/clock_net_check.tcl'}}}\n")
    assert (r.returncode == 0) == (mutation == ""), (r.stdout, r.stderr)
    assert "CLKNET" in r.stdout + r.stderr, (r.stdout, r.stderr)


@pytest.mark.parametrize("period,valid", [(3.333, True), (4.0, False)])
def test_k1_keeps_its_300_mhz_request(tmp_path, period, valid):
    r = run_tcl(tmp_path, NET_MOCK + f"\nset kernel_ref tsr_u200_k1\nset periods(core) {period}\n" +
                f"source {{{REPO / 'hw/u200/clock_net_check.tcl'}}}\n")
    assert (r.returncode == 0) == valid, (r.stdout, r.stderr)


@pytest.mark.parametrize("records,valid", [
    (["kernel:0:ss_ucs/aclk_kernel_00:300.0", "kernel:1:ss_ucs/aclk_kernel_01:166.666667"], True),
    (["kernel:0:ss_ucs/aclk_kernel_00:276.3", "kernel:1:ss_ucs/aclk_kernel_01:166.6"], True),
    (["kernel:0:ss_ucs/aclk_kernel_00:276.29", "kernel:1:ss_ucs/aclk_kernel_01:166.6"], True),
    (["kernel:0:ss_ucs/aclk_kernel_00:250.0", "kernel:1:ss_ucs/aclk_kernel_01:166.0"], True),
    (["kernel:0:ss_ucs/aclk_kernel_00:249.99", "kernel:1:ss_ucs/aclk_kernel_01:166.0"], False),
    (["kernel:0:ss_ucs/aclk_kernel_00:250.0", "kernel:1:ss_ucs/aclk_kernel_01:158.33"], True),
    (["kernel:0:ss_ucs/aclk_kernel_00:250.0", "kernel:1:ss_ucs/aclk_kernel_01:158.32"], False),
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
    core = 250 if kernel == "k4" else 300
    assert f"kernel_frequency=0:{core}|1:166.666667\n" in text
    assert not any(line.startswith("id=") for line in text.splitlines())
    for hook in ("clock_bd_check.tcl", "clock_vitis_compat.tcl", "impl_pre.tcl", "cdc_check.tcl", "clock_final_check.tcl"):
        assert f"=../../../hw/u200/{hook}\n" in text
    assert ("prop=run.impl_1.STEPS.INIT_DESIGN.TCL.PRE="
            "../../../hw/u200/clock_vitis_compat.tcl\n") in text
    if kernel == "k4":
        # Placement constraints are the Pblocks of whole major blocks (AGENTS.md); the earlier per-cell
        # ownership audit (k4_locality_check.tcl) is not part of the link.
        assert "prop=run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=../../../hw/u200/k4_major_blocks.tcl\n" in text
        assert "k4_locality_check" not in text


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


CDC_LINK_MOCK = NET_MOCK + r'''
# Model the confirmed post-override failure: clocks exist, but the clock-pair
# max-delay exceptions do not. The actual source XDC must restore them.
set read_calls 0
set max_delays {}
set bus_skews {}
set cdc_unsafe 0
set drop_constraints 0
set read_scope {}
set no_paths 0
set wrong_requirement 0
set no_gray 0
set no_sync 0
set wrong_interaction 0
set missing_skew_report 0
set loose_skew 0
array set clock_for {kernel/ap_clk core kernel/ap_clk_2 rv}
rename get_cells net_get_cells
proc get_cells {args} {
    set f [lindex $args end]
    if {[string match *u_sync_lv* $f]} {return [net_get_cells {*}$args]}
    if {[string match *gray_q_reg* $f]} {
        if {$::no_gray} {return {}}
        return kernel/gray_q_reg
    }
    if {[string match *u_sync*/s_q_reg* $f]} {
        if {$::no_sync} {return {}}
        return kernel/u_sync/s_q_reg
    }
    if {[string match *u_rst*/g_s*.s_q_reg* $f]} {return kernel/u_rst/g_s.s_q_reg}
    return [net_get_cells {*}$args]
}
rename get_pins net_get_pins
proc get_pins {args} {
    if {[lindex $args end] in {kernel/ap_clk kernel/ap_clk_2}} {
        return [lindex $args end]
    }
    return [net_get_pins {*}$args]
}
proc get_ports {name} {
    if {$::read_scope ne "kernel" || $name ni {ap_clk ap_clk_2}} {
        error "unscoped CDC port $name"
    }
    return kernel/$name
}
proc set_property {args} {}
proc set_max_delay {args} {
    if {!$::drop_constraints} {lappend ::max_delays $args}
}
proc set_bus_skew {args} {
    if {!$::drop_constraints} {lappend ::bus_skews $args}
}
proc read_xdc {args} {
    incr ::read_calls
    set i [lsearch $args -cells]
    if {$i < 0 || [lindex $args [expr {$i+1}]] ne "kernel"} {
        error "CDC XDC must be scoped to actual kernel"
    }
    set ::read_scope kernel
    try {uplevel #0 [list source [lindex $args end]]} finally {set ::read_scope {}}
}
proc create_waiver {args} {}
rename get_property net_get_property
proc get_property {prop obj} {
    if {$prop eq "REQUIREMENT"} {
        if {$::wrong_requirement || [llength $::max_delays] != 2} {return 0.040}
        return [lindex [lindex $::max_delays 0] end]
    }
    return [net_get_property $prop $obj]
}
proc get_timing_paths {args} {
    if {$::no_paths} {return {}}
    return crossing_path
}
proc report_cdc {args} {
    if {$::cdc_unsafe} {return "Critical core rv unsafe"}
    return "Info core rv safe\nInfo rv core safe"
}
proc report_clock_interaction {args} {
    if {$::wrong_interaction || [llength $::max_delays] != 2} {return "core rv Timed\nrv core Timed"}
    return "core rv Max Delay Datapath Only\nrv core Max Delay Datapath Only"
}
proc report_bus_skew {args} {
    set i [lsearch $args -file]
    set f [open [lindex $args [expr {$i+1}]] w]
    if {!$::missing_skew_report && [llength $::bus_skews] > 0} {
        puts $f {1 17 kernel/gray_q_reg}
        puts $f {        kernel/u_sync/s_q_reg}
        set bound [lindex [lindex $::bus_skews 0] end]
        if {$::loose_skew} {set bound 99}
        puts $f "        Slow $bound 0.073 3.260"
    } else {
        puts $f "platform/src_gray_ff platform/dest_gray_ff"
    }
    puts $f {2. Bus Skew Report Per Constraint}
    close $f
}
'''


def test_impl_pre_restores_cdc_after_clock_override(tmp_path):
    body = CDC_LINK_MOCK + f"\ncd {{{tmp_path}}}\nsource {{{REPO / 'hw/u200/impl_pre.tcl'}}}\n"
    body += r'''
if {$read_calls != 1} {error "missing post-override CDC XDC restore"}
if {[llength $max_delays] != 2 || [llength $bus_skews] != 1} {
    error "missing max-delay or bus-skew constraints"
}
foreach args $max_delays {
    if {[lsearch $args -datapath_only] < 0 || [lindex $args end] != 3.333} {
        error "CDC restore changed datapath-only min-period contract"
    }
}
puts "CDC restore fixture PASS"
'''
    r = run_tcl(tmp_path, body)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "CDC restore fixture PASS" in r.stdout
    assert (tmp_path / "tsr_ctrl_cdc_link.xdc").read_text().endswith(
        (REPO / "rtl/ctrl/viv/tsr_ctrl_cdc.xdc").read_text())


@pytest.mark.parametrize("mutation", ["drop_constraints", "no_paths", "wrong_requirement",
                                       "cdc_unsafe", "wrong_interaction", "no_gray", "no_sync",
                                       "missing_skew_report", "loose_skew"])
def test_impl_pre_cdc_restore_fails_closed(tmp_path, mutation):
    body = CDC_LINK_MOCK + f"\nset {mutation} 1\nset ::env(TSR_CDC_GATE) warn\n"
    body += f"cd {{{tmp_path}}}\nsource {{{REPO / 'hw/u200/impl_pre.tcl'}}}\n"
    r = run_tcl(tmp_path, body)
    assert r.returncode != 0, (r.stdout, r.stderr)
    assert "CDCBIND" in r.stdout + r.stderr


@pytest.mark.parametrize("missing", [False, True])
def test_pre_route_checks_without_restoring(tmp_path, missing):
    body = CDC_LINK_MOCK + "\n"
    if not missing:
        body += "set max_delays {{-datapath_only 3.333} {-datapath_only 3.333}}\n"
        body += "set bus_skews {{3.333}}\n"
    body += f"cd {{{tmp_path}}}\nsource {{{REPO / 'hw/u200/cdc_pre_route.tcl'}}}\n"
    body += 'if {$read_calls != 0} {error "pre-route gate unexpectedly repaired constraints"}\n'
    r = run_tcl(tmp_path, body)
    assert (r.returncode == 0) == (not missing), (r.stdout, r.stderr)
    assert "CDCBIND" in r.stdout + r.stderr


@pytest.mark.parametrize("mutation", ["alias", "missing", "zero", "nan", "inf"])
def test_cdc_restore_rejects_invalid_clock_pair(tmp_path, mutation):
    changes = {"alias": "set clock_for(kernel/ap_clk_2) core",
               "missing": "unset clock_for(kernel/ap_clk_2)",
               "zero": "set periods(rv) 0", "nan": "set periods(rv) NaN",
               "inf": "set periods(rv) Inf"}
    body = CDC_LINK_MOCK + f"\n{changes[mutation]}\n"
    body += f"source {{{REPO / 'hw/u200/cdc_restore.tcl'}}}\n"
    body += f"tsr_cdc_link::restore invalid {{{tmp_path}}}\n"
    r = run_tcl(tmp_path, body)
    assert r.returncode != 0, (r.stdout, r.stderr)
    assert "CDCBIND" in r.stdout + r.stderr or "CLKCHK" in r.stderr
    assert not (tmp_path / "tsr_ctrl_cdc_link.xdc").exists()


@pytest.mark.parametrize("core,rv,bound", [(4.0, 6.023, 3.333), (3.333, 6.023, 3.333),
                                         (3.0, 6.023, 3.0), (4.0, 2.5, 2.5)])
def test_cdc_restore_preserves_cap_and_faster_clock_bounds(tmp_path, core, rv, bound):
    body = CDC_LINK_MOCK + f"\nset periods(core) {core}\nset periods(rv) {rv}\n"
    body += f"source {{{REPO / 'hw/u200/cdc_restore.tcl'}}}\n"
    body += f"tsr_cdc_link::restore actual_periods {{{tmp_path}}}\n"
    body += f"set expected_bound {bound}\n" + r'''
foreach args $max_delays {
    if {[lindex $args end] != $expected_bound} {error "CDC bound lost the cap or the faster clock"}
}
if {[lindex [lindex $bus_skews 0] end] != $expected_bound} {error "bus skew lost the bound"}
'''
    r = run_tcl(tmp_path, body)
    assert r.returncode == 0, (r.stdout, r.stderr)

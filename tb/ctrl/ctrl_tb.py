"""Helpers of the tb/ctrl bench: firmware build, ELF segments, the cocotb drivers of tb_ctrl
(clocks, resets, loader, DMI) and the RTL <-> simulator comparisons (CC §15.2, §15.4)."""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FW_DIR = Path(__file__).resolve().parent / "fw"
TOOLS = REPO / "build" / "tools" / "riscv" / "xpack-riscv-none-elf-gcc-14.2.0-3" / "bin" / "riscv-none-elf-"
FW_FLAGS = ["-march=rv32imc_zicsr_zifencei", "-mabi=ilp32", "-Os", "-ffreestanding",
            "-fno-tree-loop-distribute-patterns", "-ffp-contract=off", "-nostdlib", "-nostartfiles", "-g",
            "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wl,--no-warn-rwx-segments"]

DTCM_BASE = 0x00100000


def build_fw(out_dir: Path) -> Path:
    """Builds tb/ctrl/fw (crt0.S + fw_ctrl.c) with the pinned xPack GCC (docs/design/toolchain.lock)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = out_dir / "fw_ctrl.elf"
    srcs = [FW_DIR / "crt0.S", FW_DIR / "fw_ctrl.c", FW_DIR / "link.ld"]
    if elf.exists() and all(elf.stat().st_mtime >= s.stat().st_mtime for s in srcs):
        return elf
    cmd = [str(TOOLS) + "gcc"] + FW_FLAGS + ["-I" + str(REPO / "sw" / "sim" / "include"), "-T",
                                             str(FW_DIR / "link.ld"), str(FW_DIR / "crt0.S"),
                                             str(FW_DIR / "fw_ctrl.c"), "-lgcc", "-o", str(elf)]
    subprocess.run(cmd, check=True)
    return elf


def elf_segments(path):
    """PT_LOAD segments of an ELF32 little-endian file: [(paddr, bytes)] (filesz bytes)."""
    b = Path(path).read_bytes()
    phoff = struct.unpack_from("<I", b, 28)[0]
    phentsize, phnum = struct.unpack_from("<HH", b, 42)
    out = []
    for i in range(phnum):
        p_type, p_off, _va, p_pa, p_fsz, _msz, _fl, _al = struct.unpack_from("<8I", b, phoff + i * phentsize)
        if p_type == 1 and p_fsz:
            data = b[p_off:p_off + p_fsz]
            data += b"\0" * (-len(data) % 4)
            out.append((p_pa, data))
    return out


def elf_symbols(path):
    """name -> value of the ELF's .symtab (ELF32 LE)."""
    b = Path(path).read_bytes()
    shoff = struct.unpack_from("<I", b, 32)[0]
    shentsize, shnum, _shstrndx = struct.unpack_from("<HHH", b, 46)
    secs = [struct.unpack_from("<10I", b, shoff + i * shentsize) for i in range(shnum)]
    syms = {}
    for s in secs:
        if s[1] == 2:   # SHT_SYMTAB
            strtab = secs[s[6]]
            for k in range(s[5] // 16):
                name_off, value, _size, _info, _other, _shndx = struct.unpack_from("<IIIBBH", b, s[4] + 16 * k)
                end = b.index(b"\0", strtab[4] + name_off)
                name = b[strtab[4] + name_off:end].decode()
                if name:
                    syms[name] = value
    return syms


# ---------------------------------------------------------------- RVFI records
def read_rvfi(path):
    """rvfi.log (tb_ctrl.sv) -> list of dicts, debug-mode instructions removed."""
    recs = []
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        f = [int(x, 16) for x in line.split()]
        r = {"order": f[0], "pc": f[1], "next_pc": f[2], "insn": f[3], "trap": f[4], "intr": f[5], "rd": f[6],
             "rd_val": f[7], "mem_addr": f[8], "rmask": f[9], "wmask": f[10], "wdata": f[11], "rdata": f[12],
             "dbg": f[13]}
        if not r["dbg"]:
            recs.append(r)
    return recs


def read_ififo(path):
    out = []
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        f = line.split()
        v = [int(x, 16) for x in f[1:]]
        out.append({"kind": int(f[0]), "w": v[0:8], "dep0": v[8], "dep1": v[9], "rec": v[10], "ticket": v[11],
                    "pc": v[12]})
    return out


def compare_records(rtl, sim, limit=20):
    """RVFI (Ibex) against tsr_rv retire records: pc, instruction, trap, next pc, rd and its value,
    the memory address and the store data (CC §15.4)."""
    errs = []
    n = min(len(rtl), len(sim))
    for k in range(n):
        a, b = rtl[k], sim[k]
        bad = []
        if a["pc"] != b["pc"]:
            bad.append("pc %08x / %08x" % (a["pc"], b["pc"]))
        insn_b = b["insn"] if b["len"] == 4 else b["insn"] & 0xFFFF
        if b["len"] and a["insn"] != insn_b:      # len 0: instruction fetch fault (no instruction)
            bad.append("insn %08x / %08x" % (a["insn"], insn_b))
        if bool(a["trap"]) != bool(b["trap"]):
            bad.append("trap %d / %d" % (a["trap"], b["trap"]))
        # the RTL's next pc is the next record's pc (Ibex's rvfi_pc_wdata is pc + 4 for a trap and
        # for mret, not the handler or mepc)
        if k + 1 < len(rtl) and rtl[k + 1]["pc"] != b["next_pc"]:
            bad.append("next pc %08x / %08x" % (rtl[k + 1]["pc"], b["next_pc"]))
        if not b["trap"]:
            rd_a = a["rd"] if a["rd"] else 0
            if rd_a != b["rd"]:
                bad.append("rd x%d / x%d" % (rd_a, b["rd"]))
            elif b["rd"] and a["rd_val"] != b["rd_val"]:
                bad.append("x%d = %08x / %08x" % (b["rd"], a["rd_val"], b["rd_val"]))
            if b["mem_size"]:
                if a["mem_addr"] != b["mem_addr"]:
                    bad.append("mem addr %08x / %08x" % (a["mem_addr"], b["mem_addr"]))
                if b["mem_write"]:
                    m = (1 << (8 * b["mem_size"])) - 1
                    if (a["wdata"] & m) != (b["mem_val"] & m):
                        bad.append("store %08x / %08x" % (a["wdata"] & m, b["mem_val"] & m))
        if bad:
            errs.append("#%d pc %08x: %s" % (k, b["pc"], "; ".join(bad)))
            if len(errs) >= limit:
                break
    if len(rtl) != len(sim):
        errs.append("record count RTL %d / simulator %d" % (len(rtl), len(sim)))
    return errs


def compare_pushes(rtl, sim, limit=20):
    errs = []
    for k in range(min(len(rtl), len(sim))):
        a, b = rtl[k], sim[k]
        diffs = [f for f in ("kind", "w", "dep0", "dep1", "rec", "ticket", "pc") if a[f] != b[f]]
        if diffs:
            errs.append("entry %d differs in %s: RTL %s / simulator %s" % (
                k, ",".join(diffs), {f: a[f] for f in diffs}, {f: b[f] for f in diffs}))
            if len(errs) >= limit:
                break
    if len(rtl) != len(sim):
        errs.append("entry count RTL %d / simulator %d" % (len(rtl), len(sim)))
    return errs


# ---------------------------------------------------------------- cocotb drivers
async def start(dut, rv_ps=6000, core_ps=3620, seed=1, lat=6):
    """Clocks (independent, random phase), knobs, resets (CC §12.2)."""
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ClockCycles, Timer
    for name, val in (("cfg_seed", seed), ("cfg_lat", lat), ("cfg_pop_mute", 0), ("cfg_acc_mute", 0),
                      ("cfg_spm_mute", 0), ("cfg_cwin_mute", 0), ("cfg_lq_mute", 0), ("rv_run", 0), ("rv_boot", 0),
                      ("dbg_stop", 0), ("dbg_halt_tgl", 0), ("dbg_cancel_tgl", 0), ("dbg_halt_on_boot", 0),
                      ("irq_lq", 0), ("irq_lcq", 0), ("perf_time_en", 1), ("perf_evt_en", 1), ("q_hold", 0),
                      ("q_err", 0), ("q_nonempty", 0), ("err_valid", 0), ("err_rec", 0), ("ld_req", 0),
                      ("ld_we", 0), ("ld_addr", 0), ("ld_wdata", 0), ("dmi_req", 0), ("dmi_addr", 0),
                      ("dmi_op", 0), ("dmi_wdata", 0)):
        getattr(dut, name).value = val
    dut.rst_n.value = 0
    dut.rst_rv_n.value = 0
    clk = Clock(dut.clk, core_ps, unit="ps")
    clk_rv = Clock(dut.clk_rv, rv_ps, unit="ps")
    if os.environ.get("TB_CTRL_SYNC") == "1":      # CTRL_SYNC = 1 builds: clk_rv = clk (CC §11.1)
        cocotb.start_soon(clk.start())
        rv_task = cocotb.start_soon(clk_rv.start())
    else:                                           # independent clocks, random phase
        cocotb.start_soon(clk.start())
        await Timer((seed * 977) % rv_ps + 1, unit="ps")
        rv_task = cocotb.start_soon(clk_rv.start())
    await ClockCycles(dut.clk, 20)
    dut.rst_rv_n.value = 1
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 20)
    return rv_task


async def ld_access(dut, we, addr, wdata=0):
    from cocotb.triggers import RisingEdge, ReadOnly
    while not int(dut.ld_ready.value):
        await RisingEdge(dut.clk)
    dut.ld_req.value = 1
    dut.ld_we.value = we
    dut.ld_addr.value = addr
    dut.ld_wdata.value = wdata
    await RisingEdge(dut.clk)
    dut.ld_req.value = 0
    while True:
        await ReadOnly()
        if int(dut.ld_rsp_valid.value):
            r = (int(dut.ld_rsp_rdata.value), int(dut.ld_rsp_err.value), int(dut.ld_rsp_timeout.value))
            await RisingEdge(dut.clk)
            return r
        await RisingEdge(dut.clk)


async def dmi(dut, op, addr, wdata=0):
    """One DMI access (op 1 read, 2 write) through the core-side DMI port. Returns (data, resp, timeout)."""
    from cocotb.triggers import RisingEdge, ReadOnly
    while not int(dut.dmi_ready.value):
        await RisingEdge(dut.clk)
    dut.dmi_req.value = 1
    dut.dmi_op.value = op
    dut.dmi_addr.value = addr
    dut.dmi_wdata.value = wdata
    await RisingEdge(dut.clk)
    dut.dmi_req.value = 0
    while True:
        await ReadOnly()
        if int(dut.dmi_rsp_valid.value):
            r = (int(dut.dmi_rsp_rdata.value), int(dut.dmi_rsp_resp.value), int(dut.dmi_rsp_timeout.value))
            await RisingEdge(dut.clk)
            return r
        await RisingEdge(dut.clk)


async def load_image(dut, elf, patches=None):
    """Writes every PT_LOAD segment through the load port, then reads it back (EX §3.3 VERIFY's
    reads); patches = {addr: word} overrides words after the load."""
    words = {}
    for pa, data in elf_segments(elf):
        for i in range(0, len(data), 4):
            words[pa + i] = struct.unpack_from("<I", data, i)[0]
    for a, v in (patches or {}).items():
        words[a] = v
    for a, v in sorted(words.items()):
        _, err, to = await ld_access(dut, 1, a, v)
        assert not err and not to, "load write %08x refused (err %d, timeout %d)" % (a, err, to)
    bad = 0
    for a, v in sorted(words.items()):
        r, err, to = await ld_access(dut, 0, a)
        if err or to or r != v:
            bad += 1
    assert bad == 0, "%d words read back wrong through the load port" % bad
    return len(words)


async def wait_rt(dut, value, max_cycles=2_000_000, step=500):
    from cocotb.triggers import ClockCycles
    n = 0
    while int(dut.rt_state.value) != value:
        await ClockCycles(dut.clk, step)
        n += step
        if n > max_cycles:
            raise AssertionError("RT_STATE %#x not reached (now %#x)" % (value, int(dut.rt_state.value)))
    return n


def vec(sig, w, i):
    return (int(sig.value) >> (w * i)) & ((1 << w) - 1)


def run_dir():
    return Path(os.getcwd())


# ---------------------------------------------------------------- DMI helpers (DB §3.5, §3.6)
DMI_DATA0, DMI_DMCONTROL, DMI_DMSTATUS, DMI_HARTINFO, DMI_ABSTRACTCS, DMI_COMMAND = 0x04, 0x10, 0x11, 0x12, 0x16, 0x17
DMI_SBCS, DMI_SBADDRESS0, DMI_SBDATA0 = 0x38, 0x39, 0x3C


class DmiPort:
    """DMI accesses on the RTL (core-side DMI port of tsr_ctrl) or on the simulator's DM model, with
    one interface, so one sequence runs on both. `step` lets time pass between accesses."""

    def __init__(self, dut=None, dmsim=None):
        self.dut, self.sim = dut, dmsim
        self.timeouts = []

    async def rd(self, a):
        if self.sim is not None:
            self.sim.advance(20)
            return self.sim.dmi_read(a)
        d, _resp, to = await dmi(self.dut, 1, a)
        assert not to, "DMI read %#x timed out" % a
        return d

    async def wr(self, a, v):
        if self.sim is not None:
            self.sim.dmi_write(a, v)
            self.sim.advance(20)
            return
        _d, _resp, to = await dmi(self.dut, 2, a, v)
        assert not to, "DMI write %#x timed out" % a

    async def step(self, n):
        if self.sim is not None:
            self.sim.advance(n)
        else:
            from cocotb.triggers import ClockCycles
            await ClockCycles(self.dut.clk, n)

    async def poll(self, a, mask, want, tries=200):
        """Reads a until (value & mask) == want; returns the last value read (the comparison of the
        two sides then shows a condition that was never reached)."""
        v = 0
        for _ in range(tries):
            v = await self.rd(a)
            if v & mask == want:
                return v
            await self.step(50)
        self.timeouts.append("DMI %#x: %#x never reached %#x under %#x" % (a, v, want, mask))
        return v

    async def sba_read(self, addr, size=2):
        await self.wr(DMI_SBCS, (size << 17) | (1 << 20) | 0x7000 | (1 << 22))   # readonaddr, clear errors
        await self.wr(DMI_SBADDRESS0, addr)
        await self.step(40)
        v = await self.rd(DMI_SBDATA0)
        cs = await self.rd(DMI_SBCS)
        return v, cs

    async def sba_write(self, addr, val, size=2):
        await self.wr(DMI_SBCS, (size << 17) | 0x7000 | (1 << 22))
        await self.wr(DMI_SBADDRESS0, addr)
        await self.wr(DMI_SBDATA0, val)
        await self.step(40)
        return await self.rd(DMI_SBCS)


async def dmi_sequence(p: DmiPort, fw_mode_addr):
    """The DMI smoke of tb/ctrl (DB §3.4-§3.6): returns [(label, value)] of the reads compared
    between the RTL and the DM model, and the halted pc (dpc) for a range check."""
    out = []
    await p.wr(DMI_DMCONTROL, 0x1)                                   # dmactive
    out.append(("dmcontrol", await p.rd(DMI_DMCONTROL)))
    st = await p.poll(DMI_DMSTATUS, 0xC00, 0xC00)                     # allrunning, anyrunning
    out.append(("dmstatus running", st))
    out.append(("hartinfo", await p.rd(DMI_HARTINFO)))
    out.append(("abstractcs", await p.rd(DMI_ABSTRACTCS)))
    await p.wr(DMI_DMCONTROL, 0x80000001)                            # haltreq
    st = await p.poll(DMI_DMSTATUS, 0x300, 0x300)                     # allhalted, anyhalted
    out.append(("dmstatus halted", st))
    await p.wr(DMI_DMCONTROL, 0x1)
    out.append(("dmcontrol after halt", await p.rd(DMI_DMCONTROL)))
    regs = {}
    for name, regno in (("x5", 0x1005), ("x0", 0x1000), ("dcsr", 0x7B0), ("misa", 0x301), ("dpc", 0x7B1)):
        await p.wr(DMI_COMMAND, 0x00220000 | regno)                   # Access Register, 32-bit, transfer
        cs = await p.poll(DMI_ABSTRACTCS, 1 << 12, 0)
        regs[name] = await p.rd(DMI_DATA0)
        out.append(("abstractcs after %s" % name, cs))
        if name != "dpc":
            out.append((name, regs[name]))
    # write x7 and read it back
    await p.wr(DMI_DATA0, 0xFEEDBEEF)
    await p.wr(DMI_COMMAND, 0x00230000 | 0x1007)
    await p.poll(DMI_ABSTRACTCS, 1 << 12, 0)
    await p.wr(DMI_DATA0, 0)
    await p.wr(DMI_COMMAND, 0x00220000 | 0x1007)
    await p.poll(DMI_ABSTRACTCS, 1 << 12, 0)
    out.append(("x7 written", await p.rd(DMI_DATA0)))
    # an unsupported command (cmdtype 1): cmderr 2, then W1C
    await p.wr(DMI_COMMAND, 0x01000000)
    out.append(("abstractcs after cmdtype 1", await p.poll(DMI_ABSTRACTCS, 1 << 12, 0)))
    await p.wr(DMI_ABSTRACTCS, 0x700)
    out.append(("abstractcs after W1C", await p.rd(DMI_ABSTRACTCS)))
    # SBA (CC §4.3, CC-33)
    v, cs = await p.sba_read(fw_mode_addr)
    out += [("sba read fw_mode", v), ("sbcs after DTCM read", cs)]
    v, cs = await p.sba_read(0x20000034)                              # ISSUE_UNITS
    out += [("sba read ISSUE_UNITS", v), ("sbcs after window read", cs)]
    out.append(("sbcs after window write", await p.sba_write(0x20000000, 0x5A)))
    out.append(("sbcs after core-window write", await p.sba_write(0x20001000, 0x5A)))
    out.append(("sbcs after DM write", await p.sba_write(0x10000380, 0x5A)))
    out.append(("sbcs after DTCM write", await p.sba_write(0x00100200, 0x600DF00D)))
    v, cs = await p.sba_read(0x00100200)
    out += [("sba read back", v), ("sbcs", cs)]
    v, cs = await p.sba_read(0x00100201, size=0)                      # byte read
    out += [("sba byte read", v & 0xFF), ("sbcs after byte read", cs)]
    v, cs = await p.sba_read(0x00100202, size=2)                      # misaligned word: sberror 3
    out += [("sbcs after misaligned", cs)]
    v, cs = await p.sba_read(0x30000000)                              # unmapped
    out += [("sbcs after unmapped", cs)]
    # resume
    await p.wr(DMI_DMCONTROL, 0x40000001)
    st = await p.poll(DMI_DMSTATUS, (1 << 17) | (1 << 11), (1 << 17) | (1 << 11))
    out.append(("dmstatus resumed", st))
    await p.wr(DMI_DMCONTROL, 0x1)
    return out, regs["dpc"]

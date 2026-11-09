#!/usr/bin/env python3
"""rsp_client.py — a scripted GDB remote-serial-protocol client against tsr-dbg (docs/design/05-debug.md
"DB" §4). Python stdlib only.

Standalone (default): starts `tsr-dbg gdb --device sim --elf <fw_rt.elf> --halt-on-boot --port 0`, then
  qSupported, '?' and `monitor why` (halted at boot), the target description and memory map, g (pc and
  x1-x31), p (CSRs), QStartNoAckMode, m (ITCM against the ELF, __tsr_dbg), M / X (DTCM, read back; the
  control window refused with E04), Z0 on the firmware symbol tsr_rt_main, c, the hit (swbreak, pc),
  z0, s, Z1 on rt_copy with `monitor launch rt_copy ...` and `monitor trw start`, c (hwbreak), a tile
  breakpoint that halts the hart (`monitor tile break op ST halt`: the O note names the op and its
  push line), `monitor hist`, `monitor units`, `monitor tile step / delete / cont`, c (the launch
  answers in an O packet), Ctrl-C (T02), `monitor trw dump` (the JSON timeline), D, and the server's
  exit status.
Embedded (--embedded): the server is an owning process (test-tsr-dbg embed-host) with TSR_GDB=tcp:0
  and TSR_GDB_WAIT=launch: every launch halts at __tsr_launch_halt; the client sees three launches,
  then detaches and the owner finishes.
Drop (--drop, with or without --embedded): the client sends a request and vanishes (RST) before the
  reply: the server must end the session with the detach, never die of SIGPIPE; in the embedded form
  the owner must finish its launches and exit 0 (DB §4.1, §4.2).
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


# ---------------------------------------------------------------------------------------------
# ELF32 symbols and loaded bytes
# ---------------------------------------------------------------------------------------------
class Elf:
    def __init__(self, path):
        self.b = open(path, "rb").read()
        b = self.b
        assert b[:4] == b"\x7fELF" and b[4] == 1, "ELF32 expected"
        (e_phoff, e_shoff) = struct.unpack_from("<II", b, 0x1C)
        (e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from("<HHHHH", b, 0x2A)
        self.segs = []
        for i in range(e_phnum):
            p_type, p_off, p_vaddr, p_paddr, p_filesz = struct.unpack_from("<IIIII", b, e_phoff + i * e_phentsize)
            if p_type == 1:
                self.segs.append((p_paddr, p_off, p_filesz))
        secs = [struct.unpack_from("<IIIIIIIIII", b, e_shoff + i * e_shentsize) for i in range(e_shnum)]
        self.syms = {}
        for s in secs:
            if s[1] == 2:  # SHT_SYMTAB
                strtab = secs[s[6]]
                for k in range(s[5] // 16):
                    name, value, size, info, other, shndx = struct.unpack_from("<IIIBBH", b, s[4] + 16 * k)
                    n = b[strtab[4] + name:b.index(b"\0", strtab[4] + name)].decode()
                    if n:
                        self.syms[n] = value

    def sym(self, n):
        return self.syms[n]

    def bytes_at(self, a, n):
        for (pa, off, sz) in self.segs:
            if pa <= a and a + n <= pa + sz:
                return self.b[off + a - pa:off + a - pa + n]
        return None


# ---------------------------------------------------------------------------------------------
# RSP
# ---------------------------------------------------------------------------------------------
class Rsp:
    def __init__(self, sock):
        self.s = sock
        self.buf = b""
        self.ack = True
        self.outputs = []  # O packets seen since the last call (decoded)

    def _read(self, timeout):
        self.s.settimeout(timeout)
        d = self.s.recv(65536)
        if not d:
            raise EOFError("connection closed")
        self.buf += d

    def send(self, payload):
        pkt = b"$" + payload.encode("latin1") + b"#%02x" % (sum(payload.encode("latin1")) & 0xFF)
        self.s.sendall(pkt)
        if self.ack:
            while b"+" not in self.buf[:1]:
                if self.buf[:1] == b"-":
                    self.buf = self.buf[1:]
                    self.s.sendall(pkt)
                    continue
                if not self.buf:
                    self._read(60)
                else:
                    break
            if self.buf[:1] == b"+":
                self.buf = self.buf[1:]

    def recv(self, timeout=120):
        """The next packet that is not an O packet; O packets are collected in self.outputs."""
        deadline = time.time() + timeout
        while True:
            i = self.buf.find(b"$")
            j = self.buf.find(b"#", i + 1) if i >= 0 else -1
            if i >= 0 and j >= 0 and len(self.buf) >= j + 3:
                body = self.buf[i + 1:j].decode("latin1")
                cs = int(self.buf[j + 1:j + 3], 16)
                self.buf = self.buf[j + 3:]
                assert cs == sum(body.encode("latin1")) & 0xFF, "checksum"
                if self.ack:
                    self.s.sendall(b"+")
                if body.startswith("O") and body != "OK" and all(c in "0123456789abcdef" for c in body[1:]):
                    self.outputs.append(bytes.fromhex(body[1:]).decode("utf-8", "replace"))
                    continue
                return body
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError("no reply")
            self._read(left)

    def cmd(self, payload, timeout=120):
        self.outputs = []
        self.send(payload)
        return self.recv(timeout)

    def monitor(self, line, timeout=120):
        r = self.cmd("qRcmd," + line.encode().hex(), timeout)
        return r, "".join(self.outputs)

    def interrupt(self):
        self.outputs = []
        self.s.sendall(b"\x03")


def le32(h):
    return struct.unpack("<I", bytes.fromhex(h))[0]


def hex32(v):
    return struct.pack("<I", v).hex()


def stop_regs(t):
    """{regnum: value} of a T packet's expedited registers, and the reason."""
    regs, reason = {}, ""
    for part in t[3:].split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        if k in ("swbreak", "hwbreak", "watch", "rwatch", "awatch"):
            reason = k
        elif k != "thread" and all(c in "0123456789abcdef" for c in k):
            regs[int(k, 16)] = le32(v)
    return regs, reason


def kind_of(elf, a):
    return 2 if (elf.bytes_at(a, 1)[0] & 3) != 3 else 4


def connect(port_file, proc, timeout=60):
    t0 = time.time()
    while not os.path.exists(port_file):
        if proc.poll() is not None:
            raise RuntimeError("the server exited early (status %s)" % proc.returncode)
        if time.time() - t0 > timeout:
            raise TimeoutError("no port file")
        time.sleep(0.05)
    port = int(open(port_file).read())
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


# ---------------------------------------------------------------------------------------------
# Standalone session
# ---------------------------------------------------------------------------------------------
def standalone(args, elf):
    pf = os.path.join(args.scratch, "port")
    if os.path.exists(pf):
        os.unlink(pf)
    log = open(os.path.join(args.scratch, "server.log"), "w")
    proc = subprocess.Popen([args.server, "gdb", "--device", "sim", "--elf", args.elf, "--halt-on-boot", "--port", "0", "--port-file", pf],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        g = Rsp(connect(pf, proc))
        r = g.cmd("qSupported:multiprocess+;swbreak+;hwbreak+;qRelocInsn+;xmlRegisters=i386")
        check("PacketSize=" in r and "swbreak+" in r and "qXfer:features:read+" in r and "QStartNoAckMode+" in r, "qSupported: " + r)
        r = g.cmd("?")
        regs, _ = stop_regs(r)
        check(r.startswith("T05") and regs.get(32) == 0x80, "'?' after attach: T05 at 0x80 (halt-on-boot): " + r)
        r, o = g.monitor("why")
        check(r == "OK" and "halted at boot" in o, "monitor why: " + o.strip())
        x = g.cmd("qXfer:features:read:target.xml:0,ffff")
        check(x.startswith("l") and "riscv:rv32" in x and "org.gnu.gdb.riscv.csr" in x and 'regnum="833"' in x, "target.xml (DB §4.4)")
        x = g.cmd("qXfer:memory-map:read::0,ffff")
        check(x.startswith("l") and "0x00100000" in x and "0x40000000" in x, "memory map (DB §4.5)")
        r = g.cmd("g")
        check(len(r) == 33 * 8, "g: 33 registers (%d hex digits)" % len(r))
        gpr = [le32(r[8 * i:8 * i + 8]) for i in range(33)]
        check(gpr[0] == 0 and gpr[32] == 0x80, "x0 = 0, pc = 0x%08x" % gpr[32])
        print("     x1-x31: " + " ".join("%08x" % v for v in gpr[1:32]))
        misa = le32(g.cmd("p%x" % (65 + 0x301)))
        check((misa >> 30) == 1 and misa & (1 << 2) and misa & (1 << 8) and misa & (1 << 12), "p misa 0x%08x: RV32IMC" % misa)
        ms = g.cmd("p%x" % (65 + 0x300))
        check(len(ms) == 8, "p mstatus " + ms)
        check(g.cmd("p%x" % (65 + 0x7B0)) == "E04", "p dcsr refused (the server's, DB §4.4)")
        check(g.cmd("P5=%s" % hex32(0xCAFEF00D)) == "OK" and le32(g.cmd("p5")) == 0xCAFEF00D, "P / p t0")
        # Ctrl-C while halted (a breakpoint hit racing the user's Ctrl-C) is ignored: no second stop reply
        # may stand in for the answer to the next request.
        g.interrupt()
        time.sleep(0.3)
        r1 = g.cmd("p20")
        r2 = g.cmd("p20")
        check(r1 == "80000000" and r2 == "80000000", "Ctrl-C while halted is ignored: the next replies are p20 answers (%s, %s)" % (r1[:20], r2[:20]))
        check(len(g.cmd("g")) == 33 * 8, "g after the Ctrl-C")
        # An ack that trails the client's next packet is still taken as the ack (no 5 s retransmission).
        raw = g.s
        raw.sendall(b"$?#3f")
        buf = b""
        raw.settimeout(5)
        while b"#" not in buf or len(buf) < buf.index(b"#") + 3:
            buf += raw.recv(4096)
        check(buf.startswith(b"+$T05"), "'?' answered (raw): " + buf[:12].decode("latin1"))
        t0 = time.time()
        raw.sendall(b"$qC#b4+")   # the next request first, the ack of the T packet behind it
        buf = b""
        while buf.count(b"$") < 1 or b"QC1#c5" not in buf:
            buf += raw.recv(4096)
        raw.settimeout(1.0)
        raw.sendall(b"+")
        try:
            extra = raw.recv(4096)
        except socket.timeout:
            extra = b""
        check(time.time() - t0 < 4.0 and buf.count(b"$T05") == 0 and extra == b"",
              "a trailing ack is accepted: no retransmission (%.2f s, %r)" % (time.time() - t0, (buf + extra)[:40]))
        raw.settimeout(30)
        check(g.cmd("QStartNoAckMode") == "OK", "QStartNoAckMode")
        g.ack = False

        a_main, a_copy, a_dbg = elf.sym("tsr_rt_main"), elf.sym("rt_copy"), elf.sym("__tsr_dbg")
        m = g.cmd("m%x,10" % a_main)
        check(bytes.fromhex(m) == elf.bytes_at(a_main, 16), "m ITCM at tsr_rt_main = the ELF")
        m = bytes.fromhex(g.cmd("m%x,10" % a_dbg))
        check(m[:4] == b"TDBG" and struct.unpack_from("<I", m, 12)[0] == 1, "m __tsr_dbg: magic TDBG, attached 1")
        check(g.cmd("M%x,4:11223344" % (a_dbg + 32)) == "OK" and g.cmd("m%x,4" % (a_dbg + 32)) == "11223344", "M / m DTCM")
        xb = bytes([0x7D, 0x23, 0x24, 0x2A])  # escaped in X
        esc = "".join("}" + chr(c ^ 0x20) if c in (0x7D, 0x23, 0x24, 0x2A) else chr(c) for c in xb)
        check(g.cmd("X%x,4:%s" % (a_dbg + 36, esc)) == "OK" and bytes.fromhex(g.cmd("m%x,4" % (a_dbg + 36))) == xb, "X binary write")
        check(g.cmd("M20000100,4:01000000") == "E04", "control-window write refused (D15)")
        check(g.cmd("m10000000,4") == "E04", "the DM range refused")

        # A software breakpoint on a firmware symbol, continue, hit.
        k = kind_of(elf, a_main)
        check(g.cmd("Z0,%x,%d" % (a_main, k)) == "OK", "Z0 tsr_rt_main (kind %d)" % k)
        t = g.cmd("c")
        regs, why = stop_regs(t)
        check(t.startswith("T05") and why == "swbreak" and regs.get(32) == a_main, "c: stopped at tsr_rt_main (swbreak): " + t)
        check(regs.get(2, 0) >= 0x00100000, "expedited sp 0x%08x in DTCM" % regs.get(2, 0))
        check(g.cmd("z0,%x,%d" % (a_main, k)) == "OK", "z0")
        t = g.cmd("s")
        regs, _ = stop_regs(t)
        check(t.startswith("T05") and regs.get(32, a_main) != a_main, "s: pc 0x%08x" % regs.get(32, 0))

        # A hardware breakpoint on a kernel; the launch and the trace writer from monitor commands.
        check(g.cmd("Z1,%x,2" % a_copy) == "OK", "Z1 rt_copy")
        r, o = g.monitor("trw start 1")
        check(r == "OK" and "trace writer on" in o, "monitor trw start: " + o.strip())
        r, o = g.monitor("launch rt_copy 0x100000 0x100100 2 0")
        check(r == "OK" and ("pushed as seq" in o or "queued" in o), "monitor launch (queued until HELLO): " + o.strip())
        t = g.cmd("c")
        regs, why = stop_regs(t)
        check(t.startswith("T05") and why == "hwbreak" and regs.get(32) == a_copy, "c: hwbreak at rt_copy: " + t)
        check(g.cmd("z1,%x,2" % a_copy) == "OK", "z1")

        # Tile level (DB §5): a breakpoint on the ST that halts the hart, the history, the probes, QSTEP.
        r, o = g.monitor("tile break op ST halt")
        check(r == "OK", "monitor tile break op ST halt: " + o.strip())
        t = g.cmd("c")
        note = "".join(g.outputs)
        check(t.startswith("T05") and "tile stop at pidx" in note and "ST ch0" in note and "in rt_copy" in note, "tile stop: " + note.strip())
        r, o = g.monitor("hist 2")
        check(r == "OK" and "LD ch0" in o and "ST ch0" in o and "queued" in o, "monitor hist:\n" + o)
        r, o = g.monitor("units")
        check(r == "OK" and "DMA" in o and "held by" in o, "monitor units (QSTATUS)")
        r, o = g.monitor("tile step")
        check(r == "OK" and "tile step" in o, "monitor tile step: " + o.strip())
        check(g.monitor("tile delete 0")[0] == "OK" and g.monitor("tile cont")[0] == "OK", "tile delete / cont")
        r, o = g.monitor("bogus command")
        check(r.startswith("E") and "unknown monitor command" in o, "an unknown monitor command: " + r)

        # Continue: the launch answers (O packet); the hart idles; Ctrl-C.
        g.outputs = []
        g.send("c")
        t0 = time.time()
        while time.time() - t0 < 60 and "answered" not in "".join(g.outputs):
            try:
                g.s.settimeout(0.2)
                g.buf += g.s.recv(65536)
                while b"$" in g.buf and b"#" in g.buf[g.buf.find(b"$"):]:
                    i = g.buf.find(b"$")
                    j = g.buf.find(b"#", i)
                    if len(g.buf) < j + 3:
                        break
                    body = g.buf[i + 1:j].decode("latin1")
                    g.buf = g.buf[j + 3:]
                    if body.startswith("O"):
                        g.outputs.append(bytes.fromhex(body[1:]).decode())
                    else:
                        check(False, "unexpected packet while running: " + body)
            except socket.timeout:
                pass
        check("launch seq" in "".join(g.outputs) and "answered: OK" in "".join(g.outputs), "the launch answered: " + "".join(g.outputs).strip())
        g.interrupt()
        t = g.recv(60)
        check(t.startswith("T02"), "Ctrl-C: T02: " + t)
        js = os.path.join(args.scratch, "rsp_trw.json")
        r, o = g.monitor("trw dump " + js)
        check(r == "OK" and os.path.exists(js), "monitor trw dump: " + o.strip())
        if os.path.exists(js):
            ev = json.load(open(js))["traceEvents"]
            names = [e.get("name") for e in ev if e.get("ph") == "X"]
            check("LD" in names and "ST" in names, "the timeline has LD and ST slices (%d events)" % len(ev))
        r, o = g.monitor("status")
        check(r == "OK" and "verdict (DB §7.2)" in o and "attached" in o, "monitor status")
        check(g.cmd("D") == "OK", "D")
        g.s.close()
        rc = proc.wait(60)
        check(rc == 0, "the server exits 0 after the detach (status %s)" % rc)
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
    print(open(os.path.join(args.scratch, "server.log")).read())


# ---------------------------------------------------------------------------------------------
# Embedded session (TSR_GDB, TSR_GDB_WAIT=launch)
# ---------------------------------------------------------------------------------------------
def embedded(args, elf):
    pf = os.path.join(args.scratch, "port")
    if os.path.exists(pf):
        os.unlink(pf)
    env = dict(os.environ, TSR_GDB="tcp:0", TSR_GDB_WAIT="launch", TSR_GDB_PORT_FILE=pf)
    log = open(os.path.join(args.scratch, "host.log"), "w")
    proc = subprocess.Popen([args.server, "embed-host", args.elf, "3"], stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        g = Rsp(connect(pf, proc))
        g.cmd("qSupported:swbreak+")
        lh = elf.sym("__tsr_launch_halt")
        t = g.cmd("?")
        regs, _ = stop_regs(t)
        if regs.get(32) != lh:  # attached before the first launch was taken: continue to it
            t = g.cmd("c")
            regs, _ = stop_regs(t)
        r, o = g.monitor("why")
        check(regs.get(32) == lh and "launch rt_nop seq" in o, "the first launch halts at __tsr_launch_halt: " + o.strip())
        seqs = [o]
        for _ in range(2):
            t = g.cmd("c")
            regs, _ = stop_regs(t)
            seqs.append("".join(g.outputs))
            check(t.startswith("T05") and regs.get(32) == lh, "the next launch halts too: " + "".join(g.outputs).strip())
        r = g.cmd("g")
        check(len(r) == 33 * 8, "g while the owner waits")
        check(g.cmd("D") == "OK", "D")
        g.s.close()
        rc = proc.wait(120)
        check(rc == 0, "the owner finishes its launches after the detach (status %s)" % rc)
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
    txt = open(os.path.join(args.scratch, "host.log")).read()
    print(txt)
    check("3 launches, 0 failed" in txt, "the owner's three launches answered OK")


def drop_once(args, mode):
    """One session in which the client vanishes with a request outstanding. mode "rst": RST at once
    (the first write fails with ECONNRESET, a second would raise SIGPIPE). mode "fin-noack": after
    QStartNoAckMode (what GDB does) the client closes normally; the first reply write succeeds, the
    peer answers it with an RST and the next write fails with EPIPE (SIGPIPE without MSG_NOSIGNAL)."""
    pf = os.path.join(args.scratch, "port")
    if os.path.exists(pf):
        os.unlink(pf)
    log = open(os.path.join(args.scratch, "drop_%s.log" % mode), "w")
    if args.embedded:
        env = dict(os.environ, TSR_GDB="tcp:0", TSR_GDB_WAIT="launch", TSR_GDB_PORT_FILE=pf)
        cmd = [args.server, "embed-host", args.elf, "3"]
    else:
        env = dict(os.environ)
        cmd = [args.server, "gdb", "--device", "sim", "--elf", args.elf, "--halt-on-boot", "--port", "0", "--port-file", pf]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    who = "owner process" if args.embedded else "server"
    try:
        s = connect(pf, proc)
        if mode == "fin-noack":
            g = Rsp(s)
            g.cmd("qSupported:swbreak+")
            check(g.cmd("QStartNoAckMode") == "OK", "QStartNoAckMode before the drop")
        body = b"qRcmd," + b"hist 64".hex().encode()
        s.sendall(b"$" + body + b"#%02x" % (sum(body) & 0xFF))
        if mode == "rst":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))   # RST on close
        s.close()
        try:
            rc = proc.wait(180)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = "timeout"
        check(rc == 0, "[%s] the %s exits 0 after the connection dropped mid-reply (status %s; -13 = SIGPIPE)" % (mode, who, rc))
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
    txt = open(os.path.join(args.scratch, "drop_%s.log" % mode)).read()
    print(txt)
    check("GDB session ended" in txt, "[%s] the session ended through the detach path" % mode)
    if args.embedded:
        check("3 launches, 0 failed" in txt, "[%s] the owner's three launches answered OK after the drop" % mode)


def drop(args, elf):
    """The client vanishes with a request outstanding: the server ends the session with the detach and
    exits 0; in the embedded form the owner finishes its launches (DB §4.1, §4.2)."""
    for mode in ("rst", "fin-noack"):
        drop_once(args, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--elf", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--embedded", action="store_true")
    ap.add_argument("--drop", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    elf = Elf(args.elf)
    try:
        (drop if args.drop else embedded if args.embedded else standalone)(args, elf)
    except Exception as e:  # noqa: BLE001 - report every failure as a test failure
        check(False, "exception: %r" % e)
    print("%s: %d failure(s)" % ("FAIL" if FAILS else "PASS", len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

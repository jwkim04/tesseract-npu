"""Driver for the core's CSR port (spec 코어 경계 인터페이스; docs/rtl-m2.md §6).

Port: csr_valid, csr_ready, csr_we, csr_addr[11:0], csr_wdata[31:0], csr_rdata[31:0].
Protocol: the master holds valid/addr/we/wdata until it sees ready; ready is a one-cycle pulse and
rdata is valid in that cycle; the transfer completes at the rising edge where valid && ready.
This driver drives right after a rising edge and samples at the falling edge (the same discipline
as mem_model.MemModel). Accesses are serialised with a lock, so several coroutines may share it.
"""
from __future__ import annotations

from cocotb.triggers import FallingEdge, Lock, RisingEdge

from . import isa


class CsrTimeout(TimeoutError):
    pass


class CsrDriver:
    def __init__(self, dut, clk=None, prefix: str = "csr_", timeout_cycles: int = 1000):
        self.dut = dut
        self.clk = clk if clk is not None else dut.clk
        self.timeout = timeout_cycles
        self.valid = getattr(dut, prefix + "valid")
        self.ready = getattr(dut, prefix + "ready")
        self.we = getattr(dut, prefix + "we")
        self.addr = getattr(dut, prefix + "addr")
        self.wdata = getattr(dut, prefix + "wdata")
        self.rdata = getattr(dut, prefix + "rdata")
        self._lock = Lock()
        self.valid.value = 0
        self.we.value = 0
        self.addr.value = 0
        self.wdata.value = 0

    async def _xfer(self, addr: int, we: int, wdata: int) -> int:
        async with self._lock:
            await RisingEdge(self.clk)
            self.addr.value = addr
            self.we.value = we
            self.wdata.value = wdata & 0xFFFFFFFF
            self.valid.value = 1
            for _ in range(self.timeout):
                await FallingEdge(self.clk)
                if int(self.ready.value):
                    rdata = int(self.rdata.value)
                    await RisingEdge(self.clk)      # handshake edge
                    self.valid.value = 0
                    self.we.value = 0
                    return rdata
            self.valid.value = 0
            raise CsrTimeout("CSR %s 0x%03x: no csr_ready within %d cycles"
                             % ("write" if we else "read", addr, self.timeout))

    async def write(self, addr: int, data: int):
        await self._xfer(addr, 1, data)

    async def read(self, addr: int) -> int:
        return await self._xfer(addr, 0, 0)

    async def read64(self, addr_lo: int) -> int:
        """64-bit counter: LO first (the core snapshots HI on the LO read)."""
        lo = await self.read(addr_lo)
        hi = await self.read(addr_lo + 4)
        return (hi << 32) | lo

    # ---- convenience ----
    async def start_run(self, cmd_addr_beat: int, cmd_count: int, args=None, ctrl: int = isa.CTRL_START):
        await self.write(isa.CSR_CMD_ADDR_LO, cmd_addr_beat & 0xFFFFFFFF)
        await self.write(isa.CSR_CMD_ADDR_HI, (cmd_addr_beat >> 32) & 0xFFFFFFFF)
        await self.write(isa.CSR_CMD_COUNT, cmd_count & 0xFFFFFFFF)
        args = list(args or []) + [0] * 8
        for i in range(8):
            await self.write(isa.CSR_ARG0 + 4 * i, args[i])
        await self.write(isa.CSR_CTRL, ctrl)

    async def final_state(self) -> dict:
        return dict(status=await self.read(isa.CSR_STATUS), err_code=await self.read(isa.CSR_ERR_CODE),
                    err_pc=await self.read(isa.CSR_ERR_PC), err_info=await self.read(isa.CSR_ERR_INFO),
                    pc=await self.read(isa.CSR_PC))

    async def perf(self, num_ch: int) -> list:
        return [await self.read64(isa.perf_csr(i)) for i in range(isa.perf_count(num_ch))]

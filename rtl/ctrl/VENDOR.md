# Vendored IP of the RISC-V control subsystem

The control subsystem (`rtl/ctrl`, docs/design/04-control-core.md "CC" §2 to §5) builds three third-party trees.
All three are Git submodules, used unmodified (CC §3.1). This file holds what CC §3.1 asks for: the pinned
commit, why it was chosen, the Verilator build check and the verification grade. CC §3.1 names
`third_party/ibex/VENDOR.md` for it, but a file there would modify the submodule, so it is here instead.

| Tree | Upstream | Pinned commit | Why this commit |
|---|---|---|---|
| `third_party/ibex` | https://github.com/lowRISC/ibex | `7a4697f578c67c550fcad5e5e3078415e1f088d7` | The commit that OpenTitan's `hw/vendor/lowrisc_ibex.lock.hjson` records ("Update lowrisc_ibex to lowRISC/ibex@7a4697f5"). It is recent enough for the age rule of CC §3.1 (at most 6 months), and its GitHub checks are green: CLA, verible lint, "Run quality checks (Lint and DV)" and the private CI trigger |
| `third_party/riscv-dbg` | https://github.com/pulp-platform/riscv-dbg | `deed17e18c0d61ce8dc554300bb4dd5c39be2a13` (tag `v0.10.1`) | CC §3.1 asks for the latest release tag, which is v0.10.1. OpenTitan's `pulp_riscv_dbg.lock.hjson` records v0.9.0 (`2bc55e1e`), which is older than the age rule allows. master has later commits (cmderr and hartsel compliance fixes) but no release |
| `third_party/common_cells` | https://github.com/pulp-platform/common_cells | `1281545696eb3fcba50ec5b4275993476a3c710e` (tag `v1.40.0`) | `dm_csrs` needs `fifo_v3`. This is the latest 1.x release that satisfies riscv-dbg's Bender.yml (`common_cells` 1.24.0, that is ^1.24). Only `src/fifo_v3.sv` is built |

## What the build uses

`rtl/ctrl/filelist.f` lists every file:

- Ibex. The `ibex_top.core`, `ibex_core.core` and `ibex_pkg.core` file sets. From `vendor/lowrisc_ip` the
  generic primitives `prim_buf`, `prim_flop`, `prim_and2` and `prim_clock_mux2`, then `prim_count`, `prim_lfsr`
  and `prim_fifo_sync`, the packages `prim_util`, `prim_mubi`, `prim_secded`, `prim_count`, `prim_cipher` and
  `prim_ram_1p`, and the `prim_assert` macros. From `vendor/pulp_common_cells`, `stream_fork` and
  `stream_join_dynamic` (SHL-0.51, used by `ibex_trvk`, which is not elaborated with CHERIoT off).
  `prim_clock_gating` is bound to `rtl/ctrl/prim/prim_clock_gating.sv`, a pass-through (CC §3.1). The generic
  latch-based gate is not built.
- riscv-dbg. `dm_pkg`, `debug_rom` (the two-scratch ROM, because `DmBaseAddress` is not 0),
  `debug_rom_one_scratch`, `dm_csrs`, `dm_mem`, `dm_sba` and `dm_top`. The JTAG DTM (`dmi_jtag*`, `dmi_cdc`) is
  not built (CC §3.1).
- common_cells. `src/fifo_v3.sv`, read with `+define+COMMON_CELLS_ASSERTIONS_SVH` and
  `+define+COMMON_CELLS_ASSERTS_OFF`. Its assertion header would otherwise redefine lowRISC's `ASSERT*` macros
  for every file read after it, which would leave `fifo_v3` without its three assertions.

## Verilator build check

`rtl/ctrl/lint.sh` runs `--lint-only -Wall` on every `rtl/ctrl` module and on `tsr_ctrl` with RVFI, in jitter
mode, in the small and `RV32MSingleCycle` configurations, with the flop register file, `CTRL_SYNC` 1, small
TCMs, a reduced unit set and the Xilinx SRAM wrapper. The waiver file comes before the sources
(`rtl/ctrl/waivers/vendor.vlt`, listed first in `rtl/filelist.f`). It covers the trees under `third_party/`
only, plus `UNOPTFLAT` on Ibex's ID-stage loop. Ibex's own `lint/verilator_waiver.vlt` is not read, because it
switches `PINCONNECTEMPTY` off for the whole design, which would also hide the rule in Tesseract's RTL.
Tesseract's own files need no file waiver. The inline waivers are `SYNCASYNCNET` on `tsr_ctrl.rst_n` and
`tsr_core.rst_n`, `PINCONNECTEMPTY` around the `ibex_top` instance in `tsr_ctrl_rv.sv` (Ibex's unused output
ports), `DECLFILENAME` for the second module of `tsr_issue.sv` and the package `UNUSEDPARAM` waivers. The
`tb/ctrl` cocotb build (Verilator `--public-flat-rw`, `RVFI` defined) runs `tb/ctrl/test_ctrl.py`.

## Configuration and its verification grade

The Ibex README's table rates "maxperf" (RV32IMC with the 1-cycle multiplier, branch target ALU and writeback
stage) Green, and "small" Green too. Tesseract deviates from maxperf in five places (CC §3.2, CC-34):

- `RV32M` is `RV32MFast` instead of `RV32MSingleCycle`. This is the first fallback of CC §3.2: maxperf's
  single-cycle multiplier limits `clk_rv` at the 166.7 MHz planning frequency. `rtl/ctrl/viv/ctrl_ooc.sh mfast`
  builds this configuration and `maxperf` the original one.
- `RegFileFPGA` instead of `RegFileFF`.
- `DbgTriggerEn` 1 with 2 triggers instead of 0.
- `RV32ZC` is `RV32Zca` instead of `RV32ZcaZcbZcmp`, so the ISA stays RV32IMC.
- `BaseIsa` is `BaseIsaRV32I`, with `cheriot_enable_i` off. The pinned commit is the CHERIoT-merged Ibex.

## Facts checked on the pinned commits (CC §3.1 and §3.2)

- `ibex_top` parameters. `RV32ZC` exists. `DmBaseAddr` and `DmAddrMask` are set to 0x1000_0000 and 0xFFF. New
  ports are tied off: `cheriot_enable_i`, `trvk_*`, `data_tag_*`, `mcounteren_writable_i` (off, so mcounteren
  reads 0 as in `tsr_rv`), `scan_rst_ni` and the `ram_cfg_*` ports. `crash_dump_o.current_pc` is `pc_id`, which
  is the store's own pc in the cycle its data request is granted (the push PC of CC §7.1).
- riscv-dbg `dm_top`.
  - It has `ndmreset_ack_i`, which is pulsed whenever the hart leaves a reset (`havereset`).
  - It exports no halted level, so `RV_HALTED` is derived from the fetch address (CC-29). `RV_HALTED` matches
    `dmstatus.allhalted` in the dmi test of `tb/ctrl`.
  - `hartinfo` is 0x0021_2380, and `DmHaltAddr` and `DmExceptionAddr` are 0x1000_0800 and 0x1000_0808.
  - The debug ROM reads its per-hart flags with `lbu`, so the DM range takes byte and half accesses from the
    hart in debug mode (`tsr_rv_bus.sv`, CC §4.1).
  - The program buffer, the abstract-command buffer and the ROM all lie in 0x1000_0000 to 0x1000_0FFF, so the
    `RV_HALTED` heuristic holds.
- riscv-dbg SBA. A bus error answer gives `sberror` 2 and a misaligned access gives 3. Sub-word reads are
  returned in the low bits, as the simulator's DM model does. Every DMI read of the `tb/ctrl` sequence equals
  `tsr_dm.c`'s.

# Changelog

Changes to Tesseract Core, the RTL in `rtl/` and the Alveo U200 image that `hw/u200` builds from it.
The format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is described in
[docs/versioning.md](../docs/versioning.md).

## [Unreleased]

## [1.1.0] - 2026-11-12

Core capability 1.0. The image is `tsr_u200_k4.xclbin` for the platform `xilinx_u200_gen3x16_xdma_2_202110_1`,
with the core clock at 200 MHz, the control core at 100 MHz and `CAPS` 0x10121 (VU, ATT, GDN, MVM_DUAL).

| Item | Value |
|---|---|
| sha256 | `628559646594efde3dc04e40ef79db01f4bc2659d6c5242d3b80ac1b819eb9cc` |
| UUID | `c2920184-1184-3844-f252-4c1789b48b2a` |

The image needs Tesseract Basis 1.2.0 or later, because an older Basis refuses the kernel's `chb<c>` arguments.

### Added

- The dual weight stream of the MVM unit (`MVM_DUAL`, [docs/rtl-mvm.md](../docs/rtl-mvm.md) §15). A cluster reads its rows
  as two weight streams at once, in lockstep, each through its own front end, and the gather has a second row assembler
  per cluster. It applies to a weight MVM with T ≤ `T_MAX` / 2. Results and the memory layout do not change, only timing.
- A read-only second memory port per channel on the core and on the kernel (`m_axi_chb<c>`, with the arguments `chb<c>`),
  each with its own read-only AXI4 bridge. `PERF_CH_RD_BEATS` counts the beats of both ports.
- `CAPS` bit 16 `MVM_DUAL`, and `ADAPTER_CTRL` bit 1 `DUAL_DIS`, which turns the dual weight stream off at run time.
  `CAPS` does not change with it.
- Benches for the dual weight stream: the MVM unit, its back end and the MAC lanes, the core against the simulator, the
  read-only bridge and the kernel.
- A bandwidth link with two compute units on every DDR bank (`hw/u200/bw/bw2x200.cfg`, `hw/u200/bw/measure_bw2.sh`, and
  `TSR_BW_CFG` and `TSR_BW_OUT` of `build_bw_xclbin.sh`).

### Changed

- The Pblock plan of the K4 image. The AXI bridges of channels 1 and 2 with their read-only bridges and the central
  gather receiver are in the SLR1 Pblock. The row assemblers of stream B are in the Pblock of their cluster.
- The link hook clears the `DONT_PARTITION` mark that the link flow sets on the kernel, before placement.

## [1.0.0] - 2026-11-10

Core capability 1.0. The image is `tsr_u200_k4.xclbin` for the platform `xilinx_u200_gen3x16_xdma_2_202110_1`,
with the core clock at 200 MHz, the control core at 100 MHz and `CAPS` 0x121 (VU, ATT, GDN).

| Item | Value |
|---|---|
| sha256 | `944ea2b59a436c7e7a60b9915d499e6a501cbefd1673c5ac2ca6fdd6d0f7a5f9` |
| UUID | `7f5e7240-e3e6-de7a-55aa-42fc777c128d` |

### Added

- Four MVM clusters of 128 MAC lanes, each streaming Q8_0 or Q4_K weights from its own DDR4 channel.
- The 4 MiB scratchpad, the DMA with KV cache stores, and QACT.
- The vector unit, with the attention modes and the fused Gated DeltaNet op.
- The RISC-V control core (Ibex RV32IMC) with its ITCM and DTCM, the issue path with one queue per unit, and the
  debug module.
- The XRT platform adapter and the U200 kernel tops.

[Unreleased]: https://github.com/jwkim04/tesseract-npu/compare/core-v1.1.0...HEAD
[1.1.0]: https://github.com/jwkim04/tesseract-npu/compare/core-v1.0.0...core-v1.1.0
[1.0.0]: https://github.com/jwkim04/tesseract-npu/releases/tag/core-v1.0.0

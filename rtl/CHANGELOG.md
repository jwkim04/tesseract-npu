# Changelog

Changes to Tesseract Core, the RTL in `rtl/` and the Alveo U200 image that `hw/u200` builds from it.
The format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is described in
[docs/versioning.md](../docs/versioning.md).

## [Unreleased]

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

[Unreleased]: https://github.com/jwkim04/tesseract-npu/compare/core-v1.0.0...HEAD
[1.0.0]: https://github.com/jwkim04/tesseract-npu/releases/tag/core-v1.0.0

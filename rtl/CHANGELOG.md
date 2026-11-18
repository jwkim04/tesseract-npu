# Changelog

Changes to Tesseract Core, the RTL in `rtl/` and the Alveo U200 image that `hw/u200` builds from it.
The format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is described in
[docs/versioning.md](../docs/versioning.md).

## [Unreleased]

## [1.2.1] - 2026-11-18

Core capability 1.0. The image is `tsr_u200_k4.xclbin` for the platform `xilinx_u200_gen3x16_xdma_2_202110_1`,
with the core clock at 200 MHz, the control core at 100 MHz and `CAPS` 0x30121 (VU, ATT, GDN, MVM_DUAL, QACT_PIPE).

| Item | Value |
|---|---|
| sha256 | `7f1223a6dd51b3c3adebb187cf85c7b5f22b05afef521f7d34fa067956b1a2d4` |
| UUID | `b35a35ad-1a16-d76e-ac75-d7ad01ea158f` |

The image needs Tesseract Basis 1.2.0 or later, as Core 1.2.0 does. It computes the same results as the Core 1.2.0
image, and `CAPS` is the same.

### Changed

- Structure changes for the wiring of the K4 image. None changes a result bit. None changes a cycle count of a run
  without the trace writer. While the trace writer writes, its bursts and the DMA's bursts on channel 0 reach the
  arbiter in a different order than before, so the cycle counts of a traced run can differ by a fraction of a percent.
  - The DMA read response of channels 0 and 3 (`DDR_LOCAL_PIPE`) is one valid and payload register (`tsr_vpipe`) without a
    skid entry: the DMA's read response ready is constant 1.
  - The VU command link (`XSLR_REG`) is a delay line of three register stages without a ready path, because the VU takes every
    command that reaches it. With `LOCAL_INTERMEDIATES` or `LOCAL_SCALE` it stays a valid and ready pipe.
  - The trace writer's write mux on channel 0 sits behind the DMA's locality pipe slices, on the arbiter's side.
  - The pair link transmitter (`tsr_mvm_pair_tx`, `RPC`) has pointer copies in its queue and one kept flit select copy per
    slice of the flit, with the payload permuted to match.
  - `MVM_BC_HEADS` (`tsr_mvm`, `tsr_core_dp`, `tsr_core`, `tsr_xrt_kernel`, default 1) makes `BC_HEADS` kept copies of the first
    broadcast stage. Cluster `c` reads copy `c` mod `BC_HEADS`. `tsr_u200_k4` sets 2.
  - The MVM request line register loads from a register that holds the next line, and `cfg_ok` of the GDN engine is a register.
  - The VU's write link carries a 16-bit word mask to `tsr_core_dp`, which expands it to the SPM client port's byte enables.
  - The accumulator select and the P register's enable term of the MAC lane (`PVR`) are registers of their own.
  - `hw/u200/k4_floorplan.tcl` lists every broadcast head copy.
- The routed kernel of the image has 402,384 LUT and 510,465 FF, where the Core 1.2.0 image has 406,288 and 518,825.
  The design uses 15,679 SLLs, where it used 16,482.

## [1.2.0] - 2026-11-17

Core capability 1.0. The image is `tsr_u200_k4.xclbin` for the platform `xilinx_u200_gen3x16_xdma_2_202110_1`,
with the core clock at 200 MHz, the control core at 100 MHz and `CAPS` 0x30121 (VU, ATT, GDN, MVM_DUAL, QACT_PIPE).

| Item | Value |
|---|---|
| sha256 | `8e90db3b593edb16e867c5c637916d88dc7aca586048d0eb1a99a5c283685656` |
| UUID | `65e69091-1b5e-4f7f-443f-bf3b4c456199` |

The image needs Tesseract Basis 1.2.0 or later, as Core 1.1.0 does. Tesseract Basis 1.5.0 is the first that knows
`CAPS` bit 17.

### Added

- The QACT scale pipeline (`QACT_PIPE`, [docs/rtl-core.md](../docs/rtl-core.md) §8.1). The A32 block scales of QACT
  (`d = amax / 127` and `id = 1 / d`) come from the new `tsr_qact_scale` instead of the two serial dividers: a multiply by
  a constant on two DSP48E2 and the normal rounding for `d`, the reciprocal core of the vector unit (`tsr_vu_rcp`) for `id`.
  The pipeline takes one block per cycle, has a fixed latency of 27 cycles and never stalls, because the reader reserves an
  entry of a scale queue (`SQ_DEPTH`, 32) before it reads a block. Results do not change by one bit. A256 commands keep the
  dividers.
- `QACT_PIPE` is a parameter of `tsr_core_dp`, `tsr_core` and `tsr_xrt_kernel` (`A32_PIPE` of `tsr_qact`) and a knob of
  `gen_tops.py`. `tsr_u200_k4` sets it to 1, so the image reads `CAPS` 0x30121 (VU, ATT, GDN, MVM_DUAL, QACT_PIPE).
  `core_ooc.sh` has the preset `k4qpipe`.
- `CAPS` bit 17 `QACT_PIPE`. It is information for cycle models and tools: no result depends on it. A Tesseract Basis that
  does not know the bit still runs models on the image: it prints no name for the bit and charges QACT at the divider cost
  in Tesseract Pivot. Its tools that build a simulator from the core's configuration refuse the bit: the shadow of
  `tsr-selftest` (`--no-shadow` runs without it), `tsr-trace-replay` and `tsr-trace-conv`.
- Timing of an A32 op with the pipeline: 2 cycles per 32-block plus 2 for every D line (16 blocks, and the end of a token),
  where the dividers take 16 cycles per block. For a vector of 1,536 elements that is 151 cycles from accept to done
  instead of 808.
- Logic of the QACT unit out of context (`xcu200-fsgd2104-2-e`, 300 MHz): 442 LUT, 525 FF, 10 DSP48E2 and 1 RAMB18
  more than without the pipeline.
- Benches: `tb/qact` runs every configuration with the pipeline as well (the tight queue, T_MAX 8, the timing of the K4
  image, the real SPM), `tb/qact/test_qact_scale.py` compares `tsr_qact_scale` with `tsr_fdiv`, `tsr_frecip` and
  `tsr_f32_to_fp16` for every possible `amax` (0 mismatches over 2,139,095,041 patterns), and `tb/core` and `tb/platform`
  run the K4 core and kernel with `QACT_PIPE` 1 against the simulator and read bit 17.

### Changed

- A K4 link asks for 0.060 ns more setup slack on the shell's host data clock `dma_ip_axi_aclk` during placement and
  routing (`hw/u200/dma_clk_margin.tcl`, [docs/platform-u200.md](../docs/platform-u200.md) §8.3). That clock is the
  tightest one of every K4 link. The requirement is removed after routing, so the final timing check is the platform's
  own. `TSR_DMA_MARGIN_NS=0` switches it off.

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

[Unreleased]: https://github.com/jwkim04/tesseract-npu/compare/core-v1.2.1...HEAD
[1.2.1]: https://github.com/jwkim04/tesseract-npu/compare/core-v1.2.0...core-v1.2.1
[1.2.0]: https://github.com/jwkim04/tesseract-npu/compare/core-v1.1.0...core-v1.2.0
[1.1.0]: https://github.com/jwkim04/tesseract-npu/compare/core-v1.0.0...core-v1.1.0
[1.0.0]: https://github.com/jwkim04/tesseract-npu/releases/tag/core-v1.0.0

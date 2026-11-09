# Changelog

Changes to Tesseract Basis, the firmware SDK in `sw/ctrl` and the host side that loads and drives it.
The format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is described in
[docs/versioning.md](docs/versioning.md).

## [Unreleased]

## [1.0.0] - 2026-11-10

Basis ABI 1. Targets Tesseract Core capability 1.0. The op list has 21 kinds (kind table signature 0x2381).
The release image is `tsr_u200_k4.xclbin` (identity in [docs/platform-u200.md](docs/platform-u200.md) §9).

### Added

- MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F) at numerics revision 1. MOE_ROUTE takes the softmax of the router logits, selects the k largest (ties to the smaller index, NaN first) and renormalises their weights. MUL_MAT_ID runs a MUL_MAT per (token, slot) against the expert named by an id in DRAM, with an optional per-expert output scale.
- Gated DeltaNet kinds SOFTPLUS (0x2D), SSM_CONV (0x2E) and GATED_DELTA_NET (0x2F), composed from existing VU operations and the DMA. UNARY accepts func 5, SIGMOID, at revision 1.
- GATED_DELTA_NET revision 2 (SW-78): with `CAPS` bit 8 and S_v of 64 or more, each head and token runs the VU's fused GDN op. Other devices and shapes keep revision 1's bits.
- Optional kinds MUL_MAT_SCALE and MUL_MAT_SOFTCAP at revision 1, and capability-gated FAST attention score and probability slots. MUL_MAT, FLASH_ATTN and Basis ABI 1 keep their contracts.
- ggml-tesseract lowers the MoE graphs of Gemma 4 26B-A4B and Qwen3-30B-A3B (SW-77) and the Gated DeltaNet layers of Qwen3.5 and Qwen3.8, with the recurrent-state caches as device windows.
- GATED_DELTA_NET reads and writes its state on the cache cell when every sequence keeps its cell (SW-82). `TSR_GDN_CELL=0` turns it off.
- Multi-token prediction with rollback snapshots (SW-83, SW-85): a context with `n_rs_seq` = K lowers to chains of K = 1 records and CPY records of window rows. The pinned llama.cpp drafter runs unchanged (SW-86).
- Weight matrices that do not fit one planner tile (the output head) are stored in 2, 4 or 8 row blocks with one MUL_MAT record per block (SW-88). `TSR_HEAD_SPLIT=0` keeps them whole.
- The PyTorch backend runs on the Basis firmware: `tesseract_torch.QuantLinear` and the op `tesseract::linear_blocks` take ggml Q8_0 and Q4_K blocks without conversion (SW-81).
- Tools: `tsr-mtp`, `tsr-top` (live monitor, SW-87), the profile report (`TSR_DUMP_PLAN`, `sw/tools/prof/prof_report.py`, SW-84), `sw/tools/trw_vs_sim.py`, `klcheck --verify-k`. The backend hook `ggml_backend_tesseract_counters`.
- Variables: `TSR_SIM_VU=0` (a core without VU and attention, `CAPS` 0), `TSR_SIM_GDN`, `TSR_SIM_CARD` and `TSR_SIM_VU_TIMING` for the `sim` transport; `TSR_DUMP_RE`, `TSR_DUMP_DIR`, `TSR_OPL_PROF`, `TSR_OPL_KINDS` and `TSR_MON`. `TSR_STATS` prints the units' busy cycles.

### Changed

- The ITCM budget of the firmware image is 96 KiB instead of 80 KiB (SW-75, amends SW-38).
- The simulator's cycle models follow the RTL (SW-80): VU lanes and latencies, QACT, stores and MVM fill. Results do not change, only cycle counts.
- `TSR_SIM_LOG` takes a `w,` prefix (the words of every push). `trw dump` of `tsr-dbg` reads a ring that an earlier process wrote.

### Fixed

- A MOE_ROUTE of more than 64 tokens (several tiles) runs on the device (SW-76). Gemma 4 26B-A4B and Qwen3-30B-A3B prefill at the default `-ub 512`.
- klcheck writes the four M-RoPE position streams in its embedding path, so its embd and floor runs equal the token path.

## [0.1.0] - 2026-10-31

Basis ABI 1. Targets Tesseract Core capability 1.0.

### Added

- The firmware runtime, the `graph` executor, the `selftest` and `replay` kernels, and the C++20 kernel headers in `sw/ctrl/include/tsr`.
- Op list format 1 with 14 kinds at numerics revision 1: NOP, JOIN, MUL_MAT, ADD, MUL, SCALE, RMS_NORM, GLU, ROPE, GET_ROWS, SET_ROWS_KV, FLASH_ATTN, CPY, UNARY.
- Versions: the Core capability in the `VERSION` register, the Basis version and the ABI in `sw/ctrl/include/tsr/version.h`, a numerics revision per kind in `opl::kind_rev`. An image carries them in `.note.tsr` and `.tsr_kinds`, and the runtime reports them in HELLO.
- The test `fw_kind_revs` and `sw/ctrl/tests/kind_revs.txt`, which fail when a kind's output bits change without a revision bump.
- Transport checks of the capability, control core, ABI, Basis version and kind table at open, image open, boot and HELLO (`tsr_versions`, `tsr_basis_version`, `tsr_image_versions`, `tsr_transport_versions`, `tsr_versions_str`).
- ggml-tesseract refuses an image with an older Basis version than its own, sends only the kinds the booted image has, and names the capability and the Basis version in the device description.
- Tools: `tsr-selftest --version`, `tsr-elf --version` and `tsr-elf kinds`. klcheck and numcheck record the versions of a device run.

[Unreleased]: https://github.com/jwkim04/tesseract-npu/compare/basis-v1.0.0...HEAD
[1.0.0]: https://github.com/jwkim04/tesseract-npu/compare/basis-v0.1.0...basis-v1.0.0
[0.1.0]: https://github.com/jwkim04/tesseract-npu/releases/tag/basis-v0.1.0

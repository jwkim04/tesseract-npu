# Changelog

Changes to Tesseract Basis, the firmware SDK in `sw/ctrl` and the host side that loads and drives it.
The format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is described in
[docs/versioning.md](docs/versioning.md).

## Unreleased

- Added `TSR_SIM_VU=0` to the `sim` and `mock` transport kinds: the core of the reduced first card image
  `tsr_u200_k4` (no vector unit, no attention; CAPS 0 and no VU queue). The host then runs MUL_MAT on the
  device and every other op on the CPU, as in M6. Behaviour without the variable is unchanged.
- Added capability-gated FAST attention score and probability slots. FLASH_ATTN
  keeps its record format and numerical order.
- Added the optional MUL_MAT_SCALE kind at numerical revision 1. The kind
  scales completed FP32 results before their final export. Legacy MUL_MAT and
  Basis ABI 1 retain their existing contracts.
- Added the optional MUL_MAT_SOFTCAP kind at numerical revision 1. The two
  scale coefficients are the original FP32 graph constants.
- Added the MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F) at numerical revision 1. MOE_ROUTE computes
  the softmax over the router logits, selects the k largest probabilities (value descending, ties by the
  smaller index, NaN first, by integer compares on the RISC-V) and renormalises their weights.
  MUL_MAT_ID runs a MUL_MAT per (token, slot) against the expert chosen by an id read from DRAM, with an
  optional per-expert output scale. The image grows by about 5.4 KiB of ITCM. No RTL change.

## Basis 0.1.0 (2026-10-31)

Basis ABI 1. Targets Tesseract Core capability 1.0. Runs on the simulator; there is no card with this
design yet.

### Added

- The firmware runtime, the `graph` executor, the `selftest` and `replay` kernels, and the C++20
  kernel headers in `sw/ctrl/include/tsr`.
- Op list format 1 with 14 kinds, all at numerics revision 1: NOP, JOIN, MUL_MAT, ADD, MUL, SCALE,
  RMS_NORM, GLU, ROPE, GET_ROWS, SET_ROWS_KV, FLASH_ATTN, CPY, UNARY.
- Versions: the Tesseract Core capability in the `VERSION` register, the Basis version and the ABI in
  `sw/ctrl/include/tsr/version.h`, a numerics revision per kind in `opl::kind_rev`. An image carries
  them in `.note.tsr` and `.tsr_kinds`, and the runtime reports them in HELLO.
- The test `fw_kind_revs` and `sw/ctrl/tests/kind_revs.txt`, which fail when a kind's output bits
  change without a revision bump.
- Transport: the capability, control core, ABI, Basis version and kind table are checked at open,
  image open, boot and HELLO. The API is `tsr_versions`, `tsr_basis_version`, `tsr_image_versions`,
  `tsr_transport_versions` and `tsr_versions_str`.
- ggml-tesseract: refuses an image with an older Basis version than its own, sends only the kinds the
  booted image has, and names the capability and the Basis version in the device description.
- Tools: `tsr-selftest --version`, `tsr-elf --version` and `tsr-elf kinds`. klcheck and numcheck record
  the versions of a device run.

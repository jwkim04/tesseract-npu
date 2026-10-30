# Changelog

Changes to Tesseract Basis, the firmware SDK in `sw/ctrl` and the host side that loads and drives it.
The format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is described in
[docs/versioning.md](docs/versioning.md).

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

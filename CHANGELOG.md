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
- Added the Gated DeltaNet kinds SOFTPLUS (0x2D), SSM_CONV (0x2E) and GATED_DELTA_NET (0x2F), each at numerics
  revision 1, composed from existing VU operations and the DMA: no new RTL operation. They need the VU (CAPS bit 0).
- GATED_DELTA_NET r2 (SW-78): with `CAPS` bit 8 (the VU's fused GDN op, VU §7.12) and S_v of 64 or more, each head and token runs
  one `GDN` op instead of the 13 VU operations. The dot products become exact block fixed-point sums rounded once, α is folded
  into k, and the output comes from the old state and κ = k · q, so the output bits change on a device with the bit. A device
  without it, and every record with S_v below 64, produces revision 1's bits (the composed row of `kind_revs.txt` keeps
  digest `b040cc2f4da8c0ca`). The record and the planner do not change. The `sim` transport has `TSR_SIM_GDN=0` to clear
  `CAPS` bit 8. The kind table signature is now 0x2381. The image takes 85,504 B of ITCM.
- The ggml-tesseract backend puts a GATED_DELTA_NET on its recurrent-state cache cell when the ids of the state gather say that every
  sequence keeps its cell (SW-82): the record reads the cell and writes the new state back into it, and the `GET_ROWS` gather and the state
  `CPY` of the pinned graph are absorbed. One pass over the state instead of three. No record, firmware or ABI change; the results are
  bit-identical. A plan that depends on the ids is checked again at every reuse. `TSR_GDN_CELL=0` turns it off.
- UNARY accepts func 5, SIGMOID (VOP SIGMOID). The kind's revision stays 1: no existing record changes.
- The ggml-tesseract backend lowers the Gated DeltaNet layer graph of the pinned llama.cpp (`qwen35`): SOFTPLUS, SIGMOID and SILU
  on gate tensors, `GATED_DELTA_NET`, the `CONCAT` / `SSM_CONV` / state `CPY` group (two window `CPY` records and one
  SSM_CONV record), the regrouped state `CPY`, IMROPE as NEOX for text positions, per-token `RMS_NORM` for strided views,
  and the recurrent-state caches as device windows. The conv taps are stored as tap rows at weight upload. Whole decode
  tokens of Qwen3.8 2B run in one launch. No llama.cpp patch.
- Raised the ITCM budget of the production firmware image from 80 KiB to 96 KiB (SW-75, amends SW-38). The image with the MoE and the
  Gated DeltaNet kinds together takes 84,056 B; each set alone fit the old budget. The kind table has 21 kinds (signature 0x804f).
- `TSR_DUMP_RE` / `TSR_DUMP_DIR` dump device values of matching nodes; `TSR_STATS` prints the units' busy cycles.
- klcheck writes the four M-RoPE position streams in its embedding path (the embd and floor runs of Qwen3.5 / 3.8 were not
  bit-identical to the token path before).
- The simulator's cycle models follow the RTL and the card (SW-80): the VU from its lanes (`TSR_SIM_VU_TIMING`, `u200` 16 / 16 / 8 by
  default, `k4` for the card image's 8 / 8 / 4 behind two lane-pipe stages) with latencies measured on the RTL unit, QACT (A32 is
  divider bound: 16 cycles per 32-block), stores and MVM fill. `TSR_SIM_CARD=k4full` sets the card image's configuration for the
  `sim` transport. Results do not change: only the cycle counts. `TSR_SIM_LOG` takes a `w,` prefix (the words of every push);
  `TSR_OPL_PROF=1` / `TSR_OPL_KINDS=<file>` make the backend profile its launches and list their record kinds; `tsr-dbg`'s
  `trw dump` reads a ring that an earlier process wrote; `sw/tools/trw_vs_sim.py` compares a card run with a simulator run.

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

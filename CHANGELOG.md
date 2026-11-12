# Changelog

Changes to Tesseract Basis, the firmware SDK in `sw/ctrl` and the host side that loads and drives it.
The format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is described in
[docs/versioning.md](docs/versioning.md).

## [Unreleased]

### Added

- The profile dump (`TSR_DUMP_PLAN`) records the weight streams that the ideal MVM cycles are counted on. `meta.json` has `mvm_dual`, 1 when MVMs of T ≤ T_MAX / 2 ran on two weight streams. Each MVM record of a plan file has `mvm_passes`, the ideal cycles of its MVMs as pairs of a count and the cycles of one MVM. `summary.json` has `mvm_dual` and `t_max`, and the Summary tab of the profile report has the row "MVM ideal streams". A dump without these fields is read as one stream.

### Changed

- A MUL_MAT_ID record of a batch (T > 1: a prompt, a verification batch of speculative decoding) reads each expert once per tile of the (token, slot) pairs that use it, up to T_MAX rows per tile, and not once per pair. The pairs are grouped while the ids are read, and a decode token (T = 1) is unchanged. The outputs do not change, and the kind revision of MUL_MAT_ID stays 1. The profile dump marks such a record `grouped`: its ideal MVM cycles and weight bytes are the lower bound of the grouping, and it has no `mvm_passes`.
- `sw/tools/prof/capture_sim.sh` runs the simulator as the release image by default: `TSR_SIM_CARD=k4dual` (was `k4full`, one weight stream) and `TSR_SIM_GDN=1` (the fused GDN op, which a card configuration left off). The cycles of a capture differ from those of one made with the old default: MVMs of T ≤ T_MAX / 2 take fewer, and a Gated DeltaNet model runs the op stream of the card. `SIM_CARD=` (empty) still gives the plain simulator, and `TSR_SIM_GDN=0` the composed GDN path.

### Fixed

- `tsr-top` shows the generation rate of a speculative decoding run. The publisher counted a verification batch of K + 1 tokens as prompt tokens, and the passes of the draft context in either count, so `TOK/S` showed `pp n` and no decode rate. `tok_decode` now counts the tokens the main context generates (per verification batch one token plus the drafts the previous batch accepted, from the `inp_pos` and `out_ids` inputs of the pass), `tok_prefill` its prompt tokens, and a pass of the draft context (less than 0.9 of the largest weight volume of a pass) counts in `passes` only. Plain single-sequence decoding counts as before; a decode batch of several sequences counts one token per row, and the drafts accepted by the last verification batch of a generation are not counted. The ctest `tools_top_spec` checks it.
- The profile report (`sw/tools/prof`, `TSR_DUMP_PLAN`) compares an MVM with the ideal of the weight streams it ran on. It counted one weight beat per cycle and cluster, so a decode profile of a core with the dual weight stream (`CAPS` bit 16) showed an MVM efficiency above 100 % and a negative fill term F. `sw/ggml-tesseract/tsr-prof.cpp` takes two streams for an MVM of T ≤ T_MAX / 2 when `CAPS` bit 16 is set and `TSR_MVM_DUAL` is not 0, and one stream otherwise. The report takes the bytes of an op from the weight bytes of its record, and the stream rate ceiling (E at clock) from the weight bytes over the ideal cycles. The ideal is the limit of the streams, not of the memory. On a card with two streams the DDR gives less than the streams ask for, so an MVM stays below 100 % efficiency, F holds the time the DDR cannot follow, and E is above 100 % (`docs/prof-dbg.md` §3 and §3.1). The ctest `tools_prof_report_*` check the ideal against the simulated run time of every MVM op, with the dual weight stream on and off.

## [1.2.0] - 2026-11-12

Basis ABI 1. Targets Tesseract Core capability 1.0. The op list has 21 kinds (kind table signature 0x2381).
It runs on Tesseract Core 1.1.0 and 1.0.0 ([rtl/CHANGELOG.md](rtl/CHANGELOG.md)). Core 1.1.0 needs this release or a later one.

### Added

- `TSR_MVM_DUAL` (default 1) controls the dual weight stream of a core with `CAPS` bit 16 `MVM_DUAL`, the feature of Tesseract Core 1.1.0. `TSR_MVM_DUAL=0` turns it off: the xrt transport writes `ADAPTER_CTRL` bit 1 `DUAL_DIS` at bind (`sw/transport/tsr_kdev.cpp`), and the `sim` transport runs its cycle model with one stream. Output bits do not change, only timing.
- The xrt transport accepts the kernel arguments `chb<c>` (offset 0x100 + 8c), the read-only port of the second weight stream of channel c. It checks that a kernel has one for every channel or none, and that each uses the memory bank of `ch<c>` (`chb_by_slot` in `sw/transport/tsr_transport_xrt.cpp`). A Basis before 1.2.0 refuses a kernel that has them.
- The simulator models `CAPS` bit 16: with the run-time enable on, a weight MVM with T ≤ T_MAX / 2 reads two weight streams per channel (`docs/rtl-mvm.md` §15.7). `TSR_SIM_CARD=k4dual` is the card image (`k4full`) with the feature, `TSR_SIM_MVM_DUAL` sets (1) or clears (0) the bit, and `TSR_SIM_MEM_RATE` bounds the memory rate in weight beats per cycle and channel. The `sim` transport has the bit only with `TSR_SIM_CARD=k4dual` or `TSR_SIM_MVM_DUAL=1`, and `TSR_SIM_VU=0` keeps it. Results do not change.
- `tsr_versions_str` prints `MVM_DUAL` among the `CAPS` names (`Tesseract Core 1.0 (VU ATT GDN MVM_DUAL)` for the release image), and so do `tsr-selftest` and the llama.cpp device list. The generated headers have `CAPS_MVM_DUAL` and `ADAPTER_CTRL_DUAL_DIS` (`sw/ctrl/include/tsr/mmio.hpp`).
- `tsr_bw --cu NAME[,NAME]` drives one named compute unit of a bandwidth image that has several (`hw/u200/bw/bw2x200.cfg`), or two at once. Two units read the same banks at the same time, in read mode only, and the tool reports per bank the read rate of each unit and their sum (`sw/tools/tsr_bw.cpp`).
- `tsr-top` shows the `MEM BW` meter, the read rate of all DDR channels as a share of what the DRAM gives, and a second line in the time series box for the device memory in use. The three meters share one row from 96 columns and take two rows on a narrower terminal.

### Changed

- `tsr-top` takes the stream limit of its DDR panel and of `--json` (`rates.ddr.stream_limit_gbs`, `pct_stream`) as two beats of 64 B per core cycle and channel on a core with `CAPS` bit 16, the dual weight stream. The new key `rates.ddr.streams` gives the number of weight streams of a cluster. Counted with one beat, the read rate of such a core came out above 100 % of the limit.
- Matching a launch trace to a device ignores `CAPS.MVM_DUAL`, since the bit changes timing only: traces made on a core without the bit replay on a core with it and the reverse (`tsr-trace-replay --device`, `tsr-selftest --traces`).
- The profile report page (`sw/tools/prof`) is a profiler window: toolbar, launch tree, timeline with a row per unit, and a bottom pane with Summary, Records, Graph and Details tabs. `capture_card.sh` defaults to `build/u200/k4/tsr_u200_k4.xclbin`, the image that `hw/u200/build_xclbin.sh k4` writes, and no longer replays the first-ATTPV trace unless `WARM=1`.

### Fixed

- `tsr-mtp` runs targets without recurrent layers (Gemma 4): it asks for rollback snapshots only when the target is recurrent or hybrid.
- `tsr_mon_read` tries for 200 ms and gives the CPU away while the writer stays stopped inside an update. It made 100,000 tries without a pause and returned -3 when none found the page consistent. The ctest `transport_mon`, whose writer never pauses, failed now and then for that reason. The test now counts a give-up (-3) apart from a torn page and ends on a number of consistent reads.
- The capacity probe of the xrt transport and the capacity check of `tsr-selftest` no longer fail on a block that cannot be read. The shell's DDR4 controllers use ECC, and a block that was not written since the xclbin was loaded can hold no valid ECC word: its read fails until it is written. The probe read both blocks of a pair before it wrote them, so the open stopped with `capacity probe ... cannot read the buffer`, and the failed read moved the transport to `staged` DMA. The new `kdev::mem_try_read` reads bytes that may never have been written. A block it cannot read is taken as never written: the test writes it, it holds zeros afterward, and one line on stderr names it. The DMA mode stays as it was.
- The ctest `tools_top_identity` reads PERF after every launch (`TSR_MON_PERF_MS=0`). With the default interval of 100 ms it failed on a quiet host: the run of the tiny model ended before the first read after the attach was due, and the record it left held no PERF counts.

## [1.1.0] - 2026-11-10

Basis ABI 1. Targets Tesseract Core capability 1.0. The op list has 21 kinds (kind table signature 0x2381).
It runs on Tesseract Core 1.0.0 ([rtl/CHANGELOG.md](rtl/CHANGELOG.md)).

### Added

- Transport hooks for a channel buffer that is larger than the memory behind it: `kdev::buf_bytes` and `kdev::detect_capacity`. Their defaults keep `ch_bytes`, so the simulated kinds do not change. The xrt kind measures the capacity at open (`detect_ch_capacity`). `tsr_mon_cap_read` and `tsr_mon_cap_write` record it per card in `tsr-cap-<bdf>`, so an attach open and `tsr-top` use what an owner open measured.
- `tsr-selftest` checks the channel capacity: distinct patterns at 0 and at P for every power of two P below the buffer, and the transport's channel size must equal the capacity found. Its memory checks wrote and read back at the same address and could not see a wrap.

### Changed

- The nominal core clock of the transport and the tools is 200 MHz, the core clock of the release image, instead of 276.3 MHz. The simulated kinds count device time at it, so a timeout of the same length covers 28 percent fewer core cycles. On xrt the PERF summary takes the clock that the driver reports in sysfs (`kdev::core_clock_hz`), and the clock fitted from `DBG_MTIME` once the session is long enough. `tsr_perf_summary.core_hz_fitted` says which one it is: 0 nominal, 1 fitted, 2 platform (`TSR_CORE_HZ_*`). `tsr-selftest` reports the PONG DURATION at that clock. `tsr-top`, `tsr-prof` and the profile report fall back to 200 MHz where no card clock is known. Results do not change.
- `tsr-top` shows the device memory in use against the card's DDR as the MEMORY meter. The DDR bar in the header and the bandwidth meter and series are gone.

### Fixed

- The xrt transport measures the capacity of each channel at open. The U200 card holds 8 GiB per channel, but the shell's memory topology says 16 GB per bank, so addresses above 8 GiB wrapped and a model of more than about 8 GiB per channel overwrote its own weights. The heap, the KV placement, the range checks and the monitor record now use the measured size, `TSR_XRT_CH_GIB` above it is refused, and a model that does not fit stops with `WEIGHT heap full`. `TSR_ATTACH` writes nothing and takes the capacity that an owner open recorded in `tsr-cap-<bdf>`. `tsr-top` shows 32 GiB for the card also when no process owns it.
- Models that share KV between layers (Gemma 4 E4B) no longer abort in prefill at a large ubatch: the backend copies the K indices that ggml-alloc overwrites before a sharing layer's FLASH_ATTN (SW-89).

## [1.0.0] - 2026-11-10

Basis ABI 1. Targets Tesseract Core capability 1.0. The op list has 21 kinds (kind table signature 0x2381).
It runs on Tesseract Core 1.0.0 ([rtl/CHANGELOG.md](rtl/CHANGELOG.md)).

### Added

- MoE kinds MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F) at numerics revision 1. MOE_ROUTE takes the softmax of the router logits, selects the k largest (ties to the smaller index, NaN first) and renormalizes their weights. MUL_MAT_ID runs a MUL_MAT per (token, slot) against the expert named by an id in DRAM, with an optional per-expert output scale.
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

[Unreleased]: https://github.com/jwkim04/tesseract-npu/compare/basis-v1.2.0...HEAD
[1.2.0]: https://github.com/jwkim04/tesseract-npu/compare/basis-v1.1.0...basis-v1.2.0
[1.1.0]: https://github.com/jwkim04/tesseract-npu/compare/basis-v1.0.0...basis-v1.1.0
[1.0.0]: https://github.com/jwkim04/tesseract-npu/compare/basis-v0.1.0...basis-v1.0.0
[0.1.0]: https://github.com/jwkim04/tesseract-npu/releases/tag/basis-v0.1.0

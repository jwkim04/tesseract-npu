# Claude handoff checkpoint, November 6, 2026

Read this page, then [handoff.md](handoff.md). Dated entries in the handoff are
an experiment history. A later entry supersedes an earlier statement that a
build is active, stopped or not yet launched. This checkpoint is not hardware
signoff. The user requested a source/documentation checkpoint before changing
assistants, not a new build, rollback or automatic retry.

## Checkout and source provenance

| Checkout | Branch | Role |
|---|---|---|
| `/home/limagin-dev/tesseract-local-intermediates-20261105` | `codex/local-intermediates` | Latest cumulative implementation; active native build; checkpoint commit belongs here. |
| `/home/limagin-dev/tesseract` | `codex/k4-route-fixes` | Previous RV-local/VU-SLR2 implementation, HEAD `70bc1d6` plus retained uncommitted work. Its last native attempt was stopped. |
| `tesseract-kvst-masked-20261105`, `tesseract-rv-local-20261105` | Isolated feature branches | Earlier feature candidates; their validated changes are integrated into the current candidate. |
| Other `tesseract-k4-*` worktrees | Detached experiments | Historical comparisons, not the active build source. |

Both principal checkouts started this checkpoint from `70bc1d6`. The candidate
was created from the original checkout's complete dirty source, not HEAD alone.
Its cumulative commit includes the earlier RTL/platform work and the new
local-intermediate, masked-KVST and software changes. It is not merely a docs
commit. The original checkout remains available for comparison; its contents
are not asserted identical to the candidate.

The original checkout has an additional recovery copy under its
`build/analysis/checkpoint_20261106/`: `original-main.patch`,
`original-main-untracked-source.tar.gz` and `original-main-source.json`.
These contain the tracked diff, allowed untracked source/docs and file hashes.
They exclude vendor contents, models, traces and generated build trees.
The older `build/analysis/local_intermediates/source_delta.patch` predates the
final hierarchy-preservation/config-comment changes. Use the checkpoint source
and actual Git diff as the current authority, not that older patch alone.

`third_party/llama.cpp` is intentionally dirty from the repository's applied
patch. Do not commit its working tree or change its gitlink. The pinned vendor
state, generators and numerical-version rules in `AGENTS.md` still apply.

## Current user decisions

- Four clusters, T_MAX=4, VU 8 front / 8 EXP / 4 back lanes. Keep MVM and VU
  arithmetic independent and the four 512-bit DDR weight streams.
- Core at least 250 MHz is success if route, setup/hold and CDC also pass.
  Do not repeat builds merely to reach 300 MHz. RV is requested near 166.7 MHz;
  Vitis uses integer-MHz clock settings, so inspect the actual clock reports.
- Placement intervention stops at Pblocks for major functional blocks. Show
  the Pblock map before every new build. Do not add individual-cell/stage/site
  constraints, ownership audits or placement ECOs.
- Keep the current build running. No new long-running goal, automatic retry
  or rollback is requested. Luna was stopped; no LLM polling monitor is wanted.
  Implementation uses Sol and review uses Astra when agents are needed.
- Use existing swap, no sudo, build/implementation jobs 8, IP synthesis jobs 16.
  All Vivado commands use the shared two-top-level-slot wrapper; internal native
  workers do not count as additional top-level runs.
- Preserve llama.cpp and PyTorch compatibility. A single-digit throughput loss
  is acceptable if implementation becomes easier. This is an acceptance target,
  not a performance result. Numerical-order changes are allowed with explicit
  oracle/version/model validation; this candidate kept the existing arithmetic.
- MoE is the first model extension after this build. GDN and MTP/speculation
  remain planned. Do not claim they are implemented by the dense locality work.

## Changes since the initial `58bf76d` handoff

| Work | Durable record and result |
|---|---|
| Initial defects and link gates | `47832fc` fixed LCQ/debug handling and link constraints. Later commits repaired linked CDC binding, clock-calculator compatibility and >=250 MHz acceptance. See `docs/handoff.md` and `docs/design/README.md` §12. Known remaining debug gaps are listed below. |
| Architecture/code audit | `docs/design/README.md` §12 maps RTL, firmware, host, tests and boundaries. File classification is not a claim that every vendor line was reviewed. |
| Interconnect and locality trials | Shared broadcast/return links, DDR0/3 localization, paired gather and VU banking were implemented and compared. SLL reduction alone did not make routing legal. `docs/platform-u200.md` §8.6.1–§8.6.9 records the measurements. |
| Compute/placement experiments | Full VU, half VU, VU-absent, T4 and several major-Pblock assignments were tried. Some were stopped by request; others failed with illegal overlapping routes. Preserve that distinction. See `docs/handoff.md`'s dated attempt table and `docs/platform-u200.md` §8.6.10–§8.6.11. |
| Compact VU and RV control transport | VU input compression/bank-local coefficients, narrower RV-SPM/issue/completion paths, RV in SLR0. These do not reduce MVM lane count. `docs/platform-u200.md` §8.6.12 and §8.6.18–§8.6.19; `docs/design/04-control-core.md` §11.4. |
| DDR partial writes | KVST word masks reach AXI WSTRB; central metadata read-modify-write is removed. Included in the current build, unlike the stopped 19:59 attempt. `docs/platform-u200.md` §8.6.14–§8.6.16. |
| TDM and distributed-SPM research | Standalone TDM and local-SPM/software-stack feasibility were investigated. They are not enabled in this build. `docs/platform-u200.md` §8.6.17 and §8.6.20–§8.6.21. |
| Local attention and MVM consumers | Local FAST score/P banks, existing-VU SCALE/softcap epilogues, 2-flit gather and capability/image-guarded host lowering. Exact scope in `docs/intermediate-locality-plan.md` §0 and its normative contract links. |
| Actual-model decode | Candidate firmware through llama.cpp M7 on the C simulator: Qwen Q8/Q4 and Gemma 12B pass sampled CPU numerical gates and exact local-on/off replays. See below. |

## Active native attempt and notifications

The attempt is under the candidate checkout:
`build/link/k4_20261106_000734_t4_local_masked_core250_rv166/`.
It started at 00:07:34 KST. Supervisor PID 1111688; wrapper 1111699;
event logger 1111691; Telegram notifier 1111692. Verify process identity and
`status.json` before acting; saved PIDs are not durable process ownership.

The packaged snapshot has 141 manifest entries and 142 verified HDL/XDC files.
Build inputs are frozen. Source snapshot, XO, source hashes, generated link
config and native reports are retained under the attempt. Committing the same
source bytes does not change the running design. Do not edit its RTL or hooks.

Placement finished at 02:37 KST, physical optimization at 02:51. The pre-route
clock/reset and CDC gates passed (unsafe 0, unconstrained 0). At 03:18 the
router was in phase 2.5.1 Update Timing, with no terminal failure. The live log is
`tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log`; the outer `build.log` can lag.

| Same-stage metric | Stopped 19:59 build | Current candidate |
|---|---:|---:|
| SLR1 occupied CLBs | 49,168 / 99.81% | 48,942 / 99.35% |
| SLR1 LUTs | 282,026 | 284,877 |
| Placed SLLs | 14,240 | 12,947 |
| Placement estimated WNS | -1.793 ns | -2.790 ns |
| Post-physopt estimated WNS | -1.344 ns | -2.124 ns |
| SLR1 north/south global congestion region | 64x64 / 64x64 | 64x64 / 64x64 |
| SLR1 west global congestion region | 16x16 | 32x32 |

The 9.1% reduction in placed crossings did not clear the central congestion.
Estimated timing worsened. These are full-device reports including the platform,
not kernel-only resources or final route timing. CLB site occupancy is not LUT
utilization. Router per-column demand is a different metric from placed SLLs.

The model-free Telegram script checks every 60 seconds, sends no unchanged
status, and records successful placement-comparison and route-step notifications.
Its source, delivery state and frozen baseline are under
`build/analysis/k4_local_build_20261106/`. The route-step alert includes pre-route
hooks; it does not assert route completion. Credentials stay outside the repo in
`~/.config/tesseract/`. Do not print them. This watcher sends notifications; it
does not wake an LLM or automatically repair/retry a failed build.

## Functional and performance evidence

Candidate firmware image ID is `f7bb1fc039d7ee0c`, SHA256
`0d86cd4cf41a5bba4a0ae5fb53b3416f84f3fec95fba74826ec46dab3431e45e`.
Old numerical kind fingerprints remain unchanged. New SCALE/softcap kinds are
revision 1. Old-image and unsupported-pattern fallbacks are retained.

| Small RTL graph, same clock configuration | Legacy cycles | Candidate cycles |
|---|---:|---:|
| FAST attention | 10,337 | 10,542 (+1.98%) |
| MVM then SCALE | 6,428 | 5,726 (-10.92%) |
| MVM then softcap | 32,616 | 31,862 (-2.31%) |
| Short prefill, gather narrowing only | 30,569 | 30,653 (+0.28%) |

These real-core/firmware graphs compare final DDR bytes with the C oracle.
Local-lane abort/drain/restart and legacy regressions pass. One gather pair now
peaks at half a line/cycle; both pairs aggregate one line/cycle. Uneven workloads
can expose this bandwidth limit. These tests do not establish whole-model tok/s.

| Actual llama.cpp M7 model | Sampled logit rows | KL / three-seed floor | CPU top-1 match | Local-on/off exact rows |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B Q8_0, 4 prompts x 32 generated tokens | 128 | 1.105141 | 124/128 | 128/128 |
| Qwen2.5-0.5B Q4_K_M requant, same scope | 128 | 1.376901 | 124/128 | 128/128 |
| Gemma 4 12B Q4_K_M, 8 generated tokens, prompt-tail 2 | 9 | 0.355755 | 9/9 | 9/9 |

All pass the existing sampled criterion and CPU token/embedding identity check.
Queue logs confirm local QK/PV and Gemma's local softcap. Both replay sides retain
the candidate gather and masked KVST; this is not a whole-design old/new A/B.
The C simulator cycle model is not the native VU 8/8/4 timing model. Primary
Qwen runs used the default RV/core ratio; shorter queue samples and Gemma used
166.666667/250. No long-context/SWA-wrap, model-quality, MoE or FPGA throughput
qualification follows. Receipts and exact commands are under
`build/analysis/model_decode_local_20261106/`; independent Astra review is in
`build/analysis/local_intermediates/review.md`.

M7 already performs one device launch per decode token. RV executes the supported
operation sequence without host intervention between operations. The host still
prepares/submits each graph, receives logits and selects the next token. Local
intermediates reduce FPGA-internal traffic; autonomous multi-token decode is not
implemented.

## Next worker's decisions

1. Read the existing native result before starting another build. A zero wrapper
   exit is not signoff: check legal route, setup/hold, core >=250 MHz, RV clocks
   and final CDC. Preserve a failed DCP and its exact source/report relationship.
2. Do not infer that a full rollback solves routing: the preceding configuration
   also lacks a successful routed image. Keep masked KVST as a useful independent
   change. If a new comparison is requested, isolate the local-attention RAM and
   control cost versus retained central paths, then epilogues/gather/placement.
   No single feature has been proven responsible for the current timing result.
   Switching software capabilities off leaves the hardware present; an area or
   routing comparison must actually remove the corresponding RTL at elaboration.
3. After a usable image, follow W6 card bring-up, identity/VERIFY/HELLO, selftest,
   GDB and llama.cpp/PyTorch model/numerical/performance checks. Historical
   pre-control-core card performance is not a result for this image.
4. Prioritize MoE expert packing/strides, MUL_MAT_ID, grouping and combine in the
   common host/firmware path. TOPK exists conditionally but is disabled in K4 and
   needs a newly qualified image to use. GDN state tiling and MTP state rollback
   requirements are recorded in `docs/design/README.md` §12.6. Do not drop
   llama.cpp/PyTorch compatibility while extending the hardware interface.

Known separate debug gaps remain: early RV_SAMPLE/RVWP_APPLY acknowledgement,
missing RTL SBA_REFUSED setter, and unverified integrated current-card GDB.
Simulator GDB and module-level RTL debug tests are not full-system card evidence.
The two-clock vkernel server and full-model RTL performance remain unfinished.

## Checkpoint validation

The user explicitly authorized immediate WIP commit and stopping the remaining
checks after the full-gate requirement and discovered failure were explained.
This handoff-only exception does not waive future implementation/release gates.

- Software build passed; non-slow CTest: 295 passed, one skipped fixture and one
  disabled tokenizer-vocabulary test, zero failures. Both generators passed.
- Full lint was stopped after 102 passing Verilator cases and three failing C16
  variants. All fail WIDTHCONCAT at `rtl/core/tsr_vu.sv:302`: zeroing the packed
  local request bus expands to 8,672 bits, above the 8,192 warning threshold.
  K4 C4 is not that configuration. XSim and Vivado multi-driver checks were not
  reached. A per-owner zero assignment is a proposed fix, not an applied fix.
- The full RTL pytest suite collected 650 tests and was stopped before completion.
  No whole-suite pass is claimed. Existing focused regression receipts remain
  evidence only for their recorded scope.
- All 142 frozen native input hashes still match. Only checkpoint test processes
  were stopped; FPGA routing and Telegram monitoring continue.

Receipts/logs are under `build/analysis/checkpoint_20261106/` in the candidate,
including `rtl-lint-receipt.json`, `rtl-lint.log`, `ctest.log`, `rtl_all.log`
and `native_inputs_unchanged.json`. The ignored `build/dev -> local_sw` symlink
uses the existing candidate software build; the pinned RISC-V toolchain is
reused read-only. These checks are separate from native physical signoff.
Build/model/trace files, credentials, simulator output trees and vendor dirty
contents are not part of the commit.

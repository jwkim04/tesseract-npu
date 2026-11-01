# Handoff: where the work stands (2026-11-02)

Work package numbers (W0-W6, P*.*) are those of [design/README.md](design/README.md) §10.

## Done

| Package | State | Commit |
|---|---|---|
| W0-W3: tables, simulator, firmware SDK (Basis), M6 and M7 backend on the simulator | done; one launch per decode token; KL within the criterion for 0.5B, Gemma 4 12B, 32B | up to `667153f` |
| Versions: Tesseract Core capability 1.0, Basis 0.1.0, ABI 1, kind revisions | done, tag `basis-v0.1.0` | `13dbd2e` |
| llama-server on the backend, TSR_KINDS lists, klcheck multi-seed floor | done | `b58fc03`, `7175f61` |
| W4-A RTL: issue path, 2R2W SPM, control subsystem (Ibex, riscv-dbg, TCMs, CDC) | done, gate T1 met at 166.7 MHz (RV32MFast) | `4149614` |
| W4-B RTL: VU, attention modes, DB block, two-clock XRT adapter; tsr-dbg (GDB), tsr-prof | done; ctest 297/297, lint clean, all benches green | `6435e8a` |
| W5 preparation: VU lane pipes (`tsr_vu_lanes.sv`, `VU_LANE_PIPE`), floorplan C'' (`hw/u200/k4_floorplan.tcl`), TCM RAMB36 constraint in every flow (`tsr_ctrl_tcm.xdc`, `hw/u200/tcm_opt.tcl`), `build_xclbin.sh` enabled for k1/k4 | checkpoint; first K4 route failed | `58bf76d` |

## Current blocker: the k4 link does not route

`hw/u200/build_xclbin.sh k4` (4 channels, T_MAX 8, the final image) synthesised and placed, then
`route_design` failed: "The router is unable to resolve localized SLL routing demand"
(Route 35-3339), 5.4 M unrouted pins.

- Utilisation after synthesis (kernel plus shell): CLB LUT 835,805 (70.7 %), FF 43 %, BRAM tiles
  66.8 %, URAM 15.4 %, DSP 49.4 %. Kernel alone: 544,920 LUT, 3,360 DSP, 1,013 BRAM tiles, 128 URAM.
- SLL demand per boundary (17,280 SLLs each, 12 columns of 1,440):

| Boundary | Demand | Columns over 100 % |
|---|---|---|
| SLR1-SLR2 | 17,479 | 7 of 12 (up to 180 %) |
| SLR0-SLR1 | 14,261 | 6 of 12 (up to 158 %) |

- For comparison, the core before the control core (floorplan C') used 7,010 SLLs in total
  (`docs/rtl-m3.md` §14). The failed checkpoint confirms extra ATT job round trips, misplaced
  broadcast decode and VU consumer LUTs. Shell/interconnect nets also contribute substantially
  to the upper boundary; attention alone does not explain the column pressure.
- Artefacts: `build/link/k4/build.log`, `build/link/k4/v++_link.log`,
  `build/link/k4/tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log` (SLL table near "Estimated SLL
  Demand Per Column"), placed and failed-route checkpoint
  `.../impl_1/level0_wrapper_routed_error.dcp`. Opening it needs about 30 GB of memory.

### Source corrections and overnight retry

The existing failed checkpoint was opened for queries and reports only. Both kernel clock
ports use clock ID 0 at 300 MHz; all 4,227 RV FF clock pins use it too. The requested RV clock
was 166.7 MHz. The generated `ulp.v` confirms the alias. The deprecated frequency option was
present, but its being the sole cause is not established: frequency requests reached the BD
script while the secondary reference-clock selection was absent.

The corrections after `58bf76d` are on branch `codex/k4-route-fixes`:

- LCQ payload/classification reads propagate I/O errors. A complete batch is acknowledged
  before host answers/events/credits change. The batch survives ambiguous ack failure;
  uncertain reset blocks consumption until counters are resynchronised.
- RV watchpoint FIFO depth is 16, covering CDC credit-return latency for continuous access
  at the K4 clock ratio. Dense 4,096-access tests preserve both counters and a single HALT
  event at several phases, with CDC jitter too. A sustained RV source faster than the core
  still needs an aggregation/interface change for general losslessness; the simulation
  assertion reports overflow. No wider clock-ratio guarantee is claimed by this fix.
- Hart reset release replaces old pending halt state with HOB or a simultaneous new request.
  The RTL regression follows the existing simulator stale-request test.
- Whole-core and packaged-IP CDC failures now terminate the flow. Link configs use
  `kernel_frequency=0:300|1:166.666667` for U200 scalable clocks. BD and synthesized clock/reset checks run before
  implementation; the final pre-bitstream check reads Vitis's scaled `_new_clk_freq`.
- K4 attention job pipes and selectors stay with their clusters. Residual broadcast decode
  goes to its destination cluster and VU writer/reduction helpers to SLR1. Crossing stage
  counts and reset connections stay the same. Locality is rechecked after placement and
  physical optimisation; missing scopes, unowned replicas and wrong SLRs fail the run.

Validation (2026-11-02): `cmake --build build/dev -j8`, CTest (297 executed, one disabled),
both generated-file checks, and the RTL lint checks pass. The full RTL pytest run passed
290 tests and skipped 115 opt-in tests. Its technology-lint node was stopped to correct a
missing wrapper environment and passed separately. The latest clock/floorplan fixtures
pass 48 tests. The first standalone lint invocation also used a relative wrapper path;
its nine Xilinx-tool checks passed on the absolute-path rerun. `AGENTS.md` now gives the
absolute wrapper and explicit Python for both commands. Logs are in
`build/analysis/implementation_fix/`.

The user authorised an overnight placement/route attempt on 2026-11-02. The first retry
(`build/link/k4_20261102_003426_core300_rv166`) stopped at CFGEN: `[clock] id=0/1` is invalid
for this platform's scalable clocks; only fixed reference IDs 2/3/4 are accepted. The earlier
static review and fixtures missed that distinction. The configs and their regression now
use `kernel_frequency` for both scalable IDs.

The replacement run is `build/link/k4_20261102_003908_core300_rv166`, started at 00:39 KST.
It repackaged the corrected XO from 137 sources. The BD clock/reset check passed on the
generated design: core uses `ss_ucs/aclk_kernel_00`, RV uses `ss_ucs/aclk_kernel_01`, and each
uses its corresponding reset generator. Vitis truncates the RV request to 166 MHz in
`ipirun.tcl` and the generated `ulp_ooc_copy.xdc` (6.024096 ns). The UCS IP's default 500 MHz
parameter is not the implementation request; scalable clocks use the generated constraints
and later runtime programming. At 00:47 KST, block-level synthesis had completed 45 of
176 jobs; eight were running. Placement and routing have not started. The linked-net clock
check and floorplan/locality checks are still pending. No corrected placed/routed result or
xclbin exists yet. The old checkpoint remains evidence of the baseline failure.

The detached supervisor is PID 741389. `status.json`, `build.log`, `memory.jsonl`, source
snapshot/hash manifest and the eventual `exit_code` are in that run directory. The pointer
`build/analysis/overnight_route_latest.txt` identifies it. The run uses eight jobs, the Vivado
slot wrapper and the existing 72 GiB swap. It stops at `vpl.impl.route_design`, preserving
the clock/reset, floorplan/locality and post-route CDC gates. Bitstream generation and the
final frequency gate are deferred; negative setup slack is not by itself a routing failure.

Known current debug-interface gaps remain: RV_SAMPLE/RVWP_APPLY writes acknowledge before
their CDC work completes, and DBG_STATUS.SBA_REFUSED has no RTL setter. These are separate
from the deferred profiler/report work and were outside this correction.

### Next steps

1. Check the active run's `status.json` and logs before starting another link. A zero exit status
   and a routed checkpoint are required before calling the route successful.
2. Check the corrected constraints on the new synthesized design before measuring crossing counts,
   column demand and per-clock timing again. The old DCP suggests about 2,220 fewer ATT payload
   crossings per boundary from removing the extra round trip; this is not a routed result.
3. An implementation-only resume of the old failed run cannot repair its clock wiring. The active
   run rebuilds from the corrected RTL and clock selection. Constraint-only changes on a
   matching synthesized design can use `TSR_VPP_FROM_STEP=vpl.impl`. A full link takes 7-10 h on
   this host because of swap. A machine with
   64 GB RAM or more (Linux, or WSL2 with Linux Vivado/Vitis 2024.2: the Alveo `v++` flow is Linux
   only) runs it much faster and allows two variants at once.
4. Gate T3: kernel clock at least 276.3 MHz (`grep 'compiler selected the following frequencies'
   build/link/k4/v++_link.log`). Known tight paths from the out-of-context placement (WNS -0.614 ns
   placed): DB logic 9-16 levels, `att_stall` crossing an SLR unregistered, an MVM reset copy into
   SLR2.

The k1 link (one channel, all in SLR1) was stopped: 170 K LUT in a 199,680-site pblock, placed WNS
-3.5 ns. If a quick bring-up image is wanted, k1 needs the k4 treatment (lanes out of SLR1).

## After the image (W6 and later)

- W6 card bring-up: identity, image load with VERIFY, HELLO, launch round trip, `tsr-selftest`, the
  GDB smoke over PCIe; M6 gate (0.5B logits bit-identical to the simulator, 32B KL), M7 gate (one
  launch per token, KL), DDR utilisation. Then README performance numbers on the new image and the
  Basis 1.0.0 release.
- Roadmap after the card gate: speculative decoding (n-gram, draft model, MTP) verified in one T_MAX 8
  pass; M8 MoE with Gemma 4 26B-A4B as the target; Gated DeltaNet kinds for Qwen3.8-27B; multi-card
  QSFP mesh; DeepSeek-V4-Flash and Qwen3.8-Flash-Next (multi-card).
- Optimisation levers kept for after the card gate: SPM banks as true dual-port URAM; about 812 free
  URAM (per-cluster attention buffers, larger preload area, T_MAX 16 buffers); the 256-line preload
  area (FLASH_ATTN above 680 tokens runs on the CPU today).
- Smaller open items: the `tsr_rv` cycle table still models the single-cycle multiplier (the RTL uses
  RV32MFast: MUL 3, MULH 4 cycles); 2K/16K long-prompt runs on the simulator; `tsr-prof report` and
  firmware PC sampling are designed but not built; an nvtop-like monitor (`tsr-top`) is designed but
  deferred.

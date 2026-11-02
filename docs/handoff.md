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

## Current: K4 interconnect optimization; physical validation pending

The user resumed work after the intentional route stop. Use the earlier four-cluster
276.3 MHz image as the baseline and reduce the added interconnect before another long route.
Two changes are implemented and pass the functional checks below:

- Broadcast shares only its central first stage (`u_bc_head`) for BC_PIPE >= 2. Every
  cluster retains its own kept local tail. Total latency, reset flush and one line per
  cycle are unchanged. BC_PIPE 0/1 retain the original path. The floorplan distinguishes
  this topology from archived checkpoints with two per-cluster stages.
- VU front/back exits share one 512-bit return after selection beside the lanes. The
  injector already orders these exits. A resettable S0+LB token delay selects the result;
  the separate 512-bit reduction drain is unchanged. BACK8 saves 256 data connections
  and adds one input token before synthesis; actual SLL savings are pending.

The four independent weight streams retain 512 bits per cycle each. The new K4 config
requests core 250 MHz / RV 166.666667 MHz. The early clock gate reads the kernel's config
and rejects a stale 300 MHz model. CDC datapath/skew bounds are capped at 3.333 ns, so the
lower core request cannot loosen the previous crossing budget. K1 retains its 300 MHz request.
Focused MVM/ATT oracle and cycle checks, VU oracle tests and 132 clock/floorplan fixtures pass.
The software build and CTest pass (297 executed, one disabled). Full RTL lint passes,
including nine XSim elaborations and eight Vivado synthesis checks. The full default RTL
run finished with 386 passes, 115 opt-in skips and four failures in the new broadcast reset
test. It initially injected reset too late in four
BC_PIPE=1/RST_LOCAL=0 configurations. The Python stimulus now observes the producer and
asserts the intended occupied stage at each reset. The complete MVM test module passes
all 14 configurations after that fix, with the original assertions and golden/cycle digests.
The four failed nodes also pass in the default build directory (`--lf`, exit 0).
All 390 executed nodes are therefore covered by passing results across the full run and
reruns. Seven new isolated build-script tests pass separately; they were added after the
full run collected its cases. The original failures remain in the first full-run log.

The default RTL suite includes block tests and integrated RTL-Ibex firmware tests.
`core_m7` covers a two-channel tiny decode/prefill layer, including the VU lane-pipe variant,
against the C simulator. It is not a full-size K4 simulation. Added checks cover the
four-channel broadcast/reset configuration, attention traffic and the VU shared-return
latency/reset behavior. Long opt-in mutation/sweep campaigns and card tests are separate.
A small actual-MVM synthesis comparison replaces only the four clusters and gather with
black boxes. Its 530-bit broadcast banks fall from eight to five (4,240 to 2,650 FF); every
source/head/tail bit and the four independent tail outputs check correctly. These black boxes
retain all 530 bits, unlike the 529 surviving bits in the integrated baseline below. This
proves the shared synthesized topology, not actual K4 placement or SLL savings. The native
U200 calculator also passes at 250 MHz (input 100, M=25, D=2, O=5, effective period 4 ns).
Both small CAD jobs exited with code 0 and returned their slots. No new
placement, routing or 250 MHz timing result is claimed. Logs are under
`build/analysis/k4_interconnect_gates_20261102/` and
`build/analysis/k4_bc_shared_20261102/`. Gather arbitration and the DDR0/3 response detour
remain candidates; they are not part of these two RTL changes.

### Preserved attempt and baseline measurements

The corrected attempt completed placement and physical optimisation, with actual SLR locality
checks passing. It was intentionally stopped at 05:25 KST on 2026-11-02: the new timing report
shows kernel FIFO crossings timed against a 0.040 ns clock-edge separation instead of the
required datapath-only maximum delay. Restoring those constraints and moving the five ATT
stall reduction LUTs pass real-DCP checks, with no new DRC violations. The corrected analysis
checkpoint is saved. A native restart from the preserved placement completed physical
optimisation and its locality check. The route-pre clock/reset and CDC gates passed;
`route_design` entered `Phase 1 Build RT Design` at 08:24 KST. No new routed result exists yet.
By 09:57 KST it passed SLL assignment and five initial net-routing passes and entered
`Phase 5.3 Global Iteration 2`. General routing congestion remains; this is not a timing pass.
That iteration and the Phase 5 footer were printed at 10:46 KST, with 2,572,256 overlapping
routing nodes, up from 2,311,139. At 12:51 KST the log was unchanged and the router had not
returned. Two workers each used nearly a full CPU throughout a seven-minute sample, with
little I/O and no measured memory pressure. This shows continued computation, not convergence.
The user requested a stop and investigation of the next attempt. SIGTERM was sent to the
implementation worker at 14:00:55 KST; the native supervisor finished at 14:01:01 with code 1.
The worker, parent Vivado and supervisor have all exited. This is an intentional stop, not a
new confirmed router error or tool hang. There is no routed checkpoint or 250 MHz timing result.
The 25 preserved files (1,242,547,736 bytes), including the placed and corrected physopt DCPs,
are under `attempts/20261102_140055_route_congestion_stop/`. Its `archive_manifest.json`
records hashes; both checkpoint hashes match their earlier identities. `status.json` retains
the native failure code and records the user-requested stop classification. No new build is
running. The current investigation is SLR interconnect reduction and its performance cost,
before choosing another long implementation attempt.
The prior four-cluster image ran at 276.3 MHz before RV/VU/attention integration. Preserve
that weight-stream throughput while investigating the added paths. Historical whole-core
synthesis records show 275,657 CLB LUTs before integration and 533,573 in the W4-B OOC run;
this is not an isolated measurement of each added feature's cost.

A read-only query of the preserved corrected physopt DCP gives the following same-checkpoint
counts. Kernel/core totals are 534,801/529,186 LUTs. These are physical-optimisation counts,
not a new synthesis, route or card result. Cross-hierarchy LUT combining limits exact sums.

| Scope | LUTs | Note |
|---|---:|---|
| RV subsystem | 6,206 | 4,234 FFs, 48 RAMB36; subset of control |
| VU | 83,398 | 433 DSP; includes lanes |
| VU lane array and pipes | 49,128 | 368 DSP; subset of VU, reported LUT difference 34,270 |
| Four ATT PV modules | 86,829 | excludes shared MAC-mode and other attention logic |

The scoped crossing census finds 529 surviving broadcast nets per cluster: 1,058 SLR1->0
and 1,058 SLR1->2. The candidate shared broadcast could remove about 1,058 duplicated
logical boundary-net incidences in total, rather than the raw-width estimate of 1,060.
VU input is 2,084 surviving nets (four 512-bit operand vectors plus 36 control nets); return
is 1,280 (512 drain + 512 front + 256 back), with one reset crossing. This totals 3,365
stage-to-stage VU boundary connections. These are logical-net incidences; they do not count
physical SLLs or prove a column reduction. No sampled source net was excluded from these
selected BC/VU scopes. Other weight/gather/AXI scopes were not enumerated.
The reports, clock metadata and input-hash record are under
`build/analysis/k4_current_interconnect_20261102/`. Core/RV clocks remain 3.333/6.023 ns.
Source-level options and conditional performance models are
`build/analysis/repository_map_20261102/slr_pipe_bandwidth_options.txt` and
`slr_width_performance_tradeoff.txt`. These measurements predate the two changes above.
The query session exited normally with code 0, returned both tool slots and left the input
hash unchanged. No CAD process from the route or this investigation remains.

The next bounded native run is prepared but has not started. There is a process-limit
ambiguity to resolve before launch. `vivado_slot.sh` limits top-level invocations, while
Vivado also launches internal synthesis workers. A small production-DBG probe with
`general.maxThreads=8` and `synth.maxThreads=1` still launched a second actual Vivado process.
The valid `synth.enableParallelHelperSpawn=none` setting removed the early helper but still
launched a direct worker. Adding Vitis's parent would exceed a literal two-process limit.
Both probes were stopped immediately within their own process groups; they do not provide
completed synthesis results. No such tool parameters are applied to the production config.
An earlier launcher-classification error and an invalid `false` enum trial are preserved
separately and excluded from that conclusion. Logs and `/proc` samples are under
`build/analysis/k4_synth_process_probe_20261102/`. No probe process remains.
Clarify whether the two-slot limit includes internal workers before starting the prepared
250 MHz run. It stops after physical optimisation, then preserves the checkpoint and runs
a read-only connectivity/utilisation/timing census. It never starts routing automatically.

The VU alternatives need more than a narrower wire declaration. Front/back result sharing
is a candidate for 256 fewer surviving return nets; the independent reduction drain must
remain separate unless buffering/admission is proved. Moving the boundary to the SPM client
retains two read lines and one write line per cycle, but also carries 512 SREG bits, a
256-bit command and control/address/byte-enable signals: roughly 2.5 K declared boundary
bits, not 1.5 K. It also moves logic into already busy outer SLRs. Moving only angle/coefficient
selection is insufficient: AMERGE's AM_L mode needs two lane-varying coefficient vectors.
Keeping a full-width fallback preserves the original wires. Front/back sharing is now
implemented as described above. The other options remain source-level proposals, without
equivalence proofs, timing passes or measured SLL reductions.

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

The corrected router estimates 8,872 SLLs at SLR1-SLR2 (51.34 % of total capacity) and
10,052 at SLR0-SLR1 (58.17 %). One column per boundary still exceeds capacity: 121 % and
175 %, respectively. `Route 35-3311` warns of localized demand, but assignment completes;
the previous `Route 35-3339` failure was at `Phase 3.1.2`. Passing that stage does not prove
all SLL or general wire congestion is resolved. Initial routing reports 1,317,850 unrouted
nets and global/short congestion level 7 (128x128); it continues with rip-up and reroute.
The first global iteration's WNS is -8.246 ns with hold unavailable, an intermediate
incomplete-route result. Keep the original net/pin units separate when comparing failures.

The congestion windows overlap c2 PV/MAC and VU lane 0 in the lower region, c1/c3 ATT and
assembler 3 in the upper region, and central gather/QACT/SPM/issue plus platform DDR logic.
These are hierarchy correlations, not identified drivers of the remaining failed nets.
The README hero is a logical diagram: c0/c2 and all VU lanes are in SLR0, c1/c3 and their
assemblers in SLR2, and the VU reader/writer/reduction and central gather in SLR1. The
shell's fixed-DMA timing path must not be attributed to the hero's LD/ST DMA block.
The local evidence is `build/analysis/repository_map_20261102/`:
`current_router_congestion_triage.txt`, `readme_bottleneck_mapping.txt` and
`route_silence_assessment.txt`. AMD's [Router Messaging, UG904 2024.2](https://docs.amd.com/r/2024.2-English/ug904-vivado-implementation/Router-Messaging)
describes iterations lasting hours under difficult congestion/hold fixing. That profile
matches the symptoms; it does not establish progress or exclude a tool defect. The later
user-requested stop is recorded above. Any eventual result still needs fully-routed/no-route-error checks.

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
and later runtime programming. All 176 block-level synthesis jobs completed and the corrected
kernel OOC checkpoint was written. At 01:25 KST, top-level synthesis completed and implementation
started. At 01:39 KST, the pre-opt clock gate stopped the run: RV period was 9.999 ns, outside
5.99–6.32 ns. Vitis's native clock-calculator helper passed `Project` where the installed Vivado
API expects a part name, reported `Invalid part string Project`, then emitted a 1:1 generated
clock. Clock/reset BD wiring had passed; this failure is in implementation clock constraints.
No corrected placed/routed result or xclbin exists yet. The old checkpoint remains evidence of
the baseline routing failure.

The repository `clock_vitis_compat.tcl` hook adapts the calculator API for Vivado 2024.2 and the
pinned U200 part. It supplies PART and the input period, checks the solve result/effective ratio,
and cleans up on errors. The native generated-XDC flow is preserved. The focused fixtures now
pass 66 tests, and the real calculator/shim probes pass at 166 MHz. These are local workaround
checks, not a routed result or an AMD-published fix. The failed logs/XDC/status are preserved in
`attempts/20261102_015229_pre_compat/` under the run directory. Configuration-only resume
completed at 01:53 KST with exit 0; `.user_init_pre.dat` and `_full_init_pre.tcl` both contain
the compatibility hook. The implementation resume started at 01:56 KST from `abc4afe`,
supervisor PID 804838. It reused the 176 block-level synthesis results and regenerated only
the top-level synthesis. The actual pre-opt `CLKNET` check passed: core 3.333 ns, RV 6.023 ns,
all 4,225 RV FFs on the RV clock, and separate reset sources in their respective domains.
The calculator selected M=83, D=10, O=5 from 100 MHz, giving 166 MHz. At 02:29 KST,
`opt_design` and its checkpoint write completed. The revised floorplan passed its initial
membership check (`K4FP locality PASS (placed=0)`); placement started after two pre-place
DRC passes with zero errors. By 04:45 KST, `place_design` completed with zero errors and the
actual SLR check passed (`K4FP locality PASS (placed=1)`). The placed reports show:

| `report_utilization -slr`, placed | Previous failed build | Corrected build |
|---|---:|---:|
| SLR0-SLR1 SLLs used | 13,325 | 9,830 |
| SLR1-SLR2 SLLs used | 15,181 | 7,375 |
| Total SLLs used | 28,506 | 17,205 |
| SLR1 occupied CLB sites | 99.58 % | 99.84 % |
| SLR1 LUT utilisation | 76.46 % | 77.09 % |

These are matching placed-report counts, not the router's per-column estimated SLL demand
above. The combined changes reduce placed crossings, but central packing pressure remains:
the placer reports high congestion. Physical optimisation also completed with zero errors
and another `placed=1` locality PASS. Its timing report has global WNS -2.960 ns. The worst
core path is the ATT stall aggregation, routed in the placement estimate through
SLR0 -> SLR2 -> SLR1; 97.1 % of its data delay is wiring. RV setup is -0.367 ns, fixed DMA
clock setup is -0.821 ns, and core/RV/platform hold violations remain in this pre-route
estimate. A lower core clock alone is not yet evidence of timing closure.
The reports are `slr_util_placed.rpt` and `kernel_util_placed.rpt` in the active
implementation run directory. The current attempt, PID and source commit are in `status.json`.
After locality passes, the hook now writes `k4_<step>_timing_summary.rpt` and
`k4_<step>_congestion.rpt` in that directory. These only query the open design. The first
place hook had already finished when diagnostics were added. The first reports were produced
after `phys_opt_design`, and the placed and physopt DCPs are complete. Its runtime hook
snapshot/hash is preserved beside `placement_milestone.json` under the attempt directory.

The reports expose a confirmed constraint gap: sampled FIFO paths in both core/RV directions
have a 0.040 ns natural-edge requirement and no max-delay exception; clock interaction calls
both directions `Clean Timed`, not `Max Delay Datapath Only`. The clocks themselves remain
3.333/6.023 ns. A later real-DCP experiment restored the exceptions, then reread the native
RV clock XDC: the bounds and full CDC gate still passed. Clock redefinition did not reproduce
the loss. The stage that loses the packaged constraints during integration remains unproven.
At 05:24:56 KST, the implementation worker
was stopped after its physopt DCP was complete. The supervisor exited at 05:25:10 with code 1.
Logs, reports, generated scripts and all three implementation checkpoints are archived under
`attempts/20261102_052510_cdc_unbound_stop/`, with hashes. The intended correction restores
the original CDC XDC on the actual clocks and checks its effect before routing. A separate
cycle-identical floorplan correction keeps the exclusive ATT stall aggregation beside its
central consumer. Neither changes RTL behaviour or pipeline depth.

The archived physopt DCP now passes the restored constraint gate: both directions have a
3.333 ns datapath-only requirement and zero unsafe/unconstrained CDC paths. A kernel Gray-bus
skew constraint is present, with a reported bound of 3.333 ns and skew of 0.073 ns. The
`cdc_restore.tcl` hook generates a scoped run-local copy of the original CDC XDC and validates
the linked clocks, timing requirement, CDC structure and kernel skew coverage. `impl_pre.tcl`
uses it before optimisation; `cdc_pre_route.tcl` rechecks before routing and fails on loss.
No false paths, clock groups or new CDC waivers are added.

The remaining floorplan correction selects all exclusive ATT stall reduction LUTs from the
actual endpoint cone. It keeps their source FFs in the original cluster SLRs. A saved-placement
hook moves only the reduction LUTs, preserves their REF/INIT and every other primitive's
LOC/BEL, and checks locality. At 07:02 KST the real-DCP ECO passed: five LUTs moved, all REF/INIT
values matched, and every other primitive's LOC/BEL was unchanged (2,131,692 cells). Actual
SLR locality and the following clock/reset and fresh CDC checks passed. Post-ECO DRC has zero
errors; all 4,368 violation detail records match the baseline, including the three existing
unrouted-state critical warnings. ATT stall slack improves from -2.960 to -1.048 ns. Global
WNS is -2.557 ns, WHS is -0.324 ns, and RV setup is still -0.367 ns. These are pre-route
estimates, not a 250 MHz pass. The remaining worst core path is c0's local ATT QK gate
(10 levels, 84.1 % wire delay); the next two are DB queue/control paths within SLR1.
The corrected DCP, the CDC-only DCP and the immutable source have recorded hashes in
`build/analysis/cdc_restore_20261102/checkpoint_hashes.json`. Vivado exited with code 0.

The native restart began at 07:34:26 KST under supervisor PID 1151282, using
`resume_impl.tcl` on the existing project from `phys_opt_design` through `route_design`.
It preserves the native route-post hook and original placement and does not repeat synthesis.
The GateLvl fileset has no TOP; the driver validates the native `gen_run.xml` checkpoint
records, run/part and matching filenames instead. The actual project API, hook settings,
step reset and launch passed. Child PID 1151638 opened `level0_wrapper_placed.dcp`.
The attempt is `attempts/20261102_073426_cdc_eco_native_resume/`; its source snapshot includes
the validated working-tree hooks. `status.json` points to its `native_resume.log`.
The 25 implementation source snapshots match pushed commit `c2055f6`; the mapping is in
`committed_hook_source.json` in that attempt. On the native placed input, the five-LUT ECO
preserved all 2,131,648 other primitive placements. Physical optimisation and its P1 check
then passed. Setup estimates are core -1.606 ns, RV -0.230 ns and fixed DMA -0.821 ns;
hold remains -0.324/-0.127/-0.201 ns. The remaining core worst path is the local `qk_gate`
feedback (8 levels, 87 % estimated wire delay). These are still pre-route results.
The new physopt DCP was written at 08:19 KST. Route-pre checks passed with core/RV periods
3.333/6.023 ns, all 4,234 hart FFs on RV and both CDC directions bounded at 3.333 ns.
The router started at 08:24 KST. Its entry evidence and checkpoint hash are in
`route_entry_milestone.json`; the completed physopt DCP is 595,717,524 bytes.
The fresh commit gates pass the software build, CTest (297 executed), both generated checks,
full RTL lint (including nine XSim and eight Vivado checks), and the full RTL pytest run
(355 passed, 115 opt-in skips, exit 0). Focused link checks pass 73 tests; floorplan checks pass 32, including
two repeated-application regressions added after the full pytest run collected its cases.
Actual BEL probes cover occupied, free and physical-only resources. Actual floorplan source
applied twice preserves all four pblock ownership sets. No 250 MHz timing result is claimed.
Fresh gate logs and their hashes are in `build/analysis/implementation_fix/cdc_floorplan_gates/`.

The original detached supervisor was PID 741389 and exited with status 1. `status.json`, `build.log`, `memory.jsonl`, source
snapshot/hash manifest and the eventual `exit_code` are in that run directory. The pointer
`build/analysis/overnight_route_latest.txt` identifies it. The run uses eight jobs, the Vivado
slot wrapper and the existing 72 GiB swap. It stops at `vpl.impl.route_design`, preserving
the clock/reset, floorplan/locality and post-route CDC gates. Bitstream generation and the
final frequency gate are deferred. The user revised K4 acceptance on 2026-11-02 to a core
clock of at least 250 MHz. The stopped attempt retained its 300 MHz request. Do not repeat
implementation just to reach 300 MHz once 250 MHz or more is valid. A route
with negative setup slack at 300 MHz is useful evidence; it is successful timing only after
checking an accepted frequency of at least 250 MHz, with RV, hold and CDC checks intact.
The old 276.3 MHz floor is superseded for K4; historical card measurements are unchanged.
The final-frequency hook accepts 250.0 MHz and rejects 249.99 MHz. At that point the early netlist
check rejected a 4 ns core period for the then-active 300 MHz request; the new 250 MHz request
and matching gate above supersede it. The clock/floorplan
fixtures pass 71 tests after this change; the build and both generated-file checks pass too.

Known current debug-interface gaps remain: RV_SAMPLE/RVWP_APPLY writes acknowledge before
their CDC work completes, and DBG_STATUS.SBA_REFUSED has no RTL setter. These are separate
from the deferred profiler/report work and were outside this correction.

GDB on the C simulator is implemented and verified. The full CTest run's `tools_dbg_gdb`
used the pinned `riscv-none-elf-gdb` through `tsr-dbg gdb --stdio --device sim` and passed
boot/symbol stops, step, memory/CSR reads, a launched kernel hardware breakpoint, backtrace,
`finish`, tile history and detach. The RSP, embedded-service and dropped-connection tests
also passed. This is separate from the RTL module tests; integrated RTL/card GDB remains
unverified. `prof-dbg.md` §1 and the user-facing environment-variable docs now reflect this
status instead of the old P3.7-planned wording.

The broader repository map is [design/README.md](design/README.md) §12: code/contract boundaries,
normal decode and failure/recovery paths, simulator/RTL/card evidence and W0–W6/deferred work.
It reuses the earlier audit and covers all 759 tracked entries by role. Review depth is explicit:
63 body/call-path reviews, 144 specific prior reviews reused, 468 interface inspections,
18 generated files, 55 assets, four vendor boundaries and seven test fixtures without body review.
This is not a claim of exhaustive line/input coverage. The detailed local reports and file-level
records are in `build/analysis/repository_map_20261102/`, referenced from that implementation map.
No runtime/RTL changes or extra build/test/Vivado/DCP/link runs were made for this mapping.

### Next steps

1. Check `status.json`, the stop record and archive hashes before reusing the stopped run.
   No CAD process from this attempt remains. A future zero exit status and a checkpoint still
   need full-route/no-route-error validation before calling the route successful.
2. Functional gates for the shared broadcast head and VU return pass as recorded above.
   Resolve the internal-worker process limit above. Check the new K4 netlist, then placement/physical optimisation at
   core 250 MHz before deciding whether to route. Actual SLL savings, local fanout and new
   ownership on the full placed design remain unverified. Keep all four 512-bit weight streams.
   The shared head could remove about 1,058 duplicated logical boundary-net incidences from
   the measured BC scope; VU sharing removes 256 data connections and adds one token before
   synthesis. Subsequent candidates are per-SLR gather arbitration and the DDR0/3 central
   weight-response detour (`docs/platform-u200.md` §8.6).
   Reducing register depth alone does not reduce a bus's crossing width. Narrowing a weight
   link from 512 to 256 bits at 250 MHz lowers its ideal payload rate from 16 to 8 GB/s;
   an active slow channel can limit the whole equal-row MVM. The earlier combined floorplan changes reduced
   placed SLL counts by 11,301, but did not establish routing success or isolate their causes.
3. An implementation-only resume of the old failed run cannot repair its clock wiring. The corrected
   run rebuilt from the corrected RTL and clock selection. Its preserved placed/physopt DCPs
   have the correct clocks. Native `vpl.impl.phys_opt_design` restart is supported, but needs
   the new PRE hook to restore constraints, K4FP Tcl state and legal LUT placement. A
   changed RTL topology needs a new kernel XO/netlist; the old synthesis cannot be reused
   as if it contained that change. The existing DCPs remain analysis references. A
   `config_hw_runs` reset can invalidate those saved run results; back them up first. A full link takes 7-10 h on
   this host because of swap. A machine with
   64 GB RAM or more (Linux, or WSL2 with Linux Vivado/Vitis 2024.2: the Alveo `v++` flow is Linux
   only) runs it much faster and allows two variants at once.
4. Gate T3: K4 kernel clock at least 250 MHz, per the user's 2026-11-02 decision. A route-only
   stop precedes Vitis's final frequency selection. Check routed timing at the accepted core
   frequency and retain the RV, hold and CDC gates. Do not retry merely for 300 MHz if this
   passes. Known tight paths from the out-of-context placement (WNS -0.614 ns
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

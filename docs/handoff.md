# Handoff: where the work stands (2026-11-01)

Work package numbers (W0-W6, P*.*) are those of [design/README.md](design/README.md) §10.

## Done

| Package | State | Commit |
|---|---|---|
| W0-W3: tables, simulator, firmware SDK (Basis), M6 and M7 backend on the simulator | done; one launch per decode token; KL within the criterion for 0.5B, Gemma 4 12B, 32B | up to `667153f` |
| Versions: Tesseract Core capability 1.0, Basis 0.1.0, ABI 1, kind revisions | done, tag `basis-v0.1.0` | `13dbd2e` |
| llama-server on the backend, TSR_KINDS lists, klcheck multi-seed floor | done | `b58fc03`, `7175f61` |
| W4-A RTL: issue path, 2R2W SPM, control subsystem (Ibex, riscv-dbg, TCMs, CDC) | done, gate T1 met at 166.7 MHz (RV32MFast) | `4149614` |
| W4-B RTL: VU, attention modes, DB block, two-clock XRT adapter; tsr-dbg (GDB), tsr-prof | done; ctest 297/297, lint clean, all benches green | `6435e8a` |
| W5 preparation: VU lane pipes (`tsr_vu_lanes.sv`, `VU_LANE_PIPE`), floorplan C'' (`hw/u200/k4_floorplan.tcl`), TCM RAMB36 constraint in every flow (`tsr_ctrl_tcm.xdc`, `hw/u200/tcm_opt.tcl`), `build_xclbin.sh` enabled for k1/k4 | checkpoint, committed with this file | this commit |

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
  (`docs/rtl-m3.md` §14). The VU lanes are in SLR0 now, yet the SLR1-SLR2 boundary (clusters 1 and
  3, no VU) is the worse one, so the attention additions (per-cluster gather ATT path: job inputs,
  row maximum, row-vector writes; cluster attention data paths) are the likely main cause. Not yet
  measured per net group.
- Artefacts: `build/link/k4/build.log`, `build/link/k4/v++_link.log`,
  `build/link/k4/tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log` (SLL table near "Estimated SLL
  Demand Per Column"), placed and failed-route checkpoint
  `.../impl_1/level0_wrapper_routed_error.dcp`. Opening it needs about 30 GB of memory.

### Next steps

1. Count SLR-crossing nets by hierarchy on the checkpoint (e.g. `get_nets -filter {CROSSING_SLRS != ""}`
   grouped by the driver cell's hierarchy) to confirm which buses dominate.
2. Cut the crossings in RTL and floorplan. Candidates: cross each SLR boundary once per bus and fan
   out inside the SLR (two clusters per outer SLR share one crossing); narrow or time-multiplex the
   gather ATT buses; reduce small results (row maximum, sums) next to the clusters before they cross;
   keep the SLR crossings registered (Laguna). The same kind of fix closed M3 (`docs/rtl-m3.md` §14:
   one cluster's `wd_q` had used 1,560 SLLs).
3. Re-link. The link resumes from synthesis with `TSR_VPP_FROM_STEP=vpl.impl` when only constraints
   change; RTL changes need a full link (7-10 h on this host because of swap). A machine with
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

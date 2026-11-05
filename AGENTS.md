# Notes for coding agents

Read [docs/handoff.md](docs/handoff.md) first: it says where the work stands and what is next.

## Ground rules

- The C simulator in `sw/sim` is the bit-exact oracle. RTL blocks, firmware kinds and the host stack
  are checked against it; a change that moves an output bit must say why (see
  [docs/versioning.md](docs/versioning.md): kind numerics revisions, `fw_kind_revs`).
- Design contract: `docs/spec.md` and `docs/design/*.md` (EX, VU, AT, CC, DB, SW; decision IDs like
  `CC-24`, `SW-53`). Cite sections as `docs/design/04-control-core.md §14`, never by line number.
- Never edit `third_party/` by hand. `third_party/llama.cpp` is intentionally dirty: patch
  `third_party/patches/0001-ggml-tesseract.patch` is applied by `third_party/apply-patches.sh`; a new
  llama.cpp change is a new numbered patch file. `third_party/ibex`, `riscv-dbg`, `common_cells` are
  vendored unmodified.
- Generated files are never edited by hand: `rtl/core/tsr_tables_pkg.sv`, `sw/sim/include/tsr_tables.h`,
  `sw/ctrl/include/tsr/{mmio,vu,at}.hpp`, `tb/common/tables.py` come from `sw/tables/gen_tables.py`;
  `rtl/platform/alveo_xrt/tsr_u200_k*.v` from `gen_tops.py`.
- No sudo. No `pip install` into existing interpreters. Python: `/home/limagin-dev/miniconda3/bin/python`
  (numpy, cocotb); `build/tools/svgvenv` (figures); `build/tools/torchvenv`. `python3` on PATH is an
  unrelated venv.
- Build with `-j8` (`TSR_BUILD_JOBS=8`). Run Vivado through `hw/u200/vivado_slot.sh <cmd>` (at most two
  top-level executions; internal synthesis workers are allowed, per the user's 2026-11-02 clarification).
  Use available logical CPUs for concurrent K4 IP synthesis jobs (currently 16), per the user's
  later 2026-11-02 instruction. `TSR_U200_SYNTH_JOBS` controls that concurrency independently;
  `TSR_BUILD_JOBS=8` still sets build jobs and implementation threads. Internal workers are allowed.
  The host has 14 GB RAM and 72 GB swap; use the existing swap.
  A K4 link peaks at about 33 GB.
- Vivado/Vitis 2024.2 under `/tool/Xilinx` (not on PATH; source the settings script in a subshell),
  XRT under `/opt/xilinx/xrt`. Read and use only.
- For new K4 builds, limit placement constraints to Pblocks for major functional blocks.
  Show the planned Pblocks to the user before launching each build. Regenerate the map if
  the block assignments or platform change. The map shows allowed regions, not placed cells.
  Let Vivado place the logic and route the nets within those regions. Do not assign individual
  LUTs, registers, pipeline stages, sites or BELs, or run helper ownership audits or placement ECOs.
  Use the normal implementation reports and the required clock, CDC and final timing checks.
- Do not commit build outputs (`build/` is ignored), model files (`models/`) or trace data (`traces/*/`
  except scripts and `MANIFEST.sha256`).
- Docs style: plain short sentences, like a long-maintained large repository; no "v1"/"v2" labels for
  the core (there is one design, the one with the RISC-V control core).

## Gates before a commit

```sh
cmake --build build/dev -j8
ctest --test-dir build/dev -LE slow -j8
/home/limagin-dev/miniconda3/bin/python sw/tables/gen_tables.py --check
/home/limagin-dev/miniconda3/bin/python rtl/platform/alveo_xrt/gen_tops.py --check
TSR_VIVADO_WRAP="$PWD/hw/u200/vivado_slot.sh" rtl/lint.sh   # Verilator, XSim, Vivado multi-driven check
TSR_BUILD_JOBS=8 TSR_PYTHON=/home/limagin-dev/miniconda3/bin/python \
  TSR_VIVADO_WRAP="$PWD/hw/u200/vivado_slot.sh" tb/run_all.sh --no-lint
```

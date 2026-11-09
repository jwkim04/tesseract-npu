# Notes for coding agents

Start with [docs/README.md](docs/README.md), the index of the documentation, and
[docs/design/README.md](docs/design/README.md), which says how the design documents fit together.

## Ground rules

- The C simulator in `sw/sim` is the bit-exact oracle. RTL blocks, firmware kinds and the host stack
  are checked against it. A change that moves an output bit must say why (see
  [docs/versioning.md](docs/versioning.md): kind numerics revisions, `fw_kind_revs`).
- Design contract: `docs/spec.md` and `docs/design/*.md` (EX, VU, AT, CC, DB, SW; decision IDs like
  `CC-24`, `SW-53`). Cite sections as `docs/design/04-control-core.md §14`, never by line number.
  Do not renumber a numbered heading: code cites them.
- Basis versions are MAJOR.MINOR.PATCH. MAJOR is a large change (an ABI or interface break), MINOR a
  small addition or change of functions (new kinds, ops, flags, tools, a kind numerics revision),
  PATCH a small fix that leaves every function's inputs and outputs unchanged. Releases are tagged
  `basis-vX.Y.Z` and recorded in `CHANGELOG.md`.
- Never edit `third_party/` by hand. `third_party/llama.cpp` is intentionally dirty: the patch
  `third_party/patches/0001-ggml-tesseract.patch` is applied by `third_party/apply-patches.sh`. A new
  llama.cpp change is a new numbered patch file. `third_party/ibex`, `riscv-dbg` and `common_cells`
  are vendored unmodified.
- Generated files are never edited by hand: `rtl/core/tsr_tables_pkg.sv`, `sw/sim/include/tsr_tables.h`,
  `sw/ctrl/include/tsr/{mmio,vu,at}.hpp` and `tb/common/tables.py` come from `sw/tables/gen_tables.py`;
  `rtl/platform/alveo_xrt/tsr_u200_k*.v` come from `gen_tops.py`.
- No sudo. No `pip install` into existing interpreters. Python: `/home/limagin-dev/miniconda3/bin/python`
  (numpy, cocotb); `build/tools/svgvenv` (figures); `build/tools/torchvenv`. `python3` on PATH is an
  unrelated venv.
- Build with `-j8` (`TSR_BUILD_JOBS=8`). The host has 14 GB RAM and 72 GB swap; use the swap.
- Run Vivado through `hw/u200/vivado_slot.sh <cmd>`, at most two top-level executions at a time.
  Internal synthesis workers are allowed. `TSR_U200_SYNTH_JOBS` sets the number of concurrent K4 IP
  synthesis jobs (use the available logical CPUs, currently 16); `TSR_BUILD_JOBS` sets the build jobs
  and the implementation threads. A K4 link peaks at about 33 GB.
- Vivado/Vitis 2024.2 is under `/tool/Xilinx` (not on PATH; source the settings script in a
  subshell) and XRT under `/opt/xilinx/xrt`. Read and use only.
- K4 builds use placement constraints only as Pblocks for major functional blocks. Show the planned
  Pblocks to the maintainer before launching each build, and regenerate the map when the block assignments
  or the platform change. The map shows allowed regions, not placed cells. Let Vivado place the logic
  and route the nets within those regions. Do not assign individual LUTs, registers, pipeline stages,
  sites or BELs, and do not run helper ownership audits or placement ECOs. Use the normal
  implementation reports and the required clock, CDC and final timing checks.
- Do not commit build outputs (`build/` is ignored), model files (`models/`) or trace data
  (`traces/*/` except scripts and `MANIFEST.sha256`).
- Docs describe what the system is and how to use it, in plain short sentences. They carry no project
  history (dates outside `CHANGELOG.md`, commit hashes, branch names, work packages, milestones) and
  no benchmark results. M6 and M7 are names of execution modes only; `docs/design/README.md` defines
  them. There is one core design, the one with the RISC-V control core: no version labels for it.

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

# hw/viz — floorplan images without the Vivado GUI

Two steps turn a routed checkpoint into a device-view image like Vivado's Device window. Every
used site is drawn at its real position in the colour of the module that owns most of its cells.

```sh
source /tool/Xilinx/Vivado/2024.2/settings64.sh
vivado -mode batch -nojournal -log OUT/vivado.log -source hw/viz/fp_dump.tcl \
  -tclargs <routed.dcp> OUT
build/tools/svgvenv/bin/python hw/viz/fp_render.py OUT out.png --mode core|link --title "..."
```

- `fp_dump.tcl` writes `cells.tsv` (placed primitives: name, site, cell type), `sites.tsv` (every
  site of the part with its RPM position and clock region) and `pblocks.tsv`. It reads each
  property with one call on the whole collection: 2–4 min and 6–8 GB for a full VU9P design,
  mostly to open the checkpoint.
- `fp_render.py --mode core` groups an out-of-context `tsr_core` (clusters, gather, SPM, DMA,
  QACT, arbiters, control, SLR-crossing pipes). `--mode link` groups a full device after the
  `v++` link: XRT static shell, shell glue, memory subsystem (DDR4 controllers, interconnect),
  platform adapter, and the core's modules inside the kernel.
- Legend totals: DSP, RAMB36 (+ RAMB18 / 2) and URAM count used sites and match
  `report_utilization`. LUT and FF count cells, so a LUT6_2 counts as two LUTs.

Checkpoints: the whole-core OOC runs (`tb/core/viv/core_ooc.sh`) write `post_route.dcp`; a `v++`
link leaves `build/u200/<k>/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp`.
The images in `docs/fig/floorplan/` were made from the 2026-10-24 runs (k4 OOC with floorplan C',
and the k1 and M0 links).

## Terminal demo videos (`term2mp4.py`)

`term2mp4.py` turns a `script` recording (util-linux, classic timing) into an MP4. It runs in
real time and keeps every delay of the recording, so the rates on screen are the measured ones.
It uses pyte as the terminal emulator, Pillow and the ffmpeg of imageio-ffmpeg, all in
`build/tools/svgvenv`.

```sh
script -q -T demo.tim -c 'stty cols 132 rows 36; ./demo.sh' demo.raw
build/tools/svgvenv/bin/python hw/viz/term2mp4.py demo.raw demo.tim demo.mp4 --cols 132 --rows 36 --fps 20 --size 16
```

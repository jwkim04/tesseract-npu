# Floorplan images and terminal videos (`hw/viz`)

Two steps turn a routed Vivado checkpoint into a device-view image like Vivado's Device window,
without the GUI. Every used site is drawn at its real position in the colour of the module that owns
most of its cells. A third script turns a terminal recording into a video.

> **Scope.** Tool documentation. The images in `docs/fig/floorplan/` are **v1**: they show the v1
> core (tag `v1-m4`) as routed by Vivado 2024.2; the v2 core has no RTL yet, so there is no v2
> floorplan. The v2 placement is a design item, not an image: [CC](../../docs/v2/04-control-core.md)
> §14.2. The scripts need Vivado only for `fp_dump.tcl`; `fp_render.py` and `term2mp4.py` run in the
> figure environment `build/tools/svgvenv`.

```sh
source /tool/Xilinx/Vivado/2024.2/settings64.sh
vivado -mode batch -nojournal -log OUT/vivado.log -source hw/viz/fp_dump.tcl \
  -tclargs <routed.dcp> OUT
build/tools/svgvenv/bin/python hw/viz/fp_render.py OUT out.png --mode core|link --title "..."
```

- `fp_dump.tcl` writes `cells.tsv` (placed primitives: name, site, cell type), `sites.tsv` (every
  site of the part with its RPM position and clock region) and `pblocks.tsv`. It reads each
  property with one call on the whole collection: 2 to 4 min and 6 to 8 GB for a full VU9P design,
  mostly to open the checkpoint.
- `fp_render.py` takes the dump directory and the output image (`dump`, `out`), a required `--title`
  and `--mode {core,link}` (`build/tools/svgvenv/bin/python hw/viz/fp_render.py --help`).
  `--mode core` groups an out-of-context `tsr_core` (clusters, gather, SPM, DMA, QACT, arbiters,
  control, SLR-crossing pipes). `--mode link` groups a full device after the `v++` link: XRT static
  shell, shell glue, memory subsystem (DDR4 controllers, interconnect), platform adapter, and the
  core's modules inside the kernel.
- Legend totals: DSP, RAMB36 (+ RAMB18 / 2) and URAM count used sites and match
  `report_utilization`. LUT and FF count cells, so a LUT6_2 counts as two LUTs.

## The images

All three were made on 2026-10-24 from routed checkpoints of the v1 core (measured in Vivado 2024.2,
v1):

| Image | Checkpoint | Shows |
| --- | --- | --- |
| `docs/fig/floorplan/k4-core-ooc.png` | the out-of-context run of the k4 core (`tb/core/viv/core_ooc.sh`, which writes `post_route.dcp`), floorplan C' of `hw/u200/k4_floorplan.tcl` | clusters 0 and 2 in SLR0, 1 and 3 in SLR2, the rest of the core in SLR1 |
| `docs/fig/floorplan/k1-linked.png` | the `tsr_u200_k1` link (`build/u200/k1/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp`) | the one-channel kernel inside the XRT shell |
| `docs/fig/floorplan/bw-linked.png` | the `tsr_bw_k4` link (the M0 bandwidth kernel) | the memory-bandwidth kernel inside the shell |

A `v++` link leaves its routed checkpoint at
`build/u200/<k>/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp` (not in git). The
linked k4 design (276.3 MHz, the card's bitstream) has no image; its placement and timing are in
[`docs/platform-u200.md`](../../docs/platform-u200.md) §8 and §14 and
[`docs/performance.md`](../../docs/performance.md).

## Terminal demo videos (`term2mp4.py`)

`term2mp4.py` turns a `script` recording (util-linux, classic timing) into an MP4. It runs in
real time and keeps every delay of the recording, so the rates on screen are the measured ones.
It uses pyte as the terminal emulator, Pillow and the ffmpeg of imageio-ffmpeg, all in
`build/tools/svgvenv`. The card demo videos were made with it (v1 card runs).

```sh
script -q -T demo.tim -c 'stty cols 132 rows 36; ./demo.sh' demo.raw
build/tools/svgvenv/bin/python hw/viz/term2mp4.py demo.raw demo.tim demo.mp4 --cols 132 --rows 36 --fps 20 --size 16
```

Arguments: `typescript timing out`, and `--cols`, `--rows`, `--fps`, `--size` (font size in px); `-h`
lists them.

# Floorplan images and terminal videos (`hw/viz`)

Two steps turn a routed Vivado checkpoint into a device-view image like Vivado's Device window, without the
GUI. Every used site is drawn at its real position in the color of the module that owns most of its cells. A
third script draws the Pblock map of an image from its placement hook, and a fourth turns a terminal
recording into a video. `fp_dump.tcl` needs Vivado. The other scripts run in the figure environment
`build/tools/svgvenv`.

```sh
source /tool/Xilinx/Vivado/2024.2/settings64.sh
vivado -mode batch -nojournal -log OUT/vivado.log -source hw/viz/fp_dump.tcl \
  -tclargs <routed.dcp> OUT
build/tools/svgvenv/bin/python hw/viz/fp_render.py OUT out.png --mode core|link --title "..."
```

- `fp_dump.tcl` writes `cells.tsv` (placed primitives: name, site, cell type), `sites.tsv` (every site of the
  part with its RPM position and clock region) and `pblocks.tsv`. It reads each property with one call on the
  whole collection. A full VU9P design takes a few minutes and several GB, mostly to open the checkpoint.
- `fp_render.py` takes the dump directory and the output image (`dump`, `out`), a required `--title` and
  `--mode {core,link}` (`--help` lists the options). `--mode core` groups an out-of-context `tsr_core`
  (clusters, gather, SPM, DMA, QACT, arbiters, control, SLR-crossing pipes). `--mode link` groups a full device
  after the `v++` link: the XRT static shell, shell glue, the memory subsystem (DDR4 controllers,
  interconnect), the platform adapter, and the core's modules inside the kernel.
- The legend totals for DSP, RAMB36 (plus RAMB18 / 2) and URAM count used sites and match
  `report_utilization`. LUT and FF count cells, so a LUT6_2 counts as two LUTs.

A `v++` link leaves its routed checkpoint at
`build/u200/<k>/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp`, which is not in git. The
out-of-context run of the core (`tb/core/viv/core_ooc.sh`) writes `post_route.dcp`.

## Pblock map (`k4_pblock_preview.py`)

`k4_pblock_preview.py` draws the allowed regions of the Pblock plan of `tsr_u200_k4`
([docs/platform-u200.md](../../docs/platform-u200.md) §8.1). It interprets the hook
`hw/u200/k4_major_blocks.tcl` with a Tcl mock, so it needs no Vivado, checkpoint or placement. The Pblock
shapes come from a properties file that `tb/platform/viv/k4_pblock_check.tcl` writes (`PBLOCK` sections with
`DERIVED_RANGES`, and an optional `CAPACITY` line): the platform's three SLR Pblocks and the Pblocks the hook
creates, as Vivado resolves them.

```sh
build/tools/svgvenv/bin/python hw/viz/k4_pblock_preview.py hw/u200/k4_major_blocks.tcl <properties> <png>
```

The colors show where a block may be placed, not where it is placed. The outlines mark the clock regions of
the platform's DDR0 and DDR3 controllers.

## Terminal demo videos (`term2mp4.py`)

`term2mp4.py` turns a `script` recording (util-linux, classic timing) into an MP4. It runs in real time and
keeps every delay of the recording, so the rates on screen are the recorded ones. It uses pyte as the terminal
emulator, Pillow and the ffmpeg of imageio-ffmpeg, all in `build/tools/svgvenv`. Box drawing and block
characters are drawn so that lines meet across cells. Glyphs that the mono font lacks come from DejaVu Sans
Mono, and emoji from Noto Color Emoji.

```sh
script -q -T demo.tim -c 'stty cols 132 rows 36; ./demo.sh' demo.raw
build/tools/svgvenv/bin/python hw/viz/term2mp4.py demo.raw demo.tim demo.mp4 --cols 132 --rows 36 --fps 20 --size 18
```

The arguments are `typescript timing out` and the options `--cols`, `--rows`, `--fps`, `--size` (font size
in pixels), `--hold` (seconds to hold the last screen) and `--ff START:END:SECONDS`. `--ff` plays the span
START..END of the recording (seconds from its start) in SECONDS of video and shows `sped up Nx` in the top
right corner while it plays. Use it for a model load, never for a span that shows a rate. `-h` lists the
options.

To record a session with several panes, run it in tmux on a private socket (`tmux -L NAME`), attach the
client inside `script` and type into a pane with `tmux send-keys`. The terminal type of the client must be
`screen-256color` or `tmux-256color`, since pyte does not implement the repeat-character sequence of
`xterm-256color`.

An animated image for a README comes from the MP4 with two ffmpeg passes in one command (the ffmpeg binary is
`imageio_ffmpeg.get_ffmpeg_exe()`): a shared palette, then the frames at 12.5 fps, which is a whole number of
centiseconds per frame.

```sh
ffmpeg -i demo.mp4 -vf "fps=12.5,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];\
[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 demo.gif
```

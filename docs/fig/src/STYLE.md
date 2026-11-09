# Figure style guide

The figures in `docs/fig/*.svg` are drawn as illustrations, not flowcharts. The look follows
Tenstorrent's architecture art: a warm card, a lighter inner panel, flat pastel blocks with cut
corners and a thin dark outline, fat hollow arrows, and small uppercase callouts on leader lines.
Every figure is a Python script in this directory built on `tsrfig.py`. Use the library for every
mark, so the figures stay consistent and the output stays deterministic.

## Files and running

| What | Where |
| --- | --- |
| Figure script | `docs/fig/src/<name>.py` (one figure per script, `Fig("<name>", ...)`) |
| Output | `docs/fig/<name>.svg` (committed); the README figures also have `docs/fig/<name>.png` (see "README figures") and `docs/fig/<name>-dark.svg` (see "Dark variants") |
| Previews | `$TSR_FIG_PNG_DIR/<name>.png` at 1:1 and `<name>_gh.png` at 880 px (about GitHub's README width). The default is `build/fig/png`, which git ignores. Previews are never committed. |
| Interpreter | `build/tools/svgvenv/bin/python docs/fig/src/<name>.py` (needs cairosvg, fontTools) |
| Logos | `docs/fig/logos/<key>.svg` (sanitised third-party logos) and `docs/fig/logos/SOURCES.md` (where each came from); `docs/fig/src/logoclean.py` makes the files. See "Logos". |
| README PNGs | `docs/fig/<name>.png` for the figures in `README_FIGS` of `pngs.py`, made from the light SVGs |
| All figures | `docs/fig/src/build.sh` runs every script, light and dark where a dark variant exists, then `pngs.py` |

- Start each script with `sys.dont_write_bytecode = True` before `from tsrfig import ...`, so
  no `__pycache__` is left in the repo.
- Look at both previews after every change. If text is hard to read in `_gh.png`, the figure is
  too dense.
- Output must be deterministic: no timestamps, random numbers or set iteration in the drawing
  order. Running a script twice must give the same SVG bytes.
- Text is converted to outlines by the library, and the SVG never uses `<text>`. Do not add raw
  `<text>` with `Fig.raw`.

## Canvas

- Card: 1600 units wide; the height is whatever the content needs, rounded to a multiple of 8. The
  card is a rounded rectangle (radius 36, outline 2) that carries its own background, so the figure
  reads the same on GitHub's light and dark themes.
- Outer padding: keep at least 56 between the card edge and the content.
- Panels: chamfered (cut 20), lighter (`PANEL`) and outlined, inset in the card. Name a panel with
  its dark tag on the top-left edge (`panel(..., label="HOST PC")`). A region nested in a panel, such
  as `tsr_core` inside the board, is an outline-only panel (`fill=None`). When an arrow crosses the
  top edge where the tag would sit, draw the panel without a label and place the tag yourself with
  `tag(x, y, ..., anchor="start")` further right (`layers`: TSR_CORE; `verify`: RTL VS SIMULATOR).
  Exceptions: `alloc` centres its three panel tags because its layout is symmetric and a top-left
  tag would hit the ch0 arrowhead, and `hero` centres the tag of its zoom panel.
- Callout margins: if a figure has callouts on leader lines, leave about 224 units free on the side
  that holds them. `system` uses both sides: its panels run from x 224 to 1376. Callout text should
  stay at least 20 units inside the card edge.
- Scale check: GitHub shows the card at about 0.55×, so 19 units become about 10 px. Keep body text
  at 17 or more; use 15 only for tags.
- Long figures: a swimlane or a stack may be taller than it is wide (`launch`, `stack`, `verify`:
  2000 units and more). Keep the width and the type scale, and stack panels instead of shrinking
  text.
- Large type: `stack` is the product-style figure and uses the `*_l` roles of the type table
  (28 / 21 / 17 / 19 units) with fewer words per tile, so it stays readable at README width. At
  phone width every 1600-unit figure is small; the SVG scales without loss, and the product
  figures (`hero`, `stack`, `system`) keep their text short for that reason.

## Palette

Fills mark a block's role. Use the role, not a raw hex, so all figures agree.

| Token / role | Hex | Use for |
| --- | --- | --- |
| `INK` | `#2B2B2B` | outlines, text, dots, dark tags |
| `INK_SOFT` | `#5C5B55` | secondary text: band labels, footnotes |
| `CARD` | `#ECEBE4` | outer card |
| `PANEL` | `#F8F8F5` | inner panels |
| `BAND` | `#E7E5DC` | flat tinted regions inside a panel (SLRs, lanes, pipeline stages) |
| `ARROW` | `#EAF4F1` | fill of hollow arrows |
| `control` | `#A8E0D5` mint | sequencing and control: the control core, issue path, checker, dispatcher, CSRs, FSMs, and firmware (the kernels and the runtime that run on the control core) |
| `compute` | `#9EDCF2` sky | MVM clusters, MAC lanes, fp units, functional units |
| `memory` | `#F8AFA3` salmon | SPM, DRAM channels, buffers, FIFOs |
| `io` | `#DCD5F7` lavender | data movement: DMA (LD, ST), weight-stream requester, broadcast / gather buses and pipes, arbiters, ports |
| `host` | `#F4DE9C` butter | host software: llama.cpp, `ggml-tesseract`, `tsr_rt`, transport, simulator, code under test in `verify` |
| `adapter` | `#F9C89B` peach | per-target platform adapter: control bridge, AXI4 bridges |
| `vendor` | `#DAD8CF` stone | vendor-specific parts: the XRT shell, PCIe, tech wrappers |
| `neutral` | `#FFFFFF` | plain blocks, such as data items in a memory map; references in `verify`; developer tools in `stack` |

Do not use gradients, shadows, transparency or 3D effects. Text is always `INK` (or `INK_SOFT`
for secondary notes) on the pastel fills; on dark tags it is `CARD`. The one exception is a
third-party logo, which keeps its own colours (see "Logos").

## Strokes and shapes

| Element | Value |
| --- | --- |
| Block, panel and card outline, hollow arrow outline | 2, miter joins |
| Thin connector | 2, solid arrowhead 12 long × 11 wide |
| Callout leader | 1.5, ending in a dot of radius 5 |
| Dash pattern (unit-queue buses, reset lines, zoom lines) | `9 6` |
| Block corner cut | 14 (`CUT`); 10–12 for blocks under about 64 tall |
| Panel corner cut | 20 (`CUT_PANEL`) |
| Tag corner cut | 5 |

## Type

The only font is Noto Sans Mono (see `FONTS.md`), with a slashed zero. "Medium" is Regular plus
a same-colour stroke of 34 font units; the library does this.

| Role (`TYPE` key) | Size | Weight | Tracking | Use |
| --- | --- | --- | --- | --- |
| `title` | 28 | bold | 0.02 em | optional heading inside the card (most figures have none; the README gives the title) |
| `label` | 24 | bold | 0.02 em | block title, UPPERCASE, 1–2 lines |
| `label_s` | 21 | bold | 0.02 em | block title in a small or crowded block |
| `sub` | 19 | regular | 0 | block detail lines under the title |
| `callout` | 19 | medium | 0.02 em | leader-line callouts and arrow names, UPPERCASE |
| `panel` | 20 | bold | 0.10 em | band labels (`SLR0`), UPPERCASE, `INK_SOFT` |
| `note` | 17 | regular | 0 | legend entries, footnotes |
| `tag` | 15 | bold | 0.08 em | badges: panel names, SLR names, `× 4` |
| `label_l` | 28 | bold | 0.02 em | tile title in `stack` |
| `sub_l` | 21 | regular | 0 | tile detail lines in `stack` |
| `panel_l` | 21 | bold | 0.10 em | layer names in `stack`, `INK_SOFT` |
| `note_l` | 19 | regular | 0 | layer captions and the footnote in `stack` |
| `tag_l` | 17 | bold | 0.08 em | badges and layer names in `stack` |

Casing:

- Block titles, callouts, tags and band labels are UPPERCASE. Examples: `ISSUE PATH`,
  `WEIGHT STREAM`, `TSR_CORE`.
- Detail lines (`sub`) are lower or sentence case.
- Code identifiers keep their case everywhere except block titles: `tsr_mem_arb`, `m_axi_ch0`,
  `s_axi_control`. So do units: `4 MiB`, `64 B`, `19.2 GB/s`, `300 MHz`.
- Put a block title on two centred lines when it is long (`MEMORY\nARBITER`).

Glyphs:

- Avoid `→` and `↔`: they are tiny in this font. Write `<->`, draw an arrow, or use words.
- Use `·` as a separator, `×` for counts and products, and `≥` / `≤`. `–` is fine in ranges
  (`R0–R15`).
- Callout lines should be at most about 15 characters, so they fit the 224-unit margins.

## Spacing

- Put every coordinate and size on the 8-unit grid (`GRID`); use 4 only for fine nudges.
- Keep at least 20 units between a block's edge and its text, and at least 48 between blocks.
- Arrow lengths: at least 56 between blocks for a one-headed hollow arrow, and 80 for a
  two-headed one, so some shaft shows between the heads.
- Size blocks to their text with `measure()`. Never let text touch an outline or an arrow.

## Arrows and lines

| Kind | Call | Use for |
| --- | --- | --- |
| Hollow arrow | `arrow(pts)` (shaft 24, head 54 × 26) | normal data or command flow between blocks. `tail=True` makes it two-way. Elbow it with more points (axis-aligned only). |
| Fat hollow arrow | `arrow(pts, fat=True)` (shaft 40, head 84 × 36) | bulk data: weight streams and DRAM beat streams. Use it for the traffic the figure is about. |
| Thin connector | `line(pts)` (solid head; `start=True` adds a head at the start) | side paths and signals: `done`, control strobes, a shared port. Make it dashed when the path is secondary or shared (`core`: the unit-queue bus). |
| Leader + dot | `callout(text, xy, target, via=...)` | annotation only, never flow. The dot sits on the thing named. Leaders are horizontal or vertical, with at most two bends. |
| Bracket | `bracket(x1, x2, y, label)` | a span: a group of channels, `× NUM_CH`, a stage range |

- Hollow arrows go on top of the blocks they join. They start and end 8–12 units inside each
  block, and the text in the block must stay clear of that overlap (at least 10 units from a
  baseline or cap top). Where a block is short, start the arrow 6 inside or make the block taller.
- An arrow drawn before a panel is hidden by the panel's fill. Draw arrows that end inside a
  panel after that panel (`alloc`: the weight streams).
- To pass "through" a block (a stream through a bridge), draw the arrow first and the block over
  it.
- Name an arrow with a `callout`-role text beside it (`AXI4-LITE`, `CSR PORT · DONE`), without a
  leader. Use a leader only for a longer explanation.

## What a figure shows

A figure shows what the release contains. Every block, arrow and path is drawn solid in its role
colour. A figure carries no status, history, milestone, date or benchmark number: no "planned",
"simulator only" or "on the card" tags, no paler or dashed blocks, no work package names. A block
that is not in `rtl/` and in the release image is not drawn.

- A block that has a mode (DMA with KVST, SPM with 2R2W banks, the MAC lanes with ATTQK and ATTPV)
  names it in its detail lines.
- Dashed lines are for paths, never for blocks: the unit-queue bus in `stack`, `core` and
  `dataflow`, the reset lines in `clocks`, zoom lines and the `W_OFF` line.
- A fact that is a configuration of the release image (core clock 200 MHz, `clk_rv` 100 MHz) is
  drawn as a fact. Do not draw a measured result.
- A figure that shows other targets (`layers`) draws the pieces a port writes, as `docs/spec.md`
  §2 lists them, in the same solid style. The lane heads say which target this tree provides.

## Repeated units and channel counts

- Channels: when the count is small and the layout means something (four DDR4 channels, each with
  its own cluster), draw every instance in its own column and label it (`CH0`–`CH3`,
  `CLUSTER 0`–`3`).
- Many identical units: draw one block with `stack=2` (two copies behind it, 8 up and right each)
  and a dark count tag such as `× 4` or `× NUM_CH`. Put the tag on the back copy
  (`tag(x + w + 16, y - 16, ..., anchor="end")`) when an arrow meets the block's top edge. A
  repeated panel uses the same offsets (`cluster`). Alternatively, put a `bracket` labelled
  `× NUM_CH` under a row, or say "one per cluster" in the block's detail line.
- Parameters: show them by name where `docs/spec.md` §3 does (`NUM_CH`, `T_MAX`, `MVM_BC_PIPE`).
  Give the U200 value only when it is fixed (`NUM_CH = 4`).

## Bands and regions

- Use `band()` for flat, outline-free regions behind blocks: SLRs, pipeline stages, clock
  domains.
- Label a band at its bottom or top in the `panel` role, in `INK_SOFT`.
- Keep a gap of at least 12 between adjacent bands.
- Draw bands before the blocks they hold. If an outline-only panel crosses them, draw it after
  the bands.
- Region tags instead of bands: use these when the regions are not contiguous in the layout. In
  `system`, the Pblock plan of the release image (`hw/u200/k4_major_blocks.tcl`) puts clusters 0
  and 2 in SLR0 and clusters 1 and 3 in SLR2, so clusters 1 and 2 sit away from their channels'
  SLR (SLR1), and the regions are not contiguous in the layout. Give each block a dark tag on the
  right of its top edge (`box(..., tag="SLR1")`). Run the arrows that meet that edge 32 left of the
  block's centre, so the tag stays clear of them. Leave a block untagged when the plan leaves it
  to Vivado (`system`: the XRT shell, the control bridge, the bridges of channels 1 and 2). Say in
  the footnote what the tags show.

## Legend

A row of swatches at the bottom of the card, made with `legend()`, is optional. Use one when a
figure relies on colour roles that the text around it does not explain. List only the roles the
figure uses. Name roles, not instances, with the same words in every figure:

| Role | Legend label |
| --- | --- |
| `control` | `CONTROL` |
| `compute` | `COMPUTE` |
| `memory` | `MEMORY` |
| `io` | `DATA MOVEMENT` |
| `host` | `HOST SOFTWARE` |
| `adapter` | `PLATFORM ADAPTER` |
| `vendor` | `VENDOR-SPECIFIC` |

`verify` is the one figure about tests rather than hardware, and it keeps its own small key:
`REFERENCE` (neutral), `CODE UNDER TEST` (host), `CHECK` (mint), `TRACES` (salmon) and `RTL`
(sky). It never reuses lavender, so a reader carrying the key over from the architecture figures
is not misled. `stack` has no legend.

## Logos

A third-party logo is the one mark in a figure that is not drawn in the figure style: it is the
official file, in its own colours, unmodified. Only `stack` has logos (llama.cpp, PyTorch,
Hugging Face); the architecture figures have none.

- Which: only software that the figure names and that a reader would look for: a program, library
  or tool. Never a mark that implies endorsement or needs permission: no OpenAI logo (the
  `llama-server` API is written as text, "OpenAI-compatible API"), no AMD, Xilinx or Alveo logo, no
  RISC-V International logo, no lowRISC or Ibex logo, and no logo of the card, the board or the
  tools that built it. Hardware, ISAs and IP cores stay as text blocks.
- Official and unmodified: take the file from the project's own repository or brand page. No
  recolouring, no stretching (`Fig.logo` keeps the aspect ratio), no cropping beyond the file's own
  padding, no effects, no partial use of a lockup. Do not mix in an older version of a mark (the
  `pytorch-logo-flame.svg` in the PyTorch repository is the old flame, so the lockup is used).
- Sources are written down: every logo has a section in `docs/fig/logos/SOURCES.md` with the URL
  (pinned to a commit where there is one), the sha256 of the download, the licence or trademark
  note (and any attribution a licence asks for), and what was changed. Read the owner's brand rules
  first; if they forbid the use, leave the logo out. Say in the note that the logo only refers to
  the project.
- Sanitised vector, no raster: the download is untrusted data. Put it in its own empty directory
  outside the repository, never run anything from it, and turn it into `docs/fig/logos/<key>.svg`
  with `logoclean.py` (it refuses scripts, event handlers, external links, `<foreignObject>`,
  `<image>`, `<text>` and DOCTYPE or ENTITY declarations, and folds CSS into attributes so two
  logos cannot collide). `logoclean.py --compare` must show 0 differing pixels against the
  download. A logo that cannot be sanitised (a raster, or an SVG with text) is not used.
- Drawing: `Fig.logo("<key>", x, y, h=...)` embeds the file once in `<defs>` and places it with
  `<use>`, so the SVG stays self-contained and the bytes are the same on every run. It returns the
  size of the mark, and `logo_box()` gives that size before drawing, for layout.
- Backing tile: a logo that would be illegible on the block colour, because it is dark (the PyTorch
  wordmark), white, or close in hue to the fill (the yellow Hugging Face face on butter), uses
  `backing=True`: a white rounded tile (radius 10, outline 1.5 in `INK`, padding 4 to 10). The tile
  is white on both themes, and the logo never changes with the theme. A logo that carries its own
  tile (the llama.cpp icon is a dark rounded square) has no backing.
- Size and place: at least 40 units tall including the backing (about 22 px at README width); 80
  for a square mark, 44 for a wide lockup. Put it at the left of its tile, 24 from the edge and
  centred in the tile's height, with the tile's text left-aligned beside it, 20 from the logo and
  at least 20 from the right edge (`stack.py`: `logo_tile()`). A lockup that contains the name
  replaces the tile title (PyTorch). A small logo may head a line of text (Hugging Face, 32 units on
  a 40-unit tile). Draw each program once in a figure.
- Check: look at `<name>.png` and `<name>_gh.png`, light and dark. The logo must be recognisable at
  880 px, clear of the tile's outline and tag, and the text beside it at least 20 units from the
  tile edge.

## README figures

`README.md` shows the figures listed in `README_FIGS` of `pngs.py` as one plain `<img>` of a PNG
each, inside a centred `<p>`. Do not use a `<picture>` element with SVG sources there: the GitHub
mobile app shows a broken image for it.

- Source: the light SVG is the source, rendered by `docs/fig/src/pngs.py` to `docs/fig/<name>.png`.
  The card carries its own background, so one PNG reads on both GitHub themes; its rounded corners
  are transparent. `build.sh` calls `pngs.py` after the figure scripts, so the PNGs are regenerated
  with the SVGs. `pngs.py <name> ...` renders only the named figures.
- Format: 1760 px wide (twice the 880 px README column), 8-bit palette (256 colours, octree, no
  dithering) with a transparency table, which keeps each file at 100 to 150 KB. The script fails over
  500 KB. There are no timestamps or text chunks, so the same SVG gives the same bytes. Commit the
  PNGs together with the SVGs.
- Embed: `<p align="center"><img src="docs/fig/<name>.png" alt="..." width="100%"></p>`, with the
  alt text that describes the figure.
- Technical documents: the figures under `docs/` (`spec.md`, `ctrl-riscv.md` and the others) embed
  the SVGs with a plain `<img>`. A new README figure needs its name in `README_FIGS`, a line in
  `build.sh` for its script, and the embed above.

## Dark variants

The README figures (`hero`, `stack`, `system`) also exist as `<name>-dark.svg`. The README does
not use them (see "README figures"). The card carries its own background, so the light files read
fine on GitHub's dark theme; the dark files are for pages that want the figure to match the theme
and that can use a `<picture>` element. The same script draws both: `tsrfig.py` reads
`TSR_FIG_THEME` (`light` by default, `dark`) once at import and swaps the palette; nothing else
changes, so the layout, the text and the roles are identical.

| Token / role | Dark hex | Token / role | Dark hex |
| --- | --- | --- | --- |
| `INK` | `#ECEBE4` | `io` | `#574D8B` |
| `INK_SOFT` | `#B4B3A9` | `host` | `#76651F` |
| `CARD` | `#1D1E1B` | `adapter` | `#8F5528` |
| `PANEL` | `#292A26` | `vendor` | `#55544C` |
| `BAND` | `#33342F` | `neutral` | `#3C3D38` |
| `ARROW` | `#33403C` | `control` | `#2D6A5F` |
| `compute` | `#276683` | `memory` | `#8E4539` |

Text is `INK` on every role fill with a contrast of at least 4.8 : 1; the dark tags are `INK` on
`CARD`. Logos and their white backing tiles are the same in both themes. A page that uses the dark files embeds them like
this (not the README, and not for a GitHub page that has to show in the mobile app):

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/fig/hero-dark.svg">
  <img src="docs/fig/hero.svg" alt="..." width="100%">
</picture>
```

## Accuracy

Every block, arrow, label and number must be backed by `docs/spec.md`, `docs/design/*` (the normative
design), `docs/rtl-core.md`, `docs/rtl-mvm.md`, `docs/platform-u200.md`, `docs/xrt-host.md`, the
`rtl/core` sources or the code under `sw/`. The script's docstring names its sources by document
and section, never by line number. Cite English documents only. When a figure adds a fact that the
cited documents do not show, say where it comes from in the docstring. Design facts follow
`docs/design` (4 channels, 128 lanes per cluster, SPM 4 MiB; the release image has `T_MAX` 4, core
clock 200 MHz and `clk_rv` 100 MHz). If a figure and the code disagree, the code wins.

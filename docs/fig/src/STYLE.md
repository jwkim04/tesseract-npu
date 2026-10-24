# Figure style guide

The architecture figures in `docs/fig/*.svg` are drawn as illustrations, not flowcharts. The model
is Tenstorrent's architecture art: a warm card, a lighter inner panel, flat pastel blocks with cut
corners and a thin dark outline, fat hollow arrows, and small uppercase callouts on leader lines.
Every figure is a Python script in this directory built on `tsrfig.py`. Use the library for every
mark, so the figures stay consistent and the output stays deterministic.

## Files and running

| What | Where |
| --- | --- |
| Figure script | `docs/fig/src/<name>.py` (one figure per script, `Fig("<name>", ...)`) |
| Output | `docs/fig/<name>.svg` (committed) |
| Previews | `$TSR_FIG_PNG_DIR/<name>.png` at 1:1 and `<name>_gh.png` at 880 px (about GitHub's README width). The default is `build/fig/png`, which git ignores. Previews are never committed. |
| Interpreter | `build/tools/svgvenv/bin/python docs/fig/src/<name>.py` (needs cairosvg, fontTools) |

- Start each script with `sys.dont_write_bytecode = True` before `from tsrfig import ...`, so
  no `__pycache__` is left in the repo.
- Look at both previews after every change. If text is hard to read in `_gh.png`, the figure is
  too dense.
- Output must be deterministic: no timestamps, random numbers or set iteration in the drawing
  order. Running a script twice must give the same SVG bytes.
- Text is converted to outlines by the library, and the SVG never uses `<text>`. Do not add raw
  `<text>` with `Fig.raw`.

## Canvas

- **Card.** It is 1600 units wide; the height is whatever the content needs, rounded to a
  multiple of 8. The card is a rounded rectangle (radius 36, outline 2) that carries its own
  background, so the figure reads the same on GitHub's light and dark themes.
- **Outer padding.** Keep at least 56 between the card edge and the content.
- **Panels.** These are chamfered (cut 20), lighter (`PANEL`) and outlined, inset in the card.
  Name a panel with its dark tag on the top-left edge (`panel(..., label="HOST PC")`). A region
  nested in a panel, such as `tsr_core` inside the board, is an outline-only panel
  (`fill=None`). When an arrow crosses the top edge where the tag would sit, draw the panel
  without a label and place the tag yourself with `tag(x, y, ..., anchor="start")` further right
  (`layers`: TSR_CORE; `verify`: RTL VS SIMULATOR). Exceptions: `alloc` centres its three panel
  tags because its layout is symmetric and a top-left tag would hit the ch0 arrowhead, and `hero`
  centres the tag of its zoom panel.
- **Callout margins.** If a figure has callouts on leader lines, leave about 224 units free on
  the side that holds them. `system` uses both sides: its panels run from x 224 to 1376.
  Callout text should stay at least 20 units inside the card edge.
- **Scale check.** GitHub shows the card at about 0.55×, so 19 units become about 10 px. Keep
  body text at 17 or more; use 15 only for tags.

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
| `control` | `#A8E0D5` mint | sequencing and control: command processor, fetch, CSRs, FSMs |
| `compute` | `#9EDCF2` sky | MVM clusters, MAC lanes, fp units, functional units |
| `memory` | `#F8AFA3` salmon | SPM, DRAM channels, buffers, FIFOs |
| `io` | `#DCD5F7` lavender | data movement: DMA (LD, ST), weight-stream requester, broadcast / gather buses and pipes, arbiters, ports |
| `host` | `#F4DE9C` butter | host software: llama.cpp, `ggml-tesseract`, transport, simulator |
| `adapter` | `#F9C89B` peach | per-target platform adapter: control bridge, AXI4 bridges |
| `vendor` | `#DAD8CF` stone | vendor-specific parts: the XRT shell, PCIe, tech wrappers |
| `neutral` | `#FFFFFF` | plain blocks, such as data items in a memory map; references in `verify` |

Do not use gradients, shadows, transparency or 3D effects. Text is always `INK` (or `INK_SOFT`
for secondary notes) on the pastel fills; on dark tags it is `CARD`.

## Strokes and shapes

| Element | Value |
| --- | --- |
| Block, panel and card outline, hollow arrow outline | 2, miter joins |
| Thin connector | 2, solid arrowhead 12 long × 11 wide |
| Callout leader | 1.5, ending in a dot of radius 5 |
| Dash pattern (planned, optional) | `9 6` |
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
| `tag` | 15 | bold | 0.08 em | badges: panel names, `PLANNED · M6`, `× 4` |

**Casing.**

- Block titles, callouts, tags and band labels are UPPERCASE. Examples: `COMMAND PROCESSOR`,
  `WEIGHT STREAM`, `TSR_CORE`.
- Detail lines (`sub`) are lower or sentence case.
- Code identifiers keep their case everywhere except block titles: `tsr_mem_arb`, `m_axi_ch0`,
  `s_axi_control`. So do units: `4 MiB`, `64 B`, `19.2 GB/s`, `300 MHz`.
- Put a block title on two centred lines when it is long (`COMMAND\nPROCESSOR`).

**Glyphs.**

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
| Dashed hollow arrow | `arrow(pts, dashed=True)` | flow to or from a planned block |
| Thin connector | `line(pts)` (solid head; `start=True` adds a head at the start) | side paths and signals: instruction fetch, `done`, control strobes, a shared port. Make it dashed when the path is secondary or shared (`system`: fetch / LD / ST on ch0). |
| Leader + dot | `callout(text, xy, target, via=...)` | annotation only, never flow. The dot sits on the thing named. Leaders are horizontal or vertical, with at most two bends. |
| Bracket | `bracket(x1, x2, y, label)` | a span: a group of channels, `× NUM_CH`, a stage range |

- Hollow arrows go on top of the blocks they join. They start and end 8–12 units inside each
  block, as in the reference art, and the text in the block must stay clear of that overlap
  (at least 10 units from a baseline or cap top). Where a block is short, start the arrow 6
  inside or make the block taller.
- An arrow drawn before a panel is hidden by the panel's fill. Draw arrows that end inside a
  panel after that panel (`alloc`: the weight streams).
- To pass "through" a block (a stream through a bridge), draw the arrow first and the block over
  it.
- Name an arrow with a `callout`-role text beside it (`AXI4-LITE`, `CSR PORT · DONE`), without a
  leader. Use a leader only for a longer explanation.

## Planned and unconfirmed things

- **Planned blocks.** Use `box(..., planned=True, tag="PLANNED · M6")`. This gives a dashed
  outline, a fill 55 % paler than the role colour, and a light dashed tag on the top edge. Always
  name the milestone in the tag.
- **Planned paths.** Draw them as dashed arrows or connectors.
- **Unconfirmed facts.** Expected but unmeasured facts, such as the shell flashed on the card
  (to be checked with `xbutil examine`), are drawn as expected and named as such in a footnote
  (`note` role, `INK_SOFT`) or a callout. Never present them as fact.
- **Hardware not yet run** (the k4 link, the card) is drawn as the README describes it, with no
  internals beyond what the README states.

## Repeated units and channel counts

- **Channels.** When the count is small and the layout means something (four DDR4 channels,
  each with its own cluster), draw every instance in its own column and label it (`CH0`–`CH3`,
  `CLUSTER 0`–`3`).
- **Many identical units.** Draw one block with `stack=2` (two copies behind it, 8 up and right
  each) and a dark count tag such as `× 4` or `× NUM_CH`. Put the tag on the back copy
  (`tag(x + w + 16, y - 16, ..., anchor="end")`) when an arrow meets the block's top edge.
  A repeated panel uses the same offsets (`cluster`). Alternatively, put a `bracket` labelled
  `× NUM_CH` under a row, or say "one per cluster" in the block's detail line.
- **Parameters.** Show them by name where the README does (`NUM_CH`, `T_MAX`, `MVM_BC_PIPE`).
  Give the U200 value only when it is fixed (`NUM_CH = 4`).

## Bands and regions

- Use `band()` for flat, outline-free regions behind blocks: SLRs, pipeline stages, clock
  domains.
- Label a band at its bottom or top in the `panel` role, in `INK_SOFT`.
- Keep a gap of at least 12 between adjacent bands.
- Draw bands before the blocks they hold. If an outline-only panel crosses them, draw it after
  the bands.
- **Region tags instead of bands.** Use these when the regions are not contiguous in the layout.
  In `system`, floorplan C' puts clusters 0 and 2 in SLR0 and clusters 1 and 3 in SLR2, so
  clusters 1 and 2 sit away from their channels' SLR (SLR1), and the regions are not contiguous
  in the layout. Give each block a dark tag on the right of its top edge
  (`box(..., tag="SLR1")`). Run the arrows that meet that edge 32 left of the block's centre, so
  the tag stays clear of them. Leave a block untagged when it has no single region (`system`: the
  XRT shell, the planned vector unit). Say in the footnote what the tags show and what validated
  them.

## Legend

A row of swatches at the bottom of the card, made with `legend()`, is optional. Use one when a
figure relies on colour roles that the README text does not explain. List only the roles the
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
| planned | `("neutral", "PLANNED", True)`, a dashed pale swatch; `layers` says `PLANNED / IDEA` |

`verify` is the one figure about tests rather than hardware, and it keeps its own small key:
`REFERENCE` (neutral), `HOST SOFTWARE` (host), `CHECK` (mint), `TRACES` (salmon), `RTL` (sky)
and `PLANNED`. It never reuses lavender, so a reader carrying the key over from the architecture
figures is not misled.

## Accuracy

Every block, arrow, label and number must be backed by the README (the fact-checked text and
the Mermaid blocks it replaces), `docs/spec.md`, `docs/rtl-m2.md`, `docs/rtl-m3.md`,
`docs/platform-u200.md` or the `rtl/core` sources. The script's docstring names its sources.
When a figure adds a fact the replaced Mermaid block did not show, cite it in the docstring.

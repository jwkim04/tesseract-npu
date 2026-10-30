"""Figure `stack`: the Tesseract software stack, from the applications to the FPGA card.

The product-style companion of the hero figure: fewer words, larger type (TYPE roles `*_l`), one
band per layer and a column of developer tools beside the middle layers. Top to bottom:

1. applications: llama.cpp (`llama-cli` with `-ngl` offload of ggml graphs on GGUF models, and
   `llama-server`, its OpenAI-compatible API; written as text, no OpenAI mark) and PyTorch
   (`torch.compile(model, backend="tesseract")` on Hugging Face models; sw/torch/README.md);
2. backends: `ggml-tesseract` (backend device `Tesseract0`, op-list builder and cache;
   sw/ggml-tesseract/README.md) and `tsr_rt` (weight upload, linear; sw/rt/README.md);
3. transport: `tsr_transport`, one protocol layer (image load, launch, wait, log) over four kinds:
   `sim`, `mock`, `vkernel`, `xrt` (docs/v2/01-execution-model.md §10, docs/v2/
   06-software-verification.md §9.1; sw/transport/tsr_transport.h);
4. firmware on the control core: runtime (launch protocol, traps, logs), graph executor (op lists,
   OPL v1), kernels (`MUL_MAT`, norms, RoPE, flash attention, KV; docs/v2/
   06-software-verification.md §7), `selftest` and `replay`; the tile / VU header library
   (`sw/ctrl/include/tsr/*.hpp`, freestanding C++20; SW §6.7);
5. NPU core: control core (Ibex RV32IMC, clock `clk_rv`), issue path with one in-order queue per
   unit, the units DMA, QACT, MVM clusters and VU, and the SPM (docs/v2/04-control-core.md §2.1,
   §7.4; docs/spec.md §2). The dashed bus from the issue path to the units stands for the unit
   queues (CC §7.4); the SPM bar under the units for their SPM clients (CC §10);
6. the Alveo U200 card: XRT shell (PCIe), four DDR4 channels, 64 GB (docs/platform-u200.md §1).
Developer tools (docs/v2/06-software-verification.md §9.5, §10; docs/v2/05-debug.md §9):
simulator (bit-exact oracle, cycle model, race detector; sw/sim), `tsr-selftest`, `klcheck` and
`numcheck`, trace tools (`tsr-trace-gen`, `tsr-trace-replay`, `tsr-trace-conv`), `tsr-log`,
GDB debugging (designed, DB §4, §9) and the profiler (`tsr-prof`, DB §8.6; both planned).

Logos (STYLE.md "Logos"; the only marks that are not drawn in the figure style): the official
logos of the software the figure names, embedded unmodified from docs/fig/logos (sources and
trademark notes in docs/fig/logos/SOURCES.md): llama.cpp (icon, on its own dark tile), PyTorch
(the lockup, on a white tile; it replaces the tile title), Hugging Face (beside "Hugging Face
models"). No logo of a company, an ISA or an IP core.

Status tags (STYLE.md "Planned and unconfirmed things"; at most three texts): ON THE CARD · V1
(dark badge: ran on the Alveo U200 with the v1 core, tag v1-m4), SIMULATOR · V2 (outlined badge:
implemented in this tree and verified on the simulator; commits 92f0b17, 83ac601), PLANNED
(dashed badge on a dashed pale block: designed in docs/v2, RTL or tool still to do; work packages
W3b, W4-W6). A panel carries the status of its tiles on its top edge; a tile carries its own tag
only where it differs. The xrt kind is tagged PLANNED: the v1 transport ran on the card, but the
v2 kind never has; it is written and unit-tested against the XRT shim (commit 83ac601) and runs on
the card with the v2 bitstream (W6). The applications carry no tag: they are third-party programs. The core panel
says ON THE CARD · V1 for the datapath (MVM clusters, DMA LD / ST, QACT, SPM: rtl/core); its
control core, issue path and vector unit are drawn as planned blocks (rtl/core has no control
core, issue path or VU yet). Colour: host software butter, firmware and control mint, tools white,
hardware by role (STYLE.md palette).

Run: build/tools/svgvenv/bin/python docs/fig/src/stack.py      (TSR_FIG_THEME=dark: stack-dark.svg)
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK, INK_SOFT, TYPE, Fig, font, logo_box, measure, wrap  # noqa: E402

W = 1600
FX0, FX1 = 56, 1544                   # full-width panels: applications, the card
MX0, MX1 = 56, 1168                   # main column panels
TX0, TX1 = 1208, 1544                 # tools column panel
PAD = 28                              # panel side padding
GAP_V = 56                            # between layer panels (one two-way arrow in each gap)
HEAD = 64                             # panel top to tile top (layer name row)
FOOT = 24                             # tile bottom to panel bottom
TILE_GAP = 24
LOGO_X, LOGO_GAP = 24, 20             # tile edge to logo; logo to the tile's text

CARD_TAG = ("ON THE CARD · V1", "dark")
SIM_TAG = ("SIMULATOR · V2", "outline")
PLAN_TAG = ("PLANNED", "planned")


# ---- layout helpers ----------------------------------------------------------------------------
def natural(t):
    """Width a tile needs for its text (and its own tag and logo), before the row is stretched."""
    w = measure(t["label"], "label_l")[0]
    if t.get("sub"):
        w = max(w, measure(t["sub"], "sub_l")[0])
    if t.get("badge"):
        w = max(w, logo_box(**t["badge"][0])[0] + 12 + measure(t["badge"][1], "sub_l")[0])
    w += 2 * 24
    if t.get("logo"):
        w += logo_box(**t["logo"])[0] + LOGO_GAP
    if t.get("tag"):
        w = max(w, measure(t["tag"][0], "tag_l")[0] + 24 + 14 + 8 + 14)
    return w


def place(tiles, x0, x1, gap=TILE_GAP):
    """Widths of a row of tiles: natural widths stretched evenly to fill [x0, x1], on the grid."""
    nat = [natural(t) for t in tiles]
    room = (x1 - x0) - gap * (len(tiles) - 1)
    if sum(nat) > room:
        raise SystemExit(f"row does not fit: needs {sum(nat):.0f}, has {room:.0f}: "
                         + ", ".join(t["label"].replace("\n", " ") for t in tiles))
    extra = (room - sum(nat)) / len(tiles)
    ws = [int((n + extra) // 8 * 8) for n in nat]
    ws[-1] += int(room - sum(ws))          # the grid remainder goes to the last tile
    xs, x = [], x0
    for w in ws:
        xs.append(x)
        x += w + gap
    return list(zip(xs, ws))


def tile(f, x, y, w, h, t):
    """One tile: solid, or dashed and pale when planned; an own tag only where it differs."""
    planned = t.get("tag") == PLAN_TAG
    tag, style = t["tag"] if t.get("tag") else ("", "")
    lg = t.get("logo")
    f.box(x, y, w, h, t["role"], "" if lg else t["label"], sub="" if lg else t.get("sub", ""),
          planned=planned, tag=tag, tag_style=style, tag_role="tag_l", label_role="label_l",
          sub_role="sub_l")
    if lg:
        logo_tile(f, x, y, w, h, t)


def logo_tile(f, x, y, w, h, t):
    """The inside of a tile with a logo: the logo at the left, centred in the tile's height, and
    the text left-aligned beside it. A tile without a label (the PyTorch lockup carries the name)
    may have a badge line: a small logo and a few words under the detail line."""
    lg = t["logo"]
    ow, oh = logo_box(**lg)
    f.logo(x=x + LOGO_X, y=y + (h - oh) / 2, **lg)
    tx = x + LOGO_X + ow + LOGO_GAP
    if not t.get("badge"):
        f.label_block(tx, y, x + w - tx, h, t["label"], t.get("sub", ""), "label_l", "sub_l",
                      align="start", pad=0)
        return
    bg, text = t["badge"]
    fo = font("regular")
    cap = fo.cap * TYPE["sub_l"][0] / fo.upm
    bw, bh = logo_box(**bg)
    gap = 18
    top = y + (h - (cap + gap + bh)) / 2
    f.text(tx, top + cap, t["sub"], "sub_l")
    f.logo(x=tx, y=top + cap + gap, **bg)
    f.text(tx + bw + 12, top + cap + gap + bh / 2, text, "sub_l", valign="middle")


def panel(f, x, y, w, h, name, status=None, caption="", logo=None):
    """Layer panel: outlined card, the layer name inside at the top left, an optional caption at
    the top right and the panel's status tag on the top edge."""
    f.panel(x, y, w, h)
    f.text(x + 32, y + 40, name, "panel_l", color=INK_SOFT)
    cx = x + w - 32
    if logo:
        ow, oh = logo_box(**logo)
        f.logo(x=cx - ow, y=y + 38 - oh / 2, **logo)
        cx -= ow + 28
    if caption:
        f.text(cx, y + 40, caption, "note_l", anchor="end", color=INK_SOFT)
    if status:
        f.tag(x + w - 36, y, status[0], anchor="end", style=status[1], role="tag_l")


def T(role, label, sub="", tag=None, logo=None, badge=None):
    return {"role": role, "label": label, "sub": sub, "tag": tag, "logo": logo, "badge": badge}


# ---- content -----------------------------------------------------------------------------------
TSR_BASIS = {"key": "tesseract-basis", "h": 36, "backing": True, "pad": 4}
TSR_CORE = {"key": "tesseract-core", "h": 36, "backing": True, "pad": 4}
HF_BADGE = {"key": "huggingface", "h": 32, "backing": True, "pad": 4}
apps = [T("host", "LLAMA.CPP", "llama-cli · GGUF models · -ngl\nllama-server (OpenAI-compatible API)",
          logo={"key": "llamacpp", "h": 80}),
        T("host", "", 'torch.compile(backend="tesseract")',
          logo={"key": "pytorch", "h": 44, "backing": True, "pad": 10},
          badge=(HF_BADGE, "Hugging Face models"))]
backends = [T("host", "GGML-TESSERACT", "device Tesseract0 · op lists"),
            T("host", "TSR_RT", "weight upload · linear")]
transport = [T("host", "SIM", "in process"), T("host", "MOCK", "adapter model"),
             T("host", "VKERNEL", "Verilated RTL", tag=PLAN_TAG),
             T("host", "XRT", "xrt::ip · PCIe", tag=PLAN_TAG)]
firmware = [T("control", "RUNTIME", "launch · logs"),
            T("control", "EXECUTOR", "op lists"),
            T("control", "KERNELS", "MUL_MAT · norms · RoPE\nflash attention · KV"),
            T("control", "SELFTEST\nREPLAY")]
core_ctl = [T("control", "CONTROL CORE", "Ibex RV32IMC · clk_rv", tag=PLAN_TAG),
            T("control", "ISSUE PATH", "checker · unit queues", tag=PLAN_TAG)]
core_units = [T("io", "DMA", "LD · ST"), T("compute", "QACT", "f32 to QA"),
              T("compute", "MVM CLUSTERS", "4 × 128 MAC lanes"),
              T("compute", "VECTOR UNIT", "norms · RoPE", tag=PLAN_TAG)]
card = [T("vendor", "ALVEO U200", "FPGA · VU9P"), T("vendor", "XRT SHELL", "PCIe · registers"),
        T("memory", "DDR4 × 4", "64 GB · 4 channels")]
tools = [T("neutral", "KLCHECK\nNUMCHECK", "accuracy vs the CPU"),
         T("neutral", "TRACE TOOLS", "gen · replay · conv"),
         T("neutral", "TSR-SELFTEST", "device self-tests"),
         T("neutral", "TSR-LOG", "firmware log decoder"),
         T("neutral", "SIMULATOR", "bit-exact oracle\ncycle model · races"),
         T("neutral", "GDB", "debugging\nover PCIe", tag=PLAN_TAG),
         T("neutral", "TSR-PROF", "timeline · counters", tag=PLAN_TAG)]

# ---- vertical layout -----------------------------------------------------------------------------
Y = 56
H1, H2 = 96, 128                      # tile heights: one detail line / firmware (and apps)
ROW = 96                              # tile height in the core panel
BUS = 64                              # control row to units row: room for the unit-queue bus
SPM_GAP, SPM_H = 56, 80               # units row to SPM bar; SPM bar height
panels = {}                           # name -> (y, h)
for name, h in (("apps", HEAD + H2 + FOOT), ("backends", HEAD + H1 + FOOT),
                ("transport", HEAD + H1 + FOOT), ("firmware", HEAD + H2 + FOOT),
                ("core", HEAD + ROW + BUS + ROW + SPM_GAP + SPM_H + FOOT), ("card", HEAD + H1 + FOOT)):
    panels[name] = (Y, h)
    Y += h + GAP_V
Y -= GAP_V
KEY_Y = Y + 72
H = KEY_Y + 3 * 40 + 28 + 2 * 26 + 48

f = Fig("stack", W, H,
        title="The Tesseract software stack",
        desc="Layered software stack, top to bottom: llama.cpp (llama-cli and llama-server, an "
             "OpenAI-compatible API) and PyTorch with Hugging Face models, each with its official "
             "logo; the ggml-tesseract "
             "backend and tsr_rt; tsr_transport with the sim, mock, vkernel and xrt kinds; "
             "firmware on the RISC-V control core (runtime, graph executor, kernels for MUL_MAT, "
             "norms, RoPE and flash attention, selftest and replay); the NPU core (control core, "
             "issue path with unit queues, DMA, QACT, MVM clusters, vector unit, SPM); the Alveo "
             "U200 card with the XRT shell and four DDR4 channels. A column of developer tools "
             "sits beside the middle layers: bit-exact simulator, tsr-selftest, klcheck and "
             "numcheck, trace tools, tsr-log, and planned GDB debugging and "
             "profiler. Tags give "
             "each block's status: on the card (v1), on the simulator (v2) or planned; the xrt "
             "kind is planned because only the v1 transport has run on the card.")

IX0, IX1 = MX0 + PAD, MX1 - PAD       # tile area of the main panels
mid = (MX0 + MX1) / 2

# applications: full width
y, h = panels["apps"]
panel(f, FX0, y, FX1 - FX0, h, "APPLICATIONS")
for t, (x, w) in zip(apps, place(apps, FX0 + PAD, FX1 - PAD)):
    tile(f, x, y + HEAD, w, H2, t)

# backends
y, h = panels["backends"]
panel(f, MX0, y, MX1 - MX0, h, "BACKENDS", SIM_TAG, "weights · op lists · launches")
for t, (x, w) in zip(backends, place(backends, IX0, IX1)):
    tile(f, x, y + HEAD, w, H1, t)

# transport: the four kinds of tsr_transport
y, h = panels["transport"]
panel(f, MX0, y, MX1 - MX0, h, "TRANSPORT · tsr_transport", SIM_TAG,
      "one protocol layer")
for t, (x, w) in zip(transport, place(transport, IX0, IX1)):
    tile(f, x, y + HEAD, w, H1, t)

# firmware
y, h = panels["firmware"]
panel(f, MX0, y, MX1 - MX0, h, "FIRMWARE ON THE CONTROL CORE", SIM_TAG,
      "kernel SDK · custom kernels in C++20", logo=TSR_BASIS)
for t, (x, w) in zip(firmware, place(firmware, IX0, IX1)):
    tile(f, x, y + HEAD, w, H2, t)

# NPU core: control row, units row with the unit-queue bus, SPM
y, h = panels["core"]
panel(f, MX0, y, MX1 - MX0, h, "TESSERACT CORE · tsr_core", CARD_TAG,
      "vendor-independent SystemVerilog", logo=TSR_CORE)
cy0 = y + HEAD
ctl = place(core_ctl, IX0, IX0 + 768)
for t, (x, w) in zip(core_ctl, ctl):
    tile(f, x, cy0, w, ROW, t)
f.arrow([(ctl[0][0] + ctl[0][1] - 10, cy0 + ROW / 2), (ctl[1][0] + 10, cy0 + ROW / 2)])
f.text(ctl[1][0] + ctl[1][1] + 40, cy0 + ROW / 2, "FIRMWARE ISSUES\nEVERY TILE OP", "callout",
       valign="middle")
cy1 = cy0 + ROW + BUS
units = place(core_units, IX0, IX1)
for t, (x, w) in zip(core_units, units):
    tile(f, x, cy1, w, ROW, t)
# the unit queues: a dashed bus from the issue path to every unit
bus_y = cy0 + ROW + BUS / 2
is_c = ctl[1][0] + ctl[1][1] / 2
ucs = [x + w / 2 for x, w in units]
f.line([(is_c, cy0 + ROW - 6), (is_c, bus_y)], end=False, dashed=True)
f.hline(min(ucs[0], is_c), max(ucs[-1], is_c), bus_y, dashed=True, color=INK, sw=2)
for ux in ucs:
    f.line([(ux, bus_y), (ux, cy1 + 8)], dashed=True)
f.text(is_c + 24, bus_y - 16, "TILE OPS", "callout", valign="middle")
cy2 = cy1 + ROW + SPM_GAP
f.box(IX0, cy2, IX1 - IX0, SPM_H, "memory", "SPM", sub="4 MiB · read and written by every unit",
      label_role="label_l", sub_role="sub_l")
for ux in ucs:
    f.arrow([(ux, cy1 + ROW - 8), (ux, cy2 + 12)], tail=True)

# the card: full width
y, h = panels["card"]
panel(f, FX0, y, FX1 - FX0, h, "FPGA CARD", CARD_TAG,
      "weights stream from DDR4, one channel per cluster")
for t, (x, w) in zip(card, place(card, FX0 + PAD, FX1 - PAD)):
    tile(f, x, y + HEAD, w, H1, t)

# arrows between the layers (two-way: work goes down, results come back up)
names = ["apps", "backends", "transport", "firmware", "core", "card"]
for a, b in zip(names, names[1:]):
    ya, ha = panels[a]
    yb, _ = panels[b]
    f.arrow([(mid, ya + ha - 8), (mid, yb + 8)], tail=True)

# developer tools column, beside layers 2 to 5
ty0 = panels["backends"][0]
ty1 = panels["core"][0] + panels["core"][1]
panel(f, TX0, ty0, TX1 - TX0, ty1 - ty0, "DEVELOPER TOOLS", SIM_TAG)
n = len(tools)
tgap = 16
th_ = int((ty1 - ty0 - HEAD - FOOT - tgap * (n - 1)) / n // 8 * 8)
slack = (ty1 - ty0 - HEAD - FOOT - tgap * (n - 1) - th_ * n) / (n - 1)
yy = ty0 + HEAD
for t in tools:
    tile(f, TX0 + 24, round(yy), TX1 - TX0 - 48, th_, t)
    yy += th_ + tgap + slack

# status key and footnote
ky = KEY_Y
key = [(CARD_TAG, "ran on the Alveo U200 with the v1 core (tag v1-m4)"),
       (SIM_TAG, "implemented and bit-exact on the simulator; the card follows with the v2 core"),
       (PLAN_TAG, "designed in docs/v2; simulated or in firmware where it exists; RTL and card planned (W4 to W6)")]
kx = 56 + 32
for (txt, style), desc in key:
    f.tag(kx, ky, txt, anchor="start", style=style, role="tag_l")
    f.text(kx + 260, ky, desc, "note_l", valign="middle")
    ky += 40
f.text(kx, ky + 8, wrap(
    "A panel's tag applies to its blocks; a block with its own tag differs from its panel. "
    "xrt: the v1 transport ran on the card; the v2 kind has not, it awaits the v2 bitstream. "
    "Throughput numbers measured on the card are v1 numbers.", 1400, "note_l"),
    "note_l", color=INK_SOFT)

f.save(scale=1)

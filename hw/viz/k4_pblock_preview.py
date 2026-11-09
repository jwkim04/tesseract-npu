"""Draw the planned hierarchy assignments on the U200's retained SLICE Pblock ranges.

The hook (hw/u200/k4_major_blocks.tcl) is interpreted with a Tcl mock: no Vivado executable, checkpoint or placement.
--no-db draws an image built without the DB block (HAS_DB 0): the mock has no g_db.* cell, as the netlist has none.
The Pblock shapes come from a properties file (PBLOCK sections with DERIVED_RANGES, and an optional CAPACITY line):
the platform's three SLR pblocks and, when the hook makes one, its control Pblock, as Vivado resolved them
(tb/platform/viv/k4_pblock_check.tcl writes the file).
"""
import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {0: "#40a79d", 1: "#db9958", 2: "#659ad0"}
CTRL_COLOR = "#14544f"
VU_COLOR = "#274d86"
INK = "#172e44"
GRAY = "#475b6c"
SCALE = 2
# SLICE columns of the clock-region columns X0 to X5 of the U200; every clock-region row is 60 SLICE rows (Y0 to Y14)
CR_X = ((0, 30), (31, 54), (55, 87), (88, 111), (112, 141), (142, 168))
# The DDR0 memory controller of the platform: clock regions X2Y1 to X2Y3 of SLR0 (docs/platform-u200.md §8.7)
MIG0 = (2, 1, 2, 3)
# The SLR2 controller (DDR[3], "MIG 2" in docs/platform-u200.md §8.7): its logic is in X3Y10 to X3Y11 and X4Y10 to X4Y13
MIG2 = ((3, 10, 3, 11), (4, 10, 4, 13))
# Pblocks of the full-feature image's hook: name -> (parent SLR, color, label, DDR controller outline)
CHILDREN = {
    "pblock_tsr_ctrl_plane": (0, CTRL_COLOR, "제어 Pblock", (MIG0,)),
    "pblock_tsr_vu": (2, VU_COLOR, "VU Pblock", MIG2),
}


def assignments(hook, no_db=False):
    # Interpret the hierarchy-only hook with Tcl. Record add_cells_to_pblock and the Pblocks the hook creates
    # (create_pblock / resize_pblock / PARENT). get_sites only satisfies the hook's own capacity checks.
    script = """
proc current_instance {args} {}
set ::pblocks {pblock_dynamic_SLR0 pblock_dynamic_SLR1 pblock_dynamic_SLR2}
array set ::parent {}
array set ::ranges {}
proc get_cells {args} {
    if {[lsearch -exact $args -filter] >= 0} {return kernel}
    if {[info exists ::env(PBLOCK_PREVIEW_NO_DB)] && [string first g_db. [lindex $args end]] >= 0} {return {}}
    return [lindex $args end]
}
proc get_property {key obj} {
    if {$key eq "IS_PRIMITIVE"} {return 0}
    if {$key eq "PARENT"} {return [expr {[info exists ::parent($obj)] ? $::parent($obj) : {}}]}
    return $obj
}
proc get_pblocks {args} {
    set n [lindex $args end]
    if {[lsearch -exact $::pblocks $n] >= 0} {return $n}
    return {}
}
proc create_pblock {n} {lappend ::pblocks $n; set ::ranges($n) {}; return $n}
proc resize_pblock {pb args} {lappend ::ranges($pb) [lindex $args 1]}
proc set_property {key value obj} {if {$key eq "PARENT"} {set ::parent($obj) $value}}
proc get_sites {args} {
    set out {}
    for {set i 0} {$i < 100000} {incr i} {lappend out site$i}
    return $out
}
proc add_cells_to_pblock {pb obj} {puts [join [list assignment $pb $obj] "\\t"]}
if {[catch {source $::env(PBLOCK_PREVIEW_HOOK)} err]} {
    puts stderr $err
    exit 1
}
foreach n $::pblocks {
    if {[info exists ::parent($n)]} {puts [join [list pblock $n $::parent($n) $::ranges($n)] "\\t"]}
}
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TSR_K4", "PBLOCK_PREVIEW"))}
    env["PBLOCK_PREVIEW_HOOK"] = str(hook.resolve())
    if no_db:
        env["PBLOCK_PREVIEW_NO_DB"] = "1"
    result = subprocess.run(["tclsh"], input=script, text=True, capture_output=True, env=env, check=True)
    groups = defaultdict(list)
    children = {}
    for line in result.stdout.splitlines():
        if line.startswith("assignment\t"):
            _, pblock, path = line.split("\t")
            # the SLR pblocks by their digit, a Pblock the hook made by its name
            groups[int(pblock[-1]) if re.fullmatch(r"pblock_dynamic_SLR\d", pblock) else pblock].append(path)
        elif line.startswith("pblock\t"):
            _, name, parent, *ranges = line.split("\t")
            children[name] = (parent, ranges[0].split() if ranges else [])
    return groups, children


def labels(paths, reduced=False):
    classes = defaultdict(list)
    for path in paths:
        tail = path.rsplit("/", 1)[-1]
        match = re.fullmatch(r'g_(cl|asm|bank|ch)\[(\d+)\]\.(u_cl|u_asm|u_bank|u_br|u_arb)', tail)
        if match:
            classes[match[3]].append(match[2])
        else:
            pair = re.fullmatch(r'g_(?:local_att|pairs)\.g_pair\[(\d+)\]\.(u_local_asmax|u_source)', tail)
            if pair:
                classes[pair[2]].append(pair[1])
            else:
                classes[tail].append("")
    frontend = ("u_gen", "u_rd", "u_ang", "u_inj", "u_lma", "u_lmb")
    if all(block in classes for block in frontend):
        for block in frontend:
            del classes[block]
        classes["vu_frontend"] = [""]
    for key, blocks in (("rv_link", ("g_rv_local.u_spm_rx", "g_rv_local.u_issue_tx", "g_rv_local.u_lq_tx")),
                        ("core_link", ("g_rv_local.u_spm_tx", "g_rv_local.u_issue_rx", "g_rv_local.u_lq_rx"))):
        if all(block in classes for block in blocks):
            for block in blocks:
                del classes[block]
            classes[key] = [""]
    names = (("u_cl", "클러스터"), ("u_asm", "결과 조립기"), ("u_bank", "VU·계수 bank"),
             ("u_local_asmax", "로컬 어텐션"), ("u_source", "결과 수집·후처리"),
             ("u_br", "DDR 브리지"), ("u_arb", "DDR 중재기"), ("u_spm", "SPM"),
             ("u_bridge", "RV↔SPM 접속부"),
             ("u_ctrl", "RV·TCM·CDC"), ("g_db.u_dbg_rv", "RV 디버그"),
             ("rv_link", "제어 전송 RV측"), ("core_link", "제어 전송 중앙측"),
             ("u_dma", "DMA" if reduced else "DMA·KVST"), ("u_qact", "QACT"),
             ("g_rv_local.u_spm_tx", "SPM 링크 끝"), ("g_rv_local.u_issue_rx", "issue 링크 끝"),
             ("g_rv_local.u_lq_rx", "LCQ 링크 끝"),
             ("u_issue", "issue 경로"), ("g_xc.u_xc", "VU 검사기"), ("g_xc.g_at.u_chk_at", "AT 검사기"),
             ("u_csr", "CSR"), ("g_db.u_dbg", "DB 코어측"),
             ("g_vu.u_vu", "벡터 유닛 전체(입력·두 lane bank·GDN 전용부)"),
             ("vu_frontend", "VU 입력·각도 생성·LM"))
    return [label + (" " + "·".join(classes[key]) if any(classes[key]) else "")
            for key, label in names if key in classes] + [k for k in classes if k not in dict(names)]


def slice_ranges(report):
    """SLICE ranges and capacity of every pblock in the properties file: {name: (ranges, capacity or None)}."""
    sections = re.split(r"(?m)^PBLOCK ", report.read_text())
    shapes = {}
    for section in sections:
        lines = section.splitlines()
        if not lines or not re.fullmatch(r"pblock_dynamic_SLR[0-2]|pblock_\w+", lines[0]):
            continue
        ranges = next((line for line in lines if line.startswith("DERIVED_RANGES ")), None)
        if ranges is None:
            continue
        capacity = next((line.split()[1:] for line in lines if line.startswith("CAPACITY ")), None)
        shapes[lines[0]] = ([tuple(map(int, match)) for match in
                             re.findall(r"SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)", ranges)],
                            [int(v) for v in capacity] if capacity else None)
    return shapes


def clock_region_slices(ranges):
    """SLICE rectangles of clock-region ranges 'CLOCKREGION_XaYb:CLOCKREGION_XcYd'."""
    out = []
    for r in ranges:
        m = re.fullmatch(r"CLOCKREGION_X(\d+)Y(\d+):CLOCKREGION_X(\d+)Y(\d+)", r)
        x0, y0, x1, y1 = map(int, m.groups())
        out.append((CR_X[x0][0], y0 * 60, CR_X[x1][1], y1 * 60 + 59))
    return out


def intersect(boxes, parent):
    out = []
    for ax0, ay0, ax1, ay1 in boxes:
        for bx0, by0, bx1, by1 in parent:
            x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
            if x0 <= x1 and y0 <= y1:
                out.append((x0, y0, x1, y1))
    return out


def cr_label(r):
    m = re.fullmatch(r"CLOCKREGION_(X\d+Y\d+):CLOCKREGION_(X\d+Y\d+)", r)
    return m[1] if m[1] == m[2] else f"{m[1]}–{m[2]}"


def render(hook, report, output, title=None, header=None, no_db=False):
    groups, children = assignments(hook, no_db)
    shapes = slice_ranges(report)
    placed = [path for paths in groups.values() for path in paths]
    has_vu_banks = any(path.endswith(".u_bank") for path in placed)
    # The full-feature image places the whole vector unit; the reduced first card image has no VU and no KVST
    # and its hook also places QACT, issue and CSR.
    full = any(path.endswith("g_vu.u_vu") for path in placed)
    reduced = not full and not has_vu_banks and any(path.endswith("/u_qact") for path in placed)
    # Pblocks the hook creates inside an SLR pblock (the control Pblock; in the full image also the VU Pblock).
    kids = {name: CHILDREN[name] for name in children if name in groups and name in CHILDREN}
    ctrl = "pblock_tsr_ctrl_plane" if "pblock_tsr_ctrl_plane" in kids else None
    img = Image.new("RGB", (1120 * SCALE, (880 if full else 820) * SCALE), "#ffffff")
    draw = ImageDraw.Draw(img)
    fontdir = Path("/usr/share/fonts/opentype/noto")
    fonts = {}

    def font(size, bold=False):
        key = (size, bold)
        if key not in fonts:
            name = "NotoSansCJK-" + ("Bold" if bold else "Regular") + ".ttc"
            fonts[key] = ImageFont.truetype(str(fontdir / name), size * SCALE, index=1)
        return fonts[key]

    def word(x, y, value, size=16, color=INK, bold=False):
        draw.text((x*SCALE, y*SCALE), value, font=font(size, bold), fill=color, anchor="lt")

    def rect(x0, y0, x1, y1, fill=None, outline=None, width=1):
        draw.rectangle(tuple(round(v*SCALE) for v in (x0, y0, x1, y1)), fill=fill, outline=outline,
                       width=width * SCALE)

    def wrap_lines(items, max_width=524):
        # Whole labels remain together. Long region lists must not disappear
        # beyond the image edge when another functional block is assigned.
        lines, line = [], ""
        for item in items:
            candidate = line + " / " + item if line else item
            if line and draw.textlength(candidate, font=font(16)) > max_width * SCALE:
                lines.append(line)
                line = item
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines

    def wrapped_items(x, y, items, max_width=524, max_lines=4):
        lines = wrap_lines(items, max_width)
        if len(lines) > max_lines:
            raise ValueError(f"Pblock labels need more than {max_lines} lines")
        for i, line in enumerate(lines):
            word(x, y + i * 22, line, 16, INK)
        return len(lines)

    word(44, 30, header or ("ALVEO U200   /   K4   /   CORE 250 MHz · CLK_RV 100 MHz   /   PRE-BUILD" if full else
                  "ALVEO U200   /   K4-MIN   /   CORE 200 MHz · CLK_RV 100 MHz   /   PRE-BUILD" if ctrl else
                  "ALVEO U200   /   K4-MIN   /   PRE-BUILD") if (full or reduced) else
         "ALVEO U200   /   K4   /   PRE-BUILD", 15, GRAY, True)
    title = title or ("풀 구성 카드 이미지(250 MHz, GDN 포함)의 Pblock 계획" if full else
             "첫 카드 이미지(200 MHz)의 Pblock 계획" if ctrl else
             "첫 카드 이미지(VU·어텐션 제외)의 Pblock 계획" if reduced else
             "다음 빌드의 Pblock 계획" if has_vu_banks else "VU 제외 진단 빌드의 Pblock 계획")
    word(42, 60, title, 36, INK, True)
    if full:
        word(44, 112, "색은 큰 블록을 배치할 수 있는 범위입니다. 진한 청록은 SLR0의 제어부 Pblock, 진한 파랑은 SLR2의 VU Pblock입니다.", 17, GRAY)
        word(44, 136, "개별 유닛의 실제 위치는 배치 후 결정됩니다." +
             (" 이 이미지에는 디버그(DB) 블록이 없습니다(HAS_DB 0)." if no_db else ""), 17, GRAY)
    elif ctrl:
        word(44, 112, "색은 큰 블록을 배치할 수 있는 범위입니다. 진한 청록은 SLR0 Pblock 안에 둔 제어부 전용 Pblock입니다.", 17, GRAY)
        word(44, 136, "개별 유닛의 실제 위치는 배치 후 결정됩니다.", 17, GRAY)
    else:
        word(44, 119, "색은 큰 블록을 배치할 수 있는 범위입니다. 개별 유닛의 실제 위치는 배치 후 결정됩니다.", 17, GRAY)
    word(44, 167, "플랫폼 동적 영역", 17, INK, True)
    word(553, 167, "각 Pblock에 지정할 블록", 17, INK, True)

    left, top, width, height = 60, 207, 410, 405

    def fx(x):          # SLICE column -> picture x
        return left + width * x / 169

    def fy(y):          # SLICE row -> picture y
        return top + height * (900 - y) / 900

    rect(left, top, left+width, top+height, "#e4e9ee")
    for s in (2, 1, 0):
        for x0, y0, x1, y1 in shapes[f"pblock_dynamic_SLR{s}"][0]:
            rect(fx(x0), fy(y1 + 1), fx(x1 + 1), fy(y0), COLORS[s])
    for i in (1, 2):
        y = top + i * height/3
        rect(left, y-1, left+width, y+1, "#ffffff")
    for name, (_, color, _, _) in kids.items():
        parent = children[name][0]
        # Vivado's resolved ranges when the properties file has them, else the clock regions cut by the parent
        boxes = (shapes[name][0] if name in shapes else
                 intersect(clock_region_slices(children[name][1]), shapes[parent][0]))
        for x0, y0, x1, y1 in boxes:
            rect(fx(x0), fy(y1 + 1), fx(x1 + 1), fy(y0), color)
    for name, (slr, _, _, migs) in kids.items():
        # the DDR controller's clock regions stay outside the Pblock of that SLR
        for mx0, my0, mx1, my1 in migs:
            rect(fx(CR_X[mx0][0]), fy(my1 * 60 + 60), fx(CR_X[mx1][1] + 1), fy(my0 * 60), outline=INK, width=2)
        mx0, my0, mx1, my1 = migs[0]
        word(fx(CR_X[mx0][0]) + 5, fy(my1 * 60 + 60) + 5, f"DDR{3 if slr == 2 else 0}", 12, "#ffffff", True)
        word(fx(CR_X[mx0][0]) + 5, fy(my1 * 60 + 60) + 20, "컨트롤러", 11, "#ffffff", True)
    # clock-region grid and labels
    for i, (x0, x1) in enumerate(CR_X):
        word(fx(x0) + (fx(x1 + 1) - fx(x0)) / 2 - 8, top - 16, f"X{i}", 11, GRAY)
        if i:
            rect(fx(x0) - 0.4, top, fx(x0) + 0.4, top + height, "#ffffff")
    for r in range(15):
        word(left - 27, fy(r * 60 + 60) + 5, f"Y{r}", 10, GRAY)
        if r % 5:
            rect(left, fy(r * 60) - 0.4, left + width, fy(r * 60) + 0.4, "#ffffff")
    word(left+width*0.59, top+height/2-9, "플랫폼 셸", 16, GRAY, True)
    for s in (2, 1, 0):
        y = top + (2-s)*135
        word(79, y+52, f"SLR{s}", 21, "#ffffff" if (s == 0 and ctrl) or (s == 2 and full) else "#122d3c", True)
    # the Pblock list: one section per SLR, as tall as its text needs
    section_y = top
    for s in (2, 1, 0):
        y = section_y
        word(550, y+9, f"SLR{s}", 21, INK, True)
        items = labels(groups[s], reduced)
        kid = next((name for name, (slr, *_ ) in kids.items() if slr == s), None)
        text_end = y + 39
        if kid:
            n = wrapped_items(550, y+39, items or ["커널 블록 별도 지정 없음"], max_lines=3)
            y2 = y + 39 + n * 22 + 8
            _, color, name, _ = kids[kid]
            word(550, y2, name + "  " + ", ".join(cr_label(r) for r in children[kid][1]), 16, color, True)
            cap = shapes[kid][1] if kid in shapes else None
            lines = 1
            if cap:
                word(550, y2 + 22, f"CLB {cap[0]:,} · RAMB36 {cap[1]} · URAM {cap[4]} · DSP {cap[3]}", 15, GRAY)
                lines = 2
            m = wrapped_items(550, y2 + 22 * lines, labels(groups[kid], reduced), max_lines=4)
            rect(517, y2 - 2, 523, y2 + 22 * (lines + m) + 2, color)
            text_end = y2 + 22 * (lines + m) + 2
        else:
            n = wrapped_items(550, y+39, items or ["커널 블록 별도 지정 없음"], max_lines=4)
            text_end = y + 39 + 22 * n
        # the SLR's own color bar spans its section down to the child bar
        bar_end = (y2 - 6) if kid else text_end + 4
        rect(517, y+9, 523, max(bar_end, y + 60), COLORS[s])
        section_y = max(text_end, y + 125) + 30
    bottom = max(section_y, top + height + 80)
    word(60, 631, "실제 SLICE 범위 기준 · 회색은 해당 Pblock 밖", 14, GRAY)
    if full:
        word(60, 651, "진한 청록: 제어 Pblock · 진한 파랑: VU Pblock", 14, GRAY)
        word(60, 671, "외곽선: DDR0 / DDR3 컨트롤러 영역", 14, GRAY)
    elif kids:
        word(60, 651, "진한 청록: 제어 Pblock · 외곽선: DDR0 컨트롤러 영역", 14, GRAY)
    box_y = max(709, bottom - 8) if full else 709
    rect(42, box_y, 1078, box_y + 71, "#eff4f7")
    word(61, box_y + 10, "Vivado 자율 배치", 16, INK, True)
    placed_dma = any(path.endswith("/u_dma") for path in placed)
    placed_issue = any(path.endswith("/u_issue") for path in placed)
    if full:
        word(61, box_y + 39, "중앙 gather 수신부, DDR1·2 브리지, DMA 측 DDR 경계 슬라이스, 어댑터 제어, 기타 연결 로직 등", 16, GRAY)
    elif reduced:
        free = "중앙 gather, DDR1·2 브리지, 어댑터 제어, " + ("" if placed_issue else "issue·CSR·중앙 debug, ")
        word(61, box_y + 39, free + "기타 연결 로직 등", 16, GRAY)
    else:
        free = ("" if placed_dma else "DMA·KVST, ") + "중앙 gather, "
        free += ("VU 출력·중앙 축약, " if placed_dma else "VU 중앙 제어·축약, ") if has_vu_banks else ""
        if not groups[1] and has_vu_banks:
            free = "SPM, RV·제어, DDR 브리지·중재기, DMA·KVST, gather, VU 중앙부, "
        word(61, box_y + 39, free + "issue·중앙 debug, 기타 연결 로직 등", 16, GRAY)
    word(44, box_y + 86, ("K4" if full else "K4-MIN" if reduced else "K4") +
         " major-block floorplan   ·   단위 셀 / 개별 배선 지정 없음", 12, GRAY)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hook", type=Path)
    parser.add_argument("pblock_properties", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", help="the picture's title (default: by the blocks the hook places)")
    parser.add_argument("--header", help="the line above the title")
    parser.add_argument("--no-db", action="store_true", help="draw the image without the DB block (HAS_DB 0)")
    args = parser.parse_args()
    render(args.hook, args.pblock_properties, args.output, args.title, args.header, args.no_db)

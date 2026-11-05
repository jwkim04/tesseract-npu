"""Draw the planned hierarchy assignments on the U200's retained SLICE Pblock ranges."""
import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {0: "#40a79d", 1: "#db9958", 2: "#659ad0"}
INK = "#172e44"
GRAY = "#475b6c"
SCALE = 2


def assignments(hook):
    # Interpret the hierarchy-only hook with Tcl. Record add_cells_to_pblock;
    # no Vivado executable, checkpoint or actual placement is involved.
    script = """
proc current_instance {args} {}
proc get_cells {args} {
    if {[lsearch -exact $args -filter] >= 0} {return kernel}
    return [lindex $args end]
}
proc get_property {key obj} {
    if {$key eq "IS_PRIMITIVE"} {return 0}
    return $obj
}
proc get_pblocks {args} {return [lindex $args end]}
proc add_cells_to_pblock {pb obj} {puts [join [list assignment $pb $obj] "\\t"]}
if {[catch {source $::env(PBLOCK_PREVIEW_HOOK)} err]} {
    puts stderr $err
    exit 1
}
"""
    result = subprocess.run(["tclsh"], input=script, text=True, capture_output=True,
                            env={**os.environ, "PBLOCK_PREVIEW_HOOK": str(hook.resolve())}, check=True)
    groups = defaultdict(list)
    for line in result.stdout.splitlines():
        if line.startswith("assignment\t"):
            _, pblock, path = line.split("\t")
            groups[int(pblock[-1])].append(path)
    return groups


def labels(paths):
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
             ("rv_link", "제어 전송 RV측"), ("core_link", "제어 전송 중앙측"), ("u_dma", "DMA·KVST"),
             ("vu_frontend", "VU 입력·각도 생성·LM"))
    return [label + (" " + "·".join(classes[key]) if any(classes[key]) else "")
            for key, label in names if key in classes] + [k for k in classes if k not in dict(names)]


def slice_ranges(report):
    sections = re.split(r"(?m)^PBLOCK ", report.read_text())
    shapes = {}
    for section in sections:
        lines = section.splitlines()
        if not lines or not re.fullmatch(r"pblock_dynamic_SLR[0-2]", lines[0]):
            continue
        ranges = next(line for line in lines if line.startswith("DERIVED_RANGES "))
        shapes[int(lines[0][-1])] = [
            tuple(map(int, match)) for match in
            re.findall(r"SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)", ranges)]
    return shapes


def render(hook, report, output):
    groups, shapes = assignments(hook), slice_ranges(report)
    has_vu_banks = any(path.endswith(".u_bank") for paths in groups.values() for path in paths)
    img = Image.new("RGB", (1120 * SCALE, 820 * SCALE), "#ffffff")
    draw = ImageDraw.Draw(img)
    fontdir = Path("/usr/share/fonts/opentype/noto")
    fonts = {}

    def word(x, y, value, size=16, color=INK, bold=False):
        key = (size, bold)
        if key not in fonts:
            name = "NotoSansCJK-" + ("Bold" if bold else "Regular") + ".ttc"
            fonts[key] = ImageFont.truetype(str(fontdir / name), size * SCALE, index=1)
        draw.text((x*SCALE, y*SCALE), value, font=fonts[key], fill=color, anchor="lt")

    def rect(x0, y0, x1, y1, fill):
        draw.rectangle(tuple(round(v*SCALE) for v in (x0, y0, x1, y1)), fill=fill)

    def wrapped_items(x, y, items, max_width=524):
        # Whole labels remain together. Long region lists must not disappear
        # beyond the image edge when another functional block is assigned.
        font = ImageFont.truetype(str(fontdir / "NotoSansCJK-Regular.ttc"), 16 * SCALE, index=1)
        lines, line = [], ""
        for item in items:
            candidate = line + " / " + item if line else item
            if line and draw.textlength(candidate, font=font) > max_width * SCALE:
                lines.append(line)
                line = item
            else:
                line = candidate
        if line:
            lines.append(line)
        if len(lines) > 4:
            raise ValueError("Pblock labels need more than four lines")
        for i, line in enumerate(lines):
            word(x, y + i * 22, line, 16, INK)

    word(44, 30, "ALVEO U200   /   K4   /   PRE-BUILD", 15, GRAY, True)
    title = "다음 빌드의 Pblock 계획" if has_vu_banks else "VU 제외 진단 빌드의 Pblock 계획"
    word(42, 60, title, 36, INK, True)
    word(44, 119, "색은 큰 블록을 배치할 수 있는 범위입니다. 개별 유닛의 실제 위치는 배치 후 결정됩니다.", 17, GRAY)
    word(44, 167, "플랫폼 동적 영역", 17, INK, True)
    word(553, 167, "각 Pblock에 지정할 블록", 17, INK, True)

    left, top, width, height = 60, 207, 410, 405
    rect(left, top, left+width, top+height, "#e4e9ee")
    for s in (2, 1, 0):
        for x0, y0, x1, y1 in shapes[s]:
            rect(left + width*x0/169, top + height*(899-y1)/900,
                 left + width*(x1+1)/169, top + height*(900-y0)/900, COLORS[s])
    for i in (1, 2):
        y = top + i * height/3
        rect(left, y-1, left+width, y+1, "#ffffff")
    word(left+width*0.59, top+height/2-9, "플랫폼 셸", 16, GRAY, True)
    for s in (2, 1, 0):
        y = top + (2-s)*135
        word(79, y+52, f"SLR{s}", 21, "#122d3c", True)
        rect(517, y+9, 523, y+125, COLORS[s])
        word(550, y+9, f"SLR{s}", 21, INK, True)
        items = labels(groups[s])
        wrapped_items(550, y+39, items or ["커널 블록 별도 지정 없음"])
    word(60, 631, "실제 SLICE 범위 기준 · 회색은 해당 Pblock 밖", 14, GRAY)
    rect(42, 709, 1078, 780, "#eff4f7")
    word(61, 719, "Vivado 자율 배치", 16, INK, True)
    placed_dma = any(path.endswith("/u_dma") for paths in groups.values() for path in paths)
    free = ("" if placed_dma else "DMA·KVST, ") + "중앙 gather, "
    free += ("VU 출력·중앙 축약, " if placed_dma else "VU 중앙 제어·축약, ") if has_vu_banks else ""
    if not groups[1] and has_vu_banks:
        free = "SPM, RV·제어, DDR 브리지·중재기, DMA·KVST, gather, VU 중앙부, "
    word(61, 748, free + "issue·중앙 debug, 기타 연결 로직 등", 16, GRAY)
    word(44, 795, "K4 major-block floorplan   ·   단위 셀 / 개별 배선 지정 없음", 12, GRAY)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hook", type=Path)
    parser.add_argument("pblock_properties", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.hook, args.pblock_properties, args.output)

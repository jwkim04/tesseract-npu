"""logoclean: turn a downloaded SVG logo into the sanitised file docs/fig/logos/<key>.svg.

A logo is third-party data. This tool never executes or follows anything in it. It reads the file
as XML, keeps only drawing elements and presentation attributes from a fixed whitelist, folds CSS
(`<style>` classes and `style="..."`) into presentation attributes (so two logos in one figure
cannot collide on class names), prefixes every id with the logo key, and refuses (exit 1) on
anything it does not know: scripts, event handlers, external or non-fragment links,
`<foreignObject>`, `<image>`, `<text>` (and so remote fonts), DOCTYPE or ENTITY declarations. It
drops metadata, editor namespaces, comments, titles and descriptions. Coordinates and colours are
copied as they are: the logo is not recoloured, stretched or cropped. Output is deterministic.

    python -I docs/fig/src/logoclean.py <downloaded.svg> <key> [--out docs/fig/logos]

Then check that the stored file renders like the download (docs/fig/logos/SOURCES.md lists the
check, `logoclean.py --compare <downloaded.svg> <stored.svg>` runs it; needs cairosvg, Pillow).
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
MAX_BYTES = 1_000_000

# Elements that are drawing content (kept) and elements that carry no drawing (dropped).
KEEP = {"svg", "g", "defs", "path", "circle", "ellipse", "rect", "polygon", "polyline", "line",
        "clipPath", "linearGradient", "radialGradient", "stop"}
DROP = {"style", "metadata", "title", "desc", "namedview"}      # namedview: Inkscape editor state

# Presentation properties (kept as attributes; from `style` and from CSS classes) and the other
# attributes a kept element may carry.
PRESENTATION = {"fill", "fill-opacity", "fill-rule", "clip-rule", "stroke", "stroke-width",
                "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "stroke-opacity",
                "stroke-dasharray", "stroke-dashoffset", "opacity", "stop-color", "stop-opacity",
                "clip-path", "display"}
GEOMETRY = {"d", "cx", "cy", "r", "rx", "ry", "x", "y", "width", "height", "points", "x1", "y1",
            "x2", "y2", "fx", "fy", "offset", "transform", "gradientUnits", "gradientTransform",
            "spreadMethod", "clipPathUnits", "id", "viewBox", "preserveAspectRatio", "href",
            "version", "xmlns"}
IGNORED_PROPS = {"enable-background", "block-progression", "color", "direction", "text-indent",
                 "text-align", "text-transform", "mix-blend-mode", "isolation", "overflow",
                 "marker", "marker-start", "marker-mid", "marker-end"}  # no effect on a flat logo
# Attributes that are dropped without comment (editor state, no drawing effect).
IGNORED_ATTRS = {"class", "style", "enable-background", "baseProfile", "data-name"}
SAFE_VALUE = re.compile(r"^[A-Za-z0-9#%.,;:()\s_/+\-'\"]*$")


def local(tag: str) -> tuple[str, str]:
    """('{ns}name' -> (ns, name))."""
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        return ns, name
    return "", tag


def fail(msg: str):
    raise SystemExit(f"logoclean: refused: {msg}")


def css_rules(root: ET.Element) -> dict[str, dict[str, str]]:
    """`.class { prop: value; }` rules of every <style>; anything else in a style sheet is refused."""
    rules: dict[str, dict[str, str]] = {}
    for el in root.iter():
        if local(el.tag)[1] != "style":
            continue
        sheet = re.sub(r"/\*.*?\*/", "", el.text or "", flags=re.S)
        if re.search(r"@|url\(|expression|javascript", sheet, flags=re.I):
            fail("style sheet with @-rule, url() or script")
        for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", sheet):
            names = [s.strip() for s in sel.split(",")]
            if not all(re.fullmatch(r"\.[A-Za-z_][\w-]*", n) for n in names):
                fail(f"style selector {sel.strip()!r} is not a plain class")
            decl = parse_decls(body)
            for n in names:
                rules.setdefault(n[1:], {}).update(decl)
    return rules


def parse_decls(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in body.split(";"):
        if not part.strip():
            continue
        k, sep, v = part.partition(":")
        if not sep:
            fail(f"bad CSS declaration {part.strip()!r}")
        out[k.strip()] = v.strip()
    return out


def check_value(name: str, value: str):
    low = value.lower()
    if "javascript" in low or "data:" in low or "<" in value or "http:" in low or "https:" in low:
        fail(f"attribute {name}={value[:60]!r} looks external or active")
    if not SAFE_VALUE.match(value):
        fail(f"attribute {name} has unexpected characters")


def clean(el: ET.Element, key: str, rules, ids: dict[str, str], root: bool = False):
    """Return (tag, [(attr, value)], children, text-free) of the sanitised element, or None."""
    ns, name = local(el.tag)
    if ns not in ("", SVG_NS):
        if name == "namedview":
            return None
        fail(f"element {{{ns}}}{name} (editor or foreign namespace)")
    if name in DROP:
        return None
    if name not in KEEP:
        fail(f"element <{name}> is not whitelisted")
    props: dict[str, str] = {}
    for cls in (el.get("class") or "").split():
        props.update(rules.get(cls, {}))
    props.update(parse_decls(el.get("style") or ""))
    attrs: list[tuple[str, str]] = []
    for aname, value in el.attrib.items():
        ans, an = local(aname)
        if ans == XLINK_NS and an == "href":
            an = "href"
        elif ans not in ("", XLINK_NS) and ans:
            continue                                   # inkscape:, sodipodi:, xmlns decls
        if an.lower().startswith("on"):
            fail(f"event handler attribute {aname}")
        if an in IGNORED_ATTRS or aname.startswith("xmlns"):
            continue
        if an == "href":
            if not value.startswith("#"):
                fail(f"non-fragment href {value[:60]!r}")
        elif an not in PRESENTATION and an not in GEOMETRY:
            fail(f"attribute {aname} on <{name}> is not whitelisted")
        if an in PRESENTATION and an not in props:
            props[an] = value
            continue
        if an in PRESENTATION:
            continue                                   # style/class has priority over attributes
        value = re.sub(r"\s+", " ", value).strip()
        check_value(an, value)
        attrs.append((an, value))
    for k, v in props.items():
        if k in IGNORED_PROPS:
            continue
        if k not in PRESENTATION:
            fail(f"CSS property {k} on <{name}> is not whitelisted")
        v = re.sub(r"\s+", " ", v).strip()
        check_value(k, v)
        attrs.append((k, v))
    children = []
    for ch in el:
        if not isinstance(ch.tag, str):
            continue                                   # comments, processing instructions
        c = clean(ch, key, rules, ids, False)
        if c:
            children.append(c)
    if (el.text or "").strip() and name != "style":
        fail(f"text content in <{name}>")
    return (name, attrs, children)


def drop_unused_ids(node):
    """Remove `id` attributes that no url(#id) or href="#id" refers to (editor leftovers)."""
    refs: set[str] = set()

    def walk(n):
        for a, v in n[1]:
            refs.update(re.findall(r"url\(#([^)]+)\)", v))
            if a == "href":
                refs.add(v[1:])
        for c in n[2]:
            walk(c)
    walk(node)

    def strip(n):
        return (n[0], [(a, v) for a, v in n[1] if a != "id" or v in refs], [strip(c) for c in n[2]])
    return strip(node)


def rename_ids(node, key: str):
    """Prefix every id with the key and rewrite url(#id) / href="#id" references."""
    name, attrs, children = node
    ids = {}

    def collect(n):
        for a, v in n[1]:
            if a == "id":
                ids[v] = f"{key}-{v}"
        for c in n[2]:
            collect(c)
    collect(node)

    def rewrite(n):
        out = []
        for a, v in n[1]:
            if a == "id":
                v = ids[v]
            elif a == "href":
                if v[1:] not in ids:
                    fail(f"href to unknown id {v}")
                v = "#" + ids[v[1:]]
            else:
                def sub(m):
                    if m.group(1) not in ids:
                        fail(f"{a} refers to unknown id {m.group(1)}")
                    return f"url(#{ids[m.group(1)]})"
                v = re.sub(r"url\(#([^)]+)\)", sub, v)
            out.append((a, v))
        return (n[0], out, [rewrite(c) for c in n[2]])
    return rewrite(node)


def esc(v: str) -> str:
    return v.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def write(node, indent: int = 0) -> str:
    name, attrs, children = node
    pad = " " * indent
    a = "".join(f' {k}="{esc(v)}"' for k, v in attrs)
    if not children:
        return f"{pad}<{name}{a}/>\n"
    return f"{pad}<{name}{a}>\n" + "".join(write(c, indent + 1) for c in children) + f"{pad}</{name}>\n"


def sanitise(data: bytes, key: str) -> str:
    if len(data) > MAX_BYTES:
        fail("file larger than 1 MB")
    if re.search(rb"<!DOCTYPE|<!ENTITY", data, flags=re.I):
        fail("DOCTYPE or ENTITY declaration")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        fail(f"not well-formed XML: {e}")
    if local(root.tag) != (SVG_NS, "svg"):
        fail("root element is not an SVG <svg>")
    vb = root.get("viewBox")
    if not vb:
        w, h = (re.sub(r"[a-z]+$", "", root.get(k, "")) for k in ("width", "height"))
        if not (w and h):
            fail("no viewBox and no width/height")
        vb = f"0 0 {w} {h}"
    nums = vb.replace(",", " ").split()
    if len(nums) != 4 or not all(re.fullmatch(r"-?[0-9.]+", n) for n in nums):
        fail(f"bad viewBox {vb!r}")
    rules = css_rules(root)
    name, attrs, children = clean(root, key, rules, {}, True)
    keep = [(a, v) for a, v in attrs if a in ("fill", "fill-rule", "stroke", "opacity")]
    keep = [("xmlns", SVG_NS), ("viewBox", " ".join(nums))] + keep
    tree = (name, keep, [c for c in children if not (c[0] == "defs" and not c[2])])
    node = rename_ids(drop_unused_ids(tree), key)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + write(node)


def compare(a: Path, b: Path, size: int = 512) -> int:
    """Render both files with cairosvg on white at the same height; return the number of pixels
    that differ by more than 2 grey levels (0: the sanitised logo draws like the download)."""
    import io

    import cairosvg
    from PIL import Image, ImageChops
    ims = []
    for p in (a, b):
        png = cairosvg.svg2png(bytestring=p.read_bytes(), output_height=size)
        im = Image.open(io.BytesIO(png)).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        ims.append(bg.convert("RGB"))
    if ims[0].size != ims[1].size:
        return -1
    hist = ImageChops.difference(*ims).convert("L").histogram()
    return sum(hist[3:])


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "--compare":
        n = compare(Path(argv[1]), Path(argv[2]))
        print(f"differing pixels: {n}")
        return 0 if n == 0 else 1
    out_dir = Path(__file__).resolve().parent.parent / "logos"
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = Path(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) != 2 or not re.fullmatch(r"[a-z][a-z0-9-]*", argv[1]):
        print(__doc__, file=sys.stderr)
        return 2
    src, key = Path(argv[0]), argv[1]
    text = sanitise(src.read_bytes(), key)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{key}.svg"
    dst.write_text(text, encoding="utf-8")
    print("wrote", dst, f"({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

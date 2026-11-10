#!/usr/bin/env python3
"""hw/viz/term2mp4.py — render a `script` recording (util-linux, classic timing) to an MP4 video.

usage: term2mp4.py <typescript> <timing> <out.mp4> [--cols 120] [--rows 34] [--fps 15]
                   [--size 17] [--ff START:END:SECONDS ...] [--hold 3]

Record with:  script -q -T timing.log -c 'stty cols 120 rows 34; ./demo.sh' typescript.log
The video is in real time: every delay of the recording is kept, so the rates shown are the ones
measured. `--ff` is the one exception: it plays the span START..END of the recording (seconds from
its start) in SECONDS of video, and shows "sped up Nx" in the top right corner while it plays (use
it for a model load, never for a span that shows a rate). Glyphs that the mono font lacks come from
DejaVu Sans Mono, and emoji from Noto Color Emoji.
Interpreter: build/tools/svgvenv/bin/python (Pillow, pyte, fontTools, imageio-ffmpeg).
"""
import argparse
import subprocess

import imageio_ffmpeg
import pyte
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

BG, FG = (24, 26, 33), (220, 223, 228)
NAMED = {
    "black": (40, 42, 54), "red": (255, 110, 103), "green": (95, 250, 104), "brown": (254, 251, 103),
    "yellow": (254, 251, 103), "blue": (87, 199, 255), "magenta": (255, 119, 198), "cyan": (154, 237, 254),
    "white": (241, 241, 240), "brightblack": (104, 104, 104), "brightred": (255, 110, 103),
    "brightgreen": (95, 250, 104), "brightyellow": (254, 251, 103), "brightblue": (87, 199, 255),
    "brightmagenta": (255, 119, 198), "brightcyan": (154, 237, 254), "brightwhite": (255, 255, 255),
}
FONT_DIR = "/usr/share/fonts/truetype/noto/"
FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf"
EMOJI = FONT_DIR + "NotoColorEmoji.ttf"       # a bitmap font, only valid at size 109
ZERO_WIDTH = {c: None for c in (*range(0xFE00, 0xFE10), *range(0x200B, 0x2010), 0x2060, 0xFEFF)}
# Box drawing is drawn, not typeset, so that lines meet across cells: (up, down, left, right) weights,
# 0 none, 1 light, 2 heavy. The rounded corners are arcs.
BOX = {"─": (0, 0, 1, 1), "━": (0, 0, 2, 2), "│": (1, 1, 0, 0), "┃": (2, 2, 0, 0),
       "┌": (0, 1, 0, 1), "┏": (0, 2, 0, 2), "┐": (0, 1, 1, 0), "┓": (0, 2, 2, 0),
       "└": (1, 0, 0, 1), "┗": (2, 0, 0, 2), "┘": (1, 0, 1, 0), "┛": (2, 0, 2, 0),
       "├": (1, 1, 0, 1), "┤": (1, 1, 1, 0), "┬": (0, 1, 1, 1), "┴": (1, 0, 1, 1), "┼": (1, 1, 1, 1),
       "╭": (0, 1, 0, 1), "╮": (0, 1, 1, 0), "╯": (1, 0, 1, 0), "╰": (1, 0, 0, 1)}
ROUND = {"╭": 180, "╮": 270, "╯": 0, "╰": 90}           # start angle of the quarter circle
QUADRANT = {"▘": 0b1000, "▝": 0b0100, "▖": 0b0010, "▗": 0b0001, "▚": 0b1001, "▞": 0b0110,
            "▙": 0b1011, "▛": 0b1110, "▜": 0b1101, "▟": 0b0111}   # bits: upper left, upper right, lower left, lower right
SHADE = {"░": 0.25, "▒": 0.5, "▓": 0.75}


def colour(c, default):
    if c == "default":
        return default
    if c in NAMED:
        return NAMED[c]
    if len(c) == 6:
        try:
            return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return default


class Glyphs:
    """Draws one character in a cell: the mono font, else DejaVu Sans Mono, else a colour emoji."""

    def __init__(self, size, cw, ch):
        self.cw, self.ch = cw, ch
        self.font = {b: ImageFont.truetype(FONT_DIR + "NotoSansMono-%s.ttf" % ("Bold" if b else "Regular"), size)
                     for b in (False, True)}
        self.alt = {b: ImageFont.truetype(FALLBACK % ("-Bold" if b else ""), size) for b in (False, True)}
        self.has = TTFont(FONT_DIR + "NotoSansMono-Regular.ttf").getBestCmap()
        self.alt_has = TTFont(FALLBACK % "").getBestCmap()
        self.emoji = ImageFont.truetype(EMOJI, 109)
        self.emoji_cache = {}
        self.lw = max(1, round(size / 10))                # light line width

    def cell_shape(self, dr, x, y, c, fill, bg):
        """Block elements and box drawing; False if c is neither."""
        cw, ch, lw = self.cw, self.ch, self.lw
        o = ord(c)
        if c in SHADE:
            f = SHADE[c]
            dr.rectangle([x, y, x + cw - 1, y + ch - 1], fill=tuple(int(b + (g - b) * f) for g, b in zip(fill, bg)))
        elif c == "▀":
            dr.rectangle([x, y, x + cw - 1, y + ch // 2 - 1], fill=fill)
        elif 0x2581 <= o <= 0x2588:                      # lower eighths, up to the full block
            dr.rectangle([x, y + ch - ch * (o - 0x2580) // 8, x + cw - 1, y + ch - 1], fill=fill)
        elif 0x2589 <= o <= 0x258F:                      # left eighths
            dr.rectangle([x, y, x + cw * (0x2590 - o) // 8 - 1, y + ch - 1], fill=fill)
        elif c == "▐":
            dr.rectangle([x + cw // 2, y, x + cw - 1, y + ch - 1], fill=fill)
        elif c in QUADRANT:
            for bit, (qx, qy) in enumerate(((1, 1), (0, 1), (1, 0), (0, 0))):
                if QUADRANT[c] >> bit & 1:
                    dr.rectangle([x + qx * cw // 2, y + qy * ch // 2, x + (qx + 1) * cw // 2 - 1,
                                  y + (qy + 1) * ch // 2 - 1], fill=fill)
        elif c in BOX:
            up, down, left, right = BOX[c]
            cx, cy = x + cw // 2, y + ch // 2
            r = cw // 2 if c in ROUND else 0
            for wt, box in ((up, lambda w: [cx - w // 2, y, cx - w // 2 + w - 1, cy - r]),
                            (down, lambda w: [cx - w // 2, cy + r, cx - w // 2 + w - 1, y + ch - 1]),
                            (left, lambda w: [x, cy - w // 2, cx - r, cy - w // 2 + w - 1]),
                            (right, lambda w: [cx + r, cy - w // 2, x + cw - 1, cy - w // 2 + w - 1])):
                if wt:
                    dr.rectangle(box(lw * wt), fill=fill)
            if r:                                        # the corner arc, centre one radius inside the bend
                a = ROUND[c]
                ax = cx + (r if c in "╭╰" else -r)
                ay = cy + (r if c in "╭╮" else -r)
                dr.arc([ax - r, ay - r, ax + r, ay + r], a, a + 90, fill=fill, width=lw)
        else:
            return False
        return True

    def emoji_image(self, text, cells):
        key = (text, cells)
        if key not in self.emoji_cache:
            big = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
            ImageDraw.Draw(big).text((0, 0), text, font=self.emoji, embedded_color=True)
            img = big.crop(big.getbbox() or (0, 0, 1, 1))
            scale = min(cells * self.cw / img.width, self.ch * 0.9 / img.height)
            self.emoji_cache[key] = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                                               Image.LANCZOS)
        return self.emoji_cache[key]

    def draw(self, img, dr, x, y, text, fill, bg, bold, cells):
        cp = ord(text[0])
        if 0x2500 <= cp < 0x25A0 and self.cell_shape(dr, x, y, text[0], fill, bg):
            return
        if cp in self.has:
            dr.text((x, y), text, fill=fill, font=self.font[bold])
        elif cp in self.alt_has:
            dr.text((x, y), text, fill=fill, font=self.alt[bold])
        elif cp >= 0x2190:
            em = self.emoji_image(text, cells)
            img.paste(em, (x + (cells * self.cw - em.width) // 2, y + (self.ch - em.height) // 2), em)
        else:
            dr.text((x, y), text, fill=fill, font=self.font[bold])


class Screen(pyte.Screen):
    def draw(self, data):
        """pyte drops the rest of a chunk at a zero width character that does not combine (emoji
        selectors, joiners): take them out first."""
        super().draw(data.translate(ZERO_WIDTH))

    def report_device_status(self, mode, **kwargs):
        """Queries such as CSI ? 996 n go to the terminal, not the screen: ignore them (pyte rejects them)."""


def parse_ff(spec):
    s, e, d = (float(v) for v in spec.split(":"))
    if not 0 <= s < e or d <= 0:
        raise SystemExit("--ff %s: need 0 <= START < END and SECONDS > 0" % spec)
    return s, e, d


def video_time(t, segs):
    """Video time of recording time t, with the --ff spans (sorted, not overlapping) squeezed."""
    v, prev = 0.0, 0.0                 # video and recording time at the end of the last span passed
    for s, e, d in segs:
        if t <= s:
            break
        v += s - prev
        if t < e:
            return v + (t - s) * d / (e - s)
        v += d
        prev = e
    return v + t - prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("typescript")
    ap.add_argument("timing")
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--rows", type=int, default=34)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, default=17, help="font size in px")
    ap.add_argument("--ff", action="append", default=[], metavar="START:END:SECONDS",
                    help="play recording seconds START..END in SECONDS of video (repeatable)")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the last screen")
    a = ap.parse_args()

    reg = ImageFont.truetype(FONT_DIR + "NotoSansMono-Regular.ttf", a.size)
    cw = int(round(reg.getlength("M")))
    ch = int(a.size * 1.35)
    pad = 16
    W = (a.cols * cw + 2 * pad + 1) // 2 * 2
    H = (a.rows * ch + 2 * pad + 1) // 2 * 2
    glyphs = Glyphs(a.size, cw, ch)
    tag_font = ImageFont.truetype(FONT_DIR + "NotoSansMono-Bold.ttf", a.size)

    segs = sorted(parse_ff(s) for s in a.ff)
    for (_, e0, _), (s1, _, _) in zip(segs, segs[1:]):
        if s1 < e0:
            raise SystemExit("--ff spans overlap")
    # the spans in video time: (start, end, speed-up)
    squeezed = [(video_time(s, segs), video_time(e, segs), (e - s) / d) for s, e, d in segs]

    data = open(a.typescript, "rb").read()
    nl = data.find(b"\n")  # `script` writes a "Script started on ..." header line first
    if data.startswith(b"Script started"):
        data = data[nl + 1:]
    events = []
    for line in open(a.timing):
        d, n = line.split()
        events.append((float(d), int(n)))

    screen = Screen(a.cols, a.rows)
    stream = pyte.ByteStream(screen)

    def frame(tag):
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)
        for y in range(a.rows):
            row = screen.buffer[y]
            for x in range(a.cols):
                c = row[x]
                fg, bg = colour(c.fg, FG), colour(c.bg, BG)
                if c.reverse:
                    fg, bg = bg, fg
                X, Y = pad + x * cw, pad + y * ch
                cells = 2 if x + 1 < a.cols and row[x + 1].data == "" else 1     # a wide character
                if bg != BG:
                    dr.rectangle([X, Y, X + cells * cw - 1, Y + ch - 1], fill=bg)
                if c.data.strip():
                    glyphs.draw(img, dr, X, Y, c.data, fg, bg, c.bold, cells)
                if c.underscore and c.data:
                    dr.line([X, Y + ch - 3, X + cells * cw - 1, Y + ch - 3], fill=fg)
        if not screen.cursor.hidden:
            X, Y = pad + screen.cursor.x * cw, pad + screen.cursor.y * ch
            dr.rectangle([X, Y + ch - 4, X + cw - 1, Y + ch - 2], fill=FG)
        if tag:
            w = int(tag_font.getlength(tag))
            dr.rectangle([W - pad - w - 4, pad, W - pad, pad + ch - 2], fill=(254, 251, 103))
            dr.text((W - pad - w - 2, pad), tag, fill=BG, font=tag_font)
        return img.tobytes()

    def tag_at(vt):
        for s, e, f in squeezed:
            if s <= vt < e:
                return " sped up %dx " % round(f)
        return ""

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out],
                           stdin=subprocess.PIPE)
    version, nframes, cache = 0, 0, (None, None, b"")   # screen version; frames written; last render

    def emit():
        nonlocal nframes, cache
        tag = tag_at(nframes / a.fps)
        if cache[:2] != (version, tag):                 # render only when the screen or the tag changed
            cache = (version, tag, frame(tag))
        enc.stdin.write(cache[2])
        nframes += 1

    t, pos = 0.0, 0
    for d, n in events:
        t += d
        while nframes / a.fps < video_time(t, segs):    # frames up to the time this chunk arrives
            emit()
        stream.feed(data[pos:pos + n])
        pos += n
        version += 1
    for _ in range(int(a.fps * a.hold)):                # hold the last screen
        emit()
    enc.stdin.close()
    enc.wait()
    print(f"wrote {a.out}: {W}x{H}, {nframes} frames, {nframes / a.fps:.1f} s")


if __name__ == "__main__":
    main()

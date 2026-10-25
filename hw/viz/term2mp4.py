#!/usr/bin/env python3
"""hw/viz/term2mp4.py — render a `script` recording (util-linux, classic timing) to an MP4 video.

usage: term2mp4.py <typescript> <timing> <out.mp4> [--cols 120] [--rows 34] [--fps 15]

Record with:  script -q -T timing.log -c 'stty cols 120 rows 34; ./demo.sh' typescript.log
The video is in real time: every delay of the recording is kept, so the rates shown are the ones
measured. Interpreter: build/tools/svgvenv/bin/python (Pillow, pyte, imageio-ffmpeg).
"""
import argparse
import subprocess

import imageio_ffmpeg
import pyte
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("typescript")
    ap.add_argument("timing")
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--rows", type=int, default=34)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, default=17, help="font size in px")
    a = ap.parse_args()

    reg = ImageFont.truetype(FONT_DIR + "NotoSansMono-Regular.ttf", a.size)
    bold = ImageFont.truetype(FONT_DIR + "NotoSansMono-Bold.ttf", a.size)
    cw = int(round(reg.getlength("M")))
    ch = int(a.size * 1.35)
    pad = 16
    W = (a.cols * cw + 2 * pad + 1) // 2 * 2
    H = (a.rows * ch + 2 * pad + 1) // 2 * 2

    data = open(a.typescript, "rb").read()
    nl = data.find(b"\n")  # `script` writes a "Script started on ..." header line first
    if data.startswith(b"Script started"):
        data = data[nl + 1:]
    events = []
    for line in open(a.timing):
        d, n = line.split()
        events.append((float(d), int(n)))

    screen = pyte.Screen(a.cols, a.rows)
    stream = pyte.ByteStream(screen)

    def frame():
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
                if bg != BG:
                    dr.rectangle([X, Y, X + cw - 1, Y + ch - 1], fill=bg)
                if c.data.strip():
                    dr.text((X, Y), c.data, fill=fg, font=bold if c.bold else reg)
        if not screen.cursor.hidden:
            X, Y = pad + screen.cursor.x * cw, pad + screen.cursor.y * ch
            dr.rectangle([X, Y + ch - 4, X + cw - 1, Y + ch - 2], fill=FG)
        return img.tobytes()

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out],
                           stdin=subprocess.PIPE)
    t, pos, nframes, last = 0.0, 0, 0, None
    for d, n in events:
        t += d
        while nframes / a.fps < t:          # frames up to the time this chunk arrives
            last = last if last is not None else frame()
            enc.stdin.write(last)
            nframes += 1
        stream.feed(data[pos:pos + n])
        pos += n
        last = frame()
    for _ in range(a.fps * 3):              # hold the last screen for 3 s
        enc.stdin.write(last or frame())
        nframes += 1
    enc.stdin.close()
    enc.wait()
    print(f"wrote {a.out}: {W}x{H}, {nframes} frames, {nframes / a.fps:.1f} s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Video of a Sherrington-Kirkpatrick spin glass run of tsr-ising.

Reads a dump of `tsr-ising --snap-every K --dump FILE` and draws, for every snapshot, the spins as an image, the energy
per spin against the step with Parisi's value, and the histogram of the local stabilities s_i h_i / sqrt(N) (h_i is the
field of all the other spins on spin i). Writes a GIF, a PNG of the last frame, and with --ffmpeg an MP4. The number of
spins must be a square (1024, 4096, 16384).
"""
import argparse
import math
import struct
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

PARISI = -0.7632
BG = '#FBFBF9'
INK = '#1F2328'
MUTED = '#6E7781'
BLUE = '#0B6FB3'
ORANGE = '#D9620D'
GRID = '#E3E3DE'


class Dump:
    def __init__(self, path):
        raw = open(path, 'rb').read()
        h = struct.unpack_from('<32I', raw, 0)
        if h[0] != 0x50445349 or h[1] != 2:
            sys.exit(f'{path}: not a tsr-ising dump of version 2')
        self.kind, self.n_orig, self.n, self.t = h[2], h[3], h[4], h[5]
        self.steps, self.snap_every, self.flags, self.nsnap = h[6], h[7], h[8], h[9]
        self.seed = h[12] | (h[13] << 32)
        self.a0, self.c0, self.dt = struct.unpack_from('<3f', raw, 14 * 4)
        rows, self.curve_first, self.curve_every, nmeta = h[17], h[18], h[19], h[20]
        off = 32 * 4
        res = np.frombuffer(raw, '<f8', self.t * 2, off).reshape(self.t, 2)
        self.energy, self.score = res[:, 0], res[:, 1]
        off += self.t * 16
        self.curve = np.frombuffer(raw, '<f8', rows * self.t, off).reshape(rows, self.t)
        off += rows * self.t * 8 + nmeta * 4
        self.x = np.frombuffer(raw, '<f4', self.t * self.n, off).reshape(self.t, self.n)
        off += self.t * self.n * 4
        self.snaps = np.frombuffer(raw, '<f4', self.nsnap * self.t * self.n, off).reshape(self.nsnap, self.t, self.n)
        if self.kind != 2:
            sys.exit(f'{path}: not an sk run')
        if self.nsnap == 0:
            sys.exit(f'{path}: no snapshots (run tsr-ising with --snap-every)')


def couplings_times(seed, n, s, rows_per_block=512):
    """J s for the columns of s (n x frames, +-1 as float32); J is the matrix of tsr-ising's sk kind."""
    out = np.empty(s.shape, np.float32)
    j = np.arange(n, dtype=np.uint64)[None, :]
    with np.errstate(over='ignore'):
        for r0 in range(0, n, rows_per_block):
            i = np.arange(r0, min(n, r0 + rows_per_block), dtype=np.uint64)[:, None]
            a, b = np.minimum(i, j), np.maximum(i, j)
            z = (np.uint64(seed) ^ ((a << np.uint64(32)) | b)) + np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z ^= z >> np.uint64(31)
            jb = np.where((z >> np.uint64(32)) & np.uint64(1), np.float32(1), np.float32(-1))
            jb[i == j] = 0
            out[r0:r0 + jb.shape[0]] = jb @ s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dump')
    ap.add_argument('--out', default='sk', help='prefix of the output files')
    ap.add_argument('--replica', type=int, default=-1, help='replica to show (default: the best one)')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--hold', type=float, default=2.5, help='seconds on the last frame')
    ap.add_argument('--ffmpeg', help='ffmpeg program for the MP4 (no MP4 without it)')
    ap.add_argument('--gif-frames', type=int, default=70)
    ap.add_argument('--stills', default='', help='comma list of frame numbers to save as PNG')
    a = ap.parse_args()

    d = Dump(a.dump)
    n = d.n_orig
    side = int(round(math.sqrt(n)))
    if side * side != n:
        sys.exit('the spin image needs a square number of spins (--n 4096, 16384, ...)')
    rep = int(np.argmin(d.energy)) if a.replica < 0 else a.replica
    norm = n ** 1.5
    x = d.snaps[:, rep, :n]                                   # frames x n
    s = np.where(x >= 0, 1.0, -1.0).astype(np.float32)
    steps = (np.arange(d.nsnap) + 1) * d.snap_every

    # The fields of every frame, and a check of this file's copy of the generator against the run's own energies.
    h = couplings_times(d.seed, n, np.ascontiguousarray(s.T)).T    # frames x n
    lam = s * h / math.sqrt(n)
    e_frames = -0.5 * (s * h).sum(1) / norm
    # The run's energy of a state: its curve row (the state after k steps is row (k - first) / every), or the final
    # energy for the state after the last step.
    rows = (steps - d.curve_first) // d.curve_every
    known = ((steps - d.curve_first) % d.curve_every == 0) & (rows < d.curve.shape[0])
    ref = np.where(known, d.curve[np.minimum(rows, d.curve.shape[0] - 1), rep], np.nan) / norm
    if steps[-1] == d.steps:
        ref[-1] = d.energy[rep] / norm
    ok = ~np.isnan(ref)
    if not ok.any() or not np.allclose(e_frames[ok], ref[ok], rtol=0, atol=2e-6):
        sys.exit('the energies of the snapshots differ from the run\'s: the couplings here are not the run\'s')
    against = (lam < 0).sum(1)

    e_all = d.curve / norm                                    # rows x t
    e_steps = d.curve_first + d.curve_every * np.arange(d.curve.shape[0])

    plt.rcParams.update({'font.size': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8})
    W, H = 1280, 720
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    cmap = LinearSegmentedColormap.from_list('spin', ['#F3F0E8', '#8FA6BA', '#16324F'])

    fig.text(60 / W, 668 / H, 'Sherrington–Kirkpatrick spin glass', fontsize=24, weight='bold', va='center')

    ax_i = fig.add_axes([60 / W, 84 / H, 512 / W, 512 / H])
    im = ax_i.imshow(np.zeros((side, side)), cmap=cmap, vmin=-1, vmax=1, interpolation='nearest', aspect='auto')
    ax_i.set_xticks([]); ax_i.set_yticks([])
    for sp in ax_i.spines.values():
        sp.set_color(MUTED)
    t_step = fig.text(1220 / W, 668 / H, '', fontsize=15, color=MUTED, va='center', ha='right')

    ax_e = fig.add_axes([690 / W, 372 / H, 530 / W, 190 / H], facecolor=BG)
    ax_e.set_xlim(0, d.steps); ax_e.set_ylim(-0.83, -0.28)
    ax_e.axhline(PARISI, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax_e.text(d.steps * 0.015, PARISI - 0.012, 'ground state of an infinite system (Parisi): −0.7632',
              fontsize=10.5, color=INK, ha='left', va='top', bbox=dict(facecolor=BG, edgecolor='none', pad=1.5))
    ax_e.grid(color=GRID, lw=0.8); ax_e.set_axisbelow(True)
    for k in ('top', 'right'):
        ax_e.spines[k].set_visible(False)
    ax_e.set_title('energy per spin, E / N$^{3/2}$', fontsize=13.5, loc='left', pad=8)
    ax_e.tick_params(labelsize=10.5)
    line_e, = ax_e.plot([], [], color=BLUE, lw=1.8)
    dot_e, = ax_e.plot([], [], 'o', color=BLUE, ms=6)
    t_e = ax_e.text(0.985, 0.93, '', transform=ax_e.transAxes, fontsize=15, color=BLUE, weight='bold',
                    ha='right', va='top')

    ax_h = fig.add_axes([690 / W, 84 / H, 530 / W, 190 / H], facecolor=BG)
    edges = np.linspace(-3.0, 4.5, 76)
    mid = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    hist = np.stack([np.histogram(lam[f], edges)[0] for f in range(d.nsnap)]) / (n * width)
    bars = ax_h.bar(mid, hist[0], width=width * 0.92, color=[ORANGE if m < 0 else BLUE for m in mid], linewidth=0)
    ax_h.set_xlim(edges[0], edges[-1]); ax_h.set_ylim(0, max(0.62, hist.max() * 1.05))
    ax_h.axvline(0, color=INK, lw=0.9)
    ax_h.grid(color=GRID, lw=0.8, axis='y'); ax_h.set_axisbelow(True)
    for k in ('top', 'right'):
        ax_h.spines[k].set_visible(False)
    ax_h.set_title('how firmly each spin sits in the field of the others', fontsize=13.5, loc='left', pad=8)
    ax_h.set_xlabel('s$_i$ h$_i$ / √N   (below zero: the spin points against its field)', fontsize=11)
    ax_h.set_yticks([]); ax_h.tick_params(labelsize=10.5)
    t_h = ax_h.text(0.015, 0.95, '', transform=ax_h.transAxes, fontsize=13, color=ORANGE, weight='bold',
                    ha='left', va='top')

    def draw(f):
        im.set_data(x[f].reshape(side, side))
        t_step.set_text(f'step {steps[f]:,} / {d.steps:,}')
        k = int(np.searchsorted(e_steps, steps[f], side='right'))
        line_e.set_data(e_steps[:k], e_all[:k, rep])
        dot_e.set_data([steps[f]], [e_frames[f]])
        t_e.set_text(f'{e_frames[f]:.4f}'.replace('-', '−'))
        for b, v in zip(bars, hist[f]):
            b.set_height(v)
        t_h.set_text(f'{against[f]:,} spins\nagainst their field')
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    frames = range(d.nsnap)
    hold = int(round(a.hold * a.fps))
    ff = None
    if a.ffmpeg:
        ff = subprocess.Popen([a.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s', f'{W}x{H}', '-r', str(a.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'slow',
                               '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out + '.mp4'],
                              stdin=subprocess.PIPE)
    gif_pick = set(np.unique(np.round(np.linspace(0, d.nsnap - 1, min(a.gif_frames, d.nsnap))).astype(int)).tolist())
    stills = [int(v) for v in a.stills.split(',') if v]
    gif = []
    last = None
    for f in frames:
        last = draw(f)
        if ff:
            ff.stdin.write(last.tobytes())
        if f in gif_pick:
            gif.append(Image.fromarray(last).resize((W // 2, H // 2), Image.BOX))   # every spin stays a sharp square
        if f in stills:
            Image.fromarray(last).save(f'{a.out}_frame{f:04d}.png')
    Image.fromarray(last).save(a.out + '_last.png')
    if ff:
        for _ in range(hold):
            ff.stdin.write(last.tobytes())
        ff.stdin.close()
        if ff.wait() != 0:
            sys.exit('ffmpeg failed')
    if gif:
        pal = gif[len(gif) * 2 // 3].quantize(colors=48,method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q = [g.quantize(palette=pal, dither=Image.Dither.NONE) for g in gif]
        per = max(20, int(round(1000 * d.nsnap / (len(gif) * a.fps))))
        q[0].save(a.out + '.gif', save_all=True, append_images=q[1:], loop=0, optimize=False,
                  duration=[per] * (len(q) - 1) + [int(a.hold * 1000)])
    print(f'replica {rep}: e {e_frames[-1]:.4f} ({e_frames[-1] / PARISI * 100:.1f} % of Parisi), '
          f'{against[-1]} of {n} spins against their field at the last snapshot, {d.nsnap} frames')


if __name__ == '__main__':
    main()

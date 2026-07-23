#!/usr/bin/env python3
"""Visual aids for the FFT-as-coherent-sum intuition.

Two figures, both from one seeded simulation of the real geometry (L = 128
window phasors, pilot below per-window noise, off-grid rotation):

fig_phasor_walk   The head-to-tail picture. Each window's row sum is an
                  arrow; a coherent sum chains them tip to tail. Unshifted,
                  the off-grid rotation curls the chain into a ball (this is
                  the naive sum, bin 0). With the right phase schedule the
                  chain straightens and the resultant is ~L long. With a
                  wrong schedule it curls again. One FFT bin = one chain.

fig_hypothesis_bank  All 256 chains' resultants at once, which is what the
                  FFT is. The pilot's bin spikes (its schedule undoes the
                  rotation), bin 0 shows what the naive sum would have seen,
                  the rest form the noise floor. Same energy in and out
                  (Parseval); the FFT only concentrates it.

    python3 scripts/plot_coherence_aids.py --out out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise.plots import (                              # noqa: E402
    BASELINE, GRID, INK, INK2, MUTED, SERIES, SURFACE, _save, setup_style)
import matplotlib.pyplot as plt                           # noqa: E402

L, LF = 128, 256          # windows per frame, padded fine bins
DELTA = 37.6 / LF         # off-grid rotation, cycles per window
SNR_W = 0.5               # per-window amplitude SNR: pilot invisible alone


def make_frame(seed=7):
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
    tone = SNR_W * np.exp(2j * np.pi * DELTA * np.arange(L))
    return tone + noise


def walk(z):
    return np.concatenate([[0], np.cumsum(z)])


def fig_phasor_walk(outfile: Path):
    setup_style()
    z = make_frame()
    ell = np.arange(L)
    cases = [
        ("(a) no phase schedule: the naive sum\n(= bin 0 of the FFT)",
         z, MUTED),
        ("(b) the right schedule: every arrow\nderotated into line ---"
         " the pilot's bin", z * np.exp(-2j * np.pi * DELTA * ell), SERIES[0]),
        ("(c) a wrong schedule: curls again ---\nevery other bin",
         z * np.exp(-2j * np.pi * (DELTA + 6.0 / LF) * ell), SERIES[1]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    for ax, (title, zz, c) in zip(axes, cases):
        w = walk(zz)
        ax.plot(w.real, w.imag, color=c, lw=1.2, alpha=0.85, zorder=3)
        tot = w[-1]
        ax.annotate("", xy=(tot.real, tot.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.plot([0], [0], "o", ms=4, color=INK, zorder=6)
        ax.annotate(f"$|\\Sigma| = {abs(tot):.1f}$",
                    xy=(0.04, 0.90), xycoords="axes fraction", fontsize=10,
                    color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
        ax.set_title(title, fontsize=9.5, loc="left", pad=8)
        cx = np.concatenate([w.real, [0]])
        cy = np.concatenate([w.imag, [0]])
        mx, my = 0.5 * (cx.min() + cx.max()), 0.5 * (cy.min() + cy.max())
        half = 0.58 * max(cx.max() - cx.min(), cy.max() - cy.min(), 12.0)
        ax.set_xlim(mx - half, mx + half)
        ax.set_ylim(my - half, my + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, color=GRID, lw=0.5)
    fig.suptitle("One frame's 128 window phasors, chained tip to tail: "
                 "a coherent sum is a walk, and the phase schedule decides "
                 "whether it goes anywhere",
                 fontsize=10.5, y=1.02)
    fig.text(0.5, -0.045,
             "Same 128 arrows in every panel (pilot at half the per-window "
             "noise amplitude --- invisible in any single window). "
             "The FFT computes all 256 possible walks at once.",
             ha="center", fontsize=9, color=INK2)
    fig.subplots_adjust(wspace=0.12, bottom=0.06)
    return _save(fig, outfile)


def fig_hypothesis_bank(outfile: Path, n_feeds=32):
    """The bank, after the real pipeline's last step: the incoherent feed
    sum. Each feed carries the same pilot with a different (random)
    geometric phase; |FFT|^2 erases that phase, so every feed's spike lands
    in the same bin, while the noise floor averages flat. The figure is the
    reason the incoherent sum costs the detection nothing."""
    setup_style()
    rng = np.random.default_rng(11)
    ell = np.arange(L)
    spec = np.zeros(LF)
    for m in range(n_feeds):
        noise = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
        tone = SNR_W * np.exp(2j * np.pi * (DELTA * ell + rng.uniform()))
        spec += np.abs(np.fft.fft(tone + noise, LF)) ** 2
    spec /= n_feeds * L                     # per-bin noise power -> 1
    bins = np.arange(LF)
    pilot_bin = int(round(DELTA * LF))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                  gridspec_kw=dict(width_ratios=[1.55, 1.0]))

    ax.semilogy(bins, spec, color=SERIES[0], lw=1.0, zorder=3,
                drawstyle="steps-mid")
    ax.plot([pilot_bin], [spec[pilot_bin]], "o", ms=7, color=SERIES[1],
            zorder=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.annotate("the pilot's bin: the one schedule that undoes\nthe "
                f"rotation --- $\\times{spec[pilot_bin]:.0f}$ over the "
                "floor, in every feed at once\n(the geometric phase moves "
                "no energy between bins)",
                xy=(pilot_bin, spec[pilot_bin]),
                xytext=(pilot_bin + 26, spec[pilot_bin] * 1.35),
                fontsize=9, color=SERIES[1],
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.8))
    ax.plot([0], [spec[0]], "s", ms=7, color=MUTED, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.annotate("bin 0: the naive unshifted\nsum --- lost in the floor",
                xy=(0, spec[0]), xytext=(9, 3.1), fontsize=9,
                color=INK2, va="bottom",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkB=3))
    ax.axhline(1.0, color=BASELINE, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("noise floor: 255 wrong schedules, each a\ncurled walk, "
                "averaged flat by the feed sum",
                xy=(215, 1.0), xytext=(0.985, 0.42),
                textcoords="axes fraction", ha="right", fontsize=9,
                color=INK2,
                arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.8))
    ax.set_xlim(-4, LF + 2)
    ax.set_ylim(0.25, spec[pilot_bin] * 12)
    ax.set_xlabel("fine bin $f$  (= assumed rotation rate = one phase "
                  "schedule)")
    ax.set_ylabel("power in that bin's walk  [noise = 1]")
    ax.set_title("The FFT is all 256 coherent sums at once",
                 loc="left", fontsize=10.5, pad=8)

    # ---- right panel: the two-axes map --------------------------------
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.add_patch(plt.Rectangle((1.2, 2.2), 6.2, 5.6, facecolor=GRID,
                                edgecolor=MUTED, lw=0.8))
    for x in np.linspace(1.2, 7.4, 9):
        ax2.plot([x, x], [2.2, 7.8], color=SURFACE, lw=0.6)
    for y in np.linspace(2.2, 7.8, 8):
        ax2.plot([1.2, 7.4], [y, y], color=SURFACE, lw=0.6)
    ax2.add_patch(plt.Rectangle((1.2, 4.6), 6.2, 0.8, facecolor=SERIES[0],
                                alpha=0.45, edgecolor="none"))
    ax2.add_patch(plt.Rectangle((3.5, 2.2), 0.78, 5.6, facecolor=SERIES[2],
                                alpha=0.45, edgecolor="none"))
    ax2.annotate("windows $\\ell$ (time) $\\rightarrow$", xy=(4.3, 1.62),
                 ha="center", fontsize=9, color=INK2)
    ax2.annotate("feeds $m$ $\\rightarrow$", xy=(0.75, 5.0), ha="center",
                 fontsize=9, color=INK2, rotation=90)
    ax2.annotate("FFT along this row = fine spectrum:\nbeamforming in "
                 "time (this work)", xy=(7.7, 5.0), fontsize=9,
                 color=SERIES[0], va="center")
    ax2.annotate("FFT along a column = sky beams:\nbeamforming in space "
                 "(CHIME/FRB)", xy=(3.9, 8.6), fontsize=9, color=SERIES[2],
                 ha="center", va="bottom")
    ax2.annotate("feeds add in power rather than phase: each feed sees\nthe tower "
                 "at a geometric phase we decline to model",
                 xy=(4.3, 0.35), ha="center", va="bottom", fontsize=8.5,
                 color=INK2)
    ax2.set_title("Same maneuver, two axes", loc="left", fontsize=10.5,
                  pad=8)

    fig.subplots_adjust(wspace=0.26, bottom=0.16)
    return _save(fig, outfile)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print("wrote", fig_phasor_walk(args.out / "fig_phasor_walk.png"))
    print("wrote", fig_hypothesis_bank(args.out / "fig_hypothesis_bank.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

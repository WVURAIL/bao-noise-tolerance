#!/usr/bin/env python3
"""Spectral anatomy of an ATSC A/53 channel, with the detector-scale inset.

A schematic amplitude response drawn from the standard's numbers rather
than from data:

    - the 8-VSB payload shelf occupies B_N ~ 5.381 MHz of the 6 MHz
      allocation, with raised-cosine skirts filling the allocation exactly
      (each skirt spans twice the pilot offset, so the -6 dB points sit at
      the pilot offset and its mirror);
    - the pilot line sits at Delta f = 177/572 MHz above the lower edge,
      the -6 dB point of the lower skirt, carrying 7.4 percent of the
      payload power at about 56 dB power-spectral-density contrast;
    - the legacy NTSC aural position (250 kHz below the upper edge, at
      5.75 MHz) falls in the upper roll-off.

The inset draws the detector's operating scale about the pilot, three
orders of magnitude below the allocation scale: the target tone, the
references displaced +/-2 coarse-grid bins (+/-6.1 kHz), and one guard
bin on each side.

    python3 scripts/plot_vsb_layout.py --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


from baonoise.plots import (
    INK, INK2, MUTED, SERIES, _save, setup_style)
import matplotlib.pyplot as plt

DF_PILOT = 177.0 / 572.0        # MHz above the lower allocation edge
ALLOC = 6.0                     # MHz
NTSC_AURAL = ALLOC - 0.25       # legacy aural carrier, 5.75 MHz
FINE_BIN_KHZ = 390.625 / 128.0  # coarse-grid bin at the detector, ~3.05 kHz


def shelf(f: np.ndarray) -> np.ndarray:
    """Raised-cosine-skirted amplitude response filling the allocation."""
    w = 2.0 * DF_PILOT          # each skirt spans twice the pilot offset
    a = np.ones_like(f)
    lo = f < w
    hi = f > ALLOC - w
    a[lo] = 0.5 * (1.0 - np.cos(np.pi * f[lo] / w))
    a[hi] = 0.5 * (1.0 - np.cos(np.pi * (ALLOC - f[hi]) / w))
    a[(f < 0) | (f > ALLOC)] = 0.0
    return a


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    fig, ax = plt.subplots(figsize=(6.4, 3.1))

    f = np.linspace(-0.05, ALLOC + 0.25, 2001)
    a = shelf(f)
    ax.fill_between(f, 0, a, color=SERIES[0], alpha=0.16, lw=0, zorder=2)
    ax.plot(f, a, color=SERIES[0], lw=1.6, zorder=3)

    # pilot line at the -6 dB point of the lower skirt
    ax.plot([DF_PILOT, DF_PILOT], [0, 1.06], color=INK, lw=2.4, zorder=4,
            solid_capstyle="butt")
    ax.annotate("pilot line, $\\Delta f = 177/572$ MHz\n"
                "7.4% of payload power;\n"
                "$\\approx 56$ dB PSD contrast",
                xy=(0.40, 1.43), ha="left", va="top", fontsize=8.5, color=INK)
    ax.annotate("8-VSB payload shelf ($B_N \\approx 5.381$ MHz)",
                xy=(0.42, 0.66), ha="left", fontsize=8.5, color=INK2)

    # legacy NTSC aural position in the upper roll-off
    ax.plot([NTSC_AURAL, NTSC_AURAL], [0, 0.92], color=INK, lw=1.2,
            ls=(0, (1.6, 1.6)), zorder=4)
    ax.annotate("legacy NTSC aural\n(co-channel), 5.75 MHz",
                xy=(5.60, 0.26), ha="right", fontsize=8.5, color=INK2)

    # pilot-offset bracket under the axis
    ax.annotate("", xy=(DF_PILOT, 0.10), xytext=(0, 0.10),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.annotate("$\\Delta f_{\\mathrm{pilot}}$", xy=(0.5 * DF_PILOT, 0.032),
                ha="center", fontsize=8, color=INK)

    ax.set_xlim(-0.08, 6.28)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([])
    ax.set_xticks(range(7))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("frequency above lower allocation edge (MHz)")

    # ---- inset: the detector's operating scale about the pilot -----------
    ib = fig.add_axes([0.455, 0.55, 0.30, 0.33])
    ib.set_facecolor("#f4f3ef")
    ref = 2.0 * FINE_BIN_KHZ                     # +/-6.1 kHz
    for sign in (-1, 1):
        ib.axvspan(sign * 0.5 * FINE_BIN_KHZ, sign * 1.5 * FINE_BIN_KHZ,
                   color="#dddbd3", lw=0, zorder=1)
        ib.annotate("guard", xy=(sign * FINE_BIN_KHZ, 0.34), rotation=90,
                    ha="center", va="center", fontsize=8, color=INK2)
        ib.plot([sign * ref, sign * ref], [0, 0.58], color=SERIES[0],
                lw=2.0, zorder=3, solid_capstyle="butt")
        ib.annotate("ref", xy=(sign * ref, 0.62), ha="center", fontsize=8.5,
                    color=SERIES[0])
    ib.plot([0, 0], [0, 0.82], color=INK, lw=2.4, zorder=4,
            solid_capstyle="butt")
    ib.annotate("target", xy=(0, 0.86), ha="center", fontsize=8.5,
                color=INK)
    ib.set_xlim(-9.5, 9.5)
    ib.set_ylim(0, 1.05)
    ib.set_yticks([])
    ib.set_xticks([-ref, 0, ref])
    ib.set_xticklabels(["$-6.1$", "0", "$+6.1$"], fontsize=8)
    ib.grid(False)
    for s in ib.spines.values():
        s.set_visible(False)
    ib.set_xlabel("kHz about the pilot", fontsize=8.5, labelpad=2)
    ib.set_title("detector scale (Ch. IV)", fontsize=8.5, color=MUTED,
                 pad=4)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.15)
    return _save(fig, args.out / "fig_vsb_layout.png")


if __name__ == "__main__":
    print("wrote", main())

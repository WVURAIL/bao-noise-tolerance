#!/usr/bin/env python3
"""The two +/-2 displacements as dual applications of the Dirichlet response.

One spectral response function, two readings:

    (a) On the coarse grid, the references are placed at the kernel's
        decayed skirts. A tone at worst-case half-bin offset leaks -3.9 dB
        into the adjacent bin but only -13.5 dB at +/-2, so the one-bin
        guard bounds self-contamination of the background estimate below
        5 percent in power.
    (b) On the padded fine grid, the window-axis kernel's main lobe spans
        exactly +/-2 bins, so the designated detection window is the
        main-lobe extent: an exclusion guard on one grid, a capture window
        on the other.

Everything is computed from the K-tap Dirichlet kernel

    D_K(x) = sin(pi x) / (K sin(pi x / K)),

with K = 128 taps on the coarse grid and the same kernel read on the
2x-padded fine grid in panel (b).

    python3 scripts/plot_dirichlet_duality.py --out out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise.plots import (                              # noqa: E402
    GRID, INK, INK2, SERIES, _save, setup_style)
import matplotlib.pyplot as plt                           # noqa: E402

K = 128


def dirichlet(x: np.ndarray) -> np.ndarray:
    """|D_K(x)| for displacement x in bins, with the removable singularity."""
    x = np.asarray(x, dtype=float)
    num = np.sin(np.pi * x)
    den = K * np.sin(np.pi * x / K)
    out = np.ones_like(x)
    nz = np.abs(den) > 1e-12
    out[nz] = np.abs(num[nz] / den[nz])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.8))

    # ---- (a) coarse grid: reference placement at the decayed skirts ------
    x = np.linspace(-4.3, 4.3, 4001)
    db = lambda v: 20 * np.log10(np.maximum(v, 1e-12))  # noqa: E731
    ax.axvspan(-2, 2, color=GRID, alpha=0.55, lw=0, zorder=1)
    (dotted,) = ax.plot(x, db(dirichlet(x)), color=INK2, lw=0.9,
                        ls=(0, (1.5, 1.5)), zorder=3,
                        label="tone at $\\delta = 0$")
    (solid,) = ax.plot(x, db(dirichlet(x - 0.5)), color=SERIES[0], lw=1.4,
                       zorder=4,
                       label="tone at $\\delta = 1/2$ (worst case)")
    for dk, txt, dy in ((1, "$-3.9$ dB", 2.2), (2, "$-13.5$ dB", 2.2)):
        v = db(dirichlet(np.array([dk - 0.5])))[0]
        ax.plot([dk], [v], "o", ms=4.5, color=INK, zorder=5)
        ax.annotate(txt, xy=(dk, v), xytext=(dk + 0.25, v + dy),
                    fontsize=8.5, color=INK)
    ax.annotate("exclusion guard (references at $\\pm 2$)", xy=(0, 1.0),
                ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-44, 4)
    ax.set_xticks(range(-4, 5, 2))
    ax.set_xlabel("coarse-grid displacement $\\Delta k$ (bins)")
    ax.set_ylabel("$|D_K|$ (dB)")
    ax.set_title("(a) reference placement: kernel decay", loc="left",
                 fontsize=9.5, pad=6)
    ax.legend(handles=[solid, dotted], loc="lower left", fontsize=8)

    # ---- (b) padded fine grid: the same kernel as a capture window -------
    xf = np.linspace(-6.5, 6.5, 4001)
    bx.axvspan(-2, 2, color=SERIES[0], alpha=0.15, lw=0, zorder=1)
    for edge in (-2, 2):
        bx.axvline(edge, color=INK2, lw=0.9, dashes=(3.0, 1.6), zorder=2)
    bx.plot(xf, dirichlet(xf / 2), color=SERIES[0], lw=1.4, zorder=3)
    bx.annotate("designated window $\\pm 2$ = main-lobe extent",
                xy=(0, 1.035), ha="center", va="bottom", fontsize=8.5,
                color=INK2)
    bx.set_xlim(-6.5, 6.5)
    bx.set_ylim(0, 1.12)
    bx.set_xticks(range(-6, 7, 2))
    bx.set_xlabel("padded fine-grid offset from tone (bins)")
    bx.set_ylabel("$|D_L|$")
    bx.set_title("(b) capture window: main-lobe extent", loc="left",
                 fontsize=9.5, pad=6)

    fig.subplots_adjust(wspace=0.30, bottom=0.17, top=0.90)
    return _save(fig, args.out / "fig_dirichlet_duality.png")


if __name__ == "__main__":
    print("wrote", main())

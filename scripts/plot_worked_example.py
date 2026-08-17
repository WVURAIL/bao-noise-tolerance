#!/usr/bin/env python3
"""The pipeline chapter's worked example: two channel-506 fine spectra.

Two frames from the channel-506 calibration-era cohort, drawn one above the
other with everything the decision rule sees: the designated window
f_a +/- 2 shaded, the usable-bulk bins marked, the bulk median (solid) and
an illustrative threshold eta * T_(rho) (dashed) drawn.

    (a) the exemplar masked frame (2025-07-31 15:52:22 UT): window maximum
        18.615, a factor of 18.4 over the bulk median 1.009; the coarse
        flag concurs (F/mu0 = 1.258).
    (b) the cohort's weakest valid frame (2025-05-16): the coarse statistic
        sits below the positive-excess boundary (F/mu0 = 0.897), yet the
        pilot stands at 3.1x the bulk median on the fine axis. This is the
        coherent-integration sensitivity on survey data, and the reason the
        fine designated-set rule rather than the coarse flag is deployed.

The usable bulk is the alternate (independent) bins of the padded fine
grid, less the window bins they would overlap: the 128 even bins minus
{60, 62, 64}, giving |B| = 125. The secondary leakage feature near bin 168
of the exemplar frame lies on a bulk bin, and the rank baseline does not
move: the robustness property, exhibited incidentally on real data.

Data: data/worked_example_ch506.csv (provenance in its header). The script
asserts every statistic the dissertation quotes before drawing, so a stale
or wrongly regenerated data file fails loudly rather than plotting.

    python3 scripts/plot_worked_example.py --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from baonoise.plots import (
    INK, INK2, SERIES, _save, setup_style)
import matplotlib.pyplot as plt

DATA = ROOT / "data" / "worked_example_ch506.csv"
ANCHOR = 62                 # survey-measured anchor f_a for channel 506
WINDOW = (60, 64)           # designated window f_a +/- 2
ETA = 1.5                   # illustrative threshold multiplier, median rank


def usable_bulk() -> np.ndarray:
    """Alternate (independent) bins minus the window bins they overlap."""
    return np.array([b for b in range(0, 256, 2) if b not in (60, 62, 64)])


def require(condition, message: str) -> None:
    """Keep reconstruction gates active even when Python runs with ``-O``."""
    if not condition:
        raise RuntimeError(message)


def check(T_a: np.ndarray, T_b: np.ndarray, bulk: np.ndarray) -> None:
    """Validate every statistic the dissertation quotes for this example."""
    med_a = float(np.median(T_a[bulk]))
    med_b = float(np.median(T_b[bulk]))
    require(len(bulk) == 125, f"expected 125 bulk samples, got {len(bulk)}")
    require([round(v, 3) for v in T_a[60:65]] ==
            [1.066, 8.730, 18.615, 7.598, 1.083],
            f"worked-example pilot bins changed: {T_a[60:65]}")
    require(round(med_a, 3) == 1.009,
            f"unexpected panel-A median: {med_a}")
    require(abs(float(np.percentile(T_a[bulk], 90)) - 1.165) < 0.005,
            "unexpected panel-A 90th percentile")
    require(round(med_b, 3) == 0.833,
            f"unexpected panel-B median: {med_b}")
    require(round(T_a[60:65].max() / med_a, 1) == 18.4,
            "unexpected panel-A peak ratio")
    require(round(float(T_b[60:65].max()), 2) == 2.59,
            "unexpected panel-B peak")
    require(round(T_b[60:65].max() / med_b, 1) == 3.1,
            "unexpected panel-B peak ratio")


def panel(ax, T, med, title, label_med, label_eta):
    bulk = usable_bulk()
    bins = np.arange(256)
    ax.axvspan(WINDOW[0] - 0.5, WINDOW[1] + 0.5, color=SERIES[0],
               alpha=0.15, lw=0, zorder=1)
    ax.semilogy(bins, T, color=SERIES[0], lw=0.9, zorder=3)
    ax.plot(bulk, T[bulk], "o", ms=2.6, color=INK2, alpha=0.7, mew=0,
            zorder=4)
    ax.axhline(med, color=INK, lw=0.9, zorder=2)
    ax.axhline(ETA * med, color=INK, lw=0.9, dashes=(3.0, 1.3), zorder=2)
    ax.annotate(label_med, xy=(253, med), ha="right", va="bottom",
                fontsize=8.5, color=INK, zorder=6)
    ax.annotate(label_eta, xy=(253, ETA * med), ha="right", va="bottom",
                fontsize=8.5, color=INK2, zorder=6)
    ax.set_xlim(0, 255)
    ax.set_ylabel("$T[f]$")
    ax.set_title(title, loc="left", fontsize=9.5, pad=6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    d = np.genfromtxt(DATA, delimiter=",")
    T_a, T_b = d[:, 1], d[:, 2]
    bulk = usable_bulk()
    check(T_a, T_b, bulk)
    med_a = float(np.median(T_a[bulk]))
    med_b = float(np.median(T_b[bulk]))

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 4.4))

    panel(ax, T_a, med_a,
          "(a) exemplar frame, 2025-07-31: $F/\\mu_0 = 1.258$",
          "bulk median 1.009",
          "$\\eta\\,T_{(\\rho)}$ (illustrative, $\\eta = 1.5$, median rank)")
    ax.set_ylim(0.797, 21.6)
    ax.annotate("window max 18.62\n(18.4$\\times$ bulk median)",
                xy=(67, 15.5), ha="left", va="top", fontsize=8.5, color=INK)
    ax.annotate("designated window\n$f_a \\pm 2$", xy=(56, 2.6), ha="right",
                fontsize=8.5, color=SERIES[0])

    panel(bx, T_b, med_b,
          "(b) weakest valid frame, 2025-05-16 ($F/\\mu_0 = 0.897$): "
          "coarse retains, fine rejects",
          "bulk median 0.833",
          "$\\eta\\,T_{(\\rho)}$ (illustrative, $\\eta = 1.5$, median rank)")
    bx.set_ylim(0.274, 3.92)
    bx.annotate("window max 2.59\n(3.1$\\times$ bulk median)",
                xy=(67, 3.1), ha="left", va="top", fontsize=8.5, color=INK)
    bx.set_xlabel("fine bin $f$")

    fig.subplots_adjust(left=0.125, right=0.90, hspace=0.46, top=0.93,
                        bottom=0.11)
    return _save(fig, args.out / "fig_worked_example.png")


if __name__ == "__main__":
    print("wrote", main())

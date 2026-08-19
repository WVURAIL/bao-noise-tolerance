#!/usr/bin/env python3
"""The whole argument in one panel: what each policy leaves behind, against
what the science can absorb.

The natural way to pitch a masking method is "it gets you there faster." That
undersells this one and invites the obvious rebuttal: integrate longer. The
measured result is stronger and different in kind: every alternative leaves a
residual four to five orders of magnitude above the tolerance, and the
tolerance is flat in integration time, so longer does not help. The choice is
not fast versus slow. It is a measurement versus a wrong number with small
error bars.

So the figure is a dot plot on a log residual axis with the tolerance drawn as
a rule. The ten-decade gap is the message, which is why the axis has to show
ten decades. Integration time rides alongside as an annotation rather than a
second axis; two scales in one frame would invite reading a trade-off
between them that does not exist here, since every failing policy fails on
residual regardless of its time cost.

    python3 scripts/plot_the_case.py --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baonoise import incumbent as I
from baonoise import products as _products
from baonoise import residual as R
from baonoise.npzio import load_npz

# Validated categorical pair:
#   node scripts/validate_palette.js "#2a78d6,#eb6834" --pairs all  -> ALL PASS
OTHER, OURS = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"

R_TOL = 1.5e-3          # binding fs8 tolerance at zeta = 1 (Amara & Refregier)
# tau_c is taken from the product rather than pinned: ch33 bounds it at <= 5 min.


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def _require_floor(st):
    """The kept-frame floor, or a refusal. Never a fabricated stand-in.

    This previously fell back to -75.99 dB when the floor was unmeasured,
    which is the *minimum detected shelf* on ch35, a sampling artifact rather than
    a floor, and thirty decibels below the floors actually measured on other
    channels. It made a failing channel look like it passed by four orders of
    magnitude.
    """
    if not np.isfinite(st.floor_db):
        raise R.NoMeasuredFloor(
            f"ch{st.channel} has no null population; its masked residual is "
            f"unmeasured. Run this on a channel with a transmitter-off epoch, "
            f"or state a substituted floor explicitly.")
    return st.floor_db


def policies(npz):
    """(label, residual, time penalty, is_ours) through the full chain."""
    st = R.shelf_statistics(npz)
    corr = R.correlation_time(npz)
    gain = (st.intraday_fraction
            * R.n_coh_from_correlation_time(corr.tau_for_budget)
            + st.fast_fraction)
    d = load_npz(npz)
    valid = d["valid"][:, 0].astype(bool)
    f_dep = float(d["reject_mask"][valid, 0].astype(bool).mean())
    res = {r.name: r for r in I.compare_flaggers(npz)[0]}

    def chain(db):
        return 10.0 ** (db / 10.0) * gain

    rows = [
        ("keep everything", chain(st.on_shelf_db), 0.0, False),
        ("MAD $1.8\\times$ (incumbent)",
         chain(res["MAD 1.8x within acquisition"].shelf_kept_db),
         res["MAD 1.8x within acquisition"].f, False),
        ("spectral kurtosis (incumbent)",
         chain(res["SK 3sigma within acquisition"].shelf_kept_db),
         res["SK 3sigma within acquisition"].f, False),
        ("pilot proxy", chain(_require_floor(st)), f_dep, True),
        # Excision leaves nothing behind and measures nothing. It cannot be
        # plotted on a log residual axis, and the figure presents it that way: it
        # wins the axis the figure shows and loses the one it cannot.
        ("excise the channel", 0.0, 1.0, False),
    ]
    return [(n, r, float("inf") if f >= 1.0 else (1.0 + r) / (1.0 - f), o)
            for n, r, f, o in rows]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=_products.paths(channels=(33,), announce=False).get(33))
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    if args.npz is None:
        ap.error("channel-33 product not found by the manifest; pass --npz")

    style()
    rows = policies(args.npz)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))

    ys = np.arange(len(rows))[::-1]
    for y, (name, r, pen, ours) in zip(ys, rows):
        c = OURS if ours else OTHER
        if r <= 0.0:                       # excision: no residual, no measurement
            xleft = 6.0e-4
            ax.plot([xleft], [y], "<", ms=8, color=c, zorder=4, alpha=0.85)
            ax.annotate("no residual", xy=(xleft, y), xytext=(0, 11),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=c)
            ax.annotate("never", xy=(1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(8, -3), textcoords="offset points",
                        ha="left", fontsize=8, color=INK2)
            continue
        ax.plot([R_TOL, r], [y, y], color=c, lw=1.0, alpha=0.35, zorder=2,
                solid_capstyle="butt")
        ax.plot([r], [y], "o", ms=9, color=c, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        over = r / R_TOL
        txt = (f"{1/over:.0f}$\\times$ inside" if over < 1
               else f"{over:,.0f}$\\times$ over")
        ax.annotate(txt, xy=(r, y), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=7.5, color=c)
        ax.annotate(f"{pen:,.1f}$\\times$ time" if np.isfinite(pen) else "never",
                    xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(8, -3), textcoords="offset points",
                    ha="left", fontsize=8, color=INK2)

    ax.axvline(R_TOL, color=INK, lw=1.2, zorder=3)
    ax.annotate("bias tolerance\n$r_{\\rm tol} = 1.5\\times10^{-3}$",
                xy=(R_TOL, len(rows) - 0.42), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=INK)
    ax.axvspan(1e-9, R_TOL, color=GRID, alpha=0.55, zorder=1, lw=0)

    ax.set_xscale("log")
    ax.set_xlim(4e-4, 3e3)
    ax.set_ylim(-0.90, len(rows) - 0.25)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _, _, _ in rows])
    ax.set_xlabel("residual DTV power surviving to the power spectrum, "
                  "in units of system noise")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Channel 33, full residual chain at the measured floor "
                 "($-44.95$ dB) and $\\tau_c \\leq 5$ min", loc="left", pad=16)
    ax.annotate("only this side is a measurement",
                xy=(R_TOL / 1.4, -0.70), ha="right", fontsize=7.5, color=MUTED)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"fig8_the_case.{ext}", dpi=220,
                    bbox_inches="tight")
    print(f"wrote {args.out}/fig8_the_case.pdf")
    for n, r, pen, _ in rows:
        print(f"  {n:32s} r={r:11.4g}  {pen:9.2f}x time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

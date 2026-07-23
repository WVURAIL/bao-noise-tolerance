#!/usr/bin/env python3
"""The two ways a channel fails, drawn as the two walls of one plane.

Every masking policy for a channel is a point in the (f, r) plane: the
fraction of time it discards, and the residual it leaves in what it keeps.
Tolerance is a box in that plane, and its two edges are two different
failures:

    the bias wall      r > r_tol      the kept data are too dirty
    the occupancy wall f -> 1         nothing is kept at all

The detector's threshold eta traces a curve between them: eta = 1 (positive
excess) is the minimum-r, maximum-f end; raising eta walks toward keep-
everything. A channel is salvageable iff its curve enters the box. The two
impossibility arguments of the introduction are the two walls: no detector
sees below the sensitivity floor (the bias wall's position is fixed), and no
detector reschedules a transmitter (the occupancy wall is where the channel
puts it).

The two failures are nearly mutually exclusive in any one era, because both
are driven by the same variable pulled in opposite directions: duty cycle.
And a heavily occupied channel starves the null sample, so failing the
occupancy wall also destroys the measurement of the bias wall: ch34 keeps 355
frames, ch36 keeps 34, and their r values are bounds from starved estimates.

    python3 scripts/plot_two_walls.py --out out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import residual as R                        # noqa: E402
from baonoise.plots import (                              # noqa: E402
    CRITICAL, GRID, INK, INK2, MUTED, SERIES, SURFACE, _save, setup_style)
import matplotlib.pyplot as plt                           # noqa: E402

# Stable zeta = 1 tolerances on the binding dilation, alpha_perp, per z bin
# (scripts/bias_tolerance.py --zeta 1.0, unrefused entries).
TOL_APERP = {27: 0.014, 28: 0.014, 29: 0.014, 30: 0.0144, 31: 0.0156, 32: 0.0156,
             33: 0.0156, 34: 0.0156, 35: 0.0352, 36: 0.0352}
# fs8 tolerance relative to alpha_perp's, per bin, drawn as a band because
# the ratio differs between the two z bins the five channels occupy.
FS8_REL = (0.00156 / 0.0352, 0.00153 / 0.0156)     # (0.044, 0.098)

FINE_DB = 10.0                                     # measured 9.4-10.0
from baonoise import products as P                 # noqa: E402
PATHS = P.paths(channels=sorted(TOL_APERP))
# Verdict classes: the three threshold-feasible channels in full color, the
# tau_c-hostage marginal in its own color, the occupancy-pinned five in gray.
COLORS = {32: SERIES[0], 33: SERIES[1], 35: SERIES[2], 29: SERIES[3]}
GRAY = MUTED


def channel_curve(ch, p):
    """(f, r/tol) along the threshold family, with provenance flags."""
    prov = R.floor_provenance(p)
    kw = {}
    stated = not np.isfinite(prov.reported_db)
    if stated:                                    # mu0 < 1: state the bound
        kw["floor_db"] = prov.sigma_implied_db
    etas = np.concatenate([np.linspace(1.0, 1.8, 17), np.geomspace(2, 300, 12)])
    sweep = R.threshold_sweep(p, etas=etas, **kw)
    tol = TOL_APERP[ch]
    pts = [(row["f"], row["r_masked"] / 10 ** (FINE_DB / 10) / tol,
            row["eta"]) for row in sweep]
    corr = R.correlation_time(p)
    return dict(ch=ch, pts=pts, stated_floor=stated,
                tau_bound=(corr.quality != "measured"),
                n_kept=prov.n_kept)


def era_point(path, year_from, floor_db, gain):
    """f at eta = 1 restricted to an era, at the same floor and gain."""
    import datetime as dt
    d = np.load(path, allow_pickle=True)
    v = d["valid"][:, 0].astype(bool)
    rej = d["reject_mask"][:, 0].astype(bool)
    t = d["unit_time0_ctime"][d["frame_unit_index"]]
    yr = np.array([dt.datetime.utcfromtimestamp(x).year for x in t])
    m = v & (yr >= year_from)
    return float(rej[m].mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)

    curves = [channel_curve(ch, p) for ch, p in PATHS.items()]

    setup_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    # the admissible box: under the bias wall, left of the occupancy wall
    ax.axhspan(1e-3, 1.0, xmin=0, xmax=1, color=GRID, alpha=0.5, zorder=0, lw=0)
    ax.axhline(1.0, color=INK, lw=1.3, zorder=5)
    ax.axvline(1.0, color=INK, lw=1.3, zorder=5)
    ax.annotate("the bias wall: kept data too dirty "
                r"($r = r_{\rm tol}$, transverse dilation, $\zeta = 1$)",
                xy=(0.02, 1.0), xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=INK, va="bottom")
    ax.annotate("the occupancy wall: nothing left to keep",
                xy=(0.994, 4e-3), xytext=(-10, 0), textcoords="offset points",
                fontsize=9, color=INK, ha="right", va="bottom", rotation=90)
    ax.annotate("tolerable", xy=(0.03, 0.55), fontsize=10, color=INK2,
                style="italic")

    # where fs8's wall would sit, relative to alpha_perp's
    ax.axhspan(*FS8_REL, color=CRITICAL, alpha=0.10, zorder=0, lw=0)
    ax.annotate(r"the $f\sigma_8$ wall falls in this band "
                "(10-25x lower, bin-dependent)",
                xy=(0.02, FS8_REL[0] * 1.15), fontsize=8.5, color=CRITICAL,
                va="bottom")

    for cur in curves:
        ch, pts = cur["ch"], cur["pts"]
        if not pts:
            continue
        c = COLORS.get(ch, GRAY)
        f = np.array([p[0] for p in pts])
        rr = np.array([p[1] for p in pts])
        order = np.argsort([p[2] for p in pts])
        f, rr = f[order], rr[order]
        dashed = cur["stated_floor"] or cur["tau_bound"]
        ax.plot(f, rr, color=c, lw=2.0, zorder=4,
                ls=(0, (4, 2)) if dashed else "-",
                solid_capstyle="round")
        ax.plot([f[0]], [rr[0]], "o", ms=8, color=c, zorder=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        lab_x, lab_y = f[0], rr[0]
        big = ch in COLORS
        off = {30: (6, -13), 34: (6, 6), 28: (6, 7), 31: (6, -14)}.get(ch, (6, 7))
        ax.annotate(f"ch{ch}", xy=(lab_x, lab_y), xytext=off,
                    textcoords="offset points",
                    fontsize=9.5 if big else 8, color=c,
                    fontweight="semibold" if big else "normal")

    # ch35's era dependence: the same channel, 2022 onward
    c35 = next(c for c in curves if c["ch"] == 35)
    if c35["pts"]:
        f_fwd = era_point(PATHS[35], 2022, None, None)
        r35 = c35["pts"][0][1]
        ax.plot([f_fwd], [r35], "X", ms=9, color=COLORS[35], zorder=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate("ch35, 2022 onward:\nthe station lit up and the same\n"
                    "channel moved to the other wall",
                    xy=(f_fwd, r35), xytext=(0.42, 6e-3), fontsize=8.5,
                    color=COLORS[35],
                    arrowprops=dict(arrowstyle="-", color=COLORS[35], lw=0.8,
                                    shrinkA=2, shrinkB=3))

    ax.annotate("Filled dots mark positive excess ($\\eta = 1$): minimum "
                "residual, maximum cost.  Raising $\\eta$ walks each curve "
                "right-to-left toward keep-everything.",
                xy=(0.5, -0.14), xycoords="axes fraction", ha="center",
                fontsize=8.5, color=INK2)

    ax.set_yscale("log")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(2e-3, 3e4)
    ax.set_xlabel("Masked fraction of observing time, $f$")
    ax.set_ylabel(r"Residual over tolerance, "
                  r"$r \, / \, r_{\rm tol}$ (transverse dilation)   [fine stage]")
    ax.set_title("Two ways to fail, one plane: every threshold is a point, "
                 "every channel a curve")
    return _save(fig, args.out / "fig_bao_two_walls.png")


if __name__ == "__main__":
    raise SystemExit(main())

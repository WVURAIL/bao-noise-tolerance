#!/usr/bin/env python3
"""Why more integration does not rescue a contaminated channel.

The intuition everyone reaches for is that a systematic sits at a fixed level
while the statistical error falls, so the two cross and the systematic takes
over. That is not what happens here, and the real answer is worse.

Both fall. In the noise-dominated regime the Fisher information grows as t^2,
so sigma(fs8) falls roughly as 1/t, and the bias a residual induces falls at
essentially the same rate, because it is built from the same integrals. Over
the survey's entire realistic range the ratio |Delta fs8| / sigma(fs8) moves by
a few percent. Integration shrinks the error bar and the bias in lockstep.

So the answer is not "eventually the systematic wins." It is that the answer
is the same number of sigma wrong on day one and after ten years, and it looks
more converged the whole way. That is what the second panel shows: flat lines
where a reader expects descending ones.

The bias-response bank is not shipped. Build it explicitly first:

    python scripts/build_bank.py --config chime2022 --cosmology planck2018 \\
        --p-res 1.0 --dense-knee \\
        --out data/fisher_bank_chime2022_pres_dense.npz
    python scripts/plot_convergence.py --out out/
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baonoise import incumbent as I
from baonoise import products as _products
from baonoise import residual as R
from baonoise import survey
from baonoise.npzio import load_npz

# Validated four-slot categorical palette:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#a8518a" --pairs all
#   ALL CHECKS PASS (worst normal-vision dE 19.0, worst CVD dE 9.2). The aqua
#   sits below 3:1 on this surface, so every series is direct-labeled.
SERIES = {"keep everything": "#2a78d6", "MAD 1.8x": "#a8518a",
          "spectral kurtosis": "#1baf7a", "pilot proxy": "#eb6834"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"

ZETA = 1.0     # Amara & Refregier b <= sigma


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def load_bias_tools():
    spec = importlib.util.spec_from_file_location(
        "bt", str(ROOT / "scripts" / "bias_tolerance.py"))
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    return bt


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


def policy_table(npz):
    st = R.shelf_statistics(npz)
    corr = R.correlation_time(npz)
    gain = (st.intraday_fraction
            * R.n_coh_from_correlation_time(corr.tau_for_budget)
            + st.fast_fraction)
    d = load_npz(npz)
    valid = d["valid"][:, 0].astype(bool)
    f_dep = float(d["reject_mask"][valid, 0].astype(bool).mean())
    res = {r.name: r for r in I.compare_flaggers(npz)[0]}
    ch = lambda db: 10.0 ** (db / 10.0) * gain          # noqa: E731
    return [
        ("keep everything", ch(st.on_shelf_db), 0.0),
        ("MAD 1.8x", ch(res["MAD 1.8x within acquisition"].shelf_kept_db),
         res["MAD 1.8x within acquisition"].f),
        ("spectral kurtosis", ch(res["SK 3sigma within acquisition"].shelf_kept_db),
         res["SK 3sigma within acquisition"].f),
        ("pilot proxy", ch(_require_floor(st)), f_dep),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=_products.paths(channels=(33,), announce=False).get(33))
    ap.add_argument("--bank", default=str(ROOT / "data" /
                                          "fisher_bank_chime2022_pres_dense.npz"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    bt = load_bias_tools()
    try:
        bank = bt.load_bias_bank(args.bank, expected_kfg_fac=None)
    except ValueError as exc:
        ap.error(str(exc))
    if args.npz is None:
        ap.error("channel-33 product not found by the manifest; pass --npz")
    names = list(bank.paramnames)
    zs = bank.zs
    ib = [i for i in range(len(zs) - 1) if abs(zs[i] - 1.40) < 1e-9][0]

    yrs = np.logspace(np.log10(0.05), np.log10(10.0), 40)
    hours = yrs * survey.OVERVIEW_ONSKY_YEAR_HOURS

    def sig_and_slope(t):
        dth, sig = bt.bias_per_unit_r(bank.F(ib, t), names)
        return sig["fs8"], abs(dth["fs8"])

    sig_clean = np.array([sig_and_slope(t)[0] for t in hours])

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax1, ax2 = axes

    # ---- (a) integration does work: the error bar falls -------------------
    ax1.loglog(yrs, sig_clean, color=INK, lw=1.8, zorder=4)
    drop = sig_clean[0] / sig_clean[-1]
    ax1.annotate(f"falls {drop:,.0f}$\\times$\nover this range",
                 xy=(yrs[len(yrs) // 2], sig_clean[len(yrs) // 2]),
                 xytext=(14, 16), textcoords="offset points",
                 fontsize=8, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax1.set_xlabel("integration time [on-sky yr]")
    ax1.set_ylabel("$\\sigma(f\\sigma_8)$, clean survey")
    ax1.set_title("(a)  integration does what it should", loc="left", pad=8)
    ax1.grid(True, which="major", color=GRID, lw=0.5, zorder=0)

    # ---- (b) and buys nothing against a coherent residual -----------------
    rows = policy_table(args.npz)
    drifts = []
    print(f"{'policy':22s} {'r':>11s} {'bias/sigma @0.05yr':>19s} {'@10yr':>11s} {'drift':>7s}")
    for name, r, f in rows:
        t_eff = hours * (1.0 - f) / (1.0 + r)
        ratio = np.array([r * sig_and_slope(t)[1] / sig_and_slope(t)[0]
                          for t in t_eff])
        c = SERIES[name]
        ax2.loglog(yrs, ratio, color=c, lw=1.8, zorder=4, label=name)
        if name == "pilot proxy":
            ax2.annotate("best of the four,\nstill $24\\times$ over",
                         xy=(yrs[-1], ratio[-1]), xytext=(6, -4),
                         textcoords="offset points", fontsize=7.5, color=c,
                         va="center")
        drifts.append(max(ratio) / min(ratio))
        print(f"{name:22s} {r:11.4g} {ratio[0]:19.4g} {ratio[-1]:11.4g} "
              f"{max(ratio)/min(ratio):7.2f}")

    ax2.axhline(ZETA, color=INK, lw=1.2, zorder=5)
    ax2.annotate(f"$\\zeta = {ZETA:g}$, the published criterion",
                 xy=(yrs[0], ZETA), xytext=(2, 7), textcoords="offset points",
                 ha="left", fontsize=7.5, color=INK)
    ax2.axhspan(1e-4, ZETA, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax2.set_xlabel("integration time [on-sky yr]")
    ax2.set_ylabel("$|\\Delta f\\sigma_8| \\, / \\, \\sigma(f\\sigma_8)$")
    ax2.set_title("(b)  and it changes nothing that matters", loc="left", pad=8)
    ax2.set_ylim(1e-3, 1e6)
    ax2.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    ax2.legend(loc="center left", frameon=False, handlelength=1.6,
               labelspacing=0.35, borderaxespad=0.4,
               bbox_to_anchor=(0.03, 0.14))

    fig.subplots_adjust(right=0.76, wspace=0.44)
    fig.subplots_adjust(bottom=0.30)
    fig.text(0.5, -0.02,
             "Channel 33, floor and $\\tau_c$ both measured. Integration shrinks the "
             "error bar and the bias together, so the right-hand curves barely "
             "move:\nthey rise by a third while the error bar falls twentyfold. "
             "A contaminated channel does not converge slowly --- it converges "
             "to the wrong\nanswer, and stays the same number of sigma wrong "
             "the whole way.",
             ha="center", fontsize=7.5, color=INK2)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"fig9_convergence.{ext}", dpi=220,
                    bbox_inches="tight")
    print(f"\nwrote {args.out}/fig9_convergence.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

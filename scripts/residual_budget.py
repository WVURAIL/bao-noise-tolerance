#!/usr/bin/env python3
"""Residual contamination budget from pilot-proxy survey products.

Prints the dB chain that takes a measured DTV shelf to the residual an analysis
actually sees, then sweeps the one term left open, the intra-day correlation
time tau_c, to show what it costs the survey.

    python3 scripts/residual_budget.py 521.npz --off-through 2021-08
    python3 scripts/residual_budget.py 5*.npz --delay bao_peak1 --plot

The sweep is the purpose of this script. Every other term is measured from the product or fixed
by the delay-filter choice. tau_c is bounded on both sides: below by the
acquisition duration, since the products resolve everything faster than that
directly, and above by a sidereal day, since anything slower is m = 0 and has
already been removed as ground-filter suppression. Inside that window it
still spans the difference between "the residual is irrelevant" and "the
residual dominates".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import residual as R          # noqa: E402
from baonoise import forecast, scenarios    # noqa: E402
from baonoise.fisherbank import FisherBank  # noqa: E402
from baonoise.resources import DEFAULT_BANK  # noqa: E402

TAU_GRID = np.array([
    R.CHIME_FRAME_SECONDS, 1.0, 10.0, 60.0, 300.0, 900.0, 3600.0,
    4 * 3600.0, 12 * 3600.0, R.MAX_TAU_C_SECONDS,
])
TAU_LABEL = ["frame", "1 s", "10 s", "1 min", "5 min", "15 min", "1 hr",
             "4 hr", "12 hr", "1 sid.day"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", type=Path,
                    help="pilot-proxy per-pilot survey products")
    ap.add_argument("--off-through", default=None, metavar="YYYY-MM",
                    help="last month of a transmitter-off epoch (sets the "
                         "single-frame sensitivity floor)")
    ap.add_argument("--floor-percentile", type=float, default=90.0,
                    help="percentile of the null-epoch shelf used as the "
                         "kept-frame bound (default 90; 50 is the median "
                         "floor and is not a bound)")
    ap.add_argument("--delay", default=R.DEFAULT_DELAY_KEY,
                    choices=sorted(R.DELAY_SUPPRESSION_DB),
                    help="which BAO feature the delay filter must preserve")
    ap.add_argument("--tau-intraday", type=float, default=None,
                    help="fix the intra-day correlation time [s] instead of "
                         "sweeping (default: sweep)")
    ap.add_argument("--bank", type=Path,
                    default=DEFAULT_BANK)
    ap.add_argument("--target", type=float, default=5.0)
    ap.add_argument("--zbin", type=int, default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    args = ap.parse_args()

    stats, corrs = [], []
    print("=" * 72)
    print("MEASURED TERMS (from the survey products)")
    print("=" * 72)
    for p in args.npz:
        st = R.shelf_statistics(p, off_through=args.off_through,
                                floor_percentile=args.floor_percentile)
        ct = R.correlation_time(p, off_through=args.off_through)
        stats.append(st)
        corrs.append(ct)
        print(st.summary())
        print(ct.summary())
        print()

    usable = [s for s in stats if np.isfinite(s.floor_db)]
    if not usable:
        print("no product yielded a measurable shelf floor; nothing to budget")
        return 1

    print("=" * 72)
    print(f"CHAIN (delay = {args.delay}, "
          f"{R.DELAY_SUPPRESSION_DB[args.delay]:.1f} dB)")
    print("=" * 72)
    worst = max(usable, key=lambda s: s.floor_db)
    worst_ct = corrs[stats.index(worst)]
    tau_star = (args.tau_intraday if args.tau_intraday
                else worst_ct.tau_for_budget)
    print(R.budget_from_statistics(
        worst, args.delay, tau_intraday=tau_star,
        tau_measured=(worst_ct.is_measured and not args.tau_intraday)).chain())
    print()
    if worst_ct.is_measured:
        print(f"tau_c is measured on this channel; the sweep below is a "
              f"sensitivity check, and\nthe operating point is the "
              f"{worst_ct.tau_c / 60:.0f} min row "
              f"({worst_ct.tau_lo / 60:.0f}-{worst_ct.tau_hi / 60:.0f} at 68%).")
    else:
        print("tau_c is NOT measured on this channel; every row below is a "
              "bound, and the\nsidereal-day row is the one to quote.")
    print()

    # ------------------------------------------------------------------
    print("=" * 72)
    print("SWEEP over the residual correlation time")
    print("=" * 72)
    bank = FisherBank(args.bank)
    style = "perbin_A" if bank.meta.get("config") == "chime2022" else "shared_A"
    if style != "perbin_A":
        print("note: this bank needs RadioFisher for the shared_A path")
    fc = forecast.Forecast(bank, None, style=style)
    dtv_chans = scenarios.band_channels("dtv")

    def sweep(bins):
        clean_h = fc.required_hours_metric(
            lambda t: fc.significance(scenarios.clean(), t, bins=bins),
            args.target)
        h0 = fc.required_hours_metric(
            lambda t: fc.significance(scenarios.measured(), t, bins=bins),
            args.target)
        rows = []
        for tau in TAU_GRID:
            n = R.n_coh_from_correlation_time(float(tau))
            b = R.budget_from_statistics(worst, args.delay,
                                         tau_intraday=float(tau))
            sc = scenarios.measured(residuals={c: b.ratio for c in dtv_chans})
            h = fc.required_hours_metric(
                lambda t: fc.significance(sc, t, bins=bins), args.target)
            rows.append((float(tau), n, b.ratio_db, b.ratio, h,
                         h / clean_h if np.isfinite(h) else np.inf))
        return clean_h, h0 / clean_h, rows

    targets = [("survey", None)]
    zbin = args.zbin if args.zbin is not None else _worst_dtv_bin(bank)
    targets.append((f"z bin {zbin} "
                    f"({bank.zs[zbin]:.2f}-{bank.zs[zbin + 1]:.2f})", [zbin]))

    results = {}
    for label, bins in targets:
        _, pen0, rows = sweep(bins)
        results[label] = (pen0, rows)
        print(f"--- {label} ---")
        print(f"masking only (measured rates): penalty {pen0:.4f}")
        print(f"{'tau_c':>8s} {'n_coh':>10s} {'r (dB)':>9s} {'r':>11s} "
              f"{'hours':>10s} {'penalty':>9s}")
        for lab, row in zip(TAU_LABEL, rows):
            print(f"{lab:>8s} {row[1]:10.3g} {row[2]:9.2f} {row[3]:11.4g} "
                  f"{row[4]:10.2f} {row[5]:9.4f}")
        print()

    print("The masking cost is flat down these tables; the residual cost is")
    print("not. Where they cross is where the threshold stops being free to")
    print("lower and starts being the thing you tune.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for (label, (pen0, rows)), c in zip(results.items(), ("C0", "C3")):
            tau = np.array([r[0] for r in rows])
            pen = np.array([r[5] for r in rows])
            ax.axhline(pen0, color=c, ls="--", lw=1.0, alpha=0.6)
            ax.loglog(tau, pen, "o-", color=c, label=label)
        if worst_ct.is_measured:
            ax.axvspan(worst_ct.tau_lo, worst_ct.tau_hi, color="0.85", zorder=0)
            ax.axvline(worst_ct.tau_c, color="0.4", lw=1.0, zorder=0)
            ax.annotate(f"measured $\\tau_c$ = {worst_ct.tau_c / 60:.0f} min",
                        xy=(worst_ct.tau_c, ax.get_ylim()[0]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color="0.3", rotation=90)
        ax.set_xlabel(r"residual correlation time $\tau_c$  [s]")
        ax.set_ylabel(r"time penalty  $t_{\rm req}/t_{\rm req}^{\rm clean}$")
        ax.set_title(f"DTV residual cost vs coherence ({args.delay});"
                     f"\ndashed = masking only")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(args.out / f"fig5_residual_coherence.{ext}", dpi=160)
        print(f"\nwrote {args.out / 'fig5_residual_coherence.png'}")

    return 0


def _worst_dtv_bin(bank) -> int:
    """Index of the redshift bin most fully covered by the ATSC DTV band."""
    from baonoise import channels as chn
    lo_dtv, hi_dtv = chn.channel_edges(14)[0], chn.channel_edges(36)[1]
    best, frac_best = 0, -1.0
    for i in range(bank.nbins):
        nu_lo = chn.NU_LINE / (1.0 + bank.zs[i + 1])
        nu_hi = chn.NU_LINE / (1.0 + bank.zs[i])
        ov = max(0.0, min(nu_hi, hi_dtv) - max(nu_lo, lo_dtv))
        frac = ov / (nu_hi - nu_lo)
        if frac > frac_best:
            best, frac_best = i, frac
    return best


if __name__ == "__main__":
    sys.exit(main())

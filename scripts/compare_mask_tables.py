#!/usr/bin/env python3
"""Diff the vendored quarterly rate table against the survey products.

The forecast's fiducial masking fractions come from a CSV column named
``hi_rate_all`` that records no statistic and no threshold. The survey products
carry their own decision (``reject_mask``) plus the rule that produced it,
in ``detector_contract_json``. When the two disagree, the forecast is pricing a
detector nobody can name.

    python3 scripts/compare_mask_tables.py 506.npz 521.npz 537.npz
    python3 scripts/compare_mask_tables.py products/*.npz --forecast

Run it over every product the scan produces. Any channel with a ratio far from
1 is one where the chapter has to say which number it means.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import channels as chn        # noqa: E402
from baonoise.resources import DEFAULT_BANK  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", type=Path,
                    help="pilot-proxy per-pilot survey products")
    ap.add_argument("--rates-csv", type=Path, default=None,
                    help="override the vendored quarterly rate table")
    ap.add_argument("--warn-ratio", type=float, default=2.0,
                    help="flag channels disagreeing by more than this factor")
    ap.add_argument("--forecast", action="store_true",
                    help="also price both tables through the Fisher bank")
    ap.add_argument("--bank", type=Path,
                    default=DEFAULT_BANK)
    args = ap.parse_args()

    kw = {} if args.rates_csv is None else {"rates_csv": args.rates_csv}
    csv_t = chn.measured_mask_table(**kw)
    prod_t = chn.mask_table_from_products(args.npz)

    print("=" * 70)
    print("SOURCES")
    print("=" * 70)
    print(f"  csv      rule: {csv_t.rule}")
    for n in csv_t.notes:
        print(f"           ! {n}")
    print(f"  products rule: {prod_t.rule}")
    print(f"           detector: {prod_t.detector_version}")
    for n in prod_t.notes:
        print(f"           ! {n}")

    rows = chn.compare_mask_tables(csv_t, prod_t)
    covered = [r for r in rows if r[0] in prod_t.n_frames]
    print()
    print("=" * 70)
    print("PER-CHANNEL (channels the products actually measured)")
    print("=" * 70)
    print(f"{'ch':>4s} {'csv':>9s} {'products':>10s} {'ratio':>9s} "
          f"{'frames':>8s}")
    flagged = []
    for ch, a, b, ratio in covered:
        mark = "  <-- disagree" if ratio > args.warn_ratio else ""
        if mark:
            flagged.append(ch)
        print(f"{ch:4d} {a:9.4f} {b:10.4f} {ratio:9.1f}x "
              f"{prod_t.n_frames.get(ch, 0):8d}{mark}")

    print()
    if flagged:
        print(f"{len(flagged)} of {len(covered)} measured channels disagree by "
              f"more than {args.warn_ratio:g}x: "
              f"{', '.join('ch%d' % c for c in flagged)}")
        print("The two tables are not the same detector at a different "
              "threshold --")
        print("no single cut on any statistic in these products reproduces the "
              "CSV.")
    else:
        print(f"all {len(covered)} measured channels agree within "
              f"{args.warn_ratio:g}x")

    uncovered = sorted(set(csv_t.fractions) - set(prod_t.fractions))
    if uncovered:
        print(f"\nnot covered by these products, so still CSV-only: "
              f"{', '.join('ch%d' % c for c in uncovered)}")

    if args.forecast:
        from baonoise import forecast, scenarios          # noqa: PLC0415
        from baonoise.fisherbank import FisherBank        # noqa: PLC0415

        bank = FisherBank(args.bank)
        fc = forecast.Forecast(bank, None, style="perbin_A")

        def hours(fr, bins=None):
            sc = scenarios.Scenario("x", "x", fractions=fr)
            return fc.required_hours_metric(
                lambda t: fc.significance(sc, t, bins=bins), 5.0)

        clean = hours({})
        merged = dict(csv_t.fractions)
        merged.update(prod_t.fractions)
        print()
        print("=" * 70)
        print("WHAT IT COSTS THE FORECAST (5 sigma, survey level)")
        print("=" * 70)
        for name, fr in [("csv table", csv_t.fractions),
                         ("products where measured", merged)]:
            print(f"  {name:26s} x{hours(fr) / clean:.4f}")
        worst = int(np.argmax([r[3] for r in covered])) if covered else None
        if worst is not None:
            ch = covered[worst][0]
            print(f"\n  worst single channel: ch{ch} "
                  f"({covered[worst][1]:.4f} vs {covered[worst][2]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

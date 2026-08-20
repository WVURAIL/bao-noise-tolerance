#!/usr/bin/env python3
"""Regenerate bao_two_walls.csv from the survey products (bridge retirement).

The two-walls figure plots every channel's coarse threshold sweep in the
occupancy-versus-residual plane: masked fraction f against the kept-frame
residual over the binding tolerance. The committed table was recovered from
the published artwork of the ten-channel era; this generator replaces it,
computed from the released products for all 23 channels under one stated
convention:

* full-archive sweep of F > eta * mu0 (``baonoise.residual.threshold_sweep``);
  channel 35's kept-frame floor comes from its verified pre-sign-on era
  (``off_through = 2021-08``), exactly as the published curve's did, while
  its sweep still covers the full archive;
* tau_c measured where the structure-function estimator's gates pass, else
  the sidereal-day cap (the budget's own convention);
* the kept-frame floor is the product's own null-population floor where at
  least 30 null frames support it (evidence ``measured``); thinner or absent
  floors use the ``floor_provenance`` sigma-implied substitute (evidence
  ``stated``, drawn dashed by the figure);
* the ordinate is the published plane's: the fine-credited residual over the
  stable zeta = 1 dilation tolerance of the channel's own redshift bin,
  (r_masked / 10) / tol_aperp --- the same axis as the operating-point
  optimization. The first-measured block keeps the released TOL_APERP
  constants (scripts/optimal_thresholds.py); the lower band's bins take the
  stable zeta = 1 alpha_perp minima of the dense bias-response bank
  (scripts/bias_tolerance.py --zeta 1.0 on the --p-res 1.0 --dense-knee
  build). Verified against the recovered artwork: channels 29, 32, and 33
  reproduce the published eta = 1 endpoints to the printed digits.

Row order is the figure's: order 0 is the highest threshold and the last
row is the eta = 1 floor, where the figure draws each channel's dot.

    python3 scripts/dissertation/make_two_walls.py --products DIR

STATUS --- generated, verified, not yet adopted by the figure. Against the
recovered artwork the eta = 1 endpoints of channels 29, 32, 33, 34, and 36
reproduce to the printed digits, and the endpoints recovered as "ch27" and
"ch30" match this generator's ch30 and ch27 respectively (the artwork
recovery appears to have swapped those two near-overlapping gray curves).
Three channels move under this generator's stated conventions and block
adoption until reconciled: ch31 and ch28 shift because their 2- and 6-frame
floors fall below the >= 30-null bar and take the sigma-implied substitute;
ch35's endpoint moves from 0.05 to 4.4 because threshold_sweep's residual
budget and optimal_thresholds' kept-frame bound book its measured 45-min
correlation time differently --- the same split that makes Table 9.5's
6.6x margin unreachable on this axis. Reconcile the two budgets (or declare
which one the plane uses) before pointing the figure at this table.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from baonoise import residual as res  # noqa: E402

FINE_DB = 10.0                    # measured fine-stage credit, booked as 10
FLOOR_OFF_THROUGH = {35: "2021-08"}   # sign-on channel: floor from the off era
MIN_MEASURED_NULLS = 30           # thinner floors count as stated, not measured

# Stable zeta = 1 alpha_perp tolerances per channel's redshift bin. Channels
# 27-36: the released constants of scripts/optimal_thresholds.py. Channels
# 14-26: the stable zeta = 1 minima from the dense bias-response bank
# (bins 1.60-1.70: 0.00672, 1.70-1.80: 0.00757, 1.80-1.90: 0.012,
# 1.90-2.04: 0.0201).
TOL_APERP = {
    14: 0.0201, 15: 0.0201, 16: 0.0201, 17: 0.0201,
    18: 0.012, 19: 0.012, 20: 0.012,
    21: 0.00757, 22: 0.00757, 23: 0.00757,
    24: 0.00672, 25: 0.00672, 26: 0.00672,
    27: 0.014, 28: 0.014, 29: 0.014, 30: 0.0144,
    31: 0.0156, 32: 0.0156, 33: 0.0156, 34: 0.0156,
    35: 0.0352, 36: 0.0352,
}


def sweep_channel(path):
    with np.load(path, allow_pickle=False) as z:
        ch = int(z["physical_channel"][0])
    fot = FLOOR_OFF_THROUGH.get(ch)
    ct = res.correlation_time(path)
    tau = ct.tau_for_budget
    _, st, _ = res.budget_from_products(path, off_through=fot)
    measured_floor = (np.isfinite(st.floor_db)
                      and st.n_off_frames >= MIN_MEASURED_NULLS)
    if measured_floor and fot is None:
        rows = res.threshold_sweep(path, tau_intraday=tau)
        return ch, "measured", rows
    if measured_floor:
        rows = res.threshold_sweep(path, tau_intraday=tau,
                                   floor_db=float(st.floor_db))
        return ch, "measured", rows
    fp = res.floor_provenance(path)
    rows = res.threshold_sweep(path, tau_intraday=tau,
                               floor_db=float(fp.sigma_implied_db))
    return ch, "stated", rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent
                    / "data" / "bao_two_walls.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(str(args.products), "*.npz")),
                   key=lambda p: int(Path(p).stem))
    out_rows = []
    for path in paths:
        ch, evidence, rows = sweep_channel(path)
        tol = TOL_APERP[ch]
        # figure order: high threshold first, the eta = 1 floor last (the dot)
        rows = sorted(rows, key=lambda r: -r["eta"])
        for i, r in enumerate(rows):
            out_rows.append(dict(
                channel=ch, evidence=evidence, order=i,
                masked_fraction=f"{r['f']:.8g}",
                r_over_rtol=f"{(r['r_masked'] / FINE_DB) / tol:.6g}"))
        end = out_rows[-1]
        print(f"ch{ch}: {evidence}, {len(rows)} sweep points, "
              f"eta=1 end (f, r/rtol) = ({end['masked_fraction']}, "
              f"{end['r_over_rtol']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["channel", "evidence", "order",
                                               "masked_fraction",
                                               "r_over_rtol"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"{args.out}: {len(out_rows)} rows, "
          f"{len({r['channel'] for r in out_rows})} channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

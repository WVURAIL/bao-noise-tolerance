#!/usr/bin/env python3
"""The threshold the forecast selects, per channel.

This is the closing move of the framework: minimize the survey-time cost

    T / T_clean = (1 + r) / (1 - f)

over the coarse threshold family F > eta * mu0, subject to the bias-tolerance
constraint r <= r_tol on the binding acoustic dilation (alpha_perp, zeta = 1),
with the fine stage's measured sensitivity credit applied to the bound.

Selection discipline:

* The residual bound is computed on TWO floor bases and both are reported.
  ``product`` uses each product's own kept-frame floor where one exists (the
  mu0 sliver), which is the convention the chapter's verdicts currently stand
  on. ``sigma_null`` bounds every undetected frame at the level a threshold
  sitting at the null center can actually resolve; defensible everywhere,
  including the mu0 < 1 channels, and stricter. Where the two disagree about
  feasibility, that disagreement is the finding rather than a nuisance.

* Among near-optimal thresholds (within 2% of the minimum cost) the SMALLEST
  eta is selected: equal cost, more residual margin. Optima on flat plateaus
  are otherwise spuriously precise.

* Channels whose tau_c was refused carry a bound rather than a measurement, and the
  bound is conservative in one direction only: a feasible eta is truly
  feasible, but the reported optimum may be pessimistic; a measured tau_c
  can only enlarge the feasible set. The output marks these.

    python3 scripts/optimal_thresholds.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import residual as R                        # noqa: E402

# Stable zeta = 1 tolerances (scripts/bias_tolerance.py --zeta 1.0).
# per channel, from the stable zeta=1 entries of the channel's z bin:
# ch27-29 z 1.51-1.59 (bin 1.5-1.6); ch30 straddles; ch31-34 bin 1.4-1.5;
# ch35/36 bin 1.3-1.4.
TOL_APERP = {27: 0.014, 28: 0.014, 29: 0.014, 30: 0.0144, 31: 0.0156, 32: 0.0156,
             33: 0.0156, 34: 0.0156, 35: 0.0352, 36: 0.0352}
TOL_FS8 = {27: 0.0016, 28: 0.0016, 29: 0.0016, 30: 0.0016, 31: 0.00153, 32: 0.00153,
           33: 0.00153, 34: 0.00153, 35: 0.00156, 36: 0.00156}
FINE_DB = 10.0                       # measured fine-stage credit, 9.4-10.0 dB
DEPLOYED_DELAY_DB = 11.4             # CHIME's 200 ns cut; NOT booked in the
                                     # verdicts; shown as a labeled scenario.
PLATEAU = 1.02                       # "within 2% of optimal" tie-break window

from baonoise import products as _products               # noqa: E402

DEFAULT_PRODUCTS = _products.paths()

ETAS = np.unique(np.concatenate([
    np.arange(1.00, 1.101, 0.01),          # fine grid at the knee
    np.arange(1.10, 2.01, 0.05),
    np.geomspace(2.0, 300.0, 16),
]))


def recent_f(path, eta, mu0, year_from):
    d = np.load(path, allow_pickle=True)
    v = d["valid"][:, 0].astype(bool)
    F = d["fstat_raw"][:, 0]
    t = d["unit_time0_ctime"][d["frame_unit_index"]]
    yr = np.array([dt.datetime.utcfromtimestamp(x).year for x in t])
    m = v & (yr >= year_from)
    if m.sum() < 100:
        return float("nan")
    return float((F[m] > eta * mu0).mean())


def optimize(path, ch):
    prov = R.floor_provenance(path)
    corr = R.correlation_time(path)
    tol = TOL_APERP[ch]
    d = np.load(path, allow_pickle=True)
    t = d["unit_time0_ctime"]
    yr_max = dt.datetime.utcfromtimestamp(float(t.max())).year
    era_from = max(yr_max - 2, 2018)

    out = {"ch": ch, "mu0": prov.mu0, "tau_bound": corr.quality != "measured",
           "tol_aperp": tol, "era_from": era_from, "bases": {}}

    for basis in ("product", "sigma_null"):
        kw = {}
        if basis == "sigma_null" or not np.isfinite(prov.reported_db):
            kw["floor_db"] = prov.sigma_implied_db
        sweep = R.threshold_sweep(path, etas=ETAS, **kw)
        rows = [dict(eta=s["eta"], f=s["f"],
                     r_fine=s["r_masked"] / 10 ** (FINE_DB / 10))
                for s in sweep]
        for row in rows:
            row["penalty"] = ((1 + row["r_fine"]) / (1 - row["f"])
                              if row["f"] < 1 else float("inf"))
        feas = [r for r in rows if r["r_fine"] <= tol]
        rec = {"n_grid": len(rows), "feasible": len(feas)}
        if feas:
            pmin = min(r["penalty"] for r in feas)
            best = min((r for r in feas if r["penalty"] <= PLATEAU * pmin),
                       key=lambda r: r["eta"])
            at1 = rows[0] if rows and abs(rows[0]["eta"] - 1.0) < 1e-9 else None
            rec.update(
                eta=best["eta"], F_thresh=best["eta"] * prov.mu0,
                f=best["f"], r_fine=best["r_fine"],
                margin=tol / best["r_fine"], penalty=best["penalty"],
                penalty_pe=(at1["penalty"] if at1 else float("nan")),
                f_recent=recent_f(path, best["eta"], prov.mu0, era_from),
                # the labeled scenario: does this eta also satisfy fs8 if the
                # deployed delay filter were booked on both sides?
                fs8_with_delay=(best["r_fine"] / 10 ** (DEPLOYED_DELAY_DB / 10)
                                <= TOL_FS8[ch]),
            )
        out["bases"][basis] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", nargs="+", default=None,
                    metavar="CH=PATH", help="override e.g. 30=/path/598.npz")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)

    products = dict(DEFAULT_PRODUCTS)
    for spec in args.products or []:
        ch, _, path = spec.partition("=")
        products[int(ch)] = path

    results = [optimize(p, ch) for ch, p in sorted(products.items())]

    print(f"objective: min (1+r)/(1-f)  s.t.  r_fine <= r_tol(alpha_perp), "
          f"zeta = 1, fine stage {FINE_DB:.0f} dB\n")
    hdr = (f"{'ch':>3} {'basis':>10} {'eta*':>6} {'F>':>10} {'f*':>7} "
           f"{'r_fine':>9} {'margin':>7} {'cost':>6} {'@eta=1':>7} "
           f"{'recent f':>9} {'fs8+delay':>9}")
    print(hdr)
    rows_csv = []
    for res in results:
        for basis, rec in res["bases"].items():
            tag = f"ch{res['ch']}" + ("*" if res["tau_bound"] else "")
            if rec.get("feasible"):
                line = (f"{tag:>3} {basis:>10} {rec['eta']:6.2f} "
                        f"{rec['F_thresh']:10.6f} {rec['f']:7.1%} "
                        f"{rec['r_fine']:9.3g} {rec['margin']:6.1f}x "
                        f"{rec['penalty']:6.2f} {rec['penalty_pe']:7.2f} "
                        f"{rec['f_recent']:9.1%} "
                        f"{'yes' if rec['fs8_with_delay'] else 'no':>9}")
            else:
                line = (f"{tag:>3} {basis:>10} {'--':>6} {'--':>10} "
                        f"{'--':>7} {'--':>9} {'--':>7} {'--':>6} {'--':>7} "
                        f"{'--':>9} {'--':>9}   no feasible eta -> excise")
            print(line)
            rec2 = {k: rec.get(k) for k in
                    ("eta", "F_thresh", "f", "r_fine", "margin", "penalty",
                     "penalty_pe", "f_recent", "feasible", "fs8_with_delay")}
            rows_csv.append({"ch": res["ch"], "basis": basis,
                             "mu0": res["mu0"], "tau_bound": res["tau_bound"],
                             "tol_aperp": res["tol_aperp"],
                             "era_from": res["era_from"], **rec2})
        print()

    print("*  tau_c refused (capped at one sidereal day): r is a bound, so a "
          "feasible eta is truly feasible\n   but the optimum may be "
          "pessimistic; a measured tau_c only enlarges the feasible set.\n"
          "fs8+delay: whether eta* also meets the fs8 tolerance if the "
          f"deployed {DEPLOYED_DELAY_DB} dB delay filter were booked\n"
          "(a labeled scenario; the chapter's verdicts book zero).")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "optimal_thresholds.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

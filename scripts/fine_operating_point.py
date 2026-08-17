#!/usr/bin/env python3
"""The joint (rho, multiplier) operating point of the fine decision, derived.

The fine decision has two knobs: the rank rho (which order statistic of
the null bulk anchors the bar) and the multiplier m (how far above it the
bar sits). Neither is a free constant here. Both fall out of the same
objective that already selected the coarse eta*:

    minimize (1 + r) / (1 - f)   s.t.   r <= r_tol(alpha_perp), zeta = 1,

evaluated on the survey products' stored per-bin ``fstat_fine``, the data
the kernel repo notes were kept precisely so these quantiles are computable.

The anchor is measured rather than assumed. The products' ``fine_designated_bins``
is the nominal [0] scaffolding ("per-channel anchors were unmeasured before
the survey ran"); each channel's real pilot sits at its own offset, and with
the window parked at 0 the pilot lands in the jury, inflates the bar, and
the rule self-censors (ch32: 2% fired at the tightest setting while the
coarse rule masks 71%). Anchors are localized from the stored spectra --
per bin, the median T over coarse-detected frames minus the median over
coarse-quiet frames, argmax (static structure common to both cohorts
cancels); channels with no usable quiet cohort fall back to a plain median
argmax, marked in the output. Because the same estimator run on the two
archive halves shows the anchor MOVES (ch32: 141 -> 152, the 2020 station
handover), the designated window is the UNION of the per-era windows --
data rather than a choice.

Decision model (the chapter's usable-null-bulk definition, union window):

    fire  iff  max_{f in D} T[f] > m * T_(rho),
    D = union over era anchors a of [a-2, a+2] mod 256 (half-width = guard);
    B = independent (even) bins minus D minus census, all-finite rows only.

Residual booking mirrors ``residual.threshold_sweep`` exactly (kept
frames book their measured shelf where one exists and the stated floor
where none does), and the measured fine-stage credit (9.4--10.0 dB,
booked FINE_DB) divides the aggregate ratio, the same convention as
``optimal_thresholds.py``, so the two tables are directly comparable.
``r_nocredit`` reports the bare-floor booking alongside.

Selection order (every stage is data or a named ordering):

  1. Feasible: r_fine <= r_tol on the full archive AND in each archive
     half (the mask-side twin of the Fisher stability gate). A channel
     whose halves cannot both certify keeps its archive-wide point but is
     flagged era_stable = False (ch35: the 2022+ era is the other wall).
  2. Cost: minimize (1 + r)/(1 - f); plateau = within 2% of the minimum
     ("equal cost" band, inherited from the coarse optimizer).
  3. Tie-break BY DATA: among plateau points, minimize the WORST era-half
     cost (minimax). On clean channels the archive-wide ridge is flat in
     rho and this is where the choice is actually made; on contaminated
     channels (ch32) the in-sample objective already has an interior
     optimum: breakdown priced by the archive, no tail constant anywhere.
  4. Residual exact ties only, deterministic order: m nearest 1 (keep the
     bar inside the measured staircase; no tail extrapolation), then
     largest headroom |B| - rho, then smallest rho, then smallest m.

The rho grid is every integer in [|B|/2, |B|]; the per-rank ridge is
written out (out/fine_ridge.csv) so Rohling's 0.75N-0.9N radar optimum
serves as a cross-check rather than an input.

    python3 scripts/fine_operating_point.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from baonoise import residual as R
from baonoise.npzio import load_npz

spec = importlib.util.spec_from_file_location(
    "ot", str(ROOT / "scripts" / "optimal_thresholds.py"))
ot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ot)

TOL = ot.TOL_APERP
FINE_DB = ot.FINE_DB
PLATEAU = ot.PLATEAU
ETAS = ot.ETAS                       # reuse the coarse multiplier grid
PRODUCTS = dict(ot.DEFAULT_PRODUCTS)   # resolved by the products manifest

HALF = 2                             # designated half-width = guard (chapter)
LF = 256
MIN_KEPT = R.MIN_THRESHOLD_SWEEP_KEPT_FRAMES
MIN_COHORT = 100
INF = float("inf")


def window(anchor):
    return sorted({(anchor + k) % LF for k in range(-HALF, HALF + 1)})


def _anchor_of(ff, rej, quiet):
    """Argmax of the per-bin median excess of coarse-detected frames over
    coarse-quiet frames (static structure common to both cohorts cancels);
    plain median argmax when no usable quiet cohort exists."""
    if rej.sum() >= MIN_COHORT and quiet.sum() >= MIN_COHORT:
        exc = np.median(ff[rej], axis=0) - np.median(ff[quiet], axis=0)
        return int(np.argmax(exc)), "rej_minus_quiet"
    src = rej if rej.sum() >= MIN_COHORT else ~rej
    return int(np.argmax(np.median(ff[src], axis=0))), "median_only"


def measure_anchor(ff, rej, quiet, cal_lo):
    """(anchor, method, early, late). The era split is the CALENDAR
    midpoint of the archive; product frame order is only locally
    chronological (whole units land out of order), so an index split mixes
    eras, and a frame-median split puts most of calendar time in one half
    because cadence ramped up. Each half re-runs the SAME estimator, so a
    drift flag means the pilot moved rather than the estimator changing."""
    anchor, method = _anchor_of(ff, rej, quiet)
    early = late = None
    if (rej & cal_lo).sum() >= MIN_COHORT:
        early = _anchor_of(ff, rej & cal_lo, quiet & cal_lo)[0]
    if (rej & ~cal_lo).sum() >= MIN_COHORT:
        late = _anchor_of(ff, rej & ~cal_lo, quiet & ~cal_lo)[0]
    return anchor, method, early, late


def channel_tables(path):
    """Per-frame pieces shared by every (rho, m), with the measured
    (era-unioned) designated window."""
    d = load_npz(path)
    valid = d["valid"][:, 0].astype(bool)
    ff = d["fstat_fine"]
    fin = np.isfinite(ff).all(axis=1)
    on = valid & fin                          # clause (iv): live rows
    rej = d["reject_mask"][:, 0].astype(bool)
    shelf = d["snr_shelf_db"][:, 0]
    t0 = d["unit_time0_ctime"]
    t_frame = t0[d["frame_unit_index"]]
    yr = np.array([dt.datetime.utcfromtimestamp(x).year
                   for x in t0])[d["frame_unit_index"]]
    t_on = t_frame[on]
    cal_lo = t_frame <= 0.5 * (t_on.min() + t_on.max())

    anchor, method, early, late = measure_anchor(ff, on & rej, on & ~rej,
                                                 cal_lo)
    desg = set(window(anchor))
    for a in (early, late):
        if a is not None:
            desg |= set(window(a))
    census = set(int(c) for c in d["fine_census_excluded_bins"])
    desg_arr = np.array(sorted(desg))
    bulk = np.array([b for b in range(0, LF, 2) if b not in desg | census])
    enbw = float(d["bin_enbw_hz"]) / LF
    offset_hz = (anchor if anchor < LF // 2 else anchor - LF) * enbw

    stats = R.shelf_statistics(path)
    corr = R.correlation_time(path)
    n_slow = R.n_coh_from_correlation_time(corr.tau_for_budget)
    comps = ((stats.intraday_fraction, n_slow), (stats.fast_fraction, 1.0))

    def r_of(mean_lin):
        db = 10.0 * np.log10(max(mean_lin, 1e-30))
        return R.ResidualBudget(
            shelf_floor_db=db,
            delay_filter_db=R.DELAY_SUPPRESSION_DB[R.DEFAULT_DELAY_KEY],
            components=comps).ratio

    return dict(
        on=on, rej=rej, shelf=shelf, yr=yr, cal_lo=cal_lo,
        anchor=anchor, anchor_method=method, anchor_early=early,
        anchor_late=late, offset_hz=offset_hz, n_bulk=len(bulk),
        n_desg=len(desg_arr),
        maxD=ff[:, desg_arr].max(axis=1),
        srt=np.sort(ff[:, bulk], axis=1),
        prov=R.floor_provenance(path),
        tau_bound=corr.quality != "measured",
        r_of=r_of, n_on=int(on.sum()))


def _half_cost(fired_col, half, nocr, r_of, tol):
    """(cost, r) of one candidate restricted to one archive half; cost is
    inf when the half cannot certify (too few kept frames or r > tol)."""
    kept = ~fired_col & half
    if kept.sum() < MIN_KEPT or half.sum() < MIN_COHORT:
        return INF, float("nan")
    r = r_of(float(nocr[kept].mean())) / 10.0 ** (FINE_DB / 10.0)
    f = 1.0 - kept.sum() / half.sum()
    return ((1.0 + r) / (1.0 - f) if r <= tol else INF), r


def optimize_channel(ch, path):
    tab = channel_tables(path)
    on = tab["on"]
    maxD, srt = tab["maxD"][on], tab["srt"][on]
    rej_on, yr_on = tab["rej"][on], tab["yr"][on]
    lo = tab["cal_lo"][on]
    era_from = max(int(tab["yr"].max()) - 2, 2018)
    n_bulk = tab["n_bulk"]
    rhos = np.arange(n_bulk // 2, n_bulk + 1)
    tol = TOL[ch]
    out = {"ch": ch, "mu0": tab["prov"].mu0, "tau_bound": tab["tau_bound"],
           "tol_aperp": tol, "era_from": era_from, "bases": {}, "ridge": [],
           **{k: tab[k] for k in ("anchor", "anchor_method", "anchor_early",
                                  "anchor_late", "offset_hz", "n_bulk",
                                  "n_desg")}}
    if tab["n_on"] < 50:
        return out

    m_grid = np.asarray(ETAS)
    fired_cache = {int(rho): maxD[:, None] > m_grid[None, :]
                   * srt[:, rho - 1][:, None] for rho in rhos}
    shelf_on = tab["shelf"][on]
    meas = np.isfinite(shelf_on)
    lin_meas = np.where(meas, 10.0 ** (shelf_on / 10.0), 0.0)

    for basis in ("product", "sigma_null"):
        prov = tab["prov"]
        if basis == "sigma_null" or not np.isfinite(prov.reported_db):
            floor_db = prov.sigma_implied_db
        else:
            floor_db = prov.reported_db
        nocr = lin_meas + ~meas * 10.0 ** (floor_db / 10.0)

        rows = []
        for rho in rhos:
            fired = fired_cache[int(rho)]
            kept = ~fired
            n_kept = kept.sum(axis=0)
            with np.errstate(invalid="ignore"):
                mean_nocr = (nocr @ kept) / n_kept
            for j, m in enumerate(m_grid):
                if n_kept[j] < MIN_KEPT:
                    continue
                r_nc = tab["r_of"](float(mean_nocr[j]))
                r_c = r_nc / 10.0 ** (FINE_DB / 10.0)
                f = 1.0 - n_kept[j] / len(maxD)
                rows.append(dict(rho=int(rho), j=j, m=float(m), f=float(f),
                                 r_fine=r_c, r_nocredit=r_nc,
                                 penalty=(1.0 + r_c) / (1.0 - f)))
        feas = [r for r in rows if r["r_fine"] <= tol]
        rec = {"n_grid": len(rows), "feasible": len(feas)}
        if feas:
            pmin = min(r["penalty"] for r in feas)
            plateau = [r for r in feas if r["penalty"] <= PLATEAU * pmin]
            for c in plateau:                    # stage 3: data breaks ties
                col = fired_cache[c["rho"]][:, c["j"]]
                ce, c["r_early"] = _half_cost(col, lo, nocr, tab["r_of"], tol)
                cl, c["r_late"] = _half_cost(col, ~lo, nocr, tab["r_of"], tol)
                c["worst_era"] = max(ce, cl)
                c["era_stable"] = np.isfinite(c["worst_era"])
                c["best_era"] = min(ce, cl)
            best = min(plateau, key=lambda c: (
                c["worst_era"] if c["era_stable"] else INF,
                c["best_era"],                    # all-inf: least-bad half
                round(abs(c["m"] - 1.0), 9),      # stage 4: named orderings
                -(n_bulk - c["rho"]), c["rho"], c["m"]))
            fired_b = fired_cache[best["rho"]][:, best["j"]]
            recent = yr_on >= era_from
            best = {k: v for k, v in best.items() if k != "j"}
            rec.update(best, headroom=n_bulk - best["rho"],
                       quantile=best["rho"] / (n_bulk + 1),
                       multiplier_q16=int(round(best["m"] * 2 ** 16)),
                       margin=tol / best["r_fine"],
                       f_recent=(float(fired_b[recent].mean())
                                 if recent.sum() >= MIN_COHORT
                                 else float("nan")),
                       coarse_covered=(float(fired_b[rej_on].mean())
                                       if rej_on.sum() else float("nan")))
        out["bases"][basis] = rec

        if basis == "product":               # ridge: best m per rho
            for rho in rhos:
                sub = [r for r in rows
                       if r["rho"] == rho and r["r_fine"] <= tol]
                if sub:
                    b = min(sub, key=lambda r: r["penalty"])
                    out["ridge"].append(dict(ch=ch, rho=int(rho), m=b["m"],
                                             f=b["f"], r_fine=b["r_fine"],
                                             penalty=b["penalty"]))
    return out


CSV_REC = ("rho", "headroom", "quantile", "m", "multiplier_q16", "f",
           "r_fine", "r_nocredit", "margin", "penalty", "r_early", "r_late",
           "worst_era", "era_stable", "f_recent", "coarse_covered",
           "feasible")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(ROOT / "out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"objective: min (1+r)/(1-f)  s.t.  r_fine <= r_tol(alpha_perp), "
          f"zeta = 1;  rho grid = [|B|/2, |B|]\n"
          f"selection: era-stable feasibility -> cost plateau (2%) -> "
          f"minimax era-half cost -> m nearest 1 -> max headroom\n"
          f"designated window: union of per-era measured anchors +/- {HALF}; "
          f"credit = measured {FINE_DB:.0f} dB (aggregate)\n")
    hdr = (f"{'ch':>4} {'basis':>10} {'anch':>5} {'offHz':>7} {'|D|':>4} "
           f"{'|B|':>4} {'rho*':>5} {'m*':>6} {'f':>7} {'r_fine':>9} "
           f"{'cost':>6} {'worstE':>7} {'eraOK':>6} {'recent':>7} {'cov':>6}")
    print(hdr)
    rows_csv, ridge_csv = [], []
    for ch, path in sorted(PRODUCTS.items()):
        res = optimize_channel(ch, path)
        for basis, rec in res.get("bases", {}).items():
            tag = f"ch{res['ch']}" + ("*" if res["tau_bound"] else "")
            if rec.get("feasible"):
                we = rec["worst_era"]
                print(f"{tag:>4} {basis:>10} {res['anchor']:>5} "
                      f"{res['offset_hz']:7.0f} {res['n_desg']:>4} "
                      f"{res['n_bulk']:>4} {rec['rho']:>5} {rec['m']:6.2f} "
                      f"{rec['f']:7.1%} {rec['r_fine']:9.3g} "
                      f"{rec['penalty']:6.2f} "
                      f"{we if np.isfinite(we) else float('nan'):7.2f} "
                      f"{'yes' if rec['era_stable'] else 'NO':>6} "
                      f"{rec['f_recent']:7.1%} {rec['coarse_covered']:6.1%}")
            else:
                print(f"{tag:>4} {basis:>10} {res['anchor']:>5} "
                      f"{res['offset_hz']:7.0f} {res['n_desg']:>4} "
                      f"{res['n_bulk']:>4} {'--':>5} {'--':>6} {'--':>7} "
                      f"{'--':>9} {'--':>6} {'--':>7} {'--':>6} {'--':>7} "
                      f"{'--':>6}   no feasible (rho, m) -> excise")
            rows_csv.append({"ch": res["ch"], "basis": basis,
                             "mu0": res["mu0"], "tau_bound": res["tau_bound"],
                             "tol_aperp": res["tol_aperp"],
                             "era_from": res["era_from"],
                             **{k: res.get(k) for k in
                                ("anchor", "offset_hz", "anchor_method",
                                 "anchor_early", "anchor_late", "n_desg",
                                 "n_bulk")},
                             **{k: rec.get(k) for k in CSV_REC}})
        if res.get("anchor_early") is not None and res.get("anchor_late") is not None:
            dr = min((res["anchor_early"] - res["anchor_late"]) % LF,
                     (res["anchor_late"] - res["anchor_early"]) % LF)
            if dr > HALF:
                print(f"     !! anchor era-drift: early {res['anchor_early']}"
                      f" vs late {res['anchor_late']} -> union window")
        if res.get("anchor_method") == "median_only":
            print("     !! no quiet cohort: anchor from plain median "
                  "(static structure does not cancel)")
        ridge_csv.extend(res.get("ridge", []))
        print()

    print("*  tau_c refused (capped at one sidereal day): r is a bound.\n"
          "eraOK = the chosen point certifies r <= tol with >= 30 kept "
          "frames in BOTH archive halves;\n  NO marks an archive-only "
          "operating point (kept for the record rather than for deployment).\n"
          "cov = fraction of coarse-masked frames the fine rule also fires "
          "on (the audit quantity).")

    for name, rows in (("fine_operating_points.csv", rows_csv),
                       ("fine_ridge.csv", ridge_csv)):
        if not rows:
            continue
        with open(args.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-channel working threshold, priced against the BAO noise tolerance.

``mu`` says where the null is; it does not say how far above the null the
threshold belongs.  That is a science question, and it is answered exactly
the way ``bao-noise-tolerance/scripts/optimal_thresholds.py`` answers it:
minimise the survey-time cost

    T / T_clean = (1 + r) / (1 - f)

over the threshold family ``F > eta * mu``, subject to the bias tolerance
``r <= r_tol``, with the measured fine-stage sensitivity credit applied to
the bound.  ``r`` grows with ``eta`` and ``f`` shrinks with it, so the
minimum is interior wherever the residual is large enough to matter -- and
where it is not, no masking pays and the tolerance alone sets the ceiling.

Two things make the answer per-channel rather than global:

* each channel's own occupancy and residual, which differ by orders of
  magnitude across the band;
* each channel's own tolerance, from the redshift bins its 6 MHz allocation
  overlaps (``tolerances.py``).  Both tiers are carried: the acoustic
  dilation tolerance the released selector constrains on, and the stricter
  growth-rate tolerance the dissertation's verdicts quote.

Everything is evaluated on the latest era only, through a single-era product
view (``ppcal.era_view``).

**Adopted floor basis: the quiet era, which is the bounded choice.**  Three
bases are available and they do not agree:

* the product's own kept-frame floor (the ``mu0`` sliver, 1 < F <= mu0).
  ``baonoise.residual.FloorProvenance`` shows this one is fixed by the weight
  bank rather than by the sky -- it lands on 10log10(mu0 - 1) plus a constant
  offset -- and it is the *lowest* of the three, so it is the least
  conservative.
* the sigma-implied level, the excess a threshold sitting at the null centre
  can actually resolve.  Defensible everywhere, including the channels where
  mu0 < 1 leaves the sliver empty for any dataset.
* the p90 shelf of the channel's own quietest era: a measurement taken on
  frames the transmitter was demonstrably off for, and the *highest* of the
  three, so the most conservative.

This run takes the quiet era wherever the channel has one and falls back to
the sigma-implied substitute where it does not, which on this archive is
exactly the five always-on carriers (ch17, 22, 24, 30, 31 -- ch35 has a
pre-sign-on quiet era).  Both choices push the residual up rather than down,
so the floor is bounded in the same one-sided sense as the coherence cap
below, and ``note`` records per channel which basis a row used.

**Adopted basis: the bound, not the measurement.**  The residual depends on
how long contamination stays coherent, and only three channels carry a
measured coherence time.  Everywhere else the chain is evaluated at the
sidereal-day cap, which is the physical ceiling: anything longer-lived has
already been removed as m = 0.  Since tau_cap >= tau_true, and the coherent
amplification n_coh grows with tau, the residual reported at the cap is an
**upper bound** on the true residual, not an estimate of it.

Two consequences follow, and both matter for how the columns are read:

* ``r > r_tol`` at the cap does **not** demonstrate that a channel violates
  its tolerance.  It demonstrates that the tolerance is *not certified* on
  present evidence.  A measured tau can only lower the residual, so it can
  only enlarge the feasible set.
* Nothing in the keep/excise disposition rests on this.  Every excised
  channel fails on carrier dominance -- the densest population in its latest
  era is the carrier, so no null exists to threshold against -- which is a
  tau-free measurement.  The bound governs what the kept channels cost, not
  which channels are kept.

The thermal end (one frame, n_coh = 1) is carried alongside as the
optimistic limit, so the width of the bracket is visible; it is not the
operating basis.  ``residual_basis`` names, per channel, which of the two a
row's cap-end numbers actually rest on.

**What an exact residual still needs**, in descending order of how much it
would move the answer:

1. a measured coherence time per channel.  Twenty of twenty-three sit at the
   sidereal-day cap, and the bracket between that and the thermal limit spans
   four to six orders of magnitude -- far more than any other term here.  The
   acquisition is specified but not scheduled.
2. a directly measured sensitivity floor, from a control-frequency or null
   trawl, rather than one bounded from whichever era happened to be quietest.
   This matters most on the five always-on carriers, which have no quiet era
   at all and therefore no floor measurement of any kind.
3. the pilot-to-shelf transfer, which is assumed exact here through the ATSC
   constants and is the one step in the chain with no error bar attached.

Until those land, every residual this script reports is an upper limit, and
every verdict that depends on one is "not certified" rather than "failed".

    python3 scripts/calibrated_thresholds.py [--products DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# This is the tolerance-layer half of the calibration suite and belongs in
# bao-noise-tolerance; the calibration package it reads products through
# lives in pilot-proxy, which owns the products. Point PP_ANALYSIS at that
# repo's analysis/ directory (and PP_SRC at its src/) to relocate either.
sys.path.insert(0, os.environ.get(
    "PP_SRC", os.path.expanduser("~/rail/pilot-proxy/src")))
sys.path.insert(0, os.environ.get(
    "PP_ANALYSIS", os.path.expanduser("~/rail/pilot-proxy/analysis")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from ppcal import era_view as _probe  # noqa: F401
except ImportError as exc:                # pragma: no cover - setup guidance
    raise SystemExit(
        "this script reads per-pilot products through the calibration package "
        "`ppcal`, which lives in pilot-proxy (the repository that owns the "
        "products). Point PP_ANALYSIS at that repo's analysis/ directory and "
        "PP_SRC at its src/, e.g.\n"
        "  export PP_ANALYSIS=~/rail/pilot-proxy/analysis\n"
        "  export PP_SRC=~/rail/pilot-proxy/src\n"
        f"underlying import error: {exc}") from exc

from baonoise import residual as res  # noqa: E402

from ppcal import era_view as EV, eras as E  # noqa: E402
from ppcal.calib import calibrate  # noqa: E402
from ppcal.products import load_all  # noqa: E402
from channel_tolerances import TOL_APERP_PUBLISHED, channel_tolerances  # noqa: E402

ETA_GRID = np.concatenate([[1.0], np.geomspace(1.01, 60.0, 90)])
DAY_CAP = 86164.0
FRAME_S = 16384 * 2.56e-6
FINE_DB = 10.0        # measured fine-stage credit, 9.4-10.0 dB (MC verified)
PLATEAU = 1.02        # "within 2% of optimal" tie-break, smallest eta wins


def r_tolerances():
    """{ch: (r_tol_dilation, r_tol_growth)}.

    The released selector's hard-coded dilation table is preferred wherever
    it exists (ch27-36) so this run stands on the published numbers; the
    lower band is extended from the completed forecast ledger.
    """
    derived = channel_tolerances()
    out = {}
    for ch, rec in derived.items():
        dil = TOL_APERP_PUBLISHED.get(ch, rec["aperp"])
        out[ch] = (dil, rec["fs8"], rec["z_low"], rec["z_high"],
                   ch in TOL_APERP_PUBLISHED)
    return out


def sweep(c, segs, fmask, mu, tau):
    """Threshold sweep over the latest era on the calibrated ``mu`` scale."""
    scale = mu / c.mu0
    floor_db, floor_era, n_floor = EV.quiet_era_floor_db(c, segs)
    if np.isfinite(floor_db):
        note = "floor from era %s (%d frames, p90 %.2f dB)" % (
            floor_era, n_floor, floor_db)
    else:
        try:
            fp = res.floor_provenance(c.path)
            floor_db = fp.sigma_implied_db
            note = ("floor substituted: sigma-implied %.2f dB (no quiet era)"
                    % floor_db)
        except Exception as exc:                   # noqa: BLE001
            return [], "no floor available: %s" % exc
    try:
        with EV.era_product_view(c, fmask) as view:
            rows = res.threshold_sweep(view, etas=ETA_GRID * scale,
                                       tau_intraday=tau, floor_db=floor_db)
    except Exception as exc:                       # noqa: BLE001
        return [], "sweep failed: %s" % exc
    credit = 10.0 ** (FINE_DB / 10.0)
    for r in rows:
        r["eta_mu"] = r["eta"] / scale
        r["r_fine"] = r["r_masked"] / credit
        r["penalty"] = (1.0 + r["r_fine"]) / max(1.0 - r["f"], 1e-12)
    return rows, note


def select(rows, r_tol=None):
    """Cheapest threshold, smallest eta on a 2% cost plateau.

    With ``r_tol`` the choice is restricted to thresholds that meet the
    tolerance; without it the cost optimum is returned unconstrained, which
    is the per-channel operating point the survey-time trade alone picks and
    is defined even where no threshold meets the bound.
    """
    ok = rows if r_tol is None else [r for r in rows if r["r_fine"] <= r_tol]
    if not ok:
        return None
    best = min(r["penalty"] for r in ok)
    near = [r for r in ok if r["penalty"] <= best * PLATEAU]
    return min(near, key=lambda r: r["eta_mu"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=os.environ.get("PP_PER_PILOT"))
    ap.add_argument("--out", default=os.environ.get(
        "PP_CALIB_OUT", os.path.join(ROOT, "out")))
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    if not args.products:
        raise SystemExit("pass --products, or set PP_PER_PILOT, to the "
                         "directory of per-pilot survey products")

    tol = r_tolerances()
    only = ({int(x) for x in args.only.split(",")} if args.only else None)
    rows = []
    print("%3s %-9s %10s %10s | %8s %8s %9s %9s | %8s %8s | %s"
          % ("ch", "z range", "r_tol_dil", "r_tol_gro", "eta_cost", "mask",
             "penalty", "r/r_dil", "eta_feas", "mask", "tau"))
    for c in sorted(load_all(args.products), key=lambda c: c.ch):
        if only and c.ch not in only:
            continue
        segs = E.segment(c)
        fmask = E.final_era_frame_mask(c, segs)
        cal = calibrate(c, fmask, segs[-1].label, 0)
        dil, gro, z_lo, z_hi, published = tol.get(
            c.ch, (float("nan"),) * 2 + (float("nan"),) * 2 + (False,))

        rec = dict(ch=c.ch, era=segs[-1].label, mu=cal.mu, z_low=z_lo,
                   z_high=z_hi, r_tol_dilation=dil, r_tol_growth=gro,
                   dilation_tol_published=published, note="")
        try:
            with EV.era_product_view(c, fmask) as view:
                corr = res.correlation_time(view)
            rec["tau_seconds"] = float(corr.tau_for_budget)
            rec["tau_measured"] = bool(corr.is_measured)
        except Exception:                          # noqa: BLE001
            rec["tau_seconds"], rec["tau_measured"] = DAY_CAP, False
        # The adopted basis is the bound wherever tau was refused: the
        # cap-end residual is then an upper limit on the true one, and a
        # later measurement can only move it down.
        rec["residual_basis"] = ("measured tau" if rec["tau_measured"]
                                 else "upper bound (tau at sidereal cap)")

        for tag, tau in (("cap", None), ("thermal", FRAME_S)):
            s, note = sweep(c, segs, fmask, cal.mu, tau)
            if note:
                rec["note"] = note
            if not s:
                continue
            rec["r_unmasked_%s" % tag] = s[0]["r_unmasked"]
            rec["r_floor_%s" % tag] = min(r["r_fine"] for r in s)
            cost = select(s)                       # unconstrained cost optimum
            if cost is not None:
                rec["eta_cost_%s" % tag] = cost["eta_mu"]
                rec["mask_cost_%s" % tag] = cost["f"]
                rec["penalty_cost_%s" % tag] = cost["penalty"]
                rec["r_cost_%s" % tag] = cost["r_fine"]
            for tier, rt in (("dilation", dil), ("growth", gro)):
                pick = select(s, rt)
                pre = "%s_%s" % (tier, tag)
                if pick is None:
                    rec["eta_" + pre] = ""
                    rec["mask_" + pre] = ""
                    rec["penalty_" + pre] = ""
                    rec["r_" + pre] = ""
                else:
                    rec["eta_" + pre] = pick["eta_mu"]
                    rec["mask_" + pre] = pick["f"]
                    rec["penalty_" + pre] = pick["penalty"]
                    rec["r_" + pre] = pick["r_fine"]

        def num(key, fmt):
            v = rec.get(key, "")
            return (fmt % v) if isinstance(v, float) else "-"

        rr = rec.get("r_cost_cap")
        print("%3d %4.2f-%4.2f %10.3g %10.3g | %8s %8s %9s %9s | %8s %8s | "
              "%6.0f s%s  %s"
              % (c.ch, z_lo, z_hi, dil, gro,
                 num("eta_cost_cap", "%.3f"), num("mask_cost_cap", "%.4f"),
                 num("penalty_cost_cap", "%.4g"),
                 ("%.3g" % (rr / dil)) if isinstance(rr, float) else "-",
                 num("eta_dilation_cap", "%.3f"),
                 num("mask_dilation_cap", "%.4f"),
                 rec["tau_seconds"], "*" if rec["tau_measured"] else " ",
                 rec["note"][:30]))
        rows.append(rec)

    cols = ["ch", "era", "mu", "z_low", "z_high", "r_tol_dilation",
            "r_tol_growth", "dilation_tol_published", "tau_seconds",
            "tau_measured", "residual_basis",
            "eta_cost_cap", "mask_cost_cap", "penalty_cost_cap", "r_cost_cap",
            "eta_cost_thermal", "mask_cost_thermal", "penalty_cost_thermal",
            "r_cost_thermal", "r_unmasked_cap", "r_floor_cap",
            "eta_dilation_cap", "mask_dilation_cap", "penalty_dilation_cap",
            "r_dilation_cap", "eta_growth_cap", "mask_growth_cap",
            "penalty_growth_cap", "r_growth_cap", "r_unmasked_thermal",
            "r_floor_thermal", "eta_dilation_thermal", "mask_dilation_thermal",
            "penalty_dilation_thermal", "r_dilation_thermal",
            "eta_growth_thermal", "mask_growth_thermal",
            "penalty_growth_thermal", "r_growth_thermal", "note"]
    tdir = os.path.join(args.out, "tables")
    os.makedirs(tdir, exist_ok=True)
    # A --only run is a spot check, not the table every downstream consumer
    # reads; writing it to eta_bao.csv would silently drop the other channels
    # and send them back to the global fallback threshold.
    name = "eta_bao.csv" if only is None else "eta_bao_subset.csv"
    with open(os.path.join(tdir, name), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("\nwrote", os.path.join(tdir, "eta_bao.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-channel Fisher-bias tolerances, by target parameter.

``scripts/optimal_thresholds.py`` in bao-noise-tolerance selects each
channel's threshold subject to the *acoustic-dilation* tolerance
(``alpha_perp``, zeta = 1), not the growth-rate one, and applies the measured
fine-stage sensitivity credit to the bound.  Its hard-coded table covers only
ch27-36; this script derives the same quantity for all 23 channels from the
completed forecast run, so the lower band is priced on the same footing.

    python3 scripts/channel_tolerances.py
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "out", "forecast_completion_all_dtv_bins.json")
MAPPING = os.path.join(ROOT, "out", "forecast_completion_channel_mapping.csv")
ESTIMATOR = "perbin_noise_normalized"
TARGETS = ("aperp", "apar", "fs8")

# the hard-coded table optimal_thresholds.py stands on, for cross-checking
TOL_APERP_PUBLISHED = {27: 0.014, 28: 0.014, 29: 0.014, 30: 0.0144,
                       31: 0.0156, 32: 0.0156, 33: 0.0156, 34: 0.0156,
                       35: 0.0352, 36: 0.0352}


def bin_tolerances(ledger_path=LEDGER, estimator=ESTIMATOR):
    """{bin_index: {target: conservative accepted r_tolerance}}."""
    with open(ledger_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    out = {}
    for b in doc["ledgers"][estimator]["bins"]:
        idx = int(b["bin_index"])
        rec = {"z_low": b["z_low"], "z_high": b["z_high"],
               "z_center": b["z_center"]}
        for target in TARGETS:
            vals = []
            for point in b["points"]:
                entry = point.get("parameters", {}).get(target)
                if not entry:
                    continue
                for label, det in entry.items():
                    if label == "accepted":
                        continue
                    if not isinstance(det, dict):
                        continue
                    if det.get("failure_reason") is not None:
                        continue
                    r = det.get("r_tolerance")
                    if r is None:
                        continue
                    vals.append(float(r))
            rec[target] = min(vals) if vals else None
        out[idx] = rec
    return out


def channel_bins(mapping_path=MAPPING, family="noise_shaped"):
    """{channel: [bin indices it overlaps]}, from the released mapping."""
    out = {}
    with open(mapping_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["family"] != family:
                continue
            ch = int(row["channel"])
            bins = [int(x) for x in row["overlap_bin_indices"].split(";") if x]
            out[ch] = bins
    return out


def channel_tolerances():
    """{channel: {target: conservative tolerance over its overlapping bins}}."""
    bt = bin_tolerances()
    cb = channel_bins()
    out = {}
    for ch, bins in sorted(cb.items()):
        rec = {"bins": bins}
        for target in TARGETS:
            vals = [bt[b][target] for b in bins
                    if b in bt and bt[b].get(target) is not None]
            rec[target] = min(vals) if vals else None
        rec["z_low"] = min(bt[b]["z_low"] for b in bins if b in bt)
        rec["z_high"] = max(bt[b]["z_high"] for b in bins if b in bt)
        out[ch] = rec
    return out


def main():
    tol = channel_tolerances()
    print("%3s %-10s %8s %11s %11s %11s %12s"
          % ("ch", "z range", "bins", "aperp", "apar", "fs8", "published"))
    for ch, rec in sorted(tol.items()):
        pub = TOL_APERP_PUBLISHED.get(ch)
        print("%3d %4.2f-%4.2f %8s %11.5g %11.5g %11.5g %12s"
              % (ch, rec["z_low"], rec["z_high"],
                 ";".join(str(b) for b in rec["bins"]),
                 rec["aperp"] or float("nan"), rec["apar"] or float("nan"),
                 rec["fs8"] or float("nan"),
                 ("%.4g" % pub) if pub else "-"))
    out = os.environ.get("PP_CALIB_OUT", os.path.join(ROOT, "out"))
    out = os.path.join(out, "tables", "channel_tolerances.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["ch", "z_low", "z_high", "bins", "r_tol_aperp",
                    "r_tol_apar", "r_tol_fs8", "r_tol_aperp_published"])
        for ch, rec in sorted(tol.items()):
            w.writerow([ch, rec["z_low"], rec["z_high"],
                        ";".join(str(b) for b in rec["bins"]),
                        rec["aperp"], rec["apar"], rec["fs8"],
                        TOL_APERP_PUBLISHED.get(ch, "")])
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

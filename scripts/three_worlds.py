#!/usr/bin/env python3
"""The delay filter booked on both sides: three worlds, one verdict table.

Each world is one published delay-cut choice carried self-consistently:
the Fisher bank loses the cut modes (kfg_fac = tau_cut * bandwidth) and the
residual chain claims the matching DTV suppression derived from the same
k_par <-> delay mapping. Nothing is invented: the cut values are CHIME's
(200 ns deployed; 55/110 ns are the first- and second-peak-preserving
design points any BAO analysis faces), and the suppression numbers are the
chapter's DELAY_SUPPRESSION_DB.

Verdicts are quoted at the minimum-residual operating point (eta = 1,
product-basis floors, fine-stage credit) for the threshold-feasible
channels and the tau_c-hostage ch29. Tolerances are the per-parameter
minima over the entries that survive the +/-10% stability gate.

    python3 scripts/three_worlds.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import residual as R                        # noqa: E402
from baonoise import survey                               # noqa: E402
from baonoise.fisherbank import FisherBank                # noqa: E402

spec = importlib.util.spec_from_file_location(
    "bt", str(ROOT / "scripts" / "bias_tolerance.py"))
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

WORLDS = [
    ("none",   "no filter (Fig. 31 baseline)", "fisher_bank_chime2022_pres_dense.npz", 0.0),
    ("peak1",  "55 ns, preserves 1st peak",    "fisher_bank_chime2022_pres_kfg22.npz", 3.6),
    ("peak2",  "110 ns, preserves 2nd peak",   "fisher_bank_chime2022_pres_kfg44.npz", 8.2),
    ("deployed", "200 ns, CHIME's cut",        "fisher_bank_chime2022_pres_kfg80.npz", 11.4),
]
FINE_DB = 10.0
PARAMS = ("aperp", "apar", "fs8")
from baonoise import products as P                        # noqa: E402
Z_BIN = {32: 1.4, 33: 1.4, 35: 1.3, 29: 1.5}
PATHS = P.paths(channels=sorted(Z_BIN))
YEARS = (1.0, 2.0, 3.0, 5.0, 8.0)


def stable_minima(bank_path):
    """Per-(bin, parameter) minimum tolerance over stability-gated entries."""
    bank = FisherBank(bank_path)
    names = list(bank.paramnames)
    zs = bank.zs
    out = {}
    for zlo in (1.3, 1.4, 1.5):
        ib = [i for i in range(len(zs) - 1) if abs(zs[i] - zlo) < 1e-9][0]
        vals = {p: [] for p in PARAMS}
        refused = 0
        for yr in YEARS:
            t = yr * survey.ONSKY_YEAR_HOURS
            dth, sig = bt.bias_per_unit_r(bank.F(ib, t), names)
            for p in PARAMS:
                drift, nsign = bt.stability(bank, ib, t, names, p)
                if drift <= 1.2 and nsign == 1:
                    vals[p].append(sig[p] / abs(dth[p]))
                else:
                    refused += 1
        out[zlo] = ({p: (min(v) if v else float("nan"))
                     for p, v in vals.items()}, refused)
    return out


def channel_r_eta1(p):
    """Fine-stage residual at eta = 1, product-basis floor (as in the
    optimizer): the minimum-residual point of the threshold family."""
    prov = R.floor_provenance(p)
    kw = {} if np.isfinite(prov.reported_db) else \
        {"floor_db": prov.sigma_implied_db}
    sweep = R.threshold_sweep(p, etas=np.array([1.0]), **kw)
    corr = R.correlation_time(p)
    return (sweep[0]["r_masked"] / 10 ** (FINE_DB / 10),
            corr.quality != "measured")


def main():
    rows = []
    r1 = {ch: channel_r_eta1(PATHS[ch]) for ch in sorted(Z_BIN)}
    print(f"channels at eta=1, fine stage: " +
          "  ".join(f"ch{c}={v[0]:.4g}{'*' if v[1] else ''}"
                    for c, v in r1.items()) + "   (* = tau_c capped)\n")

    for key, label, bank, sup_db in WORLDS:
        tol = stable_minima(ROOT / "data" / bank)
        sup = 10 ** (sup_db / 10)
        print(f"--- world '{key}': {label}  (suppression {sup_db} dB, "
              f"refusals per bin: " +
              ", ".join(f"{z}:{tol[z][1]}" for z in (1.3, 1.4, 1.5)) + ")")
        for ch, zlo in Z_BIN.items():
            r = r1[ch][0] / sup
            t = tol[zlo][0]
            verdicts = {}
            cells = []
            for p in PARAMS:
                ok = r <= t[p]
                verdicts[p] = ok
                cells.append(f"{p} {'PASS x%.1f' % (t[p]/r) if ok else 'x%.2g over' % (r/t[p])}")
            print(f"    ch{ch}{'*' if r1[ch][1] else ' '}  r={r:9.4g}   " +
                  "   ".join(f"{c:>16}" for c in cells))
            rows.append(dict(world=key, suppression_db=sup_db, ch=ch,
                             tau_capped=r1[ch][1], r_fine=r,
                             **{f"tol_{p}": t[p] for p in PARAMS},
                             **{f"pass_{p}": verdicts[p] for p in PARAMS}))
        print()

    out = ROOT / "out" / "three_worlds.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

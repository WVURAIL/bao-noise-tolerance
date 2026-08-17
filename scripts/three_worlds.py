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

The four bias-response banks are not shipped. Build matched strict-v2,
unit-response banks first:

    python scripts/build_bank.py --config chime2022 --cosmology planck2018 \\
        --p-res 1.0 --dense-knee \\
        --out data/fisher_bank_chime2022_pres_dense.npz
    python scripts/build_bank.py --config chime2022 --cosmology planck2018 \\
        --p-res 1.0 --kfg-fac 22 --dense-knee \\
        --out data/fisher_bank_chime2022_pres_kfg22_dense.npz
    python scripts/build_bank.py --config chime2022 --cosmology planck2018 \\
        --p-res 1.0 --kfg-fac 44 --dense-knee \\
        --out data/fisher_bank_chime2022_pres_kfg44_dense.npz
    python scripts/build_bank.py --config chime2022 --cosmology planck2018 \\
        --p-res 1.0 --kfg-fac 80 --dense-knee \\
        --out data/fisher_bank_chime2022_pres_kfg80_dense.npz
    python scripts/three_worlds.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from baonoise import residual as R
from baonoise import survey

spec = importlib.util.spec_from_file_location(
    "bt", str(ROOT / "scripts" / "bias_tolerance.py"))
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

WORLDS = [
    ("none",   "no filter (Fig. 31 baseline)", "fisher_bank_chime2022_pres_dense.npz", 0.0),
    ("peak1",  "55 ns, preserves 1st peak",    "fisher_bank_chime2022_pres_kfg22_dense.npz", 3.6),
    ("peak2",  "110 ns, preserves 2nd peak",   "fisher_bank_chime2022_pres_kfg44_dense.npz", 8.2),
    ("deployed", "200 ns, CHIME's cut",        "fisher_bank_chime2022_pres_kfg80_dense.npz", 11.4),
]
FINE_DB = 10.0
PARAMS = ("aperp", "apar", "fs8")
from baonoise import products as P
Z_BIN = {32: 1.4, 33: 1.4, 35: 1.3, 29: 1.5}
YEARS = (1.0, 2.0, 3.0, 5.0, 8.0)


def stable_minima(bank):
    """Per-(bin, parameter) minimum tolerance over stability-gated entries."""
    names = list(bank.paramnames)
    zs = bank.zs
    out = {}
    for zlo in (1.3, 1.4, 1.5):
        ib = [i for i in range(len(zs) - 1) if abs(zs[i] - zlo) < 1e-9][0]
        vals = {p: [] for p in PARAMS}
        refused = 0
        for yr in YEARS:
            t = yr * survey.OVERVIEW_ONSKY_YEAR_HOURS
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bank-dir", type=Path, default=ROOT / "data",
        help="directory containing all four locally built strict-v2 "
             "unit-response banks listed in this command's description")
    args = ap.parse_args(argv)

    banks = {}
    kfg_by_world = {"none": None, "peak1": 22.0, "peak2": 44.0,
                    "deployed": 80.0}
    for key, _, filename, _ in WORLDS:
        expected_kfg = kfg_by_world[key]
        kfg = "" if expected_kfg is None else f" --kfg-fac {expected_kfg:g}"
        command = (
            "python scripts/build_bank.py --config chime2022 "
            "--cosmology planck2018 --p-res 1.0"
            f"{kfg} --dense-knee --out data/{filename}"
        )
        try:
            banks[key] = bt.load_bias_bank(
                args.bank_dir / filename, build_command=command,
                expected_kfg_fac=expected_kfg)
        except ValueError as exc:
            ap.error(str(exc))

    rows = []
    paths = P.paths(channels=sorted(Z_BIN))
    missing_products = sorted(set(Z_BIN) - set(paths))
    if missing_products:
        ap.error(
            "survey products are missing for channels "
            + ", ".join(map(str, missing_products))
            + "; configure products.json or its local override")
    r1 = {ch: channel_r_eta1(paths[ch]) for ch in sorted(Z_BIN)}
    print(f"channels at eta=1, fine stage: " +
          "  ".join(f"ch{c}={v[0]:.4g}{'*' if v[1] else ''}"
                    for c, v in r1.items()) + "   (* = tau_c capped)\n")

    for key, label, _, sup_db in WORLDS:
        tol = stable_minima(banks[key])
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

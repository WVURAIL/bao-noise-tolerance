#!/usr/bin/env python3
"""Densify an existing Fisher bank on additional integration-time points
(default: half-step offsets through the CV-saturation knee, 10^3.5..10^5.83)."""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import fisherbank as fb  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ROOT / "data" / "fisher_bank_chime.npz"))
    ap.add_argument("--nproc", type=int, default=2)
    args = ap.parse_args()

    dat = np.load(args.bank, allow_pickle=False)
    F_old, t_old = dat["F"], dat["t_grid"]
    zs, zc = dat["zs"], dat["zc"]
    paramnames = [str(p) for p in dat["paramnames"]]
    meta = json.loads(str(dat["meta"]))

    t_new = 10.0 ** (3.5 + np.arange(8) / 3.0)
    t_new = np.array([t for t in t_new
                      if np.min(np.abs(np.log10(t_old) - np.log10(t))) > 0.05])
    print(f"[augment] adding {len(t_new)} t-points x {len(zc)} bins")

    fb._init_context(None, ROOT / "data" / "cache_pk.dat", "bull2015",
                     meta["epsilon_fg"], meta["k_nl0"])
    tasks = [(i, t) for i in range(len(zc)) for t in t_new]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.nproc) as pool:
        results = list(pool.imap_unordered(fb._one_fisher, tasks))
    print(f"[augment] {len(tasks)} matrices in {(time.time()-t0)/60:.1f} min")

    t_all = np.concatenate([t_old, t_new])
    order = np.argsort(t_all)
    npar = F_old.shape[-1]
    F_all = np.zeros((len(zc), len(t_all), npar, npar))
    F_all[:, :len(t_old)] = F_old
    tidx = {t: len(t_old) + j for j, t in enumerate(t_new)}
    for ibin, t, Fmat, names in results:
        assert names == paramnames
        F_all[ibin, tidx[t]] = Fmat
    F_all = F_all[:, order]
    t_all = t_all[order]

    meta["augmented"] = True
    np.savez_compressed(args.bank, F=F_all, t_grid=t_all, zs=zs, zc=zc,
                        paramnames=np.array(paramnames), meta=json.dumps(meta))
    print(f"[augment] saved {args.bank} with {len(t_all)} t-points")

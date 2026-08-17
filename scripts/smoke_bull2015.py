#!/usr/bin/env python3
"""Smoke test: load the Bull-2015 cache and run one Fisher bin."""
import time

from baonoise.compat import import_radiofisher
from baonoise import cosmologies, pkcache, survey
from baonoise.resources import filesystem_data_file

rf, rf_dir = import_radiofisher()
print("radiofisher imported from", rf_dir)

cache = filesystem_data_file("cache_pk.dat")
t0 = time.time()
base = cosmologies.with_astrophysical_profile(
    rf.experiments.cosmo, "bull2015", rf=rf)
cosmo = pkcache.load_fiducial_cosmology(rf, cache, cosmo=base)
print(f"P(k) cache ready in {time.time()-t0:.1f}s; "
      f"k range {cosmo['k_in_min']:.2e}..{cosmo['k_in_max']:.2f} Mpc^-1")

cosmo_fns = rf.background_evolution_splines(cosmo)

expt = survey.chime_experiment(rf, rf_dir, ttot_hours=1e4)
zs, zc = survey.chime_zbins(rf, expt)
print("zbins:", len(zc), "edges:", np.round(zs, 3) if (np := __import__('numpy')) else zs)

i = len(zc) // 2
t0 = time.time()
F, paramnames = rf.fisher(zs[i], zs[i + 1], cosmo, expt, cosmo_fns,
                          return_pk=False, kbins=None)
dt = time.time() - t0
print(f"fisher() bin z=[{zs[i]:.2f},{zs[i+1]:.2f}] took {dt:.1f}s")
print("paramnames:", paramnames)
print("F shape:", F.shape, "diag:", ["%.3e" % d for d in F.diagonal()])

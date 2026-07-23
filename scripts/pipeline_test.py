#!/usr/bin/env python3
"""Fast end-to-end plumbing test with a synthetic mini-bank: one real Fisher
matrix tiled across (bins, times) with the exact t^2 noise-dominated scaling.
Validates scenarios -> bank interpolation -> combined matrix -> sigma(A)."""
import sys, time, json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from baonoise.compat import import_radiofisher
from baonoise import pkcache, survey, scenarios, forecast
from baonoise.fisherbank import FisherBank

rf, rf_dir = import_radiofisher()
cosmo = pkcache.load_fiducial_cosmology(rf, root / "data" / "cache_pk.dat")
cosmo_fns = rf.background_evolution_splines(cosmo)
expt = survey.chime_experiment(rf, rf_dir, ttot_hours=1e4)
zs, zc = survey.chime_zbins(rf, expt)

i = 8
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    F0, paramnames = rf.fisher(zs[i], zs[i+1], cosmo, expt, cosmo_fns)

# synthetic bank: same matrix in every bin, F(t) = F0 * (t/1e4)^2
t_grid = np.logspace(0, 6, 13)
nb = len(zc)
F = np.zeros((nb, len(t_grid), F0.shape[0], F0.shape[0]))
for b in range(nb):
    for j, t in enumerate(t_grid):
        F[b, j] = F0 * (t / 1e4) ** 2
bank_path = root / "data" / "_minibank.npz"
np.savez_compressed(bank_path, F=F, t_grid=t_grid, zs=zs, zc=zc,
                    paramnames=np.array(paramnames), meta=json.dumps({"v": 0}))

bank = FisherBank(bank_path)
fc = forecast.Forecast(bank, rf)

# interpolation sanity: recover the t^2 law at an off-grid point
Fq = bank.F(3, 3.3e3)
ref = F0 * (3.3e3 / 1e4) ** 2
err = np.max(np.abs(Fq - ref) / (np.abs(ref) + 1e-30))
print(f"interp max rel err at off-grid t: {err:.2e}")

for scen in [scenarios.clean(), scenarios.measured(),
             scenarios.uniform(0.5, "dtv"), scenarios.uniform(0.97, "all")]:
    t0 = time.time()
    sig = fc.significance(scen, 1e4)
    req = fc.required_hours(scen, target=5.0)
    print(f"{scen.name:20s} S(1e4hr)={sig:8.2f}  t_req(5sig)={req:12.1f} hr "
          f"[{time.time()-t0:.2f}s]")

fac = scenarios.measured().bin_factors_for_zbins(zs)
print("measured per-bin (v_frac, w_bar):")
for k in range(nb):
    print(f"  z=[{zs[k]:.2f},{zs[k+1]:.2f}] v={fac[k,0]:.4f} w={fac[k,1]:.4f}")
bank_path.unlink()

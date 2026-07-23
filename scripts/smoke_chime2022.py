#!/usr/bin/env python3
"""Smoke test for the Amiri et al. (2022) Appendix-A configuration:
build the Planck-2018+mnu P(k) cache, run one bin's Fisher matrix on the
rfi-noise-model-chime branch, and check the per-bin marginalisation."""
import contextlib, io, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from baonoise.compat import import_radiofisher
from baonoise import pkcache, survey, forecast

rf, rf_dir = import_radiofisher()
import subprocess
branch = subprocess.run(["git", "-C", str(rf_dir), "branch", "--show-current"],
                        capture_output=True, text=True).stdout.strip()
print("RadioFisher branch:", branch)

cosmo = pkcache.load_fiducial_cosmology(
    rf, ROOT / "data" / "cache_pk_chime2022.dat",
    cosmo=survey.chime2022_cosmo(rf, rf_dir))
print(f"P(k): k in [{cosmo['k_in_min']:.1e}, {cosmo['k_in_max']:.1f}] Mpc^-1")
cosmo_fns = rf.background_evolution_splines(cosmo)

zs, zc = survey.chime2022_zbins()
expt = survey.chime2022_experiment(rf, rf_dir, ttot_hours=8760.0)  # 1 yr
print(f"zbins: {len(zc)}; Tsys(z=1)={expt['Tsys_tot(z)'](1.0)/1e3:.0f} K; "
      f"Sarea={expt['Sarea']:.3f} sr; eps_fg={expt['epsilon_fg']}")

i = 6  # z = 1.4-1.5 (ch30 + cluster)
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    F, names = rf.fisher(zs[i], zs[i + 1], cosmo, expt, cosmo_fns)
print(f"fisher() bin z=[{zs[i]},{zs[i+1]}] took {time.time()-t0:.1f}s")
print("paramnames:", names)
print("diag:", ["%.2e" % d for d in np.asarray(F).diagonal()])

# per-bin marginalisation (Appendix-A style) on the raw matrix


class _B:  # minimal bank stand-in for the marginal helper
    paramnames = names


fc = forecast.Forecast.__new__(forecast.Forecast)
fc.bank = _B(); fc.rf = rf; fc.style = "perbin_A"
cov, kn = fc._marginal_cov_bin(np.asarray(F))
print("kept params:", kn)
sA = np.sqrt(cov[kn.index("A"), kn.index("A")])
ip, il = kn.index("aperp"), kn.index("apar")
sdv = np.sqrt((4/9)*cov[ip, ip] + (1/9)*cov[il, il] + (4/9)*cov[ip, il])
print(f"1 yr, clean, bin z=[{zs[i]},{zs[i+1]}]: sigma_A={sA:.3f} "
      f"(per-bin BAO {1/sA:.1f} sigma), sigma_DV/DV={100*sdv:.2f}%")

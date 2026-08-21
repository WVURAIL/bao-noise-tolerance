#!/usr/bin/env python3
"""Verification pass for a Fisher bank and the forecast machinery.

Targets the bank's recorded configuration, cosmology, and canonical H I
profile. RadioFisher is accepted through its versioned capability contract,
not a historical branch name.

Checks:
1. Interpolation: bank F(z,t) at off-grid t vs direct rf.fisher() calls.
2. Physical scaling: S ~ t in the noise-dominated regime, saturating in
   the cosmic-variance limit.
3. Magnitudes: per-cent-level distance errors, tens-of-sigma BAO.
4. Scenario ordering: clean >= excised >= kept(fourier).
5. In-fork RFI hooks vs the bank-rescaling path (must agree).
"""
import argparse
import contextlib
import io
import sys
import time

import numpy as np

from baonoise import cosmologies, forecast, pkcache, scenarios, survey
from baonoise.compat import import_radiofisher
from baonoise.fisherbank import FisherBank
from baonoise.resources import DEFAULT_BANK, filesystem_data_file

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default=DEFAULT_BANK)
args = ap.parse_args()

rf, rf_dir = import_radiofisher()
bank = FisherBank(args.bank)
config = bank.meta["config"]
style = "perbin_A" if config == "chime2022" else "shared_A"
fc = forecast.Forecast(bank, rf, style=style, rf_dir=rf_dir)
print(f"bank config={config} style={style}; RadioFisher={rf_dir}")

# matching cosmology
if config == "chime2022":
    cosmology_name = bank.meta["cosmology"]
    cosmo = pkcache.load_fiducial_cosmology(
        rf, filesystem_data_file(
            "cache_pk_chime2022.dat" if cosmology_name == "planck2018"
            else f"cache_pk_chime2022_{cosmology_name}.dat"),
        cosmo=cosmologies.get(cosmology_name, rf, rf_dir))
else:
    base = cosmologies.with_astrophysical_profile(
        rf.experiments.cosmo, "bull2015", rf=rf)
    cosmo = pkcache.load_fiducial_cosmology(
        rf, filesystem_data_file("cache_pk.dat"), cosmo=base)
make_expt = lambda t: survey.experiment_from_bank_metadata(
    rf, rf_dir, bank.meta, ttot_hours=t)
cosmo_fns = rf.background_evolution_splines(cosmo)
ok = True


def sigma_A_bin_from(F):
    if style == "perbin_A":
        return fc._sigma_A_from_bin_matrix(np.asarray(F))
    Ftot, nn = rf.combined_fisher_matrix(
        [np.asarray(F)], names=list(bank.paramnames),
        exclude=list(forecast.EXCLUDE), expand=list(forecast.EXPAND))
    return float(np.sqrt(np.linalg.inv(Ftot)[nn.index("A"), nn.index("A")]))


# ---------------------------------------------------------------- 1. interp
print("== 1. bank interpolation vs direct fisher() at off-grid t ==")
nb = bank.nbins
for ibin, t in [(nb // 3, 7.7e2), (nb // 2, 3.3e3), (nb - 2, 6.1e4)]:
    expt = make_expt(t)
    with contextlib.redirect_stdout(io.StringIO()):
        Fd, names = rf.fisher(bank.zs[ibin], bank.zs[ibin + 1], cosmo, expt,
                              cosmo_fns)
    sA_d = sigma_A_bin_from(Fd)
    sA_b = sigma_A_bin_from(bank.F(ibin, t))
    rel = abs(sA_b / sA_d - 1.0)
    flag = "PASS" if rel < 0.01 else "FAIL"
    ok &= flag == "PASS"
    print(f"  bin {ibin} t={t:9.1f} hr  d(sigma_A)/sigma_A={rel:.2e}  [{flag}]")

# ------------------------------------------------------------- 2. scaling
print("== 2. S(t) scaling ==")
sc = scenarios.clean()
s_lo = [fc.significance(sc, t) for t in (30.0, 60.0)]
slope_lo = np.log(s_lo[1] / s_lo[0]) / np.log(2.0)
s_hi = [fc.significance(sc, t) for t in (2e5, 4e5)]
slope_hi = np.log(s_hi[1] / s_hi[0]) / np.log(2.0)
flag = "PASS" if (0.85 < slope_lo <= 1.05 and slope_hi < 0.35) else "FAIL"
ok &= flag == "PASS"
print(f"  noise-dominated slope dlnS/dlnt = {slope_lo:.3f} (expect ~1)")
print(f"  CV-saturated slope              = {slope_hi:.3f} (expect <<1)  [{flag}]")

# -------------------------------------------------- 3. magnitude checks
print("== 3. clean-survey magnitudes ==")
S10k = fc.significance(scenarios.clean(), 1e4)
param = "aperp" if style == "perbin_A" else "aperp0"
sig_ap = np.array([fc.sigma_param_bin(scenarios.clean(), 1e4, i, param)
                   for i in range(bank.nbins)])
med = float(np.median(sig_ap[np.isfinite(sig_ap)]))
print(f"  median per-bin sigma(aperp) at 10k hr: {100*med:.2f}%")
print(f"  total BAO significance at 10k hr: {S10k:.1f} sigma")
flag = "PASS" if (0.003 < med < 0.10 and S10k > 10) else "FAIL"
ok &= flag == "PASS"
print(f"  [{flag}] (expect percent-level distance errors, tens-of-sigma BAO)")

# ------------------------------------------------------------ 4. ordering
print("== 4. scenario ordering ==")
t2 = survey.years_to_hours(2.0, 0.75)
s_clean = fc.significance(scenarios.clean(), t2)
s_exc = fc.significance(scenarios.single_channel(30, 0.97, keep=False), t2)
s_kept_f = fc.significance(
    scenarios.single_channel(30, 0.97, keep=True, mode="fourier"), t2)
s_meas = fc.significance(scenarios.measured(), t2)
print(f"  S(clean)={s_clean:.2f}  S(ch30 excised)={s_exc:.2f}  "
      f"S(ch30 kept, fourier)={s_kept_f:.2f}  S(measured)={s_meas:.2f}")
flag = ("PASS" if s_clean >= s_exc >= s_kept_f and s_clean >= s_meas
        else "FAIL")
ok &= flag == "PASS"
print(f"  [{flag}] clean >= excised >= kept(fourier); clean >= measured")

# ---------------------------------------------- 5. fork-hook equivalence
print("== 5. in-fork RFI hooks vs bank-rescaling path ==")
IB = 6
cases = [
    ("measured", scenarios.measured(), [IB]),
    ("uniform50_dtv", scenarios.uniform(0.50, scenarios.DTV_BAND), [IB]),
    ("ch30_kept_fourier",
     scenarios.single_channel(30, 0.97, keep=True, mode="fourier"), [IB]),
    ("measured (all bins)", scenarios.measured(), None),
]
for name, sc, bins in cases:
    t0 = time.time()
    sA_bank = fc.sigma_A(sc, 1e4, bins=bins)
    sA_hook = fc.sigma_A_direct(sc, 1e4, bins=bins, cosmo=cosmo,
                                cosmo_fns=cosmo_fns, rf_dir=rf_dir)
    rel = abs(sA_hook / sA_bank - 1.0)
    flag = "PASS" if rel < 0.015 else "FAIL"
    ok &= flag == "PASS"
    print(f"  {name:22s} sigma_A bank={sA_bank:.5f} hook={sA_hook:.5f} "
          f"rel diff={rel:.2e} [{flag}] ({time.time()-t0:.0f}s)")

print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

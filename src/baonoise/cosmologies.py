"""Named fiducial cosmologies for the forecast banks.

'planck2018': the CHIME Overview forecast fiducial (Planck 2018 CMB-only
                 best fit; Amiri et al. 2022 Eq. A8, as implemented in
                 S. Foreman's chime2021/experiments_CHIME.py). This is the
                 fiducial of record: it is what makes the Fig. 31
                 validation exact.
'pact2025': ACT DR6 + Planck + CMB lensing + DESI BAO (P-ACT-LB;
                 ACT Collaboration 2025, arXiv:2503.14452): h=0.6822,
                 Obh2=0.0226, Och2=0.118, ns=0.974, sigma8=0.813,
                 sum mnu=0.06 eV. Provided to demonstrate that masking
                 penalties are insensitive to the fiducial.
"""
from __future__ import annotations

import copy

_PACT = dict(h=0.6822, obh2=0.0226, och2=0.118, ns=0.974, sigma8=0.813,
             mnu=0.06)


def get(name: str, rf, rf_dir) -> dict:
    """Return a RadioFisher cosmo dict for a named fiducial."""
    from . import survey

    base = survey.chime2022_cosmo(rf, rf_dir)   # planck2018 structure
    if name in (None, "planck2018"):
        return base
    if name == "pact2025":
        c = copy.deepcopy(base)
        h = _PACT["h"]
        onuh2 = _PACT["mnu"] / 93.04
        omega_b = _PACT["obh2"] / h**2
        omega_m = (_PACT["och2"] + _PACT["obh2"] + onuh2) / h**2
        c.update({
            "h": h,
            "omega_b_0": omega_b,
            "omega_M_0": omega_m,
            "omega_lambda_0": 1.0 - omega_m,
            "ns": _PACT["ns"],
            "sigma_8": _PACT["sigma8"],
            "mnu": _PACT["mnu"],
        })
        return c
    raise ValueError(f"unknown cosmology: {name}")

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
from numbers import Real

import numpy as np

from ._validation import nonnegative_scalar, positive_scalar

_PACT = dict(h=0.6822, obh2=0.0226, och2=0.118, ns=0.974, sigma8=0.813,
             mnu=0.06)

NEUTRINO_MASS_DENSITY_EV = 93.04

ASTROPHYSICAL_PROFILE_KEY = "astrophysical_model_profile"


def with_astrophysical_profile(cosmo: dict, profile: str, rf) -> dict:
    """Apply a backend-owned canonical H I model profile."""
    helper = getattr(rf, "with_astrophysical_profile", None)
    if not callable(helper):
        raise RuntimeError(
            "RadioFisher advertises no with_astrophysical_profile helper")
    return helper(cosmo, profile)


def physical_densities(cosmo: dict) -> dict[str, float]:
    """Resolve CAMB baryon, cold-dark-matter, and neutrino densities.

    ``omega_M_0`` is total matter, including massive neutrinos. Historical
    RadioFisher code treated ``(Omega_m - Omega_b) h^2`` as cold dark matter
    and then supplied ``mnu`` separately, adding the neutrino density twice.
    The explicit ``ombh2``/``omch2``/``omnuh2`` triplet is authoritative and
    must be supplied all-or-none.  Otherwise all three values are derived
    using the total-matter convention above.  This matches RadioFisher's
    backend contract and prevents a partially overridden cosmology.
    """
    h = positive_scalar(cosmo.get("h"), "cosmology h")

    explicit_keys = ("ombh2", "omch2", "omnuh2")
    present = {key for key in explicit_keys if key in cosmo}
    if present and present != set(explicit_keys):
        missing = sorted(set(explicit_keys) - present)
        raise ValueError(
            "explicit physical densities are all-or-none; missing: "
            + ", ".join(missing))

    if present:
        ombh2 = cosmo["ombh2"]
        omch2 = cosmo["omch2"]
        omnuh2 = cosmo["omnuh2"]
    else:
        omega_b = nonnegative_scalar(
            cosmo.get("omega_b_0"), "cosmology omega_b_0")
        omega_m = nonnegative_scalar(
            cosmo.get("omega_M_0"), "cosmology omega_M_0")
        mnu = nonnegative_scalar(cosmo.get("mnu", 0.0), "cosmology mnu")
        includes_neutrinos = cosmo.get(
            "omega_M_0_includes_neutrinos", True)
        if not isinstance(includes_neutrinos, (bool, np.bool_)):
            raise TypeError(
                "omega_M_0_includes_neutrinos must be a boolean")
        ombh2 = omega_b * h**2
        omnuh2 = mnu / NEUTRINO_MASS_DENSITY_EV
        omch2 = (omega_m * h**2 - ombh2
                 - (omnuh2 if includes_neutrinos else 0.0))

    out = {
        key: nonnegative_scalar(value, f"cosmology {key}")
        for key, value in {
            "ombh2": ombh2, "omch2": omch2, "omnuh2": omnuh2}.items()
    }
    return out


def with_explicit_physical_densities(cosmo: dict) -> dict:
    """Copy a RadioFisher cosmology and attach unambiguous CAMB densities."""
    out = copy.deepcopy(cosmo)
    densities = physical_densities(out)
    out.update(densities)
    out["mnu"] = densities["omnuh2"] * NEUTRINO_MASS_DENSITY_EV
    h2 = float(out["h"]) ** 2
    out["omega_cdm_0"] = densities["omch2"] / h2
    out["omega_nu_0"] = densities["omnuh2"] / h2
    return out


def get(name: str, rf, rf_dir) -> dict:
    """Return a RadioFisher cosmo dict for a named fiducial."""
    from . import survey

    base = with_explicit_physical_densities(with_astrophysical_profile(
        survey.chime2022_cosmo(rf, rf_dir), "chime_overview_2022", rf=rf))
    if name in (None, "planck2018"):
        return base
    if name == "pact2025":
        c = copy.deepcopy(base)
        h = _PACT["h"]
        onuh2 = _PACT["mnu"] / NEUTRINO_MASS_DENSITY_EV
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
            "ombh2": _PACT["obh2"],
            "omch2": _PACT["och2"],
            "omnuh2": onuh2,
            "omega_cdm_0": _PACT["och2"] / h**2,
            "omega_nu_0": onuh2 / h**2,
        })
        return with_explicit_physical_densities(c)
    raise ValueError(f"unknown cosmology: {name}")

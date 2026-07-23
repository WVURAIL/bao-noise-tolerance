"""Generate the fiducial matter power spectrum cache file that RadioFisher's
``cached_camb_output`` expects, using the *Python* camb package instead of the
legacy Fortran CAMB executable.

File format (see ``baofisher.cached_camb_output``):
    two columns:  k [Mpc^-1],  P(k) [Mpc^3]
with P(k) at z=0 renormalized to the fiducial sigma_8, and a one-line header
"# <hash>#".  We write a sentinel hash and always load with force_load=True.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SENTINEL_HASH = "pythoncamb"


def build_pk_cache(cosmo: dict, cachefile: str | Path, kmax_h: float = 20.0,
                   npoints: int = 1400) -> Path:
    """Write a RadioFisher-format P(k) cache for the given fiducial cosmology.

    Parameters mirror RadioFisher's ``experiments.cosmo`` dictionary. The
    output is renormalized so sigma_8 equals ``cosmo['sigma_8']`` exactly,
    reproducing what ``cached_camb_output`` does with Fortran CAMB output.
    """
    try:
        import camb
    except ModuleNotFoundError as exc:      # pragma: no cover - install path
        raise ModuleNotFoundError(
            "camb is needed only to regenerate the fiducial P(k) cache, and "
            f"no cache was found at the requested path. Either restore the "
            f"committed cache under data/, or install the extra: "
            f"pip install -e '.[pk]'") from exc

    cachefile = Path(cachefile)
    h = cosmo["h"]
    ombh2 = cosmo["omega_b_0"] * h**2
    omch2 = (cosmo["omega_M_0"] - cosmo["omega_b_0"]) * h**2
    mnu = cosmo.get("mnu", 0.0) or 0.0

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0,
                       mnu=mnu, nnu=cosmo.get("N_eff", 3.046), tau=0.06)
    pars.InitPower.set_params(As=2.1e-9, ns=cosmo["ns"])
    pars.set_matter_power(redshifts=[0.0], kmax=kmax_h, accurate_massive_neutrino_transfers=False)
    pars.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(pars)
    kh, _z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax_h,
                                                   npoints=npoints)
    sigma8_camb = results.get_sigma8_0()

    # Renormalize to the fiducial sigma_8 and convert h units -> Mpc units.
    renorm = (cosmo["sigma_8"] / sigma8_camb) ** 2
    k_mpc = kh * h                      # k [h/Mpc] -> [1/Mpc]
    pk_mpc = pk[0] * renorm / h**3      # P [(Mpc/h)^3] -> [Mpc^3]

    hdr = f"{SENTINEL_HASH}#"
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(cachefile, np.column_stack([k_mpc, pk_mpc]), header=hdr)
    return cachefile


def load_fiducial_cosmology(rf, cachefile: str | Path, force_build: bool = False,
                            cosmo: dict | None = None):
    """Return a fiducial cosmo dict with pk_nobao/fbao attached.

    Default: RadioFisher's Planck-2013 fiducial with mnu=0 (Bull et al.
    2015). Pass cosmo= for other configs (e.g. the Planck-2018 + mnu=0.06
    fiducial of the CHIME Overview forecasts). Builds the P(k) cache with
    python-camb on first use, mirroring convert_to_camb's decomposition
    (omch2 = (Om-Ob)h^2 with the neutrino density added on top).
    """
    import copy

    if cosmo is None:
        cosmo = copy.deepcopy(rf.experiments.cosmo)
        cosmo["mnu"] = 0.0
    else:
        cosmo = copy.deepcopy(cosmo)
    cachefile = Path(cachefile)
    if force_build or not cachefile.exists():
        build_pk_cache(cosmo, cachefile)
    cosmo = rf.load_power_spectrum(cosmo, str(cachefile), force_load=True)
    return cosmo

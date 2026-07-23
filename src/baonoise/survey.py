"""CHIME survey definition and redshift binning, mirroring how
``full_experiment.py`` drives the 'yCHIME' (cylinder interferometer) case.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

HRS_MHZ = 3.6e9          # 1 hour in MHz^-1 (radiofisher.units)
NU_LINE = 1420.406       # 21cm rest frequency [MHz]
HOURS_PER_YEAR = 8766.0  # mean calendar year, hours

# Literature-anchored time accounting (see paper, duty-cycle paragraph):
# * ONSKY_YEAR_HOURS: the CHIME Overview forecast normalization: t_tot =
#   "1 yr" means 8,760 on-sky hours with no duty factor (Amiri et al. 2022,
#   ApJS 261, 29, Table 2 / Appendix A; 365*24 in Foreman's chime2021 code).
# * DUTY_2019_PRACTICE: empirical cosmology-quality fraction of the 2019
#   CHIME dataset (Amiri et al. 2025, arXiv:2511.19620): 94 of 309 sidereal
#   days retained x ~0.5 night-only ~= 0.152. Their additional 38.7% masking
#   of surviving night data INCLUDES RFI flagging and must not be applied on
#   top of the masking scenarios here (double counting); including it gives
#   0.093. The Overview's daily-processing rule ("any day with less than 70%
#   coverage after masking is discarded", Amiri et al. 2022) is the
#   mechanism behind the day-retention factor; the 102-night dataset of the
#   2023 detection (ApJ 947, 16) corroborates the ~100-night/year scale.
ONSKY_YEAR_HOURS = 8760.0
DUTY_2019_PRACTICE = 0.152


def chime_experiment(rf, rf_dir: str | Path, ttot_hours: float = 1e4,
                     epsilon_fg: float = 1e-6, k_nl0: float = 0.14,
                     nx_file: str | Path | None = None) -> dict:
    """Return the CHIME experiment dict configured like full_experiment's
    'yCHIME' entry (mode 'icyl'), with an absolute n(u) file path so we can
    run from any working directory."""
    expt = copy.deepcopy(rf.experiments.CHIME)
    expt["mode"] = "icyl"
    expt["ttot"] = ttot_hours * HRS_MHZ
    expt["epsilon_fg"] = epsilon_fg
    expt["k_nl0"] = k_nl0
    if nx_file is None:
        from .layout import ensure_chime_nx
        nx_file = ensure_chime_nx(rf_dir, Path(__file__).resolve().parents[2] / "data")
    expt["n(x)"] = str(nx_file)
    return expt


def chime_zbins(rf, expt: dict, dz: float = 0.1):
    """Equal-dz redshift bins over the CHIME band (400-800 MHz)."""
    zs, zc = rf.zbins_equal_spaced(rf.overlapping_expts(expt), dz=dz)
    return np.asarray(zs), np.asarray(zc)


# ----------------------------------------------------------------------
# CHIME Overview configuration (Amiri et al. 2022, ApJS 261, 29, App. A;
# implemented in sjforeman/RadioFisher branch chime-update, chime2021/)
# ----------------------------------------------------------------------

def _import_experiments_chime(rf_dir: str | Path):
    import importlib
    import sys
    chime_dir = str(Path(rf_dir) / "chime2021")
    for p in (str(rf_dir), chime_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module("experiments_CHIME")


def chime2022_experiment(rf, rf_dir: str | Path,
                         ttot_hours: float = 8760.0) -> dict:
    """CHIME as forecast in the Overview paper: as-built 4x256 geometry,
    Tsys_tot = 55 K, S_sky = 31,000 deg^2, BAO-shift-only USE flags,
    epsilon_fg = 0, Simon Foreman's as-built n(u)."""
    exps = _import_experiments_chime(rf_dir)
    expt = copy.deepcopy(exps.CHIME)
    expt["Tsys_tot(z)"] = exps.CHIME["Tsys_tot(z)"]  # deepcopy keeps lambda ref
    expt["ttot"] = ttot_hours * HRS_MHZ
    expt["n(x)"] = str(Path(rf_dir) / "chime2021" / expt["n(x)"])
    return expt


def chime2022_cosmo(rf, rf_dir: str | Path) -> dict:
    """Planck-2018 fiducial cosmology of the Overview forecasts."""
    exps = _import_experiments_chime(rf_dir)
    return copy.deepcopy(exps.cosmo)


def chime2022_zbins():
    """The 15 redshift bins of Amiri et al. (2022) Table 2 (dz=0.1 to
    z=1.8, dz~0.16 above, matching DESI binning)."""
    zs = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                   1.9, 2.04, 2.20, 2.355, 2.51])
    return zs, 0.5 * (zs[1:] + zs[:-1])


def zbin_freq_range(zmin: float, zmax: float) -> tuple[float, float]:
    """Frequency interval [MHz] covered by a redshift bin (lo, hi)."""
    return NU_LINE / (1.0 + zmax), NU_LINE / (1.0 + zmin)


def hours_to_years(hours: np.ndarray | float, duty: float = 0.75):
    """Calendar years for a 24/7 transit survey with the given duty cycle."""
    return np.asarray(hours) / (HOURS_PER_YEAR * duty)


def years_to_hours(years: np.ndarray | float, duty: float = 0.75):
    return np.asarray(years) * HOURS_PER_YEAR * duty

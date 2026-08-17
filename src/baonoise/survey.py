"""CHIME survey definition and redshift binning, mirroring how
``full_experiment.py`` drives the 'yCHIME' (cylinder interferometer) case.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ._validation import positive_scalar
from .constants import HI_REST_FREQUENCY_MHZ

HRS_MHZ = 3.6e9                  # 1 hour in MHz^-1 (radiofisher.units)
MEAN_CALENDAR_YEAR_HOURS = 8766.0  # 365.25 days
OVERVIEW_ONSKY_YEAR_HOURS = 8760.0  # 365 days, Overview normalization

# Literature-anchored time accounting (see paper, duty-cycle paragraph):
# * OVERVIEW_ONSKY_YEAR_HOURS: the CHIME Overview normalization: t_tot =
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
        from .resources import SYNTHETIC_BASELINE_NAME, filesystem_data_file
        nx_file = filesystem_data_file(SYNTHETIC_BASELINE_NAME)
    expt["n(x)"] = str(nx_file)
    return expt


def chime_zbins(rf, expt: dict, dz: float = 0.1):
    """Equal-dz redshift bins over the CHIME band (400-800 MHz)."""
    # CHIME is a single instrument (no ``overlap`` experiment component), so
    # the backend's supported binning helper consumes its experiment directly.
    zs, zc = rf.zbins_equal_spaced(expt, dz=dz)
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
    module = importlib.import_module("experiments_CHIME")
    module_file = Path(module.__file__).resolve()
    expected = Path(chime_dir).resolve()
    try:
        module_file.relative_to(expected)
    except ValueError as exc:
        raise RuntimeError(
            "experiments_CHIME checkout mismatch: requested data from "
            f"{expected}, but Python already imported {module_file}. Start a "
            "fresh process with one RadioFisher checkout.") from exc
    return module


def chime2022_experiment(rf, rf_dir: str | Path,
                         ttot_hours: float = OVERVIEW_ONSKY_YEAR_HOURS) -> dict:
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
    return (HI_REST_FREQUENCY_MHZ / (1.0 + zmax),
            HI_REST_FREQUENCY_MHZ / (1.0 + zmin))


def hours_to_years(hours: np.ndarray | float, duty: float = 1.0,
                   hours_per_year: float = MEAN_CALENDAR_YEAR_HOURS):
    """Years on an explicit hour basis, optionally adjusted by ``duty``."""
    duty = positive_scalar(duty, "duty")
    hours_per_year = positive_scalar(hours_per_year, "hours_per_year")
    return np.asarray(hours) / (hours_per_year * duty)


def years_to_hours(years: np.ndarray | float, duty: float = 1.0,
                   hours_per_year: float = MEAN_CALENDAR_YEAR_HOURS):
    """Hours on an explicit year basis, optionally adjusted by ``duty``."""
    duty = positive_scalar(duty, "duty")
    hours_per_year = positive_scalar(hours_per_year, "hours_per_year")
    return np.asarray(years) * hours_per_year * duty

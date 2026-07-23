"""High-level one-call API: the entry point for users of the tool.

Minimal usage (with the CHIME Fisher bank shipped in the repository):

    from baonoise import api

    fc = api.load()                              # CHIME bank, RadioFisher fork
    mask = {17: 0.33, 30: 0.97, 31: 0.24}        # ATSC channel -> masked frac
    print(api.required_time(fc, mask))           # hours/years/penalty to 5sigma

Anything more elaborate (custom experiments, new banks, direct hook
evaluation) drops down to the underlying modules: scenarios, fisherbank,
forecast, survey.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import forecast as _forecast
from . import scenarios as _scenarios
from . import survey as _survey
from .compat import import_radiofisher
from .fisherbank import FisherBank

_DATA = Path(__file__).resolve().parents[2] / "data"
# Prefer the CHIME Overview (Amiri et al. 2022, Appendix A) configuration;
# fall back to the Bull et al. (2015) design bank.
_C22 = _DATA / "fisher_bank_chime2022.npz"
DEFAULT_BANK = _C22 if _C22.exists() else _DATA / "fisher_bank_chime.npz"


def load(bank: str | Path = DEFAULT_BANK, rf_dir=None) -> _forecast.Forecast:
    """Load a Fisher bank and return a Forecast ready for scenario queries.
    The marginalisation style follows the bank's config: per-bin BAO
    amplitudes (Amiri et al. 2022 Appendix A) for 'chime2022', shared
    amplitude (Bull et al. 2015) otherwise."""
    rf, _ = import_radiofisher(rf_dir)
    b = FisherBank(bank)
    style = "perbin_A" if b.meta.get("config") == "chime2022" else "shared_A"
    return _forecast.Forecast(b, rf, style=style)


def scenario_from(mask=None, uniform=None, band: str = "dtv",
                  excise_threshold: float = 0.5,
                  mode: str = "time", residuals=None,
                  residual: float = 0.0,
                  residual_excise_threshold: float = np.inf
                  ) -> _scenarios.Scenario:
    """Build a Scenario from either a {channel: masked_fraction} dict or a
    uniform masked fraction over a band ('dtv' or 'all').

    ``residuals`` ({channel: r}) or ``residual`` (uniform r) adds the
    contamination surviving the mask; both default to none, which reproduces
    masking-only behavior exactly.
    """
    if (mask is None) == (uniform is None):
        raise ValueError("provide exactly one of mask= or uniform=")
    if residuals is not None and residual:
        raise ValueError("provide at most one of residuals= or residual=")
    if mask is not None:
        res = dict(residuals) if residuals else (
            {int(c): float(residual) for c in mask} if residual else {})
        return _scenarios.Scenario("user", "user scenario",
                                   fractions=dict(mask),
                                   excise_threshold=excise_threshold,
                                   mode=mode, residuals=res,
                                   residual_excise_threshold=residual_excise_threshold)
    sc = _scenarios.uniform(float(uniform), band, mode=mode,
                            residual=float(residual))
    if residuals:
        sc.residuals = dict(residuals)
    sc.residual_excise_threshold = residual_excise_threshold
    return sc


def required_time(fc: _forecast.Forecast, mask=None, uniform=None,
                  band: str = "dtv", target: float = 5.0,
                  zbin: int | None = None, mode: str = "time",
                  excise_threshold: float = 0.5,
                  duty: float = 1.0, residuals=None, residual: float = 0.0,
                  residual_excise_threshold: float = np.inf) -> dict:
    """Observing time needed to reach a BAO detection target.

    target : detection significance A/sigma_A (e.g. 5.0)
    zbin   : None for the full survey, or a bin index for that bin alone
    duty   : 1.0 quotes on-sky years at the Overview normalization
             (8,766 hr/yr basis); use survey.DUTY_2019_PRACTICE (0.152)
             for calendar years at demonstrated CHIME practice.
    Returns a dict with on-sky hours, years at `duty`, and the penalty
    relative to the RFI-free survey (same target, same bins).
    """
    sc = scenario_from(mask, uniform, band, excise_threshold, mode,
                       residuals=residuals, residual=residual,
                       residual_excise_threshold=residual_excise_threshold)
    bins = None if zbin is None else [int(zbin)]
    metric = lambda s: (lambda t: fc.significance(s, t, bins=bins))
    hours = fc.required_hours_metric(metric(sc), target)
    clean = fc.required_hours_metric(metric(_scenarios.clean()), target)
    years = float(_survey.hours_to_years(hours, duty)) if np.isfinite(hours) else np.inf
    return dict(hours=float(hours), years=years, duty=duty,
                penalty_vs_clean=float(hours / clean) if np.isfinite(hours) else np.inf,
                target_sigma=target,
                zbin="survey" if zbin is None else int(zbin))


def significance(fc: _forecast.Forecast, years: float, mask=None,
                 uniform=None, band: str = "dtv", zbin: int | None = None,
                 mode: str = "time", excise_threshold: float = 0.5,
                 duty: float = 1.0, residuals=None, residual: float = 0.0,
                 residual_excise_threshold: float = np.inf) -> float:
    """BAO detection significance after `years` calendar years at `duty`."""
    sc = scenario_from(mask, uniform, band, excise_threshold, mode,
                       residuals=residuals, residual=residual,
                       residual_excise_threshold=residual_excise_threshold)
    bins = None if zbin is None else [int(zbin)]
    t = float(_survey.years_to_hours(years, duty))
    return fc.significance(sc, t, bins=bins)


def tolerance_curve(fc: _forecast.Forecast, fracs=None, band: str = "dtv",
                    target: float = 5.0, zbin: int | None = None,
                    duty: float = 1.0):
    """(fracs, years) arrays: required calendar years vs uniform masked
    fraction of `band`, the noise-tolerance curve."""
    if fracs is None:
        fracs = np.concatenate([np.arange(0.0, 0.96, 0.05), [0.97]])
    fracs = np.asarray(fracs, dtype=float)
    yrs = np.array([required_time(fc, uniform=float(f), band=band,
                                  target=target, zbin=zbin,
                                  duty=duty)["years"] for f in fracs])
    return fracs, yrs


def threshold_curve(fc: _forecast.Forecast, operating_points: dict,
                    target: float = 5.0, zbin: int | None = None,
                    duty: float = 1.0, mode: str = "time",
                    excise_threshold: float = 0.5,
                    residual_excise_threshold: float = np.inf) -> dict:
    """Required time as a function of detector threshold, the closed loop.

    ``operating_points`` maps a threshold (any orderable label, e.g. eta) to
    ``{channel: (masked_fraction, residual_ratio)}``. Both halves of the cost
    move with the threshold in opposite directions, so unlike
    :func:`tolerance_curve` this has an interior minimum: the threshold that
    minimises total time to the target.

    Returns ``{'eta': [...], 'years': [...], 'penalty': [...],
    'best_eta': ..., 'best_years': ...}``. A threshold whose residual makes
    the target unreachable yields ``inf`` rather than being dropped, so the
    caller can see where the wall is.
    """
    etas = sorted(operating_points)
    bins = None if zbin is None else [int(zbin)]
    clean_hours = fc.required_hours_metric(
        lambda t: fc.significance(_scenarios.clean(), t, bins=bins), target)

    years, penalty = [], []
    for eta in etas:
        sc = _scenarios.at_threshold(
            operating_points[eta], eta=eta, mode=mode,
            excise_threshold=excise_threshold,
            residual_excise_threshold=residual_excise_threshold)
        h = fc.required_hours_metric(
            lambda t: fc.significance(sc, t, bins=bins), target)
        years.append(float(_survey.hours_to_years(h, duty))
                     if np.isfinite(h) else np.inf)
        penalty.append(float(h / clean_hours) if np.isfinite(h) else np.inf)

    years = np.asarray(years)
    ibest = int(np.argmin(years)) if np.any(np.isfinite(years)) else -1
    return dict(eta=np.asarray(etas, dtype=float), years=years,
                penalty=np.asarray(penalty),
                best_eta=(float(etas[ibest]) if ibest >= 0 else np.nan),
                best_years=(float(years[ibest]) if ibest >= 0 else np.inf),
                target_sigma=target,
                zbin="survey" if zbin is None else int(zbin))

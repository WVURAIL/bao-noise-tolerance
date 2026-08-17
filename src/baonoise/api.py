"""High-level one-call API: the entry point for users of the tool.

Minimal usage (with the CHIME Fisher bank shipped in the package):

    from baonoise import api

    fc = api.load()                              # shipped CHIME bank; no backend
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
from ._validation import (nonnegative_scalar as _nonnegative_scalar,
                          positive_scalar as _positive_scalar)
from .fisherbank import ARTIFACT_FORECAST, FisherBank
from .resources import bank_file


def load(bank: str | Path | None = None, *, cosmology: str | None = None,
         rf_dir=None) -> _forecast.Forecast:
    """Load a Fisher bank and return a Forecast ready for scenario queries.
    The marginalisation style follows the bank's config: per-bin BAO
    amplitudes (Amiri et al. 2022 Appendix A) for 'chime2022', shared
    amplitude (Bull et al. 2015) otherwise. With ``rf_dir=None``, the per-bin
    path does not import RadioFisher. Passing ``rf_dir`` explicitly loads and
    retains that backend for subsequent direct evaluations."""
    if bank is not None and cosmology is not None:
        raise ValueError("provide either bank= or cosmology=, not both")
    source = bank if bank is not None else bank_file(cosmology or "planck2018")
    b = FisherBank(source, expected_artifact_kind=ARTIFACT_FORECAST)
    style = "perbin_A" if b.meta["config"] == "chime2022" else "shared_A"
    rf = None
    resolved_rf_dir = None
    if style == "shared_A" or rf_dir is not None:
        from .compat import import_radiofisher
        rf, resolved_rf_dir = import_radiofisher(rf_dir)
    return _forecast.Forecast(b, rf, style=style, rf_dir=resolved_rf_dir)


def scenario_from(mask=None, uniform=None, band: _scenarios.FrequencyBand =
                  _scenarios.DTV_BAND,
                  excise_threshold: float | None = None,
                  mode: str = "time", residuals=None,
                  residual: float = 0.0,
                  residual_excise_threshold: float = np.inf
                  ) -> _scenarios.Scenario:
    """Build a Scenario from either a {channel: masked_fraction} dict or a
    uniform masked fraction over an explicit :class:`FrequencyBand`.

    ``mask`` also accepts a prebuilt :class:`baonoise.scenarios.Scenario`
    (from :func:`scenarios.measured`, :func:`scenarios.at_threshold`, ...),
    which passes through unchanged. A Scenario already carries its own
    policy, so combining one with the other arguments here would silently
    ignore half of it; that combination is refused.

    ``residuals`` ({channel: r}) is available with a per-channel ``mask``;
    ``residual`` adds one uniform ratio to a frequency-band scenario.

    For a per-channel ``mask``, omitting ``excise_threshold`` uses the
    measured-scenario default (0.5). Uniform scenarios are retained-time
    stress tests by default and therefore do not excise; pass an explicit
    ``excise_threshold`` to apply excision to a uniform scenario. This keeps
    the retained-time tolerance-curve convention while ensuring that a
    supplied threshold is never ignored.
    """
    if (mask is None) == (uniform is None):
        raise ValueError("provide exactly one of mask= or uniform=")
    if isinstance(mask, _scenarios.Scenario):
        overridden = (excise_threshold is not None or residuals is not None
                      or float(residual) != 0.0 or mode != "time"
                      or np.isfinite(residual_excise_threshold))
        if overridden:
            raise ValueError(
                "mask= is already a Scenario carrying its own policy; build "
                "the overrides into the Scenario instead of passing them "
                "alongside it")
        return mask
    residual = _scenarios._residual(residual, "residual")
    if residuals is not None and residual != 0.0:
        raise ValueError("provide at most one of residuals= or residual=")
    if mask is not None:
        res = (dict(residuals) if residuals is not None else
               ({c: residual for c in mask} if residual else {}))
        threshold = (_scenarios.DEFAULT_EXCISE_THRESHOLD
                     if excise_threshold is None else excise_threshold)
        return _scenarios.Scenario("user", "user scenario",
                                   fractions=dict(mask),
                                   excise_threshold=threshold,
                                   mode=mode, residuals=res,
                                   residual_excise_threshold=residual_excise_threshold)
    if residuals is not None:
        raise ValueError(
            "residuals= is channel-specific and requires mask=; use the "
            "scalar residual= argument with uniform=")
    return _scenarios.uniform(
        uniform, band, mode=mode, residual=residual,
        excise_threshold=excise_threshold,
        residual_excise_threshold=residual_excise_threshold)


def required_time(fc: _forecast.Forecast, mask=None, uniform=None,
                  band: _scenarios.FrequencyBand = _scenarios.DTV_BAND,
                  target: float = 5.0,
                  zbin: int | None = None, mode: str = "time",
                  excise_threshold: float | None = None,
                  duty: float = 1.0, residuals=None, residual: float = 0.0,
                  residual_excise_threshold: float = np.inf,
                  hours_per_year: float = _survey.MEAN_CALENDAR_YEAR_HOURS) -> dict:
    """Observing time needed to reach a BAO detection target.

    target : detection significance A/sigma_A (e.g. 5.0)
    zbin   : None for the full survey, or a bin index for that bin alone
    duty   : 1.0 quotes uninterrupted years; use
             survey.DUTY_2019_PRACTICE (0.152)
             for calendar years at demonstrated CHIME practice.
    hours_per_year : defaults to the 365.25-day mean calendar year (8,766 h).
                     Pass survey.OVERVIEW_ONSKY_YEAR_HOURS for the CHIME
                     Overview's 365-day normalization.
    Returns a dict with on-sky hours, years at `duty`, and the penalty
    relative to the RFI-free survey (same target, same bins).
    """
    target = _positive_scalar(target, "target")
    duty = _positive_scalar(duty, "duty")
    hours_per_year = _positive_scalar(hours_per_year, "hours_per_year")
    sc = scenario_from(mask, uniform, band, excise_threshold, mode,
                       residuals=residuals, residual=residual,
                       residual_excise_threshold=residual_excise_threshold)
    bins = None if zbin is None else [int(zbin)]
    metric = lambda s: (lambda t: fc.significance(s, t, bins=bins))
    hours = fc.required_hours_metric(metric(sc), target)
    clean = fc.required_hours_metric(metric(_scenarios.clean()), target)
    years = (float(_survey.hours_to_years(hours, duty, hours_per_year))
             if np.isfinite(hours) else np.inf)
    return dict(hours=float(hours), years=years, duty=duty,
                hours_per_year=hours_per_year,
                penalty_vs_clean=float(hours / clean) if np.isfinite(hours) else np.inf,
                target_sigma=target,
                zbin="survey" if zbin is None else int(zbin))


def significance(fc: _forecast.Forecast, years: float, mask=None,
                 uniform=None, band: _scenarios.FrequencyBand =
                 _scenarios.DTV_BAND, zbin: int | None = None,
                 mode: str = "time", excise_threshold: float | None = None,
                 duty: float = 1.0, residuals=None, residual: float = 0.0,
                 residual_excise_threshold: float = np.inf,
                 hours_per_year: float = _survey.MEAN_CALENDAR_YEAR_HOURS) -> float:
    """BAO detection significance after `years` calendar years at `duty`."""
    years = _nonnegative_scalar(years, "years")
    duty = _positive_scalar(duty, "duty")
    hours_per_year = _positive_scalar(hours_per_year, "hours_per_year")
    sc = scenario_from(mask, uniform, band, excise_threshold, mode,
                       residuals=residuals, residual=residual,
                       residual_excise_threshold=residual_excise_threshold)
    bins = None if zbin is None else [int(zbin)]
    t = float(_survey.years_to_hours(years, duty, hours_per_year))
    return fc.significance(sc, t, bins=bins)


def tolerance_curve(fc: _forecast.Forecast, fracs=None,
                    band: _scenarios.FrequencyBand = _scenarios.DTV_BAND,
                    target: float = 5.0, zbin: int | None = None,
                    duty: float = 1.0,
                    hours_per_year: float = _survey.MEAN_CALENDAR_YEAR_HOURS):
    """(fracs, years) arrays: required calendar years vs uniform masked
    fraction of `band`, the noise-tolerance curve."""
    target = _positive_scalar(target, "target")
    duty = _positive_scalar(duty, "duty")
    hours_per_year = _positive_scalar(hours_per_year, "hours_per_year")
    if fracs is None:
        fracs = np.concatenate([np.arange(0.0, 0.96, 0.05), [0.97]])
    fracs = np.asarray(fracs, dtype=float)
    yrs = np.array([required_time(fc, uniform=float(f), band=band,
                                  target=target, zbin=zbin,
                                  duty=duty, hours_per_year=hours_per_year)["years"]
                    for f in fracs])
    return fracs, yrs


def threshold_curve(fc: _forecast.Forecast, operating_points: dict,
                    target: float = 5.0, zbin: int | None = None,
                    duty: float = 1.0, mode: str = "time",
                    excise_threshold: float = _scenarios.DEFAULT_EXCISE_THRESHOLD,
                    residual_excise_threshold: float = np.inf,
                    hours_per_year: float = _survey.MEAN_CALENDAR_YEAR_HOURS) -> dict:
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
    target = _positive_scalar(target, "target")
    duty = _positive_scalar(duty, "duty")
    hours_per_year = _positive_scalar(hours_per_year, "hours_per_year")
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
            lambda t, scenario=sc: fc.significance(
                scenario, t, bins=bins), target)
        years.append(float(_survey.hours_to_years(h, duty, hours_per_year))
                     if np.isfinite(h) else np.inf)
        penalty.append(float(h / clean_hours) if np.isfinite(h) else np.inf)

    years = np.asarray(years)
    ibest = int(np.argmin(years)) if np.any(np.isfinite(years)) else -1
    best_eta = etas[ibest] if ibest >= 0 else None
    if isinstance(best_eta, np.generic):
        best_eta = best_eta.item()
    return dict(eta=np.asarray(etas), years=years,
                penalty=np.asarray(penalty),
                best_eta=best_eta,
                best_years=(float(years[ibest]) if ibest >= 0 else np.inf),
                target_sigma=target,
                zbin="survey" if zbin is None else int(zbin))

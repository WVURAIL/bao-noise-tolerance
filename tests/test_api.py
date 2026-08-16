"""Validation tests for the high-level public API wrappers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baonoise import api, compat, scenarios


class _NoForecastWork:
    """Fail if invalid wrapper input reaches the forecast implementation."""

    def required_hours_metric(self, *args, **kwargs):
        raise AssertionError("forecast work started before validation")

    def significance(self, *args, **kwargs):
        raise AssertionError("forecast work started before validation")


def test_default_banked_api_does_not_import_radiofisher(monkeypatch):
    def unexpected_backend_import(_rf_dir=None):
        raise AssertionError("the per-bin bank does not need RadioFisher")

    monkeypatch.setattr(compat, "import_radiofisher", unexpected_backend_import)
    fc = api.load()

    assert fc.style == "perbin_A"
    assert fc.rf is None
    assert np.isfinite(fc.sigma_A(scenarios.clean(), 8_766.0))


def test_explicit_radiofisher_path_is_still_honored(monkeypatch):
    backend = object()
    requested = Path("/explicit/RadioFisher")

    def import_backend(rf_dir=None):
        assert Path(rf_dir) == requested
        return backend, requested

    monkeypatch.setattr(compat, "import_radiofisher", import_backend)
    fc = api.load(rf_dir=requested)
    assert fc.rf is backend
    assert fc.rf_dir == requested


@pytest.mark.parametrize("target", [0.0, -1.0, np.nan, np.inf, True])
def test_required_time_rejects_invalid_target_before_forecast(target):
    with pytest.raises(ValueError, match="target"):
        api.required_time(_NoForecastWork(), uniform=0.0, target=target)


@pytest.mark.parametrize("duty", [0.0, -1.0, np.nan, np.inf, True])
def test_required_time_rejects_invalid_duty_before_forecast(duty):
    with pytest.raises(ValueError, match="duty"):
        api.required_time(_NoForecastWork(), uniform=0.0, duty=duty)


@pytest.mark.parametrize("years", [-1.0, np.nan, np.inf, True])
def test_significance_rejects_invalid_years_before_forecast(years):
    with pytest.raises(ValueError, match="years"):
        api.significance(_NoForecastWork(), years, uniform=0.0)


@pytest.mark.parametrize("duty", [0.0, -1.0, np.nan, np.inf, True])
def test_significance_rejects_invalid_duty_before_forecast(duty):
    with pytest.raises(ValueError, match="duty"):
        api.significance(_NoForecastWork(), 1.0, uniform=0.0, duty=duty)


@pytest.mark.parametrize("target", [0.0, -1.0, np.nan, np.inf, True])
def test_tolerance_curve_validates_target_even_for_empty_curve(target):
    with pytest.raises(ValueError, match="target"):
        api.tolerance_curve(_NoForecastWork(), fracs=[], target=target)


@pytest.mark.parametrize("duty", [0.0, -1.0, np.nan, np.inf, True])
def test_tolerance_curve_validates_duty_even_for_empty_curve(duty):
    with pytest.raises(ValueError, match="duty"):
        api.tolerance_curve(_NoForecastWork(), fracs=[], duty=duty)


@pytest.mark.parametrize("target", [0.0, -1.0, np.nan, np.inf, True])
def test_threshold_curve_rejects_invalid_target_before_forecast(target):
    with pytest.raises(ValueError, match="target"):
        api.threshold_curve(_NoForecastWork(), {}, target=target)


@pytest.mark.parametrize("duty", [0.0, -1.0, np.nan, np.inf, True])
def test_threshold_curve_rejects_invalid_duty_before_forecast(duty):
    with pytest.raises(ValueError, match="duty"):
        api.threshold_curve(_NoForecastWork(), {}, duty=duty)


def test_significance_accepts_zero_years():
    class Forecast:
        def significance(self, scenario, hours, bins=None):
            assert hours == pytest.approx(0.0)
            return 0.0

    assert api.significance(Forecast(), 0.0, uniform=0.0) == 0.0

"""Validation tests for the high-level public API wrappers."""
from pathlib import Path

import numpy as np
import pytest

from baonoise import api, compat, scenarios
from baonoise.fisherbank import ARTIFACT_FORECAST


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


@pytest.mark.parametrize("cosmology", ["planck2018", "pact2025"])
def test_named_cosmology_routes_through_packaged_bank_registry(
        monkeypatch, cosmology):
    resource = object()
    loaded_bank = type("Bank", (), {"meta": {"config": "chime2022"}})()
    result = object()
    seen = {}

    def bank_file(name):
        seen["cosmology"] = name
        return resource

    def fisher_bank(source, *, expected_artifact_kind):
        seen["source"] = source
        seen["artifact_kind"] = expected_artifact_kind
        return loaded_bank

    def forecast(bank, rf, *, style, rf_dir):
        seen.update(bank=bank, rf=rf, style=style, rf_dir=rf_dir)
        return result

    monkeypatch.setattr(api, "bank_file", bank_file)
    monkeypatch.setattr(api, "FisherBank", fisher_bank)
    monkeypatch.setattr(api._forecast, "Forecast", forecast)

    assert api.load(cosmology=cosmology) is result
    assert seen == {
        "cosmology": cosmology,
        "source": resource,
        "artifact_kind": ARTIFACT_FORECAST,
        "bank": loaded_bank,
        "rf": None,
        "style": "perbin_A",
        "rf_dir": None,
    }


def test_api_rejects_explicit_and_named_bank_together(tmp_path):
    with pytest.raises(ValueError, match="either bank=.*cosmology"):
        api.load(tmp_path / "bank.npz", cosmology="pact2025")


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


def test_threshold_curve_preserves_orderable_string_labels():
    class Forecast:
        def required_hours_metric(self, metric, target):
            return 10.0 if metric(1.0) else 20.0

        def significance(self, scenario, hours, bins=None):
            return float(hours) + sum(scenario.residuals.values())

    points = {"loose": {30: (0.1, 0.3)},
              "strict": {30: (0.3, 0.1)}}
    result = api.threshold_curve(Forecast(), points)
    assert result["eta"].tolist() == ["loose", "strict"]
    assert result["best_eta"] in points


def test_scenario_passthrough_is_identity():
    sc = scenarios.uniform(0.3, scenarios.DTV_BAND)
    assert api.scenario_from(sc) is sc


def test_scenario_passthrough_refuses_overrides():
    """A Scenario carries its own policy; overrides beside it would be
    silently ignored, so they are refused instead."""
    sc = scenarios.uniform(0.3, scenarios.DTV_BAND)
    for kw in (dict(residual=0.5), dict(residuals={35: 0.5}),
               dict(excise_threshold=0.5), dict(mode="fourier"),
               dict(residual_excise_threshold=10.0)):
        with pytest.raises(ValueError, match="its own policy"):
            api.scenario_from(sc, **kw)
    with pytest.raises(ValueError, match="exactly one"):
        api.scenario_from(sc, uniform=0.5)


def test_required_time_accepts_a_scenario():
    fc = api.load()
    out = api.required_time(fc, mask=scenarios.clean(), target=5.0)
    assert np.isfinite(out["hours"])
    assert out["penalty_vs_clean"] == pytest.approx(1.0)

"""Safety checks for Fisher marginalisation and forecast inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baonoise.forecast import Forecast


class _Bank:
    def __init__(self, matrix, names):
        self.matrix = np.asarray(matrix, dtype=float)
        self.paramnames = list(names)
        self.zs = np.array([0.8, 0.9])
        self.nbins = 1
        self.artifact_kind = "forecast"

    def F(self, ibin, t_hours):
        assert ibin == 0
        return self.matrix * float(t_hours) ** 2


class _Scenario:
    def bin_factors_for_zbins(self, zs):
        assert len(zs) == 2
        return [(1.0, 1.0)]


class _CombinedFisher:
    def __init__(self, names):
        self.names = list(names)

    def combined_fisher_matrix(self, matrices, **_kwargs):
        return np.sum(matrices, axis=0), list(self.names)


SCENARIO = _Scenario()


def _forecast(matrix, names=("A", "sigma_NL"), style="perbin_A"):
    bank = _Bank(matrix, names)
    rf = _CombinedFisher(names)
    return Forecast(bank, rf, style=style)


def test_invalid_marginalisation_style_is_rejected():
    with pytest.raises(ValueError, match="style must be one of"):
        _forecast(np.eye(2), style="unsupported")


def test_perbin_forecast_does_not_require_radiofisher_backend():
    fc = Forecast(_Bank([[4.0, 0.0], [0.0, 1.0]], ["A", "sigma_NL"]),
                  style="perbin_A")
    assert fc.rf is None
    assert fc.sigma_A(SCENARIO, 1.0) == pytest.approx(0.5)


def test_shared_forecast_requires_radiofisher_backend():
    bank = _Bank(np.eye(2), ["A", "sigma_NL"])
    with pytest.raises(RuntimeError, match="shared_A.*RadioFisher"):
        Forecast(bank, style="shared_A")


def test_direct_forecast_reports_missing_radiofisher_clearly(monkeypatch):
    fc = Forecast(_Bank(np.eye(2), ["A", "sigma_NL"]), style="perbin_A")

    def missing_backend(_rf_dir=None):
        raise FileNotFoundError("not installed")

    monkeypatch.setattr("baonoise.compat.import_radiofisher", missing_backend)
    with pytest.raises(RuntimeError, match="direct Fisher.*RadioFisher"):
        fc.sigma_A_direct(SCENARIO, 1.0)


def test_direct_forecast_reuses_stored_radiofisher_path(monkeypatch):
    class LookupObserved(Exception):
        pass

    requested = Path("/nonstandard/RadioFisher")
    fc = Forecast(_Bank(np.eye(2), ["A", "sigma_NL"]), object(),
                  style="perbin_A", rf_dir=requested)

    def observe_lookup(explicit=None):
        assert Path(explicit) == requested
        raise LookupObserved

    monkeypatch.setattr("baonoise.compat.find_radiofisher_dir", observe_lookup)
    with pytest.raises(LookupObserved):
        fc.sigma_A_direct(SCENARIO, 1.0)


def test_direct_forecast_uses_bank_cosmology_and_matching_cache(monkeypatch):
    from baonoise.scenarios import clean

    class Backend:
        def background_evolution_splines(self, cosmo):
            assert cosmo == {"loaded": "pact"}
            return (object(),) * 4

        def fisher(self, _zlo, _zhi, cosmo, expt, _fns):
            assert cosmo == {"loaded": "pact"}
            assert expt["noise_freq_mode"] == "invvar"
            return np.eye(2), ["A", "sigma_NL"]

    bank = _Bank(np.eye(2), ["A", "sigma_NL"])
    bank.meta = {"config": "chime2022", "cosmology": "pact2025",
                 "astrophysical_model_profile": "chime_overview_2022",
                 "expt_overrides": {},
                 "provenance": {"experiment": {"settings": {}}}}
    backend = Backend()
    fc = Forecast(bank, backend, style="perbin_A", rf_dir=Path("/rf"))
    seen = {}
    monkeypatch.setattr("baonoise.compat.bind_radiofisher",
                        lambda rf, explicit=None: Path("/rf"))
    monkeypatch.setattr("baonoise.compat.require_backend_capabilities",
                        lambda *args, **kwargs: frozenset())

    def get_cosmology(name, rf, rf_dir):
        seen["name"] = name
        return {"requested": name}

    def load_cosmology(rf, cachefile, cosmo=None, **_kwargs):
        seen["cache"] = Path(cachefile).name
        seen["cosmo"] = cosmo
        return {"loaded": "pact"}

    monkeypatch.setattr("baonoise.cosmologies.get", get_cosmology)
    monkeypatch.setattr("baonoise.pkcache.load_fiducial_cosmology",
                        load_cosmology)
    monkeypatch.setattr("baonoise.survey.chime2022_experiment",
                        lambda rf, rf_dir, ttot_hours: {})
    assert fc.sigma_A_direct(clean(), 1.0) == pytest.approx(1.0)
    assert seen == {"name": "pact2025",
                    "cache": "cache_pk_chime2022_pact2025.dat",
                    "cosmo": {"requested": "pact2025"}}


def test_direct_custom_cosmology_applies_the_banks_recorded_profile(monkeypatch):
    from baonoise.scenarios import clean

    class Backend:
        @staticmethod
        def with_astrophysical_profile(cosmo, profile):
            assert profile == "bull2015"
            return dict(
                cosmo, astrophysical_model_profile=profile,
                Tb_model="powerlaw", bias_HI_model="powerlaw",
                omega_HI_model="powerlaw")

        @staticmethod
        def background_evolution_splines(cosmo):
            assert cosmo["mnu"] == pytest.approx(0.09304)
            return ("resolved-background",)

        @staticmethod
        def fisher(_zlo, _zhi, cosmo, _expt, cosmo_fns):
            assert cosmo_fns == ("resolved-background",)
            assert cosmo["astrophysical_model_profile"] == "bull2015"
            assert {cosmo[key] for key in
                    ("Tb_model", "bias_HI_model", "omega_HI_model")} \
                == {"powerlaw"}
            return np.eye(2), ["A", "sigma_NL"]

    bank = _Bank(np.eye(2), ["A", "sigma_NL"])
    bank.meta = {
        "config": "bull2015", "cosmology": "planck2013",
        "astrophysical_model_profile": "bull2015", "expt_overrides": {},
        "provenance": {"experiment": {"settings": {
            "epsilon_fg": 1e-6, "k_nl0": 0.14}}}}
    fc = Forecast(bank, Backend(), style="perbin_A", rf_dir=Path("/rf"))
    monkeypatch.setattr("baonoise.compat.bind_radiofisher",
                        lambda rf, explicit=None: Path("/rf"))
    monkeypatch.setattr("baonoise.compat.require_backend_capabilities",
                        lambda *args, **kwargs: frozenset())
    monkeypatch.setattr("baonoise.survey.chime_experiment",
                        lambda rf, rf_dir, ttot_hours, epsilon_fg, k_nl0: {
                            "epsilon_fg": epsilon_fg, "k_nl0": k_nl0})
    custom = {
        "h": 0.7, "omega_M_0": 0.3, "omega_b_0": 0.05,
        "mnu": 3.0, "ns": 0.96, "sigma_8": 0.8,
        "ombh2": 0.0245, "omch2": 0.1225, "omnuh2": 0.001}
    assert fc.sigma_A_direct(
        clean(), 1.0, cosmo=custom,
        cosmo_fns=("matching-background",)) == pytest.approx(1.0)


def test_direct_full_mask_skips_zero_weight_bins(monkeypatch):
    from baonoise.scenarios import DTV_BAND, uniform

    class Backend:
        @staticmethod
        def with_astrophysical_profile(cosmo, profile):
            return dict(
                cosmo, astrophysical_model_profile=profile,
                Tb_model="hall", bias_HI_model="castorina",
                omega_HI_model="crighton")

        @staticmethod
        def background_evolution_splines(_cosmo):
            return ("resolved-background",)

        @staticmethod
        def fisher(*_args, **_kwargs):
            raise AssertionError("zero-information bin reached the backend")

    bank = _Bank(np.eye(2), ["A", "sigma_NL"])
    bank.zs = np.array([1.4, 1.5])  # wholly inside the measured DTV band
    bank.meta = {
        "config": "chime2022", "cosmology": "planck2018",
        "astrophysical_model_profile": "chime_overview_2022"}
    fc = Forecast(bank, Backend(), style="perbin_A", rf_dir=Path("/rf"))
    monkeypatch.setattr("baonoise.compat.bind_radiofisher",
                        lambda rf, explicit=None: Path("/rf"))
    monkeypatch.setattr("baonoise.compat.require_backend_capabilities",
                        lambda *args, **kwargs: frozenset())
    custom = {
        "h": 0.7, "omega_M_0": 0.3, "omega_b_0": 0.05,
        "mnu": 0.0, "ns": 0.96, "sigma_8": 0.8}
    scenario = uniform(1.0, band=DTV_BAND)
    assert np.isinf(fc.sigma_A(scenario, 1.0))
    assert np.isinf(fc.sigma_A_direct(
        scenario, 1.0, cosmo=custom, cosmo_fns=("background",)))


def test_direct_forecast_fails_closed_on_missing_backend_capability(monkeypatch):
    bank = _Bank(np.eye(2), ["A", "sigma_NL"])
    bank.meta = {"config": "chime2022", "cosmology": "planck2018"}
    fc = Forecast(bank, object(), style="perbin_A", rf_dir=Path("/rf"))
    monkeypatch.setattr("baonoise.compat.bind_radiofisher",
                        lambda rf, explicit=None: Path("/rf"))

    def reject(*_args, **_kwargs):
        raise RuntimeError("lacks required capability(s): vol_frac")

    monkeypatch.setattr("baonoise.compat.require_backend_capabilities", reject)
    with pytest.raises(RuntimeError, match="vol_frac"):
        fc.sigma_A_direct(SCENARIO, 1.0)


@pytest.mark.parametrize("style", ["perbin_A", "shared_A"])
def test_degenerate_amplitude_is_unconstrained_not_pseudoinverse_finite(style):
    """A and its nuisance are identical, so A has no marginal constraint."""
    fc = _forecast([[1.0, 1.0], [1.0, 1.0]], style=style)
    assert np.isinf(fc.sigma_A(SCENARIO, 1.0))
    assert fc.significance(SCENARIO, 1.0) == 0.0


def test_numerically_ill_conditioned_amplitude_is_unconstrained():
    fc = _forecast([[1.0, 1.0], [1.0, 1.0 + 1e-14]])
    assert np.isinf(fc.sigma_A(SCENARIO, 1.0))


def test_constraint_orthogonal_to_a_nuisance_null_space_remains_finite():
    matrix = [[4.0, 0.0, 0.0],
              [0.0, 1.0, 1.0],
              [0.0, 1.0, 1.0]]
    fc = _forecast(matrix, names=("A", "sigma_NL", "aperp"))
    assert fc.sigma_A(SCENARIO, 1.0) == pytest.approx(0.5)


def test_full_rank_marginalisation_retains_inverse_result():
    fc = _forecast([[4.0, 1.0], [1.0, 3.0]])
    assert fc.sigma_A(SCENARIO, 1.0) == pytest.approx(np.sqrt(3.0 / 11.0))


def test_genuinely_zero_nuisance_row_is_dropped():
    fc = _forecast([[4.0, 0.0], [0.0, 0.0]])
    assert fc.sigma_A(SCENARIO, 1.0) == pytest.approx(0.5)


@pytest.mark.parametrize("matrix", [
    [[1.0, 0.0], [0.0, -1.0]],
    [[1.0, 1.0], [1.0, 0.0]],
])
def test_invalid_nonpositive_rows_are_not_mistaken_for_zero_information(matrix):
    fc = _forecast(matrix)
    assert np.isinf(fc.sigma_A(SCENARIO, 1.0))


@pytest.mark.parametrize("t_hours", [-1.0, np.nan, np.inf])
def test_forecast_time_must_be_nonnegative_and_finite(t_hours):
    fc = _forecast(np.eye(2))
    with pytest.raises(ValueError, match="t_hours"):
        fc.sigma_A(SCENARIO, t_hours)
    # Direct evaluation validates before attempting to import its backend.
    with pytest.raises(ValueError, match="t_hours"):
        fc.sigma_A_direct(SCENARIO, t_hours)


def test_zero_time_retains_zero_information_semantics():
    fc = _forecast(np.eye(2))
    assert np.isinf(fc.sigma_A(SCENARIO, 0.0))
    assert fc.significance(SCENARIO, 0.0) == 0.0
    assert np.isinf(fc.sigma_A_direct(SCENARIO, 0.0))


@pytest.mark.parametrize("target", [0.0, -1.0, np.nan, np.inf])
def test_detection_target_must_be_positive_and_finite(target):
    fc = _forecast(np.eye(2))
    with pytest.raises(ValueError, match="target"):
        fc.required_hours(SCENARIO, target=target)


@pytest.mark.parametrize("t_lo,t_hi", [
    (0.0, 10.0),
    (1.0, np.inf),
    (10.0, 10.0),
    (11.0, 10.0),
])
def test_root_bracket_must_be_positive_finite_and_ordered(t_lo, t_hi):
    fc = _forecast(np.eye(2))
    with pytest.raises(ValueError, match="t_lo|t_hi"):
        fc.required_hours(SCENARIO, target=1.0, t_lo=t_lo, t_hi=t_hi)
    with pytest.raises(ValueError, match="t_lo|t_hi"):
        fc.required_hours_metric(lambda t: t, 1.0, t_lo=t_lo, t_hi=t_hi)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_generic_metric_threshold_must_be_finite(threshold):
    fc = _forecast(np.eye(2))
    with pytest.raises(ValueError, match="threshold"):
        fc.required_hours_metric(lambda t: t, threshold)


def test_generic_metric_rejects_nan_but_allows_infinite_unreachable_values():
    fc = _forecast(np.eye(2))
    with pytest.raises(ValueError, match="metric_fn"):
        fc.required_hours_metric(lambda _t: np.nan, 1.0)
    assert np.isinf(fc.required_hours_metric(lambda _t: -np.inf, 1.0))


@pytest.mark.parametrize("duty", [0.0, -0.5, np.nan, np.inf])
def test_calendar_duty_must_be_positive_and_finite(duty):
    fc = _forecast(np.eye(2))
    with pytest.raises(ValueError, match="duty"):
        fc.required_years(SCENARIO, target=1.0, duty=duty)


def test_valid_required_time_root_is_unchanged():
    fc = _forecast([[1.0]], names=("A",))
    assert fc.required_hours(SCENARIO, target=5.0, t_lo=1.0, t_hi=10.0) \
        == pytest.approx(5.0, rel=2e-4)


def test_satisfied_lower_endpoint_wins_even_if_upper_endpoint_fails():
    """Do not call an already-reached target unreachable due to later decline."""
    fc = _forecast([[1.0]], names=("A",))

    def nonmonotonic_endpoint_metric(t_hours):
        return 2.0 - np.log10(t_hours / 10.0)

    fc.significance = lambda _scenario, t, bins=None: \
        nonmonotonic_endpoint_metric(t)
    assert fc.required_hours(SCENARIO, target=1.0) == 10.0
    assert fc.required_hours_metric(nonmonotonic_endpoint_metric, 1.0) == 10.0


def test_significance_curve_rejects_invalid_times():
    fc = _forecast(np.eye(2))
    with pytest.raises(ValueError, match="t_hours"):
        fc.significance_curve(SCENARIO, np.array([1.0, -1.0]))
    assert np.array_equal(fc.significance_curve(SCENARIO, np.array([0.0])),
                          np.array([0.0]))

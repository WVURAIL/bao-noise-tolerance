"""Safety checks for Fisher marginalisation and forecast inputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baonoise.forecast import Forecast


class _Bank:
    def __init__(self, matrix, names):
        self.matrix = np.asarray(matrix, dtype=float)
        self.paramnames = list(names)
        self.zs = np.array([0.8, 0.9])
        self.nbins = 1

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
        _forecast(np.eye(2), style="legacy")


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

"""The floor is a property of the weight bank, and these tests pin that down.

Every survey product reports a sensitivity floor by reading the shelf level of
frames the detector kept. Those frames are the interval 1 < F <= mu0, because
the mask fires on F > mu0 while the level estimator is only defined above
F = 1. The interval is narrow when mu0 > 1 and empty when mu0 < 1, and neither
fact has anything to do with the sky.

The tests below build products where the truth is known by construction, so
the claim can be checked rather than argued: a channel whose transmitter is off
for most of the archive still reports no floor if mu0 < 1, and a channel whose
transmitter never stops still reports one if mu0 > 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baonoise import residual                             # noqa: E402

BELOW_DB = 11.3
ENBW_HZ = 3051.7578125
DTV_HZ = 6.0e6
OFFSET_DB = BELOW_DB - 10.0 * np.log10(DTV_HZ / ENBW_HZ)     # -21.636 dB


def _write(path, F, mu0, channel=99, shelf=None, rejected=None):
    """A product carrying only what floor_provenance reads."""
    F = np.asarray(F, dtype=float)
    n = F.size
    if shelf is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            shelf = 10.0 * np.log10(np.where(F > 1.0, F - 1.0, np.nan)) + OFFSET_DB
    if rejected is None:
        rejected = F > mu0
    np.savez(
        path,
        valid=np.ones((n, 1), dtype=np.uint8),
        reject_mask=np.asarray(rejected).reshape(n, 1).astype(np.uint8),
        fstat_raw=F.reshape(n, 1),
        snr_shelf_db=np.asarray(shelf, dtype=float).reshape(n, 1),
        mu0=np.array([mu0]),
        physical_channel=np.array([channel], dtype=np.int32),
        freq_id=np.array([521], dtype=np.int64),
        pilot_below_data_db=np.array(BELOW_DB),
        bin_enbw_hz=np.array(ENBW_HZ),
        dtv_bandwidth_hz=np.array(DTV_HZ),
    )
    return path


def _channel(path, mu0, sigma, n_off, n_on, excess=0.5, seed=0):
    """n_off frames of pure null plus n_on frames carrying a real excess."""
    rng = np.random.default_rng(seed)
    off = mu0 + sigma * rng.standard_normal(n_off)
    on = mu0 + excess * rng.gamma(2.0, 1.0, n_on)
    return _write(path, np.concatenate([off, on]), mu0)


# ----------------------------------------------------------------------
# mu0 > 1: a floor exists, and it is the weight bank talking
# ----------------------------------------------------------------------

def test_floor_tracks_mu0_not_the_sky(tmp_path):
    """Two channels, same mu0, wildly different transmitters, same floor."""
    quiet = _channel(tmp_path / "quiet.npz", 1.004, 0.02, 20_000, 2_000, seed=1)
    busy = _channel(tmp_path / "busy.npz", 1.004, 0.02, 200, 40_000, seed=2)

    pq = residual.floor_provenance(quiet)
    pb = residual.floor_provenance(busy)

    predicted = 10.0 * np.log10(0.004) + OFFSET_DB
    for p in (pq, pb):
        assert p.n_sliver > 0
        # The sliver's upper edge is exactly the predicted value, so a high
        # percentile of it lands just below and can never land above.
        assert p.reported_db <= p.mu0_implied_db + 1e-9
        assert p.reported_db == pytest.approx(predicted, abs=1.5)
        assert p.mu0_implied_db == pytest.approx(predicted, rel=1e-12)
        assert p.mu0_determined
        assert "mu0-determined" in p.verdict

    # A hundredfold difference in how much quiet time the archive holds moves
    # the reported floor by about a decibel. It is not measuring the sky.
    assert pq.n_sliver > 50 * pb.n_sliver
    assert abs(pq.reported_db - pb.reported_db) < 1.5


def test_floor_moves_when_the_weight_bank_moves(tmp_path):
    """Change mu0 alone and the floor follows it exactly, in dB."""
    a = residual.floor_provenance(
        _channel(tmp_path / "a.npz", 1.004, 0.02, 20_000, 2_000, seed=3))
    b = residual.floor_provenance(
        _channel(tmp_path / "b.npz", 1.040, 0.02, 20_000, 2_000, seed=3))

    assert b.mu0_implied_db - a.mu0_implied_db == pytest.approx(10.0, abs=1e-9)
    assert b.reported_db - a.reported_db == pytest.approx(10.0, abs=1.0)


# ----------------------------------------------------------------------
# mu0 < 1: no floor is possible, however quiet the channel was
# ----------------------------------------------------------------------

def test_mu0_below_one_leaves_no_floor_even_when_mostly_quiet(tmp_path):
    """The interval 1 < F <= mu0 is empty for every dataset when mu0 < 1."""
    p = residual.floor_provenance(
        _channel(tmp_path / "sub.npz", 0.997, 0.004, 30_000, 500, seed=4))

    assert p.n_kept > 10_000                     # plenty of quiet frames
    assert p.n_sliver == 0                       # none of them can set a floor
    assert not np.isfinite(p.reported_db)
    assert "mu0 < 1" in p.verdict
    assert not p.mu0_determined

    # And the frames the mask fired on nearest the threshold carry no level
    # either, which is why the gap cannot be closed from the detected side.
    assert p.n_masked_without_level > 0


def test_null_scale_survives_where_the_floor_does_not(tmp_path):
    """The measurable alternative: the null's own scatter, mu0 either side."""
    for mu0 in (0.997, 1.004):
        p = residual.floor_provenance(
            _channel(tmp_path / f"s{mu0}.npz", mu0, 0.01, 40_000, 1_000, seed=5))
        assert p.sigma_null == pytest.approx(0.01, rel=0.15)
        assert p.sigma_spread < 1.3              # clean, well-sampled left tail
        assert p.sigma_implied_db == pytest.approx(
            10.0 * np.log10(0.01 / mu0) + OFFSET_DB, abs=1.0)


def test_null_scale_flags_a_left_tail_that_is_not_the_null(tmp_path):
    """The spread catches a non-Gaussian tail, which is the failure that bites.

    A small clean sample still reads consistently; what breaks the estimate is
    a left tail with something else in it: reference bins carrying an
    adjacent allocation's power, which drags F down without being null
    scatter. Probing at three widely separated quantiles turns that into a
    disagreement instead of a confident wrong number.
    """
    clean = residual.floor_provenance(
        _channel(tmp_path / "clean.npz", 1.004, 0.01, 400, 40_000, seed=6))
    assert clean.sigma_spread < 1.5

    rng = np.random.default_rng(7)
    null = 1.004 + 0.01 * rng.standard_normal(400)
    leak = 1.004 - 0.20 * rng.gamma(2.0, 1.0, 60)      # adjacent-channel drag
    on = 1.004 + 0.5 * rng.gamma(2.0, 1.0, 40_000)
    dirty = residual.floor_provenance(
        _write(tmp_path / "dirty.npz", np.concatenate([null, leak, on]), 1.004))
    assert dirty.sigma_spread > 1.5


# ----------------------------------------------------------------------
# The function refuses when its premises do not hold
# ----------------------------------------------------------------------

def test_refuses_a_product_whose_mask_is_not_the_deployed_rule(tmp_path):
    F = np.linspace(0.9, 1.5, 500)
    path = _write(tmp_path / "odd.npz", F, 1.004, rejected=F > 1.2)
    with pytest.raises(ValueError, match="not the rule"):
        residual.floor_provenance(path)


def test_refuses_a_product_whose_level_is_not_the_documented_formula(tmp_path):
    F = np.linspace(0.9, 1.5, 500)
    with np.errstate(invalid="ignore", divide="ignore"):
        bogus = np.where(F > 1.0, 10.0 * np.log10(F - 1.0) - 3.0, np.nan)
    path = _write(tmp_path / "bogus.npz", F, 1.004, shelf=bogus)
    with pytest.raises(ValueError, match="not 10log10"):
        residual.floor_provenance(path)


def test_null_scale_returns_nan_rather_than_a_negative_width():
    """A kept sample entirely above mu0 is not a null; say so, do not invent."""
    sigma, spread = residual.null_scale(np.full(100, 2.0), 1.004)
    assert not np.isfinite(sigma) and not np.isfinite(spread)

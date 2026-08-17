"""Residual contamination: the second half of the tolerance cost.

Two things this suite has to establish. First, that adding the term changes
nothing when it is absent; every published number in ``out/`` predates it and
must survive byte-for-byte. Second, that with it present the forecast stops
being monotone in the masked fraction, which is the whole point: only then is a
detector threshold something you can optimise rather than assume.
"""
import numpy as np
import pytest

from baonoise import residual, scenarios


# ----------------------------------------------------------------------
# Backward compatibility: no residuals -> identical behavior
# ----------------------------------------------------------------------

def test_absent_residuals_reproduce_masking_only():
    """The regression gate for every number already in out/."""
    sc = scenarios.measured()
    assert sc.residuals == {}
    for nu_lo, nu_hi in [(470.0, 500.0), (566.6, 591.4), (400.0, 470.0),
                         (560.0, 610.0), (700.0, 800.0)]:
        v, w = sc.bin_factors(nu_lo, nu_hi)
        # recompute the pre-residual formula directly
        width = nu_hi - nu_lo
        excised = sum(
            max(0.0, min(nu_hi, scenarios.chn.channel_edges(c)[1])
                - max(nu_lo, scenarios.chn.channel_edges(c)[0]))
            for c, f in sc.fractions.items() if f >= sc.excise_threshold)
        assert v == pytest.approx((width - excised) / width)


def test_zero_residual_is_exactly_identity():
    no_excision = scenarios.NO_EXCISION_THRESHOLD
    a = scenarios.Scenario("a", "a", fractions={30: 0.2},
                           excise_threshold=no_excision)
    b = scenarios.Scenario("b", "b", fractions={30: 0.2},
                           excise_threshold=no_excision,
                           residuals={30: 0.0})
    assert a.bin_factors(566.0, 592.0) == b.bin_factors(566.0, 592.0)
    assert a.keep_weight(30) == pytest.approx(0.8)


def test_uniform_constructor_defaults_to_no_residual():
    assert scenarios.uniform(0.5, scenarios.DTV_BAND).frequency_residuals == {}
    assert scenarios.uniform(
        0.5, scenarios.DTV_BAND,
        residual=0.1).frequency_residuals[scenarios.DTV_BAND] == 0.1


# ----------------------------------------------------------------------
# The coupling itself
# ----------------------------------------------------------------------

def test_residual_acts_as_effective_time_loss():
    """r raises noise by (1 + r), i.e. costs time by the same factor."""
    sc = scenarios.Scenario("t", "t", fractions={30: 0.0},
                            residuals={30: 1.0},
                            excise_threshold=scenarios.NO_EXCISION_THRESHOLD)
    assert sc.keep_weight(30) == pytest.approx(0.5)
    # a bin entirely inside ch30
    _, w = sc.bin_factors(567.0, 571.0)
    assert w == pytest.approx(0.5)


def test_residual_and_masking_multiply():
    sc = scenarios.Scenario("t", "t", fractions={30: 0.5},
                            residuals={30: 0.25},
                            excise_threshold=scenarios.NO_EXCISION_THRESHOLD)
    assert sc.keep_weight(30) == pytest.approx(0.5 / 1.25)


def test_negative_residual_is_rejected():
    with pytest.raises(ValueError):
        scenarios.Scenario(
            "t", "t", fractions={30: 0.1}, residuals={30: -0.1})


def test_residual_can_force_excision():
    """A quiet channel with a large residual is the dangerous case."""
    sc = scenarios.Scenario("t", "t", fractions={30: 0.01},
                            residuals={30: 9.0},
                            excise_threshold=0.5,
                            residual_excise_threshold=1.0)
    assert sc.is_excised(30)
    v, w = sc.bin_factors(566.0, 592.0)
    assert v < 1.0 and w == pytest.approx(1.0)


def test_residual_only_channel_is_seen():
    """A channel listed only in residuals still enters the band average."""
    sc = scenarios.Scenario("t", "t", residuals={30: 1.0},
                            excise_threshold=scenarios.NO_EXCISION_THRESHOLD)
    _, w = sc.bin_factors(567.0, 571.0)
    assert w == pytest.approx(0.5)


def test_fourier_mode_folds_residual_too():
    sc = scenarios.Scenario("t", "t", fractions={30: 0.5}, residuals={30: 1.0},
                            excise_threshold=scenarios.NO_EXCISION_THRESHOLD,
                            mode="fourier")
    width, ov = 592.0 - 566.0, 6.0
    _, w = sc.bin_factors(566.0, 592.0)
    expected = width / ((width - ov) + ov / 0.25)
    assert w == pytest.approx(expected)


def test_hook_stays_consistent_with_bin_factors():
    """Forecast.sigma_A_direct validates the bank path through this hook."""
    sc = scenarios.measured(residuals={17: 0.4, 31: 0.15})
    w = sc.freq_weight_fn()
    nu_lo, nu_hi = 566.6, 591.4
    nn = np.linspace(nu_lo, nu_hi, 200001)
    ww = w(nn)
    surviving = np.isfinite(ww)
    v_frac, w_bar = sc.bin_factors(nu_lo, nu_hi)
    assert surviving.mean() == pytest.approx(v_frac, abs=2e-4)
    assert np.mean(ww[surviving]) == pytest.approx(w_bar, abs=2e-4)


def test_at_threshold_constructor():
    sc = scenarios.at_threshold({30: (0.3, 0.2), 17: (0.1, 0.05)}, eta=1.23)
    assert sc.fractions == {30: 0.3, 17: 0.1}
    assert sc.residuals == {30: 0.2, 17: 0.05}
    assert "1.23" in sc.label


# ----------------------------------------------------------------------
# The budget
# ----------------------------------------------------------------------

def test_budget_chain_is_additive_in_db():
    b = residual.ResidualBudget(shelf_floor_db=-26.0, ground_filter_db=14.3,
                                delay_filter_db=3.6)
    assert b.suppressed_db == pytest.approx(-43.9)
    assert b.ratio == pytest.approx(10 ** (-4.39))
    assert b.ratio_db == pytest.approx(-43.9)


def test_coherence_amplifies_but_does_not_change_the_chain():
    base = residual.ResidualBudget(shelf_floor_db=-40.0, ground_filter_db=14.3,
                                   delay_filter_db=3.6)
    amp = residual.ResidualBudget(shelf_floor_db=-40.0, ground_filter_db=14.3,
                                  delay_filter_db=3.6, n_coh=1000.0)
    assert amp.suppressed_db == base.suppressed_db
    assert amp.ratio_db == pytest.approx(base.ratio_db + 30.0)
    assert amp.ratio == pytest.approx(base.ratio * 1000.0)


def test_n_coh_must_be_positive():
    with pytest.raises(ValueError):
        residual.ResidualBudget(shelf_floor_db=-40.0, n_coh=0.0).ratio


def _ch35_stats(**kw):
    """The measured 521.npz split, as ShelfStatistics."""
    base = dict(channel=35, freq_id=521, nu_mhz=596.48, n_valid=39775,
                n_kept=6484, on_shelf_db=-10.54, floor_db=-26.18,
                floor_percentile=90.0, dc_fraction=0.944396,
                interday_fraction=0.045157, intraday_fraction=0.008603,
                fast_fraction=0.001844, n_off_frames=3167,
                n_units=3852, n_days=1361)
    base.update(kw)
    return residual.ShelfStatistics(**base)


def test_delay_key_is_validated_and_ordered():
    d = residual.DELAY_SUPPRESSION_DB
    # a shorter delay cutoff (larger protected scale) buys less suppression
    assert d["bao_peak1"] < d["bao_peak2"] < d["aggressive_200ns"]
    with pytest.raises(ValueError):
        residual.budget_from_statistics(_ch35_stats(), delay_key="nope")


def test_ground_filter_uses_the_sidereal_day_boundary():
    """Inter-day drift is m = 0 too: the common-mode filter runs per day."""
    stats = _ch35_stats()
    assert stats.surviving_fraction == pytest.approx(0.010447, abs=1e-6)
    assert stats.ground_filter_db == pytest.approx(19.81, abs=0.05)
    # splitting at the acquisition instead lumps inter-day drift in with
    # intra-day variation and understates the filter by ~7 dB
    naive = 10 * np.log10(1.0 / stats.slow_fraction)
    assert naive == pytest.approx(12.70, abs=0.05)
    assert stats.ground_filter_db - naive == pytest.approx(7.11, abs=0.05)


def test_components_do_not_double_count_ground_filtered_power():
    """tau_c above a sidereal day is already priced as m = 0 suppression."""
    stats = _ch35_stats()
    capped = residual.budget_from_statistics(
        stats, "bao_peak1", tau_intraday=residual.MAX_TAU_C_SECONDS)
    beyond = residual.budget_from_statistics(
        stats, "bao_peak1", tau_intraday=100 * residual.MAX_TAU_C_SECONDS)
    assert beyond.ratio == pytest.approx(capped.ratio)


def test_fast_component_keeps_its_own_coherence():
    """One lumped n_coh would amplify the fast population by the slow factor."""
    stats = _ch35_stats()
    b = residual.budget_from_statistics(stats, "bao_peak1", tau_intraday=3600.0)
    n_slow = residual.n_coh_from_correlation_time(3600.0)
    lumped = (stats.intraday_fraction + stats.fast_fraction) * n_slow
    assert b._gain < lumped
    assert b._gain == pytest.approx(
        stats.intraday_fraction * n_slow + stats.fast_fraction * 1.0)


def test_budget_matches_the_measured_chain_at_one_hour():
    b = residual.budget_from_statistics(_ch35_stats(), "bao_peak1",
                                        tau_intraday=3600.0)
    assert b.ratio_db == pytest.approx(-1.09, abs=0.05)
    assert b.delay_filter_db == pytest.approx(3.6)


def test_component_fractions_are_validated():
    with pytest.raises(ValueError):
        residual.ResidualBudget(shelf_floor_db=-26.0, components=((-0.1, 1.0),))
    with pytest.raises(ValueError):
        residual.ResidualBudget(shelf_floor_db=-26.0, components=((0.1, 0.0),))


def test_n_coh_from_correlation_time():
    n = residual.n_coh_from_correlation_time(3600.0)
    assert n == pytest.approx(3600.0 / residual.CHIME_FRAME_SECONDS, rel=1e-9)
    assert 10 * np.log10(n) == pytest.approx(49.3, abs=0.1)
    # never below 1: a residual cannot average down faster than thermal noise
    assert residual.n_coh_from_correlation_time(1e-6) == 1.0
    # never above one sidereal day: longer is m = 0 and already removed
    assert residual.n_coh_from_correlation_time(1e9) == pytest.approx(
        residual.MAX_TAU_C_SECONDS / residual.CHIME_FRAME_SECONDS)
    with pytest.raises(ValueError):
        residual.n_coh_from_correlation_time(1.0, frame_seconds=0.0)


def test_chain_and_summary_render():
    stats = _ch35_stats()
    txt = stats.summary()
    assert "ch 35" in txt or "ch35" in txt
    assert "ground filter" in txt and "inter-day" in txt and "intra-day" in txt
    chain = residual.budget_from_statistics(stats, "bao_peak1",
                                            tau_intraday=3600.0).chain()
    assert "delay filter" in chain and "r = P_res/P_N" in chain


# ----------------------------------------------------------------------
# The correlation-time estimator
# ----------------------------------------------------------------------

SID = 86164.0905


def _write_product(path, shelf_db, unit_of_frame, unit_t0, channel=35,
                   rejected=None):
    """Minimal survey product with the keys the residual module reads."""
    n = shelf_db.size
    if rejected is None:
        rejected = (shelf_db > np.percentile(shelf_db, 10)).astype(np.uint8)
    np.savez(
        path,
        valid=np.ones((n, 1), dtype=np.uint8),
        reject_mask=rejected.reshape(n, 1).astype(np.uint8),
        snr_shelf_db=shelf_db.reshape(n, 1),
        frame_unit_index=unit_of_frame.astype(np.int32),
        unit_time0_ctime=unit_t0,
        physical_channel=np.array([channel], dtype=np.int32),
        freq_id=np.array([521], dtype=np.int64),
        chime_frequency_hz=np.array([596.48e6]),
    )
    return path


def _epoch_product(tmp_path, name, on_first: bool):
    """One frame per unit: an on epoch at -10 dB and an off epoch at -40 dB.

    ``on_first=True`` is a sign-off channel (on through 2020-12, off from
    2021-01); ``on_first=False`` is the time-mirrored sign-on channel.
    """
    import datetime as _dt
    rng = np.random.default_rng(7)
    n = 300
    on_shelf = -10.0 + 0.1 * rng.standard_normal(n)
    off_shelf = -40.0 + 0.1 * rng.standard_normal(n)
    t_on = _dt.datetime(2020, 6, 1, tzinfo=_dt.timezone.utc).timestamp() \
        + np.arange(n) * 3600.0
    t_off = _dt.datetime(2021, 6, 1, tzinfo=_dt.timezone.utc).timestamp() \
        + np.arange(n) * 3600.0
    if on_first:
        shelf = np.concatenate([on_shelf, off_shelf])
        t0 = np.concatenate([t_on, t_off])
        rejected = np.r_[np.ones(n), np.zeros(n)].astype(np.uint8)
    else:
        shelf = np.concatenate([off_shelf, on_shelf])
        t0 = np.concatenate([t_off - 2 * 366 * 86400.0, t_on])
        rejected = np.r_[np.zeros(n), np.ones(n)].astype(np.uint8)
    return _write_product(tmp_path / name, shelf,
                          np.arange(2 * n), t0, rejected=rejected)


def test_off_from_mirrors_off_through(tmp_path):
    """A sign-off channel calibrates exactly like its time-mirrored sign-on."""
    signoff = _epoch_product(tmp_path, "signoff.npz", on_first=True)
    signon = _epoch_product(tmp_path, "signon.npz", on_first=False)
    a = residual.shelf_statistics(signoff, off_from="2021-01")
    b = residual.shelf_statistics(signon, off_through="2020-05")
    assert a.floor_db == pytest.approx(b.floor_db, abs=0.05)
    assert a.floor_db == pytest.approx(-40.0, abs=0.5)
    assert a.on_shelf_db == pytest.approx(-10.0, abs=0.5)
    # without the epoch, the off sample is empty (every on frame is rejected,
    # every off frame kept) and the floor comes from the kept frames instead
    c = residual.shelf_statistics(signoff)
    assert c.floor_db == pytest.approx(-40.0, abs=0.5)


def test_off_epoch_specs_are_mutually_exclusive(tmp_path):
    p = _epoch_product(tmp_path, "signoff.npz", on_first=True)
    with pytest.raises(ValueError, match="at most one"):
        residual.shelf_statistics(p, off_through="2020-12", off_from="2021-01")
    with pytest.raises(ValueError, match="at most one"):
        residual.correlation_time(p, off_through="2020-12", off_from="2021-01")


def test_threshold_sweep_accepts_off_from(tmp_path):
    import datetime as _dt
    rng = np.random.default_rng(11)
    units, fpu = 100, 3        # per epoch: 100 acquisitions x 3 frames
    n = units * fpu
    shelf = np.concatenate([-10.0 + 0.1 * rng.standard_normal(n),
                            -40.0 + 0.1 * rng.standard_normal(n)])
    # on epoch: 60 strong units (F=50) over 40 weak ones (F=2)
    F = np.concatenate([np.full(60 * fpu, 50.0), np.full(40 * fpu, 2.0),
                        np.full(n, 1.0)])
    rejected = np.r_[np.ones(n), np.zeros(n)].astype(np.uint8)
    uof = np.concatenate([np.repeat(np.arange(units), fpu),
                          np.repeat(np.arange(units, 2 * units), fpu)])
    t0 = np.concatenate([
        _dt.datetime(2020, 6, 1, tzinfo=_dt.timezone.utc).timestamp()
        + np.arange(units) * 3600.0,
        _dt.datetime(2021, 6, 1, tzinfo=_dt.timezone.utc).timestamp()
        + np.arange(units) * 3600.0])
    np.savez(
        tmp_path / "sweep_signoff.npz",
        valid=np.ones((2 * n, 1), dtype=np.uint8),
        reject_mask=rejected.reshape(-1, 1),
        snr_shelf_db=shelf.reshape(-1, 1),
        fstat_raw=F.reshape(-1, 1),
        mu0=np.array([1.0]),
        frame_unit_index=uof.astype(np.int32),
        unit_time0_ctime=t0,
        physical_channel=np.array([19], dtype=np.int32),
        freq_id=np.array([767], dtype=np.int64),
        chime_frequency_hz=np.array([500.39e6]),
    )
    sweep = residual.threshold_sweep(tmp_path / "sweep_signoff.npz",
                                     off_from="2021-01")
    assert sweep, "sweep must not be empty with an off_from epoch"
    assert all(np.isfinite(row["net"]) for row in sweep)
    # thresholds between the two on-epoch F populations mask the strong 60%
    mid = [r for r in sweep if 2.5 <= r["eta"] <= 40.0]
    assert mid and all(r["f"] == pytest.approx(0.6, abs=0.01) for r in mid)
    # thresholds above every F keep everything
    assert sweep[-1]["f"] == pytest.approx(0.0, abs=1e-6)


def _stationary_product(tmp_path, tau_true, n_days=400, per_day=6,
                        frames_per_unit=6, seed=1):
    """Shelf = constant + slow day offset + AR(1) intra-day term + noise."""
    rng = np.random.default_rng(seed)
    shelf, uof, t0 = [], [], []
    u = 0
    for d in range(n_days):
        day_start = d * SID + 3000.0
        day_off = 0.02 * rng.standard_normal()
        # acquisitions spread over ~5 h, AR(1) correlated with time constant tau
        times = np.sort(rng.uniform(0.0, 5 * 3600.0, per_day))
        x, prev, prev_t = [], rng.standard_normal(), times[0]
        for tt in times:
            a = np.exp(-(tt - prev_t) / tau_true)
            prev = a * prev + np.sqrt(max(1 - a * a, 0.0)) * rng.standard_normal()
            prev_t = tt
            x.append(prev)
        for tt, xi in zip(times, x):
            t0.append(day_start + tt)
            lin = 1.0 * (1.0 + day_off + 0.06 * xi)
            frames = lin * (1.0 + 0.004 * rng.standard_normal(frames_per_unit))
            shelf.append(10.0 * np.log10(np.maximum(frames, 1e-12)))
            uof.append(np.full(frames_per_unit, u))
            u += 1
    return _write_product(tmp_path / "stationary.npz",
                          np.concatenate(shelf), np.concatenate(uof),
                          np.array(t0))


def _episodic_product(tmp_path, n_days=400, per_day=6, frames_per_unit=6,
                      seed=2):
    """Quiet baseline punctuated by rare strong bursts, ch34/ch36's shape."""
    rng = np.random.default_rng(seed)
    shelf, uof, t0 = [], [], []
    u = 0
    for d in range(n_days):
        day_start = d * SID + 3000.0
        times = np.sort(rng.uniform(0.0, 5 * 3600.0, per_day))
        for tt in times:
            t0.append(day_start + tt)
            lin = 1e-4 * (1.0 + 0.3 * rng.standard_normal())
            if rng.random() < 0.25:                 # burst, broad amplitude
                lin *= 10.0 ** rng.uniform(0.5, 4.5)
            frames = np.abs(lin * (1.0 + 0.1 * rng.standard_normal(frames_per_unit)))
            shelf.append(10.0 * np.log10(np.maximum(frames, 1e-18)))
            uof.append(np.full(frames_per_unit, u))
            u += 1
    return _write_product(tmp_path / "episodic.npz",
                          np.concatenate(shelf), np.concatenate(uof),
                          np.array(t0), channel=34)


def test_correlation_time_recovers_a_known_timescale(tmp_path):
    path = _stationary_product(tmp_path, tau_true=2400.0)
    ct = residual.correlation_time(path, n_boot=60)
    assert ct.is_measured, ct.reason
    assert 1200.0 < ct.tau_c < 4800.0, ct.tau_c
    assert ct.tau_lo <= ct.tau_c <= ct.tau_hi
    assert ct.trim_spread < 2.0 and ct.surviving_spread < 2.0


def test_correlation_time_orders_two_known_timescales(tmp_path):
    slow = residual.correlation_time(
        _stationary_product(tmp_path / "a", tau_true=3600.0, seed=11)
        if (tmp_path / "a").mkdir() or True else None, n_boot=40)
    fast = residual.correlation_time(
        _stationary_product(tmp_path / "b", tau_true=600.0, seed=12)
        if (tmp_path / "b").mkdir() or True else None, n_boot=40)
    assert slow.is_measured and fast.is_measured
    assert slow.tau_c > fast.tau_c


def test_correlation_time_refuses_an_episodic_shelf(tmp_path):
    ct = residual.correlation_time(_episodic_product(tmp_path), n_boot=40)
    assert not ct.is_measured
    assert any(k in ct.reason for k in ("episodic", "artefact", "cut",
                                       "unresolved"))
    # and the refusal hands back the conservative cap rather than a guess
    assert ct.tau_for_budget == residual.MAX_TAU_C_SECONDS
    assert np.isnan(ct.tau_c)


def test_refusal_takes_no_ground_filter_credit(tmp_path):
    """Non-stationarity breaks the split as well as the timescale."""
    b, _, ct = residual.budget_from_products(_episodic_product(tmp_path),
                                             n_boot=40)
    assert not ct.is_measured and not b.tau_measured
    assert sum(f for f, _ in b.components) == pytest.approx(1.0)
    assert "BOUND" in b.chain()


def test_measured_path_sets_the_flag_and_uses_the_measurement(tmp_path):
    path = _stationary_product(tmp_path, tau_true=2400.0)
    b, st, ct = residual.budget_from_products(path, n_boot=60)
    assert ct.is_measured and b.tau_measured
    assert "BOUND" not in b.chain()
    expected = residual.budget_from_statistics(
        st, residual.DEFAULT_DELAY_KEY, tau_intraday=ct.tau_c)
    assert b.ratio == pytest.approx(expected.ratio)


def test_noise_correction_stops_sparse_units_faking_a_short_tau(tmp_path):
    """Unit-mean noise inflates D at every lag; uncorrected it looks fast."""
    path = _stationary_product(tmp_path, tau_true=2400.0, frames_per_unit=2,
                               seed=7)
    ct = residual.correlation_time(path, n_boot=40)
    assert ct.is_measured, ct.reason
    assert ct.tau_c > 900.0, "noise correction failed: tau collapsed"


def test_bootstrap_must_preserve_within_day_ordering(tmp_path):
    """A day-block resample keeps each day's sequence; the interval is finite."""
    ct = residual.correlation_time(
        _stationary_product(tmp_path, tau_true=2400.0), n_boot=80)
    assert np.isfinite(ct.tau_lo) and np.isfinite(ct.tau_hi)
    assert ct.tau_hi > ct.tau_lo


def _fast_stationary_product(tmp_path, seed=21):
    """Stationary shelf that decorrelates faster than the acquisition cadence.

    The two failure modes of the estimator point opposite ways, and this is the
    favourable one: a short tau_c is an upper bound worth ~24 dB against the
    sidereal-day cap, so it must not be discarded like an episodic shelf.
    """
    rng = np.random.default_rng(seed)
    shelf, uof, t0 = [], [], []
    u = 0
    for d in range(400):
        day_start = d * SID + 3000.0
        day_off = 0.02 * rng.standard_normal()
        times = np.sort(rng.uniform(0.0, 5 * 3600.0, 6))
        for tt in times:                       # independent between acquisitions
            t0.append(day_start + tt)
            lin = 1.0 * (1.0 + day_off + 0.06 * rng.standard_normal())
            frames = lin * (1.0 + 0.004 * rng.standard_normal(6))
            shelf.append(10.0 * np.log10(np.maximum(frames, 1e-12)))
            uof.append(np.full(6, u))
            u += 1
    return _write_product(tmp_path / "fast.npz", np.concatenate(shelf),
                          np.concatenate(uof), np.array(t0))


def test_fast_stationary_shelf_is_bounded_not_refused(tmp_path):
    ct = residual.correlation_time(_fast_stationary_product(tmp_path), n_boot=40)
    assert ct.quality == "bounded_above", (ct.quality, ct.reason)
    assert ct.is_usable and not ct.is_measured
    assert ct.tau_c == pytest.approx(residual.STRUCTURE_LAG_EDGES[1])
    # and the bound is worth using: far below the cap it would otherwise get
    assert ct.tau_for_budget < residual.MAX_TAU_C_SECONDS / 100


def test_bounded_above_budget_uses_the_bound_and_flags_it(tmp_path):
    b, _, ct = residual.budget_from_products(_fast_stationary_product(tmp_path),
                                             n_boot=40)
    assert ct.quality == "bounded_above"
    assert not b.tau_measured and "BOUND" in b.chain()
    # the ground-filter split is still trusted; the shelf is stationary
    assert sum(f for f, _ in b.components) < 1.0
    capped = residual.budget_from_statistics(
        _, residual.DEFAULT_DELAY_KEY, tau_intraday=residual.MAX_TAU_C_SECONDS)
    assert b.ratio < capped.ratio / 100


def test_stationarity_is_checked_before_resolution(tmp_path):
    """An episodic shelf must refuse, never claim a favourable short bound."""
    ct = residual.correlation_time(_episodic_product(tmp_path), n_boot=40)
    assert ct.quality == "refused"
    assert ct.surviving_spread > 2.0 or ct.trim_spread > 2.0
    assert ct.tau_for_budget == residual.MAX_TAU_C_SECONDS


# ----------------------------------------------------------------------
# Is masking worth it?
# ----------------------------------------------------------------------

def test_mask_benefit_weighs_noise_against_data():
    """Masking pays only when the contamination removed beats the time lost."""
    # 10x cleaner for 50% of the data: pays
    d = residual.mask_benefit(35, f=0.5, r_unmasked=20.0, r_masked=1.0)
    assert d.noise_gain == pytest.approx(21.0 / 2.0)
    assert d.data_cost == pytest.approx(2.0)
    assert d.net == pytest.approx(5.25) and d.should_mask
    # the same cleaning for 99% of the data: does not
    d2 = residual.mask_benefit(35, f=0.99, r_unmasked=20.0, r_masked=1.0)
    assert d2.net < 1.0 and not d2.should_mask


def test_break_even_fraction():
    d = residual.mask_benefit(35, f=0.0, r_unmasked=20.0, r_masked=1.0)
    assert d.break_even_f == pytest.approx(1.0 - 2.0 / 21.0)
    at = residual.mask_benefit(35, f=d.break_even_f, r_unmasked=20.0, r_masked=1.0)
    assert at.net == pytest.approx(1.0)
    # nothing to remove -> no masked fraction is worth paying
    assert residual.mask_benefit(35, 0.5, 1.0, 1.0).break_even_f == 0.0


def test_masking_a_clean_channel_never_pays():
    """A transmitter already below the detection floor gains nothing."""
    d = residual.mask_benefit(34, f=0.99, r_unmasked=1e-5, r_masked=5e-6)
    assert not d.should_mask and d.net < 0.02


def test_mask_benefit_validates_inputs():
    with pytest.raises(ValueError):
        residual.mask_benefit(35, f=1.5, r_unmasked=1.0, r_masked=0.1)
    with pytest.raises(ValueError):
        residual.mask_benefit(35, f=0.5, r_unmasked=-1.0, r_masked=0.1)


def test_total_masking_is_infinitely_expensive():
    d = residual.mask_benefit(35, f=1.0, r_unmasked=1e6, r_masked=0.0)
    assert not np.isfinite(d.data_cost) and d.net == 0.0


def test_selective_scenario_keeps_declined_channels_contaminated(tmp_path):
    """A channel the mask declines must carry its full contamination."""
    yes = residual.mask_benefit(35, f=0.5, r_unmasked=20.0, r_masked=1.0)
    no = residual.mask_benefit(34, f=0.99, r_unmasked=0.01, r_masked=0.005)
    sc = scenarios.from_mask_decisions(
        [yes, no], excise_threshold=scenarios.NO_EXCISION_THRESHOLD)
    assert sc.fractions[35] == pytest.approx(0.5)
    assert sc.residuals[35] == pytest.approx(1.0)
    assert sc.fractions[34] == 0.0                  # not masked
    assert sc.residuals[34] == pytest.approx(0.01)  # but still contaminated
    forced = scenarios.from_mask_decisions(
        [yes, no], excise_threshold=scenarios.NO_EXCISION_THRESHOLD,
        force=True)
    assert forced.fractions[34] == pytest.approx(0.99)
    assert forced.keep_weight(34) < sc.keep_weight(34)   # forcing costs time


# ----------------------------------------------------------------------
# The four-way policy comparison
# ----------------------------------------------------------------------

def test_policy_time_penalty_counts_both_costs():
    """Contamination and lost frames both cost integration time."""
    clean = residual.Policy("clean", f=0.0, r=0.0)
    assert clean.time_penalty == 1.0
    # a residual at the noise level doubles the time
    assert residual.Policy("dirty", f=0.0, r=1.0).time_penalty == 2.0
    # discarding half the frames doubles it too
    assert residual.Policy("half", f=0.5, r=0.0).time_penalty == 2.0
    # excision is not a slow measurement, it is no measurement
    assert np.isinf(residual.Policy("gone", f=1.0, r=0.0, kind="excise").time_penalty)


def test_policy_validates_inputs():
    with pytest.raises(ValueError):
        residual.Policy("bad f", f=1.5, r=0.0)
    with pytest.raises(ValueError):
        residual.Policy("bad r", f=0.0, r=-1.0)
    with pytest.raises(ValueError):
        residual.Policy("bad kind", f=0.0, r=0.0, kind="wishful")


def test_loose_tolerance_keeps_everything():
    """A residual that only adds noise is cheaper to integrate through."""
    c = residual.compare_policies(
        35, keep_r=0.057,
        proxy=("pilot proxy", 0.826, 2.5e-8),
        incumbents=[("MAD", 0.056, 0.0568), ("SK", 0.203, 0.0541)])
    assert c.best().kind == "keep"
    assert not c.saves_the_band
    # the detector's mask is not what wins here, and the code should say so
    assert "measurement of r" in c.verdict()


def test_tight_tolerance_makes_the_detector_the_only_option():
    c = residual.compare_policies(
        35, keep_r=0.057,
        proxy=("pilot proxy", 0.826, 2.5e-8),
        incumbents=[("MAD", 0.056, 0.0568), ("SK", 0.203, 0.0541)],
        bias_tolerance=1e-4)
    assert c.best().kind == "proxy"
    assert c.best_without_proxy().kind == "excise"
    assert c.saves_the_band
    assert np.isinf(c.proxy_advantage)


def test_tolerance_below_the_detector_forces_excision():
    c = residual.compare_policies(
        35, keep_r=0.057,
        proxy=("pilot proxy", 0.826, 2.5e-8),
        incumbents=[("MAD", 0.056, 0.0568)],
        bias_tolerance=1e-12)
    assert c.best().kind == "excise"
    assert not c.saves_the_band
    assert c.proxy_advantage == 1.0


def test_tolerance_map_partitions_the_whole_range():
    c = residual.compare_policies(
        35, keep_r=0.057,
        proxy=("pilot proxy", 0.826, 2.5e-8),
        incumbents=[("MAD", 0.056, 0.0568), ("SK", 0.203, 0.0541)])
    spans = c.tolerance_map()
    assert spans[0][0] == 0.0
    assert not np.isinf(spans[-1][2].time_penalty)        # loosest keeps data
    for (lo, hi), (lo2, _) in zip([(a, b) for a, b, _ in spans],
                                  [(a, b) for a, b, _ in spans][1:]):
        assert hi == lo2                                  # no gaps
    assert spans[0][2].kind == "excise"                   # tightest excises
    assert spans[-1][2].kind == "keep"                    # loosest keeps
    # and every span's winner is feasible at its own lower edge
    for lo, _, p in spans:
        assert p.kind == "excise" or p.r <= lo or lo == 0.0


def test_proxy_decades_measures_the_detector_only_window():
    c = residual.compare_policies(
        35, keep_r=0.057,
        proxy=("pilot proxy", 0.826, 2.5e-8),
        incumbents=[("SK", 0.203, 0.0541)])
    assert c.proxy_decades == pytest.approx(np.log10(0.0541 / 2.5e-8), rel=1e-6)
    # an incumbent that reaches the same residual leaves no window
    tied = residual.compare_policies(
        35, keep_r=0.057,
        proxy=("pilot proxy", 0.826, 0.0541),
        incumbents=[("SK", 0.203, 0.0541)])
    assert tied.proxy_decades == 0.0


def test_comparison_rejects_duplicate_names():
    with pytest.raises(ValueError):
        residual.PolicyComparison(35, (residual.Policy("a", 0.0, 1.0),
                                       residual.Policy("a", 0.5, 0.1)))


def test_masked_residual_refuses_without_a_measured_floor():
    """The bug that turned a failing channel into a passing one.

    ch35 has no null population, so its kept-frame floor is unmeasured. The
    code used to substitute the minimum *detected* shelf, the weakest
    positive excess in 31,607 detections, -76 dB against measured floors of
    -45 to -49 dB elsewhere. That is a sampling artefact that falls without
    limit as the detection count grows, and it made the masked residual four
    orders of magnitude too small.
    """
    stats = residual.ShelfStatistics(
        channel=35, freq_id=521, nu_mhz=596.484, n_valid=39775, n_kept=6484,
        on_shelf_db=-10.69, floor_db=float("nan"), floor_percentile=90.0,
        dc_fraction=0.775, interday_fraction=0.215,
        intraday_fraction=0.0099, fast_fraction=0.00018, n_off_frames=0)
    with pytest.raises(residual.NoMeasuredFloor) as exc:
        residual.masked_residual(stats, gain=636.4)
    assert "ch35" in str(exc.value)
    assert "0 frames" in str(exc.value) or "unmeasured" in str(exc.value)


def test_masked_residual_computes_with_a_measured_floor():
    stats = residual.ShelfStatistics(
        channel=33, freq_id=552, nu_mhz=584.375, n_valid=34206, n_kept=6659,
        on_shelf_db=-33.75, floor_db=-44.95, floor_percentile=90.0,
        dc_fraction=0.405, interday_fraction=0.423,
        intraday_fraction=0.157, fast_fraction=0.0152, n_off_frames=1388)
    r = residual.masked_residual(stats, gain=1123.0)
    assert r == pytest.approx(10 ** (-4.495) * 1123.0, rel=1e-9)


def test_an_explicit_substituted_floor_is_allowed_but_must_be_passed():
    """Substitution is permitted; it just cannot happen silently."""
    stats = residual.ShelfStatistics(
        channel=35, freq_id=521, nu_mhz=596.484, n_valid=1, n_kept=1,
        on_shelf_db=-10.69, floor_db=float("nan"), floor_percentile=90.0,
        dc_fraction=0.775, interday_fraction=0.215,
        intraday_fraction=0.0099, fast_fraction=0.00018, n_off_frames=0)
    r = residual.masked_residual(stats, gain=642.7, floor_db=-45.59)
    assert r == pytest.approx(10 ** (-4.559) * 642.7, rel=1e-9)


def test_threshold_sweep_refuses_without_floor_but_accepts_a_stated_one(tmp_path):
    """A mu0 < 1 product yields no sweep unless the caller states the bound.

    This is the same refusal discipline as masked_residual(): the sweep must
    not invent a sensitivity floor, but it must accept one stated explicitly,
    because the null-scatter bound of floor_provenance() is exactly the
    defensible substitute on such channels.
    """
    rng = np.random.default_rng(21)
    mu0 = 0.997
    F = np.concatenate([mu0 + 0.004 * rng.standard_normal(3000),
                        mu0 + 0.5 * rng.gamma(2.0, 1.0, 3000)])
    n = F.size
    with np.errstate(invalid="ignore", divide="ignore"):
        shelf = 10.0 * np.log10(np.where(F > 1.0, F - 1.0, np.nan)) - 21.636
    uof = np.repeat(np.arange(n // 6), 6)[:n]
    t0 = 3000.0 + np.arange(n // 6 + 1) * 900.0
    np.savez(tmp_path / "sub.npz",
             valid=np.ones((n, 1), np.uint8),
             reject_mask=(F > mu0).reshape(n, 1).astype(np.uint8),
             fstat_raw=F.reshape(n, 1),
             snr_shelf_db=shelf.reshape(n, 1),
             mu0=np.array([mu0]),
             frame_unit_index=uof.astype(np.int32),
             unit_time0_ctime=t0,
             physical_channel=np.array([35], np.int32),
             freq_id=np.array([521], np.int64),
             chime_frequency_hz=np.array([596.48e6]))
    path = tmp_path / "sub.npz"

    assert residual.threshold_sweep(path) == []          # no invented floor

    sweep = residual.threshold_sweep(path, floor_db=-45.0)
    assert len(sweep) > 3
    rows = sorted(sweep, key=lambda r: r["eta"])
    # raising the threshold keeps more frames, monotonically
    fs = [row["f"] for row in rows]
    assert all(a >= b - 1e-12 for a, b in zip(fs, fs[1:]))
    # and across the full family the loosest threshold keeps far more
    # contamination than the tightest. (r need not be locally monotone just
    # above F = 1, where newly kept frames carry measured levels *below* the
    # stated floor bound and briefly pull the kept-set mean down.)
    assert rows[-1]["r_masked"] > 5 * rows[0]["r_masked"]

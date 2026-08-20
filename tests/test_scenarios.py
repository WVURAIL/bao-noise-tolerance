import numpy as np
import pytest

from baonoise import channels as chn
from baonoise import scenarios


def test_channel_edges():
    assert chn.channel_edges(14) == (470.0, 476.0)
    assert chn.channel_edges(30) == (566.0, 572.0)
    zlo, zhi = chn.channel_z_range(30)
    assert 1.48 < zlo < 1.49 and 1.50 < zhi < 1.52


def test_clean_scenario():
    sc = scenarios.clean()
    v, w = sc.bin_factors(500.0, 530.0)
    assert v == 1.0 and w == 1.0


def test_uniform_time_mode():
    sc = scenarios.uniform(0.5, scenarios.DTV_BAND)
    # bin fully inside the DTV band
    v, w = sc.bin_factors(500.0, 530.0)
    assert v == pytest.approx(1.0)
    assert w == pytest.approx(0.5)
    # fourier convention agrees for uniform masking
    sc_f = scenarios.uniform(0.5, scenarios.DTV_BAND, mode="fourier")
    v2, w2 = sc_f.bin_factors(500.0, 530.0)
    assert w2 == pytest.approx(0.5)


def test_uniform_outside_dtv_band():
    sc = scenarios.uniform(0.5, scenarios.DTV_BAND)
    v, w = sc.bin_factors(410.0, 440.0)   # below 470 MHz: clean
    assert v == 1.0 and w == 1.0


def test_excision_costs_volume_not_noise():
    sc = scenarios.Scenario("t", "t", fractions={30: 0.97},
                            excise_threshold=0.5)
    # 24.8 MHz bin containing ch30's 566-572 slice
    v, w = sc.bin_factors(566.6, 591.4)
    assert v == pytest.approx((591.4 - 566.6 - 5.4) / (591.4 - 566.6))
    assert w == pytest.approx(1.0)


def test_kept_channel_costs_time():
    sc = scenarios.Scenario("t", "t", fractions={30: 0.97},
                            excise_threshold=scenarios.NO_EXCISION_THRESHOLD)
    width = 591.4 - 566.6
    ov = 572.0 - 566.6
    v, w = sc.bin_factors(566.6, 591.4)
    assert v == pytest.approx(1.0)
    expected = ((width - ov) + ov * 0.03) / width
    assert w == pytest.approx(expected)
    # fourier convention is much harsher
    sc_f = scenarios.Scenario("t", "t", fractions={30: 0.97},
                              excise_threshold=scenarios.NO_EXCISION_THRESHOLD,
                              mode="fourier")
    _, wf = sc_f.bin_factors(566.6, 591.4)
    assert wf == pytest.approx(width / ((width - ov) + ov / 0.03))
    assert wf < 0.2


def test_measured_fractions_load():
    fr = chn.measured_mask_fractions()
    assert fr[30] == pytest.approx(0.97)
    assert fr[24] == pytest.approx(0.97)
    assert 0.005 < fr[36] < 0.02
    assert 0.25 < fr[17] < 0.45
    assert len(fr) == 23


def test_zbin_factor_array_shape():
    zs = np.array([0.78, 0.88, 0.98])
    fac = scenarios.measured().bin_factors_for_zbins(zs)
    assert fac.shape == (2, 2)


def test_freq_weight_fn_matches_radiofisher_hook_semantics():
    sc = scenarios.measured()
    w = sc.freq_weight_fn()
    # outside any DTV channel: clean
    assert w(450.0)[0] == 1.0 and w(700.0)[0] == 1.0
    # inside refused ch30 (566-572): excised -> NaN
    assert np.isnan(w(569.0)[0])
    # inside a kept channel: 1 - f
    f36 = sc.fractions[36]
    lo, _ = scenarios.chn.channel_edges(36)
    assert w(lo + 3.0)[0] == pytest.approx(1.0 - f36)
    # vectorised call preserves shape
    nu = np.linspace(400.0, 800.0, 1001)
    assert w(nu).shape == nu.shape
    assert sc.rf_mode() == "invvar"
    assert scenarios.measured(mode="fourier").rf_mode() == "fourier"


def test_hook_band_average_equals_bin_factors():
    """The uniform-sampled band average RadioFisher's hook computes must
    reproduce Scenario.bin_factors' piecewise-exact w_bar."""
    sc = scenarios.measured()
    w = sc.freq_weight_fn()
    nu_lo, nu_hi = 566.6, 591.4        # representative ch30-overlap bin
    nn = np.linspace(nu_lo, nu_hi, 200001)
    ww = w(nn)
    surviving = np.isfinite(ww)
    v_frac, w_bar = sc.bin_factors(nu_lo, nu_hi)
    assert surviving.mean() == pytest.approx(v_frac, abs=2e-4)
    assert np.mean(ww[surviving]) == pytest.approx(w_bar, abs=2e-4)


def test_api_scenario_from():
    from baonoise import api
    sc = api.scenario_from(mask={30: 0.97, 17: 0.33})
    v, w = sc.bin_factors(566.6, 591.4)
    assert v < 1.0 and w == 1.0      # ch30 excised, ch17 outside this bin
    sc2 = api.scenario_from(uniform=0.5)
    assert sc2.excise_threshold == scenarios.NO_EXCISION_THRESHOLD
    assert sc2.bin_factors(500.0, 530.0)[1] == pytest.approx(0.5)
    # An explicit threshold is applied rather than silently ignored.
    sc3 = api.scenario_from(uniform=0.5, excise_threshold=0.5)
    assert sc3.bin_factors(500.0, 530.0) == pytest.approx((0.0, 1.0))
    with pytest.raises(ValueError):
        api.scenario_from()


@pytest.mark.parametrize("fractions", [
    {30: -0.01}, {30: 1.01}, {30: np.nan}, {30: np.inf},
])
def test_scenario_rejects_invalid_masked_fractions(fractions):
    with pytest.raises(ValueError, match=r"fractions\[30\].*\[0, 1\]"):
        scenarios.Scenario("bad", "bad", fractions=fractions)


@pytest.mark.parametrize("residuals", [
    {30: -0.01}, {30: np.nan}, {30: np.inf},
])
def test_scenario_rejects_invalid_residuals(residuals):
    with pytest.raises(ValueError, match=r"residuals\[30\].*non-negative"):
        scenarios.Scenario("bad", "bad", residuals=residuals)


@pytest.mark.parametrize("mode", ["invvar", "Time", "", None, []])
def test_scenario_rejects_unknown_modes(mode):
    with pytest.raises(ValueError, match="mode must be one of"):
        scenarios.Scenario("bad", "bad", mode=mode)


@pytest.mark.parametrize("threshold", [-0.01, np.nan, -np.inf])
def test_scenario_rejects_invalid_mask_excision_thresholds(threshold):
    with pytest.raises(ValueError, match="excise_threshold"):
        scenarios.Scenario("bad", "bad", excise_threshold=threshold)


def test_threshold_above_one_disables_excision():
    sc = scenarios.Scenario(
        "kept", "kept", fractions={30: 1.0}, excise_threshold=1.01)
    assert not sc.is_excised(30)
    assert sc.excise_threshold == pytest.approx(1.01)


@pytest.mark.parametrize("threshold", [-0.01, np.nan, -np.inf])
def test_scenario_rejects_invalid_residual_excision_thresholds(threshold):
    with pytest.raises(ValueError, match="residual_excise_threshold"):
        scenarios.Scenario(
            "bad", "bad", residual_excise_threshold=threshold)


def test_scenario_copies_and_normalises_inputs():
    fractions = {np.int64(30): np.float64(0.25)}
    residuals = {np.int64(30): np.float64(0.5)}
    sc = scenarios.Scenario(
        "ok", "ok", fractions=fractions, residuals=residuals)
    fractions[30] = 0.75
    residuals[30] = 1.0
    assert sc.fractions == {30: 0.25}
    assert sc.residuals == {30: 0.5}


def test_api_uniform_residual_mapping_is_validated_at_construction():
    from baonoise import api

    with pytest.raises(ValueError, match="channel-specific"):
        api.scenario_from(uniform=0.2, residuals={30: np.nan})


@pytest.mark.parametrize("residual", [-0.1, np.nan, np.inf, True])
def test_api_rejects_invalid_uniform_residual(residual):
    from baonoise import api

    with pytest.raises(ValueError, match="residual"):
        api.scenario_from(uniform=0.2, residual=residual)


def test_uniform_excision_policies_are_explicit():
    kept = scenarios.uniform(0.8)
    thresholded = scenarios.uniform(0.8, excise_threshold=0.5)
    assert not kept.is_band_excised(scenarios.DTV_BAND)
    assert thresholded.is_band_excised(scenarios.DTV_BAND)
    with pytest.raises(TypeError, match="unexpected keyword argument 'excise'"):
        scenarios.uniform(0.8, excise=True)


def test_full_chime_band_is_a_physical_frequency_interval():
    sc = scenarios.uniform(0.5, scenarios.CHIME_BAND)
    assert sc.fractions == {}
    assert sc.frequency_fractions == {scenarios.CHIME_BAND: 0.5}
    assert sc.bin_factors(400.0, 800.0) == pytest.approx((1.0, 0.5))
    weight = sc.freq_weight_fn()
    assert weight(399.0)[0] == pytest.approx(1.0)
    assert weight(500.0)[0] == pytest.approx(0.5)


def test_uniform_rejects_string_band_aliases():
    with pytest.raises(ValueError, match="FrequencyBand"):
        scenarios.uniform(0.5, band="dtv")


def test_uniform_label_reflects_residual_excision_policy():
    sc = scenarios.uniform(
        0.2, residual=2.0, residual_excise_threshold=1.0)
    assert sc.is_band_excised(scenarios.DTV_BAND)
    assert "excised" in sc.label
    assert "r=2" in sc.label


def test_frequency_band_rejects_invalid_edges():
    with pytest.raises(ValueError, match="frequency-band edges"):
        scenarios.FrequencyBand("bad", 800.0, 400.0)


def test_channel_and_frequency_band_masks_must_not_overlap():
    with pytest.raises(ValueError, match="must not overlap"):
        scenarios.Scenario(
            "bad", "bad", fractions={30: 0.2},
            frequency_fractions={scenarios.DTV_BAND: 0.2})


# ----------------------------------------------------------------------
# Mask-table provenance
# ----------------------------------------------------------------------

def _product(path, channel, masked, n=400, kernel="aaaa", pkg="pp/1.0.0",
             months=None, rejected=None, fstat=None, mu0=1.0):
    """Minimal survey product carrying a detector contract.

    ``months`` (one ``YYYY-MM`` per frame) adds unit timestamps so the
    product can be windowed; ``rejected`` overrides the fraction-derived
    reject mask frame by frame; ``fstat`` adds the stored coarse statistic
    so the product can be rethresholded at eta != 1.
    """
    import datetime as dt
    import json
    if rejected is None:
        rej = np.zeros(n, dtype=np.uint8)
        rej[: int(round(masked * n))] = 1
    else:
        rej = np.asarray(rejected, dtype=np.uint8)
        n = rej.size
    arrays = dict(
        valid=np.ones((n, 1), dtype=np.uint8),
        reject_mask=rej.reshape(n, 1),
        physical_channel=np.array([channel], dtype=np.int32),
        detector_version=np.array(f"{pkg} kernel=2.1.0 kernel_sha256={kernel}"),
        detector_contract_json=np.array(json.dumps(
            {"equivalent_mask_rule": "F > mu0", "threshold_mode": "none"})),
    )
    if months is not None:
        assert len(months) == n
        t0 = np.array([
            dt.datetime.strptime(m + "-15", "%Y-%m-%d")
            .replace(tzinfo=dt.timezone.utc).timestamp() for m in months])
        arrays["unit_time0_ctime"] = t0
        arrays["frame_unit_index"] = np.arange(n, dtype=np.int32)
    if fstat is not None:
        assert len(fstat) == n
        arrays["fstat_raw"] = np.asarray(fstat, dtype=float).reshape(n, 1)
        arrays["mu0"] = np.array([mu0])
    np.savez(path, **arrays)
    return path


def test_product_table_carries_its_rule(tmp_path):
    t = chn.mask_table_from_products([_product(tmp_path / "a.npz", 35, 0.25)])
    assert t.is_traceable and t.rule == "F > mu0"
    assert t.fractions[35] == pytest.approx(0.25)
    assert t.n_frames[35] == 400
    # refused channels are assumed, and say so
    assert t.fractions[30] == pytest.approx(chn.REFUSED_FRACTION)
    assert any("ch30" in n for n in t.notes)


def test_csv_table_carries_legacy_epoch_provenance():
    t = chn.measured_mask_table()
    # the rule is now identified (traceable) but it is the mistuned legacy
    # epoch, so the table is not an occupancy measurement
    assert t.is_traceable and t.rule == chn.LEGACY_CSV_RULE
    assert not t.is_occupancy_measurement and t.epoch == chn.LEGACY_CSV_EPOCH
    assert any("legacy fs/2-mistuned" in n for n in t.notes)
    assert "NOT an occupancy measurement" in t.summary()


def test_measured_scenario_from_csv_warns_about_legacy_epoch():
    with pytest.warns(UserWarning, match="legacy fs/2-mistuned"):
        scenarios.measured()


def test_mixed_kernels_are_refused(tmp_path):
    ps = [_product(tmp_path / "a.npz", 35, 0.2, kernel="aaaa"),
          _product(tmp_path / "b.npz", 34, 0.9, kernel="bbbb")]
    with pytest.raises(ValueError, match="kernels"):
        chn.mask_table_from_products(ps)
    t = chn.mask_table_from_products(ps, require_same_detector=False)
    assert set(t.fractions) >= {34, 35}


def test_harness_drift_over_one_kernel_is_a_note_not_an_error(tmp_path):
    """The kernel decides the frames; the packaging around it does not."""
    t = chn.mask_table_from_products([
        _product(tmp_path / "a.npz", 35, 0.2, pkg="pp/0.3.0.dev0"),
        _product(tmp_path / "b.npz", 34, 0.9, pkg="pp/1.0.0")])
    assert t.fractions[34] == pytest.approx(0.9)
    assert any("harness versions" in n for n in t.notes)


def test_duplicate_channel_is_refused(tmp_path):
    """Two files for one channel: the second silently won before this gate."""
    ps = [_product(tmp_path / "a.npz", 35, 0.2),
          _product(tmp_path / "b.npz", 35, 0.9)]
    with pytest.raises(ValueError, match="two products cover ch35"):
        chn.mask_table_from_products(ps)


def test_measured_refuses_to_mix_sources_by_default(tmp_path):
    p = [_product(tmp_path / "a.npz", 35, 0.837)]
    with pytest.raises(ValueError, match="mixes two detectors"):
        scenarios.measured(products=p)
    omitted = scenarios.measured(products=p, fill_missing="omit")
    assert 17 not in omitted.fractions
    assert omitted.fractions[35] == pytest.approx(0.837, abs=2e-3)
    mixed = scenarios.measured(products=p, fill_missing="csv")
    assert mixed.fractions[35] == pytest.approx(0.837, abs=2e-3)
    assert 17 in mixed.fractions and "CSV" in mixed.label


def test_window_selects_the_epoch(tmp_path):
    """A sign-off channel: masked through 2019, clean in 2025."""
    months = ["2019-06"] * 200 + ["2025-06"] * 200
    rej = np.concatenate([np.ones(200), np.zeros(200)])
    p = _product(tmp_path / "w.npz", 35, 0.5, months=months, rejected=rej)
    full = chn.mask_table_from_products([p])
    assert full.fractions[35] == pytest.approx(0.5)
    assert full.window == "full span"
    late = chn.mask_table_from_products([p], since="2025-01")
    assert late.fractions[35] == pytest.approx(0.0)
    assert late.n_frames[35] == 200
    assert late.window == "2025-01..end"
    assert "2025-01" in late.summary()
    early = chn.mask_table_from_products([p], until="2019-12")
    assert early.fractions[35] == pytest.approx(1.0)


def test_window_with_no_frames_omits_and_notes(tmp_path):
    dead = _product(tmp_path / "dead.npz", 35, 1.0, months=["2019-06"] * 100,
                    rejected=np.ones(100))
    live = _product(tmp_path / "live.npz", 34, 0.1, months=["2025-06"] * 100,
                    rejected=np.r_[np.ones(10), np.zeros(90)])
    t = chn.mask_table_from_products([dead, live], since="2025-01")
    assert 35 not in t.n_frames
    assert t.fractions[34] == pytest.approx(0.1)
    assert any("ch35 has no valid frames" in n for n in t.notes)
    # a table that is empty inside the window refuses, and says which window
    with pytest.raises(ValueError, match="2025-01"):
        chn.mask_table_from_products([dead], since="2025-01")


def test_window_requires_timestamps(tmp_path):
    p = _product(tmp_path / "a.npz", 35, 0.25)
    with pytest.raises(ValueError, match="unit timestamps"):
        chn.mask_table_from_products([p], since="2025-01")


def test_windowed_scenario_carries_the_window(tmp_path):
    months = ["2019-06"] * 100 + ["2025-06"] * 100
    rej = np.concatenate([np.ones(100), np.zeros(100)])
    p = _product(tmp_path / "w.npz", 35, 0.5, months=months, rejected=rej)
    sc = scenarios.measured(products=[p], fill_missing="omit", since="2025-01")
    assert sc.fractions[35] == pytest.approx(0.0)
    assert "2025-01" in sc.label
    with pytest.raises(ValueError, match="require products"):
        scenarios.measured(since="2025-01")
    with pytest.raises(ValueError, match="two decisions"):
        scenarios.measured(products=[p], fill_missing="csv", since="2025-01")


def test_eta_rethresholds_from_the_stored_statistic(tmp_path):
    """A sign-off channel at eta=1 stays masked on faint residual excess;
    a threshold above the residual population frees it."""
    F = np.r_[np.full(100, 50.0),    # transmitter-on frames
              np.full(300, 1.02)]    # faint residual excess, F barely > mu0
    rej = (F > 1.0).astype(np.uint8)
    p = _product(tmp_path / "e.npz", 19, 1.0, rejected=rej, fstat=F)
    deployed = chn.mask_table_from_products([p])
    assert deployed.fractions[19] == pytest.approx(1.0)
    thresholded = chn.mask_table_from_products([p], eta=1.4)
    assert thresholded.fractions[19] == pytest.approx(0.25)
    assert "eta=1.4" in thresholded.source
    assert "rethresholded" in thresholded.rule
    assert thresholded.is_traceable


def test_eta_requires_the_statistic_and_coarse_stage(tmp_path):
    bare = _product(tmp_path / "bare.npz", 19, 0.5)
    with pytest.raises(ValueError, match="fstat_raw"):
        chn.mask_table_from_products([bare], eta=1.4)
    with pytest.raises(ValueError, match="stage"):
        chn.mask_table_from_products([bare], eta=1.4, stage="fine")
    with pytest.raises(ValueError, match="positive"):
        chn.mask_table_from_products([bare], eta=0.0)


def test_eta_scenario_carries_the_threshold(tmp_path):
    F = np.r_[np.full(100, 50.0), np.full(300, 1.02)]
    p = _product(tmp_path / "e.npz", 19, 1.0,
                 rejected=(F > 1.0).astype(np.uint8), fstat=F)
    sc = scenarios.measured(products=[p], fill_missing="omit", eta=1.4)
    assert sc.fractions[19] == pytest.approx(0.25)
    assert "eta=1.4" in sc.label
    with pytest.raises(ValueError, match="require products"):
        scenarios.measured(eta=1.4)
    with pytest.raises(ValueError, match="two decisions"):
        scenarios.measured(products=[p], fill_missing="csv", eta=1.4)


def test_compare_mask_tables_orders_by_disagreement():
    a = chn.MaskTable({14: 0.01, 35: 0.10}, "csv", "unrecorded")
    b = chn.MaskTable({14: 0.02, 35: 0.90}, "products", "F > mu0")
    rows = chn.compare_mask_tables(a, b)
    assert rows[0][0] == 35 and rows[0][3] == pytest.approx(9.0)
    assert rows[1][0] == 14 and rows[1][3] == pytest.approx(2.0)

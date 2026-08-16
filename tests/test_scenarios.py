import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    sc = scenarios.uniform(0.5, "dtv")
    # bin fully inside the DTV band
    v, w = sc.bin_factors(500.0, 530.0)
    assert v == pytest.approx(1.0)
    assert w == pytest.approx(0.5)
    # fourier convention agrees for uniform masking
    sc_f = scenarios.uniform(0.5, "dtv", mode="fourier")
    v2, w2 = sc_f.bin_factors(500.0, 530.0)
    assert w2 == pytest.approx(0.5)


def test_uniform_outside_dtv_band():
    sc = scenarios.uniform(0.5, "dtv")
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
    nu_lo, nu_hi = 566.6, 591.4        # the z=1.40-1.51 bin
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


def test_threshold_above_one_compatibly_disables_excision():
    """Existing callers using the old 1.01 idiom retain their behavior."""
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

    with pytest.raises(ValueError, match=r"residuals\[30\]"):
        api.scenario_from(uniform=0.2, residuals={30: np.nan})


@pytest.mark.parametrize("residual", [-0.1, np.nan, np.inf, True])
def test_api_rejects_invalid_uniform_residual(residual):
    from baonoise import api

    with pytest.raises(ValueError, match="residual"):
        api.scenario_from(uniform=0.2, residual=residual)


def test_uniform_excision_policies_are_explicit():
    kept = scenarios.uniform(0.8)
    thresholded = scenarios.uniform(0.8, excise_threshold=0.5)
    forced = scenarios.uniform(0.8, excise=True)
    assert not kept.is_excised(30)
    assert thresholded.is_excised(30)
    assert forced.is_excised(30)
    with pytest.raises(ValueError, match="either excise=True"):
        scenarios.uniform(0.8, excise=True, excise_threshold=0.5)


def test_uniform_label_reflects_residual_excision_policy():
    sc = scenarios.uniform(
        0.2, residual=2.0, residual_excise_threshold=1.0)
    assert sc.is_excised(30)
    assert "excised" in sc.label
    assert "r=2" in sc.label


def test_uniform_label_is_neutral_for_channel_dependent_residuals():
    sc = scenarios.uniform(
        0.2, residuals={30: 2.0}, residual_excise_threshold=1.0)
    assert sc.is_excised(30)
    assert not sc.is_excised(29)
    assert sc.label == (
        "20% uniform masked fraction, dtv band; channel-dependent residuals")


# ----------------------------------------------------------------------
# Mask-table provenance
# ----------------------------------------------------------------------

def _product(path, channel, masked, n=400, kernel="aaaa", pkg="pp/1.0.0"):
    """Minimal survey product carrying a detector contract."""
    import json
    rej = np.zeros(n, dtype=np.uint8)
    rej[: int(round(masked * n))] = 1
    np.savez(
        path,
        valid=np.ones((n, 1), dtype=np.uint8),
        reject_mask=rej.reshape(n, 1),
        physical_channel=np.array([channel], dtype=np.int32),
        detector_version=np.array(f"{pkg} kernel=2.1.0 kernel_sha256={kernel}"),
        detector_contract_json=np.array(json.dumps(
            {"equivalent_mask_rule": "F > mu0", "threshold_mode": "none"})),
    )
    return path


def test_product_table_carries_its_rule(tmp_path):
    t = chn.mask_table_from_products([_product(tmp_path / "a.npz", 35, 0.25)])
    assert t.is_traceable and t.rule == "F > mu0"
    assert t.fractions[35] == pytest.approx(0.25)
    assert t.n_frames[35] == 400
    # refused channels are assumed, and say so
    assert t.fractions[30] == pytest.approx(chn.REFUSED_FRACTION)
    assert any("ch30" in n for n in t.notes)


def test_csv_table_is_marked_untraceable():
    t = chn.measured_mask_table()
    assert not t.is_traceable and t.rule == "unrecorded"
    assert any("records no statistic" in n for n in t.notes)


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


def test_compare_mask_tables_orders_by_disagreement():
    a = chn.MaskTable({14: 0.01, 35: 0.10}, "csv", "unrecorded")
    b = chn.MaskTable({14: 0.02, 35: 0.90}, "products", "F > mu0")
    rows = chn.compare_mask_tables(a, b)
    assert rows[0][0] == 35 and rows[0][3] == pytest.approx(9.0)
    assert rows[1][0] == 14 and rows[1][3] == pytest.approx(2.0)

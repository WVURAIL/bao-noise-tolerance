"""Survey-adapter contracts that bridge Bao to RadioFisher's public API."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from baonoise import resources, survey


def test_bull_redshift_bins_use_supported_backend_helper_directly():
    experiment = {"survey_numax": 800.0, "survey_dnutot": 400.0}
    seen = {}

    class Backend:
        @staticmethod
        def zbins_equal_spaced(expt, *, dz):
            seen.update(expt=expt, dz=dz)
            return [0.8, 0.9, 1.0], [0.85, 0.95]

    edges, centers = survey.chime_zbins(Backend(), experiment, dz=0.1)

    assert seen == {"expt": experiment, "dz": 0.1}
    assert np.array_equal(edges, [0.8, 0.9, 1.0])
    assert np.array_equal(centers, [0.85, 0.95])


def test_bull_experiment_ignores_radiofisher_root_array_config(tmp_path):
    stale = tmp_path / "array_config" / "nx_CHIME_800.dat"
    stale.parent.mkdir()
    stale.write_text("historical checkout data")
    backend = SimpleNamespace(
        experiments=SimpleNamespace(
            CHIME={"n(x)": "array_config/nx_CHIME_800.dat"}))

    experiment = survey.chime_experiment(backend, tmp_path)

    expected = resources.filesystem_data_file(
        resources.SYNTHETIC_BASELINE_NAME)
    assert Path(experiment["n(x)"]).resolve() == expected.resolve()
    assert Path(experiment["n(x)"]).resolve() != stale.resolve()


def test_recorded_bull_experiment_restores_foregrounds_and_overrides(
        monkeypatch, tmp_path):
    baseline = tmp_path / "nx.dat"
    baseline.write_text("baseline")
    seen = []

    def make_experiment(_rf, _rf_dir, ttot_hours, epsilon_fg, k_nl0):
        seen.append((ttot_hours, epsilon_fg, k_nl0))
        return {
            "ttot": ttot_hours * survey.HRS_MHZ,
            "epsilon_fg": epsilon_fg,
            "k_nl0": k_nl0,
            "n(x)": str(baseline),
        }

    monkeypatch.setattr(survey, "chime_experiment", make_experiment)
    meta = {
        "config": "bull2015",
        "expt_overrides": {"kfg_fac": 80.0},
        "provenance": {"experiment": {"settings": {
            "ttot": survey.HRS_MHZ,
            "epsilon_fg": 1e-5,
            "k_nl0": 0.2,
            "kfg_fac": 80.0,
            "n(x)": baseline.name,
        }}},
    }

    experiment = survey.experiment_from_bank_metadata(
        object(), tmp_path, meta, ttot_hours=25.0)

    assert seen == [(1.0, 1e-5, 0.2), (25.0, 1e-5, 0.2)]
    assert experiment["ttot"] == 25.0 * survey.HRS_MHZ
    assert experiment["kfg_fac"] == 80.0


def test_recorded_overview_experiment_restores_kfg_override(monkeypatch, tmp_path):
    baseline = tmp_path / "chime2021" / "array_config" / "nx.dat"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("baseline")

    def make_experiment(_rf, _rf_dir, ttot_hours):
        return {
            "ttot": ttot_hours * survey.HRS_MHZ,
            "epsilon_fg": 0.0,
            "k_nl0": 0.14,
            "n(x)": str(baseline),
        }

    monkeypatch.setattr(survey, "chime2022_experiment", make_experiment)
    meta = {
        "config": "chime2022",
        "expt_overrides": {"kfg_fac": 44.0},
        "provenance": {"experiment": {"settings": {
            "ttot": survey.HRS_MHZ,
            "epsilon_fg": 0.0,
            "k_nl0": 0.14,
            "kfg_fac": 44.0,
            "n(x)": "chime2021/array_config/nx.dat",
        }}},
    }

    experiment = survey.experiment_from_bank_metadata(
        object(), tmp_path, meta, ttot_hours=12.0)

    assert experiment["ttot"] == 12.0 * survey.HRS_MHZ
    assert experiment["kfg_fac"] == 44.0


def test_recorded_experiment_fails_closed_on_unexplained_setting_drift(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        survey, "chime2022_experiment",
        lambda _rf, _rf_dir, ttot_hours: {
            "ttot": ttot_hours * survey.HRS_MHZ, "epsilon_fg": 0.0})
    meta = {
        "config": "chime2022", "expt_overrides": {},
        "provenance": {"experiment": {"settings": {
            "ttot": survey.HRS_MHZ, "epsilon_fg": 1e-5}}},
    }

    with np.testing.assert_raises_regex(ValueError, "cannot be reconstructed"):
        survey.experiment_from_bank_metadata(
            object(), tmp_path, meta, ttot_hours=1.0)

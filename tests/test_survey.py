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

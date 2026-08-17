"""Focused tests for the main forecast driver's backend selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baonoise_test_run_forecast", ROOT / "scripts" / "run_forecast.py"
)
assert SPEC is not None and SPEC.loader is not None
run_forecast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_forecast)


def test_perbin_driver_path_does_not_import_radiofisher(monkeypatch):
    def unexpected_backend_import():
        raise AssertionError("the per-bin bank does not need RadioFisher")

    monkeypatch.setattr(
        run_forecast, "import_radiofisher", unexpected_backend_import
    )
    bank, fc, style = run_forecast._load_forecast()

    assert bank.meta["config"] == "chime2022"
    assert style == "perbin_A"
    assert fc.rf is None
    assert np.isfinite(fc.sigma_A(run_forecast.scenarios.clean(), 8_766.0))


def test_shared_driver_path_imports_radiofisher(monkeypatch):
    class SharedBank:
        def __init__(self):
            self.meta = {"config": "bull2015"}
            self.artifact_kind = "forecast"
            self.paramnames = ()

    backend = object()
    monkeypatch.setattr(run_forecast, "FisherBank", lambda _path: SharedBank())
    monkeypatch.setattr(
        run_forecast, "import_radiofisher", lambda: (backend, Path("/rf"))
    )

    bank, fc, style = run_forecast._load_forecast("shared-bank.npz")

    assert bank.meta["config"] == "bull2015"
    assert style == "shared_A"
    assert fc.rf is backend
    assert fc.rf_dir == Path("/rf")

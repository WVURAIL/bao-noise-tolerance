"""Fail-closed prerequisites for optional coherent-bias research scripts."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _source_environment():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


@pytest.mark.parametrize(
    ("script", "arguments", "expected_fragment"),
    [
        ("bias_tolerance.py", ["--bank", "missing.npz"],
         "fisher_bank_chime2022_pres_dense.npz"),
        ("plot_convergence.py", ["--bank", "missing.npz"],
         "fisher_bank_chime2022_pres_dense.npz"),
        ("three_worlds.py", ["--bank-dir", "missing"],
         "fisher_bank_chime2022_pres_dense.npz"),
    ],
)
def test_bias_workflows_fail_with_exact_build_prerequisite(
        tmp_path, script, arguments, expected_fragment):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=tmp_path, env=_source_environment(), text=True,
        capture_output=True, check=False)

    assert completed.returncode == 2
    assert "Bias-response banks are deliberately not shipped" in completed.stderr
    assert "--config chime2022 --cosmology planck2018 --p-res 1.0" \
        in completed.stderr
    assert expected_fragment in completed.stderr


def test_three_worlds_help_lists_all_four_strict_v2_prerequisites():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "three_worlds.py"), "--help"],
        cwd=ROOT, env=_source_environment(), text=True,
        capture_output=True, check=False)

    assert completed.returncode == 0
    for suffix in ("pres_dense.npz", "pres_kfg22_dense.npz",
                   "pres_kfg44_dense.npz", "pres_kfg80_dense.npz"):
        assert suffix in completed.stdout


def test_bias_workflow_rejects_an_ordinary_forecast_bank():
    forecast_bank = ROOT / "src" / "baonoise" / "data" \
        / "fisher_bank_chime2022.npz"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bias_tolerance.py"),
         "--bank", str(forecast_bank)],
        cwd=ROOT, env=_source_environment(), text=True,
        capture_output=True, check=False)

    assert completed.returncode == 2
    assert "artifact_kind must be 'bias_response'" in completed.stderr
    assert "--p-res 1.0 --dense-knee" in completed.stderr

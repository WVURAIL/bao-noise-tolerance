"""Combined-estimator, time-family, and refusal-ledger contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baonoise_test_bias_tolerance", ROOT / "scripts" / "bias_tolerance.py")
assert SPEC is not None and SPEC.loader is not None
bt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bt)


PARAMETERS = [
    "A", "b_HI", "Tb", "sigma_NL", "sigma8tot", "n_s", "f",
    "aperp", "apar", "fs8", "bs8", "pk", "_Pres",
]


class Bank:
    def __init__(self):
        matrix = np.diag(np.linspace(4.0, 16.0, len(PARAMETERS)))
        ip = PARAMETERS.index("_Pres")
        matrix[PARAMETERS.index("aperp"), ip] = 0.8
        matrix[ip, PARAMETERS.index("aperp")] = 0.8
        matrix[PARAMETERS.index("apar"), ip] = -0.4
        matrix[ip, PARAMETERS.index("apar")] = -0.4
        matrix[PARAMETERS.index("fs8"), ip] = 0.5
        matrix[ip, PARAMETERS.index("fs8")] = 0.5
        self.matrix = matrix
        self.paramnames = list(PARAMETERS)
        self.nbins = 2
        self.zs = np.array([0.8, 0.9, 1.0])
        self.zc = np.array([0.85, 0.95])
        self.schema_version = 2
        self.artifact_kind = "bias_response"
        self.meta = {
            "config": "chime2022",
            "cosmology": "planck2018",
            "astrophysical_model_profile": "chime_overview_2022",
            "expt_overrides": {"P_res": 1.0},
            "provenance": {
                "built_utc": "2026-08-20T00:00:00+00:00",
                "baonoise": {"working_tree_sha256": "a" * 64},
                "radiofisher": {
                    "backend_id": "radiofisher", "backend_version": "1.0.0",
                    "api_version": 1, "git_commit": "b" * 40,
                    "working_tree_sha256": "c" * 64,
                },
                "experiment": {"settings": {"P_res": 1.0}},
            },
        }

    def F(self, ibin, t_hours):
        return self.matrix * (ibin + 1.0) * float(t_hours) ** 2


class FakeRadioFisher:
    """Minimal implementation that makes every API call observable."""

    def __init__(self):
        self.calls = []

    def eos_fisher_matrix_derivs(self, cosmo, cosmo_fns, fsigma8=False):
        self.calls.append(("eos", fsigma8))
        return "eos-derivatives"

    def expand_fisher_matrix(self, z, derivs, F, names, exclude, fsigma8):
        self.calls.append(("expand", z, derivs, tuple(exclude), fsigma8))
        return np.array(F, copy=True), list(names)

    def transform_to_lss_distances(self, z, F, names, **kwargs):
        self.calls.append(("transform", z, tuple(sorted(kwargs))))
        renamed = list(names)
        renamed[renamed.index("DA")] = "DV"
        renamed[renamed.index("H")] = "F"
        return np.array(F, copy=True), renamed

    def combined_fisher_matrix(self, matrices, names, exclude, expand):
        self.calls.append(("combine", len(matrices), tuple(exclude), tuple(expand)))
        kept = [name for name in names if name not in set(exclude)]
        nbins = len(matrices)
        output_names = []
        for name in kept:
            if name in expand:
                output_names.extend(f"{name}{ibin}" for ibin in range(nbins))
            else:
                output_names.append(name)
        total = np.zeros((len(output_names), len(output_names)))
        for ibin, matrix in enumerate(matrices):
            local = [names.index(name) for name in kept]
            projected = np.asarray(matrix)[np.ix_(local, local)]
            mapping = [
                output_names.index(f"{name}{ibin}")
                if name in expand else output_names.index(name)
                for name in kept
            ]
            total[np.ix_(mapping, mapping)] += projected
        return total, output_names


COSMO_FNS = (
    lambda z: np.asarray(z) * 0.0 + 100.0,
    lambda z: np.asarray(z) * 1000.0 + 1000.0,
    lambda z: np.asarray(z) * 0.0 + 1.0,
    lambda z: np.asarray(z) * 0.0 + 0.8,
)


def test_appendix_a_path_retains_legacy_numerical_result():
    bank = Bank()
    matrix = bank.F(0, 3.0)
    old_derivative, old_sigma = bt.bias_per_unit_r(matrix, bank.paramnames)
    estimator = bt.PerBinAppendixAEstimator(bank)
    result = bt.evaluate_raw(
        estimator, 0, 3.0, "aperp", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)

    assert result["valid"] is True
    assert result["sigma"] == pytest.approx(old_sigma["aperp"])
    assert result["dtheta_d_current_noise_ratio"] \
        == pytest.approx(old_derivative["aperp"])


def test_fixed_physical_and_noise_normalized_families_coincide_only_at_reference():
    estimator = bt.PerBinAppendixAEstimator(Bank())
    reference = 4.0
    at_reference_noise = bt.evaluate_raw(
        estimator, 0, reference, "fs8", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    at_reference_fixed = bt.evaluate_raw(
        estimator, 0, reference, "fs8", zeta=1.0,
        time_scaling=bt.FIXED_PHYSICAL_AT_REFERENCE_TIME,
        reference_hours=reference)
    later_noise = bt.evaluate_raw(
        estimator, 0, 2 * reference, "fs8", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    later_fixed = bt.evaluate_raw(
        estimator, 0, 2 * reference, "fs8", zeta=1.0,
        time_scaling=bt.FIXED_PHYSICAL_AT_REFERENCE_TIME,
        reference_hours=reference)

    assert at_reference_fixed["dtheta_d_reported_amplitude"] \
        == pytest.approx(at_reference_noise["dtheta_d_reported_amplitude"])
    assert later_fixed["dtheta_d_reported_amplitude"] \
        == pytest.approx(2.0 * later_noise["dtheta_d_reported_amplitude"])
    assert later_fixed["r_tolerance"] \
        == pytest.approx(0.5 * later_noise["r_tolerance"])


def test_combined_path_exercises_expand_transform_and_combine_once_per_time():
    bank = Bank()
    rf = FakeRadioFisher()
    estimator = bt.OverviewCombinedMultibinEstimator(
        bank, rf=rf, rf_dir="/radiofisher", cosmo={},
        cosmo_fns=COSMO_FNS, eos_derivs="eos-derivatives")
    first = bt.evaluate_raw(
        estimator, 0, 2.0, "DV", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    second = bt.evaluate_raw(
        estimator, 1, 2.0, "fs8", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)

    assert first["valid"] is True
    assert first["target_name"] == "DV0"
    assert first["response_name"] == "_Pres0"
    assert second["valid"] is True
    assert second["target_name"] == "fs81"
    assert second["response_name"] == "_Pres1"
    system = estimator.system(0, 2.0)
    assert not any(name.startswith(bt.PRES) for name in system.names)
    assert system.names.count("sigma_NL") == 1
    assert sum(call[0] == "expand" for call in rf.calls) == bank.nbins
    assert sum(call[0] == "transform" for call in rf.calls) == bank.nbins
    assert sum(call[0] == "combine" for call in rf.calls) == 1
    combine = next(call for call in rf.calls if call[0] == "combine")
    assert combine[3] == bt.COMBINED_EXPAND


def test_every_requested_point_has_diagnostics_and_explicit_rejection_reason():
    result = bt.evaluate_fisher_point(
        bt.PerBinAppendixAEstimator(Bank()), 0, 2.0, "not_a_parameter",
        zeta=1.0, stability_fraction=0.1, max_drift=1.2)

    assert result["accepted"] is False
    assert len(result["perturbations"]) == 3
    assert [point["label"] for point in result["perturbations"]] \
        == ["lower", "central", "upper"]
    assert result["central"]["r_tolerance"] is None
    assert result["rejection_reasons"] == [
        "lower:requested_parameter_not_in_estimator",
        "central:requested_parameter_not_in_estimator",
        "upper:requested_parameter_not_in_estimator",
    ]


def test_report_is_strict_json_and_counts_rejections(tmp_path):
    path = tmp_path / "bank.npz"
    path.write_bytes(b"test-bank")
    bank = Bank()
    report = bt.build_report(
        bank, path, bt.PerBinAppendixAEstimator(bank), bins=[0], years=[1.0],
        params=["aperp", "missing"], zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None, stability_fraction=0.1, max_drift=1.3)

    encoded = json.dumps(report, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["schema"] == bt.REPORT_SCHEMA
    assert decoded["residual_amplitude"]["reported_amplitude_application"] \
        .startswith("applied exactly once")
    assert decoded["bins"][0]["summary"] == {
        "requested_parameter_points": 2,
        "accepted_parameter_points": 1,
        "rejected_parameter_points": 1,
        "binding_accepted_tolerance": pytest.approx(
            report["bins"][0]["summary"]["binding_accepted_tolerance"]),
    }


def test_fixed_physical_cli_requires_reference_before_loading_bank():
    with pytest.raises(SystemExit):
        bt.main(["--time-scaling", bt.FIXED_PHYSICAL_AT_REFERENCE_TIME])


def test_default_cli_json_retains_pre_v1_consumer_shape(
        monkeypatch, tmp_path, capsys):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test-bank")
    output = tmp_path / "legacy.json"
    bank = Bank()
    monkeypatch.setattr(bt, "load_bias_bank", lambda _path: bank)

    assert bt.main([
        "--bank", str(bank_path), "--bins", "0", "--years", "1",
        "--params", "aperp", "--json", str(output),
    ]) == 0
    capsys.readouterr()
    decoded = json.loads(output.read_text())

    assert not output.read_bytes().endswith(b"\n")
    assert set(decoded) == {"zeta", "bank", "bins"}
    assert decoded["bank"] == "bank.npz"
    assert set(decoded["bins"][0]) == {"zlo", "zhi", "rows", "binding"}
    assert set(decoded["bins"][0]["rows"]["1.0"]["aperp"]) == {
        "r_tol", "sigma", "dtheta_dr", "drift", "stable",
    }


def test_complete_v1_cli_json_requires_explicit_format(
        monkeypatch, tmp_path, capsys):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test-bank")
    output = tmp_path / "complete.json"
    bank = Bank()
    monkeypatch.setattr(bt, "load_bias_bank", lambda _path: bank)

    assert bt.main([
        "--bank", str(bank_path), "--bins", "0", "--years", "1",
        "--params", "aperp", "--json-format", "complete-v1",
        "--json", str(output),
    ]) == 0
    capsys.readouterr()
    decoded = json.loads(output.read_text())

    assert output.read_bytes().endswith(b"\n")
    assert decoded["schema"] == bt.REPORT_SCHEMA
    assert decoded["schema_version"] == 1


def test_legacy_json_rejects_nonhistorical_stability_fraction(
        monkeypatch, tmp_path):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test-bank")
    monkeypatch.setattr(bt, "load_bias_bank", lambda _path: Bank())

    with pytest.raises(SystemExit):
        bt.main([
            "--bank", str(bank_path), "--bins", "0",
            "--stability-fraction", "0.05",
        ])

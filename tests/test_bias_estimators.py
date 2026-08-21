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
        self.t_grid = np.array([1e-3, 1e9])
        self.schema_version = 2
        self.artifact_kind = "bias_response"
        digest = "a" * 64
        build_identity = {
            "built_utc": "2026-08-20T00:00:00+00:00",
            "baonoise": {
                "version": "1.0.0", "git_commit": "b" * 40,
                "git_dirty": False, "working_tree_sha256": digest,
                "source_manifest": {
                    key: list(value) for key, value in
                    bt.BAONOISE_SOURCE_MANIFEST.items()},
            },
            "radiofisher": {
                "backend_id": "radiofisher", "backend_version": "1.0.0",
                "api_version": 1, "capabilities": ["P_res"],
                "git_commit": "b" * 40, "git_dirty": False,
                "working_tree_sha256": "c" * 64,
                "source_manifest": {
                    key: list(value) for key, value in
                    bt.RADIOFISHER_SOURCE_MANIFEST.items()},
            },
            "cosmology": {
                "name": "planck2018", "sha256": digest,
                "parameters": {},
                "astrophysical_model_profile": "chime_overview_2022",
                "astrophysical_models": {},
            },
            "pk_cache": {
                "filename": "cache_pk_chime2022.dat", "sha256": digest,
                "cache_id": digest,
            },
            "experiment": {
                "sha256": digest, "settings": {"P_res": 1.0},
                "baseline_sha256": None,
            },
        }
        self.meta = {
            "config": "chime2022",
            "cosmology": "planck2018",
            "astrophysical_model_profile": "chime_overview_2022",
            "expt_overrides": {"P_res": 1.0},
            "foreground_settings": {
                key: None for key in bt.FOREGROUND_KEYS},
            "provenance": build_identity,
        }
        self.evaluation_identity = {
            key: value for key, value in build_identity.items()
            if key != "built_utc"}

    def F(self, ibin, t_hours):
        return self.matrix * (ibin + 1.0) * float(t_hours) ** 2


class FakeRadioFisher:
    """Minimal implementation that makes every API call observable."""

    C = 3e5

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

    dv = bt.evaluate_raw(
        estimator, 0, 3.0, "DV", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    assert dv["dtheta_d_current_noise_ratio"] == pytest.approx(
        -(2.0 / 3.0) * old_derivative["aperp"]
        - (1.0 / 3.0) * old_derivative["apar"])
    assert estimator.provenance["derived_targets"]["DV"].startswith("-")


@pytest.mark.parametrize("parameter", ["aperp", "apar", "fs8", "DV"])
def test_fixed_physical_and_noise_normalized_families_coincide_only_at_reference(
        parameter):
    estimator = bt.PerBinAppendixAEstimator(Bank())
    reference = 4.0
    at_reference_noise = bt.evaluate_raw(
        estimator, 0, reference, parameter, zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    at_reference_fixed = bt.evaluate_raw(
        estimator, 0, reference, parameter, zeta=1.0,
        time_scaling=bt.FIXED_PHYSICAL_AT_REFERENCE_TIME,
        reference_hours=reference)
    later_noise = bt.evaluate_raw(
        estimator, 0, 2 * reference, parameter, zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    later_fixed = bt.evaluate_raw(
        estimator, 0, 2 * reference, parameter, zeta=1.0,
        time_scaling=bt.FIXED_PHYSICAL_AT_REFERENCE_TIME,
        reference_hours=reference)

    assert at_reference_fixed["dtheta_d_reported_amplitude"] \
        == pytest.approx(at_reference_noise["dtheta_d_reported_amplitude"])
    assert later_fixed["dtheta_d_reported_amplitude"] \
        == pytest.approx(2.0 * later_noise["dtheta_d_reported_amplitude"])
    assert later_fixed["r_tolerance"] \
        == pytest.approx(0.5 * later_noise["r_tolerance"])


def test_diagonal_preconditioning_makes_eigencut_unit_rescaling_invariant():
    fisher = np.array([[9.0, 1.2], [1.2, 4.0]])
    response = np.array([0.7, -0.3])
    coefficients = np.array([0.4, 1.1])
    native = bt._solve_target(
        bt.FisherSystem(fisher, response, ("x", "y"), "_Pres"),
        coefficients, "target")

    units = np.array([1e12, 1e-12])
    rescaled = bt._solve_target(
        bt.FisherSystem(
            fisher / np.outer(units, units), response / units,
            ("x_scaled", "y_scaled"), "_Pres"),
        coefficients / units, "target")

    assert native["valid"] is True
    assert rescaled["valid"] is True
    assert rescaled["sigma"] == pytest.approx(native["sigma"])
    assert rescaled["dtheta_d_current_noise_ratio"] \
        == pytest.approx(native["dtheta_d_current_noise_ratio"])
    assert rescaled["condition_number"] \
        == pytest.approx(native["condition_number"])
    assert rescaled["discarded_eigenmodes"] \
        == native["discarded_eigenmodes"]


def test_complete_evaluation_refuses_both_sides_of_bank_time_grid():
    bank = Bank()
    bank.t_grid = np.array([1.0, 10.0])
    estimator = bt.PerBinAppendixAEstimator(bank)

    below = bt.evaluate_raw(
        estimator, 0, 0.5, "aperp", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    above = bt.evaluate_raw(
        estimator, 0, 11.0, "aperp", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None)
    legacy_above = bt.evaluate_raw(
        estimator, 0, 11.0, "aperp", zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None, enforce_bank_bounds=False)

    assert below["failure_reason"] == "outside_bank_time_grid"
    assert below["bank_time_grid_position"] == "below_minimum"
    assert above["failure_reason"] == "outside_bank_time_grid"
    assert above["bank_time_grid_position"] == "above_maximum"
    assert legacy_above["valid"] is True
    assert legacy_above["bank_time_grid_position"] == "above_maximum"


def test_one_outside_stability_perturbation_rejects_complete_point():
    bank = Bank()
    bank.t_grid = np.array([1.0, 10.0])
    result = bt.evaluate_fisher_point(
        bt.PerBinAppendixAEstimator(bank), 0, 1.05, "aperp",
        stability_fraction=0.10)

    assert result["accepted"] is False
    assert result["perturbations"][0]["bank_time_grid_position"] \
        == "below_minimum"
    assert result["rejection_reasons"] == [
        "lower:outside_bank_time_grid"]


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
    assert decoded["bank"]["expt_overrides"] == {"P_res": 1.0}
    assert decoded["bank"]["foreground_settings"]["kfg_fac"] is None
    assert decoded["bank"]["time_grid"] == {
        "minimum_hours": 1e-3,
        "maximum_hours": 1e9,
        "number_of_samples": 2,
    }
    assert decoded["bank"]["scientific_identity"]["verified_equal"] is True
    assert decoded["bins"][0]["summary"] == {
        "requested_parameter_points": 2,
        "accepted_parameter_points": 1,
        "rejected_parameter_points": 1,
        "binding_accepted_tolerance": pytest.approx(
            report["bins"][0]["summary"]["binding_accepted_tolerance"]),
    }
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "docs" / "bias-tolerance.schema.json").read_text(
            encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(decoded)


def test_report_provenance_distinguishes_no_delay_filter_from_kfg_80(
        tmp_path):
    path = tmp_path / "bank.npz"
    path.write_bytes(b"test-bank")
    bank = Bank()
    no_filter = bt.build_report(
        bank, path, bt.PerBinAppendixAEstimator(bank), bins=[0], years=[1.0],
        params=["aperp"], zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None, stability_fraction=0.1, max_drift=1.3)

    bank.meta["expt_overrides"]["kfg_fac"] = 80.0
    bank.meta["foreground_settings"]["kfg_fac"] = 80.0
    delay_filter = bt.build_report(
        bank, path, bt.PerBinAppendixAEstimator(bank), bins=[0], years=[1.0],
        params=["aperp"], zeta=1.0,
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None, stability_fraction=0.1, max_drift=1.3)

    assert no_filter["bank"]["foreground_settings"]["kfg_fac"] is None
    assert delay_filter["bank"]["foreground_settings"]["kfg_fac"] == 80.0
    assert no_filter["bank"]["foreground_settings_sha256"] \
        != delay_filter["bank"]["foreground_settings_sha256"]
    assert no_filter["bank"]["expt_overrides_sha256"] \
        != delay_filter["bank"]["expt_overrides_sha256"]


def test_fixed_physical_cli_requires_reference_before_loading_bank():
    with pytest.raises(SystemExit):
        bt.main(["--time-scaling", bt.FIXED_PHYSICAL_AT_REFERENCE_TIME])


def test_default_cli_json_retains_pre_v1_consumer_shape(
        monkeypatch, tmp_path, capsys):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test-bank")
    output = tmp_path / "legacy.json"
    bank = Bank()
    monkeypatch.setattr(
        bt, "load_bias_bank", lambda _path, **_kwargs: bank)

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
    monkeypatch.setattr(
        bt, "load_bias_bank", lambda _path, **_kwargs: bank)

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


@pytest.mark.parametrize(
    ("year", "position"), [(0.5, "below_minimum"), (2.0, "above_maximum")])
def test_complete_v1_cli_refuses_outside_grid_with_explicit_position(
        monkeypatch, tmp_path, capsys, year, position):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test-bank")
    output = tmp_path / f"complete-{position}.json"
    bank = Bank()
    bank.t_grid = np.array([8000.0, 10000.0])
    monkeypatch.setattr(
        bt, "load_bias_bank", lambda _path, **_kwargs: bank)

    assert bt.main([
        "--bank", str(bank_path), "--bins", "0", "--years", str(year),
        "--params", "aperp", "--json-format", "complete-v1",
        "--json", str(output),
    ]) == 0
    capsys.readouterr()
    central = json.loads(output.read_text())["bins"][0]["points"][0][
        "parameters"]["aperp"]["central"]
    assert central["valid"] is False
    assert central["failure_reason"] == "outside_bank_time_grid"
    assert central["bank_time_grid_position"] == position


def test_legacy_json_rejects_nonhistorical_stability_fraction(
        monkeypatch, tmp_path):
    bank_path = tmp_path / "bank.npz"
    bank_path.write_bytes(b"test-bank")
    monkeypatch.setattr(
        bt, "load_bias_bank", lambda _path, **_kwargs: Bank())

    with pytest.raises(SystemExit):
        bt.main([
            "--bank", str(bank_path), "--bins", "0",
            "--stability-fraction", "0.05",
        ])


def test_bank_build_source_manifest_mismatch_fails_before_backend_import():
    bank = Bank()
    bank.meta["provenance"]["baonoise"]["source_manifest"] = {
        "include": ["one-file.py"], "exclude": []}

    with pytest.raises(ValueError, match="Bao scientific-source manifest"):
        bt._evaluation_identity(bank)

"""Versioned Fisher-bank schema and artifact-type safety."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from baonoise import api, fisherbank
from baonoise.fisherbank import (ARTIFACT_BIAS_RESPONSE, ARTIFACT_FORECAST,
                                 BANK_SCHEMA_VERSION, FisherBank)
from baonoise.forecast import Forecast
from baonoise.resources import DEFAULT_BANK


def _write_bank(path, *, kind=ARTIFACT_FORECAST,
                names=("A", "sigma_NL"), mutate=None):
    F = np.repeat(np.eye(len(names))[None, None, :, :], 4, axis=1)
    arrays = dict(
        F=F, t_grid=np.array([1.0, 2.0, 4.0, 8.0]),
        zs=np.array([0.8, 0.9]), zc=np.array([0.85]),
        paramnames=np.asarray(names))
    digest = "a" * 64
    meta = {
            "schema_version": BANK_SCHEMA_VERSION,
            "artifact_kind": kind, "config": "chime2022",
            "cosmology": "planck2018",
            "astrophysical_model_profile": "chime_overview_2022",
            "provenance": {
                "built_utc": "2026-08-16T00:00:00+00:00",
                "baonoise": {"version": "1.0.0", "git_commit": None,
                             "git_dirty": None,
                             "working_tree_sha256": digest,
                             "source_manifest": {
                                 key: list(value) for key, value in
                                 fisherbank.BAONOISE_SOURCE_MANIFEST.items()}},
                "radiofisher": {
                    "backend_id": "radiofisher", "backend_version": "1.0.0",
                    "api_version": 1,
                    "capabilities": sorted(
                        ["astrophysical_model_profiles",
                         "explicit_physical_densities"]
                        + (["P_res"] if kind ==
                           ARTIFACT_BIAS_RESPONSE else [])),
                    "git_commit": None, "git_dirty": None,
                    "working_tree_sha256": digest,
                    "source_manifest": {
                        key: list(value) for key, value in
                        fisherbank.RADIOFISHER_SOURCE_MANIFEST.items()}},
                "cosmology": {
                    "name": "planck2018", "sha256": digest,
                    "parameters": {
                        "h": 0.6732, "ns": 0.96605, "sigma_8": 0.812,
                        "N_eff": 3.046, "mnu": 0.06,
                        "ombh2": 0.022383, "omch2": 0.120092,
                        "omnuh2": 0.000645},
                    "astrophysical_model_profile": "chime_overview_2022",
                    "astrophysical_models": {
                        "Tb_model": "hall", "bias_HI_model": "castorina",
                        "omega_HI_model": "crighton"}},
                "pk_cache": {"filename": "cache_pk_chime2022.dat",
                             "sha256": digest, "cache_id": digest},
                "experiment": {"sha256": digest, "settings": {},
                               "baseline_sha256": None},
            },
        }
    if mutate is not None:
        mutate(arrays, meta)
    np.savez(path, **arrays, meta=json.dumps(meta))
    return path


def test_shipped_bank_is_provenance_complete_v2():
    bank = FisherBank(DEFAULT_BANK)
    assert bank.schema_version == BANK_SCHEMA_VERSION
    assert bank.artifact_kind == ARTIFACT_FORECAST
    assert bank.meta["astrophysical_model_profile"] == "chime_overview_2022"


def test_shipped_bank_matches_current_scientific_source_digest():
    """Docs, tests, and generated outputs must not enter the bank digest."""
    root = Path(__file__).resolve().parents[1]
    current = fisherbank._git_state(
        root, **fisherbank.BAONOISE_SOURCE_MANIFEST,
    )["working_tree_sha256"]
    recorded = FisherBank(DEFAULT_BANK).meta["provenance"]["baonoise"][
        "working_tree_sha256"]
    assert current == recorded
    assert FisherBank(DEFAULT_BANK).meta["provenance"]["baonoise"][
        "source_manifest"] == {
            key: list(value) for key, value in
            fisherbank.BAONOISE_SOURCE_MANIFEST.items()}


def test_scientific_source_digest_is_checkout_newline_independent(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "model.py"
    project = tmp_path / "pyproject.toml"
    source.write_bytes(b"VALUE = 1\n")
    project.write_bytes(b"[project]\nname = 'example'\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "model.py", "pyproject.toml"],
        check=True)
    manifest = {"include": ("*.py", "*.toml"), "exclude": ()}
    lf_digest = fisherbank._git_state(
        tmp_path, **manifest)["working_tree_sha256"]
    source.write_bytes(b"VALUE = 1\r\n")
    project.write_bytes(b"[project]\r\nname = 'example'\r\n")
    crlf_digest = fisherbank._git_state(
        tmp_path, **manifest)["working_tree_sha256"]
    assert lf_digest == crlf_digest


def test_scientific_source_digest_survives_committing_a_deletion(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "model.py"], check=True)
    manifest = {"include": ("*.py",), "exclude": ()}

    source.unlink()
    unstaged_deletion = fisherbank._git_state(
        tmp_path, **manifest)["working_tree_sha256"]
    subprocess.run(
        ["git", "-C", str(tmp_path), "rm", "--cached", "model.py"],
        check=True, capture_output=True)
    committed_tree = fisherbank._git_state(
        tmp_path, **manifest)["working_tree_sha256"]

    assert unstaged_deletion == committed_tree


def test_schema_one_bank_requires_regeneration(tmp_path):
    def mutate(_arrays, meta):
        meta["schema_version"] = 1

    path = _write_bank(tmp_path / "obsolete.npz", mutate=mutate)
    with pytest.raises(ValueError, match="only supported version is 2"):
        FisherBank(path)


def test_retired_version_alias_is_rejected(tmp_path):
    def mutate(_arrays, meta):
        meta["version"] = meta.pop("schema_version")

    path = _write_bank(tmp_path / "alias.npz", mutate=mutate)
    with pytest.raises(ValueError, match="retired 'version' alias"):
        FisherBank(path)


@pytest.mark.parametrize("version", [0, 1, 3, "2", None])
def test_unknown_or_malformed_schema_version_is_rejected(tmp_path, version):
    path = _write_bank(tmp_path / "bank.npz")
    with np.load(path, allow_pickle=False) as dat:
        arrays = {name: np.array(dat[name], copy=True) for name in dat.files
                  if name != "meta"}
    arrays["meta"] = json.dumps({"schema_version": version})
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="schema version"):
        FisherBank(path)


def test_bias_response_bank_is_rejected_by_generic_forecast(tmp_path):
    path = _write_bank(
        tmp_path / "bias.npz",
        kind=ARTIFACT_BIAS_RESPONSE, names=("A", "_Pres"))
    bank = FisherBank(path)
    assert bank.artifact_kind == ARTIFACT_BIAS_RESPONSE
    with pytest.raises(ValueError, match="bias|_Pres"):
        Forecast(bank, style="perbin_A")
    with pytest.raises(ValueError, match="artifact_kind|_Pres"):
        api.load(path)


@pytest.mark.parametrize(
    "kind,names",
    [(ARTIFACT_FORECAST, ("A", "_Pres")),
     (ARTIFACT_BIAS_RESPONSE, ("A", "sigma_NL"))])
def test_v2_artifact_kind_must_match_bias_row(tmp_path, kind, names):
    path = _write_bank(tmp_path / "bad.npz", kind=kind, names=names)
    with pytest.raises(ValueError, match="inconsistent"):
        FisherBank(path)


def test_unknown_config_is_rejected(tmp_path):
    def mutate(_arrays, meta):
        meta["config"] = "retired"

    path = _write_bank(tmp_path / "bad-config.npz", mutate=mutate)
    with pytest.raises(ValueError, match="unknown config"):
        FisherBank(path)


def test_v2_requires_complete_provenance(tmp_path):
    def mutate(_arrays, meta):
        meta["provenance"]["radiofisher"].pop("capabilities")

    path = _write_bank(tmp_path / "bad-provenance.npz", mutate=mutate)
    with pytest.raises(ValueError, match="capabilities"):
        FisherBank(path)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["radiofisher"].update(api_version=2), "api_version"),
    (lambda p: p["radiofisher"].update(capabilities=[]), "capability"),
    (lambda p: p["cosmology"].update(astrophysical_models={}),
     "canonical profile"),
    (lambda p: p["cosmology"].update(parameters={}), "parameters are missing"),
    (lambda p: p["baonoise"].update(source_manifest={}), "source_manifest"),
])
def test_v2_rejects_false_scientific_provenance(tmp_path, mutation, match):
    def mutate(_arrays, meta):
        mutation(meta["provenance"])

    path = _write_bank(tmp_path / "false-provenance.npz", mutate=mutate)
    with pytest.raises(ValueError, match=match):
        FisherBank(path)


def test_bias_response_requires_pres_backend_capability(tmp_path):
    def mutate(_arrays, meta):
        meta["provenance"]["radiofisher"]["capabilities"].remove("P_res")

    path = _write_bank(
        tmp_path / "bias.npz", kind=ARTIFACT_BIAS_RESPONSE,
        names=("A", "_Pres"), mutate=mutate)
    with pytest.raises(ValueError, match="P_res"):
        FisherBank(path)


@pytest.mark.parametrize("field", ["F", "t_grid", "zs", "zc", "paramnames"])
def test_malformed_bank_arrays_are_rejected(tmp_path, field):
    def mutate(arrays, _meta):
        if field == "F":
            arrays[field][0, 0, 0, 0] = np.nan
        elif field == "t_grid":
            arrays[field][2] = arrays[field][1]
        elif field == "zs":
            arrays[field] = arrays[field][:-1]
        elif field == "zc":
            arrays[field][0] = 1.1
        else:
            arrays[field] = np.array(["A", "A"])

    path = _write_bank(tmp_path / "bad.npz", mutate=mutate)
    with pytest.raises(ValueError):
        FisherBank(path)

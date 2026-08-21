"""Named analytic residual templates remain callable and auditable."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from baonoise import fisherbank, residual_templates as templates
from baonoise.compat import find_radiofisher_dir

ROOT = Path(__file__).resolve().parents[1]
BIAS_SPEC = importlib.util.spec_from_file_location(
    "baonoise_test_named_template_bias_tolerance",
    ROOT / "scripts" / "bias_tolerance.py")
assert BIAS_SPEC is not None and BIAS_SPEC.loader is not None
bias_tolerance = importlib.util.module_from_spec(BIAS_SPEC)
BIAS_SPEC.loader.exec_module(bias_tolerance)


@pytest.mark.parametrize("family", templates.FAMILIES)
def test_named_templates_are_unit_normalized_callable_json(family):
    template = templates.make_template(family)
    k = np.array([[0.01], [0.10], [0.30]])
    u = np.array([[-1.0, 0.0, 1.0]])
    noise = np.full((3, 3), 7.0)
    values = template(k, u, noise, np.ones_like(noise))

    assert callable(template)
    assert isinstance(template, dict)
    assert template["amplitude"] == 1.0
    assert template["normalization"] == "thermal_noise_at_evaluation_time"
    assert template["template_api_version"] == templates.TEMPLATE_API_VERSION
    assert template["implementation_sha256"] \
        == templates.implementation_sha256()
    assert values.shape == noise.shape
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= noise))
    assert json.loads(json.dumps(template))["family"] == family


def test_noise_shaped_is_exactly_the_scalar_unit_response():
    template = templates.make_template(templates.NOISE_SHAPED)
    noise = np.arange(1.0, 7.0).reshape(2, 3)
    assert np.array_equal(
        template(np.ones((2, 1)), np.ones((1, 3)), noise, noise * 2), noise)


def test_template_parameters_refuse_unknown_or_duplicate_inputs():
    with pytest.raises(ValueError, match="does not accept"):
        templates.make_template(templates.LOW_KPARALLEL, {"typo": 1.0})
    with pytest.raises(ValueError, match="duplicated"):
        templates.parse_parameter_assignments(["slope=1", "slope=2"])


def test_loaded_metadata_fails_closed_if_formula_identity_changes():
    metadata = json.loads(json.dumps(templates.make_template(
        templates.LOW_KPARALLEL)))
    assert templates.validate_template_metadata(metadata) == metadata
    metadata["implementation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authenticated implementation"):
        templates.validate_template_metadata(metadata)


def test_named_template_builds_authenticated_response_bank_end_to_end(
        tmp_path):
    try:
        rf_dir = find_radiofisher_dir()
    except FileNotFoundError:
        pytest.skip("end-to-end template build requires RadioFisher")

    template = templates.make_template(
        templates.LOW_KPARALLEL,
        {"k_parallel_scale_mpc_inv": 0.04})
    output = tmp_path / "named-template-response.npz"
    fisherbank.build_bank(
        output, rf_dir=rf_dir, config="chime2022", cosmology="planck2018",
        t_grid_hours=np.array([8000.0, 10000.0]), nproc=4,
        expt_overrides={"P_res": template}, verbose=False)

    bank = fisherbank.FisherBank(output)
    recorded = bank.meta["expt_overrides"]["P_res"]
    provenance_record = bank.meta["provenance"]["experiment"]["settings"][
        "P_res"]
    assert "_Pres" in bank.paramnames
    assert bank.artifact_kind == fisherbank.ARTIFACT_BIAS_RESPONSE
    assert templates.validate_template_metadata(recorded) == dict(template)
    assert templates.validate_template_metadata(provenance_record) \
        == dict(template)
    assert bank.meta["provenance"]["baonoise"]["working_tree_sha256"] \
        == fisherbank._git_state(
            ROOT,
            **fisherbank.BAONOISE_SOURCE_MANIFEST)["working_tree_sha256"]

    authenticated = bias_tolerance.load_bias_bank(output, rf_dir=rf_dir)
    assert authenticated.evaluation_identity["baonoise"][
        "working_tree_sha256"] == bank.meta["provenance"]["baonoise"][
            "working_tree_sha256"]
    assert authenticated.evaluation_identity["radiofisher"][
        "working_tree_sha256"] == bank.meta["provenance"]["radiofisher"][
            "working_tree_sha256"]
    assert authenticated.evaluation_identity["cosmology"] \
        == bank.meta["provenance"]["cosmology"]
    assert authenticated.evaluation_identity["pk_cache"] \
        == bank.meta["provenance"]["pk_cache"]
    assert authenticated.evaluation_identity["experiment"] \
        == bank.meta["provenance"]["experiment"]

    bank.meta["provenance"]["baonoise"]["working_tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="baonoise.working_tree_sha256"):
        bias_tolerance._evaluation_identity(bank, rf_dir=rf_dir)


def test_no_empirical_or_sidereal_family_is_fabricated():
    assert not any("empirical" in family or "sidereal" in family
                   for family in templates.FAMILIES)

"""Named analytic residual templates remain callable and auditable."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baonoise_test_residual_templates",
    ROOT / "scripts" / "residual_templates.py")
assert SPEC is not None and SPEC.loader is not None
templates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(templates)


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


def test_no_empirical_or_sidereal_family_is_fabricated():
    assert not any("empirical" in family or "sidereal" in family
                   for family in templates.FAMILIES)

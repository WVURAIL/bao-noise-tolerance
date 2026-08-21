"""Dissertation rendering checks for the all-template forecast assets."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baonoise_test_render_forecast_template_assets",
    ROOT / "scripts" / "render_forecast_template_assets.py")
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_renderer_exports_figure_table_and_complete_caption(tmp_path):
    figure = tmp_path / "comparison.png"
    table = tmp_path / "summary.tex"
    caption = tmp_path / "caption.txt"
    manifest = tmp_path / "manifest.json"
    assert renderer.main([
        "--comparison",
        str(ROOT / "out" / "forecast_completion_template_comparison.csv"),
        "--channels",
        str(ROOT / "out" / "forecast_completion_channel_mapping.csv"),
        "--status",
        str(ROOT / "out" / "forecast_completion_template_status.csv"),
        "--figure", str(figure),
        "--table", str(table),
        "--caption", str(caption),
        "--manifest", str(manifest),
    ]) == 0

    assert figure.stat().st_size > 20_000
    assert figure.with_suffix(".pdf").stat().st_size > 10_000
    tex = table.read_text(encoding="utf-8")
    assert "13/8" in tex
    assert "15/6" in tex
    assert "42/21" in tex
    assert "40/23" in tex
    assert "39/24" in tex
    assert "sensitivity-envelope ordering only" in tex
    assert all(line.endswith(r"\\") for line in tex.splitlines()
               if any(label in line for label in (
                   "Noise-shaped unit", "Low-$k_", "Wedge-like",
                   "Localized $k$ shell")))
    text = caption.read_text(encoding="utf-8")
    assert "not an overlap-weighted average" in text
    assert "data-dependent and incomplete" in text
    release = json.loads(manifest.read_text(encoding="utf-8"))
    assert release["schema"] == renderer.MANIFEST_SCHEMA
    assert release["artifact_count"] == 12
    assert release["wall_clock_fields_included"] is False
    assert release["absolute_paths_included"] is False
    assert all(not item["path"].startswith("/")
               for item in release["artifacts"])
    assert len(release["empirical_template_refusals"]) == 3
    assert release["scientific_scope"]["physical_channels"] \
        == list(range(14, 37))
    identities = release["scientific_identities"]
    assert identities["all_build_evaluation_pairs_verified_equal"] is True
    assert identities["baonoise"]["working_tree_sha256"] \
        == "ae448067c9eaf60d2dde3e0d7a57110db41759a62f2ee89e7ac4c9072e13e1a3"
    assert identities["radiofisher"]["clean_git_commit"] \
        == "3cc9f34e183db9e04820c8a2e7932395ec3a0441"
    assert len(identities["per_evidence"]) == 4
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(renderer.MANIFEST_SCHEMA_PATH.read_text(
        encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(release)


def test_aggregate_retains_exact_disposition_counts():
    comparisons = renderer._read_rows(
        ROOT / "out" / "forecast_completion_template_comparison.csv",
        renderer.COMPARISON_SCHEMA)
    channels = renderer._read_rows(
        ROOT / "out" / "forecast_completion_channel_mapping.csv",
        renderer.CHANNEL_SCHEMA)
    renderer._validate(comparisons, channels)
    summary = renderer._aggregate(comparisons, channels)

    assert summary["noise_shaped"]["perbin_accepted"] == 13
    assert summary["noise_shaped"]["perbin_rejected"] == 8
    assert summary["low_kparallel"]["perbin_accepted"] == 15
    assert summary["low_kparallel"]["perbin_rejected"] == 6
    assert summary["wedge_like"]["combined_noise_accepted"] == 40
    assert summary["wedge_like"]["combined_noise_rejected"] == 23
    assert summary["k_shell_localized"]["combined_fixed_accepted"] == 39
    assert summary["k_shell_localized"]["combined_fixed_rejected"] == 24

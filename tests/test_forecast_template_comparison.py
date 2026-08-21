"""All-bin template comparison and channel-propagation contracts."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baonoise_test_forecast_template_comparison",
    ROOT / "scripts" / "forecast_template_comparison.py")
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


EVIDENCE = {
    "noise_shaped": ROOT / "out" / "forecast_completion_all_dtv_bins.json",
    "low_kparallel": ROOT / "out"
        / "forecast_completion_all_dtv_bins_low_kparallel.json",
    "wedge_like": ROOT / "out"
        / "forecast_completion_all_dtv_bins_wedge_like.json",
    "k_shell_localized": ROOT / "out"
        / "forecast_completion_all_dtv_bins_k_shell_localized.json",
}


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _arguments(tmp_path):
    arguments = []
    for family, path in EVIDENCE.items():
        arguments.extend(["--evidence", f"{family}={path}"])
    arguments.extend([
        "--comparison-out", str(tmp_path / "comparison.csv"),
        "--channel-out", str(tmp_path / "channels.csv"),
        "--status-out", str(tmp_path / "status.csv"),
    ])
    return arguments


def test_actual_evidence_exports_complete_template_and_channel_tables(tmp_path):
    assert comparison.main(_arguments(tmp_path)) == 0
    bin_rows = _read_csv(tmp_path / "comparison.csv")
    channel_rows = _read_csv(tmp_path / "channels.csv")
    status_rows = _read_csv(tmp_path / "status.csv")

    assert len(bin_rows) == 4 * 7
    assert len(channel_rows) == 4 * 23
    assert len(status_rows) == 4 + 3
    assert {row["family"] for row in bin_rows} == set(comparison.FAMILIES)
    assert {int(row["bin_index"]) for row in bin_rows} \
        == {5, 6, 7, 8, 9, 10, 11}
    assert all(row["perbin_binding_parameter"] == "fs8"
               for row in bin_rows)
    assert all(row["combined_binding_parameter"] == "fs8"
               for row in bin_rows)
    assert all(row["fixed_equals_noise_at_reference"] == "true"
               for row in bin_rows)
    assert all(int(row["combined_noise_grid_accepted"])
               + int(row["combined_noise_grid_rejected"]) == 9
               for row in bin_rows)
    assert all(int(row["combined_fixed_grid_accepted"])
               + int(row["combined_fixed_grid_rejected"]) == 9
               for row in bin_rows)

    assert all(float(row["coverage_fraction"]) == pytest.approx(1.0)
               for row in channel_rows)
    assert all(row["shared_target"] == "fs8" for row in channel_rows)
    assert all(row["perbin_status"] == "accepted"
               and row["combined_status"] == "accepted"
               for row in channel_rows)
    assert all(row["existing_policy_status_change"] == "none"
               and row["existing_policy_ranking_change"] == "none"
               for row in channel_rows)

    noise_rows = {
        int(row["channel"]): row for row in channel_rows
        if row["family"] == "noise_shaped"}
    assert noise_rows[14]["overlap_bin_indices"] == "11"
    assert noise_rows[17]["overlap_bin_indices"] == "10;11"
    assert noise_rows[20]["overlap_bin_indices"] == "9;10"
    assert noise_rows[23]["overlap_bin_indices"] == "8;9"
    assert noise_rows[26]["overlap_bin_indices"] == "7;8"
    assert noise_rows[30]["overlap_bin_indices"] == "6;7"
    assert noise_rows[34]["overlap_bin_indices"] == "5;6"
    assert noise_rows[36]["overlap_bin_indices"] == "5"

    rank_by_family = {
        row["family"]: int(row["combined_strictness_rank"])
        for row in channel_rows if row["channel"] == "29"}
    assert rank_by_family == {
        "wedge_like": 1,
        "noise_shaped": 2,
        "low_kparallel": 3,
        "k_shell_localized": 4,
    }

    empirical = [
        row for row in status_rows
        if row["category"] == "empirical_visibility"]
    assert len(empirical) == 3
    assert all(row["execution_status"] == "data_dependent_incomplete"
               and row["scope"]
               == "explicit_refusal_no_fabricated_visibility_template"
               and not row["evidence_file"]
               for row in empirical)


def test_nonnoise_template_must_authenticate_against_installed_source(tmp_path):
    payload = json.loads(EVIDENCE["low_kparallel"].read_text(encoding="utf-8"))
    payload["bank"]["P_res"]["implementation_sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="template authentication failed"):
        comparison._validate_evidence(path, "low_kparallel")


@pytest.mark.parametrize(
    ("channel", "expected_bins"),
    [
        (14, [11]), (17, [10, 11]), (20, [9, 10]),
        (23, [8, 9]), (26, [7, 8]), (30, [6, 7]),
        (34, [5, 6]), (36, [5]),
    ],
)
def test_physical_channel_overlap_geometry(channel, expected_bins):
    payload = comparison._validate_evidence(
        EVIDENCE["noise_shaped"], "noise_shaped")
    geometry = {
        int(item["bin_index"]): {
            "z_low": float(item["z_low"]),
            "z_high": float(item["z_high"]),
        }
        for item in payload["evidence_scope"]["redshift_bins"]}
    frequency = comparison.channels.channel_edges(channel)
    actual = [
        ibin for ibin, bounds in sorted(geometry.items())
        if comparison._overlap(
            frequency,
            comparison._bin_frequency(bounds["z_low"], bounds["z_high"]))
        > 0.0
    ]
    assert actual == expected_bins

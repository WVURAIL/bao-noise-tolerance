#!/usr/bin/env python3
"""Summarize authenticated all-DTV-bin residual-template forecasts.

This command consumes the complete-v2 evidence ledgers; it never evaluates a
Fisher matrix itself.  The bin table preserves every reference-time
acceptance count and binding tolerance.  The channel table maps the common
``fs8`` target onto physical ATSC channels 14--36 using every redshift bin
with non-zero frequency overlap.  A channel's reported tolerance is the
minimum across its overlaps, not an overlap-weighted average, because a
residual in any covered part of the channel can bias that bin.

The three empirical rows are explicit refusals.  RadioFisher's current
``P_res(k, u, P_N, P_signal)`` callable has no frequency, baseline, or
sidereal-time coordinate, and no visibility product is accepted here as a
substitute for those missing measurements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from baonoise import channels, residual_templates
from baonoise.constants import HI_REST_FREQUENCY_MHZ


EVIDENCE_SCHEMA = "baonoise-forecast-completion-evidence-v2"
COMPARISON_SCHEMA = "baonoise-forecast-template-comparison-csv-v1"
CHANNEL_SCHEMA = "baonoise-forecast-channel-mapping-csv-v1"
STATUS_SCHEMA = "baonoise-forecast-template-status-csv-v1"
FAMILIES = residual_templates.FAMILIES

EMPIRICAL_REFUSALS = (
    (
        "empirical_frequency_localized",
        "pilot-proxy residual spectra with authenticated physical-frequency "
        "coordinates, plus a Fisher response interface exposing frequency",
    ),
    (
        "empirical_baseline_localized",
        "baseline-resolved residual power/covariance, plus a Fisher response "
        "interface exposing baseline coordinates",
    ),
    (
        "empirical_sidereal_coherent",
        "sidereal-time-resolved residual covariance/coherence and the window "
        "coupling needed to propagate it into the Fisher integrand",
    ),
)


COMPARISON_FIELDS = (
    "schema", "family", "template_provenance", "evidence_file",
    "evidence_sha256", "bank_sha256", "numerical_grid_sha256",
    "bin_index", "z_low", "z_high", "frequency_low_mhz",
    "frequency_high_mhz", "overlapping_channels", "reference_years",
    "perbin_parameters", "perbin_accepted", "perbin_rejected",
    "perbin_rejected_parameters", "perbin_binding_parameter",
    "perbin_binding_tolerance", "combined_parameters",
    "combined_accepted", "combined_rejected",
    "combined_rejected_parameters", "combined_binding_parameter",
    "combined_binding_tolerance", "perbin_fs8_accepted",
    "perbin_fs8_tolerance", "combined_fs8_accepted",
    "combined_fs8_tolerance", "combined_to_perbin_fs8_ratio",
    "fixed_equals_noise_at_reference",
    "combined_noise_grid_accepted", "combined_noise_grid_rejected",
    "combined_fixed_grid_accepted", "combined_fixed_grid_rejected",
)

CHANNEL_FIELDS = (
    "schema", "family", "channel", "frequency_low_mhz",
    "frequency_high_mhz", "overlap_bin_indices", "overlap_mhz_by_bin",
    "coverage_fraction", "reference_years", "shared_target",
    "mapping_rule", "perbin_status", "perbin_binding_bin",
    "perbin_conservative_tolerance", "combined_status",
    "combined_binding_bin", "combined_conservative_tolerance",
    "combined_to_perbin_ratio", "perbin_strictness_rank",
    "combined_strictness_rank", "existing_policy_status_change",
    "existing_policy_ranking_change", "policy_interpretation",
)

STATUS_FIELDS = (
    "schema", "family", "category", "execution_status", "evidence_file",
    "evidence_sha256", "bank_sha256", "template_authentication",
    "scope", "external_data_or_interface_required",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_float(value: float) -> str:
    # Complete binary64 values remain in the evidence JSON.  Twelve
    # significant digits are ample for this compact propagation table and
    # avoid presenting artifacts such as 1.3999999999999999 as bin edges.
    return format(float(value), ".12g")


def _all_true(value) -> bool:
    if isinstance(value, dict):
        return all(_all_true(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_true(item) for item in value)
    return value is True


def _parse_evidence_assignment(value: str) -> tuple[str, Path]:
    if value.count("=") != 1:
        raise argparse.ArgumentTypeError(
            "--evidence must use FAMILY=PATH exactly once")
    family, raw_path = value.split("=", 1)
    if family not in FAMILIES:
        raise argparse.ArgumentTypeError(
            f"unknown family {family!r}; choose from {FAMILIES}")
    if not raw_path:
        raise argparse.ArgumentTypeError("--evidence PATH must not be empty")
    return family, Path(raw_path)


def _template_identity(payload: dict, requested_family: str) -> tuple[str, str]:
    p_res = payload["bank"]["P_res"]
    if isinstance(p_res, (int, float)) and not isinstance(p_res, bool):
        if requested_family != residual_templates.NOISE_SHAPED \
                or not np.isclose(float(p_res), 1.0, rtol=0.0, atol=0.0):
            raise ValueError(
                "only a scalar unit P_res may stand for noise_shaped")
        return requested_family, (
            "scalar_unit_P_res_equivalent_to_noise_shaped; "
            "not_named_template_metadata")
    if not isinstance(p_res, dict):
        raise ValueError("evidence bank P_res must be a scalar or object")
    try:
        authenticated = residual_templates.validate_template_metadata(p_res)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{requested_family} template authentication failed: {exc}") \
            from exc
    if authenticated["family"] != requested_family:
        raise ValueError(
            f"evidence labeled {requested_family!r} contains "
            f"{authenticated['family']!r}")
    provenance = (
        f"authenticated_named_template_api_v"
        f"{authenticated['template_api_version']}:"
        f"{authenticated['implementation_sha256']}")
    return requested_family, provenance


def _ledger_bins(ledger: dict) -> dict[int, dict]:
    bins = {int(item["bin_index"]): item for item in ledger["bins"]}
    if len(bins) != len(ledger["bins"]):
        raise ValueError("ledger contains duplicate bin indices")
    return bins


def _reference_point(bin_report: dict, reference_years: float) -> dict:
    matches = [
        point for point in bin_report["points"]
        if np.isclose(
            float(point["years"]), reference_years, rtol=0.0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"bin {bin_report['bin_index']} must contain exactly one "
            f"reference-time point at {reference_years:g} years")
    return matches[0]


def _counts(records: dict[str, dict]) -> tuple[int, int]:
    accepted = sum(record["accepted"] is True for record in records.values())
    rejected = sum(record["accepted"] is False for record in records.values())
    if accepted + rejected != len(records):
        raise ValueError("every summarized record needs a Boolean disposition")
    return accepted, rejected


def _binding(records: dict[str, dict]) -> tuple[str, float]:
    candidates = []
    for parameter, record in records.items():
        tolerance = record["central"]["r_tolerance"]
        if record["accepted"] and tolerance is not None \
                and np.isfinite(tolerance) and tolerance > 0.0:
            candidates.append((float(tolerance), parameter))
    if not candidates:
        raise ValueError("reference point has no accepted finite tolerance")
    tolerance, parameter = min(candidates)
    return parameter, tolerance


def _grid_counts(bin_report: dict) -> tuple[int, int]:
    records = {
        f"{point['years']}:{parameter}": record
        for point in bin_report["points"]
        for parameter, record in point["parameters"].items()
    }
    return _counts(records)


def _bin_frequency(z_low: float, z_high: float) -> tuple[float, float]:
    return (
        HI_REST_FREQUENCY_MHZ / (1.0 + z_high),
        HI_REST_FREQUENCY_MHZ / (1.0 + z_low),
    )


def _overlap(first: tuple[float, float],
             second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _validate_evidence(path: Path, requested_family: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EVIDENCE_SCHEMA \
            or payload.get("schema_version") != 2:
        raise ValueError(f"{path.name} is not complete-v2 evidence")
    scope = payload.get("evidence_scope", {})
    if scope.get("selection") \
            != "all_physical_470_608_MHz_DTV_overlaps" \
            or scope.get("bin_count") != 7:
        raise ValueError(f"{path.name} is not an all-seven-DTV-bin export")
    if scope.get("new_telescope_data_used") is not False \
            or scope.get("empirical_visibility_template_used") is not False:
        raise ValueError(f"{path.name} is not model-only evidence")
    if not _all_true(payload.get("checks", {})):
        raise ValueError(f"{path.name} contains a failed evidence check")
    _template_identity(payload, requested_family)

    bin_indices = [int(value) for value in scope["bin_indices"]]
    if len(bin_indices) != 7 or len(set(bin_indices)) != 7:
        raise ValueError("DTV evidence bin indices must be seven unique values")
    for name in (
            "perbin_noise_normalized", "combined_noise_normalized",
            "combined_fixed_physical"):
        ledger = payload["ledgers"][name]
        if [int(value) for value in ledger["request"]["bin_indices"]] \
                != bin_indices:
            raise ValueError(f"{name} bin request differs from evidence scope")
        if set(_ledger_bins(ledger)) != set(bin_indices):
            raise ValueError(f"{name} ledger does not cover all requested bins")
    return payload


def _comparison_rows(inputs: dict[str, tuple[Path, dict]]) \
        -> tuple[list[dict], dict[int, dict]]:
    rows = []
    geometry: dict[int, dict] | None = None
    for family in FAMILIES:
        path, payload = inputs[family]
        _, provenance = _template_identity(payload, family)
        scope_bins = {
            int(item["bin_index"]): item
            for item in payload["evidence_scope"]["redshift_bins"]}
        current_geometry = {
            ibin: {
                "z_low": float(item["z_low"]),
                "z_high": float(item["z_high"]),
            }
            for ibin, item in scope_bins.items()
        }
        if geometry is None:
            geometry = current_geometry
        elif current_geometry != geometry:
            raise ValueError("template evidences disagree on redshift bins")

        perbin = payload["ledgers"]["perbin_noise_normalized"]
        combined = payload["ledgers"]["combined_noise_normalized"]
        fixed = payload["ledgers"]["combined_fixed_physical"]
        reference_values = [float(value) for value in perbin["request"]["years"]]
        if len(reference_values) != 1:
            raise ValueError("per-bin evidence must have one reference year")
        reference = reference_values[0]
        perbin_bins = _ledger_bins(perbin)
        combined_bins = _ledger_bins(combined)
        fixed_bins = _ledger_bins(fixed)

        for ibin in payload["evidence_scope"]["bin_indices"]:
            ibin = int(ibin)
            perbin_records = _reference_point(
                perbin_bins[ibin], reference)["parameters"]
            combined_records = _reference_point(
                combined_bins[ibin], reference)["parameters"]
            fixed_records = _reference_point(
                fixed_bins[ibin], reference)["parameters"]
            if set(combined_records) != set(fixed_records):
                raise ValueError("combined time families have different targets")
            fixed_matches = all(
                fixed_records[name]["accepted"] == record["accepted"]
                and np.isclose(
                    fixed_records[name]["central"]["r_tolerance"],
                    record["central"]["r_tolerance"],
                    rtol=1e-12, atol=0.0)
                for name, record in combined_records.items())
            if not fixed_matches:
                raise ValueError(
                    "fixed and noise-normalized families must coincide at "
                    "the reference time")

            perbin_accepted, perbin_rejected = _counts(perbin_records)
            combined_accepted, combined_rejected = _counts(combined_records)
            perbin_binding_parameter, perbin_binding = _binding(perbin_records)
            combined_binding_parameter, combined_binding = _binding(
                combined_records)
            perbin_fs8 = perbin_records["fs8"]
            combined_fs8 = combined_records["fs8"]
            if not perbin_fs8["accepted"] or not combined_fs8["accepted"]:
                raise ValueError(
                    f"family {family} bin {ibin} has no accepted common fs8 "
                    "target for channel propagation")
            perbin_fs8_tolerance = float(
                perbin_fs8["central"]["r_tolerance"])
            combined_fs8_tolerance = float(
                combined_fs8["central"]["r_tolerance"])
            noise_grid_accepted, noise_grid_rejected = _grid_counts(
                combined_bins[ibin])
            fixed_grid_accepted, fixed_grid_rejected = _grid_counts(
                fixed_bins[ibin])

            z_low = current_geometry[ibin]["z_low"]
            z_high = current_geometry[ibin]["z_high"]
            frequency = _bin_frequency(z_low, z_high)
            overlapping_channels = [
                channel for channel in channels.ATSC_DTV_CHANNELS
                if _overlap(frequency, channels.channel_edges(channel)) > 0.0
            ]
            rows.append({
                "schema": COMPARISON_SCHEMA,
                "family": family,
                "template_provenance": provenance,
                "evidence_file": path.name,
                "evidence_sha256": _file_sha256(path),
                "bank_sha256": payload["bank"]["sha256"],
                "numerical_grid_sha256":
                    payload["bank"]["numerical_grid_sha256"],
                "bin_index": str(ibin),
                "z_low": _format_float(z_low),
                "z_high": _format_float(z_high),
                "frequency_low_mhz": _format_float(frequency[0]),
                "frequency_high_mhz": _format_float(frequency[1]),
                "overlapping_channels": ";".join(
                    str(channel) for channel in overlapping_channels),
                "reference_years": _format_float(reference),
                "perbin_parameters": ";".join(sorted(perbin_records)),
                "perbin_accepted": str(perbin_accepted),
                "perbin_rejected": str(perbin_rejected),
                "perbin_rejected_parameters": ";".join(sorted(
                    name for name, record in perbin_records.items()
                    if not record["accepted"])),
                "perbin_binding_parameter": perbin_binding_parameter,
                "perbin_binding_tolerance": _format_float(perbin_binding),
                "combined_parameters": ";".join(sorted(combined_records)),
                "combined_accepted": str(combined_accepted),
                "combined_rejected": str(combined_rejected),
                "combined_rejected_parameters": ";".join(sorted(
                    name for name, record in combined_records.items()
                    if not record["accepted"])),
                "combined_binding_parameter": combined_binding_parameter,
                "combined_binding_tolerance": _format_float(combined_binding),
                "perbin_fs8_accepted": "true",
                "perbin_fs8_tolerance": _format_float(
                    perbin_fs8_tolerance),
                "combined_fs8_accepted": "true",
                "combined_fs8_tolerance": _format_float(
                    combined_fs8_tolerance),
                "combined_to_perbin_fs8_ratio": _format_float(
                    combined_fs8_tolerance / perbin_fs8_tolerance),
                "fixed_equals_noise_at_reference": "true",
                "combined_noise_grid_accepted": str(noise_grid_accepted),
                "combined_noise_grid_rejected": str(noise_grid_rejected),
                "combined_fixed_grid_accepted": str(fixed_grid_accepted),
                "combined_fixed_grid_rejected": str(fixed_grid_rejected),
            })
    assert geometry is not None
    return rows, geometry


def _rank_channel_rows(rows: list[dict], tolerance_field: str,
                       rank_field: str) -> None:
    by_channel: dict[str, list[dict]] = {}
    for row in rows:
        by_channel.setdefault(row["channel"], []).append(row)
    for channel_rows in by_channel.values():
        ranked = sorted(
            channel_rows, key=lambda row: float(row[tolerance_field]))
        for rank, row in enumerate(ranked, start=1):
            row[rank_field] = str(rank)


def _channel_rows(comparison_rows: list[dict], geometry: dict[int, dict]) \
        -> list[dict]:
    by_family_bin = {
        (row["family"], int(row["bin_index"])): row
        for row in comparison_rows}
    rows = []
    for family in FAMILIES:
        reference_years = {
            by_family_bin[(family, ibin)]["reference_years"]
            for ibin in geometry}
        if len(reference_years) != 1:
            raise ValueError("reference year changes between bins")
        for channel in channels.ATSC_DTV_CHANNELS:
            channel_frequency = channels.channel_edges(channel)
            overlaps = []
            for ibin, bounds in sorted(geometry.items()):
                bin_frequency = _bin_frequency(
                    bounds["z_low"], bounds["z_high"])
                overlap_mhz = _overlap(channel_frequency, bin_frequency)
                if overlap_mhz > 0.0:
                    overlaps.append((ibin, overlap_mhz))
            coverage = sum(value for _, value in overlaps) \
                / channels.ATSC_WIDTH
            if not overlaps or not np.isclose(
                    coverage, 1.0, rtol=0.0, atol=2e-12):
                raise ValueError(
                    f"channel {channel} is not completely covered by the "
                    "seven DTV-overlap bins")

            perbin_values = [
                (float(by_family_bin[(family, ibin)][
                    "perbin_fs8_tolerance"]), ibin)
                for ibin, _ in overlaps]
            combined_values = [
                (float(by_family_bin[(family, ibin)][
                    "combined_fs8_tolerance"]), ibin)
                for ibin, _ in overlaps]
            perbin_tolerance, perbin_bin = min(perbin_values)
            combined_tolerance, combined_bin = min(combined_values)
            perbin_status = "accepted" if all(
                by_family_bin[(family, ibin)]["perbin_fs8_accepted"] == "true"
                for ibin, _ in overlaps) else "rejected"
            combined_status = "accepted" if all(
                by_family_bin[(family, ibin)]["combined_fs8_accepted"] == "true"
                for ibin, _ in overlaps) else "rejected"
            rows.append({
                "schema": CHANNEL_SCHEMA,
                "family": family,
                "channel": str(channel),
                "frequency_low_mhz": _format_float(channel_frequency[0]),
                "frequency_high_mhz": _format_float(channel_frequency[1]),
                "overlap_bin_indices": ";".join(
                    str(ibin) for ibin, _ in overlaps),
                "overlap_mhz_by_bin": ";".join(
                    f"{ibin}:{_format_float(value)}"
                    for ibin, value in overlaps),
                "coverage_fraction": _format_float(coverage),
                "reference_years": next(iter(reference_years)),
                "shared_target": "fs8",
                "mapping_rule":
                    "minimum accepted tolerance over every nonzero-overlap bin",
                "perbin_status": perbin_status,
                "perbin_binding_bin": str(perbin_bin),
                "perbin_conservative_tolerance": _format_float(
                    perbin_tolerance),
                "combined_status": combined_status,
                "combined_binding_bin": str(combined_bin),
                "combined_conservative_tolerance": _format_float(
                    combined_tolerance),
                "combined_to_perbin_ratio": _format_float(
                    combined_tolerance / perbin_tolerance),
                "perbin_strictness_rank": "",
                "combined_strictness_rank": "",
                "existing_policy_status_change": "none",
                "existing_policy_ranking_change": "none",
                "policy_interpretation": (
                    "analytic sensitivity envelope only; no empirical "
                    "template-selection evidence"),
            })
    _rank_channel_rows(
        rows, "perbin_conservative_tolerance", "perbin_strictness_rank")
    _rank_channel_rows(
        rows, "combined_conservative_tolerance", "combined_strictness_rank")
    return rows


def _status_rows(inputs: dict[str, tuple[Path, dict]]) -> list[dict]:
    rows = []
    for family in FAMILIES:
        path, payload = inputs[family]
        _, provenance = _template_identity(payload, family)
        rows.append({
            "schema": STATUS_SCHEMA,
            "family": family,
            "category": "model_only_analytic",
            "execution_status": "executed_all_seven_dtv_bins",
            "evidence_file": path.name,
            "evidence_sha256": _file_sha256(path),
            "bank_sha256": payload["bank"]["sha256"],
            "template_authentication": provenance,
            "scope": (
                "perbin_noise_normalized;combined_noise_normalized;"
                "combined_fixed_physical"),
            "external_data_or_interface_required": "none",
        })
    for family, required in EMPIRICAL_REFUSALS:
        rows.append({
            "schema": STATUS_SCHEMA,
            "family": family,
            "category": "empirical_visibility",
            "execution_status": "data_dependent_incomplete",
            "evidence_file": "",
            "evidence_sha256": "",
            "bank_sha256": "",
            "template_authentication": "not_constructed",
            "scope": "explicit_refusal_no_fabricated_visibility_template",
            "external_data_or_interface_required": required,
        })
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n",
            extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", action="append", type=_parse_evidence_assignment,
        required=True, metavar="FAMILY=PATH",
        help="complete-v2 all-seven-bin evidence; repeat once per family")
    parser.add_argument(
        "--comparison-out", type=Path,
        default=Path("out/forecast_completion_template_comparison.csv"))
    parser.add_argument(
        "--channel-out", type=Path,
        default=Path("out/forecast_completion_channel_mapping.csv"))
    parser.add_argument(
        "--status-out", type=Path,
        default=Path("out/forecast_completion_template_status.csv"))
    args = parser.parse_args(argv)

    assignments = dict(args.evidence)
    if len(assignments) != len(args.evidence):
        parser.error("each evidence family must be specified exactly once")
    missing = sorted(set(FAMILIES) - set(assignments))
    extra = sorted(set(assignments) - set(FAMILIES))
    if missing or extra:
        parser.error(
            f"evidence families must be exactly {FAMILIES}; "
            f"missing={missing}, extra={extra}")
    try:
        inputs = {
            family: (assignments[family], _validate_evidence(
                assignments[family], family))
            for family in FAMILIES}
        comparison_rows, geometry = _comparison_rows(inputs)
        channel_rows = _channel_rows(comparison_rows, geometry)
        status_rows = _status_rows(inputs)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) \
            as exc:
        parser.error(str(exc))

    _write_csv(args.comparison_out, COMPARISON_FIELDS, comparison_rows)
    _write_csv(args.channel_out, CHANNEL_FIELDS, channel_rows)
    _write_csv(args.status_out, STATUS_FIELDS, status_rows)
    print(f"wrote {args.comparison_out} ({len(comparison_rows)} rows)")
    print(f"wrote {args.channel_out} ({len(channel_rows)} rows)")
    print(f"wrote {args.status_out} ({len(status_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

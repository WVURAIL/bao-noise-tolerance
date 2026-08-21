#!/usr/bin/env python3
"""Run a bounded, reproducible comparison of the completed forecast paths.

The input is a deliberately small three-time strict-v2 unit-response bank.
This command does not use pilot visibilities or any other telescope product;
it exercises only the Fisher model and the authenticated RadioFisher backend.
The output retains the complete lower/central/upper diagnostic ledger for each
requested point and adds machine-checked cross-family invariants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import bias_tolerance as bt
from baonoise import survey


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid_sha256(bank) -> str:
    """Content identity for numerical bank arrays, independent of ZIP metadata."""
    digest = hashlib.sha256(b"baonoise-fisher-grid-v1\0")
    for name in ("t_grid", "zs", "zc", "F_grid"):
        values = np.ascontiguousarray(getattr(bank, name), dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(values.shape).encode("ascii") + b"\0")
        digest.update(values.tobytes())
    digest.update(json.dumps(
        list(bank.paramnames), separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _ledger(report: dict) -> dict:
    return {
        "estimator": report["estimator"],
        "residual_amplitude": report["residual_amplitude"],
        "request": report["request"],
        "bins": report["bins"],
    }


def _portable_bank_report(report_bank: dict) -> dict:
    """Retain exact run/scientific identities while omitting absolute paths."""
    portable = json.loads(json.dumps(
        report_bank, sort_keys=True, allow_nan=False))
    portable.pop("path", None)
    return portable


EVIDENCE_SCHEMA = "baonoise-forecast-completion-evidence-v2"
EVIDENCE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs" / "forecast-completion-evidence.schema.json")


def _records(report: dict):
    for bin_report in report["bins"]:
        for point in bin_report["points"]:
            for name, record in point["parameters"].items():
                yield bin_report["bin_index"], point["years"], name, record


def _record(report: dict, ibin: int, year: float, parameter: str) -> dict:
    for actual_bin, actual_year, actual_parameter, record in _records(report):
        if (actual_bin == ibin and actual_year == year
                and actual_parameter == parameter):
            return record
    raise KeyError((ibin, year, parameter))


def _finite_close(first, second) -> bool:
    return bool(
        first is not None and second is not None
        and np.isfinite(first) and np.isfinite(second)
        and np.isclose(first, second, rtol=1e-12, atol=0.0)
    )


def _scaled_or_matching_null(first, second, multiplier: float) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return _finite_close(first, second * multiplier)


def _all_points_dispositioned(report: dict) -> bool:
    records = list(_records(report))
    requested = sum(
        bin_report["summary"]["requested_parameter_points"]
        for bin_report in report["bins"])
    return (
        len(records) == requested
        and all(isinstance(record["accepted"], bool)
                and isinstance(record["rejection_reasons"], list)
                and len(record["perturbations"]) == 3
                for _, _, _, record in records)
    )


def _time_family_scaling_checks(noise_report: dict, fixed_report: dict,
                                reference_years: float) -> dict:
    comparisons = {}
    for ibin in noise_report["request"]["bin_indices"]:
        comparisons[str(ibin)] = {}
        for year in noise_report["request"]["years"]:
            comparisons[str(ibin)][str(year)] = {}
            for parameter in noise_report["request"]["parameters"]:
                noise_record = _record(
                    noise_report, ibin, year, parameter)
                fixed_record = _record(
                    fixed_report, ibin, year, parameter)
                noise_by_label = {
                    point["label"]: point
                    for point in noise_record["perturbations"]}
                fixed_by_label = {
                    point["label"]: point
                    for point in fixed_record["perturbations"]}
                comparisons[str(ibin)][str(year)][parameter] = {}
                for label in ("lower", "central", "upper"):
                    noise = noise_by_label[label]
                    fixed = fixed_by_label[label]
                    multiplier = (
                        noise["t_hours"]
                        / (reference_years
                           * survey.OVERVIEW_ONSKY_YEAR_HOURS))
                    comparisons[str(ibin)][str(year)][parameter][label] = {
                        "validity_equal": noise["valid"] == fixed["valid"],
                        "point_acceptance_equal": (
                            noise_record["accepted"]
                            == fixed_record["accepted"]),
                        "failure_reason_equal": (
                            noise["failure_reason"]
                            == fixed["failure_reason"]),
                        "bank_native_response_equal":
                            _scaled_or_matching_null(
                                fixed["dtheta_d_current_noise_ratio"],
                                noise["dtheta_d_current_noise_ratio"], 1.0),
                        "bank_native_tolerance_equal":
                            _scaled_or_matching_null(
                                fixed["r_tolerance_current_noise_ratio"],
                                noise["r_tolerance_current_noise_ratio"],
                                1.0),
                        "fixed_response_equals_noise_response_times_t_over_t_ref":
                            _scaled_or_matching_null(
                                fixed["dtheta_d_reported_amplitude"],
                                noise["dtheta_d_reported_amplitude"],
                                multiplier),
                        "fixed_tolerance_equals_noise_tolerance_over_t_over_t_ref":
                            _scaled_or_matching_null(
                                fixed["r_tolerance"], noise["r_tolerance"],
                                1.0 / multiplier),
                        "fixed_multiplier_equals_t_over_t_ref": _finite_close(
                            fixed["time_scaling_multiplier"], multiplier),
                        "noise_multiplier_equals_one": _finite_close(
                            noise["time_scaling_multiplier"], 1.0),
                    }
    return comparisons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--radiofisher-dir", type=Path)
    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument(
        "--bin", type=int,
        help="one zero-based evidence bin (default: 6, z=1.4-1.5)")
    bin_group.add_argument(
        "--all-dtv-bins", action="store_true",
        help="export every bank bin overlapping the physical 470--608 MHz "
             "DTV band")
    parser.add_argument("--reference-years", type=float, default=1.0)
    parser.add_argument("--years", nargs="+", type=float,
                        default=[0.9, 1.0, 1.1])
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--out", type=Path,
                        default=Path("out/forecast_completion_evidence.json"))
    args = parser.parse_args(argv)

    if args.reference_years not in args.years:
        parser.error("--years must include --reference-years exactly")
    if any(not np.isfinite(year) or year <= 0.0 for year in args.years):
        parser.error("--years must contain only positive finite values")
    if len(set(args.years)) != len(args.years):
        parser.error("--years must not contain duplicates")
    if not np.isfinite(args.reference_years) or args.reference_years <= 0.0:
        parser.error("--reference-years must be positive and finite")
    if not np.isfinite(args.zeta) or args.zeta <= 0.0:
        parser.error("--zeta must be positive and finite")

    try:
        bank = bt.load_bias_bank(
            args.bank, rf_dir=args.radiofisher_dir)
    except ValueError as exc:
        parser.error(str(exc))
    bins = bt.dtv_bin_indices(bank) if args.all_dtv_bins else [
        6 if args.bin is None else args.bin]
    if (not bins or len(set(bins)) != len(bins)
            or any(ibin < 0 or ibin >= bank.nbins for ibin in bins)):
        parser.error("evidence bins must be unique valid bank-bin indices")
    grid_low = min(args.years) * (1.0 - 0.10) \
        * survey.OVERVIEW_ONSKY_YEAR_HOURS
    grid_high = max(args.years) * (1.0 + 0.10) \
        * survey.OVERVIEW_ONSKY_YEAR_HOURS
    if grid_low < bank.t_grid[0] or grid_high > bank.t_grid[-1]:
        parser.error(
            "bank time grid must bracket every +/-10% evidence evaluation: "
            f"need {grid_low:g}-{grid_high:g} hours, have "
            f"{bank.t_grid[0]:g}-{bank.t_grid[-1]:g}")

    perbin = bt.PerBinAppendixAEstimator(bank)
    combined = bt.OverviewCombinedMultibinEstimator(
        bank, rf_dir=args.radiofisher_dir)
    reference_hours = (
        args.reference_years * survey.OVERVIEW_ONSKY_YEAR_HOURS)
    common = dict(
        bank=bank, bank_path=args.bank, bins=bins, zeta=args.zeta,
        stability_fraction=0.10, max_drift=1.2)
    perbin_noise = bt.build_report(
        estimator=perbin, years=[args.reference_years],
        params=["aperp", "apar", "fs8"],
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None, **common)
    combined_noise = bt.build_report(
        estimator=combined, years=args.years, params=["DV", "F", "fs8"],
        time_scaling=bt.NOISE_NORMALIZED_AT_EACH_TIME,
        reference_hours=None, **common)
    combined_fixed = bt.build_report(
        estimator=combined, years=args.years, params=["DV", "F", "fs8"],
        time_scaling=bt.FIXED_PHYSICAL_AT_REFERENCE_TIME,
        reference_hours=reference_hours, **common)

    ledgers = {
        "perbin_noise_normalized": _ledger(perbin_noise),
        "combined_noise_normalized": _ledger(combined_noise),
        "combined_fixed_physical": _ledger(combined_fixed),
    }
    scaling_checks = _time_family_scaling_checks(
        combined_noise, combined_fixed, args.reference_years)
    disposition_checks = {
        name: _all_points_dispositioned(report)
        for name, report in (
            ("perbin_noise_normalized", perbin_noise),
            ("combined_noise_normalized", combined_noise),
            ("combined_fixed_physical", combined_fixed),
        )
    }
    bank_report = _portable_bank_report(perbin_noise["bank"])
    bank_report["numerical_grid_sha256"] = _grid_sha256(bank)

    payload = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": 2,
        "implementation": {
            "bias_tolerance_sha256": _file_sha256(Path(bt.__file__)),
            "evidence_runner_sha256": _file_sha256(Path(__file__)),
            "build_bank_wrapper_sha256": _file_sha256(
                Path(__file__).with_name("build_bank.py")),
            "evidence_schema_sha256": _file_sha256(EVIDENCE_SCHEMA_PATH),
            "report_schema": bt.REPORT_SCHEMA,
        },
        "bank": bank_report,
        "reproducibility": {
            "exact_execution_bank_sha256_retained": True,
            "scientific_content_reproducible": True,
            "archive_bytes_reproducible": False,
            "archive_variability_sources": [
                "bank build timestamp", "NPZ ZIP member timestamps"],
            "absolute_checkout_paths_omitted": True,
        },
        "evidence_scope": {
            "bin_indices": bins,
            "bin_count": len(bins),
            "redshift_bins": [
                {
                    "bin_index": ibin,
                    "z_low": float(bank.zs[ibin]),
                    "z_center": float(bank.zc[ibin]),
                    "z_high": float(bank.zs[ibin + 1]),
                }
                for ibin in bins
            ],
            "selection": (
                "all_physical_470_608_MHz_DTV_overlaps"
                if args.all_dtv_bins else "explicit_single_bin"),
            "new_telescope_data_used": False,
            "empirical_visibility_template_used": False,
            "absolute_checkout_paths_included": False,
        },
        "checks": {
            "all_requested_points_have_complete_dispositions":
                disposition_checks,
            "time_family_response_and_tolerance_scaling": scaling_checks,
        },
        "ledgers": ledgers,
    }
    if not (
            all(disposition_checks.values())
            and all(
                all(
                    all(
                        all(all(checks.values()) for checks in labels.values())
                        for labels in parameters.values())
                    for parameters in years.values())
                for years in scaling_checks.values())):
        raise RuntimeError("forecast-completion evidence invariant failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

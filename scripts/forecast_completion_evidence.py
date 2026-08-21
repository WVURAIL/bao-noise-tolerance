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


def _records(report: dict):
    for bin_report in report["bins"]:
        for point in bin_report["points"]:
            for name, record in point["parameters"].items():
                yield point["years"], name, record


def _record(report: dict, year: float, parameter: str) -> dict:
    for actual_year, actual_parameter, record in _records(report):
        if actual_year == year and actual_parameter == parameter:
            return record
    raise KeyError((year, parameter))


def _finite_close(first, second) -> bool:
    return bool(
        first is not None and second is not None
        and np.isfinite(first) and np.isfinite(second)
        and np.isclose(first, second, rtol=1e-12, atol=0.0)
    )


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
                for _, _, record in records)
    )


def _reference_agreement(noise_report: dict, fixed_report: dict,
                         reference_years: float) -> dict:
    comparisons = {}
    for parameter in noise_report["request"]["parameters"]:
        noise = _record(noise_report, reference_years, parameter)["central"]
        fixed = _record(fixed_report, reference_years, parameter)["central"]
        comparisons[parameter] = {
            "dtheta_d_reported_amplitude_equal": _finite_close(
                noise["dtheta_d_reported_amplitude"],
                fixed["dtheta_d_reported_amplitude"]),
            "r_tolerance_equal": _finite_close(
                noise["r_tolerance"], fixed["r_tolerance"]),
        }
    return comparisons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--radiofisher-dir", type=Path)
    parser.add_argument("--bin", type=int, default=6,
                        help="zero-based evidence bin (default: 6, z=1.4-1.5)")
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
        bank = bt.load_bias_bank(args.bank)
    except ValueError as exc:
        parser.error(str(exc))
    if args.bin < 0 or args.bin >= bank.nbins:
        parser.error("--bin is outside the bank")
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
        bank=bank, bank_path=args.bank, bins=[args.bin], zeta=args.zeta,
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
    reference_checks = _reference_agreement(
        combined_noise, combined_fixed, args.reference_years)
    disposition_checks = {
        name: _all_points_dispositioned(report)
        for name, report in (
            ("perbin_noise_normalized", perbin_noise),
            ("combined_noise_normalized", combined_noise),
            ("combined_fixed_physical", combined_fixed),
        )
    }
    multiplier_checks = {}
    for year in args.years:
        multiplier = _record(combined_fixed, year, "DV")["central"][
            "time_scaling_multiplier"]
        multiplier_checks[str(year)] = bool(np.isclose(
            multiplier, year / args.reference_years,
            rtol=1e-12, atol=0.0))

    payload = {
        "schema": "baonoise-forecast-completion-evidence-v1",
        "schema_version": 1,
        "implementation": {
            "bias_tolerance_sha256": _file_sha256(Path(bt.__file__)),
            "evidence_runner_sha256": _file_sha256(Path(__file__)),
            "build_bank_wrapper_sha256": _file_sha256(
                Path(__file__).with_name("build_bank.py")),
            "report_schema": bt.REPORT_SCHEMA,
        },
        "bank": {
            "filename": args.bank.name,
            "numerical_grid_sha256": _grid_sha256(bank),
            "t_grid_hours": [float(value) for value in bank.t_grid],
            "P_res": bank.meta["expt_overrides"]["P_res"],
            "baonoise_working_tree_sha256": bank.meta["provenance"]
                ["baonoise"]["working_tree_sha256"],
            "radiofisher_working_tree_sha256": bank.meta["provenance"]
                ["radiofisher"]["working_tree_sha256"],
        },
        "evidence_scope": {
            "bin_index": args.bin,
            "z_low": float(bank.zs[args.bin]),
            "z_high": float(bank.zs[args.bin + 1]),
            "new_telescope_data_used": False,
            "empirical_visibility_template_used": False,
        },
        "checks": {
            "all_requested_points_have_complete_dispositions":
                disposition_checks,
            "time_families_equal_at_reference": reference_checks,
            "fixed_physical_multiplier_equals_t_over_t_ref":
                multiplier_checks,
        },
        "ledgers": ledgers,
    }
    if not (
            all(disposition_checks.values())
            and all(all(values.values()) for values in reference_checks.values())
            and all(multiplier_checks.values())):
        raise RuntimeError("forecast-completion evidence invariant failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

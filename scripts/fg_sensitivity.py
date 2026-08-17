#!/usr/bin/env python3
"""Compare Bull-2015 masking penalties across two foreground settings."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baonoise import forecast, scenarios
from baonoise.compat import import_radiofisher
from baonoise.fisherbank import FisherBank


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIDUCIAL = (
    ROOT / "data" / "fisher_bank_bull2015_planck2013_epsfg1e-6.npz"
)
DEFAULT_COMPARISON = (
    ROOT / "data" / "fisher_bank_bull2015_planck2013_epsfg1e-5.npz"
)


def _load_bull_bank(path: Path, expected_epsilon: float) -> FisherBank:
    bank = FisherBank(path)
    if bank.meta["config"] != "bull2015" \
            or bank.meta["cosmology"] != "planck2013":
        raise ValueError(f"{path} is not a Bull-2015/Planck-2013 bank")
    settings = bank.meta["provenance"]["experiment"]["settings"]
    actual = settings.get("epsilon_fg")
    if actual != expected_epsilon:
        raise ValueError(
            f"{path} records epsilon_fg={actual!r}; expected "
            f"{expected_epsilon!r}")
    return bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fiducial-bank", type=Path, default=DEFAULT_FIDUCIAL)
    parser.add_argument("--comparison-bank", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "out" / "foreground_sensitivity.csv")
    args = parser.parse_args()

    fiducial = _load_bull_bank(args.fiducial_bank, 1e-6)
    comparison = _load_bull_bank(args.comparison_bank, 1e-5)
    if (not np.array_equal(fiducial.t_grid, comparison.t_grid)
            or not np.array_equal(fiducial.zs, comparison.zs)
            or fiducial.paramnames != comparison.paramnames):
        raise ValueError("foreground banks must use identical grids and parameters")

    rf, rf_dir = import_radiofisher()
    forecasts = {
        "epsfg1e-6": forecast.Forecast(
            fiducial, rf, style="shared_A", rf_dir=rf_dir),
        "epsfg1e-5": forecast.Forecast(
            comparison, rf, style="shared_A", rf_dir=rf_dir),
    }
    scenario_set = {
        "measured": scenarios.measured(),
        "uniform50_dtv": scenarios.uniform(0.50, scenarios.DTV_BAND),
        "uniform97_dtv": scenarios.uniform(0.97, scenarios.DTV_BAND),
    }

    clean_hours = {
        tag: fc.required_hours(scenarios.clean(), 5.0)
        for tag, fc in forecasts.items()
    }
    clean_da_hours = {
        tag: [
            fc.required_hours_metric(
                lambda t, i=i, fc=fc: fc.sigma_param_bin(
                    scenarios.clean(), t, i, "aperp0"),
                0.02,
                decreasing=True,
            )
            for i in range(fc.bank.nbins)
        ]
        for tag, fc in forecasts.items()
    }
    rows = []
    print(f"{'scenario':>15s} {'epsfg1e-6':>12s} {'epsfg1e-5':>12s} "
          f"{'shift':>10s}")
    for name, scenario in scenario_set.items():
        penalties = {}
        worst_da = {}
        for tag, fc in forecasts.items():
            penalties[tag] = (
                fc.required_hours(scenario, 5.0) / clean_hours[tag]
            )
            scenario_da_hours = [
                fc.required_hours_metric(
                    lambda t, i=i, fc=fc, scenario=scenario:
                        fc.sigma_param_bin(scenario, t, i, "aperp0"),
                    0.02,
                    decreasing=True,
                )
                for i in range(fc.bank.nbins)
            ]
            da_penalties = np.asarray(scenario_da_hours) / np.asarray(
                clean_da_hours[tag])
            ibin = int(np.argmax(da_penalties))
            worst_da[tag] = {
                "ibin": ibin,
                "z_lo": fc.bank.zs[ibin],
                "z_hi": fc.bank.zs[ibin + 1],
                "penalty": da_penalties[ibin],
            }
        shift = 100.0 * (
            penalties["epsfg1e-5"] / penalties["epsfg1e-6"] - 1.0
        )
        rows.append({
            "scenario": name,
            "fiducial_clean_hours": clean_hours["epsfg1e-6"],
            "comparison_clean_hours": clean_hours["epsfg1e-5"],
            "fiducial_penalty": penalties["epsfg1e-6"],
            "comparison_penalty": penalties["epsfg1e-5"],
            "relative_penalty_shift_pct": shift,
            "fiducial_worst_da_bin": worst_da["epsfg1e-6"]["ibin"],
            "fiducial_worst_da_z_lo": worst_da["epsfg1e-6"]["z_lo"],
            "fiducial_worst_da_z_hi": worst_da["epsfg1e-6"]["z_hi"],
            "fiducial_worst_da_penalty":
                worst_da["epsfg1e-6"]["penalty"],
            "comparison_worst_da_bin": worst_da["epsfg1e-5"]["ibin"],
            "comparison_worst_da_z_lo": worst_da["epsfg1e-5"]["z_lo"],
            "comparison_worst_da_z_hi": worst_da["epsfg1e-5"]["z_hi"],
            "comparison_worst_da_penalty":
                worst_da["epsfg1e-5"]["penalty"],
        })
        print(f"{name:>15s} {penalties['epsfg1e-6']:12.6f} "
              f"{penalties['epsfg1e-5']:12.6f} {shift:+9.3f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

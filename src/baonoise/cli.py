"""Small supported command-line surface for forecasts and bank generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__, api, resources, scenarios, survey
from .fisherbank import build_bank


def forecast_main(argv=None) -> int:
    """Evaluate one uniform masking scenario and print machine-readable JSON."""
    parser = argparse.ArgumentParser(
        prog="baonoise-forecast",
        description="Evaluate a uniform RFI-masking scenario from a Fisher bank.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    bank_group = parser.add_mutually_exclusive_group()
    bank_group.add_argument(
        "--bank", type=Path,
        help="explicit bank path (must be strict schema v2)")
    bank_group.add_argument(
        "--cosmology", choices=tuple(resources.BANK_NAMES),
        help="packaged CHIME bank (default: planck2018)")
    parser.add_argument("--uniform", type=float, required=True,
                        help="uniform masked fraction in [0,1]")
    parser.add_argument("--band", choices=("dtv", "chime"), default="dtv")
    parser.add_argument("--target", type=float, default=5.0)
    parser.add_argument("--duty", type=float, default=1.0)
    parser.add_argument(
        "--hours-per-year", type=float,
        default=survey.MEAN_CALENDAR_YEAR_HOURS,
        help="8766 for mean calendar years; 8760 for Overview on-sky years")
    args = parser.parse_args(argv)
    bank = (args.bank if args.bank is not None
            else resources.bank_file(args.cosmology or "planck2018"))
    forecast = api.load(bank)
    band = {"dtv": scenarios.DTV_BAND, "chime": scenarios.CHIME_BAND}[
        args.band]
    result = api.required_time(
        forecast, uniform=args.uniform, band=band, target=args.target,
        duty=args.duty, hours_per_year=args.hours_per_year)
    print(json.dumps(result, sort_keys=True))
    return 0


def build_bank_main(argv=None) -> int:
    """Generate a v2 bank without exposing the historical research scripts."""
    parser = argparse.ArgumentParser(
        prog="baonoise-build-bank",
        description="Generate a provenance-complete Fisher bank (slow).")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    parser.add_argument("--out", type=Path, required=True,
                        help="new output path; no implicit canonical overwrite")
    parser.add_argument("--radiofisher-dir", type=Path)
    parser.add_argument("--config", choices=("bull2015", "chime2022"),
                        default="chime2022")
    parser.add_argument(
        "--cosmology", choices=("planck2013", "planck2018", "pact2025"),
        default=None,
        help="named fiducial (default: planck2013 for bull2015, "
             "planck2018 for chime2022)")
    parser.add_argument("--tmin", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=1e6)
    parser.add_argument("--nt", type=int, default=19)
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--epsilon-fg", type=float, default=1e-6)
    parser.add_argument("--k-nl0", type=float, default=0.14)
    parser.add_argument("--kfg-fac", type=float)
    parser.add_argument("--p-res", type=float)
    args = parser.parse_args(argv)
    if args.nt < 2:
        parser.error("--nt must be at least 2")
    valid_cosmologies = {
        "bull2015": {None, "planck2013"},
        "chime2022": {None, "planck2018", "pact2025"},
    }
    if args.cosmology not in valid_cosmologies[args.config]:
        parser.error(
            f"--config {args.config} does not support --cosmology "
            f"{args.cosmology}")
    overrides = {}
    if args.kfg_fac is not None:
        overrides["kfg_fac"] = args.kfg_fac
    if args.p_res is not None:
        overrides["P_res"] = args.p_res
    t_grid = np.logspace(np.log10(args.tmin), np.log10(args.tmax), args.nt)
    build_bank(
        args.out, rf_dir=args.radiofisher_dir, t_grid_hours=t_grid,
        nproc=args.nproc, epsilon_fg=args.epsilon_fg, k_nl0=args.k_nl0,
        config=args.config, cosmology=args.cosmology,
        expt_overrides=overrides or None)
    return 0

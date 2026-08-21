#!/usr/bin/env python3
"""Build the per-zbin Fisher bank over a grid of integration times."""
import argparse
from pathlib import Path

import numpy as np

from baonoise.fisherbank import build_bank
from baonoise.residual_templates import (FAMILIES, make_template,
                                         parse_parameter_assignments)
from baonoise.resources import BANK_NAMES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--nt", type=int, default=19, help="time grid points")
    ap.add_argument("--tmin", type=float, default=1.0, help="hours")
    ap.add_argument("--tmax", type=float, default=1e6, help="hours")
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--epsilon-fg", type=float, default=1e-6)
    ap.add_argument("--config", default="chime2022",
                    choices=["bull2015", "chime2022"])
    ap.add_argument("--cosmology", default=None,
                    help="planck2013 for bull2015; planck2018 (default) or "
                         "pact2025 for chime2022")
    ap.add_argument("--dense-knee", action="store_true",
                    help="add half-step t points through the CV knee")
    ap.add_argument("--knee-range", nargs=2, type=float, default=(3.5, 5.83),
                    metavar=("LOG10_LO", "LOG10_HI"),
                    help="log10(hours) span the --dense-knee points cover; the "
                         "default reproduces every bank built before this flag "
                         "existed. Widen it to cover wherever the quantity you "
                         "are refining actually moves.")
    ap.add_argument("--knee-n", type=int, default=8,
                    help="number of --dense-knee points")
    ap.add_argument("--kfg-fac", type=float, default=None, metavar="KFG",
                    help="delay-filter mode cut, kfg_fac = tau_cut_s * survey "
                         "bandwidth_Hz (CHIME: 22 for the 55 ns first-peak-"
                         "preserving cut, 44 for 110 ns, 80 for the deployed "
                         "200 ns). Modeled on BOTH sides: the forecast loses "
                         "the cut modes and the residual chain claims the "
                         "matching suppression.")
    residual = ap.add_mutually_exclusive_group()
    residual.add_argument("--p-res", type=float, default=None,
                    metavar="AMPLITUDE",
                    help="add a residual-contamination template at this "
                         "multiple of the noise power, carried as a '_Pres' "
                         "row in the Fisher matrix. Use 1.0: the bias is "
                         "linear in the amplitude, so one build serves every "
                         "r, and scripts/bias_tolerance.py assumes unit "
                         "normalization.")
    residual.add_argument(
        "--residual-template", choices=FAMILIES,
        help="named unit-amplitude analytic P_res family; unlike an empirical "
             "visibility template, these use only the (k,u,P_N,P_signal) "
             "coordinates available to RadioFisher")
    ap.add_argument(
        "--template-param", action="append", default=[], metavar="NAME=VALUE",
        help="override one named analytic-template parameter; repeat as needed")
    args = ap.parse_args()

    ctag = (f"_{args.cosmology}" if args.cosmology
            and args.cosmology != "planck2018" else "")
    # A P_res bank carries an extra parameter row and is not interchangeable
    # with a plain one, so it never lands on the plain default name.
    if args.residual_template is not None:
        ptag = f"_pres_{args.residual_template}"
    else:
        ptag = "_pres" if args.p_res is not None else ""
    if args.kfg_fac is not None:
        ptag += f"_kfg{args.kfg_fac:g}"
    ktag = "_dense" if args.dense_knee else ""
    fg_tag = (f"{args.epsilon_fg:.0e}"
              .replace("e-0", "e-").replace("e+0", "e+"))
    default_name = (
        f"fisher_bank_chime2022{ctag}{ptag}{ktag}.npz"
        if args.config == "chime2022"
        else f"fisher_bank_bull2015_planck2013_epsfg{fg_tag}{ptag}{ktag}.npz")
    if args.out:
        out = args.out
    elif default_name in BANK_NAMES.values():
        # Named CHIME-2022 cosmologies are package resources. The caller
        # remains responsible for the intended grid and scientific settings.
        out = (Path(__file__).resolve().parents[1] / "src" / "baonoise" /
               "data" / default_name)
    else:
        out = Path(__file__).resolve().parents[1] / "data" / default_name
    tgrid = np.logspace(np.log10(args.tmin), np.log10(args.tmax), args.nt)
    if args.dense_knee:
        lo, hi = args.knee_range
        extra = 10.0 ** np.linspace(lo, hi, args.knee_n)
        tgrid = np.unique(np.concatenate([tgrid, extra]))
    overrides = {}
    if args.p_res is not None:
        overrides["P_res"] = args.p_res
    if args.residual_template is not None:
        try:
            template_parameters = parse_parameter_assignments(
                args.template_param)
            overrides["P_res"] = make_template(
                args.residual_template, template_parameters)
        except (TypeError, ValueError) as exc:
            ap.error(str(exc))
    elif args.template_param:
        ap.error("--template-param requires --residual-template")
    if args.kfg_fac is not None:
        overrides["kfg_fac"] = args.kfg_fac
    overrides = overrides or None
    print(f"[bank] {len(tgrid)} time points, {tgrid[0]:.3g}-{tgrid[-1]:.3g} hr"
          + (f", P_res={overrides.get('P_res')}"
             if overrides and "P_res" in overrides else ""))
    build_bank(out, t_grid_hours=tgrid, nproc=args.nproc,
               epsilon_fg=args.epsilon_fg, config=args.config,
               cosmology=args.cosmology, expt_overrides=overrides)

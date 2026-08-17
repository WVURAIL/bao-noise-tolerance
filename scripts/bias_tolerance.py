#!/usr/bin/env python
"""The residual a BAO analysis can absorb before it is biased rather than just noisier.

Every cost in the noise-tolerance framework so far has been a variance: a
residual raises sigma and the survey integrates longer. That accounting misses
the failure mode that matters. A residual coherent enough to survive the ground
filter does not average down, so it shifts the recovered acoustic dilations
rather than widening them, and a shift does not care how long you integrate.
sigma_alpha falls as t^-1/2 against a fixed Delta-alpha, so any coherent
residual eventually dominates.

This evaluates the first-order bias

    Delta-theta_i = Delta-A * sum_j (F_theta-theta^-1)_ij F_j,A

from a Fisher bank carrying the '_Pres' row (built with expt['P_res'] = 1, so
the template amplitude is in units of the system noise power and Delta-A is
just r), and inverts it for the tolerance

    r_tol(T) = max { r : |Delta-alpha(r)| <= zeta * sigma_alpha(T) }.

r_tol is what closes the loop of the whole framework: it fixes the science
contamination tolerance, the proxy margin converts that to a per-channel
threshold, and the threshold fixes (f, r) and hence the time penalty.

Bias-response banks are intentionally not distributed. Build the exact
strict-v2, unit-response prerequisite and then run:

    python scripts/build_bank.py --config chime2022 --cosmology planck2018 \\
        --p-res 1.0 --dense-knee \\
        --out data/fisher_bank_chime2022_pres_dense.npz
    python scripts/bias_tolerance.py \\
        --bank data/fisher_bank_chime2022_pres_dense.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from baonoise import channels, survey
from baonoise.constants import HI_REST_FREQUENCY_MHZ
from baonoise.fisherbank import ARTIFACT_BIAS_RESPONSE, FisherBank

PRES = "_Pres"
DEFAULT_BIAS_BANK = ROOT / "data" / "fisher_bank_chime2022_pres_dense.npz"
DEFAULT_BUILD_COMMAND = (
    "python scripts/build_bank.py --config chime2022 "
    "--cosmology planck2018 --p-res 1.0 --dense-knee "
    "--out data/fisher_bank_chime2022_pres_dense.npz"
)
_ANY_KFG = object()
# The per-bin parameter set, identical to the one the forecast marginalises
# over: {A, sigma_NL, aperp, apar, bs8, fs8}. Keeping b_HI, f and sigma8tot
# alongside the derived bs8/fs8 combinations makes the matrix rank-deficient
# by construction; the first pass here did exactly that and produced a
# condition number of 1e17 (double precision is ~1e16), with d(aperp)/dr
# scattering over four orders of magnitude and changing sign. Those were null
# vectors rather than physics.
EXCLUDE = ("b_HI", "f", "Tb", "sigma8tot", "n_s", "pk")


def _exact_numeric(value, expected):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and float(value) == float(expected))


def load_bias_bank(path, *, build_command=DEFAULT_BUILD_COMMAND,
                   expected_kfg_fac=_ANY_KFG):
    """Load an explicitly generated strict-v2 unit-response bank."""
    path = Path(path)
    instruction = (
        "Bias-response banks are deliberately not shipped. Build the exact "
        "prerequisite with:\n  " + build_command
    )
    if not path.is_file():
        raise ValueError(f"required bias-response bank is missing: {path}\n"
                         f"{instruction}")
    try:
        bank = FisherBank(path)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{path} is not a valid strict-v2 Fisher bank: {exc}\n"
            f"{instruction}") from exc

    overrides = bank.meta.get("expt_overrides")
    provenance_settings = bank.meta["provenance"]["experiment"]["settings"]
    problems = []
    if bank.artifact_kind != ARTIFACT_BIAS_RESPONSE or PRES not in bank.paramnames:
        problems.append("artifact_kind must be 'bias_response' with a '_Pres' row")
    if bank.meta.get("config") != "chime2022" \
            or bank.meta.get("cosmology") != "planck2018":
        problems.append("configuration must be chime2022/planck2018")
    if bank.meta.get("astrophysical_model_profile") != "chime_overview_2022":
        problems.append("the canonical chime_overview_2022 profile is required")
    if (not isinstance(overrides, dict)
            or not _exact_numeric(overrides.get("P_res"), 1.0)):
        problems.append("expt_overrides.P_res must equal the unit response 1.0")
    if not _exact_numeric(provenance_settings.get("P_res"), 1.0):
        problems.append("experiment provenance must record P_res=1.0")
    actual_kfg = (overrides.get("kfg_fac")
                  if isinstance(overrides, dict) else None)
    if expected_kfg_fac is _ANY_KFG:
        kfg_matches = True
    elif expected_kfg_fac is None:
        kfg_matches = actual_kfg is None
    else:
        kfg_matches = _exact_numeric(actual_kfg, expected_kfg_fac)
    if not kfg_matches:
        problems.append(
            f"expt_overrides.kfg_fac must equal {expected_kfg_fac!r}")
    if problems:
        raise ValueError(
            f"{path} is incompatible with the bias workflow: "
            + "; ".join(problems) + f"\n{instruction}")
    return bank


def split(F, names, targets=("aperp", "apar", "fs8")):
    """(F_theta-theta, F_theta-A, kept names) with zero-information rows cut."""
    ia = names.index(PRES)
    keep = [i for i, n in enumerate(names)
            if n != PRES and n not in EXCLUDE and F[i, i] > 0.0]
    Ftt = F[np.ix_(keep, keep)]
    Ftt = 0.5 * (Ftt + Ftt.T)
    FtA = F[keep, ia]
    return Ftt, FtA, [names[i] for i in keep]


def condition(F, names):
    Ftt, _, _ = split(F, names)
    return float(np.linalg.cond(Ftt))


def stability(bank, ib, t_hours, names, param, frac=0.10):
    """How much r_tol moves under a +/-10% perturbation of the integration time.

    The bias estimate is a ratio of a marginalised error to a cross-term, and
    the cross-term passes through zero as sample variance overtakes the noise.
    Near that crossing the ratio is a difference of cancelling quantities and
    the bank's interpolation cannot be trusted, so the answer is checked rather
    than reported: a tolerance that moves under a small change of t is not a
    tolerance, it is a null direction. This is the same refusal discipline the
    correlation-time estimator uses.
    """
    vals, signs = [], []
    for scale in (1.0 - frac, 1.0, 1.0 + frac):
        dth, sig = bias_per_unit_r(bank.F(ib, t_hours * scale), names)
        if param not in dth or dth[param] == 0.0:
            return np.inf, 0
        vals.append(sig[param] / abs(dth[param]))
        signs.append(np.sign(dth[param]))
    lo, hi = min(vals), max(vals)
    return (hi / lo if lo > 0 else np.inf), len(set(signs))


def bias_per_unit_r(F, names):
    """d(theta)/dA and sigma(theta), for one bin's Fisher matrix."""
    Ftt, FtA, kept = split(F, names)
    try:
        cov = np.linalg.inv(Ftt)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Ftt)
    dtheta = cov @ FtA
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return dict(zip(kept, dtheta)), dict(zip(kept, sigma))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bank", type=Path, default=DEFAULT_BIAS_BANK,
        help="strict-v2 unit-P_res bias-response bank (not shipped; exact "
             f"build prerequisite: {DEFAULT_BUILD_COMMAND})")
    ap.add_argument("--zeta", type=float, default=0.3,
                    help="admissible bias as a fraction of the statistical error")
    ap.add_argument("--params", nargs="+", default=["aperp", "apar", "fs8"])
    ap.add_argument("--years", nargs="+", type=float,
                    default=[0.25, 1.0, 5.0, 10.0])
    ap.add_argument("--max-drift", type=float, default=1.2,
                    help="largest r_tol ratio across a +/-10%% change of t "
                         "before the entry is refused")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        bank = load_bias_bank(args.bank)
    except ValueError as exc:
        ap.error(str(exc))
    names = list(bank.paramnames)
    meta = bank.meta
    print(f"bank {args.bank.name}  P_res={meta.get('expt_overrides')}  "
          f"zeta={args.zeta}")

    # the DTV band, 470-608 MHz, in redshift
    nu21 = HI_REST_FREQUENCY_MHZ
    z_dtv = (nu21 / channels.ATSC_DTV_UPPER_EDGE - 1.0,
             nu21 / channels.ATSC_CH14_LOWER_EDGE - 1.0)
    zs = bank.zs
    dtv_bins = [i for i in range(len(zs) - 1)
                if zs[i + 1] > z_dtv[0] and zs[i] < z_dtv[1]]
    print(f"DTV band z = {z_dtv[0]:.3f}-{z_dtv[1]:.3f}  ->  bins "
          f"{[f'{zs[i]:.2f}-{zs[i+1]:.2f}' for i in dtv_bins]}\n")

    out = []
    for ib in dtv_bins:
        print(f"z = {zs[ib]:.2f}-{zs[ib+1]:.2f}")
        print(f"  {'T (on-sky yr)':>14} " +
              "".join(f"{p:>14}" for p in args.params) +
              "      ('!' = refused, moves under a 10% change of t)")
        rows = {}
        for yr in args.years:
            t = yr * survey.OVERVIEW_ONSKY_YEAR_HOURS
            F = bank.F(ib, t)
            dth, sig = bias_per_unit_r(F, names)
            cells, rec = [], {}
            for p in args.params:
                if p not in dth or dth[p] == 0.0:
                    cells.append(f"{'--':>14}")
                    continue
                r_tol = args.zeta * sig[p] / abs(dth[p])
                drift, nsign = stability(bank, ib, t, names, p)
                ok = (drift <= args.max_drift) and nsign == 1
                rec[p] = dict(r_tol=float(r_tol), sigma=float(sig[p]),
                              dtheta_dr=float(dth[p]), drift=float(drift),
                              stable=bool(ok))
                cells.append(f"{r_tol:12.3g}{'  ' if ok else ' !'}")
            print(f"  {yr:14.2f} " + "".join(cells))
            rows[yr] = rec
        stable = [v["r_tol"] for rec in rows.values() for v in rec.values()
                  if v["stable"]]
        nun = sum(1 for rec in rows.values() for v in rec.values()
                  if not v["stable"])
        binding = min(stable) if stable else float("nan")
        if stable:
            print(f"  binding tolerance over the stable entries: "
                  f"r <= {binding:.3g}   ({nun} entries refused as unstable)")
        else:
            print(f"  REFUSED: every entry moves under a 10% change of t; "
                  f"no tolerance can be quoted here")
        print(f"  cond(F) at 1 yr: "
              f"{condition(bank.F(ib, survey.OVERVIEW_ONSKY_YEAR_HOURS), names):.2e}\n")
        out.append(dict(zlo=float(zs[ib]), zhi=float(zs[ib + 1]),
                        rows={str(k): v for k, v in rows.items()},
                        binding=float(binding)))

    if args.json:
        args.json.write_text(json.dumps(
            dict(zeta=args.zeta, bank=args.bank.name, bins=out), indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

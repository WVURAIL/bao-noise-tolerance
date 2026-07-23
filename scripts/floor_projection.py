#!/usr/bin/env python3
"""What would have to be true for the mask to clear the bias tolerance.

The measured verdict is negative. On every channel with a measured kept-frame
floor, the pilot-proxy mask leaves a residual above the tolerance, because
the mask cannot remove contamination the detector cannot see, so the residual
is floored by detection sensitivity, and after coherence amplification that
floor is still too high.

That is a statement about the *bootstrap coarse rule*, which is what produced
the survey products: it decides on the coarse axis, roughly 10 dB less
sensitive than the fine coherent stage the detector is designed around. This
script asks what the remaining gap is in terms anyone can check, by walking
each channel through the improvements that are actually on the table and
marking where it crosses.

Every step here is a *conditional* rather than a measurement, and the output labels
them as such. The point is not to rescue the verdict; it is to say precisely
how far away it is and which of the open items would close it.

    python3 scripts/floor_projection.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import products as _products  # noqa: E402
from baonoise import residual as R                       # noqa: E402

# The bias tolerance at the published criterion, b <= sigma (Amara &
# Refregier 2008), binding on f-sigma-8 across the DTV redshift bins.
R_TOL = 1.5e-3

# Measured coherent-integration gain of the fine axis over the coarse axis,
# from the Monte-Carlo characterisation through the deployed geometry
# (scalloping, window maximum, finite-P_fa thresholds included). The ideal
# sqrt(L) bound is 10.5 dB; the measured range is quoted here instead, because
# a projection should not spend gain the implementation has not demonstrated.
FINE_GAIN_DB = (9.4, 10.0)


def channel_state(npz):
    st = R.shelf_statistics(npz)
    corr = R.correlation_time(npz)
    gain = (st.intraday_fraction
            * R.n_coh_from_correlation_time(corr.tau_for_budget)
            + st.fast_fraction)
    return st, corr, gain


def r_at(floor_db, gain, extra_db=0.0):
    return 10.0 ** ((floor_db - extra_db) / 10.0) * gain


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+",
                    default=[p for _c, p in sorted(_products.paths(
                        channels=(32, 33, 34, 35, 36),
                        announce=False).items())])
    args = ap.parse_args(argv)

    print(f"bias tolerance r_tol = {R_TOL:.3g}  (zeta = 1, Amara & Refregier)")
    print(f"fine-stage gain over coarse: {FINE_GAIN_DB[0]}-{FINE_GAIN_DB[1]} dB "
          f"(measured rather than the {10.5:.1f} dB ideal)\n")

    measured, unmeasured = [], []
    for path in args.products:
        st, corr, gain = channel_state(path)
        (measured if np.isfinite(st.floor_db) else unmeasured).append(
            (st, corr, gain))

    print("MEASURED FLOORS: these carry verdicts")
    print(f"  {'ch':>3} {'tau_c':>14} {'gain':>10} {'coarse (now)':>14} "
          f"{'+fine stage':>16} {'+fine +1st peak':>17} {'+fine +2nd peak':>17}")
    for st, corr, gain in measured:
        row = [f"  {st.channel:3d}"]
        tau = (f"{corr.tau_for_budget/60:.0f} min" if corr.tau_for_budget < 3600
               else f"{corr.tau_for_budget/3600:.1f} h")
        row.append(f"{tau + ('' if corr.quality == 'measured' else '*'):>14}")
        row.append(f"{gain:10.4g}")
        for extra in (0.0, FINE_GAIN_DB[1], FINE_GAIN_DB[1] + 3.6,
                      FINE_GAIN_DB[1] + 8.2):
            r = r_at(st.floor_db, gain, extra)
            mark = "PASS" if r <= R_TOL else f"x{r/R_TOL:,.0f}"
            row.append(f"{r:8.3g} {mark:>7s}")
        print(" ".join(row))
    print("  * tau_c is a bound (refused or bounded-above), so the row is a "
          "bound on r rather than a measurement")

    print("\nUNMEASURED FLOORS: no verdict is available")
    for st, corr, gain in unmeasured:
        need = 10.0 * np.log10(R_TOL / gain)
        print(f"  ch{st.channel}: no null population ({st.n_off_frames} frames). "
              f"Would need a floor below {need:.1f} dB to clear the tolerance "
              f"at the present gain of {gain:.4g}.")

    # ---- the other lever: tau_c ----------------------------------------
    print("\nTHE OTHER LEVER. Two channels sit at the sidereal cap because "
          "tau_c was refused.\nSubstituting the one bound that was measured "
          "(ch33, tau_c <= 5 min) in place of the cap:")
    tau33 = 300.0
    n33 = R.n_coh_from_correlation_time(tau33)
    for st, corr, gain in measured:
        if corr.quality == "measured" or corr.tau_for_budget <= tau33:
            continue
        g2 = st.intraday_fraction * n33 + st.fast_fraction
        for label, extra in (("as-is", 0.0), ("+fine", FINE_GAIN_DB[1])):
            r = r_at(st.floor_db, g2, extra)
            mark = "PASS" if r <= R_TOL else f"x{r/R_TOL:,.0f} over"
            print(f"  ch{st.channel} with tau_c <= 5 min, {label:6s}: "
                  f"gain {g2:9.4g}  r = {r:9.3g}  {mark}")
    print("\nThat substitution is a what-if rather than a measurement. It is listed "
          "because it identifies\nwhich open item moves the verdict most: on "
          "these channels tau_c does rather than the floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

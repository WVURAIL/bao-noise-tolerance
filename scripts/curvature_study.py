#!/usr/bin/env python3
"""Measure the local background polynomial around each pilot, and price the
declined +/-2,+/-4 Richardson reference design against measurement.

The symmetric pair cancels odd orders exactly; what survives is curvature:
the pair's background estimate is biased by  eps = a2*Delta^2  (fraction of
the local level) at reference distance Delta = 6.104 kHz. The Richardson
design (weights 4/3, -1/3 at +/-2, +/-4 bins) would null a2 and leave
-4*a4*Delta^4. This script fits the archive-averaged within-channel
spectrum (products' integrated_spectrum_before_mask) on the flanks
5-12 kHz either side of the pilot (excluding the pilot cluster and any
secondary forest) and reports each term as a fraction of the background,
to be compared against the eta - 1 = 1% decision knee and the measured
null widths.

    python3 scripts/curvature_study.py
"""
from __future__ import annotations
import numpy as np
DF = 390.625e3 / 16384            # 23.84 Hz per spectrum bin
DELTA = 2 * 390.625e3 / 128       # 6.104 kHz reference distance
WIN = 390.625e3 / 128             # 3.05 kHz window integration width
FIT_IN, FIT_OUT = 5e3, 12e3       # fit annulus: past the cluster, local

from baonoise import products as P
from baonoise.npzio import load_npz
PRODUCTS = P.paths()


def main():
    from scipy.ndimage import median_filter
    print(f"reference distance Delta = {DELTA/1e3:.3f} kHz; fit annulus "
          f"{FIT_IN/1e3:.0f}-{FIT_OUT/1e3:.0f} kHz both sides\n")
    print(f"{'ch':>3} {'slope a1*D':>11} {'curv a2*D^2':>12} "
          f"{'rich 4a4*D^4':>13} {'direct pair':>12} {'clip%':>6}"
          "   (fractions of local background)")
    res = []
    for ch, p in sorted(PRODUCTS.items()):
        d = load_npz(p)
        s = d["integrated_spectrum_before_mask"].astype(float)
        dfhz = (float(d["pilot_frequency_hz"][0])
                - float(d["chime_frequency_hz"][0]))
        nom = int(round(int(d["sense"]) * dfhz / DF)) % 16384
        lo, hi = max(0, nom - 209), min(16384, nom + 209)
        pk = lo + int(np.argmax(s[lo:hi]))   # metadata-anchored: the frame
        # DC artifact at bin 0 can exceed the archive-averaged pilot
        off = (np.arange(16384) - pk) * DF
        sm = median_filter(s, 21)            # suppress narrow lines
        idx = np.arange(16384)
        m = ((np.abs(off) >= FIT_IN) & (np.abs(off) <= FIT_OUT)
             & (idx > 100) & (idx < 16284))  # keep clear of the DC artifact
        x, y = off[m] / DELTA, sm[m]
        b0 = np.median(y)
        yn = y / b0
        keep = np.ones(len(x), bool)
        for _ in range(4):                   # sigma-clip residual structure
            c = np.polynomial.polynomial.polyfit(x[keep], yn[keep], 4)
            r = yn - np.polynomial.polynomial.polyval(x, c)
            keep = np.abs(r) < 3 * np.std(r[keep])
        a1, a2, a4 = c[1], c[2], c[4]

        def wavg(f0):
            w = ((np.abs(off - f0) <= WIN / 2) & (idx > 100) & (idx < 16284))
            return s[w].mean() if w.any() else np.nan

        center = np.polynomial.polynomial.polyval(0.0, c) * b0
        direct = (0.5 * (wavg(DELTA) + wavg(-DELTA)) - center) / center
        res.append((ch, a1, a2, -4 * a4, direct))
        print(f"{ch:>3} {a1:>11.2e} {a2:>12.2e} {-4*a4:>13.2e} "
              f"{direct:>12.2e} {1-keep.mean():>6.1%}")
    a2s = [abs(r[2]) for r in res]
    ds = [abs(r[4]) for r in res]
    print(f"\nsmooth curvature |a2*D^2|: median {np.median(a2s):.1e}, worst "
          f"{max(a2s):.1e} (ch{res[int(np.argmax(a2s))][0]});  knee = 1e-2")
    print(f"direct pair bias:          median {np.median(ds):.1e}, worst "
          f"{max(ds):.1e} (ch{res[int(np.argmax(ds))][0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

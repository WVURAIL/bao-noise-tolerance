#!/usr/bin/env python3
"""Per-channel distributions of the decision statistic and the reported level.

One row per channel. The left panel is the coarse statistic F near its null;
the right panel is the shelf level the product reports for each frame. Both
are log-count, because the interesting parts are the tails.

Two vertical rules matter on the left panel, and the whole plate exists to
show where they sit relative to one another:

    F = mu0    the decision line. mu0 = 2||w0||^2 / (||w1||^2 + ||w2||^2) is an
               exact rational constant fixed by the quantised weight bank, and
               the deployed rule is reject <=> F > mu0.

    F = 1      where the *level* estimator becomes defined. The product sets
               pnr_bin_db = 10 log10(F - 1) exactly, so it references the null
               to unity rather than to mu0.

Those are not the same number, and on two of the five channels they are in the
opposite order. Everything the right panel can and cannot say follows from
that, including the missing floors on channels 35 and 36, which are an
arithmetic consequence of mu0 < 1 rather than a statement about the
transmitter.

Drawing is left to baonoise.plots, which carries the manuscript's figure
style; this script only prepares the per-channel arrays it takes.

    python3 scripts/plot_channel_histograms.py --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


from baonoise import plots
from baonoise import residual as R

# 470-608 MHz DTV allocations, by physical channel number.
ALLOCATION_MHZ = {32: "578-584", 33: "584-590", 34: "590-596",
                  35: "596-602", 36: "602-608"}

from baonoise import products as _P
from baonoise.npzio import load_npz

DEFAULT_PRODUCTS = [p for _c, p in sorted(
    _P.paths(channels=(32, 33, 34, 35, 36), announce=False).items())]


def channel_row(path) -> dict:
    """Everything the figure needs, with the provenance checks run first.

    floor_provenance re-derives the shelf offset from the product and verifies
    both the deployed rule and the level formula before answering, so the
    numbers annotated on the plate are checked rather than assumed.
    """
    prov = R.floor_provenance(path)
    d = load_npz(path)
    v = d["valid"][:, 0].astype(bool)
    rej = d["reject_mask"][:, 0].astype(bool)
    F = d["fstat_raw"][:, 0]
    shelf = d["snr_shelf_db"][:, 0]
    mu0 = prov.mu0

    kept, hit = v & ~rej, v & rej
    lev = v & np.isfinite(shelf)
    band = v & (F > min(1.0, mu0)) & (F <= max(1.0, mu0))

    return dict(
        ch=prov.channel, freq_id=prov.freq_id,
        band=ALLOCATION_MHZ.get(prov.channel, "?"),
        mu0=mu0, mu0_sigma=(mu0 - 1.0) / prov.sigma_null,
        x=(F - 1.0) / prov.sigma_null, shelf=shelf,
        kept=kept, hit=hit, sliver=kept & lev, hit_lev=hit & lev,
        n_valid=int(v.sum()), n_kept=prov.n_kept, n_sliver=prov.n_sliver,
        n_band=int(band.sum()), n_undef=int((v & ~lev).sum()),
        f_masked=float(rej[v].mean()),
        floor_db=prov.reported_db, floor_mu0_db=prov.mu0_implied_db,
        sigma=prov.sigma_null, sigma_spread=prov.sigma_spread,
        floor_sigma_db=prov.sigma_implied_db, verdict=prov.verdict,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--singles", action="store_true",
                    help="also write one figure per channel")
    args = ap.parse_args(argv)

    rows = [channel_row(p) for p in args.products]
    args.out.mkdir(parents=True, exist_ok=True)

    out = plots.fig_channel_histograms(rows, args.out / "fig5_channel_histograms.png")
    print(f"wrote {out} (+ .pdf)")
    if args.singles:
        for r in rows:
            plots.fig_channel_histogram(
                r, args.out / f"fig5{chr(ord('a') + r['ch'] - 32)}_hist_ch{r['ch']}.png")
        print(f"wrote {args.out}/fig5*_hist_ch*.png (+ .pdf)")

    print(f"\n{'ch':>3} {'mu0':>12} {'masked':>7} {'kept':>7} {'sliver':>7} "
          f"{'no level':>9} {'floor':>8} {'10lg(mu0-1)':>12} "
          f"{'sigma_null':>11} {'x-spread':>8} {'floor(sig)':>10}")
    for r in rows:
        print(f"{r['ch']:3d} {r['mu0']:12.9f} {r['f_masked']*100:6.1f}% "
              f"{r['n_kept']:7,d} {r['n_sliver']:7,d} {r['n_undef']:9,d} "
              f"{r['floor_db']:8.2f} {r['floor_mu0_db']:12.2f} "
              f"{r['sigma']:11.3e} {r['sigma_spread']:8.1f} "
              f"{r['floor_sigma_db']:10.2f}")
    print()
    for r in rows:
        print(f"  ch{r['ch']}: {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

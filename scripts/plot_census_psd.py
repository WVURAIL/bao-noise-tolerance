#!/usr/bin/env python3
"""Per-channel time-averaged PSD around each pilot, with the K = 128
analysis window and reference placements overlaid, the census's spectral
face, and the direct visual for the fine-span sizing of the K bracket.

Data: each survey product's integrated_spectrum_before_mask (the archive
average of every frame's 16384-point within-channel spectrum).

    python3 scripts/plot_census_psd.py --out /tmp/diss/figs
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baonoise.plots import (GRID, INK, INK2, MUTED, SERIES, SURFACE,
                            CRITICAL, _save, setup_style)  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from baonoise import products as P                        # noqa: E402
PRODUCTS = P.paths()
FS = 390.625e3
NFFT = 16384
DF = FS / NFFT                      # 23.84 Hz per bin
SPAN_HZ = 15e3                      # plotted neighborhood
K_SPAN = FS / 256 / 2               # +/-1.526 kHz fine span (K = 128)
REF_HZ = 2 * FS / 128               # +/-6.104 kHz reference displacement


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    n = len(PRODUCTS)
    nrows = (n + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(9.4, 2.5 * nrows),
                             sharex=True, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, (ch, path) in zip(axes.ravel(), sorted(PRODUCTS.items())):
        d = np.load(path, allow_pickle=True)
        s = d["integrated_spectrum_before_mask"]
        # anchor at the metadata pilot position (DC sits at bin 0 and can
        # exceed the pilot in the archive average), refined to the local
        # maximum within +/-5 kHz of nominal
        dfhz = float(d["pilot_frequency_hz"][0]) - float(d["chime_frequency_hz"][0])
        nom = int(round(int(d["sense"]) * dfhz / DF)) % NFFT
        lo, hi = max(0, nom - int(5e3/DF)), min(NFFT, nom + int(5e3/DF))
        pk = lo + int(np.argmax(s[lo:hi]))
        w = int(SPAN_HZ / DF)
        idx = np.arange(pk - w, pk + w + 1)
        good = (idx >= 0) & (idx < NFFT)
        f_khz = (idx[good] - pk) * DF / 1e3
        p = s[idx[good]]
        p_db = 10 * np.log10(np.maximum(p, p[p > 0].min()) / np.median(p))
        ax.axvspan(-K_SPAN/1e3, K_SPAN/1e3, color=SERIES[0], alpha=0.10, lw=0)
        for r in (-REF_HZ, REF_HZ):
            ax.axvline(r/1e3, color=SERIES[2], lw=1.0, ls=(0, (4, 3)))
        ax.plot(f_khz, p_db, color=INK, lw=0.7)
        ax.annotate(f"ch {ch}", xy=(0.04, 0.86), xycoords="axes fraction",
                    fontsize=10, color=INK, fontweight="semibold")
        ax.set_xlim(-SPAN_HZ/1e3, SPAN_HZ/1e3)
        ax.grid(True, color=GRID, lw=0.4)
    for k in range(3):
        col = axes[:, k]
        vis = [a for a in col if a.get_visible()]
        if vis:
            vis[-1].set_xlabel("offset from pilot peak [kHz]")
            vis[-1].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("dB rel. median")
    fig.suptitle("Archive-averaged spectrum around each pilot: main lobe, "
                 "secondary sidelobes, and the $K=128$ geometry\n"
                 "(shaded: the $\\pm 1.526$ kHz fine span; dashed: the $\\pm 2$-bin "
                 "reference placements at $\\pm 6.1$ kHz)", fontsize=10.5, y=0.995)
    fig.subplots_adjust(hspace=0.10, wspace=0.22, top=0.92)
    return _save(fig, args.out / "fig_census_psd.png")


if __name__ == "__main__":
    print("wrote", main())

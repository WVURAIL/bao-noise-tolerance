#!/usr/bin/env python3
"""The footprint geometry: one bin measured, the whole allocation protected.

The pilot occupies a single CHIME coarse channel; the payload it announces
spreads across the whole 6 MHz allocation. That asymmetry is the method's
leverage and it is easier to see than to describe, so it gets a figure.

  (a) one allocation, at the receiver's own raster: the pilot's bin against
      the fourteen it speaks for, with the shelf drawn at the level the proxy
      relation infers
  (b) the same geometry at band scale: 23 monitored bins against the 353 the
      DTV allocation covers

    python3 scripts/plot_footprint.py --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from baonoise import channels as chn
from baonoise.constants import (CHIME_COARSE_CHANNEL_COUNT,
                                CHIME_FREQUENCY_MAX_MHZ,
                                CHIME_FREQUENCY_MIN_MHZ,
                                HI_REST_FREQUENCY_MHZ)

# Same validated categorical slots as the residual figure:
#   node scripts/validate_palette.js "#2a78d6,#eb6834" --pairs all
#   ALL CHECKS PASS (worst normal-vision dE 33.6, worst CVD dE 24.7)
PROTECTED, MONITORED = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"

FS = ((CHIME_FREQUENCY_MAX_MHZ - CHIME_FREQUENCY_MIN_MHZ)
      / CHIME_COARSE_CHANNEL_COUNT)
ATSC_GUARD = 0.31          # MHz, allocation edge to payload edge
ATSC_PAYLOAD = 5.38        # MHz, RRC Nyquist passband
PILOT_TO_SHELF_INTEGRATED_RATIO = 13.4


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def bin_index(nu):
    """CHIME coarse-channel index whose centre is nearest nu (MHz)."""
    return int(round((CHIME_FREQUENCY_MAX_MHZ - nu) / FS))


def bin_center(idx):
    return CHIME_FREQUENCY_MAX_MHZ - idx * FS


def panel_allocation(ax, ch=35):
    """One 6 MHz allocation at the receiver's raster."""
    lo, hi = chn.channel_edges(ch)
    pilot = lo + ATSC_GUARD
    pay_lo, pay_hi = lo + ATSC_GUARD, lo + ATSC_GUARD + ATSC_PAYLOAD
    pilot_bin = bin_index(pilot)

    # every coarse bin overlapping the allocation
    idx = [i for i in range(bin_index(hi) - 1, bin_index(lo) + 2)
           if bin_center(i) + FS / 2 > lo and bin_center(i) - FS / 2 < hi]

    shelf_h = 1.0                      # per-bin shelf, arbitrary linear units
    pilot_h = (shelf_h * (chn.ATSC_WIDTH / FS)
               / PILOT_TO_SHELF_INTEGRATED_RATIO)

    for i in idx:
        c = bin_center(i)
        blo, bhi = max(c - FS / 2, lo), min(c + FS / 2, hi)
        inpay = (c > pay_lo) and (c < pay_hi)
        h = shelf_h if inpay else shelf_h * 0.06     # guard bins: rolloff only
        is_pilot = (i == pilot_bin)
        ax.add_patch(Rectangle(
            (blo + 0.004, 0), (bhi - blo) - 0.008, h + (pilot_h if is_pilot else 0),
            facecolor=MONITORED if is_pilot else PROTECTED,
            edgecolor=SURFACE, linewidth=0.8,
            alpha=0.95 if is_pilot else 0.72, zorder=3))

    ax.axvline(pilot, color=MONITORED, lw=1.4, zorder=4)
    ax.annotate("ATSC pilot", xy=(pilot, shelf_h + pilot_h),
                xytext=(pilot + 0.55, shelf_h + pilot_h + 0.62),
                fontsize=7.5, color=MONITORED, ha="left",
                arrowprops=dict(arrowstyle="-", color=MONITORED, lw=0.7))
    ax.annotate("the one bin the detector reads",
                xy=(bin_center(pilot_bin), 0.55), xytext=(pilot + 0.62, 1.86),
                fontsize=7.5, color=MONITORED, ha="left",
                arrowprops=dict(arrowstyle="->", color=MONITORED, lw=0.7))
    ax.annotate("the 14.4 bins it speaks for: inferred shelf, flat in PSD",
                xy=(pay_lo + 3.4, shelf_h), xytext=(pay_lo + 1.35, 1.30),
                fontsize=7.5, color=PROTECTED, ha="left",
                arrowprops=dict(arrowstyle="->", color=PROTECTED, lw=0.7))

    ax.set_xlim(lo - 0.12, hi + 0.12)
    ax.set_ylim(0, 2.35)
    ax.set_yticks([])
    ax.set_xlabel(f"frequency [MHz], ATSC channel {ch} allocation, "
                  f"{chn.ATSC_WIDTH / FS:.2f} CHIME coarse channels")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    zax = ax.twiny()
    zax.set_xlim(ax.get_xlim())
    zt = [lo, lo + 2, lo + 4, hi]
    zax.set_xticks(zt)
    zax.set_xticklabels(
        [f"{HI_REST_FREQUENCY_MHZ / v - 1:.3f}" for v in zt])
    zax.set_xlabel("21 cm redshift $z$", labelpad=4)
    zax.tick_params(length=2)
    for s in ("top", "right", "left", "bottom"):
        zax.spines[s].set_visible(False)
    ax.set_title("(a)  one allocation: what is measured, and what the measurement covers",
                 loc="left", pad=22)


def panel_band(ax):
    """470-608 MHz: 23 monitored bins against 353 covered."""
    lo, hi = chn.ATSC_CH14_LOWER_EDGE, chn.ATSC_DTV_UPPER_EDGE
    n_bins = (hi - lo) / FS
    for ch in chn.ATSC_DTV_CHANNELS:
        clo, chi = chn.channel_edges(ch)
        pilot = clo + ATSC_GUARD
        ax.add_patch(Rectangle((clo + 0.05, 0.0), chn.ATSC_WIDTH - 0.1, 1.0,
                               facecolor=PROTECTED, edgecolor=SURFACE,
                               linewidth=0.7, alpha=0.60, zorder=2))
        ax.add_patch(Rectangle((bin_center(bin_index(pilot)) - FS / 2, 0.0),
                               FS, 1.0, facecolor=MONITORED, edgecolor="none",
                               zorder=3))
    for ch in chn.ATSC_DTV_CHANNELS[::2]:
        clo, _ = chn.channel_edges(ch)
        ax.text(clo + 3.0, 1.06, str(ch), ha="center", va="bottom",
                fontsize=6.0, color=MUTED)
    ax.text(lo - 1.5, 1.06, "ch", ha="right", va="bottom",
            fontsize=6.0, color=MUTED)

    ax.set_xlim(lo - 4, hi + 2)
    ax.set_ylim(-0.15, 2.05)
    ax.set_yticks([])
    ax.set_xlabel("frequency [MHz]", labelpad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.text(lo - 4, 1.52,
            f"23 monitored bins  →  {n_bins:.0f} covered   "
            f"({n_bins / 23:.2f}$\\times$ leverage; the monitored bins are "
            f"{23 / n_bins * 100:.1f}% of the band)",
            fontsize=8, color=INK, va="bottom")

    zax = ax.twiny()
    zax.set_xlim(ax.get_xlim())
    zt = [470, 500, 540, 580, 608]
    zax.set_xticks(zt)
    zax.set_xticklabels(
        [f"{HI_REST_FREQUENCY_MHZ / v - 1:.2f}" for v in zt])
    zax.set_xlabel("21 cm redshift $z$", labelpad=4)
    zax.tick_params(length=2)
    for s in ("top", "right", "left", "bottom"):
        zax.spines[s].set_visible(False)
    ax.set_title("(b)  the same geometry across the DTV band", loc="left", pad=22)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel", type=int, default=35)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)

    style()
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0),
                             gridspec_kw=dict(hspace=1.15))
    panel_allocation(axes[0], args.channel)
    panel_band(axes[1])

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MONITORED, alpha=0.95),
               plt.Rectangle((0, 0), 1, 1, facecolor=PROTECTED, alpha=0.72)]
    fig.legend(handles,
               ["monitored bin: carries the pilot, sacrificed to the measurement",
                "protected bins: contaminated by the shelf, saved by the decision"],
               loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, -0.065), handlelength=1.1)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"fig7_footprint.{ext}", dpi=220,
                    bbox_inches="tight")
    print(f"wrote {args.out}/fig7_footprint.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

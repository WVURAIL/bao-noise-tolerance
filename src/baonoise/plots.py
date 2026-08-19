"""Publication figures for the noise-tolerance forecast.

Styling follows a validated categorical palette (CVD-checked) and quiet-chart
specs: 2px lines, hairline solid gridlines, recessive axes, selective direct
labels, no dual axes. Colors follow the *entity* (scenario) across figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- palette
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# color follows the entity across all figures
SCENARIO_COLORS = {
    "clean": SERIES[0],
    "measured": SERIES[1],
    "uniform50_dtv": SERIES[2],
    "uniform75_dtv": SERIES[3],
    "uniform97_dtv": SERIES[4],
}


def setup_style() -> None:
    """Quiet-chart styling with Computer Modern text to match the LaTeX
    manuscript (matplotlib's bundled cmr10 + 'cm' mathtext; no external
    TeX required, so the figures render identically for every tool user).
    cmr10 lacks some unicode glyphs: use ASCII hyphens and $\times$ in
    labels, and keep unicode_minus off."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 10.5, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 2.0, "lines.solid_joinstyle": "round",
        "lines.solid_capstyle": "round", "legend.frameon": False,
        "legend.fontsize": 9.5,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _threshold(ax, y, label, x_text):
    ax.axhline(y, color=MUTED, lw=1.0)
    ax.text(x_text, y, f" {label}", color=INK2, fontsize=9,
            va="bottom", ha="left")


# ---------------------------------------------------------------- figures
def fig_significance_vs_time(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                             labels: dict[str, str], outfile: Path):
    """curves: name -> (years, significance). Draw order: heaviest masking
    first so the clean/measured pair stays on top where curves bunch."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = list(curves)[::-1]
    for name in order:
        yrs, sig = curves[name]
        c = SCENARIO_COLORS.get(name, MUTED)
        ax.plot(yrs, sig, color=c, label=labels.get(name, name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    x0 = min(v[0].min() for v in curves.values())
    _threshold(ax, 5.0, r"5$\sigma$ detection", x0 * 1.05)
    _threshold(ax, 3.0, r"3$\sigma$", x0 * 1.05)
    ax.set_yticks([1, 3, 5, 10, 20, 40])
    ax.set_yticklabels(["1", "3", "5", "10", "20", "40"])
    ax.set_xlabel("Observing time [on-sky yr; 1 yr = 8,760 hr (Overview normalization)]")
    ax.set_ylabel(r"BAO detection significance $A/\sigma_A$")
    ax.set_title("CHIME BAO significance vs observing time under DTV masking")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lbls[::-1], loc="upper left")
    return _save(fig, outfile)


def fig_required_time(fracs: np.ndarray, series: list[dict], outfile: Path,
                      annotate_f: float = 0.5):
    """series: [{label, years (array), color (idx), annotate (bool),
    measured_years (float|None)}]. Years on log axis vs masked fraction."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for k, s in enumerate(series):
        c = SERIES[s.get("color", k)]
        ax.plot(100 * fracs, s["years"], color=c, label=s["label"])
        if s.get("annotate", False):
            i = int(np.argmin(np.abs(fracs - annotate_f)))
            ax.plot(100 * fracs[i], s["years"][i], "o", ms=8, color=c,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            ax.annotate(f"{100*fracs[i]:.0f}% masked" + r" $\rightarrow$ " + f"{s['years'][i]:.1f} yr",
                        (100 * fracs[i], s["years"][i]), xytext=(12, -18),
                        textcoords="offset points", color=INK2, fontsize=9.5)
        my = s.get("measured_years")
        if my is not None and np.isfinite(my):
            ax.plot([0.0], [my], "o", ms=8, color=c,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=6,
                    clip_on=False)
            ax.annotate(f"pilot-proxy-derived: {my:.2f} yr", (0.0, my),
                        xytext=(9, -13), textcoords="offset points",
                        color=INK2, fontsize=9)
    ax.set_yscale("log")
    lo = min(np.nanmin(s["years"]) for s in series)
    hi = max(np.nanmax(s["years"]) for s in series)
    ax.set_ylim(0.65 * lo, 2.0 * hi)
    yticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
              if 0.65 * lo <= t <= 2.0 * hi]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:g}" for t in yticks])
    ax.set_xlabel("Masked fraction of the DTV band 470-608 MHz [%]")
    ax.set_ylabel("Required observing time [on-sky yr; 1 yr = 8,760 hr]")
    ax.set_title("Noise tolerance: observing time to reach BAO targets\n"
                 "vs uniform masking of the DTV band")
    ax.legend(loc="upper left")
    return _save(fig, outfile)


def fig_channel_masking(fractions: dict[int, float], excised: set[int],
                        outfile: Path):
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.set_axisbelow(True)
    chans = sorted(fractions)
    for ch in chans:
        f = fractions[ch]
        color = CRITICAL if ch in excised else SERIES[0]
        ax.bar(ch, 100 * f, width=0.8, color=color, linewidth=0)
        if ch in excised:
            ax.annotate("excised", (ch, 100 * f), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        color=CRITICAL, fontsize=8.5, fontweight="semibold")
        elif f > 0.05:
            ax.annotate(rf"$\times${1/(1-f):.2f}", (ch, 100 * f), xytext=(0, 4),
                        textcoords="offset points", ha="center", color=INK2,
                        fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 300)
    ax.set_yticks([1, 3, 10, 30, 100])
    ax.set_yticklabels(["1%", "3%", "10%", "30%", "100%"])
    ax.set_xticks(chans)
    ax.set_xticklabels([str(c) for c in chans], fontsize=8.5)
    ax.set_xlabel("ATSC physical channel (470-608 MHz, 6 MHz each)")
    ax.set_ylabel("Masked fraction of observing time")
    ax.set_title("Adopted DTV-channel masking configuration\n"
                 r"($\times N$ = integration-time multiplier to recover "
                 "clean-equivalent depth)")
    return _save(fig, outfile)


def fig_perbin_significance(zc: np.ndarray, curves: dict[str, np.ndarray],
                            labels: dict[str, str], outfile: Path,
                            t_label: str = "2 yr", ylab: str | None = None,
                            title: str | None = None):
    """Concentric marker sizes keep all series legible where curves coincide
    (outside the DTV band the scenarios are identical by construction).
    Pass precomputed values in `curves` (significance or uncertainty) with a
    matching `ylab`/`title`."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # shade the redshift range where ATSC DTV can contaminate (470-608 MHz)
    from .channels import ATSC_CH14_LOWER_EDGE, ATSC_DTV_UPPER_EDGE
    from .constants import HI_REST_FREQUENCY_MHZ
    z_dtv = (HI_REST_FREQUENCY_MHZ / ATSC_DTV_UPPER_EDGE - 1.0,
             HI_REST_FREQUENCY_MHZ / ATSC_CH14_LOWER_EDGE - 1.0)
    ax.axvspan(z_dtv[0], z_dtv[1], color=GRID, alpha=0.45, zorder=0, lw=0)
    ax.text(0.5 * (z_dtv[0] + z_dtv[1]), 0.015, "ATSC DTV band",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            color=MUTED, fontsize=8.5)
    sizes = [10.5, 7.5, 4.5]
    for k, (name, sig) in enumerate(curves.items()):
        c = SCENARIO_COLORS.get(name, MUTED)
        ms = sizes[k] if k < len(sizes) else 4
        ax.plot(zc, sig, color=c, label=labels.get(name, name), marker="o",
                ms=ms, lw=1.6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3 + k)
    sec = ax.secondary_xaxis(
        "top", functions=(
            lambda z: HI_REST_FREQUENCY_MHZ /
            (1.0 + np.maximum(z, -0.99)),
            lambda nu: HI_REST_FREQUENCY_MHZ /
            np.maximum(nu, 1e-3) - 1.0))
    sec.set_xlabel("Frequency [MHz]", color=INK2)
    sec.tick_params(colors=MUTED, labelcolor=INK2)
    ax.set_xlabel("Redshift bin center")
    ax.set_ylabel(ylab or rf"Per-bin BAO significance at {t_label}")
    ax.set_title(title or
                 "Where masking bites: per-redshift-bin BAO significance")
    ax.legend(loc="upper right")
    return _save(fig, outfile)


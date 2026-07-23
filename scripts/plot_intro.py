#!/usr/bin/env python3
"""Pedagogical figures for the dissertation's opening section.

These four are illustrations rather than measurements, and each caption in the
manuscript says so. They exist to give a reader who has never met the problem
the pictures the prose leans on: where the measurement lives, what bias means
against an error bar, why averaging does not remove a coherent signal, and
what a television channel actually looks like to the telescope.

Every number that *is* real comes from the same constants the analysis uses:
the band edges, the pilot offset and its 11.3 dB deficit, the 390.625 kHz
CHIME raster, the 21.6 dB concentration advantage. Nothing here is tuned to
look good; the averaging panel is a single seeded realization.

    python3 scripts/plot_intro.py --out out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise.plots import (                              # noqa: E402
    BASELINE, CRITICAL, GRID, INK, INK2, MUTED, SERIES, SURFACE,
    _save, setup_style)
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.patches import Rectangle                  # noqa: E402

NU21 = 1420.406             # MHz
DTV_LO, DTV_HI = 470.0, 608.0
CHIME_LO, CHIME_HI = 400.0, 800.0
BIN_MHZ = 400.0 / 1024.0    # 390.625 kHz raster
PILOT_OFFSET_MHZ = 0.309441  # above the lower allocation edge (A/53)
PILOT_DEFICIT_DB = 11.3      # pilot power below the data shelf
CONCENTRATION_DB = 21.636    # deficit + 10log10(6 MHz / 3051.76 Hz)


def z_of(nu):
    return NU21 / np.asarray(nu, dtype=float) - 1.0


def nu_of(z):
    return NU21 / (1.0 + np.asarray(z, dtype=float))


# ---------------------------------------------------------------- fig: band
def fig_band(outfile: Path):
    setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    y0, y1 = 0.0, 1.0

    ax.add_patch(Rectangle((CHIME_LO, y0), CHIME_HI - CHIME_LO, y1 - y0,
                           facecolor=GRID, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((DTV_LO, y0), DTV_HI - DTV_LO, y1 - y0,
                           facecolor=SERIES[0], alpha=0.30, edgecolor="none",
                           zorder=2))
    # the five allocations measured in this work: ch32-36, 578-608 MHz
    ax.add_patch(Rectangle((578.0, y0), 30.0, y1 - y0, facecolor=SERIES[1],
                           alpha=0.55, edgecolor="none", zorder=3))
    for k in range(24):                       # allocation edges, ch14-36
        x = DTV_LO + 6.0 * k
        ax.plot([x, x], [y0, y1], color=SURFACE, lw=0.7, zorder=4)

    ax.annotate("CHIME observing band, 400-800 MHz",
                xy=(404, 1.10), fontsize=9.5, color=INK2, ha="left")
    ax.annotate("digital television:\n23 allocations of 6 MHz",
                xy=(539, 0.5), fontsize=9.5, color=INK, ha="center",
                va="center", zorder=6, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.88, pad=1.6))
    ax.annotate("the five channels\nmeasured here (ch32-36)",
                xy=(593, 1.0), xytext=(650, 1.28), fontsize=9,
                color=SERIES[1], ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.8,
                                shrinkA=2, shrinkB=1))

    sec = ax.secondary_xaxis("top", functions=(z_of, nu_of))
    sec.set_xticks([0.8, 1.0, 1.2, 1.5, 2.0, 2.5])
    sec.set_xlabel("Redshift of the 21 cm line", color=INK2, fontsize=9.5)
    sec.tick_params(colors=MUTED, labelcolor=INK2)

    ax.set_xlim(CHIME_LO - 4, CHIME_HI + 4)
    ax.set_ylim(-0.28, 1.75)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Frequency [MHz]")
    for x, lab in ((DTV_LO, "470"), (DTV_HI, "608")):
        ax.annotate(lab, xy=(x, -0.06), ha="center", va="top", fontsize=8.5,
                    color=SERIES[0])
    return _save(fig, outfile)


# ------------------------------------------------- fig: bias vs error bar
def fig_two_errors(outfile: Path):
    setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.linspace(-3.2, 3.2, 900)

    def gauss(mu, sig):
        return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

    b, sig_late = 0.60, 0.25
    ax.plot(x, gauss(0, 1.0), color=MUTED, lw=1.8)
    ax.plot(x, gauss(0, sig_late), color=SERIES[0], lw=2.0)
    ax.plot(x, gauss(b, sig_late), color=SERIES[1], lw=2.0)
    ax.axvline(0.0, color=INK, lw=1.2, zorder=5)
    ax.annotate("truth", xy=(0.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-6, -2), textcoords="offset points", ha="right",
                va="top", fontsize=9.5, color=INK)

    ax.annotate("early: a wide error bar,\ncentered on the truth",
                xy=(-1.15, gauss(0, 1.0)[np.argmin(abs(x + 1.15))]),
                xytext=(-2.9, 0.75), fontsize=9.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("16x the data: the error bar\nshrank fourfold, still centered",
                xy=(-0.28, 1.30), xytext=(-2.9, 1.42), fontsize=9.5,
                color=SERIES[0],
                arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.8))
    ax.annotate("16x the data with a systematic:\njust as narrow, and "
                "$2.4\\sigma$ from the truth",
                xy=(b + 0.28, 1.30), xytext=(1.15, 1.42), fontsize=9.5,
                color=SERIES[1],
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.8))

    # the bias arrow, drawn at the height where both narrow curves live
    ya = 0.55
    ax.annotate("", xy=(b, ya), xytext=(0.0, ya),
                arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.6))
    ax.annotate("bias $b$: does not shrink with time",
                xy=(b / 2, ya), xytext=(0, 6), textcoords="offset points",
                ha="left", fontsize=9.5, color=CRITICAL, zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.88, pad=1.6))

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0, 1.85)
    ax.set_yticks([])
    ax.set_xlabel("Measured value [units of the early error bar]")
    ax.set_ylabel("Probability of quoting this value")
    ax.set_title("Two ways to be wrong. Time fixes one of them.")
    return _save(fig, outfile)


# ------------------------------------------------- fig: coherent averaging
def fig_averaging(outfile: Path, a=0.01, n=2 ** 20, seed=11):
    setup_style()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    k = np.arange(1, n + 1)
    idx = np.unique(np.logspace(0, np.log10(n - 1), 700).astype(int))

    cm_noise = np.abs(np.cumsum(noise) / k)[idx]
    cm_sig = np.abs(np.cumsum(noise + a) / k)[idx]

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.loglog(k[idx], 1.0 / np.sqrt(k[idx]), color=BASELINE, lw=1.2,
              ls=(0, (5, 3)), zorder=2)
    ax.loglog(k[idx], cm_noise, color=SERIES[0], lw=1.8, zorder=3)
    ax.loglog(k[idx], cm_sig, color=SERIES[1], lw=1.8, zorder=4)
    ax.axhline(a, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=2)

    ax.annotate("thermal noise alone:\naverages down forever",
                xy=(0.025, 0.05), xycoords="axes fraction", fontsize=9.5,
                color=SERIES[0], ha="left", va="bottom", zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.88, pad=1.6))
    ax.annotate("plus a coherent signal at 1/100 the\nnoise amplitude "
                "(40 dB down in power)",
                xy=(0.985, 0.90), xycoords="axes fraction", fontsize=9.5,
                color=SERIES[1], ha="right", va="top", zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.88, pad=1.6))
    ax.annotate("the transmitter's level",
                xy=(2.2, a), xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=INK2, ha="left", zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.88, pad=1.6))
    ncross = int(1 / a ** 2)
    ax.axvline(ncross, color=MUTED, lw=0.9, zorder=1)
    ax.annotate("$N = 1/a^2$: past this point the\naverage belongs to the "
                "transmitter",
                xy=(ncross, 1.15e-1), xytext=(-8, 0),
                textcoords="offset points", fontsize=9, color=INK2,
                ha="right", zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.88, pad=1.6))

    ax.set_xlim(1, n)
    ax.set_ylim(3e-4, 3)
    ax.set_xlabel("Samples averaged, $N$")
    ax.set_ylabel(r"$|$running average$|$  [noise RMS]")
    ax.set_title("Noise averages down. A transmitter does not.")
    return _save(fig, outfile)


# ------------------------------------------------------- fig: ATSC anatomy
def fig_atsc(outfile: Path):
    setup_style()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))

    # Schematic 8-VSB spectrum: flat shelf with root-raised-cosine style
    # edges. Occupied width 5.38 MHz plus 11.5% rolloff filling the 6 MHz.
    edge = 0.62
    x = np.linspace(0, 6, 1200)
    shelf = np.ones_like(x)
    lo = x < edge
    hi = x > 6 - edge
    shelf[lo] = 0.5 * (1 - np.cos(np.pi * x[lo] / edge))
    shelf[hi] = 0.5 * (1 - np.cos(np.pi * (6 - x[hi]) / edge))
    shelf_db = 10 * np.log10(np.maximum(shelf, 1e-6))

    # CHIME's raster, and the one bin that carries the pilot
    nbins = 6.0 / BIN_MHZ                       # 15.36
    for k in range(1, int(nbins) + 1):
        ax.axvline(k * BIN_MHZ, color=GRID, lw=0.7, zorder=1)
    ax.axvspan(0.0, BIN_MHZ, color=SERIES[0], alpha=0.20, zorder=1, lw=0)

    ax.plot(x, shelf_db, color=INK2, lw=1.8, zorder=3)
    ax.fill_between(x, -9, shelf_db, color=BASELINE, alpha=0.35, zorder=2, lw=0)

    # The pilot: all of its ~7% of station power at one known frequency.
    pilot_top = CONCENTRATION_DB
    ax.plot([PILOT_OFFSET_MHZ] * 2,
            [10 * np.log10(0.5 * (1 - np.cos(np.pi * PILOT_OFFSET_MHZ / edge))),
             pilot_top], color=SERIES[0], lw=2.4, zorder=5,
            solid_capstyle="butt")
    ax.plot([PILOT_OFFSET_MHZ], [pilot_top], "o", ms=4.5, color=SERIES[0],
            zorder=6)

    ax.annotate("the pilot: $\\sim$7% of the station's power,\nall at one "
                "frequency the standard fixes ---\n21.6 dB above the shelf "
                "in a 3 kHz bin",
                xy=(PILOT_OFFSET_MHZ, pilot_top), xytext=(0.75, 21.0),
                fontsize=9.5, color=SERIES[0], va="top",
                arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.8,
                                shrinkA=4, shrinkB=2))
    ax.annotate("data payload: randomized by the broadcast standard ---\n"
                "deliberately indistinguishable from noise",
                xy=(3.3, 0.0), xytext=(3.3, 7.5), fontsize=9.5, color=INK2,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("the monitored bin:\npilot power is compared against\ntwo "
                "reference frequencies\na few kHz away inside it",
                xy=(0.5 * BIN_MHZ, -8.4), xytext=(1.55, -7.6), fontsize=9,
                color=SERIES[0], va="bottom",
                arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.8))

    # leverage bracket along the top
    yb = 25.6
    ax.annotate("", xy=(0.0, yb), xytext=(6.0, yb),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25",
                                color=INK2, lw=1.0))
    ax.annotate("one 6 MHz allocation = 15.36 CHIME bins --- watching the "
                "pilot's bin decides for all of them",
                xy=(3.0, yb), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9.5, color=INK2)

    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(-9, 29.5)
    ax.set_xlabel("Frequency above the allocation's lower edge [MHz]")
    ax.set_ylabel("Spectral density [dB, shelf = 0]")
    ax.set_title("What a digital television channel looks like "
                 "(schematic, to the measured proportions)")
    return _save(fig, outfile)


# ---------------------------------------------------- fig: the sound wave
def fig_soundwave(outfile: Path):
    """Schematic: one overdense point launches a sound wave that freezes."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    r = np.linspace(0, 200, 900)
    S = 147.0                                     # sound horizon, Mpc

    def profile(shell_r, shell_amp, shell_w=11.0):
        core = 1.00 * np.exp(-0.5 * (r / 7.0) ** 2)
        shell = shell_amp * np.exp(-0.5 * ((r - shell_r) / shell_w) ** 2)
        return core + shell

    rows = (
        ("just after the Big Bang: one overdense spot",
         profile(8.0, 0.55, 8.0), 2.30, MUTED),
        ("the plasma era: pressure drives a sound wave outward",
         profile(85.0, 0.30), 1.15, SERIES[0]),
        ("atoms form, the pressure vanishes, the wave freezes:\n"
         "a shell of extra matter at 150 Mpc --- the ruler",
         profile(S, 0.26), 0.0, SERIES[1]),
    )
    for label, prof, off, c in rows:
        ax.plot(r, prof + off, color=c, lw=2.0, zorder=4)
        ax.fill_between(r, off, prof + off, color=c, alpha=0.12, lw=0)
        ax.annotate(label, xy=(199, off + 0.72), ha="right", va="top",
                    fontsize=9.5, color=c)

    ax.axvline(S, color=INK, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.annotate("$s \\approx 150$ Mpc", xy=(S, 3.42), xytext=(6, 0),
                textcoords="offset points", fontsize=9.5, color=INK)
    ax.annotate("time", xy=(-0.085, 0.83), xycoords="axes fraction",
                fontsize=9, color=MUTED, rotation=90, va="center")
    ax.annotate("", xy=(-0.07, 0.12), xytext=(-0.07, 0.88),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    ax.set_xlim(0, 200)
    ax.set_ylim(-0.12, 3.75)
    ax.set_yticks([])
    ax.set_xlabel("Distance from the initial overdensity [Mpc]")
    ax.set_ylabel("Matter density (offset per epoch)")
    ax.set_title("Where the ruler comes from (schematic)")
    return _save(fig, outfile)


# ------------------------------------------- the forecast's own P(k), used
# by the two figures below. CAMB output cached by the Fisher-bank build,
# Planck-2018 fiducial (Om=0.316, Ob=0.049, h=0.67), the identical spectrum
# inside every forecast of Section VII. k is in Mpc^-1 (verified: the wiggle
# spacing gives s = 145 Mpc and the turnover sits at k_eq = 0.011 Mpc^-1).
PK_CACHE = ROOT / "data" / "cache_pk_chime2022.dat"
OM, OB, H = 0.316, 0.049, 0.67


def _load_pk():
    k, p = np.genfromtxt(PK_CACHE).T
    good = (k > 1e-4) & (k < 5.0)
    return k[good], p[good]


def _eh98_nowiggle(k_mpc):
    """Eisenstein & Hu (1998) zero-baryon transfer function, eqs. 26-31.

    k in Mpc^-1 throughout, matching the cache.
    """
    omh2, obh2 = OM * H * H, OB * H * H
    theta = 2.7255 / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10.0 * obh2 ** 0.75)
    a_g = (1 - 0.328 * np.log(431.0 * omh2) * (OB / OM)
           + 0.38 * np.log(22.3 * omh2) * (OB / OM) ** 2)
    gamma_eff = OM * H * (a_g + (1 - a_g) / (1 + (0.43 * k_mpc * s) ** 4))
    q = (k_mpc / H) * theta ** 2 / gamma_eff
    L = np.log(2 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1 + 62.5 * q)
    return L / (L + C * q * q)


def _nowiggle_pk(k, p):
    """No-wiggle P(k): EH98 shape, amplitude/tilt matched to the cache.

    The quadratic-in-ln k correction absorbs the primordial tilt and any
    slow shape mismatch; its curvature is far too gentle to absorb the
    wiggles themselves (period 0.043 Mpc^-1 in k, decades shorter).
    """
    t0 = _eh98_nowiggle(k)
    base = np.log(np.maximum(k * t0 * t0, 1e-30))
    fit = (k > 8e-3) & (k < 0.6)
    co = np.polyfit(np.log(k[fit]), np.log(p[fit]) - base[fit], 2)
    return np.exp(base + np.polyval(co, np.log(k)))


def _xi_r2(k, p, r_mpc, damp=2.0):
    """r^2 xi(r), by direct spherical transform with a Gaussian taper."""
    lnk = np.log(k)
    w = k ** 3 * p * np.exp(-(k * damp) ** 2) / (2 * np.pi ** 2)
    xi = np.array([np.trapezoid(w * np.sinc(k * rr / np.pi), lnk)
                   for rr in r_mpc])
    return xi * r_mpc ** 2


def fig_wiggle(outfile: Path):
    """The same feature two ways: the bump in xi(r), the wiggles in P(k)."""
    setup_style()
    k, p = _load_pk()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    r = np.linspace(40, 200, 320)
    xi2 = _xi_r2(k, p, r)
    ax1.plot(r, xi2, color=SERIES[0], lw=2.0, zorder=4)
    win = (r > 125) & (r < 170)
    ib = np.flatnonzero(win)[np.argmax(xi2[win])]
    ax1.axvline(r[ib], color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax1.annotate("the bump: pairs of regions\nprefer this separation",
                 xy=(r[ib], xi2[ib]), xytext=(-10, -34),
                 textcoords="offset points", ha="right", fontsize=9,
                 color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                 shrinkA=2, shrinkB=3))
    ax1.annotate(f"{r[ib]:.0f} Mpc", xy=(r[ib], 0.03),
                 xycoords=("data", "axes fraction"), xytext=(4, 0),
                 textcoords="offset points", fontsize=9, color=MUTED)
    ax1.set_xlabel("Separation [Mpc]")
    ax1.set_ylabel(r"$r^2 \xi(r)$")
    ax1.set_title("(a) as a preferred separation", loc="left", fontsize=10.5)

    kk = (k > 0.012) & (k < 0.45)
    ax2.plot(k[kk], (p / _nowiggle_pk(k, p))[kk], color=SERIES[0], lw=2.0,
             zorder=4)
    ax2.axhline(1.0, color=BASELINE, lw=1.0)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Wavenumber $k$ [Mpc$^{-1}$]")
    ax2.set_ylabel(r"$P(k)\,/\,P_{\rm smooth}(k)$")
    ax2.set_title("(b) as wiggles in the power spectrum", loc="left",
                  fontsize=10.5)
    ax2.annotate("one length scale $\\rightarrow$ many harmonics,\n"
                 "spaced $\\Delta k = 2\\pi/s$",
                 xy=(0.97, 0.97), xycoords="axes fraction", ha="right",
                 va="top", fontsize=9, color=INK2, zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1.6))
    fig.subplots_adjust(wspace=0.34, bottom=0.16)
    return _save(fig, outfile)


def fig_dilation(outfile: Path):
    """What each parameter does to the bump: position versus height."""
    setup_style()
    k, p = _load_pk()
    r = np.linspace(95, 195, 220)
    xi2 = _xi_r2(k, p, r)
    alpha, boost = 1.06, 1.25
    xi2_dil = _xi_r2(k, p, r * alpha) / alpha ** 2      # xi(alpha r) * r^2

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.plot(r, xi2, color=INK, lw=2.2, zorder=5)
    ax.plot(r, xi2_dil, color=SERIES[1], lw=2.0, zorder=4)
    ax.plot(r, boost * xi2, color=SERIES[2], lw=2.0, zorder=4,
            ls=(0, (5, 2)))

    win = (r > 120) & (r < 175)
    ib = np.flatnonzero(win)[np.argmax(xi2[win])]
    ibd = np.flatnonzero(win)[np.argmax(xi2_dil[win])]
    for i, c in ((ib, INK), (ibd, SERIES[1])):
        ax.axvline(r[i], color=c, lw=0.8, ls=(0, (2, 2)), zorder=2)
    ax.annotate("", xy=(r[ibd], xi2[ib] * 1.06), xytext=(r[ib], xi2[ib] * 1.06),
                arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1.4))

    ax.annotate("fiducial", xy=(96, xi2[0]), xytext=(2, 6),
                textcoords="offset points", fontsize=9.5, color=INK,
                ha="left", va="bottom")
    ax.annotate("dilation $\\alpha = 1.06$: the bump moves.\n"
                "The BAO ruler --- a geometry measurement.",
                xy=(0.025, 0.96), xycoords="axes fraction", va="top",
                fontsize=9.5, color=SERIES[1], zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1.6))
    ax.annotate("growth $\\times 1.25$: the bump rises where it stands.\n"
                "$f\\sigma_8$ --- an amplitude measurement.",
                xy=(0.025, 0.80), xycoords="axes fraction", va="top",
                fontsize=9.5, color=SERIES[2], zorder=7, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1.6))

    ax.set_yticks([])
    ax.set_xlim(95, 195)
    ax.set_xlabel("Separation [Mpc]")
    ax.set_ylabel(r"$r^2 \xi(r)$")
    ax.set_title("Two ways to read the wiggle")
    fig.subplots_adjust(bottom=0.15)
    return _save(fig, outfile)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig_band, "fig_intro_band.png"),
                     (fig_two_errors, "fig_intro_two_errors.png"),
                     (fig_averaging, "fig_intro_averaging.png"),
                     (fig_atsc, "fig_intro_atsc.png"),
                     (fig_soundwave, "fig_intro_soundwave.png"),
                     (fig_wiggle, "fig_intro_wiggle.png"),
                     (fig_dilation, "fig_intro_dilation.png")):
        print("wrote", fn(args.out / name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



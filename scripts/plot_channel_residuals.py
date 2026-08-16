#!/usr/bin/env python3
"""Four-panel summary of the residual budget for the analysed DTV channels.

Reads pilot-proxy survey products and renders the whole argument in one figure:
where the transmitter-off epochs come from, why one channel admits a stationary
timescale and the others do not, the dB chain from shelf to residual, and what
it costs the forecast.

    python3 scripts/plot_channel_residuals.py 506.npz 521.npz 537.npz \
        --off 35=2021-09 --off 34=2019-03 --out out/

Panels
  (a) quarterly masked fraction: where the off epochs are
  (b) quarterly median shelf: the transmitter switch-on, same x axis
  (c) shelf power by timescale: stationary vs episodic
  (d) the dB chain: kept-frame bound to r
  (e) forecast cost: keep vs excise, with and without residual

Reject rate and shelf level are separate axes rather than one twin-axis plot:
two measures of different scale sharing a frame invite reading a crossing that
does not exist.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.gridspec import GridSpec             # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

from baonoise import residual as R                   # noqa: E402
from baonoise import forecast, scenarios             # noqa: E402
from baonoise.fisherbank import FisherBank           # noqa: E402
from baonoise.resources import DEFAULT_BANK          # noqa: E402

# Validated categorical slots 1-3 (all-pairs, light surface):
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --pairs all
# Aqua sits below 3:1 on the light surface, so every series is direct-labelled.
# Reference categorical order, slots 1-5 (dataviz palette.md), validated:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4"
#   ALL CHECKS PASS. Three of the five sit below 3:1 against this surface,
#   so every series is direct-labelled rather than legend-only.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })


def tidy(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def quarterly(path, off_through):
    """(quarter_float, masked_fraction, median_shelf_db) per quarter."""
    d = np.load(path, allow_pickle=True)
    v = d["valid"][:, 0].astype(bool)
    rej = d["reject_mask"][:, 0].astype(bool)
    s = d["snr_shelf_db"][:, 0]
    t = d["unit_time0_ctime"][d["frame_unit_index"]]
    ym = np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc) for x in t])
    qf = np.array([a.year + (a.month - 1) // 3 * 0.25 for a in ym])
    out = []
    for q in np.unique(qf[v]):
        sel = v & (qf == q)
        if sel.sum() < 30:
            continue
        sh = s[sel & np.isfinite(s)]
        out.append((q, rej[sel].mean(),
                    float(np.median(sh)) if sh.size else np.nan))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", type=Path)
    ap.add_argument("--off", action="append", default=[], metavar="CH=YYYY-MM",
                    help="transmitter-off epoch end for a channel; repeatable")
    ap.add_argument("--delay", default=R.DEFAULT_DELAY_KEY,
                    choices=sorted(R.DELAY_SUPPRESSION_DB))
    ap.add_argument("--bank", type=Path,
                    default=DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    args = ap.parse_args()
    offs = {int(k): v for k, v in (o.split("=") for o in args.off)}

    # ---------------- gather -------------------------------------------
    chans = []
    for p in args.npz:
        ch = int(np.load(p, allow_pickle=True)["physical_channel"][0])
        off = offs.get(ch)
        budget, st, ct = R.budget_from_products(p, off_through=off,
                                                delay_key=args.delay)
        chans.append(dict(ch=ch, path=p, off=off, b=budget, st=st, ct=ct,
                          q=quarterly(p, off)))
    chans.sort(key=lambda c: c["ch"])
    color = {c["ch"]: SERIES[i] for i, c in enumerate(chans)}

    style()
    fig = plt.figure(figsize=(6.5, 7.9))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.95, 1.05, 1.15],
                  hspace=0.70, wspace=0.34,
                  left=0.100, right=0.975, top=0.925, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[0:2, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    # ---------------- (a) masked fraction ------------------------------
    for c in chans:
        q = c["q"]
        ax_a.plot(q[:, 0], 100 * q[:, 1], lw=1.4, color=color[c["ch"]],
                  solid_capstyle="round")
    ax_a.set_ylabel("masked  [%]")
    ax_a.set_ylim(0, 105)
    ax_a.set_title("(a)  masked per quarter, coarse stage $F>\\mu_0$",
                   loc="left", color=INK)
    tidy(ax_a)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    # direct labels (relief rule). All three sit at 100% by the end, so these
    # go in the right margin as a key rather than pretending to mark line ends.
    for i, c in enumerate(chans):
        ax_a.annotate(f"ch{c['ch']}", xy=(1.02, 0.92 - 0.13 * i),
                      xycoords="axes fraction", color=color[c["ch"]],
                      fontsize=7, va="center", annotation_clip=False)

    # ---------------- (b) median shelf ---------------------------------
    for c in chans:
        q = c["q"]
        ax_b.plot(q[:, 0], q[:, 2], lw=1.4, color=color[c["ch"]],
                  solid_capstyle="round")
        if c["off"]:
            yr = int(c["off"][:4]) + (int(c["off"][5:7]) - 1) // 3 * 0.25 + 0.25
            ax_b.axvline(yr, color=color[c["ch"]], lw=0.8, ls=(0, (2, 2)),
                         alpha=0.75)
    ax_b.set_ylabel("median shelf  [dB]")
    ax_b.set_xlabel("year")
    ax_b.set_title("(b)  shelf level; dashed = off-epoch cut",
                   loc="left", color=INK)
    tidy(ax_b)
    for c in chans:
        q = c["q"]
        ax_b.annotate(f"ch{c['ch']}", xy=(q[-1, 0], q[-1, 2]),
                      xytext=(4, 0), textcoords="offset points",
                      color=color[c["ch"]], fontsize=7, va="center")
    ax_b.annotate("transmitter on", xy=(2021.9, -10.6), xytext=(14, -22),
                  textcoords="offset points", fontsize=6.5, color=INK2,
                  arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))

    # ---------------- (c) power split by timescale ---------------------
    keys = [("dc_fraction", "constant"), ("interday_fraction", "inter-day"),
            ("intraday_fraction", "intra-day"), ("fast_fraction", "sub-acq.")]
    n, h = len(chans), 0.24
    ypos = np.arange(len(keys))
    for i, c in enumerate(chans):
        vals = [max(100.0 * getattr(c["st"], k), 1e-3) for k, _ in keys]
        off = (i - (n - 1) / 2) * (h + 0.035)
        ax_c.barh(ypos + off, vals, height=h, color=color[c["ch"]],
                  edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax_c.set_yticks(ypos, [lab for _, lab in keys])
    ax_c.invert_yaxis()
    ax_c.set_xscale("log")
    ax_c.set_xlim(1e-2, 900)
    ax_c.set_xlabel("share of shelf power  [%]")
    ax_c.set_title("(c)  where the shelf power lives", loc="left",
                   color=INK, pad=18)
    ax_c.set_ylim(3.6, -0.6)
    ax_c.axhspan(1.5, 3.6, color="#f3f2ee", zorder=0)
    ax_c.annotate("survives\nthe m = 0\nfilter", xy=(0.985, 0.25),
                  xycoords="axes fraction", ha="right", va="center",
                  fontsize=6.5, color=INK2, linespacing=1.4)
    ax_c.annotate("removed", xy=(0.985, 0.75), xycoords="axes fraction",
                  ha="right", va="center", fontsize=6.5, color=INK2)
    tidy(ax_c)
    ax_c.grid(axis="y", visible=False)
    ax_c.legend(handles=[Line2D([], [], color=color[c["ch"]], lw=4,
                                label=(f"ch{c['ch']} "
                                       + ("stationary" if c["ct"].is_measured
                                          else "episodic")))
                         for c in chans],
                loc="lower center", bbox_to_anchor=(0.5, 1.085), ncol=3,
                frameon=False, handlelength=0.9, handletextpad=0.4,
                columnspacing=1.0, borderpad=0.0)

    # ---------------- (d) the dB chain ---------------------------------
    stages = ["shelf\non air", "after\nmasking", "after\nground",
              "after\ndelay", "$r$ after\ncoherence"]
    x = np.arange(len(stages))
    for c in chans:
        st, b = c["st"], c["b"]
        if not np.isfinite(st.floor_db):
            continue
        gf = 10 * np.log10(1.0 / sum(f for f, _ in b.components))
        y = [st.on_shelf_db, st.floor_db, st.floor_db - gf,
             st.floor_db - gf - b.delay_filter_db, b.ratio_db]
        solid = c["ct"].is_usable
        ax_d.plot(x, y, lw=1.5, color=color[c["ch"]], zorder=3,
                  ls="-" if solid else (0, (3, 2)),
                  marker="o", ms=3.4, mfc=color[c["ch"]], mec=SURFACE, mew=0.7)
        ax_d.annotate(f"{b.ratio_db:+.1f} dB", xy=(x[-1], y[-1]),
                      xytext=(5, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=6.5,
                      color=color[c["ch"]], annotation_clip=False)
    missing = [c for c in chans if not np.isfinite(c["st"].floor_db)]
    ax_d.axhline(0, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=2)
    ax_d.annotate("$P_{\\rm res}=P_{\\rm N}$", xy=(0.02, 0), xytext=(2, 3),
                  textcoords="offset points", fontsize=6.5, color=MUTED)
    ax_d.set_xticks(x, stages)
    ax_d.set_xlim(-0.35, len(stages) - 0.35)
    ax_d.set_ylabel("power vs thermal  [dB]")
    ax_d.set_title("(d)  the residual chain", loc="left", color=INK)
    tidy(ax_d)
    ax_d.grid(axis="x", visible=False)
    if missing:
        ax_d.annotate("ch" + ", ch".join(str(c["ch"]) for c in missing)
                      + ": no off epoch, floor not measurable",
                      xy=(0.03, 0.86), xycoords="axes fraction", fontsize=6.5,
                      color=MUTED)
    ax_d.annotate("dashed = bound", xy=(0.03, 0.94), xycoords="axes fraction",
                  fontsize=6.5, color=MUTED)
    ax_d.set_ylim(top=ax_d.get_ylim()[1] + 14)

    # ---------------- (e) forecast cost --------------------------------
    bank = FisherBank(args.bank)
    fc = forecast.Forecast(bank, None, style="perbin_A")

    def hours(sc, bins=None):
        return fc.required_hours_metric(
            lambda t: fc.significance(sc, t, bins=bins), 5.0)

    ibin = 5
    clean_s = hours(scenarios.clean())
    clean_b = hours(scenarios.clean(), [ibin])
    # Masking fractions come from the forecast's own fiducial table rather than from
    # the products' reject_mask. The products record the coarse positive-excess
    # stage (and a fine rank-CFAR stage); neither reproduces the quarterly
    # table the forecast is built on, and mixing them would price two
    # different detectors in one scenario. What the products contribute here
    # is the residual r, which is a property of the shelf rather than of the
    # threshold.
    fr = dict(scenarios.measured().fractions)
    kept = max(chans, key=lambda c: c["b"].ratio if c["ct"].is_measured else -1)
    kch = kept["ch"]
    cases = [
        (f"ch{kch} clean\n($r$ = 0)", {}, 0.5, MUTED),
        (f"ch{kch} masked only\n$f$ = {fr[kch]:.3f}", {}, 0.5, "#b9b8b2"),
        (f"ch{kch} + residual\n$r$ = {kept['b'].ratio:.2f}",
         {kch: kept["b"].ratio}, 0.5, color[kch]),
    ]
    labels, sv, bn, cols = [], [], [], []
    for i, (lab, res, exc, col) in enumerate(cases):
        f_use = dict(fr)
        if i == 0:
            f_use[kch] = 0.0          # the clean reference for this channel
        sc = scenarios.Scenario("x", lab, fractions=f_use, residuals=res,
                                excise_threshold=exc)
        labels.append(lab)
        sv.append(hours(sc) / clean_s)
        bn.append(hours(sc, [ibin]) / clean_b)
        cols.append(col)
    xx = np.arange(len(labels))
    ax_e.bar(xx, bn, width=0.6, color=cols, edgecolor=SURFACE, linewidth=0.8,
             zorder=3)
    for xi, val in zip(xx, bn):
        ax_e.annotate(f"{val:.3f}", xy=(xi, val), xytext=(0, 3),
                      textcoords="offset points", ha="center", fontsize=6.5,
                      color=INK)
    ax_e.set_xticks(xx, labels, fontsize=6.0)
    ax_e.set_ylim(min(bn) - 0.05, max(bn) + 0.055)
    ax_e.set_ylabel(r"time penalty $t_{\rm req}/t_{\rm req}^{\rm clean}$")
    ax_e.set_title(f"(e)  cost in $z$ = {bank.zs[ibin]:.1f}--"
                   f"{bank.zs[ibin + 1]:.1f}, fiducial $f$",
                   loc="left", color=INK, pad=28)
    tidy(ax_e)
    ax_e.grid(axis="x", visible=False)
    grew = (bn[2] - bn[1]) / (bn[1] - bn[0]) if bn[1] != bn[0] else np.nan
    note = (f"residual costs {grew:.1f}x what the masking does"
            if np.isfinite(grew)
            else "masking alone moves this bin by nothing; the residual is "
                 "the whole cost")
    ax_e.annotate(note,
                  xy=(0.0, 1.13), xycoords="axes fraction",
                  fontsize=6.5, color=INK2)
    ax_e.annotate(f"survey level  {sv[0]:.3f} / {sv[1]:.3f} / {sv[2]:.3f}",
                  xy=(0.0, 1.045), xycoords="axes fraction",
                  fontsize=6.5, color=MUTED)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "fig6_channel_residuals"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=220, facecolor=SURFACE)
    print(f"wrote {stem}.pdf / .png")

    print("\nnumbers behind the figure")
    for c in chans:
        st, ct, b = c["st"], c["ct"], c["b"]
        tau = (f"{ct.tau_c / 60:.0f} min" if ct.is_measured
               else (f"<= {ct.tau_c / 60:.0f} min" if ct.is_usable else "refused"))
        print(f"  ch{st.channel}: masked {st.masked_fraction:.4f}  "
              f"floor {st.floor_db:7.2f} dB ({st.n_off_frames} nulls)  "
              f"ground {st.ground_filter_db:5.2f} dB  tau_c {tau:>10s}  "
              f"r {b.ratio:.4g}")
    for lab, s_, b_ in zip(labels, sv, bn):
        print(f"  {lab.replace(chr(10), ' '):32s} survey {s_:.4f}  "
              f"bin{ibin} {b_:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

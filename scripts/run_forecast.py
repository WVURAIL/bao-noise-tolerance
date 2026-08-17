#!/usr/bin/env python3
"""Main driver: evaluate masking scenarios against the Fisher bank and produce
the noise-tolerance figures, CSV tables, and a results summary (out/results.md).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from baonoise import channels, forecast, plots, scenarios, survey
from baonoise.compat import import_radiofisher
from baonoise.fisherbank import FisherBank
from baonoise.resources import DEFAULT_BANK, bank_file

DUTY = 1.0  # on-sky years at the Overview normalization (8,760 hr)


def yrs(hours: float) -> float:
    """On-sky years at the Overview normalization (1 yr = 8,760 hr)."""
    return float(hours) / survey.OVERVIEW_ONSKY_YEAR_HOURS


def fmt_t(hours: float) -> str:
    if not np.isfinite(hours):
        return "never (>1e6 hr)"
    return f"{hours:,.0f} hr ({yrs(hours):.2f} yr)"


def _load_forecast(bank_path=DEFAULT_BANK):
    """Load a strict-v2 bank, importing RadioFisher only for shared-A banks."""
    bank = FisherBank(bank_path)
    style = ("perbin_A" if bank.meta["config"] == "chime2022"
             else "shared_A")
    rf = None
    rf_dir = None
    if style == "shared_A":
        rf, rf_dir = import_radiofisher()
    fc = forecast.Forecast(bank, rf, style=style, rf_dir=rf_dir)
    return bank, fc, style


def _fiducial_comparison_rows(planck, pact, scenario_set, bins=(6, 8)):
    """Compare absolute times and masking penalties under two cosmologies."""
    metrics = [("survey_5sigma", None)]
    metrics.extend((metric, ibin) for ibin in bins
                   for metric in ("bin_5sigma", "da_2pct"))
    rows = []
    for metric, ibin in metrics:
        selected_bins = None if ibin is None else [ibin]

        def required(fc, scenario):
            if metric in {"survey_5sigma", "bin_5sigma"}:
                return fc.required_hours_metric(
                    lambda t: fc.significance(
                        scenario, t, bins=selected_bins), 5.0)
            return fc.required_hours_metric(
                lambda t: fc.sigma_param_bin(
                    scenario, t, ibin, "aperp0"),
                0.02, decreasing=True)

        clean_hours = {
            "planck2018": required(planck, scenarios.clean()),
            "pact2025": required(pact, scenarios.clean()),
        }
        for name, scenario in scenario_set.items():
            planck_hours = required(planck, scenario)
            pact_hours = required(pact, scenario)
            planck_penalty = planck_hours / clean_hours["planck2018"]
            pact_penalty = pact_hours / clean_hours["pact2025"]
            rows.append({
                "metric": metric,
                "zbin": "survey" if ibin is None else ibin,
                "scenario": name,
                "planck_hours": planck_hours,
                "pact_hours": pact_hours,
                "planck_penalty": planck_penalty,
                "pact_penalty": pact_penalty,
                "relative_penalty_shift_pct":
                    100.0 * (pact_penalty / planck_penalty - 1.0),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--outdir", default=str(ROOT / "out"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bank, fc, style = _load_forecast(args.bank)
    print(f"[cfg] bank config={bank.meta['config']} "
          f"style={style} nbins={bank.nbins}", flush=True)

    # ------------------------------------------------------------ scenarios
    scen_main = {
        "clean": scenarios.clean(),
        "measured": scenarios.measured(),
        "uniform50_dtv": scenarios.uniform(0.50, scenarios.DTV_BAND),
        "uniform75_dtv": scenarios.uniform(0.75, scenarios.DTV_BAND),
        "uniform97_dtv": scenarios.uniform(0.97, scenarios.DTV_BAND),
    }
    labels = {
        "clean": "No masking",
        "measured": "Pilot-proxy-derived mask",
        "uniform50_dtv": "50% of DTV band masked",
        "uniform75_dtv": "75% of DTV band masked",
        "uniform97_dtv": "97% of DTV band masked",
    }

    # ------------------------------------------------- table of req. times
    table_scens = dict(scen_main)
    ms = scenarios.measured(mode="fourier")
    ms.name, ms.label = "measured_fourier", "Measured (Fourier-mode noise convention)"
    table_scens["measured_fourier"] = ms
    table_scens["uniform25_dtv"] = scenarios.uniform(0.25, scenarios.DTV_BAND)
    table_scens["uniform90_dtv"] = scenarios.uniform(0.90, scenarios.DTV_BAND)
    table_scens["uniform50_chime"] = scenarios.uniform(
        0.50, scenarios.CHIME_BAND)
    table_scens["ch30_excised"] = scenarios.single_channel(30, 0.97, keep=False)
    table_scens["ch30_kept"] = scenarios.single_channel(30, 0.97, keep=True)
    sc30f = scenarios.single_channel(30, 0.97, keep=True, mode="fourier")
    sc30f.name = "ch30_kept_fourier"
    table_scens["ch30_kept_fourier"] = sc30f

    h5_clean = fc.required_hours(scen_main["clean"], 5.0)
    rows = []
    for key, sc in table_scens.items():
        h5 = fc.required_hours(sc, 5.0)
        h3 = fc.required_hours(sc, 3.0)
        s1sky = fc.significance(
            sc, survey.OVERVIEW_ONSKY_YEAR_HOURS)  # 1 on-sky yr
        s2yr = fc.significance(
            sc, 2.0 * survey.OVERVIEW_ONSKY_YEAR_HOURS)
        rows.append(dict(
            scenario=key, label=sc.label, sig_at_2yr=round(s2yr, 2),
            sig_at_1yr_onsky=round(s1sky, 2),
            hours_3sig=round(h3, 1) if np.isfinite(h3) else "inf",
            years_3sig=round(yrs(h3), 4) if np.isfinite(h3) else "inf",
            hours_5sig=round(h5, 1) if np.isfinite(h5) else "inf",
            years_5sig=round(yrs(h5), 4) if np.isfinite(h5) else "inf",
            time_penalty_vs_clean=round(h5 / h5_clean, 3) if np.isfinite(h5) else "inf"))
        print(f"[req] {key:22s} S(2yr)={s2yr:7.2f}  5sig={fmt_t(h5)}  "
              f"penalty x{rows[-1]['time_penalty_vs_clean']}", flush=True)
    with open(outdir / "required_times.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------ bin-level targets (DTV z's)
    # bin 6: z=[1.4,1.5] contains ch30 + the ch31-35 cluster
    # bin 8: z=[1.6,1.7] contains refused ch24
    bin_targets = []
    for ib in (6, 8):
        zlo, zhi = bank.zs[ib], bank.zs[ib + 1]
        for key in ["clean", "measured", "measured_fourier", "uniform50_dtv",
                    "ch30_excised", "ch30_kept", "ch30_kept_fourier"]:
            sc = table_scens[key]
            h_5s = fc.required_hours_metric(
                lambda t, sc=sc, ib=ib: fc.significance(sc, t, bins=[ib]), 5.0)
            h_a2 = fc.required_hours_metric(
                lambda t, sc=sc, ib=ib: fc.sigma_param_bin(sc, t, ib, "aperp0"),
                0.02, decreasing=True)
            bin_targets.append(dict(
                zbin=f"{zlo:.2f}-{zhi:.2f}", scenario=key,
                hours_bin5sig=round(h_5s, 1) if np.isfinite(h_5s) else "inf",
                years_bin5sig=round(yrs(h_5s), 3) if np.isfinite(h_5s) else "inf",
                hours_da2pct=round(h_a2, 1) if np.isfinite(h_a2) else "inf",
                years_da2pct=round(yrs(h_a2), 3) if np.isfinite(h_a2) else "inf"))
            print(f"[bin {zlo:.2f}-{zhi:.2f}] {key:20s} "
                  f"bin-5sig={fmt_t(h_5s)}  sigma(DA)=2%: {fmt_t(h_a2)}",
                  flush=True)
    with open(outdir / "bin_level_targets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bin_targets[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(bin_targets)

    # ----------------------- matched Planck-2018 vs P-ACT-LB comparison
    fiducial_rows = []
    if (bank.meta["config"] == "chime2022"
            and bank.meta["cosmology"] == "planck2018"):
        pact_bank = FisherBank(bank_file("pact2025"))
        if (pact_bank.meta["config"] != "chime2022"
                or pact_bank.meta["cosmology"] != "pact2025"):
            raise ValueError("the packaged P-ACT-LB bank is mislabeled")
        if (not np.array_equal(bank.t_grid, pact_bank.t_grid)
                or not np.array_equal(bank.zs, pact_bank.zs)
                or bank.paramnames != pact_bank.paramnames):
            raise ValueError(
                "Planck-2018 and P-ACT-LB banks must use matched grids and "
                "parameter schemas")
        pact_forecast = forecast.Forecast(pact_bank, style="perbin_A")
        fiducial_rows = _fiducial_comparison_rows(
            fc, pact_forecast, table_scens)
        with open(outdir / "fiducial_comparison.csv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(fiducial_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(fiducial_rows)

    # ------------------------------------------------------- fig 1: S(t)
    years_grid = np.logspace(np.log10(0.02), np.log10(20.0), 40)
    hours_grid = survey.years_to_hours(years_grid, DUTY)
    curves = {k: (years_grid, fc.significance_curve(s, hours_grid))
              for k, s in scen_main.items()}
    plots.fig_significance_vs_time(curves, labels,
                                   outdir / "fig1_significance_vs_time.png")

    # -------------------------------------------- fig 2: tolerance curves
    IB = 6   # z = 1.40-1.50 bin: ch30 + the ch31-35 cluster
    fracs = np.concatenate([np.arange(0.0, 0.96, 0.05), [0.97]])
    yrs_s5, yrs_b5, yrs_da = [], [], []
    for f in fracs:
        sc = scenarios.uniform(float(f), scenarios.DTV_BAND)
        yrs_s5.append(yrs(fc.required_hours(sc, 5.0)))
        yrs_b5.append(yrs(fc.required_hours_metric(
            lambda t, sc=sc: fc.significance(sc, t, bins=[IB]), 5.0)))
        yrs_da.append(yrs(fc.required_hours_metric(
            lambda t, sc=sc: fc.sigma_param_bin(sc, t, IB, "aperp0"),
            0.02, decreasing=True)))
    yrs_s5, yrs_b5, yrs_da = map(np.array, (yrs_s5, yrs_b5, yrs_da))

    meas = scen_main["measured"]
    meas_s5 = yrs(fc.required_hours(meas, 5.0))
    meas_b5 = yrs(fc.required_hours_metric(
        lambda t: fc.significance(meas, t, bins=[IB]), 5.0))
    meas_da = yrs(fc.required_hours_metric(
        lambda t: fc.sigma_param_bin(meas, t, IB, "aperp0"), 0.02,
        decreasing=True))
    zlo, zhi = bank.zs[IB], bank.zs[IB + 1]
    series = [
        dict(label=rf"$\sigma(D_A)\leq 2\%$ in the $z$={zlo:.2f}-{zhi:.2f} bin",
             years=yrs_da, color=0, annotate=True, measured_years=meas_da),
        dict(label=rf"BAO amplitude $S/N=5$ in the $z$={zlo:.2f}-{zhi:.2f} bin",
             years=yrs_b5, color=1, measured_years=meas_b5),
        dict(label=r"BAO amplitude $S/N=5$, full survey", years=yrs_s5,
             color=2, measured_years=meas_s5),
    ]
    plots.fig_required_time(fracs, series,
                            outdir / "fig2_required_time_vs_masking.png")
    with open(outdir / "tolerance_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["masked_fraction_dtv_band", "years_survey_5sig",
                    "years_bin_z145_5sig", "years_bin_z145_da2pct",
                    "penalty_vs_clean_da2pct"])
        for f, a, b, c in zip(fracs, yrs_s5, yrs_b5, yrs_da):
            w.writerow([round(float(f), 3), round(float(a), 4),
                        round(float(b), 4), round(float(c), 4),
                        round(float(c / yrs_da[0]), 3)])

    # --------------------------- Fig-31-style validation: sigma_DV per bin
    if style == "perbin_A":
        t1yr = survey.OVERVIEW_ONSKY_YEAR_HOURS
        with open(outdir / "fig31_validation.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["z_center", "sigma_dv_clean_pct",
                        "sigma_dv_representative_pct"])
            for i, z in enumerate(bank.zc):
                sc_cl = fc.sigma_dv_bin(scen_main["clean"], t1yr, i)
                sc_ms = fc.sigma_dv_bin(scen_main["measured"], t1yr, i)
                w.writerow([round(float(z), 3),
                            round(100 * sc_cl, 3) if np.isfinite(sc_cl) else "inf",
                            round(100 * sc_ms, 3) if np.isfinite(sc_ms) else "inf"])
        print("[fig31] per-bin sigma_DV/DV at 1 on-sky yr written", flush=True)

    # ------------------------------------------ fig 3: channel masking
    meas = scenarios.measured()
    plots.fig_channel_masking(meas.fractions,
                              set(channels.REFUSED_CHANNELS),
                              outdir / "fig3_channel_masking.png")

    # ------------------------------------------ fig 4: per-bin profile
    t2 = survey.OVERVIEW_ONSKY_YEAR_HOURS
    perbin = {}
    for k in ["clean", "measured", "uniform50_dtv"]:
        S = fc.per_bin_significance(scen_main[k], t2)
        with np.errstate(divide="ignore"):
            perbin[k] = np.where(S > 0, 100.0 / S, np.nan)  # sigma_A [%]
    plots.fig_perbin_significance(
        bank.zc, perbin, labels, outdir / "fig4_perbin_significance.png",
        t_label="1 on-sky yr (8,760 hr)",
        ylab=r"BAO amplitude uncertainty $\sigma_A$ [%]",
        title="Where masking increases BAO-amplitude uncertainty\n"
              "after 1 on-sky yr (8,760 hr)")
    with open(outdir / "perbin_significance_1onskyyr.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["z_center"] + [f"sigmaA_pct_{k}" for k in perbin])
        for i, z in enumerate(bank.zc):
            w.writerow([round(float(z), 3)] +
                       [round(float(perbin[k][i]), 3) for k in perbin])

    # --------------------------------------------------------- results.md
    def md_table(rws, cols):
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rws]
        return "\n".join([head, sep] + body)

    ch30_fraction = table_scens["ch30_kept"].fractions[30]
    ch30_mult = 1.0 / (1.0 - ch30_fraction)
    ch30_volume, _ = table_scens["ch30_excised"].bin_factors_for_zbins(
        bank.zs)[IB]
    ib_z_label = f"{bank.zs[IB]:.2f}-{bank.zs[IB + 1]:.2f}"
    fiducial_summary = ""
    if fiducial_rows:
        clean_fiducial = next(
            row for row in fiducial_rows
            if row["metric"] == "survey_5sigma"
            and row["scenario"] == "clean")
        max_penalty_shift = max(
            abs(row["relative_penalty_shift_pct"])
            for row in fiducial_rows if row["scenario"] != "clean")
        absolute_shift = 100.0 * (
            clean_fiducial["pact_hours"] / clean_fiducial["planck_hours"]
            - 1.0)
        fiducial_summary = f"""
## Matched fiducial-cosmology comparison

The provenance-complete, matched-grid Planck-2018 and P-ACT-LB banks require
{clean_fiducial['planck_hours']:.1f} and
{clean_fiducial['pact_hours']:.1f} clean on-sky hours, respectively, for the
survey-level 5-sigma target ({absolute_shift:+.1f}%). Across the survey and
both reported bin-level metrics, the largest relative change in a masking
penalty is {max_penalty_shift:.3f}%.
"""
    md = f"""# Noise-tolerance forecast results

CHIME (RadioFisher 'yCHIME' spec) BAO forecast under ATSC DTV masking.
Times are on-sky hours / on-sky years at the Overview normalization
(1 yr = 8,760 hr, no duty factor; Amiri et al. 2022 Table 2). Multiply
on-sky years by ~6.6 (duty 0.152, the empirical 2019-practice fraction
from Amiri et al. 2025, RFI component excluded) for calendar years.
Absolute times
inherit the optimism of Fisher forecasting (smooth foreground residuals
eps_FG=1e-6, no calibration systematics); the robust currency is the
**time penalty relative to the clean survey**, last column below.

## Survey-level BAO detection (A/sigma_A, all z bins combined)

{md_table(rows, ["scenario", "label", "sig_at_2yr", "years_5sig", "time_penalty_vs_clean"])}

Reading: the measured pilot-proxy masking costs the survey only
**{100*(next(r for r in rows if r['scenario']=='measured')['time_penalty_vs_clean']-1):.0f}%
extra integration time** to any fixed BAO-amplitude target, and even masking
50% of the whole DTV band costs
{100*(next(r for r in rows if r['scenario']=='uniform50_dtv')['time_penalty_vs_clean']-1):.0f}%.
The full survey reaches nominal 5-sigma almost immediately because 400-470 and
608-800 MHz are DTV-free and CHIME's mapping speed is enormous; the binding
constraints live at bin level, below.

## Bin-level targets in the DTV-affected redshifts

Per-bin BAO detection (5 sigma from that Delta-z=0.1 bin alone) and
sigma(alpha_perp) <= 2% (a per-bin BAO distance measurement):

{md_table(bin_targets, ["zbin", "scenario", "years_bin5sig", "years_da2pct"])}

## The channel-30 story

Channel 30 (566-572 MHz, z = 1.48-1.51) is masked ~97% of the time. Options:

* **Integrate through it**: recovering clean-equivalent depth in that 6 MHz
  requires x{ch30_mult:.0f} the integration time: if the survey needs T years,
  that slice needs {ch30_mult:.0f} T. Ten years of integration buys the depth of
  {120/ch30_mult:.1f} clean months. That is "never" for any practical purpose,
  and under the Fourier-mode noise convention the surviving 97% of the
  containing z-bin is dragged down with it (see ch30_kept_fourier rows).
* **Excise it** (what the forecast prices by default): costs the overlapped
  bandwidth, a {100*(1-ch30_volume):.0f}% volume hit to the z={ib_z_label} bin, a
  {100*(next(r for r in rows if r['scenario']=='ch30_excised')['time_penalty_vs_clean']-1):.1f}%
  time penalty at survey level.

Excision wins by an enormous margin: throw the channel out and pay in volume,
never in noise.

{fiducial_summary}

## Files

required_times.csv, bin_level_targets.csv, tolerance_curve.csv,
perbin_significance_1onskyyr.csv, fiducial_comparison.csv,
fig1-fig4 (png + pdf).
"""
    (outdir / "results.md").write_text(md)
    print("[done] outputs in", outdir, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Consistency check: every number quoted in the paper regenerates from
the machine-readable tables in out/ (written by scripts/run_forecast.py).

Two guarantees, so neither side can drift silently:
  1. RECOMPUTE: each paper value is recomputed from out/*.csv (or from
     first principles, e.g. the channel-30 overlap geometry) and compared
     against the expected literal at the paper's own rounding.
  2. TEX MATCH: the literal string is confirmed to appear in the .tex
     source (paper/forecast_section.tex + paper/main.tex).

Run from the repository root (or anywhere):  python3 scripts/check_paper_numbers.py
Exit status 0 = all checks pass; 1 = at least one mismatch.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
PAPER = ROOT / "paper"
from baonoise import channels, survey
from baonoise.constants import HI_REST_FREQUENCY_MHZ


# ---------------------------------------------------------------- helpers
def read_csv(name: str) -> list[dict]:
    with open(OUT / name, newline="") as fh:
        return list(csv.DictReader(fh))


def fmt(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


class Checker:
    def __init__(self, tex: str):
        self.tex = tex
        self.failures = 0
        self.n = 0

    def check(self, label: str, computed: float, literal: str, nd: int,
              tex_needle: str | None = None):
        """computed (rounded to nd places) must equal `literal`, and the
        literal (or tex_needle override) must appear in the tex source."""
        self.n += 1
        got = fmt(computed, nd)
        ok_val = got == literal
        needle = literal if tex_needle is None else tex_needle
        ok_tex = needle in self.tex
        status = "PASS" if (ok_val and ok_tex) else "FAIL"
        if status == "FAIL":
            self.failures += 1
        detail = ""
        if not ok_val:
            detail += f"  [recomputed {got} != paper {literal}]"
        if not ok_tex:
            detail += f"  [literal '{needle}' not found in .tex]"
        print(f"{status}  {label:58s} {literal:>8s}{detail}")

    def literal_only(self, label: str, needle: str):
        """Presence check only (for numbers with no CSV source here)."""
        self.n += 1
        ok = needle in self.tex
        if not ok:
            self.failures += 1
        print(f"{'PASS' if ok else 'FAIL'}  {label:58s} {needle:>8s}"
              + ("" if ok else "  [not found in .tex]"))


def main() -> int:
    tex = ((PAPER / "forecast_section.tex").read_text()
           + (PAPER / "main.tex").read_text())
    ck = Checker(tex)

    # ------------------------------------------------ survey-level table
    rt = {r["scenario"]: r for r in read_csv("required_times.csv")}
    sig = lambda s: float(rt[s]["sig_at_2yr"])
    pen = lambda s: float(rt[s]["time_penalty_vs_clean"])

    print("-- Table 2: survey level --")
    ck.check("clean (S/N)_A at 2 yr", sig("clean"), "63.57", 2)
    ck.check("pilot-proxy (S/N)_A at 2 yr", sig("measured"), "61.96", 2)
    ck.check("pilot-proxy Fourier (S/N)_A", sig("measured_fourier"), "61.89", 2)
    ck.check("uniform 25% DTV (S/N)_A", sig("uniform25_dtv"), "60.98", 2)
    ck.check("uniform 50% DTV (S/N)_A", sig("uniform50_dtv"), "57.88", 2)
    ck.check("uniform 75% DTV (S/N)_A", sig("uniform75_dtv"), "53.89", 2)
    ck.check("uniform 97% DTV (S/N)_A", sig("uniform97_dtv"), "49.70", 2)
    ck.check("uniform 50% whole band (S/N)_A", sig("uniform50_chime"), "48.84", 2)
    ck.check("ch30 excised (S/N)_A", sig("ch30_excised"), "63.02", 2)
    ck.check("ch30 retained (S/N)_A", sig("ch30_kept"), "63.19", 2)
    ck.check("ch30 retained Fourier (S/N)_A", sig("ch30_kept_fourier"), "60.26", 2)
    ck.check("pilot-proxy survey penalty", pen("measured"), "1.032", 3)
    ck.check("pilot-proxy Fourier penalty", pen("measured_fourier"), "1.034", 3)
    ck.check("uniform 25% penalty", pen("uniform25_dtv"), "1.076", 3)
    ck.check("uniform 50% penalty", pen("uniform50_dtv"), "1.152", 3)
    ck.check("uniform 75% penalty", pen("uniform75_dtv"), "1.219", 3)
    ck.check("uniform 97% penalty", pen("uniform97_dtv"), "1.256", 3)
    ck.check("whole-band 50% penalty (x2 law)", pen("uniform50_chime"), "2.000", 3)
    ck.check("ch30 excised penalty", pen("ch30_excised"), "1.008", 3)
    ck.check("ch30 retained penalty", pen("ch30_kept"), "1.013", 3)
    ck.check("ch30 retained Fourier penalty", pen("ch30_kept_fourier"), "1.064", 3)
    ck.check("clean (S/N)_A at 1 on-sky yr",
             float(rt["clean"]["sig_at_1yr_onsky"]), "48.84", 2)
    ck.check("clean survey 5-sigma hours",
             float(rt["clean"]["hours_5sig"]), "208.8", 1)

    # prose percentages derived from the penalties
    print("-- Prose: survey penalties as percentages --")
    ck.check("pilot-proxy penalty as per cent", 100 * (pen("measured") - 1),
             "3.2", 1, tex_needle="3.2 per cent")
    ck.check("uniform 50% penalty as per cent", 100 * (pen("uniform50_dtv") - 1),
             "15.2", 1, tex_needle="15.2 per cent")
    ck.check("ch30-excision penalty as per cent", 100 * (pen("ch30_excised") - 1),
             "0.8", 1, tex_needle="0.8 per cent")
    ck.check("uniform 50% DTV factor (abstract)", pen("uniform50_dtv"),
             "1.15", 2, tex_needle="1.15")

    # ------------------------------------------------ per-bin table
    bl = {(r["zbin"], r["scenario"]): r for r in read_csv("bin_level_targets.csv")}
    yr5 = lambda z, s: (float(bl[(z, s)]["hours_bin5sig"])
                        / survey.OVERVIEW_ONSKY_YEAR_HOURS)
    yrda = lambda z, s: (float(bl[(z, s)]["hours_da2pct"])
                         / survey.OVERVIEW_ONSKY_YEAR_HOURS)

    print("-- Table 3: bin level (on-sky years) --")
    for z, s, v5, vda in [
        ("1.40-1.50", "clean", "0.175", "0.315"),
        ("1.40-1.50", "measured", "0.235", "0.421"),
        ("1.40-1.50", "measured_fourier", "0.237", "0.424"),
        ("1.40-1.50", "uniform50_dtv", "0.350", "0.629"),
        ("1.40-1.50", "ch30_kept_fourier", "1.091", "1.963"),
        ("1.40-1.50", "ch30_excised", "0.201", "0.359"),
        ("1.60-1.70", "clean", "0.209", "0.411"),
        ("1.60-1.70", "measured", "0.286", "0.547"),
    ]:
        ck.check(f"{z} {s}: 5-sigma yr", yr5(z, s), v5, 3)
        ck.check(f"{z} {s}: D_A<=2% yr", yrda(z, s), vda, 3)
    ck.check("1.40-1.50 ch30 kept (IV): D_A yr",
             yrda("1.40-1.50", "ch30_kept"), "0.373", 3)

    print("-- Prose: derived factors --")
    ck.check("worst-bin D_A factor (pilot-proxy)",
             yrda("1.40-1.50", "measured") / yrda("1.40-1.50", "clean"),
             "1.34", 2)
    ck.check("ch30-excision direct factor (fix #2)",
             yrda("1.40-1.50", "ch30_excised") / yrda("1.40-1.50", "clean"),
             "1.14", 2)
    ck.check("ch30 retained-F factor vs clean",
             yrda("1.40-1.50", "ch30_kept_fourier") / yrda("1.40-1.50", "clean"),
             "6.24", 2)

    # ------------------------------------------------ channel-30 geometry
    print("-- Channel-30 analytic geometry --")
    lo, hi = (HI_REST_FREQUENCY_MHZ / 2.50,
              HI_REST_FREQUENCY_MHZ / 2.40)   # z in [1.40, 1.50]
    c30 = channels.channel_edges(30)
    q = (min(hi, c30[1]) - max(lo, c30[0])) / (hi - lo)
    ck.check("ch30 overlap bandwidth [MHz]",
             min(hi, c30[1]) - max(lo, c30[0]), "3.84", 2)
    ck.check("ch30 overlap fraction q", q, "0.162", 3)
    ck.check("ch30 volume cost as per cent", 100 * q, "16.2", 1,
             tex_needle="16.2 per cent")
    ck.check("crossover f_star", 1.0 / (1.0 + math.sqrt(1.0 - q)), "0.522", 3)
    ck.check("analytic R_exc = 1/sqrt(1-q)", 1.0 / math.sqrt(1.0 - q),
             "1.09", 2)

    # ------------------------------------------------ Fig-31 regression
    fv = {float(r["z_center"]): r for r in read_csv("fig31_validation.csv")}
    print("-- Sec 5.2: Overview regression --")
    ck.check("sigma(D_V)/D_V clean z=0.85",
             float(fv[0.85]["sigma_dv_clean_pct"]), "0.47", 2)
    ck.check("sigma(D_V)/D_V clean z=1.45",
             float(fv[1.45]["sigma_dv_clean_pct"]), "0.61", 2)
    ck.check("sigma(D_V)/D_V clean z=2.43",
             float(fv[2.433]["sigma_dv_clean_pct"]), "1.03", 2)
    ck.check("sigma(D_V)/D_V pilot-proxy z=1.45",
             float(fv[1.45]["sigma_dv_representative_pct"]), "0.733", 3)
    ck.check("sigma(D_V)/D_V clean z=1.65",
             float(fv[1.65]["sigma_dv_clean_pct"]), "0.706", 3)
    ck.check("sigma(D_V)/D_V pilot-proxy z=1.65",
             float(fv[1.65]["sigma_dv_representative_pct"]), "0.852", 3)

    # -------------------------------------- matched-fiducial comparison
    comparison = {
        (r["metric"], r["zbin"], r["scenario"]): r
        for r in read_csv("fiducial_comparison.csv")
    }
    clean = comparison[("survey_5sigma", "survey", "clean")]
    planck_clean = float(clean["planck_hours"])
    pact_clean = float(clean["pact_hours"])
    max_penalty_shift = max(
        abs(float(row["relative_penalty_shift_pct"]))
        for row in comparison.values()
    )
    print("-- Sec 5.4: matched Planck/P-ACT comparison --")
    ck.check("Planck clean survey 5-sigma hours", planck_clean,
             "208.8", 1)
    ck.check("P-ACT clean survey 5-sigma hours", pact_clean,
             "186.9", 1)
    ck.check("P-ACT absolute-time shift",
             100.0 * (pact_clean / planck_clean - 1.0), "-10.5", 1)
    ck.check("maximum fiducial masking-penalty shift",
             max_penalty_shift, "0.420", 3)

    # -------------------------------------- Bull foreground comparison
    foreground = {
        row["scenario"]: row
        for row in read_csv("foreground_sensitivity.csv")
    }
    measured_fg = foreground["measured"]
    clean_fg_shift = 100.0 * (
        float(measured_fg["comparison_clean_hours"])
        / float(measured_fg["fiducial_clean_hours"]) - 1.0
    )
    max_fg_penalty_shift = max(
        abs(float(row["relative_penalty_shift_pct"]))
        for row in foreground.values()
    )
    print("-- Sec 5.4: Bull-2015 foreground comparison --")
    ck.check("Bull measured worst-bin D_A penalty",
             float(measured_fg["fiducial_worst_da_penalty"]),
             "1.46", 2)
    ck.check("Bull uniform-50 survey penalty",
             float(foreground["uniform50_dtv"]["fiducial_penalty"]),
             "1.28", 2)
    ck.check("Bull foreground clean-time shift", clean_fg_shift,
             "3.3", 1, tex_needle="3.3 per cent")
    ck.check("maximum Bull survey-penalty shift", max_fg_penalty_shift,
             "0.43", 2, tex_needle="0.43 per cent")

    # ------------------------------------------------ normalizations
    print("-- Normalisations --")
    ck.literal_only("on-sky year definition", "8,760")
    ck.literal_only("two on-sky years", "17,520")

    print(f"\n{ck.n - ck.failures}/{ck.n} checks passed.")
    return 1 if ck.failures else 0


if __name__ == "__main__":
    sys.exit(main())

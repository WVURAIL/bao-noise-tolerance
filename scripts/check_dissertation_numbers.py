#!/usr/bin/env python3
"""Dissertation number gate: every load-bearing number in the dissertation
must match its machine-readable source, and known-stale values must be gone.

The sibling `check_paper_numbers.py` guards the paper against `out/*.csv`;
this script does the same for the dissertation, whose LaTeX lives outside the
repository (Overleaf). Point it at the source:

    # Overleaf has a git bridge (Menu -> Git), so a local clone works:
    #   git clone https://git.overleaf.com/<project-id> dissertation-tex
    python3 scripts/check_dissertation_numbers.py \
        --tex ../dissertation-tex \
        --summary-json ../pilot-proxy/data/provenance/dissertation_summary_v2.json

`--tex` accepts .tex files, directories (searched recursively for *.tex), or a
plain-text export (e.g. pdftotext output) -- table layout can scramble in text
extraction, so .tex is authoritative and extraction runs are advisory.

Three check kinds, all on a normalized text (unicode dashes -> '-',
multiplication sign -> 'x', digit-group commas removed, TeX comments stripped):

  REQUIRE      a value or phrase that must appear (source-of-truth quotes,
               evidence anchors that are still to be added);
  FORBID       a known-stale literal that must be gone;
  FORBID-PAIR  two numbers that cannot both be right; fails only while both
               are present, so fixing either side clears it.

CSV-driven checks recompute their needles from `out/optimal_thresholds.csv`
and `out/fine_operating_points.csv` at the dissertation's rounding, so a
forecast rerun moves the expectation automatically. JSON checks read the
pilot-proxy snapshot (`--summary-json`); they SKIP when it is not supplied.

Exit status 0 = all checks pass; 1 = at least one FAIL. A red run is the
to-do list: each FAIL line says what to change and where the truth lives.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


# ---------------------------------------------------------------- normalize
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"),
                        "-")
_QUOTES = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'}
_SPACES = dict.fromkeys(map(ord, "\u00a0\u2009\u202f\u2005\u2006"), " ")
_MULT = {0x00d7: "x", 0x2248: "~", 0x223c: "~"}


def normalize(text: str, *, tex: bool = False) -> str:
    """One matching surface for .tex and extracted text.

    Unicode dashes/minus -> '-', multiplication sign -> 'x', approx signs ->
    '~', curly quotes straightened, hard/thin spaces -> ' ', digit-group
    commas removed (1,566 -> 1566), TeX comments stripped when ``tex=True``,
    whitespace collapsed. Case is preserved.
    """
    s = text.translate({**_DASHES, **_QUOTES, **_SPACES, **_MULT})
    if tex:
        s = re.sub(r"(?<!\\)%.*", "", s)          # comments, not literal \%
        s = s.replace("\\%", "%").replace("\\,", " ").replace("~", " ")
        s = s.replace("\\times", " x ")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return re.sub(r"\s+", " ", s)


def load_tex(paths: list[str]) -> tuple[str, list[Path]]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("*.tex"))
            if not found:
                sys.exit(f"error: no .tex files under {p}")
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"error: --tex path not found: {p}")
    body = "".join(
        normalize(f.read_text(encoding="utf-8", errors="replace"),
                  tex=f.suffix == ".tex")
        for f in files)
    return body, files


# ---------------------------------------------------------------- checker
class Checker:
    def __init__(self, text: str):
        self.text = text
        self.n = 0
        self.failures = 0
        self._section = None

    def _emit(self, status: str, label: str, msg: str) -> None:
        self.n += 1
        if status == "FAIL":
            self.failures += 1
        tail = f"  [{msg}]" if (msg and status != "PASS") else ""
        print(f"{status:4s}  {label:62s}{tail}")

    def section(self, title: str) -> None:
        if title != self._section:
            print(f"\n-- {title} --")
            self._section = title

    def require(self, label: str, pattern: str, msg: str) -> None:
        ok = re.search(pattern, self.text) is not None
        self._emit("PASS" if ok else "FAIL", label,
                   f"missing; {msg}" if not ok else msg)

    def value(self, label: str, needles: list[str], msg: str) -> None:
        """Any one of several plain-substring renderings must appear."""
        ok = any(n in self.text for n in needles)
        self._emit("PASS" if ok else "FAIL", label,
                   "" if ok else f"none of {needles} found; {msg}")

    def forbid(self, label: str, pattern: str, msg: str) -> None:
        hit = re.search(pattern, self.text)
        self._emit("PASS" if hit is None else "FAIL", label,
                   "" if hit is None else f"stale '{hit.group(0)}'; {msg}")

    def forbid_pair(self, label: str, pat_a: str, pat_b: str,
                    msg: str) -> None:
        both = (re.search(pat_a, self.text) is not None
                and re.search(pat_b, self.text) is not None)
        self._emit("FAIL" if both else "PASS", label,
                   f"both present, at most one can be right; {msg}"
                   if both else "")

    def skip(self, label: str, msg: str) -> None:
        self._emit("SKIP", label, msg)


# ---------------------------------------------------------------- sources
def read_csv(name: str) -> list[dict]:
    with open(OUT / name, newline="") as fh:
        return list(csv.DictReader(fh))


def threshold_rows() -> dict[int, dict]:
    """Operating rows of out/optimal_thresholds.csv.

    The dissertation's Table 9.4 quotes the *product-basis* operating points;
    the sigma_null rows differ in margin/penalty for the same eta.
    """
    rows = {}
    for r in read_csv("optimal_thresholds.csv"):
        if r.get("eta") and r.get("basis") == "product":
            rows[int(r["ch"])] = r
    return rows


def fine_rows() -> dict[int, dict]:
    """Product-basis operating rows of out/fine_operating_points.csv."""
    rows = {}
    for r in read_csv("fine_operating_points.csv"):
        if r.get("multiplier_q16") and r.get("basis") == "product":
            rows[int(r["ch"])] = r
    return rows


def frac_needles(x: float) -> list[str]:
    """A fraction as quoted raw (4 dp) or as a percentage (1 dp)."""
    return [f"{100 * x:.1f}%", f"{100 * x:.1f} %", f"{x:.4f}"]


def num_needles(x: float, nds: tuple[int, ...] = (3, 2)) -> list[str]:
    """A number at fixed roundings; every needle keeps >= 3 significant
    characters so short strings like '2.1' can never match by accident."""
    out = []
    for nd in nds + ((1,) if x >= 100 else ()):
        s = f"{x:.{nd}f}"
        if len(s.replace(".", "").lstrip("0")) >= 3 and s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------- registry
def run_checks(ck: Checker, summary: dict | None) -> None:
    # ---- Fig. 9.4 / SS9.7: one comparison population --------------------
    ck.section("Fig. 9.4 / SS9.7 -- keep-everything on one population")
    if summary is None:
        ck.skip("summary_v2 policy invariants",
                "pass --summary-json <pilot-proxy>/data/provenance/"
                "dissertation_summary_v2.json")
    else:
        pol = {p["policy_key"]: p
               for p in summary["bao_policy_case"]["policies"]}
        keep = float(pol["keep_everything"]["residual_multiple"])
        mad = float(pol["mad_1p8"]["residual_multiple"])
        ratio = max(keep, mad) / min(keep, mad)
        ok = ratio < 2.0
        ck._emit("PASS" if ok else "FAIL",
                 "keep vs MAD residual multiple within 2x (one population)",
                 "" if ok else
                 f"keep={keep:g}x vs MAD={mad:g}x (ratio {ratio:.1f}); "
                 "regenerate keep_everything via scripts/policy_comparison.py"
                 " --json (expect ~1566x / 3.35x), then re-export Fig 9.4")
        for key, p in pol.items():
            ck.value(f"policy '{key}' multiple quoted",
                     [f"{int(p['residual_multiple'])}x"],
                     "text/figure must quote the snapshot; if the snapshot is"
                     " being regenerated, rerun this gate after")
    ck.forbid("stale keep-everything caption", r"316x over",
              "Fig 9.4 caption is the all-frames r_keep population; regenerate"
              " after the snapshot fix")
    ck.require("SS9.7 incumbent-comparison multiple", r"1566x",
               "keep 2.35 -> 1566x on the acquisitions>=8 base")

    # ---- Table 9.4 <- out/optimal_thresholds.csv ------------------------
    ck.section("Table 9.4 <- out/optimal_thresholds.csv")
    for ch, r in sorted(threshold_rows().items()):
        # Needles follow the table's own renderings: eta 2 dp, f as a 1 dp
        # percentage, r at 4 dp, margin as "N.Nx", penalty 2 dp (whole "Nx"
        # when it is quoted in prose, e.g. channel 31's 177x time cost).
        pen = float(r["penalty"])
        ck.value(f"ch {ch}: eta*", [f"{float(r['eta']):.2f}"],
                 "quote the CSV at the table's rounding")
        ck.value(f"ch {ch}: kept fraction f", frac_needles(float(r["f"])),
                 "quote the CSV at the table's rounding")
        ck.value(f"ch {ch}: residual r", [f"{float(r['r_fine']):.4f}"],
                 "quote the CSV at the table's rounding")
        ck.value(f"ch {ch}: margin",
                 [f"{float(r['margin']):.1f}x", f"{float(r['margin']):.2f}"],
                 "quote the CSV at the table's rounding")
        ck.value(f"ch {ch}: penalty",
                 num_needles(pen) + ([f"{pen:.0f}x"] if pen >= 100 else []),
                 "quote the CSV at the table's rounding")

    # ---- Table 8.1 <- out/fine_operating_points.csv ---------------------
    ck.section("Table 8.1 <- out/fine_operating_points.csv")
    for ch, r in sorted(fine_rows().items()):
        ck.value(f"ch {ch}: eta_q16 filled",
                 [str(int(float(r["multiplier_q16"])))],
                 "fill the pending cells from the epoch-restricted rerun of"
                 " scripts/fine_operating_point.py; CSV is authoritative")
        if r.get("r_late"):
            ck.value(f"ch {ch}: r_late", [f"{float(r['r_late']):.3f}"],
                     "quote the CSV at the table's rounding")

    # ---- SS9.3 quarterly-table provenance --------------------------------
    ck.section("SS9.3 quarterly-table provenance")
    ck.require("fs/2 legacy epoch named next to the quarterly table",
               r"(fs\s*/\s*2|half-?band).{0,600}quarterly"
               r"|quarterly.{0,600}(fs\s*/\s*2|half-?band)",
               "rewrite per PAPER_PLAN.md Amendment A1: legacy bank b0dce17a,"
               " center-at-Nyquist, pilots suppressed 39-47 dB except ch 30")
    ck.forbid("'unrecorded' provenance claim", r"unrecorded",
              "the generating rule IS recorded"
              " (analysis/survey_composition.py, 2026-07-18)")

    # ---- SS9.4 hand-back range & SS6.3 factor of ten ----------------------
    ck.section("SS9.4 / SS6.3 -- ranges and factors")
    ck.forbid("stale hand-back range", r"5\.9-7\.8",
              "with 11.4 dB at 200 ns, 8.2/3.6 dB at the cuts the hand-back"
              " is 3.2-7.8 dB (2.1-6x); SS9.9 already says 3.2 dB")
    ck.require("corrected hand-back range", r"3\.2-7\.8", "see above")
    ck.forbid_pair("fxfft statistic-move dB self-consistent",
                   r"0\.00026 ?dB", r"5\.9 ?x ?10",
                   "10 log10(1 - 5.9e-4) = -0.0026 dB: one of the two is 10x"
                   " off (also quoted in SS6.7 and docs/DESIGN_DECISIONS.md)")

    # ---- Abstract / Ch.1 / Ch.11 vs Tables 9.6/9.8 ----------------------
    ck.section("Abstract / Ch.1 / Ch.11 vs Tables 9.6/9.8")
    ck.forbid("abstract: 'Ten contiguous channels'", r"Ten contiguous",
              "21 of 23 (16-36) have products; rewrite from Tables 9.6/9.8")
    ck.forbid("abstract: 'remaining thirteen channels'",
              r"remaining thirteen", "14-15 remain; the rest are measured")
    ck.forbid("'ten measured channels'", r"ten measured channels",
              "21 measured channels; Table 11.1 needs rows for 16-26")
    ck.forbid("DTV-vs-noise range understated", r"10(-| to )35 dB",
              "measured range is roughly +0 to -44 dB (Tables 9.6/9.8)")
    ck.forbid("tolerance-excess range", r"three hundred thousand",
              "measured spread is ~3e2 (ch 33) to ~5e6 (ch 30)")
    ck.forbid("'about 42 ms' as a CHIME property", r"about 42 ms",
              "current X-engine integrates ~31 ms; 41.94 ms is the detector"
              " frame -- reword")
    ck.forbid("sign-off count", r"three transmitter sign",
              "Fig 8.1 names four (19, 20, 26, 27) and SS9.10 adds ch 32")
    ck.forbid("Eisenstein 2005 sample size", r"half a million",
              "Eisenstein et al. 2005: 46748 SDSS LRGs"
              " (Cole et al. 2005: ~221000 2dFGRS)")
    ck.require("corrected LRG count", r"46748|46,748", "see above")
    ck.forbid("'eight years'", r"eight years",
              "Dec 2018 - Jul 2026 = 7.6 yr; the repos say 7.6 everywhere")
    ck.require("7.6 yr archive span", r"7\.6 (yr|years)", "see above")
    ck.forbid("snapshot count", r"8500 snapshots",
              "SS7.1 says 9192 probed / 8962 valid; pick one census")
    ck.forbid("Vancouver driving distance", r"389 km",
              "great-circle DRAO->Mt. Seymour ~240 km; 389 km is the road")
    ck.forbid("Seattle driving distance", r"451 km",
              "great-circle DRAO->Seattle ~275 km; 451 km is the road")

    # ---- SS9.5 / SS9.7 chain arithmetic ------------------------------------
    ck.section("SS9.5 / SS9.7 chain arithmetic")
    ck.forbid("net chain gain", r"22\.9 dB",
              "r_proxy/p_kept = 0.0359/10^-4.495 = +30.5 dB NET (already"
              " includes the -7.6 dB ground filter); redo the ledger")
    ck.forbid("stale three-channel frame-stage r_proxy list",
              r"0\.057, 0\.0022",
              "Table 9.6 on-air shelves give 0.085 / 3.9e-4 / 1.35e-3")

    # ---- SS6.2 encoding ----------------------------------------------------
    ck.section("SS6.2 encoding / SS10.1 packer duties")
    ck.forbid("'consumes the receiver's native format directly'",
              r"native format directly",
              "native samples are excess-8; the adapter repacks losslessly")
    ck.forbid("'sign extension' as the unpack story", r"sign extension",
              "repack to two's complement (per byte, XOR 0x88); the kernel"
              " sign-extends nibbles")
    ck.require("XOR 0x88 stated", r"XOR 0x88", "state the packer conversion")

    # ---- evidence anchors still to add ----------------------------------
    ck.section("Evidence anchors (red until the artifact exists and is cited)")
    ck.require("bootstrap-rule P_fa stated", r"48\.5 ?(%|per ?cent)",
               "the null_power_ratio point spends 48.5% of verified-quiet"
               " time (docs/DESIGN_DECISIONS.md); put it in SS5.5 and Ch. 9")
    ck.require("fine-gain Monte Carlo cited", r"fine_gain_mc|measure_fine_gain",
               "run tools/measure_fine_gain.py, commit docs/evidence/"
               "fine_gain_mc_<date>/, cite it for the coherent-gain credit")
    ck.require("ROC / Youden-J analysis cited", r"[Yy]ouden",
               "commit youden_j.py with the survey analysis and cite the"
               " coarse-vs-fine ROC table")

    # ---- revision-artifact phrasing --------------------------------------
    ck.section("Revision-artifact phrasing (editor's notes to remove)")
    for phrase in ("supplied for this revision", "not invented here",
                   "the present draft", "left pending", "remembered analysis",
                   "dissertation-source bundle", "the revised analysis"):
        ck.forbid(f"'{phrase}'", re.escape(phrase),
                  "replace with a plain evidence-status statement")


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tex", nargs="+", required=True,
                    help=".tex files, directories of .tex (an Overleaf git"
                         " clone), or a plain-text export")
    ap.add_argument("--summary-json", default=None,
                    help="pilot-proxy data/provenance/"
                         "dissertation_summary_v2.json (policy invariants"
                         " SKIP without it)")
    args = ap.parse_args(argv)

    text, files = load_tex(args.tex)
    print(f"checking {len(files)} source file(s), {len(text):,} chars"
          " normalized")
    summary = None
    if args.summary_json:
        summary = json.loads(Path(args.summary_json).read_text())

    ck = Checker(text)
    run_checks(ck, summary)
    print(f"\n{ck.n - ck.failures}/{ck.n} checks passed.")
    if ck.failures:
        print("Each FAIL line above names the fix and the authoritative"
              " source.")
    return 1 if ck.failures else 0


if __name__ == "__main__":
    sys.exit(main())

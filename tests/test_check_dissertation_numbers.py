# coding=utf-8
"""Gate the dissertation number checker itself.

The checker is only trustworthy if its matching surface is: normalization
must fold the unicode variants LaTeX and PDF extraction produce into one
form, the three check kinds must fail in exactly their advertised
directions, and the CSV row selectors must keep resolving the operating rows
of the shipped out/ tables (so a column rename breaks CI here, loudly,
rather than silently green-lighting the dissertation).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_dissertation_numbers",
    ROOT / "scripts" / "check_dissertation_numbers.py")
cdn = importlib.util.module_from_spec(spec)
sys.modules["check_dissertation_numbers"] = cdn
spec.loader.exec_module(cdn)


def test_normalize_folds_unicode_and_commas():
    s = "1,566\u00d7 over \u2014 5.9\u20137.8 dB \u2248 \u22120.00026"
    n = cdn.normalize(s)
    assert "1566x over" in n
    assert "5.9-7.8 dB" in n
    assert "~ -0.00026" in n


def test_normalize_tex_strips_comments_not_escaped_percent():
    n = cdn.normalize("48.5\\% kept % trailing comment\nnext", tex=True)
    assert "48.5%" in n and "trailing" not in n and "next" in n


def test_check_kinds_fail_in_their_directions(capsys):
    ck = cdn.Checker("keep 1566x and 316x over; both 104 ns and 110 ns")
    ck.require("present", r"1566x", "m")
    ck.require("absent", r"9999x", "m")
    ck.forbid("stale present", r"316x over", "m")
    ck.forbid("stale absent", r"555x", "m")
    ck.forbid_pair("both present", r"104 ns", r"110 ns", "m")
    ck.forbid_pair("one present", r"104 ns", r"777 ns", "m")
    out = capsys.readouterr().out
    assert ck.n == 6 and ck.failures == 3
    assert out.count("FAIL") == 3 and out.count("PASS") == 3


def test_value_accepts_any_rendering():
    ck = cdn.Checker("kept fraction 51.8% of frames")
    ck.value("f", cdn.frac_needles(0.5183), "m")
    assert ck.failures == 0


def test_num_needles_never_too_short():
    for x in (2.099, 0.237, 1.4008, 177.31, 6.587):
        assert all(len(n.replace(".", "").lstrip("0")) >= 3
                   for n in cdn.num_needles(x))


def test_csv_operating_rows_resolve():
    thr = cdn.threshold_rows()
    assert set(thr) >= {31, 32, 33, 35}
    assert abs(float(thr[32]["eta"]) - 1.01) < 0.005
    fine = cdn.fine_rows()
    assert fine and all(r["multiplier_q16"] for r in fine.values())


def test_end_to_end_exit_codes(tmp_path):
    # A source containing every stale literal must fail; the same source
    # with requires satisfied and forbids absent must pass. Build the green
    # text from the registry's own CSV needles so the test tracks out/.
    thr = cdn.threshold_rows()
    fine = cdn.fine_rows()
    green = " ".join(
        [f"{float(r['eta']):.2f} {100 * float(r['f']):.1f}% "
         f"{float(r['r_fine']):.4f} {float(r['margin']):.1f}x "
         f"{float(r['penalty']):.2f} {float(r['penalty']):.0f}x"
         for r in thr.values()]
        + [str(int(float(r["multiplier_q16"]))) + " "
           + (f"{float(r['r_late']):.3f}" if r.get("r_late") else "")
           for r in fine.values()]
        + ["1566x over now", "the fs/2 legacy epoch quarterly table",
           "3.2-7.8 dB", "46748 LRGs", "7.6 yr", "XOR 0x88",
           "48.5% of verified-quiet time", "fine_gain_mc evidence",
           "Youden-J table"])
    red = green + " eight years; 316x over; 5.9-7.8 dB; ten measured channels"
    g, r = tmp_path / "g.txt", tmp_path / "r.txt"
    g.write_text(green)
    r.write_text(red)
    assert cdn.main(["--tex", str(g)]) == 0
    assert cdn.main(["--tex", str(r)]) == 1


def test_summary_invariant_flags_split_population(tmp_path, capsys):
    src = tmp_path / "s.txt"
    src.write_text("316x 1587x 382x 24x 1566x over; " + _green_min())
    sj = tmp_path / "summary.json"
    sj.write_text(
        '{"bao_policy_case": {"policies": ['
        '{"policy_key": "keep_everything", "residual_multiple": 316},'
        '{"policy_key": "mad_1p8", "residual_multiple": 1587},'
        '{"policy_key": "sk", "residual_multiple": 382},'
        '{"policy_key": "pilot", "residual_multiple": 24}]}}')
    rc = cdn.main(["--tex", str(src), "--summary-json", str(sj)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "one population" in out and "ratio 5.0" in out


def _green_min() -> str:
    thr = cdn.threshold_rows()
    fine = cdn.fine_rows()
    return " ".join(
        [f"{float(r['eta']):.2f} {100 * float(r['f']):.1f}% "
         f"{float(r['r_fine']):.4f} {float(r['margin']):.1f}x "
         f"{float(r['penalty']):.2f} {float(r['penalty']):.0f}x"
         for r in thr.values()]
        + [str(int(float(r["multiplier_q16"]))) + " "
           + (f"{float(r['r_late']):.3f}" if r.get("r_late") else "")
           for r in fine.values()]
        + ["fs/2 legacy quarterly", "3.2-7.8", "46748", "7.6 yr",
           "XOR 0x88", "48.5%", "fine_gain_mc", "Youden"])

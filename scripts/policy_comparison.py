#!/usr/bin/env python
"""The four-way decision for each measured channel.

Runs every available flagger on the same frames, then asks which of the four
things a survey can actually do (keep, excise, flag with the incumbent, flag
on the pilot) is cheapest at a given residual-bias tolerance.

    python scripts/policy_comparison.py products/*.npz
    python scripts/policy_comparison.py products/*.npz --bias-tolerance 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                            # noqa: E402

from baonoise import incumbent, residual                      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("products", nargs="+", type=Path)
    ap.add_argument("--bias-tolerance", type=float, default=None,
                    help="largest residual r whose parameter bias is acceptable; "
                         "omit for a pure noise trade")
    ap.add_argument("--mad-k", type=float, default=incumbent.DEFAULT_MAD_K)
    ap.add_argument("--sk-nsigma", type=float, default=3.0)
    ap.add_argument("--min-frames", type=int, default=8,
                    help="shortest acquisition a block statistic may use")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    out = []
    for path in args.products:
        try:
            results, meta = incumbent.compare_flaggers(
                path, mad_k=args.mad_k, sk_nsigma=args.sk_nsigma,
                min_frames=args.min_frames)
        except ValueError as exc:
            print(f"\n{path.name}: skipped ({exc})")
            continue

        by = {r.name: r for r in results}
        keep = next(r for r in results if r.name == "keep everything")
        proxy = by["pilot proxy"]
        incs = [(r.name, r.f, r.r) for r in results
                if r.name not in ("keep everything", "pilot proxy")]

        cmp_ = residual.compare_policies(
            meta["channel"], keep_r=keep.r,
            proxy=("pilot proxy", proxy.f, proxy.r),
            incumbents=incs, bias_tolerance=args.bias_tolerance)

        print(f"\n{'=' * 80}")
        print(f"ch{meta['channel']}  freq_id={meta['freq_id']}  "
              f"{meta['nu_mhz']:.3f} MHz  duty cycle {meta['duty_cycle']:.4f}")
        print(f"  scored on {meta['n_scored']} of {meta['n_frames']} frames "
              f"({meta['n_blocks']} acquisitions >= {args.min_frames} frames, "
              f"longest {meta['block_max_seconds']:.3f} s)")
        print(f"  SK null calibrated to n_accum={meta['sk_n_accum']:.4g}, "
              f"median SK {meta['sk_median']:.4f}")
        print(f"{'=' * 80}")
        for r in results:
            print(f"  {r.name:34s} f={r.f:7.4f}  shelf {r.shelf_kept_db:8.2f} dB "
                  f"({r.reduction_db:6.2f} dB removed)")
        print()
        print(cmp_.table())

        print(f"\n  which policy wins, as the bias tolerance tightens:")
        for lo, hi, p in reversed(cmp_.tolerance_map()):
            hi_s = "inf" if np.isinf(hi) else f"{hi:.4g}"
            print(f"    r tolerance {lo:10.4g} .. {hi_s:>10s}   {p.name}")
        span = cmp_.winning_span("proxy")
        if span is None:
            print("    the detector's mask never wins on its own")
        else:
            print(f"    detector-only window: {cmp_.proxy_decades:.2f} decades "
                  f"of tolerance where nothing else keeps the band")

        out.append(dict(meta=meta,
                        flaggers=[vars(r) for r in results],
                        best=cmp_.best().name,
                        best_without_proxy=cmp_.best_without_proxy().name,
                        proxy_advantage=cmp_.proxy_advantage,
                        saves_the_band=cmp_.saves_the_band,
                        verdict=cmp_.verdict()))

    if args.json and out:
        args.json.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

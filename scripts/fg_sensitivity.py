#!/usr/bin/env python3
"""Foreground-residual sensitivity: compare masking *penalties* between the
fiducial bank (eps_FG=1e-6) and a bank built with eps_FG=1e-5 (x10 residuals).
The claim under test: penalties t_req/t_req_clean are stable even where
absolute times shift."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baonoise import forecast, scenarios  # noqa: E402
from baonoise.compat import import_radiofisher  # noqa: E402
from baonoise.fisherbank import FisherBank  # noqa: E402

rf, _ = import_radiofisher()
scens = {
    "measured": scenarios.measured(),
    "uniform50_dtv": scenarios.uniform(0.50, "dtv"),
    "uniform97_dtv": scenarios.uniform(0.97, "dtv"),
}

print(f"{'bank':>10s} {'scenario':>15s} {'t5_clean[hr]':>12s} "
      f"{'t5[hr]':>10s} {'penalty':>8s}")
rows = {}
for tag, path in [("fg1e-6", "fisher_bank_chime.npz"),
                  ("fg1e-5", "fisher_bank_chime_fg1e5.npz")]:
    bank = FisherBank(ROOT / "data" / path)
    fc = forecast.Forecast(bank, rf)
    t_clean = fc.required_hours(scenarios.clean(), 5.0)
    for name, sc in scens.items():
        t = fc.required_hours(sc, 5.0)
        rows[(tag, name)] = t / t_clean
        print(f"{tag:>10s} {name:>15s} {t_clean:12.1f} {t:10.1f} "
              f"{t/t_clean:8.3f}")

print("\npenalty shifts (fg1e-5 vs fg1e-6):")
for name in scens:
    a, b = rows[("fg1e-6", name)], rows[("fg1e-5", name)]
    print(f"  {name:>15s}: {a:.3f} -> {b:.3f}  ({100*(b/a-1):+.2f}%)")

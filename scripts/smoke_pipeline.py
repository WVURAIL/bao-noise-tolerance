#!/usr/bin/env python3
"""Fast schema-v2 smoke run through bank interpolation and scenarios."""
from __future__ import annotations

import time

import numpy as np

from baonoise import forecast, scenarios
from baonoise.fisherbank import FisherBank
from baonoise.resources import DEFAULT_BANK


bank = FisherBank(DEFAULT_BANK)
fc = forecast.Forecast(bank, style="perbin_A")

# Interpolation sanity at an off-grid time: finite and symmetric.
matrix = bank.F(3, 3.3e3)
if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
    raise RuntimeError("interpolated Fisher matrix is not finite and symmetric")
print(f"schema={bank.schema_version} interpolation finite/symmetric")

for scenario in [
    scenarios.clean(),
    scenarios.measured(),
    scenarios.uniform(0.5, scenarios.DTV_BAND),
    scenarios.uniform(0.97, scenarios.CHIME_BAND),
]:
    started = time.time()
    significance = fc.significance(scenario, 1e4)
    required = fc.required_hours(scenario, target=5.0)
    print(
        f"{scenario.name:20s} S(1e4hr)={significance:8.2f}  "
        f"t_req(5sig)={required:12.1f} hr [{time.time() - started:.2f}s]"
    )

factors = scenarios.measured().bin_factors_for_zbins(bank.zs)
print("measured per-bin (v_frac, w_bar):")
for index in range(bank.nbins):
    print(
        f"  z=[{bank.zs[index]:.2f},{bank.zs[index + 1]:.2f}] "
        f"v={factors[index, 0]:.4f} w={factors[index, 1]:.4f}"
    )

#!/usr/bin/env python3
"""Minimal baonoise example: from a masking table to observing-time cost.

Requires a RadioFisher checkout (sibling directory or RADIOFISHER_DIR) and
the CHIME Fisher bank shipped in data/. Runs in seconds -- no bank build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baonoise import api

fc = api.load()  # CHIME bank shipped with the repository

# Your masking measurements: ATSC channel -> fraction of time masked.
# Channels above the excision threshold (default 0.5) are excised
# (volume loss); the rest cost effective integration time.
mask = {17: 0.33, 24: 0.97, 30: 0.97, 31: 0.24, 32: 0.14, 33: 0.10, 35: 0.14}

# Survey-level 5-sigma BAO detection
print(api.required_time(fc, mask, target=5.0))

# The worst-affected redshift bin alone (bin 6: z = 1.40-1.51)
print(api.required_time(fc, mask, target=5.0, zbin=6))

# Significance after 2 calendar years (75% duty)
print(f"S(2 yr) = {api.significance(fc, 2.0, mask):.1f} sigma")

# Noise-tolerance curve: years to 5 sigma vs uniform DTV-band masking
fracs, years = api.tolerance_curve(fc, fracs=[0.0, 0.25, 0.5, 0.75, 0.97])
for f, y in zip(fracs, years):
    print(f"  {100*f:4.0f}% masked -> {y:6.3f} yr to 5 sigma (survey)")

#!/usr/bin/env python3
"""Minimal baonoise example: from a masking table to observing-time cost.

Uses the CHIME Fisher bank and quarterly rates shipped inside the package.
Runs in seconds without a RadioFisher checkout or a bank build.
"""
from baonoise import api

fc = api.load()  # shipped CHIME bank; RadioFisher is not required

# Your masking measurements: ATSC channel -> fraction of time masked.
# Channels above the excision threshold (default 0.5) are excised
# (volume loss); the rest cost effective integration time.
mask = {17: 0.33, 24: 0.97, 30: 0.97, 31: 0.24, 32: 0.14, 33: 0.10, 35: 0.14}

# Survey-level 5-sigma BAO detection
print(api.required_time(fc, mask, target=5.0))

# The worst-affected redshift bin alone (bin 6: z = 1.40-1.50)
print(api.required_time(fc, mask, target=5.0, zbin=6))

# Significance after 2 on-sky years at the Overview normalization
print(f"S(2 yr) = {api.significance(fc, 2.0, mask):.1f} sigma")

# Noise-tolerance curve: years to 5 sigma vs uniform DTV-band masking
fracs, years = api.tolerance_curve(fc, fracs=[0.0, 0.25, 0.5, 0.75, 0.97])
for f, y in zip(fracs, years):
    print(f"  {100*f:4.0f}% masked -> {y:6.3f} yr to 5 sigma (survey)")

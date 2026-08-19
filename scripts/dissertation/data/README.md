# Dissertation figure tables (tolerance side)

Frozen snapshots behind `../figures.py`; the dissertation bundle vendors the
rendered PDFs.

| table | status |
|---|---|
| bao_time_vs_masking.csv | forecast-derived (pilot-proxy `tools/make_dissertation_tables.py --bao-time-vs-masking`, computed through this package's released forecast) |
| bao_policy_case.csv | curated snapshot (channel-33 residual-policy comparison; from the pilot-proxy summary export) |
| bao_convergence.csv | LEGACY BRIDGE recovered from the published vector artwork; replacement path: direct fixed-target tolerance calculation via the Pres research bank |
| bao_two_walls.csv | LEGACY BRIDGE recovered from the published vector artwork; replacement path: regeneration from threshold_sweeps.json (all 21 channels) |

The two bridges reproduce the published curves exactly; they are recorded as
bridges, not measurements, until the direct regenerations land.

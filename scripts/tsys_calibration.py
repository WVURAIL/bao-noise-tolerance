#!/usr/bin/env python3
"""Report the CHIME system-temperature calibration and check the packaged Tsys.

The measurement file (``20190530_and_20190614_system_temperature_measurement.h5``,
CHIME calibrator transits of 2019-05-30 and 2019-06-14) carries, per frequency
channel and correlator input: Tsys for each of the two transits, the receiver
temperature, the effective area, and the Jy/K conversion. It is CHIME
collaboration calibration data and lives in the team SharePoint under
``RFI Mitigation/``; pass its local path here. Nothing in the packaged
forecasts *reads* it -- they work in ratio units with Tsys_tot = 55 K
(``baonoise.survey``) -- so this script is the calibration's one consumer: it
prints the measured summary and checks that the packaged constant sits inside
the measured spread, band-restricted to the DTV band this package prices.

    python3 scripts/tsys_calibration.py --file /path/to/..._measurement.h5

Requires h5py (not a package dependency; install it in the analysis
environment).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGED_TSYS_K = 55.0          # baonoise.survey CHIME experiment dict
DTV_BAND_MHZ = (470.0, 608.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, required=True,
                    help="path to the system-temperature measurement h5 "
                         "(team SharePoint: RFI Mitigation/)")
    args = ap.parse_args(argv)

    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required: pip install h5py")

    with h5py.File(args.file, "r") as f:
        freq = np.asarray(f["index_map/freq"])          # MHz, 1024 channels
        tsys = np.asarray(f["Tsys"])                    # (freq, input, transit)
        trecv = np.asarray(f["Trecv"])                  # (freq, input)
        jy_per_k = np.asarray(f["Jy_per_K"])            # (freq, input)
        pol = [p.decode() for p in np.asarray(f["polarisation"])]

    good = np.isfinite(tsys) & (tsys > 0)
    t = tsys[good]
    print(f"frequency coverage: {freq.min():.1f}-{freq.max():.1f} MHz "
          f"({freq.size} channels); inputs: {len(pol)} ({''.join(pol)}); "
          f"transits: {tsys.shape[2]}")
    print(f"Tsys over all usable samples ({good.sum():,}/{tsys.size:,}): "
          f"median {np.median(t):.1f} K, range {t.min():.1f}-{t.max():.1f} K")
    for i, p in enumerate(pol):
        ti = tsys[:, i, :][good[:, i, :]]
        if ti.size:
            print(f"  input {i} ({p}): median {np.median(ti):.1f} K")
    tr = trecv[np.isfinite(trecv) & (trecv > 0)]
    if tr.size:
        print(f"Trecv: median {np.median(tr):.1f} K")
    jk = jy_per_k[np.isfinite(jy_per_k) & (jy_per_k > 0)]
    if jk.size:
        print(f"Jy/K: median {np.median(jk):.1f}")

    in_band = (freq >= DTV_BAND_MHZ[0]) & (freq <= DTV_BAND_MHZ[1])
    tb = tsys[in_band][good[in_band]]
    med = float(np.median(tb))
    lo, hi = float(np.percentile(tb, 10)), float(np.percentile(tb, 90))
    print(f"\nDTV band {DTV_BAND_MHZ[0]:g}-{DTV_BAND_MHZ[1]:g} MHz: "
          f"Tsys median {med:.1f} K (10th-90th pct {lo:.1f}-{hi:.1f} K)")
    ok = lo <= PACKAGED_TSYS_K <= hi
    print(f"packaged Tsys_tot = {PACKAGED_TSYS_K:g} K "
          f"{'INSIDE' if ok else 'OUTSIDE'} the measured in-band 10th-90th "
          f"percentile band -> {'PASS' if ok else 'FAIL'}")
    print("(forecast verdicts are ratio-based and do not depend on this "
          "number; it matters for absolute-unit conversion of shelf and "
          "floor levels)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

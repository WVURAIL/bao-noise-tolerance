"""Generate the CHIME baseline-density file n(x) that RadioFisher's CHIME
experiment expects (``array_config/nx_CHIME_800.dat``).

The original file was produced by ``process_chime_baselines.py`` from a raw
baseline list that is not distributed with the repository (and the auxiliary
tarball URL is dead). This module reproduces the same processing recipe from a
synthetic feed layout matching the RadioFisher CHIME spec used in
Bull, Ferreira, Patel & Santos (2015): 5 cylinders x 256 feeds, 20 m wide,
80 m instrumented length.

Processing recipe replicated exactly from process_chime_baselines.py:
  * u = d / lambda at nu = 800 MHz
  * cut baselines with d <= Dcut = Ddish = 20 m  (the 'nx_CHIME_800.dat' case)
  * ring histogram with du = (1/30) / sqrt(FOV),
    FOV = 180deg * 1.22 * (lambda/D) * (pi/180)^2   [cylinder strip beam]
  * n(u) = counts / (2 pi u du), no renormalisation, no small-u averaging
  * saved as columns  x = u/nu,  n_x = n(u) * nu^2   (nu in MHz)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

C_MS = 3e8


def chime_feed_positions(ncyl: int = 5, nfeed: int = 256, cyl_spacing: float = 20.0,
                         cyl_length: float = 80.0) -> np.ndarray:
    """Feed (x, y) positions [m] for a CHIME-like cylinder array."""
    xs = np.arange(ncyl) * cyl_spacing
    ys = np.arange(nfeed) * (cyl_length / nfeed)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def fov_cyl(nu_mhz: float, ddish: float) -> float:
    """Cylinder field of view [rad^2] as in process_chime_baselines.py."""
    lam = C_MS / (nu_mhz * 1e6)
    return 180.0 * 1.22 * (lam / ddish) * (np.pi / 180.0) ** 2


def build_nx_file(outfile: str | Path, ncyl: int = 5, nfeed: int = 256,
                  cyl_spacing: float = 20.0, cyl_length: float = 80.0,
                  ddish: float = 20.0, nu_mhz: float = 800.0,
                  dcut: float | None = None) -> Path:
    """Compute n(x) for the cylinder layout and write the RadioFisher file."""
    outfile = Path(outfile)
    dcut = ddish if dcut is None else dcut
    lam = C_MS / (nu_mhz * 1e6)

    pos = chime_feed_positions(ncyl, nfeed, cyl_spacing, cyl_length)
    d = pdist(pos)                       # all pairwise separations [m]
    d = d[d > dcut]                      # strict cut, as in the original
    u = d / lam

    du = (1.0 / 30.0) / np.sqrt(fov_cyl(nu_mhz, ddish))
    imax = int(np.max(u) / du) + 1
    edges = np.linspace(0.0, imax * du, imax + 1)
    counts, edges = np.histogram(u, edges)
    uc = 0.5 * (edges[1:] + edges[:-1])

    n_u = counts / (2.0 * np.pi * uc * du)

    x = uc / nu_mhz
    n_x = n_u * nu_mhz**2
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(outfile, np.column_stack([x, n_x]))
    return outfile


def ensure_chime_nx(rf_dir: str | Path, data_dir: str | Path,
                    layout: str = "bull2015") -> Path:
    """Return path to a CHIME n(x) file, generating it if needed.

    layout='bull2015'  : 5 cyl x 256 feeds, 80 m  (RadioFisher paper spec)
    layout='asbuilt'   : 4 cyl x 256 feeds, 22 m spacing, 78 m instrumented
    """
    rf_official = Path(rf_dir) / "array_config" / "nx_CHIME_800.dat"
    if layout == "bull2015" and rf_official.exists():
        return rf_official
    data_dir = Path(data_dir)
    if layout == "bull2015":
        out = data_dir / "nx_CHIME_800_synth.dat"
        if not out.exists():
            build_nx_file(out)
    elif layout == "asbuilt":
        out = data_dir / "nx_CHIME_800_asbuilt.dat"
        if not out.exists():
            build_nx_file(out, ncyl=4, nfeed=256, cyl_spacing=22.0,
                          cyl_length=78.0)
    else:
        raise ValueError(f"unknown layout: {layout}")
    return out

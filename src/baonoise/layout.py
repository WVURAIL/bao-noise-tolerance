"""Generate Bao-owned CHIME baseline-density files n(x).

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from .constants import CHIME_FREQUENCY_MAX_MHZ

C_MS = 3e8


@dataclass(frozen=True)
class CylinderLayout:
    """Named geometry used to generate a baseline-density table."""
    ncyl: int
    nfeed: int
    cyl_spacing_m: float
    cyl_length_m: float
    cylinder_width_m: float


BULL2015_LAYOUT = CylinderLayout(5, 256, 20.0, 80.0, 20.0)
CHIME_ASBUILT_LAYOUT = CylinderLayout(4, 256, 22.0, 78.0, 20.0)
LAYOUTS = {"bull2015": BULL2015_LAYOUT, "asbuilt": CHIME_ASBUILT_LAYOUT}


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
                  ddish: float = 20.0,
                  nu_mhz: float = CHIME_FREQUENCY_MAX_MHZ,
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


def ensure_chime_nx(data_dir: str | Path,
                    layout: str = "bull2015") -> Path:
    """Return a Bao-owned CHIME n(x) path, generating it if needed.

    layout='bull2015'  : 5 cyl x 256 feeds, 80 m  (RadioFisher paper spec)
    layout='asbuilt'   : 4 cyl x 256 feeds, 22 m spacing, 78 m instrumented

    RadioFisher's historical root-level ``array_config`` directory is not an
    input. Forecast adapters bind Bull-2015 explicitly to Bao's packaged
    synthetic table, so an unrelated checkout file cannot change the science.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout: {layout}; choose from {sorted(LAYOUTS)}")
    data_dir = Path(data_dir)
    spec = LAYOUTS[layout]
    if layout == "bull2015":
        out = data_dir / "nx_CHIME_800_synth.dat"
    else:
        out = data_dir / "nx_CHIME_800_asbuilt.dat"
    if not out.exists():
        build_nx_file(
            out, ncyl=spec.ncyl, nfeed=spec.nfeed,
            cyl_spacing=spec.cyl_spacing_m, cyl_length=spec.cyl_length_m,
            ddish=spec.cylinder_width_m)
    return out

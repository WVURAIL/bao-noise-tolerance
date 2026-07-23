"""Compatibility shims so the (2014-era) RadioFisher code runs on a modern
scientific Python stack (scipy >= 1.14, numpy >= 2.0, no Fortran CAMB, no MPI).

Import this module *before* importing ``radiofisher``.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

_DEFAULT_RF_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "RadioFisher",   # sibling of project
    Path.home() / "work" / "RadioFisher",
    Path.cwd() / "RadioFisher",
)


def install_scipy_shims() -> None:
    """Restore scipy APIs that RadioFisher expects but scipy removed.

    * ``scipy.misc.derivative`` (removed in scipy 1.12): central finite
      difference, same signature and behavior as the old function for n=1.
    * ``scipy.integrate.simps`` (removed in scipy 1.14): an alias of simpson.
    """
    import numpy as np
    import scipy
    import scipy.integrate as si

    if not hasattr(si, "simps"):
        si.simps = si.simpson  # type: ignore[attr-defined]
    if not hasattr(si, "cumtrapz"):
        si.cumtrapz = si.cumulative_trapezoid  # type: ignore[attr-defined]
    if not hasattr(si, "trapz"):
        si.trapz = si.trapezoid  # type: ignore[attr-defined]
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]

    try:
        from scipy.misc import derivative  # noqa: F401  (older scipy: fine)
    except Exception:
        _CD_WEIGHTS = {
            3: np.array([-0.5, 0.0, 0.5]),
            5: np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0,
        }

        def derivative(func, x0, dx=1.0, n=1, args=(), order=3):  # noqa: ANN001
            if n != 1:
                raise NotImplementedError("shim only implements first derivative")
            if order not in _CD_WEIGHTS:
                raise NotImplementedError(f"order={order} not supported by shim")
            w = _CD_WEIGHTS[order]
            half = len(w) // 2
            val = 0.0
            for k, wk in enumerate(w):
                if wk == 0.0:
                    continue
                val += wk * func(x0 + (k - half) * dx, *args)
            return val / dx

        mod = types.ModuleType("scipy.misc")
        mod.derivative = derivative
        sys.modules["scipy.misc"] = mod
        scipy.misc = mod  # type: ignore[attr-defined]


def find_radiofisher_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Locate a RadioFisher checkout (env RADIOFISHER_DIR overrides)."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("RADIOFISHER_DIR"):
        candidates.append(Path(os.environ["RADIOFISHER_DIR"]))
    candidates.extend(_DEFAULT_RF_CANDIDATES)
    for cand in candidates:
        if (Path(cand) / "radiofisher" / "baofisher.py").exists():
            return Path(cand).resolve()
    raise FileNotFoundError(
        "Could not find a RadioFisher checkout. Clone "
        "https://github.com/djgormley/RadioFisher and set RADIOFISHER_DIR."
    )


def import_radiofisher(explicit: str | os.PathLike | None = None):
    """Install shims, put the RadioFisher checkout on sys.path and import it."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    install_scipy_shims()
    rf_dir = find_radiofisher_dir(explicit)
    if str(rf_dir) not in sys.path:
        sys.path.insert(0, str(rf_dir))
    import radiofisher  # noqa: PLC0415

    return radiofisher, rf_dir

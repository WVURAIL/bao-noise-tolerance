"""Precompute per-redshift-bin RadioFisher Fisher matrices on a grid of
integration times, so that masking scenarios can be evaluated instantly.

Key fact exploited here: for a fixed redshift bin, the experiment's total
integration time ``ttot`` enters the Fisher integrand only through the thermal
noise power  P_N ~ Tsys^2 / ttot .  A masking scenario rescales the bin's
effective time (ttot -> ttot * w_bar) and its usable volume (F -> v_frac * F),
so a one-off bank of F(z_bin, ttot) covers every scenario and every requested
observing time.

In the deep noise-dominated limit every Fisher element scales as ttot^2 (each
dP/dtheta term carries one power of P_S/P_N); the bank interpolates elements
with monotone splines in log(ttot) and uses the t^2 law below the grid.
"""
from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import pkcache, survey

BANK_VERSION = 1

# module-level context for fork-based workers
_CTX: dict = {}


def _init_context(rf_dir, cachefile, config, epsilon_fg, k_nl0,
                  cosmology=None, expt_overrides=None):
    """Build cosmology/experiment context once (in the parent, pre-fork).

    config='bull2015'  : Bull et al. (2015) CHIME spec, Planck-2013 cosmo
    config='chime2022' : Amiri et al. (2022) Appendix A spec (as-built
                         geometry, Tsys_tot=55 K, Planck-2018 + mnu=0.06,
                         BAO-shift-only, epsilon_fg=0)
    """
    from .compat import import_radiofisher

    rf, rf_dir = import_radiofisher(rf_dir)
    if config == "chime2022":
        from . import cosmologies
        cosmo = pkcache.load_fiducial_cosmology(
            rf, cachefile, cosmo=cosmologies.get(cosmology, rf, rf_dir))
        zs, zc = survey.chime2022_zbins()
        make_expt = lambda t: survey.chime2022_experiment(rf, rf_dir,
                                                          ttot_hours=t)
    else:
        cosmo = pkcache.load_fiducial_cosmology(rf, cachefile)
        expt0 = survey.chime_experiment(rf, rf_dir, ttot_hours=1e4,
                                        epsilon_fg=epsilon_fg, k_nl0=k_nl0)
        zs, zc = survey.chime_zbins(rf, expt0)
        make_expt = lambda t: survey.chime_experiment(
            rf, rf_dir, ttot_hours=t, epsilon_fg=epsilon_fg, k_nl0=k_nl0)
    if expt_overrides:
        base_make = make_expt

        def make_expt(t, _base=base_make, _ov=dict(expt_overrides)):
            e = _base(t)
            e.update(_ov)
            return e

    cosmo_fns = rf.background_evolution_splines(cosmo)
    _CTX.update(rf=rf, rf_dir=rf_dir, cosmo=cosmo, cosmo_fns=cosmo_fns,
                zs=zs, zc=zc, config=config, make_expt=make_expt,
                epsilon_fg=epsilon_fg, k_nl0=k_nl0,
                expt_overrides=dict(expt_overrides or {}))
    return rf, zs, zc


# The settings a reader has to check to know whether a bank is a noise-only
# forecast. Recorded from the experiment dict that was actually built rather than from
# build_bank's arguments: the chime2022 path takes epsilon_fg and k_nl0 from
# the experiment definition and ignores the arguments entirely, so banks built
# that way used to advertise epsilon_fg = 1e-6 while in fact carrying 0. The
# claim "system temperature and nothing else" has to be auditable from the
# file, which means the file has to report what it did rather than what it was
# asked to do.
FOREGROUND_KEYS = ("epsilon_fg", "k_nl0", "kfg_fac", "wedge", "use")


def _foreground_provenance() -> dict:
    """The foreground/systematics settings in force, read back from the expt."""
    try:
        expt = _CTX["make_expt"](1.0)
    except Exception:                            # pragma: no cover
        return {"foreground_settings": "unavailable"}
    out = {}
    for k in FOREGROUND_KEYS:
        v = expt.get(k, None)
        out[k] = v if (v is None or isinstance(v, (int, float, str, dict, list))) \
            else repr(v)
    return {"foreground_settings": out}


def _one_fisher(task):
    """Worker: Fisher matrix for one (zbin, ttot) grid point."""
    ibin, t_hours = task
    rf = _CTX["rf"]
    zs = _CTX["zs"]
    expt = _CTX["make_expt"](t_hours)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        F, paramnames = rf.fisher(zs[ibin], zs[ibin + 1], _CTX["cosmo"], expt,
                                  _CTX["cosmo_fns"], return_pk=False)
    return ibin, t_hours, np.asarray(F), list(paramnames)


# Overrides that must add a parameter row, and the row each one adds. An
# override RadioFisher does not implement is silently ignored by
# ``expt.update()``; the build then runs to completion and writes a file
# whose metadata advertises an override its Fisher matrices do not carry.
# One such bank cost a 7-minute build and looked correct by filename and by
# metadata; only the parameter count gave it away.
ROW_ADDING_OVERRIDES = {"P_res": "_Pres"}


def _preflight(expt_overrides, t_hours):
    """Fail before the grid runs if an override did not take effect.

    Evaluates one Fisher matrix and checks that every override expected to add
    a parameter row actually added it. Costs one grid point; saves the whole
    build and, worse, saves a mislabeled artifact.
    """
    wanted = {k: v for k, v in (expt_overrides or {}).items()
              if k in ROW_ADDING_OVERRIDES}
    if not wanted:
        return
    _, _, _, paramnames = _one_fisher((0, float(t_hours)))
    missing = {k: ROW_ADDING_OVERRIDES[k] for k in wanted
               if ROW_ADDING_OVERRIDES[k] not in paramnames}
    if missing:
        rf_dir = _CTX.get("rf_dir")
        raise RuntimeError(
            "expt override(s) had no effect: "
            + ", ".join(f"{k!r} did not add the {v!r} row"
                        for k, v in missing.items())
            + f".\nThe RadioFisher checkout at {rf_dir} does not implement "
            f"them. The P_res hook lives on the 'rfi-noise-model-chime' "
            f"branch; check that branch out, or point RADIOFISHER_DIR at a "
            f"checkout that has it.\nParameters this checkout returns: "
            f"{paramnames}")


def build_bank(outfile: str | Path, rf_dir=None,
               cachefile: str | Path | None = None,
               t_grid_hours: np.ndarray | None = None,
               nproc: int | None = None, epsilon_fg: float = 1e-6,
               k_nl0: float = 0.14, config: str = "bull2015",
               cosmology: str | None = None, expt_overrides: dict | None = None,
               verbose: bool = True) -> Path:
    """Compute the (Nbins x Nt) Fisher bank and save it as .npz.

    ``expt_overrides`` are merged into the experiment dict at every grid point.
    Use it for the foreground treatment the deployed pipeline actually applies:
    ``{'kfg_fac': tau_cut_s * survey_dnutot_Hz}`` for a delay filter (200 ns on
    CHIME gives 80), or ``{'wedge': 'horizon'}`` for the interferometer wedge.
    Without either, the bank prices BAO using radial modes the pipeline
    filters away."""
    outfile = Path(outfile)
    here = Path(__file__).resolve().parents[2]
    ctag = f"_{cosmology}" if cosmology not in (None, "planck2018") else ""
    default_cache = (f"cache_pk_chime2022{ctag}.dat" if config == "chime2022"
                     else "cache_pk.dat")
    cachefile = Path(cachefile) if cachefile else here / "data" / default_cache
    if t_grid_hours is None:
        t_grid_hours = np.logspace(0.0, 6.0, 19)   # 1 hr .. 1e6 hr
    t_grid_hours = np.asarray(sorted(t_grid_hours))

    rf, zs, zc = _init_context(rf_dir, cachefile, config, epsilon_fg, k_nl0,
                               cosmology=cosmology,
                               expt_overrides=expt_overrides)
    _preflight(expt_overrides, t_grid_hours[0])
    nbins, nt = len(zc), len(t_grid_hours)
    tasks = [(i, t) for i in range(nbins) for t in t_grid_hours]

    nproc = nproc or max(1, (os.cpu_count() or 2))
    t0 = time.time()
    results = []
    if nproc > 1:
        with mp.get_context("fork").Pool(nproc) as pool:
            for k, res in enumerate(pool.imap_unordered(_one_fisher, tasks)):
                results.append(res)
                if verbose and (k + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"[bank] {k + 1}/{len(tasks)} done "
                          f"({el / 60:.1f} min elapsed)", flush=True)
    else:
        for k, task in enumerate(tasks):
            results.append(_one_fisher(task))
            if verbose and (k + 1) % 10 == 0:
                print(f"[bank] {k + 1}/{len(tasks)} done", flush=True)

    paramnames = results[0][3]
    npar = len(paramnames)
    F = np.zeros((nbins, nt, npar, npar))
    t_index = {t: j for j, t in enumerate(t_grid_hours)}
    for ibin, t_hours, Fmat, names in results:
        assert names == paramnames
        F[ibin, t_index[t_hours]] = Fmat

    meta = dict(version=BANK_VERSION,
                expt_overrides=dict(expt_overrides or {}),
                config=config, cosmology=cosmology or "planck2018",
                experiment="CHIME (RadioFisher 'yCHIME', mode icyl)",
                built_minutes=round((time.time() - t0) / 60.0, 2),
                **_foreground_provenance())
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outfile, F=F, t_grid=t_grid_hours, zs=zs, zc=zc,
                        paramnames=np.array(paramnames),
                        meta=json.dumps(meta))
    if verbose:
        print(f"[bank] saved {outfile} ({meta['built_minutes']} min)")
    return outfile


class FisherBank:
    """Loads a bank file and interpolates F(z_bin, ttot) in log time."""

    def __init__(self, path: str | Path):
        # ``importlib.resources`` can return an archive-backed Traversable,
        # not an OS path. Read through its public ``open`` interface and copy
        # every array while the NPZ is open; Forecast retains no file handle.
        opener = getattr(path, "open", None)
        stream = (opener("rb") if opener is not None
                  else contextlib.nullcontext(path))
        with stream as source, np.load(source, allow_pickle=False) as dat:
            self.F_grid = np.array(dat["F"], copy=True)
            self.t_grid = np.array(dat["t_grid"], copy=True)
            self.zs = np.array(dat["zs"], copy=True)
            self.zc = np.array(dat["zc"], copy=True)
            self.paramnames = [str(p) for p in dat["paramnames"]]
            self.meta = json.loads(str(dat["meta"]))
        self._logt = np.log10(self.t_grid)
        nbins, nt, npar, _ = self.F_grid.shape
        # Interpolate the shape function G(t) = F(t)/t^2: exactly constant in
        # the noise-dominated regime, slowly varying through CV saturation.
        G = self.F_grid / (self.t_grid[None, :, None, None] ** 2)
        self._interps = [
            PchipInterpolator(self._logt, G[i].reshape(nt, npar * npar),
                              axis=0, extrapolate=False)
            for i in range(nbins)
        ]
        self.npar = npar

    @property
    def nbins(self) -> int:
        return self.F_grid.shape[0]

    def F(self, ibin: int, t_hours: float) -> np.ndarray:
        """Fisher matrix for a bin at arbitrary effective integration time."""
        tmin, tmax = self.t_grid[0], self.t_grid[-1]
        if t_hours <= 0:
            return np.zeros((self.npar, self.npar))
        if t_hours < tmin:
            # deep noise-dominated limit: F scales as t^2
            return self.F_grid[ibin, 0] * (t_hours / tmin) ** 2
        if t_hours > tmax:
            t_hours = tmax
        vec = self._interps[ibin](np.log10(t_hours))
        return vec.reshape(self.npar, self.npar) * t_hours**2

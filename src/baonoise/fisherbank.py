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
import datetime as dt
import fnmatch
import hashlib
import io
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import pkcache, survey

BANK_SCHEMA_VERSION = 2
ARTIFACT_FORECAST = "forecast"
ARTIFACT_BIAS_RESPONSE = "bias_response"
ARTIFACT_KINDS = frozenset({ARTIFACT_FORECAST, ARTIFACT_BIAS_RESPONSE})
BIAS_PARAMETER = "_Pres"
SUPPORTED_CONFIGS = frozenset({"bull2015", "chime2022"})
SUPPORTED_COSMOLOGIES = {
    "bull2015": frozenset({None, "planck2013"}),
    "chime2022": frozenset({None, "planck2018", "pact2025"}),
}
PROFILE_MODEL_CONTRACT = {
    "bull2015": {
        "Tb_model": "powerlaw",
        "bias_HI_model": "powerlaw",
        "omega_HI_model": "powerlaw",
    },
    "chime_overview_2022": {
        "Tb_model": "hall",
        "bias_HI_model": "castorina",
        "omega_HI_model": "crighton",
    },
}
BANK_BUILD_CAPABILITIES = frozenset(
    {"explicit_physical_densities", "astrophysical_model_profiles"})
COSMOLOGY_PARAMETER_KEYS = frozenset(
    {"h", "ns", "sigma_8", "N_eff", "mnu", "ombh2", "omch2", "omnuh2"})
BAONOISE_SOURCE_MANIFEST = {
    "include": ("pyproject.toml", "src/baonoise/*.py"),
    "exclude": ("src/baonoise/data/fisher_bank*.npz",),
}
RADIOFISHER_SOURCE_MANIFEST = {
    "include": ("pyproject.toml", "radiofisher/*.py",
                "chime2021/experiments_CHIME.py"),
    "exclude": (),
}

_V2_PROVENANCE_KEYS = frozenset(
    {"baonoise", "radiofisher", "cosmology", "pk_cache", "experiment"})
_V2_PROVENANCE_FIELDS = {
    "baonoise": frozenset(
        {"version", "git_commit", "git_dirty", "working_tree_sha256",
         "source_manifest"}),
    "radiofisher": frozenset(
        {"backend_id", "backend_version", "api_version", "capabilities",
         "git_commit", "git_dirty", "working_tree_sha256",
         "source_manifest"}),
    "cosmology": frozenset(
        {"name", "sha256", "parameters", "astrophysical_model_profile",
         "astrophysical_models"}),
    "pk_cache": frozenset({"filename", "sha256", "cache_id"}),
    "experiment": frozenset(
        {"sha256", "settings", "baseline_sha256"}),
}


# module-level context for fork-based workers
_CTX: dict = {}


def _validate_configuration(config: str, cosmology: str | None) -> None:
    """Reject mislabeled config/cosmology pairs before backend work starts."""
    if config not in SUPPORTED_CONFIGS:
        raise ValueError(
            f"unknown config {config!r}; choose from {sorted(SUPPORTED_CONFIGS)}")
    if cosmology not in SUPPORTED_COSMOLOGIES[config]:
        allowed = sorted(
            name for name in SUPPORTED_COSMOLOGIES[config] if name is not None)
        raise ValueError(
            f"config={config!r} does not support cosmology={cosmology!r}; "
            f"choose from {allowed}")


def _init_context(rf_dir, cachefile, config, epsilon_fg, k_nl0,
                  cosmology=None, expt_overrides=None):
    """Build cosmology/experiment context once (in the parent, pre-fork).

    config='bull2015'  : Bull et al. (2015) CHIME spec, Planck-2013 cosmo
    config='chime2022' : Amiri et al. (2022) Appendix A spec (as-built
                         geometry, Tsys_tot=55 K, Planck-2018 + mnu=0.06,
                         BAO-shift-only, epsilon_fg=0)
    """
    from . import cosmologies
    from .compat import import_radiofisher, require_backend_capabilities

    _validate_configuration(config, cosmology)

    rf, rf_dir = import_radiofisher(rf_dir)
    required = set(BANK_BUILD_CAPABILITIES)
    if "P_res" in (expt_overrides or {}):
        required.add("P_res")
    require_backend_capabilities(rf, required, rf_dir=rf_dir)
    if config == "chime2022":
        cosmo = pkcache.load_fiducial_cosmology(
            rf, cachefile, cosmo=cosmologies.get(cosmology, rf, rf_dir))
        zs, zc = survey.chime2022_zbins()
        make_expt = lambda t: survey.chime2022_experiment(rf, rf_dir,
                                                          ttot_hours=t)
    else:
        base_cosmo = cosmologies.with_astrophysical_profile(
            rf.experiments.cosmo, "bull2015", rf=rf)
        cosmo = pkcache.load_fiducial_cosmology(
            rf, cachefile, cosmo=base_cosmo)
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
                cachefile=Path(cachefile), cosmology_name=cosmology,
                astrophysical_model_profile=
                    cosmo[cosmologies.ASTROPHYSICAL_PROFILE_KEY],
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
SOURCE_TEXT_SUFFIXES = frozenset({".py", ".toml"})


def _git_state(path: Path, *, include: tuple[str, ...] = (),
               exclude: tuple[str, ...] = ()) -> dict:
    """Best-effort immutable revision state for an input checkout.

    A commit plus a dirty flag cannot reproduce a bank built before the
    changes are committed. Hash the caller-selected tracked and non-ignored
    source files as well. This content identity survives committing the same
    tree, unlike a hash of its patch against ``HEAD``. Generated data and
    golden outputs are excluded so a rebuild does not hash its own result.
    """
    def run_bytes(*args):
        result = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True,
            check=False)
        return result.stdout if result.returncode == 0 else None

    commit_bytes = run_bytes("rev-parse", "HEAD")
    status = run_bytes("status", "--porcelain=v1", "-z")
    files = run_bytes(
        "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    commit = (commit_bytes.decode("ascii").strip()
              if commit_bytes is not None else None)
    digest = None
    if files is not None:
        hasher = hashlib.sha256()
        hasher.update(b"scientific-source-manifest\0")
        for relative_bytes in sorted(filter(None, files.split(b"\0"))):
            relative = os.fsdecode(relative_bytes)
            if include and not any(
                    fnmatch.fnmatchcase(relative, pattern)
                    for pattern in include):
                continue
            if any(fnmatch.fnmatchcase(relative, pattern)
                   for pattern in exclude):
                continue
            source = path / relative
            try:
                if not source.exists() and not source.is_symlink():
                    # An unstaged deletion is the same scientific source tree
                    # as that deletion after it is staged and committed.
                    continue
                if source.is_symlink():
                    content = os.fsencode(os.readlink(source))
                    kind = b"symlink"
                elif source.is_file():
                    content = source.read_bytes()
                    kind = b"file"
                else:
                    content = b""
                    kind = b"missing"
            except OSError:
                digest = None
                break
            if kind == b"file" and source.suffix in SOURCE_TEXT_SUFFIXES:
                # Git's checkout newline policy is not a scientific input.
                # Hash a canonical LF representation so Windows and POSIX
                # checkouts of identical source produce the same identity.
                content = content.replace(b"\r\n", b"\n").replace(
                    b"\r", b"\n")
            hasher.update(kind)
            hasher.update(b"\0")
            hasher.update(relative_bytes)
            hasher.update(b"\0")
            hasher.update(content)
        else:
            digest = hasher.hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "working_tree_sha256": digest,
    }


def _jsonable(value):
    """Stable JSON projection of an experiment dictionary."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if callable(value):
        return "<callable>"
    return repr(value)


def experiment_settings_payload(expt: dict, rf_dir: str | Path) -> dict:
    """Return the canonical experiment projection stored in bank provenance.

    Absolute baseline paths are checkout details, not scientific identities.
    Paths inside RadioFisher are recorded relative to that checkout; external
    packaged baselines are recorded by filename and separately content-hashed.
    """
    payload = _jsonable(expt)
    baseline = expt.get("n(x)")
    if baseline is not None:
        baseline_path = Path(baseline)
        try:
            baseline_label = baseline_path.resolve().relative_to(
                Path(rf_dir).resolve()).as_posix()
        except ValueError:
            baseline_label = baseline_path.name
        payload["n(x)"] = baseline_label
    return payload


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_provenance() -> dict:
    """Scientific and software identity of the context that built a bank."""
    from . import __version__
    from .compat import backend_capabilities

    rf = _CTX["rf"]
    rf_dir = Path(_CTX["rf_dir"])
    package_root = Path(__file__).resolve().parents[2]
    cosmo = _CTX["cosmo"]
    cachefile = Path(_CTX["cachefile"])
    expt = _CTX["make_expt"](1.0)
    baseline = expt.get("n(x)")
    baseline_path = Path(baseline) if baseline else None
    expt_payload = experiment_settings_payload(expt, rf_dir)
    cache_meta = pkcache.inspect_pk_cache(cachefile)
    return {
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baonoise": {
            "version": __version__,
            "source_manifest": {
                key: list(value) for key, value in
                BAONOISE_SOURCE_MANIFEST.items()},
            **_git_state(
                package_root,
                **BAONOISE_SOURCE_MANIFEST),
        },
        "radiofisher": {
            "backend_id": getattr(rf, "BACKEND_ID", None),
            "backend_version": getattr(rf, "BACKEND_VERSION", None),
            "api_version": getattr(rf, "BACKEND_API_VERSION", None),
            "capabilities": sorted(backend_capabilities(rf)),
            "source_manifest": {
                key: list(value) for key, value in
                RADIOFISHER_SOURCE_MANIFEST.items()},
            **_git_state(
                rf_dir,
                **RADIOFISHER_SOURCE_MANIFEST),
        },
        "cosmology": {
            "name": (_CTX.get("cosmology_name")
                     or ("planck2018" if _CTX["config"] == "chime2022"
                         else "planck2013")),
            "sha256": pkcache.cosmology_fingerprint(cosmo),
            "parameters": pkcache.cosmology_payload(cosmo),
            "astrophysical_model_profile":
                _CTX["astrophysical_model_profile"],
            "astrophysical_models": {
                key: cosmo[key] for key in
                ("Tb_model", "bias_HI_model", "omega_HI_model")},
        },
        "pk_cache": {
            "filename": cachefile.name,
            "sha256": pkcache.file_sha256(cachefile),
            "cache_id": _sha256_json(cache_meta),
        },
        "experiment": {
            "sha256": _sha256_json(expt_payload),
            "settings": expt_payload,
            "baseline_sha256": (
                pkcache.file_sha256(baseline_path)
                if baseline_path is not None and baseline_path.is_file()
                else None),
        },
    }


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
            f"them. Point RADIOFISHER_DIR at a backend that declares and "
            f"implements the required P_res capability.\nParameters this "
            f"checkout returns: "
            f"{paramnames}")


def build_bank(outfile: str | Path, rf_dir=None,
               cachefile: str | Path | None = None,
               t_grid_hours: np.ndarray | None = None,
               nproc: int | None = None, epsilon_fg: float = 1e-6,
               k_nl0: float = 0.14, config: str = "chime2022",
               cosmology: str | None = None, expt_overrides: dict | None = None,
               verbose: bool = True) -> Path:
    """Compute the (Nbins x Nt) Fisher bank and save it as .npz.

    ``expt_overrides`` are merged into the experiment dict at every grid point.
    Use it for the foreground treatment the deployed pipeline actually applies:
    ``{'kfg_fac': tau_cut_s * survey_dnutot_Hz}`` for a delay filter (200 ns on
    CHIME gives 80), or ``{'wedge': 'horizon'}`` for the interferometer wedge.
    Without either, the bank prices BAO using radial modes the pipeline
    filters away."""
    _validate_configuration(config, cosmology)
    outfile = Path(outfile)
    ctag = f"_{cosmology}" if cosmology not in (None, "planck2018") else ""
    default_cache = (f"cache_pk_chime2022{ctag}.dat" if config == "chime2022"
                     else "cache_pk.dat")
    if cachefile is None:
        from .resources import filesystem_data_file
        cachefile = filesystem_data_file(default_cache)
    else:
        cachefile = Path(cachefile)
    if t_grid_hours is None:
        t_grid_hours = np.logspace(0.0, 6.0, 19)   # 1 hr .. 1e6 hr
    t_grid_hours = np.asarray(sorted(t_grid_hours), dtype=float)
    if (t_grid_hours.ndim != 1 or t_grid_hours.size < 2
            or not np.all(np.isfinite(t_grid_hours))
            or np.any(t_grid_hours <= 0.0)
            or np.any(np.diff(t_grid_hours) <= 0.0)):
        raise ValueError(
            "t_grid_hours must contain at least two distinct, increasing, "
            "positive finite values")

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
        if names != paramnames:
            raise RuntimeError(
                f"RadioFisher parameter schema changed during the build: "
                f"expected {paramnames}, received {names}")
        F[ibin, t_index[t_hours]] = Fmat

    artifact_kind = (ARTIFACT_BIAS_RESPONSE if BIAS_PARAMETER in paramnames
                     else ARTIFACT_FORECAST)
    meta = dict(schema_version=BANK_SCHEMA_VERSION,
                artifact_kind=artifact_kind,
                expt_overrides=dict(expt_overrides or {}),
                config=config,
                cosmology=(cosmology or
                           ("planck2018" if config == "chime2022"
                            else "planck2013")),
                experiment="CHIME (RadioFisher 'yCHIME', mode icyl)",
                astrophysical_model_profile=
                    _CTX["astrophysical_model_profile"],
                built_minutes=round((time.time() - t0) / 60.0, 2),
                provenance=_build_provenance(),
                **_foreground_provenance())
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outfile, F=F, t_grid=t_grid_hours, zs=zs, zc=zc,
                        paramnames=np.array(paramnames),
                        meta=json.dumps(meta, sort_keys=True))
    if verbose:
        print(f"[bank] saved {outfile} ({meta['built_minutes']} min)")
    return outfile


def _validate_bank_arrays(F_grid, t_grid, zs, zc, paramnames) -> None:
    """Reject malformed banks before interpolation or marginalisation."""
    if F_grid.ndim != 4:
        raise ValueError(f"F must be four-dimensional, got shape {F_grid.shape}")
    nbins, nt, nrow, ncol = F_grid.shape
    if nrow != ncol:
        raise ValueError(f"F matrices must be square, got shape {F_grid.shape}")
    if not np.all(np.isfinite(F_grid)):
        raise ValueError("F contains non-finite values")
    if not np.allclose(F_grid, np.swapaxes(F_grid, -1, -2),
                       rtol=1e-10, atol=1e-12):
        raise ValueError("F matrices must be symmetric")
    if (t_grid.ndim != 1 or len(t_grid) != nt or nt < 2
            or not np.all(np.isfinite(t_grid)) or np.any(t_grid <= 0.0)
            or np.any(np.diff(t_grid) <= 0.0)):
        raise ValueError(
            "t_grid must be one-dimensional, positive, finite, and strictly "
            "increasing, with one entry per Fisher-grid time")
    if (zs.ndim != 1 or len(zs) != nbins + 1
            or not np.all(np.isfinite(zs)) or np.any(np.diff(zs) <= 0.0)):
        raise ValueError(
            "zs must contain one more strictly increasing finite edge than "
            "there are Fisher redshift bins")
    if (zc.ndim != 1 or len(zc) != nbins or not np.all(np.isfinite(zc))
            or np.any(zc <= zs[:-1]) or np.any(zc >= zs[1:])):
        raise ValueError("zc must contain one finite center inside each z bin")
    if len(paramnames) != nrow:
        raise ValueError(
            f"paramnames has {len(paramnames)} entries for {nrow} Fisher rows")
    if any(not name or name.strip() != name for name in paramnames):
        raise ValueError("paramnames must be non-empty, stripped strings")
    if len(set(paramnames)) != len(paramnames):
        raise ValueError("paramnames must be unique")


def _normalise_metadata(meta: dict, paramnames: list[str]) -> dict:
    """Validate strict schema-v2 metadata."""
    if not isinstance(meta, dict):
        raise ValueError("bank metadata must decode to a JSON object")
    if "schema_version" not in meta:
        if "version" in meta:
            raise ValueError(
                "bank metadata uses the retired 'version' alias; rebuild it "
                "with the schema-v2 writer")
        raise ValueError("bank metadata is missing required schema_version=2")
    raw_version = meta.get("schema_version")
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError("bank metadata must contain an integer schema version")
    if raw_version != BANK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Fisher bank schema version {raw_version}; "
            f"the only supported version is {BANK_SCHEMA_VERSION}")

    out = dict(meta)
    has_bias_row = BIAS_PARAMETER in paramnames
    inferred_kind = (ARTIFACT_BIAS_RESPONSE if has_bias_row
                     else ARTIFACT_FORECAST)
    if out.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError("bank metadata must set schema_version=2")
    if "version" in out:
        raise ValueError(
            "bank metadata uses the retired 'version' alias; rebuild it with "
            "the schema-v2 writer")
    kind = out.get("artifact_kind")
    if kind not in ARTIFACT_KINDS:
        raise ValueError(
            f"v2 bank artifact_kind must be one of {sorted(ARTIFACT_KINDS)}")
    if kind != inferred_kind:
        raise ValueError(
            f"artifact_kind={kind!r} is inconsistent with parameter row "
            f"{BIAS_PARAMETER!r} {'being present' if has_bias_row else 'being absent'}")
    if not isinstance(out.get("config"), str) or not out["config"]:
        raise ValueError("v2 bank metadata requires a non-empty config")
    if out["config"] not in SUPPORTED_CONFIGS:
        raise ValueError(
            f"unknown config {out['config']!r}; choose from "
            f"{sorted(SUPPORTED_CONFIGS)}")
    if not isinstance(out.get("cosmology"), str) or not out["cosmology"]:
        raise ValueError("v2 bank metadata requires a named cosmology")
    _validate_configuration(out["config"], out["cosmology"])
    expected_profile = ("chime_overview_2022"
                        if out["config"] == "chime2022" else "bull2015")
    if out.get("astrophysical_model_profile") != expected_profile:
        raise ValueError(
            f"config={out['config']!r} requires astrophysical_model_profile="
            f"{expected_profile!r}")
    provenance = out.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("v2 bank metadata requires provenance")
    missing = sorted(_V2_PROVENANCE_KEYS - set(provenance))
    if missing:
        raise ValueError("v2 bank provenance is missing: " + ", ".join(missing))
    built_utc = provenance.get("built_utc")
    if not isinstance(built_utc, str) or not built_utc:
        raise ValueError("v2 bank provenance requires built_utc")
    try:
        built_timestamp = dt.datetime.fromisoformat(built_utc)
    except ValueError as exc:
        raise ValueError("v2 bank provenance built_utc must be ISO-8601") from exc
    if built_timestamp.tzinfo is None:
        raise ValueError("v2 bank provenance built_utc must include a timezone")
    for section, required_fields in _V2_PROVENANCE_FIELDS.items():
        value = provenance.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"v2 bank provenance {section} must be an object")
        section_missing = sorted(required_fields - set(value))
        if section_missing:
            raise ValueError(
                f"v2 bank provenance {section} is missing: "
                + ", ".join(section_missing))
    for section in ("baonoise", "radiofisher"):
        if provenance[section]["git_dirty"] not in {True, False, None}:
            raise ValueError(
                f"v2 bank provenance {section}.git_dirty must be boolean or null")
        commit = provenance[section]["git_commit"]
        if commit is not None and not (
                isinstance(commit, str) and len(commit) in {40, 64}
                and set(commit.lower()) <= set("0123456789abcdef")):
            raise ValueError(
                f"v2 bank provenance {section}.git_commit must be null or a "
                "Git object ID")
    if (not isinstance(provenance["baonoise"]["version"], str)
            or not provenance["baonoise"]["version"]):
        raise ValueError("v2 bank provenance requires a Bao package version")
    if (not isinstance(provenance["radiofisher"]["backend_version"], str)
            or not provenance["radiofisher"]["backend_version"]):
        raise ValueError(
            "v2 bank provenance requires a RadioFisher backend version")
    hexdigits = set("0123456789abcdef")
    for section, field in (("cosmology", "sha256"), ("pk_cache", "sha256"),
                           ("pk_cache", "cache_id"),
                           ("experiment", "sha256")):
        digest = provenance[section][field]
        if not (isinstance(digest, str) and len(digest) == 64
                and set(digest.lower()) <= hexdigits):
            raise ValueError(
                f"v2 bank provenance {section}.{field} must be a SHA-256")
    for section in ("baonoise", "radiofisher"):
        digest = provenance[section]["working_tree_sha256"]
        if digest is not None and not (
                isinstance(digest, str) and len(digest) == 64
                and set(digest.lower()) <= hexdigits):
            raise ValueError(
                f"v2 bank provenance {section}.working_tree_sha256 must be "
                "null or SHA-256")
        manifest = provenance[section]["source_manifest"]
        if (not isinstance(manifest, dict)
                or set(manifest) != {"include", "exclude"}
                or not isinstance(manifest["include"], list)
                or not manifest["include"]
                or not isinstance(manifest["exclude"], list)
                or not all(isinstance(pattern, str) and pattern
                           for patterns in manifest.values()
                           for pattern in patterns)
                or any(len(patterns) != len(set(patterns))
                       for patterns in manifest.values())):
            raise ValueError(
                f"v2 bank provenance {section}.source_manifest must contain "
                "unique, non-empty include/exclude pattern lists")
    baseline_digest = provenance["experiment"]["baseline_sha256"]
    if baseline_digest is not None and not (
            isinstance(baseline_digest, str) and len(baseline_digest) == 64
            and set(baseline_digest.lower()) <= hexdigits):
        raise ValueError(
            "v2 bank provenance experiment.baseline_sha256 must be null or SHA-256")
    cache_filename = provenance["pk_cache"]["filename"]
    if (not isinstance(cache_filename, str) or not cache_filename
            or Path(cache_filename).name != cache_filename):
        raise ValueError("v2 bank pk_cache.filename must be a plain filename")
    if provenance["radiofisher"]["backend_id"] != "radiofisher":
        raise ValueError("v2 bank provenance has the wrong RadioFisher backend_id")
    from .compat import SUPPORTED_BACKEND_API_VERSION
    api_version = provenance["radiofisher"]["api_version"]
    if (isinstance(api_version, bool) or not isinstance(api_version, int)
            or api_version != SUPPORTED_BACKEND_API_VERSION):
        raise ValueError(
            "v2 bank provenance has an unsupported RadioFisher api_version")
    capabilities = provenance["radiofisher"]["capabilities"]
    if (not isinstance(capabilities, list)
            or not all(isinstance(item, str) for item in capabilities)
            or capabilities != sorted(set(capabilities))):
        raise ValueError(
            "v2 bank RadioFisher capabilities must be sorted unique strings")
    required_capabilities = set(BANK_BUILD_CAPABILITIES)
    if kind == ARTIFACT_BIAS_RESPONSE:
        required_capabilities.add("P_res")
    missing_capabilities = sorted(required_capabilities - set(capabilities))
    if missing_capabilities:
        raise ValueError(
            "v2 bank RadioFisher provenance is missing required capability(s): "
            + ", ".join(missing_capabilities))
    if provenance["cosmology"]["name"] != out["cosmology"]:
        raise ValueError("v2 bank cosmology name disagrees with its provenance")
    if (provenance["cosmology"]["astrophysical_model_profile"]
            != out["astrophysical_model_profile"]):
        raise ValueError(
            "v2 bank astrophysical profile disagrees with its provenance")
    models = provenance["cosmology"]["astrophysical_models"]
    if models != PROFILE_MODEL_CONTRACT[expected_profile]:
        raise ValueError(
            f"v2 bank {expected_profile!r} astrophysical model provenance "
            "does not match the canonical profile")
    parameters = provenance["cosmology"]["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("v2 bank cosmology parameters must be an object")
    missing_parameters = sorted(COSMOLOGY_PARAMETER_KEYS - set(parameters))
    if missing_parameters:
        raise ValueError(
            "v2 bank cosmology parameters are missing: "
            + ", ".join(missing_parameters))
    if any(isinstance(parameters[key], bool)
           or not isinstance(parameters[key], (int, float))
           or not np.isfinite(parameters[key])
           for key in COSMOLOGY_PARAMETER_KEYS):
        raise ValueError("v2 bank cosmology parameters must be finite numbers")
    if not isinstance(provenance["experiment"]["settings"], dict):
        raise ValueError("v2 bank experiment settings must be an object")
    return out


class FisherBank:
    """Loads a bank file and interpolates F(z_bin, ttot) in log time."""

    def __init__(self, path: str | Path, *, expected_artifact_kind: str | None = None):
        # ``importlib.resources`` can return an archive-backed Traversable,
        # not an OS path. Read through its public ``open`` interface and copy
        # every array while the NPZ is open; Forecast retains no file handle.
        opener = getattr(path, "open", None)
        stream = (opener("rb") if opener is not None
                  else contextlib.nullcontext(path))
        try:
            with stream as source, np.load(source, allow_pickle=False) as dat:
                required = {"F", "t_grid", "zs", "zc", "paramnames", "meta"}
                missing = sorted(required - set(dat.files))
                if missing:
                    raise ValueError("bank archive is missing: " + ", ".join(missing))
                self.F_grid = np.array(dat["F"], dtype=float, copy=True)
                self.t_grid = np.array(dat["t_grid"], dtype=float, copy=True)
                self.zs = np.array(dat["zs"], dtype=float, copy=True)
                self.zc = np.array(dat["zc"], dtype=float, copy=True)
                raw_names = np.array(dat["paramnames"], copy=True)
                if raw_names.ndim != 1 or raw_names.dtype.kind not in "US":
                    raise ValueError("paramnames must be a one-dimensional string array")
                self.paramnames = [str(p) for p in raw_names]
                raw_meta = np.array(dat["meta"], copy=True)
                if raw_meta.shape != () or raw_meta.dtype.kind not in "US":
                    raise ValueError("meta must be a scalar JSON string")
                try:
                    decoded_meta = json.loads(str(raw_meta))
                except json.JSONDecodeError as exc:
                    raise ValueError("bank metadata is not valid JSON") from exc
        except ValueError as exc:
            if "Object arrays cannot be loaded" in str(exc):
                raise ValueError("bank contains unsafe pickle-backed object arrays") from exc
            raise
        _validate_bank_arrays(self.F_grid, self.t_grid, self.zs, self.zc,
                              self.paramnames)
        self.meta = _normalise_metadata(decoded_meta, self.paramnames)
        self.schema_version = BANK_SCHEMA_VERSION
        self.artifact_kind = self.meta["artifact_kind"]
        if expected_artifact_kind is not None:
            self.require_artifact_kind(expected_artifact_kind)
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

    def require_artifact_kind(self, expected: str) -> None:
        """Reject accidental use of a response/bias bank as a forecast bank."""
        if expected not in ARTIFACT_KINDS:
            raise ValueError(f"unknown expected artifact kind: {expected!r}")
        if self.artifact_kind != expected:
            raise ValueError(
                f"bank artifact_kind={self.artifact_kind!r}; expected "
                f"{expected!r}. A {BIAS_PARAMETER} response row is a bias "
                "template and must not be marginalized as a forecast parameter.")

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

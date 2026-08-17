"""Build and validate RadioFisher-compatible CAMB power-spectrum caches.

Version-2 caches bind the numerical table to the resolved physical densities,
CAMB settings, generator version, and a SHA-256 digest of the table itself.
The historical ``# pythoncamb#`` sentinel does not identify any of those
inputs and is therefore treated as stale rather than silently force-loaded.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np

from .cosmologies import (NEUTRINO_MASS_DENSITY_EV, physical_densities,
                          with_explicit_physical_densities)

CACHE_FORMAT = "baonoise-pk-v2"
LEGACY_SENTINEL_HASH = "pythoncamb"
DEFAULT_KMAX_H = 20.0
DEFAULT_NPOINTS = 1400
DEFAULT_AS = 2.1e-9
DEFAULT_TAU = 0.06


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of an on-disk scientific input or generated artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosmology_payload(cosmo: dict) -> dict:
    """The scalar cosmology inputs that determine this linear P(k) table."""
    densities = physical_densities(cosmo)
    return {
        "h": float(cosmo["h"]),
        "ns": float(cosmo["ns"]),
        "sigma_8": float(cosmo["sigma_8"]),
        "N_eff": float(cosmo.get("N_eff", 3.046)),
        "mnu": densities["omnuh2"] * NEUTRINO_MASS_DENSITY_EV,
        **densities,
    }


def cosmology_fingerprint(cosmo: dict) -> str:
    """Stable SHA-256 identifier for the resolved scalar cosmology."""
    return _sha256_bytes(_canonical_json(cosmology_payload(cosmo)).encode())


def _request_payload(cosmo: dict, kmax_h: float, npoints: int) -> dict:
    return {
        "format": CACHE_FORMAT,
        "cosmology": cosmology_payload(cosmo),
        "settings": {
            "As": DEFAULT_AS,
            "tau": DEFAULT_TAU,
            "kmax_h_mpc": float(kmax_h),
            "minkh_h_mpc": 1e-4,
            "npoints": int(npoints),
            "redshift": 0.0,
            "nonlinear": False,
            "accurate_massive_neutrino_transfers": False,
            "neutrino_input": "omnuh2_active",
        },
    }


def _set_camb_cosmology(pars, cosmo: dict, densities: dict) -> None:
    """Configure CAMB from the authoritative density triplet."""
    pars.set_cosmology(
        H0=100.0 * cosmo["h"], ombh2=densities["ombh2"],
        omch2=densities["omch2"], omk=0.0, mnu=None,
        omnuh2_active=densities["omnuh2"],
        nnu=cosmo.get("N_eff", 3.046), tau=DEFAULT_TAU)
    if not np.isclose(
            float(pars.omnuh2), densities["omnuh2"], rtol=0.0, atol=1e-15):
        raise RuntimeError(
            "CAMB did not preserve the requested authoritative omnuh2: "
            f"requested {densities['omnuh2']!r}, got {pars.omnuh2!r}")


def inspect_pk_cache(cachefile: str | Path, *, verify_content: bool = True) \
        -> dict:
    """Return validated v2 metadata, or raise for unversioned/corrupt caches."""
    cachefile = Path(cachefile)
    with cachefile.open("rb") as stream:
        first = stream.readline().decode("utf-8").strip()
        second = stream.readline().decode("utf-8").strip()
        body = stream.read()
    if not (first.startswith("# ") and first.endswith("#")):
        raise ValueError(f"{cachefile} has no RadioFisher cache hash header")
    cache_id = first[2:-1]
    prefix = "# baonoise-meta "
    if not second.startswith(prefix):
        if cache_id == LEGACY_SENTINEL_HASH:
            raise ValueError(
                f"{cachefile} is an unversioned CAMB cache")
        raise ValueError(f"{cachefile} has no {CACHE_FORMAT} metadata")
    try:
        meta = json.loads(second[len(prefix):])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{cachefile} has malformed cache metadata") from exc
    if meta.get("format") != CACHE_FORMAT:
        raise ValueError(f"unsupported P(k) cache format: {meta.get('format')!r}")
    recorded_content = meta.get("data_sha256")
    if verify_content and _sha256_bytes(body) != recorded_content:
        raise ValueError(f"{cachefile} data digest does not match its header")
    expected_id = _sha256_bytes(_canonical_json(meta).encode())
    if cache_id != expected_id:
        raise ValueError(f"{cachefile} metadata digest does not match its header")
    return meta


def cache_matches(cachefile: str | Path, cosmo: dict,
                  kmax_h: float = DEFAULT_KMAX_H,
                  npoints: int = DEFAULT_NPOINTS) -> bool:
    """Whether a cache is intact and was built for the requested inputs."""
    try:
        meta = inspect_pk_cache(cachefile)
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return False
    request = _request_payload(cosmo, kmax_h, npoints)
    return all(meta.get(key) == value for key, value in request.items())


def build_pk_cache(cosmo: dict, cachefile: str | Path,
                   kmax_h: float = DEFAULT_KMAX_H,
                   npoints: int = DEFAULT_NPOINTS) -> Path:
    """Write a content-verified RadioFisher P(k) cache for ``cosmo``."""
    try:
        import camb
    except ModuleNotFoundError as exc:      # pragma: no cover - install path
        raise ModuleNotFoundError(
            "camb is required to build or refresh a P(k) cache; install the "
            "optional dependency with: pip install -e '.[pk]'") from exc

    cosmo = with_explicit_physical_densities(cosmo)
    cachefile = Path(cachefile)
    h = cosmo["h"]
    densities = physical_densities(cosmo)

    pars = camb.CAMBparams()
    _set_camb_cosmology(pars, cosmo, densities)
    pars.InitPower.set_params(As=DEFAULT_AS, ns=cosmo["ns"])
    pars.set_matter_power(
        redshifts=[0.0], kmax=kmax_h,
        accurate_massive_neutrino_transfers=False)
    pars.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(pars)
    kh, _z, pk = results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=kmax_h, npoints=npoints)
    sigma8_camb = results.get_sigma8_0()

    renorm = (cosmo["sigma_8"] / sigma8_camb) ** 2
    k_mpc = kh * h
    pk_mpc = pk[0] * renorm / h**3

    body_stream = io.StringIO(newline="\n")
    np.savetxt(body_stream, np.column_stack([k_mpc, pk_mpc]))
    body = body_stream.getvalue().encode("utf-8")
    meta = {
        **_request_payload(cosmo, kmax_h, npoints),
        "generator": {
            "package": "camb",
            "version": getattr(camb, "__version__", None)
                       or importlib.metadata.version("camb"),
        },
        "data_sha256": _sha256_bytes(body),
    }
    cache_id = _sha256_bytes(_canonical_json(meta).encode())
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    with cachefile.open("wb") as stream:
        stream.write(f"# {cache_id}#\n".encode())
        stream.write(f"# baonoise-meta {_canonical_json(meta)}\n".encode())
        stream.write(body)
    return cachefile


def load_fiducial_cosmology(rf, cachefile: str | Path,
                            force_build: bool = False,
                            cosmo: dict | None = None,
                            kmax_h: float = DEFAULT_KMAX_H,
                            npoints: int = DEFAULT_NPOINTS):
    """Return a fiducial cosmology with ``pk_nobao``/``fbao`` attached.

    An existing cache is reused only if its content digest and complete CAMB
    request match. Legacy sentinel caches must be regenerated once; this is
    deliberate because they may contain the former double-counted-neutrino
    spectrum while claiming no particular cosmology.
    """
    import copy

    if cosmo is None:
        cosmo = copy.deepcopy(rf.experiments.cosmo)
        cosmo["mnu"] = 0.0
    cosmo = with_explicit_physical_densities(cosmo)
    cachefile = Path(cachefile)
    if force_build or not cache_matches(cachefile, cosmo, kmax_h, npoints):
        try:
            build_pk_cache(cosmo, cachefile, kmax_h=kmax_h, npoints=npoints)
        except ModuleNotFoundError as exc:
            state = "stale or unversioned" if cachefile.exists() else "missing"
            raise ModuleNotFoundError(
                f"the requested P(k) cache is {state}: {cachefile}. "
                "Install baonoise[pk] to regenerate a content-verified cache") \
                from exc
    return rf.load_power_spectrum(cosmo, str(cachefile), force_load=True)

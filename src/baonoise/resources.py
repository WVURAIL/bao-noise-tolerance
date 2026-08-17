"""Access the immutable data files distributed with :mod:`baonoise`.

The returned objects implement :class:`importlib.resources.abc.Traversable`.
Callers should use ``open()`` rather than converting them to filesystem paths:
that keeps resource access valid for every import loader, including archives.
"""
from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType

DATA_PACKAGE = "baonoise.data"
DEFAULT_BANK_NAME = "fisher_bank_chime2022.npz"
PACT2025_BANK_NAME = "fisher_bank_chime2022_pact2025.npz"
BANK_NAMES = MappingProxyType({
    "planck2018": DEFAULT_BANK_NAME,
    "pact2025": PACT2025_BANK_NAME,
})
DEFAULT_RATES_NAME = "survey_quarterly_rates_all23.csv"
PRODUCTS_MANIFEST_NAME = "products.json"
SYNTHETIC_BASELINE_NAME = "nx_CHIME_800_synth.dat"
CACHE_NAMES = frozenset({
    "cache_pk.dat",
    "cache_pk_chime2022.dat",
    "cache_pk_chime2022_pact2025.dat",
})
RADIOFISHER_FILESYSTEM_NAMES = CACHE_NAMES | {SYNTHETIC_BASELINE_NAME}


def data_file(name: str) -> Traversable:
    """Return a packaged data file, failing clearly if packaging is broken."""
    resource = files(DATA_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(
            f"baonoise package data is missing {name!r}; reinstall baonoise "
            "from a complete wheel or sdist")
    return resource


def bank_file(cosmology: str = "planck2018") -> Traversable:
    """Return a packaged CHIME bank by its named fiducial cosmology."""
    try:
        name = BANK_NAMES[cosmology]
    except KeyError as exc:
        raise ValueError(
            f"unknown packaged bank cosmology {cosmology!r}; choose from "
            f"{sorted(BANK_NAMES)}") from exc
    return data_file(name)


def filesystem_data_file(name: str) -> Path:
    """Return packaged data that an external filesystem-only API can read.

    Installed wheels are unpacked and therefore provide ordinary paths.  A
    zip-imported package can still use :func:`data_file` for read-only package
    resources, but RadioFisher requires a persistent real filename for P(k)
    caches and cannot run directly from an import archive.
    """
    resource = data_file(name)
    try:
        path = Path(resource)
    except TypeError as exc:
        raise RuntimeError(
            f"{name!r} must be materialized on a filesystem before invoking "
            "RadioFisher; install the baonoise wheel instead of importing it "
            "directly from an archive") from exc
    if not path.is_file():  # pragma: no cover - guarded by data_file
        raise FileNotFoundError(f"package data is missing {name!r}: {path}")
    return path


DEFAULT_BANK = data_file(DEFAULT_BANK_NAME)
PACT2025_BANK = data_file(PACT2025_BANK_NAME)
DEFAULT_RATES_CSV = data_file(DEFAULT_RATES_NAME)
PRODUCTS_MANIFEST = data_file(PRODUCTS_MANIFEST_NAME)

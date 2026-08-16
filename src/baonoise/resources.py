"""Access the immutable data files distributed with :mod:`baonoise`.

The returned objects implement :class:`importlib.resources.abc.Traversable`.
Callers should use ``open()`` rather than converting them to filesystem paths:
that keeps resource access valid for every import loader, including archives.
"""
from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable

DATA_PACKAGE = "baonoise.data"
DEFAULT_BANK_NAME = "fisher_bank_chime2022.npz"
DEFAULT_RATES_NAME = "survey_quarterly_rates_all23.csv"


def data_file(name: str) -> Traversable:
    """Return a packaged data file, failing clearly if packaging is broken."""
    resource = files(DATA_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(
            f"baonoise package data is missing {name!r}; reinstall baonoise "
            "from a complete wheel or sdist")
    return resource


DEFAULT_BANK = data_file(DEFAULT_BANK_NAME)
DEFAULT_RATES_CSV = data_file(DEFAULT_RATES_NAME)

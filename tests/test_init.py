"""Release identity and intentionally supported top-level imports."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

import baonoise


RELEASE_VERSION = "1.0.0"
PUBLIC_MODULES = (
    "api",
    "channels",
    "compat",
    "constants",
    "cosmologies",
    "fisherbank",
    "forecast",
    "incumbent",
    "layout",
    "pkcache",
    "products",
    "residual",
    "resources",
    "scenarios",
    "survey",
)


def test_release_version_is_consistent_across_public_metadata():
    root = Path(__file__).resolve().parents[1]

    assert baonoise.__version__ == RELEASE_VERSION
    assert f'version = "{RELEASE_VERSION}"' in (
        root / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version: "{RELEASE_VERSION}"' in (
        root / "CITATION.cff").read_text(encoding="utf-8")


def test_installed_distribution_matches_public_version_when_available():
    try:
        installed = metadata.version("baonoise")
    except metadata.PackageNotFoundError:
        return
    assert installed == RELEASE_VERSION


def test_public_module_exports_are_exact_and_importable():
    assert tuple(baonoise.__all__) == PUBLIC_MODULES
    for name in PUBLIC_MODULES:
        assert getattr(baonoise, name) is not None

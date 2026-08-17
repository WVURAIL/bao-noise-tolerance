"""RadioFisher backend-contract validation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from baonoise.compat import backend_capabilities


def _backend(api_version=1, capabilities=frozenset({"feature"})):
    return SimpleNamespace(
        BACKEND_ID="radiofisher",
        BACKEND_API_VERSION=api_version,
        get_backend_capabilities=lambda: capabilities)


def test_backend_api_contract_is_exact_not_forward_guessed():
    assert backend_capabilities(_backend()) == frozenset({"feature"})
    for version in (None, True, 0, 2):
        with pytest.raises(RuntimeError, match="must equal 1"):
            backend_capabilities(_backend(api_version=version))


def test_backend_capabilities_are_an_immutable_string_set():
    for declared in ({"feature"}, ["feature"], frozenset({1})):
        with pytest.raises(RuntimeError, match=r"frozenset\[str\]"):
            backend_capabilities(_backend(capabilities=declared))

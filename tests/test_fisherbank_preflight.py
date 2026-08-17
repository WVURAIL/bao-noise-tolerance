"""The build must refuse to advertise an override that did no work.

``expt.update()`` accepts any key. A RadioFisher checkout that does not
implement one silently ignores it, and the build then runs to completion and
writes a bank whose filename and metadata both claim the override while its
Fisher matrices lack the parameter row it was supposed to add. That happened
once, cost a seven-minute build, and was caught only by counting parameters
downstream. The preflight makes it a first-grid-point error instead.
"""
from __future__ import annotations

import pytest

from baonoise import fisherbank


@pytest.fixture
def stub_fisher(monkeypatch):
    """Replace the worker with one returning a chosen parameter list."""
    def install(paramnames):
        monkeypatch.setattr(
            fisherbank, "_one_fisher",
            lambda task: (0, task[1], None, list(paramnames)))
        monkeypatch.setitem(fisherbank._CTX, "rf_dir", "/fake/RadioFisher")
    return install


def test_no_overrides_needs_no_check(stub_fisher):
    """The preflight must not cost a Fisher evaluation when nothing needs it."""
    calls = []
    fisherbank._preflight(None, 1.0)
    fisherbank._preflight({}, 1.0)
    fisherbank._preflight({"wedge": "horizon"}, 1.0)   # not a row-adding one
    assert calls == []


def test_missing_row_is_an_error(stub_fisher):
    stub_fisher(["A", "b_HI", "aperp", "apar", "pk"])
    with pytest.raises(RuntimeError) as exc:
        fisherbank._preflight({"P_res": 1.0}, 1e4)
    msg = str(exc.value)
    assert "_Pres" in msg
    assert "P_res capability" in msg           # tells the user the actual fix
    assert "/fake/RadioFisher" in msg          # and which checkout is at fault


def test_present_row_passes(stub_fisher):
    stub_fisher(["A", "b_HI", "aperp", "apar", "pk", "_Pres"])
    fisherbank._preflight({"P_res": 1.0}, 1e4)


def test_unrelated_overrides_are_not_policed(stub_fisher):
    """Only overrides that are supposed to add a row are checked."""
    stub_fisher(["A", "aperp", "apar", "pk"])
    fisherbank._preflight({"wedge": "horizon", "kfg_fac": 80.0}, 1e4)


def test_foreground_provenance_reports_the_experiment_not_the_arguments(monkeypatch):
    """The claim 'Tsys and nothing else' must be checkable from the file.

    build_bank takes epsilon_fg as an argument, but the chime2022 path ignores
    it and uses the experiment definition's own value. Banks built that way
    recorded the argument, so every one of them advertised a 1e-6 foreground
    residual while in fact carrying zero. Provenance now comes from the built
    experiment.
    """
    monkeypatch.setitem(
        fisherbank._CTX, "make_expt",
        lambda t: {"epsilon_fg": 0, "k_nl0": 0.14,
                   "use": {"alpha_bao_shift": True}, "Tsys_tot(z)": lambda z: 55.0})
    prov = fisherbank._foreground_provenance()["foreground_settings"]
    assert prov["epsilon_fg"] == 0            # not the 1e-6 default argument
    assert prov["kfg_fac"] is None            # absent means no delay filter
    assert prov["wedge"] is None              # absent means no wedge excision
    assert prov["use"] == {"alpha_bao_shift": True}


def test_foreground_provenance_survives_a_missing_context():
    fisherbank._CTX.pop("make_expt", None)
    assert fisherbank._foreground_provenance() == {
        "foreground_settings": "unavailable"}


def test_unknown_config_fails_before_importing_backend(monkeypatch, tmp_path):
    def unexpected_import(*_args, **_kwargs):
        raise AssertionError("backend import started before config validation")

    monkeypatch.setattr("baonoise.compat.import_radiofisher", unexpected_import)
    with pytest.raises(ValueError, match="unknown config"):
        fisherbank._init_context(
            None, tmp_path / "cache.dat", "unknown", 1e-6, 0.14)


def test_invalid_config_cosmology_pair_fails_before_backend(monkeypatch,
                                                            tmp_path):
    def unexpected_import(*_args, **_kwargs):
        raise AssertionError("backend import started before config validation")

    monkeypatch.setattr("baonoise.compat.import_radiofisher", unexpected_import)
    with pytest.raises(ValueError, match="does not support cosmology"):
        fisherbank.build_bank(
            tmp_path / "wrong.npz", config="bull2015", cosmology="pact2025")

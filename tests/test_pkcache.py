"""Cosmology-density and power-cache provenance regressions."""
from __future__ import annotations

import json
import pytest

from baonoise import cosmologies, pkcache, resources


def _planck_like():
    return {"h": 0.6732, "omega_M_0": 0.3158,
            "omega_b_0": 0.022383 / 0.6732**2,
            "mnu": 0.06, "N_eff": 3.046, "ns": 0.96605,
            "sigma_8": 0.8120}


def test_massive_neutrinos_are_not_counted_as_cold_dark_matter():
    cosmo = _planck_like()
    densities = cosmologies.physical_densities(cosmo)
    total = cosmo["omega_M_0"] * cosmo["h"]**2
    assert densities["omnuh2"] == pytest.approx(
        cosmo["mnu"] / cosmologies.NEUTRINO_MASS_DENSITY_EV)
    assert densities["omch2"] == pytest.approx(
        total - densities["ombh2"] - densities["omnuh2"])
    assert sum(densities.values()) == pytest.approx(total)


def test_matter_total_can_explicitly_exclude_neutrinos():
    cosmo = _planck_like() | {"omega_M_0_includes_neutrinos": False}
    densities = cosmologies.physical_densities(cosmo)
    declared_non_neutrino = cosmo["omega_M_0"] * cosmo["h"]**2
    assert densities["ombh2"] + densities["omch2"] \
        == pytest.approx(declared_non_neutrino)
    assert densities["omnuh2"] > 0.0


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_matter_total_neutrino_convention_must_be_boolean(value):
    with pytest.raises(TypeError, match="must be a boolean"):
        cosmologies.physical_densities(
            _planck_like() | {"omega_M_0_includes_neutrinos": value})


@pytest.mark.parametrize("field", ["omega_b_0", "omega_M_0", "mnu"])
@pytest.mark.parametrize("bad", [True, "0.1", -0.1, float("nan"), float("inf")])
def test_derived_cosmology_inputs_are_strict_nonnegative_scalars(field, bad):
    with pytest.raises(ValueError, match=field):
        cosmologies.physical_densities(_planck_like() | {field: bad})


def test_complete_explicit_density_triplet_is_authoritative():
    cosmo = _planck_like() | {"ombh2": 0.022383, "omch2": 0.13,
                             "omnuh2": 0.001}
    assert cosmologies.physical_densities(cosmo) == {
        "ombh2": 0.022383, "omch2": 0.13, "omnuh2": 0.001}


@pytest.mark.parametrize("present", [
    {"ombh2": 0.02},
    {"ombh2": 0.02, "omch2": 0.12},
    {"omnuh2": 0.001},
])
def test_partial_explicit_density_triplet_is_rejected(present):
    with pytest.raises(ValueError, match="all-or-none"):
        cosmologies.physical_densities(_planck_like() | present)


@pytest.mark.parametrize("bad", [True, -0.1, float("nan"), float("inf")])
def test_explicit_physical_densities_are_strict_nonnegative_scalars(bad):
    with pytest.raises(ValueError, match="ombh2"):
        cosmologies.physical_densities(
            _planck_like() |
            {"ombh2": bad, "omch2": 0.12, "omnuh2": 0.001})


def test_explicit_neutrino_density_overrides_inconsistent_mass():
    cosmo = _planck_like() | {
        "mnu": 3.0, "ombh2": 0.022383, "omch2": 0.12, "omnuh2": 0.001}
    expected_mass = 0.001 * cosmologies.NEUTRINO_MASS_DENSITY_EV
    assert pkcache.cosmology_payload(cosmo)["mnu"] == pytest.approx(
        expected_mass)
    assert cosmologies.with_explicit_physical_densities(cosmo)["mnu"] \
        == pytest.approx(expected_mass)


def test_camb_receives_the_authoritative_neutrino_density():
    class Parameters:
        omnuh2 = None

        def set_cosmology(self, **kwargs):
            assert kwargs["mnu"] is None
            self.omnuh2 = kwargs["omnuh2_active"]

    densities = {"ombh2": 0.02, "omch2": 0.12, "omnuh2": 0.001}
    pars = Parameters()
    pkcache._set_camb_cosmology(pars, _planck_like(), densities)
    assert pars.omnuh2 == densities["omnuh2"]


def test_camb_density_round_trip_is_checked():
    class Parameters:
        omnuh2 = 0.0

        def set_cosmology(self, **_kwargs):
            self.omnuh2 = 0.002

    with pytest.raises(RuntimeError, match="did not preserve"):
        pkcache._set_camb_cosmology(
            Parameters(), _planck_like(),
            {"ombh2": 0.02, "omch2": 0.12, "omnuh2": 0.001})


def test_backend_owned_profiles_pin_every_radiofisher_model_choice():
    profiles = {
        "bull2015": ("powerlaw", "powerlaw", "powerlaw"),
        "chime_overview_2022": ("hall", "castorina", "crighton"),
    }

    class Backend:
        @staticmethod
        def with_astrophysical_profile(cosmo, profile):
            try:
                models = profiles[profile]
            except KeyError as exc:
                raise ValueError("unknown astrophysical profile") from exc
            out = dict(cosmo)
            out["astrophysical_model_profile"] = profile
            out.update(dict(zip(
                ("Tb_model", "bias_HI_model", "omega_HI_model"), models)))
            return out

    rf = Backend()
    bull = cosmologies.with_astrophysical_profile(
        _planck_like(), "bull2015", rf)
    assert bull["astrophysical_model_profile"] == "bull2015"
    assert {bull[key] for key in
            ("Tb_model", "bias_HI_model", "omega_HI_model")} == {"powerlaw"}
    overview = cosmologies.with_astrophysical_profile(
        _planck_like(), "chime_overview_2022", rf)
    assert (overview["Tb_model"], overview["bias_HI_model"],
            overview["omega_HI_model"]) == ("hall", "castorina", "crighton")
    with pytest.raises(ValueError, match="unknown astrophysical profile"):
        cosmologies.with_astrophysical_profile(
            _planck_like(), "unknown_profile", rf)


def test_unversioned_sentinel_is_not_a_cache_identity(tmp_path):
    cache = tmp_path / "unversioned.dat"
    cache.write_text("# pythoncamb#\n1 2\n", encoding="utf-8")
    assert not pkcache.cache_matches(cache, _planck_like())
    with pytest.raises(ValueError, match="unversioned"):
        pkcache.inspect_pk_cache(cache)


def test_cache_content_tampering_is_detected(tmp_path):
    body = b"1 2\n3 4\n"
    request = pkcache._request_payload(_planck_like(), 20.0, 1400)
    meta = {**request, "generator": {"package": "camb", "version": "x"},
            "data_sha256": pkcache._sha256_bytes(body)}
    cache_id = pkcache._sha256_bytes(pkcache._canonical_json(meta).encode())
    cache = tmp_path / "cache.dat"
    cache.write_bytes(
        f"# {cache_id}#\n# baonoise-meta {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n".encode()
        + body)
    assert pkcache.cache_matches(cache, _planck_like())
    cache.write_bytes(cache.read_bytes() + b"5 6\n")
    with pytest.raises(ValueError, match="data digest"):
        pkcache.inspect_pk_cache(cache)


def test_committed_caches_are_content_verified_v2():
    expected_omch2 = {
        "cache_pk.dat": 0.1198563,
        "cache_pk_chime2022.dat": 0.12009212027110579,
        "cache_pk_chime2022_pact2025.dat": 0.118,
    }
    for name, expected in expected_omch2.items():
        meta = pkcache.inspect_pk_cache(resources.filesystem_data_file(name))
        assert meta["format"] == pkcache.CACHE_FORMAT
        assert meta["cosmology"]["omch2"] == pytest.approx(expected)
        assert meta["settings"]["neutrino_input"] == "omnuh2_active"

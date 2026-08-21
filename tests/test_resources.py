"""Package-data regressions for source, wheel, and archive installations."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest

from baonoise import channels, fisherbank, forecast, resources, scenarios
from baonoise.compat import find_radiofisher_dir
from baonoise.fisherbank import (ARTIFACT_FORECAST, BANK_SCHEMA_VERSION,
                                 FisherBank)


EXPECTED_SHA256 = {
    resources.DEFAULT_BANK_NAME:
        "25272a9f0fc9d59266e48314503bfb5d5fa424119bc0560128db38e5ef3a0379",
    resources.PACT2025_BANK_NAME:
        "693c2f974ee47c2aeebce9ef63824e2cf622c78183cbc0f945ab1278c884bfa0",
    resources.DEFAULT_RATES_NAME:
        "da8c1c1df1f3929920ac132ea037adaa7cad5f5edb215e046ec5a40281d6bde3",
    resources.PRODUCTS_MANIFEST_NAME:
        "cede0afd1dd7e79bac8f94d198645081fc014cb07eead3ca26e6128c3fbeb165",
    resources.SYNTHETIC_BASELINE_NAME:
        "101a156d88c212781b098f89a86bfb11ada58433d5dcc5cc2fb86956c930790b",
    "cache_pk.dat":
        "c33b7e8e9b5e5abff63476e518a0a4376ac1a5043aa32a811e037dce5c81506f",
    "cache_pk_chime2022.dat":
        "aaa8a35e42723c2df364616e40a73f340fb1887a7db35507496e6f17451daadf",
    "cache_pk_chime2022_pact2025.dat":
        "f7bdfdc9c241864432b888b670cc99a03d961c164e4e27a0c37e43b08c16e708",
}
CANONICAL_TEXT_RESOURCES = frozenset({
    resources.DEFAULT_RATES_NAME,
    resources.SYNTHETIC_BASELINE_NAME,
})
BULL_BANK_SHA256 = {
    "fisher_bank_bull2015_planck2013_epsfg1e-6.npz":
        "7e21a37435c6d47e83f328883c8234342a85030025ce8f316220b6cbd47b8dff",
    "fisher_bank_bull2015_planck2013_epsfg1e-5.npz":
        "da5f188cad52526c5835b6eb2c6b7e91fd1af3fb01622bc47d269465b3b03f27",
}
EXPECTED_RADIOFISHER_SOURCE_SHA256 = (
    "864980e9658c475093fc34c7da02e456bf2849d1fd239ef6152fe6508b9a68d7"
)


def _sha256(resource, canonical_text: bool = False) -> str:
    with resource.open("rb") as stream:
        data = stream.read()
    if canonical_text:
        # Scientific text content is independent of the checkout's newline
        # convention. .gitattributes enforces LF for future checkouts, while
        # this normalization also handles an already-populated Windows tree.
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_packaged_data_bytes_are_unchanged():
    """Moving the canonical files into the package must not alter them."""
    for name, expected in EXPECTED_SHA256.items():
        canonical_text = name in CANONICAL_TEXT_RESOURCES
        assert _sha256(resources.data_file(name), canonical_text) == expected


def test_canonical_text_hash_is_newline_independent(tmp_path):
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"channel,rate\n14,0.1\n")
    crlf.write_bytes(b"channel,rate\r\n14,0.1\r\n")
    assert _sha256(lf, canonical_text=True) == _sha256(
        crlf, canonical_text=True)


def test_scientific_dat_resources_are_checkout_stable_lf():
    attributes = (Path(__file__).resolve().parents[1] / ".gitattributes") \
        .read_text(encoding="utf-8")
    assert "src/baonoise/data/*.dat text eol=lf" in attributes


def test_source_checkout_defaults_load_from_package_data():
    bank = FisherBank(resources.DEFAULT_BANK)
    assert bank.schema_version == BANK_SCHEMA_VERSION
    assert "version" not in bank.meta
    assert bank.meta["config"] == "chime2022"
    assert bank.meta["cosmology"] == "planck2018"
    assert bank.nbins == 15
    assert len(channels.measured_mask_fractions()) == 23


def test_resource_annotations_resolve_on_supported_python():
    """The Traversable type moved after Python 3.10."""
    assert get_type_hints(resources.data_file)["return"].__name__ \
        == "Traversable"
    assert resources.DEFAULT_BANK.is_file()


def test_named_bank_registry_is_exact_and_immutable():
    assert dict(resources.BANK_NAMES) == {
        "planck2018": resources.DEFAULT_BANK_NAME,
        "pact2025": resources.PACT2025_BANK_NAME,
    }
    assert resources.bank_file("planck2018") == resources.DEFAULT_BANK
    assert resources.PACT2025_BANK.name == resources.PACT2025_BANK_NAME
    with pytest.raises(TypeError):
        resources.BANK_NAMES["alias"] = resources.DEFAULT_BANK_NAME


def test_unknown_named_bank_fails_before_resource_lookup():
    with pytest.raises(ValueError, match="unknown packaged bank cosmology"):
        resources.bank_file("unknown")


@pytest.mark.parametrize("cosmology", ["planck2018", "pact2025"])
def test_named_packaged_banks_are_strict_v2_forecasts(cosmology):
    expected = (resources.DEFAULT_BANK if cosmology == "planck2018"
                else resources.PACT2025_BANK)
    if not expected.is_file():
        pytest.skip(
            f"{cosmology} bank is awaiting its reproducible release build")

    resource = resources.bank_file(cosmology)
    bank = FisherBank(resource)

    assert resource.name == resources.BANK_NAMES[cosmology]
    assert bank.schema_version == BANK_SCHEMA_VERSION
    assert bank.artifact_kind == ARTIFACT_FORECAST
    assert "version" not in bank.meta
    assert bank.meta["config"] == "chime2022"
    assert bank.meta["cosmology"] == cosmology
    assert bank.meta["provenance"]["cosmology"]["name"] == cosmology


def test_named_banks_are_distinct_matched_1_0_builds():
    planck_resource = resources.bank_file("planck2018")
    pact_resource = resources.bank_file("pact2025")
    assert _sha256(planck_resource) != _sha256(pact_resource)

    planck = FisherBank(planck_resource)
    pact = FisherBank(pact_resource)
    assert planck.meta["cosmology"] == "planck2018"
    assert pact.meta["cosmology"] == "pact2025"
    assert not np.array_equal(planck.F_grid, pact.F_grid)
    assert np.array_equal(planck.t_grid, pact.t_grid)
    assert np.array_equal(planck.zs, pact.zs)
    assert planck.paramnames == pact.paramnames

    root = Path(__file__).resolve().parents[1]
    current_bao_digest = fisherbank._git_state(
        root, **fisherbank.BAONOISE_SOURCE_MANIFEST)["working_tree_sha256"]
    expected_caches = {
        "planck2018": "cache_pk_chime2022.dat",
        "pact2025": "cache_pk_chime2022_pact2025.dat",
    }
    try:
        radiofisher_root = find_radiofisher_dir()
    except FileNotFoundError:
        radiofisher_root = None
    radio_digests = set()
    for cosmology, bank in (("planck2018", planck), ("pact2025", pact)):
        provenance = bank.meta["provenance"]
        assert provenance["baonoise"]["version"] == "1.0.0"
        assert provenance["radiofisher"]["backend_version"] == "1.0.0"
        assert provenance["baonoise"]["working_tree_sha256"] \
            == current_bao_digest
        assert provenance["baonoise"]["source_manifest"] == {
            key: list(value) for key, value in
            fisherbank.BAONOISE_SOURCE_MANIFEST.items()}
        assert provenance["radiofisher"]["source_manifest"] == {
            key: list(value) for key, value in
            fisherbank.RADIOFISHER_SOURCE_MANIFEST.items()}
        cache_name = expected_caches[cosmology]
        assert provenance["pk_cache"]["filename"] == cache_name
        assert provenance["pk_cache"]["sha256"] == _sha256(
            resources.data_file(cache_name))
        radio_digests.add(
            provenance["radiofisher"]["working_tree_sha256"])
    assert radio_digests == {EXPECTED_RADIOFISHER_SOURCE_SHA256}
    if radiofisher_root is not None:
        current_radio_digest = fisherbank._git_state(
            radiofisher_root,
            **fisherbank.RADIOFISHER_SOURCE_MANIFEST,
        )["working_tree_sha256"]
        assert current_radio_digest == EXPECTED_RADIOFISHER_SOURCE_SHA256


def test_packaged_pact_bank_matches_direct_backend_for_masked_bin():
    try:
        find_radiofisher_dir()
    except FileNotFoundError:
        pytest.skip("direct P-ACT validation requires a RadioFisher checkout")
    bank = FisherBank(resources.PACT2025_BANK)
    calculator = forecast.Forecast(bank, style="perbin_A")
    scenario = scenarios.measured()
    bank_sigma = calculator.sigma_A(scenario, 1.0e4, bins=[6])
    direct_sigma = calculator.sigma_A_direct(
        scenario, 1.0e4, bins=[6])
    assert direct_sigma == pytest.approx(bank_sigma, rel=0.015)


def test_bull_research_banks_are_matched_strict_v2_1_0_builds():
    root = Path(__file__).resolve().parents[1]
    current_bao_digest = fisherbank._git_state(
        root, **fisherbank.BAONOISE_SOURCE_MANIFEST)["working_tree_sha256"]
    banks = []
    radio_digests = set()
    for name, expected_sha256 in BULL_BANK_SHA256.items():
        path = root / "data" / name
        assert _sha256(path) == expected_sha256
        bank = FisherBank(path)
        banks.append(bank)

        assert bank.schema_version == BANK_SCHEMA_VERSION
        assert bank.artifact_kind == ARTIFACT_FORECAST
        assert "version" not in bank.meta
        assert bank.meta["config"] == "bull2015"
        assert bank.meta["cosmology"] == "planck2013"
        assert bank.meta["astrophysical_model_profile"] == "bull2015"
        provenance = bank.meta["provenance"]
        assert provenance["cosmology"]["astrophysical_models"] == {
            "Tb_model": "powerlaw",
            "bias_HI_model": "powerlaw",
            "omega_HI_model": "powerlaw",
        }
        assert provenance["baonoise"]["version"] == "1.0.0"
        assert provenance["radiofisher"]["backend_version"] == "1.0.0"
        assert provenance["baonoise"]["working_tree_sha256"] \
            == current_bao_digest
        assert provenance["pk_cache"]["filename"] == "cache_pk.dat"
        assert provenance["pk_cache"]["sha256"] == EXPECTED_SHA256[
            "cache_pk.dat"]
        radio_digests.add(
            provenance["radiofisher"]["working_tree_sha256"])

    assert radio_digests == {EXPECTED_RADIOFISHER_SOURCE_SHA256}
    assert np.array_equal(banks[0].t_grid, banks[1].t_grid)
    assert np.array_equal(banks[0].zs, banks[1].zs)
    assert banks[0].paramnames == banks[1].paramnames
    assert not np.array_equal(banks[0].F_grid, banks[1].F_grid)
    epsilon_fg = {
        bank.meta["provenance"]["experiment"]["settings"]["epsilon_fg"]
        for bank in banks
    }
    assert epsilon_fg == {1e-6, 1e-5}


@pytest.mark.parametrize("name", sorted(BULL_BANK_SHA256))
def test_each_bull_research_bank_matches_its_recorded_direct_backend(name):
    try:
        rf_dir = find_radiofisher_dir()
    except FileNotFoundError:
        pytest.skip("direct Bull-bank validation requires a RadioFisher checkout")
    from baonoise.compat import import_radiofisher

    rf, rf_dir = import_radiofisher(rf_dir)
    bank = FisherBank(Path(__file__).resolve().parents[1] / "data" / name)
    calculator = forecast.Forecast(
        bank, rf=rf, style="shared_A", rf_dir=rf_dir)
    scenario = scenarios.clean()
    bank_sigma = calculator.sigma_A(scenario, 1.0e4, bins=[6])
    direct_sigma = calculator.sigma_A_direct(
        scenario, 1.0e4, bins=[6], rf_dir=rf_dir)
    assert direct_sigma == pytest.approx(bank_sigma, rel=0.015)


@pytest.mark.parametrize("name", sorted(BULL_BANK_SHA256))
def test_each_bull_bank_interpolates_bin8_below_one_percent(name):
    try:
        rf_dir = find_radiofisher_dir()
    except FileNotFoundError:
        pytest.skip("Bull interpolation validation requires RadioFisher")
    from baonoise.compat import import_radiofisher

    rf, rf_dir = import_radiofisher(rf_dir)
    bank = FisherBank(Path(__file__).resolve().parents[1] / "data" / name)
    calculator = forecast.Forecast(
        bank, rf=rf, style="shared_A", rf_dir=rf_dir)
    scenario = scenarios.clean()
    bank_sigma = calculator.sigma_A(scenario, 3300.0, bins=[8])
    direct_sigma = calculator.sigma_A_direct(
        scenario, 3300.0, bins=[8], rf_dir=rf_dir)

    assert abs(bank_sigma / direct_sigma - 1.0) < 0.01


def test_missing_packaged_data_fails_clearly():
    with pytest.raises(FileNotFoundError, match="package data is missing"):
        resources.data_file("not-distributed.dat")


def test_defaults_work_through_a_zip_importer(tmp_path):
    """Exercise archive-backed Traversables, not only pathlib resources."""
    package_root = Path(__file__).resolve().parents[1] / "src" / "baonoise"
    archive = tmp_path / "baonoise-test.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source in package_root.rglob("*"):
            if (source.is_file() and "__pycache__" not in source.parts
                    and source.suffix != ".pyc"):
                destination = (Path("baonoise")
                               / source.relative_to(package_root))
                zf.write(source, destination)

    code = f"""
import sys
sys.path.insert(0, {str(archive)!r})
from baonoise import api, channels, products, resources
assert '.zip/' in str(resources.DEFAULT_BANK)
forecast = api.load()
assert forecast.bank.meta['config'] == 'chime2022'
assert len(channels.measured_mask_fractions()) == 23
found, missing = products.load()
assert products.freq_id(35) == 521
assert len(found) + len(missing) == 23
for resource_name in resources.RADIOFISHER_FILESYSTEM_NAMES:
    with resources.data_file(resource_name).open('rb') as stream:
        assert stream.read(1)
try:
    resources.filesystem_data_file('cache_pk.dat')
except RuntimeError as exc:
    assert 'install the baonoise wheel' in str(exc)
else:
    raise AssertionError('zip-imported cache unexpectedly had a filesystem path')
result = api.required_time(forecast, uniform=0.0)
assert result['hours'] > 0.0
print(result['hours'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_built_wheel_contains_and_loads_named_banks_and_manifest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--wheel-dir", str(wheel_dir), str(root)],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(wheel_dir.glob("baonoise-*.whl"))
    code = f"""
import sys
sys.path.insert(0, {str(wheel)!r})
import baonoise
from baonoise import products, resources
from baonoise.fisherbank import BANK_SCHEMA_VERSION, FisherBank
assert baonoise.__version__ == '1.0.0'
assert products.freq_id(35) == 521
found, missing = products.load()
assert len(found) + len(missing) == 23
assert set(resources.BANK_NAMES) == {{'planck2018', 'pact2025'}}
for cosmology in resources.BANK_NAMES:
    bank = FisherBank(resources.bank_file(cosmology))
    assert bank.schema_version == BANK_SCHEMA_VERSION
    assert bank.meta['cosmology'] == cosmology
for resource_name in resources.RADIOFISHER_FILESYSTEM_NAMES:
    with resources.data_file(resource_name).open('rb') as stream:
        assert stream.read(1)
"""
    loaded = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        text=True, capture_output=True, check=False)
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr

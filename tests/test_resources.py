"""Package-data regressions for source, wheel, and archive installations."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baonoise import channels, resources
from baonoise.fisherbank import FisherBank


EXPECTED_SHA256 = {
    resources.DEFAULT_BANK_NAME:
        "9ad94cf506b135c6ad93c9c46163203818b8810038b688869be12db5f69bff5f",
    resources.DEFAULT_RATES_NAME:
        "da8c1c1df1f3929920ac132ea037adaa7cad5f5edb215e046ec5a40281d6bde3",
}
CANONICAL_TEXT_RESOURCES = frozenset({resources.DEFAULT_RATES_NAME})


def _sha256(resource, canonical_text: bool = False) -> str:
    with resource.open("rb") as stream:
        data = stream.read()
    if canonical_text:
        # The scientific CSV content is independent of the checkout's newline
        # convention.  .gitattributes enforces LF for future checkouts, while
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


def test_source_checkout_defaults_load_from_package_data():
    bank = FisherBank(resources.DEFAULT_BANK)
    assert bank.meta["config"] == "chime2022"
    assert bank.nbins == 15
    assert len(channels.measured_mask_fractions()) == 23


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
from baonoise import api, channels, resources
assert '.zip/' in str(resources.DEFAULT_BANK)
forecast = api.load()
assert forecast.bank.meta['config'] == 'chime2022'
assert len(channels.measured_mask_fractions()) == 23
result = api.required_time(forecast, uniform=0.0)
assert result['hours'] > 0.0
print(result['hours'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

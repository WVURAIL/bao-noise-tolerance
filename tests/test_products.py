import json

from baonoise import products


def _write(tmp_path, manifest, local=None):
    m = tmp_path / "products.json"
    m.write_text(json.dumps(manifest))
    l = tmp_path / "products.local.json"
    if local is not None:
        l.write_text(json.dumps(local))
    return m, l


def test_fid_and_prefixed_resolution(tmp_path):
    d = tmp_path / "prod"
    d.mkdir()
    (d / "552.npz").write_bytes(b"x")
    (d / "abc123-598.npz").write_bytes(b"x")
    m, l = _write(tmp_path, {
        "search_dirs": [str(d)],
        "channels": {"33": {"freq_id": 552}, "30": {"freq_id": 598},
                     "29": {"freq_id": 614}}})
    found, missing = products.load(m, l)
    assert found[33].endswith("552.npz")
    assert found[30].endswith("abc123-598.npz")
    assert missing == [29]


def test_explicit_path_precedence(tmp_path):
    d = tmp_path / "prod"
    d.mkdir()
    (d / "552.npz").write_bytes(b"x")
    override = tmp_path / "elsewhere.npz"
    override.write_bytes(b"x")
    m, l = _write(tmp_path,
                  {"search_dirs": [str(d)],
                   "channels": {"33": {"freq_id": 552}}},
                  {"channels": {"33": {"path": str(override)}}})
    found, _ = products.load(m, l)
    assert found[33] == str(override)


def test_env_dirs_searched_first(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "552.npz").write_bytes(b"env wins")
    (b / "552.npz").write_bytes(b"manifest dir")
    m, l = _write(tmp_path, {"search_dirs": [str(b)],
                             "channels": {"33": {"freq_id": 552}}})
    monkeypatch.setenv(products.ENV_DIRS, str(a))
    found, _ = products.load(m, l)
    assert found[33] == str(a / "552.npz")


def test_repo_manifest_registers_all_23():
    manifest = json.loads(products.MANIFEST.read_text())
    chans = manifest["channels"]
    assert sorted(int(c) for c in chans) == list(range(14, 37))
    fids = [chans[str(c)]["freq_id"] for c in range(14, 37)]
    assert fids[0] == 844 and fids[-1] == 506
    assert all(fids[i] > fids[i+1] for i in range(len(fids)-1))

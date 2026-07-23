"""One registry for the survey products.

Every analysis script used to carry its own hardcoded product paths; this
module replaces all of them with a single manifest (``data/products.json``,
tracked) plus an optional machine-local overlay (``data/products.local.json``,
gitignored) and an environment hook (``$BAONOISE_PRODUCT_DIRS``,
colon-separated, searched first).

Resolution, per channel: an explicit ``path`` (local overlay first, then
manifest) that exists on disk wins; otherwise each search directory is tried
for ``{freq_id}.npz`` and then ``*-{freq_id}.npz`` (survey exports sometimes
carry a hash prefix). Channels with no file are *reported* rather than defaulted:
the registry follows the same discipline as everything else here --- refusal
over silence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = _ROOT / "data" / "products.json"
LOCAL = _ROOT / "data" / "products.local.json"
ENV_DIRS = "BAONOISE_PRODUCT_DIRS"


def _search_dirs(manifest: dict, local: dict) -> list[Path]:
    dirs: list[Path] = []
    for d in os.environ.get(ENV_DIRS, "").split(":"):
        if d:
            dirs.append(Path(d))
    for src in (local, manifest):
        for d in src.get("search_dirs", []):
            p = Path(d)
            dirs.append(p if p.is_absolute() else _ROOT / p)
    return dirs


def load(manifest_path: Path = MANIFEST,
         local_path: Path = LOCAL) -> tuple[dict[int, str], list[int]]:
    """(found, missing): every registered channel resolved to a file path,
    and the sorted channels whose products are absent everywhere."""
    manifest = json.loads(Path(manifest_path).read_text())
    local = (json.loads(Path(local_path).read_text())
             if Path(local_path).exists() else {})
    dirs = _search_dirs(manifest, local)
    local_ch = local.get("channels", {})
    found: dict[int, str] = {}
    missing: list[int] = []
    for ch_s, meta in sorted(manifest["channels"].items(),
                             key=lambda kv: int(kv[0])):
        ch = int(ch_s)
        explicit = (local_ch.get(ch_s, {}).get("path")
                    or meta.get("path"))
        if explicit and Path(explicit).exists():
            found[ch] = str(explicit)
            continue
        fid = meta["freq_id"]
        hit = None
        for d in dirs:
            if not d.is_dir():
                continue
            cand = d / f"{fid}.npz"
            if cand.exists():
                hit = cand
                break
            matches = sorted(d.glob(f"*-{fid}.npz"))
            if matches:
                hit = matches[0]
                break
        if hit is not None:
            found[ch] = str(hit)
        else:
            missing.append(ch)
    return found, missing


def paths(channels=None, announce: bool = True) -> dict[int, str]:
    """Resolved product paths, optionally restricted to ``channels``.
    Absent channels are printed once (a report rather than an error): scripts
    proceed on what exists, and the printout says what is still awaited."""
    found, missing = load()
    if channels is not None:
        missing = [c for c in channels if c not in found]
        found = {c: found[c] for c in channels if c in found}
    if announce and missing:
        print("[products] awaiting: "
              + ", ".join(f"ch{c}" for c in missing))
    return found


def freq_id(ch: int) -> int:
    manifest = json.loads(MANIFEST.read_text())
    return int(manifest["channels"][str(ch)]["freq_id"])

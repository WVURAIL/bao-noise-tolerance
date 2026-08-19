#!/usr/bin/env python3
"""Re-pin the shipped Fisher banks' SHA-256 hashes after a rebuild.

The four shipped banks are pinned in tests/test_resources.py (the two
packaged CHIME banks under EXPECTED_SHA256, the two repo-data Bull-2015
banks under BULL_BANK_SHA256). After scripts/rebuild_shipped_banks.sh
replaces the bank files, this rewrites each pin in place, keyed by the
dict entry's anchor rather than the old hash value, so it works from any
starting state. ``--check`` verifies the current pins against the current
files and changes nothing.

    python scripts/restamp_bank_pins.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests" / "test_resources.py"

# pin anchor in tests/test_resources.py -> bank file in the checkout
BANKS = {
    "resources.DEFAULT_BANK_NAME:":
        "src/baonoise/data/fisher_bank_chime2022.npz",
    "resources.PACT2025_BANK_NAME:":
        "src/baonoise/data/fisher_bank_chime2022_pact2025.npz",
    '"fisher_bank_bull2015_planck2013_epsfg1e-6.npz":':
        "data/fisher_bank_bull2015_planck2013_epsfg1e-6.npz",
    '"fisher_bank_bull2015_planck2013_epsfg1e-5.npz":':
        "data/fisher_bank_bull2015_planck2013_epsfg1e-5.npz",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify pins against the bank files; write nothing")
    args = ap.parse_args(argv)

    text = TESTS.read_text(encoding="utf-8")
    stale = 0
    for anchor, rel in BANKS.items():
        new = sha(ROOT / rel)
        pattern = re.compile(
            "(" + re.escape(anchor) + r'\s*\n\s*")([0-9a-f]{64})(")')
        m = pattern.search(text)
        if not m:
            sys.exit(f"pin anchor not found in {TESTS}: {anchor}")
        old = m.group(2)
        name = Path(rel).name
        if old == new:
            print(f"{name}: pin current")
            continue
        stale += 1
        if args.check:
            print(f"{name}: STALE pin\n  pinned {old}\n  actual {new}")
        else:
            text = pattern.sub(lambda mm: mm.group(1) + new + mm.group(3),
                               text, count=1)
            print(f"{name}: re-pinned\n  {old} -> {new}")
    if args.check:
        print("pins", "STALE" if stale else "OK")
        return 1 if stale else 0
    if stale:
        TESTS.write_text(text, encoding="utf-8")
        print(f"updated {stale} pin(s) in {TESTS}")
    else:
        print("all pins already current; nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

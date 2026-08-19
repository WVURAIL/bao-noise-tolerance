#!/bin/bash
# Rebuild the four shipped Fisher banks with their exact release recipe,
# copy them into place, and re-pin tests/test_resources.py.
#
# Banks record a working_tree_sha256 over pyproject + src/baonoise/*.py at
# build time, so run this AFTER all source changes are final, from a clean
# tree, and commit the banks + pins together as a re-stamp commit.
#
# Requires an installed baonoise (baonoise-build-bank on PATH) and a
# RadioFisher checkout: $RADIOFISHER_DIR, or the ../RadioFisher sibling
# that baonoise finds automatically.
#
#   NPROC=24 scripts/rebuild_shipped_banks.sh [workdir]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain -- pyproject.toml src/baonoise)" ]; then
  echo "WARNING: pyproject/src/baonoise not clean; the banks will stamp a" >&2
  echo "         dirty tree. Commit or stash first for a release re-stamp." >&2
fi

WORK=${1:-$(mktemp -d)}
NPROC=${NPROC:-$(nproc)}
RF_ARGS=()
[ -n "${RADIOFISHER_DIR:-}" ] && RF_ARGS=(--radiofisher-dir "$RADIOFISHER_DIR")
baonoise-build-bank --version

build() {  # outfile, then baonoise-build-bank args
  local out=$1; shift
  echo "=== building $out ($(date +%H:%M:%S)) ==="
  baonoise-build-bank --out "$WORK/$out" --nt 27 --nproc "$NPROC" \
    "${RF_ARGS[@]}" "$@" 2>&1 | tail -3
  echo "done $out ($(date +%H:%M:%S))"
}

build fisher_bank_chime2022.npz          --config chime2022 --cosmology planck2018
build fisher_bank_chime2022_pact2025.npz --config chime2022 --cosmology pact2025
build fisher_bank_bull2015_planck2013_epsfg1e-6.npz \
      --config bull2015 --cosmology planck2013 --epsilon-fg 1e-6
build fisher_bank_bull2015_planck2013_epsfg1e-5.npz \
      --config bull2015 --cosmology planck2013 --epsilon-fg 1e-5

cp "$WORK/fisher_bank_chime2022.npz"          src/baonoise/data/
cp "$WORK/fisher_bank_chime2022_pact2025.npz" src/baonoise/data/
cp "$WORK/fisher_bank_bull2015_planck2013_epsfg1e-6.npz" data/
cp "$WORK/fisher_bank_bull2015_planck2013_epsfg1e-5.npz" data/
python scripts/restamp_bank_pins.py
echo "ALL SHIPPED BANKS REBUILT AND RE-PINNED (workdir: $WORK)"
echo "Now: python scripts/verify_bank.py && python -m pytest tests/ -q,"
echo "then commit banks + pins together as the re-stamp commit."

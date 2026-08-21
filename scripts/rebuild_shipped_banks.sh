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
# One targeted point resolves the Bull bin-8 interpolation knee at 3,300 hr
# without moving any of the 27 release-grid points. Keep the decimal literal:
# it is Python's round-trip representation of 10**3.5 hours.
BULL_KNEE_HOURS=3162.2776601683795
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
      --config bull2015 --cosmology planck2013 --epsilon-fg 1e-6 \
      --extra-time-hours "$BULL_KNEE_HOURS"
build fisher_bank_bull2015_planck2013_epsfg1e-5.npz \
      --config bull2015 --cosmology planck2013 --epsilon-fg 1e-5 \
      --extra-time-hours "$BULL_KNEE_HOURS"

cp "$WORK/fisher_bank_chime2022.npz"          src/baonoise/data/
cp "$WORK/fisher_bank_chime2022_pact2025.npz" src/baonoise/data/
cp "$WORK/fisher_bank_bull2015_planck2013_epsfg1e-6.npz" data/
cp "$WORK/fisher_bank_bull2015_planck2013_epsfg1e-5.npz" data/
python scripts/fg_sensitivity.py
python scripts/restamp_bank_pins.py
python scripts/check_paper_numbers.py
echo "ALL SHIPPED BANKS REBUILT AND RE-PINNED (workdir: $WORK)"
echo "foreground_sensitivity.csv regenerated and paper-number gate passed."
echo "Now: python scripts/verify_bank.py && python -m pytest tests/ -q,"
echo "then commit banks + pins together as the re-stamp commit."

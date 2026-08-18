"""ATSC DTV channel <-> 21cm frequency/redshift mapping, and ingestion of
pilot-proxy masking statistics.

A masking fraction is only meaningful next to the rule that produced it, and
the two sources here differ by up to 90x on the same channels:

* :func:`mask_table_from_products` reads the survey products directly and takes
  the rule out of each product's own detector contract. The number and the rule
  travel together and cannot drift apart.
* :func:`measured_mask_fractions` reads a quarterly rate CSV. Nothing in that
  file records what statistic or threshold produced its ``hi_rate_all``
  column, so a forecast built on it silently inherits an unidentified detector.
  It is kept for the published numbers; :func:`measured_mask_table` wraps it
  and marks the rule ``unrecorded`` so the gap is visible rather than implied.

Channels 24 and 30 are "refused" in pilot-proxy (no calibrated zero point: the
DTV transmitter is essentially always on); we model them as masked at
REFUSED_FRACTION.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .npzio import load_npz
from .constants import HI_REST_FREQUENCY_MHZ

from .resources import DEFAULT_RATES_CSV

ATSC_CH14_LOWER_EDGE = 470.0  # MHz
ATSC_WIDTH = 6.0              # MHz
ATSC_DTV_CHANNELS = tuple(range(14, 37))
ATSC_DTV_UPPER_EDGE = (
    ATSC_CH14_LOWER_EDGE + len(ATSC_DTV_CHANNELS) * ATSC_WIDTH)

# Channels with no calibrated zero point in pilot-proxy (persistently on-air).
REFUSED_CHANNELS = (24, 30)
REFUSED_FRACTION = 0.97       # masked fraction adopted for refused channels


def _source_name(source) -> str:
    """Display name for a path or package resource."""
    name = getattr(source, "name", None)
    return str(name) if name is not None else Path(source).name


def channel_edges(ch: int) -> tuple[float, float]:
    """Lower/upper frequency edge [MHz] of an ATSC UHF physical channel."""
    lo = ATSC_CH14_LOWER_EDGE + (ch - 14) * ATSC_WIDTH
    return lo, lo + ATSC_WIDTH


def channel_z_range(ch: int) -> tuple[float, float]:
    """21cm redshift interval covered by an ATSC channel (z_lo at upper edge)."""
    lo, hi = channel_edges(ch)
    return (HI_REST_FREQUENCY_MHZ / hi - 1.0,
            HI_REST_FREQUENCY_MHZ / lo - 1.0)


def measured_mask_fractions(rates_csv: str | Path = DEFAULT_RATES_CSV,
                            refused_fraction: float = REFUSED_FRACTION,
                            rate_col: str = "hi_rate_all") -> dict[int, float]:
    """Exposure-weighted mean masking fraction per ATSC channel.

    Weights each quarterly hi-rate by its n_valid_frames, then adds the
    refused channels at ``refused_fraction``.
    """
    num = defaultdict(float)
    den = defaultdict(float)
    opener = getattr(rates_csv, "open", None)
    fh = (opener("r", encoding="utf-8", newline="") if opener is not None
          else open(rates_csv, encoding="utf-8", newline=""))
    with fh:
        for row in csv.DictReader(fh):
            ch = int(row["atsc_channel"])
            n = float(row["n_valid_frames"])
            num[ch] += n * float(row[rate_col])
            den[ch] += n
    fractions = {ch: num[ch] / den[ch] for ch in num if den[ch] > 0}
    for ch in REFUSED_CHANNELS:
        fractions[ch] = refused_fraction
    return dict(sorted(fractions.items()))


@dataclass
class MaskTable:
    """Per-channel masked fractions, carrying the rule that produced them.

    The ``rule`` field is the reason this class exists. A bare
    ``{channel: fraction}`` dict cannot be
    checked against the detector it is supposed to describe; this can.
    """
    fractions: dict[int, float]
    source: str                     # 'products' | 'csv'
    rule: str                       # the detector's own rule, or 'unrecorded'
    detector_version: str = "unrecorded"
    n_frames: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    window: str = "full span"       # epoch the fractions describe

    @property
    def is_traceable(self) -> bool:
        return self.rule != "unrecorded"

    def summary(self) -> str:
        head = (f"{len(self.fractions)} channels from {self.source}; "
                f"rule: {self.rule}")
        if self.window != "full span":
            head += f"; window: {self.window}"
        body = "\n".join(
            f"    ch{ch:>3d}  {f:7.4f}"
            + (f"   ({self.n_frames[ch]} frames)" if ch in self.n_frames else "")
            for ch, f in sorted(self.fractions.items()))
        tail = "".join(f"\n    ! {n}" for n in self.notes)
        return f"{head}\n{body}{tail}"


def _frame_months(d) -> np.ndarray:
    """UTC ``YYYY-MM`` label per frame, from the product's unit timestamps."""
    months = np.array([
        dt.datetime.fromtimestamp(float(t), dt.timezone.utc).strftime("%Y-%m")
        for t in np.asarray(d["unit_time0_ctime"], dtype=float)])
    return months[np.asarray(d["frame_unit_index"])]


def mask_table_from_products(paths, refused_channels=REFUSED_CHANNELS,
                             refused_fraction: float = REFUSED_FRACTION,
                             require_same_detector: bool = True,
                             stage: str = "coarse",
                             since: str | None = None,
                             until: str | None = None,
                             eta: float = 1.0) -> MaskTable:
    """Masked fractions straight from the survey products.

    ``since``/``until`` restrict the fractions to frames in UTC months
    ``since <= YYYY-MM <= until``. A masked fraction is an epoch statement as
    much as a rule statement: a channel whose transmitter signed off carries
    its dead epoch in the full-span fraction, and a forward-looking forecast
    wants the fraction from the epoch that still describes the sky (e.g. the
    trailing year). The window travels in ``MaskTable.window`` next to the
    rule, and a channel with no valid frames inside the window is omitted
    (with a note) rather than reported from the wrong epoch. Products that
    carry no unit timestamps cannot be windowed and are refused when a window
    is requested.

    ``eta`` recomputes the coarse fractions at the threshold
    ``F > eta * mu0`` from the product's stored statistic instead of
    reporting the shipped ``reject_mask`` (the deployed ``eta = 1``
    zero-excess rule). The deployed rule fires on faint residual signal even
    where a transmitter has signed off, so an epoch window alone cannot
    lower those channels' fractions; the threshold can. Requires
    ``fstat_raw`` and ``mu0`` in every product and ``stage='coarse'``; the
    recomputation is recorded in the table's rule so an ``eta != 1`` table
    can never be mistaken for the detector's own decision.

    ``stage`` picks which decision to report, and the returned table names it,
    because the pipeline makes more than one and they differ by two orders of
    magnitude on the same channel:

    * ``'coarse'``: the product's ``reject_mask``, i.e. the positive-excess
      rule ``F > mu0`` with no threshold above it. This is what the kernel
      applies per frame.
    * ``'fine'``: the deployed fine rank-CFAR, ``fine_detected_count > 0``
      at the product's own ``fine_p_fa``.

    Neither reproduces the vendored quarterly ``hi_rate_all`` table the
    forecast is fiducially built on, so a table from here and a table from the
    CSV are not interchangeable and must not be merged. Naming the stage is
    the point: a bare ``{channel: fraction}`` dict cannot be checked against
    the detector it claims to describe.

    ``require_same_detector`` refuses a mixed set. The hard gate is the
    *kernel* (``kernel_sha256`` plus the mask rule) because that is what
    decides each frame; two products that disagree there cannot be averaged
    into one table. A differing harness *package* version over an identical
    kernel is recorded as a note instead of an error: it usually means one
    product predates a release, which is worth knowing but does not change
    what F was computed to be.
    """
    eta = float(eta)
    if eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    if eta != 1.0 and stage != "coarse":
        raise ValueError(
            "eta rethresholds the coarse statistic F; it does not apply to "
            f"stage={stage!r}")
    windowed = since is not None or until is not None
    window = ("full span" if not windowed
              else f"{since or 'start'}..{until or 'end'}")
    fractions, n_frames, rules = {}, {}, set()
    kernels, packages, seen = set(), set(), {}
    window_notes = []
    for p in paths:
        d = load_npz(p)
        ch = int(d["physical_channel"][0])
        valid = d["valid"][:, 0].astype(bool)
        if windowed:
            if "unit_time0_ctime" not in d or "frame_unit_index" not in d:
                raise ValueError(
                    f"{Path(p).name} carries no unit timestamps, so its "
                    f"fractions cannot be windowed; drop since/until or "
                    f"regenerate the product with time provenance")
            month = _frame_months(d)
            if since is not None:
                valid = valid & (month >= since)
            if until is not None:
                valid = valid & (month <= until)
        if eta != 1.0:
            if "fstat_raw" not in d or "mu0" not in d:
                raise ValueError(
                    f"{Path(p).name} carries no fstat_raw/mu0, so its "
                    f"fractions cannot be rethresholded; drop eta or "
                    f"regenerate the product with the statistic")
            rejected = (d["fstat_raw"][:, 0]
                        > eta * float(np.asarray(d["mu0"]).ravel()[0]))
        elif stage == "coarse":
            rejected = d["reject_mask"][:, 0].astype(bool)
        elif stage == "fine":
            if "fine_detected_count" not in d:
                raise ValueError(f"{Path(p).name} carries no fine-stage "
                                 f"decision; use stage='coarse'")
            rejected = d["fine_detected_count"][:, 0] > 0
        else:
            raise ValueError(f"unknown stage {stage!r}; use 'coarse' or 'fine'")
        if valid.sum() == 0:
            if windowed:
                window_notes.append(
                    f"ch{ch} has no valid frames in {window}; omitted")
            continue
        if ch in fractions:
            raise ValueError(
                f"two products cover ch{ch} ({seen[ch].name} and "
                f"{Path(p).name}); one would silently overwrite the other; "
                f"pass exactly one product per channel")
        seen[ch] = Path(p)
        fractions[ch] = float(rejected[valid].mean())
        n_frames[ch] = int(valid.sum())
        if eta != 1.0:
            rules.add(f"F > {eta:g}*mu0 (rethresholded from fstat_raw; "
                      f"deployed decision is F > mu0)")
        elif stage == "fine":
            pfa = float(d["fine_p_fa"]) if "fine_p_fa" in d else float("nan")
            rules.add(f"fine rank-CFAR detection (p_fa={pfa:g})")
        else:
            try:
                contract = json.loads(str(d["detector_contract_json"]))
                rules.add(str(contract.get("equivalent_mask_rule",
                                           contract.get("mask_rule",
                                                        "unrecorded"))))
            except Exception:
                rules.add(str(d["mask_rule"]) if "mask_rule" in d
                          else "unrecorded")
        dv = str(d["detector_version"]) if "detector_version" in d else ""
        packages.add(dv.split(" ")[0] or "unrecorded")
        kernels.add(next((tok.split("=", 1)[1] for tok in dv.split()
                          if tok.startswith("kernel_sha256=")), "unrecorded"))

    if not fractions:
        raise ValueError(
            "no product yielded any valid frames"
            + (f" in {window}" if windowed else ""))
    if require_same_detector and len(rules) > 1:
        raise ValueError(f"products disagree on the mask rule: {sorted(rules)}; "
                         f"combining them would average two detectors")
    if require_same_detector and len(kernels) > 1:
        raise ValueError(
            f"products span detector kernels {sorted(k[:12] for k in kernels)}; "
            f"the frames were decided by different code; re-run them under "
            f"one kernel, or pass require_same_detector=False to override")

    notes = list(window_notes)
    if len(packages) > 1:
        notes.append(f"products span harness versions {sorted(packages)} over "
                     f"one kernel ({sorted(kernels)[0][:12]}); F is comparable, "
                     f"but the older products predate a release")
    for ch in refused_channels:
        if ch not in fractions:
            fractions[ch] = refused_fraction
            notes.append(f"ch{ch} refused by pilot-proxy; assumed "
                         f"{refused_fraction:g} masked rather than measured")
    src = (f"products[coarse@eta={eta:g}]" if eta != 1.0
           else f"products[{stage}]")
    return MaskTable(fractions=dict(sorted(fractions.items())),
                     source=src, rule=sorted(rules)[0],
                     detector_version="+".join(sorted(packages))
                     + f" kernel={sorted(kernels)[0][:12]}",
                     n_frames=n_frames, notes=notes, window=window)


def measured_mask_table(rates_csv: str | Path = DEFAULT_RATES_CSV,
                        refused_fraction: float = REFUSED_FRACTION,
                        rate_col: str = "hi_rate_all") -> MaskTable:
    """The vendored quarterly CSV, wrapped so its missing provenance shows."""
    fr = measured_mask_fractions(rates_csv, refused_fraction, rate_col)
    return MaskTable(
        fractions=fr, source="csv", rule="unrecorded",
        notes=[f"column {rate_col!r} in {_source_name(rates_csv)} records no "
               f"statistic or threshold; the detector behind it is unverified"])


def merge_mask_tables(*tables: MaskTable) -> MaskTable:
    """Refuse to merge tables from different detectors. There is no valid
    merge: a forecast half from one stage and half from another is not a
    forecast of anything."""
    rules = {t.rule for t in tables}
    if len(rules) > 1:
        raise ValueError(f"cannot merge tables built under different rules: "
                         f"{sorted(rules)}")
    out, n = {}, {}
    for t in tables:
        out.update(t.fractions)
        n.update(t.n_frames)
    return MaskTable(fractions=dict(sorted(out.items())),
                     source="+".join(sorted({t.source for t in tables})),
                     rule=rules.pop(), n_frames=n,
                     notes=[m for t in tables for m in t.notes])


def compare_mask_tables(a: MaskTable, b: MaskTable) -> list[tuple]:
    """(channel, a, b, ratio) for channels both tables cover, worst first."""
    out = []
    for ch in sorted(set(a.fractions) & set(b.fractions)):
        x, y = a.fractions[ch], b.fractions[ch]
        lo, hi = min(x, y), max(x, y)
        out.append((ch, x, y, (hi / lo) if lo > 0 else float("inf")))
    return sorted(out, key=lambda r: -r[3])


def dtv_band(fractions: dict[int, float]) -> tuple[float, float]:
    """Frequency range [MHz] spanned by the channels in a mask table."""
    lo = min(channel_edges(ch)[0] for ch in fractions)
    hi = max(channel_edges(ch)[1] for ch in fractions)
    return lo, hi

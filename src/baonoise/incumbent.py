"""What the flaggers already in the pipeline do to a DTV shelf.

The detector has to be worth deploying over what CHIME already runs, so the
incumbents need to be measured rather than argued about. Two families cover
the ground:

MAD outlier rejection
    Integrated power per block, cut at ``median + k * MAD``, which is the
    style of the flagging described in CHIME's overview paper.
spectral kurtosis
    The Nita & Gary statistic over ``M`` accumulated power estimates, standard
    in single-dish and VLBI RFI excision.

Both estimate their own reference from the data. That is the property that
decides this comparison: a transmitter with a duty cycle above ~50% moves the
median and the kurtosis together with the data, so the shelf is absorbed into
the baseline and read as sky. The pilot matched filter has no baseline to
corrupt (the tone is at a known offset from the band edge whether or not it
was there a second ago), so it is indifferent to duty cycle. The measurements
below are that argument in numbers.

Blocks never cross an acquisition boundary. The trawl archive is ~8500
baseband snapshots with a median length of 3 frames, spaced about three hours
apart over 7.6 years; a block built across that gap measures the archive's
sampling pattern rather than the sky. The first version of this comparison did cross
those boundaries and returned spectral kurtosis values of 11-100 against an
expectation of 1. The consequence is a real limit on what this dataset can
settle: the longest snapshot is 33 frames, so CHIME's few-second flagging
cadence cannot be reproduced here at all. That needs the contiguous scan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .npzio import load_npz

FRAME_SECONDS = 16384 * 2.56e-6           # 41.94304 ms, upgraded backend
MAD_TO_SIGMA = 1.4826
DEFAULT_MAD_K = 1.8                       # CHIME overview paper, sec 3.2.3


# ----------------------------------------------------------------------
# The common measurement every flagger is scored against
# ----------------------------------------------------------------------

def shelf_per_frame(d) -> tuple[np.ndarray, float]:
    """(shelf power / system noise per frame, sensitivity floor in dB).

    Frames with no pilot detection have no shelf measurement. They are not
    known to be clean (they are frames where the shelf sits at or below the
    floor), so they are assigned the floor rather than zero. That choice is
    conservative in the direction that matters: it raises the residual left
    behind by *every* flagger equally, including the pilot proxy's own, so it
    can only understate the detector's advantage.
    """
    shelf_db = d["snr_shelf_db"][:, 0]
    valid = d["valid"][:, 0].astype(bool)
    rejected = d["reject_mask"][:, 0].astype(bool)
    seen = valid & rejected & np.isfinite(shelf_db)
    clean = valid & ~rejected & np.isfinite(shelf_db)
    if clean.sum():
        floor_db = float(np.percentile(shelf_db[clean], 90.0))
    elif seen.sum():
        floor_db = float(np.nanmin(shelf_db[seen]))
    else:
        raise ValueError("no finite shelf measurements in this product")
    lin = np.full(shelf_db.shape, 10.0 ** (floor_db / 10.0))
    lin[seen] = 10.0 ** (shelf_db[seen] / 10.0)
    return lin, floor_db


def acquisition_blocks(unit: np.ndarray, min_frames: int = 2):
    """Contiguous frame runs within one acquisition, as (start, stop) pairs."""
    if len(unit) == 0:
        return []
    edges = np.flatnonzero(np.diff(unit)) + 1
    return [(int(a), int(b))
            for a, b in zip(np.r_[0, edges], np.r_[edges, len(unit)])
            if b - a >= min_frames]


# ----------------------------------------------------------------------
# The flaggers
# ----------------------------------------------------------------------

def mad_flag(power, unit, k: float = DEFAULT_MAD_K, min_frames: int = 4):
    """Per-frame flag: power above ``median + k*MAD`` of its own acquisition."""
    out = np.zeros(len(power), bool)
    for a, b in acquisition_blocks(unit, min_frames):
        p = power[a:b]
        med = float(np.median(p))
        mad = float(np.median(np.abs(p - med))) * MAD_TO_SIGMA
        if mad > 0:
            out[a:b] = p > med + k * mad
    return out


def spectral_kurtosis(p: np.ndarray, n_accum: float) -> float:
    """Nita & Gary generalised SK for one block of accumulated powers."""
    M = p.size
    if M < 2:
        return 1.0
    s1 = float(p.sum())
    if s1 <= 0:
        return 1.0
    s2 = float((p.astype(float) ** 2).sum())
    return ((M * n_accum + 1) / (M - 1)) * (M * s2 / s1 ** 2 - 1)


def sk_sigma(M: int, n_accum: float) -> float:
    """Standard deviation of SK under the Gaussian null, mean 1."""
    nm = n_accum * M
    return float(np.sqrt((2 * nm * (nm + 1)) / ((M - 1) * (nm + 2) * (nm + 3))))


def calibrate_sk_null(power, unit, min_frames: int = 8) -> float:
    """Effective accumulation length implied by the data's own scatter.

    The analytic ``n_accum`` for this product is not recoverable from it: the
    power accumulates over 2048 input streams whose correlation depends on
    whether the source is the sky or a distant transmitter, and the product
    does not record which normalization was applied. Rather than assume one,
    the null is calibrated so the *median* block sits at SK = 1. That makes SK
    a fair detector by construction: it is handed the best possible
    normalization, tuned on this very data, which the pilot proxy is not.
    """
    blocks = acquisition_blocks(unit, min_frames)
    if not blocks:
        raise ValueError("no acquisition long enough to calibrate SK")
    # SK is linear in n_accum once the +1 and +2/+3 corrections are dropped,
    # so solve for the value that puts the median raw ratio at 1.
    ratios = []
    for a, b in blocks:
        p = power[a:b].astype(float)
        M = p.size
        s1 = float(p.sum())
        if s1 <= 0:
            continue
        s2 = float((p ** 2).sum())
        ratios.append((M / (M - 1)) * (M * s2 / s1 ** 2 - 1))
    if not ratios:
        raise ValueError("no usable acquisitions for SK calibration")
    med = float(np.median(ratios))
    if med <= 0:
        raise ValueError("degenerate SK null: non-positive median ratio")
    return 1.0 / med


def sk_flag(power, unit, n_accum: float | None = None, nsigma: float = 3.0,
            min_frames: int = 8):
    """Per-frame flag: acquisitions whose SK departs from the null.

    Returns ``(flag, sk_values, n_accum)``. ``n_accum=None`` calibrates the
    null on this channel; see :func:`calibrate_sk_null`.
    """
    if n_accum is None:
        n_accum = calibrate_sk_null(power, unit, min_frames)
    out = np.zeros(len(power), bool)
    sks = []
    for a, b in acquisition_blocks(unit, min_frames):
        p = power[a:b].astype(float)
        sk = spectral_kurtosis(p, n_accum)
        sks.append(sk)
        if abs(sk - 1.0) > nsigma * sk_sigma(p.size, n_accum):
            out[a:b] = True
    return out, np.array(sks), float(n_accum)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

@dataclass
class FlaggerResult:
    name: str
    f: float
    r: float
    n_kept: int
    shelf_kept_db: float
    reduction_db: float


def score_flagger(name, flag, lin, base) -> FlaggerResult:
    """Masked fraction and surviving shelf for one flagger, over ``base``."""
    keep = base & ~flag
    f = 1.0 - keep.sum() / base.sum()
    r = float(lin[keep].mean()) if keep.sum() else float("inf")
    r_all = float(lin[base].mean())
    db = 10 * np.log10(r) if np.isfinite(r) and r > 0 else float("-inf")
    return FlaggerResult(name=name, f=float(f), r=r, n_kept=int(keep.sum()),
                         shelf_kept_db=float(db),
                         reduction_db=float(10 * np.log10(r_all / r))
                         if np.isfinite(r) and r > 0 else float("inf"))


def duty_cycle(d) -> float:
    """Fraction of valid frames carrying a positive pilot excess.

    Above 0.5 a self-calibrating flagger's own reference is contaminated, so
    the shelf is measured as sky. This single number predicts most of what the
    incumbent comparison finds.
    """
    valid = d["valid"][:, 0].astype(bool)
    return float(d["reject_mask"][valid, 0].astype(bool).mean())


def compare_flaggers(npz_path, mad_k: float = DEFAULT_MAD_K,
                     sk_nsigma: float = 3.0, min_frames: int = 8):
    """Every flagger on the same frames of one product.

    Restricted to acquisitions long enough for a block statistic to exist, so
    the incumbents are scored on ground they can actually stand on. Returns
    ``(results, meta)``.
    """
    d = load_npz(npz_path)
    lin, floor_db = shelf_per_frame(d)
    valid = d["valid"][:, 0].astype(bool)
    power = d["baseband_power_linear"][:, 0].astype(float)
    unit = d["frame_unit_index"]
    proxy = d["reject_mask"][:, 0].astype(bool)

    blocks = acquisition_blocks(unit, min_frames)
    inblock = np.zeros(len(unit), bool)
    for a, b in blocks:
        inblock[a:b] = True
    base = valid & inblock
    if base.sum() == 0:
        raise ValueError(f"no acquisition reaches {min_frames} frames")

    sk, sk_values, n_accum = sk_flag(power, unit, nsigma=sk_nsigma,
                                     min_frames=min_frames)
    results = [
        score_flagger("keep everything", np.zeros(len(unit), bool), lin, base),
        score_flagger(f"MAD {mad_k}x within acquisition",
                      mad_flag(power, unit, k=mad_k), lin, base),
        score_flagger(f"SK {sk_nsigma:g}sigma within acquisition", sk, lin, base),
        score_flagger("pilot proxy", proxy, lin, base),
    ]
    sizes = np.array([b - a for a, b in blocks])
    meta = dict(
        channel=int(d["physical_channel"][0]),
        freq_id=int(d["freq_id"][0]),
        nu_mhz=float(d["chime_frequency_hz"][0]) / 1e6,
        floor_db=floor_db,
        duty_cycle=duty_cycle(d),
        n_frames=int(valid.sum()),
        n_scored=int(base.sum()),
        n_blocks=len(blocks),
        block_med_frames=float(np.median(sizes)) if len(sizes) else 0.0,
        block_max_frames=int(sizes.max()) if len(sizes) else 0,
        block_max_seconds=float(sizes.max() * FRAME_SECONDS) if len(sizes) else 0.0,
        sk_n_accum=n_accum,
        sk_median=float(np.median(sk_values)) if sk_values.size else float("nan"),
    )
    return results, meta

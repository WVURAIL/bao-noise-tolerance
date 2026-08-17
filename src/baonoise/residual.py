"""Residual contamination: what the mask *leaves behind*, and what it costs.

Why this module exists
----------------------
:mod:`scenarios` prices the cost of the data a mask removes. That is only half
of the tolerance question, and on its own it is monotone: masking less is
always cheaper, so a forecast built from :mod:`scenarios` alone is minimised by
never masking at all. The missing term is the cost of the contamination that
survives the mask. A detector threshold sets *both*: lower threshold, more
data removed but a cleaner remainder; higher threshold, less data removed but a
dirtier one. Only with both terms is the threshold an optimisation variable
rather than an input.

The residual chain
------------------
The pilot-proxy survey products report, per frame, ``snr_shelf_db``: the DTV
shelf power relative to system noise in the DTV bandwidth, inferred from the
pilot via the frozen proxy relation

    snr_shelf_db = pnr_bin_db + pilot_below_data_db - 10 log10(B_DTV / bin_enbw)

(a fixed -21.636 dB offset for the deployed geometry). Four terms take that to
the residual an analysis actually sees:

1. **Transmitter-on shelf.** What is there when nothing is done. Measured.
2. **Masking.** A *kept* frame is one with no pilot excess, so its shelf is not
   measured; it is bounded by the single-frame sensitivity floor. That floor
   is measurable: it is the ``snr_shelf_db`` distribution over a
   transmitter-off epoch, where the reported value is pure noise. Use an upper
   percentile of that distribution rather than the median, as the bound.
3. **Ground / common-mode filter.** A terrestrial transmitter is stationary in
   the telescope frame, so its residual is overwhelmingly m = 0 and is removed
   by the standard ground filter. The surviving fraction is *measurable* from
   the same product; see :func:`shelf_statistics`, which splits the shelf
   power into constant, slow-variable and fast-variable parts.
4. **Delay filter.** Foreground delay filtering suppresses the shelf by an
   amount that depends on the delay cutoff, and therefore on which BAO scale
   is being protected. This is the term that is *not* free: the suppression at
   today's tau_cut = 200 ns is not the suppression at the first BAO peak.

Integration scaling
-------------------
Thermal noise averages down; a residual only averages down over its own
correlation time. If a surviving component decorrelates on timescale tau_c it
is amplified relative to thermal noise by ``n_coh = tau_c / T_frame``, the
number of frames per correlation time, and that amplification is
*independent of total integration time*, so the residual-to-noise ratio
saturates rather than growing without bound.

tau_c is bounded above rather than free. Anything correlated for longer than a
sidereal day is m = 0 within each day and is already removed by term 3, so
capping tau_c at :data:`MAX_TAU_C_SECONDS` is what keeps the ground filter and
the coherence factor from pricing the same power twice. Within that cap the
two surviving populations are handled separately, because they are physically
different: intra-day variation carries a long coherence, sub-acquisition
variation averages like noise, and one lumped n_coh would apply the former's
amplification to the latter's power.

What remains genuinely unmeasured is narrow: where inside [acquisition
duration, one sidereal day] the intra-day component decorrelates. The default
is the pessimistic end of that window. Nothing here guesses.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .npzio import load_npz

# Delay-filter suppression of the DTV shelf, dB, as a function of which BAO
# feature the filter must preserve. Derived from the k_par <-> delay relation
# k_par = 2 pi nu_21 H0 E(z) tau / [c (1 + z)^2] (Amiri et al. 2023, Eq. 64)
# evaluated at the DTV band. Preserving a *larger* scale (lower k_par, the
# first acoustic peak) forces a shorter delay cutoff and buys less suppression.
DELAY_SUPPRESSION_DB = {
    "none": 0.0,                # claim no foreground-removal help (default)
    "aggressive_200ns": 11.4,   # CHIME's deployed tau_cut = 200 ns
    "bao_peak2": 8.2,           # second acoustic peak, tau ~ 104 ns
    "bao_peak1": 3.6,           # first acoustic peak, tau ~ 56 ns
}
# Default to claiming nothing. The delay filter is a foreground-removal stage,
# and the forecast this feeds follows the Overview paper's Figure 31
# convention: thermal noise from Tsys and nothing else: no foreground
# residual, no wedge, BAO-shift only. Crediting the chain with a filter the
# forecast does not model would take suppression from a stage that is not in
# the budget. The non-zero entries stay available as an explicit opt-in and
# bound the help that *could* be claimed if the filter were modelled on both
# sides. Note the direction: claiming nothing makes every residual larger,
# which is the conservative direction for a contamination budget.
DEFAULT_DELAY_KEY = "none"

CHIME_FRAME_SECONDS = 16384 * 2.56e-6   # 41.94304 ms
SIDEREAL_DAY = 86164.0905               # s

# Hard cap on the residual correlation time. Anything correlated for longer
# than a sidereal day is m = 0 within each day and is removed by the
# common-mode filter, so it never reaches the coherence term; it has already
# been priced as ground-filter suppression. Capping here is what keeps the two
# terms from double-counting the same power.
MAX_TAU_C_SECONDS = SIDEREAL_DAY


# ----------------------------------------------------------------------
# Survey-product measurements
# ----------------------------------------------------------------------

@dataclass
class ShelfStatistics:
    """What a pilot-proxy survey product says about one channel's shelf.

    All powers are linear ratios to system noise in the DTV bandwidth; the
    ``_db`` fields are 10 log10 of the corresponding ratio.
    """
    channel: int
    freq_id: int
    nu_mhz: float
    n_valid: int
    n_kept: int
    on_shelf_db: float            # median shelf, transmitter on
    floor_db: float               # sensitivity floor -> bound on kept frames
    floor_percentile: float
    dc_fraction: float            # time-constant share of shelf power
    interday_fraction: float      # varies between sidereal days
    intraday_fraction: float      # varies between acquisitions within a day
    fast_fraction: float          # varies within an acquisition
    n_off_frames: int = 0
    n_units: int = 0
    n_days: int = 0
    trim_percentile: float | None = None

    @property
    def masked_fraction(self) -> float:
        return 1.0 - self.n_kept / self.n_valid if self.n_valid else 1.0

    @property
    def slow_fraction(self) -> float:
        """All variation slower than an acquisition."""
        return self.interday_fraction + self.intraday_fraction

    @property
    def surviving_fraction(self) -> float:
        """Share of shelf power the ground filter cannot remove.

        A residual constant *within* a sidereal day is m = 0 no matter how it
        drifts from one day to the next, because the common-mode filter is
        applied per day. So DC and inter-day both go; only variation on
        timescales shorter than a day lands at m != 0.
        """
        return self.intraday_fraction + self.fast_fraction

    @property
    def ground_filter_db(self) -> float:
        """Suppression from removing the m = 0 component."""
        var = self.surviving_fraction
        if var <= 0.0:
            return np.inf
        return float(10.0 * np.log10(1.0 / var))

    def summary(self) -> str:
        return (
            f"ch{self.channel:>3d} ({self.nu_mhz:7.2f} MHz, freq_id {self.freq_id})  "
            f"masked {100 * self.masked_fraction:5.2f}%\n"
            f"    shelf on-air      {self.on_shelf_db:7.2f} dB\n"
            f"    kept-frame bound  {self.floor_db:7.2f} dB "
            f"(p{self.floor_percentile:g} of {self.n_off_frames} null frames)\n"
            f"    power split       DC       {100 * self.dc_fraction:7.4f}%\n"
            f"                      inter-day{100 * self.interday_fraction:7.4f}%  "
            f"(m=0, removed)\n"
            f"                      intra-day{100 * self.intraday_fraction:7.4f}%  "
            f"(survives)\n"
            f"                      fast     {100 * self.fast_fraction:7.4f}%  "
            f"(averages as noise)\n"
            f"    ground filter     {self.ground_filter_db:7.2f} dB "
            f"({self.n_units} acquisitions over {self.n_days} sidereal days)"
        )


def _month_of(ctime: np.ndarray) -> np.ndarray:
    return np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc).strftime("%Y-%m")
                     for x in ctime])


def _on_epoch(d, off_through: str | None, trim_percentile: float | None):
    """Transmitter-on frames, optionally with the burst tail trimmed.

    Gating on ``reject_mask`` is what makes this a transmitter-on selection
    without needing an off epoch: a rejected frame carries a positive pilot
    excess, so the shelf is demonstrably there. ``off_through`` narrows it
    further when a clean off epoch is known.

    Trimming matters for intermittent transmitters. On a channel whose shelf is
    a quiet baseline punctuated by strong bursts, the linear-power moments are
    entirely the bursts (on ch34 the top 1% of frames carry 99.5% of the
    variance), and a variance decomposition of that is meaningless. Trimming
    restricts to the regime the residual budget is actually about, since the
    bursts are exactly what the detector flags and removes.
    """
    valid = d["valid"][:, 0].astype(bool)
    rejected = d["reject_mask"][:, 0].astype(bool)
    shelf = d["snr_shelf_db"][:, 0]
    unit = d["frame_unit_index"]
    t0 = d["unit_time0_ctime"]

    on = valid & rejected & np.isfinite(shelf)
    if off_through is not None:
        on &= _month_of(t0)[unit] > off_through
    if on.sum() and trim_percentile is not None:
        on &= shelf <= np.percentile(shelf[on], trim_percentile)
    return on, shelf, unit, t0


def _nested_split(lin, unit, t0):
    """(dc, interday, intraday, fast, n_units, n_days, v_fast_abs, groups).

    ``groups`` is (unit_mean, n_frames, t_start, sidereal_day) per acquisition,
    which the structure-function estimator reuses.
    """
    sday = np.floor(t0[unit] / SIDEREAL_DAY).astype(np.int64)
    key = sday * (int(unit.max()) + 1) + unit
    order = np.argsort(key, kind="stable")
    lin_s, key_s, day_s, unit_s = lin[order], key[order], sday[order], unit[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    parts = list(zip(np.split(lin_s, bounds), np.split(day_s, bounds),
                     np.split(unit_s, bounds)))
    groups = [(g, dd[0], uu[0]) for g, dd, uu in parts if g.size >= 2]
    if not groups:
        return (np.nan,) * 4 + (0, 0, np.nan, None)

    within = np.concatenate([g - g.mean() for g, _, _ in groups])
    unit_means = np.array([g.mean() for g, _, _ in groups])
    n_frames = np.array([g.size for g, _, _ in groups])
    unit_days = np.array([dd for _, dd, _ in groups])
    unit_t = np.array([t0[uu] for _, _, uu in groups])

    day_means, intra = [], []
    for dd in np.unique(unit_days):
        m = unit_means[unit_days == dd]
        day_means.append(m.mean())
        if m.size >= 2:
            intra.append(m - m.mean())
    day_means = np.array(day_means)
    intra = np.concatenate(intra) if intra else np.zeros(1)

    grand = float(day_means.mean())
    v_day, v_intra, v_fast = day_means.var(), intra.var(), within.var()
    total = grand ** 2 + v_day + v_intra + v_fast
    if total <= 0:
        return (np.nan,) * 4 + (0, 0, np.nan, None)
    return (float(grand ** 2 / total), float(v_day / total),
            float(v_intra / total), float(v_fast / total),
            len(groups), int(np.unique(unit_days).size), float(v_fast),
            (unit_means, n_frames, unit_t, unit_days, grand))


def shelf_statistics(npz_path: str | Path, off_through: str | None = None,
                     floor_percentile: float = 90.0,
                     trim_percentile: float | None = 90.0) -> ShelfStatistics:
    """Measure the residual chain's data-driven terms from a survey product.

    ``off_through`` is the last ``YYYY-MM`` of a transmitter-off epoch for this
    channel; frames at or before it define the sensitivity floor. Without one
    the floor is taken from frames the detector *kept*, which is a weaker bound
    (it is the floor only where the pilot estimate stayed positive).

    The power split is a three-level nested variance decomposition of the
    linear shelf over the transmitter-on epoch, keyed on sidereal day and
    acquisition unit:

        DC        = (grand mean)^2
        inter-day = variance of the per-sidereal-day means
        intra-day = variance of unit means about their own day's mean
        fast      = variance of frames about their own unit's mean

    The sidereal-day boundary is the one that matters. A residual constant
    within a day is m = 0 however much it drifts day to day, so DC and
    inter-day are both removed by the common-mode filter; only intra-day and
    faster variation reaches m != 0. Splitting at the acquisition instead --
    which lumps day-to-day drift in with intra-day variation, understates the
    ground filter and double-counts that power in the coherence term.
    """
    d = load_npz(npz_path)
    valid = d["valid"][:, 0].astype(bool)
    rejected = d["reject_mask"][:, 0].astype(bool)
    shelf = d["snr_shelf_db"][:, 0]
    unit = d["frame_unit_index"]
    t0 = d["unit_time0_ctime"]
    month = _month_of(t0)[unit]

    if off_through is not None:
        off = valid & (month <= off_through)
    else:
        off = valid & ~rejected

    finite_off = off & np.isfinite(shelf)
    if finite_off.sum() == 0:
        floor_db = float("nan")
    else:
        floor_db = float(np.percentile(shelf[finite_off], floor_percentile))

    on, _, _, _ = _on_epoch(d, off_through, trim_percentile)
    on_db = float(np.median(shelf[on])) if on.sum() else float("nan")

    dc = interday = intraday = fast = float("nan")
    n_units = n_days = 0
    if on.sum() > 10:
        dc, interday, intraday, fast, n_units, n_days, _, _ = _nested_split(
            10.0 ** (shelf[on] / 10.0), unit[on], t0)

    return ShelfStatistics(
        channel=int(d["physical_channel"][0]),
        freq_id=int(d["freq_id"][0]),
        nu_mhz=float(d["chime_frequency_hz"][0]) / 1e6,
        n_valid=int(valid.sum()),
        n_kept=int((valid & ~rejected).sum()),
        on_shelf_db=on_db,
        floor_db=floor_db,
        floor_percentile=float(floor_percentile),
        trim_percentile=trim_percentile,
        dc_fraction=dc, interday_fraction=interday,
        intraday_fraction=intraday, fast_fraction=fast,
        n_off_frames=int(finite_off.sum()),
        n_units=n_units, n_days=n_days,
    )


# ----------------------------------------------------------------------
# The intra-day correlation time
# ----------------------------------------------------------------------

# Lag bins for the same-day structure function, seconds. The short end is set
# by the acquisition cadence (pairs closer than ~5 min are rare); the long end
# by the sidereal-day boundary, past which inter-day drift contaminates.
STRUCTURE_LAG_EDGES = np.array(
    [0.0, 300.0, 900.0, 1800.0, 2700.0, 3600.0, 5400.0, 7200.0, 14400.0, 28800.0])
PLATEAU_LAG_SECONDS = 7200.0     # lags beyond this estimate the full variance
TRIM_PROBES = (75.0, 90.0, 95.0)


@dataclass
class CorrelationTime:
    """Intra-day correlation time of the shelf, or a refusal with a reason.

    ``quality`` is one of:

    * ``'measured'``: every gate passed; ``tau_c`` is a measurement with
      a bootstrap interval.
    * ``'bounded_above'``: the shelf is stationary but already decorrelated
      at the shortest lag the acquisition cadence resolves. ``tau_c`` is that
      lag, and it is an *upper* bound, so the budget built from it is a bound
      in the favorable direction rather than a measurement.
    * ``'refused'``: the shelf is episodic and admits no stationary
      timescale at all; ``tau_c`` is nan.

    Only the last is a refusal, and it is not a failure of the channel; it is
    the estimator declining to put a number on a process that does not admit
    one. See :func:`correlation_time`.
    """
    channel: int
    tau_c: float                 # s; nan when refused
    tau_lo: float                # 16th percentile, day-block bootstrap
    tau_hi: float                # 84th percentile
    plateau_fraction: float      # intra-day variance, share of DC power
    n_days: int
    n_pairs: int
    trim_spread: float           # max/min tau_c across TRIM_PROBES
    surviving_spread: float      # max/min (intra + fast) across TRIM_PROBES
    quality: str                 # 'measured' | 'refused'
    reason: str = ""

    @property
    def is_measured(self) -> bool:
        return self.quality == "measured"

    @property
    def is_usable(self) -> bool:
        """Measured or bounded above; either way the budget can use it."""
        return self.quality in ("measured", "bounded_above")

    @property
    def tau_for_budget(self) -> float:
        """The measured value or upper bound; the cap only when refused."""
        return self.tau_c if self.is_usable else MAX_TAU_C_SECONDS

    def summary(self) -> str:
        if self.quality == "refused":
            return (f"ch{self.channel:>3d}  tau_c REFUSED: {self.reason}\n"
                    f"    falling back to the {MAX_TAU_C_SECONDS / 3600:.1f} h cap "
                    f"(bound rather than measurement)")
        if self.quality == "bounded_above":
            return (
                f"ch{self.channel:>3d}  tau_c <= {self.tau_c / 60:.0f} min "
                f"(upper bound): {self.reason}\n"
                f"    plateau {100 * self.plateau_fraction:.4f}% of DC power, "
                f"{self.n_pairs} same-day pairs over {self.n_days} sidereal days\n"
                f"    stability across trim: tau x{self.trim_spread:.2f}, "
                f"surviving x{self.surviving_spread:.2f}")
        return (
            f"ch{self.channel:>3d}  tau_c = {self.tau_c / 60:.0f} min "
            f"[{self.tau_lo / 60:.0f}-{self.tau_hi / 60:.0f} at 68%]\n"
            f"    plateau {100 * self.plateau_fraction:.4f}% of DC power, "
            f"{self.n_pairs} same-day pairs over {self.n_days} sidereal days\n"
            f"    stability across trim: tau x{self.trim_spread:.2f}, "
            f"surviving x{self.surviving_spread:.2f}")


def _same_day_structure(groups, v_fast_abs, edges=STRUCTURE_LAG_EDGES,
                        days_subset=None):
    """Noise-corrected structure function D(dt) over same-sidereal-day pairs.

    D(dt) = <[x(t+dt) - x(t)]^2> / 2 rises from 0 to the intra-day variance as
    dt passes the correlation time. Each unit mean carries estimation noise
    V_fast / n_frames, which inflates every squared difference by a known
    amount; subtracting it is what keeps sparsely-sampled acquisitions from
    faking a short correlation time.

    Pairs are restricted to a single sidereal day so inter-day drift (which
    is m = 0 and already priced as ground-filter suppression) cannot leak in.
    """
    unit_means, n_frames, unit_t, unit_days, _ = groups
    lags, sq = [], []
    for dd in (np.unique(unit_days) if days_subset is None else days_subset):
        idx = np.flatnonzero(unit_days == dd)
        if idx.size < 2:
            continue
        tt, xx, nn = unit_t[idx], unit_means[idx], n_frames[idx]
        i, j = np.triu_indices(idx.size, 1)
        lags.append(np.abs(tt[j] - tt[i]))
        sq.append((xx[j] - xx[i]) ** 2 - v_fast_abs * (1.0 / nn[j] + 1.0 / nn[i]))
    if not lags:
        return np.array([]), np.array([]), np.array([]), np.nan, 0
    lags = np.concatenate(lags)
    D = 0.5 * np.concatenate(sq)
    plateau = float(D[lags > PLATEAU_LAG_SECONDS].mean()) \
        if (lags > PLATEAU_LAG_SECONDS).any() else np.nan

    centers, values, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lags >= a) & (lags < b)
        if m.sum() >= 40:
            centers.append(0.5 * (a + b))
            values.append(float(D[m].mean()))
            counts.append(int(m.sum()))
    return (np.array(centers), np.array(values), np.array(counts),
            plateau, int(lags.size))


def _tau_from_structure(centers, values, plateau):
    """Lag at which D reaches (1 - 1/e) of its plateau, linearly interpolated."""
    if not np.isfinite(plateau) or plateau <= 0 or centers.size < 3:
        return np.nan
    target = (1.0 - 1.0 / np.e) * plateau
    if values.max() < target:
        return np.nan
    k = int(np.argmax(values >= target))
    if k == 0:
        return float(centers[0])
    x0, x1, y0, y1 = centers[k - 1], centers[k], values[k - 1], values[k]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def _measure_at_trim(d, off_through, trim):
    on, shelf, unit, t0 = _on_epoch(d, off_through, trim)
    if on.sum() < 100:
        return None
    split = _nested_split(10.0 ** (shelf[on] / 10.0), unit[on], t0)
    if split[-1] is None:
        return None
    _, _, intraday, fast, _, n_days, v_fast_abs, groups = split
    centers, values, _, plateau, n_pairs = _same_day_structure(groups, v_fast_abs)
    grand = groups[4]
    return dict(tau=_tau_from_structure(centers, values, plateau),
                surviving=intraday + fast, n_days=n_days, n_pairs=n_pairs,
                plateau_frac=(plateau / grand ** 2 if grand else np.nan),
                groups=groups, v_fast_abs=v_fast_abs)


def correlation_time(npz_path: str | Path, off_through: str | None = None,
                     trim_percentile: float = 90.0,
                     trim_probes=TRIM_PROBES, max_trim_spread: float = 2.0,
                     min_days: int = 100, min_pairs: int = 200,
                     n_boot: int = 200, seed: int = 20260807) -> CorrelationTime:
    """Measure the intra-day correlation time, or refuse and say why.

    The estimator is a noise-corrected same-sidereal-day structure function of
    the acquisition-mean shelf power, read at the (1 - 1/e) crossing, with a
    day-block bootstrap for the interval. Resampling has to be by whole day
    with each day's time ordering intact; shuffling acquisitions within a day
    destroys exactly the structure being measured and drives tau_c to the
    shortest lag bin.

    Four gates must pass, and the last two are the ones that matter:

    * enough same-day pairs and enough days;
    * a positive plateau the structure function actually reaches, at a lag the
      acquisition cadence can resolve; a crossing in the first bin means
      tau_c is *below* the shortest measurable lag, which is a bound rather than a
      measurement;
    * **tau_c stable across trim level**; and
    * **the surviving power fraction stable across trim level**.

    The stability gates are what separate a stationary shelf from an episodic
    one. On ch35 the answer moves by a factor 1.2 as the trim runs 75-95%; on
    ch34 and ch36, whose transmitters burst rather than sit on, it moves by 10x
    and 15x, because there is no stationary variance to decompose and the
    answer is entirely an artefact of where the tail was cut. Those channels
    get a refusal and the conservative cap rather than a number.
    """
    d = load_npz(npz_path)
    channel = int(d["physical_channel"][0])

    probes = {}
    for tp in trim_probes:
        r = _measure_at_trim(d, off_through, tp)
        if r is not None and np.isfinite(r["tau"]) and r["surviving"] > 0:
            probes[tp] = r

    def refuse(reason, ts=np.nan, ss=np.nan, nd=0, npair=0, pf=np.nan):
        return CorrelationTime(channel, np.nan, np.nan, np.nan, pf, nd, npair,
                               ts, ss, "refused", reason)

    if len(probes) < len(trim_probes):
        return refuse(f"only {len(probes)}/{len(trim_probes)} trim probes "
                      f"yielded a finite estimate")

    taus = np.array([r["tau"] for r in probes.values()])
    survs = np.array([r["surviving"] for r in probes.values()])
    trim_spread = float(taus.max() / taus.min())
    surv_spread = float(survs.max() / survs.min())

    main = _measure_at_trim(d, off_through, trim_percentile)
    if main is None or not np.isfinite(main["tau"]):
        return refuse("no finite estimate at the requested trim",
                      trim_spread, surv_spread)
    if main["n_days"] < min_days:
        return refuse(f"only {main['n_days']} sidereal days with a same-day "
                      f"pair (need {min_days})", trim_spread, surv_spread,
                      main["n_days"], main["n_pairs"], main["plateau_frac"])
    if main["n_pairs"] < min_pairs:
        return refuse(f"only {main['n_pairs']} same-day pairs (need {min_pairs})",
                      trim_spread, surv_spread, main["n_days"], main["n_pairs"],
                      main["plateau_frac"])
    # Stationarity is checked BEFORE resolution, because the two failure modes
    # point opposite ways. An episodic shelf gives a meaningless answer at any
    # lag. A stationary shelf whose crossing lands in the first bin has a
    # *short* correlation time (an upper bound in the favorable direction,
    # worth ~24 dB against the cap) and must not be thrown away as if it were
    # the same kind of failure.
    if trim_spread > max_trim_spread:
        return refuse(f"tau_c moves x{trim_spread:.1f} across trim probes "
                      f"(max {max_trim_spread:g}); the shelf is episodic, "
                      f"not stationary", trim_spread, surv_spread,
                      main["n_days"], main["n_pairs"], main["plateau_frac"])
    if surv_spread > max_trim_spread:
        return refuse(f"surviving power fraction moves x{surv_spread:.1f} "
                      f"across trim probes (max {max_trim_spread:g}); the "
                      f"variance split is set by where the tail was cut",
                      trim_spread, surv_spread, main["n_days"], main["n_pairs"],
                      main["plateau_frac"])
    if main["tau"] <= STRUCTURE_LAG_EDGES[1]:
        # Stationary, and already decorrelated at the shortest lag the
        # acquisition cadence resolves: tau_c <= that lag.
        return CorrelationTime(
            channel=channel, tau_c=float(STRUCTURE_LAG_EDGES[1]),
            tau_lo=np.nan, tau_hi=float(STRUCTURE_LAG_EDGES[1]),
            plateau_fraction=float(main["plateau_frac"]),
            n_days=int(main["n_days"]), n_pairs=int(main["n_pairs"]),
            trim_spread=trim_spread, surviving_spread=surv_spread,
            quality="bounded_above",
            reason=f"decorrelated by {STRUCTURE_LAG_EDGES[1] / 60:.0f} min, the "
                   f"shortest resolvable lag; tau_c is an upper bound")

    groups, v_fast_abs = main["groups"], main["v_fast_abs"]
    days = np.unique(groups[3])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=days.size, replace=True)
        c, v, _, pl, _ = _same_day_structure(groups, v_fast_abs, days_subset=pick)
        t = _tau_from_structure(c, v, pl)
        if np.isfinite(t):
            boots.append(t)
    if len(boots) < max(20, n_boot // 10):
        return refuse("bootstrap did not converge", trim_spread, surv_spread,
                      main["n_days"], main["n_pairs"], main["plateau_frac"])
    boots = np.array(boots)

    return CorrelationTime(
        channel=channel, tau_c=float(main["tau"]),
        tau_lo=float(np.percentile(boots, 16)),
        tau_hi=float(np.percentile(boots, 84)),
        plateau_fraction=float(main["plateau_frac"]),
        n_days=int(main["n_days"]), n_pairs=int(main["n_pairs"]),
        trim_spread=trim_spread, surviving_spread=surv_spread,
        quality="measured")


# ----------------------------------------------------------------------
# The budget
# ----------------------------------------------------------------------

@dataclass
class ResidualBudget:
    """The dB chain from a kept-frame shelf bound to a residual/noise ratio.

    Every term is either measured (``shelf_floor_db``, ``ground_filter_db``)
    or an explicit modelling choice (``delay_filter_db``, ``n_coh``). There
    are no hidden defaults: ``n_coh = 1.0`` asserts the residual averages
    exactly like thermal noise, which is the optimistic end of the bracket.

    ``components`` optionally replaces the single ``(ground_filter_db, n_coh)``
    pair with per-timescale ``(power_fraction, n_coh)`` entries, which is what
    :func:`budget_from_statistics` builds: the surviving power is not one
    population but two (intra-day variation that carries a long coherence,
    and sub-acquisition variation that averages like noise), and lumping them
    applies the slow term's amplification to the fast term's power.
    """
    shelf_floor_db: float
    ground_filter_db: float = 0.0
    delay_filter_db: float = DELAY_SUPPRESSION_DB[DEFAULT_DELAY_KEY]
    n_coh: float = 1.0
    label: str = ""
    components: tuple = ()      # ((power_fraction, n_coh), ...)
    tau_measured: bool = False  # False => tau_c is the cap, i.e. a bound

    def __post_init__(self):
        for frac, n in self.components:
            if frac < 0.0:
                raise ValueError(f"component power fraction is negative: {frac}")
            if n <= 0.0:
                raise ValueError(f"component n_coh must be positive: {n}")

    @property
    def _gain(self) -> float:
        """Total power gain applied to the shelf bound, linear."""
        if self.components:
            return float(sum(frac * n for frac, n in self.components))
        if self.n_coh <= 0:
            raise ValueError("n_coh must be positive")
        return float(10.0 ** (-self.ground_filter_db / 10.0) * self.n_coh)

    @property
    def suppressed_db(self) -> float:
        """Residual relative to per-frame system noise, before coherence."""
        if self.components:
            surviving = sum(frac for frac, _ in self.components)
            gf = np.inf if surviving <= 0 else 10.0 * np.log10(1.0 / surviving)
            return float(self.shelf_floor_db - gf - self.delay_filter_db)
        return self.shelf_floor_db - self.ground_filter_db - self.delay_filter_db

    @property
    def ratio(self) -> float:
        """r = P_res / P_thermal in the integrated estimate."""
        return float(10.0 ** ((self.shelf_floor_db - self.delay_filter_db) / 10.0)
                     * self._gain)

    @property
    def ratio_db(self) -> float:
        return float(10.0 * np.log10(self.ratio)) if self.ratio > 0 else -np.inf

    def chain(self) -> str:
        head = (
            f"{self.label or 'residual'}\n"
            f"    kept-frame shelf bound   {self.shelf_floor_db:+7.2f} dB\n"
        )
        if self.components:
            body = ""
            for name, (frac, n) in zip(("intra-day", "fast", "extra"),
                                       self.components):
                body += (f"    {name:<12s} {100 * frac:8.4f}%  x n_coh {n:9.4g}"
                         f"  -> {10 * np.log10(max(frac * n, 1e-300)):+7.2f} dB\n")
            surviving = sum(f for f, _ in self.components)
            body += (f"    ground / m=0 filter      "
                     f"{-10 * np.log10(1.0 / max(surviving, 1e-300)):+7.2f} dB "
                     f"(from the split above)\n")
        else:
            body = (f"    ground / m=0 filter      "
                    f"{-self.ground_filter_db:+7.2f} dB\n"
                    f"    coherence (n_coh={self.n_coh:g})"
                    f"{10 * np.log10(self.n_coh):+9.2f} dB\n")
        return (
            head + body
            + f"    delay filter             {-self.delay_filter_db:+7.2f} dB\n"
            f"    ---------------------------------\n"
            f"    => r = P_res/P_N         {self.ratio_db:+7.2f} dB "
            f"({self.ratio:.4g})"
            + ("" if self.tau_measured else
               "   [BOUND: tau_c not measured, capped at one sidereal day]")
        )


class NoMeasuredFloor(ValueError):
    """Raised when a masked-policy residual is asked for without a floor.

    A masked frame is one the detector did not fire on, so its shelf was never
    measured. What bounds it is the *sensitivity floor*: the shelf level the
    estimator reports on frames where the transmitter is known quiet. That
    requires a null population (valid frames with no detection that still
    carry a finite shelf estimate), and some channels have none at all.

    The tempting substitute is the smallest shelf among the frames that *were*
    detected. It is not a floor. It is the weakest positive excess that
    happened to occur in the sample, it falls without limit as the detection
    count grows, and on ch35 it came out at -76 dB against measured floors of
    -45 to -49 dB elsewhere: thirty decibels of pure sampling artifact,
    which propagated into a masked residual four orders of magnitude too
    small and turned a failing verdict into a passing one. Refusing is the
    only safe behaviour: a channel with no null population has an unmeasured
    masked residual, and that is a statement the analysis has to carry rather
    than paper over.
    """


def masked_residual(stats: ShelfStatistics, gain: float,
                    delay_key: str = DEFAULT_DELAY_KEY,
                    floor_db: float | None = None) -> float:
    """Residual left by a policy that masks every detected frame.

    ``floor_db`` overrides the measured floor for an explicit what-if; pass it
    only with the substitution stated in the caller, never to fill a gap
    silently.
    """
    fl = stats.floor_db if floor_db is None else floor_db
    if not np.isfinite(fl):
        raise NoMeasuredFloor(
            f"ch{stats.channel} has no null population "
            f"({stats.n_off_frames} frames with a finite shelf among the "
            f"non-detections), so its kept-frame floor is unmeasured and the "
            f"masked residual cannot be computed. Supply floor_db explicitly "
            f"if substituting a value from elsewhere, and say so.")
    return float(10.0 ** ((fl - DELAY_SUPPRESSION_DB[delay_key]) / 10.0) * gain)


@dataclass
class FloorProvenance:
    """Where a channel's sensitivity floor actually comes from.

    Two constants govern every floor in these products and they are not the
    same number:

    ``mu0``   the decision line. The deployed rule is reject <=> F > mu0, and
              mu0 = 2||w0||^2 / (||w1||^2 + ||w2||^2) is an exact rational
              constant of the quantised weight bank.

    ``1``     the level estimator's reference. The product sets
              pnr_bin_db = 10 log10(F - 1) exactly, so it reports no level at
              all for F <= 1, whatever mu0 happens to be.

    The floor is read from frames that are kept *and* carry a level, which is
    the interval 1 < F <= mu0. Two consequences follow, and both are arithmetic
    rather than empirical:

    * Where mu0 > 1 the interval is non-empty but narrow, so the floor lands on
      10 log10(mu0 - 1) plus the fixed shelf offset, a property of the weight
      bank rather than a measurement of the sky. ``agreement_db`` is how closely the
      reported floor tracks that prediction; a few tenths of a decibel means
      the floor carries no information about the transmitter or the noise.

    * Where mu0 < 1 the interval is empty for every possible dataset, so no
      frame can set a floor however long the transmitter stays off. Reporting
      that as "no null population" invites the reading that the channel was
      never quiet, which the data do not say.

    ``sigma_null`` is the alternative that is a measurement: the scatter of the
    decision statistic under the null, estimated from the kept sample, which is
    the null's lower half. A threshold sitting at the null centre cannot
    resolve anything below its own scatter, so ``sigma_implied_db`` is the
    level the mask can actually be held to, and it exists on every channel
    with a usable kept sample, including those where mu0 < 1.
    """
    channel: int
    freq_id: int
    mu0: float
    n_kept: int
    n_sliver: int
    n_masked_without_level: int
    shelf_offset_db: float
    reported_db: float
    mu0_implied_db: float
    sigma_null: float
    sigma_spread: float
    sigma_implied_db: float

    @property
    def agreement_db(self) -> float:
        """Reported floor minus the value mu0 alone predicts."""
        return self.reported_db - self.mu0_implied_db

    # The reported floor is the high percentile of a sliver whose upper edge
    # is exactly 10log10(mu0-1) + offset, so it can never sit above that value
    # and sits a few tenths below it: 10log10(0.9) = -0.46 dB for a flat
    # sliver at the 90th percentile, plus sampling noise on thin slivers. A
    # decibel and a half is loose enough for a starved channel and far tighter
    # than anything a floor that tracked the sky would satisfy.
    MU0_AGREEMENT_DB = 1.5

    @property
    def mu0_determined(self) -> bool:
        return bool(np.isfinite(self.reported_db)
                    and abs(self.agreement_db) < self.MU0_AGREEMENT_DB)

    @property
    def verdict(self) -> str:
        if self.mu0 < 1.0:
            return ("unavailable: mu0 < 1, so 1 < F <= mu0 is empty for any "
                    "dataset and no frame can set a floor")
        if self.mu0_determined:
            return (f"mu0-determined: within {abs(self.agreement_db):.2f} dB "
                    f"of 10log10(mu0-1), so it measures the weight bank")
        if not np.isfinite(self.reported_db):
            return "unavailable: no kept frame carries a level"
        return "not explained by mu0 alone"


# Quantile probes for the null scale, and the standard-normal deviates they
# correspond to *in the full null*; the kept sample is its lower half, so the
# p-th percentile of the kept frames is the (p/2)-th percentile of the null.
NULL_SCALE_PROBES = ((32.0, 1.0000), (5.0, 1.9600), (0.3, 2.9677))
MIN_THRESHOLD_SWEEP_KEPT_FRAMES = 30


def null_scale(f_kept: np.ndarray, mu0: float) -> tuple[float, float]:
    """Robust scale of the decision statistic under the null, left side only.

    Returns ``(sigma, spread)``, where ``spread`` is the ratio of the largest
    to the smallest of the individual quantile estimates. A spread near one
    means the left tail is Gaussian and well sampled; a large spread means the
    channel is masked so heavily that the kept frames cannot characterise the
    null, and the scale should be treated as indicative rather than measured.
    """
    ests = [(mu0 - float(np.percentile(f_kept, p))) / z
            for p, z in NULL_SCALE_PROBES]
    ests = [e for e in ests if e > 0.0]
    if not ests:
        return float("nan"), float("nan")
    return float(np.median(ests)), float(max(ests) / min(ests))


def floor_provenance(npz_path: str | Path,
                     floor_percentile: float = 90.0) -> FloorProvenance:
    """Trace a channel's reported floor back to the constant that fixes it."""
    d = load_npz(npz_path)
    valid = d["valid"][:, 0].astype(bool)
    rejected = d["reject_mask"][:, 0].astype(bool)
    F = d["fstat_raw"][:, 0]
    shelf = d["snr_shelf_db"][:, 0]
    mu0 = float(d["mu0"][0])

    if not np.array_equal(rejected[valid], F[valid] > mu0):
        raise ValueError("product's reject_mask is not the rule F > mu0; "
                         "the provenance argument below does not apply")

    # Read the shelf offset off the product rather than assuming it, then check
    # that the level really is 10log10(F - 1) shifted by it.
    offset = (float(d["pilot_below_data_db"])
              - 10.0 * np.log10(float(d["dtv_bandwidth_hz"])
                                / float(d["bin_enbw_hz"])))
    lev = valid & np.isfinite(shelf)
    if lev.sum():
        resid = shelf[lev] - (10.0 * np.log10(F[lev] - 1.0) + offset)
        if not np.all(np.abs(resid) < 1e-6):
            raise ValueError("product's shelf is not 10log10(F-1) + offset; "
                             "the provenance argument below does not apply")

    kept = valid & ~rejected
    sliver = kept & lev                       # exactly the frames 1 < F <= mu0
    sigma, spread = null_scale(F[kept], mu0)

    return FloorProvenance(
        channel=int(d["physical_channel"][0]),
        freq_id=int(d["freq_id"][0]),
        mu0=mu0,
        n_kept=int(kept.sum()),
        n_sliver=int(sliver.sum()),
        n_masked_without_level=int((valid & rejected & ~lev).sum()),
        shelf_offset_db=offset,
        reported_db=(float(np.percentile(shelf[sliver], floor_percentile))
                     if sliver.sum() else float("nan")),
        mu0_implied_db=10.0 * np.log10(abs(mu0 - 1.0)) + offset,
        sigma_null=sigma, sigma_spread=spread,
        sigma_implied_db=10.0 * np.log10(sigma / mu0) + offset,
    )


def budget_from_statistics(stats: ShelfStatistics, delay_key: str = DEFAULT_DELAY_KEY,
                           tau_intraday: float = MAX_TAU_C_SECONDS,
                           tau_fast: float = CHIME_FRAME_SECONDS,
                           frame_seconds: float = CHIME_FRAME_SECONDS,
                           tau_measured: bool = False) -> ResidualBudget:
    """Assemble a :class:`ResidualBudget` from measured shelf statistics.

    The two surviving populations get their own coherence. ``tau_intraday``
    defaults to the sidereal-day cap: the pessimistic end, and the only value
    that needs no assumption at all, since anything longer has already been
    removed as m = 0. Narrow it if the acquisition cadence resolves the
    intra-day structure.
    """
    if delay_key not in DELAY_SUPPRESSION_DB:
        raise ValueError(f"unknown delay_key {delay_key!r}; "
                         f"choose from {sorted(DELAY_SUPPRESSION_DB)}")
    comps = (
        (float(stats.intraday_fraction),
         n_coh_from_correlation_time(tau_intraday, frame_seconds)),
        (float(stats.fast_fraction),
         n_coh_from_correlation_time(tau_fast, frame_seconds)),
    )
    return ResidualBudget(
        shelf_floor_db=stats.floor_db,
        delay_filter_db=DELAY_SUPPRESSION_DB[delay_key],
        components=comps,
        tau_measured=tau_measured,
        label=(f"ch{stats.channel} ({delay_key}, "
               f"tau_intraday={tau_intraday:g} s"
               f"{'' if tau_measured else ', CAP'})"))


def n_coh_from_correlation_time(tau_c_seconds: float,
                                frame_seconds: float = CHIME_FRAME_SECONDS) -> float:
    """Frames per residual correlation time.

    This is the whole integration-scaling model in one number. ``tau_c`` at the
    frame scale gives 1 (residual behaves as thermal noise); ``tau_c`` of an
    hour gives ~8.6e4, i.e. +49 dB, which is what makes a slowly-varying
    residual dangerous even when it is far below the per-frame noise.
    """
    if frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive")
    tau = min(float(tau_c_seconds), MAX_TAU_C_SECONDS)
    return max(1.0, tau / float(frame_seconds))


def residuals_from_products(paths, off_through=None, delay_key=DEFAULT_DELAY_KEY,
                            floor_percentile: float = 90.0,
                            trim_percentile: float = 90.0, **ct_kwargs):
    """Per-channel ``{channel: r}`` plus the statistics and correlation times.

    ``off_through`` may be a single ``YYYY-MM`` applied to every product, or a
    ``{channel: 'YYYY-MM'}`` mapping. Channels with no measurable shelf floor
    are omitted rather than defaulted; channels whose correlation time is
    refused are included at the conservative cap and flagged by
    ``corrs[ch].is_measured``.
    """
    out, stats, corrs = {}, {}, {}
    for p in paths:
        head = shelf_statistics(p, off_through=None,
                                floor_percentile=floor_percentile)
        ot = (off_through.get(head.channel) if isinstance(off_through, dict)
              else off_through)
        budget, st, ct = budget_from_products(
            p, off_through=ot, delay_key=delay_key,
            floor_percentile=floor_percentile,
            trim_percentile=trim_percentile, **ct_kwargs)
        stats[st.channel] = st
        corrs[st.channel] = ct
        if np.isfinite(st.floor_db):
            out[st.channel] = budget.ratio
    return out, stats, corrs

def budget_from_products(npz_path: str | Path, off_through: str | None = None,
                         delay_key: str = DEFAULT_DELAY_KEY,
                         floor_percentile: float = 90.0,
                         trim_percentile: float = 90.0,
                         **ct_kwargs):
    """One call: shelf statistics, correlation time, and the assembled budget.

    Returns ``(budget, stats, corr)``. The correlation time is measured where
    the data supports one and falls back to the sidereal-day cap where it does
    not, so a refused channel yields a *bound* rather than a wrong number --
    ``budget.tau_measured`` says which you got.
    """
    stats = shelf_statistics(npz_path, off_through=off_through,
                             floor_percentile=floor_percentile,
                             trim_percentile=trim_percentile)
    corr = correlation_time(npz_path, off_through=off_through,
                            trim_percentile=trim_percentile, **ct_kwargs)
    if corr.is_usable:
        budget = budget_from_statistics(stats, delay_key,
                                        tau_intraday=corr.tau_c,
                                        tau_measured=corr.is_measured)
    else:
        # The estimator refuses when the shelf is episodic, and the same
        # non-stationarity that makes tau_c unmeasurable makes the variance
        # split unmeasurable, since both are moments of the same process. So
        # the fallback takes no ground-filter credit either: all shelf power
        # surviving, at the sidereal-day cap. That is an actual bound rather
        # than a cap applied to a split that is an artefact of the trim.
        budget = ResidualBudget(
            shelf_floor_db=stats.floor_db,
            delay_filter_db=DELAY_SUPPRESSION_DB[delay_key],
            components=((1.0, n_coh_from_correlation_time(MAX_TAU_C_SECONDS)),),
            tau_measured=False,
            label=f"ch{stats.channel} ({delay_key}, BOUND: no stationary split)")
    return budget, stats, corr


# ----------------------------------------------------------------------
# Is masking worth it?
# ----------------------------------------------------------------------

# What can actually be done with a DTV-contaminated channel. These are the
# alternatives a survey has rather than a menu of detector settings: masking is only
# worth deploying if it beats every one of them.
POLICY_KINDS = ("keep", "excise", "incumbent", "proxy")


@dataclass(frozen=True)
class Policy:
    """One thing a survey can do with a contaminated channel.

    ``f`` is the fraction of frames the policy discards and ``r`` the residual
    shelf power left in what survives, in units of system noise. Excision is
    ``f = 1``; keeping everything is ``f = 0`` with ``r`` at the raw
    contamination level.
    """
    name: str
    f: float
    r: float
    kind: str = "proxy"
    note: str = ""

    def __post_init__(self):
        if not 0.0 <= self.f <= 1.0:
            raise ValueError(f"{self.name}: masked fraction out of range: {self.f}")
        if self.r < 0:
            raise ValueError(f"{self.name}: residual ratio must be non-negative")
        if self.kind not in POLICY_KINDS:
            raise ValueError(f"{self.name}: unknown policy kind {self.kind!r}")

    @property
    def time_penalty(self) -> float:
        """Integration time to reach a fixed noise level, vs a clean channel.

        Contamination raises the variance by (1+r) and discarding frames
        lowers the integration by (1-f), so the time needed to reach a given
        error scales as ``(1+r)/(1-f)``. Excision is infinite: no amount of
        time on a discarded channel produces a measurement in that band.
        """
        if self.f >= 1.0:
            return float("inf")
        return float((1.0 + self.r) / (1.0 - self.f))


@dataclass
class PolicyComparison:
    """The four-way decision for one channel, and which option wins.

    Masking is not being weighed against leaving the data dirty. A survey with
    a DTV-contaminated channel has four moves, and the detector has to beat
    all of them to be worth deploying:

    keep
        Integrate the dirty data. Costs ``(1+r)`` in time and puts the full
        residual into the map.
    excise
        Drop the channel. Costs the band (infinite time to any measurement
        inside it) but leaves nothing behind.
    incumbent
        Flag with what the pipeline already runs: a MAD cut on integrated
        power, or spectral kurtosis. Both estimate their own baseline from the
        data, so a transmitter with a duty cycle above ~50% is absorbed into
        that baseline and passes through.
    proxy
        Flag on the pilot. A matched filter on a tone of known frequency needs
        no baseline estimate and so does not care about duty cycle.

    Which wins depends on ``bias_tolerance``: the largest residual whose
    first-order parameter bias stays acceptable, computed on the Fisher side
    from ``expt['P_res']``. Without it this is a pure noise trade and the
    answer is nearly always "keep everything", because a residual that merely
    adds variance is cheaper to integrate through than to cut around. With it
    the question becomes whether any policy reaches the tolerance at all, and
    that is where a policy the incumbent cannot reach starts to matter.
    """
    channel: int
    policies: tuple[Policy, ...]
    bias_tolerance: float | None = None

    def __post_init__(self):
        self.policies = tuple(self.policies)
        if not self.policies:
            raise ValueError("no policies to compare")
        seen = [p.name for p in self.policies]
        if len(set(seen)) != len(seen):
            raise ValueError(f"duplicate policy names: {seen}")

    def of_kind(self, kind: str) -> tuple[Policy, ...]:
        return tuple(p for p in self.policies if p.kind == kind)

    def meets_bias(self, policy: Policy) -> bool:
        """Excision always meets it; without a tolerance nothing is excluded."""
        if policy.kind == "excise":
            return True
        if self.bias_tolerance is None:
            return True
        return policy.r <= self.bias_tolerance

    def feasible(self) -> tuple[Policy, ...]:
        return tuple(p for p in self.policies if self.meets_bias(p))

    def best(self, exclude_kinds: tuple[str, ...] = ()) -> Policy:
        """Cheapest feasible policy. Excision is the fallback, never empty."""
        pool = [p for p in self.feasible() if p.kind not in exclude_kinds]
        if not pool:
            raise ValueError("no feasible policy and no excision to fall back on")
        return min(pool, key=lambda p: p.time_penalty)

    def best_without_proxy(self) -> Policy:
        """What the survey would do if the detector did not exist."""
        return self.best(exclude_kinds=("proxy",))

    @property
    def proxy_advantage(self) -> float:
        """Time the detector saves, as a multiple. inf means it saves the band.

        1.0 means the best thing to do is what would have been done anyway.
        """
        without = self.best_without_proxy().time_penalty
        with_ = self.best().time_penalty
        if with_ == 0:
            return float("inf")
        if np.isinf(without):
            return float("inf") if np.isfinite(with_) else 1.0
        return float(without / with_)

    @property
    def saves_the_band(self) -> bool:
        """The detector turns an excised channel into a measured one."""
        return (self.best_without_proxy().kind == "excise"
                and self.best().kind != "excise")

    def verdict(self) -> str:
        best, without = self.best(), self.best_without_proxy()
        if self.saves_the_band:
            return (f"ch{self.channel}: excise without the detector; "
                    f"'{best.name}' with it (time penalty {best.time_penalty:.2f}x)")
        adv = self.proxy_advantage
        if best.kind != "proxy":
            return (f"ch{self.channel}: '{best.name}' wins without needing the "
                    f"detector to mask (time penalty {best.time_penalty:.2f}x); "
                    f"the detector's contribution here is the measurement of r "
                    f"that decides this rather than the mask")
        return (f"ch{self.channel}: '{best.name}' wins, "
                f"{adv:.2f}x faster than '{without.name}'")

    def tolerance_map(self):
        """Which policy wins, over every bias tolerance. [(lo, hi, policy)].

        The feasible set only changes as the tolerance crosses one of the
        measured residuals, so the whole decision is a handful of intervals.
        This is the answer to "when does the detector matter": the width of
        the interval it wins on, in decades of tolerance, is how much room
        there is between the residual the incumbent can reach and the one that
        forces excision.
        """
        breaks = sorted({p.r for p in self.policies if p.kind != "excise"})
        edges = [0.0] + breaks + [float("inf")]
        spans = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            feasible = [p for p in self.policies
                        if p.kind == "excise" or p.r <= lo]
            win = min(feasible, key=lambda p: p.time_penalty)
            if spans and spans[-1][2] is win:
                spans[-1] = (spans[-1][0], hi, win)
            else:
                spans.append((lo, hi, win))
        return spans

    def winning_span(self, kind: str = "proxy"):
        """(lo, hi) tolerance range where a kind of policy wins, or None."""
        hits = [(lo, hi) for lo, hi, p in self.tolerance_map() if p.kind == kind]
        if not hits:
            return None
        return (min(h[0] for h in hits), max(h[1] for h in hits))

    @property
    def proxy_decades(self) -> float:
        """Decades of bias tolerance over which only the detector keeps the band.

        Zero means every tolerance is either loose enough for the incumbent or
        tight enough to force excision, and the mask buys nothing.
        """
        span = self.winning_span("proxy")
        if span is None:
            return 0.0
        lo, hi = span
        if lo <= 0 or not np.isfinite(hi):
            return float("inf")
        return float(np.log10(hi / lo))

    def table(self) -> str:
        tol = ("none" if self.bias_tolerance is None
               else f"r <= {self.bias_tolerance:.4g}")
        lines = [f"ch{self.channel}  bias tolerance: {tol}",
                 f"  {'policy':32s} {'kind':10s} {'f':>8s} {'r':>11s} "
                 f"{'time':>9s}  bias",
                 "  " + "-" * 78]
        best = self.best()
        for p in sorted(self.policies, key=lambda q: q.time_penalty):
            t = "inf" if np.isinf(p.time_penalty) else f"{p.time_penalty:.3f}"
            ok = "ok" if self.meets_bias(p) else "FAILS"
            star = " <-" if p is best else ""
            lines.append(f"  {p.name:32s} {p.kind:10s} {p.f:8.4f} {p.r:11.5g} "
                         f"{t:>9s}  {ok}{star}")
        lines.append("  " + self.verdict())
        return "\n".join(lines)


def compare_policies(channel, keep_r, proxy=None, incumbents=(),
                     bias_tolerance=None) -> PolicyComparison:
    """Assemble the four-way comparison from measured (f, r) pairs.

    ``keep_r`` is the raw contamination with nothing masked. ``proxy`` and each
    entry of ``incumbents`` are ``(name, f, r)``. Excision is added
    automatically: it is always available and its cost never has to be
    measured.
    """
    pols = [Policy("keep everything", 0.0, float(keep_r), kind="keep"),
            Policy("excise the channel", 1.0, 0.0, kind="excise")]
    for name, f, r in incumbents:
        pols.append(Policy(name, float(f), float(r), kind="incumbent"))
    if proxy is not None:
        name, f, r = proxy
        pols.append(Policy(name, float(f), float(r), kind="proxy"))
    return PolicyComparison(int(channel), tuple(pols),
                            None if bias_tolerance is None else float(bias_tolerance))


@dataclass
class MaskDecision:
    """Whether masking beats leaving the data dirty.

    This answers one edge of :class:`PolicyComparison` (masking against
    keeping everything) and is kept because that edge has a closed form
    worth having (``break_even_f``). It is not the deployment question. A
    channel where masking loses to keeping everything may still be one where
    both lose to excision, and the choice between *those* is what decides
    whether the band survives. Use :func:`compare_policies` to decide; use
    this to understand the trade at a fixed operating point.

    The test is not whether masking lowers the residual; it almost always
    does) but whether the noise it removes beats the data it costs. Masking
    a fraction f raises the noise by 1/(1-f) through lost integration, and
    lowers it by (1+r_masked)/(1+r_unmasked) through reduced contamination.
    The product of those is the net multiplier on effective noise power::

        net = (1 + r_unmasked) / (1 + r_masked) * (1 - f)

    and masking pays only when ``net > 1``. Below 1 the mask is discarding
    more signal than contamination, which is what happens on a channel whose
    transmitter is on nearly all the time: frame selection needs variance to
    exploit, and a steady transmitter offers none.
    """
    channel: int
    f: float
    r_unmasked: float
    r_masked: float

    @property
    def noise_gain(self) -> float:
        """Factor by which masking reduces contamination."""
        return (1.0 + self.r_unmasked) / (1.0 + self.r_masked)

    @property
    def data_cost(self) -> float:
        """Factor by which masking raises noise through lost integration."""
        return np.inf if self.f >= 1.0 else 1.0 / (1.0 - self.f)

    @property
    def net(self) -> float:
        """> 1 means masking pays; < 1 means it costs more than it saves."""
        return float(self.noise_gain / self.data_cost)

    @property
    def should_mask(self) -> bool:
        return self.net > 1.0

    @property
    def break_even_f(self) -> float:
        """Largest masked fraction worth paying at this contamination level."""
        if self.r_unmasked <= self.r_masked:
            return 0.0
        return float(max(0.0, 1.0 - (1.0 + self.r_masked) / (1.0 + self.r_unmasked)))

    def summary(self) -> str:
        return (f"ch{self.channel:>3d}  f={self.f:.4f}  "
                f"r {self.r_unmasked:.4g} -> {self.r_masked:.4g}  "
                f"gain {self.noise_gain:.2f}x  cost {self.data_cost:.1f}x  "
                f"net {self.net:.3f}  "
                f"{'MASK' if self.should_mask else 'DO NOT MASK'} "
                f"(break-even f = {self.break_even_f:.3f})")


def mask_benefit(channel, f, r_unmasked, r_masked) -> MaskDecision:
    """Does masking this channel pay? See :class:`MaskDecision`."""
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"masked fraction out of range: {f}")
    if r_unmasked < 0 or r_masked < 0:
        raise ValueError("residual ratios must be non-negative")
    return MaskDecision(int(channel), float(f), float(r_unmasked), float(r_masked))


def threshold_sweep(npz_path, off_through=None, etas=None,
                    delay_key=DEFAULT_DELAY_KEY, floor_percentile=90.0,
                    tau_intraday=None, floor_db=None,
                    min_kept: int = MIN_THRESHOLD_SWEEP_KEPT_FRAMES):
    """(eta, f, kept-shelf dB, r, net) as the coarse threshold F > eta*mu0 moves.

    This is what the framework exists to compute: the detector's operating
    point mapped onto a science decision. Frames with no pilot detection have
    no shelf measurement, so they are bounded at the transmitter-off
    sensitivity floor, an upper bound that makes the result conservative.

    ``floor_db`` supplies that bound explicitly when the product cannot: on a
    channel whose mu0 < 1 the interval 1 < F <= mu0 is empty and no kept frame
    carries a level (see :func:`floor_provenance`), so there is nothing to
    take a percentile of. The caller states the substitute (the natural one
    is ``floor_provenance(...).sigma_implied_db``, the level a threshold at
    the null center can actually resolve), and the substitution is theirs to
    defend. Without it, such a product returns an empty sweep rather than a
    fabricated one.
    """
    d = load_npz(npz_path)
    valid = d["valid"][:, 0].astype(bool)
    shelf = d["snr_shelf_db"][:, 0]
    F = d["fstat_raw"][:, 0]
    mu0 = float(d["mu0"][0])
    t0 = d["unit_time0_ctime"]
    month = _month_of(t0)[d["frame_unit_index"]]

    if off_through is not None:
        on = valid & (month > off_through)
        off = valid & (month <= off_through)
    else:
        on, off = valid, valid & ~d["reject_mask"][:, 0].astype(bool)
    fin_off = off & np.isfinite(shelf)
    if on.sum() < 50:
        return []
    if floor_db is None:
        if fin_off.sum() == 0:
            return []
        floor_db = float(np.percentile(shelf[fin_off], floor_percentile))
    lin = np.where(np.isfinite(shelf), 10.0 ** (shelf / 10.0),
                   10.0 ** (floor_db / 10.0))

    stats = shelf_statistics(npz_path, off_through=off_through,
                             floor_percentile=floor_percentile)
    corr = correlation_time(npz_path, off_through=off_through)
    tau = tau_intraday if tau_intraday is not None else corr.tau_for_budget
    n_slow = n_coh_from_correlation_time(tau)
    comps_for = lambda db: ((stats.intraday_fraction, n_slow),
                            (stats.fast_fraction, 1.0))

    def r_of(mean_lin):
        db = 10.0 * np.log10(max(mean_lin, 1e-30))
        return ResidualBudget(shelf_floor_db=db,
                              delay_filter_db=DELAY_SUPPRESSION_DB[delay_key],
                              components=comps_for(db)).ratio

    r_un = r_of(float(lin[on].mean()))
    if etas is None:
        etas = np.concatenate([[1.0], np.geomspace(1.05, 500.0, 24)])
    out = []
    for eta in etas:
        keep = on & (F <= eta * mu0)
        if keep.sum() < min_kept:
            continue
        f = 1.0 - keep.sum() / on.sum()
        kept_lin = float(lin[keep].mean())
        dec = mask_benefit(int(d["physical_channel"][0]), f, r_un, r_of(kept_lin))
        out.append(dict(eta=float(eta), f=f,
                        kept_shelf_db=10.0 * np.log10(max(kept_lin, 1e-30)),
                        r_unmasked=r_un, r_masked=dec.r_masked,
                        net=dec.net, should_mask=dec.should_mask))
    return out


def best_operating_point(sweep):
    """The row of a :func:`threshold_sweep` with the largest net benefit."""
    rows = [r for r in sweep if np.isfinite(r["net"])]
    return max(rows, key=lambda r: r["net"]) if rows else None

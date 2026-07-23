"""Masking scenarios: translate per-channel masked-time fractions into
per-redshift-bin noise/volume factors for the Fisher forecast.

Model
-----
RFI masking removes time-frequency samples. For a frequency slice with masked
fraction f, the effective integration time is t_eff = ttot * (1 - f), so its
thermal noise power grows by 1/(1-f).

Channels masked more than ``excise_threshold`` are treated as *excised*: the
slice is dropped from the analysis entirely. That costs survey volume
(Fisher information scales linearly with the excised bandwidth fraction) but
does not degrade the noise of the surviving band. This mirrors real practice:
a 97%-masked channel (e.g. ATSC 30) is cut rather than integrated 33x longer.

For each RadioFisher redshift bin we return:
  v_frac : surviving bandwidth fraction (scales the bin's Fisher matrix)
  w_bar  : effective clean-time fraction of the *surviving* band
           (rescales ttot -> ttot * w_bar)

Two conventions for w_bar over a bin containing slices with different f:
  mode='time'    : w_bar = <1 - f>            (inverse-variance / sample
                   counting; standard forecasting assumption)
  mode='fourier' : w_bar = 1 / <1/(1-f)>      (radial Fourier modes see the
                   arithmetic mean of per-slice noise power; pessimistic)
The two agree when f is uniform across the bin.

Residual contamination
----------------------
Masking is only half the cost. What survives the mask adds power to the band,
and a slice carrying a residual-to-thermal power ratio r has its effective
noise raised to P_N (1 + r), exactly equivalent in this framework to
shortening that slice's integration time by the same factor. So a residual
folds in as

    (1 - f)  ->  (1 - f) / (1 + r)

with no other change to the machinery. ``residuals`` is empty by default, so
every number in this repository predating it is reproduced exactly.

That coupling is what makes a detector threshold optimisable. Lowering the
threshold raises f (priced here from the start) and lowers r (priced now);
raising it does the reverse. Without the r term the forecast is monotone in f
and its own optimum is "never mask", which is not a physical answer. See
:mod:`baonoise.residual` for where r comes from and what in it is measured
rather than assumed.

Treating a residual as excess *variance* is the conservative reading only if
the residual is incoherent; a coherent residual is a *bias*, and a bias is not
bounded by this construction. ``residual_excise_threshold`` exists for that
case: a slice whose r exceeds it is dropped rather than integrated through.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import channels as chn


def _overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


@dataclass
class Scenario:
    """A per-channel masking scenario."""
    name: str
    label: str
    fractions: dict[int, float] = field(default_factory=dict)  # ch -> f_masked
    excise_threshold: float = 0.5
    mode: str = "time"           # 'time' or 'fourier'
    residuals: dict[int, float] = field(default_factory=dict)  # ch -> r
    residual_excise_threshold: float = np.inf

    # ------------------------------------------------------------------
    def keep_weight(self, ch: int) -> float:
        """Surviving-time weight of a kept channel: (1 - f) / (1 + r)."""
        f = self.fractions.get(ch, 0.0)
        r = self.residuals.get(ch, 0.0)
        if r < 0.0:
            raise ValueError(f"residual for channel {ch} is negative: {r}")
        return (1.0 - f) / (1.0 + r)

    def is_excised(self, ch: int) -> bool:
        return (self.fractions.get(ch, 0.0) >= self.excise_threshold
                or self.residuals.get(ch, 0.0) >= self.residual_excise_threshold)

    # ------------------------------------------------------------------
    def bin_factors(self, nu_lo: float, nu_hi: float) -> tuple[float, float]:
        """Return (v_frac, w_bar) for a frequency bin [nu_lo, nu_hi] MHz."""
        width = nu_hi - nu_lo
        if width <= 0:
            return 1.0, 1.0

        excised = 0.0
        masked_slices = []   # (bandwidth, weight) for surviving DTV slices
        dtv_total = 0.0
        for ch in set(self.fractions) | set(self.residuals):
            lo, hi = chn.channel_edges(ch)
            ov = _overlap(nu_lo, nu_hi, lo, hi)
            if ov <= 0.0:
                continue
            dtv_total += ov
            if self.is_excised(ch):
                excised += ov
            else:
                masked_slices.append((ov, self.keep_weight(ch)))

        v_frac = (width - excised) / width
        surv = width - excised
        if surv <= 0.0:
            return 0.0, 1.0

        clean_bw = width - dtv_total          # band with no DTV allocation
        if self.mode == "time":
            acc = clean_bw * 1.0
            acc += sum(bw * wt for bw, wt in masked_slices)
            w_bar = acc / surv
        elif self.mode == "fourier":
            if any(wt <= 0.0 for _, wt in masked_slices):
                return v_frac, 0.0     # a slice with infinite noise power
            acc = clean_bw * 1.0
            acc += sum(bw / wt for bw, wt in masked_slices)
            w_bar = surv / acc
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return v_frac, float(np.clip(w_bar, 0.0, 1.0))

    def bin_factors_for_zbins(self, zs: np.ndarray) -> np.ndarray:
        """(Nbins, 2) array of (v_frac, w_bar) for RadioFisher z-bin edges."""
        out = []
        for i in range(len(zs) - 1):
            nu_lo = chn.NU_LINE / (1.0 + zs[i + 1])
            nu_hi = chn.NU_LINE / (1.0 + zs[i])
            out.append(self.bin_factors(nu_lo, nu_hi))
        return np.array(out)

    # ------------------------------------------------------------------
    def freq_weight_fn(self):
        """Vectorised w(nu_MHz) for RadioFisher's expt['noise_freq_weight']
        hook: surviving time weight (1 - f)/(1 + r) inside kept DTV channels,
        1.0 outside any listed channel, NaN inside excised channels (the hook
        excludes NaN slices from the band average; their loss is priced via
        expt['vol_frac']).

        Must stay consistent with :meth:`bin_factors`, since
        ``Forecast.sigma_A_direct`` uses this hook to validate the bank path."""
        chans = sorted(set(self.fractions) | set(self.residuals))
        edges, values = [], []
        for ch in chans:
            lo, hi = chn.channel_edges(ch)
            w = np.nan if self.is_excised(ch) else self.keep_weight(ch)
            edges.append((lo, hi))
            values.append(w)
        edges = np.array(edges)
        values = np.array(values)

        def w_of_nu(nu):
            nu = np.atleast_1d(np.asarray(nu, dtype=float))
            out = np.ones_like(nu)
            for (lo, hi), w in zip(edges, values):
                sel = (nu >= lo) & (nu < hi)
                out[sel] = w
            return out

        return w_of_nu

    def rf_mode(self) -> str:
        """RadioFisher noise_freq_mode string for this scenario."""
        return "fourier" if self.mode == "fourier" else "invvar"


# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------

def clean() -> Scenario:
    return Scenario("clean", "No masking (RFI-free)")


def measured(rates_csv=None, refused_fraction: float = chn.REFUSED_FRACTION,
             excise_threshold: float = 0.5, mode: str = "time",
             residuals: dict[int, float] | None = None,
             residual_excise_threshold: float = np.inf,
             products=None, fill_missing: str = "error") -> Scenario:
    """Fiducial: pilot-proxy exposure-weighted rates; refused ch24/30 excised.

    ``residuals`` optionally adds the contamination that survives the mask
    (see :mod:`baonoise.residual`); omitted, the result is masking-only and
    identical to every published number in ``out/``.

    ``products`` takes the masking fractions from survey products instead of
    the vendored CSV, so the forecast's input is the detector's own decision
    rather than an unlabeled rate column. Channels the products do not cover
    are an error by default: filling them from the CSV would put two different
    detectors in one table, which is the failure this argument exists to stop.
    Pass ``fill_missing='csv'`` to do it anyway (the scenario is then tagged),
    or ``'omit'`` to forecast on the covered channels alone.
    """
    kw = {} if rates_csv is None else {"rates_csv": rates_csv}
    table = chn.measured_mask_table(refused_fraction=refused_fraction, **kw)
    label, name = "Measured pilot-proxy masking", "measured"

    if products is not None:
        prod = chn.mask_table_from_products(products,
                                            refused_fraction=refused_fraction)
        missing = sorted(set(table.fractions) - set(prod.fractions))
        if missing and fill_missing == "error":
            raise ValueError(
                f"products cover {sorted(prod.fractions)} but the band needs "
                f"{missing} as well; filling those from the CSV mixes two "
                f"detectors; pass fill_missing='csv' to accept that, or "
                f"'omit' to forecast on the covered channels only")
        fr = dict(prod.fractions)
        if missing and fill_missing == "csv":
            fr.update({ch: table.fractions[ch] for ch in missing})
            label += f" (products; {len(missing)} channels from CSV)"
            name = "measured_mixed"
        elif missing and fill_missing == "omit":
            label += f" (products; {len(missing)} channels omitted)"
            name = "measured_products"
        else:
            label += " (from products)"
            name = "measured_products"
        if fill_missing not in ("error", "csv", "omit"):
            raise ValueError(f"unknown fill_missing: {fill_missing!r}")
    else:
        fr = table.fractions

    return Scenario(name, label,
                    fractions=fr, excise_threshold=excise_threshold, mode=mode,
                    residuals=dict(residuals or {}),
                    residual_excise_threshold=residual_excise_threshold)


def band_channels(band: str = "dtv") -> list[int]:
    """ATSC channel numbers spanning a band."""
    if band == "dtv":
        return [c for c in range(14, 37)]
    if band == "all":
        # tile 400-800 MHz in 6 MHz slices using the same edge convention
        return [c for c in range(2, 69) if chn.channel_edges(c)[1] > 400.0
                and chn.channel_edges(c)[0] < 800.0]
    raise ValueError(band)


def uniform(f: float, band: str = "dtv", excise: bool = False,
            mode: str = "time", residual: float = 0.0) -> Scenario:
    """Uniform masked fraction f across a band.

    band='dtv'  : ATSC channels 14-36 (470-608 MHz), the pilot-proxy survey band
    band='all'  : the entire CHIME band, approximated by ATSC-width slices
                  from 400-800 MHz (channels -2..68 in extended numbering)
    residual    : uniform residual-to-thermal ratio r left in the kept data
    """
    chans = band_channels(band)
    tag = "excised" if excise else "masked"
    res = {c: float(residual) for c in chans} if residual else {}
    rtag = f", r={residual:g}" if residual else ""
    return Scenario(f"uniform{int(round(100 * f))}_{band}",
                    f"{100 * f:.0f}% {tag}, {band} band{rtag}",
                    fractions={c: f for c in chans},
                    excise_threshold=(f if excise else 1.01), mode=mode,
                    residuals=res)


def at_threshold(per_channel: dict[int, tuple[float, float]],
                 eta: float | None = None, excise_threshold: float = 0.5,
                 residual_excise_threshold: float = np.inf,
                 mode: str = "time") -> Scenario:
    """Scenario from a detector operating point: ``{channel: (f, r)}``.

    This is the constructor the threshold study wants. A pilot-proxy threshold
    sweep produces, per channel, a masked fraction ``f`` and the residual ``r``
    left in what survives; feeding both here prices the two halves against each
    other, so the required time as a function of threshold has an interior
    minimum instead of running away to "never mask".
    """
    fr = {int(c): float(v[0]) for c, v in per_channel.items()}
    rs = {int(c): float(v[1]) for c, v in per_channel.items()}
    tag = "" if eta is None else f" (eta={eta:g})"
    return Scenario(f"threshold{'' if eta is None else f'_{eta:g}'}",
                    f"Detector operating point{tag}",
                    fractions=fr, residuals=rs,
                    excise_threshold=excise_threshold,
                    residual_excise_threshold=residual_excise_threshold,
                    mode=mode)


def from_mask_decisions(decisions, excise_threshold: float = 0.5,
                        residual_excise_threshold: float = np.inf,
                        mode: str = "time", force: bool = False) -> Scenario:
    """Apply the mask only on channels where it pays.

    Each :class:`baonoise.residual.MaskDecision` carries both outcomes, so a
    channel it declines enters the scenario unmasked (f = 0 and the *full*
    contamination) rather than being dropped. Leaving it out would quietly
    credit the forecast with contamination nobody removed.

    ``force=True`` masks every channel regardless, which is what a uniform
    threshold does and what the comparison in the chapter needs.
    """
    fr, rs, masked = {}, {}, []
    for d in decisions:
        if force or d.should_mask:
            fr[d.channel], rs[d.channel] = d.f, d.r_masked
            masked.append(d.channel)
        else:
            fr[d.channel], rs[d.channel] = 0.0, d.r_unmasked
    tag = "all channels" if force else f"{len(masked)}/{len(fr)} channels"
    return Scenario("selective" if not force else "uniform",
                    f"Mask applied to {tag}",
                    fractions=fr, residuals=rs,
                    excise_threshold=excise_threshold,
                    residual_excise_threshold=residual_excise_threshold,
                    mode=mode)


def single_channel(ch: int, f: float, keep: bool = True,
                   mode: str = "time") -> Scenario:
    """One contaminated channel; keep=True integrates through it (noise
    penalty), keep=False excises it (volume penalty)."""
    thr = 1.01 if keep else 0.0
    verb = "kept" if keep else "excised"
    return Scenario(f"ch{ch}_{int(round(100 * f))}_{verb}",
                    f"ch{ch} {100 * f:.0f}% masked ({verb})",
                    fractions={ch: f}, excise_threshold=thr, mode=mode)

# bao-noise-tolerance

**How much observing time does RFI masking cost a 21 cm BAO survey?**
A general framework + tool: feed it per-channel masked-time fractions, get
back required integration time, detection significance, and the penalty
relative to an RFI-free survey.

## Quickstart (seconds; the CHIME Fisher bank ships with the repo)

```python
from baonoise import api

fc = api.load()                                   # CHIME bank + RadioFisher
mask = {17: 0.33, 30: 0.97, 31: 0.24}             # ATSC channel -> masked frac
api.required_time(fc, mask, target=5.0)           # survey-level 5-sigma BAO
api.required_time(fc, mask, target=5.0, zbin=6)   # worst bin (z=1.40-1.51)
api.significance(fc, 2.0, mask)                   # sigma after 2 on-sky yr
api.tolerance_curve(fc, band="dtv")               # years-to-target vs masking
```

`examples/minimal_example.py` runs this end-to-end (~2 s). In per-channel
scenarios, channels masked at or above the excision threshold (default 50%) are
excised and priced as survey volume; the rest cost effective integration
time. Uniform tolerance curves deliberately retain the affected slices as a
lost-time stress test. Pass an explicit `excise_threshold` to
`api.scenario_from(uniform=...)` when uniform excision is the intended policy.
Everything below explains the physics, the CHIME + DTV worked example, and
how to regenerate or extend it.

[pilot-proxy](https://github.com/WVURAIL/pilot-proxy) measures, per ATSC DTV
channel, the fraction of observing time an F-statistic positive excess flags
the channel as contaminated, i.e. the fraction of data that gets masked.
What it cannot tell you is what that masking *costs* in observing time. This
project closes the loop: it feeds per-channel masked-time fractions into a
[RadioFisher](https://github.com/djgormley/RadioFisher) Fisher forecast for
CHIME and inverts it into a **noise-tolerance statement**: the observing time
required to reach a target BAO detection significance, as a function of how
much data masking removes.

The headline case: ATSC channel 30 (566–572 MHz, 21cm z ≈ 1.48–1.51) is
masked ~97% of the time. No amount of integration rescues that slice; it
must be excised, and the forecast prices the excision in survey volume
instead. Most channels are masked at the 1–3% level, and the forecast shows
the survey barely notices.

## Method

1. **Fisher bank.** RadioFisher's CHIME configuration ('yCHIME', cylinder
   interferometer mode, 25,000 deg², T_inst = 50 K, 400–800 MHz in Δz = 0.1
   bins) is evaluated once per redshift bin on a log grid of total
   integration times, 1–10⁶ on-sky hours. Thermal noise enters the Fisher
   integrand only through P_N ∝ T_sys²/t_tot, so every masking scenario and
   every requested observing time can be evaluated instantly from the bank
   by rescaling each bin's effective time and volume. Elements are
   interpolated as F(t)/t² in log t (exact in the noise-dominated limit).

2. **Masking → per-bin factors.** A channel masked a fraction *f* of the
   time keeps t_eff = t·(1−f) of its integration; its noise power grows by
   1/(1−f). Channels masked above an excision threshold (default 50%) are
   dropped entirely: that costs the overlapped bandwidth fraction of the
   bin's Fisher information (volume), but leaves the surviving band's noise
   untouched, matching real practice for persistent transmitters like
   ch24/ch30. Within a bin, surviving slices combine under one of two
   conventions:
   * `time` (default): w̄ = ⟨1−f⟩, sample counting / inverse-variance
     weighting; the standard forecasting assumption.
   * `fourier`: w̄ = 1/⟨1/(1−f)⟩: radial Fourier modes see the arithmetic
     mean of per-slice noise power; pessimistic. The two bracket reality and
     agree when masking is uniform across the bin.

3. **Detection metric.** BAO detection significance = A/σ(A) with fiducial
   A = 1, where A is RadioFisher's BAO wiggle amplitude. Following the
   published RadioFisher analysis (`plotting/plot_Abao_zbins.py`; Bull,
   Ferreira, Patel & Santos 2015, ApJ 803, 21), σ(A) is marginalised over
   {b_HI, f, α_⊥, α_∥} per redshift bin and a shared σ_NL, with
   {T_b, σ₈, n_s} fixed (externally constrained); A is shared across bins.

4. **Inversion.** Required time to a target (5σ detection, 3σ evidence) is
   found by root-finding S(t) = target. Times are quoted in on-sky years at
   the Overview normalization (1 yr = 8,760 on-sky hours); the CSVs carry
   raw hours, and `survey.DUTY_2019_PRACTICE` (0.152, literature-anchored)
   converts to calendar years at demonstrated CHIME practice.

## Inputs

The fiducial "measured" scenario uses the survey products committed in
pilot-proxy (`data/provenance/survey_stratum_20260718/
survey_quarterly_rates_all23.csv`, vendored under `data/`): per channel, the
exposure-weighted mean of quarterly `hi_rate_all` (weights = `n_valid_frames`).
Channels 24 and 30 are *refused* there (no calibrated zero point; the
transmitter is essentially always on); they are assigned a 97% masked
fraction, which puts them above the excision threshold. Exposure-weighted
fractions, for reference: ch17 ≈ 33%, ch31 ≈ 24%, ch32 ≈ 14%, ch35 ≈ 14%,
ch33 ≈ 10%, everything else ≲ 4%.

## Results

Run `scripts/run_forecast.py` to regenerate. Headline numbers from the
current bank (see `out/required_times.csv`, `out/results.md` for the full
table):

Fiducial configuration: **the CHIME collaboration's own BAO forecast**
(Amiri et al. 2022, ApJS 261, 29, Appendix A; as-built 4×256 geometry,
T_sys = 55 K, 31,000 deg², Planck-2018, per-bin BAO amplitudes), which the
pipeline reproduces before masking is applied (per-bin σ(D_V)/D_V of
0.47–1.04% at 1 on-sky year, matching Fig. 31).

| Scenario | Time penalty vs clean (any BAO target) | σ(D_A)≤2% in the z=1.40–1.50 bin |
|---|---|---|
| No masking | ×1.00 | 0.32 on-sky yr |
| **Measured masking (representative table)** | **×1.03** | **0.43 on-sky yr** |
| 50% of DTV band masked | ×1.15 | 0.64 on-sky yr |
| 97% of DTV band masked | ×1.26 | ~10 on-sky yr |
| 50% of the *entire* band masked | ×2.00 (exactly, noise-dominated) | – |
| ch30 (97% masked) excised | ×1.008 | 0.36 on-sky yr |
| ch30 kept in analysis (Fourier convention) | ×1.06 | 1.98 on-sky yr |

Times are on-sky years at the Overview normalization ("1 yr" = 8,760
on-sky hours, no duty factor; Amiri et al. 2022 Table 2); multiply by
~6.6 for calendar years at the demonstrated 2019-practice duty factor
(0.152 = 94/309 days × night-only; Amiri et al. 2025; their additional
38.7% night masking includes RFI and must not be double-counted against
these scenarios). Read ratios rather than absolute times, since Fisher forecasts are
idealized. The measured DTV masking costs ~3% extra
integration at survey level and ~34% extra for the distance measurement in
the worst-hit redshift bin (z≈1.45, where ch30, ch24 and the 10–33%-masked
ch17/31/32/33/35 all live). Integrating through a 97%-masked channel
instead of excising it would need ×33 the time (ten years of integration
buys 3.6 clean-equivalent months), while excision costs a 16% volume hit
to one Δz=0.1 bin (0.8% at survey level). Even the absurd worst case (97%
of the whole DTV band) only multiplies survey-level time by ×1.26, because
400–470 and 608–800 MHz carry the detection. A second bank built on the
Bull et al. (2015) design study ships alongside for cross-configuration
robustness checks.

Figures produced in `out/`:

* `fig1_significance_vs_time`: survey BAO significance vs observing time
  for the measured masking and uniform DTV-band masking scenarios.
* `fig2_required_time_vs_masking`: the noise-tolerance curve, years to
  3σ/5σ vs the masked fraction of the DTV band, with the 50% stress case
  annotated.
* `fig3_channel_masking`: measured per-channel masked fractions and the
  integration-time multiplier 1/(1−f) each implies.
* `fig4_perbin_significance`: per-redshift-bin significance profile,
  showing where ch24/ch30 excision and the ch31–35 cluster bite (z ≈ 1.3–1.7).

## Usage

```bash
git clone https://github.com/djgormley/RadioFisher   # sibling checkout
git clone https://github.com/WVURAIL/bao-noise-tolerance
cd bao-noise-tolerance
pip install -e ".[test]"        # numpy, scipy, matplotlib, camb, pytest

export RADIOFISHER_DIR=../RadioFisher   # optional; sibling dir is found automatically

python scripts/build_bank.py            # ~20 min on 2 cores, writes data/fisher_bank_chime.npz
python scripts/verify_bank.py           # interpolation + physics sanity checks
python scripts/run_forecast.py          # figures + CSVs + results.md in out/
python scripts/check_paper_numbers.py   # every number in the paper regenerates from out/*.csv
python -m pytest tests/ -q
```

To swap in new masking measurements, point `baonoise.scenarios.measured()`
at any CSV with `atsc_channel`, `n_valid_frames`, `hi_rate_all` columns
(quarterly or otherwise), or build a `Scenario` directly from a
`{channel: masked_fraction}` dict. The bank does not need to be rebuilt when
masks change, only when the instrument/survey definition does.

## Where the RFI model lives

The physics is implemented **inside the RadioFisher fork** (branch
`rfi-noise-model`) as three optional experiment-dict hooks, all no-ops when
absent: `expt['noise_freq_weight']` (a w(ν) callable, surviving time
fraction per frequency, NaN = excised slice), `expt['noise_freq_mode']`
(`'invvar'` or `'fourier'` band averaging), and `expt['vol_frac']`
(surviving bandwidth fraction, scales bin volume). `Scenario.freq_weight_fn()`
produces the callable; `Forecast.sigma_A_direct()` drives the hooks with
direct `rf.fisher()` calls.

The Fisher bank is the fast path: because both hooks reduce, per bin and
scenario, to a time rescaling plus a volume scale, bank evaluation is
mathematically identical to the in-fork hooks. `scripts/verify_bank.py`
check 5 enforces this (agreement to <0.2% on σ_A, including a full-survey
comparison). Use the bank for sweeps and inversions; use the hooks when you
drive RadioFisher directly or extend the model (e.g. k∥-resolved noise from
mask gaps), at which point the bank gains a dimension.

No Fortran CAMB, MPI, or Python 2 required. The fork itself runs on modern
scipy/numpy; `baonoise.compat`'s shims remain only as a fallback for running
against *upstream* (philbull) RadioFisher, and `baonoise.pkcache` generates
RadioFisher's P(k) cache with the Python `camb` package.

## Where the masking fractions come from

A masking fraction is only meaningful next to the rule that produced it, and
the two available sources disagree by up to 87× on the same channels:

```
python3 scripts/compare_mask_tables.py 506.npz 521.npz 537.npz --forecast
```

| ch | vendored CSV | survey products | ratio |
|---|---|---|---|
| 34 | 0.0231 | 0.9909 | 43× |
| 35 | 0.1380 | 0.8370 | 6× |
| 36 | 0.0115 | 0.9991 | 87× |

This is **not** the same detector at a different threshold. The products'
contract says `threshold_mode: none`, `mask_source: positive_excess`,
`equivalent_mask_rule: F > mu0`, the raw zero-excess rule. No cut on any
statistic in the products reproduces the CSV: matching it would need η = 1.40
on ch34, 15.2 on ch35 and 20.8 on ch36, and masking at the median gives 50% by
construction. The CSV's per-channel exposures also don't match the products
(37,474 frames for ch34 against 39,017) and run 7.7k–37k across the band, so it
came from a different trawl.

The defect is that nothing recorded which. `measured_mask_fractions` reads a
column named `hi_rate_all` and hands it to the Fisher machinery, so the
forecast silently inherits an unidentified detector.
`channels.MaskTable` fixes that by making provenance part of the value:

```python
from baonoise import channels as chn, scenarios

prod = chn.mask_table_from_products(["506.npz", "521.npz", "537.npz"])
prod.rule           # 'F > mu0; mu0 = 2*target_norm_sq/ref_norm_sum_sq'
prod.is_traceable   # True
chn.measured_mask_table().is_traceable   # False (rule 'unrecorded')

scenarios.measured(products=[...])       # forecast on the detector's own decision
```

Three gates stop the ways this goes wrong quietly. Products decided by
different **kernels** (`kernel_sha256` or mask rule) are refused, because
averaging them averages two detectors; a differing harness *package* version
over one kernel is only a note, since the kernel is what decides each frame.
Two products covering the **same channel** are refused rather than letting the
second silently overwrite the first. And `scenarios.measured(products=…)`
refuses by default to backfill uncovered channels from the CSV, since a table
half from one detector and half from another is the original bug in a new
place; `fill_missing='csv'` accepts it and tags the scenario, `'omit'`
forecasts on the covered channels alone.

At survey level the choice is ×1.032 (CSV) against ×1.063 (products where
measured); this is small, but the chapter has to say which detector it means.

## Residual contamination: the other half of the cost

Masking cost alone is monotone (mask less, integrate less), so a forecast
built from `scenarios` by itself is minimised by never masking at all. The
missing term is the contamination that *survives* the mask. A detector
threshold sets both: lower it and you remove more data but keep a cleaner
remainder; raise it and you do the reverse. Only with both terms is the
threshold something you can optimise rather than assume.

`baonoise.residual` supplies the second term. A slice carrying residual power
*r* relative to thermal has its noise raised by (1 + *r*), which is exactly a
loss of effective time, so it folds into the existing machinery as

```
(1 - f)  ->  (1 - f) / (1 + r)
```

and nothing else changes. `residuals` defaults to empty, so every number in
`out/` is reproduced exactly.

The chain from a pilot-proxy survey product to *r* has four terms. Three are
measured or fixed; one is not:

| term | ch35 (596.5 MHz) | source |
|---|---|---|
| shelf, transmitter on | −10.6 dB | measured |
| kept-frame bound | −26.2 dB | p90 of the transmitter-off epoch; a kept frame has no pilot excess, so its shelf is *bounded* by the single-frame sensitivity floor rather than measured |
| ground / m=0 filter | −20.6 dB | measured (see the split below) |
| delay filter | −3.6 to −11.4 dB | choice: 11.4 dB at CHIME's deployed τ_cut = 200 ns, only 3.6 dB if the filter must preserve the first acoustic peak |
| coherence | +27.5 dB | measured: τ_c = 46 min (32–58 at 68%) |
| **r = P_res/P_N** | **0.59** (0.41–0.74) | |

The ground-filter term comes from a three-level nested decomposition of the
shelf power, keyed on **sidereal day** and acquisition. For ch35, over 5,647
acquisitions on 1,438 sidereal days:

| timescale | share of shelf power | fate |
|---|---|---|
| constant | 94.52% | m = 0, removed |
| inter-day drift | 4.62% | m = 0 within each day, removed |
| intra-day (acquisition to acquisition) | 0.85% | **survives** |
| sub-acquisition | 0.02% | survives, averages like noise |

The sidereal-day boundary is the one that matters, and it is why splitting at
the acquisition instead (which lumps day-to-day drift in with intra-day
variation) understates the filter by ~7 dB and then double-counts that same
power in the coherence term.

Coherence is what remains. Thermal noise averages down; a residual only
averages down over its own correlation time τ_c, so a component is amplified by
`n_coh` = τ_c / T_frame relative to thermal, and, usefully, that amplification
is *independent of total integration time*, so the residual saturates rather
than growing without bound. τ_c is capped at one sidereal day by construction
(anything slower is already gone as m = 0), and the two surviving populations
carry their own coherence rather than one lumped factor.

### Measuring τ_c, or declining to

`residual.correlation_time` measures τ_c from a noise-corrected
**same-sidereal-day structure function** of the acquisition-mean shelf power,
read at the (1 − 1/e) crossing, with a day-block bootstrap for the interval.
Two details are load-bearing. Each acquisition mean carries estimation noise
`V_fast / n_frames`, which inflates every squared difference; subtracting it is
what stops sparsely sampled acquisitions from faking a short correlation time.
And the bootstrap must resample **whole days with their time ordering intact**;
shuffling acquisitions within a day destroys exactly the structure being
measured and drives τ_c to the shortest lag bin.

It refuses rather than guesses. Four gates: enough same-day pairs and days; a
crossing at a lag the acquisition cadence can actually resolve; τ_c stable
across trim level; and the surviving power fraction stable across trim level.
The stability gates separate a stationary shelf from an episodic one: on ch35
the answer moves by ×1.08 as the trim runs 75–95%, while on ch34 and ch36,
whose transmitters burst rather than sit on, the top 1% of frames carry 99.5%
and 91.4% of the linear variance and the "answer" is entirely an artifact of
where the tail was cut. Those channels get a refusal, and the refusal takes no
ground-filter credit either: the same non-stationarity that makes τ_c
unmeasurable makes the variance split unmeasurable, since both are moments of
the same process. The fallback is all shelf power surviving at the
sidereal-day cap, an actual bound, flagged in the chain as `[BOUND]`.

```
python3 scripts/residual_budget.py 521.npz --off-through 2021-08 --plot
```

For ch35 that gives r = 0.59, and the cost lands at:

| | masking only | + residual at measured τ_c |
|---|---|---|
| survey | ×1.032 | **×1.133** (1.111–1.147) |
| z = 1.40–1.50 | ×1.347 | **×2.14** (1.90–2.34) |

against ×1.25 and ×26 if τ_c had to be quoted at the sidereal-day cap. Measuring
it is worth an order of magnitude in the worst bin, which is why the estimator
exists rather than a conservative constant.

## Is masking worth it?

Lowering the residual is not the test. Masking a fraction *f* raises noise by
1/(1−*f*) through lost integration and lowers it by (1+r_masked)/(1+r_unmasked)
through reduced contamination, so it pays only when

```
net = (1 + r_unmasked) / (1 + r_masked) * (1 - f)   >   1
```

`residual.mask_benefit` evaluates that per channel and `residual.threshold_sweep`
traces it as the detector's threshold moves, which is the framework's whole
purpose: an operating point mapped onto a science decision.

On the two channels with a measurable transmitter-off epoch, at the deployed
threshold (`F > mu0`, no threshold above it):

| ch | *f* | r unmasked → masked | noise gain | data cost | net | break-even *f* |
|---|---|---|---|---|---|---|
| 35 | 0.988 | 20.7 → 0.563 | 13.9× | 85.7× | **0.162** | 0.928 |
| 34 | 0.995 | 51.4 → 5.03 | 8.7× | 214.9× | **0.041** | 0.885 |

Both say do not mask: the cleaning is real (14× and 9×) but it is bought at
86× and 215× the data. A steady transmitter offers frame selection no variance
to exploit, which is the same fact as "94.5% of the shelf power is
time-constant" seen from the cost side rather than the coherence side. Break-even
needs *f* ≲ 0.93, so the deployed threshold is past the optimum rather than
wildly wrong.

`scenarios.from_mask_decisions` builds the selective policy: mask where it
pays, and carry the *full* contamination where it does not, since dropping a
declined channel would quietly credit the forecast with contamination nobody
removed. `force=True` gives the uniform policy for comparison.

## Assumptions and caveats

* **A residual treated as excess variance is conservative only if it is
  incoherent.** A coherent residual is a *bias* on the recovered BAO
  amplitude, and a bias is not bounded by the (1 + *r*) construction.
  `residual_excise_threshold` drops a slice rather than integrating through it
  when *r* is large. Propagating a bias properly needs the per-mode Fisher
  integrand (a `P_res` hook in the RadioFisher fork alongside the existing
  `noise_freq_weight` / `vol_frac` hooks), which is not written yet.
* **Masking is treated as a per-channel duty cycle**, uncorrelated with the
  sky signal. Time-correlated masking (e.g. tropospheric ducting seasons)
  reduces effective time exactly the same way at this level of forecast;
  what it can add is mode coupling not modeled here.
* **Excised 6 MHz channels are priced as lost volume.** Frequency gaps also
  mix radial Fourier modes; a quadratic-estimator analysis handles this with
  modest extra losses. The `fourier` convention gives you the pessimistic
  bracket for *kept* noisy channels.
* **RadioFisher's CHIME is the Bull et al. (2015) design**: 5 cylinders ×
  256 feeds (1,280 receivers), 80 m cylinders, 25,000 deg², T_inst = 50 K.
  The as-built instrument (4 × 256 dual-pol over 78 m instrumented, ~22 m
  spacing) is available via `layout.ensure_chime_nx(..., layout="asbuilt")`,
  but the fiducial keeps the published spec. Absolute times shift with the
  spec; the *ratios* vs masking (the object of this project) are robust.
* **Baseline density n(u) is synthesized** from the layout above using the
  exact recipe in RadioFisher's `process_chime_baselines.py` (the original
  file and the auxiliary-files URL are no longer distributed).
* Foreground residuals at RadioFisher's fiducial ε_FG = 10⁻⁶; nonlinear
  cutoff k_NL(0) = 0.14 Mpc⁻¹; Planck-2013-era fiducial cosmology
  (as shipped in `radiofisher.experiments.cosmo`). The 3σ/5σ **amplitude**
  metric is a detection statement rather than a distance-precision statement;
  `scripts/verify_bank.py` also reports per-bin α_⊥/α_∥ errors.

## Layout

```
src/baonoise/
  compat.py      modern-stack shims + RadioFisher import
  pkcache.py     python-camb P(k) cache in RadioFisher's format
  layout.py      CHIME cylinder baseline density n(u) generator
  survey.py      CHIME experiment dict, z bins, time conversions
  channels.py    ATSC <-> frequency/z mapping, pilot-proxy ingestion
  scenarios.py   masking scenarios -> per-bin (volume, time) factors
  fisherbank.py  per-bin Fisher matrices on a time grid (+ interpolation)
  forecast.py    sigma(A), significance curves, required-time inversion
  plots.py       publication figures
scripts/         build_bank / verify_bank / run_forecast / tests
data/            P(k) cache, n(u), vendored pilot-proxy rates, Fisher bank
out/             figures, CSVs, results.md
```

## Citations

If this feeds a paper, cite RadioFisher (P. Bull, P. G. Ferreira, P. Patel
& M. G. Santos, ApJ 803, 21 (2015), arXiv:1405.1452) for the forecasting
formalism, and pilot-proxy (WVURAIL) for the masking measurements.

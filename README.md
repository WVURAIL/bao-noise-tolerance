# bao-noise-tolerance

**How much observing time does RFI masking cost a 21 cm BAO survey?**
A general framework + tool: feed it per-channel masked-time fractions, get
back required integration time, detection significance, and the penalty
relative to an RFI-free survey.

## Quickstart (seconds; two CHIME Fisher banks ship with the package)

```python
from baonoise import api, scenarios

fc = api.load()                                   # Planck-2018 (default)
pact = api.load(cosmology="pact2025")             # matched P-ACT-LB bank
mask = {17: 0.33, 30: 0.97, 31: 0.24}             # ATSC channel -> masked frac
api.required_time(fc, mask, target=5.0)           # survey-level 5-sigma BAO
api.required_time(fc, mask, target=5.0, zbin=6)   # worst bin (z=1.40-1.50)
api.significance(fc, 2.0, mask)                   # sigma after 2 on-sky yr
api.tolerance_curve(fc, band=scenarios.DTV_BAND)  # years-to-target vs masking
```

`examples/minimal_example.py` runs this end-to-end (~2 s). Banks must use the
strict schema-v2 format; pre-provenance banks are rejected and must be rebuilt.
Both packaged banks record explicit baryon, cold-dark-matter, and neutrino
densities, code-tree identities, backend capabilities, the canonical H I
profile, cache/input hashes, and an explicit `forecast` kind. In per-channel
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
[RadioFisher](https://github.com/WVURAIL/RadioFisher) Fisher forecast for
CHIME and inverts it into a **noise-tolerance statement**: the observing time
required to reach a target BAO detection significance, as a function of how
much data masking removes.

The headline case: ATSC channel 30 (566–572 MHz, 21cm z ≈ 1.48–1.51) is
masked ~97% of the time. No amount of integration rescues that slice; it
must be excised, and the forecast prices the excision in survey volume
instead. Most channels are masked at the 1–3% level, and the forecast shows
the survey barely notices.

## Method

1. **Fisher bank.** The shipped fiducial uses the CHIME Overview
   configuration (Amiri et al. 2022): as-built 4×256 geometry,
   31,000 deg², total system temperature 55 K, and Planck-2018 cosmology.
   RadioFisher is evaluated once per redshift bin on a log grid of total
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
   A = 1. The shipped Overview bank follows Appendix A: each redshift bin has
   an independent BAO amplitude, marginalised over its nuisance parameters,
   and bin significances add in quadrature. The Bull-2015 configuration uses
   its original shared-A treatment, but its banks obey the same strict-v2
   provenance contract.

4. **Inversion.** Required time to a target (5σ detection, 3σ evidence) is
   found by root-finding S(t) = target. Times are quoted in on-sky years at
   the Overview normalization (1 yr = 8,760 on-sky hours); the CSVs carry
   raw hours, and `survey.DUTY_2019_PRACTICE` (0.152, literature-anchored)
   converts to calendar years at demonstrated CHIME practice.

## Inputs

The fiducial "measured" scenario uses the survey products committed in
pilot-proxy (`data/provenance/survey_stratum_20260718/
survey_quarterly_rates_all23.csv`, distributed as package data): per channel,
the exposure-weighted mean of quarterly `hi_rate_all` (weights =
`n_valid_frames`).
Channels 24 and 30 are *refused* there (no calibrated zero point; the
transmitter is essentially always on); they are assigned a 97% masked
fraction, which puts them above the excision threshold. Exposure-weighted
fractions, for reference: ch17 ≈ 33%, ch31 ≈ 24%, ch32 ≈ 14%, ch35 ≈ 14%,
ch33 ≈ 10%, everything else ≲ 4%.

The CHIME system-temperature calibration
(`20190530_and_20190614_system_temperature_measurement.h5`: Tsys over
400--800 MHz from two 2019 calibrator transits, median 54.6 K, with
receiver temperatures, effective area, and Jy/K) lives in the team
SharePoint under `RFI Mitigation/`, not in this public repository. The
packaged forecasts use Tsys_tot = 55 K and work in ratio units
throughout, so the file changes no verdict; it matters for
absolute-unit conversion of shelf and floor levels and for validating
the forecast Tsys. `scripts/tsys_calibration.py` reads the file
and runs that validation (in the DTV band the measured median
is 56.5 K; 55 K sits inside the measured 10th--90th percentile).

## Results

Run `scripts/run_forecast.py` to regenerate. Headline numbers from the
current bank (see `out/required_times.csv`, `out/results.md` for the full
table):

Fiducial configuration: **the CHIME collaboration's own BAO forecast**
(Amiri et al. 2022, ApJS 261, 29, Appendix A; as-built 4×256 geometry,
T_sys = 55 K, 31,000 deg², Planck-2018, per-bin BAO amplitudes), which the
pipeline reproduces before masking is applied (per-bin σ(D_V)/D_V of
0.47–1.03% at 1 on-sky year, matching Fig. 31).

| Scenario | Time penalty vs clean (any BAO target) | σ(D_A)≤2% in the z=1.40–1.50 bin |
|---|---|---|
| No masking | ×1.00 | 0.32 on-sky yr |
| **Measured masking (representative table)** | **×1.03** | **0.42 on-sky yr** |
| 50% of DTV band masked | ×1.15 | 0.63 on-sky yr |
| 97% of DTV band masked | ×1.25 | ~10 on-sky yr |
| 50% of the *entire* band masked | ×2.00 (exactly, noise-dominated) | – |
| ch30 (97% masked) excised | ×1.008 | 0.36 on-sky yr |
| ch30 kept in analysis (Fourier convention) | ×1.06 | 1.96 on-sky yr |

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
of the whole DTV band) only multiplies survey-level time by ×1.25, because
400–470 and 608–800 MHz carry the detection. The package includes matched
Planck-2018 and P-ACT-LB strict-v2 CHIME banks. The source repository also
keeps matched-grid strict-v2 Bull-2015 banks at $\epsilon_{FG}=10^{-6}$ and
$10^{-5}$ for the foreground sensitivity check; these are research comparison
artifacts rather than installed defaults. In the matched cosmology comparison,
the clean five-sigma time is 208.8 hours for Planck-2018 and 186.9 hours for
P-ACT-LB, while relative masking penalties differ by at most 0.420%. In the
Bull-2015 foreground check, raising $\epsilon_{FG}$ by a factor of ten changes
the clean survey time by 3.3% and the tested survey-level masking penalties by
at most 0.43%.

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
git clone https://github.com/WVURAIL/bao-noise-tolerance
cd bao-noise-tolerance
pip install -e ".[test]"        # numpy, scipy, matplotlib, pytest

# The shipped CHIME bank and masking rates work without RadioFisher.
python examples/minimal_example.py
python scripts/run_forecast.py          # figures + CSVs + results.md in out/
baonoise-forecast --uniform 0.25        # Planck-2018; JSON output
baonoise-forecast --cosmology pact2025 --uniform 0.25
baonoise-forecast --version             # baonoise-forecast 1.0.0

# RadioFisher is needed only to build or directly verify a bank. The three
# supported P(k) caches ship in the wheel; new or stale caches also require
# the optional Python CAMB dependency.
git clone https://github.com/WVURAIL/RadioFisher ../RadioFisher

export RADIOFISHER_DIR=../RadioFisher   # optional; sibling dir is found automatically
pip install -e ".[pk]"                 # only when generating/refreshing P(k)

baonoise-build-bank --help              # strict-v2 bank builder
bash scripts/rebuild_shipped_banks.sh   # the four shipped banks: exact recipe + pin re-stamp
python scripts/verify_bank.py           # interpolation + physics sanity checks
python scripts/check_paper_numbers.py   # every number in the paper regenerates from out/*.csv
python -m pytest tests/ -q
```

### Optional coherent-bias research banks

The `_Pres` workflows are useful but are not part of the installed forecast
API, and their large response banks are deliberately not shipped. They accept
only strict-v2 `bias_response` artifacts built with the CHIME-Overview-2022
profile, Planck-2018, and the unit normalization `P_res=1.0`. Build the exact
matched prerequisites locally before running the research scripts:

```bash
python scripts/build_bank.py --config chime2022 --cosmology planck2018 \
  --p-res 1.0 --dense-knee \
  --out data/fisher_bank_chime2022_pres_dense.npz
python scripts/build_bank.py --config chime2022 --cosmology planck2018 \
  --p-res 1.0 --kfg-fac 22 --dense-knee \
  --out data/fisher_bank_chime2022_pres_kfg22_dense.npz
python scripts/build_bank.py --config chime2022 --cosmology planck2018 \
  --p-res 1.0 --kfg-fac 44 --dense-knee \
  --out data/fisher_bank_chime2022_pres_kfg44_dense.npz
python scripts/build_bank.py --config chime2022 --cosmology planck2018 \
  --p-res 1.0 --kfg-fac 80 --dense-knee \
  --out data/fisher_bank_chime2022_pres_kfg80_dense.npz

python scripts/bias_tolerance.py --zeta 1.0
python scripts/plot_convergence.py --out out/
python scripts/three_worlds.py
```

Each script checks the artifact kind, cosmology, astrophysical profile,
unit-response normalization, and (where applicable) `kfg_fac` before reading
Fisher matrices. Missing or incompatible banks fail with the exact rebuild
command. The historical `bias_tolerance*.json` and `three_worlds.csv` outputs
made from absent pre-1.0 banks are not distributed; regenerate them only from
the locally built prerequisites above.

To swap in new masking measurements, point `baonoise.scenarios.measured()`
at any CSV with `atsc_channel`, `n_valid_frames`, `hi_rate_all` columns
(quarterly or otherwise), or build a `Scenario` directly from a
`{channel: masked_fraction}` dict. The bank does not need to be rebuilt when
masks change, only when the instrument/survey definition does. Uniform masks
use physical frequency intervals rather than synthetic channel identifiers:

```python
scenarios.uniform(0.5, scenarios.DTV_BAND)    # 470--608 MHz
scenarios.uniform(0.5, scenarios.CHIME_BAND)  # 400--800 MHz
```

Pass an explicit `excise_threshold` when a uniform band should be excised;
omitting it retains the band as a lost-time stress test.

## Where the RFI model lives

The physics is implemented **inside the supported WVURAIL RadioFisher
backend** as three experiment-dict hooks: `noise_freq_weight`,
`noise_freq_mode`, and `vol_frac`. RadioFisher publishes a versioned backend
API and immutable capability set. Direct masked calculations fail before doing
work when any required capability is absent; unknown keys are never treated as
successful masking. The hooks are no-ops when
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

No Fortran CAMB, MPI, or Python 2 is required. `baonoise.compat` also binds the
imported module to one checkout path, preventing code from one checkout from
being combined with configuration data from another. `baonoise.pkcache`
generates RadioFisher's P(k) table with Python CAMB. Its v2 header hashes the
resolved baryon, cold-dark-matter, and neutrino densities, CAMB settings and
table bytes. CAMB receives the authoritative `omnuh2_active` value directly;
an old `pythoncamb` sentinel is deliberately considered stale.

Each Fisher bank also records the exact computational-source manifest used
for its code fingerprint. For Bao this is `pyproject.toml` plus
`src/baonoise/*.py`; for RadioFisher it is `pyproject.toml`,
`radiofisher/*.py`, and `chime2021/experiments_CHIME.py`. Generated banks,
tests, documentation, paper text, and output tables are deliberately outside
that digest, avoiding a self-reference while every numerical input is hashed
separately in the bank provenance. Source text is canonicalized to LF before
hashing, so the identity is unchanged by the checkout platform's newline
convention.

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
  when *r* is large. The RadioFisher `P_res` hook propagates coherent bias into
  a `_Pres` response row. Such files are schema-v2 `bias_response` artifacts;
  the generic `Forecast`/`api.load()` path rejects them so the response cannot
  be accidentally marginalized as an ordinary parameter.
* **Masking is treated as a per-channel duty cycle**, uncorrelated with the
  sky signal. Time-correlated masking (e.g. tropospheric ducting seasons)
  reduces effective time exactly the same way at this level of forecast;
  what it can add is mode coupling not modeled here.
* **Excised 6 MHz channels are priced as lost volume.** Frequency gaps also
  mix radial Fourier modes; a quadratic-estimator analysis handles this with
  modest extra losses. The `fourier` convention gives you the pessimistic
  bracket for *kept* noisy channels.
* **The shipped fiducials use CHIME Overview 2022**, with its as-built geometry,
  BAO-shift-only flags, ε_FG = 0, and either Planck-2018 or P-ACT-LB
  cosmology. The Bull-2015 design remains an explicitly named configuration
  with provenance-complete research comparison banks.
  Absolute times shift with the instrument and foreground model; the masking
  ratios are the principal result.
* **Physical densities are explicit.** `omega_M_0` means total matter,
  including massive neutrinos. CAMB receives separate `ombh2`, `omch2`, and
  `omnuh2`; neutrino density is not hidden in cold dark matter and added again.
  Set `omega_M_0_includes_neutrinos=False` only when a source dictionary's
  matter total intentionally excludes neutrinos. An explicit density triplet
  is authoritative and must be supplied all-or-none.
* The 3σ/5σ **amplitude**
  metric is a detection statement rather than a distance-precision statement;
  `scripts/verify_bank.py` also reports per-bin α_⊥/α_∥ errors.

## Layout

```
src/baonoise/
  data/          installed banks, caches, baseline, rates, and product manifest
  api.py         high-level load/scenario/time interface
  resources.py   importlib.resources access to installed data
  products.py    external survey-product registry and resolution
  compat.py      modern-stack shims + backend capability/path binding
  pkcache.py     content-verified Python-CAMB P(k) caches
  layout.py      CHIME cylinder baseline density n(u) generator
  survey.py      CHIME experiment dict, z bins, time conversions
  channels.py    ATSC <-> frequency/z mapping, pilot-proxy ingestion
  scenarios.py   masking scenarios -> per-bin (volume, time) factors
  fisherbank.py  versioned forecast/bias artifacts + interpolation
  forecast.py    sigma(A), significance curves, required-time inversion
  residual.py    residual statistics, provenance, budgets, and policies
  incumbent.py   incumbent-flagger comparisons
  plots.py       publication figures
scripts/         reproducibility and strict-v2 bank-management commands
data/            strict-v2 research comparison banks
out/             figures, CSVs, results.md
```

## Scope and historical material

Detector-kernel design, non-pilot/control-frequency selection, and new survey
product generation belong in `pilot-proxy`; this repository validates and
prices the resulting measurements. The survey-plate figures (census
PSDs, per-channel histograms, the worked example, the coherence aids)
accordingly live in pilot-proxy's `analysis/`. The old in-repo non-pilot draft and mixed
dissertation task list were removed to avoid competing contracts. The small
[archived-roadmap note](docs/archive/legacy-roadmap.md) retains unresolved
cross-project dependencies; Git history retains the completed rationale.

Generated bank logs, status sentinels, private email drafts, and the compiled
paper PDF are likewise omitted. Source inputs, LaTeX, tests, and reproducible
tables remain tracked.

## Citations

If this feeds a paper, cite RadioFisher (P. Bull, P. G. Ferreira, P. Patel
& M. G. Santos, ApJ 803, 21 (2015), arXiv:1405.1452) for the forecasting
formalism, and pilot-proxy (WVURAIL) for the masking measurements.

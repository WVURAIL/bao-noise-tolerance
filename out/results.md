# Noise-tolerance forecast results

CHIME (RadioFisher 'yCHIME' spec) BAO forecast under ATSC DTV masking.
Times are on-sky hours / on-sky years at the Overview normalization
(1 yr = 8,760 hr, no duty factor; Amiri et al. 2022 Table 2). Multiply
on-sky years by ~6.6 (duty 0.152, the empirical 2019-practice fraction
from Amiri et al. 2025, RFI component excluded) for calendar years.
Absolute times
inherit the optimism of Fisher forecasting (smooth foreground residuals
eps_FG=1e-6, no calibration systematics); the robust currency is the
**time penalty relative to the clean survey**, last column below.

## Survey-level BAO detection (A/sigma_A, all z bins combined)

| scenario | label | sig_at_2yr | years_5sig | time_penalty_vs_clean |
|---|---|---|---|---|
| clean | No masking (RFI-free) | 63.57 | 0.0238 | 1.0 |
| measured | Measured pilot-proxy masking | 61.96 | 0.0246 | 1.032 |
| uniform50_dtv | 50% masked, dtv band | 57.88 | 0.0275 | 1.152 |
| uniform75_dtv | 75% masked, dtv band | 53.89 | 0.029 | 1.219 |
| uniform97_dtv | 97% masked, dtv band | 49.7 | 0.0299 | 1.256 |
| measured_fourier | Measured (Fourier-mode noise convention) | 61.89 | 0.0247 | 1.034 |
| uniform25_dtv | 25% masked, dtv band | 60.98 | 0.0256 | 1.076 |
| uniform90_dtv | 90% masked, dtv band | 51.01 | 0.0297 | 1.247 |
| uniform50_chime | 50% masked, chime band | 48.84 | 0.0477 | 2.0 |
| ch30_excised | ch30 97% masked (excised) | 63.02 | 0.024 | 1.008 |
| ch30_kept | ch30 97% masked (kept) | 63.19 | 0.0241 | 1.013 |
| ch30_kept_fourier | ch30 97% masked (kept) | 60.26 | 0.0254 | 1.064 |

Reading: the measured pilot-proxy masking costs the survey only
**3%
extra integration time** to any fixed BAO-amplitude target, and even masking
50% of the whole DTV band costs
15%.
The full survey reaches nominal 5-sigma almost immediately because 400-470 and
608-800 MHz are DTV-free and CHIME's mapping speed is enormous; the binding
constraints live at bin level, below.

## Bin-level targets in the DTV-affected redshifts

Per-bin BAO detection (5 sigma from that Delta-z=0.1 bin alone) and
sigma(alpha_perp) <= 2% (a per-bin BAO distance measurement):

| zbin | scenario | years_bin5sig | years_da2pct |
|---|---|---|---|
| 1.40-1.50 | clean | 0.175 | 0.315 |
| 1.40-1.50 | measured | 0.235 | 0.421 |
| 1.40-1.50 | measured_fourier | 0.237 | 0.424 |
| 1.40-1.50 | uniform50_dtv | 0.35 | 0.629 |
| 1.40-1.50 | ch30_excised | 0.201 | 0.359 |
| 1.40-1.50 | ch30_kept | 0.207 | 0.373 |
| 1.40-1.50 | ch30_kept_fourier | 1.091 | 1.963 |
| 1.60-1.70 | clean | 0.209 | 0.411 |
| 1.60-1.70 | measured | 0.286 | 0.547 |
| 1.60-1.70 | measured_fourier | 0.286 | 0.547 |
| 1.60-1.70 | uniform50_dtv | 0.419 | 0.821 |
| 1.60-1.70 | ch30_excised | 0.209 | 0.411 |
| 1.60-1.70 | ch30_kept | 0.209 | 0.411 |
| 1.60-1.70 | ch30_kept_fourier | 0.209 | 0.411 |

## The channel-30 story

Channel 30 (566-572 MHz, z = 1.48-1.51) is masked ~97% of the time. Options:

* **Integrate through it**: recovering clean-equivalent depth in that 6 MHz
  requires x33 the integration time: if the survey needs T years,
  that slice needs 33 T. Ten years of integration buys the depth of
  3.6 clean months. That is "never" for any practical purpose,
  and under the Fourier-mode noise convention the surviving 97% of the
  containing z-bin is dragged down with it (see ch30_kept_fourier rows).
* **Excise it** (what the forecast prices by default): costs the overlapped
  bandwidth, a 16% volume hit to the z=1.40-1.50 bin, a
  0.8%
  time penalty at survey level.

Excision wins by an enormous margin: throw the channel out and pay in volume,
never in noise.


## Matched fiducial-cosmology comparison

The provenance-complete, matched-grid Planck-2018 and P-ACT-LB banks require
208.8 and
186.9 clean on-sky hours, respectively, for the
survey-level 5-sigma target (-10.5%). Across the survey and
both reported bin-level metrics, the largest relative change in a masking
penalty is 0.420%.


## Files

required_times.csv, bin_level_targets.csv, tolerance_curve.csv,
perbin_significance_1onskyyr.csv, fiducial_comparison.csv,
fig1-fig4 (png + pdf).

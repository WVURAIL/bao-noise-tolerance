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
| clean | No masking (RFI-free) | 63.06 | 0.0241 | 1.0 |
| measured | Measured pilot-proxy masking | 61.46 | 0.0249 | 1.032 |
| uniform50_dtv | 50% masked, dtv band | 57.42 | 0.0278 | 1.151 |
| uniform75_dtv | 75% masked, dtv band | 53.46 | 0.0294 | 1.218 |
| uniform97_dtv | 97% masked, dtv band | 49.3 | 0.0303 | 1.255 |
| measured_fourier | Measured (Fourier-mode noise convention) | 61.4 | 0.025 | 1.034 |
| uniform25_dtv | 25% masked, dtv band | 60.52 | 0.026 | 1.075 |
| uniform90_dtv | 90% masked, dtv band | 50.61 | 0.0301 | 1.247 |
| uniform50_all | 50% masked, all band | 48.45 | 0.0483 | 2.0 |
| ch30_excised | ch30 97% masked (excised) | 62.51 | 0.0243 | 1.008 |
| ch30_kept | ch30 97% masked (kept) | 62.68 | 0.0245 | 1.013 |
| ch30_kept_fourier | ch30 97% masked (kept) | 59.78 | 0.0257 | 1.064 |

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
| 1.40-1.50 | clean | 0.177 | 0.318 |
| 1.40-1.50 | measured | 0.238 | 0.426 |
| 1.40-1.50 | measured_fourier | 0.239 | 0.428 |
| 1.40-1.50 | uniform50_dtv | 0.353 | 0.636 |
| 1.40-1.50 | ch30_excised | 0.203 | 0.363 |
| 1.40-1.50 | ch30_kept | 0.21 | 0.377 |
| 1.40-1.50 | ch30_kept_fourier | 1.103 | 1.984 |
| 1.60-1.70 | clean | 0.212 | 0.416 |
| 1.60-1.70 | measured | 0.289 | 0.554 |
| 1.60-1.70 | measured_fourier | 0.289 | 0.554 |
| 1.60-1.70 | uniform50_dtv | 0.424 | 0.831 |
| 1.60-1.70 | ch30_excised | 0.212 | 0.416 |
| 1.60-1.70 | ch30_kept | 0.212 | 0.416 |
| 1.60-1.70 | ch30_kept_fourier | 0.212 | 0.416 |

## The channel-30 story

Channel 30 (566-572 MHz, z = 1.48-1.51) is masked ~97% of the time. Options:

* **Integrate through it**: recovering clean-equivalent depth in that 6 MHz
  requires x33 the integration time: if the survey needs T years,
  that slice needs 33 T. Ten years of integration buys the depth of
  3.6 clean months. That is "never" for any practical purpose,
  and under the Fourier-mode noise convention the surviving 97% of the
  containing z-bin is dragged down with it (see ch30_kept_fourier rows).
* **Excise it** (what the forecast prices by default): costs the overlapped
  bandwidth, a 22% volume hit to the z=1.40-1.51 bin, a
  0.8%
  time penalty at survey level.

Excision wins by an enormous margin: throw the channel out and pay in volume,
never in noise.

## Files

required_times.csv, bin_level_targets.csv, tolerance_curve.csv,
perbin_significance_1onskyyr.csv, fig1-fig4 (png + pdf).

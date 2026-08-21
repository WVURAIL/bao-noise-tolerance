# Model-only forecast completion

This tranche closes the forecast tasks that do not require new telescope data.
It does not replace the missing pilot-to-shelf transfer measurement or the
visibility-domain calibration. Every result remains conditional on the
residual template carried by its response bank.

## Estimators

`scripts/bias_tolerance.py` exposes two estimator names. The name is written to
every complete-v1 JSON report; there is no implicit mode selected from the
bank.

### `perbin_appendix_a`

This is the pre-existing dissertation calculation. For each requested
redshift bin it removes `{b_HI, f, Tb, sigma8tot, n_s, pk}`, inverts that bin
alone, and reports targets in the remaining
`{A, sigma_NL, aperp, apar, fs8, bs8}` system. `DV` is also available as the
linear logarithmic combination `(2/3) aperp + (1/3) apar`. The historical
`split`, `bias_per_unit_r`, `condition`, and `stability` functions remain
available with unchanged signatures for the existing figure scripts.

### `overview_combined_multibin`

This path reproduces the Overview/Fig.-31 API sequence using the local,
path-bound RadioFisher checkout:

1. `expand_fisher_matrix(..., fsigma8=True)` projects every bank bin through
   the equation-of-state expansion used by the published workflow.
2. The native dilations are converted to physical `DA` in Gpc and `H/100`,
   then `transform_to_lss_distances` produces `DV` and `F`.
3. `combined_fisher_matrix` expands `A`, `bs8`, `fs8`, `DV`, `F`, and `_Pres`
   per redshift bin while retaining `sigma_NL` as a shared nuisance.
4. Every `_PresN` row is removed before inversion. Its cross-column is the
   bias source for contamination in bin `N`; it is never marginalized as a
   fitted parameter.

The bank always contains a unit response: its callable supplies the shape of
`C_res/P_N`, with no fitted or measured residual amplitude embedded in that
shape. The reported amplitude is applied exactly once, as `Delta A`, after the
unit bank response has been evaluated. In particular, an external `r_sys` must
not also be multiplied into `C_res`; doing both would count the same amplitude
twice.

The current RadioFisher spelling `sigma_8` is explicitly excluded along with
the historical global rows. The published script used the older spelling
`sigma8`; leaving that stale spelling in the current matrices produces a
numerically dependent row. The report records the full API sequence,
expanded parameters, exclusions, backend revision, and backend source digest.

The combined target `DV` is an absolute distance internally and the per-bin
target is logarithmic. This does not affect `r_tolerance = zeta sigma/|dtheta/dr|`
because a linear rescaling multiplies the numerator and denominator equally.
Absolute errors should nevertheless be compared only after converting both to
the same fractional convention.

## Residual-amplitude time families

The scalar `P_res=1` bank uses RadioFisher's unit thermal-noise template at
each evaluated integration time. The report requires one of two names:

### `noise_normalized_at_each_time`

The reported amplitude is `r(t) = P_res(t)/P_N(t)`. A unit response is tied to
the contemporaneous thermal power at every time. This is the convention used
by the legacy convergence figure; it is a family of differently normalized
physical templates, not one contaminant followed through time.

### `fixed_physical_at_reference_time`

The reported amplitude is `r_ref = P_res/P_N(t_ref)` at the explicitly supplied
`--reference-years`. Since the bank's thermal power obeys `P_N proportional to
1/t`, its response is multiplied by

```
r(t) / r_ref = t / t_ref.
```

The two families agree exactly at `t_ref`. Away from it they answer different
questions. A fixed physical contaminant becomes larger relative to thermal
noise as the integration grows.

## Stability and refusals

Every requested `(bin, time, parameter)` record contains the lower, central,
and upper evaluations at `t * (1-delta, 1, 1+delta)`. No failed point is
dropped. Each perturbation includes the Fisher eigenvalue range, cutoff,
condition number, number of discarded modes, statistical error, raw response,
time-family multiplier, reported response, and tolerance.

The refusal gate is applied to `r_tolerance_current_noise_ratio`, before the
deterministic `t/t_ref` conversion. This tests numerical movement and response
zero-crossings without falsely rejecting the fixed-physical family for the
physical time dependence that defines it. The JSON also reports the movement
of the final, reported-amplitude tolerance.

Possible refusal reasons are explicit strings, including a missing target,
invalid or negative Fisher information, target/response overlap with a
discarded null mode, failed inversion, zero/non-finite response, response sign
change, and excessive bank-native tolerance drift. Summary minima use accepted
points only and become `null` if every requested point is refused.

## Named analytic template banks

`scripts/build_bank.py` can build four unit-amplitude analytic families through
RadioFisher's callable `P_res(k, u, P_N, P_signal)` interface:

- `noise_shaped`: exactly `P_N`;
- `low_kparallel`: a Gaussian envelope in `|k u|`;
- `wedge_like`: unity inside a declared linear wedge boundary with a Gaussian
  roll-off outside it; and
- `k_shell_localized`: a Gaussian shell in `|k|`.

For example:

```bash
PYTHONPATH=src RADIOFISHER_DIR=/path/to/RadioFisher \
python3 scripts/build_bank.py \
  --config chime2022 --cosmology planck2018 \
  --residual-template low_kparallel \
  --template-param k_parallel_scale_mpc_inv=0.04 \
  --dense-knee --out data/fisher_bank_low_kparallel.npz
```

The callable object is also a JSON dictionary, so the bank records its family,
normalization, unit amplitude, and parameters in both `expt_overrides` and the
experiment-provenance digest. A named-template bank passes the same strict-v2
unit-response checks as the scalar noise-shaped bank.

The callable interface has no observing frequency, redshift-bin identity,
baseline vector, sidereal time, or measured visibility residual. Therefore it
cannot honestly construct a frequency-localized, sidereal-coherent, or
empirical visibility template. Those families are deliberately absent rather
than synthesized from unrelated coordinates. They remain external-data gates.

## CLI examples

The existing per-bin result is still the default:

```bash
PYTHONPATH=src python3 scripts/bias_tolerance.py \
  --bank data/fisher_bank_chime2022_pres_dense.npz \
  --estimator perbin_appendix_a \
  --time-scaling noise_normalized_at_each_time \
  --zeta 1 --years 0.25 1 5 10 \
  --json-format complete-v1 \
  --json out/perbin_noise_normalized.json
```

The combined fixed-physical family is explicit:

```bash
PYTHONPATH=src RADIOFISHER_DIR=/path/to/RadioFisher \
python3 scripts/bias_tolerance.py \
  --bank data/fisher_bank_chime2022_pres_dense.npz \
  --estimator overview_combined_multibin \
  --time-scaling fixed_physical_at_reference_time \
  --reference-years 1 --zeta 1 \
  --params DV F fs8 --years 0.25 1 5 10 \
  --json-format complete-v1 \
  --json out/combined_fixed_physical.json
```

`--bins` accepts zero-based bank-bin indices for a bounded evidence run. Its
default is every redshift bin overlapping the physical 470--608 MHz DTV band.
An explicit combined path fails if the local RadioFisher checkout is absent or
lacks any required API.

## Reproducible bounded evidence run

The committed `out/forecast_completion_evidence.json` is generated from only
three integration-time grid points and one reported redshift bin. The combined
estimator still assembles every redshift bin before its survey-wide inversion;
`--bin 6` limits the number of independent residual injections and output
records, not the information included in the combined Fisher matrix.

```bash
PYTHONPATH=src RADIOFISHER_DIR=/path/to/RadioFisher \
python3 scripts/build_bank.py \
  --config chime2022 --cosmology planck2018 --p-res 1 \
  --tmin 7000 --tmax 11000 --nt 3 --nproc 8 \
  --out /tmp/bias_response_small.npz

PYTHONPATH=src RADIOFISHER_DIR=/path/to/RadioFisher \
python3 scripts/forecast_completion_evidence.py \
  --bank /tmp/bias_response_small.npz \
  --bin 6 --years 0.9 1 1.1 --reference-years 1 \
  --out out/forecast_completion_evidence.json
```

The evidence command refuses a bank whose time grid would require
extrapolation at any lower or upper stability perturbation. It records the
bank's numerical-grid digest, the authenticated source digests, every
lower/central/upper evaluation, every acceptance or rejection, equality of the
two amplitude conventions at the reference time, and the `t/t_ref` scaling
away from it. Content hashes bind the evidence to the exact evaluator,
evidence runner, and bank-building wrapper even though research scripts are
intentionally outside the bank's authenticated scientific-source manifest. It
does not read pilot data and is not evidence for an empirical visibility
residual shape.

## JSON contract

For backward compatibility, `--json-format legacy` remains the default. It
preserves the original top-level `{zeta, bank, bins}` shape and represents only
the historical `perbin_appendix_a` plus
`noise_normalized_at_each_time` calculation with its fixed 10% perturbation.
Existing commands and JSON consumers therefore do not silently switch
semantics. Select `--json-format complete-v1` explicitly for either new
estimator, either named time-family provenance record, adjustable stability
perturbations, or the complete refusal ledger.

The versioned output identifier is `baonoise-bias-tolerance-v1`; its
machine-readable schema is `docs/bias-tolerance.schema.json`. Complete-v1 JSON
is emitted with
`allow_nan=False`: unavailable or refused quantities are `null`, never the
non-standard tokens `NaN` or `Infinity`.

The top-level provenance binds the bank digest, bank/source identities,
estimator/backend identity, template metadata, time family, reference time,
request, and stability policy. `bins[].points[].parameters` contains one entry
for every requested target. Consumers must test `accepted` and must not infer
acceptance merely from the presence of a central numerical value.

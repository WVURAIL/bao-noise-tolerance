# Open items

## K = 128 formalization: LANDED 2026-08-11
Chapter II carries the population/propagation subsection with all exhibits:
census table (132 facilities, 49 pri / 83 sec = 63% secondary, from Dylan's
dtv_500.xlsx aggregated by class heuristic on call signs), 120-mile map
(Dylan's xmtrs_120miles.png, cropped), per-channel archive-averaged PSD
(scripts/plot_census_psd.py from the products' integrated spectra), and the
Doppler-by-class table (2v/lambda at 581 MHz). Chapter III item 3(b) has
the three-sided bracket + the out-of-span caveat (ch33's second
primary at -4 kHz).
Remaining enrichments (optional):
- ERP / assigned-offset columns per facility (his docx documents the LMS
  app_antenna_frequency route); would let the PSD sidelobes be matched to
  specific facilities.
- Class heuristic spot-check by Dylan (call-sign rules: CHnnnn-, -N,
  -LD/-LP/-CD/-CA, K##XX(-D) -> secondary).
- Attribution study (campaign): shelf power vs refractivity soundings
  (ducting) and fine spectra vs ADS-B tracks (flutter); stated in the
  chapter as separable by timescale but unproven.

## Non-pilot channel mode (kernel + weights)
SPEC WRITTEN 2026-08-11: pilot-proxy docs/nonpilot_mode_spec.md (patch
prepared; selection algorithm,
bundle/schema fields, provenance refusal rules, validation gates; ready
for repo integration). Original requirements, all addressed there:

- A weight-synthesis mode for an arbitrary stated target frequency. Candidate
  default: the bin-center via the existing off-grid machinery, same reference
  placement. "Use fine bin 0" is on the table but MUST be gated by
  instrument-bin avoidance: fine bin 0 is the frame DFT's DC, where receiver
  artifacts (offsets, 1/f, PFB edge structure) concentrate. Selection logic
  should exclude DC, the fine-span edges, and any census-excluded bins, then
  pick the nearest clean fine bin to the stated target.
- Kernel: non-pilot channel currently disables the detector (bundle rule);
  needs a mode that instead selects a stated-target weight bank.
- Provenance: products must record that the target is a control frequency,
  not a pilot, so floor_provenance()-style checks do not misread them.

## Waiting on data
- tau_c via contiguous scan: freq_id 614 first (ch29's verdict flips on it),
  then 568 (era-resolved). After all pilot scans, per Dylan.
- ch33 wide-null diagnosis (bins 553/551) and across-allocation proxy
  verification (mid-shelf bins): after all pilot scans.

## Reference-axis design study (forward epoch only)
Chapter III item 4 now records the sizing law (knee at N_ref = 2 because
feeds are cells) and the declined alternatives. Two measurable follow-ups,
both next-epoch (any new reference = new weight bank = new mu0 = survey
epoch boundary):
- Curvature study: DONE 2026-08-11 (scripts/curvature_study.py).
  Median smooth curvature bias 2e-3 of background (1/5 of the 1% knee),
  <= 3e-3 on every threshold-feasible channel; slope terms (cancelled
  exactly by the pair) run 10x larger. Crowded-channel excess is discrete
  structure the Richardson design cannot remove -> plain pair stands,
  measured; residual concern redirected to the censoring option.
  SIDE FINDING: the frame-spectrum DC artifact (bin 0) exceeds the
  archive-averaged pilot on ch28/29/34/36; the census PSD figure and any
  argmax-based spectrum handling must anchor on metadata pilot position
  (fixed in plot_census_psd.py); the non-pilot mode spec's DC exclusion
  cites this.
- Censoring third reference: median-of-3 references (comparisons only,
  integer-exact) bounds the denominator bias a contaminant on one
  reference can cause. The one reference-axis upgrade with qualitative
  value; assess against the multi-pilot/era environment.

## Forecast side
- Three-worlds (delay filter both sides) verdict table: banks building,
  wakeup scheduled.
- Optimal thresholds: production T lands as multiplier_q16 (fine axis) /
  rational eta*mu0 (coarse dev check); regenerate after each new channel.

## Fine-only deployment gap (from scripts/fine_operating_point.py)
The joint (rho, m) argmin under the survey-cost objective works, and its
FA machinery validates on real quiet frames (ch29: empirical exceedance
matches (N+1-rho)/(N+1) at all tested ranks). But the deployed
single-fixed-anchor rule does not yet reproduce the coarse accounting the
chapter's verdicts stand on:

- Anchors are per-channel AND per-era bundle data, now measured: ch32
  drifts 141 -> 152 (~130 Hz, the 2020 station handover), ch35 112 -> 107
  (~60 Hz, the 2021 light-up), ch33 splits 180 / 116 (two structures).
  Anchor localization + era epochs belong in the pending campaign; all
  fixes are bundle data, no kernel change.
- Selection convention SETTLED (2026-08-11): (1) feasible = r <= tol on
  the archive AND in each calendar half (the mask-side stability gate);
  (2) cost plateau 2%; (3) ties broken BY DATA (minimax era-half cost);
  (4) residual exact ties: m nearest 1, then max headroom, then smallest
  rho. Product frame order is NOT chronological (whole units land out of
  order), so era splits must be by time; the calendar midpoint is the
  named split (frame-median is cadence-biased toward recent years).
- Era certification localized each channel's usable epoch (union windows,
  per-era anchors): ch32 early half (station era, pilot at 152) keeps 7
  frames -> uncertifiable; late half (pilot at 141) certifies at f=43.3%,
  r=0.0155. ch35 early half (distant-station era, pilot at 82) certifies
  at f=18.1%, r=0.017; late half (2021 station, pilot at 111) fails BOTH
  walls (f=98.8% and r=0.237). Neither channel is single-bundle
  era-stable: deployment needs per-epoch bundles with boundaries at the
  measured station events (2020 departure, 2021 arrival); the repo's
  "contract change lands at a survey epoch boundary" rule is the hook.
  Archive-wide points (ch32: rho=80, m=1.15, cost 2.54 vs coarse 2.10;
  ch35: era-fallback m=7.61) are records rather than deployment candidates.
- ch33: NO concentrated schedule; designated coverage of coarse-masked
  frames is 2-5% under any single anchor (18% era-matched); early/late
  anchors 123/180 are two different structures. Wide-null confirmed on
  the fine axis; ch33's feasible verdict rests on the coarse family only.
- ch31's "feasible" fine point (f = 99.5%, cost ~185) is vacuous --
  occupancy wall, same as its coarse point.
- FA machinery validated on real quiet frames (ch29: empirical exceedance
  matches (N+1-rho)/(N+1) at rho = 100/115/124, three pseudo-designated
  bins).

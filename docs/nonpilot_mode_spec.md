# Non-Pilot (Control-Target) Kernel Mode — Design Specification

Status: DRAFT for pilot-proxy integration. Written against kernel core 2.3.x,
product schema v3, fine decision v1. Nothing here changes decision
arithmetic; the mode adds a weight-synthesis path, a bundle flag, and
provenance so control products can never masquerade as pilot products.

## 1. Purpose

The control/verification program (out/nonpilot_scan_list.csv: 15 bins, P1–P4)
scans coarse channels that carry **no pilot**: mid-shelf bins inside DTV
allocations, the protected radio-astronomy quiet zone (608–614 MHz), and
far-from-DTV controls. Goals: (a) empirical false-alarm rates on bins where
the detector should see nothing; (b) floor calibration cross-checks on bins
whose noise is genuinely clean; (c) across-allocation support for the proxy
transfer premise (P1 shelf-adjacent bins). The standard weight bank does not
apply — there is no pilot frequency to synthesize against — and the current
bundle rule *disables* the detector on non-pilot channels. This mode replaces
that disable with a stated-target run that records statistics with full
provenance.

## 2. Target-frequency selection (offline, weight-generation time)

Deterministic algorithm, recorded in the manifest:

1. Candidate grid: the K-tap filter's bin centers within the coarse channel
   (normalized frequencies k/K, k = 0..K−1) — reuse of the existing off-grid
   synthesis machinery with zero off-grid term.
2. Exclusions, in order:
   - **Frame-DFT DC neighborhood**: candidates whose synthesized tone lands
     within one fine span (±f_s/2K) of baseband DC — receiver offsets and
     quantizer bias concentrate there (the integrated spectra show bin-0
     artifacts exceeding archive-averaged pilots on four of nine measured
     channels; the same artifact must not sit under a control target).
   - **Channel-edge neighborhood**: candidates within one K-bin of the coarse
     channel edges (PFB rolloff, wrap ambiguity).
   - **Census bins**: any candidate whose designated window would intersect a
     census-excluded fine bin, using the same census source as the fine
     decision.
   - **Measured structure** (when an integrated spectrum exists for the
     channel): candidates where the archive-averaged spectrum within the
     would-be designated window exceeds the channel median by more than a
     stated threshold (default 3 dB) — reject known lines.
3. Selection: the surviving candidate nearest the channel center; ties break
   toward lower frequency. If no candidate survives, the channel is declared
   uncontrollable and recorded as such (no silent fallback).
4. Reference placement: the standard ±2-bin rule with the existing
   edge-wrap/DC-shift resolution, applied to the chosen target.

## 3. Bundle & kernel

- New bundle fields per channel: `control_target: bool` (default false),
  `control_freq_hz: float64` (the stated target's RF frequency),
  `control_selection: json` (candidate list, exclusion reasons, chosen k).
- Kernel behavior is **unchanged**: same projections, same fine reduction,
  same decision arithmetic against the bundle's (anchor, window, bulk, rank,
  multiplier). For control targets the recommended operating point is the
  survey scaffolding configuration (designated bin 0, diagnostic CFAR)
  so control products are comparable to the pilot survey's bootstrap era.
- The mask output line is computed but **must not gate** any consumer: the
  bundle validator enforces `control_target == true → mask_consumers == none`.
  A control channel that fires is a finding, not a flag.

## 4. Product provenance (schema additions)

- `control_target` (bool), `control_freq_hz`, `control_selection_json`
  copied from the bundle into every product.
- `detector_contract_json` gains `"target_kind": "pilot" | "control"`.
- Analysis-side contract: `floor_provenance()`, `shelf_statistics()`, and
  every proxy-relation consumer MUST refuse products with
  `target_kind == "control"` unless called with an explicit
  `allow_control=True` — there is no shelf behind a control target, so the
  proxy relation (Eq. shelfproxy) does not apply, and a control product
  interpreted as a pilot product would fabricate contamination.

## 5. Validation gates

1. Injection: a synthesized tone at the stated control frequency (signed
   offset) must peak in the predicted fine bin — same acceptance form as the
   pilot path's sense gate.
2. Exactness: μ0 recomputed from the control weight bank's integer norms;
   cross-product decision arithmetic unchanged (fuzz gate reuse).
3. Selection determinism: re-running selection on the same inputs must
   reproduce the manifest byte-for-byte (the candidate walk is ordered).
4. Refusal: a channel with no surviving candidate must produce a validator
   error, not a product.

## 6. Analysis deliverables (once scans run)

- Empirical designated-bin and any-bin false-alarm rates vs the
  exchangeability prediction, per control class P1–P4.
- Floor distribution on P2 (quiet-zone) bins vs the σ-null basis used for
  μ0<1 pilot channels.
- P1 shelf-adjacent bins: kept-frame shelf estimates across the allocation
  vs the pilot-inferred level — the first data on the footprint premise
  ahead of the deployment-side verification.

## 7. Out of scope

No changes to fine decision v1 arithmetic, no new kernel entry points, no
per-channel K. The mode is bundle data + weight synthesis + provenance.

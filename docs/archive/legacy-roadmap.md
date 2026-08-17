# Archived roadmap context

The former root `TODO.md` mixed completed dissertation notes, pilot-proxy
detector work, data-collection plans, and Bao forecast tasks. Git history is
the authoritative archive for that detailed chronology. This short record
preserves only the unresolved context that can still affect Bao results.

- Correlation-time measurements still need contiguous scans, first for
  `freq_id=614` (ATSC ch29) and then `freq_id=568`. Those inputs belong to the
  survey-product producer; Bao consumes their resulting provenance and
  residual budgets.
- Per-era detector bundles and non-pilot/control-frequency selection are
  pilot-proxy responsibilities. The former `docs/nonpilot_mode_spec.md` was
  removed here because keeping a second design specification in the forecast
  repository allowed the producer and consumer contracts to drift.
- New measurement products must record their detector rule, kernel identity,
  frequency target, and survey epoch. Bao deliberately refuses to infer those
  fields from filenames.
- Publication products should be regenerated from machine-readable inputs.
  Build logs, status sentinels, private email drafts, and compiled PDFs are not
  source artifacts and were removed from the maintained tree.
- `fine_operating_point.py`, `plot_convergence.py`, and `three_worlds.py` are
  research-only scripts that still load helpers from sibling scripts. The
  coherent-bias scripts now fail closed unless their exact locally built
  strict-v2 prerequisites are present (see the README); neither the coupling
  nor those large response banks are part of the installed API. Factor the
  helpers into `baonoise` if a workflow is promoted to a maintained interface.

Future implementation work should be tracked as issues in the repository that
owns it. Completed rationale remains recoverable from Git and should not be
copied back into an active task list.

"""BAO detection significance and time-to-detection from a Fisher bank plus a
masking scenario.

Detection metric
----------------
The BAO detection significance is 1/sigma(A), where A is the BAO wiggle
amplitude parameter in RadioFisher (fiducial A=1, so A/sigma_A = 1/sigma_A).
Following the published RadioFisher analysis (plot_Abao_zbins.py; Bull,
Ferreira, Patel & Santos 2015), we marginalise per redshift bin over
{b_HI, f, aperp, apar} (expanded as functions of z) and the shared sigma_NL,
while fixing {Tb, sigma_8, n_s} (externally constrained shape parameters).
A is shared across bins, so the combined matrix directly yields the
survey-level sigma(A).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import survey
from ._validation import (finite_scalar as _finite_scalar,
                          nonnegative_scalar as _nonnegative_finite_scalar,
                          positive_scalar as _positive_finite_scalar)
from .fisherbank import ARTIFACT_FORECAST, BIAS_PARAMETER, FisherBank
from .scenarios import Scenario

EXCLUDE = ["Tb", "sigma8tot", "n_s", "fs8", "bs8", "pk"]
EXPAND = ["b_HI", "f", "aperp", "apar"]

# Appendix-A style (Amiri et al. 2022): each bin analyzed independently,
# marginalising over {A, sigma_NL, aperp, apar, bs8, fs8} with no priors;
# b_HI/f enter through the (b+f mu^2) sigma8 combinations bs8/fs8.
EXCLUDE_PERBIN = ["b_HI", "f", "Tb", "sigma8tot", "n_s", "pk"]

V_FRAC_MIN = 1e-6

FORECAST_STYLES = frozenset({"shared_A", "perbin_A"})

# Eigenmodes below this relative scale cannot be distinguished reliably from
# a Fisher-matrix null space. Treating them as information and blindly taking
# a pseudoinverse can turn an unconstrained parameter into a small, finite
# error bar. A 1e12 condition-number limit is conservative relative to the
# committed banks in their canonical forecast styles (O(1e6) or better).
FISHER_CONDITION_LIMIT = 1e12
FISHER_NULLSPACE_RTOL = np.sqrt(np.finfo(float).eps)


def _time_bracket(t_lo, t_hi) -> tuple[float, float]:
    lo = _positive_finite_scalar(t_lo, "t_lo")
    hi = _positive_finite_scalar(t_hi, "t_hi")
    if lo >= hi:
        raise ValueError("t_lo must be less than t_hi")
    return lo, hi


def _variance_from_fisher(F: np.ndarray, coefficients: np.ndarray) -> float:
    """Variance of a linear parameter combination after marginalisation.

    A singular Fisher matrix can still constrain a combination that is
    orthogonal to its null space. Conversely, ``pinv(F)`` alone returns an
    artificially finite variance for a combination that overlaps a discarded
    eigenmode. Test estimability first, and use the inverse/pseudoinverse only
    for combinations that are actually constrained.
    """
    F = np.asarray(F, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if (F.ndim != 2 or F.shape[0] != F.shape[1]
            or coefficients.shape != (F.shape[0],)):
        raise ValueError("Fisher matrix and coefficient vector shapes disagree")
    if F.size == 0 or not np.all(np.isfinite(F)) \
            or not np.all(np.isfinite(coefficients)):
        return np.inf

    F = 0.5 * (F + F.T)
    eigenvalues, eigenvectors = np.linalg.eigh(F)
    largest = float(eigenvalues[-1])
    if largest <= 0.0:
        return np.inf

    cutoff = largest / FISHER_CONDITION_LIMIT
    # A materially negative eigenvalue is not a valid Fisher matrix. Small
    # roundoff-level negatives are treated as null modes below.
    if eigenvalues[0] < -cutoff:
        return np.inf
    constrained = eigenvalues > cutoff
    null_vectors = eigenvectors[:, ~constrained]
    if null_vectors.size:
        projected = np.linalg.norm(null_vectors.T @ coefficients)
        scale = max(1.0, float(np.linalg.norm(coefficients)))
        if projected > FISHER_NULLSPACE_RTOL * scale:
            return np.inf

    if np.all(constrained):
        # Retain the established full-rank calculation path so valid forecast
        # results do not move merely because null-space checking was added.
        try:
            covariance = np.linalg.inv(F)
        except np.linalg.LinAlgError:  # defensive: eigh already found full rank
            return np.inf
        variance = float(coefficients @ covariance @ coefficients)
    else:
        projections = eigenvectors[:, constrained].T @ coefficients
        variance = float(np.sum(projections**2 / eigenvalues[constrained]))
    return variance if np.isfinite(variance) and variance > 0.0 else np.inf


class Forecast:
    """style='shared_A'  : Bull et al. (2015) treatment: A shared across
                           bins, {b_HI,f,aperp,apar} expanded per bin.
       style='perbin_A'  : Amiri et al. (2022) Appendix A: each bin
                           independent with per-bin A; survey significance
                           adds per-bin sigma_A^-2 in quadrature.

       Banked ``perbin_A`` calculations do not need a RadioFisher backend,
       so ``rf`` may be omitted. ``shared_A`` requires the backend at
       construction; direct Fisher evaluation imports it lazily when needed.
    """

    def __init__(self, bank: FisherBank, rf=None, style: str = "shared_A",
                 rf_dir=None):
        if style not in FORECAST_STYLES:
            choices = ", ".join(sorted(FORECAST_STYLES))
            raise ValueError(f"style must be one of: {choices}")
        if style == "shared_A" and rf is None:
            raise RuntimeError(
                "shared_A forecasts require a RadioFisher backend; pass the "
                "imported radiofisher module as rf"
            )
        if BIAS_PARAMETER in getattr(bank, "paramnames", ()):
            raise ValueError(
                f"{BIAS_PARAMETER} is a residual-bias response row, not a "
                "forecast parameter; use the dedicated bias workflow")
        bank_kind = bank.artifact_kind
        if bank_kind != ARTIFACT_FORECAST:
            raise ValueError(
                f"bank artifact_kind={bank_kind!r}; Forecast requires "
                f"{ARTIFACT_FORECAST!r}")
        if rf is not None and getattr(rf, "__file__", None):
            from .compat import bind_radiofisher
            rf_dir = bind_radiofisher(rf, rf_dir)
        self.bank = bank
        self.rf = rf
        self.rf_dir = rf_dir
        self.style = style

    # ------------------------------------------------------------------
    def _marginal_fisher_bin(self, F: np.ndarray):
        """Fisher matrix of one bin's kept parameters
        (perbin_A style). Exactly zero-information rows are dropped.

        A non-positive diagonal is not by itself evidence of zero
        information: a negative diagonal or a zero diagonal coupled to
        another parameter makes the Fisher matrix invalid. Keep those rows so
        the eigenvalue check can refuse the resulting uncertainty estimate.
        """
        names = list(self.bank.paramnames)
        keep = [i for i, n in enumerate(names) if n not in EXCLUDE_PERBIN]
        Fk = F[np.ix_(keep, keep)]
        kn = [names[i] for i in keep]
        nz = np.flatnonzero(np.any(Fk != 0.0, axis=0)
                            | np.any(Fk != 0.0, axis=1))
        Fk = Fk[np.ix_(nz, nz)]
        kn = [kn[i] for i in nz]
        Fk = 0.5 * (Fk + Fk.T)
        return Fk, kn

    def _sigma_A_from_bin_matrix(self, F: np.ndarray) -> float:
        Fk, kn = self._marginal_fisher_bin(F)
        if "A" not in kn:
            return np.inf
        coefficients = np.zeros(len(kn))
        coefficients[kn.index("A")] = 1.0
        var = _variance_from_fisher(Fk, coefficients)
        return float(np.sqrt(var)) if np.isfinite(var) else np.inf

    # ------------------------------------------------------------------
    def _scenario_matrices(self, scenario: Scenario, t_hours: float,
                           bins: list[int] | None = None):
        t_hours = _nonnegative_finite_scalar(t_hours, "t_hours")
        factors = scenario.bin_factors_for_zbins(self.bank.zs)
        F_list = []
        for i, (v_frac, w_bar) in enumerate(factors):
            if bins is not None and i not in bins:
                continue
            if v_frac <= V_FRAC_MIN:
                continue
            F_list.append(v_frac * self.bank.F(i, t_hours * w_bar))
        return F_list

    def sigma_A(self, scenario: Scenario, t_hours: float,
                bins: list[int] | None = None) -> float:
        """Marginalised sigma(A) for a scenario at total time t_hours.
        In perbin_A style this is the quadrature combination
        (sum_i sigma_A_i^-2)^-1/2 of independent per-bin amplitudes."""
        F_list = self._scenario_matrices(scenario, t_hours, bins=bins)
        return self._sigma_A_from_list(F_list)

    def _sigma_A_from_list(self, F_list) -> float:
        if not F_list:
            return np.inf
        if self.style == "perbin_A":
            inv_var = 0.0
            for F in F_list:
                sA = self._sigma_A_from_bin_matrix(F)
                if np.isfinite(sA) and sA > 0:
                    inv_var += 1.0 / sA**2
            return float(1.0 / np.sqrt(inv_var)) if inv_var > 0 else np.inf
        Ftot, names = self.rf.combined_fisher_matrix(
            F_list, names=list(self.bank.paramnames),
            exclude=list(EXCLUDE), expand=list(EXPAND))
        iA = names.index("A")
        Ftot = 0.5 * (Ftot + Ftot.T)
        coefficients = np.zeros(len(names))
        coefficients[iA] = 1.0
        var = _variance_from_fisher(Ftot, coefficients)
        return float(np.sqrt(var)) if np.isfinite(var) else np.inf

    def significance(self, scenario: Scenario, t_hours: float,
                     bins: list[int] | None = None) -> float:
        """BAO detection significance A/sigma(A) (fiducial A=1)."""
        s = self.sigma_A(scenario, t_hours, bins=bins)
        return 1.0 / s if np.isfinite(s) and s > 0 else 0.0

    def significance_curve(self, scenario: Scenario,
                           t_hours: np.ndarray) -> np.ndarray:
        times = np.asarray(t_hours, dtype=float)
        if times.ndim == 0 or not np.all(np.isfinite(times)) \
                or np.any(times < 0.0):
            raise ValueError("t_hours must contain non-negative finite values")
        return np.array([self.significance(scenario, t) for t in times])

    # ------------------------------------------------------------------
    def required_hours(self, scenario: Scenario, target: float = 5.0,
                       t_lo: float = 10.0, t_hi: float = 1e6) -> float:
        """Total on-sky hours needed for significance >= target (inf if the
        target is unreachable within t_hi hours)."""
        target = _positive_finite_scalar(target, "target")
        t_lo, t_hi = _time_bracket(t_lo, t_hi)
        f = lambda logt: self.significance(scenario, 10.0 ** logt) - target
        if f(np.log10(t_lo)) >= 0.0:
            return t_lo
        if f(np.log10(t_hi)) < 0.0:
            return np.inf
        # General first-crossing detection for nonmonotonic curves is a
        # separate concern; brentq retains the established bracket behavior.
        logt = brentq(f, np.log10(t_lo), np.log10(t_hi), xtol=1e-4)
        return float(10.0 ** logt)

    def required_years(self, scenario: Scenario, target: float = 5.0,
                       duty: float = 1.0,
                       hours_per_year: float = survey.MEAN_CALENDAR_YEAR_HOURS) -> float:
        duty = _positive_finite_scalar(duty, "duty")
        hours_per_year = _positive_finite_scalar(
            hours_per_year, "hours_per_year")
        h = self.required_hours(scenario, target)
        return (float(survey.hours_to_years(h, duty, hours_per_year))
                if np.isfinite(h) else np.inf)

    # ------------------------------------------------------------------
    def sigma_A_direct(self, scenario: Scenario, t_hours: float,
                       bins: list[int] | None = None, cosmo=None,
                       cosmo_fns=None, rf_dir=None) -> float:
        """sigma(A) via *direct* rf.fisher() calls using the RadioFisher fork's
        RFI hooks (expt['noise_freq_weight'] / ['noise_freq_mode'] /
        ['vol_frac']) instead of the precomputed bank. Slow (seconds/bin);
        used to validate that the in-fork noise model and the bank-rescaling
        path agree.
        """
        t_hours = _nonnegative_finite_scalar(t_hours, "t_hours")
        if t_hours == 0.0:
            return np.inf
        import contextlib
        import io

        from . import cosmologies, pkcache, survey
        from .compat import (DIRECT_MASK_CAPABILITIES, bind_radiofisher,
                             import_radiofisher, require_backend_capabilities)

        rf = self.rf
        requested_rf_dir = rf_dir if rf_dir is not None else self.rf_dir
        if rf is None:
            try:
                rf, rf_dir = import_radiofisher(requested_rf_dir)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "direct Fisher evaluation requires a RadioFisher "
                    "checkout; set RADIOFISHER_DIR or pass rf_dir"
                ) from exc
        else:
            rf_dir = bind_radiofisher(rf, requested_rf_dir)
        require_backend_capabilities(
            rf, DIRECT_MASK_CAPABILITIES, rf_dir=rf_dir)
        cfg = self.bank.meta["config"]
        cosmology_name = self.bank.meta["cosmology"]
        recorded_profile = self.bank.meta["astrophysical_model_profile"]
        if cosmo is None and cosmo_fns is not None:
            raise ValueError("cosmo_fns cannot be supplied without cosmo")
        if cosmo is not None:
            if recorded_profile not in {"bull2015", "chime_overview_2022"}:
                raise ValueError(
                    "direct validation requires a recorded canonical "
                    "astrophysical_model_profile")
            cosmo = cosmologies.with_explicit_physical_densities(
                cosmologies.with_astrophysical_profile(
                    cosmo, recorded_profile, rf=rf))
            # Resolving an authoritative physical-density triplet can change
            # mnu/omega_cdm_0/omega_nu_0. Never reuse splines built from the
            # caller's pre-resolution dictionary.
            cosmo_fns = rf.background_evolution_splines(cosmo)
        if cosmo is None:
            from .resources import filesystem_data_file
            if cfg == "chime2022":
                ctag = (f"_{cosmology_name}"
                        if cosmology_name != "planck2018" else "")
                cosmo = pkcache.load_fiducial_cosmology(
                    rf, filesystem_data_file(
                        f"cache_pk_chime2022{ctag}.dat"),
                    cosmo=cosmologies.get(cosmology_name, rf, rf_dir))
            elif cfg == "bull2015" and cosmology_name == "planck2013":
                if recorded_profile != "bull2015":
                    raise ValueError(
                        "direct validation of a Bull-2015 bank requires its "
                        "canonical HI signal profile")
                base_cosmo = cosmologies.with_astrophysical_profile(
                    rf.experiments.cosmo, "bull2015", rf=rf)
                cosmo = pkcache.load_fiducial_cosmology(
                    rf, filesystem_data_file("cache_pk.dat"),
                    cosmo=base_cosmo)
            else:
                raise ValueError(
                    f"unsupported direct-validation configuration: "
                    f"config={cfg!r}, cosmology={cosmology_name!r}")
            cosmo_fns = rf.background_evolution_splines(cosmo)

        factors = scenario.bin_factors_for_zbins(self.bank.zs)
        wfn = scenario.freq_weight_fn()
        F_list = []
        for i, (v_frac, w_bar) in enumerate(factors):
            if bins is not None and i not in bins:
                continue
            # A retained slice with zero clean-time weight carries exactly no
            # Fisher information. Skip it just as the bank path evaluates
            # F(t=0), rather than sending a forbidden zero weight through the
            # backend's strictly-positive noise-frequency contract.
            if v_frac <= V_FRAC_MIN or w_bar <= 0.0:
                continue
            expt = survey.experiment_from_bank_metadata(
                rf, rf_dir, self.bank.meta, ttot_hours=t_hours)
            expt["noise_freq_weight"] = wfn
            expt["noise_freq_mode"] = scenario.rf_mode()
            expt["vol_frac"] = float(v_frac)
            with contextlib.redirect_stdout(io.StringIO()):
                F, names = rf.fisher(self.bank.zs[i], self.bank.zs[i + 1],
                                     cosmo, expt, cosmo_fns)
            if list(names) != list(self.bank.paramnames):
                raise RuntimeError(
                    "RadioFisher parameter schema does not match the bank: "
                    f"bank={list(self.bank.paramnames)}, direct={list(names)}")
            F_list.append(np.asarray(F))
        return self._sigma_A_from_list(F_list)

    # ------------------------------------------------------------------
    def sigma_param_bin(self, scenario: Scenario, t_hours: float, ibin: int,
                        param: str = "aperp0") -> float:
        """Marginalised error on a per-bin parameter for one redshift bin
        analyzed alone ('aperp0' in shared_A style; 'aperp' in perbin_A)."""
        F_list = self._scenario_matrices(scenario, t_hours, bins=[ibin])
        if not F_list:
            return np.inf
        if self.style == "perbin_A":
            Fk, kn = self._marginal_fisher_bin(F_list[0])
            pname = param.rstrip("0123456789") if param not in kn else param
            if pname not in kn:
                return np.inf
            coefficients = np.zeros(len(kn))
            coefficients[kn.index(pname)] = 1.0
            var = _variance_from_fisher(Fk, coefficients)
            return float(np.sqrt(var)) if np.isfinite(var) else np.inf
        Ftot, names = self.rf.combined_fisher_matrix(
            F_list, names=list(self.bank.paramnames),
            exclude=list(EXCLUDE), expand=list(EXPAND))
        Ftot = 0.5 * (Ftot + Ftot.T)
        coefficients = np.zeros(len(names))
        coefficients[names.index(param)] = 1.0
        var = _variance_from_fisher(Ftot, coefficients)
        return float(np.sqrt(var)) if np.isfinite(var) else np.inf

    def required_hours_metric(self, metric_fn, threshold: float,
                              decreasing: bool = False, t_lo: float = 10.0,
                              t_hi: float = 1e6) -> float:
        """Smallest t with metric_fn(t) >= threshold (or <= if decreasing)."""
        threshold = _finite_scalar(threshold, "threshold")
        t_lo, t_hi = _time_bracket(t_lo, t_hi)
        sign = -1.0 if decreasing else 1.0

        def f(logt):
            value = float(metric_fn(10.0 ** logt))
            if np.isnan(value):
                raise ValueError("metric_fn must return a scalar that is not NaN")
            return sign * (value - threshold)

        if f(np.log10(t_lo)) >= 0.0:
            return t_lo
        if f(np.log10(t_hi)) < 0.0:
            return np.inf
        # General first-crossing detection for nonmonotonic curves is a
        # separate concern; brentq retains the established bracket behavior.
        logt = brentq(f, np.log10(t_lo), np.log10(t_hi), xtol=1e-4)
        return float(10.0 ** logt)

    # ------------------------------------------------------------------
    def sigma_dv_bin(self, scenario: Scenario, t_hours: float,
                     ibin: int) -> float:
        """Fractional error on the volume distance D_V from one bin
        (perbin_A style): ln D_V = (2/3) ln aperp + (1/3) ln apar."""
        F_list = self._scenario_matrices(scenario, t_hours, bins=[ibin])
        if not F_list:
            return np.inf
        Fk, kn = self._marginal_fisher_bin(F_list[0])
        if "aperp" not in kn or "apar" not in kn:
            return np.inf
        ip, il = kn.index("aperp"), kn.index("apar")
        coefficients = np.zeros(len(kn))
        coefficients[ip] = 2.0 / 3.0
        coefficients[il] = 1.0 / 3.0
        var = _variance_from_fisher(Fk, coefficients)
        return float(np.sqrt(var)) if np.isfinite(var) else np.inf

    # ------------------------------------------------------------------
    def per_bin_significance(self, scenario: Scenario,
                             t_hours: float) -> np.ndarray:
        """Detection significance of each redshift bin analyzed alone."""
        out = np.zeros(self.bank.nbins)
        for i in range(self.bank.nbins):
            out[i] = self.significance(scenario, t_hours, bins=[i])
        return out

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

from .fisherbank import FisherBank
from .scenarios import Scenario
from . import survey

EXCLUDE = ["Tb", "sigma8tot", "n_s", "fs8", "bs8", "pk"]
EXPAND = ["b_HI", "f", "aperp", "apar"]

# Appendix-A style (Amiri et al. 2022): each bin analyzed independently,
# marginalising over {A, sigma_NL, aperp, apar, bs8, fs8} with no priors;
# b_HI/f enter through the (b+f mu^2) sigma8 combinations bs8/fs8.
EXCLUDE_PERBIN = ["b_HI", "f", "Tb", "sigma8tot", "n_s", "pk"]

V_FRAC_MIN = 1e-6


class Forecast:
    """style='shared_A'  : Bull et al. (2015) treatment: A shared across
                           bins, {b_HI,f,aperp,apar} expanded per bin.
       style='perbin_A'  : Amiri et al. (2022) Appendix A: each bin
                           independent with per-bin A; survey significance
                           adds per-bin sigma_A^-2 in quadrature."""

    def __init__(self, bank: FisherBank, rf, style: str = "shared_A"):
        self.bank = bank
        self.rf = rf
        self.style = style

    # ------------------------------------------------------------------
    def _marginal_cov_bin(self, F: np.ndarray):
        """Marginalised covariance of one bin's kept parameters
        (perbin_A style). Zero-information rows are dropped."""
        names = list(self.bank.paramnames)
        keep = [i for i, n in enumerate(names) if n not in EXCLUDE_PERBIN]
        Fk = F[np.ix_(keep, keep)]
        kn = [names[i] for i in keep]
        nz = [i for i in range(len(kn)) if Fk[i, i] > 0.0]
        Fk = Fk[np.ix_(nz, nz)]
        kn = [kn[i] for i in nz]
        Fk = 0.5 * (Fk + Fk.T)
        try:
            cov = np.linalg.inv(Fk)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(Fk)
        return cov, kn

    def _sigma_A_from_bin_matrix(self, F: np.ndarray) -> float:
        cov, kn = self._marginal_cov_bin(F)
        if "A" not in kn:
            return np.inf
        var = cov[kn.index("A"), kn.index("A")]
        return float(np.sqrt(var)) if var > 0 else np.inf

    # ------------------------------------------------------------------
    def _scenario_matrices(self, scenario: Scenario, t_hours: float,
                           bins: list[int] | None = None):
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
        try:
            cov = np.linalg.inv(Ftot)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(Ftot)
        var = cov[iA, iA]
        return float(np.sqrt(var)) if var > 0 else np.inf

    def significance(self, scenario: Scenario, t_hours: float,
                     bins: list[int] | None = None) -> float:
        """BAO detection significance A/sigma(A) (fiducial A=1)."""
        s = self.sigma_A(scenario, t_hours, bins=bins)
        return 1.0 / s if np.isfinite(s) and s > 0 else 0.0

    def significance_curve(self, scenario: Scenario,
                           t_hours: np.ndarray) -> np.ndarray:
        return np.array([self.significance(scenario, t) for t in t_hours])

    # ------------------------------------------------------------------
    def required_hours(self, scenario: Scenario, target: float = 5.0,
                       t_lo: float = 10.0, t_hi: float = 1e6) -> float:
        """Total on-sky hours needed for significance >= target (inf if the
        target is unreachable within t_hi hours)."""
        f = lambda logt: self.significance(scenario, 10.0 ** logt) - target
        if f(np.log10(t_hi)) < 0.0:
            return np.inf
        if f(np.log10(t_lo)) >= 0.0:
            return t_lo
        logt = brentq(f, np.log10(t_lo), np.log10(t_hi), xtol=1e-4)
        return float(10.0 ** logt)

    def required_years(self, scenario: Scenario, target: float = 5.0,
                       duty: float = 0.75) -> float:
        h = self.required_hours(scenario, target)
        return float(survey.hours_to_years(h, duty)) if np.isfinite(h) else np.inf

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
        import contextlib
        import io

        from . import pkcache, survey
        from .compat import find_radiofisher_dir

        rf = self.rf
        rf_dir = find_radiofisher_dir(rf_dir)
        cfg = self.bank.meta.get("config", "bull2015")
        if cosmo is None or cosmo_fns is None:
            from pathlib import Path
            data = Path(__file__).resolve().parents[2] / "data"
            if cfg == "chime2022":
                cosmo = pkcache.load_fiducial_cosmology(
                    rf, data / "cache_pk_chime2022.dat",
                    cosmo=survey.chime2022_cosmo(rf, rf_dir))
            else:
                cosmo = pkcache.load_fiducial_cosmology(rf, data / "cache_pk.dat")
            cosmo_fns = rf.background_evolution_splines(cosmo)

        factors = scenario.bin_factors_for_zbins(self.bank.zs)
        wfn = scenario.freq_weight_fn()
        F_list = []
        for i, (v_frac, _w_bar) in enumerate(factors):
            if bins is not None and i not in bins:
                continue
            if v_frac <= V_FRAC_MIN:
                continue
            if cfg == "chime2022":
                expt = survey.chime2022_experiment(rf, rf_dir,
                                                   ttot_hours=t_hours)
            else:
                expt = survey.chime_experiment(rf, rf_dir, ttot_hours=t_hours)
            expt["noise_freq_weight"] = wfn
            expt["noise_freq_mode"] = scenario.rf_mode()
            expt["vol_frac"] = float(v_frac)
            with contextlib.redirect_stdout(io.StringIO()):
                F, names = rf.fisher(self.bank.zs[i], self.bank.zs[i + 1],
                                     cosmo, expt, cosmo_fns)
            assert list(names) == list(self.bank.paramnames)
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
            cov, kn = self._marginal_cov_bin(F_list[0])
            pname = param.rstrip("0123456789") if param not in kn else param
            if pname not in kn:
                return np.inf
            var = cov[kn.index(pname), kn.index(pname)]
            return float(np.sqrt(var)) if var > 0 else np.inf
        Ftot, names = self.rf.combined_fisher_matrix(
            F_list, names=list(self.bank.paramnames),
            exclude=list(EXCLUDE), expand=list(EXPAND))
        Ftot = 0.5 * (Ftot + Ftot.T)
        try:
            cov = np.linalg.inv(Ftot)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(Ftot)
        var = cov[names.index(param), names.index(param)]
        return float(np.sqrt(var)) if var > 0 else np.inf

    def required_hours_metric(self, metric_fn, threshold: float,
                              decreasing: bool = False, t_lo: float = 10.0,
                              t_hi: float = 1e6) -> float:
        """Smallest t with metric_fn(t) >= threshold (or <= if decreasing)."""
        sign = -1.0 if decreasing else 1.0
        f = lambda logt: sign * (metric_fn(10.0 ** logt) - threshold)
        if f(np.log10(t_hi)) < 0.0:
            return np.inf
        if f(np.log10(t_lo)) >= 0.0:
            return t_lo
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
        cov, kn = self._marginal_cov_bin(F_list[0])
        if "aperp" not in kn or "apar" not in kn:
            return np.inf
        ip, il = kn.index("aperp"), kn.index("apar")
        var = (4.0 / 9.0) * cov[ip, ip] + (1.0 / 9.0) * cov[il, il] \
            + (4.0 / 9.0) * cov[ip, il]
        return float(np.sqrt(var)) if var > 0 else np.inf

    # ------------------------------------------------------------------
    def per_bin_significance(self, scenario: Scenario,
                             t_hours: float) -> np.ndarray:
        """Detection significance of each redshift bin analyzed alone."""
        out = np.zeros(self.bank.nbins)
        for i in range(self.bank.nbins):
            out[i] = self.significance(scenario, t_hours, bins=[i])
        return out

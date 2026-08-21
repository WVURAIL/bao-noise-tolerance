#!/usr/bin/env python3
"""Evaluate coherent-residual Fisher-bias tolerances without hiding refusals.

Two estimator families are deliberately kept separate:

``perbin_appendix_a``
    The established dissertation path. Each redshift bin is inverted alone
    over the Appendix-A parameter set.

``overview_combined_multibin``
    The CHIME Overview/Fig.-31 construction. Every bank bin is projected with
    RadioFisher's EOS expansion, converted from ``(D_A, H)`` to ``(D_V, F)``,
    and assembled with ``combined_fisher_matrix`` before one survey-wide
    inversion. The residual-response row is expanded per bin and is removed
    before inversion; it is a bias source, never a fitted parameter.

The response bank is built with a unit template proportional to the thermal
noise at the time at which each Fisher point is evaluated. That bank supports
two different, explicitly named time families:

``noise_normalized_at_each_time``
    One unit always means the contemporaneous thermal-noise power: the
    stationary finite-correlation limit in which residual and thermal power
    both average down.

``fixed_physical_at_reference_time``
    One unit means the thermal-noise power at a declared reference time. As
    thermal power scales as 1/t, the bank response at time t is multiplied by
    t/t_ref. This is the non-averaging persistent-residual limit. The two
    families are identical at t_ref and are not interchangeable away from it.

Bias-response banks are intentionally not distributed. Build one and run:

    python scripts/build_bank.py --config chime2022 --cosmology planck2018 \
        --p-res 1.0 --dense-knee \
        --out data/fisher_bank_chime2022_pres_dense.npz
    python scripts/bias_tolerance.py \
        --bank data/fisher_bank_chime2022_pres_dense.npz \
        --estimator overview_combined_multibin \
        --time-scaling fixed_physical_at_reference_time \
        --reference-years 1 --json-format complete-v1 \
        --json out/combined_fixed_physical.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from baonoise import __version__, channels, cosmologies, pkcache, survey
from baonoise.compat import (import_radiofisher,
                             require_backend_capabilities)
from baonoise.constants import HI_REST_FREQUENCY_MHZ
from baonoise.fisherbank import (
    ARTIFACT_BIAS_RESPONSE, BAONOISE_SOURCE_MANIFEST, FOREGROUND_KEYS,
    RADIOFISHER_SOURCE_MANIFEST, FisherBank, _git_state,
    experiment_settings_payload,
)
from baonoise.residual_templates import validate_template_metadata
from baonoise.resources import filesystem_data_file

PRES = "_Pres"
DEFAULT_BIAS_BANK = ROOT / "data" / "fisher_bank_chime2022_pres_dense.npz"
DEFAULT_BUILD_COMMAND = (
    "python scripts/build_bank.py --config chime2022 "
    "--cosmology planck2018 --p-res 1.0 --dense-knee "
    "--out data/fisher_bank_chime2022_pres_dense.npz"
)
_ANY_KFG = object()

PERBIN_APPENDIX_A = "perbin_appendix_a"
OVERVIEW_COMBINED_MULTIBIN = "overview_combined_multibin"
ESTIMATORS = (PERBIN_APPENDIX_A, OVERVIEW_COMBINED_MULTIBIN)

NOISE_NORMALIZED_AT_EACH_TIME = "noise_normalized_at_each_time"
FIXED_PHYSICAL_AT_REFERENCE_TIME = "fixed_physical_at_reference_time"
TIME_SCALINGS = (
    NOISE_NORMALIZED_AT_EACH_TIME,
    FIXED_PHYSICAL_AT_REFERENCE_TIME,
)

REPORT_SCHEMA = "baonoise-bias-tolerance-v1"
JSON_FORMATS = ("legacy", "complete-v1")
FISHER_CONDITION_LIMIT = 1e12
FISHER_NULLSPACE_RTOL = np.sqrt(np.finfo(float).eps)

# The per-bin parameter set is identical to the one the forecast marginalises
# over: {A, sigma_NL, aperp, apar, bs8, fs8}. Keeping b_HI, f and sigma8tot
# beside the derived bs8/fs8 combinations makes the matrix rank-deficient by
# construction.
EXCLUDE = ("b_HI", "f", "Tb", "sigma8tot", "n_s", "pk")

# This is the Overview plot_dv_forecasts.py sequence, adjusted for the current
# parameter spelling. The EOS projection is exercised before these global
# rows are removed, matching the published distance-estimator construction.
# ``sigma_8`` must be included explicitly: the historical script used the old
# spelling ``sigma8``, which leaves a numerically dependent row in current
# RadioFisher matrices.
COMBINED_EXCLUDE = (
    "Tb", "sigma8tot", "n_s", "omegak", "omegaDE", "w0", "wa",
    "h", "gamma", "sigma_8", "N_eff", "pk", "f", "b_HI",
)
COMBINED_EXPAND = ("A", "bs8", "fs8", "DV", "F", PRES)


def _exact_numeric(value, expected):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and float(value) == float(expected))


def _is_unit_response(value) -> bool:
    """Whether metadata describes a unit thermal-normalized response."""
    if _exact_numeric(value, 1.0):
        return True
    if not isinstance(value, dict):
        return False
    try:
        validate_template_metadata(value)
    except (TypeError, ValueError):
        return False
    return True


def _sha256_json(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_manifest(manifest: dict) -> dict:
    return {
        "include": list(manifest["include"]),
        "exclude": list(manifest["exclude"]),
    }


def _bank_build_identity(bank) -> dict:
    """Canonical build-time identity with bank-level settings attached."""
    identity = json.loads(json.dumps(
        bank.meta["provenance"], sort_keys=True, allow_nan=False))
    for name in ("expt_overrides", "foreground_settings"):
        settings = json.loads(json.dumps(
            bank.meta[name], sort_keys=True, allow_nan=False))
        identity[name] = {
            "sha256": _sha256_json(settings),
            "settings": settings,
        }
    return identity


def _evaluation_identity(bank, *, rf_dir=None) -> tuple[dict, dict]:
    """Reconstruct and authenticate every scientific input used at runtime."""
    build = bank.meta["provenance"]
    expected_bao_manifest = _canonical_manifest(BAONOISE_SOURCE_MANIFEST)
    expected_rf_manifest = _canonical_manifest(RADIOFISHER_SOURCE_MANIFEST)
    if build["baonoise"]["source_manifest"] != expected_bao_manifest:
        raise ValueError(
            "bank-build Bao scientific-source manifest differs from the "
            "evaluator's canonical manifest")
    if build["radiofisher"]["source_manifest"] != expected_rf_manifest:
        raise ValueError(
            "bank-build RadioFisher scientific-source manifest differs from "
            "the evaluator's canonical manifest")

    rf, resolved_rf_dir = import_radiofisher(rf_dir)
    capabilities = require_backend_capabilities(
        rf, build["radiofisher"]["capabilities"], rf_dir=resolved_rf_dir)
    bao_state = _git_state(ROOT, **BAONOISE_SOURCE_MANIFEST)
    rf_state = _git_state(resolved_rf_dir, **RADIOFISHER_SOURCE_MANIFEST)
    evaluation_bao = {
        "version": __version__,
        "source_manifest": expected_bao_manifest,
        **bao_state,
    }
    evaluation_rf = {
        "backend_id": getattr(rf, "BACKEND_ID", None),
        "backend_version": getattr(rf, "BACKEND_VERSION", None),
        "api_version": getattr(rf, "BACKEND_API_VERSION", None),
        "capabilities": sorted(capabilities),
        "source_manifest": expected_rf_manifest,
        **rf_state,
    }

    cache_name = build["pk_cache"]["filename"]
    cachefile = filesystem_data_file(cache_name)
    cache_meta = pkcache.inspect_pk_cache(cachefile)
    evaluation_cache = {
        "filename": cache_name,
        "sha256": pkcache.file_sha256(cachefile),
        "cache_id": _sha256_json(cache_meta),
    }
    resolved_cosmo = cosmologies.get(
        bank.meta["cosmology"], rf, resolved_rf_dir)
    evaluation_cosmology = {
        "name": bank.meta["cosmology"],
        "sha256": pkcache.cosmology_fingerprint(resolved_cosmo),
        "parameters": pkcache.cosmology_payload(resolved_cosmo),
        "astrophysical_model_profile":
            resolved_cosmo[cosmologies.ASTROPHYSICAL_PROFILE_KEY],
        "astrophysical_models": {
            key: resolved_cosmo[key] for key in
            ("Tb_model", "bias_HI_model", "omega_HI_model")},
    }
    if not pkcache.cache_matches(cachefile, resolved_cosmo):
        raise ValueError(
            "evaluation P(k) cache does not match the bank's resolved "
            "cosmology and canonical cache settings")
    cosmo = pkcache.load_fiducial_cosmology(
        rf, cachefile, cosmo=resolved_cosmo)

    experiment = survey.experiment_from_bank_metadata(
        rf, resolved_rf_dir, bank.meta, ttot_hours=1.0)
    experiment_payload = experiment_settings_payload(
        experiment, resolved_rf_dir)
    baseline = experiment.get("n(x)")
    baseline_path = Path(baseline) if baseline is not None else None
    evaluation_experiment = {
        "sha256": _sha256_json(experiment_payload),
        "settings": experiment_payload,
        "baseline_sha256": (
            pkcache.file_sha256(baseline_path)
            if baseline_path is not None and baseline_path.is_file() else None),
    }
    evaluation_foregrounds = {
        key: experiment_payload.get(key) for key in FOREGROUND_KEYS}
    evaluation_overrides = json.loads(json.dumps(
        bank.meta["expt_overrides"], sort_keys=True, allow_nan=False))

    evaluation = {
        "baonoise": evaluation_bao,
        "radiofisher": evaluation_rf,
        "cosmology": evaluation_cosmology,
        "pk_cache": evaluation_cache,
        "experiment": evaluation_experiment,
        "expt_overrides": {
            "sha256": _sha256_json(evaluation_overrides),
            "settings": evaluation_overrides,
        },
        "foreground_settings": {
            "sha256": _sha256_json(evaluation_foregrounds),
            "settings": evaluation_foregrounds,
        },
    }
    mismatches = []
    for section in ("baonoise", "radiofisher"):
        recorded = build[section]["working_tree_sha256"]
        current = evaluation[section]["working_tree_sha256"]
        if recorded is None or current is None or recorded != current:
            mismatches.append(
                f"{section}.working_tree_sha256 build={recorded!r} "
                f"evaluation={current!r}")
    for field in ("version",):
        if build["baonoise"][field] != evaluation_bao[field]:
            mismatches.append(
                f"baonoise.{field} build={build['baonoise'][field]!r} "
                f"evaluation={evaluation_bao[field]!r}")
    for field in ("backend_id", "backend_version", "api_version",
                  "capabilities"):
        if build["radiofisher"][field] != evaluation_rf[field]:
            mismatches.append(
                f"radiofisher.{field} build="
                f"{build['radiofisher'][field]!r} "
                f"evaluation={evaluation_rf[field]!r}")
    for section in ("cosmology", "pk_cache", "experiment"):
        if build[section] != evaluation[section]:
            mismatches.append(f"{section} build/evaluation identity differs")
    if bank.meta.get("foreground_settings") != evaluation_foregrounds:
        mismatches.append(
            "foreground_settings build/evaluation identity differs")
    if bank.meta.get("expt_overrides") != evaluation_overrides:
        mismatches.append(
            "expt_overrides build/evaluation identity differs")
    if mismatches:
        raise ValueError(
            "scientific evaluation identity does not match the response-bank "
            "build: " + "; ".join(mismatches))

    context = {
        "rf": rf,
        "rf_dir": resolved_rf_dir,
        "cosmo": cosmo,
        "cosmo_fns": rf.background_evolution_splines(cosmo),
        "cachefile": cachefile,
    }
    return evaluation, context


def load_bias_bank(path, *, build_command=DEFAULT_BUILD_COMMAND,
                   expected_kfg_fac=_ANY_KFG, rf_dir=None):
    """Load an explicitly generated strict-v2 unit-response bank."""
    path = Path(path)
    instruction = (
        "Bias-response banks are deliberately not shipped. Build the exact "
        "prerequisite with:\n  " + build_command
    )
    if not path.is_file():
        raise ValueError(f"required bias-response bank is missing: {path}\n"
                         f"{instruction}")
    try:
        bank = FisherBank(path)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{path} is not a valid strict-v2 Fisher bank: {exc}\n"
            f"{instruction}") from exc

    overrides = bank.meta.get("expt_overrides")
    provenance_settings = bank.meta["provenance"]["experiment"]["settings"]
    problems = []
    if bank.artifact_kind != ARTIFACT_BIAS_RESPONSE or PRES not in bank.paramnames:
        problems.append("artifact_kind must be 'bias_response' with a '_Pres' row")
    if bank.meta.get("config") != "chime2022" \
            or bank.meta.get("cosmology") != "planck2018":
        problems.append("configuration must be chime2022/planck2018")
    if bank.meta.get("astrophysical_model_profile") != "chime_overview_2022":
        problems.append("the canonical chime_overview_2022 profile is required")
    if not isinstance(overrides, dict) or not _is_unit_response(
            overrides.get("P_res")):
        problems.append(
            "expt_overrides.P_res must describe a unit thermal-normalized "
            "response")
    if not _is_unit_response(provenance_settings.get("P_res")):
        problems.append(
            "experiment provenance must record the unit thermal-normalized "
            "P_res response")
    actual_kfg = (overrides.get("kfg_fac")
                  if isinstance(overrides, dict) else None)
    if expected_kfg_fac is _ANY_KFG:
        kfg_matches = True
    elif expected_kfg_fac is None:
        kfg_matches = actual_kfg is None
    else:
        kfg_matches = _exact_numeric(actual_kfg, expected_kfg_fac)
    if not kfg_matches:
        problems.append(
            f"expt_overrides.kfg_fac must equal {expected_kfg_fac!r}")
    if problems:
        raise ValueError(
            f"{path} is incompatible with the bias workflow: "
            + "; ".join(problems) + f"\n{instruction}")
    try:
        evaluation, context = _evaluation_identity(bank, rf_dir=rf_dir)
    except (FileNotFoundError, KeyError, ModuleNotFoundError, RuntimeError,
            TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} cannot be authenticated for evaluation: {exc}\n"
            f"{instruction}") from exc
    bank.evaluation_identity = evaluation
    bank.evaluation_context = context
    return bank


# ---------------------------------------------------------------------------
# Legacy Appendix-A helpers. These signatures are used by plot_convergence.py
# and three_worlds.py, so they remain stable and numerically unchanged.
# ---------------------------------------------------------------------------
def split(F, names, targets=("aperp", "apar", "fs8")):
    """(F_theta-theta, F_theta-A, kept names) with zero-information rows cut."""
    del targets  # retained for the historical public signature
    ia = names.index(PRES)
    keep = [i for i, n in enumerate(names)
            if n != PRES and n not in EXCLUDE and F[i, i] > 0.0]
    Ftt = F[np.ix_(keep, keep)]
    Ftt = 0.5 * (Ftt + Ftt.T)
    FtA = F[keep, ia]
    return Ftt, FtA, [names[i] for i in keep]


def condition(F, names):
    Ftt, _, _ = split(F, names)
    return float(np.linalg.cond(Ftt))


def bias_per_unit_r(F, names):
    """d(theta)/dA and sigma(theta), for one bin's Fisher matrix."""
    Ftt, FtA, kept = split(F, names)
    try:
        cov = np.linalg.inv(Ftt)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Ftt)
    dtheta = cov @ FtA
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return dict(zip(kept, dtheta)), dict(zip(kept, sigma))


def stability(bank, ib, t_hours, names, param, frac=0.10):
    """Historical +/-frac stability tuple used by existing research scripts."""
    vals, signs = [], []
    for scale in (1.0 - frac, 1.0, 1.0 + frac):
        dth, sig = bias_per_unit_r(bank.F(ib, t_hours * scale), names)
        if param not in dth or dth[param] == 0.0:
            return np.inf, 0
        vals.append(sig[param] / abs(dth[param]))
        signs.append(np.sign(dth[param]))
    lo, hi = min(vals), max(vals)
    return (hi / lo if lo > 0 else np.inf), len(set(signs))


class FisherSystem(NamedTuple):
    fisher: np.ndarray
    response: np.ndarray
    names: tuple[str, ...]
    response_name: str


class PerBinAppendixAEstimator:
    """Existing one-bin-at-a-time Appendix-A bias estimator."""

    name = PERBIN_APPENDIX_A
    default_params = ("aperp", "apar", "fs8")

    def __init__(self, bank):
        self.bank = bank

    @property
    def provenance(self):
        return {
            "name": self.name,
            "scope": "one redshift bin inverted independently",
            "radiofisher_backend_required_at_evaluation": False,
            "excluded_parameters": list(EXCLUDE),
            "derived_targets": {
                "DV": "-(2/3 aperp + 1/3 apar) in logarithmic units"},
        }

    def response_name(self, ibin: int) -> str:
        del ibin
        return PRES

    def target_label(self, ibin: int, param: str) -> str | None:
        del ibin
        return param

    def target_metadata(self, ibin: int, param: str) -> dict:
        return {
            "target_units": (
                "dimensionless logarithmic distance shift"
                if param in {"aperp", "apar", "DV"} else "dimensionless"),
            "fiducial_DV_Mpc": (
                _fiducial_dv_mpc(self, ibin) if param == "DV" else None),
        }

    def system(self, ibin: int, t_hours: float) -> FisherSystem:
        Ftt, FtA, names = split(
            self.bank.F(ibin, t_hours), list(self.bank.paramnames))
        return FisherSystem(Ftt, FtA, tuple(names), PRES)

    def target_coefficients(self, system: FisherSystem, ibin: int,
                            param: str):
        del ibin
        coeff = np.zeros(len(system.names))
        if param == "DV":
            if "aperp" not in system.names or "apar" not in system.names:
                return None, None
            coeff[system.names.index("aperp")] = -2.0 / 3.0
            coeff[system.names.index("apar")] = -1.0 / 3.0
            return coeff, "DV"
        if param not in system.names:
            return None, None
        coeff[system.names.index(param)] = 1.0
        return coeff, param


class OverviewCombinedMultibinEstimator:
    """Survey-wide Overview/Fig.-31 estimator plus per-bin bias sources."""

    name = OVERVIEW_COMBINED_MULTIBIN
    default_params = ("DV", "F", "fs8")

    def __init__(self, bank, *, rf=None, rf_dir=None, cosmo=None,
                 cosmo_fns=None, eos_derivs=None):
        self.bank = bank
        verified = getattr(bank, "evaluation_context", None)
        if rf is None and verified is not None:
            rf = verified["rf"]
            rf_dir = verified["rf_dir"]
            cosmo = verified["cosmo"]
            cosmo_fns = verified["cosmo_fns"]
        if rf is None:
            rf, rf_dir = import_radiofisher(rf_dir)
        self.rf = rf
        self.rf_dir = Path(rf_dir).resolve() if rf_dir is not None else None
        required = (
            "expand_fisher_matrix", "transform_to_lss_distances",
            "combined_fisher_matrix", "eos_fisher_matrix_derivs",
        )
        missing = [name for name in required if not callable(getattr(rf, name, None))]
        if missing:
            raise RuntimeError(
                "RadioFisher lacks the combined-estimator API(s): "
                + ", ".join(missing))
        if cosmo is None or cosmo_fns is None:
            from baonoise import cosmologies, pkcache
            from baonoise.resources import filesystem_data_file

            cosmology_name = bank.meta["cosmology"]
            cache_tag = ("" if cosmology_name == "planck2018"
                         else f"_{cosmology_name}")
            cache = filesystem_data_file(
                f"cache_pk_chime2022{cache_tag}.dat")
            cosmo = pkcache.load_fiducial_cosmology(
                rf, cache,
                cosmo=cosmologies.get(cosmology_name, rf, self.rf_dir))
            cosmo_fns = rf.background_evolution_splines(cosmo)
        self.cosmo = cosmo
        self.cosmo_fns = cosmo_fns
        self.eos_derivs = (eos_derivs if eos_derivs is not None
                           else rf.eos_fisher_matrix_derivs(
                               cosmo, cosmo_fns, fsigma8=True))
        self._cache = {}

    @property
    def provenance(self):
        backend = self.bank.meta["provenance"]["radiofisher"]
        return {
            "name": self.name,
            "scope": "all redshift bins in one Fisher inversion",
            "api_sequence": [
                "expand_fisher_matrix(fsigma8=True)",
                "transform_to_lss_distances",
                "combined_fisher_matrix",
            ],
            "expanded_per_bin": list(COMBINED_EXPAND),
            "excluded_parameters": list(COMBINED_EXCLUDE),
            "shared_parameters": ["sigma_NL"],
            "radiofisher": {
                "backend_id": backend["backend_id"],
                "backend_version": backend["backend_version"],
                "api_version": backend["api_version"],
                "git_commit": backend["git_commit"],
                "working_tree_sha256": backend["working_tree_sha256"],
            },
        }

    def _project_bin(self, ibin: int, t_hours: float):
        z = float(self.bank.zc[ibin])
        F, names = self.rf.expand_fisher_matrix(
            z, self.eos_derivs, self.bank.F(ibin, t_hours),
            list(self.bank.paramnames), exclude=[], fsigma8=True)
        F = np.asarray(F, dtype=float)
        names = list(names)

        # The native dilation parameters are converted to the physical units
        # used by the published script before RadioFisher's LSS transform.
        i_da = names.index("aperp")
        i_h = names.index("apar")
        H_fn, r_fn, _, _ = self.cosmo_fns
        h_value = float(H_fn(z))
        da_mpc = float(r_fn(z) / (1.0 + z))
        da_gpc = da_mpc / 1000.0
        names[i_da] = "DA"
        names[i_h] = "H"
        F[i_da, :] *= -1.0 / da_gpc
        F[:, i_da] *= -1.0 / da_gpc
        F[i_h, :] *= 100.0 / h_value
        F[:, i_h] *= 100.0 / h_value
        F, names = self.rf.transform_to_lss_distances(
            z, F, names, DA=da_mpc, H=h_value,
            rescale_da=1000.0, rescale_h=100.0)
        return np.asarray(F, dtype=float), list(names)

    def _combined(self, t_hours: float):
        key = float(t_hours)
        if key in self._cache:
            return self._cache[key]
        matrices = []
        names = None
        for ibin in range(self.bank.nbins):
            projected, projected_names = self._project_bin(ibin, key)
            if names is not None and projected_names != names:
                raise RuntimeError(
                    "RadioFisher projection changed its parameter schema "
                    f"between redshift bins {ibin - 1} and {ibin}")
            matrices.append(projected)
            names = projected_names
        present_exclusions = [name for name in COMBINED_EXCLUDE
                              if name in names]
        F, names = self.rf.combined_fisher_matrix(
            matrices, names=names, exclude=present_exclusions,
            expand=list(COMBINED_EXPAND))
        array = np.asarray(F, dtype=float)
        result = (0.5 * (array + array.T), tuple(names))
        self._cache[key] = result
        return result

    def system(self, ibin: int, t_hours: float) -> FisherSystem:
        F, names = self._combined(t_hours)
        response_names = [f"{PRES}{i}" for i in range(self.bank.nbins)]
        response_name = response_names[ibin]
        if response_name not in names:
            raise RuntimeError(
                f"combined Fisher matrix has no response row {response_name!r}")
        response_index = names.index(response_name)
        keep = [i for i, name in enumerate(names)
                if name not in response_names]
        Ftt = F[np.ix_(keep, keep)]
        FtA = F[keep, response_index]
        kept_names = tuple(names[i] for i in keep)
        return FisherSystem(Ftt, FtA, kept_names, response_name)

    def response_name(self, ibin: int) -> str:
        return f"{PRES}{ibin}"

    def target_label(self, ibin: int, param: str) -> str | None:
        return f"{param}{ibin}"

    def target_metadata(self, ibin: int, param: str) -> dict:
        return {
            "target_units": "Mpc" if param == "DV" else "dimensionless",
            "fiducial_DV_Mpc": (
                _fiducial_dv_mpc(self, ibin) if param == "DV" else None),
        }

    def target_coefficients(self, system: FisherSystem, ibin: int,
                            param: str):
        target = f"{param}{ibin}"
        if target not in system.names:
            return None, None
        coeff = np.zeros(len(system.names))
        coeff[system.names.index(target)] = 1.0
        return coeff, target


def _fiducial_dv_mpc(estimator, ibin: int) -> float | None:
    """Fiducial physical D_V for the target's redshift bin, when available."""
    context = getattr(estimator.bank, "evaluation_context", None)
    rf = getattr(estimator, "rf", None)
    cosmo_fns = getattr(estimator, "cosmo_fns", None)
    if context is not None:
        rf = context["rf"]
        cosmo_fns = context["cosmo_fns"]
    light_speed = getattr(rf, "C", None)
    if light_speed is None or cosmo_fns is None:
        return None
    z = float(estimator.bank.zc[ibin])
    H_fn, r_fn, _, _ = cosmo_fns
    H = float(H_fn(z))
    DA = float(r_fn(z) / (1.0 + z))
    return float(
        ((1.0 + z) ** 2 * DA**2 * light_speed * z / H) ** (1.0 / 3.0))


def make_estimator(bank, name: str, *, rf_dir=None, **kwargs):
    if name == PERBIN_APPENDIX_A:
        return PerBinAppendixAEstimator(bank)
    if name == OVERVIEW_COMBINED_MULTIBIN:
        return OverviewCombinedMultibinEstimator(
            bank, rf_dir=rf_dir, **kwargs)
    raise ValueError(f"unknown estimator {name!r}; choose from {ESTIMATORS}")


def _solve_target(system: FisherSystem, coefficients, target_name: str) -> dict:
    """Return sigma and unit-response bias with null-space diagnostics."""
    F = np.asarray(system.fisher, dtype=float)
    response = np.asarray(system.response, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    base = {
        "target_name": target_name,
        "response_name": system.response_name,
        "sigma": None,
        "dtheta_d_current_noise_ratio": None,
        "condition_number": None,
        "minimum_eigenvalue": None,
        "maximum_eigenvalue": None,
        "eigenvalue_cutoff": None,
        "discarded_eigenmodes": None,
        "eigensystem_preconditioning": "sqrt_fisher_diagonal",
        "preconditioning_scale_minimum": None,
        "preconditioning_scale_maximum": None,
        "valid": False,
        "failure_reason": None,
    }
    if (F.ndim != 2 or F.shape[0] != F.shape[1]
            or response.shape != (F.shape[0],)
            or coefficients.shape != (F.shape[0],)):
        base["failure_reason"] = "fisher_shape_mismatch"
        return base
    if F.size == 0 or not (np.all(np.isfinite(F))
                            and np.all(np.isfinite(response))
                            and np.all(np.isfinite(coefficients))):
        base["failure_reason"] = "nonfinite_or_empty_fisher_system"
        return base

    F = 0.5 * (F + F.T)
    diagonal = np.diag(F)
    scales = np.ones_like(diagonal)
    positive_diagonal = diagonal > 0.0
    scales[positive_diagonal] = np.sqrt(diagonal[positive_diagonal])
    base["preconditioning_scale_minimum"] = float(np.min(scales))
    base["preconditioning_scale_maximum"] = float(np.max(scales))
    F = F / np.outer(scales, scales)
    response = response / scales
    coefficients = coefficients / scales
    eigenvalues, eigenvectors = np.linalg.eigh(F)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    base["minimum_eigenvalue"] = smallest
    base["maximum_eigenvalue"] = largest
    if largest <= 0.0:
        base["failure_reason"] = "nonpositive_fisher_information"
        return base
    cutoff = largest / FISHER_CONDITION_LIMIT
    base["eigenvalue_cutoff"] = float(cutoff)
    if smallest < -cutoff:
        base["failure_reason"] = "materially_negative_fisher_eigenvalue"
        return base
    constrained = eigenvalues > cutoff
    base["discarded_eigenmodes"] = int((~constrained).sum())
    positive = eigenvalues[constrained]
    base["condition_number"] = (
        float(largest / positive[0]) if positive.size else None)
    null_vectors = eigenvectors[:, ~constrained]
    if null_vectors.size:
        target_projection = float(np.linalg.norm(
            null_vectors.T @ coefficients))
        target_scale = max(1.0, float(np.linalg.norm(coefficients)))
        if target_projection > FISHER_NULLSPACE_RTOL * target_scale:
            base["failure_reason"] = "target_overlaps_fisher_nullspace"
            return base
        response_projection = float(np.linalg.norm(null_vectors.T @ response))
        response_scale = max(1.0, float(np.linalg.norm(response)))
        if response_projection > FISHER_NULLSPACE_RTOL * response_scale:
            base["failure_reason"] = "response_overlaps_fisher_nullspace"
            return base

    if np.all(constrained):
        try:
            covariance = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            base["failure_reason"] = "fisher_inverse_failed"
            return base
    else:
        vectors = eigenvectors[:, constrained]
        covariance = (vectors / eigenvalues[constrained]) @ vectors.T
    variance = float(coefficients @ covariance @ coefficients)
    derivative = float(coefficients @ covariance @ response)
    if not np.isfinite(variance) or variance <= 0.0:
        base["failure_reason"] = "nonpositive_or_nonfinite_target_variance"
        return base
    if not np.isfinite(derivative):
        base["failure_reason"] = "nonfinite_bias_response"
        return base
    base.update(
        sigma=float(np.sqrt(variance)),
        dtheta_d_current_noise_ratio=derivative,
        valid=True,
    )
    return base


def _time_scaling_multiplier(time_scaling: str, t_hours: float,
                             reference_hours: float | None) -> float:
    if time_scaling == NOISE_NORMALIZED_AT_EACH_TIME:
        return 1.0
    if time_scaling == FIXED_PHYSICAL_AT_REFERENCE_TIME:
        if reference_hours is None or not np.isfinite(reference_hours) \
                or reference_hours <= 0.0:
            raise ValueError(
                "fixed_physical_at_reference_time requires positive finite "
                "reference_hours")
        return float(t_hours / reference_hours)
    raise ValueError(
        f"unknown time scaling {time_scaling!r}; choose from {TIME_SCALINGS}")


def _bank_time_grid_position(bank, t_hours: float) -> str:
    minimum = float(bank.t_grid[0])
    maximum = float(bank.t_grid[-1])
    if t_hours < minimum:
        return "below_minimum"
    if t_hours > maximum:
        return "above_maximum"
    return "inside"


def _invalid_evaluation(estimator, ibin: int, t_hours: float, param: str,
                        *, time_scaling: str,
                        reference_hours: float | None,
                        position: str, reason: str) -> dict:
    metadata = estimator.target_metadata(ibin, param)
    return {
        "target_name": estimator.target_label(ibin, param),
        "target_units": metadata["target_units"],
        "fiducial_DV_Mpc": metadata["fiducial_DV_Mpc"],
        "response_name": estimator.response_name(ibin),
        "bank_time_grid_position": position,
        "sigma": None,
        "dtheta_d_current_noise_ratio": None,
        "time_scaling_multiplier": _time_scaling_multiplier(
            time_scaling, t_hours, reference_hours),
        "dtheta_d_reported_amplitude": None,
        "r_tolerance_current_noise_ratio": None,
        "r_tolerance": None,
        "condition_number": None,
        "minimum_eigenvalue": None,
        "maximum_eigenvalue": None,
        "eigenvalue_cutoff": None,
        "discarded_eigenmodes": None,
        "eigensystem_preconditioning": "sqrt_fisher_diagonal",
        "preconditioning_scale_minimum": None,
        "preconditioning_scale_maximum": None,
        "valid": False,
        "failure_reason": reason,
    }


def evaluate_raw(estimator, ibin: int, t_hours: float, param: str,
                 *, zeta: float, time_scaling: str,
                 reference_hours: float | None,
                 enforce_bank_bounds: bool = True) -> dict:
    """Evaluate one estimator/bin/time/parameter point before stability gating."""
    position = _bank_time_grid_position(estimator.bank, t_hours)
    if enforce_bank_bounds and position != "inside":
        return _invalid_evaluation(
            estimator, ibin, t_hours, param, time_scaling=time_scaling,
            reference_hours=reference_hours, position=position,
            reason="outside_bank_time_grid")
    system = estimator.system(ibin, t_hours)
    coefficients, target_name = estimator.target_coefficients(
        system, ibin, param)
    if coefficients is None:
        return _invalid_evaluation(
            estimator, ibin, t_hours, param, time_scaling=time_scaling,
            reference_hours=reference_hours, position=position,
            reason="requested_parameter_not_in_estimator")
    solved = _solve_target(system, coefficients, target_name)
    solved.update(estimator.target_metadata(ibin, param))
    solved["bank_time_grid_position"] = position
    multiplier = _time_scaling_multiplier(
        time_scaling, t_hours, reference_hours)
    solved["time_scaling_multiplier"] = multiplier
    derivative = solved["dtheta_d_current_noise_ratio"]
    if not solved["valid"]:
        solved["dtheta_d_reported_amplitude"] = None
        solved["r_tolerance_current_noise_ratio"] = None
        solved["r_tolerance"] = None
        return solved
    noise_normalized_tolerance = float(
        zeta * solved["sigma"] / abs(derivative)) \
        if derivative != 0.0 else None
    solved["r_tolerance_current_noise_ratio"] = noise_normalized_tolerance
    derivative = float(derivative * multiplier)
    solved["dtheta_d_reported_amplitude"] = derivative
    if derivative == 0.0:
        solved["valid"] = False
        solved["failure_reason"] = "zero_bias_response"
        solved["r_tolerance"] = None
        return solved
    tolerance = float(zeta * solved["sigma"] / abs(derivative))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        solved["valid"] = False
        solved["failure_reason"] = "nonpositive_or_nonfinite_tolerance"
        solved["r_tolerance"] = None
        return solved
    solved["r_tolerance"] = tolerance
    return solved


def evaluate_fisher_point(estimator, ibin: int, t_hours: float, param: str,
                          *, zeta: float = 1.0,
                          time_scaling: str = NOISE_NORMALIZED_AT_EACH_TIME,
                          reference_hours: float | None = None,
                          stability_fraction: float = 0.10,
                          max_drift: float = 1.2,
                          enforce_bank_bounds: bool = True) -> dict:
    """Evaluate and gate one requested Fisher point, retaining all evidence."""
    if not np.isfinite(t_hours) or t_hours <= 0.0:
        raise ValueError("t_hours must be positive and finite")
    if not np.isfinite(zeta) or zeta <= 0.0:
        raise ValueError("zeta must be positive and finite")
    if (not np.isfinite(stability_fraction)
            or not 0.0 < stability_fraction < 1.0):
        raise ValueError("stability_fraction must be finite and in (0, 1)")
    if not np.isfinite(max_drift) or max_drift < 1.0:
        raise ValueError("max_drift must be finite and at least 1")

    scales = (
        ("lower", 1.0 - stability_fraction),
        ("central", 1.0),
        ("upper", 1.0 + stability_fraction),
    )
    perturbations = []
    reasons = []
    for label, scale in scales:
        evaluated = evaluate_raw(
            estimator, ibin, t_hours * scale, param, zeta=zeta,
            time_scaling=time_scaling, reference_hours=reference_hours,
            enforce_bank_bounds=enforce_bank_bounds)
        perturbations.append({
            "label": label,
            "scale": float(scale),
            "t_hours": float(t_hours * scale),
            **evaluated,
        })
        if not evaluated["valid"]:
            reasons.append(
                f"{label}:{evaluated['failure_reason']}")

    valid = [point for point in perturbations if point["valid"]]
    drift = None
    reported_drift = None
    sign_count = None
    if len(valid) == len(perturbations):
        # The refusal gate diagnoses interpolation/cancellation in the bank's
        # native response. A fixed physical amplitude has a real t/t_ref
        # movement by definition; applying the drift threshold after that
        # deterministic rescaling would falsely reject a numerically smooth
        # curve merely for obeying the requested physical hypothesis.
        tolerances = [
            point["r_tolerance_current_noise_ratio"] for point in valid]
        drift = float(max(tolerances) / min(tolerances))
        reported_tolerances = [point["r_tolerance"] for point in valid]
        reported_drift = float(
            max(reported_tolerances) / min(reported_tolerances))
        signs = {int(np.sign(point["dtheta_d_reported_amplitude"]))
                 for point in valid}
        sign_count = len(signs)
        if sign_count != 1:
            reasons.append("bias_response_sign_change")
        if drift > max_drift:
            reasons.append("tolerance_drift_exceeds_limit")

    # Preserve order while preventing duplicate reasons.
    rejection_reasons = list(dict.fromkeys(reasons))
    central = perturbations[1]
    return {
        "parameter": param,
        "target_name": central["target_name"],
        "t_hours": float(t_hours),
        "central": central,
        "stability": {
            "fractional_time_perturbation": float(stability_fraction),
            "maximum_allowed_tolerance_ratio": float(max_drift),
            "gate_quantity": "r_tolerance_current_noise_ratio",
            "observed_tolerance_ratio": drift,
            "observed_reported_amplitude_tolerance_ratio": reported_drift,
            "bias_response_sign_count": sign_count,
        },
        "perturbations": perturbations,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dtv_bin_indices(bank) -> list[int]:
    z_dtv = (
        HI_REST_FREQUENCY_MHZ / channels.ATSC_DTV_UPPER_EDGE - 1.0,
        HI_REST_FREQUENCY_MHZ / channels.ATSC_CH14_LOWER_EDGE - 1.0,
    )
    return [i for i in range(bank.nbins)
            if bank.zs[i + 1] > z_dtv[0] and bank.zs[i] < z_dtv[1]]


def build_report(bank, bank_path: Path, estimator, *, bins, years, params,
                 zeta: float, time_scaling: str,
                 reference_hours: float | None,
                 stability_fraction: float, max_drift: float,
                 enforce_bank_bounds: bool = True) -> dict:
    """Build the complete machine-readable report; no point is dropped."""
    bin_reports = []
    for ibin in bins:
        points = []
        for year in years:
            t_hours = float(year * survey.OVERVIEW_ONSKY_YEAR_HOURS)
            points.append({
                "years": float(year),
                "t_hours": t_hours,
                "parameters": {
                    param: evaluate_fisher_point(
                        estimator, ibin, t_hours, param, zeta=zeta,
                        time_scaling=time_scaling,
                        reference_hours=reference_hours,
                        stability_fraction=stability_fraction,
                        max_drift=max_drift,
                        enforce_bank_bounds=enforce_bank_bounds)
                    for param in params
                },
            })
        accepted = [
            record["central"]["r_tolerance"]
            for point in points
            for record in point["parameters"].values()
            if record["accepted"]
        ]
        requested = len(points) * len(params)
        accepted_count = sum(
            record["accepted"]
            for point in points for record in point["parameters"].values())
        bin_reports.append({
            "bin_index": int(ibin),
            "z_low": float(bank.zs[ibin]),
            "z_center": float(bank.zc[ibin]),
            "z_high": float(bank.zs[ibin + 1]),
            "points": points,
            "summary": {
                "requested_parameter_points": int(requested),
                "accepted_parameter_points": int(accepted_count),
                "rejected_parameter_points": int(requested - accepted_count),
                "binding_accepted_tolerance": (
                    float(min(accepted)) if accepted else None),
            },
        })

    bank_response = bank.meta["expt_overrides"]["P_res"]
    canonical_overrides = json.loads(json.dumps(
        bank.meta["expt_overrides"], sort_keys=True, allow_nan=False))
    canonical_foregrounds = json.loads(json.dumps(
        bank.meta["foreground_settings"], sort_keys=True, allow_nan=False))
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bank": {
            "path": str(bank_path.resolve()),
            "filename": bank_path.name,
            "sha256": _sha256_file(bank_path),
            "schema_version": bank.schema_version,
            "artifact_kind": bank.artifact_kind,
            "config": bank.meta["config"],
            "cosmology": bank.meta["cosmology"],
            "astrophysical_model_profile":
                bank.meta["astrophysical_model_profile"],
            "P_res": bank_response,
            "expt_overrides": canonical_overrides,
            "expt_overrides_sha256": _sha256_json(canonical_overrides),
            "foreground_settings": canonical_foregrounds,
            "foreground_settings_sha256": _sha256_json(
                canonical_foregrounds),
            "time_grid": {
                "minimum_hours": float(bank.t_grid[0]),
                "maximum_hours": float(bank.t_grid[-1]),
                "number_of_samples": int(len(bank.t_grid)),
            },
            "built_utc": bank.meta["provenance"]["built_utc"],
            "scientific_identity": {
                "schema": "baonoise-scientific-evaluation-identity-v1",
                "schema_version": 1,
                "bank_build": _bank_build_identity(bank),
                "evaluation": bank.evaluation_identity,
                "verified_equal": True,
            },
        },
        "estimator": estimator.provenance,
        "residual_amplitude": {
            "time_scaling": time_scaling,
            "reported_amplitude_unit": (
                "ratio to thermal-noise power at each evaluated time"
                if time_scaling == NOISE_NORMALIZED_AT_EACH_TIME else
                "ratio to thermal-noise power at the declared reference time"),
            "physical_interpretation": (
                "stationary finite-correlation residual power averaging down "
                "with thermal power"
                if time_scaling == NOISE_NORMALIZED_AT_EACH_TIME else
                "non-averaging persistent physical residual power"),
            "reference_hours": (
                float(reference_hours) if reference_hours is not None else None),
            "reference_years": (
                float(reference_hours / survey.OVERVIEW_ONSKY_YEAR_HOURS)
                if reference_hours is not None else None),
            "thermal_power_scaling_assumption": "P_N proportional to 1/t",
            "bank_response_definition": (
                "unit C_res/P_N shape at each evaluated time; no reported "
                "amplitude is included in C_res"),
            "reported_amplitude_application": (
                "applied exactly once as Delta_A after evaluating the unit "
                "bank response"),
        },
        "request": {
            "zeta": float(zeta),
            "years": [float(year) for year in years],
            "parameters": list(params),
            "bin_indices": [int(ibin) for ibin in bins],
            "stability_fraction": float(stability_fraction),
            "maximum_tolerance_drift_ratio": float(max_drift),
        },
        "bins": bin_reports,
    }


def build_legacy_payload(bank, bank_path: Path, *, bins, years, params,
                         zeta: float, max_drift: float) -> dict:
    """The pre-v1 JSON shape, retained as the default for existing consumers."""
    names = list(bank.paramnames)
    out = []
    for ibin in bins:
        rows = {}
        for year in years:
            t_hours = float(year * survey.OVERVIEW_ONSKY_YEAR_HOURS)
            dtheta, sigma = bias_per_unit_r(bank.F(ibin, t_hours), names)
            record = {}
            for param in params:
                if param not in dtheta or dtheta[param] == 0.0:
                    continue
                r_tolerance = zeta * sigma[param] / abs(dtheta[param])
                drift, sign_count = stability(
                    bank, ibin, t_hours, names, param)
                accepted = drift <= max_drift and sign_count == 1
                record[param] = {
                    "r_tol": float(r_tolerance),
                    "sigma": float(sigma[param]),
                    "dtheta_dr": float(dtheta[param]),
                    "drift": float(drift),
                    "stable": bool(accepted),
                }
            rows[str(year)] = record
        accepted = [
            item["r_tol"] for record in rows.values()
            for item in record.values() if item["stable"]]
        out.append({
            "zlo": float(bank.zs[ibin]),
            "zhi": float(bank.zs[ibin + 1]),
            "rows": rows,
            "binding": float(min(accepted)) if accepted else float("nan"),
        })
    return {"zeta": float(zeta), "bank": bank_path.name, "bins": out}


def _validate_cli_numbers(ap, args):
    if not np.isfinite(args.zeta) or args.zeta <= 0.0:
        ap.error("--zeta must be positive and finite")
    if any(not np.isfinite(year) or year <= 0.0 for year in args.years):
        ap.error("--years must contain only positive finite values")
    if (not np.isfinite(args.stability_fraction)
            or not 0.0 < args.stability_fraction < 1.0):
        ap.error("--stability-fraction must be finite and in (0,1)")
    if not np.isfinite(args.max_drift) or args.max_drift < 1.0:
        ap.error("--max-drift must be finite and at least 1")
    if args.time_scaling == FIXED_PHYSICAL_AT_REFERENCE_TIME:
        if (args.reference_years is None
                or not np.isfinite(args.reference_years)
                or args.reference_years <= 0.0):
            ap.error(
                "--time-scaling fixed_physical_at_reference_time requires "
                "positive finite --reference-years")
    elif args.reference_years is not None:
        ap.error(
            "--reference-years is meaningful only with "
            "fixed_physical_at_reference_time")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bank", type=Path, default=DEFAULT_BIAS_BANK,
        help="strict-v2 unit-P_res bias-response bank (not shipped; exact "
             f"build prerequisite: {DEFAULT_BUILD_COMMAND})")
    ap.add_argument("--estimator", choices=ESTIMATORS,
                    default=PERBIN_APPENDIX_A)
    ap.add_argument("--time-scaling", choices=TIME_SCALINGS,
                    default=NOISE_NORMALIZED_AT_EACH_TIME)
    ap.add_argument(
        "--reference-years", type=float,
        help="reference time for fixed_physical_at_reference_time, in "
             "8,760-hour Overview on-sky years")
    ap.add_argument("--radiofisher-dir", type=Path)
    ap.add_argument("--zeta", type=float, default=0.3,
                    help="admissible bias as a fraction of the statistical error")
    ap.add_argument(
        "--params", nargs="+",
        help="targets (default: aperp apar fs8 for per-bin; DV F fs8 for "
             "combined)")
    ap.add_argument("--years", nargs="+", type=float,
                    default=[0.25, 1.0, 5.0, 10.0])
    ap.add_argument(
        "--bins", nargs="+", type=int,
        help="zero-based bank-bin indices (default: every bin overlapping "
             "the 470--608 MHz DTV band)")
    ap.add_argument("--stability-fraction", type=float, default=0.10)
    ap.add_argument("--max-drift", type=float, default=1.2,
                    help="largest tolerance ratio across the +/- time "
                         "perturbation before refusal")
    ap.add_argument("--json", type=Path)
    ap.add_argument(
        "--json-format", choices=JSON_FORMATS, default="legacy",
        help="legacy preserves the pre-v1 JSON shape for existing consumers; "
             "complete-v1 emits the versioned provenance and refusal ledger")
    args = ap.parse_args(argv)
    _validate_cli_numbers(ap, args)

    try:
        bank = load_bias_bank(args.bank, rf_dir=args.radiofisher_dir)
    except ValueError as exc:
        ap.error(str(exc))
    bins = dtv_bin_indices(bank) if args.bins is None else list(args.bins)
    if (not bins or len(set(bins)) != len(bins)
            or any(ibin < 0 or ibin >= bank.nbins for ibin in bins)):
        ap.error("--bins must be unique valid bank-bin indices")
    try:
        estimator = make_estimator(
            bank, args.estimator, rf_dir=args.radiofisher_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        ap.error(str(exc))
    params = list(args.params or estimator.default_params)
    if not params or len(set(params)) != len(params):
        ap.error("--params must contain unique target names")
    reference_hours = (
        args.reference_years * survey.OVERVIEW_ONSKY_YEAR_HOURS
        if args.reference_years is not None else None)
    if args.json_format == "legacy" and (
            args.estimator != PERBIN_APPENDIX_A
            or args.time_scaling != NOISE_NORMALIZED_AT_EACH_TIME):
        ap.error(
            "--json-format legacy represents only the historical "
            "perbin_appendix_a/noise_normalized_at_each_time calculation; "
            "use --json-format complete-v1 for this request")
    if (args.json_format == "legacy"
            and args.stability_fraction != 0.10):
        ap.error(
            "--json-format legacy has the historical fixed +/-10% stability "
            "test; use --json-format complete-v1 to change "
            "--stability-fraction")
    report = build_report(
        bank, args.bank, estimator, bins=bins, years=args.years,
        params=params, zeta=args.zeta, time_scaling=args.time_scaling,
        reference_hours=reference_hours,
        stability_fraction=args.stability_fraction,
        max_drift=args.max_drift,
        enforce_bank_bounds=args.json_format == "complete-v1")

    response = bank.meta.get("expt_overrides", {}).get("P_res")
    print(f"bank {args.bank.name}  P_res={response}  zeta={args.zeta}")
    print(f"estimator={args.estimator}  time_scaling={args.time_scaling}"
          + (f"  reference_years={args.reference_years:g}"
             if args.reference_years is not None else ""))
    print(f"bins={bins}  parameters={params}\n")
    for bin_report in report["bins"]:
        print(f"z = {bin_report['z_low']:.2f}-{bin_report['z_high']:.2f}")
        print(f"  {'T (on-sky yr)':>14} "
              + "".join(f"{param:>16}" for param in params))
        for point in bin_report["points"]:
            cells = []
            for param in params:
                record = point["parameters"][param]
                tolerance = record["central"]["r_tolerance"]
                if tolerance is None:
                    cell = "--"
                else:
                    cell = f"{tolerance:.3g}"
                cells.append(f"{cell:>14}{'  ' if record['accepted'] else ' !'}")
            print(f"  {point['years']:14.2f} " + "".join(cells))
        summary = bin_report["summary"]
        binding = summary["binding_accepted_tolerance"]
        print("  binding accepted tolerance: "
              + (f"r <= {binding:.3g}" if binding is not None else "REFUSED")
              + f"  ({summary['rejected_parameter_points']} of "
                f"{summary['requested_parameter_points']} points refused)\n")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        if args.json_format == "legacy":
            payload = build_legacy_payload(
                bank, args.bank, bins=bins, years=args.years,
                params=params, zeta=args.zeta, max_drift=args.max_drift)
            encoded = json.dumps(payload, indent=2)
        else:
            encoded = json.dumps(
                report, indent=2, sort_keys=True, allow_nan=False)
        # Preserve the historical JSON bytes, including its lack of a trailing
        # newline. Complete-v1 follows the repository's newline-terminated
        # machine-output convention.
        suffix = "" if args.json_format == "legacy" else "\n"
        args.json.write_text(encoded + suffix, encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

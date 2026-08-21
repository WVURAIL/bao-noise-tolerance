"""Authenticated analytic residual shapes for bias-response Fisher banks.

RadioFisher's callable interface receives ``(k, u, P_N, P_signal)``. These
families intentionally use only those coordinates. An empirical visibility,
frequency-bin, baseline, or sidereal-time template requires additional data
and is not represented here.

Each callable is also a JSON dictionary. Its metadata includes this module's
API version and canonical source digest, so a loaded bank can fail closed if
the formula implementation at evaluation differs from the implementation
used to build the bank.
"""
from __future__ import annotations

import hashlib
from importlib.resources import files
from numbers import Real

import numpy as np

TEMPLATE_API_VERSION = 1

NOISE_SHAPED = "noise_shaped"
LOW_KPARALLEL = "low_kparallel"
WEDGE_LIKE = "wedge_like"
K_SHELL_LOCALIZED = "k_shell_localized"
FAMILIES = (NOISE_SHAPED, LOW_KPARALLEL, WEDGE_LIKE, K_SHELL_LOCALIZED)

DEFAULT_PARAMETERS = {
    NOISE_SHAPED: {},
    LOW_KPARALLEL: {"k_parallel_scale_mpc_inv": 0.05},
    WEDGE_LIKE: {
        "slope": 1.0,
        "intercept_mpc_inv": 0.02,
        "rolloff_mpc_inv": 0.02,
    },
    K_SHELL_LOCALIZED: {
        "k_center_mpc_inv": 0.12,
        "k_width_mpc_inv": 0.03,
    },
}


def implementation_sha256() -> str:
    """Canonical digest of the exact source that implements the formulas."""
    source = files("baonoise").joinpath("residual_templates.py").read_bytes()
    source = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _nonnegative(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return value


class NamedResidualTemplate(dict):
    """Callable additive-power template with authenticated JSON provenance."""

    def __init__(self, family: str, **parameters):
        if family not in FAMILIES:
            raise ValueError(
                f"unknown residual-template family {family!r}; "
                f"choose from {FAMILIES}")
        expected = DEFAULT_PARAMETERS[family]
        unknown = sorted(set(parameters) - set(expected))
        if unknown:
            raise ValueError(
                f"{family} does not accept parameter(s): " + ", ".join(unknown))
        resolved = {**expected, **parameters}
        if family == LOW_KPARALLEL:
            resolved["k_parallel_scale_mpc_inv"] = _positive(
                resolved["k_parallel_scale_mpc_inv"],
                "k_parallel_scale_mpc_inv")
        elif family == WEDGE_LIKE:
            resolved["slope"] = _nonnegative(resolved["slope"], "slope")
            resolved["intercept_mpc_inv"] = _nonnegative(
                resolved["intercept_mpc_inv"], "intercept_mpc_inv")
            resolved["rolloff_mpc_inv"] = _positive(
                resolved["rolloff_mpc_inv"], "rolloff_mpc_inv")
        elif family == K_SHELL_LOCALIZED:
            resolved["k_center_mpc_inv"] = _nonnegative(
                resolved["k_center_mpc_inv"], "k_center_mpc_inv")
            resolved["k_width_mpc_inv"] = _positive(
                resolved["k_width_mpc_inv"], "k_width_mpc_inv")
        super().__init__(
            family=family,
            amplitude=1.0,
            normalization="thermal_noise_at_evaluation_time",
            template_api_version=TEMPLATE_API_VERSION,
            implementation_sha256=implementation_sha256(),
            parameters=resolved,
        )

    @property
    def family(self) -> str:
        return self["family"]

    @property
    def parameters(self) -> dict:
        return self["parameters"]

    def shape(self, k, u):
        k = np.asarray(k, dtype=float)
        u = np.asarray(u, dtype=float)
        if self.family == NOISE_SHAPED:
            return np.ones(np.broadcast_shapes(k.shape, u.shape), dtype=float)
        if self.family == LOW_KPARALLEL:
            scale = self.parameters["k_parallel_scale_mpc_inv"]
            k_parallel = np.abs(k * u)
            return np.exp(-0.5 * (k_parallel / scale) ** 2)
        if self.family == WEDGE_LIKE:
            k_parallel = np.abs(k * u)
            k_perpendicular = np.abs(k) * np.sqrt(np.clip(
                1.0 - u**2, 0.0, 1.0))
            boundary = (self.parameters["intercept_mpc_inv"]
                        + self.parameters["slope"] * k_perpendicular)
            outside = np.maximum(k_parallel - boundary, 0.0)
            return np.exp(
                -0.5 * (outside / self.parameters["rolloff_mpc_inv"]) ** 2)
        center = self.parameters["k_center_mpc_inv"]
        width = self.parameters["k_width_mpc_inv"]
        return np.exp(-0.5 * ((np.abs(k) - center) / width) ** 2)

    def __call__(self, k, u, noise_power, signal_power):
        del signal_power
        return np.asarray(noise_power, dtype=float) * self.shape(k, u)


def make_template(family: str, parameters: dict | None = None) \
        -> NamedResidualTemplate:
    return NamedResidualTemplate(family, **dict(parameters or {}))


def validate_template_metadata(value: dict) -> dict:
    """Return canonical metadata or reject stale/tampered formula identity."""
    if not isinstance(value, dict):
        raise TypeError("named residual-template metadata must be an object")
    family = value.get("family")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("named residual-template parameters must be an object")
    expected = dict(make_template(family, parameters))
    if value != expected:
        raise ValueError(
            "named residual-template metadata does not match the installed "
            "authenticated implementation; rebuild the response bank")
    return expected


def parse_parameter_assignments(assignments) -> dict:
    """Parse repeatable CLI ``NAME=VALUE`` assignments without silent wins."""
    out = {}
    for assignment in assignments or ():
        if assignment.count("=") != 1:
            raise ValueError(
                f"template parameter must use NAME=VALUE, got {assignment!r}")
        name, raw = assignment.split("=", 1)
        if not name or name.strip() != name or name in out:
            raise ValueError(
                "template parameter name is empty, padded, or duplicated: "
                f"{name!r}")
        try:
            out[name] = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"template parameter {name!r} must have a numeric value") \
                from exc
    return out

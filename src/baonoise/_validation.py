"""Shared scalar validation for public and internal forecast boundaries."""
from __future__ import annotations

from numbers import Real

import numpy as np


def real_scalar(value, name: str) -> float:
    """Return a real scalar while rejecting booleans and coercive strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar, got {value!r}")
    return float(value)


def finite_scalar(value, name: str) -> float:
    """Return a finite real scalar."""
    value = real_scalar(value, name)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def positive_scalar(value, name: str) -> float:
    """Return a finite real scalar greater than zero."""
    value = finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")
    return value


def nonnegative_scalar(value, name: str) -> float:
    """Return a finite real scalar greater than or equal to zero."""
    value = finite_scalar(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value

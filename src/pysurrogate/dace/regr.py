"""Regression (trend) basis for Dace -- re-exported from the shared :mod:`pysurrogate.core.regression`."""

from pysurrogate.core.regression import (
    ConstantRegression,
    LinearRegression,
    QuadraticRegression,
    Regression,
)

__all__ = ["Regression", "ConstantRegression", "LinearRegression", "QuadraticRegression"]

"""Benchmark and select surrogate models: metrics, cross-validation, function sweeps, selection."""

from pysurrogate.core.prediction import predictions_frame
from pysurrogate.selection.benchmark import (
    AutoModel,
    Benchmark,
    FunctionBenchmark,
    score,
)
from pysurrogate.selection.factory import as_named, cartesian
from pysurrogate.selection.study import StudyResult, default_kriging, default_models, study

__all__ = [
    "AutoModel",
    "Benchmark",
    "FunctionBenchmark",
    "score",
    "predictions_frame",
    "cartesian",
    "as_named",
    "study",
    "StudyResult",
    "default_models",
    "default_kriging",
]

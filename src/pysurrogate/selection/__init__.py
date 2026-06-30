"""Benchmark and select surrogate models: metrics, cross-validation, function sweeps, selection."""

from pysurrogate.core.prediction import predictions_frame
from pysurrogate.selection.benchmark import (
    Benchmark,
    FunctionBenchmark,
    ModelSelection,
    score,
)
from pysurrogate.selection.factory import as_named, cartesian
from pysurrogate.selection.study import StudyResult, default_kriging, default_models, study

__all__ = [
    "Benchmark",
    "FunctionBenchmark",
    "ModelSelection",
    "score",
    "predictions_frame",
    "cartesian",
    "as_named",
    "study",
    "StudyResult",
    "default_models",
    "default_kriging",
]

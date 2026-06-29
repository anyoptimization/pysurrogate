"""Model comparison and selection: benchmark candidates and pick the best."""

from pysurrogate.selection.benchmark import Benchmark
from pysurrogate.selection.factory import as_named, cartesian
from pysurrogate.selection.selection import ModelSelection
from pysurrogate.selection.study import StudyResult, default_models, study

__all__ = ["Benchmark", "ModelSelection", "cartesian", "as_named", "study", "StudyResult", "default_models"]

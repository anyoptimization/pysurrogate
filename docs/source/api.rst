API reference
=============

The full public surface, grouped to match the narrative pages. Everything here is importable
from the top-level ``pysurrogate`` package unless noted.

Core lifecycle
--------------

.. autoclass:: pysurrogate.Model
   :members:

.. autoclass:: pysurrogate.Prediction
   :members:

.. autofunction:: pysurrogate.predictions_frame

Kriging / DACE
--------------

.. autoclass:: pysurrogate.Dace
   :members:

.. autoclass:: pysurrogate.Kriging
   :members:

Correlation kernels
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pysurrogate.dace.corr
   :members:

Regression trends
~~~~~~~~~~~~~~~~~~

.. automodule:: pysurrogate.dace.regr
   :members:

Model backends
--------------

.. autoclass:: pysurrogate.RBF
   :members:

.. autoclass:: pysurrogate.SVR
   :members:

.. autoclass:: pysurrogate.KNN
   :members:

.. autoclass:: pysurrogate.InverseDistanceWeighting
   :members:

.. autoclass:: pysurrogate.SimpleMean
   :members:

.. autoclass:: pysurrogate.PolynomialRegression
   :members:

.. autoclass:: pysurrogate.RandomForest
   :members:

Optimizer layer
---------------

.. autoclass:: pysurrogate.Problem
   :members:

.. autoclass:: pysurrogate.Optimizer
   :members:

.. autoclass:: pysurrogate.Callback
   :members:

.. autoclass:: pysurrogate.Evaluation
   :members:

.. autoclass:: pysurrogate.Result
   :members:

.. autoclass:: pysurrogate.LBFGS
   :members:

.. autoclass:: pysurrogate.PatternSearch
   :members:

.. autoclass:: pysurrogate.Boxmin
   :members:

.. autoclass:: pysurrogate.Adam
   :members:

.. autoclass:: pysurrogate.Restart
   :members:

Sampling & partitioning
-----------------------

.. autoclass:: pysurrogate.Sampling
   :members:

.. autoclass:: pysurrogate.LHS
   :members:

.. autoclass:: pysurrogate.Random
   :members:

.. autoclass:: pysurrogate.Partitioning
   :members:

.. autoclass:: pysurrogate.CrossvalidationPartitioning
   :members:

.. autoclass:: pysurrogate.RandomPartitioning
   :members:

.. autoclass:: pysurrogate.Split
   :members:

Transformations
~~~~~~~~~~~~~~~~

.. automodule:: pysurrogate.core.transformation
   :members:

Benchmarking & selection
------------------------

.. autoclass:: pysurrogate.Benchmark
   :members:

.. autoclass:: pysurrogate.AutoModel
   :members:

.. autoclass:: pysurrogate.FunctionBenchmark
   :members:

.. autofunction:: pysurrogate.study

.. autoclass:: pysurrogate.StudyResult
   :members:

.. autofunction:: pysurrogate.score

.. autofunction:: pysurrogate.cartesian

.. autofunction:: pysurrogate.as_named

Metrics
~~~~~~~

.. automodule:: pysurrogate.selection.metrics
   :members:

Test functions
~~~~~~~~~~~~~~~

.. automodule:: pysurrogate.util.test_functions
   :members:

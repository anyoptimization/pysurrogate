"""Tests for the pysurrogate logging helpers (NullHandler default, enable/disable)."""

import logging

import numpy as np

import pysurrogate
from pysurrogate import Kriging
from pysurrogate.util.logging import disable_logging, enable_logging, get_logger


def test_root_logger_has_a_null_handler_by_default():
    # importing the package must not emit to an unconfigured root logger.
    handlers = logging.getLogger("pysurrogate").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_get_logger_names_are_under_the_package_root():
    assert get_logger().name == "pysurrogate"
    assert get_logger("dace").name == "pysurrogate.dace"
    assert get_logger("models.rbf").name == "pysurrogate.models.rbf"


def test_enable_logging_emits_records_and_disable_removes_the_handler():
    logger = get_logger()
    try:
        handler = enable_logging(logging.DEBUG)
        assert handler in logger.handlers
        assert logger.level == logging.DEBUG

        # a fit should now produce at least one INFO record on the model logger.
        records = []
        probe = logging.Handler()
        probe.emit = records.append  # type: ignore[method-assign]
        logger.addHandler(probe)
        rng = np.random.default_rng(0)
        X = rng.random((12, 2))
        Kriging().fit(X, (X**2).sum(axis=1))
        logger.removeHandler(probe)
        assert any(r.name == "pysurrogate.model" and "fit" in r.getMessage() for r in records)
    finally:
        disable_logging()
    assert not any(h.get_name() == "pysurrogate-stream" for h in logger.handlers)


def test_enable_logging_is_idempotent_and_does_not_stack_handlers():
    logger = get_logger()
    try:
        enable_logging(logging.INFO)
        enable_logging(logging.WARNING)
        streams = [h for h in logger.handlers if h.get_name() == "pysurrogate-stream"]
        assert len(streams) == 1
        assert logger.level == logging.WARNING
    finally:
        disable_logging()


def test_enable_logging_is_exported_at_the_root():
    assert pysurrogate.enable_logging is enable_logging
    assert pysurrogate.disable_logging is disable_logging

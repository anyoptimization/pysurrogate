"""Library logging: a lazily-configured logger tree rooted at the ``pysurrogate`` namespace."""

import logging

# The package root logger. Following stdlib library convention, pysurrogate never configures
# logging itself -- it only emits records and attaches a NullHandler (in the package __init__)
# so that an application that never configures logging sees nothing. Turn output on with
# ``enable_logging()`` (a convenience for exploration) or the host application's own handlers.
ROOT = "pysurrogate"


def get_logger(name=None):
    """Return the pysurrogate logger for a submodule.

    Args:
        name: Dotted sub-name below the package root (e.g. ``"dace"`` or ``"models.rbf"``).
            ``None`` returns the package-root logger.

    Returns:
        The ``logging.Logger`` named ``pysurrogate[.name]``.
    """
    return logging.getLogger(ROOT if not name else f"{ROOT}.{name}")


def enable_logging(level=logging.INFO, *, stream=None, fmt=None):
    """Attach a stream handler to the pysurrogate logger -- a one-call switch for exploration.

    This is a convenience for interactive use and debugging (e.g. ``enable_logging(logging.DEBUG)``
    to watch a fit's hyperparameter search); a real application should configure logging itself and
    leave the library's NullHandler in place. Calling it repeatedly does not stack handlers -- the
    previously installed one is replaced, and the level is updated.

    Args:
        level: The logging level for the pysurrogate logger (e.g. ``logging.DEBUG``).
        stream: Destination stream (default: ``sys.stderr`` via ``StreamHandler``).
        fmt: A ``logging.Formatter`` format string; a compact default is used when ``None``.

    Returns:
        The installed ``logging.Handler`` (so it can be removed again with :func:`disable_logging`).
    """
    logger = get_logger()
    disable_logging()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt or "%(asctime)s  %(levelname)-5s  %(name)s  %(message)s", "%H:%M:%S"))
    handler.set_name("pysurrogate-stream")
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler


def disable_logging():
    """Remove the handler installed by :func:`enable_logging` (leaving the library's NullHandler)."""
    logger = get_logger()
    for handler in [h for h in logger.handlers if h.get_name() == "pysurrogate-stream"]:
        logger.removeHandler(handler)

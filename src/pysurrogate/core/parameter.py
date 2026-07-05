"""Tunable kernel parameters: how each length-scale or shape coordinate is sized, bounded, and encoded."""

import numpy as np


class Encoding:
    """How a parameter's value maps to the optimizer's search coordinate, and how its bounds transform.

    A search runs in an encoded coordinate space (e.g. ``log10`` for a positive length-scale) while
    the kernel consumes the decoded value. An encoding is that map: :meth:`to_value` decodes a search
    coordinate to the parameter value, and :meth:`bounds` transforms value-space bounds into the
    coordinate space the optimizer searches. It is the single seam that lets a parameter declare *how*
    it is searched instead of the search problem hard-coding one convention.
    """

    def to_value(self, x):
        """Decode a search coordinate into the parameter value."""
        raise NotImplementedError

    def bounds(self, lo, hi):
        """Map value-space bounds ``(lo, hi)`` into coordinate space."""
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__


class Log10(Encoding):
    """Base-10 log encoding ``value = 10**x`` -- the positive-scale case (Dace length-scales, nugget).

    Keeps a positive parameter positive under an unconstrained search and spreads a wide dynamic
    range evenly.
    """

    def to_value(self, x):
        return 10.0**x

    def bounds(self, lo, hi):
        return np.log10(lo), np.log10(hi)


class Parameter:
    """One tunable quantity of a kernel: its name, coordinate count, bounds, and encoding.

    A kernel *declares* its parameters (``Kernel.parameters(d)``); the search layer concatenates the
    declarations of every composed component into one flat vector. ``size`` is the number of scalar
    search coordinates the parameter occupies (1 for a shared/isotropic length-scale or a scalar
    shape parameter, ``d`` for a per-dimension ARD length-scale, ``h`` for a reduced/rotated metric).

    Args:
        name: Identifier, unique within a kernel's parameter list (e.g. ``"theta"``, ``"power"``).
        size: Number of scalar coordinates. For a ``fill`` parameter this is only the default/declared
            count -- a search may resize it (see ``fill``).
        bounds: Default ``(lo, hi)`` in value space, each broadcastable to ``size``. A search may
            override these (the length-scale bounds are a search setting, not intrinsic to a kernel).
        encoding: The value<->coordinate map (defaults to :class:`Log10`, the positive-scale case).
        fill: Whether this parameter's size is *caller-driven* -- the length-scale block, whose ARD
            count comes from the bounds / start vector the caller supplies, not from the kernel's own
            ``ard`` flag (a diagonal kernel can be driven with per-dimension length-scales). A search
            sizes the one ``fill`` parameter as the total coordinate count minus the fixed shape
            parameters (e.g. an exponent). Fixed shape parameters keep ``fill=False``.
    """

    def __init__(self, name, size=1, bounds=(0.0, 100.0), encoding=None, fill=False):
        self.name = name
        self.size = size
        self.bounds = bounds
        self.encoding = encoding if encoding is not None else Log10()
        self.fill = fill

    def __repr__(self):
        return f"Parameter({self.name!r}, size={self.size}, encoding={self.encoding!r})"


class ParameterSpace:
    """An ordered concatenation of :class:`Parameter` declarations -> one flat search vector.

    Turns a list of per-component parameter declarations into the two things a search needs: the
    coordinate-space bounds, and the ``decode`` from a flat vector to a per-name value dict. It is
    pure layout/encoding bookkeeping -- it holds no data and does no kernel math.
    """

    def __init__(self, params):
        self.params = list(params)

    def bounds(self):
        """Coordinate-space ``(lo, hi)`` for the whole vector, encoding each parameter's value bounds."""
        if not self.params:
            return np.zeros(0), np.zeros(0)
        los, his = [], []
        for p in self.params:
            lo, hi = p.encoding.bounds(*p.bounds)
            los.append(np.broadcast_to(lo, p.size))
            his.append(np.broadcast_to(hi, p.size))
        return np.concatenate(los), np.concatenate(his)

    def decode(self, x):
        """Split a flat search vector into a ``{name: value}`` dict (each value decoded to value space)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out, i = {}, 0
        for p in self.params:
            out[p.name] = p.encoding.to_value(x[i : i + p.size])
            i += p.size
        return out

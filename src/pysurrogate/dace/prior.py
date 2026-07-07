"""Priors on the length-scale search: a MAP penalty folded into the DACE likelihood objective."""

import numpy as np

_LN10 = np.log(10.0)


class Prior:
    """A MAP penalty on the encoded (``log10``) length-scales, added to the search objective.

    Supplies ``penalty(Z)`` -- the per-candidate penalty added to the negative log-likelihood -- and
    ``grad(Z)`` -- its gradient in the encoded (``log10``) coordinate the optimizer searches. Both act
    on the length-scale coordinates only (never the nugget). A ``Prior`` (or the ``(mean, lam)`` tuple
    shorthand for :class:`GaussianPrior`) is what ``theta_prior`` accepts.
    """

    def penalty(self, Z):
        """Per-candidate penalty ``(J,)`` for the encoded length-scales ``Z`` of shape ``(J, p)``."""
        raise NotImplementedError

    def grad(self, Z):
        """Gradient of the penalty w.r.t. ``Z``, shape ``(J, p)``."""
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__


class GaussianPrior(Prior):
    """Gaussian (Tikhonov) prior ``lam * sum((log10 theta - mean)**2)`` on the length-scales.

    Pulls the length-scales toward ``10**mean`` and away from the short-length-scale over-fitting pure
    maximum likelihood falls into on small designs. The ``(mean, lam)`` tuple form of ``theta_prior``
    resolves to this.

    Args:
        mean: Prior centre on ``log10(theta)`` (``0`` centres on unit length-scale).
        lam: Prior strength (larger regularizes harder).
    """

    def __init__(self, mean=0.0, lam=0.01):
        self.mean = float(mean)
        self.lam = float(lam)

    def penalty(self, Z):
        return self.lam * np.sum((Z - self.mean) ** 2, axis=1)

    def grad(self, Z):
        return 2.0 * self.lam * (Z - self.mean)

    def __repr__(self):
        return f"GaussianPrior(mean={self.mean}, lam={self.lam})"


def resolve_prior(theta_prior):
    """Resolve ``theta_prior`` into a :class:`Prior` (or ``None``).

    ``None`` -> ``None``; a ``(mean, lam)`` tuple -> :class:`GaussianPrior`; a :class:`Prior` -> itself.
    """
    if theta_prior is None or isinstance(theta_prior, Prior):
        return theta_prior
    mean, lam = theta_prior
    return GaussianPrior(mean, lam)

"""The kernel zoo: GP-covariance and radial-basis kernels as first-class objects with gradients."""

import numpy as np

from pysurrogate.core.parameter import Log10, Parameter


def _asvec(theta):
    """Length-scales as a 1-D float array (accepts a scalar or a vector ``theta``)."""
    return np.atleast_1d(np.asarray(theta, dtype=float))


def _asmat(thetas):
    """A theta population as a 2-D float array, shape ``(J, p)`` (accepts a single row)."""
    return np.atleast_2d(np.asarray(thetas, dtype=float))


def calc_kernel_matrix(A, B, func, theta):
    """Correlation matrix between the rows of ``A`` and ``B`` for one kernel at one ``theta``.

    Builds the pairwise coordinate differences and evaluates ``func`` (a kernel) on them.

    Args:
        A: Left points, shape ``(nA, d)``.
        B: Right points, shape ``(nB, d)``.
        func: The correlation kernel, called as ``func(D, theta)``.
        theta: Length-scale parameters for the kernel.

    Returns:
        The correlation matrix, shape ``(nA, nB)``.
    """
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    K = func(D, theta)
    return np.reshape(K, (A.shape[0], B.shape[0]))


def calc_kernel_tensor(A, B, kernel, thetas, D=None):
    """Correlation matrices for a whole population of theta at once.

    The batched counterpart of ``calc_kernel_matrix``: the same pairwise differences,
    evaluated for every theta in ``thetas`` in one ``kernel.batch`` call.

    Args:
        A: Left points, shape ``(nA, d)``.
        B: Right points, shape ``(nB, d)``.
        kernel: The correlation kernel.
        thetas: Population of length-scales, shape ``(J, p)``.
        D: Optional precomputed componentwise differences ``(nA*nB, d)`` in the same layout
            this function would build them. A theta search reuses one ``D`` across all its
            evaluations (the differences are theta-independent), so passing it avoids rebuilding
            the ``(nA*nB, d)`` matrix on every call -- and lets a reducing kernel (KPLS) cache its
            per-fit projection keyed on that one array.

    Returns:
        The stacked correlation matrices, shape ``(J, nA, nB)``.
    """
    if D is None:
        D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    K = kernel.batch(D, thetas)
    return K.reshape(K.shape[0], A.shape[0], B.shape[0])


# -------------------------------
# Correlation Kernels
# -------------------------------


class Kernel:
    """A kernel: a callable object bundling its own gradients.

    A kernel maps componentwise distances ``D`` and length-scales ``theta`` to a
    correlation vector, and carries the two derivatives the model may need:

    - ``__call__(D, theta)`` -- the correlation itself.
    - ``grad(D, theta)`` -- derivative w.r.t. the design point, used by ``predict(grad=True)``.
    - ``theta_grad(D, theta)`` -- derivative w.r.t. ``theta``, one column per scalar in
      ``theta``, used by gradient-based optimizers (e.g. ``LBFGS``).

    ``theta_grad`` is implemented here once and delegates to the optional
    ``_dtheta_per_dim`` hook (the per-dimension theta partials); this base collapses the
    isotropic case so kernels don't each repeat it. A kernel supplies just that hook --
    or overrides ``theta_grad`` outright when its ``theta`` is laid out differently (e.g.
    ``GeneralizedExponential``, which also tunes an exponent). ``grad`` / ``theta_grad``
    are *optional*: a kernel implementing neither leaves ``theta_grad`` raising
    ``NotImplementedError`` and gradient-based consumers fall back (numerical gradient /
    derivative-free ``Boxmin``). ``theta`` carries the length-scale(s) -- scalar
    (isotropic) or per-dimension vector (ARD) -- and is the only model-tuned parameter;
    any fixed shape parameter (e.g. RQ's ``alpha``) lives on the kernel object instead.

    ``ard`` declares the default length-scale layout: ``ard=False`` is the isotropic / RBF use
    (one shared length-scale), ``ard=True`` the per-dimension / DACE use (one per input
    dimension). The math itself is driven by the *shape* of the ``theta`` actually passed (so
    the same kernel evaluates either way). A search reads a kernel's tunables from
    :meth:`parameters` (which sizes the length-scale block from ``ard`` via :meth:`n_theta`);
    :meth:`n_theta` remains as the quick per-``d`` coordinate count.
    """

    def __init__(self, ard=False):
        self.ard = ard

    def n_theta(self, d):
        """Number of length-scale coordinates for ``d`` input dimensions (``d`` if ARD, else 1)."""
        return d if self.ard else 1

    def parameters(self, d):
        """The tunable parameters this kernel exposes to a search, for ``d`` input dimensions.

        A declaration only -- names, coordinate counts, default bounds and encodings; the search
        layer concatenates the declarations of every composed component into one flat vector (see
        :class:`~pysurrogate.core.parameter.ParameterSpace`). The default is a single ``log10``
        length-scale vector sized by :meth:`n_theta`; kernels with extra shape coordinates (e.g.
        :class:`GeneralizedExponential`) or no search at all (the radial bases) override this. The
        length-scale block is a ``fill`` parameter: its ARD count is caller-driven (from the supplied
        bounds / start), so a diagonal kernel may be driven with per-dimension length-scales.
        """
        return [Parameter("theta", size=self.n_theta(d), encoding=Log10(), fill=True)]

    def __call__(self, D, theta):
        raise NotImplementedError

    def grad(self, D, theta):
        raise NotImplementedError

    def batch(self, D, thetas):
        """Correlation for a population of theta at once: ``(J, n_pairs)``.

        The batched counterpart of ``__call__``. The default stacks per-theta calls --
        always correct, and enough for the optimizers since the per-theta correlation is
        not the cost (the O(n^3) Cholesky downstream is). A kernel whose correlation
        vectorizes cleanly over theta (e.g. ``Gaussian``) overrides this.

        Args:
            D: Componentwise distances, shape ``(n_pairs, d)``.
            thetas: Population of length-scales, shape ``(J, p)``.

        Returns:
            One correlation row per theta, shape ``(J, n_pairs)``.
        """
        return np.stack([self(D, t) for t in np.atleast_2d(thetas)])

    def theta_grad(self, D, theta):
        # one column per scalar in theta. ARD (vector theta) keeps the per-dimension
        # partials; isotropic (a single shared theta) is, by the chain rule, the sum of
        # those partials -> a single column. The kernel only supplies _dtheta_per_dim.
        per_dim = self._dtheta_per_dim(D, theta)
        if np.size(theta) == 1:
            return per_dim.sum(axis=1, keepdims=True)
        return per_dim

    def _dtheta_per_dim(self, D, theta):
        # optional analytic hook: d corr / d theta_k for each input dimension k,
        # shape (n_pairs, d). Left unimplemented -> the kernel has no analytic theta
        # gradient and LBFGS uses a numeric one instead.
        raise NotImplementedError

    @property
    def has_theta_grad(self):
        """Whether this kernel provides an analytic theta-gradient.

        True if it implements the ``_dtheta_per_dim`` hook (the usual path) or overrides
        ``theta_grad`` outright (e.g. ``GeneralizedExponential``). A gradient-based
        optimizer asks the kernel this to choose between the exact Jacobian and a
        finite-difference fallback.
        """
        cls = type(self)
        return cls.theta_grad is not Kernel.theta_grad or cls._dtheta_per_dim is not Kernel._dtheta_per_dim

    def __repr__(self):
        return type(self).__name__


# Back-compat alias: the DACE layer (and its tests) know this base as ``Correlation``.
Correlation = Kernel


def _product_rule(ss, dd):
    """Gradient of a product kernel ``prod_j ss_j`` from per-dimension factors.

    Args:
        ss: Per-dimension correlation factors, shape ``(n_pairs, d)``.
        dd: Per-dimension factor derivatives (w.r.t. the point or w.r.t. theta), shape
            ``(n_pairs, d)``.

    Returns:
        Column ``k`` is ``prod_{j != k} ss_j * dd_k`` (the product rule applied per
        dimension), shape ``(n_pairs, d)``.
    """
    out = np.zeros(ss.shape)
    for k in range(ss.shape[1]):
        cols = ss.copy()
        cols[:, k] = dd[:, k]
        out[:, k] = np.prod(cols, axis=1)
    return out


class ProductKernel(Correlation):
    """A kernel that is a product of per-dimension factors ``M(t_k)``, ``t_k = theta_k |D_k|``.

    Subclasses implement only ``_factor`` -- the per-dimension factor ``M(t)`` and its
    derivative ``M'(t)`` w.r.t. the scaled distance ``t``. The product, the point-gradient
    and the theta-gradient are written once here via the product rule, so a new
    compact-support / Matern-style kernel needs no hand-rolled per-dimension loop (the
    historical source of gradient bugs). Both gradients share the same leave-one-out
    product and differ only in the chain factor: ``dt/dx_k = theta_k sign(D_k)`` for the
    point gradient, ``dt/dtheta_k = |D_k|`` for the theta gradient. ``M'(t)`` must be 0 in
    any clamped / compact-support region so the product rule self-zeroes there.
    """

    def _factor(self, D, theta):
        # returns (ss, dsdt): per-dimension factor M(t) and derivative M'(t), t=theta|D|.
        raise NotImplementedError

    def __call__(self, D, theta):
        ss, _ = self._factor(D, theta)
        return np.prod(ss, axis=1)

    def grad(self, D, theta):
        ss, dsdt = self._factor(D, theta)
        return _product_rule(ss, dsdt * theta * np.sign(D))

    def _dtheta_per_dim(self, D, theta):
        ss, dsdt = self._factor(D, theta)
        return _product_rule(ss, dsdt * np.abs(D))


# -------------------------------
# Metric x Profile composition
# -------------------------------
# A large family of stationary kernels factor as ``k(D, theta) = f(s(D, theta))`` -- a *metric*
# ``s`` (how far apart two points are) fed through a *profile* ``f`` (how correlation decays with
# that distance). Splitting the two makes them independently reusable: one ``Exp`` profile serves
# both the Gaussian (over a squared metric) and the Exponential (over an L1 metric), and one metric
# family (plain / rotated / reduced) serves every profile. The value and both gradients are written
# once in ``ComposedKernel`` via the chain rule; a concrete kernel is just a ``(metric, profile)``
# pair. Kernels that are *not* of this form (compact-support products, scale mixtures) stay on
# ``ProductKernel`` instead -- see ``Cubic``/``Matern``/``RationalQuadratic`` below.


class Metric:
    """A dissimilarity ``s(D, theta)``: coordinate differences -> one scalar per pair.

    A metric owns the length-scales and knows only how to *measure* distance; the correlation
    profile (the curve) is a separate object. Subclasses supply:

    - ``value(D, theta)`` -- ``s``, shape ``(n_pairs,)``.
    - ``spatial_grad(D, theta)`` -- ``ds/dD``, shape ``(n_pairs, d)`` (for ``predict(grad=True)``).
    - ``dtheta(D, theta)`` -- ``ds/dtheta`` per parameter, shape ``(n_pairs, p)`` (for the search).
    - ``batch(D, thetas)`` -- ``s`` for a whole theta population, shape ``(J, n_pairs)``, or
      ``None`` to route the population through the default per-theta stack (see
      :meth:`ComposedKernel.batch`). Returning ``None`` deliberately keeps a metric off the
      GEMM fast path when its exact per-theta summation order must be preserved.

    ``ard`` declares the length-scale layout exactly as on :class:`Kernel` (shared vs per-dim).
    """

    def __init__(self, ard=False):
        self.ard = ard

    def n_theta(self, d):
        """Number of length-scale coordinates for ``d`` input dimensions (``d`` if ARD, else 1)."""
        return d if self.ard else 1

    def parameters(self, d):
        """The length-scale parameter this metric owns: one caller-sized ``log10`` (fill) vector."""
        return [Parameter("theta", size=self.n_theta(d), encoding=Log10(), fill=True)]

    def value(self, D, theta):
        raise NotImplementedError

    def spatial_grad(self, D, theta):
        raise NotImplementedError

    def dtheta(self, D, theta):
        raise NotImplementedError

    def batch(self, D, thetas):
        return None

    def __repr__(self):
        return type(self).__name__


class Profile:
    """A correlation curve ``f(s)``: a scalar dissimilarity -> a correlation.

    Supplies ``f(s)`` and its derivative ``fprime(s) = df/ds``. ``f_batch`` maps a whole
    population of dissimilarities at once (default: elementwise ``f``).
    """

    def f(self, s):
        raise NotImplementedError

    def fprime(self, s):
        raise NotImplementedError

    def f_batch(self, s):
        return self.f(s)

    def parameters(self, d):
        """The profile's own tunable shape parameters (none by default; the curve is fixed)."""
        return []

    def __repr__(self):
        return type(self).__name__


class Exp(Profile):
    """Exponential profile ``f(s) = exp(-s)``.

    Over a squared metric this is the Gaussian (squared-exponential) kernel; over an L1 metric
    it is the Exponential kernel -- the same curve, a different distance.
    """

    def f(self, s):
        return np.exp(-s)

    def fprime(self, s):
        return -np.exp(-s)

    def f_batch(self, s):
        return np.exp(-s)


class ComposedKernel(Kernel):
    """A kernel ``k(D, theta) = f(s(D, theta))``: a :class:`Profile` composed with a :class:`Metric`.

    The value, the spatial gradient (chain rule ``f'(s) * ds/dD``) and the theta gradient
    (``f'(s) * ds/dtheta``, with the isotropic collapse inherited from :class:`Kernel`) are
    written once here; a concrete kernel is just a ``(metric, profile)`` pair. This is the
    ``distance x curve`` seam -- swap the metric to change *how distance is measured* (isotropic,
    ARD, rotated, reduced) without touching the curve, and vice versa.
    """

    def __init__(self, metric, profile):
        super().__init__(ard=metric.ard)
        self.metric = metric
        self.profile = profile

    def n_theta(self, d):
        return self.metric.n_theta(d)

    def parameters(self, d):
        # the composed kernel's search vector is the metric's length-scales followed by any shape
        # parameters the profile tunes -- literal concatenation, the payoff of the distance x curve split.
        return [*self.metric.parameters(d), *self.profile.parameters(d)]

    def __call__(self, D, theta):
        return self.profile.f(self.metric.value(D, theta))

    def grad(self, D, theta):
        s = self.metric.value(D, theta)
        return self.profile.fprime(s)[:, None] * self.metric.spatial_grad(D, theta)

    def _dtheta_per_dim(self, D, theta):
        s = self.metric.value(D, theta)
        return self.profile.fprime(s)[:, None] * self.metric.dtheta(D, theta)

    def batch(self, D, thetas):
        s = self.metric.batch(D, thetas)
        if s is None:
            # the metric declined the fast path -> evaluate the population one theta at a time,
            # preserving the exact per-theta summation order (e.g. the L1 metric under Exponential).
            return super().batch(D, thetas)
        return self.profile.f_batch(s)

    def __repr__(self):
        return type(self).__name__


class WeightedSquare(Metric):
    """Weighted squared-Euclidean metric ``s = sum_k theta_k * D_k**2`` (isotropic or ARD).

    Composed with :class:`Exp` this is the Gaussian kernel. It carries the batched fast path the
    Gaussian historically used: an isotropic broadcast, and the ARD case as a single GEMM.
    """

    def value(self, D, theta):
        return np.sum(np.square(D) * theta, axis=1)

    def spatial_grad(self, D, theta):
        return 2 * theta * D

    def dtheta(self, D, theta):
        # ds/d(theta_k) = D_k^2, one column per input dimension (the isotropic case is collapsed
        # to a single column by Kernel.theta_grad, exactly as the old Gaussian did).
        return np.square(D)

    def batch(self, D, thetas):
        thetas = _asmat(thetas)
        Dsq = np.square(D)
        # isotropic theta (one column) broadcasts over the d distance columns; ARD theta
        # (one per dim) contracts against them. Either way the result is (J, n_pairs).
        return thetas * np.sum(Dsq, axis=1) if thetas.shape[1] == 1 else thetas @ Dsq.T


class WeightedAbs(Metric):
    """Weighted L1 (absolute-distance) metric ``s = sum_k theta_k * |D_k|`` (isotropic or ARD).

    Composed with :class:`Exp` this is the Exponential kernel. It deliberately does **not**
    override ``batch`` (inherits ``None``), so a theta population evaluates through the per-theta
    stack -- preserving the exact summation order the Exponential kernel has always used, rather
    than introducing a GEMM whose different reduction order would perturb the likelihood in the
    low bits and disturb a gradient-free search trajectory.
    """

    def value(self, D, theta):
        return np.sum(np.abs(D) * theta, axis=1)

    def spatial_grad(self, D, theta):
        return theta * np.sign(D)

    def dtheta(self, D, theta):
        return np.abs(D)


class Gaussian(ComposedKernel):
    """Gaussian (squared-exponential) kernel: ``exp(-sum_k theta_k * D_k**2)``.

    The canonical composition ``Exp`` over :class:`WeightedSquare` -- kept as a named class because
    it is the framework's default covariance and is referenced by name throughout.
    """

    def __init__(self, ard=False):
        super().__init__(WeightedSquare(ard=ard), Exp())


class Cubic(ProductKernel):
    """Cubic kernel: compact-support smooth correlation (zero past the length scale)."""

    def _factor(self, D, theta):
        t = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - t**2 * (3 - 2 * t)
        dsdt = 6 * t * (t - 1)  # = -6t + 6t^2; 0 at the clamp t=1
        return ss, dsdt


class Exponential(ComposedKernel):
    """Exponential (absolute-distance) kernel: ``exp(-sum_k theta_k * |D_k|)``.

    The composition ``Exp`` over :class:`WeightedAbs` -- the same curve as :class:`Gaussian`, over
    an L1 metric instead of a squared one.
    """

    def __init__(self, ard=False):
        super().__init__(WeightedAbs(ard=ard), Exp())


class Linear(ProductKernel):
    """Linear correlation kernel: ``prod_k max(1 - theta_k * |D_k|, 0)`` (compact support).

    The matching regression trend is named ``LinearRegression`` (in ``pysurrogate.dace.regr``),
    so this bare ``Linear`` kernel and that trend can be imported together without a clash.
    """

    def _factor(self, D, theta):
        ss = np.maximum(1 - np.abs(D) * theta, 0)
        # d/dt of max(1 - t, 0) is -1 where 1 - t > 0, else 0 (flat in the clamped tail)
        dsdt = np.where(ss > 0, -1.0, 0.0)
        return ss, dsdt


class Spherical(ProductKernel):
    """Spherical kernel: compact-support correlation ``1 - 1.5 t + 0.5 t**3``."""

    def _factor(self, D, theta):
        t = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - t * (1.5 - 0.5 * np.power(t, 2))
        dsdt = 1.5 * (np.power(t, 2) - 1)  # 0 at the clamp t=1
        return ss, dsdt


class Spline(ProductKernel):
    """Cubic-spline kernel: piecewise-polynomial compact-support correlation."""

    def _factor(self, D, theta):
        # t = |D|*theta; the factor and its derivative M'(t) are piecewise (low / mid /
        # far). The far region (t >= 1) stays 0 in both, so the product rule self-zeroes.
        t = np.abs(D) * theta
        lo = t <= 0.2
        mid = (t > 0.2) & (t < 1.0)
        ss = np.zeros(D.shape)
        ss[lo] = 1 - t[lo] ** 2 * (15 - 30 * t[lo])
        ss[mid] = 1.25 * (1 - t[mid]) ** 3
        dsdt = np.zeros(D.shape)
        dsdt[lo] = (90 * t[lo] - 30) * t[lo]
        dsdt[mid] = -3.75 * (1 - t[mid]) ** 2
        return ss, dsdt


class GeneralizedExponential(Correlation):
    """Generalized exponential kernel: ``exp(-theta * |D|**power)``.

    Generalizes the fixed-exponent kernels by making the exponent a parameter:
    ``power = 2`` recovers ``Gaussian`` and ``power = 1`` recovers ``Exponential``, so
    ``1 <= power <= 2`` interpolates between them. Unlike the other kernels the
    exponent is tuned *with* the length-scale: ``theta`` is ``(length_scale, power)``
    (or ``(length_scales..., power)`` for ARD), so the search optimizes both together
    -- which is why ``power`` stays in ``theta`` here rather than on the object.
    """

    def n_theta(self, d):
        """Length-scale coordinates plus the shared exponent: ``(d if ard else 1) + 1`` for ``power``."""
        return super().n_theta(d) + 1

    def parameters(self, d):
        """The length-scale(s) followed by the shared exponent ``power``.

        ``theta`` is the (isotropic or ARD) length-scale block; ``power`` is one extra coordinate. It
        is declared ``log10``-encoded to match the historical Dace search (which searches every
        coordinate, exponent included, in log10 space), so promoting it to a first-class parameter
        leaves the search byte-identical.
        """
        n_ls = d if self.ard else 1
        return [
            Parameter("theta", size=n_ls, encoding=Log10(), fill=True),
            Parameter("power", size=1, bounds=(1.0, 2.0), encoding=Log10()),  # fixed shape coordinate
        ]

    def __call__(self, D, theta):
        # theta is (length_scale, power) [isotropic] or (length_scales..., power) [ARD];
        # D is the (n_pairs, d) difference matrix, so the ARD case is keyed on the input
        # dimensionality D.shape[1], NOT len(D) (which is the pair count).
        _theta, power = self._split(D, theta)
        return np.exp(np.sum(np.abs(D) ** power * -_theta, axis=1))

    def grad(self, D, theta):
        _theta, power = self._split(D, theta)
        return power * -_theta * np.sign(D) * np.abs(D) ** (power - 1) * self(D, theta)[:, None]

    def theta_grad(self, D, theta):
        # theta = (length_scale(s)..., power). One column per entry: the length-scale
        # partials first, then a final column for the shared exponent ``power``. This
        # kernel overrides theta_grad (rather than _dtheta_per_dim) because of that
        # extra power column and the special theta layout.
        _theta, power = self._split(D, theta)
        corr = self(D, theta)[:, None]
        ad = np.abs(D)
        adp = ad**power

        # length-scale partials: d corr / d theta_k = -|D_k|^power * corr
        d_ls = -adp * corr
        if np.size(_theta) == 1:  # isotropic: one shared length-scale -> a single column
            d_ls = d_ls.sum(axis=1, keepdims=True)

        # power partial: d corr / d power = -corr * sum_k theta_k |D_k|^power ln|D_k|
        # (the |D_k|=0 terms vanish: x^p ln x -> 0, but numerically 0 * -inf -> nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            plog = adp * np.log(ad)
        plog = np.where(ad > 0, plog, 0.0)
        d_pow = -np.sum(_theta * plog, axis=1, keepdims=True) * corr

        return np.concatenate([d_ls, d_pow], axis=1)

    @staticmethod
    def _split(D, theta):
        """Split theta into its length-scale(s) and the shared exponent ``power``."""
        d = D.shape[1]
        if len(theta) == 2:
            return theta[0], theta[1]
        if len(theta) == d + 1:
            return theta[:-1], theta[-1]
        raise Exception(f"For GeneralizedExponential theta is either length 2 or d+1 = {d + 1}")


class RationalQuadratic(Correlation):
    """Rational Quadratic correlation kernel: an (infinite) scale-mixture of Gaussians.

    The shape parameter ``alpha`` sets the tails -- small alpha gives heavier tails
    (more robust across length scales), and ``alpha -> inf`` recovers ``Gaussian``.
    It is a fixed construction parameter, deliberately kept out of ``theta`` (which
    the optimizer tunes by maximum likelihood): tuning a shape parameter on a small
    sample overfits, so it belongs to the kernel, not to the search vector.

    The default ``alpha=0.25`` gives noticeably heavy tails (robust across a range of
    length scales -- a safe general-purpose choice); raise it toward ``Gaussian`` for a
    tighter, single-scale fit. ``theta`` carries the length-scale(s) and may be a scalar
    (isotropic) or a per-dimension vector (ARD), like the other kernels.

    Args:
        alpha: Tail / scale-mixture parameter (> 0).
    """

    def __init__(self, alpha=0.25, ard=False):
        super().__init__(ard=ard)
        self.alpha = alpha

    def __call__(self, D, theta):
        # parameterized so alpha -> inf recovers Gaussian exactly (no 1/2 factor,
        # matching this library's gauss convention exp(-theta * D**2)).
        base = 1 + theta * np.square(D) / self.alpha
        return np.prod(base ** (-self.alpha), axis=1)

    def grad(self, D, theta):
        base = 1 + theta * np.square(D) / self.alpha
        r = np.prod(base ** (-self.alpha), axis=1)
        return -2 * theta * D / base * r[:, None]

    def _dtheta_per_dim(self, D, theta):
        # the k-th factor's log-derivative is -d_k^2 / base_k (the alpha's cancel),
        # so d corr / d theta_k = -d_k^2 / base_k * corr, per input dimension.
        base = 1 + theta * np.square(D) / self.alpha
        r = np.prod(base ** (-self.alpha), axis=1)[:, None]
        return -np.square(D) / base * r

    def __repr__(self):
        return f"RationalQuadratic(alpha={self.alpha})"


class Matern(ProductKernel):
    """Matérn kernel (product of per-dimension Matérns), smoothness ``nu`` in {0.5, 1.5, 2.5}.

    ``nu`` controls smoothness: 0.5 is the rough exponential (once-continuous), 2.5 is
    twice-differentiable (a common default for physical responses, where the Gaussian's
    infinite smoothness is unrealistic), and ``nu -> inf`` recovers ``Gaussian``. Like
    RQ's ``alpha``, ``nu`` is a fixed construction parameter kept out of ``theta`` --
    tuning smoothness on a small sample overfits. ``theta`` carries the inverse
    length-scale(s), scalar (isotropic) or per-dimension (ARD); the kernel is the product
    of per-dimension Matérns ``M(theta_k * |D_k|)`` (the standard half-integer closed
    forms, so theta=1/length recovers textbook Matérn-nu).

    Args:
        nu: Smoothness; one of 0.5, 1.5, 2.5 (the cases with a closed form).
    """

    def __init__(self, nu=2.5, ard=False):
        super().__init__(ard=ard)
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("Matern supports nu in {0.5, 1.5, 2.5} (the closed-form cases).")
        self.nu = nu

    def _factor(self, D, theta):
        # per-dimension scaled distance t = theta_k |D_k|, the Matérn factor M(t) and its
        # derivative M'(t) -- both (n_pairs, d). ProductKernel turns these into the kernel
        # and both gradients via the shared product rule.
        t = np.abs(D) * theta
        if self.nu == 0.5:
            e = np.exp(-t)
            return e, -e
        if self.nu == 1.5:
            c = np.sqrt(3.0)
            e = np.exp(-c * t)
            return (1 + c * t) * e, -3.0 * t * e
        # nu == 2.5
        c = np.sqrt(5.0)
        e = np.exp(-c * t)
        return (1 + c * t + (5.0 / 3.0) * t**2) * e, -(5.0 / 3.0) * t * (1 + c * t) * e

    def __repr__(self):
        return f"Matern(nu={self.nu})"


# -------------------------------
# Conditionally positive-definite radial bases
# -------------------------------
# These are *not* valid GP covariances (so DACE does not use them), but they are excellent
# interpolation bases when paired with a polynomial tail (the RBF use). Written in the same
# ``k(D, theta)`` style as the covariance kernels -- value on the coordinate differences ``D``
# and a spatial ``grad`` -- so they live in the one zoo. They are isotropic (``ard=False`` is
# the natural setting): they depend on the radius ``r = ||D||`` only, so they provide no
# per-theta gradient (``has_theta_grad`` is False) and are not theta-searched.


class RadialKernel(Kernel):
    """A radial basis ``phi(r2)`` of the squared radius ``r2 = sum_k D_k**2``.

    Subclasses implement ``_phi(r2, theta)`` returning ``(phi, dphi_dr2)``: the basis value
    and its derivative w.r.t. ``r2``. The spatial gradient is then
    ``d phi / d D_k = dphi_dr2 * 2 D_k`` (one chain rule, written once).
    """

    def _phi(self, r2, theta):
        raise NotImplementedError

    def parameters(self, d):
        """No searchable parameters: the radial bases depend on the radius only and are not theta-tuned."""
        return []

    def __call__(self, D, theta):
        r2 = np.sum(np.square(D), axis=1)
        return self._phi(r2, theta)[0]

    def grad(self, D, theta):
        r2 = np.sum(np.square(D), axis=1)
        _, dphi = self._phi(r2, theta)
        return 2.0 * dphi[:, None] * D


class ThinPlateSpline(RadialKernel):
    """Thin-plate spline ``r**2 * log(r)`` (the standard 2D polyharmonic basis).

    ``theta`` is unused (the spline has no shape parameter); it is accepted for a uniform
    kernel signature. Pair with a linear (or higher) polynomial tail for a well-posed fit.
    """

    def _phi(self, r2, theta):
        # phi = r^2 log r = 0.5 r2 log r2; clamp r2 away from 0 so log is finite (phi -> 0).
        r2 = np.maximum(r2, np.finfo(float).eps)
        phi = 0.5 * r2 * np.log(r2)
        dphi = 0.5 * (np.log(r2) + 1.0)  # d/dr2 [0.5 r2 log r2]
        return phi, dphi


class Multiquadric(RadialKernel):
    """Multiquadric ``sqrt(r**2 + c**2)`` with shape parameter ``c = theta`` (default 1)."""

    def __call__(self, D, theta=1.0):
        return super().__call__(D, theta)

    def grad(self, D, theta=1.0):
        return super().grad(D, theta)

    def _phi(self, r2, theta):
        c2 = np.square(np.asarray(theta, dtype=float))
        s = np.sqrt(r2 + c2)
        return s, 0.5 / s


class LinearRadial(RadialKernel):
    """Linear radial basis ``phi(r) = r`` (the 1D polyharmonic spline). ``theta`` is unused.

    Named ``LinearRadial`` to distinguish it from the compact-support product kernel :class:`Linear`:
    this depends only on the radius ``r = ||D||`` (a growing radial power), whereas :class:`Linear` is
    a per-dimension ``max(1 - theta|D|, 0)`` factor. Conditionally positive-definite -- pair with at
    least a constant polynomial tail for a well-posed interpolant.
    """

    def _phi(self, r2, theta):
        r2 = np.maximum(r2, np.finfo(float).eps)  # keep dphi/dr2 = 0.5/r finite at a center
        r = np.sqrt(r2)
        return r, 0.5 / r


class CubicRadial(RadialKernel):
    """Cubic radial basis ``phi(r) = r**3`` (the 3D polyharmonic spline). ``theta`` is unused.

    Named ``CubicRadial`` to distinguish it from the compact-support product kernel :class:`Cubic`;
    this is the growing radial power ``r**3``, not a compact-support factor. Conditionally
    positive-definite -- pair with a linear polynomial tail for a well-posed interpolant.
    """

    def _phi(self, r2, theta):
        r = np.sqrt(r2)
        return r2 * r, 1.5 * r  # phi = r^3 = r2**1.5; dphi/dr2 = 1.5 * r


# -------------------------------
# Reduction metrics (rotated / low-rank distance)
# -------------------------------
# Two ways to fold the d coordinate differences into a lower-rank squared metric before the
# length-scales weight them -- they differ only in *where* the linear map is applied relative to the
# square, and that difference is exactly rotation (cross terms) vs. no rotation. Both reduce ``D`` by
# one fixed matmul per fit, so the reduced array is cached once (see ``ReducedMetric``).


class ReducedMetric(Metric):
    """A squared metric that first reduces ``D`` by a fixed ``(d, h)`` linear map, cached per fit.

    Subclasses supply ``_transform(D)`` -- the once-per-fit reduction (a projection or a squared-
    distance mix) whose result the value/gradient/batch read. A theta search reuses one ``D`` array
    across all its evaluations, so ``_reduce`` caches on object identity (``is``): the search hits the
    cache and a fresh ``D`` (e.g. at predict) correctly misses. Always ARD (``n_theta = h``).

    Args:
        matrix: The ``(d, h)`` reduction map (a projection ``A`` or squared-weight matrix ``W2``).
    """

    def __init__(self, matrix):
        super().__init__(ard=True)
        self.matrix = np.asarray(matrix, dtype=float)  # (d, h)
        self._cache_D = None  # identity-keyed 1-slot cache of the reduced differences (per fit)
        self._cache_R = None

    def n_theta(self, d):
        return self.matrix.shape[1]

    def _transform(self, D):
        """The fixed reduction of ``D`` to cache (a projection or a squared-distance mix)."""
        raise NotImplementedError

    def _reduce(self, D):
        if D is self._cache_D:
            return self._cache_R
        R = self._transform(D)
        self._cache_D, self._cache_R = D, R
        return R


class ProjectedSquare(ReducedMetric):
    """Squared metric under a fixed linear projection: ``s = sum_k theta_k (D @ A)_k**2``.

    Projects the differences by ``A`` (shape ``(d, h)``, ``h <= d``) *then* squares, so the cross
    terms that encode a rotated (non-axis-aligned) metric survive. ``A = I`` recovers the plain ARD
    squared metric. Composed with :class:`Exp` this is the Mahalanobis kernel.
    """

    @property
    def A(self):
        """The projection matrix ``(d, h)`` (its columns are the metric's principal directions)."""
        return self.matrix

    def _transform(self, D):
        return np.asarray(D, dtype=float) @ self.matrix  # projected differences P = D @ A

    def value(self, D, theta):
        return np.square(self._reduce(D)) @ _asvec(theta)

    def spatial_grad(self, D, theta):
        # ds/dD_l = 2 sum_k theta_k P_k A_lk = 2 ((theta * P) @ A^T)_l. With A = I this is 2 theta D.
        P = self._reduce(D)
        return 2.0 * ((_asvec(theta) * P) @ self.matrix.T)

    def dtheta(self, D, theta):
        return np.square(self._reduce(D))

    def batch(self, D, thetas):
        return _asmat(thetas) @ np.square(self._reduce(D)).T


class SquareThenMix(ReducedMetric):
    """Reduced squared metric ``s = sum_k theta_k M_k`` with ``M = D**2 @ W2`` (square, then mix).

    Squares the differences *then* mixes them by ``W2`` (shape ``(d, h)``), so there are no cross
    terms -- this is the exact reduction behind KPLS: ``exp(-sum_l eta_l D_l**2)`` with
    ``eta = W2 theta`` equals ``exp(-sum_k theta_k M_k)``, evaluated in the ``h``-dim reduced space.
    """

    @property
    def w2(self):
        """The squared-weight matrix ``(d, h)`` mixing the squared distances into ``h`` coordinates."""
        return self.matrix

    def _transform(self, D):
        return np.square(D) @ self.matrix  # reduced squared distances M = D**2 @ W2

    def value(self, D, theta):
        return self._reduce(D) @ _asvec(theta)

    def spatial_grad(self, D, theta):
        # ds/dD_l = 2 D_l sum_k theta_k W2_lk = 2 D_l (W2 @ theta)_l. Not used by KPLS (which takes the
        # base kernel's full-space gradient), but provided so this composes into a working kernel.
        return 2.0 * (self.matrix @ _asvec(theta)) * D

    def dtheta(self, D, theta):
        return self._reduce(D)

    def batch(self, D, thetas):
        return _asmat(thetas) @ self._reduce(D).T


class Mahalanobis(ComposedKernel):
    """Squared-exponential kernel under a low-rank Mahalanobis metric ``M = A diag(theta) Aᵀ``.

    ARD Kriging stretches the *coordinate axes* -- its metric ``diag(theta)`` is diagonal, so it
    can only model anisotropy aligned with the input dimensions. A Mahalanobis metric adds
    **rotation**: the correlation is a Gaussian in a linearly projected difference space,

        k(x, x') = exp( -(x-x')ᵀ M (x-x') ),   M = A diag(theta) Aᵀ
                 = exp( -sum_k theta_k (a_kᵀ (x-x'))² ),   P = D @ A  (project, then ARD),

    where ``A`` (shape ``(d, h)``, ``h <= d``) is a **fixed** projection supplied at construction --
    its columns are the metric's principal directions. ``A = I`` recovers ARD-Gaussian; a rank-``h``
    ``A`` with off-axis columns is a genuinely rotated (cross-term) metric that a diagonal kernel
    cannot represent. Only the ``h`` length-scales ``theta`` are optimized, so from the engine's
    side this is an ordinary ARD kernel with ``n_theta = h``; the rotation lives entirely in ``A``.

    Mechanically it is :class:`Exp` composed with :class:`ProjectedSquare` -- the ``project, then
    square`` reduction is what distinguishes it from :class:`KPLSKernel`'s ``square, then mix``.
    Estimating ``A`` from data (PLS, PCA, active-subspace eigenvectors, ...) is the caller's job.

    Args:
        A: The projection matrix ``(d, h)`` (``h <= d``); its columns are the metric directions.
    """

    def __init__(self, A):
        super().__init__(ProjectedSquare(A), Exp())

    def __repr__(self):
        A = self.metric.A
        return f"Mahalanobis(d={A.shape[0]}, h={A.shape[1]})"


def _reducible_square_metric(base, w2):
    """A :class:`SquareThenMix` reduction if ``base`` is squared-exponential, else ``None``.

    KPLS's reduced-space evaluation ``exp(-sum_k theta_k M_k)`` is exact only for a
    squared-exponential base -- :class:`Exp` over a :class:`WeightedSquare` metric. Recognizing that
    by capability (the metric and profile types) rather than by the concrete class keeps the
    reduction available to any such composition and correctly excludes L1 / product bases.
    """
    if isinstance(base, ComposedKernel) and isinstance(base.metric, WeightedSquare) and isinstance(base.profile, Exp):
        return SquareThenMix(w2)
    return None


class KPLSKernel(Correlation):
    """A base kernel whose per-dimension length-scales are a rank-``h`` PLS subspace (KPLS).

    Ordinary ARD Kriging tunes one length-scale per input dimension -- ``d`` hyperparameters,
    which makes the ``theta`` search intractable in high dimensions. KPLS (Bouhlel et al.) keeps
    a *product-exponential* base kernel but constrains its ``d`` length-scales to a low-rank
    linear image of just ``h`` free parameters, with the mixing fixed up front by Partial Least
    Squares. For the ``Gaussian`` (squared-exponential) kernel this is exact::

        k(x, x') = prod_k prod_l exp(-theta_k w_lk^2 (x_l - x'_l)^2)
                 = exp(-sum_l eta_l (x_l - x'_l)^2),   eta = W2 @ theta

    so a KPLS-Gaussian model *is* an ARD-Gaussian model whose length-scale vector ``eta`` lives
    on the ``h``-dimensional column space of ``W2`` (the squared PLS weights, shape ``(d, h)``).
    Only ``h`` coordinates are optimized; the DACE likelihood, the theta search, ``predict`` and
    ``calibrate`` are unchanged because from their side this is just a kernel with ``n_theta = h``.

    The PLS weights ``W2`` are data-dependent, so the :class:`~pysurrogate.models.kpls.KPLS`
    backend computes them (in the model's standardized space) and constructs this wrapper; the
    kernel itself is a pure, fixed reparameterization. The base kernel must be product-exponential
    for the factorization to be exact -- ``Gaussian`` (the default) or ``Exponential``.

    Args:
        base: The product-exponential kernel to reparameterize (e.g. ``Gaussian()``).
        w2: The squared PLS weight matrix ``(d, h)`` mapping the ``h`` search parameters to the
            ``d`` effective per-dimension length-scales via ``eta = w2 @ theta``.
    """

    def __init__(self, base, w2):
        super().__init__(ard=True)
        self.base = base
        self.w2 = np.asarray(w2, dtype=float)  # (d, h)
        # the reduced correlation exp(-sum_k theta_k M_k), M = D**2 @ W2, is exact only for a
        # squared-exponential base -- Exp over a WeightedSquare metric, the one case where the
        # per-dimension factors combine into a single sum of squared distances. Recognized by that
        # *capability* (metric + profile), not by the concrete class name, so any squared-exponential
        # composition reduces and any other base (e.g. an L1 Exponential) correctly falls back to the
        # general expand-to-d path. The reused SquareThenMix carries the identity-keyed M cache.
        self._reduced = _reducible_square_metric(base, self.w2)
        self._reducible = self._reduced is not None

    def n_theta(self, d):
        return self.w2.shape[1]

    def _eta(self, theta):
        # map the h search length-scales to the d effective ARD length-scales: eta = W2 @ theta.
        return self.w2 @ _asvec(theta)

    def __call__(self, D, theta):
        if not self._reducible:
            return self.base(D, self._eta(theta))
        # exp(-sum_l eta_l D_l^2) = exp(-sum_k theta_k M_k) -- same value, in the reduced h-space.
        return self.base.profile.f(self._reduced.value(D, theta))

    def grad(self, D, theta):
        # spatial gradient w.r.t. the design point: eta are just the length-scales, so this is the
        # base kernel's gradient at the expanded eta -- a predict-only path, left in full d-space.
        return self.base.grad(D, self._eta(theta))

    def theta_grad(self, D, theta):
        if not self._reducible:
            # chain rule: d corr / d theta_k = sum_l (d corr / d eta_l) W2_lk = base_tg @ W2.
            return self.base.theta_grad(D, self._eta(theta)) @ self.w2
        # reduced: d corr / d theta_k = f'(s) * M_k, s = sum_k theta_k M_k -- no d-space detour.
        s = self._reduced.value(D, theta)
        return self.base.profile.fprime(s)[:, None] * self._reduced.dtheta(D, theta)

    def batch(self, D, thetas):
        thetas = _asmat(thetas)
        if not self._reducible:
            # population map: each row theta (h,) -> eta (d,) is thetas @ W2.T, then base batches.
            return self.base.batch(D, thetas @ self.w2.T)
        # the same (J, p) @ (p, n_pairs) GEMM the Gaussian batch uses, on the reduced M (n_pairs, h).
        return self.base.profile.f_batch(self._reduced.batch(D, thetas))

    def __repr__(self):
        return f"KPLS({self.base!r}, h={self.w2.shape[1]})"

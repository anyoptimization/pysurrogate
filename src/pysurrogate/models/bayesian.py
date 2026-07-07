"""FullyBayesianGP: marginalize the GP hyperparameters by HMC instead of point-estimating them."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.dace import ConstantRegression, Dace, Gaussian, GaussianPrior
from pysurrogate.dace.hmc import hmc_sample
from pysurrogate.dace.problem import DaceProblem

_CLIP_THETA = (1e-6, 1e6)  # length-scale clamp for the frozen posterior-sample fits (numeric safety)
_FLOOR_NOISE = 1e-8


class _GPPosterior:
    """The GP log-posterior over ``x = [log10 theta, log10 noise]`` as an HMC potential.

    Wraps a :class:`~pysurrogate.dace.problem.DaceProblem` (built with **no** prior) purely to reuse
    its batched, never-raising ``obj`` and analytic ``d(obj)/d(log10-coord)`` -- the exact profile
    likelihood the Kriging search already trusts. The DACE objective is ``obj = sigma2 * detR``, and
    the Gaussian log-likelihood is ``-(n/2) * log(obj)`` up to a constant, so the potential
    (negative log-posterior) is ``U = (n/2) * log(obj) + prior_penalty`` and its gradient follows by
    the chain rule through the ``log``. The prior is added directly in ``log10`` space on the
    length-scale coordinates only (never the nugget), matching :class:`~pysurrogate.dace.GaussianPrior`.
    """

    def __init__(self, X, Y, regr, kernel, prior, theta_bounds, noise_bounds):
        # theta_prior=None: we want the *pure* likelihood obj/grad and apply the prior ourselves in
        # the (correct) log-posterior form -- DaceProblem would otherwise add it to obj linearly.
        self.problem = DaceProblem(X, Y, regr, kernel, theta_bounds, noise_bounds=noise_bounds, theta_prior=None)
        self.n = X.shape[0]
        self.p = self.problem.p  # number of length-scale coordinates (the nugget is coordinate p)
        self.prior = prior

    def potential(self, x):
        """Negative log-posterior ``U(x)`` at the ``log10`` hyperparameter vector ``x`` (``+inf`` if infeasible)."""
        ev = self.problem(x[None])
        obj = float(ev.f[0])
        if not ev.feasible[0] or obj <= 0.0 or not np.isfinite(obj):
            return np.inf
        pen = 0.0 if self.prior is None else float(self.prior.penalty(x[None, : self.p])[0])
        return 0.5 * self.n * np.log(obj) + pen

    def grad(self, x):
        """Gradient ``dU/dx`` (zeros on an infeasible point so the leapfrog just coasts through)."""
        ev = self.problem(x[None])
        obj = float(ev.f[0])
        if not ev.feasible[0] or obj <= 0.0 or not np.isfinite(obj):
            return np.zeros_like(x)
        out = (0.5 * self.n / obj) * ev.grad[0]  # d/dx [ (n/2) log(obj) ] = (n/2)/obj * d(obj)/dx
        if self.prior is not None:
            out[: self.p] = out[: self.p] + self.prior.grad(x[None, : self.p])[0]
        return np.where(np.isfinite(out), out, 0.0)


class FullyBayesianGP(Model):
    """Fully-Bayesian GP: marginalize the length-scales and nugget by HMC, then model-average.

    A Kriging fit point-estimates the hyperparameters by maximum likelihood -- one length-scale
    vector, one nugget -- and predicts with the variance *conditional on that single estimate*. When
    the data is scarce or noisy the likelihood is flat or multi-modal, so that point estimate is
    arbitrary and the conditional variance is **overconfident** (the model does not know it guessed).
    This backend instead draws posterior samples of ``(theta, noise)`` by Hamiltonian Monte Carlo and
    predicts by **Bayesian model averaging** over them:

        ``mean = <mu_s>``   and   ``var = <sigma2_s> + Var_s(mu_s)``

    -- the average predictive mean, and the average predictive variance *plus* the between-sample
    variance of the means (the uncertainty in the hyperparameters themselves, which the MLE fit
    simply drops). The payoff is **calibrated** uncertainty and robustness to noise: on noisy data
    the model-average both predicts better than MLE-Kriging and reports honest error bars (far lower
    negative-log-predictive-density), which is exactly what a Bayesian-optimization acquisition needs.
    It is *not* a free lunch on clean, data-rich problems -- there MLE-ARD already nails the fit and
    HMC only adds cost -- so reach for it when data is scarce/noisy and calibration matters.

    The heavy lifting is reused, not reinvented: the HMC potential is the same DACE profile likelihood
    the Kriging search descends (via :class:`~pysurrogate.dace.problem.DaceProblem`), each posterior
    sample becomes a frozen :class:`~pysurrogate.dace.Dace` fit (``optimizer=None``), and prediction
    averages their :class:`~pysurrogate.core.prediction.Prediction` outputs. Single-output only.

    Args:
        regr: Regression trend for every sample's GP (default :class:`ConstantRegression`).
        corr: Correlation kernel; ARD is the point of marginalizing per-dimension length-scales
            (default ``Gaussian(ard=True)``).
        prior: A :class:`~pysurrogate.dace.Prior` on the ``log10`` length-scales that makes the
            posterior proper (HMC needs a normalizable target). Default ``GaussianPrior(0.0, 0.1)`` --
            a weakly-informative ridge toward unit length-scale. Pass a stronger/sparser prior to bias
            the marginalization (e.g. toward relevance selection). The nugget is never penalized.
        n_samples: Posterior samples to draw (after warmup).
        n_warmup: HMC warmup iterations for step-size adaptation (discarded).
        n_leapfrog: Leapfrog steps per HMC proposal.
        thin: Keep every ``thin``-th sample for the model-average (decorrelate; fewer frozen fits).
        init_noise: Starting nugget for the chain (its ``log10`` seeds the noise coordinate).
        random_state: Seed for HMC (momenta + accept/reject); the whole fit is deterministic given it.
    """

    def __init__(
        self,
        regr=None,
        corr=None,
        prior=None,
        n_samples=100,
        n_warmup=400,
        n_leapfrog=15,
        thin=2,
        init_noise=1e-3,
        random_state=0,
        **kwargs,
    ):
        super().__init__(eliminate_duplicates=True, **kwargs)
        self.regr = regr if regr is not None else ConstantRegression()
        self.corr = corr if corr is not None else Gaussian(ard=True)
        self.prior = prior if prior is not None else GaussianPrior(0.0, 0.1)
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.n_leapfrog = n_leapfrog
        self.thin = thin
        self.init_noise = init_noise
        self.random_state = random_state
        # populated by _fit
        self.samples_ = None
        self.accept_rate_ = None
        self.gps_ = None

    def _fit(self, X, y, optimize=True, **kwargs):
        if y.shape[1] != 1:
            raise ValueError("FullyBayesianGP is single-output; got y with shape " + str(y.shape))
        d = X.shape[1]

        # standardize exactly as Dace does internally (ddof=1) so the posterior's likelihood and the
        # per-sample frozen fits share ONE length-scale space -- the sampled theta means the same
        # thing to the frozen Dace (which re-standardizes the raw X,y to the identical nX,nY).
        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        mY, sY = np.mean(y, axis=0), np.std(y, axis=0, ddof=1)
        sX = np.where(sX == 0.0, 1.0, sX)
        sY = np.where(sY == 0.0, 1.0, sY)
        nX, nY = (X - mX) / sX, (y - mY) / sY

        # bounds only shape the coordinate layout (ARD count from the theta_bounds length, plus the
        # nugget coordinate); HMC itself is unbounded and the prior keeps the target proper.
        ard = getattr(self.corr, "ard", False)
        theta_bounds = (np.full(d, _CLIP_THETA[0]), np.full(d, _CLIP_THETA[1])) if ard else _CLIP_THETA
        noise_bounds = (_FLOOR_NOISE, 1.0)
        post = _GPPosterior(nX, nY, self.regr, self.corr, self.prior, theta_bounds, noise_bounds)

        x0 = np.append(np.zeros(post.p), np.log10(self.init_noise))  # log10 theta=0 (unit), log10 noise
        self.samples_, self.accept_rate_ = hmc_sample(
            post.potential,
            post.grad,
            x0,
            self.n_samples,
            n_warmup=self.n_warmup,
            n_leapfrog=self.n_leapfrog,
            seed=self.random_state,
        )

        # each (thinned) posterior sample -> a frozen Dace fit on the RAW X,y (Dace standardizes
        # internally to the same nX,nY, so the sampled theta applies unchanged). Skip any sample whose
        # correlation matrix is singular; the wide-but-finite theta clamp keeps the solve well-posed.
        self.gps_ = []
        for s in self.samples_[:: self.thin]:
            theta = np.clip(10.0 ** s[: post.p], *_CLIP_THETA)
            noise = max(float(10.0 ** s[post.p]), _FLOOR_NOISE)
            try:
                gp = Dace(regr=self.regr, corr=self.corr, theta=theta, theta_bounds=None, noise=noise, optimizer=None)
                gp.fit(X, y)
                self.gps_.append(gp)
            except Exception:
                continue
        if not self.gps_:
            raise RuntimeError("FullyBayesianGP: every posterior-sample fit was singular; check the data scaling.")

    def _predict(self, X, var=False, grad=False):
        preds = [gp.predict(X, var=var, grad=grad) for gp in self.gps_]
        y_s = np.stack([p.y for p in preds])  # (S, m, 1)
        y = y_s.mean(axis=0)

        v = None
        if var:
            within = np.stack([p.var for p in preds]).mean(axis=0)  # <sigma2_s>: average conditional variance
            between = y_s.var(axis=0)  # Var_s(mu_s): spread of the sample means (the hyperparameter uncertainty)
            v = within + between  # the Bayesian-model-averaging total predictive variance

        g = None
        if grad:
            g = np.stack([p.grad for p in preds]).mean(axis=0)  # averaged mean-gradient

        return Prediction(y=y, var=v, grad=g)

"""Hamiltonian Monte Carlo with dual-averaging step-size adaptation."""

import numpy as np


def _leapfrog(x, p, eps, n_leapfrog, grad_potential):
    """One leapfrog trajectory of ``n_leapfrog`` steps at step size ``eps``; returns ``(x, p)``."""
    p = p - 0.5 * eps * grad_potential(x)
    for _ in range(n_leapfrog - 1):
        x = x + eps * p
        p = p - eps * grad_potential(x)
    x = x + eps * p
    p = p - 0.5 * eps * grad_potential(x)
    return x, p


def hmc_sample(potential, grad_potential, x0, n_samples, n_warmup=600, n_leapfrog=20, target_accept=0.8, seed=0):
    """Sample from ``p(x) ∝ exp(-potential(x))`` by Hamiltonian Monte Carlo.

    Leapfrog dynamics with a Metropolis accept/reject on the Hamiltonian, and a warmup phase that
    adapts the step size by dual averaging (Hoffman & Gelman) toward ``target_accept``. The mass
    matrix is the identity -- fine for the ``log10`` hyperparameter space, which is roughly unit-scaled.
    An infinite potential (an infeasible point) is simply rejected, so the chain steps around
    non-positive-definite regions rather than crashing.

    Args:
        potential: ``U(x)`` -- the negative log-density (up to a constant), ``+inf`` where infeasible.
        grad_potential: ``dU/dx`` at ``x`` (shape of ``x``); used by the leapfrog integrator.
        x0: Initial point, shape ``(dim,)``.
        n_samples: Number of post-warmup samples to keep.
        n_warmup: Warmup iterations used to adapt the step size (discarded).
        n_leapfrog: Leapfrog steps per proposal (the trajectory length is ``n_leapfrog * eps``).
        target_accept: Target Metropolis acceptance rate the step size is tuned toward.
        seed: Seed for the momentum draws and accept/reject (deterministic).

    Returns:
        ``(samples, accept_rate)`` -- ``samples`` shape ``(n_samples, dim)`` and the post-warmup mean
        acceptance probability.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x0, dtype=float).copy()
    dim = x.size

    # dual-averaging state (Nesterov), targeting `target_accept`
    eps = 0.05
    mu = np.log(10.0 * eps)
    log_eps_bar = 0.0
    h_bar = 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75

    samples = np.empty((n_samples, dim))
    acc_sum = 0.0
    kept = 0
    lo_L = max(1, int(0.8 * n_leapfrog))
    hi_L = int(1.2 * n_leapfrog) + 1
    for it in range(n_warmup + n_samples):
        # jitter the trajectory length each proposal: a *fixed* leapfrog count can land on a periodic
        # orbit of the Hamiltonian and systematically under-explore the widest direction (resonance),
        # so vary it in +/-20% -- standard practice that decorrelates the chain at no extra cost.
        n_l = rng.randint(lo_L, hi_L)
        p0 = rng.standard_normal(dim)
        x_new, p_new = _leapfrog(x, p0.copy(), eps, n_l, grad_potential)
        h0 = potential(x) + 0.5 * (p0 @ p0)
        h1 = potential(x_new) + 0.5 * (p_new @ p_new)
        a = min(1.0, float(np.exp(h0 - h1))) if np.isfinite(h1) else 0.0
        if rng.random() < a:
            x = x_new
        if it < n_warmup:
            m = it + 1
            h_bar = (1.0 - 1.0 / (m + t0)) * h_bar + (1.0 / (m + t0)) * (target_accept - a)
            log_eps = mu - np.sqrt(m) / gamma * h_bar
            eta = m ** (-kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            eps = np.exp(log_eps)
        else:
            eps = np.exp(log_eps_bar)  # freeze the adapted step size for sampling
            samples[kept] = x
            kept += 1
            acc_sum += a
    return samples, acc_sum / max(1, n_samples)

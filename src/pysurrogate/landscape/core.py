"""The Landscape facade: run every criterion family on one point cloud and merge into one view."""

import importlib
import warnings
from typing import Any

import numpy as np

from ._context import Context

# The criterion families, in a stable order. Each module exposes ``compute(ctx) -> dict``.
FAMILIES: tuple[str, ...] = (
    "distribution",
    "meta_model",
    "curvature",
    "rotation",
    "active_subspace",
    "separability",
    "variogram",
    "multimodality",
    "nearest_better",
    "dispersion",
    "information_content",
    "convexity",
    "topology",
    "spectral",
    "gradient_field",
    "network",
)


class Landscape:
    """Exploratory landscape analysis for a labelled point cloud ``(X, y)``.

    Builds one shared :class:`Context`, runs every criterion family's ``compute`` on it, and
    merges the results into a single flat feature dict whose keys are namespaced
    ``"<family>.<feature>"``. The families are independent yet share the cached primitives on the
    context, so nothing expensive is recomputed.

    Args:
        X: Inputs, shape ``(n, d)``.
        y: Outputs, shape ``(n,)`` (minimization objective; lower is better).
        seed: Seed for any randomized feature.
        strict: When ``True`` a failing family re-raises instead of being skipped with a warning.

    Attributes:
        ctx: The shared :class:`Context` all families read from.
    """

    def __init__(self, X: Any, y: Any, seed: int = 0, strict: bool = False) -> None:
        self.ctx = Context(X, y, seed=seed)
        self._groups: dict[str, dict[str, float]] = {}
        for name in FAMILIES:
            module = importlib.import_module(f"{__package__}.{name}")
            try:
                group = {k: float(v) for k, v in module.compute(self.ctx).items()}
            except Exception as e:
                # A family must never take the whole analysis down (unless asked to via
                # ``strict``); its features are simply absent and read as nan via ``get``.
                if strict:
                    raise
                warnings.warn(f"landscape family {name} failed: {e!r}", stacklevel=2)
                group = {}
            self._groups[name] = group

    def features(self) -> dict[str, float]:
        """The full flat feature dict keyed ``"<family>.<feature>"``.

        Returns:
            One dict mapping each namespaced feature name to its float value (possibly ``nan``).
        """
        flat: dict[str, float] = {}
        for name, group in self._groups.items():
            for key, value in group.items():
                flat[f"{name}.{key}"] = value
        return flat

    def groups(self) -> dict[str, dict[str, float]]:
        """The features nested by family.

        Returns:
            A dict ``{family: {feature: value}}`` (a shallow copy safe to mutate).
        """
        return {name: dict(group) for name, group in self._groups.items()}

    def get(self, key: str, default: float = float("nan")) -> float:
        """Look up one namespaced feature.

        Args:
            key: A ``"<family>.<feature>"`` name.
            default: Value returned when the feature is absent.

        Returns:
            The feature's value, or ``default`` when it was not produced.
        """
        family, _, feature = key.partition(".")
        return self._groups.get(family, {}).get(feature, default)

    def report(self) -> str:
        """A readable multi-line interpretation of the headline structural criteria.

        Reads a handful of the most diagnostic features and renders plain-language verdicts on
        the questions that matter for surrogate choice: dimensionality, conditioning, rotation,
        smoothness/noise, modality, separability, and global trend.

        Returns:
            A multi-line string; each line is one structural verdict with the evidence behind it.
        """
        n, d = self.ctx.n, self.ctx.d
        g = self.get
        lines = [f"Landscape report  (n={n}, d={d})", "-" * 48]

        # -- global trend / linearity --
        lin_r2 = g("meta_model.lin_r2")
        is_linear = g("meta_model.is_linear")
        if is_linear >= 0.5 or lin_r2 >= 0.95:
            lines.append(f"Trend      : strongly LINEAR/planar (lin_r2={lin_r2:.2f})")
        elif lin_r2 >= 0.5:
            lines.append(f"Trend      : partial linear trend (lin_r2={lin_r2:.2f})")
        else:
            lines.append(f"Trend      : nonlinear (lin_r2={lin_r2:.2f})")

        # -- curvature / conditioning --
        cond = g("curvature.condition_number")
        curv_lin = g("curvature.curv_linear_ratio")
        if np.isnan(cond):
            lines.append("Curvature  : flat/linear (no reliable Hessian)")
        elif cond > 30:
            lines.append(f"Curvature  : ILL-CONDITIONED bowl (cond={cond:.1f}, stretched valley)")
        elif curv_lin > 0.3:
            lines.append(f"Curvature  : curved bowl (cond={cond:.1f}, curv/lin={curv_lin:.2f})")
        else:
            lines.append(f"Curvature  : gentle/near-linear (cond={cond:.1f})")

        # -- effective dimensionality --
        pr = g("active_subspace.participation_ratio")
        top = g("active_subspace.top_eig_frac")
        if not np.isnan(pr):
            if pr <= max(1.6, 0.4 * d):
                lines.append(
                    f"Eff. dim   : LOW effective dimension (~{pr:.1f} of {d}, "
                    f"top dir carries {top:.0%}) -- ridge/active-subspace"
                )
            elif pr >= 0.75 * d:
                lines.append(f"Eff. dim   : fully active / isotropic (~{pr:.1f} of {d})")
            else:
                lines.append(f"Eff. dim   : partially reduced (~{pr:.1f} of {d})")

        # -- rotation --
        hess_rot = g("rotation.hess_rot")
        grad_rot = g("rotation.grad_rot")
        offaxis = g("rotation.hess_offaxis")
        aniso = g("curvature.curv_anisotropy")
        rot_score = np.nanmax([hess_rot, grad_rot]) if not (np.isnan(hess_rot) and np.isnan(grad_rot)) else float("nan")
        if np.isnan(rot_score) or aniso < 0.1:
            lines.append("Rotation   : isotropic or axis-undefined (no rotation to detect)")
        elif rot_score > 0.25 or offaxis > 0.2:
            lines.append(f"Rotation   : ROTATED off the coordinate axes (rot={rot_score:.2f}, offaxis={offaxis:.2f})")
        else:
            lines.append(f"Rotation   : axis-aligned (rot={rot_score:.2f})")

        # -- smoothness / noise --
        smooth = g("variogram.smoothness_exp")
        nugget = g("variogram.nugget_ratio")
        h_max = g("information_content.h_max")
        if nugget > 0.08:
            lines.append(f"Noise      : NOISY / nugget present (nugget_ratio={nugget:.2f})")
        else:
            lines.append(f"Noise      : clean deterministic (nugget_ratio={nugget:.2f})")
        if not np.isnan(smooth):
            if smooth >= 1.7:
                lines.append(f"Smoothness : very smooth (variogram exponent={smooth:.2f})")
            elif smooth >= 0.8:
                lines.append(f"Smoothness : moderately smooth (exponent={smooth:.2f}, h_max={h_max:.2f})")
            else:
                lines.append(f"Smoothness : rough/rugged (exponent={smooth:.2f}, h_max={h_max:.2f})")

        # -- modality --
        n_basins = g("topology.n_basins")
        lmf = g("multimodality.local_min_frac")
        assort = g("network.fitness_assortativity")
        fdc = g("dispersion.fdc")
        # Two agreeing signals: neighbor graphs over smooth clouds spuriously fragment into many
        # "basins", so require both an elevated local-min fraction AND low fitness assortativity
        # (rugged neighborhoods) before calling a landscape multimodal.
        multi = (lmf > 0.03) and (not np.isnan(assort)) and (assort < 0.2)
        if multi:
            lines.append(
                f"Modality   : MULTIMODAL (local-min frac={lmf:.3f}, assortativity={assort:.2f}, fdc={fdc:.2f})"
            )
        else:
            lines.append(
                f"Modality   : unimodal / single-funnel "
                f"(local-min frac={lmf:.3f}, basins~{n_basins:.0f}, fdc={fdc:.2f})"
            )
        if not np.isnan(assort):
            lines.append(f"Neighbors  : fitness assortativity={assort:.2f} (high=smooth, low=rugged)")

        # -- separability --
        sep = g("separability.separability_index")
        offdiag = g("separability.hessian_offdiag_ratio")
        gain = g("separability.interaction_r2_gain")
        if not np.isnan(sep):
            if sep >= 0.85 and gain < 0.1:
                lines.append(f"Separable  : SEPARABLE / additive (index={sep:.2f}, offdiag={offdiag:.2f})")
            elif sep < 0.6 or gain > 0.2:
                lines.append(
                    f"Separable  : NON-separable / coupled (index={sep:.2f}, "
                    f"interaction gain={gain:.2f}, offdiag={offdiag:.2f})"
                )
            else:
                lines.append(f"Separable  : weak coupling (index={sep:.2f}, gain={gain:.2f})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact identity of this analysis."""
        return f"Landscape(n={self.ctx.n}, d={self.ctx.d}, features={len(self.features())})"

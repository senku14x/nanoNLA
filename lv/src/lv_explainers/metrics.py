"""Pure-math core for NLA reconstruction metrics.

These functions are the load-bearing measurement instruments for the whole
project (residual alignment, MSE, FVE). If any of them is subtly wrong, every
gate inherits the error coherently. They are therefore implemented in plain
numpy with no model dependency and exercised by an executable self-test
(``python -m lv_explainers.metrics``) that asserts the algebraic identities the
released NLA code relies on, in particular ``MSE == 2*(1 - cos)`` under the
released "normalize both sides to sqrt(d_model)" convention.

Conventions (matching the released NLA code, see docs/nla-method-notes.md):
  * The reward compares *individually L2-normalized* vectors. With
    ``mse_scale = sqrt(d_model)`` both vectors are placed on the sphere of
    radius sqrt(d), so the per-coordinate mean squared error equals 2*(1-cos)
    and is direction-only (magnitude is discarded by construction).
  * FVE is computed against the *un-normalized* mean of the normalized golds
    (the mean of unit-sphere vectors lies inside the sphere); do NOT normalize
    the mean. See docs/nla-method-notes.md and the released inference guide.

Nothing here imports torch; callers convert tensors to numpy float arrays first.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def _as2d(x: ArrayLike) -> np.ndarray:
    """Return x as a float64 2D array of shape (n, d)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {a.shape}")
    return a


def normalize(x: ArrayLike, scale: float | None) -> np.ndarray:
    """L2-normalize rows of x to length ``scale``.

    scale=None  -> return x unchanged (raw magnitude retained).
    scale=float -> project each row onto the sphere of that radius.

    Matches ``schema.normalize_activation`` in the released code: the default
    used by reward/critic loss is ``scale = sqrt(d_model)``.
    """
    a = _as2d(x)
    if scale is None:
        return a
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    # Guard against zero vectors (a dead/empty activation should be visible,
    # not silently turned into NaN, so we surface it).
    if np.any(norms == 0):
        raise ValueError("normalize() received a zero-norm row")
    return a / norms * float(scale)


def cosine(pred: ArrayLike, gold: ArrayLike) -> np.ndarray:
    """Row-wise cosine similarity. Shapes broadcast as (n, d) vs (n, d)."""
    p = _as2d(pred)
    g = _as2d(gold)
    pn = p / np.linalg.norm(p, axis=-1, keepdims=True)
    gn = g / np.linalg.norm(g, axis=-1, keepdims=True)
    return np.sum(pn * gn, axis=-1)


def mse_normalized(pred: ArrayLike, gold: ArrayLike, scale: float) -> np.ndarray:
    """Per-coordinate MSE after normalizing both sides to ``scale``.

    With scale=sqrt(d) this returns 2*(1-cos) elementwise per row, the released
    reward's quantity. Range [0, 4]: 0=identical, 2=orthogonal, 4=antiparallel.
    """
    p = normalize(pred, scale)
    g = normalize(gold, scale)
    return np.mean((p - g) ** 2, axis=-1)


def reward_from_mse(mse: ArrayLike, log_variant: bool = False) -> np.ndarray:
    """Released reward shaping: r = -mse (default) or r = -log(mse)."""
    m = np.asarray(mse, dtype=np.float64)
    if log_variant:
        return -np.log(np.clip(m, 1e-12, None))
    return -m


def fve(pred: ArrayLike, gold: ArrayLike, scale: float) -> float:
    """Fraction of Variance Explained on a *batch*, the released way.

    fve = 1 - mean_i ||q(pred_i) - q(gold_i)||^2 / mean_i ||q(gold_i) - mu||^2
    where q normalizes to ``scale`` and mu is the (un-normalized) mean of the
    normalized golds. Both numerator and denominator use the same per-coordinate
    mean so the d-factor cancels. Requires n >= 2 rows.
    """
    p = normalize(pred, scale)
    g = normalize(gold, scale)
    if g.shape[0] < 2:
        raise ValueError("fve needs at least 2 rows to form a denominator")
    mu = g.mean(axis=0, keepdims=True)  # un-normalized mean of normalized golds
    num = np.mean((p - g) ** 2)
    den = np.mean((g - mu) ** 2)
    if den == 0:
        raise ValueError("fve denominator is zero (all golds identical)")
    return float(1.0 - num / den)


def residual(h: ArrayLike, h_hat: ArrayLike, scale: float) -> np.ndarray:
    """Reconstruction residual e = q(h) - q(h_hat) in normalized space.

    This is the operative reward-relevant quantity: the marginal reward for a
    mention that nudges the reconstruction by delta is ~ 2<e, delta>.
    """
    return normalize(h, scale) - normalize(h_hat, scale)


def residual_alignment(
    h: ArrayLike, h_hat: ArrayLike, delta_c: ArrayLike, scale: float
) -> np.ndarray:
    """Row-wise <e, v_c>, the marginal naming reward along concept direction c.

    v_c is delta_c normalized to a *unit* vector (not to ``scale``) so the
    output is in the same units as the residual's projection onto a direction.
    """
    e = residual(h, h_hat, scale)
    dc = np.asarray(delta_c, dtype=np.float64).ravel()
    vc = dc / np.linalg.norm(dc)
    return e @ vc


def residual_alignment_z(
    h: ArrayLike,
    h_hat: ArrayLike,
    delta_c: ArrayLike,
    scale: float,
    n_random: int = 512,
    seed: int = 0,
) -> dict:
    """Residual alignment expressed as a z-score against random unit directions.

    A real concept direction must stand out above the alignment a random
    direction gets purely from the residual's overall magnitude. Returns the
    mean alignment, the random-direction null (mean/std), and the z-score.
    This is the headline Gate-0a sanity check: random ~ 0, concept >> 0 only if
    the residual genuinely concentrates along c.
    """
    e = residual(h, h_hat, scale)  # (n, d)
    d = e.shape[1]
    dc = np.asarray(delta_c, dtype=np.float64).ravel()
    vc = dc / np.linalg.norm(dc)
    concept_align = float(np.mean(e @ vc))

    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_random, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    null = np.mean(e @ R.T, axis=0)  # per random direction, mean over rows
    null_mu, null_sd = float(null.mean()), float(null.std() + 1e-12)
    return {
        "concept_alignment": concept_align,
        "null_mean": null_mu,
        "null_std": null_sd,
        "z": (concept_align - null_mu) / null_sd,
    }


# --------------------------------------------------------------------------- #
# Executable self-test: run `python -m lv_explainers.metrics`.
# Asserts the algebraic identities the project depends on. This is the sanity
# check to run before trusting any gate number.
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    rng = np.random.default_rng(0)
    d = 4096
    s = np.sqrt(d)

    # 1. MSE == 2*(1 - cos) under the normalize-to-sqrt(d) convention.
    a = rng.standard_normal((50, d))
    b = rng.standard_normal((50, d))
    m = mse_normalized(a, b, s)
    c = cosine(a, b)
    assert np.allclose(m, 2 * (1 - c), atol=1e-8), "MSE != 2(1-cos)"

    # 2. Boundary cases: identical->0, orthogonal->2, antiparallel->4.
    x = rng.standard_normal((1, d))
    assert mse_normalized(x, x, s)[0] < 1e-9, "identical MSE not ~0"
    assert abs(mse_normalized(x, -x, s)[0] - 4.0) < 1e-7, "antiparallel MSE not ~4"
    # build a vector orthogonal to x
    y = rng.standard_normal((1, d))
    y = y - (y @ x.T) / (x @ x.T) * x
    assert abs(mse_normalized(x, y, s)[0] - 2.0) < 1e-6, "orthogonal MSE not ~2"

    # 3. FVE under sphere-normalization. Because fve() re-normalizes the
    #    prediction onto the sphere of radius sqrt(d), the relevant identity is
    #    FVE ~= 2*cos - 1 (the un-normalized mean mu ~= 0 in high d). So:
    #      cos=1   -> FVE ~= 1   (perfect)
    #      cos=0.5 -> FVE ~= 0   (the zero point is cosine 0.5, NOT predict-mean)
    #      cos=0   -> FVE ~= -1  (a constant/random on-sphere predictor)
    #    NB: "predict the mean" cannot be expressed through this path because the
    #    mean is inside the sphere; that intuition is for un-normalized regression.
    gold = rng.standard_normal((400, d))
    assert abs(fve(gold, gold, s) - 1.0) < 1e-9, "FVE(perfect) != 1"

    def _pred_with_cosine(g: np.ndarray, rho: float, seed: int) -> np.ndarray:
        """Build preds with an exact per-row cosine rho to the golds."""
        r = np.random.default_rng(seed)
        gu = g / np.linalg.norm(g, axis=1, keepdims=True)
        n = r.standard_normal(g.shape)
        n = n - np.sum(n * gu, axis=1, keepdims=True) * gu  # orthogonalize
        nu = n / np.linalg.norm(n, axis=1, keepdims=True)
        return rho * gu + np.sqrt(1 - rho**2) * nu

    for rho in (0.75, 0.5, 0.25):
        f = fve(_pred_with_cosine(gold, rho, seed=7), gold, s)
        assert abs(f - (2 * rho - 1)) < 0.05, f"FVE(cos={rho}) = {f:.3f} != 2rho-1"

    # a constant on-sphere predictor (worst case) is strongly negative, ~ -1.
    const = np.repeat(normalize(rng.standard_normal((1, d)), s), gold.shape[0], 0)
    assert fve(const, gold, s) < -0.8, "constant predictor FVE not ~ -1"

    # 4. Residual alignment: a random direction is ~0 in z; a direction built
    #    to lie along the residual is strongly positive.
    h = rng.standard_normal((300, d))
    h_hat = h + 0.5 * rng.standard_normal((300, d))  # imperfect reconstruction
    rand_dir = rng.standard_normal(d)
    z_rand = residual_alignment_z(h, h_hat, rand_dir, s, seed=1)["z"]
    assert abs(z_rand) < 6, f"random direction not ~null (z={z_rand:.2f})"
    e = residual(h, h_hat, s)
    aligned_dir = e.mean(axis=0)  # direction the residual actually points
    z_aligned = residual_alignment_z(h, h_hat, aligned_dir, s, seed=1)["z"]
    assert z_aligned > z_rand + 3, "residual-aligned dir not separated from null"

    # 5. reward = -mse monotonic.
    assert reward_from_mse(0.2) > reward_from_mse(1.0), "reward not decreasing in mse"

    print("metrics self-test: PASS")
    print(f"  random-direction residual z = {z_rand:+.3f} (want ~0)")
    print(f"  residual-aligned direction z = {z_aligned:+.3f} (want >> random)")


if __name__ == "__main__":
    _selftest()

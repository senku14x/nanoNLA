"""Directional reconstruction loss + FVE, masked to a stage's active layers (spec §7).

Reuses the repo's directional convention: normalize_activation(·, mse_scale) on BOTH pred
and target, then squared error (= 2(1−cos) at unit norm) — identical to
models_multi.three_target_loss, just masked per-example to the stage's active set S_B.

Two modes:
  - progressive_stage_mean   (default): equal weight per active tap. L24 therefore gets 3x
    the total gradient mass (it's active at all 3 budgets).
  - progressive_layer_balanced: weight tap ℓ by 1/c_ℓ (c_ℓ = #budgets supervising ℓ), so the
    per-layer total supervision is equal — needed to separate a real hierarchy from
    direct-supervision imbalance (spec §7).
"""

from __future__ import annotations


def per_tap_dir_loss(pred, gold, mse_scale):
    """[B, k] per-(example, tap) directional loss. pred/gold [B,k,d] RAW (normalized here)."""
    from nla.schema import normalize_activation
    pn = normalize_activation(pred, mse_scale)
    gn = normalize_activation(gold, mse_scale)
    return ((pn - gn) ** 2).mean(dim=2)


def masked_stage_loss(pred, gold, mse_scale, active_mask, layer_weight=None):
    """Per-example weighted mean of the directional loss over the stage's ACTIVE taps; then
    .mean() over the batch by the caller.

    active_mask [B, k] in {0,1} (which target layers are supervised for each row's stage).
    layer_weight [k] or None: 1/c_ℓ for layer_balanced, None (==1) for stage_mean. Inactive
    taps contribute zero loss AND zero gradient (the head's prediction is masked out)."""
    l = per_tap_dir_loss(pred, gold, mse_scale)          # [B, k]
    w = active_mask.to(l.dtype)
    if layer_weight is not None:
        w = w * layer_weight.to(l.dtype).view(1, -1)
    denom = w.sum(dim=1).clamp_min(1e-8)
    per_example = (l * w).sum(dim=1) / denom              # [B]
    return per_example.mean()


def layer_balanced_weights(target_layers, stages):
    """w_ℓ = 1/c_ℓ over target_layers (in order), c_ℓ = supervision_counts. Returns a list."""
    from multilayer_nla.progressive_reader.schedule import supervision_counts
    c = supervision_counts(stages, tuple(target_layers))
    return [1.0 / c[l] for l in target_layers]


# ----------------------------------------------------------------- FVE (eval; spec §1.4)

def fve_from_sqerr(mean_sqerr, baseline):
    """FVE = 1 − MSE_model / baseline (the repo convention). Scalars or per-cell."""
    return 1.0 - (mean_sqerr / baseline)


def predict_mean_baseline(targets, mse_scale):
    """The repo's fve_nrm denominator = compute_predict_mean_baselines(targets, mse_scale)[1]
    (= mse_rawvar, variance of the √d-normalized targets around their mean). TRAIN-derived by
    the caller — never dev/test (spec §1.4/§9). `targets` [N, d] raw for one layer."""
    from nla.schema import compute_predict_mean_baselines
    return compute_predict_mean_baselines(targets, mse_scale)[1]

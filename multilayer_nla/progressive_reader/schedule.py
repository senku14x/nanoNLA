"""Target layers + the nested progressive schedule, with strict validation.

stdlib-only (no numpy/torch) so the validation tests run dependency-free.
"""

from __future__ import annotations

# The union of source-model residual layers the 7-tap reader reconstructs.
TARGET_LAYERS: tuple[int, ...] = (20, 22, 23, 24, 25, 26, 28)

# The three exact teacher-token-prefix budgets (token-ID counts, not characters).
# Max is 96, not 128: the gold explanations are ~112 tokens median (p90=127), so
# coverage@128 is only 9.5% — 96 keeps ~91% of rows under the strict-prefix filter
# (data_audit.json). Adjust here + the configs if a re-audit changes the distribution.
PREFIX_BUDGETS: tuple[int, ...] = (32, 64, 96)

# The genuinely NESTED progressive schedule  S_32 ⊂ S_64 ⊂ S_96.
PROGRESSIVE_STAGES: dict[int, tuple[int, ...]] = {
    32: (24,),
    64: (23, 24, 25),
    96: (20, 22, 23, 24, 25, 26, 28),
}

# Flat-All baseline: every target layer supervised at every budget.
FLAT_STAGES: dict[int, tuple[int, ...]] = {b: TARGET_LAYERS for b in PREFIX_BUDGETS}


def validate_schedule(stages: dict[int, tuple[int, ...]],
                      target_layers: tuple[int, ...] = TARGET_LAYERS,
                      *, require_nested: bool = True) -> None:
    """Fail LOUD on a malformed schedule (spec §2). Checks:

      1. budgets are unique and strictly increasing;
      2. every active layer is in `target_layers`;
      3. (if require_nested) S_{b_i} ⊂ S_{b_{i+1}} for ascending budgets.

    `require_nested=False` is for the Flat-All baseline (all layers at all budgets,
    which is trivially "nested" but we don't depend on it). Raises ValueError.
    """
    if not stages:
        raise ValueError("empty schedule")
    budgets = list(stages.keys())
    if budgets != sorted(budgets) or len(set(budgets)) != len(budgets):
        raise ValueError(f"budgets must be unique and strictly increasing, got {budgets}")

    tset = set(target_layers)
    for b, layers in stages.items():
        if len(set(layers)) != len(layers):
            raise ValueError(f"budget {b}: duplicate layers in {layers}")
        bad = [l for l in layers if l not in tset]
        if bad:
            raise ValueError(f"budget {b}: layers {bad} not in TARGET_LAYERS {target_layers}")
        if not layers:
            raise ValueError(f"budget {b}: empty active-layer set")

    if require_nested:
        asc = sorted(budgets)
        for i in range(len(asc) - 1):
            small, big = set(stages[asc[i]]), set(stages[asc[i + 1]])
            if not small.issubset(big):
                raise ValueError(
                    f"schedule not nested: S_{asc[i]}={sorted(small)} ⊄ S_{asc[i + 1]}={sorted(big)} "
                    f"(missing {sorted(small - big)})")


def active_layer_mask(active_layers, target_layers: tuple[int, ...] = TARGET_LAYERS) -> list[int]:
    """0/1 mask over `target_layers` (in order) marking which are supervised at a stage.

    Returned as a plain list[int] (stdlib) — callers cast to a tensor. Index j is 1 iff
    target_layers[j] ∈ active_layers.
    """
    aset = set(active_layers)
    bad = aset - set(target_layers)
    if bad:
        raise ValueError(f"active layers {sorted(bad)} not in TARGET_LAYERS {target_layers}")
    return [1 if l in aset else 0 for l in target_layers]


def supervision_counts(stages: dict[int, tuple[int, ...]],
                       target_layers: tuple[int, ...] = TARGET_LAYERS) -> dict[int, int]:
    """c_ℓ = number of budgets at which layer ℓ is supervised (spec §7, layer-balanced).

    For the progressive schedule: L24 -> 3, {L23,L25} -> 2, outer -> 1. The
    layer-balanced loss weights each layer by 1/c_ℓ so L24 isn't 3x over-supervised.
    """
    return {l: sum(1 for layers in stages.values() if l in layers) for l in target_layers}

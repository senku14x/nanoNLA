"""Progressive Reader v0 — dependency-free validation tests (spec §13.1, §13.2, §13.8).

These cover the correctness-critical pure logic (exact-prefix slicing, nested-schedule
validation, doc-level derangement) and run with NO torch/pyarrow/transformers, so they
gate the foundations independently of the GPU pipeline. The model/data-dependent tests
(§13.3-§13.7, §13.9) live alongside the train/eval modules once those are built.

Run:  python -m multilayer_nla.tests.test_progressive_reader
"""

from multilayer_nla.progressive_reader.schedule import (
    FLAT_STAGES, PROGRESSIVE_STAGES, TARGET_LAYERS, active_layer_mask,
    supervision_counts, validate_schedule,
)
from multilayer_nla.progressive_reader.prefix import (
    exact_prefix, effective_prefix_length, had_full_budget, teacher_ids_sha256, validate_prefix,
)
from multilayer_nla.progressive_reader.controls import assert_deranged, doc_derangement


def test_exact_prefix():
    ids = [11, 12, 13, 14, 15]
    assert exact_prefix(ids, 3) == [11, 12, 13]          # §13.1: exact slice
    assert exact_prefix(ids, 99) == ids                  # budget > len -> full
    assert effective_prefix_length(ids, 3) == 3
    assert effective_prefix_length(ids, 99) == 5
    assert had_full_budget(ids, 5) and not had_full_budget(ids, 6)
    # validate_prefix accepts the true slice, rejects a re-tokenize-style drift
    h = teacher_ids_sha256(ids)
    validate_prefix(ids, 3, [11, 12, 13], full_sha256=h)
    drifted = False
    try:
        validate_prefix(ids, 3, [11, 12, 99])            # not a true prefix
    except AssertionError:
        drifted = True
    assert drifted, "validate_prefix must reject a non-prefix"
    # sha256 is stable + sequence-sensitive
    assert teacher_ids_sha256(ids) == h
    assert teacher_ids_sha256([11, 12, 13, 14, 16]) != h


def test_nested_schedule_validation():
    validate_schedule(PROGRESSIVE_STAGES)                # the real schedule passes
    validate_schedule(FLAT_STAGES, require_nested=False)

    def must_fail(stages, why, **kw):
        try:
            validate_schedule(stages, **kw)
        except ValueError:
            return
        raise AssertionError(f"schedule should have failed: {why}")

    must_fail({32: (24,), 64: (23, 25), 128: TARGET_LAYERS}, "L24 missing at 64 (not nested)")
    must_fail({64: (23, 24, 25), 32: (24,), 128: TARGET_LAYERS}, "unsorted budgets")
    must_fail({32: (24,), 64: (23, 24, 25), 128: (20, 22, 23, 24, 25, 26, 99)}, "unknown layer 99")
    must_fail({32: (24, 24), 64: (23, 24, 25), 128: TARGET_LAYERS}, "duplicate layer in a stage")


def test_supervision_counts_and_mask():
    c = supervision_counts(PROGRESSIVE_STAGES)
    assert c[24] == 3 and c[23] == 2 and c[25] == 2
    assert c[20] == 1 and c[22] == 1 and c[26] == 1 and c[28] == 1
    assert active_layer_mask((24,)) == [0, 0, 0, 1, 0, 0, 0]          # order = TARGET_LAYERS
    assert active_layer_mask((23, 24, 25)) == [0, 0, 1, 1, 1, 0, 0]
    assert active_layer_mask(TARGET_LAYERS) == [1] * len(TARGET_LAYERS)


def test_shuffled_derangement():
    # 40 docs x 5 rows each — every shuffled row must map to a different document, none to itself.
    doc_ids = [f"doc{i // 5}" for i in range(200)]
    for seed in (0, 1, 7, 42):
        perm = doc_derangement(doc_ids, seed)
        assert_deranged(doc_ids, perm)                   # §13.8


def test_eval_metrics_match_spec():
    # M_scheduled (§9) and G_local/G_outer (§11) must match the spec formulas exactly.
    from multilayer_nla.progressive_reader.evaluate import m_scheduled, stage_gains
    from multilayer_nla.progressive_reader.schedule import PREFIX_BUDGETS
    def fve(B, l):
        return round(0.2 + 0.001 * B + 0.005 * l, 5)
    cells = {f"{B},{l}": {"fve": fve(B, l)} for B in PREFIX_BUDGETS for l in TARGET_LAYERS}
    exp_m = (fve(32, 24)
             + sum(fve(64, l) for l in (23, 24, 25)) / 3.0
             + sum(fve(128, l) for l in TARGET_LAYERS) / len(TARGET_LAYERS)) / 3.0
    assert abs(m_scheduled(cells) - exp_m) < 1e-9
    g = stage_gains(cells)
    assert abs(g["G_local"] - sum(fve(64, l) - fve(32, l) for l in (23, 25)) / 2.0) < 1e-9
    assert abs(g["G_outer"] - sum(fve(128, l) - fve(64, l) for l in (20, 22, 26, 28)) / 4.0) < 1e-9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} progressive_reader tests passed.")


if __name__ == "__main__":
    _run_all()

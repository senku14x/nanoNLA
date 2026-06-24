"""Regeneration core: published prefix -> three-layer triplet at the final token.

Pure (numpy/pyarrow) — no model. Validates the additive activation-column build
and the n_raw_tokens round-trip guard that protects the final-token == extraction
position invariant. Run: python -m multilayer_nla.tests.test_regenerate_multilayer
"""

import numpy as np
import pyarrow as pa

from multilayer_nla.regenerate_multilayer_activations import (
    SLOT_COLUMNS,
    append_triplet_columns,
)

D = 8
CENTER = 24
LAYERS = (CENTER - 1, CENTER, CENTER + 1)


def _make_table(n, n_raw):
    return pa.table({
        "detokenized_text_truncated": pa.array([f"web prefix {i}" for i in range(n)]),
        "n_raw_tokens": pa.array(n_raw, pa.int64()),
        "activation_layer": pa.array([CENTER] * n, pa.int64()),
        "doc_id": pa.array([f"doc:{i}" for i in range(n)], pa.string()),
        "response": pa.array([f"<explanation>\nfeat {i}\n</explanation>" for i in range(n)]),
    })


def _make_results(n_raw, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for seq_len in n_raw:
        hidden = {li: rng.standard_normal((seq_len, D)).astype(np.float32) for li in LAYERS}
        results.append({"token_ids": list(range(seq_len)), "hidden": hidden})
    return results


def test_appends_triplet_at_final_token():
    n_raw = [3, 5, 4]
    table = _make_table(len(n_raw), n_raw)
    results = _make_results(n_raw)
    out = append_triplet_columns(table, results, CENTER, D)
    for c in SLOT_COLUMNS:
        assert c in out.schema.names
    # centre column row i must equal the FINAL-token vector of layer 24
    centre = out.column("activation_centre").to_pylist()
    prev = out.column("activation_prev").to_pylist()
    nxt = out.column("activation_next").to_pylist()
    for i, r in enumerate(results):
        assert np.allclose(centre[i], r["hidden"][CENTER][-1])
        assert np.allclose(prev[i], r["hidden"][CENTER - 1][-1])
        assert np.allclose(nxt[i], r["hidden"][CENTER + 1][-1])


def test_center_layer_column_added():
    n_raw = [2, 2]
    out = append_triplet_columns(_make_table(2, n_raw), _make_results(n_raw), CENTER, D)
    assert out.column("center_layer").to_pylist() == [CENTER, CENTER]


def test_preserves_published_label_columns():
    n_raw = [3, 3]
    table = _make_table(2, n_raw)
    out = append_triplet_columns(table, _make_results(n_raw), CENTER, D)
    # labels + provenance carried through untouched
    assert out.column("response").to_pylist() == table.column("response").to_pylist()
    assert out.column("doc_id").to_pylist() == table.column("doc_id").to_pylist()
    assert out.column("detokenized_text_truncated").to_pylist() == \
        table.column("detokenized_text_truncated").to_pylist()


def test_roundtrip_mismatch_raises():
    n_raw = [3, 5]
    table = _make_table(2, n_raw)
    results = _make_results(n_raw)
    # corrupt row 1: re-encoded to 6 tokens, stage-0 had 5 -> must be a hard error
    results[1]["token_ids"] = list(range(6))
    results[1]["hidden"] = {li: np.zeros((6, D), np.float32) for li in LAYERS}
    try:
        append_triplet_columns(table, results, CENTER, D)
    except AssertionError as e:
        assert "round-trip" in str(e)
        assert "row 1" in str(e)
    else:
        raise AssertionError("expected AssertionError on n_raw_tokens mismatch")


def test_roundtrip_check_can_be_disabled():
    n_raw = [3]
    table = _make_table(1, n_raw)
    results = _make_results(n_raw)
    results[0]["token_ids"] = list(range(99))  # mismatch, but check off
    out = append_triplet_columns(table, results, CENTER, D, check_roundtrip=False)
    assert out.num_rows == 1  # no raise; final token still taken from hidden[-1]


def test_double_regeneration_refused():
    n_raw = [2]
    table = _make_table(1, n_raw)
    once = append_triplet_columns(table, _make_results(n_raw), CENTER, D)
    try:
        append_triplet_columns(once, _make_results(n_raw), CENTER, D)
    except AssertionError as e:
        assert "already has" in str(e)
    else:
        raise AssertionError("expected refusal to regenerate an already-populated triplet")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} regenerate-multilayer tests passed.")


if __name__ == "__main__":
    _run_all()

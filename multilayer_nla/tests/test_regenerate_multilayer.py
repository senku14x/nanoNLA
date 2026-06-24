"""Regeneration core: published prefix -> per-layer final-token activations.

Pure (numpy/pyarrow) — no model. Validates the windowed activation_L{k} build,
the final-token tap (both [seq,d] and final-token-only [d] capture formats), and
the n_raw_tokens round-trip guard (hard-fail + drop-and-log).
Run: python -m multilayer_nla.tests.test_regenerate_multilayer
"""

import numpy as np
import pyarrow as pa

from multilayer_nla.regenerate_multilayer_activations import (
    append_layer_columns,
    layer_col,
    parse_layers,
)

D = 8
TRIPLET = [23, 24, 25]
WIDE = list(range(19, 30))  # 11-layer window


def _make_table(n, n_raw):
    return pa.table({
        "detokenized_text_truncated": pa.array([f"web prefix {i}" for i in range(n)]),
        "n_raw_tokens": pa.array(n_raw, pa.int64()),
        "activation_layer": pa.array([24] * n, pa.int64()),
        "doc_id": pa.array([f"doc:{i}" for i in range(n)], pa.string()),
        "response": pa.array([f"<explanation>\nfeat {i}\n</explanation>" for i in range(n)]),
    })


def _make_results(n_raw, layers, *, final_only=False, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for seq_len in n_raw:
        if final_only:
            hidden = {li: rng.standard_normal(D).astype(np.float32) for li in layers}      # [d]
        else:
            hidden = {li: rng.standard_normal((seq_len, D)).astype(np.float32) for li in layers}  # [seq,d]
        results.append({"token_ids": list(range(seq_len)), "hidden": hidden})
    return results


def test_appends_layer_columns_at_final_token():
    n_raw = [3, 5, 4]
    table = _make_table(len(n_raw), n_raw)
    results = _make_results(n_raw, TRIPLET)
    out = append_layer_columns(table, results, TRIPLET, D)
    for li in TRIPLET:
        assert layer_col(li) in out.schema.names
    for li in TRIPLET:
        col = out.column(layer_col(li)).to_pylist()
        for i, r in enumerate(results):
            assert np.allclose(col[i], r["hidden"][li][-1])  # final token of [seq,d]


def test_final_token_only_capture_format():
    # extract_multi(..., final_token_only=True) yields [d] per layer, not [seq,d]
    n_raw = [3, 4]
    table = _make_table(2, n_raw)
    results = _make_results(n_raw, TRIPLET, final_only=True)
    out = append_layer_columns(table, results, TRIPLET, D)
    centre = out.column(layer_col(24)).to_pylist()
    for i, r in enumerate(results):
        assert np.allclose(centre[i], r["hidden"][24])  # the [d] vec used as-is


def test_wide_window_11_layers():
    n_raw = [3, 3]
    table = _make_table(2, n_raw)
    out = append_layer_columns(table, _make_results(n_raw, WIDE), WIDE, D)
    for li in WIDE:
        assert layer_col(li) in out.schema.names
    assert len([c for c in out.schema.names if c.startswith("activation_L")]) == 11


def test_does_not_add_center_layer():
    # center_layer is the caller's (main) job, not this pure column-builder's
    n_raw = [2]
    out = append_layer_columns(_make_table(1, n_raw), _make_results(n_raw, TRIPLET), TRIPLET, D)
    assert "center_layer" not in out.schema.names


def test_preserves_published_label_columns():
    n_raw = [3, 3]
    table = _make_table(2, n_raw)
    out = append_layer_columns(table, _make_results(n_raw, TRIPLET), TRIPLET, D)
    assert out.column("response").to_pylist() == table.column("response").to_pylist()
    assert out.column("doc_id").to_pylist() == table.column("doc_id").to_pylist()


def test_roundtrip_mismatch_raises():
    n_raw = [3, 5]
    table = _make_table(2, n_raw)
    results = _make_results(n_raw, TRIPLET)
    results[1]["token_ids"] = list(range(6))  # re-encoded 6 != stored 5
    try:
        append_layer_columns(table, results, TRIPLET, D)
    except AssertionError as e:
        assert "round-trip" in str(e) and "row 1" in str(e)
    else:
        raise AssertionError("expected AssertionError on n_raw_tokens mismatch")


def test_roundtrip_drop_within_threshold():
    n_raw = [3, 5, 4, 3]
    table = _make_table(4, n_raw)
    results = _make_results(n_raw, TRIPLET)
    results[2]["token_ids"] = list(range(99))  # 1/4 = 25% < 50% threshold -> dropped
    out = append_layer_columns(table, results, TRIPLET, D, max_drop_frac=0.5)
    assert out.num_rows == 3
    assert "doc:2" not in out.column("doc_id").to_pylist()
    col = out.column(layer_col(24)).to_pylist()
    assert np.allclose(col[0], results[0]["hidden"][24][-1])
    assert np.allclose(col[2], results[3]["hidden"][24][-1])  # row 3 shifted into slot 2


def test_roundtrip_drop_over_threshold_still_raises():
    n_raw = [3, 5, 4, 3]
    table = _make_table(4, n_raw)
    results = _make_results(n_raw, TRIPLET)
    results[1]["token_ids"] = list(range(99))
    results[2]["token_ids"] = list(range(99))  # 50% > 10% threshold -> raise
    try:
        append_layer_columns(table, results, TRIPLET, D, max_drop_frac=0.1)
    except AssertionError as e:
        assert "round-trip" in str(e) and "exceeds" in str(e)
    else:
        raise AssertionError("expected raise when drop fraction exceeds max_drop_frac")


def test_roundtrip_check_can_be_disabled():
    n_raw = [3]
    table = _make_table(1, n_raw)
    results = _make_results(n_raw, TRIPLET)
    results[0]["token_ids"] = list(range(99))
    out = append_layer_columns(table, results, TRIPLET, D, check_roundtrip=False)
    assert out.num_rows == 1


def test_double_regeneration_refused():
    n_raw = [2]
    table = _make_table(1, n_raw)
    once = append_layer_columns(table, _make_results(n_raw, TRIPLET), TRIPLET, D)
    try:
        append_layer_columns(once, _make_results(n_raw, TRIPLET), TRIPLET, D)
    except AssertionError as e:
        assert "already has" in str(e)
    else:
        raise AssertionError("expected refusal to re-add an existing activation_L{k} column")


def test_gather_last_real_token_equals_slice_last():
    # guards the Blocker-1 fix's indexing: under right padding, captured[batch, len-1]
    # (gathered on GPU) == captured[i, :len_i][-1] (the old full-transfer path).
    import torch
    B, T, d = 4, 6, 5
    captured = torch.randn(B, T, d)
    attn = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    lengths = attn.sum(1)
    last = (lengths - 1).clamp_min(0)
    bidx = torch.arange(B)
    gathered = captured[bidx, last]                  # the fix
    for i in range(B):
        assert torch.allclose(gathered[i], captured[i, : lengths[i]][-1])


def test_parse_layers():
    assert parse_layers("19-29") == list(range(19, 30))
    assert parse_layers("19,24,29") == [19, 24, 29]
    assert parse_layers("19-21,25,27-29") == [19, 20, 21, 25, 27, 28, 29]
    assert parse_layers("24") == [24]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} regenerate-multilayer tests passed.")


if __name__ == "__main__":
    _run_all()

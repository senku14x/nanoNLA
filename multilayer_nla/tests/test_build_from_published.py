"""build_from_published: regenerated published rows -> our 3-slot training parquets.

Validates the per-subset adaptation (prompt swap for av/rl, verbatim canonical
critic prompt for ar, label preservation) and that the output round-trips through
the real trainers' loaders. Pure pyarrow + tmp parquet — no model, no API.
Run: python -m multilayer_nla.tests.test_build_from_published
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multilayer_nla.build_from_published import assemble_published, build_one
from multilayer_nla.datasets import (
    AR_CRITIC_TEMPLATE,
    INJECT_PLACEHOLDER,
    N_SLOTS,
    SLOT_COLUMNS,
    fill_ar_prompt,
    load_ar_sft_dataset,
    load_av_sft_dataset,
)

D = 8


def _fsl(arrs):
    flat = np.concatenate([a.reshape(1, -1) for a in arrs]).reshape(-1).astype(np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), D)


def _triplet_cols(n, seed=0):
    rng = np.random.default_rng(seed)
    acts = {c: [rng.standard_normal(D).astype(np.float32) for _ in range(n)] for c in SLOT_COLUMNS}
    return {c: _fsl(acts[c]) for c in SLOT_COLUMNS}, acts


def _published_table(n, mode, *, with_response=True, with_prompt=True, with_expl=False):
    cols, acts = _triplet_cols(n)
    cols["doc_id"] = pa.array([f"doc:{i}" for i in range(n)], pa.string())
    if mode == "av" and with_response:
        cols["response"] = pa.array([f"<explanation>\nfeat {i}\n</explanation>" for i in range(n)])
    if mode == "av" and with_prompt:
        # published single-marker actor prompt (chat list) — must be discarded
        cols["prompt"] = pa.array([[{"role": "user", "content": "old single-marker ㊗ prompt"}]] * n)
    if mode == "ar" and with_prompt:
        cols["prompt"] = pa.array([fill_ar_prompt(f"feat {i}") for i in range(n)], pa.string())
    if mode == "rl" and with_prompt:
        cols["prompt"] = pa.array([[{"role": "user", "content": "old single-marker ㊗ prompt"}]] * n)
    if with_expl:
        cols["api_explanation"] = pa.array([f"feat {i}" for i in range(n)], pa.string())
    return pa.table(cols), acts


def test_av_prompt_swapped_response_kept():
    table, acts = _published_table(6, "av")
    out = assemble_published(table, "av")
    prompts = out.column("prompt").to_pylist()
    # our three-marker prompt now, not the published single-marker one
    assert prompts[0][0]["content"].count(INJECT_PLACEHOLDER) == N_SLOTS
    resp = out.column("response").to_pylist()
    assert all(r.startswith("<explanation>") for r in resp)
    for c in SLOT_COLUMNS:
        assert c in out.schema.names


def test_av_round_trips_through_loader():
    with tempfile.TemporaryDirectory() as tmp:
        table, acts = _published_table(5, "av")
        out_path = str(Path(tmp) / "av_sft.parquet")
        # write a regenerated input, run build_one, then load with the real loader
        in_path = str(Path(tmp) / "regen_av.parquet")
        pq.write_table(table, in_path)
        n = build_one(in_path, "av", out_path)
        assert n == 5
        rows = load_av_sft_dataset(out_path)
        assert len(rows) == 5
        r = rows[0]
        assert r["prompt"][0]["content"].count(INJECT_PLACEHOLDER) == N_SLOTS
        assert r["response"].startswith("<explanation>")
        for j, c in enumerate(SLOT_COLUMNS):
            assert np.allclose(r[c], acts[c][0])


def test_ar_prompt_kept_verbatim_and_canonical():
    table, _ = _published_table(4, "ar")
    out = assemble_published(table, "ar")
    prompts = out.column("prompt").to_pylist()
    assert all(p.startswith("Summary of the following text:") for p in prompts)
    assert all(p.rstrip().endswith("<summary>") for p in prompts)
    # byte-identical to our shared template (== RL-time critic format)
    assert prompts[0] == AR_CRITIC_TEMPLATE.format(explanation="feat 0")


def test_ar_round_trips_through_loader():
    with tempfile.TemporaryDirectory() as tmp:
        table, acts = _published_table(5, "ar")
        in_path = str(Path(tmp) / "regen_ar.parquet")
        out_path = str(Path(tmp) / "ar_sft.parquet")
        pq.write_table(table, in_path)
        build_one(in_path, "ar", out_path)
        rows = load_ar_sft_dataset(out_path)
        assert len(rows) == 5
        assert isinstance(rows[0]["prompt"], str)
        assert rows[0]["prompt"].rstrip().endswith("<summary>")
        assert all(c in rows[0] for c in SLOT_COLUMNS)


def test_rl_prompt_swapped_no_response():
    table, _ = _published_table(4, "rl")
    out = assemble_published(table, "rl")
    assert "response" not in out.schema.names
    prompts = out.column("prompt").to_pylist()
    assert prompts[0][0]["content"].count(INJECT_PLACEHOLDER) == N_SLOTS
    assert set(SLOT_COLUMNS).issubset(set(out.schema.names))


def test_av_explanation_fallback_wraps():
    # no response column, but api_explanation present -> wrapped into <explanation>
    table, _ = _published_table(3, "av", with_response=False, with_expl=True)
    out = assemble_published(table, "av")
    resp = out.column("response").to_pylist()
    assert all(r.startswith("<explanation>") and "feat" in r for r in resp)


def test_ar_explanation_fallback_fills_canonical():
    # no prompt column, but api_explanation present -> filled with canonical template
    table, _ = _published_table(3, "ar", with_prompt=False, with_expl=True)
    out = assemble_published(table, "ar")
    prompts = out.column("prompt").to_pylist()
    assert all(p.startswith("Summary of the following text:") for p in prompts)
    assert all(p.rstrip().endswith("<summary>") for p in prompts)


def test_ar_rejects_prompt_without_summary_anchor():
    cols, _ = _triplet_cols(2)
    cols["prompt"] = pa.array(["no anchor here", "still none"], pa.string())
    table = pa.table(cols)
    try:
        assemble_published(table, "ar")
    except AssertionError as e:
        assert "<summary>" in str(e)
    else:
        raise AssertionError("expected rejection of ar prompts lacking the <summary> anchor")


def test_missing_triplet_refused():
    table = pa.table({"response": pa.array(["<explanation>\nx\n</explanation>"])})
    try:
        assemble_published(table, "av")
    except AssertionError as e:
        assert "triplet" in str(e)
    else:
        raise AssertionError("expected refusal when the activation triplet is absent")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} build-from-published tests passed.")


if __name__ == "__main__":
    _run_all()

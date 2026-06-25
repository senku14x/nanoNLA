"""Validate multi-vector dataset loading (plan §6.1, point 2): the doc-level
split preserves ALL THREE activation columns and never splits a doc; the
slot-stacking matches the injection scan order.

Run:  python -m multilayer_nla.tests.test_datasets
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multilayer_nla.datasets import (
    CONDITIONS,
    SLOT_COLUMNS,
    apply_condition_columns,
    doc_bucket,
    split_by_document,
    stack_slot_vectors,
)


def _make_base_parquet(path, n_docs=120, ppd=5, d=8):
    rng = np.random.default_rng(0)
    prev, centre, nxt, dids, nrt, cl = [], [], [], [], [], []
    for doc in range(n_docs):
        for p in range(ppd):
            # encode (doc, pos, slot) into the values so we can verify exact preservation
            prev.append(np.full(d, doc * 1000 + p, dtype=np.float32))
            centre.append(np.full(d, doc * 1000 + p + 0.5, dtype=np.float32))
            nxt.append(np.full(d, doc * 1000 + p + 0.25, dtype=np.float32))
            dids.append(f"c:train:{doc}")
            nrt.append(p + 60)
            cl.append(24)

    def fsl(arrs):
        flat = np.concatenate([a.reshape(1, -1) for a in arrs]).reshape(-1).astype(np.float32)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat), d)

    schema = pa.schema([
        ("n_raw_tokens", pa.int64()),
        ("activation_prev", pa.list_(pa.float32(), d)),
        ("activation_centre", pa.list_(pa.float32(), d)),
        ("activation_next", pa.list_(pa.float32(), d)),
        ("center_layer", pa.int64()),
        ("doc_id", pa.string()),
    ])
    tbl = pa.table({
        "n_raw_tokens": pa.array(nrt, pa.int64()),
        "activation_prev": fsl(prev),
        "activation_centre": fsl(centre),
        "activation_next": fsl(nxt),
        "center_layer": pa.array(cl, pa.int64()),
        "doc_id": pa.array(dids, pa.string()),
    }, schema=schema)
    pq.write_table(tbl, path, row_group_size=97)  # odd size -> rows split across groups
    return n_docs, ppd, d


def test_split_doc_level_and_preserves_all_columns():
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "base.parquet")
        n_docs, ppd, d = _make_base_parquet(base)
        paths = split_by_document(base, str(Path(tmp) / "splits"),
                                  fracs=(0.25, 0.25, 0.5),
                                  names=("av_sft", "ar_sft", "rl"), seed=42)

        base_tbl = pq.read_table(base)
        # 1) all three activation columns survive in every split
        for nm, p in paths.items():
            t = pq.read_table(p)
            assert set(SLOT_COLUMNS).issubset(set(t.schema.names)), f"{nm} dropped activation columns"
            assert t.schema.names == base_tbl.schema.names, f"{nm} schema changed"

        # 2) partition is exact: every base row appears in exactly one split, no loss
        total = sum(pq.read_table(p).num_rows for p in paths.values())
        assert total == base_tbl.num_rows == n_docs * ppd

        # 3) document-level: every doc's rows are entirely within one split
        doc_to_split = {}
        for nm, p in paths.items():
            for did in set(pq.read_table(p).column("doc_id").to_pylist()):
                assert did not in doc_to_split, f"doc {did} appears in {nm} AND {doc_to_split[did]}"
                doc_to_split[did] = nm
        assert len(doc_to_split) == n_docs

        # 4) values preserved exactly for a sampled doc (centre column carries doc*1000+p+0.5)
        for nm, p in paths.items():
            t = pq.read_table(p)
            if t.num_rows == 0:
                continue
            dids = t.column("doc_id").to_pylist()
            nrts = t.column("n_raw_tokens").to_pylist()
            col = t.column("activation_centre").combine_chunks()
            centre = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(col), -1)
            for i in range(min(5, t.num_rows)):
                doc = int(dids[i].split(":")[-1])
                pos = nrts[i] - 60
                assert np.allclose(centre[i], doc * 1000 + pos + 0.5), f"value corrupted in {nm}"

        # 5) deterministic
        paths2 = split_by_document(base, str(Path(tmp) / "splits2"),
                                   fracs=(0.25, 0.25, 0.5), seed=42)
        for nm in paths:
            assert (pq.read_table(paths[nm]).column("doc_id").to_pylist()
                    == pq.read_table(paths2[nm]).column("doc_id").to_pylist())


def test_doc_bucket_routing_and_balance():
    fracs = (0.25, 0.25, 0.5)
    # same doc_id -> same bucket (document-level guarantee at the routing level)
    assert doc_bucket("c:train:5", fracs, 42) == doc_bucket("c:train:5", fracs, 42)
    counts = [0, 0, 0]
    for i in range(3000):
        counts[doc_bucket(f"c:train:{i}", fracs, 42)] += 1
    frac = [c / 3000 for c in counts]
    assert abs(frac[0] - 0.25) < 0.04 and abs(frac[1] - 0.25) < 0.04 and abs(frac[2] - 0.5) < 0.04, frac


def test_stack_slot_vectors_order():
    d = 4
    rows = [
        {"activation_prev": np.arange(d) + 0.0,
         "activation_centre": np.arange(d) + 10.0,
         "activation_next": np.arange(d) + 20.0},
        {"activation_prev": np.arange(d) + 100.0,
         "activation_centre": np.arange(d) + 110.0,
         "activation_next": np.arange(d) + 120.0},
    ]
    v = stack_slot_vectors(rows, k=3)  # [B*k, d] = [6, d]
    assert v.shape == (6, d)
    # example-major, slot order [prev, centre, next] — the exact order the injection
    # scan (row-major over markers) walks.
    assert np.allclose(v[0], np.arange(d) + 0.0)
    assert np.allclose(v[1], np.arange(d) + 10.0)
    assert np.allclose(v[2], np.arange(d) + 20.0)
    assert np.allclose(v[3], np.arange(d) + 100.0)
    assert np.allclose(v[4], np.arange(d) + 110.0)
    assert np.allclose(v[5], np.arange(d) + 120.0)


def test_apply_condition_columns():
    """§7 ablation transform: coherent is identity; duplicate makes every slot the
    centre column (for BOTH injected vectors and AR targets, since both read these
    columns); the input is never mutated; unknown conditions raise."""
    n, d = 6, 8
    rng = np.random.default_rng(1)
    acts = {c: rng.standard_normal((n, d)).astype(np.float32) for c in SLOT_COLUMNS}
    centre = acts["activation_centre"]
    assert not np.array_equal(acts["activation_prev"], centre), "fixture must have distinct slots"

    # coherent = identity
    coh = apply_condition_columns(acts, "coherent")
    for c in SLOT_COLUMNS:
        assert np.array_equal(coh[c], acts[c])

    # duplicate = every slot is the centre column; the input dict is left intact
    dup = apply_condition_columns(acts, "duplicate")
    for c in SLOT_COLUMNS:
        assert np.array_equal(dup[c], centre), f"{c} != centre under duplicate"
    assert not np.array_equal(acts["activation_prev"], acts["activation_centre"]), "input was mutated"

    # the transform propagates through slot-stacking: a duplicated row stacks to centre x3
    dup_rows = [{c: dup[c][i] for c in SLOT_COLUMNS} for i in range(n)]
    v = stack_slot_vectors(dup_rows, k=3)  # [n*3, d], example-major [prev, centre, next]
    for i in range(n):
        for slot in range(3):
            assert np.allclose(v[3 * i + slot], centre[i]), "duplicate slot != centre after stacking"

    assert set(CONDITIONS) == {"coherent", "duplicate"}
    try:
        apply_condition_columns(acts, "bogus")
        assert False, "unknown condition must raise"
    except ValueError:
        pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} dataset tests passed.")


if __name__ == "__main__":
    _run_all()

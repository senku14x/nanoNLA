"""Validate the §7 SFT control-sweep builder + splits (offline, synthetic bank).

Encodes a recognizable value per (doc, pos, layer) into a tiny synthetic L19-L29
bank, runs the document-level splits + build_all, then proves:
  - av_in_* carry the RIGHT layers per condition (local/duplicate/wide/single);
  - the AR reconstruction target is byte-identical [L23,L24,L25] for EVERY condition;
  - end-to-end eval rows share source identity + targets across local/duplicate/wide;
  - the preflight assertions pass; dev/test are doc-disjoint.

Run:  python -m multilayer_nla.tests.test_build_sweep
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.schema import wrap_explanation
from multilayer_nla.datasets import AR_TARGET_COLUMNS, av_in_columns, fill_ar_prompt
from multilayer_nla import build_sweep, splits

LAYERS = list(range(19, 30))
D = 6


def _val(doc, pos, L):
    """Unique-ish per (doc,pos,layer) so we can read back which layer landed in a slot."""
    return float(doc) * 1000.0 + float(pos) + float(L) * 0.0001


def _fsl(arrs):
    flat = np.concatenate([a.reshape(1, -1) for a in arrs]).reshape(-1).astype(np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), D)


def _make_bank(bank_dir, subset, n_docs=120, ppd=3, with_response=False, with_prompt=False):
    cols = {L: [] for L in LAYERS}
    dids, resp, prompt = [], [], []
    for doc in range(n_docs):
        did = f"{subset}:{doc}"
        for pos in range(ppd):
            for L in LAYERS:
                cols[L].append(np.full(D, _val(doc, pos, L), np.float32))
            dids.append(did)
            if with_response:
                resp.append(wrap_explanation(f"expl {doc}.{pos}"))
            if with_prompt:
                prompt.append(fill_ar_prompt(f"expl {doc}.{pos}"))
    tbl = {"doc_id": pa.array(dids)}
    for L in LAYERS:
        tbl[f"activation_L{L}"] = _fsl(cols[L])
    if with_response:
        tbl["response"] = pa.array(resp)
    if with_prompt:
        tbl["prompt"] = pa.array(prompt)
    Path(bank_dir).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(tbl), str(Path(bank_dir) / f"{subset}.shard00of01.parquet"),
                   row_group_size=53)


def _col(path, name):
    t = pq.read_table(path, columns=[name]).column(name).combine_chunks()
    return t.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)


def _expected(path, layer, ppd=3):
    """The value each row SHOULD hold for `layer`, from its (doc,pos) recovered by order."""
    dids = pq.read_table(path, columns=["doc_id"]).column("doc_id").to_pylist()
    srcs = pq.read_table(path, columns=["src_row_id"]).column("src_row_id").to_pylist()
    out = np.zeros((len(dids), D), np.float32)
    for i, s in enumerate(srcs):
        out[i] = _val(s // ppd, s % ppd, layer)
    return out


def test_build_sweep_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        bank = Path(tmp) / "bank"
        _make_bank(bank, "av_sft", with_response=True)
        _make_bank(bank, "ar_sft", with_prompt=True)
        _make_bank(bank, "rl")
        sweep = Path(tmp) / "sweep"

        splits.build_split_manifest([str(bank / "rl.shard00of01.parquet")], "rl", str(sweep),
                                    seed=42, dev_subset=None, test_subset=None)
        splits.build_split_manifest([str(bank / "ar_sft.shard00of01.parquet")], "ar", str(sweep),
                                    seed=42)
        build_sweep.build_all(str(bank), str(sweep),
                              str(sweep / "rl_split_manifest.json"),
                              str(sweep / "ar_split_manifest.json"))

        # ---- av_in_* carry the right layers per condition ----
        expect = {"local": [23, 24, 25], "duplicate": [24, 24, 24],
                  "wide": [20, 24, 28], "single": [24]}
        for cond, layers in expect.items():
            p = sweep / f"av_{cond}.parquet"
            slot_cols = av_in_columns(len(layers))
            for sc, L in zip(slot_cols, layers):
                got = _col(p, sc)
                assert np.allclose(got, _expected(p, L)), f"av_{cond}.{sc} is not layer L{L}"

        # ---- AR target is byte-identical [L23,L24,L25] across every condition's e2e set ----
        ref = {tc: _col(sweep / "rl_dev_local.parquet", tc) for tc in AR_TARGET_COLUMNS}
        for tc, L in zip(AR_TARGET_COLUMNS, [23, 24, 25]):
            assert np.allclose(ref[tc], _expected(sweep / "rl_dev_local.parquet", L)), f"{tc} != L{L}"
        for cond in ("duplicate", "wide", "single"):
            p = sweep / f"rl_dev_{cond}.parquet"
            for tc in AR_TARGET_COLUMNS:
                assert np.array_equal(_col(p, tc), ref[tc]), \
                    f"rl_dev_{cond}.{tc} target drifted (must be fixed L23/24/25)"

        # ---- duplicate really injects L24 thrice; local does not ----
        dup = sweep / "av_duplicate.parquet"
        assert np.array_equal(_col(dup, "av_in_0"), _col(dup, "av_in_1")) and \
               np.array_equal(_col(dup, "av_in_1"), _col(dup, "av_in_2")), "duplicate slots differ"
        loc = sweep / "av_local.parquet"
        assert not np.array_equal(_col(loc, "av_in_0"), _col(loc, "av_in_1")), "local slots collapsed"

        # ---- dev/test doc-disjoint ----
        dev = set(pq.read_table(sweep / "rl_dev_local.parquet", columns=["doc_id"]).column("doc_id").to_pylist())
        test = set(pq.read_table(sweep / "rl_test_local.parquet", columns=["doc_id"]).column("doc_id").to_pylist())
        assert dev and test and not (dev & test), "rl dev/test overlap or empty"


def test_locked_subset_is_deterministic_and_sized():
    docs = [f"d:{i}" for i in range(500)]
    a = splits.locked_subset(docs, 64, seed=42)
    b = splits.locked_subset(docs, 64, seed=42)
    assert a == b and len(a) == 64
    assert set(a).issubset(set(docs))
    assert splits.locked_subset(docs, 9999, seed=42) == sorted(docs)  # n>=len -> all


def test_build_sweep_pool_mean():
    """MEAN-pooled (averaged-input) condition: build_av/build_rl_eval(pool=True) emit a
    single k=1 slot av_in_0 == mean of the pooled layers, pass the assert_pool_mean gate,
    and stay paired (doc_id, src_row_id) with single[24] so the paired bootstrap holds.
    Quadratic-in-L synthetic bank so mean(L23,24,25) != any single layer (the default
    linear bank would make the mean exactly == L24 and mask a copy-vs-mean bug)."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = Path(tmp) / "bank"
        sweep = Path(tmp) / "sweep"
        bank.mkdir(parents=True, exist_ok=True)

        def valnl(doc, pos, L):
            return float(doc) * 1000.0 + float(pos) + float(L * L) * 0.001

        for subset, with_resp in (("av_sft", True), ("rl", False)):
            cols = {L: [] for L in LAYERS}
            dids, resp = [], []
            for doc in range(60):
                for pos in range(3):
                    for L in LAYERS:
                        cols[L].append(np.full(D, valnl(doc, pos, L), np.float32))
                    dids.append(f"{subset}:{doc}")
                    if with_resp:
                        resp.append(wrap_explanation(f"e{doc}.{pos}"))
            tbl = {"doc_id": pa.array(dids)}
            for L in LAYERS:
                tbl[f"activation_L{L}"] = _fsl(cols[L])
            if with_resp:
                tbl["response"] = pa.array(resp)
            pq.write_table(pa.table(tbl), str(bank / f"{subset}.shard00of01.parquet"), row_group_size=53)

        splits.build_split_manifest([str(bank / "rl.shard00of01.parquet")], "rl", str(sweep),
                                    seed=42, dev_subset=None, test_subset=None)
        seed, fracs, names, sub = build_sweep._read_split(str(sweep / "rl_split_manifest.json"))
        avb = [str(bank / "av_sft.shard00of01.parquet")]
        rlb = [str(bank / "rl.shard00of01.parquet")]

        build_sweep.build_av(avb, str(sweep / "av_mean.parquet"), [23, 24, 25], pool=True)
        nm = pq.ParquetFile(sweep / "av_mean.parquet").schema_arrow.names
        assert "av_in_0" in nm and "av_in_1" not in nm, "pooled AV must be k=1 (av_in_0 only)"

        bidx = list(names).index("dev")
        build_sweep.build_rl_eval(rlb, str(sweep / "rl_dev_mean.parquet"), [23, 24, 25], [23, 24, 25],
                                  bidx, fracs, seed, pool=True, subset=sub.get("dev"))
        build_sweep.build_rl_eval(rlb, str(sweep / "rl_dev_single.parquet"), [24], [23, 24, 25],
                                  bidx, fracs, seed, subset=sub.get("dev"))
        # exact gate: av_in_0 == mean(in-file targets L23/24/25) and != each single target
        build_sweep.assert_pool_mean(str(sweep / "rl_dev_mean.parquet"), [23, 24, 25], [23, 24, 25])
        # pooling must NOT change which rows survive -> stays paired with single
        def key(p):
            t = pq.read_table(p, columns=["doc_id", "src_row_id"])
            return list(zip(t.column("doc_id").to_pylist(), t.column("src_row_id").to_pylist()))
        assert key(sweep / "rl_dev_mean.parquet") == key(sweep / "rl_dev_single.parquet"), \
            "pooled rows not paired with single (doc_id, src_row_id) — paired bootstrap would break"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} build_sweep tests passed.")


if __name__ == "__main__":
    _run_all()

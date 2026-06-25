"""Build the §7 SFT control-sweep datasets from the regenerated wide bank (L19-L29).

AV input layers and AR target layers are INDEPENDENT. The AR reconstruction target
is ALWAYS the [L23,L24,L25] triplet, for EVERY condition; only the AV *input* slots
vary:

  local     av_in = L23,L24,L25   (k=3, 3-marker prompt)   <- adjacent local stages
  duplicate av_in = L24,L24,L24   (k=3, 3-marker prompt)   <- repeated centre (control)
  wide      av_in = L20,L24,L28   (k=3, 3-marker prompt)   <- wider depth coverage
  single    av_in = L24           (k=1, 1-marker prompt)   <- single-layer baseline

Outputs (into --out-dir, condition-named; the preflight refuses to clobber a smoke run):
  ar_common.parquet / ar_dev.parquet / ar_test.parquet         (shared AR train + AR-only gold eval)
      prompt(canonical critic, verbatim) + activation_prev/centre/next(=L23/24/25) + doc_id
  av_<cond>.parquet                                            (AV-SFT train, per condition)
      prompt(k-marker) + response(label) + av_in_*(cond layers) + doc_id + src_row_id
  rl_dev_<cond>.parquet / rl_test_<cond>.parquet               (end-to-end eval, per condition)
      av_in_*(cond layers) + activation_prev/centre/next(=L23/24/25) + doc_id + src_row_id

Document bucketing reuses datasets.doc_bucket with the SAME (seed, fracs) as
multilayer_nla.splits, so the on-disk partition matches the split manifest by
construction. Rows stream shard by shard; only the needed activation_L{} columns are
projected (the unused bank layers are never read).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.schema import EXPLANATION_OPEN, wrap_explanation
from multilayer_nla.datasets import AR_TARGET_COLUMNS, av_in_columns, build_av_prompt, doc_bucket
from multilayer_nla.build_from_published import (
    EXPLANATION_COL, PROMPT_COL, RESPONSE_COL, _AR_PREFIX, _AR_SUFFIX, _subset_inputs,
)

# Pre-registered §7 conditions: AV input layers per condition (AR target is fixed).
CONDITIONS = {
    "local":     [23, 24, 25],
    "duplicate": [24, 24, 24],
    "wide":      [20, 24, 28],
    "single":    [24],
}
# The three conditions that share source rows + AR targets and differ ONLY in av_in_*
# (single has a different marker count, so it's excluded from that identity check).
IDENTITY_CONDS = ("local", "duplicate", "wide")
AR_TARGET_LAYERS = [23, 24, 25]


def _layer_col(L: int) -> str:
    return f"activation_L{L}"


def _need_layers(schema_names, layers, what: str) -> None:
    missing = [_layer_col(L) for L in dict.fromkeys(layers) if _layer_col(L) not in schema_names]
    if missing:
        raise SystemExit(
            f"{what}: bank is missing {missing}. Regenerate with --save-layers covering "
            f"{sorted(set(layers))} (the wide bank should hold L19-L29)."
        )


def _bucket_keep(doc_ids, bucket_idx, fracs, seed, subset=None):
    """Boolean mask: rows whose doc routes to bucket_idx (and, if given, is in subset)."""
    return np.fromiter(
        ((doc_bucket(d, fracs, seed) == bucket_idx) and (subset is None or d in subset)
         for d in doc_ids),
        dtype=bool, count=len(doc_ids),
    )


def build_av(bank_paths, out_path, av_layers, *, batch_size=4096) -> int:
    """av_<cond>.parquet from the av_sft bank: k-marker prompt + response label +
    av_in_*(av_layers) + doc_id + src_row_id. No bucket filter — AV trains on ALL av_sft.

    src_row_id is the global ordinal over the av_sft stream (pre-anything), so the
    same source row gets the same id across local/duplicate/wide (the identity check).
    """
    k = len(av_layers)
    slot_cols = av_in_columns(k)
    av_prompt = build_av_prompt(k)
    schema_names = pq.ParquetFile(bank_paths[0]).schema_arrow.names
    _need_layers(schema_names, av_layers, "av")
    has_resp, has_expl = RESPONSE_COL in schema_names, EXPLANATION_COL in schema_names
    if not (has_resp or has_expl):
        raise SystemExit(f"av bank needs a {RESPONSE_COL!r} or {EXPLANATION_COL!r} column for the label")
    label_col = RESPONSE_COL if has_resp else EXPLANATION_COL
    proj = list(dict.fromkeys([_layer_col(L) for L in av_layers] + ["doc_id", label_col]))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for p in bank_paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_size, columns=proj):
            t = pa.Table.from_batches([batch])
            m = t.num_rows
            raw = t.column(label_col).to_pylist()
            resp = raw if has_resp else [wrap_explanation(e) for e in raw]
            bad = [i for i, r in enumerate(resp) if not r or EXPLANATION_OPEN not in r]
            assert not bad, (f"av: {len(bad)} rows have an empty/unwrapped response "
                             f"(first idx {bad[0]}); expected the published <explanation> label")
            cols = {
                "prompt": pa.array([av_prompt] * m),
                "response": pa.array(resp, pa.string()),
                "doc_id": t.column("doc_id"),
                "src_row_id": pa.array(list(range(n, n + m)), pa.int64()),
            }
            for sc, L in zip(slot_cols, av_layers):
                cols[sc] = t.column(_layer_col(L))
            out = pa.table(cols)
            if writer is None:
                writer = pq.ParquetWriter(out_path, out.schema)
            writer.write_table(out)
            n += m
    if writer is not None:
        writer.close()
    print(f"[sweep:av] {Path(out_path).name}  ({n} rows, av_in={av_layers}, k={k})")
    return n


def build_ar(bank_paths, out_path, target_layers, *, bucket_idx=None, fracs=None,
             seed=42, batch_size=4096) -> int:
    """ar_<bucket>.parquet from the ar_sft bank: canonical critic prompt (verbatim) +
    activation_prev/centre/next(=target_layers) + doc_id. bucket_idx filters by doc."""
    assert len(target_layers) == len(AR_TARGET_COLUMNS), "AR needs exactly 3 target layers"
    schema_names = pq.ParquetFile(bank_paths[0]).schema_arrow.names
    _need_layers(schema_names, target_layers, "ar")
    if PROMPT_COL not in schema_names:
        raise SystemExit(f"ar bank needs the published critic {PROMPT_COL!r} column")
    proj = list(dict.fromkeys([_layer_col(L) for L in target_layers] + ["doc_id", PROMPT_COL]))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for p in bank_paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_size, columns=proj):
            t = pa.Table.from_batches([batch])
            prompts = t.column(PROMPT_COL).to_pylist()
            bad = [i for i, pr in enumerate(prompts)
                   if not (pr and pr.startswith(_AR_PREFIX) and pr.endswith(_AR_SUFFIX))]
            assert not bad, (f"ar: {len(bad)} prompts are NOT the canonical critic template "
                             f"(first idx {bad[0]}: {prompts[bad[0]]!r})")
            cols = {"prompt": pa.array(prompts, pa.string()), "doc_id": t.column("doc_id")}
            for tc, L in zip(AR_TARGET_COLUMNS, target_layers):
                cols[tc] = t.column(_layer_col(L))
            out = pa.table(cols)
            if bucket_idx is not None:
                keep = _bucket_keep(out.column("doc_id").to_pylist(), bucket_idx, fracs, seed)
                out = out.filter(pa.array(keep))
            if out.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out_path, out.schema)
                writer.write_table(out)
                n += out.num_rows
    if writer is not None:
        writer.close()
    print(f"[sweep:ar] {Path(out_path).name}  ({n} rows, targets={target_layers}, bucket={bucket_idx})")
    return n


def build_rl_eval(bank_paths, out_path, av_layers, target_layers, bucket_idx, fracs, seed,
                  *, subset=None, batch_size=4096) -> int:
    """rl_<bucket>_<cond>.parquet from the rl bank: av_in_*(av_layers) +
    activation_prev/centre/next(=target_layers) + doc_id + src_row_id. Filtered to
    bucket_idx (dev/test); optional `subset` (locked doc_ids) restricts further.

    src_row_id is the global ordinal over the rl stream (pre-filter) so the same source
    row gets the same id across conditions.
    """
    k = len(av_layers)
    slot_cols = av_in_columns(k)
    assert len(target_layers) == len(AR_TARGET_COLUMNS), "rl-eval needs exactly 3 target layers"
    schema_names = pq.ParquetFile(bank_paths[0]).schema_arrow.names
    _need_layers(schema_names, list(av_layers) + list(target_layers), "rl-eval")
    proj = list(dict.fromkeys(
        [_layer_col(L) for L in dict.fromkeys(list(av_layers) + list(target_layers))] + ["doc_id"]))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, n, seen = None, 0, 0
    for p in bank_paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_size, columns=proj):
            t = pa.Table.from_batches([batch])
            m = t.num_rows
            cols = {
                "doc_id": t.column("doc_id"),
                "src_row_id": pa.array(list(range(seen, seen + m)), pa.int64()),
            }
            for sc, L in zip(slot_cols, av_layers):
                cols[sc] = t.column(_layer_col(L))
            for tc, L in zip(AR_TARGET_COLUMNS, target_layers):
                cols[tc] = t.column(_layer_col(L))
            out = pa.table(cols)
            keep = _bucket_keep(out.column("doc_id").to_pylist(), bucket_idx, fracs, seed, subset)
            out = out.filter(pa.array(keep))
            if out.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out_path, out.schema)
                writer.write_table(out)
                n += out.num_rows
            seen += m
    if writer is not None:
        writer.close()
    print(f"[sweep:rl-eval] {Path(out_path).name}  ({n} rows, av_in={av_layers}, "
          f"targets={target_layers}, bucket={bucket_idx})")
    return n


# ------------------------------------------------------------------ preflight

def _col_np(path, name):
    t = pq.read_table(path, columns=[name]).column(name).combine_chunks()
    return t.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1)


def assert_conditions(out_dir: Path, fracs, seed) -> dict:
    """Preflight invariants over the built datasets. Raises on any violation."""
    out_dir = Path(out_dir)
    report = {}

    # 1) AV training datasets local/duplicate/wide: identical source rows
    #    (doc_id, src_row_id, response, prompt) and ONLY av_in_* differ.
    base_key = None
    for c in IDENTITY_CONDS:
        p = out_dir / f"av_{c}.parquet"
        t = pq.read_table(p, columns=["doc_id", "src_row_id", "response"])
        key = (t.column("doc_id").to_pylist(),
               t.column("src_row_id").to_pylist(),
               t.column("response").to_pylist())
        if base_key is None:
            base_key = key
        else:
            assert key == base_key, f"av_{c}: source rows differ from av_{IDENTITY_CONDS[0]}"
        # prompt is the same constant 3-marker prompt across these three
    report["av_identity_rows"] = len(base_key[0])

    # 2) av_in_* layer correctness per condition + duplicate-slot equality.
    for c, layers in CONDITIONS.items():
        p = out_dir / f"av_{c}.parquet"
        names = pq.ParquetFile(p).schema_arrow.names
        slot_cols = [n for n in names if n.startswith("av_in_")]
        assert len(slot_cols) == len(layers), (
            f"av_{c}: {len(slot_cols)} av_in_* columns, expected {len(layers)} for layers {layers}")
        # duplicate: all slots byte-identical; local/wide: distinct where layers distinct
        slots = {sc: _col_np(p, sc) for sc in av_in_columns(len(layers))}
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                same = np.array_equal(slots[f"av_in_{i}"], slots[f"av_in_{j}"])
                if layers[i] == layers[j]:
                    assert same, f"av_{c}: slots {i},{j} share layer L{layers[i]} but differ"
                else:
                    assert not same, f"av_{c}: slots {i},{j} are L{layers[i]}/L{layers[j]} but are equal"

    # 3) end-to-end eval datasets (rl_dev/rl_test): across local/duplicate/wide the
    #    AR targets are byte-identical and only av_in_* differ (same source rows).
    for bucket in ("dev", "test"):
        base_tgt = None
        base_rows = None
        for c in IDENTITY_CONDS:
            p = out_dir / f"rl_{bucket}_{c}.parquet"
            if not p.exists():
                continue
            ids = pq.read_table(p, columns=["doc_id", "src_row_id"])
            rows = (ids.column("doc_id").to_pylist(), ids.column("src_row_id").to_pylist())
            tgt = {tc: _col_np(p, tc) for tc in AR_TARGET_COLUMNS}
            if base_tgt is None:
                base_tgt, base_rows = tgt, rows
            else:
                assert rows == base_rows, f"rl_{bucket}_{c}: source rows differ"
                for tc in AR_TARGET_COLUMNS:
                    assert np.array_equal(tgt[tc], base_tgt[tc]), (
                        f"rl_{bucket}_{c}: AR target {tc} not byte-identical to "
                        f"rl_{bucket}_{IDENTITY_CONDS[0]} (the target must be fixed)")
        if base_rows is not None:
            report[f"rl_{bucket}_rows"] = len(base_rows[0])

    # 4) split doc_ids disjoint across the three rl buckets we built (dev vs test).
    dev_docs = test_docs = None
    pdev, ptest = out_dir / "rl_dev_local.parquet", out_dir / "rl_test_local.parquet"
    if pdev.exists() and ptest.exists():
        dev_docs = set(pq.read_table(pdev, columns=["doc_id"]).column("doc_id").to_pylist())
        test_docs = set(pq.read_table(ptest, columns=["doc_id"]).column("doc_id").to_pylist())
        inter = dev_docs & test_docs
        assert not inter, f"rl_dev and rl_test share {len(inter)} docs (e.g. {sorted(inter)[:3]})"
    # AR train/dev/test disjoint
    ar_sets = {}
    for b in ("common", "dev", "test"):
        p = out_dir / f"ar_{b}.parquet"
        if p.exists():
            ar_sets[b] = set(pq.read_table(p, columns=["doc_id"]).column("doc_id").to_pylist())
    for a in ar_sets:
        for b in ar_sets:
            if a < b:
                inter = ar_sets[a] & ar_sets[b]
                assert not inter, f"ar_{a} and ar_{b} share {len(inter)} docs"

    print(f"[sweep:preflight] OK  {report}")
    return report


# ------------------------------------------------------------------ orchestration

def _bucket_idx(names, bucket):
    assert bucket in names, f"bucket {bucket!r} not in split names {names}"
    return list(names).index(bucket)


def _read_split(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    return (m["seed"], tuple(m["fracs"]), tuple(m["names"]),
            {k: set(v) for k, v in m.get("locked_subsets", {}).items()})


def build_all(in_dir, out_dir, rl_manifest, ar_manifest, *, ar_target_layers=AR_TARGET_LAYERS,
              batch_size=4096, allow_existing=False):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.parquet"))
    if existing and not allow_existing:
        raise SystemExit(f"--out-dir {out_dir} already holds {len(existing)} parquet(s) "
                         f"(e.g. {existing[0].name}); refusing to clobber. Use a fresh dir "
                         f"or --allow-existing.")

    rl_seed, rl_fracs, rl_names, rl_sub = _read_split(rl_manifest)
    ar_seed, ar_fracs, ar_names, _ = _read_split(ar_manifest)

    av_bank = _subset_inputs(in_dir, "av_sft")
    ar_bank = _subset_inputs(in_dir, "ar_sft")
    rl_bank = _subset_inputs(in_dir, "rl")

    counts = {"ar": {}, "av": {}, "rl_dev": {}, "rl_test": {}}

    # shared AR: train on ar_train, gold eval on ar_dev / ar_test
    counts["ar"]["common"] = build_ar(ar_bank, str(out_dir / "ar_common.parquet"), ar_target_layers,
                                      bucket_idx=_bucket_idx(ar_names, "train"), fracs=ar_fracs,
                                      seed=ar_seed, batch_size=batch_size)
    for b in ("dev", "test"):
        counts["ar"][b] = build_ar(ar_bank, str(out_dir / f"ar_{b}.parquet"), ar_target_layers,
                                   bucket_idx=_bucket_idx(ar_names, b), fracs=ar_fracs,
                                   seed=ar_seed, batch_size=batch_size)

    # per-condition AV training datasets + end-to-end eval datasets
    for c, av_layers in CONDITIONS.items():
        counts["av"][c] = build_av(av_bank, str(out_dir / f"av_{c}.parquet"), av_layers,
                                   batch_size=batch_size)
        for b in ("dev", "test"):
            counts[f"rl_{b}"][c] = build_rl_eval(
                rl_bank, str(out_dir / f"rl_{b}_{c}.parquet"), av_layers, ar_target_layers,
                _bucket_idx(rl_names, b), rl_fracs, rl_seed,
                subset=rl_sub.get(b), batch_size=batch_size)

    report = assert_conditions(out_dir, rl_fracs, rl_seed)
    (out_dir / "sweep_build_manifest.json").write_text(json.dumps({
        "in_dir": str(in_dir),
        "conditions": CONDITIONS,
        "ar_target_layers": ar_target_layers,
        "rl_split_manifest": str(rl_manifest),
        "ar_split_manifest": str(ar_manifest),
        "counts": counts,
        "preflight": report,
        "note": "AR target is fixed [L23,L24,L25] for every condition; only av_in_* varies.",
    }, indent=2))
    print(f"[sweep] all datasets -> {out_dir}")
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["av", "ar", "rl-eval", "all"], required=True)
    p.add_argument("--in", dest="inp", help="bank shard glob (single-mode)")
    p.add_argument("--in-dir", help="regen bank dir with av_sft/ar_sft/rl shards (--mode all)")
    p.add_argument("--out", help="output parquet (single-mode)")
    p.add_argument("--out-dir", help="output dir (--mode all)")
    p.add_argument("--av-slot-layers", help="comma-sep AV input layers, e.g. 23,24,25 or 24,24,24 or 24")
    p.add_argument("--ar-target-layers", default="23,24,25",
                   help="comma-sep AR target layers (MUST stay 23,24,25 for the sweep)")
    p.add_argument("--bucket", help="split bucket to keep (train|dev|test) for ar / rl-eval")
    p.add_argument("--rl-split-manifest", help="rl split manifest (--mode all / rl-eval)")
    p.add_argument("--ar-split-manifest", help="ar split manifest (--mode all)")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--allow-existing", action="store_true", help="permit writing into a non-empty out-dir")
    args = p.parse_args()

    def _layers(s):
        return [int(x) for x in s.split(",")]
    import glob as _glob

    if args.mode == "all":
        assert args.in_dir and args.out_dir and args.rl_split_manifest and args.ar_split_manifest, \
            "--mode all needs --in-dir --out-dir --rl-split-manifest --ar-split-manifest"
        build_all(args.in_dir, args.out_dir, args.rl_split_manifest, args.ar_split_manifest,
                  ar_target_layers=_layers(args.ar_target_layers), batch_size=args.batch_size,
                  allow_existing=args.allow_existing)
        return

    assert args.inp and args.out, f"--mode {args.mode} needs --in and --out"
    ins = sorted(_glob.glob(args.inp)) or [args.inp]
    if args.mode == "av":
        assert args.av_slot_layers, "--mode av needs --av-slot-layers"
        build_av(ins, args.out, _layers(args.av_slot_layers), batch_size=args.batch_size)
    elif args.mode in ("ar", "rl-eval"):
        seed, fracs, names, sub = _read_split(args.rl_split_manifest) if args.rl_split_manifest else (42, (0.8, 0.1, 0.1), ("train", "dev", "test"), {})
        bidx = _bucket_idx(names, args.bucket) if args.bucket else None
        if args.mode == "ar":
            build_ar(ins, args.out, _layers(args.ar_target_layers), bucket_idx=bidx,
                     fracs=fracs, seed=seed, batch_size=args.batch_size)
        else:
            assert args.av_slot_layers and args.bucket, "--mode rl-eval needs --av-slot-layers and --bucket"
            build_rl_eval(ins, args.out, _layers(args.av_slot_layers), _layers(args.ar_target_layers),
                          bidx, fracs, seed, subset=sub.get(args.bucket), batch_size=args.batch_size)


if __name__ == "__main__":
    main()

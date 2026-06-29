"""Resilient, standalone integrity verifier for the §7 sweep datasets.

WHY THIS EXISTS. The canonical preflight (`build_sweep.assert_conditions`) only runs under
`--mode all` and ABORTS on the first violation. The two stride-2 conditions were built
per-mode (`--mode av` / `--mode rl-eval`), so they never went through it. This re-checks
EVERY condition independently and collects ALL failures (never aborts), so a truncated
condition can't mask whether the others are clean. It re-uses the exact primitives the
builder/preflight use (`CONDITIONS`, `av_in_columns`, `AR_TARGET_COLUMNS`, `doc_bucket`,
`_col_np`) so "what is correct" stays single-sourced.

It verifies, per condition, the invariants the causal claim depends on:
  * AV row counts equal across conditions (full av_sft, no filter).
  * AV source-row identity (doc_id, src_row_id, response) + shared prompt (k=3 conds).
  * av_in_* slot count + duplicate/distinct structure.
  * FIXED-TARGET byte-identity: rl_{dev,test}_<cond> AR targets == local's.   <-- the big one
  * rl source-row identity vs local; slot pour (input layer == its target col where they overlap).
  * rl bucket re-derivation (every doc hashes to the right split bucket).
  * dev/test, AR train/dev/test, and av/ar/rl corpus document disjointness.

READ-ONLY. Heavy activation comparisons on the small rl_* tables are done in FULL; the large
av_* slot-structure comparison is SAMPLED (first --av-sample rows) to bound memory — a slot
mixup would differ in the sample with overwhelming probability.

  python -m multilayer_nla.verify_sweep_integrity --sweep-dir $SWEEP \
      --rl-split-manifest $SWEEP/rl_split_manifest.json --ar-split-manifest $SWEEP/ar_split_manifest.json
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from multilayer_nla.build_sweep import CONDITIONS, _bucket_idx, _col_np, _read_split
from multilayer_nla.datasets import AR_TARGET_COLUMNS, av_in_columns, doc_bucket

BASE = "local"  # reference condition every other is compared against
TGT_OF = {23: AR_TARGET_COLUMNS[0], 24: AR_TARGET_COLUMNS[1], 25: AR_TARGET_COLUMNS[2]}


def _exists(p):
    return Path(p).exists()


def _rows(path):
    return pq.ParquetFile(path).metadata.num_rows


def _plain(path, cols):
    t = pq.read_table(path, columns=cols)
    return {c: t.column(c).to_pylist() for c in cols}


def _head_np(path, name, n):
    """First n rows of an activation column as (rows, d) float32 — bounds memory for av_*."""
    for b in pq.ParquetFile(path).iter_batches(batch_size=n, columns=[name]):
        t = pa.Table.from_batches([b])
        return (t.column(name).combine_chunks().flatten()
                .to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(t), -1))
    return np.zeros((0, 0), np.float32)


def verify(sweep_dir, rl_manifest=None, ar_manifest=None, av_sample=2048):
    sweep = Path(sweep_dir)
    conds = list(CONDITIONS)
    R = []
    def rec(cat, name, ok, detail=""):
        R.append((cat, name, bool(ok), detail))

    # ---- AV row counts (must all equal the full av_sft) ----
    av_counts = {c: _rows(sweep / f"av_{c}.parquet") for c in conds if _exists(sweep / f"av_{c}.parquet")}
    mode = None
    if av_counts:
        mode = max(set(av_counts.values()), key=list(av_counts.values()).count)
        for c, n in av_counts.items():
            rec("AV", f"row_count[{c}]", n == mode, f"{n:,}" + ("" if n == mode else f" (expected {mode:,})"))

    # ---- AV source-row identity + shared prompt ----
    bp = sweep / f"av_{BASE}.parquet"
    if _exists(bp):
        bk = _plain(bp, ["doc_id", "src_row_id", "response"])
        bprompt = _plain(bp, ["prompt"])["prompt"][0]
        for c in conds:
            p = sweep / f"av_{c}.parquet"
            if not _exists(p):
                continue
            k = _plain(p, ["doc_id", "src_row_id", "response"])
            same = (k == bk)
            rec("AV", f"src_identity[{c}]", same,
                "" if same else f"differs from {BASE} (rows {len(k['doc_id']):,} vs {len(bk['doc_id']):,})")
            if len(CONDITIONS[c]) == 3:   # single (k=1) has a different prompt by design
                pr = _plain(p, ["prompt"])["prompt"][0]
                rec("AV", f"prompt[{c}]", pr == bprompt, "" if pr == bprompt else "AV prompt differs from local")

    # ---- AV slot count + duplicate/distinct structure (sampled) ----
    for c in conds:
        p = sweep / f"av_{c}.parquet"
        if not _exists(p):
            continue
        layers = CONDITIONS[c]
        names = pq.ParquetFile(p).schema_arrow.names
        slot_cols = [n for n in names if n.startswith("av_in_")]
        rec("AV", f"slot_count[{c}]", len(slot_cols) == len(layers),
            f"{len(slot_cols)} cols, expect {len(layers)}")
        if len(slot_cols) == len(layers) and len(layers) > 1:
            slots = {i: _head_np(p, f"av_in_{i}", av_sample) for i in range(len(layers))}
            ok, det = True, []
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    eq = np.array_equal(slots[i], slots[j])
                    if eq != (layers[i] == layers[j]):
                        ok = False
                        det.append(f"({i},{j})=L{layers[i]}/L{layers[j]} eq={eq}")
            rec("AV", f"slot_struct[{c}]", ok, (";".join(det) or f"ok (sampled {av_sample})"))

    # ---- RL: fixed target, src identity, slot pour, bucket ----
    if rl_manifest and _exists(rl_manifest):
        rl_seed, rl_fracs, rl_names, _ = _read_split(rl_manifest)
    else:
        rl_seed, rl_fracs, rl_names = 42, (0.8, 0.1, 0.1), ("train", "dev", "test")
    for bucket in ("dev", "test"):
        try:
            bidx = _bucket_idx(rl_names, bucket)
        except AssertionError:
            continue
        basep = sweep / f"rl_{bucket}_{BASE}.parquet"
        if not _exists(basep):
            continue
        btgt = {tc: _col_np(basep, tc) for tc in AR_TARGET_COLUMNS}
        brows = _plain(basep, ["doc_id", "src_row_id"])
        for c in conds:
            p = sweep / f"rl_{bucket}_{c}.parquet"
            if not _exists(p):
                continue
            # FIXED-TARGET byte-identity (the invariant the whole comparison depends on)
            try:
                same_t = all(np.array_equal(_col_np(p, tc), btgt[tc]) for tc in AR_TARGET_COLUMNS)
                det = "" if same_t else "AR target NOT byte-identical to local — FIXED-TARGET VIOLATION"
            except Exception as e:
                same_t, det = False, f"error reading targets: {e}"
            rec("RL_target", f"fixed_target[{bucket}/{c}]", same_t, det)
            # source-row identity
            rows = _plain(p, ["doc_id", "src_row_id"])
            rec("RL_target", f"src_identity[{bucket}/{c}]", rows == brows,
                "" if rows == brows else "source rows differ from local")
            # slot pour: input layer that IS a target layer must byte-equal that target column
            layers, slots = CONDITIONS[c], av_in_columns(len(CONDITIONS[c]))
            pour_ok, pdet = True, []
            for i, L in enumerate(layers):
                if L in TGT_OF:
                    if not np.array_equal(_col_np(p, slots[i]), _col_np(p, TGT_OF[L])):
                        pour_ok = False
                        pdet.append(f"av_in_{i}!=L{L}_target")
            rec("RL_target", f"slot_pour[{bucket}/{c}]", pour_ok, (";".join(pdet) or "ok"))
            # bucket re-derivation
            wrong = sum(1 for d in set(rows["doc_id"]) if doc_bucket(d, rl_fracs, rl_seed) != bidx)
            rec("RL_split", f"bucket[{bucket}/{c}]", wrong == 0, f"{wrong} docs not in bucket {bidx}")

    # ---- split / corpus disjointness ----
    pdev, ptest = sweep / f"rl_dev_{BASE}.parquet", sweep / f"rl_test_{BASE}.parquet"
    if _exists(pdev) and _exists(ptest):
        dd = set(_plain(pdev, ["doc_id"])["doc_id"])
        td = set(_plain(ptest, ["doc_id"])["doc_id"])
        rec("Split", "rl_dev∩rl_test", not (dd & td), f"{len(dd & td)} shared docs")
    ar = {}
    for b in ("common", "dev", "test"):
        p = sweep / f"ar_{b}.parquet"
        if _exists(p):
            ar[b] = set(_plain(p, ["doc_id"])["doc_id"])
    akeys = list(ar)
    for i in range(len(akeys)):
        for j in range(i + 1, len(akeys)):
            inter = ar[akeys[i]] & ar[akeys[j]]
            rec("Split", f"ar_{akeys[i]}∩{akeys[j]}", not inter, f"{len(inter)} shared docs")

    def _docs(name):
        p = sweep / name
        return set(_plain(p, ["doc_id"])["doc_id"]) if _exists(p) else set()
    corp = {"av": _docs("av_local.parquet"),
            "ar": _docs("ar_common.parquet") | _docs("ar_dev.parquet") | _docs("ar_test.parquet"),
            "rl": _docs("rl_dev_local.parquet") | _docs("rl_test_local.parquet")}
    ckeys = [k for k in corp if corp[k]]
    for i in range(len(ckeys)):
        for j in range(i + 1, len(ckeys)):
            inter = corp[ckeys[i]] & corp[ckeys[j]]
            rec("Split", f"corpora_{ckeys[i]}∩{ckeys[j]}", not inter, f"{len(inter)} shared docs")

    return R


def diagnose_av(sweep_dir, cond, ref=BASE, av_sample=2048):
    """Focused diagnosis of ONE av_<cond> parquet — how (and how badly) is it truncated?

    The decisive question for a short condition: is it a contiguous FRONT-SLICE of the
    av_sft stream (src_row_id 0..n-1) → a biased *document* subset (only the docs in the
    first shards), or does it still cover the full document set with fewer rows? A
    front-slice means the trained AV saw a non-representative doc mix → must be rebuilt.
    """
    sweep = Path(sweep_dir)
    p, rp = sweep / f"av_{cond}.parquet", sweep / f"av_{ref}.parquet"
    out = [f"## av_{cond} diagnosis (ref = av_{ref})"]
    if not _exists(p):
        return "\n".join(out + [f"- av_{cond}.parquet MISSING"])
    d = _plain(p, ["doc_id", "src_row_id", "response"])
    n = len(d["doc_id"])
    sid = d["src_row_id"]
    smin, smax = min(sid), max(sid)
    contiguous = (len(set(sid)) == n and smin == 0 and smax == n - 1)
    out.append(f"- rows: {n:,}")
    out.append(f"- src_row_id: min={smin:,} max={smax:,}  → "
               + ("CONTIGUOUS front-slice (rows 0..n-1) — build read only the first shards"
                  if contiguous else
                  "NOT a clean 0..n-1 prefix — investigate (gaps/overlap in the ordinal stream)"))
    cond_docs = set(d["doc_id"])
    if _exists(rp):
        ref_docs = set(_plain(rp, ["doc_id"])["doc_id"])
        ref_n = _rows(rp)
        miss = ref_docs - cond_docs
        cov = len(cond_docs) / max(len(ref_docs), 1)
        out.append(f"- rows vs ref: {n:,} / {ref_n:,}  ({n/max(ref_n,1):.1%})")
        out.append(f"- unique docs: {len(cond_docs):,} / {len(ref_docs):,}  (coverage {cov:.1%})")
        out.append(f"- docs in ref but MISSING here: {len(miss):,}")
        if cov < 0.95:
            out.append(f"  ⚠ DOC-BIASED subset: this AV never saw {len(miss):,} documents the others "
                       f"trained on → its verbalizer is confounded. REBUILD.")
        else:
            out.append("  ~all docs covered (fewer rows each) — bias is milder, but rebuild for a clean count.")
    # response validity + slot layer distinctness on the rows it has
    bad_resp = sum(1 for r in d["response"] if not r or "<explanation>" not in r)
    out.append(f"- responses empty/unwrapped: {bad_resp:,}  ({'ok' if bad_resp == 0 else '⚠ INVALID LABELS'})")
    layers = CONDITIONS.get(cond, [])
    names = pq.ParquetFile(p).schema_arrow.names
    slot_cols = [nm for nm in names if nm.startswith("av_in_")]
    out.append(f"- av_in_* columns: {len(slot_cols)} (expect {len(layers)} for layers {layers})")
    if len(slot_cols) == len(layers) and len(layers) > 1:
        slots = {i: _head_np(p, f"av_in_{i}", av_sample) for i in range(len(layers))}
        bad = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                eq = np.array_equal(slots[i], slots[j])
                if eq != (layers[i] == layers[j]):
                    bad.append(f"({i},{j}) L{layers[i]}/L{layers[j]} eq={eq}")
        out.append(f"- slot structure (sampled {av_sample}): "
                   + ("ok — distinct layers are distinct" if not bad else "⚠ " + ";".join(bad)))
    return "\n".join(out)


def _print(R):
    cats = []
    for cat, *_ in R:
        if cat not in cats:
            cats.append(cat)
    n_fail = 0
    for cat in cats:
        print(f"\n### {cat}")
        for c, name, ok, detail in [r for r in R if r[0] == cat]:
            mark = "✓" if ok else "✗ FAIL"
            n_fail += (0 if ok else 1)
            print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    total = len(R)
    print(f"\n===== {total - n_fail}/{total} checks passed; {n_fail} FAILED =====")
    if n_fail:
        print("INTEGRITY NOT CLEAN — see the ✗ lines above. The affected condition(s) are "
              "confounded and must be rebuilt before their numbers can be trusted.")
    else:
        print("ALL INTEGRITY CHECKS PASSED — datasets honour the fixed-target / vary-only-input design.")
    return n_fail


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", required=True, help="dir with av_<cond>/rl_*_<cond>/ar_* parquets")
    p.add_argument("--rl-split-manifest", help="rl split manifest (for bucket re-derivation)")
    p.add_argument("--ar-split-manifest", help="ar split manifest (unused today; accepted for symmetry)")
    p.add_argument("--av-sample", type=int, default=2048, help="rows sampled for av slot-structure check")
    args = p.parse_args()
    R = verify(args.sweep_dir, args.rl_split_manifest, args.ar_split_manifest, args.av_sample)
    raise SystemExit(_print(R))


if __name__ == "__main__":
    main()

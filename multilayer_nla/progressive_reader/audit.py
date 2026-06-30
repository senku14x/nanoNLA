"""Progressive Reader v0 — data preflight audit (spec §3). THE GATE.

The make-or-break question: do the gold teacher explanations actually reach 128 tokens
often enough for the 64/128 conditions to be non-degenerate? These are API summaries, so
`coverage_at_budget[128]` could be low — if it is, the strict-prefix v0 guts the data and
we rethink BEFORE building train/eval. Run this first.

Reads an existing bank corpus (av_sft / ar_sft / rl shards: per-row teacher label +
activation_L{k} + doc_id), tokenizes each explanation ONCE with the real tokenizer (the
canonical teacher-token-ID source, spec §1.1), and reports length coverage + the
document-level split + target-layer presence. No model, no GPU (tokenizer only).

  python -m multilayer_nla.progressive_reader.audit \
    --data "$REGEN/av_sft.shard*of*.parquet" --base-ckpt Qwen/Qwen3-8B \
    --out runs/progressive_reader_v0/data_audit.json
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
from pathlib import Path

from multilayer_nla.progressive_reader.schedule import PREFIX_BUDGETS, TARGET_LAYERS

# Teacher-text parsing — reuse the repo's canonical wrappers (lazy import of nla.schema).
_AR_PRE = "Summary of the following text: <text>"   # ar_sft `prompt` critic template
_AR_SUF = "</text> <summary>"


def _explanation_from_prompt(prompt: str) -> str | None:
    if isinstance(prompt, str) and prompt.startswith(_AR_PRE) and prompt.endswith(_AR_SUF):
        return prompt[len(_AR_PRE):len(prompt) - len(_AR_SUF)]
    return None


def detect_teacher_field(schema_names: list[str]) -> str:
    """av_sft ships `response` (<explanation>…</explanation>); ar_sft ships `prompt`
    (critic template). Prefer response."""
    if "response" in schema_names:
        return "response"
    if "prompt" in schema_names:
        return "prompt"
    raise SystemExit(f"no teacher field: need 'response' or 'prompt' in {schema_names[:10]}")


def _percentiles(vals, ps=(0, 10, 25, 50, 75, 90, 100)) -> dict:
    import numpy as np
    a = np.asarray(vals)
    return {f"p{p:02d}": (int(np.percentile(a, p)) if len(a) else 0) for p in ps}


def run(data_glob: str, base_ckpt: str, *, fracs=(0.8, 0.1, 0.1), seed: int = 42,
        max_rows: int | None = None, tok_batch: int = 1024) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from nla.schema import extract_explanation
    from multilayer_nla.datasets import doc_bucket  # doc-level split, stable hash

    paths = sorted(_glob.glob(data_glob)) or [data_glob]
    assert Path(paths[0]).exists(), f"no parquet matched {data_glob!r}"
    names = pq.ParquetFile(paths[0]).schema_arrow.names
    teacher_field = detect_teacher_field(names)
    present_targets = [l for l in TARGET_LAYERS if f"activation_L{l}" in names]
    missing = [l for l in TARGET_LAYERS if l not in present_targets]
    dim_col = f"activation_L{present_targets[0]}" if present_targets else None

    tok = AutoTokenizer.from_pretrained(base_ckpt)

    explanations: list[str] = []
    doc_ids: list[str] = []
    activation_dim = 0
    n_unparsed = 0
    cols = [teacher_field, "doc_id"] + ([dim_col] if dim_col else [])
    n = 0
    for fp in paths:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            if max_rows is not None and n >= max_rows:
                break
            t = pf.read_row_group(rg, columns=cols)
            take = t.num_rows if max_rows is None else min(max_rows - n, t.num_rows)
            t = t.slice(0, take)
            raw = t.column(teacher_field).to_pylist()
            dids = t.column("doc_id").to_pylist()
            if activation_dim == 0 and dim_col is not None:
                c0 = t.column(dim_col).combine_chunks()
                activation_dim = int(len(c0.flatten()) // max(len(c0), 1))
            for r, d in zip(raw, dids):
                expl = extract_explanation(r) if teacher_field == "response" else _explanation_from_prompt(r)
                if not expl or not expl.strip():
                    n_unparsed += 1
                    continue
                explanations.append(expl)
                doc_ids.append(d)
            n += take
        if max_rows is not None and n >= max_rows:
            break

    # Tokenize ONCE (add_special_tokens=False) — the canonical teacher-token-ID source.
    lengths = []
    for i in range(0, len(explanations), tok_batch):
        enc = tok(explanations[i:i + tok_batch], add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in enc)
    lengths = np.asarray(lengths, dtype=np.int64)

    # Document-level split (stable hash, seed 42) — count rows + docs per bucket.
    names3 = ("train", "dev", "test")
    buckets = np.fromiter((doc_bucket(d, fracs, seed) for d in doc_ids), dtype=np.int64, count=len(doc_ids))
    split_docs = {nm: set() for nm in names3}
    split_rows = {nm: 0 for nm in names3}
    strict_rows = {nm: 0 for nm in names3}
    strict_docs = {nm: set() for nm in names3}
    full128 = lengths >= 128
    for d, b, ok in zip(doc_ids, buckets, full128):
        nm = names3[b]
        split_rows[nm] += 1
        split_docs[nm].add(d)
        if ok:
            strict_rows[nm] += 1
            strict_docs[nm].add(d)

    def overlap(a, b):
        return len(split_docs[a] & split_docs[b])

    res = {
        "data_glob": data_glob, "n_files": len(paths), "teacher_text_field": teacher_field,
        "teacher_token_ids_present": False,  # derived by tokenizing the text (canonical, §1.1)
        "rows_total": int(len(doc_ids)), "rows_unparsed_dropped": int(n_unparsed),
        "documents_total": int(len({*doc_ids})),
        "train_rows": split_rows["train"], "dev_rows": split_rows["dev"], "test_rows": split_rows["test"],
        "train_documents": len(split_docs["train"]), "dev_documents": len(split_docs["dev"]),
        "test_documents": len(split_docs["test"]),
        "teacher_length_quantiles": _percentiles(lengths),
        "teacher_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "coverage_at_budget": {str(b): float((lengths >= b).mean()) if len(lengths) else 0.0
                               for b in PREFIX_BUDGETS},
        "target_layers_present": present_targets, "target_layers_missing": missing,
        "activation_dim": activation_dim,
        "document_overlap_train_dev": overlap("train", "dev"),
        "document_overlap_train_test": overlap("train", "test"),
        "document_overlap_dev_test": overlap("dev", "test"),
        # strict-prefix (n>=128) effect, per split (spec §3): how much data the headline keeps.
        "strict_128": {
            "train_rows": strict_rows["train"], "dev_rows": strict_rows["dev"],
            "test_rows": strict_rows["test"],
            "train_documents": len(strict_docs["train"]), "dev_documents": len(strict_docs["dev"]),
            "test_documents": len(strict_docs["test"]),
            "retained_row_frac": float(full128.mean()) if len(full128) else 0.0,
        },
        "split": {"fracs": list(fracs), "seed": seed, "names": list(names3)},
    }
    return res


def _verdict(res: dict) -> None:
    cov = res["coverage_at_budget"]
    q = res["teacher_length_quantiles"]
    print("=" * 72)
    print(f"PROGRESSIVE READER — DATA AUDIT  (field='{res['teacher_text_field']}', d={res['activation_dim']})")
    print(f"  rows {res['rows_total']} ({res['rows_unparsed_dropped']} unparsed dropped) · "
          f"docs {res['documents_total']} · split train/dev/test rows "
          f"{res['train_rows']}/{res['dev_rows']}/{res['test_rows']}")
    print(f"  teacher tokens: p10/p50/p90/p100 = {q['p10']}/{q['p50']}/{q['p90']}/{q['p100']}  "
          f"mean {res['teacher_length_mean']:.0f}")
    print(f"  coverage @ 32/64/128 tokens: {cov['32']*100:.1f}% / {cov['64']*100:.1f}% / {cov['128']*100:.1f}%")
    s = res["strict_128"]
    print(f"  STRICT n>=128 keeps {s['retained_row_frac']*100:.1f}% of rows "
          f"-> train/dev/test = {s['train_rows']}/{s['dev_rows']}/{s['test_rows']} rows "
          f"({s['test_documents']} test docs)")
    miss = res["target_layers_missing"]
    print(f"  target layers present: {res['target_layers_present']}"
          + (f"  ** MISSING {miss} (need GPU re-extraction) **" if miss else "  (all 7 present)"))
    ov = res["document_overlap_train_dev"] + res["document_overlap_train_test"] + res["document_overlap_dev_test"]
    print(f"  doc-split overlaps (must be 0): {ov}")
    print("-" * 72)
    if cov["128"] < 0.5:
        print(f"  GATE: coverage@128 = {cov['128']*100:.1f}% < 50% — strict-prefix v0 keeps "
              f"only {s['retained_row_frac']*100:.0f}% of rows.")
        print("  -> Decide BEFORE building train/eval: (a) lower the max budget to a covered")
        print("     value, (b) accept the smaller strict set if test docs are still plentiful,")
        print("     or (c) use require_full_prefix_for_max_budget=false (censored — NOT headline).")
    else:
        print(f"  GATE: coverage@128 = {cov['128']*100:.1f}% >= 50% — strict-prefix v0 is viable.")
    if miss:
        print(f"  GATE: target layers {miss} absent — re-probe within the bank window or re-extract.")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="bank corpus parquet/glob (av_sft / ar_sft / rl shards)")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B", help="tokenizer for the teacher text")
    p.add_argument("--out", required=True, help="write data_audit.json here")
    p.add_argument("--fracs", default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None, help="cap rows (quick estimate)")
    args = p.parse_args()
    fracs = tuple(float(x) for x in args.fracs.split(","))
    res = run(args.data, args.base_ckpt, fracs=fracs, seed=args.seed, max_rows=args.max_rows)
    _verdict(res)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[audit] -> {args.out}")


if __name__ == "__main__":
    main()

"""Regenerate a window of layer activations for a published NLA parquet row.

Multi-layer analog of `tools/regenerate_activations.py`. The published warmstart
dataset (ceselder/qwen3-8b-nla-L24-finefineweb-100k, over **m-a-p/FineFineWeb**)
already contains the real LABELS — av_sft `response`, ar_sft `prompt`, rl
`prompt`, the exact `detokenized_text_truncated` prefix, `n_raw_tokens`, and
`doc_id`. It intentionally OMITS only the raw layer-24 vector, because the
activation is a deterministic function of (exact prefix, layer):

    detokenized_text_truncated  --(one forward, hooks at {save-layers})-->  {a^(k) : k in window}

The label only ever depended on the text (stage2 feeds the API model just the
prefix; it never sees the activation), so the published explanation is exactly
as valid for our three-layer patch as for the original single layer-l vector.
We therefore REGENERATE the activations locally — NO API, NO paid labeling, just
H200 forward-pass time — and inherit the labels for free.

ONE forward computes EVERY layer's hidden state, so capturing a wider window
(`--save-layers`, default the center triplet) is FREE on compute — only storage
grows (≈16 KB/row/layer at d=4096 fp32). Saving e.g. 19-29 future-proofs the §5
center-selection sweep: re-probing any center in [20..28] becomes a re-slice in
build_from_published (--center), never a re-extraction. Layers are written as a
uniform `activation_L{k}` archive (build_from_published selects the {c-1,c,c+1}
triplet → activation_prev/centre/next for the trainers).

⚠️  CORPUS: this consumes the PUBLISHED rows' stored prefixes directly. Do NOT
re-sample positions from a corpus, and do NOT point this at our 30k
`HuggingFaceFW/fineweb` extraction — that is a DIFFERENT corpus from the labeled
`m-a-p/FineFineWeb` data and its positions/texts do not line up with the
published AV/AR/RL rows. The 30k fineweb run stays a plumbing/extraction check
only. Use `extract_multilayer.py` for fresh corpus sampling; use THIS for the
labeled run.

For each published row we forward `detokenized_text_truncated` and read the
windowed hidden states at the FINAL token — the original extraction position by
construction, since stage 0 truncated the prefix to end exactly there. Output =
the input table with `activation_L{k}` (RAW, norm="none" — the invariant) +
`center_layer` appended; every published column (`prompt`, `response`, `doc_id`,
`n_raw_tokens`, ...) carried through UNTOUCHED. The single→three-marker actor
prompt swap happens later, in build_from_published.py.

Usage:
    python -m multilayer_nla.regenerate_multilayer_activations \\
        --in  av_sft.parquet  --out av_sft.mlnla.parquet \\
        --base-model Qwen/Qwen3-8B --center-layer 24 --save-layers 19-29 --max-length 4096
"""

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.arch_adapters import resolve_decoder_layers
from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs

TEXT_COL = "detokenized_text_truncated"


def layer_col(k: int) -> str:
    """Uniform archive column name for layer k's final-token vector."""
    return f"activation_L{k}"


def parse_layers(spec: str) -> list[int]:
    """Parse a layer spec like "19-29" or "19,20,24" or "19-21,25,27-29" -> sorted ints."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    assert out, f"empty/invalid --save-layers spec: {spec!r}"
    return sorted(out)


def _final_vec(h) -> np.ndarray:
    """Final-token vector [d] from a per-text hidden capture.

    Accepts either the [seq_len, d] full capture (take last row) or the already
    final-token [d] capture (extract_multi(..., final_token_only=True)).
    """
    v = np.asarray(h, dtype=np.float32)
    return v[-1] if v.ndim == 2 else v


def append_layer_columns(table: pa.Table, results: list[dict], save_layers: list[int],
                         d_model: int, *, check_roundtrip: bool = True,
                         row_offset: int = 0, max_drop_frac: float = 0.0) -> pa.Table:
    """Append `activation_L{k}` (RAW, final token) for each k in `save_layers`.

    `results[i]` aligns to `table` row i and is `{token_ids: list[int],
    hidden: {layer_index -> vec}}` where `vec` is [seq_len, d] or the final-token
    [d] (MultiLayerHFExtractor.extract_multi, optionally final_token_only=True).

    Round-trip guard (when `check_roundtrip` and `n_raw_tokens` present): the
    re-encoded prefix must reproduce the stored token count, or the final token
    is NOT the position the label describes — the activation would land on the
    wrong token and is unusable. `max_drop_frac` controls the response: 0.0
    (default) hard-fails on ANY mismatch (the single-layer regenerate tool's
    behavior, and the right default — a systematic mismatch is almost always a
    --max-length misconfiguration, which silent dropping would mask). A small
    positive value (e.g. 1e-3) tolerates rare per-row tokenizer drift on a large
    run by DROPPING + logging the offending rows, failing only if the drop
    fraction exceeds the threshold.

    Pure (numpy/pyarrow only, no model) — unit-testable offline with fabricated
    `results`.
    """
    assert len(results) == table.num_rows, (
        f"results ({len(results)}) and table rows ({table.num_rows}) misaligned"
    )

    if check_roundtrip and "n_raw_tokens" in table.schema.names:
        n_raw = table.column("n_raw_tokens").to_pylist()
        bad = [(row_offset + i, len(r["token_ids"]), n)
               for i, (r, n) in enumerate(zip(results, n_raw))
               if len(r["token_ids"]) != n]
        if bad:
            frac = len(bad) / len(results)
            if frac > max_drop_frac:
                raise AssertionError(
                    f"{len(bad)} rows fail the tokenization round-trip "
                    f"(first: row {bad[0][0]} re-encoded to {bad[0][1]} tokens, "
                    f"stage-0 had {bad[0][2]}). Drop fraction {frac:.4%} exceeds "
                    f"--max-drop-frac {max_drop_frac:.4%}. These rows are not "
                    f"faithfully regenerable — check --max-length matches the "
                    f"original extraction (4096 for the published datasets)."
                )
            bad_local = {i for i, (r, n) in enumerate(zip(results, n_raw))
                         if len(r["token_ids"]) != n}
            good = [i for i in range(len(results)) if i not in bad_local]
            table = table.take(good)
            results = [results[i] for i in good]
            print(f"[regen] dropped {len(bad)} ({frac:.4%}) round-trip-mismatch rows "
                  f"(<= --max-drop-frac {max_drop_frac:.4%}); kept {len(good)}", flush=True)

    def _final_token_fsl(layer_index: int) -> pa.Array:
        # [n_rows, d] float32 — the final-token vector for this layer per row.
        mat = np.stack([_final_vec(r["hidden"][layer_index]) for r in results])
        assert mat.shape == (table.num_rows, d_model), (
            f"layer {layer_index}: built {mat.shape}, expected "
            f"({table.num_rows}, {d_model}) — wrong d_model or empty seq?"
        )
        flat = mat.reshape(-1).astype(np.float32, copy=False)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat), d_model)

    out = table
    for li in save_layers:
        name = layer_col(li)
        assert name not in out.schema.names, (
            f"input already has {name!r} — nothing to regenerate"
        )
        out = out.append_column(name, _final_token_fsl(li))
    return out


def _infer_center(pf: pq.ParquetFile, explicit: int | None) -> int:
    """center-layer from CLI, else the (constant) center_layer/activation_layer column."""
    if explicit is not None:
        return explicit
    names = pf.schema_arrow.names
    for col in ("center_layer", "activation_layer"):
        if col in names:
            vals = set(pf.read(columns=[col]).column(col).to_pylist())
            assert len(vals) == 1, (
                f"{col} is not constant across rows ({sorted(vals)[:5]}...) — "
                f"pass --center-layer explicitly."
            )
            return int(vals.pop())
    raise SystemExit(
        "no --center-layer given and neither center_layer nor activation_layer "
        "column present — cannot infer the layer triplet."
    )


def _shard_out(out: str, idx: int, n: int) -> str:
    """Per-shard output path so parallel jobs never clobber one --out (merge after)."""
    if n <= 1:
        return out
    from pathlib import Path
    p = Path(out)
    return str(p.with_name(f"{p.stem}.shard{idx:02d}of{n:02d}{p.suffix}"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", required=True, help="published slim parquet (no activation triplet)")
    p.add_argument("--out", required=True, help="output parquet (triplet appended, labels preserved)")
    p.add_argument("--base-model", required=True, help="HF base model, e.g. Qwen/Qwen3-8B")
    p.add_argument("--center-layer", type=int, default=None,
                   help="center block l (provenance + default window). Default: read from the "
                        "center_layer/activation_layer column (24 for the published L24 set).")
    p.add_argument("--save-layers", default=None,
                   help="layers to capture, e.g. '19-29' or '19,24,29' or '19-21,25'. "
                        "Default: the center triplet {l-1, l, l+1}. Wider windows are FREE on "
                        "compute (one forward) and future-proof the §5 center sweep; cost is "
                        "storage (~16 KB/row/layer at d=4096 fp32). Written as activation_L{k}.")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per write (bounds memory)")
    p.add_argument("--batch-size", type=int, default=16, help="model forward batch size")
    p.add_argument("--max-length", type=int, default=4096,
                   help="extractor context cap — MUST match the original stage-0 extraction "
                        "(4096 for the published datasets). Too small right-truncates long rows "
                        "and silently regenerates at the wrong position; the n_raw_tokens "
                        "round-trip check turns that into a hard error.")
    p.add_argument("--no-roundtrip-check", action="store_true",
                   help="disable the n_raw_tokens round-trip guard (NOT recommended)")
    p.add_argument("--max-drop-frac", type=float, default=0.0,
                   help="tolerate up to this fraction of round-trip-mismatch rows by DROPPING "
                        "(and logging) them; default 0.0 hard-fails on any mismatch. Use a tiny "
                        "value (e.g. 1e-3) for a large run where rare per-row tokenizer drift "
                        "should not abort the whole shard.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="data-parallel fan-out: split the input into N contiguous shards (one "
                        "GPU/job each). Each shard writes its own parquet; merge after. The 30k "
                        "stage0 run flagged exactly this for the 100k corpus.")
    p.add_argument("--shard-index", type=int, default=0,
                   help="which shard [0, num-shards) this job processes")
    p.add_argument("--extractor-cls",
                   default="multilayer_nla.extract_multilayer.MultiLayerHFExtractor")
    p.add_argument("--extractor-kwargs", default=None, help="JSON dict of extra extractor kwargs")
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)
    pf = pq.ParquetFile(storage.open_read(args.inp))
    names = pf.schema_arrow.names
    assert TEXT_COL in names, (
        f"input lacks {TEXT_COL!r} — cannot regenerate without the source prefix"
    )

    center = _infer_center(pf, args.center_layer)
    assert center - 1 >= 0, f"center-layer={center} has no l-1 block"
    save_layers = parse_layers(args.save_layers) if args.save_layers else [center - 1, center, center + 1]
    assert min(save_layers) >= 0, f"--save-layers has a negative layer: {save_layers}"
    triplet = {center - 1, center, center + 1}
    if not triplet.issubset(save_layers):
        print(f"WARNING: --save-layers {save_layers} omits part of the center {center} triplet "
              f"{sorted(triplet)} — build_from_published --center {center} will fail until you "
              f"either widen the window or build at a center whose triplet IS saved.", flush=True)
    for li in save_layers:
        assert layer_col(li) not in names, f"input already has {layer_col(li)!r} — nothing to regenerate"

    user_kwargs = parse_kwargs(args.extractor_kwargs)
    assert "model_name" not in user_kwargs, "pass --base-model, not model_name in --extractor-kwargs"
    extractor = load_class(args.extractor_cls)(
        model_name=args.base_model, batch_size=args.batch_size,
        max_length=args.max_length, **user_kwargs,
    )
    assert hasattr(extractor, "extract_multi"), (
        f"{args.extractor_cls} has no extract_multi(); multi-layer regeneration needs it"
    )
    d_model = extractor.d_model
    n_layers = len(resolve_decoder_layers(extractor.model))
    assert max(save_layers) < n_layers, (
        f"--save-layers max {max(save_layers)} out of range for a {n_layers}-block model"
    )

    assert args.num_shards >= 1 and 0 <= args.shard_index < args.num_shards, (
        f"bad shard config: shard_index={args.shard_index}, num_shards={args.num_shards}"
    )
    total = pf.metadata.num_rows
    per = (total + args.num_shards - 1) // args.num_shards
    lo = args.shard_index * per
    hi = min(total, lo + per)
    out_path = _shard_out(args.out, args.shard_index, args.num_shards)
    if args.num_shards > 1:
        print(f"[shard {args.shard_index}/{args.num_shards}] input rows [{lo}, {hi}) of {total} "
              f"-> {out_path}", flush=True)

    storage.ensure_parent(out_path)
    writer = None
    g = 0            # global INPUT row offset (for sharding + round-trip messages)
    n_out = 0        # output rows actually written (post drop)
    try:
        for batch in pf.iter_batches(batch_size=args.chunk_size):
            blen = batch.num_rows
            o_lo, o_hi = max(lo, g), min(hi, g + blen)
            if o_lo < o_hi:  # this batch overlaps the shard's row range
                tbl = pa.Table.from_batches([batch]).slice(o_lo - g, o_hi - o_lo)
                texts = tbl.column(TEXT_COL).to_pylist()
                results = extractor.extract_multi(texts, save_layers, final_token_only=True)
                tbl = append_layer_columns(
                    tbl, results, save_layers, d_model,
                    check_roundtrip=not args.no_roundtrip_check,
                    row_offset=o_lo, max_drop_frac=args.max_drop_frac,
                )
                # center_layer pins provenance the same way extract_multilayer does.
                if "center_layer" not in tbl.schema.names:
                    tbl = tbl.append_column("center_layer",
                                            pa.array([center] * tbl.num_rows, pa.int64()))
                if writer is None:
                    writer = pq.ParquetWriter(storage.open_write(out_path), tbl.schema)
                writer.write_table(tbl)
                n_out += tbl.num_rows
                print(f"  {n_out} rows (input @ {o_hi}/{hi})", flush=True)
            g += blen
            if g >= hi:
                break
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {n_out} rows, layers {save_layers} (center {center}, d={d_model}) -> {out_path}")
    if args.num_shards > 1:
        print(f"[shard {args.shard_index}/{args.num_shards}] done. Merge all shards with e.g.:\n"
              f"  python -c \"import pyarrow.parquet as pq, glob; "
              f"pq.write_table(pq.ParquetDataset(sorted(glob.glob('PREFIX.shard*of*.parquet'))).read(), "
              f"'MERGED.parquet', row_group_size=4096)\"")


if __name__ == "__main__":
    main()

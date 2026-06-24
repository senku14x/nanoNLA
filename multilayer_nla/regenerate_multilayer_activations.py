"""Regenerate the three-layer activation triplet for a published NLA parquet row.

Multi-layer analog of `tools/regenerate_activations.py`. The published warmstart
dataset (ceselder/qwen3-8b-nla-L24-finefineweb-100k, over **m-a-p/FineFineWeb**)
already contains the real LABELS — av_sft `response`, ar_sft `prompt`, rl
`prompt`, the exact `detokenized_text_truncated` prefix, `n_raw_tokens`, and
`doc_id`. It intentionally OMITS only the raw layer-24 vector, because the
activation is a deterministic function of (exact prefix, layer):

    detokenized_text_truncated  --(one forward, hooks at {l-1, l, l+1})-->  [a^(l-1), a^(l), a^(l+1)]

The label only ever depended on the text (stage2 feeds the API model just the
prefix; it never sees the activation), so the published explanation is exactly
as valid for our three-layer patch as for the original single layer-l vector.
We therefore REGENERATE the activations locally — NO API, NO paid labeling, just
H200 forward-pass time — and inherit the labels for free.

⚠️  CORPUS: this consumes the PUBLISHED rows' stored prefixes directly. Do NOT
re-sample positions from a corpus, and do NOT point this at our 30k
`HuggingFaceFW/fineweb` extraction — that is a DIFFERENT corpus from the labeled
`m-a-p/FineFineWeb` data and its positions/texts do not line up with the
published AV/AR/RL rows. The 30k fineweb run stays a plumbing/extraction check
only. Use `extract_multilayer.py` for fresh corpus sampling; use THIS for the
labeled run.

For each published row we forward `detokenized_text_truncated` through the base
model and read the {l-1, l, l+1} hidden states at the FINAL token — the original
extraction position by construction, since stage 0 truncated the prefix to end
exactly there. We are "just extending the hook set from layer 24 to [23,24,25]".
Output = the input table with `activation_prev/centre/next` (RAW, norm="none" —
the invariant) appended; every published column (`prompt`, `response`, `doc_id`,
`n_raw_tokens`, ...) is carried through UNTOUCHED. The single→three-marker actor
prompt swap happens later, in build_from_published.py.

Usage:
    python -m multilayer_nla.regenerate_multilayer_activations \\
        --in  av_sft.parquet  --out av_sft.mlnla.parquet \\
        --base-model Qwen/Qwen3-8B --center-layer 24 --max-length 4096
"""

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.arch_adapters import resolve_decoder_layers
from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs

# Output column names — IDENTICAL to extract_multilayer._schema so the same
# loaders/trainers consume either source interchangeably.
SLOT_COLUMNS = ("activation_prev", "activation_centre", "activation_next")
TEXT_COL = "detokenized_text_truncated"


def append_triplet_columns(table: pa.Table, results: list[dict], center: int,
                           d_model: int, *, check_roundtrip: bool = True,
                           row_offset: int = 0) -> pa.Table:
    """Append activation_prev/centre/next (RAW, at the final token) to `table`.

    `results[i]` aligns to `table` row i and is `{token_ids: list[int],
    hidden: {layer_index -> array[seq_len, d]}}` (exactly what
    MultiLayerHFExtractor.extract_multi returns; layer indices are the
    output-of-block indices {center-1, center, center+1}). The triplet is read
    at index -1 of each layer's hidden states — the final (extraction) token,
    because stage-0 truncated the prefix to end there.

    Round-trip guard (when `check_roundtrip` and `n_raw_tokens` is present): the
    re-encoded prefix must reproduce the stored token count, or the final token
    is NOT the position the label describes. This is the multi-layer equivalent
    of the single-layer regenerate tool's assertion — a hard failure, never
    silent (the usual cause is --max-length not matching the original 4096).

    Pure (numpy/pyarrow only, no model) so it is unit-testable offline with
    fabricated `results`.
    """
    assert len(results) == table.num_rows, (
        f"results ({len(results)}) and table rows ({table.num_rows}) misaligned"
    )
    layers = (center - 1, center, center + 1)

    if check_roundtrip and "n_raw_tokens" in table.schema.names:
        n_raw = table.column("n_raw_tokens").to_pylist()
        bad = [(row_offset + i, len(r["token_ids"]), n)
               for i, (r, n) in enumerate(zip(results, n_raw))
               if len(r["token_ids"]) != n]
        assert not bad, (
            f"{len(bad)} rows fail the tokenization round-trip "
            f"(first: row {bad[0][0]} re-encoded to {bad[0][1]} tokens, "
            f"stage-0 had {bad[0][2]}). These rows are not faithfully "
            f"regenerable — check --max-length matches the original extraction "
            f"(4096 for the published datasets), or filter the offending rows."
        )

    def _final_token_fsl(layer_index: int) -> pa.Array:
        # [n_rows, d] float32 — the final-token vector for this layer per row.
        mat = np.stack([np.asarray(r["hidden"][layer_index], dtype=np.float32)[-1]
                        for r in results])
        assert mat.shape == (table.num_rows, d_model), (
            f"layer {layer_index}: built {mat.shape}, expected "
            f"({table.num_rows}, {d_model}) — wrong d_model or empty seq?"
        )
        flat = mat.reshape(-1).astype(np.float32, copy=False)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat), d_model)

    out = table
    for name, li in zip(SLOT_COLUMNS, layers):
        assert name not in out.schema.names, (
            f"input already has {name!r} — nothing to regenerate (this parquet "
            f"already carries the triplet)"
        )
        out = out.append_column(name, _final_token_fsl(li))
    # center_layer pins the provenance the same way extract_multilayer does.
    if "center_layer" not in out.schema.names:
        out = out.append_column("center_layer",
                                pa.array([center] * out.num_rows, pa.int64()))
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


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="inp", required=True, help="published slim parquet (no activation triplet)")
    p.add_argument("--out", required=True, help="output parquet (triplet appended, labels preserved)")
    p.add_argument("--base-model", required=True, help="HF base model, e.g. Qwen/Qwen3-8B")
    p.add_argument("--center-layer", type=int, default=None,
                   help="center block l; patch = {l-1, l, l+1}. Default: read from the "
                        "center_layer/activation_layer column (24 for the published L24 set).")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per write (bounds memory)")
    p.add_argument("--batch-size", type=int, default=16, help="model forward batch size")
    p.add_argument("--max-length", type=int, default=4096,
                   help="extractor context cap — MUST match the original stage-0 extraction "
                        "(4096 for the published datasets). Too small right-truncates long rows "
                        "and silently regenerates at the wrong position; the n_raw_tokens "
                        "round-trip check turns that into a hard error.")
    p.add_argument("--no-roundtrip-check", action="store_true",
                   help="disable the n_raw_tokens round-trip guard (NOT recommended)")
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
    for c in SLOT_COLUMNS:
        assert c not in names, f"input already has {c!r} — nothing to regenerate"

    center = _infer_center(pf, args.center_layer)
    layers = [center - 1, center, center + 1]
    assert center - 1 >= 0, f"center-layer={center} has no l-1 block"

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
    assert center + 1 < n_layers, (
        f"center-layer={center} needs block {center + 1}, but model has {n_layers} blocks"
    )

    storage.ensure_parent(args.out)
    writer = None
    done = 0
    try:
        for batch in pf.iter_batches(batch_size=args.chunk_size):
            tbl = pa.Table.from_batches([batch])
            texts = tbl.column(TEXT_COL).to_pylist()
            results = extractor.extract_multi(texts, layers)
            tbl = append_triplet_columns(
                tbl, results, center, d_model,
                check_roundtrip=not args.no_roundtrip_check, row_offset=done,
            )
            if writer is None:
                writer = pq.ParquetWriter(storage.open_write(args.out), tbl.schema)
            writer.write_table(tbl)
            done += tbl.num_rows
            print(f"  {done} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {done} rows with triplet {layers} (center {center}, d={d_model}) -> {args.out}")


if __name__ == "__main__":
    main()

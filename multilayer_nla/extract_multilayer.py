"""Stage 0 (multi-layer): corpus -> base_multilayer.parquet with a contiguous
three-layer activation patch [a^(l-1), a^(l), a^(l+1)] per sampled token position.

Mirrors `nla.datagen.stage0_extract` but captures THREE contiguous decoder-block
outputs in ONE forward pass (three hooks, one model run — not 3x the compute).
RAW vectors are stored (norm="none") — the plan's §4 (Rev 2) is explicit that
the AV injects raw `a` and the sqrt(d)-normalized `u` is a target/FVE-only
transform applied downstream (in headroom.py), never at extraction.

Position sampling REUSES `nla.datagen.stage0_extract._sample_positions` verbatim,
so for a given (seed, doc_id) the sampled positions are byte-identical to a
single-layer run. Two consequences the plan depends on:
  - the center tap a^(l) matches legacy single-layer extraction (verify with
    `verify_center_parity.py` — plan §12.5 step 1);
  - Condition A (single layer) and Conditions B/C/D (three layers) train on the
    exact same token positions (plan §7 "identical token-position sampling").

`layer_index=K` captures the OUTPUT of decoder block K (== HF hidden_states[K+1]),
matching `nla.datagen.extractors.HFExtractor` semantics. Center default = 24
(2/3 depth of Qwen3-8B's 36 layers; the released-dataset / "legacy layer-24"
choice the parity check anchors to). Triplet = {center-1, center, center+1}.

Output schema (FixedSizeList activations — same overflow-safety rationale as
stage0_extract._schema):
    n_raw_tokens                int64
    detokenized_text_truncated  str          (kept by default; --no-keep-text drops it)
    activation_prev             list<f32>[d]  RAW a^(center-1)
    activation_centre           list<f32>[d]  RAW a^(center)
    activation_next             list<f32>[d]  RAW a^(center+1)
    center_layer                int64
    doc_id                      str
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm

from nla.arch_adapters import resolve_decoder_layers
from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs
from nla.datagen.extractors import HFExtractor
# Reuse the EXACT keyed-RNG sampler + min-position + dataset-id helpers so a
# multi-layer run is positionally identical to a single-layer run (invariant).
from nla.datagen.stage0_extract import _MIN_POSITION, _dataset_id, _sample_positions
from multilayer_nla.manifest import build_manifest


class MultiLayerHFExtractor(HFExtractor):
    """HFExtractor that captures several decoder-block outputs in one forward.

    Inherits all of HFExtractor (model load, right-padding tokenization,
    batching, d_model). Adds `extract_multi`: registers a forward hook on each
    requested layer, runs ONE forward, returns per-text {layer -> [seq_len, d]}.
    The single-layer `extract()` is left untouched so `verify_center_parity.py`
    can compare the two paths on the same model instance.
    """

    def _register_multi_hooks(self, layer_indices: list[int]):
        layers = resolve_decoder_layers(self.model)
        for li in layer_indices:
            assert 0 <= li < len(layers), (
                f"layer_index={li} out of range for model with {len(layers)} layers"
            )
        self._captured_multi: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(li: int):
            def hook(_module, _inputs, output):
                # Transformer blocks return tuples; first element is the hidden
                # state. .clone() (not bare .detach()) — storage may be reused.
                h = output[0] if isinstance(output, tuple) else output
                self._captured_multi[li] = h.detach().clone()
            return hook

        for li in layer_indices:
            handles.append(layers[li].register_forward_hook(make_hook(li)))
        return handles

    @torch.no_grad()
    def extract_multi(self, texts: list[str], layer_indices: list[int],
                      *, final_token_only: bool = False) -> list[dict[str, Any]]:
        handles = self._register_multi_hooks(layer_indices)
        try:
            return self._extract_multi_impl(texts, layer_indices,
                                            final_token_only=final_token_only)
        finally:
            for h in handles:
                h.remove()

    def _extract_multi_impl(self, texts: list[str], layer_indices: list[int],
                            *, final_token_only: bool = False) -> list[dict[str, Any]]:
        """Per-text {token_ids, hidden}. `hidden[li]` is [seq_len, d] by default;
        with `final_token_only=True` it is the FINAL real token's vector [d]
        (under right padding, index seq_len-1). The latter is what the regenerate
        path wants — it keeps CPU memory flat regardless of how many layers are
        captured (saving an 11-layer window over long prefixes would otherwise
        accumulate the full [seq, d] per layer per text).
        """
        results: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.batch_size):
            sub = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length, add_special_tokens=True,
            )
            device = self.model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            self._captured_multi = {}
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for li in layer_indices:
                assert li in self._captured_multi, (
                    f"forward hook on decoder block {li} did not fire — wrong module path?"
                )
                assert self._captured_multi[li].shape[-1] == self.d_model, (
                    f"layer {li}: captured width {self._captured_multi[li].shape[-1]} "
                    f"!= d_model {self.d_model}"
                )
            lengths = attention_mask.sum(dim=1)  # [B], on device
            lengths_cpu = lengths.cpu().tolist()
            if final_token_only:
                # Select the final REAL token ON GPU, then move only [B, d] per layer
                # to CPU — NOT the whole [B, T, d]. Right padding: the last real token
                # is at index len-1, so captured[batch, len-1] gathers it. This keeps
                # the GPU->CPU transfer ~T× smaller, which is the whole point of
                # final_token_only when capturing a wide layer window.
                last = (lengths - 1).clamp_min(0)                       # [B]
                bidx = torch.arange(input_ids.shape[0], device=last.device)
                final = {li: self._captured_multi[li][bidx, last].float().cpu()  # [B, d]
                         for li in layer_indices}
                for i, seq_len in enumerate(lengths_cpu):
                    results.append({
                        "token_ids": input_ids[i, :seq_len].cpu().tolist(),
                        "hidden": {li: final[li][i] for li in layer_indices},  # [d]
                    })
            else:
                hidden = {li: self._captured_multi[li].float().cpu() for li in layer_indices}  # [B,T,d]
                for i, seq_len in enumerate(lengths_cpu):
                    results.append({
                        "token_ids": input_ids[i, :seq_len].cpu().tolist(),
                        "hidden": {li: hidden[li][i, :seq_len].clone() for li in layer_indices},  # [seq,d]
                    })
        return results


def _schema(d_model: int, keep_text: bool) -> pa.Schema:
    fields = [("n_raw_tokens", pa.int64())]
    if keep_text:
        fields.append(("detokenized_text_truncated", pa.string()))
    fields += [
        ("activation_prev", pa.list_(pa.float32(), d_model)),
        ("activation_centre", pa.list_(pa.float32(), d_model)),
        ("activation_next", pa.list_(pa.float32(), d_model)),
        ("center_layer", pa.int64()),
        ("doc_id", pa.string()),
    ]
    return pa.schema(fields)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-model", required=True, help="HF model name/path (also the extractor provenance key)")
    p.add_argument("--corpus", required=True, help="HF dataset name or local .parquet path")
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--corpus-start", type=int, default=0)
    p.add_argument("--corpus-length", type=int, required=True, help="number of documents to process")
    p.add_argument("--text-column", default="text")
    p.add_argument("--center-layer", type=int, default=24,
                   help="center block l; patch = {l-1, l, l+1}. Default 24 (Qwen3-8B 2/3 depth).")
    p.add_argument("--positions-per-doc", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=256, help="docs per extraction call / parquet write granularity")
    p.add_argument("--seed", type=int, default=42, help="position-sampling seed (keyed per doc_id)")
    p.add_argument("--keep-text", action=argparse.BooleanOptionalAction, default=True,
                   help="keep detokenized_text_truncated (needed for the §9 tail-text baseline; "
                        "--no-keep-text shrinks the parquet for a headroom-only run)")
    p.add_argument("--extractor-cls", default="multilayer_nla.extract_multilayer.MultiLayerHFExtractor")
    p.add_argument("--extractor-kwargs", default=None, help="JSON dict of extra extractor kwargs")
    p.add_argument("--output", required=True, help="output parquet path")
    add_storage_args(p)
    args = p.parse_args()

    # Patch geometry: the half-width is fixed at 1 (exactly three layers) per
    # plan §4 ("Do not begin with five-layer patches"). The schema is the three
    # named columns prev/centre/next; widening is deliberate future work.
    center = args.center_layer
    layers = [center - 1, center, center + 1]
    assert center - 1 >= 0, f"center-layer={center} has no l-1 block"

    storage = make_storage(args)

    user_kwargs = parse_kwargs(args.extractor_kwargs)
    assert "model_name" not in user_kwargs, (
        "pass --base-model, not --extractor-kwargs '{\"model_name\": ...}' "
        "(kwargs would silently win and poison the sidecar provenance)."
    )
    extractor_kwargs = {"model_name": args.base_model, **user_kwargs}
    extractor = load_class(args.extractor_cls)(**extractor_kwargs)
    assert hasattr(extractor, "extract_multi"), (
        f"{args.extractor_cls} has no extract_multi(); multi-layer extraction needs it"
    )
    d_model = extractor.d_model
    tokenizer = extractor.tokenizer
    # l+1 must exist — assert against the live model, not a hardcoded layer count.
    n_layers = len(resolve_decoder_layers(extractor.model))
    assert center + 1 < n_layers, (
        f"center-layer={center} needs block {center + 1}, but model has {n_layers} blocks"
    )
    keep_text = args.keep_text
    schema = _schema(d_model, keep_text)

    special_ids = set(tokenizer.all_special_ids)
    pad_id_to_check = (
        tokenizer.pad_token_id
        if (tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id)
        else None
    )

    import os
    if args.corpus.endswith(".parquet") and os.path.exists(args.corpus):
        ds = Dataset.from_parquet(args.corpus)
    else:
        ds = load_dataset(args.corpus, name=args.corpus_config, split=args.corpus_split)
    assert isinstance(ds, Dataset), (
        f"expected a concrete Dataset, got {type(ds).__name__} — pass an explicit --corpus-split"
    )
    ds = ds.select(range(args.corpus_start, args.corpus_start + args.corpus_length))

    storage.ensure_parent(args.output)
    row_count = 0
    n_docs_skipped = 0
    n_docs_short_sampled = 0

    with pq.ParquetWriter(storage.open_write(args.output), schema) as writer:
        for chunk_start in tqdm(range(0, len(ds), args.chunk_size), desc="chunks"):
            chunk = ds.select(range(chunk_start, min(chunk_start + args.chunk_size, len(ds))))
            texts = chunk[args.text_column]
            results = extractor.extract_multi(texts, layers)

            # Vectorized row build: accumulate per-doc numpy slices, then build the
            # FixedSizeList columns from contiguous buffers once per chunk. The old
            # path called .tolist() on every 4096-float vector (3 per position ->
            # ~31M Python float objects per chunk), which is single-threaded and
            # dominates CPU time on a many-core box. numpy advanced-indexing +
            # FixedSizeListArray.from_arrays skips the Python-object layer entirely
            # — same RAW float32 values, just no per-float boxing.
            prev_parts, centre_parts, next_parts = [], [], []
            nrt_col, did_col, cl_col, text_col = [], [], [], []
            for doc_offset, res in enumerate(results):
                doc_idx = args.corpus_start + chunk_start + doc_offset
                doc_id = f"{args.corpus}:{args.corpus_split}:{doc_idx}"
                token_ids = res["token_ids"]
                if pad_id_to_check is not None:
                    assert pad_id_to_check not in token_ids, (
                        f"pad_token_id {pad_id_to_check} found in token_ids for {doc_id} — "
                        f"the extractor's [:seq_len] slice is broken; all positions suspect."
                    )
                positions = _sample_positions(
                    token_ids, args.positions_per_doc, special_ids, doc_id, args.seed,
                )
                if not positions:
                    n_docs_skipped += 1
                    continue
                if len(positions) < args.positions_per_doc:
                    n_docs_short_sampled += 1
                pos_idx = torch.as_tensor(positions, dtype=torch.long)
                h = res["hidden"]
                # advanced-index -> [n_pos, d] float32 (h tensors are already float cpu).
                # RAW vectors — normalization is a downstream (FVE/AR) decision.
                prev_parts.append(h[center - 1][pos_idx].numpy())
                centre_parts.append(h[center][pos_idx].numpy())
                next_parts.append(h[center + 1][pos_idx].numpy())
                for pos in positions:
                    nrt_col.append(pos + 1)
                    did_col.append(doc_id)
                    cl_col.append(center)
                    if keep_text:
                        text_col.append(
                            tokenizer.decode(token_ids[: pos + 1], skip_special_tokens=True)
                        )

            if did_col:
                def _fsl(parts):
                    flat = np.concatenate(parts, axis=0).reshape(-1).astype(np.float32, copy=False)
                    return pa.FixedSizeListArray.from_arrays(pa.array(flat), d_model)
                cols = {
                    "n_raw_tokens": pa.array(nrt_col, pa.int64()),
                    "activation_prev": _fsl(prev_parts),
                    "activation_centre": _fsl(centre_parts),
                    "activation_next": _fsl(next_parts),
                    "center_layer": pa.array(cl_col, pa.int64()),
                    "doc_id": pa.array(did_col, pa.string()),
                }
                if keep_text:
                    cols["detokenized_text_truncated"] = pa.array(text_col, pa.string())
                writer.write_table(pa.table(cols, schema=schema))
                row_count += len(did_col)

    corpus_slice = {"start": args.corpus_start, "length": args.corpus_length}
    manifest = build_manifest(
        stage="stage0_multilayer_extract",
        tokenizer=tokenizer,
        extra={
            "base_model": args.base_model,
            "layer_triplet": layers,
            "center_layer": center,
            "corpus": args.corpus,
            "corpus_slice": corpus_slice,
            "position_seed": args.seed,
            "positions_per_doc": args.positions_per_doc,
            "d_model": d_model,
        },
    )
    meta = {
        "kind": "mlnla_dataset",
        "schema_version": 1,
        "stage": "base_multilayer",
        "base_model": args.base_model,
        "d_model": d_model,
        "center_layer": center,
        "layers": layers,                 # output-of-block indices (stage0 semantics)
        "layer_offsets": [-1, 0, 1],
        "norm": "none",                   # RAW — invariant
        "corpus": args.corpus,
        "corpus_slice": corpus_slice,
        "positions_per_doc": args.positions_per_doc,
        "seed": args.seed,
        "row_count": row_count,
        "keep_text": keep_text,
        "dataset_id": _dataset_id(args.base_model, center, args.corpus, corpus_slice),
        "manifest": manifest,
    }
    meta_path = args.output + ".mlnla_meta.yaml"
    storage.write_text(meta_path, yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))

    print(f"wrote {row_count} rows ({d_model}-dim x3 layers {layers}) -> {args.output}")
    print(f"  skipped {n_docs_skipped} docs (too short / all-special past position {_MIN_POSITION})")
    print(f"  short-sampled {n_docs_short_sampled} docs (< {args.positions_per_doc} valid positions)")
    print(f"sidecar -> {meta_path}")
    print(json.dumps({"row_count": row_count, "layers": layers, "d_model": d_model}, indent=2))


if __name__ == "__main__":
    main()

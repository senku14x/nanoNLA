"""Stage-expansion dataset (spec §4, §5) — exact teacher prefixes → 7 raw layer targets.

One base row per source activation record (teacher IDs tokenized ONCE + 7 raw target
vectors + doc_id + split). Each base row expands VIRTUALLY into 3 stage views
(32/64/128); activation targets are NOT triplicated on disk. text_mode swaps the teacher
content for the eval controls: 'real', 'no_text' (empty prefix), 'shuffled' (another
document's prefix at the SAME effective length — strict-128 makes the length match exact).
"""

from __future__ import annotations

from multilayer_nla.progressive_reader.prefix import (
    effective_prefix_length, exact_prefix, teacher_ids_sha256,
)
from multilayer_nla.progressive_reader.schedule import (
    PREFIX_BUDGETS, TARGET_LAYERS, active_layer_mask,
)

_AR_PRE = "Summary of the following text: <text>"
_AR_SUF = "</text> <summary>"


def _teacher_text(raw, field):
    from nla.schema import extract_explanation
    if field == "response":
        return extract_explanation(raw)
    if isinstance(raw, str) and raw.startswith(_AR_PRE) and raw.endswith(_AR_SUF):
        return raw[len(_AR_PRE):len(raw) - len(_AR_SUF)]
    return None


def load_base_rows(data_glob, tokenizer, *, target_layers=TARGET_LAYERS, teacher_field="auto",
                   require_full_128=True, fracs=(0.8, 0.1, 0.1), seed=42,
                   names=("train", "dev", "test"), max_documents=None, max_rows=None,
                   tok_batch=1024):
    """Stream the bank corpus → per-split base rows. Tokenizes the teacher text ONCE (the
    canonical teacher-token-ID source, spec §1.1) and stores ids + sha256 + length, the 7 raw
    target vectors, doc_id, src_row_id, and the doc-level split bucket. Returns
    {split: dict-of-arrays}. require_full_128 keeps only rows with n>=128 (headline)."""
    import glob as _glob
    import numpy as np
    import pyarrow.parquet as pq
    from multilayer_nla.datasets import doc_bucket

    paths = sorted(_glob.glob(data_glob)) or [data_glob]
    schema_names = pq.ParquetFile(paths[0]).schema_arrow.names
    if teacher_field == "auto":
        teacher_field = "response" if "response" in schema_names else "prompt"
    tcols = [f"activation_L{l}" for l in target_layers]
    for c in tcols:
        assert c in schema_names, f"missing target column {c}"
    cols = [teacher_field, "doc_id"] + tcols

    texts, doc_ids, tgt_parts, n = [], [], [], 0
    for fp in paths:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            if max_rows is not None and n >= max_rows:
                break
            t = pf.read_row_group(rg, columns=cols)
            take = t.num_rows if max_rows is None else min(max_rows - n, t.num_rows)
            t = t.slice(0, take)
            raws = t.column(teacher_field).to_pylist()
            dids = t.column("doc_id").to_pylist()

            def to_np(name):
                c = t.column(name).combine_chunks()
                return c.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(c), -1)

            layer_arrs = [to_np(c) for c in tcols]          # each [take, d]
            for i in range(take):
                expl = _teacher_text(raws[i], teacher_field)
                if not expl or not expl.strip():
                    continue
                texts.append(expl)
                doc_ids.append(dids[i])
                tgt_parts.append(np.stack([a[i] for a in layer_arrs]))   # [k, d]
            n += take
        if max_rows is not None and n >= max_rows:
            break

    # tokenize ONCE
    full_ids = []
    for i in range(0, len(texts), tok_batch):
        full_ids.extend(tokenizer(texts[i:i + tok_batch], add_special_tokens=False)["input_ids"])
    lengths = np.fromiter((len(x) for x in full_ids), dtype=np.int64, count=len(full_ids))
    targets = np.stack(tgt_parts) if tgt_parts else np.zeros((0, len(target_layers), 1), np.float32)

    keep = lengths >= 128 if require_full_128 else np.ones(len(full_ids), dtype=bool)
    # smoke: limit to the first `max_documents` unique docs (deterministic by stream order)
    if max_documents is not None:
        seen, doc_keep = set(), np.zeros(len(full_ids), dtype=bool)
        for i, d in enumerate(doc_ids):
            if not keep[i]:
                continue
            if d in seen or len(seen) < max_documents:
                seen.add(d)
                doc_keep[i] = d in seen
        keep = keep & doc_keep

    buckets = np.fromiter((doc_bucket(d, fracs, seed) for d in doc_ids), dtype=np.int64, count=len(doc_ids))
    out = {}
    src_ids = np.arange(len(full_ids), dtype=np.int64)     # global ordinal (stable across views)
    for bi, nm in enumerate(names):
        sel = np.where(keep & (buckets == bi))[0]
        out[nm] = {
            "full_ids": [full_ids[i] for i in sel],
            "sha256": [teacher_ids_sha256(full_ids[i]) for i in sel],
            "lengths": lengths[sel],
            "targets": targets[sel],                       # [n, k, d]
            "doc_ids": [doc_ids[i] for i in sel],
            "src_row_ids": src_ids[sel],
            "teacher_field": teacher_field,
        }
    return out


class ProgressiveReaderDataset:
    """torch Dataset over one split. len = n_base * len(budgets); idx -> (base, stage).

    text_mode: 'real' | 'no_text' | 'shuffled'. shuffle_perm (deranged base indices) is
    required for 'shuffled'; the shuffled teacher is perm[base]'s prefix at THIS row's
    effective length (exact under strict-128)."""

    def __init__(self, split_rows, tokenizer, stages, *, target_layers=TARGET_LAYERS,
                 budgets=PREFIX_BUDGETS, text_mode="real", shuffle_perm=None,
                 prompt_prefix=None, suffix=None):
        from multilayer_nla.progressive_reader.model import (
            DEFAULT_PROMPT_PREFIX, DEFAULT_SUFFIX,
        )
        self.r = split_rows
        self.tok = tokenizer
        self.budgets = list(budgets)
        self.target_layers = list(target_layers)
        self.text_mode = text_mode
        self.perm = shuffle_perm
        self.prompt_prefix = DEFAULT_PROMPT_PREFIX if prompt_prefix is None else prompt_prefix
        self.suffix = DEFAULT_SUFFIX if suffix is None else suffix
        # precompute fixed prefix/suffix ids + per-stage active masks
        self.pre_ids = tokenizer.encode(self.prompt_prefix, add_special_tokens=False)
        self.suf_ids = tokenizer.encode(self.suffix, add_special_tokens=False)
        self.stage_masks = [active_layer_mask(stages[b], tuple(target_layers)) for b in self.budgets]
        self.n_base = len(self.r["full_ids"])
        if text_mode == "shuffled":
            assert self.perm is not None, "shuffled mode needs shuffle_perm"

    def __len__(self):
        return self.n_base * len(self.budgets)

    def __getitem__(self, idx):
        base = idx // len(self.budgets)
        stage = idx % len(self.budgets)
        budget = self.budgets[stage]
        eff = effective_prefix_length(self.r["full_ids"][base], budget)
        if self.text_mode == "no_text":
            teacher = []
        elif self.text_mode == "shuffled":
            teacher = exact_prefix(self.r["full_ids"][self.perm[base]], eff)   # other doc, same eff len
        else:
            teacher = exact_prefix(self.r["full_ids"][base], budget)
        input_ids = self.pre_ids + teacher + self.suf_ids
        return {
            "input_ids": input_ids,
            "targets": self.r["targets"][base],            # [k, d] raw
            "active_mask": self.stage_masks[stage],        # [k] 0/1
            "budget": budget,
            "effective_teacher_prefix_length": int(eff if self.text_mode != "no_text" else 0),
            "had_full_budget": bool(self.r["lengths"][base] >= budget),
            "doc_id": self.r["doc_ids"][base],
            "src_row_id": int(self.r["src_row_ids"][base]),
            "teacher_ids_sha256": self.r["sha256"][base],
        }

    def collate(self, batch, device, pad_id):
        import torch
        T = max(len(b["input_ids"]) for b in batch)
        B = len(batch)
        ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, T), dtype=torch.long, device=device)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long, device=device)
            attn[i, :L] = 1
        readout_idx = attn.sum(dim=1) - 1                  # last real token (spec §4)
        import numpy as np
        targets = torch.tensor(np.stack([b["targets"] for b in batch]), dtype=torch.float32, device=device)
        active = torch.tensor([b["active_mask"] for b in batch], dtype=torch.float32, device=device)
        return {
            "input_ids": ids, "attention_mask": attn, "readout_idx": readout_idx,
            "targets": targets, "active_mask": active,
            "budgets": [b["budget"] for b in batch],
            "doc_ids": [b["doc_id"] for b in batch],
            "src_row_ids": [b["src_row_id"] for b in batch],
            "eff_lengths": [b["effective_teacher_prefix_length"] for b in batch],
            "meta": batch,
        }

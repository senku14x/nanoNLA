"""Multi-vector dataset loading for the three-layer NLA (plan §6.1, §7, point 2).

Three responsibilities, all preserving ALL THREE activation columns
(prev/centre/next) end to end — never collapsing to a single `activation_vector`:

  1. The AV prompt with three depth slots (the same marker char x3; slot identity
     is by order). `INJECT_PLACEHOLDER` x3, swapped to the marker char at prep.
  2. `split_by_document` — document-level 3-way split (av_sft / ar_sft / rl) that
     routes every position of a doc to ONE bucket and carries every column.
  3. `stack_slot_vectors` / `prepare_av_chunk_multi` — build the [B*K, d] vector
     tensor in the exact scan order `inject_multislot_in_residual` expects
     ([a_prev, a_centre, a_next] per example, example-major), plus the AV-SFT
     (input_ids, attn, response-only loss_mask).

The base parquet (from extract_multilayer.py) has columns:
    n_raw_tokens, [detokenized_text_truncated], activation_prev/centre/next,
    center_layer, doc_id.
AV/AR SFT parquets add `prompt` (and AV adds `response`) downstream; this module
loads + preps them.
"""

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nla.schema import INJECT_PLACEHOLDER

# ---- AV prompt: three depth slots, same marker char (order = slot) ----
# Kept as a module constant so datagen (build) and training (prep) use the SAME
# string — drift here moves the marker positions and breaks injection.
AV_USER_TEMPLATE = (
    "Earlier depth: {m}\n"
    "Centre depth: {m}\n"
    "Later depth: {m}\n"
    "Describe the information represented across these local computational stages."
)
N_SLOTS = 3
# Scan/stack order — MUST match inject_multislot_in_residual's example-major,
# ascending-position walk: the prompt lists prev, centre, next top-to-bottom.
SLOT_COLUMNS = ("activation_prev", "activation_centre", "activation_next")


def av_user_content(placeholder: str = INJECT_PLACEHOLDER) -> str:
    return AV_USER_TEMPLATE.format(m=placeholder)


def build_av_prompt(placeholder: str = INJECT_PLACEHOLDER) -> list[dict]:
    """The constant AV chat prompt (one user turn with three marker slots)."""
    return [{"role": "user", "content": av_user_content(placeholder)}]


def apply_chat_template_no_think(tokenizer, msgs, *, add_generation_prompt=True) -> str:
    """Chat-template the AV prompt with Qwen3 thinking DISABLED (enable_thinking=False).

    Qwen3 is a thinking model: with thinking on it emits <think>...</think> and burns
    the whole token budget reasoning *about* the prompt, never reaching <explanation>.
    The NLA AV is a verbalizer, not a reasoner — we want the answer directly. The
    kwarg is an unused template var on non-thinking archs (Llama/Gemma), so this is
    harmless there. MUST be used identically at AV-SFT train time and RL rollout
    time, or the actor sees a different prompt than it was trained on.
    """
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


# ---- Document-level split (plan §3 invariant, point 2) ----

def doc_bucket(doc_id: str, fracs: tuple[float, ...], seed: int) -> int:
    """Route a doc_id to a bucket index by hashing (seed, doc_id) -> [0,1).

    All positions of a doc share its doc_id, so they all land in one bucket.
    """
    h = hashlib.sha256(f"{seed}|{doc_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    cum = 0.0
    for i, f in enumerate(fracs):
        cum += f
        if u < cum:
            return i
    return len(fracs) - 1


def split_by_document(base_parquet: str, out_dir: str, *,
                      fracs: tuple[float, ...] = (0.25, 0.25, 0.5),
                      names: tuple[str, ...] = ("av_sft", "ar_sft", "rl"),
                      seed: int = 42) -> dict[str, str]:
    """Stream the base multi-layer parquet, route each row to a bucket by doc_id,
    and write one parquet per bucket carrying EVERY column (all three activation
    columns + provenance). Returns {name: path}.

    Document-level (never splits a doc across buckets) and seed-deterministic.
    """
    from pathlib import Path
    assert len(fracs) == len(names), "fracs and names must align"
    assert abs(sum(fracs) - 1.0) < 1e-6, f"fracs must sum to 1, got {sum(fracs)}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(base_parquet)
    schema = pf.schema_arrow
    paths = {nm: str(out / f"{nm}.parquet") for nm in names}
    writers = {nm: pq.ParquetWriter(paths[nm], schema) for nm in names}
    counts = {nm: 0 for nm in names}
    try:
        for rg_idx in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg_idx)
            doc_ids = tbl.column("doc_id").to_pylist()
            buckets = np.fromiter((doc_bucket(d, fracs, seed) for d in doc_ids),
                                  dtype=np.int64, count=len(doc_ids))
            for bi, nm in enumerate(names):
                mask = pa.array(buckets == bi)
                sub = tbl.filter(mask)
                if sub.num_rows:
                    writers[nm].write_table(sub)
                    counts[nm] += sub.num_rows
    finally:
        for w in writers.values():
            w.close()
    print(f"[split] {base_parquet} -> " + ", ".join(f"{nm}:{counts[nm]}" for nm in names))
    return paths


# ---- Slot-vector stacking (the correctness-critical bit) ----

def stack_slot_vectors(rows: list[dict], k: int = N_SLOTS) -> np.ndarray:
    """Stack the three per-row activations into [B*k, d] in injection scan order.

    For row b: [row[prev], row[centre], row[next]] -> rows 0..k-1.
    Flatten is example-major ([b0 slots..., b1 slots..., ...]), matching
    inject_multislot_in_residual's row-major marker walk.
    """
    per_row = np.stack(
        [np.stack([np.asarray(r[c], dtype=np.float32) for c in SLOT_COLUMNS[:k]])
         for r in rows]
    )  # [B, k, d]
    B, kk, d = per_row.shape
    assert kk == k
    return per_row.reshape(B * k, d)


# ---- Loading + AV-SFT chunk prep ----

def load_av_sft_dataset(parquet_path: str, n_max: int | None = None) -> list[dict]:
    """Load AV-SFT rows: prompt (list[msg]) + response (str) + 3 activations.

    Activations via flatten->numpy (zero-copy-ish), same pattern as nla/. Slices
    row-groups so n_max takes exactly n_max rows, not the whole first group.
    """
    cols = ["prompt", "response", *SLOT_COLUMNS]
    pf = pq.ParquetFile(parquet_path)
    rows: list[dict] = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if n_max is None else min(n_max - len(rows), rg.num_rows)
        rg = rg.slice(0, take)
        prompts = rg.column("prompt").to_pylist()
        responses = rg.column("response").to_pylist()

        def to_np(name):
            col = rg.column(name).combine_chunks()
            return (col.flatten().to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(col), -1))

        acts = {c: to_np(c) for c in SLOT_COLUMNS}
        for i in range(take):
            rows.append({
                "prompt": prompts[i],
                "response": responses[i],
                **{c: acts[c][i] for c in SLOT_COLUMNS},
            })
    return rows


def prepare_av_chunk_multi(rows: list[dict], tokenizer, inject_char: str, inj_id: int,
                           device, *, max_len: int = 1024, k: int = N_SLOTS):
    """Build (input_ids, attn, loss_mask, vectors[B*k, d]) for a multi-slot AV-SFT batch.

    - prompt: chat-templated, with INJECT_PLACEHOLDER x k swapped to inject_char.
    - per-example guard: assert exactly k marker tokens in the prompt (complements
      the hook's per-row check; catches template/tokenizer drift before the forward).
    - response: trailing EOS so the model learns to stop; loss_mask = 1 on response
      tokens only (CE is response-only).
    - vectors: stack_slot_vectors -> [B*k, d] in scan order.
    """
    full_ids_list, prompt_lens = [], []
    for row in rows:
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        prompt_str = apply_chat_template_no_think(tokenizer, msgs)
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        n_mark = sum(1 for t in prompt_ids if t == inj_id)
        assert n_mark == k, (
            f"AV prompt has {n_mark} marker tokens, expected k={k}. Template drift or "
            f"the marker char split into multiple tokens under this tokenizer."
        )
        resp = row["response"] + (tokenizer.eos_token or "")
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)
        full = prompt_ids + resp_ids
        if len(full) > max_len:
            full = full[:max_len]  # truncate response from the right; prompt (markers) is preserved
        full_ids_list.append(torch.tensor(full, dtype=torch.long))
        prompt_lens.append(len(prompt_ids))

    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    loss_mask = torch.zeros((bs, T), dtype=torch.float32, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
        loss_mask[i, prompt_lens[i]:L] = 1  # response-only (in target-token space; CE shift applied in loss)

    vectors = torch.tensor(stack_slot_vectors(rows, k), dtype=torch.float32, device=device)
    return batch_ids, attn, loss_mask, vectors


# ---- AR-SFT: text z -> three activations ----
# Suffix-anchored critic prompt (plan §6.2): the AR reads ONLY the explanation,
# formatted into this template ending in a fixed suffix, and taps the LAST token.
# No marker / injection on the AR side.
AR_CRITIC_TEMPLATE = "<text>{explanation}</text> <summary>"


def fill_ar_prompt(explanation: str) -> str:
    return AR_CRITIC_TEMPLATE.format(explanation=explanation)


def load_ar_sft_dataset(parquet_path: str, n_max: int | None = None) -> list[dict]:
    """Load AR-SFT rows: prompt (filled critic text, str) + 3 activations."""
    cols = ["prompt", *SLOT_COLUMNS]
    pf = pq.ParquetFile(parquet_path)
    rows: list[dict] = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if n_max is None else min(n_max - len(rows), rg.num_rows)
        rg = rg.slice(0, take)
        prompts = rg.column("prompt").to_pylist()

        def to_np(name):
            col = rg.column(name).combine_chunks()
            return (col.flatten().to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(col), -1))

        acts = {c: to_np(c) for c in SLOT_COLUMNS}
        for i in range(take):
            rows.append({"prompt": prompts[i], **{c: acts[c][i] for c in SLOT_COLUMNS}})
    return rows


def prepare_ar_chunk_multi(rows: list[dict], tokenizer, device, *, max_len: int = 1024):
    """(input_ids, attn, gold[B, N_SLOTS, d]) for an AR-SFT batch.

    Critic prompt tokenized with add_special_tokens=False (matches RL-time
    scoring; True would prepend BOS on Llama/Gemma -> train/reward mismatch).
    Over-length rows are SKIPPED (right-truncation would cut the <summary>
    suffix and the last-token tap would land mid-text). gold is the three RAW
    activations stacked [prev, centre, next] — normalized in the loss.
    """
    ids_list, kept = [], []
    n_skipped = 0
    for row in rows:
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(ids) > max_len:
            n_skipped += 1
            continue
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        kept.append(row)
    assert ids_list, f"all {len(rows)} AR rows exceeded max_len={max_len}"
    if n_skipped:
        print(f"[ar] skipped {n_skipped}/{len(rows)} rows over max_len (suffix anchor)")
    bs = len(ids_list)
    T = max(t.numel() for t in ids_list)
    pad = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    for i, t in enumerate(ids_list):
        batch_ids[i, :t.numel()] = t.to(device)
        attn[i, :t.numel()] = 1
    gold = torch.tensor(
        np.stack([[np.asarray(r[c], dtype=np.float32) for c in SLOT_COLUMNS] for r in kept]),
        dtype=torch.float32, device=device,
    )  # [B, N_SLOTS, d]
    return batch_ids, attn, gold

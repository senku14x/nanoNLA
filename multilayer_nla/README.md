# Multi-layer NLA — FineWeb architecture-validation phase

Self-contained experiment package for the **FineWeb Multi-Layer Natural Language
Autoencoders** plan (Revision 2, design frozen). Everything here reuses the
stable `nla/` core (keyed-RNG sampling, `HFExtractor`, storage) and never mutates
it.

**Primary question.** Does a coherent contiguous activation patch
`[a^(l-1), a^(l), a^(l+1)]` preserve local cross-layer information through a
natural-language bottleneck better than matched controls?

**Non-goals for this phase (plan §15).** No safety / deception / refusal /
corrigibility / agentic claims, no faithfulness claim, no claim that this
reproduces the official NLA. The only intended claim is the architectural one
above, on passive FineWeb activations.

---

## What's built here

| Script | Role | Plan ref | Runs on |
|---|---|---|---|
| `extract_multilayer.py` | Stage 0: corpus → 3-layer patch (RAW), keyed-RNG positions | §4, §6.1, §12.5.1 | H200 |
| `verify_center_parity.py` | center tap == legacy L24 (bitwise) | §12.5.1 | H200 |
| `headroom.py` | Gate 0 probe (`H_unique`) | §5.1, §8, §13 | CPU/H200 |
| `regenerate_multilayer_activations.py` | **published prefix → layer-window activations at final token** (`--save-layers`, free labels, no API; sharded) | §6.1 | H200 |
| `build_from_published.py` | select center triplet (`--center`) → our 3-slot av/ar/rl training parquets | §6.1, §7 | anywhere |
| `build_datasets.py` | split a fresh extraction → av/ar/rl (dummy/real explanations) | §6.1 | anywhere |
| `datasets.py` | 3-slot AV prompt, doc-split, loaders, chunk prep | §6.1–6.2 | anywhere |
| `injection_multi.py` | 3-slot norm-matched ADD + per-row marker guard | §6.1 | anywhere |
| `models_multi.py` | multi-tap AR (3 depth heads) + three-target reward | §6.2 | H200 |
| `train_{av,ar,rl}_multi.py` | AV-SFT / AR-SFT / GRPO trainers (bf16+LoRA, wandb) | §6.2–6.3 | H200 |
| `manifest.py` | run provenance (§12.6) | §12.6 | — |
| `tests/` | 47 offline tests (numpy/pyarrow/torch) | — | anywhere |

**Status update.** The **§7 SFT control sweep is COMPLETE** — 6 conditions
(local / duplicate / wide / single / s2_19_21_23 / s2_20_22_24), held-out test;
results + configs in **`EXPERIMENT_REPORT.md`** and **`SWEEP_STATUS.md`** (headline:
multi-layer AV input helps reconstruction, significant but ~5× smaller than the
verbalizer→ceiling gap). The condition lives in the **data** (positional `av_in_*`
columns written by `build_sweep.py`), **not** a train/inject-time flag — the old
`coherent/duplicate/single/mismatched` flag idea was dropped. **Still pending:** the
pre-registered **Gate 0** headroom verdict (deferred); RL-per-condition (not yet wired).

---

## Frozen execution order (plan §12.5)

```
1. extract three contiguous layers  ── extract_multilayer.py
   └─ verify center tap == legacy L24 extraction (bitwise)  ── verify_center_parity.py
2. geometry + ridge + mean-update + center-only headroom  ── headroom.py  → GATE 0
   ─────────────────────────────  (this is where the NLA-free phase ends)  ─────────────
3. three-slot AV injection; coherent vs duplicate, 20 SFT steps        [built — train_av_multi.py]
4. verify marker count / slot order / no marker leakage / loss down    [built — injection guard + smoke]
5. multi-tap AR; 5k-doc full-chain smoke                               [built — train_ar_multi.py; smoke GREEN]
6. 100k corpus + 500-step RL                                           [built — train_rl_multi.py; smoke GREEN]
```

The end-to-end plumbing smoke (AV-SFT → AR-SFT → RL, dummy labels) is GREEN. For
the **real** run, source labels from the published dataset (next section) instead
of paying for API explanations; Gate 0 and the §7 condition flag remain pending.

---

## Run recipe (steps 1–2, on the H200)

Set `PYTHONPATH` to the repo root so `import nla` / `import multilayer_nla` resolve.

```bash
export PYTHONPATH=/path/to/nanoNLA
OUT=/data/mlnla/qwen3_8b_L24

# --- Step 1a: extract the 3-layer patch (RAW vectors; center=24 -> {23,24,25}) ---
python -m multilayer_nla.extract_multilayer \
    --base-model Qwen/Qwen3-8B \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --corpus-length 100000 --positions-per-doc 10 \
    --center-layer 24 --seed 42 \
    --extractor-kwargs '{"batch_size": 12, "max_length": 4096}' \
    --output $OUT/base_multilayer.parquet
#  smoke: --corpus-length 2000

# --- Step 1b: DECISIVE plumbing gate — center tap must match legacy L24 (bitwise) ---
python -m multilayer_nla.verify_center_parity \
    --base-model Qwen/Qwen3-8B \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --center-layer 24 --n-docs 16 --atol 0
#  -> "[parity] PASS" required before trusting any FVE number.

# --- Step 2: headroom probe -> GATE 0 verdict ---
python -m multilayer_nla.headroom \
    --parquet $OUT/base_multilayer.parquet \
    --out-json $OUT/headroom_L24.json \
    --dev-frac 0.2 --seed 0
```

The sidecar (`$OUT/base_multilayer.parquet.mlnla_meta.yaml`) is auto-discovered by
the probe and folded into the results manifest (plan §12.6).

---

## Real run — free labels from the published dataset (NO API)

nanoNLA's published warmstart dataset
[`ceselder/qwen3-8b-nla-L24-finefineweb-100k`](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k)
**already contains the real labels** — the AV `response` explanations, AR
`prompt`s, RL prompts, the exact `detokenized_text_truncated` prefix,
`n_raw_tokens`, and `doc_id`. It intentionally omits only the raw activation
array, because the activation is a deterministic function of (exact prefix,
layer) — and the *label* itself only ever depended on the **text** (stage 2 feeds
the API model just the prefix; it never sees the activation). So the published
explanation is exactly as valid for our three-layer patch as for the original
single layer-24 vector. We **regenerate only the activations** and inherit the
labels for free: no API key, no paid Stage-2 labeling, just H200 forward time.

> **Corpus caution.** The labeled data is over **`m-a-p/FineFineWeb`**. Our 30k
> `extract_multilayer.py` run used **`HuggingFaceFW/fineweb`** — a *different*
> corpus, kept only as a plumbing/extraction check. Do **not** join the published
> labels onto those rows or point the regenerator at that extraction; positions
> and texts will not line up. The regenerator consumes the published rows'
> *stored prefixes* directly.

The whole thing — fresh-instance bootstrap (clone + `pip install -e .`), the
small-slice smoke, and the full sharded run — is scripted in
[`scripts/run_labeled.sh`](scripts/run_labeled.sh) (smoke by default; `RUN_FULL=1`
for the ~1M-row run). The annotated steps:

```bash
export PYTHONPATH=/path/to/nanoNLA
PUB=/data/mlnla/published          # downloaded HF subsets (text + LABELS, no vectors)
REGEN=/data/mlnla/published_L24x3  # + activation_prev/centre/next
TRAIN=/data/mlnla/qwen3_8b_L24_labeled

# 1. Download the published av_sft / ar_sft / rl subsets.
python - <<'PY'
import os
from datasets import load_dataset
PUB = os.environ["PUB"]
for name in ("av_sft", "ar_sft", "rl"):
    ds = load_dataset("ceselder/qwen3-8b-nla-L24-finefineweb-100k", name, split="train")
    ds.to_parquet(f"{PUB}/{name}.parquet"); print(name, ds.num_rows)
PY

# 2. Regenerate the activation window at each prefix's FINAL token. Capturing
#    layers 19-29 is FREE on compute (one forward computes every layer) and
#    future-proofs the §5 center sweep [20,22,24,26,28]; cost is ~16 GB/layer.
#    --max-length 4096 MUST match the published extraction; the n_raw_tokens
#    round-trip check hard-fails on any drift (wrong position).
for s in av_sft ar_sft rl; do
  python -m multilayer_nla.regenerate_multilayer_activations \
      --in $PUB/$s.parquet --out $REGEN/$s.parquet \
      --base-model Qwen/Qwen3-8B --center-layer 24 --save-layers 19-29 --max-length 4096
done
#  ~1M rows: fan out with --num-shards N --shard-index i (per-shard files; tool
#  prints the merge one-liner). Saves activation_L19 .. activation_L29 (RAW).

# 3. Select the center triplet -> our 3-slot training parquets: --center 24 picks
#    activation_L{23,24,25} as prev/centre/next; swaps the single-marker actor
#    prompt for our three-marker one (av, rl); keeps the published <explanation>
#    response (av) and canonical critic prompt (ar) verbatim.
python -m multilayer_nla.build_from_published --mode all --center 24 --in-dir $REGEN --out-dir $TRAIN
#  -> $TRAIN/{av_sft,ar_sft,rl}.parquet   (NO sidecar — trainers auto-pick the marker)
#  Sweep a different center later WITHOUT re-extracting: re-run step 3 with --center 22/26/...

# 4. Train AV-SFT -> AR-SFT -> RL exactly as the point-6 smoke, but --*-parquet
#    pointed at $TRAIN (bf16 + LoRA, --quant none). The av/ar/rl document split is
#    INHERITED from the published subsets (the split the released checkpoints used).
```

**Scale (~1M rows).** The full L24 set is ~1M positions; step 2 is one forward per
prefix (up to 4096 tokens). Fan out across GPUs with `--num-shards N --shard-index i`
— each job writes its own `*.shardNNofMM.parquet` and the tool prints a one-liner to
merge them before step 3. Validate on a small `--in` slice first: with
`--max-length 4096` the `n_raw_tokens` round-trip should be 100% clean;
`--max-drop-frac 1e-3` tolerates rare per-row tokenizer drift instead of aborting a
whole shard (default `0.0` hard-fails on any mismatch — usually a `--max-length`
misconfiguration).

**Prompt length.** AR-SFT skips, and RL-time critic scoring penalizes, critic
prompts over `--max-len` (default 1024). The critic prompt embeds the *explanation*,
not the 4096-token prefix, so this rarely bites — but check the length distribution
on the regenerated `ar_sft` parquet and raise `--max-len` if a non-trivial fraction
would be dropped.

The §7 controls run on these **identical labeled rows**. NOTE (updated): the shipped
sweep makes the condition a **data** property — `build_sweep.py` writes per-condition
positional `av_in_*` columns (actual conditions: local / duplicate / wide / single /
s2_19_21_23 / s2_20_22_24), so it IS a separate cheap data build per condition, **not** a
train/inject-time flag. The AR target stays fixed at [L23,L24,L25] regardless. See
`SWEEP_STATUS.md` / `EXPERIMENT_REPORT.md`.

---

## Gate 0 — the decision (plan §5.1, §13, Rev 2)

The probe predicts the local update `delta = [u^(l)-u^(l-1), u^(l+1)-u^(l)]` from
the center `u^(l)` alone, with four predictors, and reports update FVE
(`FVE_delta = 1 - E||pred-delta||^2 / E||delta - delta_bar||^2`, §8):

| predictor | meaning | by construction |
|---|---|---|
| constant `delta_bar` | the floor | `FVE_delta = 0` |
| ridge | best **linear** center-only | — |
| MLP | best cheap **nonlinear** center-only | — |
| true neighbor | the ceiling | `FVE_delta = 1` |

**Decision variable (Rev 2):** the *unique headroom*
`H_unique = 1 - max(FVE_ridge, FVE_MLP)` on the **dev** split — the update
variance the center cannot recover, i.e. the ceiling on what a coherent patch
could add through the bottleneck.

- **STOP / pivot iff `H_unique < 0.05`** — the center already determines the
  neighbors; there is nothing for the bottleneck to carry.
- **Otherwise GO** — proceed to Stage 2 (Conditions A–D).
- The **MLP–ridge gap** is reported as a *diagnostic* (is the recoverable part
  nonlinear?), **not** the gate.

If torch is unavailable the probe runs ridge-only; since the MLP can only *raise*
the best center-only FVE, ridge-only `H_unique` is an **upper bound** — a
ridge-only STOP is decisive, a ridge-only GO is flagged provisional.

The probe self-asserts the floor (`==0`) and ceiling (`==1`) identities and
√d-normalization on every run; a failure raises before any number is trusted
(plan §5.1.6).

---

## Invariants preserved (CLAUDE.md / plan Rev 2)

- **RAW vectors stored** (`norm="none"`). The AV injects raw `a`; the
  √d-normalized `u = √d·a/‖a‖` is used **only** as AR target and in FVE (§4 Rev 2).
- **Document-level split** — all positions of a doc share one bucket
  (`doc_split_mask`), in both extraction provenance and the probe's dev split.
- **Per-doc keyed RNG** — position sampling reuses
  `nla.datagen.stage0_extract._sample_positions` verbatim, so the multi-layer
  center tap lands on the same positions a single-layer run would (Condition A
  comparability + the parity check).
- **No mutation of `nla/`** — `MultiLayerHFExtractor` subclasses `HFExtractor`;
  the single-layer `extract()` path is untouched so parity can compare both.

---

## Local validation already run (pre-H200)

- `tests/test_fve_math.py` — 6 numpy tests: floor=0, ceiling=1, √d-norm, ridge
  recovers a linear map, doc-level split, `H_unique` definition. **PASS.**
- `headroom.py --selfcheck` — synthetic GO/STOP cases (orthogonal-map neighbors
  vs an independent neighbor); verdicts and identities. **PASS** (incl. torch MLP).
- Probe on a synthetic parquet via the real CLI — streaming load, MLP path,
  JSON + manifest output. **PASS.**
- 3-hook capture on a tiny in-memory Llama — center tap matches a legacy
  single-hook capture **bitwise (diff 0.0)**, three layers distinct (closure
  binding correct). **PASS.**

The extraction and parity scripts themselves run on the H200 (need the real model
+ pinned `transformers==4.57.1`); the parity check is their designed validation.

```bash
# reproduce the local checks anywhere with numpy (+ torch for the MLP path):
python -m multilayer_nla.tests.test_fve_math
python -m multilayer_nla.headroom --selfcheck
```

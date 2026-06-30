# Progressive Multi-Layer Gold-Prefix AR — v0 implementation note

Reader-only experiment: **exact prefixes of the existing gold teacher explanation →
multi-tap reconstructor → raw source activations at a hierarchy of layers.** No AV
generation, injection, RL, joint AR–AV training, or external data. This note records the
**actual discovered repo paths/fields/classes** the implementation reuses (spec §3), the
construction decisions, the claim boundary, and the pre-registration.

## Claim boundary (preserved verbatim in every `summary.md`)
Results are **conditional gold-prefix reader ceilings** — `C_{B,ℓ}^{gold;H,D}` — conditional
on the current teacher-label distribution, AR architecture, source activation bank, exact
prefix budgets, target normalization, and optimization recipe. They are **not** absolute
information-theoretic ceilings, **not** evidence the teacher text is semantically faithful,
and **not** evidence reconstructed activations are causally faithful.

## Discovered data (the bank corpus, `$REGEN`)
The L19–L29 bank shards (`av_sft` / `ar_sft` / `rl` `.shard*of*.parquet`) — one row per
labeled (doc, position) — carry **everything this experiment needs**:

| field | source | use |
| --- | --- | --- |
| `response` (av_sft) | `<explanation>\n{summary}\n</explanation>` (`nla.schema.wrap_explanation`) | teacher text → `extract_explanation` |
| `prompt` (ar_sft) | `Summary of the following text: <text>{summary}</text> <summary>` | teacher text → strip prefix/suffix |
| `activation_L{19..29}` | RAW residual (`norm="none"`), final token | targets `L20,22,23,24,25,26,28` (all present) |
| `doc_id` | published | document-level split |
| `n_raw_tokens`, `detokenized_text_truncated`, `center_layer` | published | provenance / round-trip |

**Teacher token IDs are NOT stored.** The teacher text is API output; the canonical
token-ID source is **tokenizing that text once** with the Qwen tokenizer
(`add_special_tokens=False`), then slicing (spec §1.1). This is not a decode→re-tokenize
round-trip — there is no prior Qwen ID sequence to drift from. We persist the IDs + a
SHA-256 (`prefix.teacher_ids_sha256`) and validate slices against it.

Recommended corpus: `av_sft` (most rows, ~216k) — but the audit auto-detects the field and
any corpus works (they all carry teacher text + all 7 layers + `doc_id`).

## Reused classes / conventions
- **Split:** `multilayer_nla.datasets.doc_bucket(doc_id, fracs, seed)` — stable hash, **seed
  42**, fracs `0.8/0.1/0.1`, document-level (every stage view of a row lands in one bucket).
- **FVE:** `nla.schema.compute_predict_mean_baselines(targets, mse_scale)[1]` = `mse_rawvar`
  (the repo's `fve_nrm`), with `nla.schema.normalize_activation(v, mse_scale)` =
  `mse_scale·v/‖v‖` (directional). `FVE = 1 − MSE(pred_norm, tgt_norm)/mse_rawvar`,
  `mse_scale = √d = 64`. **Centering/baseline stats come from TRAIN only** (spec §1.4/§9).
- **AR to generalize:** `multilayer_nla.models_multi.MultiTapCriticModel` — truncated backbone
  (to `max(tap_layers)+1`), final-norm stripped, a per-tap forward hook capturing each block's
  output, and an **identity-init `Linear(d,d)` head per tap** reading the **last real token**
  (`attention_mask.sum(1)-1`). v0 generalizes `tap_layers` from `(23,24,25)` to
  `TARGET_LAYERS=(20,22,23,24,25,26,28)` → truncate to **29 blocks**, 7 heads. Default
  `reader_tap_by_target_layer` is identity (tap ℓ → target ℓ). No layer-query token, no
  low-rank/affine heads (spec §6).
- **Directional loss:** reuse `models_multi.three_target_loss` semantics — `normalize_activation`
  both pred & target, then MSE (= `2(1−cos)` at unit norm); active-set mean per stage (§7).

## Reader input (spec §4)
Fixed prompt `Explanation:\n` + exact teacher prefix IDs + fixed suffix `\n[RECONSTRUCT]`
(no new vocab in v0). Readout = the **final token of the suffix**, gathered per row at
`attention_mask.sum(1)-1` (never the padded end).

## Stage expansion (spec §5)
One base dataset (one row per activation record) + a lightweight **virtual** stage index:
each base row → exactly 3 stage views `(i,32,S_32) (i,64,S_64) (i,96,S_96)`. Activation
targets are **not** triplicated on disk; all three views expose byte-identical targets.

## The GATE (run first)
`python -m multilayer_nla.progressive_reader.audit --data "$REGEN/av_sft.shard*of*.parquet"
--base-ckpt Qwen/Qwen3-8B --out runs/progressive_reader_v0/data_audit.json`

Reports teacher-length quantiles, `coverage_at_budget{32,64,96}`, the strict-`n≥max`
retained rows/docs per split, target-layer presence, and doc-split overlaps. **If
`coverage@128 < 50%`, decide before building train/eval** (lower max budget / accept the
smaller strict set / use censored mode — not headline). These are API summaries, so this is
genuinely uncertain.


## AUDIT RESULT — budgets locked to {32, 64, 96}
The data_audit (216,570 rows, 24,974 docs) found gold explanations are short and tight: **median 112 tokens** (p10/p90 = 98/127, max 201). So **coverage@128 = 9.5%** (strict-128 would keep only ~10% of rows — the longest, a selection confound), while **coverage@32/64 = 100%** and **coverage@96 ≈ 91%**. Max budget dropped 128 → **96** (strict-96 keeps ~91%; train/dev/test ≈ shown in data_audit.json). All 7 target layers present, doc-split overlaps 0. Note for the pre-registration: the explanations being short and uniform further *weakens* the prior for a strong depth hierarchy (limited 'rate' to progressively reveal across a ~60-token span).

## Pre-registration
- **Hypothesis:** longer gold prefixes progressively unlock *deeper/outer* layers (a depth
  hierarchy), not just "more text helps everything."
- **Main alternative (predicted likely):** adjacent residual layers L20–L28 are collinear and
  all ~functions of the prefix, so reconstruction is **flat across depth** with a **uniform
  budget effect** and **no depth-specific unlocking** (`Progressive ≈ Flat`, `G_outer ≈
  G_local`). That would make the "hierarchy" an artifact of the imposed schedule.
- **Real evidence requires all three:** Progressive's outer-layer budget gain `G_outer`
  steeper than `G_local` and L24 already saturated at 32; `ΔG_outer` (Progressive−Flat) CI
  excludes 0 (paired doc-bootstrap); and `real − shuffled > 0` at those cells.
- **Headline-critical, not optional (§7):** run **both** `loss_mode`s. Stage-mean gives L24
  3× the gradient mass; only `progressive_layer_balanced` (weight 1/c_ℓ) separates a real
  hierarchy from direct-supervision imbalance.
- **Decision rule:** budget-monotone but depth-flat + `Progressive ≈ Flat` ⇒ **null on the
  hierarchy** (report as "depth-flat, budget-increasing, document-specific recoverability").
  Steeper outer gain + `ΔG_outer>0` + text-dependent ⇒ hierarchy is real ⇒ proceed.

## Status
- **Done (this slice):** `schedule.py` (nested-schedule validation), `prefix.py` (exact-prefix
  + SHA-256), `controls.py` (doc-level derangement), `audit.py` (the gate), stdlib tests
  (§13.1/§13.2/§13.8) — all passing.
- **Built:** `data.py`, `model.py`, `loss.py`, `train.py`, `evaluate.py` (+ H100-safe
  configs) — pipeline complete; budgets locked to {32,64,96} post-audit. The model/gradient/
  eval-matrix tests (§13.5–§13.7, §13.9) are exercised by the smoke run.

## Next experiment after this succeeds (do NOT auto-proceed)
AV-SFT against a **frozen** progressive AR: train an AV to emit text the frozen progressive
reader can invert per the depth schedule. Separate, pre-registered run.

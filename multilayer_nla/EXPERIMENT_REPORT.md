# §7 SFT Control Sweep — Experiment Report

All numbers below are read directly from `results/sft_control_sweep/` (the frozen eval
summaries); splits/baselines are from the code (`splits.py`, `build_sweep.py`,
`evaluate_e2e.py`); training configs are the trainer defaults plus the confirmed overrides
(authoritative per-run config is logged to **wandb** via `config=vars(args)` — see §iii).

**Substrate.** One pretrained base, **`Qwen/Qwen3-8B`, frozen.** Every model is a **LoRA
adapter** fine-tuned on top (parameter-efficient SFT) — nothing is trained from scratch. An
NLA = an **AV** (activation→text verbalizer) + an **AR** (text→activation reconstructor)
through a natural-language bottleneck. FVE = 1 − MSE(model) / MSE(predict-the-mean).

**Question (pre-registered).** Does multi-layer AV *input* improve end-to-end reconstruction
of the SAME fixed target state **[L23, L24, L25]**? Only the AV input layers vary; the AR
target is fixed for every condition. Headline test: `local` vs `duplicate`.

---

## i) Experiments — exact setup, data splits, baselines

### The 6 conditions (the AV input is the ONLY thing that varies)

| condition | AV input layers | distinct layers | span | markers (k) |
| --- | --- | ---: | ---: | ---: |
| local | 23,24,25 | 3 | 2 | 3 |
| duplicate | 24,24,24 | 1 | 0 | 3 |
| wide | 20,24,28 | 3 | 8 | 3 |
| single | 24 | 1 | 0 | 1 |
| s2_19_21_23 | 19,21,23 | 3 | 4 | 3 |
| s2_20_22_24 | 20,22,24 | 3 | 4 | 3 |

The condition lives in the **data** (positional `av_in_*` columns), not a train-time flag.
AR targets are the distinct columns `activation_prev/centre/next` (== L23/L24/L25) so the
target cannot follow the input. **Integrity:** verified 83/83 — every condition's
`rl_{dev,test}` AR target is byte-identical to `local`'s; `av_in_*` slots hold the right
layers; no split/corpus leakage.

### Three experiments (share the same 6 AVs, base, and splits)

**A. 3-tap multi-layer AV-input sweep (headline).** Per-condition AV → text → one **shared,
frozen 3-tap AR** → reconstruct [L23,L24,L25]. The AR is trained ONCE and is identical for
all conditions, so any difference is purely AV-side. End-to-end: AV emits TEXT only, AR
reconstructs from TEXT only (no activation crosses the bottleneck), greedy decoding.

**B. L24-only single-target cut.** Same 6 AVs, but a **single-target AR** (tap `[24]`)
reconstructs just **L24** — removes 3-tap averaging; the cleanest one-target version.

**C. AR-gold reconstructor ceiling.** Gold (API) explanation → shared 3-tap AR →
[L23,L24,L25] on the AR held-out sets. Localizes the bottleneck (verbalizer vs reconstructor).

### Data splits (document-level, stable `doc_bucket` hash, **seed 42**, fracs 0.8/0.1/0.1)

Three document-disjoint corpora from the L19–L29 bank (verified disjoint):

| corpus | role | split → sizes (rows) |
| --- | --- | --- |
| `av_sft` | AV SFT training | used **wholesale**, no internal split — **216,570 rows/condition** (AVs are selected on the rl-dev e2e eval, not an av-internal holdout) |
| `ar_sft` | AR training + gold ceiling | train `ar_common` **198,779** · dev `ar_dev` **24,563** · test `ar_test` **24,007** |
| `rl` | end-to-end eval | dev `rl_dev` **2,560 rows / 256 locked docs** · test `rl_test` **9,999 rows / 1,000 locked docs** (the rl-train 80% is reserved for the deferred RL phase; **unused here**) |

Never trained on dev/test; never selected on test. Locked subsets: `DEV_SUBSET=256`,
`TEST_SUBSET=1000` documents.

### Baselines (per evaluation)

1. **Predict-the-mean (the FVE denominator).** Computed on the **eval split only**. Per-tap
   predict-the-mean MSE (in `mse_scale`=64 units), from the test summaries:
   **prev 0.5970 · centre 0.5630 · next 0.5663**. FVE = 1 − model_MSE / this. FVE = 0 means
   "no better than predicting the mean activation."
2. **Shuffled-generation control.** Permute explanations **across documents**, re-score.
   Must collapse to ≈ 0 or strongly negative — observed **≈ −80%** in every condition
   (penalized), confirming the FVE is not distributional luck / leakage.
3. **Two FVE variants** reported everywhere: success-only (`fve_*`, parsed rows) and
   **failure-penalized** (`pen_fve_*`, parse-fail → 0). Bootstrap CIs resample **documents**
   (n_boot = 1000).

### Checkpoint selection (dev only)

Metric `pen_fve_overall` on `rl_dev`. Selected **shared AR = step 3000**, **per-condition
AV = step 1000** for all six (`selection.json`; mean dev metric at AR-3000 = 0.4227).
Nothing reads a test summary during selection.

---

## ii) Final FVE — dev and test

### A. 3-tap end-to-end (shared AR `ar_3tap_bs256e_3k/iter_0003000`, AV `iter_0001000`)

FVE % (success-only `overall` / `centre`; penalized `pen`). DEV = `rl_dev` (256 docs),
TEST = `rl_test` (1,000 docs).

| condition | AV in | DEV overall | DEV centre | DEV pen | TEST overall | TEST centre | TEST pen | TEST ext% | TEST shuffled |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local | 23,24,25 | 43.8 | 45.0 | 42.8 | 41.6 | 42.7 | 40.6 | 97.6 | −80.4 |
| duplicate | 24,24,24 | 41.1 | 42.4 | 40.3 | 39.9 | 41.2 | 39.0 | 97.6 | −80.3 |
| wide | 20,24,28 | 45.2 | 46.3 | 44.3 | 43.2 | 44.3 | 42.1 | 97.4 | −80.1 |
| single | 24 | 40.9 | 42.2 | 39.9 | 39.1 | 40.3 | 38.2 | 97.8 | −80.2 |
| s2_19_21_23 | 19,21,23 | 44.1 | 45.1 | 43.2 | 42.1 | 42.9 | 41.1 | 97.6 | −80.6 |
| s2_20_22_24 | 20,22,24 | 45.2 | 46.3 | 44.0 | 43.3 | 44.3 | 42.3 | 97.6 | −80.5 |

(Per-tap TEST, success-only prev/centre/next: local 38.9/42.7/43.2 · duplicate 37.0/41.2/41.7
· wide 40.7/44.3/44.8 · single 36.2/40.3/40.7 · s2_19_21_23 40.1/42.9/43.3 · s2_20_22_24
40.9/44.3/44.7.)

**Paired contrasts (TEST, Δ = A−B, penalized, bootstrapped over 1,000 shared docs; all
P(Δ>0)=1.000):**

| contrast | Δ (pp) | CI95 | isolates |
| --- | ---: | --- | --- |
| local − duplicate | **+1.63** | [+1.17, +2.08] | layer diversity @ fixed k=3 (PRE-REGISTERED) |
| duplicate − single | +0.81 | [+0.35, +1.28] | marker count @ fixed layer |
| wide − local | +1.50 | [+1.03, +1.99] | span among 3 distinct |
| s2_20_22_24 − local | +1.64 | [+1.19, +2.08] | stride-2 span vs narrow |
| s2_20_22_24 − s2_19_21_23 | +1.16 | [+0.73, +1.61] | target proximity @ equal span |
| wide − duplicate | +3.13 | [+2.64, +3.62] | diversity + span combined |

(Headline marginal CIs *overlap* — local pen 40.6 [39.7,41.6] vs duplicate 39.0 [38.0,39.9]
— but the **paired** test cleanly excludes 0; success-only +1.61 [1.18,2.04] ≈ penalized, so
not a parse-rate artifact.)

### B. L24-only single-target cut (AR `ar_l24only/iter_0003000`, tap [24]; TEST `rl_test`)

`overall ≡ centre` (one tap). Replicates the 3-tap ordering on a single fixed target.

| condition | TEST centre | TEST pen | ext% |
| --- | ---: | ---: | ---: |
| local | 42.7 | 41.7 | 97.6 |
| duplicate | 41.0 | 40.0 | 97.8 |
| wide | 44.2 | 43.1 | 97.6 |
| single | 40.0 | 39.2 | 98.0 |
| s2_19_21_23 | 42.8 | 41.8 | 97.6 |
| s2_20_22_24 | 44.1 | 43.0 | 97.7 |

Paired contrasts (TEST, penalized, all P=1.000): local−duplicate **+1.66** [+1.20,+2.11] ·
duplicate−single +0.84 [+0.39,+1.34] · wide−local +1.42 [+0.94,+1.92] · s2_20_22_24−local
+1.34 [+0.87,+1.80] · s2_20_22_24−s2_19_21_23 +1.29 [+0.85,+1.73] · wide−duplicate +3.08
[+2.56,+3.59].

### C. AR-gold reconstructor ceiling (gold explanation → shared 3-tap AR)

| split | rows | FVE prev | FVE centre | FVE next | FVE overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ar_dev | 24,563 | 60.0 | 63.7 | 64.8 | 62.8 |
| ar_test | 24,007 | 59.7 | 63.2 | 64.3 | 62.4 |

**Bottleneck (TEST e2e overall vs 62.4 ceiling):** local 41.6 (gap 20.8pp) · duplicate 39.9
(22.4) · wide 43.2 (19.2) · single 39.1 (23.3) · s2_19_21_23 42.1 (20.3) · s2_20_22_24 43.3
(19.1). The verbalizer is the dominant bottleneck (~19–23pp) — ~5× the entire
across-condition spread (~4.2pp).

---

## iii) wandb runs & training configs (final experiments)

**wandb project:** `multi layer nla` (from `run_sweep.sh` / `CLAUDE.md`). Both trainers log
the FULL arg set per run via `wandb.init(name=…, config=vars(args))`
(`train_ar_multi.py:202`, `train_av_multi.py:193`) — that is the authoritative config. The
`run name` below is the checkpoint `--save-dir` basename (exact, from the eval summaries'
`ar_ckpt`/`av_ckpt` paths); confirm the literal `--wandb-name` against the wandb project.

| model | role | ckpt (save-dir / step) | wandb run | batch × accum | steps | other |
| --- | --- | --- | --- | --- | --- | --- |
| shared 3-tap AR | reconstruct [23,24,25] | `ar_3tap_bs256e_3k` / 3000 | `ar_3tap_bs256e_3k` | **256 × 1** | **3000** | tap-layers 23,24,25 |
| L24-only AR | reconstruct [24] | `ar_l24only` / 3000 | `ar_l24only` | (confirm on wandb) | **3000** | tap-layers 24 |
| AV ×6 | verbalize | `av_<cond>` / 1000 | `av-<cond>` | **64 × 1** | **1000** | one per condition |

(`av_<cond>` ∈ local, duplicate, wide, single, s2_19_21_23, s2_20_22_24. The AR save-dir
name encodes its recipe: `bs256e` = effective batch 256, `3k` = 3000 steps.)

### Shared hyperparameters (trainer defaults, identical across AR + AV unless noted)

From `train_ar_multi.py` / `train_av_multi.py` argparse defaults:

```
--base-ckpt Qwen/Qwen3-8B   --use-lora   --quant none
--lora-r 128   --lora-alpha 16   (lora_dropout 0.0, bias none, rslora)
--lr 3e-5   --min-lr 3e-6   --lr-warmup-steps 50   --max-grad-norm 1.0
--max-len 1024   --save-every 500   --seed 0
optimizer Adam(betas=0.9,0.95, weight_decay=0)   cosine-to-min LR schedule
AR only:  --strip-final-norm (True)   value/tap heads identity-init (mse_scale=64)
AV only:  --gradient-checkpointing (True)
```

Confirmed overrides vs defaults: **AR** batch 64→**256**, num-steps 1000→**3000**; **AV**
keeps batch **64**, num-steps **1000**. Exact per-run values (incl. any per-AV batch
differences for the originally-converged vs rebuilt adapters) are on wandb under each run's
`config`.

### Invocations (reconstructed)

```bash
# shared 3-tap AR
python -m multilayer_nla.train_ar_multi --base-ckpt Qwen/Qwen3-8B --use-lora --quant none \
  --parquet $SWEEP/ar_common.parquet --save-dir $CKPT/ar_3tap_bs256e_3k \
  --tap-layers 23,24,25 --eval-parquet $SWEEP/ar_dev.parquet \
  --num-steps 3000 --save-every 500 --batch-size 256 --gradient-accumulation-steps 1 \
  --seed 0 --wandb-project "multi layer nla" --wandb-name ar_3tap_bs256e_3k

# L24-only AR (single target)
python -m multilayer_nla.train_ar_multi --base-ckpt Qwen/Qwen3-8B --use-lora --quant none \
  --parquet $SWEEP/ar_common.parquet --save-dir $CKPT/ar_l24only \
  --tap-layers 24 --eval-parquet $SWEEP/ar_dev.parquet \
  --num-steps 3000 --save-every 500 --seed 0 \
  --wandb-project "multi layer nla" --wandb-name ar_l24only

# each AV (condition lives in the data; only --parquet/--save-dir/--wandb-name change)
python -m multilayer_nla.train_av_multi --base-ckpt Qwen/Qwen3-8B --use-lora --quant none \
  --parquet $SWEEP/av_<cond>.parquet --save-dir $CKPT/av_<cond> \
  --num-steps 1000 --save-every 500 --batch-size 64 --gradient-accumulation-steps 1 \
  --seed 0 --wandb-project "multi layer nla" --wandb-name av-<cond>
```

### Eval invocation (per condition, all final evals)

```bash
python -m multilayer_nla.evaluate_e2e --base-ckpt Qwen/Qwen3-8B \
  --av-ckpt $CKPT/av_<cond>/iter_0001000 --ar-ckpt $CKPT/ar_3tap_bs256e_3k/iter_0003000 \
  --eval-parquet $SWEEP/rl_{dev,test}_<cond>.parquet --condition <cond> \
  --batch-size 1024 --seed 0 --out <...>.jsonl --summary <...>.json
# L24-only cut: same, but --ar-ckpt $CKPT/ar_l24only/iter_0003000 on rl_test
# AR-gold ceiling: multilayer_nla.eval_ar_gold --ar-ckpt <AR> --eval-parquet ar_{dev,test}.parquet
```

---

## Summary of findings

Multi-layer AV input **improves** reconstruction of the fixed target, via three additive,
individually-significant mechanisms (all paired CIs exclude 0, confirmed on both 3-tap and
single-target): **layer diversity** (+1.63pp, headline), **span** (+1.50pp), **target
proximity** (+1.16pp); raw marker count alone is smallest (+0.81pp). Effects stack
(`wide − duplicate` +3.13 ≈ diversity + span). The effect is **real but secondary** — the
~4pp across-condition spread sits under a ~20pp verbalizer-vs-ceiling gap. Scope: SFT only
(no RL); warm-start labels are layer-blind single-layer L24, so this is "input diversity
helps the verbalizer even without layer-aware supervision."

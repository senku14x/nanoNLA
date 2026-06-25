# §7 SFT control sweep — status & handoff

_Last updated: 2026-06-25. Branch `multilayer_working`, PR #3 (draft). This doc is the
single source of truth for the current experiment; read it before continuing._

---

## 1. Where this came from (the pivot)

We had a **coherent multi-layer RL run** reporting ~48% FVE — but that number was
measured on the **training rollouts**, with **no held-out set**. It could not
distinguish "the multi-layer patch carries recoverable signal" from "the
reconstructor memorized these documents." So we **stopped the RL run** and pivoted.

Because RL is the expensive part, the plan is now: **build a held-out eval set and
compare the cheap SFT warm-starts across conditions FIRST**, and only pay for RL if
the warm-start comparison justifies it. This became a **pre-registered SFT control
sweep** (one H200, sequential, no RL, no re-extraction).

## 2. The experiment

**Core question:** does the information available to the AV from **multiple layers**
improve end-to-end reconstruction of the **same fixed nearby target state**?

**The load-bearing invariant:** the AR reconstruction target is **FIXED at
[L23, L24, L25] for every condition**. Only the AV *input* slots vary. So any FVE
gap between conditions is attributable to the input, not a moving target.

| condition | AV input layers | k / prompt | AR target | role |
| --- | --- | --- | --- | --- |
| **local** | 23, 24, 25 | 3 / 3-marker | 23, 24, 25 | ceiling (AV sees the exact target) |
| **duplicate** | 24, 24, 24 | 3 / 3-marker | 23, 24, 25 | **primary control** — only the centre |
| **wide** | 20, 24, 28 | 3 / 3-marker | 23, 24, 25 | wider span (secondary; span≠depth confound) |
| **single** | 24 | 1 / 1-marker | 23, 24, 25 | single-layer baseline (secondary; marker-count confound) |

- **Headline test: `local` − `duplicate`** = the marginal value of actually seeing
  the neighbour layers (L23, L25) in the input, vs inferring them from L24 alone —
  read out through the natural-language explanation bottleneck.
- The AV emits **text only**; the AR reconstructs **from text only**. No activation
  crosses between them, so "av_in == ar_target" in `local` is a round-trip through
  language, not a copy. `local` is the best-case ceiling, not leakage.
- `wide`'s L28 is **not leakage**: the held-out split is by document (L28 of a test
  doc is as held-out as its L23-25), and nothing flows but text. It carries a
  span-vs-depth *confound*, which is why it is secondary; the headline (local vs
  duplicate, both ≤ L25) is untouched by it.

## 3. Pipeline & data flow

The regenerated **wide bank** (`$REGEN`, columns `activation_L19 … activation_L29` +
`doc_id` + labels) already exists — **we do NOT regenerate**. Everything the sweep
needs (layers 20, 23, 24, 25, 28) is inside 19-29, so re-probing any combination
within that window is a CPU re-build (`build_sweep`), never a GPU re-extraction.

```
wide bank (L19-29 shards, doc_id)
  └─ splits.py ──────────────► rl 80/10/10 + ar 80/10/10  (doc-level, stable hash, manifests)
  └─ build_sweep.py ─────────► ar_common/dev/test  (prompt + targets prev/centre/next = L23/24/25)
                               av_<cond>            (k-marker prompt + response + av_in_* = cond layers)
                               rl_dev/test_<cond>   (av_in_* + targets + doc_id + src_row_id)   + PREFLIGHT
  └─ train_ar_multi.py ──────► ONE shared AR (ar_train, 1000 steps, gold dev FVE logged)
  └─ train_av_multi.py x4 ───► av-local/duplicate/wide/single (matched settings, 1000 steps)
  └─ evaluate_e2e.py ────────► AV gen → extract → shared AR → FVE vs FIXED [23,24,25]
  └─ eval_ar_gold.py ────────► gold explanation → shared AR → [23,24,25]  (reconstructor ceiling)
  └─ select_and_report.py ──► dev-only ckpt selection → one-shot test → result_table.md
```

**Column contract (do not conflate):**
- AV input slots = positional `av_in_0 … av_in_{k-1}` (k=3 or k=1).
- AR targets = `activation_prev/centre/next` (== L23/L24/L25), **always**.
- Distinct names are what stop the target from silently following the input.

## 4. What's DONE (all committed + offline-tested)

- **`datasets.py`** — contract: `av_in_*` slots, `detect_av_slots` (legacy
  prev/centre/next fallback for the smoke pipeline), `build_av_prompt(k)` with a
  1-marker `single` template, `stack_slot_vectors(rows, slot_cols)`,
  `prepare_av_chunk_multi(..., slot_cols)`. **Removed** the old
  `apply_condition_columns` / `CONDITIONS` / `doc_holdout` (the condition lives in
  the data now, not a train-time transform).
- **`splits.py`** — doc-level 80/10/10 (reuses `doc_bucket`), JSON manifest (seed,
  doc/row counts, per-bucket doc_id-set hashes), disjointness asserted, optional
  **locked dev/test subsets** for the expensive rollout eval.
- **`build_sweep.py`** — independent `--av-slot-layers` / `--ar-target-layers`;
  builds all datasets; **preflight `assert_conditions`** (source-row identity +
  byte-identical AR targets + per-slot absolute layer identity across
  local/duplicate/wide + prompt identity + dev/test + cross-subset disjointness) and
  **`assert_marker_counts`** (renders each prompt through the real tokenizer, asserts
  k markers; gated on `--base-ckpt`).
- **Trainers** — `train_av_multi.py` is slot-count aware (k from `detect_av_slots`
  drives the hook + prep); `train_ar_multi.py` trains the shared AR + `--eval-parquet`
  gold dev/test FVE. Both dropped `--condition`.
- **`evaluate_e2e.py`** — greedy AV generation with injection (k slots) → extract →
  shared AR → per-tap FVE vs fixed targets. Two overall variants (success-only;
  failure-penalized = mean-predictor/FVE 0), extraction rate, gen-token stats,
  per-example JSONL, **bootstrap CIs resampling documents**, **document-level shuffled
  control**, baselines from the eval split only. Empty/whitespace extraction counts
  as failure.
- **`eval_ar_gold.py`** — AR-only gold held-out FVE (reconstructor ceiling, isolates
  AV/extraction from the reconstructor).
- **`select_and_report.py`** — dev-ONLY selection (shared AR by mean dev FVE across
  the four conditions, then per-condition AV; asserts full grid + non-None) → result
  table.
- **`scripts/run_sweep.sh`** — one-H200 sequential run order 1-11, **resumable**
  (skips any step whose output exists), no RL, writes only to `sweep*` dirs.
- **Tests** — `test_datasets`, `test_build_sweep` (synthetic-bank build + preflight +
  layer-placement + fixed-target), `test_eval_select` (FVE aggregate, penalized,
  bootstrap doc-resample, dev selection, doc-derangement). **Full offline suite green.**
- **Adversarial verification workflow** (5 dimensions × independent verify):
  **13 findings, all confirmed, all fixed, none high-severity.** Fixes: shuffled
  control across documents; empty-extraction = failure; per-example gen length in
  tokens; preflight marker-count + absolute-layer + prompt-identity + cross-subset
  disjointness; selection full-grid/non-None guards.
- **Smoke run** validated the full pipeline end-to-end on tiny steps; **injection
  confirmed working** (English output, not CJK).

## 5. What's HAPPENING now

The **real sweep is running on the H200, in tmux**, into fresh dirs
(`$DATA/sweep`, `$DATA/sweep_ckpt`, `$DATA/sweep_eval`). Sequence: split → build
(+ preflight) → shared AR (1000 steps) → av local/duplicate/single/wide (1000 each)
→ 16-cell dev eval grid → dev-only selection → one-shot test → `result_table.md`.
wandb project **"multi layer nla"**, runs `ar-shared`, `av-local/duplicate/wide/single`
(eval steps write JSON, not wandb). The 5 SFT trainings are the long pole; if the box
dies, re-running the same command resumes.

## 6. What's NEXT

1. Read **`$DATA/sweep_eval/result_table.md`** — local vs duplicate held-out **test**
   FVE, with the AR-gold row (reconstructor ceiling) and the shuffled-control row
   (must collapse) beside it.
2. **Interpret honestly:** trust the *gap*, not absolute FVE (these are SFT
   warm-starts, no RL). If `local` clears `duplicate` with separated bootstrap CIs
   (and AR-gold is high) → the neighbour layers carry recoverable signal and **RL is
   justified**. If CIs overlap → "no detectable effect at this N," consider a larger
   `TEST_SUBSET`. AR-gold high but end-to-end low ⇒ the verbalizer/extraction is the
   bottleneck; AR-gold low ⇒ the reconstructor is.
3. **Pre-registration discipline:** do NOT change data, hyperparameters, prompts,
   layer layouts, or evaluator logic after seeing the test table.
4. **RL is deferred.** RL-per-condition is **not yet wired** — `train_rl_multi.py`
   still uses the old fixed 3-slot scheme (SLOT_COLUMNS, k=3) for both inject and
   reward. To RL the sweep conditions it needs the **same input/target decoupling**
   the evaluator got (inject `av_in_*` with k slots; reward against fixed
   [23,24,25]). The GRPO surrogate, Fix-4 KL reference, and multitap critic scoring
   all already exist — it's a contained follow-up, only when the sweep says signal is
   there to optimise toward.
5. Optional: gdrive backup of the bank + `sweep*` dirs via rclone.

## 7. How to run (paths & commands)

```bash
# pull
git fetch origin && git checkout multilayer_working && git pull origin multilayer_working

# env (point REGEN at the EXISTING L19-29 bank)
export DATA=/data/mlnla
export REGEN=$DATA/published_L24x_window
export BASE=Qwen/Qwen3-8B
export WANDB_PROJECT="multi layer nla"

# real run (in tmux), resumable, no RL
bash multilayer_nla/scripts/run_sweep.sh 2>&1 | tee "$DATA/sweep_run.log"
```

Knobs (env): `STEPS=1000`, `SAVE_EVERY=500` (ckpts at 500+1000), `DEV_SUBSET=256`,
`TEST_SUBSET=1000`, `SPLIT_SEED=42`, `SEED=0`. Fast path-check: `STEPS=20
SAVE_EVERY=10 DEV_SUBSET=8 TEST_SUBSET=8 SWEEP=$DATA/sweep_smoke
CKPT=$DATA/sweep_smoke_ckpt EVAL=$DATA/sweep_smoke_eval bash ...run_sweep.sh`.

## 8. Guardrails (do not break)

- AR target stays **[23,24,25]** for every condition. The condition is in the data
  (`av_in_*`), never a train-time flag. `--ar-target-layers` must remain 23,24,25.
- Never train on `rl_dev`/`rl_test`/`ar_dev`/`ar_test`; never select on test.
- Predict-the-mean baselines from the **eval split only** (never AR train rows).
- Bootstrap resamples **documents**; shuffled control permutes **across documents**.
- No RL, no regen, no new objectives/penalties/architecture changes in this work.
- The smoke artifacts (`smoke/`, `ckpt/`, `sweep_smoke*`) and the sweep dirs
  (`sweep`, `sweep_ckpt`, `sweep_eval`) are separate — never overwrite.
- Loudest injection smoke test: grep generated text for CJK (silent injection
  failure ⇒ the actor free-associates Chinese).

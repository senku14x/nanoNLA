# §7 SFT control sweep — status & handoff

_Last updated: 2026-06-28. **Sweep COMPLETE** (converged run; 6 conditions). Full numbers +
configs: `multilayer_nla/EXPERIMENT_REPORT.md`. This doc = status, design & handoff; the
Results section below is the headline, the rest is the design / how-to-run record._

---

## 0. RESULTS (held-out TEST, 1,000 docs) — sweep COMPLETE

**Multi-layer AV input improves reconstruction of the fixed [L23,L24,L25] target — real,
significant, but modest.** The conclusion rests on the **paired bootstrap over shared
documents** (marginal CIs overlap; the paired Δ does not). Test overall FVE:

| condition | AV input | test overall FVE | paired Δ vs duplicate (95% CI) |
| --- | --- | ---: | --- |
| single | 24 | 39.1 | −0.8 [−1.3, −0.3] |
| duplicate | 24,24,24 | 39.9 | (control) |
| local | 23,24,25 | 41.6 | +1.6 [+1.2, +2.1] |
| s2_19_21_23 | 19,21,23 | 42.1 | +2.2 |
| wide | 20,24,28 | 43.2 | +3.1 [+2.6, +3.6] |
| s2_20_22_24 | 20,22,24 | 43.3 | +3.3 |

- **Headline local − duplicate = +1.63pp [+1.17, +2.08]** (layer diversity at fixed k=3 markers).
- Additive decomposition (all paired CIs exclude 0): marker-count +0.8, **diversity +1.6**,
  span +1.5; they stack.
- Diversity helps even WITHOUT the target layer (`s2_19_21_23`, all <L24, beats single/duplicate);
  proximity adds more (`s2_20_22_24` − `s2_19_21_23` = +1.16 [+0.73, +1.61]).
- **Replicates on a single-target L24-only AR** (local−duplicate +1.66 [+1.20, +2.11]) → not a
  3-tap-averaging artifact; the L24-only AR ≈ the 3-tap AR's L24 (no task contention).
- **Bottleneck is the verbalizer, not the AR.** The ~4pp across-condition spread sits ~20pp
  below the AR-gold ceiling (test overall 62.4; prev/centre/next 59.7/63.2/64.3). SFT only;
  warm-start labels are layer-blind (single-layer L24).
- **Integrity:** `verify_sweep_integrity.py` 83/83 — fixed-target byte-identical across all 6,
  slot layers correct, shuffled control ≈ −80%, dev/test + corpus disjoint.
- **Note (data fix):** `av_s2_19_21_23.parquet` first built truncated (74,101 rows vs 216,570,
  a partial `--mode av` build); rebuilt to full size, AV retrained + re-evaled. Result stable
  (42.0 → 42.1), so the conclusion is unaffected.

**Selected ckpts:** shared 3-tap AR `ar_3tap_bs256e_3k/iter_0003000` (batch 256, 3000 steps);
per-condition AV `av_<cond>/iter_0001000` (batch 64, 1000 steps); L24-only AR
`ar_l24only/iter_0003000`. **Published:** model `senku21x/qwen3-8b-nla-multilayer-sweep` (8
adapters on frozen Qwen3-8B); results + datacard in dataset
`senku21x/qwen3-8b-nla-multilayer-L19-29` @ `results/sft_control_sweep/`.

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
| **local** | 23, 24, 25 | 3 / 3-marker | 23, 24, 25 | adjacent triplet (AV sees the exact target) |
| **duplicate** | 24, 24, 24 | 3 / 3-marker | 23, 24, 25 | **primary control** — only the centre, ×3 |
| **wide** | 20, 24, 28 | 3 / 3-marker | 23, 24, 25 | wider span (span vs depth) |
| **single** | 24 | 1 / 1-marker | 23, 24, 25 | single-layer baseline (marker-count confound) |
| **s2_19_21_23** | 19, 21, 23 | 3 / 3-marker | 23, 24, 25 | stride-2, all below target (diversity without L24) |
| **s2_20_22_24** | 20, 22, 24 | 3 / 3-marker | 23, 24, 25 | stride-2, spans up to target (diversity + proximity) |

_The two `s2_*` conditions were added after the first 4; they're built per-mode (`--mode av/rl-eval`),
so re-run `verify_sweep_integrity.py` (or `build_sweep --mode preflight`) to gate them — the
canonical `--mode all` preflight only covers a single build._

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

## 5. What HAPPENED (run as executed — COMPLETE)

The converged run differs from the original `run_sweep.sh` defaults in three ways (all
recorded in the ckpt names / `EXPERIMENT_REPORT.md`):
- **AR retrained longer/bigger**: `ar_3tap_bs256e_3k` = batch 256, **3000** steps (vs the
  original undertrained `ar/` at 1000 steps, which is NOT used by any final number). A
  matched **L24-only AR** `ar_l24only` (tap [24], 3000 steps) was also trained for the
  single-target cut.
- **6 AVs** `av_<cond>` at batch 64, 1000 steps (selected `iter_0001000`).
- Eval lives in `$DATA/sweep_eval_converged` (`test/`, `dev/`, `test_arL24/`); selection by
  dev `pen_fve_overall` → AR 3000 / AV 1000.

wandb project **"multi layer nla"**; runs named by save-dir (`ar_3tap_bs256e_3k`, `ar_l24only`,
`av-<cond>`) with full config logged via `config=vars(args)`. Integrity verified 83/83.
Results published to HF (see §0). The old undertrained `ar/iter_000{500,1000}` is the only
stale artifact on disk; it feeds nothing and can be deleted.

## 6. What's NEXT

The verdict is in (§0): multi-layer input helps but is **secondary** to the verbalizer
bottleneck (~20pp below the AR-gold ceiling). So the next levers attack the verbalizer / the
text channel, not the input-layer choice:

1. **Better warm-start labels** (cheap, data-side). Labels are currently layer-blind
   (single-layer L24) and next-token-flavoured. Two distinct experiments: (a) *causal /
   important-info-first* structure → should close the AV→gold gap; (b) *more-informative*
   labels → could raise the 62.4 gold ceiling itself. Test separately so you know which gap moved.
2. **Multi-token bottleneck** (controlled). Vary the AV token budget (e.g. 60/120/240) with
   difficulty controlled, and check if FVE rises — raises channel capacity directly, may
   interact with multi-layer input. NB observationally, longer current outputs score *worse*
   (difficulty confound), so this needs an intervention, not a correlation.
3. **RL with a held-out eval.** RL optimises the AV text directly against reconstruction — the
   one lever SFT lacks — and is the most direct attack on the 20pp gap. **Still not wired:**
   `train_rl_multi.py` uses the old fixed 3-slot scheme; it needs the same `av_in_*`/fixed-target
   decoupling the evaluator got (inject k `av_in_*` slots; reward vs fixed [23,24,25]). GRPO
   surrogate, KL reference, multitap critic already exist. Measure on a proper held-out set —
   the earlier ~48% was on training rollouts (untrustworthy).
4. **Pre-registration discipline:** the §7 test numbers are locked; do NOT retro-tune data,
   hyperparameters, prompts, or evaluator logic against them. New levers = new pre-registered runs.
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

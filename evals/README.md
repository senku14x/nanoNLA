# `evals/` — NLA evaluation harness

## The iron rule: never train on eval data

Every input in this package is **fully disjoint** from training data. Three guards:

1. **Doc-level disjointness (stage-1 invariant)**: `av_sft` / `ar_sft` / `rl`
   parquets are split by unique `doc_id` at stage 1. Held-out eval prompts
   come from rows **past `--eval-skip-rows`** of whichever parquet they share
   (typically `rl_shuf` rows ≥ 35000, past the RL trainer's `--max-rows 30000`
   training cursor).
2. **External corpora**: `karvonen_confusion` uses Karvonen's
   `investigations.json` / `verification.json`, which were never in the
   FineFineWeb stage-0 ingest. No `doc_id` overlap is possible by construction.
3. **CI test**: `tests/test_doc_disjoint.py` (TODO) loads every training
   parquet and asserts the intersection with the held-out slice is empty.

## Data sources

| Eval | Source | Path |
|---|---|---|
| `hallucination` | held-out `rl_shuf` rows | `rl_shuf.parquet` rows ≥ `--eval-skip-rows` |
| `karvonen_confusion` | external | `/home/celeste/shared/{investigations,verification}.json` |

## Adding a new eval

1. Create `evals/<my_eval>/eval.py` with a `class MyEval(Eval)` subclass.
2. Decorate with `@register("my_eval")` from `evals.registry`.
3. Add `fixtures/` (parquet or JSON) + `fixtures/README.md` explaining
   provenance + disjointness from training.
4. Wire into `nla/train_rl_self_contained.py`'s `--eval-suite` arg (TODO).

## Cost guardrails

Sonnet 4.6 judge calls aren't free. Defaults:

- `eval_every = 10` RL steps
- `n_samples = 20` per eval per step
- Judge model: **Sonnet 4.6** (per CLAUDE.md judge rule), not Opus, unless
  `--judge-model claude-opus-4-7` is passed explicitly
- `temperature = 0` on the judge (reproducibility)
- Anthropic Batches API for `--standalone` runs over >20 samples

Rough cost: `steps/10 * 20 * 1 eval * ~$0.003 per call ≈ $0.60 per 1k RL steps`.

## Reproducibility

Every `evaluate(step)` call writes:

```
<output_dir>/step_<NNNNNNN>/<eval_id>.json
```

with: rng seed, sample indices, source texts, generated explanations, judge
scores, judge reasons, judge model name, repo SHA (TODO). Re-runs with
identical seed + step + ckpt + judge `temperature=0` reproduce metrics modulo
Anthropic API non-determinism (rare for `temperature=0`).

## Standalone usage

```bash
python -m evals.run_evals \
  --av-ckpt /path/to/av_sft/iter_0001000/hf \
  --rl-lora /path/to/rl/iter_000500 \
  --parquet /path/to/rl_shuf.parquet \
  --sidecar /path/to/rl_shuf.parquet \
  --output-dir /path/to/eval_runs/smoke \
  --evals hallucination \
  --n-samples 40 \
  --step 500
```

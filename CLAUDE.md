# NLA — instructions for Claude / AI assistants

## Constraints

- **This is an open-source repo.** Only standard libs: `pathlib.Path`,
  `pyarrow`, `transformers`, `datasets`, `httpx`, `pyyaml`, `numpy`, `orjson`,
  `safetensors`, the public `anthropic` SDK.
  No private/internal dependencies.
- Use argparse for CLIs in `nla/`.
- Storage and completion-provider backends are pluggable via import-path
  strings (`--storage-cls`, `--provider-cls`). The shipped implementations are
  `LocalStorage` and `AnthropicProvider`. Cloud storage / other LLM APIs are
  bring-your-own — don't hardcode bucket paths or vendor SDKs into `nla/`.

## Key invariants (do not break these)

- **Data-gen NEVER normalizes** — all parquets store raw vectors
  (`norm="none"`). `stage3_build` asserts input `norm == "none"`. Normalization
  happens at injection time (`injection_scale`) and at loss time (`mse_scale`),
  both read from the sidecar.
- **Stage-1 split is DOCUMENT-level** — partition by unique `doc_id`, all rows
  from the same doc go to the same bucket. Never split positions from one doc
  across `av_sft` / `ar_sft` / `rl`.
- **Stage-0 `_MIN_POSITION = 50`** — need enough left-context for the
  activation to be meaningful. Earlier positions decode to noise.
- **Critic extraction is suffix-anchored** — no scan, no marker token. The
  critic prompt template ends with `... <summary>`; training extracts at
  `tokens[-1]`. `critic_suffix_ids` in the sidecar is for sanity-checking only.
- **Per-doc keyed RNG** — same `(seed, doc_id)` → same sampled positions
  regardless of chunk boundaries, slice ordering, or process count. This is
  what makes multi-GPU stage-0 sharding bit-reproducible.
- **Injection hook scans for the token ID inside the hook** (`inputs[0]`), not
  from precomputed positions. Batching/reordering means precomputed indices
  are wrong by construction.
- **Sidecar is the contract.** Token IDs, prompt templates, `injection_scale`,
  `mse_scale`, `d_model` — all loaded from `nla_meta.yaml` and asserted
  against the live tokenizer at startup. Never hardcode them.

## §7 SFT control sweep — CURRENT FOCUS (see `multilayer_nla/SWEEP_STATUS.md`)

The active experiment. A coherent RL run reported ~48% FVE but **on training
rollouts with no held-out set** — untrustworthy. We pivoted to a **pre-registered,
held-out SFT control sweep** (one H200, sequential, **no RL, no re-extraction**) that
compares the cheap SFT warm-starts first; RL only if the warm-start gap justifies it.
As of this writing the real sweep is **running** (branch `multilayer_working`, PR #3).

**Core question:** does multi-layer AV *input* improve end-to-end reconstruction of
the SAME fixed target state? Four conditions vary ONLY the AV input layers:
`local` [23,24,25] · `duplicate` [24,24,24] · `wide` [20,24,28] · `single` [24].
Headline test = **local vs duplicate**. `single`/`wide` are secondary (marker-count /
span-vs-depth confounds).

Invariants (do not break — they are what make the comparison causal):

- **AR reconstruction target is FIXED at [L23,L24,L25] for EVERY condition.** Only the
  AV input varies. `--ar-target-layers` must stay `23,24,25`.
- **The condition lives in the DATA, not a train-time flag.** AV input = positional
  `av_in_*` columns (k=3 local/duplicate/wide, k=1 single, via the 1-marker prompt);
  AR targets = `activation_prev/centre/next` (== L23/24/25). Distinct names so the
  target can't follow the input. There is NO `--condition` flag (the old
  `apply_condition_columns` transform was removed — it wrongly rewrote the target).
- **Shared AR is trained ONCE on `ar_train`, frozen, identical for all conditions.**
  Do NOT co-train AV+AR per condition — that gives each condition its own AR and
  confounds the gap (co-training == the deferred RL phase).
- **Document-level 80/10/10 splits** (`splits.py`, reuses `doc_bucket`): rl for
  end-to-end eval, ar for AR-only gold eval. Never train on dev/test; never select on
  test. Predict-the-mean baselines from the **eval split only**.
- **Evaluator (`evaluate_e2e.py`)**: AV emits TEXT only, AR reconstructs from TEXT only
  (no activation crosses). Two FVE variants (success-only; failure-penalized = FVE 0).
  Bootstrap CIs resample **documents**; the shuffled control permutes **across
  documents** and must collapse. AR-gold (`eval_ar_gold.py`) localises the bottleneck
  (verbalizer vs reconstructor).
- **Bank is L19-L29** (`$REGEN`); covers every sweep layer. Re-probing within 19-29 is
  a CPU `build_sweep` re-run; a layer outside 19-29 needs GPU re-extraction.
- Entry point: `scripts/run_sweep.sh` (resumable, no RL, writes only to `sweep*`
  dirs). RL-per-condition is NOT wired (`train_rl_multi` still uses the fixed 3-slot
  scheme; it needs the same `av_in_*`/fixed-target decoupling as the evaluator).

## RL training: two trainers

- The **verified recipe is single-GPU 4-bit** with HF `generate()` rollouts:
  `nla/train_rl_self_contained.py` (see `scripts/sbatch_rl_fixed.sh` for the
  working invocation). `nla/train_rl_vllm.py` is the faster multi-GPU path
  (vLLM rollouts).
- For vLLM rollouts: use `--tensor-parallel-size N` to spread the rollout
  engine across all GPUs. The HF trainable side stays on one GPU (LoRA's
  ~120M trainable params don't benefit from FSDP), so GPUs 1..N-1 are
  vLLM-only during training-time forward but used during rollout.
- Weight broadcast via `llm.collective_rpc("load_weights", ...)` handles
  TP-sharding internally — same API regardless of TP size.
- If we ever move to full fine-tuning instead of LoRA, sharded training
  (FSDP) becomes worthwhile and the weight-gather path needs the TRL
  `_sync_fsdp{1,2}_params_to_vllm()` treatment (`gather_if_zero3`,
  `summon_full_params`, etc.) — note for future-Claude.

## Debugging

If injection silently fails the actor sees the literal CJK marker char and
free-associates Chinese. Grep generated text for CJK — that's the loudest
smoke test for the entire injection path. Usual causes: the marker token ID
drifted (wrong tokenizer/sidecar), the hook never registered (wrong layer
attribute path), or the prompt template lost the marker char.

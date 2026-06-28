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

## §7 SFT control sweep — COMPLETED (full report: `multilayer_nla/EXPERIMENT_REPORT.md`; status/handoff: `multilayer_nla/SWEEP_STATUS.md`)

Done. A pre-registered, **held-out SFT control sweep** (one H200, sequential, **no RL,
no re-extraction**). We pivoted here because an earlier coherent RL run reported ~48% FVE
but on **training rollouts with no held-out set** (untrustworthy).

**Core question:** does multi-layer AV *input* improve end-to-end reconstruction of the
SAME fixed target [L23,L24,L25]? **Six** conditions vary ONLY the AV input layers (the AR
target is fixed for all): `local` [23,24,25] · `duplicate` [24,24,24] · `wide` [20,24,28] ·
`single` [24] · `s2_19_21_23` [19,21,23] · `s2_20_22_24` [20,22,24]. Baseline = `single`;
skyline = `AR-gold` (gold explanation → AR). Headline test = **local vs duplicate**.

**Result — multi-layer input helps, significantly but modestly.** Held-out TEST (1,000 docs);
**paired bootstrap over shared documents** is the valid test (marginal CIs overlap, the
paired Δ does not). Test overall FVE: single 39.1 · duplicate 39.9 · local 41.6 ·
s2_19_21_23 42.1 · wide 43.2 · s2_20_22_24 43.3; AR-gold ceiling 62.4.
- Headline **local − duplicate = +1.63pp, 95% CI [+1.17, +2.08]** (layer diversity at fixed k=3 markers).
- Additive decomposition: marker-count +0.8 (duplicate−single), **diversity +1.6**, span +1.5 (wide−local); they stack (wide−duplicate +3.1). All paired CIs exclude 0.
- Diversity helps even WITHOUT the target layer (`s2_19_21_23`, all <L24, still beats single/duplicate); proximity adds more (`s2_20_22_24` > `s2_19_21_23` +1.16 [0.73,1.61]).
- Replicates on a single-target **L24-only AR** (local−duplicate +1.66 [1.20,2.11]) → not a 3-tap-averaging artifact; the L24-only AR ≈ the 3-tap AR's L24 (no task contention).
- **Real but SECONDARY:** the ~4pp across-condition spread sits ~20pp under the AR-gold ceiling (62.4). The **verbalizer**, not the AR, is the dominant bottleneck. SFT only; warm-start labels are layer-blind (single-layer L24), so diversity helps even without layer-aware supervision. RL still deferred.
- Integrity: `verify_sweep_integrity.py` **83/83** (fixed-target byte-identical across all 6; shuffled control ≈ −80%; dev/test + corpus disjoint).

**Selected checkpoints / published artifacts.** Shared 3-tap AR `ar_3tap_bs256e_3k/iter_0003000`
(batch 256, 3000 steps); per-condition AV `av_<cond>/iter_0001000` (batch 64, 1000 steps);
L24-only AR `ar_l24only/iter_0003000`. All 8 LoRA adapters on frozen `Qwen/Qwen3-8B`.
Published: model repo `senku21x/qwen3-8b-nla-multilayer-sweep`; results + datacard in dataset
repo `senku21x/qwen3-8b-nla-multilayer-L19-29` under `results/sft_control_sweep/`.
Read-only analysis tooling: `analyze_sweep.py` (table + paired contrasts + leakage + qualitative),
`make_datacard.py`, `plot_sweep.py`, `verify_sweep_integrity.py`.

Invariants (still the contract for any re-run; they are what made the comparison causal):

- **AR reconstruction target is FIXED at [L23,L24,L25] for EVERY condition.** Only the
  AV input varies. `--ar-target-layers` must stay `23,24,25`.
- **The condition lives in the DATA, not a train-time flag.** AV input = positional
  `av_in_*` columns (k=3 for local/duplicate/wide/s2_19_21_23/s2_20_22_24; k=1 single,
  via the 1-marker prompt); AR targets = `activation_prev/centre/next` (== L23/24/25).
  Distinct names so the target can't follow the input. There is NO `--condition` flag (the
  old `apply_condition_columns` transform was removed — it wrongly rewrote the target).
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
- Entry point: `scripts/run_sweep.sh` (resumable, no RL, writes only to `sweep*` dirs).
  Post-run, read-only analysis/reporting: `analyze_sweep.py`, `verify_sweep_integrity.py`
  (the report-all integrity gate), `make_datacard.py`, `plot_sweep.py`; publish via
  `scripts/push_to_hf.sh`. RL-per-condition is NOT wired (`train_rl_multi` still uses the
  fixed 3-slot scheme; it needs the same `av_in_*`/fixed-target decoupling as the evaluator).

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

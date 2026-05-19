# open-NLAs

Open-source training pipeline for **Natural Language Autoencoders** — a
minimal, self-contained fork of
[`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders).
Original work:
**[Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html)**
(Fraser-Taliente et al., Transformer Circuits 2026).

📄 [Blog post](https://www.anthropic.com/research/natural-language-autoencoders) · ▶ [Video walkthrough](https://www.youtube.com/watch?v=j2knrqAzYVY) · 🔬 [Released NLAs on Neuronpedia](https://www.neuronpedia.org/nla)

---

A Natural Language Autoencoder is a pair of fine-tuned LMs that map
residual-stream activation vectors to natural language and back:

| | direction | mechanism |
|---|---|---|
| **AV** (activation verbalizer) | `vector → text` | inject the vector as a single token embedding into a fixed prompt, autoregress a description |
| **AR** (activation reconstructor) | `text → vector` | truncated K+1-layer LM + `Linear(d, d)` head, extract at the final token |

Both vectors are L2-normalised before comparison, so the round-trip
`MSE(reconstructed, original) = 2(1 − cos)` measures direction agreement only.
Low MSE means the AR could recover the original direction from the AV's words
alone, which implies the explanation captures the information in the vector.

**What's different from upstream**:
- **No Miles dependency** — AV-SFT, AR-SFT, and GRPO RL each run from a
  single self-contained file (`nla/train_sft.py`, `nla/train_rl_self_contained.py`).
  Drops ~2,200 LOC of plugin-path wiring.
- **Single-GPU first** — fits Qwen3-8B full FT on one H200 via
  `bitsandbytes.AdamW8bit` + bf16 + gradient checkpointing. Multi-GPU and
  vLLM-rollout paths exist as alternates.
- **Karvonen-style injection** — ADD norm-matched residual injection on layer 1
  (`h'_p = h_p + ‖h_p‖ · v/‖v‖`) instead of embedding-replace; better val NLL
  and a much easier inference path (no `input_embeds` plumbing).
- **Co-trained AR with `value_head(normalize(last_hidden))`** — bounds the
  value-head's input norm so bf16+Adam can't blow up the output norm
  (the failure mode that NaN'd AR SFT 8+ times in upstream).

## Released checkpoints

This fork's primary target is **Qwen3-8B layer 24**:
- AV-SFT (Karvonen injection): val NLL 1.82 (vs 1.86 baseline embedding-replace)
- AR-SFT (frozen value-head identity init): trains stable for 1000 steps
- **RL GRPO: peak held-out FVE 71.8%** on doc-disjoint eval set
  (started from 35.9% — +35.9 pp).

Upstream's four-model release (Qwen2.5-7B, Gemma-3-12B/27B, Llama-3.3-70B)
lives at [`kitft/nla-models`](https://huggingface.co/collections/kitft/nla-models) on the HF Hub.

## Repo layout

```
nla/
  train_sft.py                  ← --mode {av,ar} SFT trainer (single file, ~555 LOC)
  train_rl_self_contained.py    ← GRPO with HF generate (primary RL path)
  train_rl_vllm.py              ← GRPO with vLLM rollout + TRL-style weight broadcast (alternate)
  injection.py                  ← Karvonen ADD norm-matched residual injection
  models.py                     ← NLACriticModel (truncated K+1 + value_head)
  schema.py, config.py          ← sidecar contract (token IDs, prompt templates, scales)
  arch_adapters.py, storage.py
  datagen/                      ← 5-stage activation → parquet pipeline (extract → split → judge → join → build)
scripts/
  sbatch_datagen.sh             ← Stage 0–3: build the SFT/RL parquets from FineFineWeb
  sbatch_av_sft.sh, sbatch_ar_sft.sh   ← SFT entry points
  sbatch_rl_long.sh             ← GRPO RL entry point
  sbatch_rl_vllm.sh             ← alternate vLLM-rollout RL
  sbatch_eval_post_rl.sh + eval_post_rl.py   ← held-out reconstruction reward
  compute_fve_baseline.py       ← reward → FVE conversion (predict-the-mean baseline)
  verify_lora_disable_eq_sft.py ← test: peft.disable_adapter() == fresh AV-SFT load
  upload_dataset.py             ← publish Stage-3 parquets to HF Hub
configs/datagen/                ← Qwen3-8B YAMLs (100k / 10k / quick-test)
configs/TRAINING_NOTES.md       ← profiling + LR scan notes (mostly Qwen2.5-7B reference)
nla_inference.py                ← standalone single-file inference client
CLAUDE.md                       ← codebase-specific notes for AI assistants
```

## Reproduction (Qwen3-8B end-to-end)

Assumes single H200 (141 GB) per stage. Cluster paths assume
`/workspace-vast/celeste/nla-experiments/`.

### 0. Data generation

```bash
sbatch scripts/sbatch_datagen.sh
# 100k FineFineWeb docs → Stage 0 (Qwen3-8B layer 24 activations)
# → Stage 1 (25% AV / 25% AR / 50% RL doc-level split)
# → Stage 2 (Sonnet 4.6 judges via Anthropic Batches API)
# → Stage 3 (av_train, av_val, ar_sft_shuf_clean, rl_shuf parquets)
# ~12h, ~$80 in batch-API tokens (one-time)
```

### 1. AV (Activation Verbalizer) SFT

```bash
sbatch scripts/sbatch_av_sft.sh
# Qwen3-8B base + Karvonen layer-1 injection hook.
# bf16 + AdamW8bit + FA2 + gradient checkpointing, batch=64.
# Cross-entropy on response tokens only. ~1.5h for 1000 steps on 1× H200.
# Writes HF format directly to qwen3_8b_L24_av_sft/iter_NNNNNNN/.
```

### 2. AR (Activation Reconstructor) SFT

```bash
sbatch scripts/sbatch_ar_sft.sh
# NLACriticModel = Qwen3-8B truncated to K+1=25 layers + Linear(d, d) value_head.
# Truncation happens in-script; identity-init the value head (critical — see notes).
# bf16 + AdamW8bit + SDPA, batch=64, no grad ckpt. ~50min on 1× H200.
# pred = value_head(normalize(backbone_last_hidden, mse_scale)).
# Loss = MSE on L2-normalised (pred, gold) at last token position.
```

### 3. GRPO RL

```bash
sbatch scripts/sbatch_rl_long.sh
# Single GPU. B=16 prompts × G=16 group samples = 256 rollouts/step.
# rsLoRA r=128 actor (122M trainable) on q/k/v/o; reference policy via
# peft.disable_adapter() (verified bit-identical to fresh AV-SFT load).
# Co-trained AR backbone+value_head (LR 5e-5).
# KL β=0.01 against AV-SFT init, PPO clip 0.2, k3 KL estimator.
# Reward = -mse_nrm on L2-normalised vectors (paper's monotonic-transform default).
# ~25h for 1000 steps. Saves LoRA every 100 steps.
```

### 4. Post-RL evaluation

```bash
sbatch scripts/sbatch_eval_post_rl.sh
# 128 prompts from rl_shuf rows past the training cursor (doc-disjoint
# from av_train AND past the trainer's --max-rows). Pre-RL vs post-RL
# reconstruction reward + extraction rate.
# Convert reward to FVE: python scripts/compute_fve_baseline.py ...
```

## Datasets

| stage | dataset | size | location |
|---|---|---|---|
| Corpus | [`m-a-p/FineFineWeb`](https://huggingface.co/datasets/m-a-p/FineFineWeb) | 100k docs sample | public HF Hub |
| Stage 0 — activations | 1.4M (doc, position, Qwen3-8B layer-24 activation) tuples | ~16 GB | regenerable via `sbatch scripts/sbatch_datagen.sh` |
| Stage 3 — SFT/RL parquets | `av_train`, `av_val`, `ar_sft_shuf_clean`, `rl_shuf` w/ Sonnet 4.6 explanations | ~4 GB | regenerable; `scripts/upload_dataset.py` to publish to HF |

## Paper-faithful invariants

- **GRPO advantage**: per-prompt group baseline `A_ij = (r_ij − μ_j) / σ_j`
- **Clipped surrogate**: `min(r·A, clip(r, 1±ε)·A)` with ε=0.2
- **KL estimator**: Schulman k3 — `exp(δ) − δ − 1` where `δ = ref_lp − new_lp`
- **KL anchor = AV-SFT init**: implemented via `peft.disable_adapter()` (verified
  bit-identical to a freshly-loaded AV-SFT checkpoint — see
  `scripts/verify_lora_disable_eq_sft.py`)
- **Reward = `-MSE(normalize(pred), normalize(gold))`**: range `[-2.0, 0]`,
  with `-2.0` as the FAILED-extraction floor (= MSE of orthogonal unit vectors)
- **Per-doc keyed RNG** in stage 0 — `(seed, doc_id) → same sampled positions`
  regardless of chunk boundaries or process count, so multi-GPU sharding is
  bit-reproducible

## Documented deviations from the paper

1. **LoRA actor (rsLoRA r=128, α=16) instead of full 8B fine-tune** for RL.
   Paper does full FT on 2× H100. Verified invariant: `disable_adapter()` is
   bit-identical to a fresh AV-SFT load, so the KL anchor exactly equals
   `D_KL(AV_φ ‖ AV_φ_init)`.

2. **AR `value_head` sees normalised input.** Paper does
   `pred = value_head(backbone_last_hidden)` directly. We do
   `pred = value_head(normalize(backbone_last_hidden, mse_scale))`. At
   identity init the two are equivalent. The normalize-before form bounds
   value_head's input norm so bf16+Adam can't blow up its output. Upstream
   works around the same numerical issue with FP32 master weights in Miles'
   FSDP setup — we use AdamW8bit and need the architectural guard.

3. **Smaller effective batch.** Paper RL: 128 × 4 = 512 / step. Ours: 16 × 16
   = 256 / step. Larger G gives lower-variance per-prompt advantage.

4. **HF `generate()` for rollout** (default) or vLLM with TRL-style colocate
   weight broadcast (alternate). Paper uses SGLang `input_embeds`. We
   benchmark vLLM at TP=4 with ~50% PPO clip frac at step 0 from kernel
   mismatch — usable but not strictly better than HF generate here. The
   HF-generate path is what produced our headline 71.8% FVE result.

## Key invariants for future agents

- **Data-gen NEVER normalises** — Stage 3 parquets store raw `activation_vector`
  with `norm="none"` in the sidecar. Normalisation happens at injection time
  (`injection_scale`) and at loss time (`mse_scale`). Never on data load.
- **Stage-1 split is DOC-LEVEL** — partition by unique `doc_id`. Every row
  from the same doc lands in the same bucket. Never split positions across
  AV / AR / RL.
- **Stage-0 `_MIN_POSITION = 50`** — earlier positions decode to noise.
- **Critic extraction is suffix-anchored** — no scan, no marker token. The
  critic prompt ends with a fixed suffix (e.g. `... <summary>`); training and
  inference extract at `tokens[-1]`. The sidecar's `critic_suffix_ids` is for
  sanity-checking only.
- **Per-doc keyed RNG** — `(seed, doc_id)` → same sampled positions regardless
  of process count or chunk boundaries. Bit-reproducible multi-GPU stage 0.
- **`cp_size == 1` only.** Context-parallel splits each sample across ranks
  and breaks the marker-neighbor check. NLA sequences are short; CP buys
  nothing.
- **Sidecar is the contract.** Token IDs, prompt templates, `injection_scale`,
  `mse_scale`, `d_model` — all loaded from `nla_meta.yaml` and asserted
  against the live tokenizer at startup. Never hardcode them.

## Debugging

If injection silently fails, the actor sees the literal CJK marker char (`㈎`)
and free-associates Chinese. **Grep generated text for CJK** — loudest smoke
test for the entire injection path.

## Citation

If you use this code or the released checkpoints, cite the original paper:

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and Mossing, Dan and Lu, Christina and Bogdan, Paul C. and Ameisen, Emmanuel and Chen, James and Kishylau, Dzmitry and Pearce, Adam and Tarng, Julius and Wu, Alex and Wu, Jeff and Zhang, Yang and Ziegler, Daniel M. and Hubinger, Evan and Batson, Joshua and Lindsey, Jack and Zimmerman, Samuel and Marks, Samuel},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Released checkpoints additionally inherit the
license of their base model (Gemma, Llama-3.3) — see the NOTICE files in each
HF repo.

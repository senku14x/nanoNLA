# open-NLAs

Open-source training pipeline for **Natural Language Autoencoders** — a minimal
fork of [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders).
Original work: **[Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html)**
(Fraser-Taliente et al., Transformer Circuits 2026).

📄 [Blog post](https://www.anthropic.com/research/natural-language-autoencoders) ·
▶ [Video walkthrough](https://www.youtube.com/watch?v=j2knrqAzYVY) ·
🔬 [Released NLAs on Neuronpedia](https://www.neuronpedia.org/nla)

---

## What's an NLA?

A Natural Language Autoencoder is a pair of fine-tuned LMs that map
residual-stream activation vectors to natural language and back:

| | direction | mechanism |
|---|---|---|
| **AV** (activation verbalizer) | `vector → text` | inject the vector as a single token embedding into a fixed prompt, autoregress a description |
| **AR** (activation reconstructor) | `text → vector` | truncated K+1-layer LM + `Linear(d, d)` head, extract at the final token |

Both vectors are L2-normalised before comparison, so the round-trip
`MSE(reconstructed, original) = 2(1 − cos)` measures **direction agreement**
only. Low MSE means the AR could recover the original direction from the AV's
words alone — i.e. the explanation captures the information in the vector.

## What's different in this fork

- **Self-contained.** No more Miles / SGLang dependency. The entire AV-SFT,
  AR-SFT, and GRPO RL pipeline runs from 3 single-file Python scripts.
- **Trained on Qwen3-8B layer 24** rather than the paper's Qwen2.5-7B layer 20.
- **bf16 + 8-bit Adam (bitsandbytes)** throughout to fit on a single H200.
- **Co-trained AR critic** (paper-faithful) with a normalize-before-value_head
  variant that fixes the AR-SFT NaN we hit on bf16+Adam.
- **`disable_adapter()` as the KL anchor** for the RL trainer, verified
  bit-identical to a fresh AV-SFT load (`scripts/verify_lora_disable_eq_sft.py`).
- **LoRA actor (rsLoRA r=128, α=16) for RL** to fit full Qwen3-8B + co-trained
  AR + activations on one GPU.

## Repo layout

```
nla/
  train_sft.py                  ← single-file AV+AR SFT (--mode av | ar)
  train_rl_self_contained.py    ← single-file GRPO with HF generate
  train_rl_vllm.py              ← alternate: GRPO with vLLM rollout + TRL-style colocate weight sync
  injection.py                  ← Karvonen ADD norm-matched injection + paper's embedding-replace
  models.py                     ← NLACriticModel (truncated K+1 backbone + Linear value_head)
  schema.py / config.py         ← sidecar contract (token IDs, prompt templates, scales)
  datagen/                      ← 4-stage parquet pipeline (corpus → activations → judges → SFT data)
scripts/                        ← sbatch wrappers + helper scripts (training + eval)
configs/                        ← datagen YAMLs + TRAINING_NOTES.md
nla_inference.py                ← standalone HF/SGLang inference for released checkpoints
```

## Datasets

| stage | dataset | size | location |
|---|---|---|---|
| **Corpus** | [`m-a-p/FineFineWeb`](https://huggingface.co/datasets/m-a-p/FineFineWeb) (67 domains, ~10 TB) | 100 k docs sampled | public on HF Hub |
| **Stage 0 — activations** | 1.4 M (doc, position, residual-stream activation @ Qwen3-8B layer 24) tuples | ~16 GB | regenerable via `scripts/sbatch_datagen.sh` |
| **Stage 3 — SFT/RL parquets** | `av_train`, `av_val`, `ar_sft_shuf_clean`, `rl_shuf` (with Sonnet 4.6 explanations) | ~4 GB | regenerable; `scripts/upload_dataset.py` publishes to HF Hub |

Stage 1 (doc-level 25/25/50 split into AV/AR/RL) and Stage 2 (Sonnet judging
via the Anthropic Batches API) are deterministic given corpus + seed.
Regeneration: `scripts/sbatch_datagen.sh` (requires `ANTHROPIC_API_KEY_BATCH`;
~12 h, ~$80 in batch-API tokens).

## Quick start: reproduce Qwen3-8B end-to-end

Single H200 unless noted. Set `HF_TOKEN` and `WANDB_API_KEY` in your shell.

```bash
# 0. Data (~12h on cluster, ~$80 Sonnet — skip if you have the parquets already)
sbatch scripts/sbatch_datagen.sh

# 1. AV SFT (~1.5h, batch=64, grad ckpt ON, Karvonen layer-1 injection)
sbatch scripts/sbatch_av_sft.sh

# 2. AR SFT (~50min, batch=64, grad ckpt OFF, truncated K+1=25 + identity-init value_head)
sbatch scripts/sbatch_ar_sft.sh

# 3. RL (GRPO, ~13-15h for ~500 useful steps, HF-generate rollout)
sbatch scripts/sbatch_rl_long.sh

# 4. Post-RL eval (~10min, pre-vs-post reward on doc-disjoint held-out prompts)
sbatch scripts/sbatch_eval_post_rl.sh
```

`scripts/sbatch_rl_vllm.sh` is an alternate vLLM-rollout variant (TRL-style
colocate weight broadcast every 20 steps + TIS clip). It works but vLLM/HF
kernel mismatch produces ~40% PPO-clip fraction; kept for reference, not the
default. See "Why HF generate over vLLM" below.

## Pipeline overview

```
Stage 0 (extract)            Stage 1 (split)            Stage 2 (judge)        Stage 3 (build SFT)
  FineFineWeb 100k docs   →  doc-level partition into  → Sonnet 4.6 via       → parquets:
  → 1.4M positions @           {av_sft, ar_sft, rl}      Batches API:           av_train.parquet
  layer 24, raw bf16         80/10/10 by doc_id         explanation per         av_val.parquet
  activations                                            (doc, position)        ar_sft_shuf_clean.parquet
                                                                                rl_shuf.parquet

Stage 4 (AV SFT)             Stage 5 (AR SFT)           Stage 6 (RL GRPO)
  Qwen3-8B + Karvonen        Truncated K+1=25 layer    LoRA actor + co-trained
  layer-1 ADD norm-match  →  backbone + Linear(d,d)  → AR critic; GRPO clipped
  injection. CE on response  value_head. MSE on        surrogate + k3 KL toward
  tokens. SFT on av_train    L2-normalised vectors     AV-SFT init.
```

## RL configuration

The reference recipe is `configs/rl.sh` (paper's Qwen2.5-7B run via Miles).
The self-contained `nla/train_rl_self_contained.py` matches it algorithmically:

| paper / `rl.sh` | open-NLAs self-contained | what it is |
|---|---|---|
| `--n-samples-per-prompt 4` | `--group-size 4` (we use 16 by default) | samples per prompt for the GRPO group baseline |
| `--advantage-estimator grpo` | per-prompt mean/std normalisation | group-relative advantage |
| `--use-kl-loss --kl-loss-coef 0.01` | `--kl-beta 0.01`, k3 (Schulman) estimator | KL penalty against reference |
| `--lr 1e-6 constant` | `--lr 1e-5` (LoRA wants ~10× higher) | actor LR |
| `--rollout-max-response-len 150` | `--max-new-tokens 150` | response cap |
| clipped surrogate (PPO-style) | `--clip-eps 0.2` + `min(ratio·A, clip(ratio,1±ε)·A)` | importance-ratio clip |
| FAILED extraction → `-2.0` | `-2.0` for missing `<explanation>` | matches `nla/reward.py` |

### Deviations from the paper

1. **LoRA actor (rsLoRA r=128, α=16)** instead of full 8B fine-tune. Paper FT
   on 2× H100; we LoRA-FT on 1× H200. Trainable: ~123 M (1.48 %). KL anchor is
   `disable_adapter()` — verified bit-identical to AV-SFT init via
   `scripts/verify_lora_disable_eq_sft.py` (max abs diff = 0.0, literally).
2. **AR critic: `value_head` sees normalized input.**
   `pred = value_head(normalize(backbone_last_hidden, mse_scale))` instead of
   the paper's direct `value_head(backbone_last_hidden)`. At identity init
   the two are equivalent (the loss re-normalizes anyway), but our variant
   bounds value_head's input norm. Without it, bf16+Adam on a near-identity
   value_head NaN'd AR SFT 8 + times. Paper uses FP32 master weights in Miles
   FSDP to work around the same problem differently; we use 8-bit Adam to fit
   on one GPU, so we need the architectural guard.
3. **Co-trained AR** (paper-faithful). Backbone + value_head both train
   against MSE on the same explanations the AV just produced. Critic LR 5e-5.
4. **B × G = 16 × 16 = 256 / step** vs paper's 128 × 4 = 512. Larger G gives
   lower-variance per-prompt advantage at the cost of fewer prompts/step.
5. **HF `generate()` instead of SGLang `input_embeds`.** ~5× slower per step
   but no infra, no off-policy drift. vLLM variant exists but kernel mismatch
   hurts more than vLLM speeds up — see below.

## Results (Qwen3-8B, layer 24, FineFineWeb 100k)

**Headline:** RL pushed held-out FVE from 35.9 % → **71.8 %** (peak at step
500, end of run before cluster storage filled).

| stage | wandb | result |
|---|---|---|
| AV SFT (Karvonen) | [`epu2zb0m`](https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/epu2zb0m) | val NLL 1.82 vs 1.86 baseline embedding-replace |
| AR SFT | [`8ea7vvfk`](https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/8ea7vvfk) | stable 1000 steps with normalize-before-value_head |
| RL GRPO (v4, 500 steps) | [`m08tbggs`](https://wandb.ai/adamkarvonen/nla-qwen3-8b/runs/m08tbggs) | held-out FVE 35.9 % → **71.8 %** (Δ **+35.9 pp**) |

Held-out eval uses 20 fixed prompts from `rl_shuf` rows 35000-35020 — past
the trainer's `--max-rows 30000` cursor AND doc-disjoint from `av_train` by
stage-1 invariant.

**FVE = 1 − mse_actual / baseline_mse**, where the baseline is the
predict-the-mean ceiling (`normalize(μ)` as constant pred). 0% = no better
than constant; 100% = perfect reconstruction. Paper's Qwen2.5-7B critic-SL
alone reports 37.5% — our AV-SFT alone hits 44.5%, and ~500 GRPO steps push
it to 71.8 %.

## What went wrong + the fixes (write-up for future-Claude)

| symptom | cause | fix |
|---|---|---|
| AR SFT NaN'd by step ~29 every retry (8+ attempts) | bf16+Adam on near-identity value_head blew pred-norm by 100× → normalize-then-MSE overflow | `pred = value_head(normalize(backbone_last_hidden, mse_scale))` in the forward; bounds value_head's input norm |
| RL step 0 had `loss=inf` then step 1 generate CUDA-asserted | HF `output_scores=True` returns post-`top_p` logits → filtered tokens get `-inf` old_logp → `ratio = exp(new_lp - old_lp) = inf` | Use `output_logits=True` (raw pre-processor logits); drop `top_p` to keep rollout/training distributions matched |
| `[data] loading …` hangs 5+ min | `.as_py()` on 100 k × 4096-float `activation_vector` column | Streaming row-group read + `.slice(0, take)` before `to_pylist()` |
| `peft 0.19` `ImportError` on `torchao 0.9` | recent peft requires torchao ≥ 0.16 | pin `peft==0.13.0` |
| OOM at B×G=256 with co-trained AR | retained autograd graphs from all micro-batches accumulating before backward | Fused per-microbatch forward+loss+backward (`grpo_update_microbatched` in `train_rl_self_contained.py`) |
| vLLM weight-sync: `'Worker' has no attribute 'load_weights'` | `collective_rpc("load_weights")` calls method on the Worker wrapper, not the model | Use `llm.apply_model(functools.partial(load_weights_chunk))` — module-level fn so it pickles |
| vLLM weight-sync: `msgspec EncodeError: data longer than 2^32-1` | 8 B-param state dict > 4 GB msgspec single-encode cap | Layer-by-layer chunked broadcast (prime-rl pattern) |
| vLLM weight-sync: `KeyError: '... .base_layer.weight'` | PEFT's `merge_adapter()` leaves `.base_layer.weight` in the state dict | Strip `.base_layer.` prefix before pushing to vLLM |
| AV-SFT held-out eval looked too optimistic | `av_val.parquet` was row-sub-sampled from `av_train.parquet` (99.99 % doc overlap) | Eval on `rl_shuf` rows past training cursor — doc-disjoint by stage-1 invariant |

## Why HF generate over vLLM

We tried vLLM rollouts with the TRL/prime-rl-style colocate weight broadcast
pattern (`nla/train_rl_vllm.py`). The plumbing works — actor.merge_adapter() →
layer-chunked `llm.apply_model(load_weights)` → unmerge — but at step 0 the
PPO clip fraction was **41 %** (vs HF generate's 0.5 %). vLLM and HF give
slightly different logprobs even at identical weights (different attention
kernels, different precision corners), and Truncated Importance Sampling
(TIS) cap=2.0 doesn't compensate enough at that scale. With `enforce_eager=True`
(required to coexist with HF training memory-wise) vLLM also loses its
CUDA-graph speedup. Net: kept it as documented alternative, not default.

## Datasets section

See **Datasets** above. The minimal reproduction inputs are:
- `m-a-p/FineFineWeb` (public, ~10 TB)
- Anthropic Batches API key (~$80 for Sonnet 4.6 judging at 100 k docs × 14
  positions each)

Output of `sbatch scripts/sbatch_datagen.sh` is what feeds Stages 4-6.
`scripts/upload_dataset.py --repo <your-hf-org>/nla-qwen3-8b-stage3`
publishes the Stage-3 parquets to HF Hub so others skip the 12 h of datagen.

## Citation

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

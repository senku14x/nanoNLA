# NLA method & released-code notes (reference)

Distilled from the released repo `kitft/natural_language_autoencoders`
(`docs/design.md`, `docs/inference.md`, `configs/TRAINING_NOTES.md`, README,
examples) and the paper *Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations* (Transformer Circuits, 2026). This is the
grounding for `src/lv_explainers/`. Where a claim is from a secondary summary
rather than the primary paper, it is marked **[secondary]**.

## What an NLA is
- **AV (Activation Verbalizer):** fine-tuned LM that maps one residual-stream
  activation `h` (single layer, ~2/3 depth) to free-text `z`. The activation is
  injected by replacing a rare marker token's embedding with the (rescaled)
  vector and feeding `input_embeds` (not `input_ids`).
- **AR (Activation Reconstructor / "critic"):** truncated backbone of `K+1`
  layers (`K` = read layer), final LayerNorm → Identity, `lm_head` stripped,
  plus `value_head = Linear(d, d, bias=False)`. Reconstruction is read at the
  **last token** of the critic-formatted text.
- Co-trained with GRPO RL; the AR is the live reward model.

## The reward is direction-only
Both predicted and gold activations are normalized to `mse_scale` (default
`sqrt(d_model)`), then mean-squared-difference. So
`reward = -MSE(q(ĥ), q(h)) = -2·(1 - cos(ĥ, h))` — **magnitude is discarded.**
Confirmed in code: `schema.normalize_activation` normalizes both sides;
`reward._mse_to_reward` takes the mean-squared diff; README states `MSE = 2(1-cos)`.
Reward variants: `-mse` (default) or `-log(mse)` (`NLA_LOG_MSE_REWARD=1`).
Extraction-failure reward: orthogonal value (`-2.0`, or `-log 2`).

## Training pipeline (4 stages)
0. **Datagen:** FineWeb docs → extract layer-L activations at ~10 positions/doc
   (min position ~50). An **oracle** (`claude-haiku-4-5`) writes initial
   explanations from the surrounding document context. Split 25% AV-SFT /
   25% AR-SFT / 50% RL. Vectors stored **raw / unnormalized**.
1. **AR (critic) SFT:** regress vectors from text, MSE on normalized vectors.
   `value_head` initialized to **identity** (`torch.eye(d)`), not random.
2. **AV SFT:** cross-entropy to imitate oracle explanations, activation injected.
3. **RL (GRPO):** AV generates; live AR scores; `reward = -MSE`. GRPO group-
   normalizes. KL-to-SFT penalty (coef ~0.01), response-length cap.

## Injection recipe (AV inference) — the gotchas
1. Tokenize once with `apply_chat_template(tokenize=True, add_generation_prompt=True)`
   (two-step path risks double-BOS on Gemma/Llama).
2. Embed, then scale: `embed_scale` = 1.0 for Qwen/Llama, **≈ sqrt(hidden)** for
   Gemma-3 (its `Gemma3TextScaledWordEmbedding` multiplies by sqrt(d) in-forward).
3. Rescale vector: `v_scaled = v_raw * injection_scale / ||v_raw||`. Find the
   marker token, **validate neighbors** `[left, marker, right]`, replace embedding.
4. SGLang `/generate` with `input_embeds` ONLY (never also `input_ids`);
   `--disable-radix-cache` (radix keys on token ids); Gemma needs `fa3` backend
   and the multimodal `input_embeds` patch or injection is silently dropped.
5. Parse `<explanation>...</explanation>`.

`injection_scale` varies wildly by model (Llama ~30 → Gemma-3-12B ~80000) because
residual-stream norms differ; **always load from the sidecar**, never hardcode.

## AR scoring (what Gate 0 needs)
AR scoring is a plain forward pass through the truncated backbone + value head —
**no SGLang**. Normalize both predicted and gold to `mse_scale` by hand
(`reconstruct()` returns raw). `MSE = 2(1-cos)`, range [0,4]; ~0.2 good, 1.0
mediocre, 2.0 orthogonal. FVE uses the **un-normalized** mean of normalized golds
in the denominator (do not normalize the mean).

## Released checkpoints (HF `kitft/nla-models`)
Qwen2.5-7B L20 d3584 · Gemma-3-12B L32 d3840 · Gemma-3-27B L41 d5376 ·
Llama-3.3-70B L53 d8192 — AV+AR each, with `nla_meta.yaml` sidecars. Released
FVE ~0.6–0.8. Single-GPU reproductions (syvb Qwen3-8B; Kameshr) reach ~0.3.

## Key results **[secondary]** (verify against primary before citing)
- Evaluation awareness surfaced internally far more than verbalized: ~16% on
  destructive-action tests, ~26% on a capability benchmark, <1% in normal use.
- Hidden motivations surfaced ~12–15% (vs <3% without).
- Vignettes: rhyme pre-planning; deliberation about avoiding detection while
  cheating. Limitation acknowledged: explanations **confabulate**; compute-heavy.

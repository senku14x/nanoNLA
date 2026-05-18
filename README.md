# open-NLAs

Open-source training pipeline for **Natural Language Autoencoders** — a fork of
[`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders)
trimmed down to a minimal, single-GPU-friendly impl, with a self-contained GRPO
RL trainer added. Original work:
**[Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html)**
(Fraser-Taliente et al., Transformer Circuits 2026).

📄 [Blog post](https://www.anthropic.com/research/natural-language-autoencoders) · ▶ [Video walkthrough](https://www.youtube.com/watch?v=j2knrqAzYVY) · 🔬 [Released NLAs on Neuronpedia](https://www.neuronpedia.org/nla) · 🧪 [Qwen3-8B reproduction guide](docs/qwen3_8b_run.md)

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

This is the **full training repo** — data generation, SFT, GRPO RL, and
checkpoint conversion. For a lightweight inference-only package (just
`NLAClient` + `NLACritic`, no training deps), see
[`kitft/nla-inference`](https://github.com/kitft/nla-inference).

> **A note on naming.** Public-facing names are **AV** / **AR**. Inside the
> `nla/` package you will see **actor** / **critic** — those are the same two
> models, named to map directly onto Miles' RL primitives (the AV *is* the
> policy actor; the AR *is* the value critic). The codebase keeps actor/critic
> so the Miles extension points read naturally; everywhere user-facing we use
> AV/AR.

---

## Released checkpoints

All eight checkpoints are gathered in the
**[`kitft/nla-models` collection](https://huggingface.co/collections/kitft/nla-models)**
on the HF Hub — four base-model families, each with an AV and an AR. We extract
from a layer roughly **two-thirds of the way through the model** in each case
— deep enough that the residual stream carries rich semantic content, shallow
enough that it hasn't yet collapsed toward the unembedding.

| base model | layer | d_model | AV | AR |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 20 / 28 | 3584 | [`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av) | [`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar) |
| Gemma-3-12B-IT | 32 / 48 | 3840 | [`kitft/nla-gemma3-12b-L32-av`](https://huggingface.co/kitft/nla-gemma3-12b-L32-av) | [`kitft/nla-gemma3-12b-L32-ar`](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar) |
| Gemma-3-27B-IT | 41 / 62 | 5376 | [`kitft/nla-gemma3-27b-L41-av`](https://huggingface.co/kitft/nla-gemma3-27b-L41-av) | [`kitft/nla-gemma3-27b-L41-ar`](https://huggingface.co/kitft/nla-gemma3-27b-L41-ar) |
| Llama-3.3-70B-Instruct | 53 / 80 | 8192 | [`kitft/Llama-3.3-70B-NLA-L53-av`](https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-av) | [`kitft/Llama-3.3-70B-NLA-L53-ar`](https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-ar) |

Each checkpoint ships an `nla_meta.yaml` sidecar with the prompt template,
injection token IDs, and scale factors that the model was trained with — load
those, never hardcode them.

---

## How it fits together

NLA training is built as a thin extension on top of two open-source projects:

- **[Miles](https://github.com/radixark/miles)** — Ray-orchestrated RL training
  (FSDP2 / Megatron backends, GRPO, async rollout). We used the FSDP backend
  for the 7B/12B/27B runs and Megatron only for Llama-70B. NLA plugs in via Miles'
  upstream `--custom-rm-path`, `--data-source-path`, and
  `--custom-generate-function-path` extension points; the integration patch in
  `nla/miles_patches/` adds `--custom-actor-cls-path` and `--force-use-critic`
  on top (see [docs/design.md §2](docs/design.md)).
- **[SGLang](https://github.com/sgl-project/sglang)** — rollout serving. We
  send `input_embeds` (not `input_ids`) so the AV sees the injected vector;
  SGLang serves it like any other request. The embed sequence is built on the
  **trainer side** — we look up the prompt tokens in the actor's own embedding
  table, splice the activation vector in at the injection slot, and ship the
  finished `[seq, d]` tensor over HTTP. SGLang never needs to know what an
  injection is. We don't apply any learned map to the injected vector in this
  work — it goes in raw (after a fixed scalar `injection_scale`) — but this
  design means a future affine `W·v + b` adapter would be a trainer-side-only
  change: apply it before sending, no SGLang modification required. (vLLM also
  supports `input_embeds` and would work as a drop-in alternative.)

We chose this stack because it is **near-frontier training infrastructure**:
Miles + Megatron is what production-scale RL post-training looks like, and
hooking onto it cleanly is what let us scale to RL-ing a 70B-parameter AV — and
likely further. The `nla/` package never modifies Miles or SGLang in place; it
only subclasses and registers function-pointer hooks, so upstream updates pull
in cleanly.

---

## Quick start

### Inference (use a released checkpoint)

```bash
uv pip install torch transformers safetensors httpx orjson pyyaml numpy
uv pip install "sglang[all]>=0.5.6"

python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 --disable-radix-cache &

python nla_inference.py kitft/nla-qwen2.5-7b-L20-av \
    --sglang-url http://localhost:30000 \
    --parquet path/to/activations.parquet
```

Don't have a parquet yet? Any file with an `activation_vector` column of
`d_model`-wide float lists will do — here's a minimal one for Qwen layer 20:

```python
import torch, pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16, device_map="cuda")
ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to("cuda")
hs = m(**ids, output_hidden_states=True).hidden_states[20][0]  # [seq, 3584]
pq.write_table(pa.table({"activation_vector": hs.float().cpu().tolist()}), "demo.parquet")
```

(Or omit `--parquet` entirely for a smoke test on a random unit vector.)

`nla_inference.py` is a single self-contained file. The full recipe —
model-specific scale factors, the Gemma `√d` embed-scale gotcha, debugging the
"output is in Chinese" failure mode, AR scoring — is in
**[docs/inference.md](docs/inference.md)**.

### Training (reproduce Qwen3-8B end-to-end)

Install Miles + this package per **[docs/setup.md](docs/setup.md)**, then run
the four stages. Full recipe + tuning notes in
**[docs/qwen3_8b_run.md](docs/qwen3_8b_run.md)**.

```bash
# 0. Generate data — 100k FineFineWeb docs + Sonnet 4.6 judges via Anthropic Batches API
sbatch launch/sbatch_datagen.sh

# 1. AV SFT (Karvonen layer-1 ADD norm-matched injection, our preferred variant)
sbatch launch/sbatch_av_sft_karvonen.sh

# 2. AR SFT (frozen-value-head variant — fixes the bf16+Adam NaN that hit step ~29 8+ times)
sbatch launch/sbatch_prepare_critic.sh
sbatch launch/sbatch_ar_safe.sh

# 3. RL: self-contained GRPO with co-trained AR, rsLoRA r=128 actor (single GPU, paper-faithful)
sbatch launch/sbatch_rl_long.sh
```

The full design — data transport through Miles' `multimodal_train_inputs`, the
injection forward-hook, simultaneous AV/AR scheduling — is in
**[docs/design.md](docs/design.md)**. Profiling and hyperparameter notes are in
[`configs/TRAINING_NOTES.md`](configs/TRAINING_NOTES.md) (Qwen2.5-7B reference)
and [`docs/qwen3_8b_run.md`](docs/qwen3_8b_run.md) (Qwen3-8B, this fork).

---

## Repo layout

```
nla/                            core package
  schema.py, config.py, models.py     — sidecar contract, NLACriticModel (AR)
  injection.py                        — Karvonen ADD norm-matched residual injection
  train_actor.py                      — NLAFSDPActor (Miles FSDP subclass, AV+AR SFT)
  train_rl_self_contained.py          — single-GPU GRPO with co-trained AR (new in fork)
  rollout/                            — SFT rollout helpers
  loss.py                             — AR MSE loss
  datagen/                            — 4-stage activation → parquet pipeline
configs/                        training shell configs + datagen YAMLs
  actor_sft.sh, critic_sft.sh         — Miles+FSDP SFT entry points
  TRAINING_NOTES.md                   — Qwen2.5-7B profiling + LR scan
launch/                         sbatch wrappers + repro scripts (Qwen3-8B path)
scripts/                        multi-GPU launch wrappers (datagen)
tools/                          FSDP-DCP → HF checkpoint converter
docs/                           design.md, inference.md, qwen3_8b_run.md
nla_inference.py                standalone single-file inference client
```

## Datasets

The Qwen3-8B pipeline in this fork uses:

| stage | dataset | size | location |
|---|---|---|---|
| **Corpus** | [`m-a-p/FineFineWeb`](https://huggingface.co/datasets/m-a-p/FineFineWeb) (67 domains, ~10TB) | sample 100k docs | public on HF Hub |
| **Stage 0 — activations** | 1.4M (doc, position, residual-stream activation @ Qwen3-8B layer 24) tuples | ~16 GB | regenerable via `launch/sbatch_datagen.sh` |
| **Stage 3 — SFT/RL parquets** | `av_train`, `av_val`, `ar_sft_shuf_clean`, `rl_shuf` (with Sonnet 4.6 explanations) | ~4 GB | regenerable; upload script at `launch/upload_dataset.py` for HF publishing |

Stage 1 (doc-level 25/25/50 split into AV/AR/RL) and Stage 2 (Sonnet judging
via the Anthropic Batches API) are deterministic given the corpus + seed.
Regenerate from FineFineWeb with `launch/sbatch_datagen.sh` (requires
`ANTHROPIC_API_KEY_BATCH`; ~12h, ~$80 in batch-API tokens). After Stage 3,
`python launch/upload_dataset.py --repo <your-hf-org>/nla-qwen3-8b-stage3`
publishes the SFT/RL parquets to HF Hub for re-use.

---

## Citation

For attribution in academic contexts, please cite this work as

> Fraser-Taliente, Kantamneni, Ong et al., "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", Transformer Circuits, 2026.

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

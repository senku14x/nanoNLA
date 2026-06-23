# Compute plan & environment

## This container vs the GPU boxes
This repo's session runs in a **CPU-only container** (no GPU, ~15 GB RAM, no
torch, and `huggingface.co` / arXiv / the NLA site return 403 — egress is
allowlisted; pypi *is* reachable). It is for **authoring code + design + the
pure-math self-tests only**. Nothing that touches a model runs here.

Two GPU boxes do the real work:

**Substrate (D2): syvb's Qwen3-8B L24 NLA for ALL gates.** One 8B model end-to-end.

| Box | Hardware | Role |
|---|---|---|
| **Vast** | 1× H200 (141 GB) | **ALL gates −1 / 0 / 1 / 2 / 3.** Inference (AR scoring + in-process AV gen) and the single-GPU RL fine-tune all fit one card. |
| **Lambda** | 2× H100/A100 (80 GB) | **Optional headroom** — a concurrent second run / sweep arm, or the `train_rl_vllm.py` path if we want faster rollout later. Not required. |

**The whole pipeline runs on the one H200 (D5 + user confirm).** Gate 3 is
single-GPU: nanoNLA's validated path (`nla.train_rl_self_contained`) is
single-process, **4-bit base + LoRA, no vLLM/SGLang**, so it fits the H200
alongside everything else — no checkpoint shuffling between boxes. The earlier
"2×H100 vLLM rollout/trainable split" was a misread; Lambda is now just optional
parallelism. See `configs/models.yaml`.

## Substrate tradeoff (recorded, D2)
We chose Qwen3-8B (syvb, ~0.3 FVE) over the released high-FVE Qwen2.5-7B NLA for
model-matching + frugality + runnability. The low FVE means the residual
measurement is noisier and the read/unread gap is *unmeasured* on this model — so
**Gate 1 is a real fork** (it must establish a gap exists, and may return "no gap").
The Gate-0 counterfactual-mention test is preferred precisely because it is a
*difference* (ΔMSE from a mention), more robust to low absolute FVE. The released
high-FVE NLAs stay in `configs/models.yaml` as an optional cross-check/fallback.

**Prerequisite before probing behavioral concepts:** `scripts/check_model.py` must
confirm the model is the **post-trained** Qwen3-8B, not `-Base` (refusal/sycophancy/
corrigibility are RLHF behaviors; on a base model they are weak/absent and the panel
pivots to truth/topic/factual concepts).

## Environment setup (on a GPU box)
```bash
# CPU-only deps (also what this container uses for self-tests):
pip install -r requirements.txt
# GPU deps (only on the GPU boxes):
pip install -r requirements-gpu.txt   # torch, transformers, safetensors, sglang, vllm
huggingface-cli login                 # needed for gated bases (Llama/Gemma)
```

## Operational practices (generic — for ephemeral/spot GPU boxes)
Standard hygiene, nothing project-specific:
- **Persist as you go.** Vast/Colab boxes are ephemeral; push every checkpoint +
  any generated dataset to HF (or GCS) *immediately* after it's written, so a host
  failure doesn't cost a run. Don't accumulate hours of state only on local disk.
- **Fit, don't fight, memory.** Use gradient checkpointing for the AR-SFT and RL
  steps before reaching for smaller models; tune micro-batch up to fill the card
  rather than leaving VRAM idle. On the H200 (141 GB) bf16 8B has plenty of room.
- **Logging from env.** Auto-enable wandb from `WANDB_API_KEY` so runs are tracked
  without per-run flags; log the per-arm metrics the design lists (reward mean,
  within-group reward std, critic loss, response length, extraction-failure rate).
- **Calibrate wall-clock first.** Run ~20 steps to estimate time before committing
  to a full sweep.

## First thing to run anywhere
```bash
bash scripts/run_tests.sh    # pure-math self-tests; no GPU needed
```
These assert `MSE == 2(1-cos)`, the FVE geometry (`FVE ≈ 2cos-1` under sphere-
normalization), residual-alignment z-scores (random ~0, aligned ≫0), the probe
(real ≫ shuffled≈0.5), and the Gate-0 decision logic on three constructed worlds.
If any fails, fix it before interpreting a single model number.

## Gate 0 smoke test (on the GPU box, before reading results)
The AR loading is `NEEDS-GPU-VALIDATION`: the released `NLACriticModel` state-dict
layout (value-head key, `K+1` layer count, no final norm) must be asserted on
first load (`nla_io.ARScorer` already asserts what it can). The functional smoke
test is in `gate0_counterfactual`: a refusal-mention on a refusal activation MUST
reconstruct better than an irrelevant mention. If it does not, the AR readout
token / normalization / checkpoint layout is wrong — stop and fix.

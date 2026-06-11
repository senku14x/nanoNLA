# nanoNLA

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/2e6dc1c6-d998-4e57-807a-4ce4a5f288e2" />

(enjoy gpt image 2.0 reading of this repo)

This is a minimal reimplementation of [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html). 

Starting from their [implementation](https://github.com/kitft/natural_language_autoencoders), I here share a minimal version that is sufficient to train NLAs on small models (no SGLang) and should lead to significantly reduced infra hassle.

**The code contains the code to warmstart an AV and an AR, and co-train them using RL**

**The warmstart dataset can be found [here](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k)** 

Hyperparameters are *close* to the paper but not identical — this repo trades some fidelity for running on modest hardware (single GPU, 4-bit base). Known deviations:

- **Injection**: Karvonen norm-matched ADD at the layer-1 output, not the paper's embedding replacement
- **LoRA** (r=128, rsLoRA, α=16) on a 4-bit quantized base, instead of full fine-tuning
- **RL batch**: 16 prompts × group size 16 (paper: 128 × G=8)
- **SFT**: batch 64, lr 3e-5 (paper: 256, lr 1e-5)
- **Reward**: −MSE (paper: −log MSE)
- **Data**: 10 positions/doc (paper's open-model runs: 5), 2-3 summary features per explanation (paper: 4-5)

FVE in this repo is reported against the paper's variance-around-mean baseline (since 2026-06; older notes/curves used a looser baseline and read several points higher).

A Natural Language Autoencoder is a pair of fine-tuned LMs that map
residual-stream activation vectors to natural language and back.

(I may or may not extend this with evals, for ease of hillclimbing)

## For agents (Claude etc.)

Agent instructions, repo invariants, and gotchas live in [CLAUDE.md](CLAUDE.md). Working launch commands live in [`scripts/`](scripts/) — see `scripts/sbatch_{av,ar}_sft_lora_fixed.sh` and `scripts/sbatch_rl_fixed.sh` for the verified pipeline, or `scripts/smoke_fixed_pipeline.sh` to exercise the whole AV-SFT → AR-SFT → RL → resume path end-to-end in a few minutes. The step-by-step recipe for a new model is in [docs/train_new_model.md](docs/train_new_model.md).

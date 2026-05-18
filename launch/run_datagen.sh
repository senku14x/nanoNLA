#!/bin/bash
# Real datagen for Qwen3-8B on 10k FineWeb docs.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
# Use BATCH key (50% off, async) — BatchAnthropicProvider reads ANTHROPIC_API_KEY_BATCH first.
: "${ANTHROPIC_API_KEY_BATCH:?set ANTHROPIC_API_KEY_BATCH (batch tier key, see CLAUDE.md)}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY (low-prio tier key, see CLAUDE.md)}"
export PYTHONUNBUFFERED=1

echo "=== START real datagen (Qwen3-8B / FineFineWeb / 100k docs / Sonnet 4.6 batch API) ==="
date
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1
echo

python -m nla.datagen.run_pipeline --config configs/datagen/qwen3_8b_finefineweb_100k.yaml --stages 2,3,shuffle

echo "=== END real datagen ==="
date
ls -la /workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k/ 2>&1 | head -30

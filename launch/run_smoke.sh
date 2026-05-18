#!/bin/bash
# Smoke test for Qwen3-8B NLA datagen pipeline.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source .venv/bin/activate

export HF_HOME=/workspace-vast/pretrained_ckpts
: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY (low-prio tier key, see CLAUDE.md)}"
export PYTHONUNBUFFERED=1

echo "=== START smoke pipeline ==="
date
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1

python -m nla.datagen.run_pipeline --config configs/datagen/qwen3_8b_quick_test.yaml

echo "=== END smoke pipeline ==="
date
ls -la /workspace-vast/celeste/nla-data/qwen3_8b_smoke/ 2>&1 | head -20

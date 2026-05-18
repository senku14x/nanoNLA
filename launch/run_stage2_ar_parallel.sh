#!/bin/bash
# Run Stage 2 (API explanations) for AR-SFT in parallel with the main datagen
# job's AV-SFT Stage 2. Each writes to its own chunks/ subdir so no conflicts.
# stage2_api_explain has resume logic — if main job catches up and finds chunks
# already written, it skips them.

set -euo pipefail

cd /workspace-vast/celeste/nla-experiments
source /workspace-vast/celeste/envs/nla/bin/activate

: "${HF_TOKEN:?set HF_TOKEN in your shell}"
: "${ANTHROPIC_API_KEY_BATCH:?set ANTHROPIC_API_KEY_BATCH (batch tier key, see CLAUDE.md)}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY (low-prio tier key, see CLAUDE.md)}"
export PYTHONUNBUFFERED=1

DATA=/workspace-vast/celeste/nla-data/qwen3_8b_finefineweb_100k

echo "=== AR Stage 2 (parallel) START ==="
date

python -m nla.datagen.stage2_api_explain \
  --input  $DATA/splits/ar_sft_raw.parquet \
  --output $DATA/splits/ar_sft_explained.parquet \
  --provider-cls nla.datagen.providers.BatchAnthropicProvider \
  --provider-kwargs '{"model": "claude-sonnet-4-6", "max_tokens": 300, "temperature": 1.0, "max_batch_size": 50000, "poll_interval_s": 30.0}' \
  --chunk-size 50000 \
  --storage-cls nla.datagen.storage.LocalStorage

echo "=== AR Stage 2 (parallel) END ==="
date

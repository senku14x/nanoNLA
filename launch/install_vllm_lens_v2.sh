#!/bin/bash
# vllm-lens with cu128-compatible torch (cluster driver is 12.8, not 13.0).
set -euo pipefail

VENV=/workspace-vast/celeste/envs/vllm-lens
# Nuke + recreate (the previous venv has cu130 torch)
rm -rf "$VENV" || true
uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"

# 1. Install torch built for cu128 first
uv pip install --python "$VENV/bin/python" "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5

# 2. Now install vllm-lens — should use existing torch instead of pulling cu130
uv pip install --python "$VENV/bin/python" vllm-lens 2>&1 | tail -10

# 3. Verify
"$VENV/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
import vllm; print('vllm', vllm.__version__)
import vllm_lens; print('vllm_lens', vllm_lens.__version__)
print('cuda available:', torch.cuda.is_available())
"

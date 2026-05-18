#!/bin/bash
# Try vllm-lens 1.0 + older vllm (<0.17) which may bundle cu128 torch.
set -euo pipefail
VENV=/workspace-vast/celeste/envs/vllm-lens
rm -rf "$VENV"
uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"
uv pip install --python "$VENV/bin/python" "vllm-lens==1.0.0" "vllm>=0.16,<0.17" 2>&1 | tail -8
"$VENV/bin/python" -c "
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail:', torch.cuda.is_available())
import vllm; print('vllm', vllm.__version__)
import vllm_lens; print('vllm_lens', vllm_lens.__version__)
"

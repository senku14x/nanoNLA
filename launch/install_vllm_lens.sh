#!/bin/bash
# Test install of vllm-lens in a fresh venv (don't touch the nla env which has sglang).
set -euo pipefail

VENV=/workspace-vast/celeste/envs/vllm-lens
if [ ! -d "$VENV" ]; then
  uv venv "$VENV" --python 3.12
fi
source "$VENV/bin/activate"

echo "=== install vllm-lens ==="
uv pip install --python "$VENV/bin/python" vllm-lens 2>&1 | tail -15

echo "=== verify import ==="
"$VENV/bin/python" -c "
import vllm_lens
from vllm_lens import SteeringVector
import vllm
print('vllm', vllm.__version__)
print('vllm_lens', getattr(vllm_lens, '__version__', '?'))
print('SteeringVector ok')
"
echo "=== done ==="

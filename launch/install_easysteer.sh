#!/bin/bash
# EasySteer install — uses a pinned vllm commit which may have an older torch
# that works with the cluster's cu128 driver.

set -euo pipefail

VENV=/workspace-vast/celeste/envs/easysteer
SRC=/workspace-vast/celeste/git/easysteer

rm -rf "$VENV" 2>/dev/null || true
uv venv "$VENV" --python 3.10
source "$VENV/bin/activate"

if [ ! -d "$SRC" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git "$SRC"
fi
cd "$SRC/vllm-steer"
export VLLM_PRECOMPILED_WHEEL_COMMIT=95c0f928cdeeaa21c4906e73cee6a156e1b3b995
VLLM_USE_PRECOMPILED=1 uv pip install --python "$VENV/bin/python" --editable . 2>&1 | tail -10
cd ..
uv pip install --python "$VENV/bin/python" --editable . 2>&1 | tail -10

echo "=== verify ==="
"$VENV/bin/python" -c "
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail:', torch.cuda.is_available())
import vllm; print('vllm', vllm.__version__)
from vllm.steer_vectors.request import SteerVectorRequest
print('SteerVectorRequest ok')
"

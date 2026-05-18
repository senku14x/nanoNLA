#!/bin/bash
# Clean rebuild of the nla-experiments venv with sglang training stack.
# Install order: sglang (sets torch/transformers pins) → miles → nla → patches.

set -euo pipefail

WORKDIR=/workspace-vast/celeste/nla-experiments
VENV=/workspace-vast/celeste/envs/nla
MILES_DIR=/workspace-vast/celeste/miles
SGLANG_DIR=/workspace-vast/celeste/sglang
MILES_PIN=051cd15

cd $WORKDIR

echo "=== 0. Fresh venv at $VENV ==="
mkdir -p $(dirname "$VENV")
# Don't try to rm an old one on VAST (delete-while-locked footgun) — fresh path.
uv venv "$VENV" --python 3.12

echo "=== 1. Bootstrap: pip + setuptools ==="
uv pip install --python "$VENV/bin/python" pip setuptools wheel pybind11

echo "=== 2. Install sglang[all] FIRST — it pins torch/transformers ==="
if [ ! -d $SGLANG_DIR/.git ]; then
  git clone https://github.com/sgl-project/sglang.git $SGLANG_DIR
fi
cd $SGLANG_DIR
git fetch origin 2>&1 | tail -2 || true
git checkout v0.5.6 2>/dev/null || echo "(v0.5.6 tag missing, staying on current HEAD)"
bash $WORKDIR/patches/apply_sglang_patches.sh $SGLANG_DIR 2>&1 | tail -5 || echo "(patches partial — ok, can apply later)"
uv pip install --python "$VENV/bin/python" -e "$SGLANG_DIR/python[all]" 2>&1 | tail -10

echo "=== 3. Verify torch/transformers from sglang's pins ==="
$VENV/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__)"

echo "=== 4. Install miles ==="
if [ ! -d $MILES_DIR/.git ]; then
  git clone https://github.com/radixark/miles.git $MILES_DIR
fi
cd $MILES_DIR
git fetch origin
git checkout "$MILES_PIN"
# Re-apply NLA's miles patches (may need to be re-applied after fresh checkout).
for patch in $WORKDIR/nla/miles_patches/*.patch; do
  if git apply --check "$patch" >/dev/null 2>&1; then
    git apply "$patch"
    echo "  applied $(basename "$patch")"
  fi
done
uv pip install --python "$VENV/bin/python" -e "$MILES_DIR" 2>&1 | tail -10

echo "=== 5. Install nla package ==="
cd $WORKDIR
uv pip install --python "$VENV/bin/python" -e . 2>&1 | tail -10

echo "=== 6. Extra deps nla/scripts use ==="
uv pip install --python "$VENV/bin/python" anthropic httpx orjson "pyyaml>=6" wandb ray[default] 2>&1 | tail -10

echo "=== 7. Verify imports ==="
$VENV/bin/python -c "
import torch; print('torch', torch.__version__)
import transformers; print('transformers', transformers.__version__)
import sglang; print('sglang', sglang.__version__)
import miles; print('miles ok')
import nla; print('nla ok')
import ray; print('ray', ray.__version__)
import wandb; print('wandb', wandb.__version__)
import anthropic; print('anthropic', anthropic.__version__)
print('cuda available:', torch.cuda.is_available())
"
echo "=== DONE ==="

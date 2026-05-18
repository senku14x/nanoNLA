#!/bin/bash
# Install SGLang with NLA's training patches.

set -euo pipefail

cd /workspace-vast/celeste
source /workspace-vast/celeste/nla-experiments/.venv/bin/activate

if [ ! -d /workspace-vast/celeste/sglang/.git ]; then
  echo "=== clone sglang ==="
  git clone https://github.com/sgl-project/sglang.git /workspace-vast/celeste/sglang
fi

cd /workspace-vast/celeste/sglang
# Pin to known-good version per nla repo.
git fetch origin
git checkout v0.5.6 2>/dev/null || echo "(v0.5.6 tag not available; using current HEAD)"

echo "=== apply nla patches ==="
bash /workspace-vast/celeste/nla-experiments/patches/apply_sglang_patches.sh /workspace-vast/celeste/sglang || echo "(patches partial — review)"

echo "=== install editable ==="
uv pip install --python /workspace-vast/celeste/nla-experiments/.venv/bin/python -e "/workspace-vast/celeste/sglang/python[all]"

echo "=== verify ==="
/workspace-vast/celeste/nla-experiments/.venv/bin/python -c "import sglang; print('sglang', sglang.__version__)"
echo "=== DONE ==="

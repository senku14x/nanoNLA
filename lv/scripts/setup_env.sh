#!/usr/bin/env bash
# Environment setup. On a CPU box (or this container) installs only the pure-math
# deps. Pass --gpu on a Vast/Lambda box to also install model deps.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -r requirements.txt
if [[ "${1:-}" == "--gpu" ]]; then
  pip install -r requirements-gpu.txt
  echo "GPU deps installed. Remember: huggingface-cli login for gated bases."
fi
echo "Running pure-math self-tests..."
bash scripts/run_tests.sh

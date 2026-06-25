# Canonical config for the 4xH200 AR/RL wide-bank regeneration ops.
# Source this from the ops scripts. Override any value via the environment.
export REPO="${REPO:-/workspace/nanoNLA}"
export WORK_ROOT="${WORK_ROOT:-/workspace/mlnla}"
export PUB="${PUB:-$WORK_ROOT/published}"                       # source parquets
export REGEN="${REGEN:-$WORK_ROOT/published_L24x_window}"       # output shards (DO NOT merge)
export LOGS="${LOGS:-$WORK_ROOT/logs}"
export MANIFESTS="${MANIFESTS:-$WORK_ROOT/manifests}"
export DRIVE_REMOTE="${DRIVE_REMOTE:-gdrive:nla-archives/qwen3-8b-finefineweb/L19-L29_center24}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export CENTER="${CENTER:-24}"
export SAVE_LAYERS="${SAVE_LAYERS:-19-29}"
export MAXLEN="${MAXLEN:-4096}"
export BATCH="${BATCH:-48}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
export MAXDROP="${MAXDROP:-0.005}"

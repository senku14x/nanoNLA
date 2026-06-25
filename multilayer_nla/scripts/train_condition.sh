#!/usr/bin/env bash
# §7 control arm: train AV-SFT -> AR-SFT -> RL under a single injection CONDITION,
# reading PRE-BUILT labeled triplet parquets — NO regen, NO build. Run this on a
# second box for the duplicate control, in parallel with the coherent run.
#
# Fresh-instance bootstrap (clone first — this script lives in the repo):
#     git clone <repo> && cd nanoNLA && git checkout multilayer_working && pip install -e .
#     export HF_TOKEN=... && huggingface-cli login --token "$HF_TOKEN" ; wandb login
#
# Then:
#   1. copy the coherent box's built triplets here (NOT the 180 GB wide bank):
#        rclone copy gdrive:nla-archives/qwen3-8b-finefineweb/labeled_triplets "$TRAIN"
#        # expect av_sft.parquet, ar_sft.parquet, rl.parquet
#   2. CONDITION=duplicate bash multilayer_nla/scripts/train_condition.sh
#
# The condition is applied at LOAD time (datasets.apply_condition_columns) to the
# injected vectors AND the AR targets, so it threads identically through all three
# stages — the ONLY variable vs the coherent run. duplicate := every slot = centre.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BASE="${BASE:-Qwen/Qwen3-8B}"
CENTER="${CENTER:-24}"
CONDITION="${CONDITION:-duplicate}"            # coherent | duplicate
WANDB_PROJECT="${WANDB_PROJECT:-multi layer nla}"
DATA="${DATA:-/data/mlnla}"
TRAIN="${TRAIN:-$DATA/labeled}"                # pre-built av_sft/ar_sft/rl parquets (copied over)
CKPT="${CKPT:-$DATA/ckpt_${CONDITION}}"        # separate ckpt tree per condition (no collision)
# RL rollout batch. DEFAULT matches the coherent run (16x16) so the ONLY variable
# is the condition — a clean A/B. A fresh box has headroom to raise these (~33GB at
# 16x16, ~85GB at 32x32 on a 140GB H200), but that adds a second variable; only do
# it if you also re-run the coherent arm at the same size.
RL_BATCH_PROMPTS="${RL_BATCH_PROMPTS:-16}"
RL_GROUP_SIZE="${RL_GROUP_SIZE:-16}"

for f in av_sft ar_sft rl; do
  [ -f "$TRAIN/$f.parquet" ] || { echo "ERROR: missing $TRAIN/$f.parquet — copy the labeled triplets first" >&2; exit 1; }
done
mkdir -p "$CKPT"
# Trainers save to {save_dir}/iter_{step+1:07d}/; RL must load that subdir, not the parent.
latest_iter() { ls -d "$1"/iter_* 2>/dev/null | sort | tail -1; }
require_dir() { [ -n "$1" ] && [ -d "$1" ] || { echo "ERROR: checkpoint dir not found: '$1'" >&2; exit 1; }; }

echo "[control] CONDITION=$CONDITION | TRAIN=$TRAIN | CKPT=$CKPT | RL ${RL_BATCH_PROMPTS}x${RL_GROUP_SIZE}"

python -m multilayer_nla.train_av_multi --base-ckpt "$BASE" --parquet "$TRAIN/av_sft.parquet" \
      --save-dir "$CKPT/av" --use-lora --quant none --num-steps 1000 --condition "$CONDITION" \
      --wandb-project "$WANDB_PROJECT" --wandb-name "av-L${CENTER}-${CONDITION}"
python -m multilayer_nla.train_ar_multi --base-ckpt "$BASE" --parquet "$TRAIN/ar_sft.parquet" \
      --save-dir "$CKPT/ar" --use-lora --quant none --num-steps 1000 \
      --tap-layers $((CENTER-1)),${CENTER},$((CENTER+1)) --condition "$CONDITION" \
      --wandb-project "$WANDB_PROJECT" --wandb-name "ar-L${CENTER}-${CONDITION}"
AVD=$(latest_iter "$CKPT/av"); require_dir "$AVD"
ARD=$(latest_iter "$CKPT/ar"); require_dir "$ARD"
python -m multilayer_nla.train_rl_multi --base-ckpt "$BASE" \
      --av-ckpt "$AVD" --ar-ckpt "$ARD" --rl-parquet "$TRAIN/rl.parquet" \
      --save-dir "$CKPT/rl" --quant none --num-steps 500 --condition "$CONDITION" \
      --batch-prompts "$RL_BATCH_PROMPTS" --group-size "$RL_GROUP_SIZE" \
      --wandb-project "$WANDB_PROJECT" --wandb-name "rl-L${CENTER}-${CONDITION}"
echo "[control] $CONDITION arm complete -> $CKPT  (compare rl-L${CENTER}-${CONDITION} FVE vs the coherent run)"

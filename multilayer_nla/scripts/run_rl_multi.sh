#!/usr/bin/env bash
# Multi-layer NLA RL (GRPO) — AV actor optimized against a FROZEN multitap AR critic.
# Reward = three_target_reward (√d-normalized directional MSE), byte-identical to the
# verified single-target recipe (k=1) generalized to k taps. Critic is FROZEN here
# (AV-only RL); for AR co-training see the --train-critic port (not yet wired into this trainer).
#
# Two configs via CONFIG:
#   CONFIG=3tap  -> reconstruct {L23,L24,L25} from 3-slot AV input   (3-tap AR, all 3 target slots)
#   CONFIG=l24   -> reconstruct L24 ONLY from the SAME 3-slot input   (1-tap L24 AR, centre slot)
# The AV injection is always 3-slot; only the AR *target* differs (--ar-target-slots).
#
#   AV_CKPT=<3-slot av dir>  AR_CKPT=<AR dir>  RL_PARQUET=<rl.parquet>  CONFIG=3tap \
#     bash multilayer_nla/scripts/run_rl_multi.sh
#
# CONFIG=3tap needs a 3-tap AR (tap_layers 23,24,25, e.g. §7 ar_3tap).
# CONFIG=l24  needs a 1-tap  AR (tap_layers 24,       e.g. §7 ar_l24only).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
CONFIG="${CONFIG:-3tap}"
BASE="${BASE:-Qwen/Qwen3-8B}"
AV_CKPT="${AV_CKPT:?set AV_CKPT to the 3-slot AV-SFT LoRA dir (e.g. av_local)}"
AR_CKPT="${AR_CKPT:?set AR_CKPT to the AR multitap dir (ar_multitap.safetensors + ar_meta.json)}"
RL_PARQUET="${RL_PARQUET:?set RL_PARQUET to the rl shard(s) with prev/centre/next slot columns}"
STEPS="${STEPS:-500}"
QUANT="${QUANT:-4bit}"              # single-GPU verified path is 4-bit
# rollout knobs — the ONLY real speed lever (rollouts dominate step time).
# per-step rollouts = BATCH_PROMPTS × GROUP_SIZE (default 256). Halve GROUP_SIZE -> ~2x faster
# (slightly noisier group-advantage z-score; 8 is a fine GRPO minimum). BATCH_PROMPTS×GROUP_SIZE=64
# (e.g. 8×8) is ~4x faster. MAX_NEW caps generated tokens (explanations run ~40-80 tok).
BATCH_PROMPTS="${BATCH_PROMPTS:-16}"
GROUP_SIZE="${GROUP_SIZE:-16}"
MAX_NEW="${MAX_NEW:-150}"
SAVE_EVERY="${SAVE_EVERY:-50}"      # lower (e.g. 25) for earlier checkpoints when you plan to early-stop
NW=""; [ "${WANDB:-1}" = 0 ] && NW="--no-wandb"

case "$CONFIG" in
  3tap) SLOTS="prev,centre,next";  echo "[rl-multi] reconstruct L23/24/25 (3 targets)";;
  l24)  SLOTS="centre";            echo "[rl-multi] reconstruct L24 only (1 target) from 3-slot input";;
  *) echo "CONFIG must be '3tap' or 'l24' (got '$CONFIG')"; exit 1;;
esac
SAVE="${SAVE:-runs/rl_multi_$CONFIG}"

echo "[rl-multi] av=$AV_CKPT ar=$AR_CKPT slots=$SLOTS -> $SAVE"
python -m multilayer_nla.train_rl_multi --base-ckpt "$BASE" \
  --av-ckpt "$AV_CKPT" --ar-ckpt "$AR_CKPT" --ar-target-slots "$SLOTS" \
  --rl-parquet "$RL_PARQUET" --save-dir "$SAVE" \
  --quant "$QUANT" --num-steps "$STEPS" --batch-prompts "$BATCH_PROMPTS" --group-size "$GROUP_SIZE" \
  --max-new-tokens "$MAX_NEW" --save-every "$SAVE_EVERY" --temperature 1.0 --lr 1e-5 --kl-beta 0.01 --clip-eps 0.2 \
  --wandb-name "rl_multi_${CONFIG}" $NW
echo "[rl-multi] DONE -> $SAVE"

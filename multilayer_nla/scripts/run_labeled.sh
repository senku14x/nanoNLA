#!/usr/bin/env bash
# Multi-layer NLA — labeled run (free labels from the published dataset, NO API).
#
# Fresh-instance bootstrap (run these FIRST, before this script — it lives inside
# the repo, so you must clone before you can run it):
#
#     git clone https://github.com/senku14x/nanoNLA.git   # use your auth if private
#     cd nanoNLA && git checkout multilayer_working
#     pip install -e .                                     # torch already on H200 images; pins transformers==4.57.1
#     export HF_TOKEN=...        && huggingface-cli login --token "$HF_TOKEN"
#     wandb login               # paste your key
#     bash multilayer_nla/scripts/run_labeled.sh           # STAGE=smoke (default); RUN_FULL=1 for the real run
#
# Pipeline: download published av/ar/rl (text + LABELS) -> regenerate the 19-29
# activation window at each prefix's FINAL token (no API) -> build_from_published
# --center 24 selects the {23,24,25} triplet -> AV-SFT -> AR-SFT -> RL (coherent).
set -euo pipefail

# ── config (override via env) ───────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-$PWD}"        # repo root
BASE="${BASE:-Qwen/Qwen3-8B}"
CENTER="${CENTER:-24}"
SAVE_LAYERS="${SAVE_LAYERS:-19-29}"            # window; covers candidate_centers [20..28]
MAXLEN="${MAXLEN:-4096}"                       # MUST match the published extraction
DATASET="${DATASET:-ceselder/qwen3-8b-nla-L24-finefineweb-100k}"
WANDB_PROJECT="${WANDB_PROJECT:-multi layer nla}"
NUM_SHARDS="${NUM_SHARDS:-8}"                  # crash-resilient regen (per-shard files; resume via skip-if-exists)
MAXDROP="${MAX_DROP_FRAC:-0.02}"               # tolerate benign detokenize->retokenize drift (~0.2% observed
                                               #   per chunk); a much higher rate = systematic error (e.g. wrong --max-length)
RUN_FULL="${RUN_FULL:-0}"                      # 0 = smoke only; 1 = also run the full ~1M-row pipeline

DATA="${DATA:-/data/mlnla}"                    # point at your big-disk mount
PUB="$DATA/published"; REGEN="$DATA/published_L${CENTER}x_window"; TRAIN="$DATA/labeled"; CKPT="$DATA/ckpt"
mkdir -p "$PUB" "$REGEN" "$TRAIN" "$CKPT"

# Trainers save to {save_dir}/iter_{step+1:07d}/ (the final step always saves), so
# RL must be pointed at that subdir, not the parent. Resolve the latest one:
latest_iter() { ls -d "$1"/iter_* 2>/dev/null | sort | tail -1; }
require_dir() { [ -n "$1" ] && [ -d "$1" ] || { echo "ERROR: checkpoint dir not found: '$1'" >&2; exit 1; }; }

# ── 1. download published av/ar/rl subsets (text + LABELS, no vectors) ──
#     This is also the existence/shape GATE: it fails instantly if the dataset id,
#     a config name, or a required column is wrong — before any GPU work. Eyeball
#     the printed rows/columns: expect ~247k/247k/500k and detokenized_text_truncated
#     + n_raw_tokens on all three (av: response; ar: prompt; no activation_vector).
python - <<PY
from datasets import load_dataset
NEED = {"av_sft": {"response", "detokenized_text_truncated", "n_raw_tokens"},
        "ar_sft": {"prompt", "detokenized_text_truncated", "n_raw_tokens"},
        "rl":     {"detokenized_text_truncated", "n_raw_tokens"}}
for name in ("av_sft", "ar_sft", "rl"):
    ds = load_dataset("$DATASET", name, split="train")
    ds.to_parquet(f"$PUB/{name}.parquet")
    cols = set(ds.column_names)
    print(f"{name}: {ds.num_rows} rows | columns = {sorted(cols)}", flush=True)
    missing = NEED[name] - cols
    assert not missing, f"{name} is missing expected column(s) {missing} — published schema changed?"
print("[check] schema OK — proceeding.", flush=True)
PY

# ════════════════════════════════════════════════════════════════════
# A. SMALL-SLICE SMOKE (~2k rows — prove the whole path cheaply FIRST)
# ════════════════════════════════════════════════════════════════════
SMK="$DATA/smoke"; mkdir -p "$SMK"/{pub,regen,train,ckpt}
python - <<PY
import pyarrow.parquet as pq
for name in ("av_sft","ar_sft","rl"):
    t = pq.read_table(f"$PUB/{name}.parquet").slice(0, 2000)
    pq.write_table(t, f"$SMK/pub/{name}.parquet"); print("smoke", name, t.num_rows, flush=True)
PY
for s in av_sft ar_sft rl; do
  python -m multilayer_nla.regenerate_multilayer_activations \
      --in "$SMK/pub/$s.parquet" --out "$SMK/regen/$s.parquet" \
      --base-model "$BASE" --center-layer "$CENTER" --save-layers "$SAVE_LAYERS" \
      --max-length "$MAXLEN" --max-drop-frac "$MAXDROP"
done   # a ~0.2% round-trip drop is EXPECTED (benign detokenize drift); a large drop = problem
python -m multilayer_nla.build_from_published --mode all --center "$CENTER" \
      --in-dir "$SMK/regen" --out-dir "$SMK/train"
python -m multilayer_nla.train_av_multi --base-ckpt "$BASE" --parquet "$SMK/train/av_sft.parquet" \
      --save-dir "$SMK/ckpt/av" --use-lora --quant none --num-steps 40 --batch-size 8 \
      --wandb-project "$WANDB_PROJECT" --wandb-name smoke-av
python -m multilayer_nla.train_ar_multi --base-ckpt "$BASE" --parquet "$SMK/train/ar_sft.parquet" \
      --save-dir "$SMK/ckpt/ar" --use-lora --quant none --num-steps 40 --batch-size 8 --tap-layers 23,24,25 \
      --wandb-project "$WANDB_PROJECT" --wandb-name smoke-ar
# RL loads the per-iter checkpoint dirs (not the parent) — resolve the latest.
AVD=$(latest_iter "$SMK/ckpt/av"); require_dir "$AVD"
ARD=$(latest_iter "$SMK/ckpt/ar"); require_dir "$ARD"
python -m multilayer_nla.train_rl_multi --base-ckpt "$BASE" \
      --av-ckpt "$AVD" --ar-ckpt "$ARD" --rl-parquet "$SMK/train/rl.parquet" \
      --save-dir "$SMK/ckpt/rl" --quant none --num-steps 5 --batch-prompts 4 --group-size 4 \
      --wandb-project "$WANDB_PROJECT" --wandb-name smoke-rl
echo "[smoke] GREEN iff: round-trip drop small (~0.2%) | AV loss down | AR FVE printed | RL Fix-4 KL≈0, ext>0, no CJK"

[ "$RUN_FULL" = "1" ] || { echo "[done] smoke complete. Re-run with RUN_FULL=1 for the full ~1M-row pipeline."; exit 0; }

# ════════════════════════════════════════════════════════════════════
# B. FULL RUN (~1M rows) — only after the smoke looks right
# ════════════════════════════════════════════════════════════════════
# B1. regenerate the window for every row. Shards are crash-resilient: each writes
#     its own file; the skip-if-exists guard resumes a re-run. One H200 -> sequential.
#     Multi-GPU: wrap the python call in  CUDA_VISIBLE_DEVICES=$((i % NGPU)) ... &  + wait.
for s in av_sft ar_sft rl; do
  for i in $(seq 0 $((NUM_SHARDS-1))); do
    shard=$(printf '%s/%s.shard%02dof%02d.parquet' "$REGEN" "$s" "$i" "$NUM_SHARDS")
    [ -f "$shard" ] && { echo "skip existing $shard"; continue; }
    python -m multilayer_nla.regenerate_multilayer_activations \
        --in "$PUB/$s.parquet" --out "$REGEN/$s.parquet" \
        --base-model "$BASE" --center-layer "$CENTER" --save-layers "$SAVE_LAYERS" \
        --max-length "$MAXLEN" --max-drop-frac "$MAXDROP" \
        --num-shards "$NUM_SHARDS" --shard-index "$i"
  done
done
# NOTE: do NOT merge the shards — that would materialize the full ~180 GB wide
# archive in RAM and on disk. The shards ARE the authoritative layer bank;
# build_from_published reads them directly (streaming, projecting only 3 layers).

# B2. select the center triplet -> 3-slot training parquets (streams the shards;
#     reads only 3 of the 11 saved layers, so the wide archive never lands in RAM)
python -m multilayer_nla.build_from_published --mode all --center "$CENTER" \
      --in-dir "$REGEN" --out-dir "$TRAIN"

# B3. coherent run: AV-SFT -> AR-SFT -> RL (bf16 + LoRA). --use-lora on AV is REQUIRED
#     (RL Fix-4 loads the AV-SFT adapter as policy + frozen KL reference).
python -m multilayer_nla.train_av_multi --base-ckpt "$BASE" --parquet "$TRAIN/av_sft.parquet" \
      --save-dir "$CKPT/av" --use-lora --quant none --num-steps 1000 \
      --wandb-project "$WANDB_PROJECT" --wandb-name av-L${CENTER}-coherent
python -m multilayer_nla.train_ar_multi --base-ckpt "$BASE" --parquet "$TRAIN/ar_sft.parquet" \
      --save-dir "$CKPT/ar" --use-lora --quant none --num-steps 1000 --tap-layers $((CENTER-1)),${CENTER},$((CENTER+1)) \
      --wandb-project "$WANDB_PROJECT" --wandb-name ar-L${CENTER}-coherent
# RL loads the per-iter checkpoint dirs (not the parent) — resolve the latest.
AVD=$(latest_iter "$CKPT/av"); require_dir "$AVD"
ARD=$(latest_iter "$CKPT/ar"); require_dir "$ARD"
python -m multilayer_nla.train_rl_multi --base-ckpt "$BASE" \
      --av-ckpt "$AVD" --ar-ckpt "$ARD" --rl-parquet "$TRAIN/rl.parquet" \
      --save-dir "$CKPT/rl" --quant none --num-steps 500 \
      --wandb-project "$WANDB_PROJECT" --wandb-name rl-L${CENTER}-coherent
echo "[done] coherent run complete. (duplicate control needs the pending §7 --condition flag.)"

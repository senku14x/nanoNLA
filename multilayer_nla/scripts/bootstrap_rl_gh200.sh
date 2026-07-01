#!/usr/bin/env bash
# Fresh-box bootstrap for multi-layer NLA RL on a GH200 (aarch64 + Hopper, ~80GB).
# Run from the repo root AFTER cloning + checking out the branch. Does: deps -> hf/wandb
# login -> download bank shard + adapters -> BUILD the RL-ready parquet (build_from_published:
# activation_L* -> prompt + prev/centre/next) -> PREFLIGHT -> smoke -> 250-step RL
# for BOTH configs (3tap + l24). bf16 by default (bitsandbytes is unreliable on aarch64;
# 80GB holds the 8B actor+critic in bf16). Set QUANT=4bit only if you confirmed bnb works.
#
#   HF_TOKEN=hf_xxx WANDB_API_KEY=xxx bash multilayer_nla/scripts/bootstrap_rl_gh200.sh
#
# Ingredient paths (HF repo subdirs) are best-guess §7 defaults — OVERRIDE if yours differ;
# the preflight fails loudly (not silently wrong) if a path/format/columns are off.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# ---- config (override via env) ----
DATA="${DATA:-/workspace/mlnla}"                                  # big-disk mount
CKPTS="$DATA/ckpts"; BANK="$DATA/bank"; mkdir -p "$CKPTS" "$BANK"
MODEL_REPO="${MODEL_REPO:-senku21x/qwen3-8b-nla-multilayer-sweep}"
DS_REPO="${DS_REPO:-senku21x/qwen3-8b-nla-multilayer-L19-29}"
AV_SUBDIR="${AV_SUBDIR:-av_local}"           # 3-slot AV-SFT (input L23,24,25)
AR3_SUBDIR="${AR3_SUBDIR:-ar_3tap_bs256e_3k}"  # 3-tap AR (tap_layers 23,24,25)
ARL24_SUBDIR="${ARL24_SUBDIR:-ar_l24only}"   # 1-tap AR (tap_layers 24)
RL_SHARD="${RL_SHARD:-rl.shard00of08.parquet}"
STEPS="${STEPS:-250}"
QUANT="${QUANT:-none}"                        # none=bf16 (GH200-safe) | 4bit (needs working bnb)
BASE="${BASE:-Qwen/Qwen3-8B}"

# ---- 1. torch/CUDA (do NOT reinstall — GH200 needs the base-image aarch64+cu12 build) ----
python -c "import torch;assert torch.cuda.is_available();print('[rl] torch',torch.__version__,'cuda OK',torch.cuda.get_device_name(0))" \
  || { echo "!! torch+CUDA not available. Launch on an NVIDIA PyTorch aarch64 base image (nvcr.io/nvidia/pytorch)."; exit 1; }

# ---- 2. deps: fast hub FIRST (before the 4.57.1 pin downgrades it), then the project ----
pip install -q "huggingface_hub[cli]"
pip install -e . -q
pip install -q wandb peft matplotlib
[ "$QUANT" = 4bit ] && { pip install -q bitsandbytes || echo "!! bitsandbytes failed on aarch64 — set QUANT=none"; }

# ---- 3. auth ----
if [ -n "${HF_TOKEN:-}" ]; then hf auth login --token "$HF_TOKEN" >/dev/null; else hf auth login; fi
if [ -n "${WANDB_API_KEY:-}" ]; then wandb login "$WANDB_API_KEY" >/dev/null; else wandb login; fi
echo "[rl] hf: $(hf auth whoami 2>/dev/null | head -1) · wandb: logged in"

# ---- 4. download ingredients ----
echo "[rl] downloading adapters from $MODEL_REPO ..."
for sub in "$AV_SUBDIR" "$AR3_SUBDIR" "$ARL24_SUBDIR"; do
  hf download "$MODEL_REPO" --include "$sub/**" --local-dir "$CKPTS" >/dev/null
done
echo "[rl] downloading RAW rl bank shard ($RL_SHARD) from $DS_REPO ..."
hf download "$DS_REPO" --repo-type dataset --include "$RL_SHARD" --local-dir "$BANK" >/dev/null

# ---- 4b. BUILD the RL-ready parquet. The bank shard has activation_L19..L29 (+ published
#          single-marker prompt); train_rl_multi needs the 3-marker AV prompt + the derived
#          activation_prev/centre/next slots. build_from_published --mode rl does exactly that
#          (--center 24 -> prev=L23, centre=L24, next=L25). CPU only. ----
TRAIN="$DATA/train"; mkdir -p "$TRAIN"; RL_PARQUET="$TRAIN/rl.parquet"
if [ ! -f "$RL_PARQUET" ]; then
  echo "[rl] building RL-ready parquet (build_from_published --mode rl --center 24) ..."
  python -m multilayer_nla.build_from_published --mode rl --center 24 \
    --in "$BANK/$RL_SHARD" --out "$RL_PARQUET"
else
  echo "[rl] RL-ready parquet already built: $RL_PARQUET"
fi

latest_iter(){ local d; d=$(ls -d "$1"/iter_* 2>/dev/null | sort | tail -1); echo "${d:-$1}"; }
AV_CKPT=$(latest_iter "$CKPTS/$AV_SUBDIR")
AR3_CKPT=$(latest_iter "$CKPTS/$AR3_SUBDIR")
ARL24_CKPT=$(latest_iter "$CKPTS/$ARL24_SUBDIR")

# ---- 5. PREFLIGHT: validate before burning GPU ----
python - "$AV_CKPT" "$AR3_CKPT" "$ARL24_CKPT" "$RL_PARQUET" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
av, ar3, arl24, rlp = map(Path, sys.argv[1:5])
ok = True
def need(cond, msg):
    global ok
    print(("  OK  " if cond else "  FAIL")+" "+msg); ok = ok and cond
need((av/"adapter_config.json").exists(), f"AV is a PEFT LoRA dir: {av}")
for nm, d, k in (("ar_3tap", ar3, 3), ("ar_l24only", arl24, 1)):
    m = d/"ar_meta.json"
    if not (m.exists() and (d/"ar_multitap.safetensors").exists()):
        need(False, f"{nm} has ar_meta.json + ar_multitap.safetensors: {d}"); continue
    taps = json.loads(m.read_text()).get("tap_layers", [])
    need(len(taps)==k, f"{nm} tap_layers={taps} (need {k} tap{'s' if k>1 else ''})")
cols = pq.ParquetFile(rlp).schema_arrow.names
need("prompt" in cols, "rl parquet has 'prompt'")
need(all(f"activation_{s}" in cols for s in ("prev","centre","next")),
     f"rl parquet has activation_prev/centre/next slot cols (got {[c for c in cols if 'activation' in c][:6]})")
if not ok:
    print("\nPREFLIGHT FAILED. Fix paths (AV_SUBDIR/AR*_SUBDIR/RL_SHARD) or build the rl parquet\n"
          "  with slot columns (multilayer_nla.build_sweep / build_from_published). Not spending GPU.")
    sys.exit(1)
print("\nPREFLIGHT OK — ingredients valid.")
PY

echo "[rl] AV=$AV_CKPT"; echo "[rl] AR3=$AR3_CKPT"; echo "[rl] ARL24=$ARL24_CKPT"; echo "[rl] RL=$RL_PARQUET"

# ---- 6. smoke (5 steps) then the real 250-step runs, for BOTH configs ----
run(){ # $1 config  $2 ar-ckpt  $3 steps  $4 save
  CONFIG="$1" AV_CKPT="$AV_CKPT" AR_CKPT="$2" RL_PARQUET="$RL_PARQUET" \
    QUANT="$QUANT" STEPS="$3" SAVE="$4" BASE="$BASE" \
    bash multilayer_nla/scripts/run_rl_multi.sh
}
echo "== SMOKE 3tap (5 steps) — watch mean_cjk≈0 (high cjk => AV/injection mismatch) =="
run 3tap "$AR3_CKPT" 5 "$DATA/rl_smoke_3tap"
echo "== SMOKE l24 (5 steps) =="
run l24 "$ARL24_CKPT" 5 "$DATA/rl_smoke_l24"

echo "== RL 3tap ($STEPS steps): reconstruct L23/24/25 =="
run 3tap "$AR3_CKPT" "$STEPS" "$DATA/rl_multi_3tap"
echo "== RL l24 ($STEPS steps): reconstruct L24 only, 3-slot input =="
run l24 "$ARL24_CKPT" "$STEPS" "$DATA/rl_multi_l24"

echo "[rl] DONE. checkpoints in $DATA/rl_multi_{3tap,l24}. Push to HF before you tear the box down."

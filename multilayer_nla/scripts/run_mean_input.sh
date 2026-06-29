#!/usr/bin/env bash
# Averaged-input (mean-pool) experiment, end to end on ONE GPU. Builds the av_mean
# datasets, trains ONE new AV (wandb run: av-mean), and runs the two FREE eval cuts that
# reuse the FROZEN published reconstructors:
#   (a) mean-input -> frozen 3-tap AR  -> FVE vs FIXED [L23,L24,L25]   (comparable to single/local)
#   (c) mean-input -> frozen L24-only AR -> FVE vs L24                  (comparable to the L24-only cut)
# No new AR is trained here. The mean-TARGET AR (b) is a separate, gated follow-up.
#
# Run AFTER `hf auth login` + `wandb login` (see the header of the PR / chat for the
# fresh-box bootstrap). Resumable: every step skips if its output already exists.
#
#   BASE=Qwen/Qwen3-8B \
#   AR=$CKPT/ar_3tap_bs256e_3k/iter_0003000  AR_L24=$CKPT/ar_l24only/iter_0003000 \
#     bash multilayer_nla/scripts/run_mean_input.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

C=mean
POOL_LAYERS="${POOL_LAYERS:-23,24,25}"          # layers averaged into the single av_in_0 slot
BASE="${BASE:-Qwen/Qwen3-8B}"
DATA="${DATA:-/data/mlnla}"
REGEN="${REGEN:-$DATA/published_L24x_window}"    # L19-29 bank: av_sft.shard*of*.parquet, rl.shard*of*.parquet
SWEEP="${SWEEP:-$DATA/sweep}"
CKPT="${CKPT:-$DATA/sweep_ckpt}"
EVALC="${EVALC:-$DATA/sweep_eval_converged}"
WANDB_PROJECT="${WANDB_PROJECT:-multi layer nla}"
WANDB_NAME="${WANDB_NAME:-av-$C}"                # <-- the wandb run name (av-mean), matches av-<cond>
EVAL_BATCH="${EVAL_BATCH:-1024}"
STEPS="${STEPS:-1000}"; SAVE_EVERY="${SAVE_EVERY:-500}"

# Frozen reconstructors — point these at the real iter dirs (local volume or HF download).
AR="${AR:?set AR = the 3-tap shared AR iter dir (e.g. \$CKPT/ar_3tap_bs256e_3k/iter_0003000)}"
AR_L24="${AR_L24:?set AR_L24 = the L24-only AR iter dir (e.g. \$CKPT/ar_l24only/iter_0003000)}"

AVDIR="$CKPT/av_$C"; AV="$AVDIR/iter_$(printf '%07d' "$STEPS")"
MAN="$SWEEP/rl_split_manifest.json"
mkdir -p "$SWEEP" "$EVALC/dev" "$EVALC/test" "$EVALC/test_arL24"

# ── preconditions (fail loud, before any compute) ────────────────────────────
shopt -s nullglob
av_shards=("$REGEN"/av_sft.shard*of*.parquet); rl_shards=("$REGEN"/rl.shard*of*.parquet)
shopt -u nullglob
[ ${#av_shards[@]} -gt 0 ] || { echo "ABORT: no av_sft.shard*of*.parquet in REGEN=$REGEN (set REGEN to the L19-29 bank)"; exit 1; }
[ ${#rl_shards[@]} -gt 0 ] || { echo "ABORT: no rl.shard*of*.parquet in REGEN=$REGEN"; exit 1; }
for d in "$AR" "$AR_L24"; do
  [ -f "$d/ar_meta.json" ] && [ -f "$d/ar_multitap.safetensors" ] || {
    echo "ABORT: $d is not an AR ckpt (need ar_meta.json + ar_multitap.safetensors)"; exit 1; }
done
echo "[mean] bank OK (${#av_shards[@]} av_sft + ${#rl_shards[@]} rl shards) | AR=$AR | AR_L24=$AR_L24"

# ── 0. split manifest — regenerate deterministically if absent (seed 42, locked 256/1000) ──
if [ ! -f "$MAN" ]; then
  echo "[mean] regenerating rl split manifest (seed 42, dev 256 / test 1000) — reproduces the locked split"
  python -m multilayer_nla.splits --source "$REGEN/rl.shard*of*.parquet" --name rl \
    --out-dir "$SWEEP" --seed 42 --fracs 0.8,0.1,0.1 --dev-subset 256 --test-subset 1000
fi

# ── 1. build the mean-pool datasets (CPU) + GATE ─────────────────────────────
if [ ! -f "$SWEEP/av_$C.parquet" ]; then
  python -m multilayer_nla.build_sweep --mode av \
    --in "$REGEN/av_sft.shard*of*.parquet" --out "$SWEEP/av_$C.parquet" \
    --av-slot-layers "$POOL_LAYERS" --av-pool
fi
for B in dev test; do
  if [ ! -f "$SWEEP/rl_${B}_$C.parquet" ]; then
    python -m multilayer_nla.build_sweep --mode rl-eval \
      --in "$REGEN/rl.shard*of*.parquet" --out "$SWEEP/rl_${B}_$C.parquet" \
      --av-slot-layers "$POOL_LAYERS" --av-pool --ar-target-layers 23,24,25 \
      --bucket "$B" --rl-split-manifest "$MAN"
  fi
done
# self-contained gate: av_in_0 == mean(in-file L23/24/25 targets), not a copy of any layer
python -m multilayer_nla.build_sweep --mode verify-pool \
  --out "$SWEEP/rl_test_$C.parquet" --av-slot-layers "$POOL_LAYERS" --ar-target-layers 23,24,25

# ── 2. train ONE AV (matched recipe: batch 64, accum 1, 1000 steps, seed 0) ───
if [ ! -f "$AV/adapter_config.json" ]; then
  python -m multilayer_nla.train_av_multi \
    --base-ckpt "$BASE" --use-lora --quant none \
    --parquet "$SWEEP/av_$C.parquet" --save-dir "$AVDIR" \
    --num-steps "$STEPS" --save-every "$SAVE_EVERY" --batch-size 64 --gradient-accumulation-steps 1 \
    --seed 0 --wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME"
else
  echo "[mean] AV adapter exists ($AV) — skipping training"
fi
[ -f "$AV/adapter_config.json" ] || { echo "ABORT: training did not produce $AV/adapter_config.json"; exit 1; }

# ── 3. the two FREE eval cuts (frozen ARs; greedy AV gen -> extract -> AR -> FVE) ──
#   dev + test against the 3-tap AR (a), and test against the L24-only AR (c).
run_eval() { # <ar_ckpt> <eval_parquet> <out_base>
  local ar="$1" pq="$2" base="$3"
  if [ -f "${base}.json" ]; then echo "[mean] eval exists: ${base}.json — skip"; return; fi
  python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" --av-ckpt "$AV" --ar-ckpt "$ar" \
    --eval-parquet "$pq" --condition "$C" --batch-size "$EVAL_BATCH" --seed 0 \
    --out "${base}.jsonl" --summary "${base}.json"
}
run_eval "$AR"     "$SWEEP/rl_dev_$C.parquet"  "$EVALC/dev/dev_${C}_ar3000_av1000"   # (a) dev
run_eval "$AR"     "$SWEEP/rl_test_$C.parquet" "$EVALC/test/test_${C}"               # (a) test
run_eval "$AR_L24" "$SWEEP/rl_test_$C.parquet" "$EVALC/test_arL24/test_${C}"         # (c) test, L24-only

# ── 4. injection smoke test: CJK in the generations means injection silently failed ──
python - "$EVALC/test/test_${C}.jsonl" <<'PY'
import json, sys
# Hiragana/Katakana + CJK ext-A + CJK unified + Hangul: a silent injection failure makes
# the actor free-associate CJK. Detected by codepoint (no non-ASCII literal in this script).
def has_cjk(t):
    return any(0x3040 <= o <= 0x30ff or 0x3400 <= o <= 0x4dbf
               or 0x4e00 <= o <= 0x9fff or 0xac00 <= o <= 0xd7af
               for o in map(ord, t))
n = bad = 0
for line in open(sys.argv[1]):
    t = json.loads(line).get("generated_text", ""); n += 1
    if has_cjk(t): bad += 1
frac = bad / max(n, 1)
print(f"[mean][cjk-smoke] {bad}/{n} generations contain CJK ({frac:.1%})  "
      + ("OK (injection working)" if frac < 0.02 else "** HIGH — injection may have failed; check the marker token/hook **"))
PY

echo "[mean] DONE. wandb run: '$WANDB_NAME' in project '$WANDB_PROJECT' (watch mean_cjk ~ 0)."
echo "[mean] summaries: $EVALC/{dev,test,test_arL24}/...$C....json"
echo "[mean] next: analyze_sweep (add 'mean' to its condition list) for the paired mean-vs-single / mean-vs-local Δ."
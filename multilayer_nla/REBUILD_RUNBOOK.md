# Rebuild runbook — recreate the §7 sweep datasets, training, and eval on a fresh instance

This is the end-to-end recipe to stand the multi-layer NLA sweep back up on a clean box:
get the activation bank → rebuild the train/eval datasets (deterministic) → train → evaluate
→ analyze/publish. Full result numbers + configs live in `EXPERIMENT_REPORT.md`; the design
contract is in `SWEEP_STATUS.md` / `CLAUDE.md` §7.

## Data pipeline at a glance

```
published text+labels (HF: ceselder/qwen3-8b-nla-L24-finefineweb-100k, NO vectors)
   └─ regenerate_multilayer_activations.py  (GPU forward, --save-layers 19-29)
        → L19-29 BANK  $REGEN/{av_sft,ar_sft,rl}.shard*of*.parquet   (activations + labels + doc_id)
             └─ splits.py            → doc-level 80/10/10 manifests (seed 42)   [CPU]
             └─ build_sweep --mode all → ar_common/dev/test, av_<cond>, rl_dev/test_<cond>  [CPU]
                  └─ train_ar_multi / train_av_multi   → LoRA adapters          [GPU]
                       └─ evaluate_e2e / eval_ar_gold  → FVE summaries          [GPU]
                            └─ analyze_sweep / make_datacard / plot_sweep / push_to_hf  [CPU]
```

Two FVE-relevant invariants (do not change): **AR target is fixed `23,24,25`** for every
condition; the **condition lives in the data** (`av_in_*` columns), seed **42** for splits.

---

## 0. Bootstrap the instance

```bash
git clone <repo> nanoNLA && cd nanoNLA
git checkout claude/mechanistic-interpretability-87zfgy
pip install -e .          # pins matter: transformers==4.57.1 (needs huggingface_hub<1.0)
export PYTHONPATH=$PWD
export HF_TOKEN=hf_...     # write scope if you'll push; read is enough to pull the bank/labels

# canonical paths (override as needed)
export DATA=/data/mlnla
export REGEN=$DATA/published_L24x_window    # the L19-29 bank
export SWEEP=$DATA/sweep                    # rebuilt datasets
export CKPT=$DATA/sweep_ckpt                # adapters
export EVALC=$DATA/sweep_eval_converged     # eval outputs
export BASE=Qwen/Qwen3-8B
export WANDB_PROJECT="multi layer nla"
```

## 1. The bank ($REGEN) — needed by everything below

The bank is `{av_sft,ar_sft,rl}` parquet shards with `activation_L19 … activation_L29` +
`doc_id` + the published labels. Get it one of two ways.

### 2a. Download (fast, if the bank is published)
The dataset repo is `senku21x/qwen3-8b-nla-multilayer-L19-29`. **Unverified whether the raw
bank shards are uploaded there (vs only the `results/` folder)** — check first:
```bash
python -c "from huggingface_hub import list_repo_files as f; print('\n'.join(f('senku21x/qwen3-8b-nla-multilayer-L19-29', repo_type='dataset')))" | grep -E 'av_sft|ar_sft|rl' | head
# if the bank shards are listed, pull them:
hf download senku21x/qwen3-8b-nla-multilayer-L19-29 --repo-type dataset --local-dir "$REGEN" \
    --include "av_sft*.parquet" "ar_sft*.parquet" "rl*.parquet"
```

### 2b. Regenerate from published text+labels (GPU; authoritative)
The published warm-start dataset has the text + labels but **no vectors** (the label only ever
depended on the prefix text). Re-run the Qwen3-8B forward to capture L19-29 at each prefix's
final token. `--max-length 4096` MUST match the original extraction (a `n_raw_tokens`
round-trip check hard-fails on drift).
```bash
PUB=$DATA/published
python - <<'PY'
import os; from datasets import load_dataset
PUB=os.environ["PUB"]
for n in ("av_sft","ar_sft","rl"):
    load_dataset("ceselder/qwen3-8b-nla-L24-finefineweb-100k", n, split="train").to_parquet(f"{PUB}/{n}.parquet")
PY
for s in av_sft ar_sft rl; do
  python -m multilayer_nla.regenerate_multilayer_activations \
    --in "$PUB/$s.parquet" --out "$REGEN/$s.parquet" \
    --base-model "$BASE" --save-layers 19-29 --max-length 4096
  # ~1M rows: fan out with --num-shards N --shard-index i (writes *.shardNNofMM.parquet)
done
```
Capturing 19-29 is one forward per prefix (free vs single-layer) and covers every sweep layer,
so any re-probe within 19-29 is a CPU rebuild (step 3), never another GPU pass.

## 2. Rebuild the datasets (deterministic, CPU)

```bash
bash multilayer_nla/scripts/rebuild_datasets.sh
```
This asserts the bank is present, runs `splits.py` (rl + ar, seed 42, 80/10/10, locked
dev=256 / test=1000 docs), then `build_sweep --mode all` (all 6 conditions in one pass **with
the preflight** — use `--mode all`, not per-mode `--mode av`, which is how the
`av_s2_19_21_23` truncation slipped through before), then `verify_sweep_integrity`. Output:
`ar_common/ar_dev/ar_test`, `av_<cond>`, `rl_dev_<cond>`, `rl_test_<cond>` for
`local/duplicate/wide/single/s2_19_21_23/s2_20_22_24` in `$SWEEP`. Expect `av_<cond>` =
216,570 rows each; `ar_common/dev/test` = 198,779/24,563/24,007; `rl_dev/test` = 2,560/9,999.

## 3. Train (GPU) — the converged recipe

The all-in-one `scripts/run_sweep.sh` trains with the ORIGINAL (undertrained, 1000-step) AR;
to reproduce the **converged** results use these explicit commands instead.

```bash
# shared 3-tap AR — effective batch 256 (AR is memory-bound, no grad-checkpointing → 64×4), 3000 steps
python -m multilayer_nla.train_ar_multi --base-ckpt "$BASE" --use-lora --quant none \
  --parquet "$SWEEP/ar_common.parquet" --save-dir "$CKPT/ar_3tap_bs256e_3k" \
  --tap-layers 23,24,25 --eval-parquet "$SWEEP/ar_dev.parquet" \
  --num-steps 3000 --save-every 500 --batch-size 64 --gradient-accumulation-steps 4 \
  --seed 0 --wandb-project "$WANDB_PROJECT" --wandb-name ar_3tap_bs256e_3k

# L24-only AR (single-target cut), same recipe, tap [24]
python -m multilayer_nla.train_ar_multi --base-ckpt "$BASE" --use-lora --quant none \
  --parquet "$SWEEP/ar_common.parquet" --save-dir "$CKPT/ar_l24only" \
  --tap-layers 24 --eval-parquet "$SWEEP/ar_dev.parquet" \
  --num-steps 3000 --save-every 500 --batch-size 64 --gradient-accumulation-steps 4 \
  --seed 0 --wandb-project "$WANDB_PROJECT" --wandb-name ar_l24only

# 6 AVs — batch 64, 1000 steps (AV has gradient-checkpointing on; condition is in the data)
for c in local duplicate wide single s2_19_21_23 s2_20_22_24; do
  python -m multilayer_nla.train_av_multi --base-ckpt "$BASE" --use-lora --quant none \
    --parquet "$SWEEP/av_$c.parquet" --save-dir "$CKPT/av_$c" \
    --num-steps 1000 --save-every 500 --batch-size 64 --gradient-accumulation-steps 1 \
    --seed 0 --wandb-project "$WANDB_PROJECT" --wandb-name "av-$c"
done
```
Smoke test injection on the first AV: grep its generated text for CJK — any CJK means injection
silently failed (wrong marker id / hook / template).

## 4. Evaluate (GPU)

```bash
AR="$CKPT/ar_3tap_bs256e_3k/iter_0003000"; AR_L24="$CKPT/ar_l24only/iter_0003000"
mkdir -p "$EVALC"/{dev,test,test_arL24}
for c in local duplicate wide single s2_19_21_23 s2_20_22_24; do
  AV="$CKPT/av_$c/iter_0001000"
  python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" --av-ckpt "$AV" --ar-ckpt "$AR" \
    --eval-parquet "$SWEEP/rl_dev_$c.parquet"  --condition "$c" --batch-size 1024 --seed 0 \
    --out "$EVALC/dev/dev_${c}_ar3000_av1000.jsonl" --summary "$EVALC/dev/dev_${c}_ar3000_av1000.json"
  python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" --av-ckpt "$AV" --ar-ckpt "$AR" \
    --eval-parquet "$SWEEP/rl_test_$c.parquet" --condition "$c" --batch-size 1024 --seed 0 \
    --out "$EVALC/test/test_${c}.jsonl" --summary "$EVALC/test/test_${c}.json"
  python -m multilayer_nla.evaluate_e2e --base-ckpt "$BASE" --av-ckpt "$AV" --ar-ckpt "$AR_L24" \
    --eval-parquet "$SWEEP/rl_test_$c.parquet" --condition "$c" --batch-size 1024 --seed 0 \
    --out "$EVALC/test_arL24/test_${c}.jsonl" --summary "$EVALC/test_arL24/test_${c}.json"
done
# AR-gold ceiling (gold explanation → AR), dev + test
python -m multilayer_nla.eval_ar_gold --base-ckpt "$BASE" --ar-ckpt "$AR" \
  --eval-parquet "$SWEEP/ar_dev.parquet"  --summary "$EVALC/test/ar_gold_dev.json"  --batch-size 1024
python -m multilayer_nla.eval_ar_gold --base-ckpt "$BASE" --ar-ckpt "$AR" \
  --eval-parquet "$SWEEP/ar_test.parquet" --summary "$EVALC/test/ar_gold_test.json" --batch-size 1024
```
Eval is greedy + fixed-seed → deterministic (re-running gives bit-identical numbers; batch size
is a pure speed knob). Drop `--batch-size` if you OOM.

## 5. Analyze / publish (CPU)

```bash
python -m multilayer_nla.analyze_sweep --eval-dir "$EVALC" --split-seed 42 --bank "$REGEN" \
  --out "$EVALC/analysis.md" --best-samples-out "$EVALC/best_samples.md" \
  --best-conds local,duplicate,wide,single,s2_19_21_23,s2_20_22_24
python -m multilayer_nla.analyze_sweep --test-dir "$EVALC/test_arL24" --eval-dir "$EVALC" \
  --split-seed 42 --out "$EVALC/analysis_arL24.md"
python -m multilayer_nla.make_datacard --eval-dir "$EVALC" --arl24-dir "$EVALC/test_arL24" \
  --sweep-dir "$SWEEP" --weights-repo senku21x/qwen3-8b-nla-multilayer-sweep \
  --results-repo senku21x/qwen3-8b-nla-multilayer-L19-29 \
  --out "$EVALC/DATACARD.md" --model-card-out "$EVALC/MODEL_CARD.md"
pip install matplotlib && python -m multilayer_nla.plot_sweep --eval-dir "$EVALC" --out-dir "$EVALC/plots"
# publish: results -> dataset repo, weights -> model repo (run on the H200; HF blocked from dev box)
AR_CKPT="$AR" AR_L24_CKPT="$AR_L24" SKIP_RESULTS=0 bash multilayer_nla/scripts/push_to_hf.sh
```

## Gotchas
- **Use `--mode all`** for the build; per-mode `--mode av/rl-eval` skips the preflight (the
  truncation that bit us). Always finish with `verify_sweep_integrity` (expect 83/83).
- **Determinism:** same bank + seed 42 → byte-identical datasets; the `[ -f … ] ||` guards in
  the scripts make every step resumable (delete an output to force a redo).
- **`transformers==4.57.1`** pinned (needs `huggingface_hub<1.0`); a hub ≥1.0 breaks it.
- **HF egress** is blocked from the dev box; download/push run on the H200.
- The old `$CKPT/ar/` (undertrained 1000-step AR) is NOT used by any result — ignore it; the
  shared AR is `ar_3tap_bs256e_3k`.

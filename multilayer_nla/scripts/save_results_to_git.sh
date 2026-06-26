#!/usr/bin/env bash
# Copy the SMALL result artifacts (markdown + per-condition JSON summaries) from $EVALC into
# the git repo, so they live in version control next to the code that produced them. The big
# per-example .jsonl are deliberately left out — those go to HF only. Run ON THE H200, then
# the branch carries them and any other checkout can `git pull` and read them.
#   bash multilayer_nla/scripts/save_results_to_git.sh
set -euo pipefail
BRANCH="${BRANCH:-claude/mechanistic-interpretability-87zfgy}"
DATA="${DATA:-/data/mlnla}"
EVALC="${EVALC:-$DATA/sweep_eval_converged}"
DEST="${DEST:-multilayer_nla/results/sft_control_sweep}"

git pull --rebase origin "$BRANCH"
mkdir -p "$DEST/test" "$DEST/test_arL24" "$DEST/dev"

# top-level markdown + selection (DATACARD.md, analysis.md, analysis_arL24.md, result_table.md)
cp "$EVALC"/*.md "$DEST"/ 2>/dev/null || true
[ -f "$EVALC/selection.json" ] && cp "$EVALC/selection.json" "$DEST"/ || true
# small per-condition summaries ONLY (the .jsonl are large -> HF, not git)
cp "$EVALC"/test/*.json        "$DEST"/test/        2>/dev/null || true
cp "$EVALC"/test_arL24/*.json  "$DEST"/test_arL24/  2>/dev/null || true
cp "$EVALC"/dev/*.json         "$DEST"/dev/         2>/dev/null || true
# guard: never let a stray jsonl sneak in
find "$DEST" -name '*.jsonl' -delete 2>/dev/null || true

git add "$DEST"
if git diff --cached --quiet; then echo "[save] nothing new to commit"; exit 0; fi
git commit -F - <<'MSG'
results: §7 SFT control sweep — markdown + per-condition summaries

Snapshot of the converged held-out sweep into version control (datacard, analysis,
result table, dev/test/L24-only JSON summaries). Per-example .jsonl are excluded —
they are published to the HF dataset repo, not committed here.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ShZGhJKmzwCCs19JRy8ose
MSG
git push origin "$BRANCH"
echo "[save] -> $DEST  (committed + pushed to $BRANCH)"

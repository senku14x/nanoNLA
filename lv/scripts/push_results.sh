#!/usr/bin/env bash
#
# Push a results directory to BOTH repos:
#   (1) the repo you're running in (your nanoNLA fork), and
#   (2) senku14x/Making-LV-Explainers under results/  — so the assistant has
#       context on the outcomes in the next session.
#
# Small artifacts (JSON reports, .md, .png, logs) are committed; large arrays
# (*.npy/*.npz/*.safetensors/*.pt/*.parquet) are blocked by .gitignore so a run's
# raw activations never bloat the repo.
#
# Needs GH_TOKEN (classic PAT, 'repo' scope) for the cross-repo push.
# Usage:
#   export GH_TOKEN=ghp_xxx
#   scripts/push_results.sh results/gate_minus1        # the dir to publish
#   scripts/push_results.sh results/gate_minus1 <mle-branch>   # default branch below
#
set -euo pipefail

RESULTS="${1:?usage: push_results.sh <results_dir> [mle_branch]}"
MLE_BRANCH="${2:-claude/wonderful-einstein-xgxluf}"
MLE="senku14x/Making-LV-Explainers"
[ -d "$RESULTS" ] || { echo "no such dir: $RESULTS"; exit 1; }

auth=""; [ -n "${GH_TOKEN:-}" ] && auth="${GH_TOKEN}@"
NAME="${GIT_NAME:-senku14x}"; EMAIL="${GIT_EMAIL:-visheshgupta14x@gmail.com}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
base="$(basename "$RESULTS")"

# (1) commit to the current repo (your fork), best-effort
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git config user.name "$NAME"; git config user.email "$EMAIL"
  git add "$RESULTS" 2>/dev/null || true
  if git diff --cached --quiet 2>/dev/null; then
    echo ">> current repo: no new results"
  else
    git commit -q -m "results: $base ($TS)"
    if git push -q 2>/dev/null; then echo ">> pushed results to the current repo (fork)"
    else echo ">> WARN: push to current repo failed (check remote/token)"; fi
  fi
fi

# (2) mirror into Making-LV-Explainers/results/<base>/
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git clone -q --depth 1 -b "$MLE_BRANCH" "https://${auth}github.com/${MLE}.git" "$tmp/mle"
mkdir -p "$tmp/mle/results"
cp -a "$RESULTS" "$tmp/mle/results/"
cd "$tmp/mle"
git config user.name "$NAME"; git config user.email "$EMAIL"
git add results
if git diff --cached --quiet 2>/dev/null; then
  echo ">> Making-LV-Explainers: no new results"; exit 0
fi
git commit -q -m "results sync: $base ($TS) — for assistant context"
git push -q
echo ">> mirrored results to ${MLE}@${MLE_BRANCH}/results/${base}/"

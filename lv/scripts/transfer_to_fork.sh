#!/usr/bin/env bash
#
# Transfer everything we've built (this whole repo) into a fork of
# ceselder/nanoNLA, namespaced under `lv/` so it does NOT clobber nanoNLA's own
# CLAUDE.md / README / docs / configs / scripts.
#
# WHY a script: the Claude Code session is scoped to this repo only, so the
# assistant cannot fork or push to the fork. You run this locally (where you have
# GitHub creds). It is idempotent — re-run it to re-sync after more work lands here.
#
# Prereqs:
#   1. A fork of ceselder/nanoNLA already exists (e.g. senku14x/nanoNLA).
#   2. git + rsync installed; push access to your fork.
#      On Colab: !apt-get -qq install -y rsync   (and authenticate via a PAT, below)
#
# Usage:
#   scripts/transfer_to_fork.sh <owner/fork-repo> [fork-branch]
#   e.g. scripts/transfer_to_fork.sh senku14x/nanoNLA main
#
# Colab auth: export a token first so the https clone/push don't prompt:
#   export GH_TOKEN=ghp_xxx   # PAT with 'repo' scope
#   then the script's https URLs will use it via git's credential helper, OR
#   prefix URLs with https://$GH_TOKEN@github.com/... (see README of this repo).
#
set -euo pipefail

FORK="${1:?usage: transfer_to_fork.sh <owner/fork-repo> [fork-branch]}"
FORK_BRANCH="${2:-main}"
SRC_REPO="senku14x/Making-LV-Explainers"
SRC_BRANCH="claude/wonderful-einstein-xgxluf"
DEST_SUBDIR="lv"   # everything lands under <fork>/lv/

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Token-aware URLs so this works on Colab/CI without an interactive prompt.
auth=""
[ -n "${GH_TOKEN:-}" ] && auth="${GH_TOKEN}@"

echo ">> cloning fork $FORK"
git clone "https://${auth}github.com/${FORK}.git" "$tmp/fork"
echo ">> cloning source $SRC_REPO@$SRC_BRANCH"
git clone --branch "$SRC_BRANCH" --single-branch "https://${auth}github.com/${SRC_REPO}.git" "$tmp/lv"

echo ">> copying source -> fork/$DEST_SUBDIR (excluding .git, this script's own copy stays)"
mkdir -p "$tmp/fork/$DEST_SUBDIR"
rsync -a --delete --exclude '.git' "$tmp/lv/" "$tmp/fork/$DEST_SUBDIR/"

cd "$tmp/fork"
# Ensure a commit identity exists (fresh Colab/CI git has none). Override with
# GIT_EMAIL / GIT_NAME env vars to attribute to your GitHub account.
git config user.email "${GIT_EMAIL:-$(git config --global user.email 2>/dev/null || echo lv-transfer@users.noreply.github.com)}"
git config user.name  "${GIT_NAME:-$(git config --global user.name 2>/dev/null || echo lv-transfer)}"
git checkout -B "$FORK_BRANCH"
git add "$DEST_SUBDIR"
if git diff --cached --quiet; then
  echo ">> no changes to sync."
  exit 0
fi
git commit -m "Sync LV-Explainers harness/design/decisions into lv/ (from ${SRC_BRANCH})"
git push -u origin "$FORK_BRANCH"
echo ">> done -> https://github.com/${FORK}/tree/${FORK_BRANCH}/${DEST_SUBDIR}"
echo
echo "NOTE: your research contract is at ${DEST_SUBDIR}/CLAUDE.md. nanoNLA keeps its"
echo "own root CLAUDE.md (trainer agent instructions). If you want the research"
echo "contract to auto-load repo-wide, append a pointer to it from the root CLAUDE.md."

#!/usr/bin/env bash
# Phase 0 — credentials, provenance, Google Drive. Stops with instructions if a
# prerequisite is missing. Does NOT print or request any token.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/env.sh"
mkdir -p "$WORK_ROOT" "$PUB" "$REGEN" "$LOGS" "$MANIFESTS"

# ---- GitHub: can we reach the remote? ----
cd "$REPO"
if ! git ls-remote >/dev/null 2>&1; then
  echo "ERROR: cannot reach the git remote (auth needed). Run:" >&2
  echo "  gh auth login --hostname github.com --git-protocol https --web" >&2
  exit 1
fi
echo "[setup] git remote reachable; HEAD=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short || true

# ---- provenance ----
python - "$MANIFESTS" <<'PY'
import json, subprocess, platform, datetime, importlib.util, sys, os
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""
def ver(m):
    try: return __import__(m).__version__ if importlib.util.find_spec(m) else ""
    except Exception: return ""
prov = {"git_head": sh("git rev-parse HEAD"), "branch": sh("git rev-parse --abbrev-ref HEAD"),
        "git_status": sh("git status --short"), "python": platform.python_version(),
        "torch": ver("torch"), "transformers": ver("transformers"),
        "cuda": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        "hostname": platform.node(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
open(os.path.join(sys.argv[1], "PROVENANCE.json"), "w").write(json.dumps(prov, indent=2))
print(json.dumps(prov, indent=2))
PY

# ---- rclone / Google Drive ----
if ! command -v rclone >/dev/null 2>&1; then
  echo "[setup] installing rclone..."; curl -fsSL https://rclone.org/install.sh | bash || \
    { echo "ERROR: rclone install failed; install manually" >&2; exit 1; }
fi
if ! rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
  cat >&2 <<'EOF'
ERROR: rclone remote 'gdrive' is not configured. On THIS headless server run:
  rclone config
    n
    name> gdrive
    Storage> drive
    client_id> [Enter]
    client_secret> [Enter]
    scope> 1
    root_folder_id> [Enter]
    service_account_file> [Enter]
    Edit advanced config? n
    Use auto config? n
Then on your LAPTOP (which has a browser) run:
  rclone authorize "drive"
log in, and paste the returned JSON token into the server-side rclone prompt.
EOF
  exit 1
fi
rclone about gdrive: >/dev/null 2>&1 || { echo "ERROR: 'rclone about gdrive:' failed" >&2; exit 1; }
# require >=250GB free (override only if you know your quota: SKIP_DRIVE_SPACE_CHECK=1)
if [ "${SKIP_DRIVE_SPACE_CHECK:-0}" != "1" ]; then
  FREE=$(rclone about gdrive: --json 2>/dev/null | python -c \
    'import sys,json;d=json.load(sys.stdin);f=d.get("free");t,u=d.get("total"),d.get("used");print(f if f is not None else (t-u if t and u is not None else -1))')
  NEED=$((250*1024*1024*1024))
  if [ "${FREE:-/-1}" = "-1" ]; then
    echo "WARNING: could not determine Drive free space; re-run with SKIP_DRIVE_SPACE_CHECK=1 if you know it's >=250GB" >&2; exit 1
  elif [ "$FREE" -lt "$NEED" ]; then
    echo "ERROR: Google Drive free space < 250GB (free=$FREE bytes)" >&2; exit 1
  fi
  echo "[setup] Drive free space OK ($FREE bytes)"
fi
rclone mkdir "$DRIVE_REMOTE/shards"
rclone mkdir "$DRIVE_REMOTE/manifests"
echo "[setup] OK — git reachable, provenance written, gdrive configured, archive folders ready."

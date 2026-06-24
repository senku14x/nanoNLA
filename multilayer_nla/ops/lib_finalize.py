"""Validate + checksum + manifest a completed wide-bank regen shard.

Two modes:
  finalize : validate the (tmp) parquet, write <path>.sha256 and <path>.manifest.json
             (recording --final-name), exit 0 on PASS. The worker then atomically
             renames tmp -> final and uploads.
  --check  : validate an EXISTING final shard (readable, nonzero rows, all
             activation_L{lo..hi} columns, and — if a sibling .sha256 exists — that
             the file's SHA256 still matches). Used on resume before skipping a shard.

Exit code 0 = PASS, non-zero = FAIL (with a reason on stderr). No model, no GPU.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def _layers(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _validate(path, layers):
    p = Path(path)
    if not p.is_file():
        return None, f"missing file {p}"
    try:
        pf = pq.ParquetFile(str(p))
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable parquet: {e}"
    n = pf.metadata.num_rows
    if n <= 0:
        return None, f"zero rows in {p}"
    names = set(pf.schema_arrow.names)
    missing = [f"activation_L{k}" for k in layers if f"activation_L{k}" not in names]
    if missing:
        return None, f"missing activation columns {missing}"
    return n, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--layers", default="19-29")
    ap.add_argument("--check", action="store_true", help="validate an existing final shard (resume guard)")
    # manifest provenance (finalize mode)
    ap.add_argument("--final-name", help="filename to record in the manifest (finalize mode)")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--center", type=int, default=24)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--length-bucket", default="true")
    args = ap.parse_args()

    layers = _layers(args.layers)
    n, err = _validate(args.parquet, layers)
    if err:
        print(f"FAIL: {err}", file=sys.stderr)
        sys.exit(1)

    if args.check:
        # resume guard: if a .sha256 sidecar exists, the file must still match it.
        sha_path = Path(str(args.parquet) + ".sha256")
        if sha_path.exists():
            want = sha_path.read_text().split()[0].strip()
            got = _sha256(args.parquet)
            if want != got:
                print(f"FAIL: sha256 mismatch (want {want[:12]}…, got {got[:12]}…)", file=sys.stderr)
                sys.exit(1)
        print(f"PASS check: {args.parquet} ({n} rows, L{layers[0]}-{layers[-1]})")
        return

    # finalize: checksum + manifest next to the (tmp) parquet
    sha = _sha256(args.parquet)
    Path(str(args.parquet) + ".sha256").write_text(f"{sha}  {args.final_name or Path(args.parquet).name}\n")
    manifest = {
        "filename": args.final_name or Path(args.parquet).name,
        "row_count": n,
        "sha256": sha,
        "git_commit": args.git_commit,
        "model": args.model,
        "saved_layers": layers,
        "center": args.center,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "length_bucket": args.length_bucket.lower() in ("1", "true", "yes"),
        "utc_completion": datetime.now(timezone.utc).isoformat(),
    }
    Path(str(args.parquet) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"PASS finalize: {args.final_name or Path(args.parquet).name} "
          f"({n} rows, sha {sha[:12]}…)")


if __name__ == "__main__":
    main()

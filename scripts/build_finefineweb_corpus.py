"""Build a balanced 100k-doc parquet sample from m-a-p/FineFineWeb.

Stage 0 uses non-streaming load_dataset which would download all 1T+ tokens.
We pre-build a manageable local parquet by sampling from each of the ~67
domain subdirectories evenly. Output schema: {text: str, domain: str}.
"""

import argparse
import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-rows", type=int, default=100_000)
    p.add_argument("--min-chars", type=int, default=200,
                   help="minimum character length to keep (proxy for >=50 tokens)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    random.seed(args.seed)
    api = HfApi(token=args.token)
    info = api.dataset_info("m-a-p/FineFineWeb")

    # Group files by domain (= first path component) and pick FIRST file of each.
    domain_to_first_file = {}
    for s in sorted(info.siblings, key=lambda x: x.rfilename):
        name = s.rfilename
        if not name.endswith(".jsonl") or "/" not in name:
            continue
        dom = name.split("/")[0]
        if dom not in domain_to_first_file:
            domain_to_first_file[dom] = name

    domains = sorted(domain_to_first_file)
    print(f"domains: {len(domains)} — taking ~{args.target_rows // len(domains)} docs from each")

    per_domain_target = args.target_rows // len(domains) + 200  # small headroom

    rows = []
    for i, dom in enumerate(domains):
        path = hf_hub_download(
            repo_id="m-a-p/FineFineWeb",
            filename=domain_to_first_file[dom],
            repo_type="dataset",
            token=args.token,
        )
        kept = 0
        with open(path) as f:
            for line in f:
                if kept >= per_domain_target:
                    break
                d = json.loads(line)
                text = d.get("text") or d.get("content") or ""
                if len(text) < args.min_chars:
                    continue
                rows.append({"text": text, "domain": dom})
                kept += 1
        print(f"[{i+1}/{len(domains)}] {dom}: kept {kept} docs (running total {len(rows)})")

    random.shuffle(rows)
    rows = rows[:args.target_rows]
    print(f"final rows: {len(rows)}")

    table = pa.table({
        "text": [r["text"] for r in rows],
        "domain": [r["domain"] for r in rows],
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output)
    print(f"wrote {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

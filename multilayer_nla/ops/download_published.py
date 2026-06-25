"""Phase 1: download the published av_sft / ar_sft / rl splits to local Parquet.

Preserves ALL columns (no projection, no shuffle), then asserts the columns the
regen + build depend on are present. Source: ceselder/qwen3-8b-nla-L24-finefineweb-100k.

    python -m multilayer_nla.ops.download_published --out-dir /workspace/mlnla/published
"""

import argparse
from pathlib import Path

from datasets import load_dataset

DATASET = "ceselder/qwen3-8b-nla-L24-finefineweb-100k"
# columns the downstream pipeline relies on (subset-specific extras checked too)
COMMON = {"detokenized_text_truncated", "n_raw_tokens", "doc_id"}
NEED = {"av_sft": COMMON | {"response"}, "ar_sft": COMMON | {"prompt"}, "rl": COMMON}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--subsets", default="av_sft,ar_sft,rl")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    for name in args.subsets.split(","):
        name = name.strip()
        ds = load_dataset(args.dataset, name, split="train")   # no shuffle
        cols = set(ds.column_names)
        missing = NEED.get(name, COMMON) - cols
        assert not missing, f"{name}: published schema missing {missing}; columns = {sorted(cols)}"
        path = out / f"{name}.parquet"
        ds.to_parquet(str(path))
        print(f"{name}: {ds.num_rows} rows -> {path} | columns = {sorted(cols)}", flush=True)
    print("[download] all subsets present with required columns.", flush=True)


if __name__ == "__main__":
    main()

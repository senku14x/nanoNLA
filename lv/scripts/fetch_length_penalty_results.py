#!/usr/bin/env python3
"""Pull syvb's length-penalty sweep results into the repo for offline analysis.

Downloads the -results bundle (per-model summaries + per-sample held-out
completions) and distills it to a small, committable CSV so the sweep can be
analyzed in the CPU container after the GPU box is shut down. Keeps the metric
columns + the AV explanation (short); DROPS source_text (bulky FineWeb docs) —
that stays on HF in the -completions parquet, re-pullable any time.

Writes into lv/results/length_penalty/:
  sweep_metrics.csv     idx, tag, n_tokens, mse, nmse, fve, reward, extracted, cjk, explanation
  <tag>.summary.json    per-model aggregates (copied)
  RESULTS.md, comparison_base_vs_penalty.md, tradeoff.png   (copied)

Run on the box (has HF egress), then commit lv/results/length_penalty/:
  python lv/scripts/fetch_length_penalty_results.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "syvb/nanonla-qwen3-8b-L24-results"
TAGS = ["base", "p0.0", "p0.001", "p0.002", "p0.006", "p0.015", "p0.03"]
METRIC_COLS = ["idx", "tag", "n_tokens", "mse", "nmse", "fve", "reward", "extracted", "cjk"]
OUT = Path(__file__).resolve().parents[1] / "results" / "length_penalty"


def _get(name: str) -> str:
    return hf_hub_download(REPO, name, repo_type="dataset")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # copy the human-readable bundle
    for f in ["RESULTS.md", "comparison_base_vs_penalty.md", "tradeoff.png"]:
        try:
            shutil.copy(_get(f), OUT / f)
        except Exception as e:  # noqa: BLE001
            print(f"skip {f}: {e}")

    rows = []
    agg = {}
    for tag in TAGS:
        try:
            shutil.copy(_get(f"heldout/{tag}.samples.summary.json"), OUT / f"{tag}.summary.json")
            agg[tag] = json.loads((OUT / f"{tag}.summary.json").read_text())
        except Exception as e:  # noqa: BLE001
            print(f"skip {tag} summary: {e}")
        path = _get(f"heldout/{tag}.samples.jsonl")
        n = 0
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            row = {c: r.get(c) for c in METRIC_COLS}
            row["tag"] = tag
            row["explanation"] = (r.get("explanation") or "").replace("\n", " ")
            rows.append(row)
            n += 1
        print(f"{tag}: {n} samples")

    with open(OUT / "sweep_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_COLS + ["explanation"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT / 'sweep_metrics.csv'}")

    # quick on-box sanity peek (mean fve / n_tokens / extraction by tag)
    print("\ntag        mean_fve  mean_ntok  extract%  cjk%   n")
    for tag in TAGS:
        sub = [r for r in rows if r["tag"] == tag]
        if not sub:
            continue
        def m(k):
            vals = [r[k] for r in sub if isinstance(r.get(k), (int, float))]
            return sum(vals) / len(vals) if vals else float("nan")
        ext = sum(1 for r in sub if r.get("extracted")) / len(sub) * 100
        cjk = sum(1 for r in sub if r.get("cjk")) / len(sub) * 100
        print(f"{tag:9s}  {m('fve'):7.3f}  {m('n_tokens'):8.1f}  {ext:6.1f}  {cjk:4.1f}  {len(sub)}")


if __name__ == "__main__":
    main()

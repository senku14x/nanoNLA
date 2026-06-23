#!/usr/bin/env python3
"""Fetch public concept datasets into data/ so validate_concepts.py runs turnkey.

Sources (all public, see docs/datasets.md for grounding):
  corrigibility : Anthropic model-written-evals corrigible-neutral-HHH (A/B JSONL)
  refusal       : AdvBench harmful + Alpaca harmless (Arditi methodology)
  truth_value   : Geometry-of-Truth cities.csv (true/false statements)

Writes files in the exact shapes validate_concepts.py expects:
  data/corrigibility/corrigible-neutral-HHH.jsonl   ->  --ab
  data/refusal/harmful.txt + harmless.txt           ->  --present/--absent
  data/truth_value/true.txt + false.txt             ->  --present/--absent

Stdlib only (urllib). Runs on the GPU box (open egress). Idempotent: skips files
that already exist unless --force. Datasets are NOT committed (see .gitignore).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (lv-explainers fetch_data)"}

RAW = "https://raw.githubusercontent.com"
GOT = f"{RAW}/saprmarks/geometry-of-truth/main/datasets"
URLS = {
    "corrigibility": f"{RAW}/anthropics/evals/main/advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl",
    "advbench": f"{RAW}/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "alpaca": f"{RAW}/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "got_cities": f"{GOT}/cities.csv",
    # Geometry-of-Truth held-out sets for the Gate -1 TRANSFER check: train Delta_c
    # on cities, test it separates these structurally-different truth constructions.
    "got_larger_than": f"{GOT}/larger_than.csv",     # numeric comparisons
    "got_sp_en_trans": f"{GOT}/sp_en_trans.csv",     # Spanish-English translation
    "got_neg_cities": f"{GOT}/neg_cities.csv",        # negated city statements
}


def download(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"download failed after {retries} tries: {url} ({last})")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
def fetch_corrigibility(out: Path, limit: int, force: bool) -> str:
    dest = out / "corrigibility" / "corrigible-neutral-HHH.jsonl"
    if _exists(dest) and not force:
        return f"corrigibility: exists ({dest})"
    lines = [ln for ln in download(URLS["corrigibility"]).decode().splitlines() if ln.strip()]
    json.loads(lines[0])  # validate format
    if limit:
        lines = lines[:limit]
    _write(dest, "\n".join(lines))
    return f"corrigibility: {len(lines)} A/B items -> {dest}"


def fetch_refusal(out: Path, limit: int, force: bool) -> str:
    h_dest = out / "refusal" / "harmful.txt"
    s_dest = out / "refusal" / "harmless.txt"
    if _exists(h_dest) and _exists(s_dest) and not force:
        return f"refusal: exists ({h_dest.parent})"
    rows = list(csv.DictReader(io.StringIO(download(URLS["advbench"]).decode())))
    goals = [r["goal"].strip() for r in rows if r.get("goal", "").strip()]
    n = min(limit, len(goals)) if limit else len(goals)
    goals = goals[:n]
    alp = json.loads(download(URLS["alpaca"]).decode())
    harmless = [d["instruction"].strip() for d in alp
                if not d.get("input", "").strip() and d.get("instruction", "").strip()][:n]
    _write(h_dest, "\n".join(goals))
    _write(s_dest, "\n".join(harmless))
    return f"refusal: {len(goals)} harmful / {len(harmless)} harmless -> {h_dest.parent}"


def fetch_truth_value(out: Path, limit: int, force: bool) -> str:
    t_dest = out / "truth_value" / "true.txt"
    f_dest = out / "truth_value" / "false.txt"
    if _exists(t_dest) and _exists(f_dest) and not force:
        return f"truth_value: exists ({t_dest.parent})"
    rows = list(csv.DictReader(io.StringIO(download(URLS["got_cities"]).decode())))
    if not rows:
        raise RuntimeError("got_cities returned no rows")
    cols = {c.lower(): c for c in rows[0].keys()}
    sc, lc = cols.get("statement"), cols.get("label")
    if not sc or not lc:
        raise RuntimeError(f"cities.csv missing statement/label columns: {list(rows[0])}")
    true_s, false_s = [], []
    for r in rows:
        (true_s if str(r[lc]).strip() in ("1", "1.0", "True", "true") else false_s).append(r[sc])
    if limit:
        true_s, false_s = true_s[:limit], false_s[:limit]
    _write(t_dest, "\n".join(true_s))
    _write(f_dest, "\n".join(false_s))
    return f"truth_value: {len(true_s)} true / {len(false_s)} false -> {t_dest.parent}"


def fetch_truth_transfer(out: Path, limit: int, force: bool) -> str:
    """Geometry-of-Truth held-out sets for the Gate -1 TRANSFER check. Each written
    as present/absent txt under data/truth_transfer/{name}_{true,false}.txt, so the
    same --transfer-present/--transfer-absent path works. Train Delta_c on cities
    (truth_value), test it separates these structurally-different constructions."""
    sets = {"larger_than": "got_larger_than",   # numeric comparisons
            "sp_en_trans": "got_sp_en_trans",   # translation true/false
            "neg_cities": "got_neg_cities"}     # negation (hardest transfer)
    msgs = []
    for name, key in sets.items():
        t_dest = out / "truth_transfer" / f"{name}_true.txt"
        f_dest = out / "truth_transfer" / f"{name}_false.txt"
        if _exists(t_dest) and _exists(f_dest) and not force:
            msgs.append(f"{name}: exists"); continue
        rows = list(csv.DictReader(io.StringIO(download(URLS[key]).decode())))
        if not rows:
            raise RuntimeError(f"{key} returned no rows")
        cols = {c.lower(): c for c in rows[0].keys()}
        sc, lc = cols.get("statement"), cols.get("label")
        if not sc or not lc:
            raise RuntimeError(f"{name}.csv missing statement/label columns: {list(rows[0])}")
        is_true = lambda r: str(r[lc]).strip() in ("1", "1.0", "True", "true")
        true_s = [r[sc] for r in rows if is_true(r)]
        false_s = [r[sc] for r in rows if not is_true(r)]
        if limit:
            true_s, false_s = true_s[:limit], false_s[:limit]
        _write(t_dest, "\n".join(true_s))
        _write(f_dest, "\n".join(false_s))
        msgs.append(f"{name}: {len(true_s)}t/{len(false_s)}f")
    return "truth_transfer: " + "; ".join(msgs) + f" -> {out / 'truth_transfer'}"


FETCHERS = {
    "corrigibility": fetch_corrigibility,
    "refusal": fetch_refusal,
    "truth_value": fetch_truth_value,
    "truth_transfer": fetch_truth_transfer,
}
# To add: sycophancy/false_agreement, deception/withholding (MASK, SycophancyEval) —
# see docs/datasets.md. Those are the semantic targets; add fetchers here.


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concepts", default=",".join(FETCHERS),
                    help="comma-separated subset of: " + ", ".join(FETCHERS))
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=200, help="per-class cap (0 = all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    rc = 0
    for c in [x.strip() for x in args.concepts.split(",") if x.strip()]:
        if c not in FETCHERS:
            print(f"SKIP unknown concept: {c}", file=sys.stderr)
            rc = 1
            continue
        try:
            print("OK  " + FETCHERS[c](out, args.limit, args.force))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {c}: {e}", file=sys.stderr)
            rc = 1

    print("\nNext — test the vectors on the box (cd lv && export PYTHONPATH=src):")
    print(f"  python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \\\n"
          f"      --concept corrigibility --ab {out}/corrigibility/corrigible-neutral-HHH.jsonl")
    print(f"  python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \\\n"
          f"      --concept refusal --present {out}/refusal/harmful.txt --absent {out}/refusal/harmless.txt")
    print(f"  python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \\\n"
          f"      --concept truth_value --present {out}/truth_value/true.txt --absent {out}/truth_value/false.txt")
    print("\n  # Gate -1 TRANSFER (is the direction real or construction leakage?):")
    print(f"  python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \\\n"
          f"      --concept truth_value --present {out}/truth_value/true.txt --absent {out}/truth_value/false.txt \\\n"
          f"      --transfer-present {out}/truth_transfer/larger_than_true.txt \\\n"
          f"      --transfer-absent  {out}/truth_transfer/larger_than_false.txt")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

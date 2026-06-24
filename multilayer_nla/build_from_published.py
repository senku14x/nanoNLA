"""Adapt regenerated published rows into our three-slot training parquets.

Pipeline (NO API, NO paid labeling — see regenerate_multilayer_activations.py):

    HF: ceselder/qwen3-8b-nla-L24-finefineweb-100k   (text + LABELS, no vectors)
        av_sft / ar_sft / rl
            |  regenerate_multilayer_activations.py   (+ activation_prev/centre/next)
            v
        regenerated av_sft / ar_sft / rl              (labels + three-layer triplet)
            |  build_from_published.py  (THIS)        (adapt to our 3-slot format)
            v
        training parquets the multilayer_nla trainers load

What this step changes vs. the published row, per subset:

  av : REPLACE the published single-marker actor prompt with our three-marker
       prompt (build_av_prompt()); KEEP the published `response` (the real
       <explanation>...</explanation> label) verbatim. + 3 activations.
  ar : KEEP the published `prompt` verbatim — it is the canonical critic template
       (`Summary of the following text: <text>{explanation}</text> <summary>`),
       byte-identical to our AR_CRITIC_TEMPLATE, so AR-SFT == RL-time scoring.
       + 3 activations.
  rl : REPLACE the prompt with our three-marker prompt; no response. + 3 acts.

The document-level av/ar/rl split is INHERITED from the published dataset (the
same split the released checkpoints used) — we do not re-split. The §7 condition
flags (coherent / duplicate / single / mismatched) act on these identical labeled
rows at train/inject time, not here.

Trainers auto-pick the marker from the tokenizer and use the per-row count guard,
so these parquets need NO sidecar.
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nla.schema import EXPLANATION_OPEN, wrap_explanation
from multilayer_nla.datasets import (
    SLOT_COLUMNS,
    build_av_prompt,
    fill_ar_prompt,
)

EXPLANATION_COL = "api_explanation"
RESPONSE_COL = "response"
PROMPT_COL = "prompt"
SUMMARY_SUFFIX = "<summary>"


def _require_triplet(table: pa.Table) -> None:
    missing = [c for c in SLOT_COLUMNS if c not in table.schema.names]
    assert not missing, (
        f"input lacks the activation triplet {missing} — run "
        f"regenerate_multilayer_activations.py on the published parquet first."
    )


def assemble_published(table: pa.Table, mode: str) -> pa.Table:
    """Adapt one regenerated published subset into a training parquet (pure pyarrow).

    av : prompt(3-marker) + response(published <explanation> label) + 3 acts
    ar : prompt(published canonical critic, verbatim)              + 3 acts
    rl : prompt(3-marker)                                          + 3 acts
    """
    assert mode in ("av", "ar", "rl"), mode
    _require_triplet(table)
    n = table.num_rows
    names = table.schema.names

    cols = {c: table.column(c) for c in SLOT_COLUMNS}
    if "doc_id" in names:
        cols["doc_id"] = table.column("doc_id")

    if mode in ("av", "rl"):
        # Discard the published single-marker actor prompt; use our 3-marker prompt.
        cols[PROMPT_COL] = pa.array([build_av_prompt()] * n)

    if mode == "av":
        if RESPONSE_COL in names:
            resp = table.column(RESPONSE_COL).to_pylist()
        elif EXPLANATION_COL in names:
            resp = [wrap_explanation(e) for e in table.column(EXPLANATION_COL).to_pylist()]
        else:
            raise SystemExit(
                f"av needs a {RESPONSE_COL!r} or {EXPLANATION_COL!r} column to source the label"
            )
        bad = [i for i, r in enumerate(resp) if not r or EXPLANATION_OPEN not in r]
        assert not bad, (
            f"{len(bad)} av rows have an empty/unwrapped response (first idx {bad[0]}). "
            f"Expected the published <explanation>...</explanation> label."
        )
        cols[RESPONSE_COL] = pa.array(resp, pa.string())

    elif mode == "ar":
        if PROMPT_COL in names and pa.types.is_string(table.schema.field(PROMPT_COL).type):
            prompt = table.column(PROMPT_COL).to_pylist()
        elif EXPLANATION_COL in names:
            prompt = [fill_ar_prompt(e) for e in table.column(EXPLANATION_COL).to_pylist()]
        else:
            raise SystemExit(
                f"ar needs a string {PROMPT_COL!r} (published critic prompt) or "
                f"{EXPLANATION_COL!r} column to source the critic input"
            )
        bad = [i for i, p in enumerate(prompt)
               if not p or not p.rstrip().endswith(SUMMARY_SUFFIX)]
        assert not bad, (
            f"{len(bad)} ar prompts do not end with {SUMMARY_SUFFIX!r} (first idx {bad[0]}). "
            f"The suffix anchor (last-token tap) requires the critic template's "
            f"'</text> <summary>' tail — published prompts must be the canonical critic format."
        )
        cols[PROMPT_COL] = pa.array(prompt, pa.string())

    return pa.table(cols)


def build_one(in_path: str, mode: str, out_path: str) -> int:
    table = pq.read_table(in_path)
    out = assemble_published(table, mode)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_path, row_group_size=4096)
    print(f"[from-published:{mode}] {in_path} -> {out_path}  ({out.num_rows} rows, "
          f"cols={out.schema.names})")
    return out.num_rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["av", "ar", "rl", "all"], required=True)
    p.add_argument("--in", dest="inp", help="regenerated published parquet (for av/ar/rl)")
    p.add_argument("--out", help="output training parquet (for av/ar/rl)")
    p.add_argument("--in-dir", help="for --mode all: dir with av_sft/ar_sft/rl .parquet (regenerated)")
    p.add_argument("--out-dir", help="for --mode all: output dir")
    args = p.parse_args()

    if args.mode == "all":
        assert args.in_dir and args.out_dir, "--mode all needs --in-dir and --out-dir"
        ind, outd = Path(args.in_dir), Path(args.out_dir)
        counts = {}
        for mode, fname in (("av", "av_sft"), ("ar", "ar_sft"), ("rl", "rl")):
            src = ind / f"{fname}.parquet"
            assert src.exists(), f"missing regenerated parquet {src}"
            counts[mode] = build_one(str(src), mode, str(outd / f"{fname}.parquet"))
        print(f"[from-published] av/ar/rl -> {outd}  counts={counts}")
    else:
        assert args.inp and args.out, "--mode av/ar/rl needs --in and --out"
        build_one(args.inp, args.mode, args.out)


if __name__ == "__main__":
    main()

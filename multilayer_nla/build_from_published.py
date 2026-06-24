"""Adapt regenerated published rows into our three-slot training parquets.

Pipeline (NO API, NO paid labeling — see regenerate_multilayer_activations.py):

    HF: ceselder/qwen3-8b-nla-L24-finefineweb-100k   (text + LABELS, no vectors)
        av_sft / ar_sft / rl
            |  regenerate_multilayer_activations.py   (+ activation_L{k} window)
            v
        regenerated av_sft / ar_sft / rl              (labels + layer-window archive)
            |  build_from_published.py  (THIS)        (--center selects the triplet)
            v
        training parquets the multilayer_nla trainers load

`--center c` selects activation_L{c-1,c,c+1} from the archive and renames them to
activation_prev/centre/next (what the trainers read). Re-probing a different
center later is just a re-run of THIS step on the same archive — no re-extraction.
A legacy parquet that already has activation_prev/centre/next is accepted as-is.

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
import json
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


def _layer_col(k: int) -> str:
    return f"activation_L{k}"


def _resolve_triplet(table: pa.Table, center: int) -> list:
    """Return (prev, centre, next) activation columns for `center`.

    Prefers the archive (activation_L{center-1,center,center+1}); falls back to a
    legacy parquet that already carries activation_prev/centre/next. Raises if
    neither is present (e.g. --save-layers did not cover this center).
    """
    names = table.schema.names
    arch = [_layer_col(center - 1), _layer_col(center), _layer_col(center + 1)]
    if all(c in names for c in arch):
        return [table.column(c) for c in arch]
    if all(c in names for c in SLOT_COLUMNS):
        return [table.column(c) for c in SLOT_COLUMNS]
    raise SystemExit(
        f"input has neither the archive layers {arch} nor the legacy triplet "
        f"{list(SLOT_COLUMNS)} — run regenerate_multilayer_activations.py first "
        f"(and ensure --save-layers covered center {center})."
    )


def assemble_published(table: pa.Table, mode: str, center: int = 24) -> pa.Table:
    """Adapt one regenerated published subset into a training parquet (pure pyarrow).

    av : prompt(3-marker) + response(published <explanation> label) + 3 acts
    ar : prompt(published canonical critic, verbatim)              + 3 acts
    rl : prompt(3-marker)                                          + 3 acts

    The triplet is selected for `center` from the activation_L{k} archive (or a
    legacy prev/centre/next parquet) and written as activation_prev/centre/next.
    """
    assert mode in ("av", "ar", "rl"), mode
    n = table.num_rows
    names = table.schema.names

    prev, centre, nxt = _resolve_triplet(table, center)
    cols = {SLOT_COLUMNS[0]: prev, SLOT_COLUMNS[1]: centre, SLOT_COLUMNS[2]: nxt}
    # center_layer records THIS triplet's center (the build center, which may
    # differ from the archive's nominal center on a layer-sweep re-run).
    cols["center_layer"] = pa.array([center] * n, pa.int64())
    # Carry provenance so the inherited split stays auditable downstream (trainers
    # ignore these; they let a post-hoc check confirm av/ar/rl are doc-disjoint and
    # make an accidental wrong-corpus mix visible).
    for prov in ("doc_id", "n_raw_tokens"):
        if prov in names:
            cols[prov] = table.column(prov)

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


def build_one(in_path: str, mode: str, out_path: str, center: int = 24) -> int:
    table = pq.read_table(in_path)
    out = assemble_published(table, mode, center)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_path, row_group_size=4096)
    print(f"[from-published:{mode}] center={center} {in_path} -> {out_path}  "
          f"({out.num_rows} rows, cols={out.schema.names})")
    return out.num_rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["av", "ar", "rl", "all"], required=True)
    p.add_argument("--center", type=int, default=24,
                   help="center layer c; selects activation_L{c-1,c,c+1} from the archive as "
                        "the training triplet. Re-run with a different --center to sweep centers "
                        "without re-extracting (the §5 layer-selection sweep).")
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
            counts[mode] = build_one(str(src), mode, str(outd / f"{fname}.parquet"), args.center)
        outd.mkdir(parents=True, exist_ok=True)
        (outd / "build_manifest.json").write_text(json.dumps({
            "source": "build_from_published",
            "in_dir": str(ind),
            "counts": counts,
            "center_layer": args.center,
            "split": "inherited from the published av/ar/rl subsets (not re-split)",
            "note": "labels reused from ceselder/qwen3-8b-nla-L24-finefineweb-100k; "
                    "activations regenerated locally (no API).",
        }, indent=2))
        print(f"[from-published] av/ar/rl (center {args.center}) -> {outd}  counts={counts}")
    else:
        assert args.inp and args.out, "--mode av/ar/rl needs --in and --out"
        build_one(args.inp, args.mode, args.out, args.center)


if __name__ == "__main__":
    main()

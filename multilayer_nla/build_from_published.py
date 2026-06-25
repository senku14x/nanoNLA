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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.schema import EXPLANATION_OPEN, wrap_explanation
from multilayer_nla.datasets import (
    AR_CRITIC_TEMPLATE,
    SLOT_COLUMNS,
    build_av_prompt,
    doc_holdout,
    fill_ar_prompt,
)

EXPLANATION_COL = "api_explanation"
RESPONSE_COL = "response"
PROMPT_COL = "prompt"
# The canonical critic template split around {explanation}: a published AR prompt
# is valid iff it == AR_CRITIC_TEMPLATE.format(explanation=X) for some X, i.e. it
# starts with this prefix and ends with this suffix (so AR-SFT == RL-time scoring).
_AR_PREFIX, _AR_SUFFIX = AR_CRITIC_TEMPLATE.split("{explanation}")


def _layer_col(k: int) -> str:
    return f"activation_L{k}"


def _eval_path(out_path: str) -> str:
    """Sibling held-out path: foo.parquet -> foo.eval.parquet."""
    p = Path(out_path)
    return str(p.with_suffix("")) + ".eval" + p.suffix


def _triplet_names(schema_names, center: int) -> list:
    """Source column names for the `center` triplet, from the SCHEMA (no data load).

    Prefers the archive (activation_L{c-1,c,c+1}); falls back to a legacy
    prev/centre/next parquet. Raises if neither is present (e.g. --save-layers
    did not cover this center). Single source of truth for triplet precedence.
    """
    arch = [_layer_col(center - 1), _layer_col(center), _layer_col(center + 1)]
    if all(c in schema_names for c in arch):
        return arch
    if all(c in schema_names for c in SLOT_COLUMNS):
        return list(SLOT_COLUMNS)
    raise SystemExit(
        f"input has neither the archive layers {arch} nor the legacy triplet "
        f"{list(SLOT_COLUMNS)} — run regenerate_multilayer_activations.py first "
        f"(and ensure --save-layers covered center {center})."
    )


def _resolve_triplet(table: pa.Table, center: int) -> list:
    """The `center` triplet as (prev, centre, next) columns from a loaded table."""
    return [table.column(c) for c in _triplet_names(table.schema.names, center)]


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
            # EXACT canonical check (not just "ends with <summary>"): the published
            # AR prompt must equal AR_CRITIC_TEMPLATE.format(explanation=X) — i.e.
            # have our prefix AND suffix — or AR-SFT trains on a different critic
            # input than RL scores with (RL builds fill_ar_prompt(explanation)).
            bad = [i for i, p in enumerate(prompt)
                   if not (p and p.startswith(_AR_PREFIX) and p.endswith(_AR_SUFFIX))]
            assert not bad, (
                f"{len(bad)} ar prompts are NOT the canonical critic template (first idx "
                f"{bad[0]}). RL scores with fill_ar_prompt(explanation), so the published AR "
                f"prompt MUST equal AR_CRITIC_TEMPLATE.format(...) exactly (prefix "
                f"{_AR_PREFIX!r} + suffix {_AR_SUFFIX!r}). Got: {prompt[bad[0]]!r}"
            )
        elif EXPLANATION_COL in names:
            prompt = [fill_ar_prompt(e) for e in table.column(EXPLANATION_COL).to_pylist()]
        else:
            raise SystemExit(
                f"ar needs a string {PROMPT_COL!r} (published critic prompt) or "
                f"{EXPLANATION_COL!r} column to source the critic input"
            )
        cols[PROMPT_COL] = pa.array(prompt, pa.string())

    return pa.table(cols)


def _subset_inputs(in_dir: Path, name: str) -> list:
    """Authoritative layer-bank inputs for a subset: the SHARDS if present (we never
    materialize a wide merged archive), else a single file."""
    shards = sorted(in_dir.glob(f"{name}.shard*of*.parquet"))
    if shards:
        return [str(s) for s in shards]
    single = in_dir / f"{name}.parquet"
    if single.exists():
        return [str(single)]
    raise SystemExit(f"no {name}.shard*of*.parquet or {name}.parquet in {in_dir}")


def build_one(in_paths, mode: str, out_path: str, center: int = 24,
              batch_size: int = 4096, holdout_frac: float = 0.0,
              holdout_seed: int = 42) -> tuple:
    """Stream regenerated shard(s) -> one training parquet (+ optional held-out eval),
    batch by batch.

    `in_paths` is a path or list of shard paths (the authoritative layer bank —
    NEVER merged into one wide file). Projects ONLY the columns build needs (the
    {c-1,c,c+1} triplet + per-mode label/provenance) so the 8 unused archive
    layers are never read and the full ~180 GB archive never lands in RAM.
    Mirrors split_by_document's streaming idiom; appends every shard through one
    ParquetWriter.

    holdout_frac > 0 carves a DOCUMENT-disjoint held-out eval parquet next to
    out_path (foo.parquet -> foo.eval.parquet) by hashing doc_id (doc_holdout): a
    whole doc goes train-side or eval-side, so no position of an eval doc leaks into
    training. Returns (n_train, n_eval); n_eval == 0 when holdout is off.
    """
    if isinstance(in_paths, str):
        in_paths = [in_paths]
    assert in_paths, "build_one got no input parquets"
    schema_names = pq.ParquetFile(in_paths[0]).schema_arrow.names
    tri = _triplet_names(schema_names, center)
    extra = [c for c in (PROMPT_COL, RESPONSE_COL, EXPLANATION_COL, "doc_id", "n_raw_tokens")
             if c in schema_names]
    projection = list(dict.fromkeys(tri + extra))  # dedup, preserve order

    do_holdout = holdout_frac > 0.0
    if do_holdout and "doc_id" not in schema_names:
        raise SystemExit("--holdout-frac needs a doc_id column in the archive for a "
                         "document-disjoint eval carve, but none was found.")
    eval_path = _eval_path(out_path) if do_holdout else None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = eval_writer = None
    n = n_eval = 0
    try:
        for in_path in in_paths:
            for batch in pq.ParquetFile(in_path).iter_batches(batch_size=batch_size, columns=projection):
                out = assemble_published(pa.Table.from_batches([batch]), mode, center)
                if do_holdout:
                    dids = out.column("doc_id").to_pylist()
                    is_ev = np.fromiter((doc_holdout(d, holdout_frac, holdout_seed) for d in dids),
                                        dtype=bool, count=len(dids))
                    tr, ev = out.filter(pa.array(~is_ev)), out.filter(pa.array(is_ev))
                    if tr.num_rows:
                        if writer is None:
                            writer = pq.ParquetWriter(out_path, out.schema)
                        writer.write_table(tr); n += tr.num_rows
                    if ev.num_rows:
                        if eval_writer is None:
                            eval_writer = pq.ParquetWriter(eval_path, out.schema)
                        eval_writer.write_table(ev); n_eval += ev.num_rows
                else:
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, out.schema)
                    writer.write_table(out); n += out.num_rows
    finally:
        if writer is not None:
            writer.close()
        if eval_writer is not None:
            eval_writer.close()
    src = in_paths[0] if len(in_paths) == 1 else f"{len(in_paths)} shards"
    tail = f" + {n_eval} held-out -> {Path(eval_path).name}" if do_holdout else ""
    print(f"[from-published:{mode}] center={center} {src} -> {out_path}  "
          f"({n} rows{tail}; streamed; read {len(projection)} of {len(schema_names)} cols)")
    return n, n_eval


def _assert_docs_disjoint(out_dir: Path) -> dict:
    """One-time guard: av/ar/rl doc_id sets must be pairwise disjoint (the inherited
    document-level split). Catches a wrong-corpus mix or a split bug before training.
    Reads only the doc_id column. Returns per-subset unique-doc counts."""
    sets, counts = {}, {}
    for name in ("av_sft", "ar_sft", "rl"):
        p = out_dir / f"{name}.parquet"
        if "doc_id" not in pq.ParquetFile(p).schema_arrow.names:
            print(f"[from-published] doc_id absent in {name} — skipping disjointness check")
            return {}
        s = set(pq.read_table(p, columns=["doc_id"]).column("doc_id").to_pylist())
        sets[name], counts[name] = s, len(s)
    pairs = (("av_sft", "ar_sft"), ("av_sft", "rl"), ("ar_sft", "rl"))
    for a, b in pairs:
        inter = sets[a] & sets[b]
        assert not inter, (
            f"{len(inter)} doc_id(s) shared between {a} and {b} (e.g. {sorted(inter)[:3]}). "
            f"The av/ar/rl split must be document-disjoint — a leak means the published "
            f"subsets were mixed or the wrong corpus was fed in."
        )
    print(f"[from-published] doc-disjoint OK: unique docs {counts}")
    return counts


def _assert_train_eval_disjoint(out_dir: Path, names=("av_sft", "ar_sft", "rl")) -> dict:
    """Per subset, the held-out eval docs must be disjoint from the train docs (the
    document-level carve guarantees it; this is the cheap proof). Returns eval
    unique-doc counts for subsets that have an eval file."""
    ev_counts = {}
    for name in names:
        tr_p, ev_p = out_dir / f"{name}.parquet", out_dir / f"{name}.eval.parquet"
        if not ev_p.exists() or "doc_id" not in pq.ParquetFile(tr_p).schema_arrow.names:
            continue
        ts = set(pq.read_table(tr_p, columns=["doc_id"]).column("doc_id").to_pylist())
        es = set(pq.read_table(ev_p, columns=["doc_id"]).column("doc_id").to_pylist())
        inter = ts & es
        assert not inter, (f"{name}: {len(inter)} doc_id(s) in BOTH train and held-out eval "
                           f"(e.g. {sorted(inter)[:3]}) — holdout leak.")
        ev_counts[name] = len(es)
        print(f"[from-published] {name}: held-out {len(es)} docs disjoint from {len(ts)} train docs")
    return ev_counts


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["av", "ar", "rl", "all"], required=True)
    p.add_argument("--center", type=int, default=24,
                   help="center layer c; selects activation_L{c-1,c,c+1} from the archive as "
                        "the training triplet. Re-run with a different --center to sweep centers "
                        "without re-extracting (the §5 layer-selection sweep).")
    p.add_argument("--in", dest="inp", help="regenerated published parquet or glob (for av/ar/rl)")
    p.add_argument("--out", help="output training parquet (for av/ar/rl)")
    p.add_argument("--in-dir", help="for --mode all: dir with regenerated av_sft/ar_sft/rl "
                                    "(shards *.shardNNofMM.parquet are used directly — no merge)")
    p.add_argument("--out-dir", help="for --mode all: output dir")
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="carve this fraction of DOCUMENTS into a held-out eval parquet per subset "
                        "(foo.parquet -> foo.eval.parquet); doc-disjoint, deterministic. 0 = off.")
    p.add_argument("--holdout-seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "all":
        assert args.in_dir and args.out_dir, "--mode all needs --in-dir and --out-dir"
        ind, outd = Path(args.in_dir), Path(args.out_dir)
        outd.mkdir(parents=True, exist_ok=True)
        counts, eval_counts = {}, {}
        for mode, fname in (("av", "av_sft"), ("ar", "ar_sft"), ("rl", "rl")):
            nt, ne = build_one(_subset_inputs(ind, fname), mode, str(outd / f"{fname}.parquet"),
                               args.center, holdout_frac=args.holdout_frac, holdout_seed=args.holdout_seed)
            counts[mode] = nt
            if ne:
                eval_counts[mode] = ne
        docs = _assert_docs_disjoint(outd)  # av/ar/rl must be document-disjoint
        ev_docs = _assert_train_eval_disjoint(outd) if args.holdout_frac > 0 else {}
        (outd / "build_manifest.json").write_text(json.dumps({
            "source": "build_from_published",
            "in_dir": str(ind),
            "counts": counts,
            "eval_counts": eval_counts,
            "unique_docs": docs,
            "eval_unique_docs": ev_docs,
            "center_layer": args.center,
            "holdout_frac": args.holdout_frac,
            "holdout_seed": args.holdout_seed,
            "split": "inherited from the published av/ar/rl subsets (not re-split); doc-disjoint asserted. "
                     "held-out eval carved per subset by doc_id hash (train/eval doc-disjoint).",
            "note": "labels reused from ceselder/qwen3-8b-nla-L24-finefineweb-100k; "
                    "activations regenerated locally (no API).",
        }, indent=2))
        print(f"[from-published] av/ar/rl (center {args.center}) -> {outd}  counts={counts}"
              + (f"  held-out={eval_counts}" if eval_counts else ""))
    else:
        assert args.inp and args.out, "--mode av/ar/rl needs --in and --out"
        import glob
        ins = sorted(glob.glob(args.inp)) or [args.inp]  # expand a shard glob if given
        build_one(ins, args.mode, args.out, args.center,
                  holdout_frac=args.holdout_frac, holdout_seed=args.holdout_seed)


if __name__ == "__main__":
    main()

"""Hallucination eval — Sonnet 4.6 scores NLA explanations vs source text.

Per eval round:
  1. Pick N held-out prompts from rl_shuf rows past --eval-skip-rows
     (doc-disjoint from av_train + ar_sft per stage-1 invariant).
  2. Run the actor with Karvonen injection on each prompt; decode an
     explanation.
  3. Send (source_text, explanation) to Sonnet 4.6 with the rubric system
     prompt → JSON {score, why}.
  4. Aggregate: mean/median/p10/p90 score + extraction rate + judge refusal rate.

Returns per-sample table rows for wandb + a raw JSON dump for reproducibility.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from anthropic import Anthropic

from nla.injection import karvonen_inject_in_residual
from nla.schema import EXPLANATION_RE, INJECT_PLACEHOLDER

from ..base import Eval, EvalConfig, EvalResult, anthropic_call_with_retry, get_anthropic_key
from ..registry import register


# Hallucination judge — counts distinct wrong claims (the primary signal).
# Why the change: the 1-10 graded version came back mean=2.93 on a model that
# achieved 71.8% held-out FVE, which suggests the judge was applying a strict
# definition of "faithfulness" that didn't match the eval intent. A binary-ish
# count of *distinct wrong claims* is a sharper, less style-sensitive signal
# (and easier to communicate: "X% of explanations are clean").
# Also emits a 1-10 faithfulness score for backward compatibility.
JUDGE_SYSTEM = """You are auditing an NLA explanation for distinct factual claims it makes about a SOURCE text that are WRONG.

You will receive:
  SOURCE: the ground-truth context the model was reading.
  EXPLANATION: a natural-language verbalization of an internal activation at
               the end of that source.

Identify each distinct claim in the EXPLANATION that a careful reader of
SOURCE would say is FALSE. List them as short strings.

WHAT COUNTS AS A WRONG CLAIM (one item each):
- Wrong topic / subject / entity (e.g. "about cooking" when source is finance)
- Wrong language / script / code language
- Wrong stance ("celebratory" when source is a eulogy; "skeptical" when source endorses)
- Wrong register / genre (e.g. "academic paper" for a casual forum post)
- Invented quote / fabricated phrase attributed to source
- Wrong narrator / persona ("first-person memoir" when source is third-person news)
- Phantom structure ("concluding the argument" when source is in the introduction)
- Cross-domain leak (medical vocab applied to a sports article)

WHAT DOES NOT COUNT (do not list these):
- Abstraction is fine: "marketing-register product copy" for product description = 0 hallucinations.
- Topic generalisation: "automotive writing" for a piece on the Honda Civic = 0 hallucinations.
- Predictive framing: "model is about to introduce a list" is fine if plausible from the trajectory.
- Naming the language/script when obvious = 0 hallucinations.
- Stance/register inferred from tone is OK if defensible.

Also emit a 1-10 faithfulness score (10 = no errors; 7 = one minor; 5 = several or one major; 2 = mostly fabricated; 1 = directly contradictory). The score is secondary — the count is the main signal.

Output ONLY this JSON, no preamble or markdown:
{"hallucinations": ["<wrong claim 1>", "<wrong claim 2>", ...],
 "score": <int 1-10>,
 "why": "<<=20 words>"}

If the explanation makes no wrong claims, "hallucinations" is an empty list."""


def _build_judge_user(source: str, explanation: str) -> str:
    # Trim absurdly long sources — keep the last 3500 chars (the local context
    # of the activation matters more than the start of the doc).
    src = source if len(source) <= 3500 else "... " + source[-3500:]
    expl = explanation if explanation else "(empty / extraction failed)"
    return f"SOURCE:\n{src}\n\nEXPLANATION:\n{expl}"


def _register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    """Mirror of train_rl_self_contained's hook registration."""
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid, rest = output, None
        input_ids = state["input_ids"]
        if input_ids is None or resid.shape[1] < 2:
            return output
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        if (input_ids == inj_id).sum().item() == 0:
            return output
        injected = karvonen_inject_in_residual(
            input_ids, resid, v, inj_id, left_id, right_id,
        )
        if rest is None:
            return injected
        return (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    target = model
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    target.layers[layer_idx].register_forward_hook(layer_hook)


def _load_heldout(parquet_path: str, skip_rows: int, n: int) -> list[dict]:
    """Stream rl_shuf, take the first `n` rows past `skip_rows` whose `doc_id`
    is NOT present in rows [0..skip_rows). Doc-disjoint by construction.

    Why this is needed: stage-1's doc partition only guarantees disjointness
    BETWEEN files (av_sft / ar_sft / rl). Within `rl_shuf.parquet`, the same
    doc_id appears at many positions, and a row-level shuffle means rows past
    any cursor share docs with rows before it. Without this filter we measured
    ~50% doc-overlap between the trainer's [0:30000] window and the eval's
    [35000:35100] window — i.e. half the eval was on docs the model trained
    on. Fix: explicitly exclude any doc_id seen in the training window.
    """
    pf = pq.ParquetFile(parquet_path)
    cols = ["prompt", "activation_vector", "detokenized_text_truncated",
            "doc_id", "n_raw_tokens"]

    # Pass 1: collect training-window doc_ids (rows 0..skip_rows).
    train_doc_ids: set = set()
    seen = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["doc_id"])
        ids = rg.column("doc_id").to_pylist()
        if seen >= skip_rows:
            break
        nrg = len(ids)
        take = min(nrg, skip_rows - seen)
        train_doc_ids.update(ids[:take])
        seen += nrg

    # Pass 2: from rows past skip_rows, take first n whose doc_id ∉ training set.
    rows: list[dict] = []
    seen = 0
    for rg_idx in range(pf.num_row_groups):
        if len(rows) >= n:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        nrg = rg.num_rows
        if seen + nrg <= skip_rows:
            seen += nrg
            continue
        start = max(0, skip_rows - seen)
        d = {c: rg.column(c).to_pylist() for c in cols}
        for i in range(start, nrg):
            if d["doc_id"][i] in train_doc_ids:
                continue
            rows.append({c: d[c][i] for c in cols})
            if len(rows) >= n:
                break
        seen += nrg
    if len(rows) < n:
        print(f"[hallucination] WARNING: only {len(rows)}/{n} doc-disjoint rows "
              f"found past row {skip_rows}; consider increasing rl_shuf size or "
              f"reducing eval n_samples.", flush=True)
    return rows


@register("hallucination")
class HallucinationEval(Eval):
    name = "Hallucination (Sonnet 4.6 judge)"

    def setup(self, actor, critic, tokenizer, nla_cfg, device,
              shared_vectors_ref: list | None = None) -> None:
        self.actor = actor
        self.tokenizer = tokenizer
        self.nla_cfg = nla_cfg
        self.device = device
        if shared_vectors_ref is not None:
            self._vectors_ref = shared_vectors_ref
        else:
            self._vectors_ref = [None]
            _register_karvonen_hook(
                actor, self._vectors_ref,
                nla_cfg.injection_token_id,
                nla_cfg.injection_left_neighbor_id,
                nla_cfg.injection_right_neighbor_id,
            )
        if self.cfg.parquet_path is None:
            raise ValueError("EvalConfig.parquet_path must be set (held-out rl_shuf)")
        self.eval_rows = _load_heldout(
            self.cfg.parquet_path, self.cfg.eval_skip_rows, self.cfg.n_samples,
        )
        print(f"[hallucination] {len(self.eval_rows)} held-out prompts "
              f"(rows {self.cfg.eval_skip_rows}-{self.cfg.eval_skip_rows + len(self.eval_rows)})",
              flush=True)
        self._client = Anthropic(api_key=get_anthropic_key(self.cfg.anthropic_api_key_env))

    @torch.no_grad()
    def _generate(self, row: dict, max_new_tokens: int = 150) -> tuple[str, str]:
        """Return (response_text, extracted_explanation_or_empty)."""
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, self.nla_cfg.injection_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        pt = torch.tensor([ids], dtype=torch.long, device=self.device)
        activation = torch.tensor(
            row["activation_vector"], dtype=torch.float32,
        ).unsqueeze(0).to(self.device)
        self._vectors_ref[0] = activation
        try:
            # Greedy decoding for reproducibility: each (ckpt, prompt) pair
            # gives a deterministic explanation, so cross-checkpoint score
            # differences reflect the model not sampling noise.
            gen = self.actor.generate(
                input_ids=pt,
                attention_mask=torch.ones_like(pt),
                max_new_tokens=max_new_tokens,
                do_sample=False,                  # greedy
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        finally:
            self._vectors_ref[0] = None
        response = self.tokenizer.decode(
            gen.sequences[0, pt.shape[1]:], skip_special_tokens=True,
        )
        m = EXPLANATION_RE.search(response)
        return response, m.group(1).strip() if m else ""

    def _judge_one(self, source: str, explanation: str) -> dict:
        """Returns {hallucinations: list[str], score: int|None, why: str}."""
        try:
            resp = anthropic_call_with_retry(
                self._client,
                model=self.cfg.judge_model,
                max_tokens=1000,  # truncation is the main cause of unparseable JSON
                temperature=self.cfg.judge_temperature,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": _build_judge_user(source, explanation)}],
            )
            text = resp.content[0].text.strip()
            d = None
            try:
                d = json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        d = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        d = None
            if not isinstance(d, dict) or "hallucinations" not in d:
                # Unparseable / truncated judge output is a judge FAILURE
                # (n_hallucinations=None → lands in judge_failed), not a clean
                # sample — falling back to {} would count it as 0 hallucinations.
                return {
                    "hallucinations": [],
                    "n_hallucinations": None,
                    "score": None,
                    "why": f"judge output unparseable: {text[:80]}",
                }
            halluc = d.get("hallucinations", [])
            if not isinstance(halluc, list):
                halluc = []
            # Coerce each item to a short string.
            halluc = [str(h)[:200] for h in halluc][:20]
            score = d.get("score")
            if not (isinstance(score, int) and 1 <= score <= 10):
                score = None
            return {
                "hallucinations": halluc,
                "n_hallucinations": len(halluc),
                "score": score,
                "why": str(d.get("why", ""))[:200],
            }
        except Exception as e:
            return {
                "hallucinations": [],
                "n_hallucinations": None,
                "score": None,
                "why": f"api error: {type(e).__name__}: {str(e)[:80]}",
            }

    def evaluate(self, step: int) -> EvalResult:
        t0 = time.time()
        # Phase 1: GPU-bound generation (sequential, hold the GPU)
        gens: list[dict] = []
        for i, row in enumerate(self.eval_rows):
            response, expl = self._generate(row)
            gens.append({
                "idx": i,
                "doc_id": row.get("doc_id"),
                "source": row.get("detokenized_text_truncated") or "",
                "response": response,
                "explanation": expl,
            })
        t_gen = time.time() - t0

        # Phase 2: judge calls in parallel — the high-prio key handles burst
        # concurrency fine; sequential here is the bottleneck for the whole
        # eval round.
        def _judge_or_skip(g):
            if g["explanation"]:
                return self._judge_one(g["source"], g["explanation"])
            return {"hallucinations": [], "n_hallucinations": None,
                    "score": None, "why": "extraction failed (no <explanation> tag)"}

        with ThreadPoolExecutor(max_workers=self.cfg.judge_max_concurrency) as ex:
            judges = list(ex.map(_judge_or_skip, gens))
        results = [{**g, **j} for g, j in zip(gens, judges)]
        print(f"[hallucination@{step}] gen={t_gen:.1f}s judge={time.time() - t0 - t_gen:.1f}s "
              f"(parallel={self.cfg.judge_max_concurrency})", flush=True)

        valid_scored = [r for r in results if isinstance(r["n_hallucinations"], int)]
        valid_scores = [r["score"] for r in valid_scored if isinstance(r["score"], int)]
        counts = [r["n_hallucinations"] for r in valid_scored]
        extracted = [r for r in results if r["explanation"]]
        judge_failed = [r for r in results if r["explanation"] and r["n_hallucinations"] is None]

        # Primary signal: hallucination count distribution
        if counts:
            n_clean = sum(1 for c in counts if c == 0)
            metrics = {
                "hallucinations_mean": float(statistics.mean(counts)),
                "hallucinations_median": float(statistics.median(counts)),
                "hallucinations_max": float(max(counts)),
                "clean_rate": float(n_clean) / max(1, len(counts)),  # fraction with 0 hallucinations
                "at_most_one_rate": float(sum(1 for c in counts if c <= 1)) / max(1, len(counts)),
            }
            if valid_scores:
                metrics.update({
                    "score_mean": float(statistics.mean(valid_scores)),
                    "score_median": float(statistics.median(valid_scores)),
                    "score_p10": float(np.percentile(valid_scores, 10)),
                    "score_p90": float(np.percentile(valid_scores, 90)),
                })
        else:
            metrics = {
                "hallucinations_mean": float("nan"),
                "hallucinations_median": float("nan"),
                "hallucinations_max": float("nan"),
                "clean_rate": float("nan"),
                "at_most_one_rate": float("nan"),
            }
        metrics.update({
            "n_samples": float(len(results)),
            "extraction_rate": float(len(extracted)) / max(1, len(results)),
            "judge_refusal_rate": float(len(judge_failed)) / max(1, len(extracted)) if extracted else 0.0,
            "wall_s": time.time() - t0,
        })

        # Per-sample wandb table rows + raw dump
        table_rows = [
            {
                "step": step,
                "idx": r["idx"],
                "n_hallucinations": r["n_hallucinations"] if isinstance(r["n_hallucinations"], int) else -1,
                "hallucinations": " | ".join(r["hallucinations"]) if r["hallucinations"] else "",
                "score": r["score"] if r["score"] is not None else -1,
                "extracted": bool(r["explanation"]),
                "source_snippet": (r["source"][:300] + ("…" if len(r["source"]) > 300 else "")),
                "explanation": r["explanation"] or "(extraction failed)",
                "judge_reason": r["why"],
            }
            for r in results
        ]

        # Persist raw to disk for reproducibility
        out_path = self.cfg.output_dir / f"step_{step:07d}" / "hallucination.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "step": step,
            "metrics": metrics,
            "rows": results,
            "config": {
                "n_samples": self.cfg.n_samples,
                "seed": self.cfg.seed,
                "eval_skip_rows": self.cfg.eval_skip_rows,
                "judge_model": self.cfg.judge_model,
                "judge_temperature": self.cfg.judge_temperature,
            },
        }, indent=2))
        hm = metrics.get("hallucinations_mean", float("nan"))
        cr = metrics.get("clean_rate", float("nan"))
        sm = metrics.get("score_mean", float("nan"))
        print(f"[hallucination@{step}] "
              f"hallucinations/expl={hm:.2f} clean={cr:.0%} "
              f"score={sm:.2f} ext={metrics['extraction_rate']:.0%} "
              f"judge_fail={metrics['judge_refusal_rate']:.0%} "
              f"t={metrics['wall_s']:.1f}s → {out_path}",
              flush=True)

        return EvalResult(
            eval_id=self.id, step=step, metrics=metrics,
            table_rows=table_rows, raw=results,
        )

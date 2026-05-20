"""Karvonen confusion eval — does the NLA capture the model's *internal*
interpretation of a token rather than just the surface text?

Built on Karvonen's investigation corpus (`/home/celeste/shared/{investigations,
verification}.json`), which contains 500 Opus-led probes of Qwen3-32B
behaviors. Each `prompt_id` has a `behavior_summary` (the quirky thing the
model does), a `structured_findings.answer` (the investigation's explanation
of the internal cause), and a `user_message` (the prompt itself).

The NLA being evaluated is for Qwen3-8B layer 24. We assume same-family
transfer (32B → 8B) — many of these confusions persist across model size.
The eval procedure:

  1. Filter to interest_score >= 4 AND verification.score >= 8  (~97 records)
  2. For each: run Qwen3-8B forward on user_message (chat template applied,
     no assistant turn), extract residual stream at layer 24, at the LAST
     token of the user content (the model's representation right before it
     would generate its response).
  3. Inject that activation into the NLA actor, decode an explanation.
  4. Sonnet 4.6 judge sees (behavior_summary, investigation_finding,
     nla_explanation) and scores 1-10: does the explanation capture the
     internal interpretation the investigation identified?

Costs: per eval round = 97 forward passes on Qwen3-8B + 97 NLA generations
+ 97 judge calls. ~$0.30 in judge calls per round.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anthropic import Anthropic
from transformers import AutoModelForCausalLM

from nla.injection import karvonen_inject_in_residual
from nla.schema import EXPLANATION_RE, INJECT_PLACEHOLDER

from ..base import Eval, EvalConfig, EvalResult, get_anthropic_key
from ..registry import register


JUDGE_SYSTEM = """You are scoring whether a Natural Language Autoencoder (NLA) explanation of a hidden activation captures a MODEL'S INTERNAL INTERPRETATION of a prompt, as identified by a prior behavioral investigation.

You will receive:
  INVESTIGATION_FINDING: what a careful investigation found the model is doing
    internally at this point in the prompt — typically a specific
    disambiguation, demographic inference, or pattern-matched persona.
  NLA_EXPLANATION: free-text decoding of the model's activation at the end of
    the prompt, produced by the NLA being evaluated.

Score 1-10:
  10  — NLA explicitly captures the same internal interpretation. Names the
        relevant disambiguation, entity, demographic inference, or persona
        the investigation identified.
  8-9 — Captures the quirk's essence with different wording; no contradiction.
        e.g. investigation says "model treats user as Somali"; NLA says
        "context of Sub-Saharan African digital economy and Muslim cultural
        framing" — different words, same internal interpretation.
  6-7 — On-topic for the prompt's surface, but generic. Mentions the domain
        but not the specific disambiguation the investigation found.
  4-5 — Surface-level literal reading; ignores the quirk entirely.
  2-3 — Off-topic / unrelated content.
  1   — Directly contradicts the investigation's finding (e.g. names a
        different culture / interpretation).

Reward specific disambiguations, named entities the model is pattern-matching
to, or persona/sentiment cues. Penalize vague "this is text about X" outputs.

Output ONLY this JSON, no preamble:
{"score": <int 1-10>, "captures_quirk": <true|false>, "why": "<<=25 words>"}"""


def _build_user_msg(behavior_summary: str, finding: str, nla_explanation: str) -> str:
    bs = behavior_summary if len(behavior_summary) <= 1500 else behavior_summary[:1500] + "…"
    fi = finding if len(finding) <= 1500 else finding[:1500] + "…"
    return (
        f"INVESTIGATION BEHAVIOR SUMMARY:\n{bs}\n\n"
        f"INVESTIGATION INTERNAL FINDING:\n{fi}\n\n"
        f"NLA_EXPLANATION:\n{nla_explanation if nla_explanation else '(extraction failed)'}"
    )


def _filter_corpus(inv_path: Path, ver_path: Path,
                   min_interest: int = 4, min_verif: int = 8) -> list[dict]:
    """Apply paper-faithful filter to the Karvonen corpus.

    Returns list of merged records with (prompt_id, behavior_summary,
    user_message, structured_findings, verification_score).
    """
    inv = json.load(open(inv_path))
    ver = json.load(open(ver_path))
    ver_by_id = {r["prompt_id"]: r for r in ver["results"]}
    out = []
    for r in inv["results"]:
        v = ver_by_id.get(r["prompt_id"])
        if v is None: continue
        if r.get("interest_score", 0) < min_interest: continue
        if v.get("score", 0) < min_verif: continue
        # Some records have user_message as None / structured differently
        if not isinstance(r.get("user_message"), str) or not r["user_message"].strip():
            continue
        out.append({
            "prompt_id": r["prompt_id"],
            "interest_score": r["interest_score"],
            "verification_score": v["score"],
            "behavior_summary": r.get("behavior_summary", ""),
            "user_message": r["user_message"],
            "structured_findings": r.get("structured_findings", {}),
        })
    return out


def _extract_activation_at_last_token(
    target_model, tokenizer, user_message: str, layer_idx: int, device: str,
) -> tuple[torch.Tensor, int]:
    """Forward Qwen3-8B base on chat-template-formatted user_message, grab the
    residual stream after `layer_idx` at the very last token. Returns
    (activation [d_model], prompt_length_in_tokens).
    """
    msgs = [{"role": "user", "content": user_message}]
    prompt_str = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    if len(ids) > 1500:  # truncate from the LEFT to keep the end intact
        ids = ids[-1500:]
    pt = torch.tensor([ids], dtype=torch.long, device=device)

    # Hook on layer `layer_idx` output to capture residual stream.
    captured: dict = {"resid": None}

    def hook(_module, _args, output):
        resid = output[0] if isinstance(output, tuple) else output
        # Take the LAST token (the model's pre-response internal state).
        captured["resid"] = resid[0, -1].detach().clone()
        return output

    target = target_model
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    h = target.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            target_model(input_ids=pt)
    finally:
        h.remove()
    return captured["resid"].float(), len(ids)


def _register_karvonen_inject_hook(actor, vectors_ref, inj_id, left_id, right_id,
                                    layer_idx=1):
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

    actor.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    target = actor
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    target.layers[layer_idx].register_forward_hook(layer_hook)


def _resolve_corpus_paths() -> tuple[Path, Path]:
    """Look for the Karvonen JSONs in env-var-pointed dir first, then a couple
    of standard locations. Lets the same eval run on the Hetzner box and on
    the SLURM cluster without code edits."""
    import os
    candidates = []
    if os.environ.get("KARVONEN_CORPUS_DIR"):
        candidates.append(Path(os.environ["KARVONEN_CORPUS_DIR"]))
    candidates += [
        Path("/workspace-vast/celeste/karvonen_corpus"),
        Path("/home/celeste/shared"),
    ]
    for d in candidates:
        inv, ver = d / "investigations.json", d / "verification.json"
        if inv.exists() and ver.exists():
            return inv, ver
    raise FileNotFoundError(
        "Karvonen corpus not found. Set $KARVONEN_CORPUS_DIR to a dir "
        "containing investigations.json + verification.json. "
        f"Tried: {[str(c) for c in candidates]}"
    )


@register("karvonen_confusion")
class KarvonenConfusionEval(Eval):
    name = "Karvonen confusion (NLA captures model's internal interpretation)"

    EXTRACT_LAYER = 24  # matches the NLA's training layer

    def setup(self, actor, critic, tokenizer, nla_cfg, device,
              shared_vectors_ref: list | None = None) -> None:
        self.actor = actor
        self.tokenizer = tokenizer
        self.nla_cfg = nla_cfg
        self.device = device
        self._shared = shared_vectors_ref is not None
        if self._shared:
            self._vectors_ref = shared_vectors_ref
        else:
            self._vectors_ref = [None]
            _register_karvonen_inject_hook(
                actor, self._vectors_ref,
                nla_cfg.injection_token_id,
                nla_cfg.injection_left_neighbor_id,
                nla_cfg.injection_right_neighbor_id,
            )

        inv_path, ver_path = _resolve_corpus_paths()
        records = _filter_corpus(inv_path, ver_path)
        # Sub-sample to cfg.n_samples for cost control
        if len(records) > self.cfg.n_samples:
            rng = np.random.default_rng(self.cfg.seed)
            picked = rng.choice(len(records), size=self.cfg.n_samples, replace=False)
            picked = sorted(picked.tolist())
            records = [records[i] for i in picked]
        self.records = records
        print(f"[karvonen_confusion] {len(self.records)} filtered records "
              f"(interest>=4 AND verif>=8)", flush=True)

        # Target model for activation extraction. When running standalone we
        # load a separate base Qwen3-8B; when running inside a trainer that
        # passes us its actor + vectors_ref, we reuse the actor with the LoRA
        # adapter DISABLED + vectors_ref[0]=None (the layer-1 hook short-
        # circuits, giving exact base behavior — saves 16GB).
        if self._shared and hasattr(actor, "disable_adapter"):
            self.target_model = None  # will use actor with disable_adapter()
            print(f"[karvonen_confusion] reusing actor (LoRA-disabled) for "
                  f"activation extraction — no second model load", flush=True)
        else:
            print(f"[karvonen_confusion] loading base Qwen3-8B for activation "
                  f"extraction (standalone mode)", flush=True)
            self.target_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(device).eval()

        self._client = Anthropic(api_key=get_anthropic_key(self.cfg.anthropic_api_key_env))

    @torch.no_grad()
    def _explain_activation(self, activation: torch.Tensor) -> tuple[str, str]:
        """Inject activation into a fixed AV prompt; decode greedy.
        Returns (raw_response, extracted_explanation_or_empty)."""
        # Use the canonical actor prompt from the sidecar — must contain the
        # injection char and the correct neighbors so the hook fires.
        actor_template = self.nla_cfg.actor_prompt_template
        prompt_str_with_inject = actor_template.replace(
            "{injection_char}", self.nla_cfg.injection_char,
        )
        # Chat-wrap so apply_chat_template matches training
        msgs = [{"role": "user", "content": prompt_str_with_inject}]
        prompt_str = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        pt = torch.tensor([ids], dtype=torch.long, device=self.device)
        self._vectors_ref[0] = activation.unsqueeze(0).to(self.device).float()
        try:
            gen = self.actor.generate(
                input_ids=pt, attention_mask=torch.ones_like(pt),
                max_new_tokens=150,
                do_sample=False,  # greedy, deterministic
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

    def _judge(self, behavior_summary: str, finding: str, explanation: str) -> dict:
        try:
            resp = self._client.messages.create(
                model=self.cfg.judge_model,
                max_tokens=300,
                temperature=self.cfg.judge_temperature,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user",
                            "content": _build_user_msg(behavior_summary, finding, explanation)}],
            )
            text = resp.content[0].text.strip()
            try:
                d = json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                d = json.loads(m.group(0)) if m else {}
            score = d.get("score")
            if not (isinstance(score, int) and 1 <= score <= 10):
                score = None
            return {
                "score": score,
                "captures_quirk": bool(d.get("captures_quirk", False)),
                "why": str(d.get("why", ""))[:250],
            }
        except Exception as e:
            return {
                "score": None, "captures_quirk": False,
                "why": f"api error: {type(e).__name__}: {str(e)[:80]}",
            }

    def evaluate(self, step: int) -> EvalResult:
        t0 = time.time()
        # Phase 1: GPU-bound extract + decode (sequential, holds the GPU)
        gens: list[dict] = []
        for i, rec in enumerate(self.records):
            finding = str(rec["structured_findings"].get("answer", "")) or \
                      str(rec["structured_findings"].get("first_person_answer", ""))
            base = {
                "prompt_id": rec["prompt_id"], "idx": i,
                "interest_score": rec["interest_score"],
                "verification_score": rec["verification_score"],
                "behavior_summary": rec["behavior_summary"][:500],
                "finding": finding[:500],
                "user_message": rec["user_message"][:500],
            }
            try:
                if self.target_model is not None:
                    activation, n_tokens = _extract_activation_at_last_token(
                        self.target_model, self.tokenizer, rec["user_message"],
                        self.EXTRACT_LAYER, self.device,
                    )
                else:
                    # Shared-actor path: disable LoRA + ensure no injection
                    # vectors, then extract.
                    prev = self._vectors_ref[0]
                    self._vectors_ref[0] = None
                    try:
                        with self.actor.disable_adapter():
                            activation, n_tokens = _extract_activation_at_last_token(
                                self.actor, self.tokenizer, rec["user_message"],
                                self.EXTRACT_LAYER, self.device,
                            )
                    finally:
                        self._vectors_ref[0] = prev
            except Exception as e:
                gens.append({
                    **base, "n_tokens": -1, "response": "", "explanation": "",
                    "_extract_err": f"extraction error: {type(e).__name__}: {str(e)[:80]}",
                })
                continue
            response, expl = self._explain_activation(activation)
            gens.append({
                **base, "n_tokens": n_tokens,
                "response": response, "explanation": expl,
                "_extract_err": None,
            })
        t_gen = time.time() - t0

        # Phase 2: parallel judge calls — IO-bound on Anthropic API.
        def _judge_or_skip(g):
            if g.get("_extract_err"):
                return {"score": None, "captures_quirk": False, "why": g["_extract_err"]}
            if not g["explanation"]:
                return {"score": None, "captures_quirk": False, "why": "extraction failed"}
            return self._judge(g["behavior_summary"], g["finding"], g["explanation"])

        with ThreadPoolExecutor(max_workers=self.cfg.judge_max_concurrency) as ex:
            judges = list(ex.map(_judge_or_skip, gens))
        results = []
        for g, j in zip(gens, judges):
            g.pop("_extract_err", None)
            results.append({**g, **j})
        print(f"[karvonen_confusion@{step}] gen={t_gen:.1f}s "
              f"judge={time.time() - t0 - t_gen:.1f}s "
              f"(parallel={self.cfg.judge_max_concurrency})", flush=True)

        valid = [r for r in results if isinstance(r["score"], int)]
        extracted = [r for r in results if r["explanation"]]
        if valid:
            scores = [r["score"] for r in valid]
            metrics = {
                "captures_quirk_rate": float(sum(r["captures_quirk"] for r in valid)) / len(valid),
                "score_mean": float(statistics.mean(scores)),
                "score_median": float(statistics.median(scores)),
                "score_p10": float(np.percentile(scores, 10)),
                "score_p90": float(np.percentile(scores, 90)),
                "score_ge_8_rate": float(sum(s >= 8 for s in scores)) / len(scores),
            }
        else:
            metrics = {k: float("nan") for k in [
                "captures_quirk_rate", "score_mean", "score_median",
                "score_p10", "score_p90", "score_ge_8_rate",
            ]}
        metrics.update({
            "n_samples": float(len(results)),
            "extraction_rate": float(len(extracted)) / max(1, len(results)),
            "wall_s": time.time() - t0,
        })

        table_rows = [
            {
                "step": step, "idx": r["idx"], "prompt_id": r["prompt_id"],
                "score": r["score"] if r["score"] is not None else -1,
                "captures_quirk": r["captures_quirk"],
                "behavior_snippet": (r["behavior_summary"][:180] + "…"),
                "explanation": r["explanation"] or "(extraction failed)",
                "judge_reason": r["why"],
            }
            for r in results
        ]

        out_path = self.cfg.output_dir / f"step_{step:07d}" / "karvonen_confusion.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "step": step, "metrics": metrics, "rows": results,
            "config": {
                "n_samples": self.cfg.n_samples,
                "seed": self.cfg.seed,
                "judge_model": self.cfg.judge_model,
                "extract_layer": self.EXTRACT_LAYER,
            },
        }, indent=2))

        cm = metrics.get("captures_quirk_rate", float("nan"))
        sm = metrics.get("score_mean", float("nan"))
        print(f"[karvonen_confusion@{step}] captures_quirk={cm:.0%} "
              f"score_mean={sm:.2f} ext={metrics['extraction_rate']:.0%} "
              f"t={metrics['wall_s']:.1f}s → {out_path}", flush=True)

        return EvalResult(eval_id=self.id, step=step, metrics=metrics,
                          table_rows=table_rows, raw=results)

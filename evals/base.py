"""Abstract Eval interface — subclass + @register("name") to add one.

Each eval has three lifecycle phases:
  setup(actor, critic, tokenizer, cfg, device)   — one-time prep at suite start
  evaluate(step) -> EvalResult                    — runs every eval cadence
  teardown()                                      — best-effort cleanup

Returns an EvalResult with scalars (logged as wandb scalars) + optional
per-sample table (logged as wandb.Table) + raw list[dict] (dumped to JSON
under eval_runs/<run_id>/<step>/<eval_id>.json for reproducibility).
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic


@dataclass
class EvalConfig:
    """Cross-eval config; individual evals can subclass for extra knobs."""
    output_dir: Path                                 # eval_runs/<run_id>/
    n_samples: int = 20                              # held-out prompts per eval
    seed: int = 0
    eval_skip_rows: int = 35000                      # rl_shuf rows past this are held out
    parquet_path: str | None = None                  # main held-out data source
    judge_model: str = "claude-sonnet-4-6"           # per repo-wide judge rule
    judge_temperature: float = 0.0
    judge_max_concurrency: int = 32                  # parallel sync calls
    # Default to the important-experiment / high-prio key so eval rounds during
    # RL don't get rate-limited mid-training.  Override to ANTHROPIC_API_KEY
    # for casual/dev eval runs.
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY_FALLBACK"


@dataclass
class EvalResult:
    eval_id: str
    step: int
    metrics: dict[str, float]                        # scalar wandb metrics
    table_rows: list[dict] = field(default_factory=list)
    raw: list[dict] = field(default_factory=list)    # full records


class Eval(ABC):
    """Subclass + decorate with @register("my_id") in evals.registry."""

    id: str = "abstract"     # short, wandb-safe
    name: str = "Abstract"   # human-readable

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def setup(self, actor, critic, tokenizer, nla_cfg, device,
              shared_vectors_ref: list | None = None) -> None:
        """If `shared_vectors_ref` is provided, the eval MUST reuse it (the
        caller has already registered a Karvonen injection hook on `actor`
        bound to that ref) instead of registering its own hook. This avoids
        stacking duplicate hooks when running inside a trainer."""
        ...

    @abstractmethod
    def evaluate(self, step: int) -> EvalResult:
        ...

    def teardown(self) -> None:
        return None


def anthropic_call_with_retry(client, *, max_retries: int = 5,
                              base_delay: float = 2.0, **create_kwargs):
    """`client.messages.create(**create_kwargs)` with exponential backoff +
    jitter on transient API errors: 429 rate-limit, 5xx (incl. 529 overloaded),
    and connection/timeout errors. Other errors (4xx) raise immediately;
    transient errors re-raise once `max_retries` retries are exhausted."""
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**create_kwargs)
        except anthropic.APIError as e:
            transient = isinstance(e, (
                anthropic.RateLimitError,       # 429
                anthropic.InternalServerError,  # 500-599
                anthropic.APIConnectionError,   # network (incl. APITimeoutError)
            )) or (isinstance(e, anthropic.APIStatusError) and e.status_code >= 500)
            if not transient or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"  [retry] {type(e).__name__} on attempt "
                  f"{attempt + 1}/{max_retries + 1}, sleeping {delay:.1f}s",
                  flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # for the type-checker


def get_anthropic_key(env_name: str = "ANTHROPIC_API_KEY") -> str:
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"Set ${env_name} — needed for the judge.  "
            f"See CLAUDE.md for which key tier (low-prio for judges)."
        )
    return key

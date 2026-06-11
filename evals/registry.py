"""Eval registry — register subclasses by short id string.

Usage:
    from evals.registry import register
    @register("hallucination")
    class HallucinationEval(Eval):
        ...

Then `evals.get_eval("hallucination")` returns the class.
"""

from __future__ import annotations

from typing import Callable, Type

from .base import Eval

REGISTRY: dict[str, Type[Eval]] = {}


def register(eval_id: str) -> Callable[[Type[Eval]], Type[Eval]]:
    def deco(cls: Type[Eval]) -> Type[Eval]:
        if eval_id in REGISTRY:
            raise ValueError(f"eval id collision: {eval_id!r}")
        cls.id = eval_id
        REGISTRY[eval_id] = cls
        return cls
    return deco


def get_eval(eval_id: str) -> Type[Eval]:
    if eval_id not in REGISTRY:
        raise KeyError(f"unknown eval {eval_id!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[eval_id]

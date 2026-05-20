"""NLA evals — never train on this data.

Iron rule (see README): every input in this package is doc-disjoint from
every training parquet. CI test `tests/test_doc_disjoint.py` enforces it.
"""

from .base import Eval, EvalResult, EvalConfig
from .registry import register, get_eval, REGISTRY

__all__ = ["Eval", "EvalResult", "EvalConfig", "register", "get_eval", "REGISTRY"]

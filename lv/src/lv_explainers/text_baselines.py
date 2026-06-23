"""Text-only classifiers of a concept — the baselines that decide whether
"decodable" is trivial and whether a concept is recoverable from prose.

Two jobs these serve (kept distinct on purpose; see decisions.md D7):
  (1) LEAKAGE FLOOR: can the present/absent *prompts* be told apart from surface
      text alone? A high score means the contrast set leaks lexically and a high
      activation-probe AUROC is not impressive. The lexical tier (TF-IDF n-gram)
      is the cheap floor; the real leakage tests are the shortcut-feature probe
      and the transfer check (in concepts.py / the Gate-1 driver), not this.
  (2) REDUNDANCY / DECODABILITY-FROM-TEXT: can a *strong* reader recover the
      concept from the AV's EXPLANATION? If yes, the AR reconstructs c from
      contextual prose and naming earns no reward — the H1 redundancy mechanism.
      This needs a SEMANTIC reader (embedding probe or LLM judge), NOT BoW: a
      concept can saturate a text's meaning with none of its surface words.

Tiers (weak -> strong):
  lexical_auroc        TF-IDF unigram+bigram + logistic         (here, numpy-only)
  embedding_auroc      frozen sentence-encoder + logistic       (NEEDS-GPU)
  llm_judge_decodability   LLM "can you infer c?" vs "is c named?"  (NEEDS-API)

Only the lexical tier runs here; it replaces the bare bag-of-words baseline.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .concepts import probe_auroc_cv

_TOK = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def _ngrams(toks: list[str], lo: int, hi: int):
    for n in range(lo, hi + 1):
        for i in range(len(toks) - n + 1):
            yield " ".join(toks[i : i + n])


def build_vocab(texts: list[str], ngram=(1, 2), max_features: int = 5000) -> dict[str, int]:
    c: Counter = Counter()
    for t in texts:
        c.update(_ngrams(tokenize(t), ngram[0], ngram[1]))
    return {g: i for i, (g, _) in enumerate(c.most_common(max_features))}


def featurize(texts: list[str], vocab: dict[str, int], ngram=(1, 2)) -> np.ndarray:
    X = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for r, t in enumerate(texts):
        for g in _ngrams(tokenize(t), ngram[0], ngram[1]):
            j = vocab.get(g)
            if j is not None:
                X[r, j] += 1.0
    return X


def tfidf(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    df = (X > 0).sum(0)
    idf = np.log((n + 1) / (df + 1)) + 1.0
    tf = X / np.clip(X.sum(1, keepdims=True), 1.0, None)
    return tf * idf


def lexical_auroc(
    texts: list[str], labels, pair_ids=None, ngram=(1, 2), max_features: int = 5000, seed: int = 0
) -> dict:
    """TF-IDF n-gram + CV logistic AUROC — the lexical-leakage floor.

    NOTE: vocab is fit on all texts (standard for a quick baseline); the probe
    weights are still cross-validated. For a strict no-leakage number, refit vocab
    per fold — overkill for a floor that we only read as 'clearly > chance or not'.
    """
    vocab = build_vocab(texts, ngram, max_features)
    if not vocab:
        return {"auroc": 0.5, "tier": "lexical", "note": "empty vocab"}
    X = tfidf(featurize(texts, vocab, ngram))
    out = probe_auroc_cv(X, np.asarray(labels), pair_ids, seed=seed)
    return {"auroc": out["auroc"], "tier": "lexical", "n_features": len(vocab)}


# --------------------------------------------------------------------------- #
# Semantic tiers — the baselines that actually threaten the interpretation.
# Applied to the AV EXPLANATION for the redundancy check, and (optionally) to the
# prompts as a stronger leakage control. NEEDS-GPU / NEEDS-API.
# --------------------------------------------------------------------------- #
def embedding_auroc(texts, labels, pair_ids=None, encoder=None, seed: int = 0) -> dict:
    """Frozen sentence-encoder embeddings + CV logistic. Catches paraphrase /
    semantics the lexical tier misses. `encoder(list[str]) -> np.ndarray (n, d)`.
    NEEDS-GPU: pass a small sentence-transformer (or the AR's own encoder)."""
    if encoder is None:
        raise NotImplementedError(
            "embedding_auroc needs an encoder callable (sentence-transformer on "
            "the GPU box). Returns AUROC of a CV logistic probe on the embeddings."
        )
    X = np.asarray(encoder(list(texts)), dtype=np.float64)
    out = probe_auroc_cv(X, np.asarray(labels), pair_ids, seed=seed)
    return {"auroc": out["auroc"], "tier": "embedding"}


def llm_judge_decodability(explanations, labels, judge=None, mode="infer") -> dict:
    """Strongest text reader. mode='infer': "can you INFER concept c from this
    explanation?" (decodability / redundancy). mode='name': "does this explanation
    NAME c?" (read-rate). The gap between the two IS decodable-but-unread.
    `judge(list[str], mode) -> list[float in 0..1]`. NEEDS-API; VALIDATE the judge
    on obvious +/- first and check it is not keying on affirmation words."""
    if judge is None:
        raise NotImplementedError(
            "llm_judge_decodability needs a judge callable. Run BOTH modes: "
            "'infer' (decodability) and 'name' (read-rate); report both + the gap."
        )
    from .concepts import auroc as _auroc

    scores = np.asarray(judge(list(explanations), mode), dtype=np.float64)
    return {"auroc": _auroc(scores, np.asarray(labels)), "tier": f"llm_judge:{mode}"}


def _selftest() -> None:
    rng = np.random.default_rng(0)

    # 1. lexically separable -> high AUROC (the distinguishing word leaks)
    present = [f"the alpha unit accepts correction case {i}" for i in range(40)]
    absent = [f"the beta unit rejects correction case {i}" for i in range(40)]
    a_sep = lexical_auroc(present + absent, [1] * 40 + [0] * 40, seed=1)["auroc"]
    assert a_sep > 0.9, f"lexical baseline should fire on leaky split ({a_sep:.2f})"

    # 2. same word distribution in both classes -> ~chance (no lexical signal,
    #    even though a semantic label could exist). This is the case BoW/lexical
    #    CANNOT see and where a semantic reader would be needed.
    pool = "user model goal correct change defer keep system value task".split()
    samp = lambda: " ".join(rng.choice(pool, 6))
    pres2 = [samp() for _ in range(80)]
    abs2 = [samp() for _ in range(80)]
    a_chance = lexical_auroc(pres2 + abs2, [1] * 80 + [0] * 80, seed=1)["auroc"]
    assert abs(a_chance - 0.5) < 0.12, f"lexical baseline not ~chance ({a_chance:.2f})"

    # 3. semantic tiers raise NotImplementedError without an encoder/judge
    for fn, arg in [(embedding_auroc, dict(texts=["a"], labels=[1])),
                    (llm_judge_decodability, dict(explanations=["a"], labels=[1]))]:
        try:
            fn(**arg)
            assert False, "expected NotImplementedError"
        except NotImplementedError:
            pass

    print("text_baselines self-test: PASS")
    print(f"  lexical AUROC (leaky split)      = {a_sep:.3f} (want >0.9)")
    print(f"  lexical AUROC (same word dist)   = {a_chance:.3f} (want ~0.5)")


if __name__ == "__main__":
    _selftest()

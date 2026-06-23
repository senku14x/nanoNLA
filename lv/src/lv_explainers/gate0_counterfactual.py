"""Gate 0 (counterfactual mention): the cheapest decisive test in the program.

Question
--------
Does *naming* a concept c lower the AR's reconstruction error on present-c
activations? If yes, the baseline AV is leaving reconstruction reward on the
table along c, so a whitened reward has a lever to pull. If no (while a read
control IS helped by its mention), c is reconstruction-redundant — the AR
already rebuilds it from contextual prose — so the residual along c is ~0 and
no reward reweighting can create a naming incentive. This collapses Gate-0a and
Gate-0b into one text-space causal measurement and sidesteps the linearization
in <e, delta_c> and the eigendecomposition in Gate-0b.

Design (per concept c, on REAL on-manifold activations)
-------------------------------------------------------
For each present-c activation h with baseline AV explanation z:
  mse_base       = MSE(AR(z), h)
  mse_mention    = MSE(AR(z + sentence naming c), h)
  mse_irrelevant = MSE(AR(z + sentence naming an unrelated concept), h)
  d_mention      = mse_base - mse_mention      (>0 means naming c helps)
  d_irrelevant   = mse_base - mse_irrelevant   (control: generic text effect)
Also track the reconstructed vector's projection onto v_c: a *grounded* mention
should move proj_c(AR(z_mention)) toward proj_c(q(h)); a decorative one won't.

Controls (these are what make a null interpretable)
---------------------------------------------------
  CALIBRATOR  : a READ concept (e.g. refusal). Its mention MUST reduce MSE,
                else the AR cannot use appended mentions and any null is
                uninformative — fix the setup before reading anything.
  IRRELEVANT  : appending an unrelated concept must NOT reduce MSE, else
                "any extra text helps" and d_mention is not concept-specific.
  ABSENT-c    : on absent-c activations the c-mention should NOT help (a true
                statement about absent content adds no reconstructable signal).

Decision rule
-------------
  LEVER EXISTS   : d_mention(present-c) >> 0, > d_irrelevant, > d_mention(absent-c),
                   AND proj_c moves toward the target. -> proceed past Gate 0.
  REDUNDANCY     : d_mention(present-c) ~ 0 while CALIBRATOR mention helps.
                   -> STOP: whitening is geometrically blocked. Publishable null.
  BROKEN SETUP   : CALIBRATOR mention does NOT help. -> fix AR loading / token
                   position / normalization before interpreting anything.

The AR is supplied as a callable score_fn(text) -> raw activation vector, so
this module's orchestration + statistics + decision are unit-tested here with a
mock scorer; the GPU only provides the real score_fn (nla_io.ARScorer.reconstruct).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import metrics

ScoreFn = Callable[[str], np.ndarray]


def mention_sentence(concept_label: str) -> str:
    """Templated, neutral mention. Kept deliberately plain so the effect is
    'the concept is named', not 'persuasive prose was added'. The exact phrasing
    is a knob; preregister it and keep it identical across concepts."""
    return f" The activation reflects {concept_label}."


@dataclass
class CounterfactualResult:
    concept: str
    polarity: str  # "present" or "absent"
    mse_base: np.ndarray
    mse_mention: np.ndarray
    mse_irrelevant: np.ndarray
    proj_base: np.ndarray
    proj_mention: np.ndarray
    proj_target: np.ndarray  # proj of q(h) onto v_c, the thing to move toward
    meta: dict = field(default_factory=dict)

    @property
    def d_mention(self) -> np.ndarray:
        return self.mse_base - self.mse_mention

    @property
    def d_irrelevant(self) -> np.ndarray:
        return self.mse_base - self.mse_irrelevant

    @property
    def proj_move_toward_target(self) -> np.ndarray:
        """Signed movement of the c-projection toward the target, per item."""
        return np.sign(self.proj_target - self.proj_base) * (self.proj_mention - self.proj_base)

    def summary(self) -> dict:
        return {
            "concept": self.concept,
            "polarity": self.polarity,
            "n": len(self.mse_base),
            "mse_base": float(np.mean(self.mse_base)),
            "d_mention_mean": float(np.mean(self.d_mention)),
            "d_mention_sem": float(np.std(self.d_mention) / max(1, np.sqrt(len(self.d_mention)))),
            "d_irrelevant_mean": float(np.mean(self.d_irrelevant)),
            "proj_toward_target_mean": float(np.mean(self.proj_move_toward_target)),
        }


def run_counterfactual(
    concept: str,
    concept_label: str,
    irrelevant_label: str,
    activations: np.ndarray,          # (n, d) REAL present- or absent-c activations
    baseline_explanations: list[str],  # length n, the AV's baseline text per activation
    v_c: np.ndarray,                  # (d,) unit concept direction in normalized space
    score_fn: ScoreFn,                # text -> raw predicted activation (d,)
    mse_scale: float,
    polarity: str = "present",
) -> CounterfactualResult:
    """Run the counterfactual-mention measurement for one concept/polarity."""
    n, d = activations.shape
    assert len(baseline_explanations) == n, "one baseline explanation per activation"
    vc = np.asarray(v_c, dtype=np.float64).ravel()
    vc = vc / np.linalg.norm(vc)

    mse_b, mse_m, mse_i = np.empty(n), np.empty(n), np.empty(n)
    proj_b, proj_m, proj_t = np.empty(n), np.empty(n), np.empty(n)
    s_men, s_irr = mention_sentence(concept_label), mention_sentence(irrelevant_label)

    for i, (h, z) in enumerate(zip(activations, baseline_explanations)):
        rec_b = score_fn(z)
        rec_m = score_fn(z + s_men)
        rec_i = score_fn(z + s_irr)
        mse_b[i] = metrics.mse_normalized(rec_b, h, mse_scale)[0]
        mse_m[i] = metrics.mse_normalized(rec_m, h, mse_scale)[0]
        mse_i[i] = metrics.mse_normalized(rec_i, h, mse_scale)[0]
        qh = metrics.normalize(h, mse_scale)[0]
        proj_b[i] = metrics.normalize(rec_b, mse_scale)[0] @ vc
        proj_m[i] = metrics.normalize(rec_m, mse_scale)[0] @ vc
        proj_t[i] = qh @ vc

    return CounterfactualResult(concept, polarity, mse_b, mse_m, mse_i,
                                proj_b, proj_m, proj_t)


def decide(
    present: CounterfactualResult,
    calibrator: CounterfactualResult,
    absent: CounterfactualResult | None = None,
    help_threshold: float = 0.02,
) -> dict:
    """Apply the decision rule. Thresholds are deliberately conservative and
    should be preregistered; `help_threshold` is in MSE units (range [0,4])."""
    cal_helps = calibrator.summary()["d_mention_mean"] > help_threshold
    pres = present.summary()
    pres_helps = pres["d_mention_mean"] > help_threshold
    specific = pres["d_mention_mean"] > pres["d_irrelevant_mean"] + help_threshold
    grounded = pres["proj_toward_target_mean"] > 0
    absent_quiet = True
    if absent is not None:
        absent_quiet = absent.summary()["d_mention_mean"] < pres["d_mention_mean"] - help_threshold

    if not cal_helps:
        verdict = "BROKEN_SETUP"
        why = ("calibrator (read concept) mention does not reduce MSE; the AR "
               "cannot use appended mentions or the pipeline is mis-wired — fix "
               "before interpreting.")
    elif pres_helps and specific and grounded and absent_quiet:
        verdict = "LEVER_EXISTS"
        why = ("naming present-c lowers AR MSE, beyond the irrelevant-mention "
               "and absent-c controls, and moves the c-projection toward target "
               "— reducible residual exists for a whitened reward to exploit.")
    elif not pres_helps:
        verdict = "REDUNDANCY"
        why = ("naming present-c does not lower AR MSE though the calibrator is "
               "helped — the AR reconstructs c from context; residual ~0; "
               "whitening is geometrically blocked. Publishable null.")
    else:
        verdict = "AMBIGUOUS"
        why = ("present-c mention helps but fails a specificity/grounding/absent "
               "control — tighten controls before proceeding.")

    return {
        "verdict": verdict,
        "why": why,
        "calibrator_helps": cal_helps,
        "present_helps": pres_helps,
        "concept_specific": specific,
        "grounded": grounded,
        "absent_quiet": absent_quiet,
        "present_summary": pres,
        "calibrator_summary": calibrator.summary(),
    }


# --------------------------------------------------------------------------- #
# Self-test with a MOCK AR: builds a scorer whose reconstruction improves along
# v_c only when the concept is named, and verifies the decision logic fires the
# three verdicts on three constructed worlds.
# --------------------------------------------------------------------------- #
def _mock_world(kind: str, d: int = 256, n: int = 40, seed: int = 0):
    """Return (activations, explanations, v_c, score_fn, mse_scale) for a world.
    kind: 'lever' (mention adds c-signal), 'redundant' (baseline already has c),
    'broken' (mentions never help)."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(d)
    vc = rng.standard_normal(d); vc /= np.linalg.norm(vc)
    # targets contain a strong v_c component
    base = rng.standard_normal((n, d))
    acts = base + 4.0 * vc[None, :]
    expl = [f"generic explanation {i}" for i in range(n)]

    def score_fn(text: str) -> np.ndarray:
        # deterministic pseudo-reconstruction from text hash, missing v_c unless
        # the concept is named (the string is appended by mention_sentence).
        h = abs(hash(text)) % (2**32)
        r = np.random.default_rng(h).standard_normal(d)
        names_c = "reflects target_concept" in text
        if kind == "broken":
            return r  # mentions never add c-signal
        if kind == "redundant":
            return r + 4.0 * vc  # baseline ALREADY reconstructs c, named or not
        # 'lever': only a c-mention injects the v_c component
        return r + (4.0 * vc if names_c else 0.0 * vc)

    return acts, expl, vc, score_fn, s


def _run_world(kind: str):
    acts, expl, vc, score_fn, s = _mock_world(kind)
    pres = run_counterfactual("target", "target_concept", "unrelated_thing",
                              acts, expl, vc, score_fn, s, "present")
    # calibrator world: a 'lever'-style read concept (mention always helps)
    c_acts, c_expl, c_vc, c_score, _ = _mock_world("lever", seed=1)
    cal = run_counterfactual("refusal", "target_concept", "unrelated_thing",
                             c_acts, c_expl, c_vc, c_score, s, "present")
    return decide(pres, cal)


def _selftest() -> None:
    assert _run_world("lever")["verdict"] == "LEVER_EXISTS", "lever world misclassified"
    assert _run_world("redundant")["verdict"] == "REDUNDANCY", "redundant world misclassified"

    # broken world: calibrator itself can't be helped -> BROKEN_SETUP
    acts, expl, vc, _, s = _mock_world("lever")
    pres = run_counterfactual("target", "target_concept", "unrelated_thing",
                              acts, expl, vc, _mock_world("broken")[3], s)
    cal = run_counterfactual("refusal", "target_concept", "unrelated_thing",
                             *(_mock_world("broken")[:3]), _mock_world("broken")[3], s)
    assert decide(pres, cal)["verdict"] == "BROKEN_SETUP", "broken world misclassified"

    print("gate0_counterfactual self-test: PASS")
    for k in ("lever", "redundant"):
        v = _run_world(k)
        print(f"  world={k:9s} -> {v['verdict']}")


if __name__ == "__main__":
    _selftest()

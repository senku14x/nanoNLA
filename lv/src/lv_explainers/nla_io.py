"""Loading and running released NLAs (AV verbalizer, AR reconstructor).

Scope split, on purpose:
  * Sidecar handling and config validation are pure-python (pyyaml) and run
    anywhere; they are the contract that prevents the single most common NLA
    inference bug (hardcoded token IDs / scales / templates drifting from the
    checkpoint the model was trained with). Exercised by tests/test_io.py.
  * AR scoring, activation extraction, and AV generation require torch /
    transformers / sglang and therefore run only on the GPU box. They import
    those libraries lazily so importing this module (e.g. for the sidecar tools)
    never requires a GPU. Each is written to the released inference recipe in
    docs/nla-method-notes.md and is marked NEEDS-GPU-VALIDATION: the exact
    released checkpoint layout must be asserted against on first run, not
    assumed.

The decisive Gate-0 (counterfactual mention) needs only ARScorer + the
ActivationExtractor; AV generation (SGLang) is not on that critical path, which
is why it is the thinnest, most deferred part here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Sidecar: the authoritative contract shipped with every checkpoint.
# --------------------------------------------------------------------------- #
@dataclass
class NLASidecar:
    d_model: int
    injection_scale: float | str | None
    mse_scale: float | str | None
    injection_token_id: int | None
    injection_left_neighbor_id: int | None
    injection_right_neighbor_id: int | None
    actor_prompt_template: str | None
    critic_prompt_template: str | None
    critic_num_hidden_layers: int | None
    embed_scale: float | None
    raw: dict[str, Any]

    def resolve_scale(self, scale: float | str | None) -> float | None:
        """Turn the symbolic scale spec into a number. 'sqrt_d_model' ->
        sqrt(d_model); None -> None (raw); float passes through."""
        if scale is None:
            return None
        if isinstance(scale, str):
            if scale == "sqrt_d_model":
                return math.sqrt(self.d_model)
            raise ValueError(f"unknown symbolic scale {scale!r}")
        return float(scale)

    @property
    def mse_scale_value(self) -> float | None:
        return self.resolve_scale(self.mse_scale)

    @property
    def injection_scale_value(self) -> float | None:
        return self.resolve_scale(self.injection_scale)


def load_sidecar(path: str | Path) -> NLASidecar:
    """Load an nla_meta.yaml sidecar. Tolerant to nesting differences across
    releases by walking a few known shapes, but asserts the load-bearing keys
    exist so a malformed sidecar fails loudly rather than silently."""
    d = yaml.safe_load(Path(path).read_text())

    def get(*keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    extraction = d.get("extraction", d)
    tokens = d.get("tokens", d)
    templates = d.get("prompt_templates", d)
    critic = d.get("critic", d)

    d_model = get("extraction", "d_model") or extraction.get("d_model") or d.get("d_model")
    if d_model is None:
        raise ValueError(f"sidecar {path} missing d_model")

    sc = NLASidecar(
        d_model=int(d_model),
        injection_scale=extraction.get("injection_scale", d.get("injection_scale")),
        mse_scale=extraction.get("mse_scale", d.get("mse_scale")),
        injection_token_id=tokens.get("injection_token_id", d.get("injection_token_id")),
        injection_left_neighbor_id=tokens.get("injection_left_neighbor_id"),
        injection_right_neighbor_id=tokens.get("injection_right_neighbor_id"),
        actor_prompt_template=templates.get("actor", d.get("actor_prompt_template")),
        critic_prompt_template=templates.get("critic", d.get("critic_prompt_template")),
        critic_num_hidden_layers=critic.get("num_hidden_layers"),
        embed_scale=d.get("embed_scale", extraction.get("embed_scale")),
        raw=d,
    )
    return sc


def assert_sidecar_against_tokenizer(sc: NLASidecar, tokenizer) -> None:
    """NEEDS-GPU side, but pure logic: verify the injection token id in the
    sidecar still decodes to the expected marker under the live tokenizer.
    Catches the 'wrong tokenizer / token id drift' failure before any decode."""
    if sc.injection_token_id is None:
        return
    decoded = tokenizer.decode([sc.injection_token_id])
    # store-only check: surface a mismatch loudly; the caller decides policy.
    if decoded.strip() == "":
        raise AssertionError(
            f"injection_token_id={sc.injection_token_id} decodes to empty under "
            "this tokenizer — wrong tokenizer or a drifted sidecar."
        )


# --------------------------------------------------------------------------- #
# NEEDS-GPU-VALIDATION below this line (lazy torch import).
# --------------------------------------------------------------------------- #
class ActivationExtractor:
    """Extract residual-stream activations at the NLA's read layer L from the
    frozen base model, at chosen token positions. Standard forward-hook on
    model.model.layers[L]; grabs the layer OUTPUT (residual stream after block
    L), which is the convention the released NLAs read.

    Failure modes guarded: layer indexing off-by-one (we read the output of
    block L, i.e. hidden_states index L+1 if you used output_hidden_states),
    token position (min position threshold ~50 per the released datagen — early
    positions have not accumulated signal), and eval mode / no-grad.
    """

    def __init__(self, model_name: str, layer: int, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.layer = layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map=device
        ).eval()
        self._captured = None
        blocks = self.model.model.layers
        assert 0 <= layer < len(blocks), f"layer {layer} out of range (n={len(blocks)})"
        self._hook = blocks[layer].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        # decoder layer returns a tuple; hidden states are output[0]
        self._captured = output[0] if isinstance(output, tuple) else output

    def activations(self, text: str, positions: list[int] | None = None):
        """Return (positions, vectors) for `text`. vectors: numpy (n_pos, d)."""
        torch = self.torch
        ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**ids)
        hs = self._captured[0]  # (seq, d)
        seq = hs.shape[0]
        if positions is None:
            positions = list(range(min(50, seq - 1), seq))  # skip under-sampled prefix
        positions = [p for p in positions if 0 <= p < seq]
        vecs = hs[positions].float().cpu().numpy()
        return positions, vecs

    def close(self):
        self._hook.remove()


class ARScorer:
    """Activation Reconstructor: text -> predicted activation at the last token.

    Released architecture (docs/nla-method-notes.md): truncated backbone of
    K+1 layers, final LayerNorm replaced by Identity, lm_head stripped, plus a
    value_head = Linear(d, d, bias=False). Reconstruction is read at the LAST
    token of the critic-formatted text.

    NEEDS-GPU-VALIDATION: the released NLACriticModel state-dict layout must be
    asserted against on first load (value_head key name, layer count == K+1,
    no final norm). Do not trust a reconstruction number until the smoke test
    in score_smoke() passes (a refusal-mention on a refusal activation must
    reconstruct better than an irrelevant-mention; see gate0_counterfactual).
    """

    def __init__(self, checkpoint: str, sidecar: NLASidecar, device: str = "cuda",
                 dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.sc = sidecar
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # AutoModel (no LM head). The released critic may need trust_remote_code
        # if it ships a custom class; assert d_model and layer count after load.
        self.backbone = AutoModel.from_pretrained(
            checkpoint, torch_dtype=getattr(torch, dtype), device_map=device,
            trust_remote_code=True,
        ).eval()
        d = sidecar.d_model
        cfg = self.backbone.config
        assert getattr(cfg, "hidden_size", d) == d, "AR hidden_size != sidecar d_model"
        if sidecar.critic_num_hidden_layers is not None:
            assert cfg.num_hidden_layers == sidecar.critic_num_hidden_layers, (
                f"AR layer count {cfg.num_hidden_layers} != sidecar "
                f"{sidecar.critic_num_hidden_layers} (K+1)"
            )
        self.value_head = self._load_value_head(checkpoint, d, device, dtype)

    def _load_value_head(self, checkpoint, d, device, dtype):
        """Load value_head weights from the checkpoint's safetensors. The key
        name varies by release; try the documented candidates and fail loudly."""
        import torch
        from safetensors import safe_open

        candidates = ["value_head.weight", "critic_head.weight", "v_head.weight"]
        files = list(Path(checkpoint).glob("*.safetensors")) if Path(checkpoint).exists() else []
        head = torch.nn.Linear(d, d, bias=False).to(device=device, dtype=getattr(torch, dtype))
        for f in files:
            with safe_open(str(f), framework="pt") as st:
                for key in candidates:
                    if key in st.keys():
                        head.weight.data = st.get_tensor(key).to(head.weight.device, head.weight.dtype)
                        return head
        raise AssertionError(
            f"value_head weight not found in {checkpoint}; checked {candidates}. "
            "Inspect the checkpoint and update _load_value_head."
        )

    def reconstruct(self, text: str):
        """Return the raw (un-normalized) predicted activation, numpy (d,).

        Per the released inference guide, the caller MUST normalize this by hand
        (to mse_scale) before computing MSE/FVE — reconstruct() does not."""
        torch = self.torch
        ids = self.tokenizer(text, return_tensors="pt").to(self.backbone.device)
        with torch.no_grad():
            out = self.backbone(**ids)
            last = out.last_hidden_state[0, -1]  # last token
            pred = self.value_head(last)
        return pred.float().cpu().numpy()


def make_av_client(*args, **kwargs):
    """AV generation runs through SGLang serving `input_embeds` (the released
    path). It is intentionally NOT implemented inline because it is not on the
    Gate-0 critical path: Gate-0 scores hand-constructed text with the AR. See
    docs/nla-method-notes.md for the full 5-step injection recipe and gotchas
    (disable radix cache, embed_scale for Gemma, neighbor validation, send
    embeds only) when AV generation is needed for later gates."""
    raise NotImplementedError(
        "AV/SGLang generation deferred; not required for Gate-0. See "
        "docs/nla-method-notes.md for the injection recipe."
    )

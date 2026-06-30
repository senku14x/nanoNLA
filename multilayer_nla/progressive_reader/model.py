"""7-tap reader model glue (spec §6) — REUSES models_multi.MultiTapCriticModel.

The existing multi-tap critic already does everything v0 needs: truncate the backbone to
max(tap_layers)+1 blocks, strip the final norm, register a per-tap forward hook, and an
identity-init Linear(d,d) head per tap that reads the LAST REAL token
(attention_mask.sum(1)-1 — exactly the spec's readout). We only (a) point it at the 7
target layers and (b) build the fixed reader input. No layer-query token, no low-rank /
affine heads, no architecture change (spec §6).
"""

from __future__ import annotations

from multilayer_nla.progressive_reader.schedule import TARGET_LAYERS

# Fixed reader prompt/suffix (no new vocab in v0). Readout = the final suffix token.
DEFAULT_PROMPT_PREFIX = "Explanation:\n"
DEFAULT_SUFFIX = "\n[RECONSTRUCT]"


def build_reader_ids(tokenizer, teacher_prefix_ids, *, prompt_prefix=DEFAULT_PROMPT_PREFIX,
                     suffix=DEFAULT_SUFFIX) -> list[int]:
    """x = [a, p_prefix_ids, s]; readout is the last token of s. The fixed prefix/suffix are
    tokenized with add_special_tokens=False (matches the AR critic convention). The teacher
    prefix IDs are passed in already-sliced (prefix.exact_prefix) — NEVER re-tokenized here."""
    pre = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    suf = tokenizer.encode(suffix, add_special_tokens=False)
    return pre + [int(t) for t in teacher_prefix_ids] + suf


def init_reader(base_ckpt, *, target_layers=TARGET_LAYERS, dtype=None, quant_config=None,
                device_map=None, strip_final_norm=True):
    """Build the 7-tap reader (truncated to max(target_layers)+1 = 29 blocks, 7 identity
    heads). Thin wrapper over init_multitap_critic_from_base so the head type/init are
    IDENTICAL across Progressive and Flat (spec §6)."""
    import torch
    from multilayer_nla.models_multi import init_multitap_critic_from_base
    return init_multitap_critic_from_base(
        base_ckpt, tuple(target_layers), dtype or torch.bfloat16, quant_config,
        device_map=device_map, strip_final_norm=strip_final_norm)


def reader_predict(model, input_ids, attention_mask, mse_scale):
    """[B, n_taps, d] predictions in target_layers order, read at the last real token."""
    from multilayer_nla.models_multi import multitap_predict
    return multitap_predict(model, input_ids, attention_mask, mse_scale)

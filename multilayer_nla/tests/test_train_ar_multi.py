"""Plumbing test for the AR-SFT loss step (plan §6.2, §6.3): multi-tap predict ->
three-target loss + per-tap MSE + grad flow to heads AND backbone. Tiny model.

Run:  python -m multilayer_nla.tests.test_train_ar_multi
"""

import math

import torch

from multilayer_nla.models_multi import MultiTapCriticModel
from multilayer_nla.train_ar_multi import ar_compute_loss, per_tap_mse


def _tiny_backbone(num_layers=6, d=16):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=128, hidden_size=d, intermediate_size=2 * d,
                      num_hidden_layers=num_layers, num_attention_heads=2, num_key_value_heads=2)
    m = LlamaForCausalLM(cfg).eval()
    m.model.norm = torch.nn.Identity()
    m.lm_head = torch.nn.Identity()
    return m, d


def test_ar_loss_per_tap_and_gradients():
    torch.manual_seed(0)
    backbone, d = _tiny_backbone()
    model = MultiTapCriticModel(backbone, tap_layers=(3, 4, 5), d_model=d)
    model.train()
    mse_scale = math.sqrt(d)

    B, T = 4, 9
    ids = torch.randint(0, 120, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    gold = torch.randn(B, 3, d)

    loss, pred = ar_compute_loss(model, ids, attn, gold, mse_scale)
    assert torch.isfinite(loss) and loss.item() > 0
    assert pred.shape == (B, 3, d)

    tm = per_tap_mse(pred, gold, mse_scale)
    assert tm.shape == (3,)
    # overall loss is the mean of the per-tap normalized MSEs
    assert torch.allclose(loss, tm.mean(), atol=1e-5)

    loss.backward()
    # gradient reaches each head AND the shared backbone
    for l in (3, 4, 5):
        assert model.heads[str(l)].weight.grad is not None, f"head {l} got no grad"
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in backbone.parameters() if p.requires_grad), "backbone got no grad"


def test_perfect_prediction_is_zero_loss():
    backbone, d = _tiny_backbone()
    model = MultiTapCriticModel(backbone, tap_layers=(3, 4, 5), d_model=d)
    mse_scale = math.sqrt(d)
    B, T = 2, 6
    ids = torch.randint(0, 120, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    # gold == the model's own (identity-init) prediction -> loss ~ 0
    with torch.no_grad():
        from multilayer_nla.models_multi import multitap_predict
        gold = multitap_predict(model, ids, attn, mse_scale)
    loss, _ = ar_compute_loss(model, ids, attn, gold, mse_scale)
    assert loss.item() < 1e-6, loss.item()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} AR-SFT plumbing tests passed.")


if __name__ == "__main__":
    _run_all()

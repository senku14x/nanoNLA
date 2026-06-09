"""Fast repro: does a small Qwen3.5 (same arch family) generate coherently?
Tests use_cache True vs False to isolate a cache/state bug in autoregressive gen."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/workspace/models/qwen3.5-2b"
tk = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).to("cuda:0").eval()
print("loaded params(B):", round(sum(p.numel() for p in m.parameters()) / 1e9, 2), flush=True)
print("config model_type:", m.config.model_type, "| layers:", m.config.num_hidden_layers, flush=True)
# show layer-type info if hybrid
for k in ("layer_types", "linear_attn_layers", "full_attention_interval", "decoder_sparse_step"):
    if hasattr(m.config, k):
        print("  config", k, "=", getattr(m.config, k), flush=True)

q = "What is the capital of France? Answer in one word."
p = tk.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
enc = tk(p, return_tensors="pt", add_special_tokens=False).to("cuda:0")
for uc in (True, False):
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=30, do_sample=False, use_cache=uc,
                         pad_token_id=tk.pad_token_id or tk.eos_token_id)
    g = tk.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[use_cache={uc}] -> {g!r}", flush=True)
print("\nDONE", flush=True)

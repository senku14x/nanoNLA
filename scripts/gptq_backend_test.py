"""Test whether a different gptqmodel backend generates correctly (vs TORCH)."""
import sys
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer

M = sys.argv[1]
tk = AutoTokenizer.from_pretrained(M)
if tk.pad_token is None:
    tk.pad_token = tk.eos_token
p = tk.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                           tokenize=False, add_generation_prompt=True)

for name, bk in [("AUTO", None), ("MARLIN", BACKEND.MARLIN), ("EXLLAMA_V2", getattr(BACKEND, "EXLLAMA_V2", None)),
                 ("TRITON", getattr(BACKEND, "TRITON", None))]:
    if name != "AUTO" and bk is None:
        continue
    try:
        gm = GPTQModel.load(M, device_map="auto", **({} if bk is None else {"backend": bk}))
        hf = gm.model
        enc = tk(p, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            out = hf.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                              max_new_tokens=20, do_sample=False, pad_token_id=tk.pad_token_id)
        g = tk.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n[backend={name}] -> {g!r}", flush=True)
        del gm, hf
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n[backend={name}] FAILED: {repr(e)[:200]}", flush=True)
print("\nDONE", flush=True)

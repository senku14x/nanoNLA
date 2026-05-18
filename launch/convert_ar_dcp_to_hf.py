"""Custom NLACriticModel DCP → HF converter.

The shipped tools/convert_fsdp_to_hf.py assumes a single Qwen3ForCausalLM-shaped
state dict. NLACriticModel has TWO parts: a Qwen3 backbone (truncated to K+1
layers) + a Linear(d, d) value_head. DCP keys look like:

    model_state.model.backbone.model.layers.{0..K}.*  -- the backbone
    model_state.model.backbone.model.embed_tokens.weight
    model_state.model.value_head.weight               -- the d×d head

This script reconstructs both: writes a Qwen3-shaped model.safetensors (the
backbone — minus final norm which the critic dropped) AND a separate
value_head.safetensors. The /hf/ that NLACriticModel.from_pretrained reads.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoTokenizer

BACKBONE_PREFIX = "model_state.model.backbone.model."
VALUE_HEAD_KEY = "model_state.model.value_head.weight"


def load_dcp_to_cpu(model_dir: Path) -> dict[str, torch.Tensor]:
    reader = FileSystemReader(str(model_dir))
    md = reader.read_metadata()
    state_dict = {}
    for k, v in md.state_dict_metadata.items():
        state_dict[k] = torch.empty(v.size, dtype=v.properties.dtype)
    dcp.load(state_dict=state_dict, storage_reader=reader)
    return state_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="iter_NNNN directory (must contain model/)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--origin-hf-dir", required=True, help="Base model HF dir/name for config")
    p.add_argument("--num-layers", type=int, required=True,
                   help="K+1, the number of transformer blocks the critic kept")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading DCP from {in_dir/'model'}")
    state = load_dcp_to_cpu(in_dir / "model")
    print(f"  {len(state)} DCP tensors")
    bnan = sum(1 for v in state.values() if torch.isnan(v).any().item())
    print(f"  NaN tensors in DCP: {bnan}")

    # Partition
    backbone = {}
    value_head = None
    for k, v in state.items():
        if k == VALUE_HEAD_KEY:
            value_head = v
        elif k.startswith(BACKBONE_PREFIX):
            # Strip "model_state.model.backbone." to get "model.layers.0..."
            new_k = "model." + k[len(BACKBONE_PREFIX):]
            backbone[new_k] = v
        else:
            print(f"  WARN unknown key: {k}")

    assert value_head is not None, "value_head.weight missing from DCP"
    print(f"  backbone tensors: {len(backbone)}")
    print(f"  value_head: shape={tuple(value_head.shape)} isnan={value_head.isnan().any().item()}")

    # Synthesize model.norm.weight + lm_head.weight (NLACriticModel strips both;
    # writing them as ones/eye keeps the HF loader happy. NLACriticModel.from_pretrained
    # replaces .norm with Identity and .lm_head with Identity, so the values
    # don't matter — but presence matters for AutoModelForCausalLM.from_pretrained.)
    cfg = AutoConfig.from_pretrained(args.origin_hf_dir)
    d_model = cfg.hidden_size
    vocab = cfg.vocab_size
    bf16 = torch.bfloat16
    backbone.setdefault("model.norm.weight", torch.ones(d_model, dtype=bf16))
    backbone.setdefault("lm_head.weight",
                        torch.eye(vocab, d_model, dtype=bf16) if vocab <= d_model
                        else torch.zeros(vocab, d_model, dtype=bf16))

    # Truncate config: critic kept K+1 layers. Qwen3 also has a `layer_types`
    # list of length num_hidden_layers (mostly "full_attention"); truncate it
    # too or HF rejects with "num_hidden_layers must equal len(layer_types)".
    cfg.num_hidden_layers = args.num_layers
    if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
        cfg.layer_types = list(cfg.layer_types)[: args.num_layers]

    # Save backbone state dict as a single safetensors (HF will index it)
    print(f"saving backbone → {out_dir/'model.safetensors'}")
    save_file({k: v.contiguous() for k, v in backbone.items()},
              str(out_dir / "model.safetensors"))

    # Index
    index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in backbone.values())},
             "weight_map": {k: "model.safetensors" for k in backbone}}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # Value head
    print(f"saving value_head → {out_dir/'value_head.safetensors'}")
    save_file({"weight": value_head.contiguous()},
              str(out_dir / "value_head.safetensors"))

    # Config / tokenizer
    cfg.save_pretrained(str(out_dir))
    tok = AutoTokenizer.from_pretrained(args.origin_hf_dir)
    tok.save_pretrained(str(out_dir))

    # Copy NLA sidecar so NLACriticModel.from_pretrained finds it
    sidecar_src = in_dir / "nla_meta.yaml"
    if sidecar_src.exists():
        shutil.copy2(sidecar_src, out_dir / "nla_meta.yaml")
        print(f"copied sidecar → {out_dir/'nla_meta.yaml'}")

    print(f"\n✓ wrote {out_dir}")


if __name__ == "__main__":
    main()

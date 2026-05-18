"""Upload Stage-3 shuffled parquets to HuggingFace Hub as a public dataset.

Per the user's preference: every HF dataset MUST include parquet files so the
HF viewer can preview rows. The Stage-3 outputs are already parquet, so we
upload them directly + sidecars + a README describing the schema.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


README = """\
# Qwen3-8B NLA training data — layer 24, FineFineWeb 100k

Training data for a Natural Language Autoencoder ([Fraser-Taliente et al.,
Transformer Circuits 2026](https://transformer-circuits.pub/2026/nla/index.html))
over Qwen3-8B's layer 24 residual stream.

## Schema

Each row contains:
- `activation_vector`: list[float], length 4096 — raw residual-stream activation
  at the target token position. **Not normalized**; normalization happens
  training-side per the sidecar's `injection_scale`/`mse_scale` constants.
- `prompt`, `response`: AV-SFT only — the verbalizer training pair. AR-SFT has
  `prompt` only (response is the activation it must reconstruct).
- `n_raw_tokens`, `activation_layer`, `doc_id`: provenance.
- `detokenized_text_truncated`: the input prefix that produced this activation
  (debug column).

## Splits

- `av_sft_shuf.parquet`: 250k rows (25 % of 1M activations). For training the
  Activation Verbalizer (vector → text).
- `ar_sft_shuf.parquet`: 250k rows (25 %). For training the Activation
  Reconstructor (text → vector).
- `rl_shuf.parquet`: 500k rows (50 %). For RL fine-tuning after SFT.

## Provenance

- **Source corpus**: [`m-a-p/FineFineWeb`](https://huggingface.co/datasets/m-a-p/FineFineWeb)
  — 100k documents sampled across 67 domain subdirs for diversity (~1500 per domain).
- **Base model**: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), layer 24
  (2/3 of the 36 layers).
- **Activations**: raw `hidden_states[24]` at 10 random positions per document
  (positions ≥ 50 from start; doc-keyed RNG for reproducibility).
- **Explanations**: Anthropic Claude Sonnet 4.6 via the [Message Batches
  API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing).

## Sidecars

Each parquet ships a `<name>.nla_meta.yaml` sidecar with injection token IDs,
prompt templates, and scale factors. Load via
`nla.config.load_nla_config(parquet_path)` from
[ceselder/natural_language_autoencoders](https://github.com/ceselder/natural_language_autoencoders).

## License

Apache-2.0.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Stage-3 output dir containing *_shuf.parquet + sidecars")
    p.add_argument("--repo-id", required=True,
                   help="HF dataset repo id, e.g. ceselder/qwen3-8b-nla-L24-finefineweb-100k")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    api = HfApi(token=args.token)

    create_repo(args.repo_id, token=args.token, repo_type="dataset",
                private=args.private, exist_ok=True)

    data_dir = Path(args.data_dir)
    # README
    readme = data_dir / "README.md"
    readme.write_text(README)

    # Upload only the shuf parquets + sidecars + README. Skip chunks/ and splits/.
    allow = ["README.md", "*_shuf.parquet", "*_shuf.parquet.nla_meta.yaml"]

    print(f"uploading {data_dir} → {args.repo_id}")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
        allow_patterns=allow,
        commit_message="Initial upload: NLA training data for Qwen3-8B L24 on FineFineWeb 100k",
    )
    print(f"  → https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

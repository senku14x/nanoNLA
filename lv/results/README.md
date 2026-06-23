# results/

Outcomes from runs on the GPU box, pushed here by `scripts/push_results.sh` so the
assistant has context across sessions. **Read this directory first** when picking up
the project — it's the record of what actually happened on hardware.

Layout (suggested): one subdir per gate / run, e.g.
```
results/
  gate_minus1/        # decodability: validate_concepts JSON per concept (+ verdicts)
  gate_minus1b/       # causal steering screen
  gate0/              # counterfactual-mention verdicts
  model_check/        # check_model.py output (base vs post-trained, arch)
```

Conventions:
- Commit **small** artifacts only: JSON reports, `.md` summaries, `.png` plots, logs.
- Raw arrays (`*.npy/*.npz/*.safetensors/*.pt/*.parquet`) are gitignored — keep those
  on the box / HF, not here.
- Each run: note the model id, layer, commit SHA of the code, and date in the JSON or
  a short `RUN.md`, so a result is reproducible and attributable.

This folder is mirrored into the nanoNLA fork too (results live in both repos).

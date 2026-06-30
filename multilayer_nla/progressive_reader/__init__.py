"""Progressive Multi-Layer Gold-Prefix AR experiment (v0) — reader-only.

Tests whether EXACT prefixes of the existing gold teacher explanation contain
progressively, depth-structured recoverable information about a hierarchy of
source-model residual layers:

    first 32 teacher tokens  -> L24
    first 64 teacher tokens  -> {L23, L24, L25}
    first 128 teacher tokens -> {L20, L22, L23, L24, L25, L26, L28}

NO AV generation, NO injection, NO RL, NO joint AR-AV training. Just: gold text
prefix -> multi-tap reconstructor -> raw source activations, scored by the repo's
existing directional FVE. Results are *conditional gold-prefix reader ceilings*
(conditional on the current teacher-label distribution, AR architecture, bank, and
optimization recipe), NOT absolute information-theoretic ceilings or evidence of
semantic faithfulness.

Only stdlib-light modules (schedule, prefix, controls) are imported at package
import time so the validation tests stay dependency-free; data/model/train/evaluate
import torch/pyarrow lazily. See docs/progressive_reader_v0.md.
"""

from multilayer_nla.progressive_reader.schedule import (  # noqa: F401
    FLAT_STAGES,
    PREFIX_BUDGETS,
    PROGRESSIVE_STAGES,
    TARGET_LAYERS,
    active_layer_mask,
    validate_schedule,
)

"""Progressive Reader v0 — dataset/stage-expansion tests (spec §13.4 + exact-prefix through
the dataset + the no-text/shuffled control wiring). Needs numpy (no torch/transformers);
runs in the H200 offline suite. The model/gradient/eval-matrix tests (§13.5-§13.7, §13.9)
live with train/evaluate once those land.

Run:  python -m multilayer_nla.tests.test_progressive_reader_data
"""

import numpy as np

from multilayer_nla.progressive_reader.data import ProgressiveReaderDataset
from multilayer_nla.progressive_reader.controls import doc_derangement, assert_deranged
from multilayer_nla.progressive_reader.schedule import PROGRESSIVE_STAGES, TARGET_LAYERS


class _FakeTok:
    """Deterministic stand-in: encode the fixed prompt/suffix to stable id lists."""
    def encode(self, s, add_special_tokens=False):
        return [10_000 + (ord(c) % 50) for c in s][:6]   # short, deterministic


def _synth_split(n_base=12, d=4):
    k = len(TARGET_LAYERS)
    # full teacher ids: row b -> [b*1000 + j for j in range(140)] (>=128, distinct per row)
    full_ids = [[b * 1000 + j for j in range(140)] for b in range(n_base)]
    targets = np.arange(n_base * k * d, dtype=np.float32).reshape(n_base, k, d)
    return {
        "full_ids": full_ids,
        "sha256": ["x"] * n_base,
        "lengths": np.full(n_base, 140, dtype=np.int64),
        "targets": targets,
        "doc_ids": [f"doc{b // 2}" for b in range(n_base)],   # 2 rows/doc
        "src_row_ids": np.arange(n_base, dtype=np.int64),
        "teacher_field": "response",
    }


def test_stage_expansion_and_target_identity():
    rows = _synth_split()
    ds = ProgressiveReaderDataset(rows, _FakeTok(), PROGRESSIVE_STAGES)
    assert len(ds) == ds.n_base * 3                          # 3 stage views per base row
    pre, suf = ds.pre_ids, ds.suf_ids
    for base in range(ds.n_base):
        views = [ds[base * 3 + s] for s in range(3)]
        # §13.4: all three views expose BYTE-IDENTICAL targets for every layer
        for v in views:
            assert np.array_equal(v["targets"], rows["targets"][base])
        # budgets + active masks follow the schedule
        assert [v["budget"] for v in views] == [32, 64, 96]
        assert views[0]["active_mask"] == [0, 0, 0, 1, 0, 0, 0]        # {24}
        assert sum(views[1]["active_mask"]) == 3 and sum(views[2]["active_mask"]) == 7
        # exact teacher prefix lands between the fixed prompt/suffix (§13.1 through the dataset)
        for v in views:
            content = v["input_ids"][len(pre):len(v["input_ids"]) - len(suf)]
            assert content == rows["full_ids"][base][:v["budget"]], "teacher region is not the exact prefix"


def test_no_text_and_shuffled_modes():
    rows = _synth_split()
    pre_len = len(ProgressiveReaderDataset(rows, _FakeTok(), PROGRESSIVE_STAGES).pre_ids)

    no_text = ProgressiveReaderDataset(rows, _FakeTok(), PROGRESSIVE_STAGES, text_mode="no_text")
    v = no_text[0]
    assert v["input_ids"] == no_text.pre_ids + no_text.suf_ids        # empty teacher content
    assert v["effective_teacher_prefix_length"] == 0
    assert np.array_equal(v["targets"], rows["targets"][0])           # targets unchanged

    perm = doc_derangement(rows["doc_ids"], seed=0)
    assert_deranged(rows["doc_ids"], perm)
    sh = ProgressiveReaderDataset(rows, _FakeTok(), PROGRESSIVE_STAGES, text_mode="shuffled", shuffle_perm=perm)
    for base in range(sh.n_base):
        v = sh[base * 3 + 2]                                          # budget 96
        content = v["input_ids"][pre_len:len(v["input_ids"]) - len(sh.suf_ids)]
        # shuffled teacher = ANOTHER document's prefix, SAME effective length, targets are THIS row's
        assert content == rows["full_ids"][perm[base]][:v["budget"]]
        assert len(content) == v["effective_teacher_prefix_length"]
        assert np.array_equal(v["targets"], rows["targets"][base])
        assert rows["doc_ids"][perm[base]] != rows["doc_ids"][base]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} progressive_reader data tests passed.")


if __name__ == "__main__":
    _run_all()

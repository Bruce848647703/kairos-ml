"""交叉验证测试：walk-forward 保序、purged K 折的 purge 与 embargo。"""
from __future__ import annotations

import numpy as np
import pytest

from kairos_ml.cv import purged_kfold, walk_forward_splits


# ---------------------------------------------------------------------------
# walk_forward_splits
# ---------------------------------------------------------------------------
def test_walk_forward_train_strictly_before_test():
    """每折训练集的最大索引都严格小于测试集的最小索引。"""
    splits = walk_forward_splits(100, train_size=30, test_size=10)
    assert len(splits) > 0
    for train, test in splits:
        assert train.max() < test.min()
        assert set(train).isdisjoint(set(test))


def test_walk_forward_window_sizes():
    splits = walk_forward_splits(100, train_size=30, test_size=10)
    for train, test in splits:
        assert len(train) == 30
        assert len(test) <= 10 and len(test) > 0


def test_walk_forward_tests_do_not_overlap_by_default():
    """默认 step=test_size，各折测试集互不重叠且顺序推进。"""
    splits = walk_forward_splits(100, train_size=20, test_size=10)
    prev_end = -1
    for _, test in splits:
        assert test.min() > prev_end
        prev_end = test.max()


def test_walk_forward_expanding_grows_train():
    """扩展窗口：训练起点恒为 0 且不断变长。"""
    splits = walk_forward_splits(100, train_size=20, test_size=10, expanding=True)
    starts = {int(tr.min()) for tr, _ in splits}
    assert starts == {0}
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    for train, test in splits:
        assert train.max() < test.min()


def test_walk_forward_accepts_index_object():
    import pandas as pd
    idx = pd.RangeIndex(50)
    splits = walk_forward_splits(idx, train_size=20, test_size=10)
    assert len(splits) > 0
    assert all(tr.max() < te.min() for tr, te in splits)


# ---------------------------------------------------------------------------
# purged_kfold
# ---------------------------------------------------------------------------
def test_purged_kfold_no_overlap_and_full_test_coverage():
    splits = purged_kfold(20, n_splits=4, embargo=0, label_horizon=1)
    assert len(splits) == 4
    seen_test = []
    for train, test in splits:
        assert set(train).isdisjoint(set(test))
        seen_test.extend(test.tolist())
    # 每个样本恰好属于一个测试块
    assert sorted(seen_test) == list(range(20))


def test_purged_kfold_purges_labels_overlapping_test():
    """测试块之前 label_horizon 期内、标签延伸进测试区的训练样本应被剔除。"""
    n, horizon = 20, 2
    splits = purged_kfold(n, n_splits=4, embargo=0, label_horizon=horizon)
    for train, test in splits:
        ts, te = int(test.min()), int(test.max()) + 1
        train_set = set(train.tolist())
        for i in range(max(0, ts - horizon), ts):
            # i 的标签期 [i, i+horizon] 与 [ts, te) 相交 -> 必须被 purge
            assert i not in train_set


def test_purged_kfold_embargo_excludes_after_test():
    """测试块之后 embargo 期内的样本不得出现在训练集。"""
    n, embargo = 24, 3
    splits = purged_kfold(n, n_splits=4, embargo=embargo, label_horizon=1)
    for train, test in splits:
        te = int(test.max()) + 1
        train_set = set(train.tolist())
        for j in range(te, min(te + embargo, n)):
            assert j not in train_set


def test_purged_kfold_zero_horizon_no_embargo_keeps_distant_train():
    """label_horizon=0 且 embargo=0 时，非测试样本应全部保留在训练集。"""
    splits = purged_kfold(20, n_splits=5, embargo=0, label_horizon=0)
    for train, test in splits:
        union = set(train.tolist()) | set(test.tolist())
        assert union == set(range(20))


def test_purged_kfold_invalid_args():
    with pytest.raises(ValueError):
        purged_kfold(20, n_splits=1)
    with pytest.raises(ValueError):
        purged_kfold(20, n_splits=4, embargo=-1)

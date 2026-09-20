"""时序交叉验证模块：walk-forward 切分与带 purge/embargo 的 K 折。

金融时序不能用普通 K 折：标签往往向前看若干期（如 n 期远期收益、三重障碍的
垂直轨），若训练样本的标签期与测试集重叠，就会把「未来信息」泄漏进训练，
造成回测虚高。本模块提供两类严格保序、防泄漏的切分：

- :func:`walk_forward_splits` : 滚动/扩展窗口训练-测试切分（训练恒在测试之前）。
- :func:`purged_kfold`        : 自研 purged K 折——剔除与测试标签期重叠的训练样本，
                                并在测试集之后加 embargo，双重阻断信息泄漏。

两者都返回 ``List[Tuple[np.ndarray, np.ndarray]]``，元素为 (train_idx, test_idx)
的**位置整数**数组，便于直接对任意等长序列做 ``iloc`` 索引。
"""
from __future__ import annotations

from typing import List, Optional, Sized, Tuple, Union

import numpy as np
import pandas as pd

Splits = List[Tuple[np.ndarray, np.ndarray]]


def _length(n: Union[int, pd.Index, np.ndarray, Sized]) -> int:
    """把「样本数」或「带长度的索引对象」统一解析成整数长度。"""
    if isinstance(n, (int, np.integer)):
        return int(n)
    return len(n)


def walk_forward_splits(n: Union[int, Sized], train_size: int, test_size: int,
                        step: Optional[int] = None, expanding: bool = False) -> Splits:
    """Walk-forward（滚动/扩展窗口）训练-测试切分。

    参数
    ----
    n:           样本总数，或任何带 ``len`` 的索引对象（如 ``DatetimeIndex``）。
    train_size:  训练窗口长度（expanding=True 时为**首个**训练窗口长度，其后不断扩展）。
    test_size:   每个测试窗口长度。
    step:        相邻两折起点的步长，默认等于 ``test_size``（各测试集不重叠）。
    expanding:   False=滚动窗口（训练长度恒定）；True=扩展窗口（训练起点恒为 0，越长越大）。

    保证
    ----
    - 严格时间顺序：每折 ``max(train_idx) < min(test_idx)``，训练全部在测试之前。
    - 训练与测试无重叠；默认步长下各测试集互不重叠。
    """
    total = _length(n)
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size / test_size 必须 >= 1")
    step = int(step) if step is not None else int(test_size)
    if step < 1:
        raise ValueError("step 必须 >= 1")

    splits: Splits = []
    if expanding:
        boundary = int(train_size)
        while boundary < total:
            te = min(boundary + test_size, total)
            train = np.arange(0, boundary)
            test = np.arange(boundary, te)
            if test.size > 0 and train.size > 0:
                splits.append((train, test))
            if te >= total:
                break
            boundary += step
    else:
        start = 0
        while start + train_size < total:
            tr_end = start + train_size
            te = min(tr_end + test_size, total)
            train = np.arange(start, tr_end)
            test = np.arange(tr_end, te)
            if test.size > 0:
                splits.append((train, test))
            if te >= total:
                break
            start += step
    return splits


def purged_kfold(n: Union[int, Sized], n_splits: int = 5, embargo: int = 0,
                 label_horizon: int = 1) -> Splits:
    """带 purge 与 embargo 的 K 折交叉验证（自研实现）。

    把样本按时间顺序均分为 ``n_splits`` 个连续块，轮流作为测试集；对每一折：

    1. **Purge（清洗）**：训练样本 i 的标签向前看 ``label_horizon`` 期，其标签区间
       为 [i, i+label_horizon]。若该区间与测试区间 [ts, te) 相交，则 i 泄露了测试期
       信息，需从训练集剔除。相交条件：``i < te`` 且 ``i + label_horizon >= ts``。
       这会剔除测试块**之前** horizon 期内的样本（它们的标签延伸进了测试期）。
    2. **Embargo（禁运）**：额外剔除测试块**之后** ``embargo`` 期内的训练样本
       ([te, te+embargo))，以阻断特征序列相关带来的前向泄漏。

    参数
    ----
    n:              样本总数或带 ``len`` 的索引对象。
    n_splits:       折数。
    embargo:        测试集之后禁运的样本数（>=0）。
    label_horizon:  每个标签向前看的期数（>=0）；0 表示标签即时、无需 purge。

    返回
    ----
    ``List[(train_idx, test_idx)]``，均为位置整数数组；训练与测试严格无重叠。
    """
    total = _length(n)
    if n_splits < 2:
        raise ValueError("n_splits 必须 >= 2")
    if embargo < 0 or label_horizon < 0:
        raise ValueError("embargo / label_horizon 不能为负")
    if total < n_splits:
        raise ValueError("样本数不足以切成 n_splits 折")

    idx_all = np.arange(total)
    blocks = np.array_split(idx_all, n_splits)
    splits: Splits = []
    for block in blocks:
        if block.size == 0:
            continue
        ts = int(block[0])
        te = int(block[-1]) + 1
        test = idx_all[ts:te]

        mask = np.ones(total, dtype=bool)
        mask[ts:te] = False                              # 排除测试块本身
        overlap = (idx_all < te) & (idx_all + label_horizon >= ts)
        mask &= ~overlap                                 # purge 标签重叠
        if embargo > 0:
            emb = (idx_all >= te) & (idx_all < te + embargo)
            mask &= ~emb                                 # embargo 禁运
        train = idx_all[mask]
        if train.size > 0 and test.size > 0:
            splits.append((train, test))
    return splits

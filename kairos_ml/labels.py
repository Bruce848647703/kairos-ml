"""标签构造模块：远期收益、三重障碍法与元标签。

三重障碍法 (Triple-Barrier) 是金融 ML 的经典标签生成范式：为每个「入场时刻」
同时设置三条障碍——止盈上轨、止损下轨与垂直时间轨，标签取决于**最先被触达**
的障碍。上下轨宽度按波动率的倍数设定，从而对不同波动环境自适应。

- :func:`forward_returns`  : n 期远期收益（把未来收益对齐到当前时刻）。
- :func:`triple_barrier`   : 三重障碍标签 {-1, 0, +1} + 首个触达的障碍。
- :func:`meta_labeling`    : 一阶方向 + 二阶「是否值得下注」的元标签 {0, 1}。

全部为 numpy 自研实现，仅用「截至入场时刻」的波动率，杜绝未来函数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._util import as_series, returns_arr, rolling_std_arr

BarrierName = str  # 'upper' | 'lower' | 'vertical' | 'none'


def _volatility(arr: np.ndarray, vol_window: int) -> np.ndarray:
    """入场时刻的波动率（收益率滚动标准差，sigma[t] 仅用 t 及以前的收益）。"""
    rets = returns_arr(arr)
    return rolling_std_arr(rets, vol_window, ddof=1)


def _positional(entries: Optional[Sequence], idx: pd.Index, n: int) -> np.ndarray:
    """把可选的入场时刻（位置整数或时间戳）转成位置索引数组；None 表示自动。"""
    if entries is None:
        return np.arange(n)
    ent = np.asarray(list(entries))
    if ent.dtype.kind in "iu":  # 整数 -> 位置
        pos = ent.astype(int)
    else:  # 时间戳/标签 -> 通过索引定位
        pos = idx.get_indexer(ent)
    pos = pos[(pos >= 0) & (pos < n)]
    return np.unique(pos)


@dataclass
class BarrierScan:
    """单次三重障碍扫描的结果。"""
    label: int                 # +1 触上轨 / -1 触下轨 / 0 触垂直轨
    barrier: BarrierName       # 首个触达的障碍名
    offset: int                # 相对入场后第几个 bar 触达（0 起始）
    ret: float                 # 从入场到出场的（按方向调整的）收益率


def _scan_barrier(entry: float, forward: np.ndarray, sigma: float,
                  pt: float, sl: float, side: int = 1) -> BarrierScan:
    """在给定前向价格路径上，找出最先触达的障碍。

    相对路径 rel = side * (forward/entry - 1)：
    - rel >= pt*sigma  -> 触达止盈上轨（对 side 而言是盈利方向）；
    - rel <= -sl*sigma -> 触达止损下轨；
    - 垂直期内都未触达 -> 垂直轨，label=0，在路径末端出场。

    波动率非正/非有限时退化为仅有垂直轨。
    """
    m = forward.shape[0]
    if not np.isfinite(sigma) or sigma <= 0:
        rel_end = float(side * (forward[-1] / entry - 1.0))
        return BarrierScan(0, "vertical", m - 1, rel_end)
    up = pt * sigma
    dn = -sl * sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = side * (forward / entry - 1.0)
    hit_up = np.nonzero(rel >= up)[0]
    hit_dn = np.nonzero(rel <= dn)[0]
    iu = int(hit_up[0]) if hit_up.size else None
    idn = int(hit_dn[0]) if hit_dn.size else None
    if iu is None and idn is None:
        return BarrierScan(0, "vertical", m - 1, float(rel[-1]))
    if iu is not None and (idn is None or iu < idn):
        return BarrierScan(1, "upper", iu, float(rel[iu]))
    if idn is not None and (iu is None or idn < iu):
        return BarrierScan(-1, "lower", idn, float(rel[idn]))
    # 同一 bar 同时触达上下轨：无法判定先后，保守记为垂直轨 0 标签。
    off = iu if iu is not None else idn
    return BarrierScan(0, "vertical", int(off), float(rel[int(off)]))


def forward_returns(prices: Union[pd.Series, pd.DataFrame], periods: int = 1
                    ) -> Union[pd.Series, pd.DataFrame]:
    """n 期远期收益：fret[t] = price[t+periods] / price[t] - 1。

    把「未来 periods 期的收益」对齐到当前时刻 t，用作监督学习的目标。
    末尾 periods 个点因无未来数据为 NaN。
    """
    if periods < 1:
        raise ValueError("periods 必须 >= 1")

    def _f(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype="float64")
        n = arr.shape[0]
        out = np.full(n, np.nan)
        if n > periods:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:-periods] = arr[periods:] / arr[:-periods] - 1.0
        return pd.Series(out, index=s.index, name=s.name)

    if isinstance(prices, pd.DataFrame):
        return pd.DataFrame({c: _f(prices[c]) for c in prices.columns}, index=prices.index)
    s = prices if isinstance(prices, pd.Series) else pd.Series(np.asarray(prices, dtype="float64"))
    return _f(s)


def triple_barrier(prices: pd.Series, pt: float = 2.0, sl: float = 2.0,
                   vertical: int = 10, vol_window: int = 20, side: int = 1,
                   entry_times: Optional[Sequence] = None) -> pd.DataFrame:
    """三重障碍法打标签（自研实现）。

    参数
    ----
    prices:      单资产价格 ``Series``。
    pt, sl:      止盈/止损轨的波动率倍数（上轨 = 入场价×(1+pt·σ)，下轨 = 入场价×(1−sl·σ)）。
    vertical:    垂直时间轨长度（bar 数），超过则强制在末端出场。
    vol_window:  计算入场波动率 σ 所用的收益率滚动窗口。
    side:        交易方向，+1 做多 / −1 做空；默认 +1，此时 upper/lower 即价格上/下轨。
    entry_times: 可选的入场时刻（位置整数或时间戳）；默认对所有可用时刻打标。

    返回
    ----
    以入场时刻为索引的 ``DataFrame``，列包括：
    ``label`` ∈ {−1, 0, +1}、``barrier``（首个触达的障碍名）、``ret``（出场收益）、
    ``t_exit``（出场时刻）、``upper`` / ``lower``（障碍价位）。
    """
    s = as_series(prices)
    arr = s.to_numpy(dtype="float64")
    idx = s.index
    n = arr.shape[0]
    sigma = _volatility(arr, vol_window)
    entries = _positional(entry_times, idx, n)

    rows: List[tuple] = []
    for t in entries:
        if t >= n - 1:            # 没有前向数据
            continue
        entry = arr[t]
        sig = sigma[t]
        if not np.isfinite(entry):
            continue
        forward = arr[t + 1:t + 1 + vertical]
        if forward.shape[0] == 0:
            continue
        scan = _scan_barrier(entry, forward, sig, pt, sl, side)
        exit_pos = min(t + 1 + scan.offset, n - 1)
        if np.isfinite(sig) and sig > 0:
            up_px = entry * (1.0 + side * pt * sig)
            dn_px = entry * (1.0 - side * sl * sig)
        else:
            up_px = dn_px = np.nan
        rows.append((idx[t], scan.label, scan.barrier, scan.ret, idx[exit_pos], up_px, dn_px))

    df = pd.DataFrame(rows, columns=["entry", "label", "barrier", "ret", "t_exit", "upper", "lower"])
    if df.empty:
        return df.set_index("entry")
    return df.set_index("entry")


def meta_labeling(prices: pd.Series, side: Union[pd.Series, Sequence[float]],
                  pt: float = 2.0, sl: float = 2.0, vertical: int = 10,
                  vol_window: int = 20) -> pd.DataFrame:
    """元标签 (Meta-Labeling)：在一阶方向之上，构造「是否值得下注」的二阶标签。

    一阶模型给出方向 ``side`` ∈ {−1, 0, +1}（0 表示不交易）。对每个 ``side≠0``
    的时刻，沿该方向执行三重障碍扫描：若在垂直期内先触达**盈利轨**，则元标签
    为 1（这注该下），否则为 0。``side=0`` 的时刻元标签恒为 0 且 ``active=0``。

    二阶模型据此学习「该信号盈利的概率」，用于决定下注与否或仓位大小，
    从而在不改变一阶召回的前提下提升精确率。

    返回
    ----
    ``DataFrame``，列包括：``meta_label`` ∈ {0,1}、``side``（一阶方向）、
    ``barrier``（首个触达障碍，从交易方向视角）、``ret``（按方向调整的出场收益）、
    ``t_exit``、``active``（该时刻是否有下注信号）。
    """
    s = as_series(prices)
    arr = s.to_numpy(dtype="float64")
    idx = s.index
    n = arr.shape[0]
    sigma = _volatility(arr, vol_window)

    if isinstance(side, pd.Series):
        side_arr = side.reindex(idx).to_numpy(dtype="float64")
    elif isinstance(side, pd.DataFrame):
        side_arr = side.iloc[:, 0].reindex(idx).to_numpy(dtype="float64")
    else:
        side_arr = np.asarray(list(side), dtype="float64")
        if side_arr.shape[0] != n:
            raise ValueError("side 长度需与 prices 一致")

    rows: List[tuple] = []
    for t in range(n):
        if t >= n - 1 or not np.isfinite(sigma[t]) or not np.isfinite(arr[t]):
            continue
        sd = side_arr[t]
        if not np.isfinite(sd) or sd == 0:
            rows.append((idx[t], 0, 0.0 if not np.isfinite(sd) else sd,
                         "none", 0.0, idx[t], 0))
            continue
        direction = int(np.sign(sd))
        forward = arr[t + 1:t + 1 + vertical]
        if forward.shape[0] == 0:
            continue
        scan = _scan_barrier(arr[t], forward, sigma[t], pt, sl, direction)
        meta = 1 if scan.label == 1 else 0
        exit_pos = min(t + 1 + scan.offset, n - 1)
        rows.append((idx[t], meta, sd, scan.barrier, scan.ret, idx[exit_pos], 1))

    df = pd.DataFrame(rows, columns=["entry", "meta_label", "side", "barrier", "ret", "t_exit", "active"])
    if df.empty:
        return df.set_index("entry")
    return df.set_index("entry")

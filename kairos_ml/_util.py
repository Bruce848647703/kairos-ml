"""内部数值工具集（纯 numpy 实现，供各模块复用）。

集中放置滚动统计、指数平滑、收益率、秩相关、正态分布等基础算法，
避免在各业务模块里重复造轮子。全部为自研实现，不依赖 scipy/sklearn；
scipy 若存在则用于加速正态 CDF，缺失时自动回退到 numpy 近似。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 基础类型转换
# ---------------------------------------------------------------------------
def to_1d_float(x) -> np.ndarray:
    """把 Series / ndarray / 列表统一转成一维 float64 数组。"""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype="float64")
    elif isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype="float64").ravel()
    else:
        arr = np.asarray(x, dtype="float64")
    return arr.ravel()


def as_series(x, name: Optional[str] = None) -> pd.Series:
    """把输入统一成 pandas.Series（保留原有索引，ndarray 则用整数索引）。"""
    if isinstance(x, pd.Series):
        s = x.astype("float64")
        if name is not None:
            s = s.rename(name)
        return s
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("as_series 仅接受单列 DataFrame")
        s = x.iloc[:, 0].astype("float64")
        return s.rename(name) if name is not None else s
    arr = np.asarray(x, dtype="float64").ravel()
    return pd.Series(arr, name=name)


# ---------------------------------------------------------------------------
# 收益率与滚动统计
# ---------------------------------------------------------------------------
def returns_arr(arr: np.ndarray) -> np.ndarray:
    """简单收益率序列：ret[t] = arr[t]/arr[t-1] - 1，ret[0] = NaN。"""
    arr = np.asarray(arr, dtype="float64")
    out = np.full(arr.shape[0], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = arr[1:] / arr[:-1] - 1.0
    return out


def _sliding(arr: np.ndarray, window: int) -> Optional[np.ndarray]:
    """返回形状 (n-window+1, window) 的滑窗视图；数据不足时返回 None。"""
    arr = np.asarray(arr, dtype="float64")
    n = arr.shape[0]
    if window < 1 or n < window:
        return None
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        return sliding_window_view(arr, window)
    except Exception:  # pragma: no cover - 极旧 numpy 回退
        return None


def rolling_mean_arr(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动均值；窗口不足或窗口内含 NaN 时该点为 NaN（等价 min_periods=window）。"""
    arr = np.asarray(arr, dtype="float64")
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if window < 1 or n < window:
        return out
    wins = _sliding(arr, window)
    if wins is not None:
        out[window - 1:] = np.mean(wins, axis=1)
    else:  # pragma: no cover
        for i in range(window - 1, n):
            out[i] = np.mean(arr[i - window + 1:i + 1])
    return out


def rolling_std_arr(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """滚动标准差；窗口不足或含 NaN 时为 NaN。"""
    arr = np.asarray(arr, dtype="float64")
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if window < 1 or n < window:
        return out
    wins = _sliding(arr, window)
    if wins is not None:
        out[window - 1:] = np.std(wins, axis=1, ddof=ddof)
    else:  # pragma: no cover
        for i in range(window - 1, n):
            out[i] = np.std(arr[i - window + 1:i + 1], ddof=ddof)
    return out


def ema_arr(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均（递归平滑，alpha=2/(span+1)，以首个有限值播种）。

    与 pandas ``ewm(span=..., adjust=False)`` 的递归形式一致；前导 NaN 原样保留。
    """
    arr = np.asarray(arr, dtype="float64")
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if span < 1 or n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    started = False
    for i in range(n):
        x = arr[i]
        if not np.isfinite(x):
            continue
        if not started:
            prev = x
            started = True
        else:
            prev = alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return out


def wilder_arr(arr: np.ndarray, window: int) -> np.ndarray:
    """Wilder 平滑（RSI 专用）：首值取前 window 个的均值，其后递归 (prev*(w-1)+x)/w。"""
    arr = np.asarray(arr, dtype="float64")
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if window < 1 or n < window:
        return out
    seed = np.mean(arr[:window])
    out[window - 1] = seed
    for i in range(window, n):
        out[i] = (out[i - 1] * (window - 1) + arr[i]) / window
    return out


# ---------------------------------------------------------------------------
# 秩与相关（自研 Spearman / Pearson）
# ---------------------------------------------------------------------------
def rankdata_avg(a: np.ndarray) -> np.ndarray:
    """一维数组的平均秩（并列取平均，1 起始），等价 scipy.stats.rankdata 的 'average'。"""
    a = np.asarray(a, dtype="float64")
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype="float64")
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """皮尔逊相关系数；任一序列方差为 0 或样本不足时返回 0.0。"""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape[0] < 2 or a.shape[0] != b.shape[0]:
        return 0.0
    am = a - a.mean()
    bm = b - b.mean()
    da = float(np.sqrt(np.sum(am * am)))
    db = float(np.sqrt(np.sum(bm * bm)))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(np.sum(am * bm) / (da * db))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """斯皮尔曼秩相关 = 秩上的皮尔逊相关。"""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape[0] < 2 or a.shape[0] != b.shape[0]:
        return 0.0
    return pearson_corr(rankdata_avg(a), rankdata_avg(b))


# ---------------------------------------------------------------------------
# 正态分布 CDF（scipy 可选，缺失回退 numpy 近似）
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 取决于运行环境
    from scipy.special import erf as _scipy_erf  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_erf = None
    _HAS_SCIPY = False


def _erf_fallback(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 误差函数近似（最大绝对误差约 1.5e-7）。"""
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                          -1.453152027, 1.061405429)
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def norm_cdf(x) -> np.ndarray:
    """标准正态分布累积分布函数 Φ(x)。优先用 scipy.erf，缺失时回退到 numpy 近似。"""
    x = np.asarray(x, dtype="float64")
    z = x / np.sqrt(2.0)
    if _HAS_SCIPY:  # pragma: no cover
        return 0.5 * (1.0 + _scipy_erf(z))
    return 0.5 * (1.0 + _erf_fallback(z))


def has_scipy() -> bool:
    """是否检测到可用的 scipy（用于文档/诊断）。"""
    return _HAS_SCIPY


# ---------------------------------------------------------------------------
# 设计矩阵工具
# ---------------------------------------------------------------------------
def check_X(X) -> Tuple[np.ndarray, Optional[list]]:
    """把 X 转成二维 float 数组，并尽力保留列名。"""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype="float64"), list(X.columns)
    arr = np.asarray(X, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, None


def check_y(y) -> np.ndarray:
    """把 y 转成一维 float 数组。"""
    return to_1d_float(y)

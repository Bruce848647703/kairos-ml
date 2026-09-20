"""模型评估模块：命中率、分类指标、信号 PnL 与信息系数 (IC)。

面向金融 ML 的常用评估口径，全部 numpy/pandas 自研实现：

- :func:`hit_rate`            : 方向命中率（预测与真实的符号一致比例）。
- :func:`precision` / :func:`recall` / :func:`f1` : 二分类指标。
- :func:`signal_pnl`          : 信号 × 远期收益的逐期盈亏（可再汇总）。
- :func:`ic`                  : 信息系数，预测与实际的（秩）相关。
- :func:`t_statistic`         : 盈亏序列的 t 统计量与显著性（正态 CDF 支持 scipy 可选）。
- :func:`value_at_risk`       : 参数法 VaR（正态分位数，scipy 可选，缺失回退）。
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

from ._util import (
    has_scipy,
    norm_cdf,
    pearson_corr,
    spearman_corr,
    to_1d_float,
)

SeriesLike = Union[pd.Series, np.ndarray]


def hit_rate(y_true: SeriesLike, y_pred: SeriesLike) -> float:
    """方向命中率：``sign(y_true) == sign(y_pred)`` 的比例。

    对连续收益即「预测涨跌方向正确率」；对 {0,1} 标签等价于准确率。
    自动按索引对齐（若均为 Series）并忽略任一为 NaN 的点。
    """
    a, b = _align_pair(y_true, y_pred)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.sign(a) == np.sign(b)))


def precision(y_true: SeriesLike, y_pred: SeriesLike, pos_label: int = 1) -> float:
    """精确率 = TP / (TP + FP)；分母为 0 时返回 0.0。"""
    a, b = _align_pair(y_true, y_pred)
    tp = int(np.sum((b == pos_label) & (a == pos_label)))
    fp = int(np.sum((b == pos_label) & (a != pos_label)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: SeriesLike, y_pred: SeriesLike, pos_label: int = 1) -> float:
    """召回率 = TP / (TP + FN)；分母为 0 时返回 0.0。"""
    a, b = _align_pair(y_true, y_pred)
    tp = int(np.sum((b == pos_label) & (a == pos_label)))
    fn = int(np.sum((b != pos_label) & (a == pos_label)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(y_true: SeriesLike, y_pred: SeriesLike, pos_label: int = 1) -> float:
    """F1 = 2·precision·recall / (precision + recall)；两者皆 0 时返回 0.0。"""
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def signal_pnl(signal: SeriesLike, forward_returns: SeriesLike) -> pd.Series:
    """信号盈亏：逐期 ``signal × forward_returns``，返回每期 PnL 序列。

    ``signal`` 可为方向 {−1,0,+1} 或仓位权重；正值表示做多、负值做空。
    结果序列的 ``sum()`` 为累计 PnL，``mean()`` 为单期平均 PnL。
    """
    s, r = _align_pair(signal, forward_returns, as_pandas=True)
    return pd.Series(s * r, name="pnl")


def ic(predictions: SeriesLike, actual: SeriesLike, method: str = "spearman") -> float:
    """信息系数 (IC)：预测值与实际值的相关性。

    method='spearman'（默认）为秩相关，对异常值更稳健；'pearson' 为线性相关。
    """
    a, b = _align_pair(predictions, actual)
    if a.size < 2:
        return 0.0
    if method == "spearman":
        return spearman_corr(a, b)
    if method == "pearson":
        return pearson_corr(a, b)
    raise ValueError("method 仅支持 'spearman' 或 'pearson'")


def t_statistic(pnl: SeriesLike) -> Tuple[float, float]:
    """盈亏序列的 t 统计量与双侧 p 值（正态近似）。

    返回 (t, p)。p 值用标准正态 CDF 计算，:func:`kairos_ml._util.norm_cdf`
    在检测到 scipy 时用 ``scipy.special.erf``，否则回退到 numpy 近似。
    """
    x = to_1d_float(pnl)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 0.0, 1.0
    sd = float(np.std(x, ddof=1))
    if sd < 1e-12:
        return 0.0, 1.0
    t = float(np.mean(x) / (sd / np.sqrt(n)))
    p = float(2.0 * (1.0 - norm_cdf(abs(t))))
    return t, p


def value_at_risk(returns: SeriesLike, confidence: float = 0.95) -> float:
    """参数法风险价值 (VaR)：返回给定置信度下的（正数）损失分位。

    假设收益近似正态，VaR = −(μ + z_{1−c}·σ)。正态分位数 z 用 scipy（若可用）
    或 Acklam 逆正态近似；此处以标准差与均值直接给出。
    """
    x = to_1d_float(returns)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    z = _norm_ppf(1.0 - float(confidence))  # 负值
    return float(-(mu + z * sd))


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _norm_ppf(q: float) -> float:
    """标准正态分位数（逆 CDF）。优先 scipy，缺失时用 Acklam 有理逼近。"""
    if not (0.0 < q < 1.0):
        return 0.0
    try:  # pragma: no cover - 取决于运行环境
        from scipy.stats import norm as _norm  # type: ignore

        return float(_norm.ppf(q))
    except Exception:  # pragma: no cover
        return float(_acklam_ppf(q))


def _acklam_ppf(q: float) -> float:
    """Acklam 逆正态 CDF 有理逼近（相对误差 < 1.15e-9）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        r = np.sqrt(-2 * np.log(q))
        return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    if q > phigh:
        r = np.sqrt(-2 * np.log(1 - q))
        return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    r = q - 0.5
    s = r * r
    return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
           (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)


def _align_pair(a: SeriesLike, b: SeriesLike, as_pandas: bool = False):
    """对齐两个序列并剔除任一为 NaN 的点。

    as_pandas=False 返回两个 ndarray；True 返回对齐后的两个 ndarray（用于 signal_pnl
    时再包装成 Series）。
    """
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        av = joined["a"].to_numpy(dtype="float64")
        bv = joined["b"].to_numpy(dtype="float64")
    else:
        av = to_1d_float(a)
        bv = to_1d_float(b)
        m = min(av.size, bv.size)
        av, bv = av[:m], bv[:m]
        mask = np.isfinite(av) & np.isfinite(bv)
        av, bv = av[mask], bv[mask]
    return av, bv

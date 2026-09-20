"""特征工程模块：滚动特征、滞后项、截面变换与自研技术指标。

约定
----
- 单资产时序函数（滚动统计、SMA/EMA/RSI/MACD 等）接受 ``Series`` 或 ``DataFrame``。
  传入 ``DataFrame`` 时按列分别计算，返回同形状的 ``DataFrame``。
- 截面函数（``cross_sectional_rank`` / ``cross_sectional_zscore``）接受
  ``index=时间, columns=资产`` 的 ``DataFrame``，在**每个时间截面**（沿列）上计算。
- 所有指标只用「截至当前 bar」的数据，不含未来函数；窗口不足的早期点为 NaN。

全部为 numpy 自研实现（见 :mod:`kairos_ml._util`），不依赖任何第三方 ML 库。
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ._util import (
    ema_arr,
    returns_arr,
    rolling_mean_arr,
    rolling_std_arr,
    wilder_arr,
    rankdata_avg,
)

SeriesOrFrame = Union[pd.Series, pd.DataFrame, np.ndarray]


def _map_columns(x: SeriesOrFrame, fn: Callable[[pd.Series], pd.Series]) -> Union[pd.Series, pd.DataFrame]:
    """把「作用于单列 Series 的函数」广播到 DataFrame 的每一列。"""
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame({c: fn(x[c]) for c in x.columns}, index=x.index)
    s = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype="float64"))
    return fn(s)


# ---------------------------------------------------------------------------
# 滚动 / 滞后特征
# ---------------------------------------------------------------------------
def rolling_mean(series: SeriesOrFrame, window: int) -> Union[pd.Series, pd.DataFrame]:
    """滚动均值（简单移动平均）。窗口不足处为 NaN。"""
    def _f(s: pd.Series) -> pd.Series:
        return pd.Series(rolling_mean_arr(s.to_numpy(dtype="float64"), window),
                         index=s.index, name=s.name)
    return _map_columns(series, _f)


def rolling_std(series: SeriesOrFrame, window: int, ddof: int = 1) -> Union[pd.Series, pd.DataFrame]:
    """滚动标准差（默认样本标准差 ddof=1）。"""
    def _f(s: pd.Series) -> pd.Series:
        return pd.Series(rolling_std_arr(s.to_numpy(dtype="float64"), window, ddof),
                         index=s.index, name=s.name)
    return _map_columns(series, _f)


def rolling_momentum(series: SeriesOrFrame, window: int) -> Union[pd.Series, pd.DataFrame]:
    """滚动动量（window 期变化率）：mom[t] = x[t] / x[t-window] - 1。"""
    def _f(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype="float64")
        n = arr.shape[0]
        out = np.full(n, np.nan)
        if window >= 1 and n > window:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[window:] = arr[window:] / arr[:-window] - 1.0
        return pd.Series(out, index=s.index, name=s.name)
    return _map_columns(series, _f)


def lags(series: pd.Series, n_lags: int, prefix: str = "lag") -> pd.DataFrame:
    """构造滞后特征：返回列 ``lag_1 .. lag_n``（lag_k[t] = series[t-k]）。

    仅使用历史值，天然无未来函数；前 k 行对应位置为 NaN。
    """
    s = series if isinstance(series, pd.Series) else pd.Series(np.asarray(series, dtype="float64"))
    if n_lags < 1:
        raise ValueError("n_lags 必须 >= 1")
    data = {}
    for k in range(1, n_lags + 1):
        arr = s.to_numpy(dtype="float64")
        out = np.full(arr.shape[0], np.nan)
        if arr.shape[0] > k:
            out[k:] = arr[:-k]
        data[f"{prefix}_{k}"] = out
    return pd.DataFrame(data, index=s.index)


# ---------------------------------------------------------------------------
# 截面变换
# ---------------------------------------------------------------------------
def cross_sectional_rank(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """在每个时间截面上对资产做秩变换（沿列，axis=1）。

    pct=True 时归一化到 (0, 1]，便于跨截面对比；并列取平均秩。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("cross_sectional_rank 需要 DataFrame(index=时间, columns=资产)")
    vals = df.to_numpy(dtype="float64")
    out = np.full_like(vals, np.nan)
    for r in range(vals.shape[0]):
        row = vals[r]
        mask = np.isfinite(row)
        m = int(mask.sum())
        if m == 0:
            continue
        ranked = rankdata_avg(row[mask])
        if pct:
            ranked = ranked / m
        out[r, mask] = ranked
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def cross_sectional_zscore(df: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """在每个时间截面上对资产做标准化 (x - 均值) / 标准差（沿列，axis=1）。

    某截面标准差为 0（或有效样本不足）时，该截面输出 0。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("cross_sectional_zscore 需要 DataFrame(index=时间, columns=资产)")
    vals = df.to_numpy(dtype="float64")
    out = np.full_like(vals, np.nan)
    for r in range(vals.shape[0]):
        row = vals[r]
        mask = np.isfinite(row)
        m = int(mask.sum())
        if m == 0:
            continue
        sub = row[mask]
        mu = float(np.mean(sub))
        sd = float(np.std(sub, ddof=ddof)) if m > ddof else 0.0
        if sd < 1e-12:
            out[r, mask] = 0.0
        else:
            out[r, mask] = (sub - mu) / sd
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ---------------------------------------------------------------------------
# 自研技术指标
# ---------------------------------------------------------------------------
def sma(series: SeriesOrFrame, window: int) -> Union[pd.Series, pd.DataFrame]:
    """简单移动平均 (SMA)，等价于滚动均值。"""
    return rolling_mean(series, window)


def ema(series: SeriesOrFrame, span: int) -> Union[pd.Series, pd.DataFrame]:
    """指数移动平均 (EMA)，alpha = 2/(span+1)，以首个有效值播种的递归平滑。"""
    def _f(s: pd.Series) -> pd.Series:
        return pd.Series(ema_arr(s.to_numpy(dtype="float64"), span), index=s.index, name=s.name)
    return _map_columns(series, _f)


def rsi(series: SeriesOrFrame, window: int = 14) -> Union[pd.Series, pd.DataFrame]:
    """相对强弱指标 (RSI)，采用 Wilder 平滑，取值 0~100。

    RSI = 100 - 100/(1+RS)，RS = 平均涨幅 / 平均跌幅。全程上涨记为 100，
    无涨无跌记为 50。前 window 个点因数据不足为 NaN。
    """
    def _f(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype="float64")
        n = arr.shape[0]
        out = np.full(n, np.nan)
        if n < 2 or window < 1:
            return pd.Series(out, index=s.index, name=s.name)
        delta = np.diff(arr)  # 长度 n-1，delta[i] 对应价格索引 i+1
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = wilder_arr(gains, window)
        avg_loss = wilder_arr(losses, window)
        rs_rsi = np.full(gains.shape[0], np.nan)
        valid = np.isfinite(avg_gain) & np.isfinite(avg_loss)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_loss > 1e-12, avg_gain / np.where(avg_loss > 1e-12, avg_loss, 1.0), np.inf)
            rsi_vals = 100.0 - 100.0 / (1.0 + rs)
        # 平均跌幅为 0：有涨幅 -> 100，无涨幅 -> 50
        rsi_vals = np.where((avg_loss <= 1e-12) & (avg_gain > 1e-12), 100.0, rsi_vals)
        rsi_vals = np.where((avg_loss <= 1e-12) & (avg_gain <= 1e-12), 50.0, rsi_vals)
        rs_rsi[valid] = rsi_vals[valid]
        out[1:] = rs_rsi  # 对齐回价格索引
        return pd.Series(out, index=s.index, name=s.name)
    return _map_columns(series, _f)


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """移动平均收敛散度 (MACD)，仅接受单资产 ``Series``。

    返回三列 DataFrame：``macd``(快线-慢线)、``signal``(macd 的 EMA)、
    ``hist``(macd - signal)。基于自研 :func:`ema_arr` 递归平滑。
    """
    s = series if isinstance(series, pd.Series) else pd.Series(np.asarray(series, dtype="float64"))
    arr = s.to_numpy(dtype="float64")
    ema_fast = ema_arr(arr, fast)
    ema_slow = ema_arr(arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_arr(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist}, index=s.index)


def realized_volatility(series: SeriesOrFrame, window: int = 20,
                        periods_per_year: Optional[int] = 252) -> Union[pd.Series, pd.DataFrame]:
    """已实现波动率：对**收益率**做滚动标准差，可选按 ``periods_per_year`` 年化。

    传入价格会先转成简单收益率再计算；``periods_per_year=None`` 则返回未年化的
    每期波动率。窗口不足处为 NaN。
    """
    def _f(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype="float64")
        rets = returns_arr(arr)
        vol = rolling_std_arr(rets, window, ddof=1)
        if periods_per_year is not None:
            vol = vol * float(np.sqrt(periods_per_year))
        return pd.Series(vol, index=s.index, name=s.name)
    return _map_columns(series, _f)

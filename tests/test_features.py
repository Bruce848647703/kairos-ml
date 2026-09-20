"""特征工程测试：滚动/滞后/截面特征与自研技术指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_ml as kml
from kairos_ml.features import (
    cross_sectional_rank,
    cross_sectional_zscore,
    ema,
    lags,
    macd,
    realized_volatility,
    rolling_mean,
    rolling_momentum,
    rolling_std,
    rsi,
    sma,
)


def test_rolling_mean_known():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rolling_mean(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_rolling_std_positive_and_nan_warmup():
    s = pd.Series(np.arange(10, dtype="float64"))
    out = rolling_std(s, 4)
    assert int(out.isna().sum()) == 3  # 前 window-1 个为 NaN
    assert np.all(out.dropna().to_numpy() > 0)


def test_sma_equals_rolling_mean():
    s = pd.Series(np.linspace(1, 50, 50))
    pd.testing.assert_series_equal(sma(s, 5), rolling_mean(s, 5))


def test_rolling_momentum_known():
    s = pd.Series([100.0, 100.0, 110.0, 121.0])
    out = rolling_momentum(s, 2)
    # mom[2] = 110/100 - 1 = 0.10 ; mom[3] = 121/100 - 1 = 0.21
    assert out.iloc[2] == pytest.approx(0.10, rel=1e-9)
    assert out.iloc[3] == pytest.approx(0.21, rel=1e-9)
    assert np.isnan(out.iloc[1])


def test_lags_structure_and_values():
    idx = pd.RangeIndex(5)
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
    df = lags(s, 2)
    assert list(df.columns) == ["lag_1", "lag_2"]
    assert df["lag_1"].iloc[1] == 10.0
    assert df["lag_2"].iloc[2] == 10.0
    assert np.isnan(df["lag_1"].iloc[0]) and np.isnan(df["lag_2"].iloc[1])


def test_ema_manual_recursion():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = ema(s, span=3)  # alpha=0.5，以首值播种
    expected = [1.0, 1.5, 2.25, 3.125]
    assert out.tolist() == pytest.approx(expected, rel=1e-9)


def test_rsi_all_gains_near_100_all_losses_near_0():
    up = pd.Series(np.linspace(100, 200, 40))
    down = pd.Series(np.linspace(200, 100, 40))
    rsi_up = rsi(up, 14).dropna()
    rsi_dn = rsi(down, 14).dropna()
    assert np.all(rsi_up.to_numpy() > 99.0)
    assert np.all(rsi_dn.to_numpy() < 1.0)


def test_rsi_within_range():
    rng = np.random.default_rng(0)
    s = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01)))
    r = rsi(s, 14).dropna()
    assert np.all(r.to_numpy() >= 0.0) and np.all(r.to_numpy() <= 100.0)


def test_macd_columns_and_relation():
    s = pd.Series(np.linspace(100, 150, 60))
    m = macd(s, fast=5, slow=10, signal=3)
    assert list(m.columns) == ["macd", "signal", "hist"]
    # hist = macd - signal
    valid = m.dropna()
    assert np.allclose(valid["hist"], valid["macd"] - valid["signal"])


def test_realized_volatility_zero_for_constant_returns():
    # 恒定涨幅 -> 收益率方差为 0 -> 已实现波动率为 0
    prices = pd.Series([100.0 * (1.01 ** i) for i in range(30)])
    rv = realized_volatility(prices, window=10).dropna()
    assert np.allclose(rv.to_numpy(), 0.0, atol=1e-12)


def test_cross_sectional_rank_pct():
    df = pd.DataFrame({"A": [10.0], "B": [20.0], "C": [30.0]})
    out = cross_sectional_rank(df, pct=True)
    assert out.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_cross_sectional_zscore_row():
    df = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
    out = cross_sectional_zscore(df)
    assert out.iloc[0].tolist() == pytest.approx([-1.0, 0.0, 1.0], abs=1e-9)


def test_dataframe_input_applies_columnwise():
    idx = pd.RangeIndex(30)
    df = pd.DataFrame({"A": np.linspace(1, 30, 30), "B": np.linspace(30, 1, 30)}, index=idx)
    out = rolling_mean(df, 5)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["A", "B"]
    assert np.isnan(out["A"].iloc[3]) and not np.isnan(out["A"].iloc[5])

"""标签测试：远期收益、三重障碍法的首触障碍、元标签。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_ml as kml
from kairos_ml.labels import forward_returns, meta_labeling, triple_barrier


def _prices_from_returns(rets, p0: float = 100.0) -> pd.Series:
    """由「逐步收益率列表」构造价格 Series（rets[i] 为 i-1→i 的收益，rets[0] 忽略）。"""
    prices = [p0]
    for i in range(1, len(rets)):
        prices.append(prices[-1] * (1.0 + rets[i]))
    return pd.Series(prices, dtype="float64")


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------
def test_forward_returns_known_values():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])
    fr = forward_returns(prices, periods=1)
    assert fr.iloc[0] == pytest.approx(0.10, rel=1e-9)
    assert fr.iloc[1] == pytest.approx(0.10, rel=1e-9)
    assert np.isnan(fr.iloc[-1])  # 最后一期无未来数据


def test_forward_returns_multi_period_and_tail_nan():
    prices = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
    fr = forward_returns(prices, periods=2)
    assert fr.iloc[0] == pytest.approx(110.0 / 100.0 - 1, rel=1e-9)
    assert int(fr.tail(2).isna().sum()) == 2


# ---------------------------------------------------------------------------
# triple_barrier：精心构造路径，验证首个触达的障碍
# ---------------------------------------------------------------------------
# 说明：入场波动率 sigma 由入场前的交替涨跌 (+c,-c) 决定，窗口 4 时
#       sigma = c*sqrt(4/3)。pt=sl=1 时上/下轨约为入场价的 ±1.1547%。
C = 0.01
SIGMA = C * np.sqrt(4.0 / 3.0)


def _alternating_prefix(t: int) -> list:
    """构造前 t 步交替涨跌的收益列表（rets[0]=0 占位）。"""
    rets = [0.0]
    for i in range(1, t + 1):
        rets.append(C if i % 2 == 1 else -C)
    return rets


def test_triple_barrier_upper_first():
    """入场后价格立刻大涨，应先触达上轨 -> label=+1, barrier='upper'。"""
    rets = _alternating_prefix(8)          # 建立 sigma（rets[1..8]）
    rets = rets + [0.02] + [0.0] * 5       # rets[9]=+2% 跳涨
    prices = _prices_from_returns(rets)
    tb = triple_barrier(prices, pt=1.0, sl=1.0, vertical=5, vol_window=4,
                        entry_times=[8])
    row = tb.iloc[0]
    assert int(row["label"]) == 1
    assert row["barrier"] == "upper"
    # 上轨价位 = 入场价 * (1 + 1*sigma)
    entry = prices.iloc[8]
    assert row["upper"] == pytest.approx(entry * (1 + SIGMA), rel=1e-9)


def test_triple_barrier_lower_first():
    """入场后价格立刻大跌，应先触达下轨 -> label=-1, barrier='lower'。"""
    rets = _alternating_prefix(8)
    rets = rets + [-0.02] + [0.0] * 5      # rets[9]=-2% 跳跌
    prices = _prices_from_returns(rets)
    tb = triple_barrier(prices, pt=1.0, sl=1.0, vertical=5, vol_window=4,
                        entry_times=[8])
    row = tb.iloc[0]
    assert int(row["label"]) == -1
    assert row["barrier"] == "lower"


def test_triple_barrier_vertical_when_no_horizontal_touch():
    """垂直期内小幅波动、始终不触上下轨 -> label=0, barrier='vertical'。"""
    rets = _alternating_prefix(8)
    rets = rets + [0.001] * 5              # 每步 +0.1%，累计 <1.15%
    prices = _prices_from_returns(rets)
    tb = triple_barrier(prices, pt=1.0, sl=1.0, vertical=5, vol_window=4,
                        entry_times=[8])
    row = tb.iloc[0]
    assert int(row["label"]) == 0
    assert row["barrier"] == "vertical"


def test_triple_barrier_upper_before_lower_ordering():
    """先小幅下探（不触下轨）再大涨触上轨，应判为上轨。"""
    rets = _alternating_prefix(8)
    rets = rets + [-0.005, 0.02, 0.0, 0.0, 0.0]  # 先 -0.5%（未达 -1.15%），再 +2%
    prices = _prices_from_returns(rets)
    tb = triple_barrier(prices, pt=1.0, sl=1.0, vertical=5, vol_window=4,
                        entry_times=[8])
    assert int(tb.iloc[0]["label"]) == 1
    assert tb.iloc[0]["barrier"] == "upper"


def test_triple_barrier_labels_within_set_on_synthetic():
    """随机路径上所有标签都应落在 {-1,0,1}，障碍名合法。"""
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(300) * 0.01
    prices = _prices_from_returns(np.r_[0.0, rets])
    tb = triple_barrier(prices, pt=1.5, sl=1.5, vertical=8, vol_window=20)
    assert set(tb["label"].unique()).issubset({-1, 0, 1})
    assert set(tb["barrier"].unique()).issubset({"upper", "lower", "vertical"})
    assert len(tb) > 0


# ---------------------------------------------------------------------------
# meta_labeling
# ---------------------------------------------------------------------------
def test_meta_labeling_long_profit_is_one():
    """一阶做多、价格大涨触盈利轨 -> 元标签 1。"""
    rets = _alternating_prefix(8) + [0.02] + [0.0] * 5
    prices = _prices_from_returns(rets)
    side = pd.Series(0.0, index=prices.index)
    side.iloc[8] = 1.0                     # 仅在 t=8 给出做多信号
    ml = meta_labeling(prices, side, pt=1.0, sl=1.0, vertical=5, vol_window=4)
    row = ml.loc[8]
    assert int(row["meta_label"]) == 1
    assert int(row["active"]) == 1


def test_meta_labeling_long_loss_is_zero():
    """一阶做多但价格大跌触止损轨 -> 元标签 0（这注不该下）。"""
    rets = _alternating_prefix(8) + [-0.02] + [0.0] * 5
    prices = _prices_from_returns(rets)
    side = pd.Series(0.0, index=prices.index)
    side.iloc[8] = 1.0
    ml = meta_labeling(prices, side, pt=1.0, sl=1.0, vertical=5, vol_window=4)
    assert int(ml.loc[8]["meta_label"]) == 0


def test_meta_labeling_short_profit_is_one():
    """一阶做空、价格大跌（对空头是盈利）-> 元标签 1。"""
    rets = _alternating_prefix(8) + [-0.02] + [0.0] * 5
    prices = _prices_from_returns(rets)
    side = pd.Series(0.0, index=prices.index)
    side.iloc[8] = -1.0                    # 做空
    ml = meta_labeling(prices, side, pt=1.0, sl=1.0, vertical=5, vol_window=4)
    assert int(ml.loc[8]["meta_label"]) == 1


def test_meta_labeling_no_signal_is_inactive_zero():
    """side=0 的时刻：不下注，元标签 0 且 active=0。"""
    rets = _alternating_prefix(8) + [0.02] + [0.0] * 5
    prices = _prices_from_returns(rets)
    side = pd.Series(0.0, index=prices.index)  # 全程无信号
    ml = meta_labeling(prices, side, pt=1.0, sl=1.0, vertical=5, vol_window=4)
    assert set(ml["meta_label"].unique()) == {0}
    assert int(ml["active"].sum()) == 0

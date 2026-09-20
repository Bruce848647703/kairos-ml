"""评估指标测试：命中率、分类指标、信号 PnL、IC、t 统计量与 VaR。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_ml as kml
from kairos_ml._util import norm_cdf
from kairos_ml.evaluate import (
    f1,
    hit_rate,
    ic,
    precision,
    recall,
    signal_pnl,
    t_statistic,
    value_at_risk,
)


def test_hit_rate_directional():
    yt = pd.Series([0.01, -0.02, 0.03, -0.01])
    yp = pd.Series([0.02, -0.01, -0.01, -0.03])  # 前两个方向对，第三个错，第四个对
    assert hit_rate(yt, yp) == pytest.approx(3 / 4)


def test_hit_rate_binary_equals_accuracy():
    yt = pd.Series([1, 0, 1, 1])
    yp = pd.Series([1, 0, 0, 1])
    assert hit_rate(yt, yp) == pytest.approx(0.75)


def test_precision_recall_f1_known():
    yt = pd.Series([1, 1, 0, 0, 1])
    yp = pd.Series([1, 0, 0, 1, 1])
    # TP: 位置0、4 => 2 ; FP: 位置3 => 1 ; FN: 位置1 => 1
    assert precision(yt, yp) == pytest.approx(2 / 3)
    assert recall(yt, yp) == pytest.approx(2 / 3)
    assert f1(yt, yp) == pytest.approx(2 / 3)


def test_precision_zero_denominator():
    yt = pd.Series([0, 0, 0])
    yp = pd.Series([0, 0, 0])
    assert precision(yt, yp, pos_label=1) == 0.0


def test_signal_pnl_elementwise_and_sum():
    sig = pd.Series([1.0, -1.0, 1.0, 0.0])
    fwd = pd.Series([0.01, 0.02, -0.01, 0.05])
    pnl = signal_pnl(sig, fwd)
    assert pnl.tolist() == pytest.approx([0.01, -0.02, -0.01, 0.0])
    assert pnl.sum() == pytest.approx(-0.02)


def test_ic_perfect_positive_and_negative():
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([1.1, 2.2, 2.9, 4.3])
    assert ic(a, b, method="spearman") == pytest.approx(1.0)
    assert ic(a, -b, method="spearman") == pytest.approx(-1.0)


def test_ic_pearson_vs_spearman_monotone():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    y = np.exp(x)  # 单调但非线性：spearman=1，pearson<1
    assert ic(x, y, method="spearman") == pytest.approx(1.0, abs=1e-9)
    assert ic(x, y, method="pearson") < 1.0


def test_norm_cdf_basic_values():
    assert float(norm_cdf(0.0)) == pytest.approx(0.5, abs=1e-6)
    assert float(norm_cdf(3.0)) == pytest.approx(0.99865, abs=1e-3)
    assert float(norm_cdf(-3.0)) == pytest.approx(1 - 0.99865, abs=1e-3)


def test_t_statistic_significant_positive():
    pnl = pd.Series([0.01] * 100)  # 恒正、零方差退化处理
    t, p = t_statistic(pnl)
    assert t == 0.0 and p == 1.0  # 方差为 0 -> 约定不显著
    noisy = pd.Series(np.full(200, 0.01) + np.linspace(-1e-4, 1e-4, 200))
    t2, p2 = t_statistic(noisy)
    assert t2 > 0 and 0.0 <= p2 <= 1.0


def test_value_at_risk_positive_for_normal_like():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.standard_normal(1000) * 0.01)
    var95 = value_at_risk(rets, confidence=0.95)
    var99 = value_at_risk(rets, confidence=0.99)
    assert var95 > 0
    assert var99 > var95  # 置信度越高，VaR 越大


def test_metric_functions_exported_at_top_level():
    for fn in (kml.hit_rate, kml.precision, kml.recall, kml.f1, kml.signal_pnl, kml.ic):
        assert callable(fn)

"""流水线测试：特征构造 → 模型 fit/predict 的端到端串联与索引对齐。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_ml as kml
from kairos_ml.pipeline import ModelPipeline


def _synthetic_prices(n=300, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    rets = rng.standard_normal(n) * 0.01
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)


def _builder(px: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "mom5": kml.rolling_momentum(px, 5),
        "rsi14": kml.rsi(px, 14),
        "vol20": kml.realized_volatility(px, 20),
    }, index=px.index)


def test_transform_builds_feature_frame():
    prices = _synthetic_prices()
    pipe = ModelPipeline(kml.LogisticRegression(), feature_builder=_builder)
    X = pipe.transform(prices)
    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == ["mom5", "rsi14", "vol20"]
    assert len(X) == len(prices)


def test_fit_predict_shapes_and_binary_output():
    prices = _synthetic_prices()
    y = (kml.forward_returns(prices, 5) > 0).astype(float)
    pipe = ModelPipeline(kml.LogisticRegression(C=1.0), feature_builder=_builder)
    pipe.fit(prices, y)
    pred = pipe.predict(prices)
    assert pred.shape == (len(prices),)
    assert set(np.unique(pred)).issubset({0, 1})
    proba = pipe.predict_proba(prices)
    assert proba.shape == (len(prices), 2)
    # 特征预热期（NaN 特征）对应 NaN 概率，仅在有效行上校验概率合法
    valid = pipe.transform(prices).notna().all(axis=1).to_numpy()
    sub = proba[valid]
    assert np.all(sub >= 0) and np.all(sub <= 1)
    assert np.allclose(sub.sum(axis=1), 1.0)


def test_fit_aligns_and_drops_nan_rows():
    """特征预热期 NaN 与标签末尾 NaN 应被自动对齐丢弃，模型仍能训练。"""
    prices = _synthetic_prices(n=200)
    y = kml.forward_returns(prices, 5)          # 末尾 5 个 NaN
    pipe = ModelPipeline(kml.RidgeRegression(alpha=1.0), feature_builder=_builder)
    pipe.fit(prices, y)                         # 特征前部也有 NaN
    assert pipe.model.coef_ is not None
    pred = pipe.predict(prices)
    assert pred.shape == (len(prices),)


def test_dict_feature_builder_concatenates():
    prices = _synthetic_prices(n=150)
    y = kml.forward_returns(prices, 3)
    builder = {
        "mom": lambda p: kml.rolling_momentum(p, 5),
        "rsi": lambda p: kml.rsi(p, 10),
    }
    pipe = ModelPipeline(kml.RidgeRegression(alpha=1.0), feature_builder=builder)
    pipe.fit(prices, y)
    X = pipe.transform(prices)
    assert list(X.columns) == ["mom", "rsi"]


def test_feature_columns_subset_and_order():
    prices = _synthetic_prices(n=120)
    y = (kml.forward_returns(prices, 3) > 0).astype(float)
    pipe = ModelPipeline(kml.LogisticRegression(), feature_builder=_builder,
                         feature_columns=["vol20", "mom5"])
    X = pipe.transform(prices)
    assert list(X.columns) == ["vol20", "mom5"]
    pipe.fit(prices, y)
    assert pipe.predict(prices).shape == (len(prices),)


def test_pipeline_beats_random_on_learnable_signal():
    """构造有可学习动量结构的数据，样本内命中率应明显高于 0.5。"""
    rng = np.random.default_rng(4)
    n = 600
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.25 * r[t - 1] + rng.standard_normal() * 0.01  # 正自相关 -> 动量可学
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)

    y = (kml.forward_returns(prices, 1) > 0).astype(float)
    pipe = ModelPipeline(kml.LogisticRegression(C=1.0), feature_builder=_builder)
    pipe.fit(prices, y)
    pred = pipe.predict(prices)
    valid = y.notna().to_numpy()
    hr = float(np.mean(pred[valid] == y.to_numpy()[valid]))
    assert hr > 0.55

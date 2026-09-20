"""模型测试：岭回归闭式解、逻辑回归、梯度提升与置换重要性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kairos_ml as kml
from kairos_ml.models import GradientBoostingRegressor, LogisticRegression, RidgeRegression, permutation_importance


# ---------------------------------------------------------------------------
# RidgeRegression
# ---------------------------------------------------------------------------
def test_ridge_matches_lstsq_when_alpha_tiny():
    """α→0 时闭式解应与 numpy.linalg.lstsq 一致。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 3))
    beta = np.array([1.5, -2.0, 0.5])
    y = X @ beta + 0.05 * rng.standard_normal(120)

    model = RidgeRegression(alpha=1e-10, fit_intercept=True).fit(X, y)
    Xd = np.column_stack([X, np.ones(120)])
    w, *_ = np.linalg.lstsq(Xd, y, rcond=None)

    assert model.coef_ == pytest.approx(w[:3], abs=1e-6)
    assert model.intercept_ == pytest.approx(w[3], abs=1e-6)


def test_ridge_alpha_zero_recovers_linear_exactly():
    """无噪声线性关系 + α=0：预测应几乎完美复现。"""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = (3.0 * X.ravel() + 2.0)
    model = RidgeRegression(alpha=0.0).fit(X, y)
    pred = model.predict(X)
    assert pred == pytest.approx(y, abs=1e-6)


def test_ridge_more_stable_on_collinear_data():
    """共线数据下，更大的 α 使系数范数更小、更稳定。"""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 3))
    X = np.column_stack([X, 2.0 * X[:, 0]])  # 完全共线列
    y = X[:, 0] - X[:, 1] + 0.1 * rng.standard_normal(100)

    small = RidgeRegression(alpha=0.01).fit(X, y)
    big = RidgeRegression(alpha=100.0).fit(X, y)

    assert np.all(np.isfinite(small.coef_)) and np.all(np.isfinite(big.coef_))
    assert np.linalg.norm(big.coef_) < np.linalg.norm(small.coef_)


def test_ridge_standardize_option_runs_and_predicts():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 4)) * np.array([1.0, 100.0, 0.01, 10.0])
    y = X[:, 1] + 0.5 * X[:, 3] + rng.standard_normal(80)
    model = RidgeRegression(alpha=1.0, standardize=True).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (80,)
    assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# LogisticRegression
# ---------------------------------------------------------------------------
def _separable(n=120, seed=5):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 2)) + np.array([3.0, 3.0])
    neg = rng.standard_normal((n, 2)) + np.array([-3.0, -3.0])
    X = np.vstack([pos, neg])
    y = np.r_[np.ones(n), np.zeros(n)]
    return X, y


def test_logistic_high_accuracy_on_separable():
    X, y = _separable()
    model = LogisticRegression(C=1.0).fit(X, y)
    acc = float((model.predict(X) == y).mean())
    assert acc > 0.9


def test_logistic_proba_in_unit_interval():
    X, y = _separable()
    model = LogisticRegression(C=1.0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_predict_matches_proba_argmax():
    X, y = _separable()
    model = LogisticRegression(C=1.0).fit(X, y)
    pred = model.predict(X)
    argmax = model.predict_proba(X).argmax(axis=1)
    assert np.array_equal(pred, argmax)


def test_logistic_accepts_dataframe_and_series_alignment():
    X, y = _separable(n=60)
    idx = pd.RangeIndex(len(y))
    Xdf = pd.DataFrame(X, columns=["f0", "f1"], index=idx)
    yser = pd.Series(y, index=idx)
    model = LogisticRegression(C=1.0).fit(Xdf, yser)
    assert float((model.predict(Xdf) == yser.to_numpy()).mean()) > 0.9


# ---------------------------------------------------------------------------
# GradientBoostingRegressor
# ---------------------------------------------------------------------------
def test_gbm_beats_constant_on_nonlinear():
    """浅树集成拟合非线性函数应显著优于常数（均值）预测。"""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((400, 2))
    y = X[:, 0] ** 2 + np.sin(2.0 * X[:, 1])

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                      max_depth=3, random_state=0).fit(X, y)
    mse_model = float(np.mean((y - model.predict(X)) ** 2))
    mse_const = float(np.mean((y - y.mean()) ** 2))
    assert mse_model < 0.5 * mse_const


def test_gbm_stump_default_depth_one():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((200, 1))
    y = np.where(X[:, 0] > 0, 1.0, -1.0)  # 阶跃函数，决策桩即可逼近
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2,
                                      max_depth=1, random_state=0).fit(X, y)
    assert model.max_depth == 1
    acc = float(np.mean(np.sign(model.predict(X)) == np.sign(y)))
    assert acc > 0.9


def test_gbm_more_estimators_reduce_train_error():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((300, 2))
    y = X[:, 0] * X[:, 1]
    few = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=2, random_state=0).fit(X, y)
    many = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0).fit(X, y)
    err_few = float(np.mean((y - few.predict(X)) ** 2))
    err_many = float(np.mean((y - many.predict(X)) ** 2))
    assert err_many < err_few


# ---------------------------------------------------------------------------
# permutation_importance
# ---------------------------------------------------------------------------
def test_permutation_importance_flags_useful_feature():
    """真正有用的特征应获得明显更高的重要性。"""
    rng = np.random.default_rng(13)
    n = 400
    X = rng.standard_normal((n, 4))
    y = 3.0 * X[:, 0] + 0.1 * rng.standard_normal(n)  # 仅 x0 有用
    df = pd.DataFrame(X, columns=["useful", "noise1", "noise2", "noise3"])

    model = RidgeRegression(alpha=0.1).fit(df, y)
    imp = permutation_importance(model, df, y, scorer="r2", n_repeats=8, random_state=0)

    assert imp.index[0] == "useful"
    assert imp.loc["useful", "importance_mean"] > imp.loc["noise1", "importance_mean"]
    assert imp.loc["useful", "importance_mean"] > imp.loc["noise2", "importance_mean"]
    assert imp.loc["useful", "importance_mean"] > 0.5


def test_permutation_importance_with_gbm_and_neg_mse():
    rng = np.random.default_rng(15)
    X = rng.standard_normal((300, 3))
    y = X[:, 1] ** 2 + 0.1 * rng.standard_normal(300)  # 仅 x1 有用（非线性）
    df = pd.DataFrame(X, columns=["a", "signal", "b"])
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                      max_depth=3, random_state=0).fit(df, y)
    imp = permutation_importance(model, df, y, scorer="neg_mse", n_repeats=6, random_state=0)
    assert imp.index[0] == "signal"


def test_permutation_importance_is_reproducible():
    rng = np.random.default_rng(17)
    X = rng.standard_normal((200, 3))
    y = X[:, 0] + 0.1 * rng.standard_normal(200)
    df = pd.DataFrame(X, columns=list("abc"))
    model = RidgeRegression(alpha=0.1).fit(df, y)
    i1 = permutation_importance(model, df, y, n_repeats=5, random_state=42)
    i2 = permutation_importance(model, df, y, n_repeats=5, random_state=42)
    pd.testing.assert_frame_equal(i1, i2)
